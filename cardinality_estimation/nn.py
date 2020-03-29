import time
import numpy as np
import pdb
import math
from db_utils.utils import *
from db_utils.query_storage import *
from utils.utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.net import *
from cardinality_estimation.losses import *
import pandas as pd
import json
import multiprocessing
# from torch.multiprocessing import Pool as Pool2
# import torch.multiprocessing as mp
# from utils.tf_summaries import TensorboardSummaries
from tensorflow import summary as tf_summary
from multiprocessing.pool import ThreadPool

# import park
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import random
from torch.nn.utils.clip_grad import clip_grad_norm_
from collections import defaultdict
import sys
import klepto
import datetime
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
from .algs import *
import sys
import gc

# dataset
from cardinality_estimation.query_dataset import QueryDataset
from torch.utils import data
from cardinality_estimation.join_loss import JoinLoss
from torch.multiprocessing import Pool as Pool2
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn")
except:
    pass

SUBQUERY_JERR_THRESHOLD = 100000
PERCENTILES_TO_SAVE = [1,5,10,25, 50, 75, 90, 95, 99]
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f

def fcnn_loss(net, use_qloss=False):
    def f(yhat,y):
        inp = torch.cat((yhat,y))
        jloss = net(inp)
        if use_qloss:
            qlosses = qloss_torch(yhat,y)
            qloss = sum(qlosses) / len(qlosses)
            # return qloss + jloss
            return (qloss / 100.0) + jloss
        else:
            return jloss
    return f

def single_train_combined_net(net, optimizer, loader, loss_fn,
        clip_gradient, load_query_together=False):
    for idx, (xbatch, ybatch,_) in enumerate(loader):
        # TODO: add handling for num_tables
        if load_query_together:
            # update the batches
            xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                    xbatch.shape[2])
            ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])

        pred = net(xbatch).squeeze(1)
        losses = loss_fn(pred, ybatch)
        loss = losses.sum() / len(losses)
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(net.parameters(), clip_gradient)
        optimizer.step()

def single_train_mscn(net, optimizer, loader, loss_fn,
        clip_gradient, load_query_together=False):
    assert not load_query_together
    for idx, (tbatch, pbatch, jbatch, ybatch,_) in enumerate(loader):
        pred = net(tbatch,pbatch,jbatch).squeeze(1)
        losses = loss_fn(pred, ybatch)
        loss = losses.sum() / len(losses)
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(net.parameters(), clip_gradient)
        optimizer.step()

def compute_subquery_priorities(qrep, true_cards, est_cards,
        explain, jerr, env, use_indexes=True):
    '''
    @return: subquery priorities, which must sum upto jerr.
    '''
    def get_sql(aliases):
        aliases = tuple(aliases)
        subgraph = qrep["join_graph"].subgraph(aliases)
        sql = nx_graph_to_query(subgraph)
        return sql, subgraph

    def handle_subtree(plan_tree, cur_node, cur_jerr):
        successors = list(plan_tree.successors(cur_node))
        # print(successors)
        if len(successors) == 0:
            return
        assert len(successors) == 2
        left = successors[0]
        right = successors[1]
        left_aliases = plan_tree.nodes()[left]["aliases"]
        right_aliases = plan_tree.nodes()[right]["aliases"]
        left_sql, left_sg = get_sql(left_aliases)
        right_sql, right_sg = get_sql(right_aliases)
        left_total_cost = plan_tree.nodes()[left]["Total Cost"]
        right_total_cost = plan_tree.nodes()[right]["Total Cost"]

        if len(left_aliases) >= 3:
            (left_est_costs, left_opt_costs,_,_,_,_) = \
                    join_loss_pg([left_sql], [left_sg], [true_cards],
                            [est_cards], env, use_indexes, None, 1)
            left_jerr = left_est_costs[0] - left_opt_costs[0]
        else:
            left_jerr = 0.0

        if len(right_aliases) >= 3:
            (right_est_costs, right_opt_costs,_,_,_,_) = \
                    join_loss_pg([right_sql], [right_sg], [true_cards],
                            [est_cards], env, use_indexes, None, 1)
            right_jerr = right_est_costs[0] - right_opt_costs[0]
        else:
            right_jerr = 0.00

        # -2.00 just added for close cases
        # assert left_jerr <= cur_jerr + 2.00
        # assert right_jerr <= cur_jerr + 2.00

        jerr_thresh = cur_jerr / 3.0
        if left_jerr < jerr_thresh \
                and right_jerr < jerr_thresh:
            # both the sides of the tree have low jerr, so take the complete
            # tree at this point and update the priorities of any subgraph
            # using the nodes in this tree (might not be in the current tree,
            # as those subqueries are important to reduce the jerr here as
            # well)
            cur_aliases = plan_tree.nodes()[left]["aliases"] + plan_tree.nodes()[right]["aliases"]
            for i, cur_nodes in enumerate(qrep["subset_graph"].nodes()):
                if set(cur_nodes) <= set(cur_aliases):
                    subpriorities[i] += cur_jerr
            return
        if left_jerr > jerr_thresh:
            handle_subtree(plan_tree, left, left_jerr)

        if right_jerr > jerr_thresh:
            handle_subtree(plan_tree, right, right_jerr)

    subpriorities = np.zeros(len(qrep["subset_graph"].nodes()))

    ## FIXME: what is a good initial priority?
    # give everyone the initial priority
    for i, _ in enumerate(qrep["subset_graph"].nodes()):
        subpriorities[i] = jerr

    for i, _ in enumerate(qrep["subset_graph"].nodes()):
        subpriorities[i] = 0

    # add sub-jerr priorities to the ones that are included in them - might at
    # most double it

    plan_tree = explain_to_nx(explain)
    root = [n for n,d in plan_tree.in_degree() if d==0]
    assert len(root) == 1
    handle_subtree(plan_tree, root[0], jerr)

    # TODO: normalize subpriorities
    subpriorities /= 1000000

    ## make sure it sums to jerr
    # subpriorities += 1
    # subpriorities /= np.sum(subpriorities)
    # subpriorities *= jerr

    return subpriorities

def compute_subquery_priorities_old(qrep, pred, env, use_indexes=True):
    priorities = np.ones(len(qrep["subset_graph"].nodes()))
    priority_dict = {}
    start = time.time()

    ests = {}
    trues = {}
    for i, node in enumerate(qrep["subset_graph"].nodes()):
        if len(node) > 4:
            continue
        alias_key = ' '.join(node)
        node_info = qrep["subset_graph"].nodes()[node]
        est_sel = pred[i]
        est_card = est_sel*node_info["cardinality"]["total"]
        ests[alias_key] = int(est_card)
        trues[alias_key] = node_info["cardinality"]["actual"]

    for i, node in enumerate(qrep["subset_graph"].nodes()):
        if len(node) != 4:
            continue
        node_info = qrep["subset_graph"].nodes()[node]
        # ests = {}
        # trues = {}
        subgraph = qrep["join_graph"].subgraph(node)
        sql = nx_graph_to_query(subgraph)
        # we need to go over descendants to get all the required cardinalities
        (est_costs, opt_costs,_,_,_,_) = \
                join_loss_pg([sql], [trues], [ests], env, use_indexes,
                        None, 1)

        # now, decide what to do next with these.
        assert len(est_costs) == 1
        jerr_diff = est_costs[0] - opt_costs[0]
        jerr_ratio = est_costs[0] / opt_costs[0]
        if jerr_diff < 20000:
            continue
        if jerr_ratio < 1.5:
            continue
        descendants = nx.descendants(qrep["subset_graph"], node)
        for desc in descendants:
            if desc not in priority_dict:
                priority_dict[desc] = 0.00
            priority_dict[desc] += jerr_ratio

    for i, node in enumerate(qrep["subset_graph"].nodes()):
        if node in priority_dict:
            priorities[i] += priority_dict[node]
    return priorities

def compute_subquery_priorities2(qrep, pred, env):
    '''
    Will use join error computation on the subqueries of @qrep to assign
    priorities to each subquery.

    Separate function to support calling this in parallel. Single threaded,
    because we don't want to iterate over all subqueries.
    @ret: priorities list for each subquery.
    '''
    pred_dict = {}
    for i, node in enumerate(qrep["subset_graph"].nodes()):
        pred_dict[node] = pred[i]

    nodes = list(qrep["subset_graph"].nodes())
    nodes.sort(key=lambda x: len(x), reverse=True)

    # sort by length, and go from longer to shorter. return array based on
    # original ordering.
    start = time.time()
    for node in nodes:
        if len(node) == 1:
            continue
        node_info = qrep["subset_graph"].nodes()[node]
        ests = {}
        trues = {}
        subgraph = qrep["join_graph"].subgraph(node)
        sql = nx_graph_to_query(subgraph)
        # we need to go over descendants to get all the required cardinalities
        descendants = nx.descendants(qrep["subset_graph"], node)
        for desc in descendants:
            alias_key = ' '.join(desc)
            node_info = qrep["subset_graph"].nodes()[desc]
            est_sel = pred_dict[desc]
            est_card = est_sel*node_info["cardinality"]["total"]
            ests[alias_key] = int(est_card)
            trues[alias_key] = node_info["cardinality"]["actual"]

        (est_costs, opt_costs,_,_,_,_) = \
                join_loss_pg([sql], [trues], [ests], env, use_indexes,
                        None, 1)

        # now, decide what to do next with these.
        assert len(est_costs) == 1
        jerr_diff = est_costs[0] - opt_costs[0]
        jerr_ratio = est_costs[0] / opt_costs[0]

    print("took: ", time.time() - start)
    pdb.set_trace()

    # convert back to the order we need

class NN(CardinalityEstimationAlg):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)
        if self.exp_prefix != "":
            self.exp_prefix += "-"
        if self.train_card_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
            self.wj_times = get_wj_times_dict(self.train_card_key)
        else:
            self.wj_times = None
        if self.load_query_together:
            assert self.num_groups == 1

        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekno = datetime.datetime.today().weekday()
        self.start_day = days[weekno]

        if self.load_query_together:
            self.mb_size = 1
        else:
            self.mb_size = 2500

        if self.nn_type == "microsoft":
            self.featurization_scheme = "combined"
        elif self.nn_type == "num_tables":
            self.featurization_scheme = "combined"
        elif self.nn_type == "mscn":
            self.featurization_scheme = "mscn"
        else:
            assert False

        if self.loss_func == "qloss":
            self.loss = qloss_torch
        elif self.loss_func == "rel":
            self.loss = rel_loss_torch
        elif self.loss_func == "weighted":
            self.loss = weighted_loss
        elif self.loss_func == "abs":
            self.loss = abs_loss_torch
        elif self.loss_func == "mse":
            self.loss = torch.nn.MSELoss(reduction="none")
        elif self.loss_func == "cm_fcnn":
            self.loss = qloss_torch
        else:
            assert False

        self.nets = []
        self.optimizers = []
        self.schedulers = []

        # each element is a list of priorities
        self.past_priorities = []

        # initialize stats collection stuff

        # will keep storing all the training information in the cache / and
        # dump it at the end
        # self.key = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # We want to only summarize and store the statistics we care about
        # header info:
        #   iter: every eval_epoch
        #   loss_type: qerr, join-loss etc.
        #   summary_type: mean, max, min, percentiles: 50,75th,90th,99th,25th
        #   template: all, OR only for specific template
        #   num_tables: all, OR t1,t2 etc.
        #   num_samples: in the given class, whose stats are being summarized.
        self.cur_stats = defaultdict(list)
        self.summary_funcs = [np.mean, np.max, np.min]
        self.summary_types = ["mean", "max", "min"]
        for q in PERCENTILES_TO_SAVE:
            self.summary_funcs.append(percentile_help(q))
            self.summary_types.append("percentile:{}".format(str(q)))

        self.query_stats = defaultdict(list)
        self.query_qerr_stats = defaultdict(list)

    def init_stats(self, samples):
        self.max_tables = 0
        self.max_val = 0
        self.min_val = 100000
        self.total_training_samples = 0

        for sample in samples:
            for node, info in sample["subset_graph"].nodes().items():
                self.total_training_samples += 1
                if len(node) > self.max_tables:
                    self.max_tables = len(node)
                card = info["cardinality"]["actual"]
                if card > self.max_val:
                    self.max_val = card
                if card < self.min_val:
                    self.min_val = card

    def init_groups(self, num_groups):
        groups = []
        for i in range(num_groups):
            groups.append([])

        tables_per_group = math.floor(self.max_tables / num_groups)
        for i in range(num_groups):
            start = i*tables_per_group
            for j in range(start,start+tables_per_group,1):
                groups[i].append(j+1)

        if j+1 < self.max_tables+1:
            for i in range(j+1,self.max_tables,1):
                print(i+1)
                groups[-1].append(i+1)

        print("nn groups: ", groups)
        return groups

    def _init_net(self, net_name, optimizer_name, sample):
        if net_name == "FCNN":
            # do training
            net = SimpleRegression(self.num_features,
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers,
                    hidden_layer_size=self.hidden_layer_size)
        elif net_name == "LinearRegression":
            net = LinearRegression(self.num_features,
                    1)
        elif net_name == "SetConv":
            if self.load_query_together:
                assert False
            net = SetConv(len(sample[0]), len(sample[1]), len(sample[2]),
                    self.hidden_layer_size, dropout= self.dropout)
        else:
            assert False

        if self.nn_weights_init_pg:
            print(net)
            new_weights = {}
            for key, weights in net.state_dict().items():
                print(key, len(weights))
                new_weights[key] = torch.zeros(weights.shape)
                if "bias" not in key:
                    new_weights[key][-1][-1] = 1.00

            net.load_state_dict(new_weights)
            print("state dict updated to pg init")

        if optimizer_name == "ams":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=True)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(),
                    lr=self.lr, momentum=0.9)
        else:
            assert False

        if self.adaptive_lr:
            scheduler = ReduceLROnPlateau(optimizer, 'min',
                    patience=self.adaptive_lr_patience,
                            verbose=True, factor=0.1, eps=self.lr)
        else:
            scheduler = None

        return net, optimizer, scheduler

    def init_nets(self, sample):
        for i in range(len(self.groups)):
            if self.nn_type == "mscn":
                net, optimizer, scheduler = \
                        self._init_net("SetConv", self.optimizer_name, sample)
            else:
                net, optimizer, scheduler = \
                        self._init_net(self.net_name, self.optimizer_name, sample)
            self.nets.append(net)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

        print("initialized {} nets for num_tables version".format(len(self.nets)))
        assert len(self.nets) == len(self.groups)

    def _eval_combined(self, net, loader):
        tstart = time.time()
        # TODO: set num threads, gradient off etc.
        torch.set_grad_enabled(False)
        all_preds = []
        all_y = []
        all_idxs = []

        for idx, (xbatch, ybatch,info) in enumerate(loader):
            pred = net(xbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)
            all_idxs.append(info["dataset_idx"])

        pred = torch.cat(all_preds).detach().numpy()
        y = torch.cat(all_y).detach().numpy()
        all_idxs = torch.cat(all_idxs).detach().numpy()
        return pred, y, all_idxs

    def _eval_mscn(self, net, loader):
        tstart = time.time()
        # TODO: set num threads, gradient off etc.
        torch.set_grad_enabled(False)
        all_preds = []
        all_y = []
        all_idxs = []

        for idx, (tbatch,pbatch,jbatch, ybatch,info) in enumerate(loader):
            pred = net(tbatch,pbatch,jbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)
            all_idxs.append(info["dataset_idx"])

        pred = torch.cat(all_preds).detach().numpy()
        y = torch.cat(all_y).detach().numpy()
        all_idxs = torch.cat(all_idxs).detach().numpy()
        return pred, y, all_idxs

    def _eval_loaders(self, loaders):
        if len(loaders) == 1:
            # could also reduce this to the second case, but we don't need to
            # reindex stuff here, so might as well avoid it
            net = self.nets[0]
            loader = loaders[0]
            if self.featurization_scheme == "combined":
                res = self._eval_combined(net, loader)
            elif self.featurization_scheme == "mscn":
                res = self._eval_mscn(net, loader)

            pred = to_variable(res[0]).float()
            y = to_variable(res[1]).float()
        else:
            # in this case, we need to fix indexes after the evaluation
            res = []
            for i, loader in enumerate(loaders):
                if self.featurization_scheme == "combined":
                    res.append(self._eval_combined(self.nets[i], loader))
                elif self.featurization_scheme == "mscn":
                    res.append(self._eval_mscn(self.nets[i], loader))

            # FIXME: this can be faster
            idxs = [r[2] for r in res]
            all_preds = [r[0] for r in res]
            all_ys = [r[1] for r in res]
            idxs = np.concatenate(idxs)
            all_preds = np.concatenate(all_preds)
            all_ys = np.concatenate(all_ys)
            max_idx = np.max(idxs)
            assert len(idxs) == max_idx+1

            pred = np.zeros(len(idxs))
            y = np.zeros(len(idxs))
            for i, val in enumerate(all_preds):
                pred[idxs[i]] = val
                y[idxs[i]] = all_ys[i]
            pred = to_variable(pred).float()
            y = to_variable(y).float()

        return pred,y

    def _eval_samples(self, loaders):
        '''
        @ret: numpy arrays
        '''
        # don't need to compute gradients, saves a lot of memory
        torch.set_grad_enabled(False)
        ret = self._eval_loaders(loaders)
        torch.set_grad_enabled(True)
        return ret

    def eval_samples(self, samples_type):
        loaders = self.eval_loaders[samples_type]
        return self._eval_samples(loaders)

    def add_row(self, losses, loss_type, epoch, template,
            num_tables, samples_type):
        for i, func in enumerate(self.summary_funcs):
            loss = func(losses)
            row = [epoch, loss_type, loss, self.summary_types[i],
                    template, num_tables, len(losses)]
            self.cur_stats["epoch"].append(epoch)
            self.cur_stats["loss_type"].append(loss_type)
            self.cur_stats["loss"].append(loss)
            self.cur_stats["summary_type"].append(self.summary_types[i])
            self.cur_stats["template"].append(template)
            self.cur_stats["num_tables"].append(num_tables)
            self.cur_stats["num_samples"].append(len(losses))
            self.cur_stats["samples_type"].append(samples_type)
            if self.summary_types[i] == "mean" and \
                    (template == "all" or num_tables == "all") \
                    and self.tfboard:
                stat_name = self.tf_stat_fmt.format(
                        samples_type = samples_type,
                        loss_type = loss_type,
                        num_tables = num_tables,
                        template = template)
                with self.tf_summary_writer.as_default():
                    tf_summary.scalar(stat_name, loss, step=epoch)

    def get_exp_name(self):
        '''
        '''
        time_hash = str(deterministic_hash(self.start_time))[0:3]
        name = "{PREFIX}{NN}-{PRIORITY}-{HASH}".format(\
                    PREFIX = self.exp_prefix,
                    NN = self.__str__(),
                    PRIORITY = self.sampling_priority_alpha,
                    HASH = time_hash)
        return name

    def save_stats(self):
        '''
        replaces the results file.
        '''
        # TODO: maybe reset cur_stats
        self.stats = pd.DataFrame(self.cur_stats)
        if not os.path.exists(self.result_dir):
            make_dir(self.result_dir)
        exp_name = self.get_exp_name()
        exp_dir = self.result_dir + "/" + exp_name
        if not os.path.exists(exp_dir):
            make_dir(exp_dir)

        fn = exp_dir + "/" + "nn.pkl"

        results = {}
        results["stats"] = self.stats
        results["config"] = self.kwargs
        results["name"] = self.__str__()
        results["query_stats"] = self.query_stats
        results["query_qerr_stats"] = self.query_qerr_stats

        with open(fn, 'wb') as fp:
            pickle.dump(results, fp,
                    protocol=pickle.HIGHEST_PROTOCOL)

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6
        num_params = 0
        for net in self.nets:
            num_params += _calc_size(net)

        return num_params

    def _single_train_combined_net_cm(self, net, optimizer, loader):
        assert self.load_query_together
        for idx, (xbatch, ybatch,_) in enumerate(loader):
            # TODO: add handling for num_tables
            xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                    xbatch.shape[2])
            ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
            pred = net(xbatch).squeeze(1)
            loss = self.cm_loss(pred, ybatch)
            assert len(loss) == 1
            optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()
            optimizer.step()

    def train_one_epoch(self):
        if self.loss_func == "cm_fcnn":
            assert len(self.nets) == 1
            self._single_train_combined_net_cm(self.nets[0], self.optimizers[0],
                    self.training_loaders[0])

        for i, net in enumerate(self.nets):
            opt = self.optimizers[i]
            loader = self.training_loaders[i]

            if self.featurization_scheme == "combined":
                single_train_combined_net(net, opt, loader, self.loss,
                        self.clip_gradient, self.load_query_together)
            elif self.featurization_scheme == "mscn":
                single_train_mscn(net, opt, loader, self.loss,
                        self.clip_gradient, self.load_query_together)
            else:
                assert False

    def save_join_loss_stats(self, join_losses, est_plans, samples,
            samples_type):
        self.add_row(join_losses, "jerr", self.epoch, "all",
                "all", samples_type)
        print("{}, join losses mean: {}".format(samples_type,
                np.mean(join_losses)))

        summary_data = defaultdict(list)

        query_idx = 0
        for i, sample in enumerate(samples):
            template = sample["template_name"]
            summary_data["template"].append(template)
            summary_data["loss"].append(join_losses[i])

        df = pd.DataFrame(summary_data)
        for template in set(df["template"]):
            tvals = df[df["template"] == template]
            self.add_row(tvals["loss"].values, "jerr", self.epoch,
                    template, "all", samples_type)

        for i, sample in enumerate(samples):
            self.query_stats["epoch"].append(self.epoch)
            self.query_stats["query_name"].append(sample["name"])
            # this is also equivalent to the priority, we can normalize it
            # later
            self.query_stats["jerr"].append(join_losses[i])
            self.query_stats["plan"].append(get_leading_hint(est_plans[i]))
            self.query_stats["explain"].append(est_plans[i])

    def periodic_eval(self, samples_type):
        pred, Y = self.eval_samples(samples_type)
        losses = self.loss(pred, Y).detach().numpy()
        # FIXME: self.loss could be various loss functions, like mse, but we care
        # about observing how the q-error is evolving

        loss_avg = round(np.sum(losses) / len(losses), 6)
        # TODO: better print, use self.cur_stats and print after evals
        print("""{}: {}, N: {}, qerr: {}""".format(
            samples_type, self.epoch, len(Y), loss_avg))
        if self.adaptive_lr and self.scheduler is not None:
            self.scheduler.step(loss_avg)

        self.add_row(losses, "qerr", self.epoch, "all",
                "all", samples_type)
        samples = self.samples[samples_type]
        summary_data = defaultdict(list)
        query_idx = 0
        for sample in samples:
            template = sample["template_name"]
            sample_losses = []
            for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
                num_tables = len(node)
                idx = query_idx + subq_idx
                loss = float(losses[idx])
                sample_losses.append(loss)
                summary_data["loss"].append(loss)
                summary_data["num_tables"].append(num_tables)
                summary_data["template"].append(template)
            query_idx += len(sample["subset_graph"].nodes())
            self.query_qerr_stats["epoch"].append(self.epoch)
            self.query_qerr_stats["query_name"].append(sample["name"])
            self.query_qerr_stats["qerr"].append(sum(sample_losses) / len(sample_losses))

        df = pd.DataFrame(summary_data)
        for template in set(df["template"]):
            tvals = df[df["template"] == template]
            self.add_row(tvals["loss"].values, "qerr", self.epoch,
                    template, "all", samples_type)

        for nt in set(df["num_tables"]):
            nt_losses = df[df["num_tables"] == nt]
            self.add_row(nt_losses["loss"].values, "qerr", self.epoch, "all",
                    str(nt), samples_type)

        if (self.epoch % self.eval_epoch_jerr == 0 \
                and self.epoch != 0):
            if (samples_type == "train" and \
                    self.sampling_priority_alpha > 0 and \
                    self.epoch % self.reprioritize_epoch == 0):
                if self.train_card_key == "actual":
                    print("not recalculating join loss for training")
                    return
                print("recalculating join loss")
            print("going to calculate join loss")
            print(self.epoch, self.eval_epoch_jerr)
            # if priority on, then stats will be saved when calculating
            # priority
            jl_eval_start = time.time()
            assert self.jl_use_postgres

            sqls, jgs, true_cardinalities, est_cardinalities = \
                    self.get_query_estimates(pred, samples)
            (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                    jgs,
                    true_cardinalities, est_cardinalities, self.env,
                    self.jl_indexes, None,
                    pool = self.join_loss_pool,
                    join_loss_data_file = self.join_loss_data_file)

            join_losses = np.array(est_costs) - np.array(opt_costs)
            # join_losses = np.maximum(join_losses, 0.00)

            self.save_join_loss_stats(join_losses, est_plans, samples,
                    samples_type)

            # TODO: what to do with prioritization?

    def _normalize_priorities(self, priorities):
        total = np.float64(np.sum(priorities))
        norm_priorities = np.zeros(len(priorities))
        norm_priorities = np.divide(priorities, total)

        # if they don't sum to 1...

        if 1.00 - sum(norm_priorities) != 0.00:
            diff = 1.00 - sum(norm_priorities)
            while True:
                random_idx = np.random.randint(0,len(norm_priorities))
                if diff < 0.00 and norm_priorities[random_idx] < abs(diff):
                    continue
                else:
                    norm_priorities[random_idx] += diff
                    break

        return norm_priorities

    def _update_sampling_weights(self, priorities):
        '''
        refer to prioritized action replay
        '''
        priorities = np.power(priorities, self.sampling_priority_alpha)
        priorities = self._normalize_priorities(priorities)

        if self.avg_jl_priority:
            self.past_priorities.append(priorities)
            if len(self.past_priorities) > 1:
                new_priorities = np.zeros(len(priorities))
                num_past = min(self.num_last, len(self.past_priorities))
                for i in range(1,num_past+1):
                    new_priorities += self.past_priorities[-i]
                priorities = self._normalize_priorities(new_priorities)

        return priorities

    def get_query_estimates(self, pred, samples, true_card_key="actual"):
        '''
        @ret:
        '''
        if not isinstance(pred, np.ndarray):
            pred = pred.detach().numpy()
        sqls = []
        true_cardinalities = []
        est_cardinalities = []
        join_graphs = []
        query_idx = 0
        for sample in samples:
            if true_card_key != "actual":
                sql = "/* {} */ ".format(true_card_key) + sample["sql"]
            else:
                sql = sample["sql"]
            sqls.append(sql)
            join_graphs.append(sample["join_graph"])
            ests = {}
            trues = {}
            # we don't need to sort these as we are returning a dict here...

            node_keys = list(sample["subset_graph"].nodes())
            node_keys.sort()
            for subq_idx, node in enumerate(node_keys):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = ' '.join(node)
                # alias_key = node
                idx = query_idx + subq_idx
                if self.normalization_type == "mscn":
                    est_card = np.exp((pred[idx] + \
                        self.min_val)*(self.max_val-self.min_val))
                else:
                    est_sel = pred[idx]
                    est_card = est_sel*cards["total"]
                ests[alias_key] = est_card
                if self.wj_times is not None:
                    ck = "wanderjoin-" + str(self.wj_times[sample["template_name"]])
                    true_val = cards[ck]
                    if true_val == 0:
                        true_val = cards["expected"]
                else:
                    true_val = cards[true_card_key]
                trues[alias_key] = true_val

            est_cardinalities.append(ests)
            true_cardinalities.append(trues)
            query_idx += len(sample["subset_graph"].nodes())

        return sqls, join_graphs, true_cardinalities, est_cardinalities

    def initialize_tfboard(self):
        name = self.get_exp_name()
        # name = self.__str__()
        log_dir = "tfboard_logs/" + name
        self.tf_summary_writer = tf_summary.create_file_writer(log_dir)
        self.tf_stat_fmt = "{samples_type}-{loss_type}-nt:{num_tables}-tmp:{template}"

    def init_dataset(self, samples, shuffle, batch_size,
            weighted=False):
        training_sets = []
        training_loaders = []
        for i in range(len(self.groups)):
            training_sets.append(QueryDataset(samples, self.db,
                    self.featurization_scheme, self.heuristic_features,
                    self.preload_features, self.normalization_type,
                    self.load_query_together,
                    min_val = self.min_val,
                    max_val = self.max_val,
                    card_key = self.train_card_key,
                    group = self.groups[i]))
            if not weighted:
                training_loaders.append(data.DataLoader(training_sets[i],
                        batch_size=batch_size, shuffle=shuffle, num_workers=0))
            else:
                weight = 1 / len(training_sets[i])
                weights = torch.DoubleTensor([weight]*len(training_sets[i]))
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                        num_samples=len(weights))
                training_loader = data.DataLoader(training_sets[i],
                        batch_size=self.mb_size, shuffle=False, num_workers=0,
                        sampler = sampler)
                training_loaders.append(training_loader)
                # priority_loader = data.DataLoader(training_set,
                        # batch_size=25000, shuffle=False, num_workers=0)

        assert len(training_sets) == len(self.groups) == len(training_loaders)
        return training_sets, training_loaders


    def train(self, db, training_samples, use_subqueries=False,
            test_samples=None, join_loss_pool = None):
        assert isinstance(training_samples[0], dict)

        self.join_loss_pool = join_loss_pool

        if self.tfboard:
            self.initialize_tfboard()
        # model is always small enough that it runs fast w/o using many cores
        torch.set_num_threads(2)
        self.db = db
        db.init_featurizer(num_tables_feature = self.num_tables_feature,
                max_discrete_featurizing_buckets =
                self.max_discrete_featurizing_buckets,
                heuristic_features = self.heuristic_features)

        self.init_stats(training_samples)
        self.groups = self.init_groups(self.num_groups)

        if self.normalization_type == "mscn":
            y = np.array(get_all_cardinalities(training_samples))
            y = np.log(y)
            self.max_val = np.max(y)
            self.min_val = np.min(y)
            print("min val: ", self.min_val)
            print("max val: ", self.max_val)
        else:
            assert self.normalization_type == "pg_total_selectivity"
            self.min_val, self.max_val = None, None

        # create a new park env, and close at the end.
        self.env = JoinLoss(self.db.user, self.db.pwd, self.db.db_host,
                self.db.port, self.db.db_name)

        self.training_samples = training_samples
        if self.sampling_priority_alpha > 0.00:
            training_sets, self.training_loaders = self.init_dataset(training_samples,
                                    False, self.mb_size, weighted=True)
            priority_loaders = []
            for i, ds in enumerate(self.training_sets):
                priority_loaders.append(data.dataloader(ds,
                        batch_size=25000, shuffle=False, num_workers=0))
        else:
            training_sets, self.training_loaders = self.init_dataset(training_samples,
                                    True, self.mb_size, weighted=False)

        assert len(self.training_loaders) == len(self.groups)

        if self.featurization_scheme == "combined":
            if self.load_query_together:
                self.num_features = len(training_sets[0][0][0][0])
            else:
                self.num_features = len(training_sets[0][0][0])
                if len(self.groups) == 1:
                    assert self.total_training_samples == len(training_sets[0])
        else:
            self.num_features = len(training_sets[0][0][0]) + \
                    len(training_sets[0][0][1]) + len(training_sets[0][0][2])


        if self.priority_normalize_type == "paths1":
            subw_start = time.time()
            subquery_rel_weights = np.zeros(len(training_set))
            qidx = 0
            template_weights = {}
            for sample in training_samples:
                subsetg = sample["subset_graph"]
                node_list = list(subsetg.nodes())
                node_list.sort(key = lambda v: len(v))
                dest = node_list[-1]
                node_list.sort()
                cur_weights = np.zeros(len(node_list))
                if sample["template_name"] in template_weights:
                    cur_weights = template_weights[sample["template_name"]]
                else:
                    for i, node in enumerate(node_list):
                        all_paths = nx.all_simple_paths(subsetg, dest, node)
                        num_paths = len(list(all_paths))
                        cur_weights[i] = num_paths
                    cur_weights = cur_weights / sum(cur_weights)
                    assert np.abs(sum(cur_weights) - 1.0) < 0.001
                    template_weights[sample["template_name"]] = cur_weights

                subquery_rel_weights[qidx:qidx+len(node_list)] = cur_weights

                qidx += len(node_list)
            print("subquery weights calculate in: ", time.time()-subw_start)

        if self.loss_func == "cm_fcnn":
            inp_len = len(training_samples[0]["subset_graph"].nodes())
            # fcnn_net = SimpleRegression(inp_len*2, 2, 1,
                    # num_hidden_layers=1)
            fcnn_net = torch.load("./cm_fcnn.pt")
            self.cm_loss = fcnn_loss(fcnn_net)

        # evaluation set, smaller
        self.samples = {}
        self.eval_loaders = {}
        random.seed(2112)
        if self.debug_set:
            eval_samples_size_divider = 1
        else:
            # eval_samples_size_divider = 10
            eval_samples_size_divider = 1

        eval_training_samples = random.sample(training_samples,
                int(len(training_samples) / eval_samples_size_divider))
        self.samples["train"] = eval_training_samples

        eval_train_sets, eval_train_loaders = \
                self.init_dataset(eval_training_samples, False, 10000, weighted=False)

        self.eval_loaders["train"] = eval_train_loaders

        # TODO: add separate dataset, dataloaders for evaluation
        if test_samples is not None and len(test_samples) > 0:
            test_samples = random.sample(test_samples, int(len(test_samples) /
                    eval_samples_size_divider))
            self.samples["test"] = test_samples
            eval_test_sets, eval_test_loaders = \
                    self.init_dataset(test_samples, False, 10000, weighted=False)
            self.eval_loaders["test"] = eval_test_loaders
        else:
            self.samples["test"] = None

        # TODO: initialize self.num_features
        self.init_nets(training_sets[0][0])

        model_size = self.num_parameters()
        print("""training samples: {}, feature length: {}, model size: {},
        max_discrete_buckets: {}, hidden_layer_size: {}""".\
                format(self.total_training_samples, self.num_features, model_size,
                    self.max_discrete_featurizing_buckets,
                    self.hidden_layer_size))

        for self.epoch in range(1,self.max_epochs):
            if self.epoch % self.eval_epoch == 0:
                eval_start = time.time()
                self.periodic_eval("train")
                if self.samples["test"] is not None:
                    self.periodic_eval("test")
                self.save_stats()

            start = time.time()
            self.train_one_epoch()
            print("train epoch took: ", time.time() - start)

            if self.sampling_priority_alpha > 0 \
                    and (self.epoch % self.reprioritize_epoch == 0 \
                            or self.epoch == self.prioritize_epoch):
                pred, _ = self._eval_samples(priority_loaders)
                pred = pred.detach().numpy()
                weights = np.zeros(len(training_set))
                if self.sampling_priority_type == "query":
                    # TODO: decompose
                    pr_start = time.time()
                    sqls, jgs, true_cardinalities, est_cardinalities = \
                            self.get_query_estimates(pred,
                                    self.training_samples,
                                    true_card_key=self.train_card_key)
                    (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                            jgs, true_cardinalities, est_cardinalities, self.env,
                            self.jl_indexes, None,
                            pool = self.join_loss_pool,
                            join_loss_data_file = self.join_loss_data_file)

                    jerr_ratio = est_costs / opt_costs
                    jerr = est_costs - opt_costs
                    # don't do this if train_card_key is not actual, as this
                    # does not reflect real join loss
                    if self.train_card_key == "actual":
                        self.save_join_loss_stats(jerr, est_plans,
                                self.training_samples, "train")

                    print("epoch: {}, jerr_ratio: {}, jerr: {}, time: {}"\
                            .format(self.epoch,
                                np.round(np.mean(jerr_ratio), 2),
                                np.round(np.mean(jerr), 2),
                                time.time()-pr_start))
                    query_idx = 0
                    for si, sample in enumerate(self.training_samples):
                        if self.priority_err_type == "jerr":
                            sq_weight = jerr[si] / 1000000.00
                        else:
                            sq_weight = jerr_ratio[si]

                        if self.priority_normalize_type == "div":
                            sq_weight /= len(sample["subset_graph"].nodes())
                        elif self.priority_normalize_type == "paths1":
                            pass

                        for subq_idx, _ in enumerate(sample["subset_graph"].nodes()):
                            weights[query_idx+subq_idx] = sq_weight

                        query_idx += len(sample["subset_graph"].nodes())

                    if self.priority_normalize_type == "paths1":
                        weights *= subquery_rel_weights

                elif self.sampling_priority_type == "subquery":
                    pr_start = time.time()
                    sqls, jgs, true_cardinalities, est_cardinalities = \
                            self.get_query_estimates(pred,
                                    self.training_samples,
                                    true_card_key = self.train_card_key)
                    (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                            jgs, true_cardinalities, est_cardinalities, self.env,
                            self.jl_indexes, None,
                            pool = self.join_loss_pool,
                            join_loss_data_file = self.join_loss_data_file)
                    jerrs = est_costs - opt_costs
                    jerr_ratio = est_costs / opt_costs
                    self.save_join_loss_stats(jerrs, est_plans,
                            self.training_samples, "train")

                    print("epoch: {}, jerr_ratio: {}, jerr: {}, time: {}"\
                            .format(self.epoch,
                                np.round(np.mean(jerr_ratio), 2),
                                np.round(np.mean(jerrs), 2),
                                time.time()-pr_start))
                    par_args = []
                    num_proc = 8
                    for si, sample in enumerate(self.training_samples):
                        par_args.append((sample,
                                    true_cardinalities[si], est_cardinalities[si],
                                    est_plans[si], jerrs[si], self.env,
                                    self.jl_indexes))
                    # with Pool(processes=num_proc) as pool:
                        # all_subps = pool.starmap(compute_subquery_priorities, par_args)
                    all_subps = self.join_loss_pool.starmap(compute_subquery_priorities,
                            par_args)

                    query_idx = 0
                    for si, sample in enumerate(self.training_samples):
                        subps = all_subps[si]
                        slen = len(sample["subset_graph"].nodes())
                        weights[query_idx:query_idx+slen] = subps
                        query_idx += slen

                elif self.sampling_priority_type == "subquery_old":
                    start = time.time()
                    par_args = []
                    query_idx = 0
                    for _, qrep in enumerate(self.training_samples):
                        sqs = len(qrep["subset_graph"].nodes())
                        par_args.append((qrep, pred[query_idx:query_idx+sqs],
                            self.env))
                        query_idx += sqs
                    with ThreadPool(processes=multiprocessing.cpu_count()) as pool:
                        all_priorities = pool.starmap(compute_subquery_priorities, par_args)
                    print("subquery sampling took: ", time.time() - start)
                    weights = np.concatenate(all_priorities)
                    assert len(weights) == len(training_set)
                else:
                    assert False

                weights = self._update_sampling_weights(weights)

                weights = torch.DoubleTensor(weights)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                        num_samples=len(weights))
                self.training_loader = data.DataLoader(training_set,
                        batch_size=self.mb_size, shuffle=False, num_workers=0,
                        sampler = sampler)

    def test(self, test_samples):
        '''
        @test_samples: [] sql_representation dicts
        '''
        datasets, loaders = \
                self.init_dataset(test_samples, False, 10000, weighted=False)
        pred, y = self._eval_samples(loaders)
        if self.preload_features:
            for dataset in datasets:
                del(dataset.X)
                del(dataset.Y)
                del(dataset.info)
        loss = self.loss(pred, y).detach().numpy()
        print("loss after test: ", np.mean(loss))
        pred = pred.detach().numpy()

        all_ests = []
        query_idx = 0
        for sample in test_samples:
            ests = {}
            node_keys = list(sample["subset_graph"].nodes())
            node_keys.sort()
            # for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
            for subq_idx, node in enumerate(node_keys):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = node
                idx = query_idx + subq_idx
                if self.normalization_type == "mscn":
                    est_card = np.exp((pred[idx] + \
                        self.min_val)*(self.max_val-self.min_val))
                elif self.normalization_type == "pg_total_selectivity":
                    est_sel = pred[idx]
                    est_card = est_sel*cards["total"]
                else:
                    assert False
                ests[alias_key] = est_card
            all_ests.append(ests)
            query_idx += len(sample["subset_graph"].nodes())
        # assert query_idx == len(dataset)
        return all_ests

    def __str__(self):
        if self.nn_type == "microsoft":
            name = "msft"
        elif self.nn_type == "num_tables":
            name = "nt"
        elif self.nn_type == "mscn":
            name = "mscn"
        else:
            name = self.__class__.__name__

        if self.max_discrete_featurizing_buckets:
            name += "-df:" + str(self.max_discrete_featurizing_buckets)
        if self.sampling_priority_alpha > 0.00:
            name += "-pr:" + str(self.sampling_priority_alpha)
        name += "-nn:" + str(self.num_hidden_layers) + ":" + str(self.hidden_layer_size)

        if self.sampling_priority_type != "query":
            name += "-spt:" + self.sampling_priority_type
        # if not self.priority_query_len_scale:
            # name += "-psqls:0"
        if not self.avg_jl_priority:
            name += "-no_pr_avg"

        if self.loss_func != "qloss":
            name += "-loss:" + self.loss_func
        if not self.heuristic_features:
            name += "-no_pg_ests"

        return name

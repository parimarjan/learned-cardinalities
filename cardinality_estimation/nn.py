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

import park
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

PERCENTILES_TO_SAVE = [25, 50, 75, 90, 99]
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f

def compute_subquery_priorities(qrep, true_cards, est_cards,
        explain, jerr, env, use_indexes=True):
    '''
    @return: subquery priorities
    '''
    def get_sql(aliases):
        aliases = tuple(aliases)
        subgraph = qrep["join_graph"].subgraph(aliases)
        sql = nx_graph_to_query(subgraph)
        return sql

    def handle_subtree(plan_tree, cur_node, cur_jerr):
        successors = list(plan_tree.successors(cur_node))
        # print(successors)
        if len(successors) == 0:
            return
        assert len(successors) == 2
        # if len(successors) != 2:
            # print(successors)
            # pdb.set_trace()
        left = successors[0]
        right = successors[1]
        left_sql = get_sql(plan_tree.nodes()[left]["aliases"])
        right_sql = get_sql(plan_tree.nodes()[right]["aliases"])
        # computer left_jerr, right_jerr
        (left_est_costs, left_opt_costs,_,_,_,_) = \
                join_loss_pg([left_sql], [trues], [ests], env, use_indexes,
                        None, 1)

        handle_subtree(plan_tree, left, cur_jerr)
        handle_subtree(plan_tree, right, cur_jerr)

    plan_tree = explain_to_nx(explain)
    root = [n for n,d in plan_tree.in_degree() if d==0]
    assert len(root) == 1
    handle_subtree(plan_tree, root[0], jerr)
    pdb.set_trace()

    # handle_subtree(plan_tree, jerr)

    # all_nodes = list(qrep["join_graph"].nodes())
    # all_nodes.sort()
    # all_nodes_key = None
    # for node, data in plan_tree.nodes(data=True):
        # aliases = data["aliases"]
        # aliases.sort()
        # if aliases == all_nodes:
            # plan_tree.nodes()[node]["jerr"] = jerr
            # all_nodes_key = node
            # break
        # aliases = tuple(aliases)
        # subgraph = qrep["join_graph"].subgraph(aliases)
        # sql = nx_graph_to_query(subgraph)
        # print(sql)

        # node_info = qrep["subset_graph"].nodes()[node]
        # for node2 in plan_tree.successors(node):
            # print("successors: ", node2)

    pdb.set_trace()

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
        # print(node, jerr_diff, jerr_ratio)
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
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekno = datetime.datetime.today().weekday()
        self.start_day = days[weekno]

        # initialize stats collection stuff
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
        else:
            assert False

        self.net = None
        self.optimizer = None
        self.scheduler = None

        # each element is a list of priorities
        self.past_priorities = []

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

    def _init_net(self, net_name, optimizer_name, sample):
        if net_name == "FCNN":
            num_features = len(sample[0])
            # do training
            net = SimpleRegression(num_features,
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers,
                    hidden_layer_size=self.hidden_layer_size)
        elif net_name == "LinearRegression":
            net = LinearRegression(num_features,
                    1)
        elif net_name == "SetConv":
            net = SetConv(len(sample[0]), len(sample[1]), len(sample[2]),
                    self.hidden_layer_size)
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
        # TODO: num_tables version, need have multiple neural nets
        if self.nn_type == "num_tables":
            self.nets = {}
            self.optimizers = {}
            self.schedulers = {}
            for num_table in self.train_num_table_mapping:
                num_table = self._map_num_tables(num_table)
                if num_table not in self.nets:
                    net, opt, scheduler = self._init_net(self.net_name,
                            self.optimizer_name, sample)
                    self.nets[num_table] = net
                    self.optimizers[num_table] = opt
                    self.schedulers[num_table] = scheduler
            print("initialized {} nets for num_tables version".format(len(self.nets)))
        elif self.nn_type == "mscn":
            self.net, self.optimizer, self.scheduler = \
                    self._init_net("SetConv", self.optimizer_name, sample)
        else:
            self.net, self.optimizer, self.scheduler = \
                    self._init_net(self.net_name, self.optimizer_name, sample)

    def _eval_combined(self, loader):
        all_preds = []
        all_y = []
        for idx, (xbatch, ybatch,_) in enumerate(loader):
            pred = self.net(xbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)
        pred = torch.cat(all_preds)
        y = torch.cat(all_y)
        return pred,y

    def _eval_mscn(self, loader):
        all_preds = []
        all_y = []
        for idx, (tbatch,pbatch,jbatch, ybatch,_) in enumerate(loader):
            pred = self.net(tbatch,pbatch,jbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)
        pred = torch.cat(all_preds)
        y = torch.cat(all_y)
        return pred,y

    def _eval_samples(self, loader):
        torch.set_grad_enabled(False)
        if self.featurization_scheme == "combined":
            ret = self._eval_combined(loader)
        elif self.featurization_scheme == "mscn":
            ret = self._eval_mscn(loader)
        torch.set_grad_enabled(True)
        return ret

    def eval_samples(self, samples_type):
        loader = self.eval_loaders[samples_type]
        return self._eval_samples(loader)

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
        name = "{PREFIX}-{NN}-{PRIORITY}-{HASH}".format(\
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

        with open(fn, 'wb') as fp:
            pickle.dump(results, fp,
                    protocol=pickle.HIGHEST_PROTOCOL)

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6

        if self.nn_type == "num_tables":
            num_params = 0
            for _,net in self.nets.items():
                num_params += _calc_size(net)
        else:
            num_params = _calc_size(self.net)
        return num_params

    def _train_combined_net(self):
        for idx, (xbatch, ybatch,_) in enumerate(self.training_loader):
            # TODO: add handling for num_tables
            pred = self.net(xbatch).squeeze(1)
            losses = self.loss(pred, ybatch)
            loss = losses.sum() / len(losses)
            # if idx % 10 == 0:
                # print(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()

    def _train_mscn(self):
        for idx, (tbatch, pbatch, jbatch, ybatch,_) in enumerate(self.training_loader):
            # TODO: add handling for num_tables
            pred = self.net(tbatch,pbatch,jbatch).squeeze(1)
            losses = self.loss(pred, ybatch)
            loss = losses.sum() / len(losses)
            # if idx % 10 == 0:
                # print(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()

    def train_one_epoch(self):
        if self.featurization_scheme == "combined":
            self._train_combined_net()
        elif self.featurization_scheme == "mscn":
            self._train_mscn()
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

    def periodic_eval(self, samples_type):
        pred, Y = self.eval_samples(samples_type)
        losses = self.loss(pred, Y).detach().numpy()
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
            for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
                num_tables = len(node)
                idx = query_idx + subq_idx
                loss = losses[idx]
                summary_data["loss"].append(loss)
                summary_data["num_tables"].append(num_tables)
                summary_data["template"].append(template)
            query_idx += len(sample["subset_graph"].nodes())

        df = pd.DataFrame(summary_data)
        for template in set(df["template"]):
            tvals = df[df["template"] == template]
            self.add_row(tvals["loss"].values, "qerr", self.epoch,
                    template, "all", samples_type)
            for nt in set(tvals["num_tables"]):
                nt_losses = tvals[tvals["num_tables"] == nt]
                self.add_row(nt_losses["loss"].values, "qerr", self.epoch, template,
                        str(nt), samples_type)

        for nt in set(df["num_tables"]):
            nt_losses = df[df["num_tables"] == nt]
            self.add_row(nt_losses["loss"].values, "qerr", self.epoch, "all",
                    str(nt), samples_type)

        if (self.epoch % self.eval_epoch_jerr == 0 \
                and self.epoch != 0):
            if (samples_type == "train" and \
                    self.sampling_priority_alpha > 0):
                print("not recalculating join loss for training")
                return
            # if priority on, then stats will be saved when calculating
            # priority
            jl_eval_start = time.time()
            assert self.jl_use_postgres

            # TODO: do we need this awkward loop. decompose?
            sqls, true_cardinalities, est_cardinalities = \
                    self.get_query_estimates(pred, samples)
            (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                    true_cardinalities, est_cardinalities, self.env,
                    self.jl_indexes, None,
                    pool = self.join_loss_pool)

            join_losses = np.array(est_costs) - np.array(opt_costs)
            # join_losses = np.maximum(join_losses, 0.00)

            self.save_join_loss_stats(join_losses, est_plans, samples,
                    samples_type)

            # TODO: what to do with prioritization?

    def _normalize_priorities(self, priorities):
        total = np.float64(np.sum(priorities))
        norm_priorities = np.zeros(len(priorities))
        # for i, priority in enumerate(priorities):
            # norm_priorities[i] = priority / total
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

        NUM_LAST = 4
        if self.avg_jl_priority:
            self.past_priorities.append(priorities)
            if len(self.past_priorities) > 1:
                new_priorities = np.zeros(len(priorities))
                num_past = min(NUM_LAST, len(self.past_priorities))
                for i in range(1,num_past+1):
                    new_priorities += self.past_priorities[-i]
                priorities = self._normalize_priorities(new_priorities)

        return priorities

    def get_query_estimates(self, pred, samples):
        '''
        @ret:
        '''
        if not isinstance(pred, np.ndarray):
            pred = pred.detach().numpy()
        sqls = []
        true_cardinalities = []
        est_cardinalities = []
        query_idx = 0
        for sample in samples:
            sqls.append(sample["sql"])
            ests = {}
            trues = {}
            for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
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
                # ests[alias_key] = int(est_card)
                ests[alias_key] = est_card
                trues[alias_key] = cards["actual"]
            est_cardinalities.append(ests)
            true_cardinalities.append(trues)
            query_idx += len(sample["subset_graph"].nodes())

        return sqls, true_cardinalities, est_cardinalities

    def initialize_tfboard(self):
        name = self.get_exp_name()
        # name = self.__str__()
        log_dir = "tfboard_logs/" + name
        self.tf_summary_writer = tf_summary.create_file_writer(log_dir)
        self.tf_stat_fmt = "{samples_type}-{loss_type}-nt:{num_tables}-tmp:{template}"

    def train(self, db, training_samples, use_subqueries=False,
            test_samples=None, join_loss_pool = None):
        assert isinstance(training_samples[0], dict)
        if not self.nn_type == "num_tables":
            self.num_threads = multiprocessing.cpu_count()
        else:
            self.num_threads = multiprocessing.cpu_count(epoch)

        print("nn train, pool: ", join_loss_pool)
        self.join_loss_pool = join_loss_pool

        if self.tfboard:
            self.initialize_tfboard()
        print("setting num threads to: ", self.num_threads)
        torch.set_num_threads(self.num_threads)
        self.db = db
        db.init_featurizer(num_tables_feature = self.num_tables_feature,
                max_discrete_featurizing_buckets =
                self.max_discrete_featurizing_buckets,
                heuristic_features = self.heuristic_features)

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
        self.env = park.make('query_optimizer')
        training_set = QueryDataset(training_samples, db,
                self.featurization_scheme, self.heuristic_features,
                self.preload_features, self.normalization_type,
                min_val = self.min_val,
                max_val = self.max_val)
        self.training_samples = training_samples
        if self.featurization_scheme == "combined":
            self.num_features = len(training_set[0][0])
        else:
            self.num_features = len(training_set[0][0]) + \
                    len(training_set[0][1]) + len(training_set[0][2])

        # TODO: only for priority case, this should be updated after every
        # epoch
        if self.sampling_priority_alpha > 0.00:
            # start with uniform weight
            weight = 1 / len(training_set)
            weights = torch.DoubleTensor([weight]*len(training_set))
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                    num_samples=len(weights))
            self.training_loader = data.DataLoader(training_set,
                    batch_size=self.mb_size, shuffle=False, num_workers=0,
                    sampler = sampler)
            priority_loader = data.DataLoader(training_set,
                    batch_size=25000, shuffle=False, num_workers=0)
        else:
            self.training_loader = data.DataLoader(training_set,
                    batch_size=self.mb_size, shuffle=True, num_workers=0)

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
        eval_train_set = QueryDataset(eval_training_samples, db,
                self.featurization_scheme, self.heuristic_features,
                self.preload_features, self.normalization_type,
                min_val = self.min_val,
                max_val = self.max_val)
        eval_train_loader = data.DataLoader(eval_train_set,
                batch_size=10000, shuffle=False,num_workers=0)
        self.eval_loaders["train"] = eval_train_loader

        # TODO: add separate dataset, dataloaders for evaluation
        if test_samples is not None and len(test_samples) > 0:
            test_samples = random.sample(test_samples, int(len(test_samples) /
                    eval_samples_size_divider))
            self.samples["test"] = test_samples
            # TODO: add test dataloader
            test_set = QueryDataset(test_samples, db,
                    self.featurization_scheme, self.heuristic_features,
                    self.preload_features, self.normalization_type,
                    min_val = self.min_val,
                    max_val = self.max_val)
            eval_test_loader = data.DataLoader(test_set,
                    batch_size=10000, shuffle=False,num_workers=0)
            self.eval_loaders["test"] = eval_test_loader
        else:
            self.samples["test"] = None

        # TODO: initialize self.num_features
        self.init_nets(training_set[0])
        model_size = self.num_parameters()
        print("""training samples: {}, feature length: {}, model size: {},
        max_discrete_buckets: {}, hidden_layer_size: {}""".\
                format(len(training_set), self.num_features, model_size,
                    self.max_discrete_featurizing_buckets,
                    self.hidden_layer_size))

        for self.epoch in range(self.max_epochs):
            start = time.time()
            if self.epoch % self.eval_epoch == 0:
                eval_start = time.time()
                self.periodic_eval("train")
                if self.samples["test"] is not None:
                    self.periodic_eval("test")
                self.save_stats()

            self.train_one_epoch()
            print("train epoch took: ", time.time() - start)

            if self.sampling_priority_alpha > 0 \
                    and self.epoch % self.reprioritize_epoch == 0:
                pred, _ = self._eval_samples(priority_loader)
                pred = pred.detach().numpy()
                if self.sampling_priority_type == "query":
                    # TODO: decompose
                    pr_start = time.time()
                    sqls, true_cardinalities, est_cardinalities = \
                            self.get_query_estimates(pred,
                                    self.training_samples)
                    (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                            true_cardinalities, est_cardinalities, self.env,
                            self.jl_indexes, None,
                            pool = self.join_loss_pool)

                    jerr_ratio = est_costs / opt_costs
                    jerr = est_costs - opt_costs
                    self.save_join_loss_stats(jerr, est_plans,
                            self.training_samples, "train")

                    print("epoch: {}, jerr_ratio: {}, jerr: {}, time: {}"\
                            .format(self.epoch,
                                np.round(np.mean(jerr_ratio), 2),
                                np.round(np.mean(jerr), 2),
                                time.time()-pr_start))
                    weights = np.zeros(len(training_set))
                    assert len(weights) == len(training_set)
                    query_idx = 0
                    for si, sample in enumerate(self.training_samples):
                        if self.priority_err_type == "jerr":
                            sq_weight = jerr[si] / 1000000.00
                        else:
                            sq_weight = jerr_ratio[si]

                        if self.priority_err_divide_len:
                            sq_weight /= len(sample["subset_graph"].nodes())

                        for subq_idx, _ in enumerate(sample["subset_graph"].nodes()):
                            weights[query_idx+subq_idx] = sq_weight
                        query_idx += len(sample["subset_graph"].nodes())

                elif self.sampling_priority_type == "subquery":
                    pr_start = time.time()
                    sqls, true_cardinalities, est_cardinalities = \
                            self.get_query_estimates(pred,
                                    self.training_samples)
                    (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                            true_cardinalities, est_cardinalities, self.env,
                            self.jl_indexes, None,
                            pool = self.join_loss_pool)
                    jerrs = est_costs - opt_costs
                    compute_subquery_priorities(self.training_samples[0], true_cardinalities[0],
                            est_cardinalities[0], est_plans[0], jerrs[0], self.env,
                            self.jl_indexes)

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

        # if self.preload_features:
            # del(training_set.X)
            # del(training_set.Y)
            # del(training_set.info)

    def test(self, test_samples):
        '''
        @test_samples: [] sql_representation dicts
        '''
        dataset = QueryDataset(test_samples, self.db,
                self.featurization_scheme, self.heuristic_features,
                self.preload_features, self.normalization_type,
                min_val = self.min_val,
                max_val = self.max_val)
        loader = data.DataLoader(dataset,
                batch_size=1000, shuffle=False,num_workers=0)
        pred, y = self._eval_samples(loader)
        if self.preload_features:
            del(dataset.X)
            del(dataset.Y)
            del(dataset.info)
        # loss = self.loss(pred, y).detach().numpy()
        pred = pred.detach().numpy()

        all_ests = []
        query_idx = 0
        for sample in test_samples:
            ests = {}
            for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
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
        assert query_idx == len(dataset)
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

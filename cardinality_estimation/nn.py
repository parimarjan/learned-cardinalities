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
from torch.multiprocessing import Pool as Pool2
import torch.multiprocessing as mp
# from utils.tf_summaries import TensorboardSummaries
from tensorflow import summary as tf_summary

try:
    mp.set_start_method("spawn")
except:
    pass

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
        if self.nn_type == "microsoft":
            self.mb_size = 2500
        elif self.nn_type == "num_tables":
            self.mb_size = 250
        else:
            assert False

        if self.loss_func == "qloss":
            self.loss = qloss_torch
        elif self.loss_func == "rel":
            self.loss = rel_loss_torch
        elif self.loss_func == "weighted":
            self.loss = weighted_loss
        else:
            assert False

        self.net = None
        self.optimizer = None
        self.scheduler = None

        # each element is a list of priorities
        self.past_priorities = []

        # number of processes used for computing train and test join losses
        # using park envs. These are computed simultaneously, while the next
        # iterations of the neural net train.
        self.num_join_loss_processes = multiprocessing.cpu_count()

        nn_results_dir = self.nn_results_dir

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

    def _map_num_tables(self, num_tables):

        if self.group_models >= 0:
            if num_tables >= 12:
                tables = 12
            else:
                tables = num_tables
        else:
            tables = num_tables

        if self.group_models == 1:
            # so 1 and 2 get mapped to 1
            tables += 1
            tables = int((tables / 2))
            return tables
        elif self.group_models == 2:
            if tables <= 2:
                return 1
            else:
                return 2
        elif self.group_models == 3:
            # return true values for all tables except the middle ones
            if tables in [5,6,7,8,9,10]:
                # should start with 1
                return tables - 4
            else:
                return -1

        elif self.group_models < 0:
            if tables <= abs(self.group_models):
                return -1
            else:
                return 1
        else:
            return tables

    def _init_net(self, net_name, optimizer_name):
        num_features = self.num_features
        if net_name == "FCNN":
            # do training
            net = SimpleRegression(num_features,
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers,
                    hidden_layer_size=self.hidden_layer_size)
        elif net_name == "LinearRegression":
            net = LinearRegression(num_features,
                    1)
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
                            verbose=True, factor=0.1, eps=0.0001)
        else:
            scheduler = None

        return net, optimizer, scheduler

    def init_nets(self):
        # TODO: num_tables version, need have multiple neural nets
        if self.nn_type == "num_tables":
            self.nets = {}
            self.optimizers = {}
            self.schedulers = {}
            for num_table in self.train_num_table_mapping:
                num_table = self._map_num_tables(num_table)
                if num_table not in self.nets:
                    net, opt, scheduler = self._init_net(self.net_name, self.optimizer_name)
                    self.nets[num_table] = net
                    self.optimizers[num_table] = opt
                    self.schedulers[num_table] = scheduler
            print("initialized {} nets for num_tables version".format(len(self.nets)))
        else:
            self.net, self.optimizer, self.scheduler = \
                    self._init_net(self.net_name, self.optimizer_name)

    def _eval_samples(self, loader):
        all_preds = []
        all_y = []
        for idx, (xbatch, ybatch) in enumerate(loader):
            pred = self.net(xbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)
        pred = torch.cat(all_preds)
        y = torch.cat(all_y)
        return pred,y

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
            if self.summary_types[i] == "mean":
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
        name = "{DAY}-{NN_TYPE}-{PRIORITY}-{HASH}".format(\
                    DAY = self.start_day,
                    NN_TYPE = self.nn_type,
                    PRIORITY = self.sampling_priority_alpha,
                    HASH = time_hash)
        return name

    def save_stats(self):
        '''
        replaces the results file.
        '''
        # TODO: maybe reset cur_stats
        self.stats = pd.DataFrame(self.cur_stats)
        if not os.path.exists(self.nn_results_dir):
            make_dir(self.nn_results_dir)
        exp_name = self.get_exp_name()
        fn = self.nn_results_dir + "/" + exp_name + ".pkl"
        results = {}
        results["stats"] = self.stats
        results["config"] = self.kwargs
        results["name"] = self.__str__()

        with open(fn, 'wb') as fp:
            pickle.dump(results, fp,
                    protocol=pickle.HIGHEST_PROTOCOL)

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6

        if self.nn_type == "microsoft":
            num_params = _calc_size(self.net)
        elif self.nn_type == "num_tables":
            num_params = 0
            for _,net in self.nets.items():
                num_params += _calc_size(net)
        return num_params

    def train_one_epoch(self):
        for idx, (xbatch, ybatch) in enumerate(self.training_loader):
            # TODO: add handling for num_tables
            pred = self.net(xbatch).squeeze(1)
            losses = self.loss(pred, ybatch)
            loss = losses.sum() / len(losses)
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()

    def periodic_eval(self, samples_type):
        pred, Y = self.eval_samples(samples_type)
        losses = self.loss(pred, Y).detach().numpy()
        loss_avg = round(np.sum(losses) / len(losses), 2)
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
            jl_eval_start = time.time()
            assert self.jl_use_postgres

            # TODO: do we need this awkward loop. decompose?
            sqls, true_cardinalities, est_cardinalities = \
                    self.get_query_estimates(pred, samples)
            (est_costs, opt_costs,_,_,_,_) = join_loss_pg(sqls,
                    true_cardinalities, est_cardinalities, self.env, None,
                    self.num_join_loss_processes)

            join_losses = np.array(est_costs) - np.array(opt_costs)
            join_losses = np.maximum(join_losses, 0.00)

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

        AVG_PRIORITIES = False
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
                est_sel = pred[idx]
                est_card = est_sel*cards["total"]
                ests[alias_key] = int(est_card)
                trues[alias_key] = cards["actual"]
            est_cardinalities.append(ests)
            true_cardinalities.append(trues)
            query_idx += len(sample["subset_graph"].nodes())

        return sqls, true_cardinalities, est_cardinalities

    def initialize_tfboard(self):
        exp_name = self.get_exp_name()
        log_dir = "tfboard_logs/" + exp_name
        self.tf_summary_writer = tf_summary.create_file_writer(log_dir)
        self.tf_stat_fmt = "{samples_type}-{loss_type}-nt:{num_tables}-tmp:{template}"

    def train(self, db, training_samples, use_subqueries=False,
            test_samples=None):
        assert isinstance(training_samples[0], dict)
        if not self.nn_type == "num_tables":
            self.num_threads = multiprocessing.cpu_count()
            # torch.set_num_threads(self.num_threads)
        else:
            # self.num_threads = -1
            self.num_threads = multiprocessing.cpu_count(epoch)

        self.initialize_tfboard()
        print("setting num threads to: ", self.num_threads)
        torch.set_num_threads(self.num_threads)
        self.db = db
        db.init_featurizer(num_tables_feature = self.num_tables_feature,
                max_discrete_featurizing_buckets =
                self.max_discrete_featurizing_buckets)
        # create a new park env, and close at the end.
        self.env = park.make('query_optimizer')
        training_set = QueryDataset(training_samples, db)
        self.training_samples = training_samples
        self.num_features = len(training_set[0][0])

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
        random.seed(1234)
        eval_training_samples = random.sample(training_samples,
                int(len(training_samples) / 5))
        self.samples["train"] = eval_training_samples
        eval_train_set = QueryDataset(eval_training_samples, db)
        eval_train_loader = data.DataLoader(eval_train_set,
                batch_size=len(training_set), shuffle=False,num_workers=0)
        self.eval_loaders["train"] = eval_train_loader

        # TODO: add separate dataset, dataloaders for evaluation
        if test_samples is not None and len(test_samples) > 0:
            test_samples = random.sample(test_samples, int(len(test_samples) /
                    5))
            self.samples["test"] = test_samples
            # TODO: add test dataloader
            test_set = QueryDataset(test_samples, db)
            eval_test_loader = data.DataLoader(test_set,
                    batch_size=len(test_set), shuffle=False,num_workers=0)
            self.eval_loaders["test"] = eval_test_loader
        else:
            self.samples["test"] = None

        # TODO: initialize self.num_features
        self.init_nets()
        model_size = self.num_parameters()
        print("""training samples: {}, feature length: {}, model size: {},
        max_discrete_buckets: {}, hidden_layer_size: {}""".\
                format(len(training_set), self.num_features, model_size,
                    self.max_discrete_featurizing_buckets,
                    self.hidden_layer_size))

        for self.epoch in range(self.max_epochs):
            if self.epoch % self.eval_epoch == 0:
                eval_start = time.time()
                self.periodic_eval("train")
                if self.samples["test"] is not None:
                    self.periodic_eval("test")
                self.save_stats()
                print("eval time: ", time.time()-eval_start)

                # print summaries
                # cur_df = self.stats[self.stats["epoch"] == self.epoch]
                # cur_df = self.stats[self.stats["summary_type"] == "mean"]
                # train_df = self.stats[self.stats["samples_type"] == "train"]
                # print(cur_df)
                # print("""epoch: {}, train_qerr: {}, test_qerr: {},
                # train_jerr: {}, test_jerr: {}""".format(
                    # self.epoch, 0, 0, 0, 9))

            epoch_start = time.time()
            self.train_one_epoch()

            if self.sampling_priority_alpha > 0 \
                    and self.epoch % self.reprioritize_epoch == 0:
                if self.sampling_priority_type == "query":
                    # TODO: decompose
                    pr_start = time.time()
                    pred, _ = self._eval_samples(priority_loader)
                    sqls, true_cardinalities, est_cardinalities = \
                            self.get_query_estimates(pred,
                                    self.training_samples)
                    (est_costs, opt_costs,_,_,_,_) = join_loss_pg(sqls,
                            true_cardinalities, est_cardinalities, self.env, None,
                            self.num_join_loss_processes)
                    jerr_ratio = est_costs / opt_costs
                    jerr = est_costs - opt_costs
                    print("epoch: {}, jerr_ratio: {}, jerr: {}, time: {}"\
                            .format(self.epoch,
                                np.round(np.mean(jerr_ratio), 2),
                                np.round(np.mean(jerr), 2),
                                time.time()-pr_start))
                    weights = np.zeros(len(training_set))
                    assert len(weights) == len(training_set)
                    query_idx = 0
                    for si, sample in enumerate(self.training_samples):
                        sq_weight = float(jerr_ratio[si])
                        for subq_idx, _ in enumerate(sample["subset_graph"].nodes()):
                            weights[query_idx+subq_idx] = sq_weight
                        query_idx += len(sample["subset_graph"].nodes())

                    weights = self._update_sampling_weights(weights)
                    weights = torch.DoubleTensor(weights)
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                            num_samples=len(weights))
                    self.training_loader = data.DataLoader(training_set,
                            batch_size=self.mb_size, shuffle=False, num_workers=0,
                            sampler = sampler)
                else:
                    assert False


    def test(self, test_samples):
        dataset = QueryDataset(test_samples, self.db)
        loader = data.DataLoader(dataset,
                batch_size=len(dataset), shuffle=False,num_workers=0)
        pred, _ = self._eval_samples(loader)
        pred = pred.detach().numpy()

        all_ests = []
        query_idx = 0
        for sample in test_samples:
            ests = {}
            for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = node
                idx = query_idx + subq_idx
                est_sel = pred[idx]
                est_card = est_sel*cards["total"]
                ests[alias_key] = int(est_card)
            all_ests.append(ests)
            query_idx += len(sample["subset_graph"].nodes())

        return all_ests

    def __str__(self):
        if self.nn_type == "microsoft":
            name = "msft"
        elif self.nn_type == "num_tables":
            name = "nt"
        else:
            name = self.__class__.__name__

        if self.max_discrete_featurizing_buckets:
            name += "-df:" + str(self.max_discrete_featurizing_buckets)
        if self.sampling_priority_alpha > 0.00:
            name += "-pr:" + str(self.sampling_priority_alpha)
        if self.hidden_layer_size:
            name += "-hls:" + str(self.hidden_layer_size)
        if self.sampling_priority_type != "query":
            name += "-spt:" + self.sampling_priority_type

        return name

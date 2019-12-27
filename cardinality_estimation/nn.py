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
# try:
    # mp.set_start_method("spawn")
# except:
    # pass

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
        if self.max_discrete_featurizing_buckets > 10:
            self.num_join_loss_processes = 4
        else:
            self.num_join_loss_processes = 8

        # TODO: right time to close these, at the end of all jobs
        self.train_join_loss_pool = multiprocessing.pool.ThreadPool()
        self.test_join_loss_pool = multiprocessing.pool.ThreadPool()
        # will be returned by pool.map_async
        self.train_join_results = None
        self.test_join_results = None

        nn_results_dir = self.nn_results_dir

        # will keep storing all the training information in the cache / and
        # dump it at the end
        # self.key = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # We want to only summarize and store the statistics we care about
        # header info:
        #   iter: every eval_iter
        #   loss_type: qerr, join-loss etc.
        #   summary_type: mean, max, min, percentiles: 50,75th,90th,99th,25th
        #   template: all, OR only for specific template
        #   num_tables: all, OR t1,t2 etc.
        #   num_samples: in the given class, whose stats are being summarized.
        self.stats = defaultdict(list)
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

    def eval_samples(self, X, nt_map):
        if self.nn_type == "num_tables":
            pass
            # assert self.net is None
            # assert self.optimizer is None
            # all_preds = []
            # for nt in nt_map:
                # start,end = nt_map[nt]
                # Xcur = X[start:end]
                # net_map = self._map_num_tables(nt)
                # pred = self.nets[net_map](Xcur).squeeze(1)
                # all_preds.append(pred)
            # pred = torch.cat(all_preds)
        else:
            pred = self.net(X).squeeze(1)
        return pred

    def add_row(self, losses, loss_type, num_iter, template,
            num_tables, samples_type):
        for i, func in enumerate(self.summary_funcs):
            loss = func(losses)
            row = [num_iter, loss_type, loss, self.summary_types[i],
                    template, num_tables, len(losses)]
            self.stats["num_iter"].append(num_iter)
            self.stats["loss_type"].append(loss_type)
            self.stats["loss"].append(loss)
            self.stats["summary_type"].append(self.summary_types[i])
            self.stats["template"].append(template)
            self.stats["num_tables"].append(num_tables)
            self.stats["num_samples"].append(len(losses))
            self.stats["samples_type"].append(samples_type)

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
        '''
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
            loss = self.loss(pred, ybatch)
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()

    def train(self, db, training_samples, use_subqueries=False,
            test_samples=None):
        assert isinstance(training_samples[0], dict)
        if not self.nn_type == "num_tables":
            self.num_threads = multiprocessing.cpu_count()
            print("setting num threads to: ", self.num_threads)
            torch.set_num_threads(self.num_threads)
        else:
            self.num_threads = -1

        self.db = db
        db.init_featurizer(num_tables_feature = self.num_tables_feature,
                max_discrete_featurizing_buckets =
                self.max_discrete_featurizing_buckets)
        # create a new park env, and close at the end.
        self.env = park.make('query_optimizer')
        self.training_set = QueryDataset(training_samples, db)
        self.num_features = len(self.training_set[0][0])
        # TODO: add appropriate parameters
        self.training_loader = data.DataLoader(self.training_set,
                batch_size=self.mb_size, shuffle=False, num_workers=1)

        # TODO: add separate dataset, dataloaders for evaluation
        # if test_samples is not None and len(test_samples) > 0:
            # # TODO: add test dataloader
            # self.test_set = QueryDataset(test_samples)
            # self.test_env = park.make('query_optimizer')

        # TODO: initialize self.num_features
        self.init_nets()
        model_size = self.num_parameters()
        print("""training samples: {}, feature length: {}, model size: {},
        max_discrete_buckets: {}, hidden_layer_size: {}
                """.format(len(self.training_set), self.num_features,
                    model_size, self.max_discrete_featurizing_buckets,
                    self.hidden_layer_size))


        for epoch in range(self.max_iter):
            # TODO: do periodic_eval, re-prioritization etc.
            print("epoch: ", epoch)
            epoch_start = time.time()
            self.train_one_epoch()
            print("epoch {} took {} seconds".format(epoch,
                time.time()-epoch_start))
            self.save_stats()

    def test(self, test_samples):
        return None

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

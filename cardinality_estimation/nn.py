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
# import multiprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
from .algs import *
import sys
import gc

def single_level_net_steps(net, opt, Xcur, Ycur, num_steps,
        mb_size, loss_func, clip_gradient):
    # TODO: make sure there are no wasteful copying
    torch.set_num_threads(1)
    start = time.time()
    Xcur = to_variable(Xcur).float()
    Ycur = to_variable(Ycur).float()
    for mini_iter in range(num_steps):
        # TODO: reweight sampling weights here
        idxs = np.random.choice(list(range(len(Xcur))),
                mb_size)
        xbatch = Xcur[idxs]
        ybatch = Ycur[idxs]
        pred = net(xbatch).squeeze(1)
        assert pred.shape == ybatch.shape
        loss = loss_func(pred, ybatch)
        opt.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(net.parameters(), clip_gradient)
        opt.step()

class NN(CardinalityEstimationAlg):
    def __init__(self, *args, **kwargs):
        for k, val in kwargs.items():
            self.__setattr__(k, val)
        # initialize stats collection stuff

        # TODO: find appropriate name based on kwargs, dt etc.
        # TODO: write out everything to text logging file

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

        self.num_threads = 8

        # number of processes used for computing train and test join losses
        # using park envs. These are computed simultaneously, while the next
        # iterations of the neural net train.
        self.num_join_loss_processes = 8

        # TODO: right time to close these, at the end of all jobs
        self.train_join_loss_pool = multiprocessing.pool.ThreadPool()
        self.test_join_loss_pool = multiprocessing.pool.ThreadPool()
        self.train_join_results = None
        self.test_join_results = None

        nn_cache_dir = self.nn_cache_dir

        # caching related stuff
        self.training_cache = klepto.archives.dir_archive(nn_cache_dir,
                cached=True, serialized=True)
        # will keep storing all the training information in the cache / and
        # dump it at the end
        # self.key = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        dt = datetime.datetime.now()
        self.key = "{}-{}-{}-{}".format(dt.day, dt.hour, dt.minute, dt.second)
        self.key += "-" + str(deterministic_hash(str(kwargs)))[0:6]

        self.stats = {}
        self.training_cache[self.key] = self.stats

        # all the configuration parameters are specified here
        self.stats["kwargs"] = kwargs
        self.stats["name"] = self.__str__()

        # iteration : value
        self.stats["gradients"] = {}
        self.stats["lr"] = {}

        # iteration : value + additional stuff, like query-string : sql
        self.stats["mb-loss"] = {}

        # iteration: qerr: val, jloss: val
        self.stats["train"] = {}
        self.stats["test"] = {}

        self.stats["train"]["eval"] = {}
        self.stats["train"]["eval"]["qerr"] = {}
        self.stats["train"]["eval"]["join-loss"] = {}
        self.stats["train"]["eval"]["est_jl"] = {}
        self.stats["train"]["eval"]["opt_jl"] = {}
        # self.stats["train"]["eval"]["join-loss-all"] = {}

        self.stats["test"]["eval"] = {}
        self.stats["test"]["eval"]["qerr"] = {}
        self.stats["test"]["eval"]["join-loss"] = {}
        # self.stats["test"]["eval"]["join-loss-all"] = {}
        self.stats["test"]["eval"]["est_jl"] = {}
        self.stats["test"]["eval"]["opt_jl"] = {}

        self.stats["train"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["train"]["tables_eval"]["qerr"] = {}
        self.stats["train"]["tables_eval"]["qerr-all"] = {}

        self.stats["test"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["test"]["tables_eval"]["qerr"] = {}
        self.stats["test"]["tables_eval"]["qerr-all"] = {}

        # TODO: store these
        self.stats["model_params"] = {}

        self.stats["est_plans"] = {}

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
        num_features = len(self.Xtrain[0])
        if net_name == "FCNN":
            # do training
            net = SimpleRegression(num_features,
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers)
        elif net_name == "LinearRegression":
            net = LinearRegression(num_features,
                    1)
        elif net_name == "Hydra":
            net = Hydra(num_features,
                    self.hidden_layer_multiple, 1,
                    len(db.aliases), False)
        elif net_name == "FatHydra":
            net = FatHydra(num_features,
                    self.hidden_layer_multiple, 1,
                    len(db.aliases))
        elif net_name == "HydraLinear":
            net = Hydra(num_features,
                    self.hidden_layer_multiple, 1,
                    len(db.aliases), True)
        else:
            assert False

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
            print("initialized {} nets".format(len(self.nets)))
        else:
            self.net, self.optimizer, self.scheduler = \
                    self._init_net(self.net_name, self.optimizer_name)

    def _get_feature_vectors(self, samples, torch=True):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.

        @ret:
            X: feature vector for every query and its subquery. sorted by
            num_tables in each query: (0:N-1 --> 1 table, N:N2 --> 2 table, and
            so on)
            Y: corresponding selectivities.
            num_table_mapping: num_tables : (start, end)
        '''
        X = []
        Y = []
        num_table_mapping = {}

        # num_table: list
        Xtmps = defaultdict(list)
        Ytmps = defaultdict(list)
        for i, sample in enumerate(samples):
            # FIXME: do more efficient way without converting to Query
            query = convert_sql_rep_to_query_rep(sample)
            for subq in query.subqueries:
                features = self.db.get_features(subq)
                aliases = tuple(sorted(subq.aliases))
                assert aliases in sample["subset_graph"].nodes()
                Xtmps[len(subq.aliases)].append((i, aliases,
                        features))
                Ytmps[len(subq.aliases)].append(subq.true_sel)

        for i in range(20):
            if i not in Xtmps:
                continue
            # start:end
            num_table_mapping[i] = (len(X), len(X)+len(Xtmps[i]))
            for xi, x in enumerate(Xtmps[i]):
                samples[x[0]]["subset_graph"].nodes()[x[1]]["idx"] = len(X)
                X.append(x[2])
                Y.append(Ytmps[i][xi])
            del(Xtmps[i])
            del(Ytmps[i])

        assert len(X) == len(Y)
        print("feature vectors created!")
        # update the actual Xs, Ys, and mappings
        if torch:
            Xtrain = to_variable(X).float()
            Ytrain = to_variable(Y).float()
            del(X)
            del(Y)

        return Xtrain,Ytrain,num_table_mapping

    def _update_sampling_weights(self, priorities):
        '''
        refer to prioritized action replay
        '''
        priorities = np.power(priorities, self.sampling_priority_alpha)
        total = float(np.sum(priorities))
        query_sampling_weights = np.zeros(len(priorities))
        for i, priority in enumerate(priorities):
            query_sampling_weights[i] = priority / total

        return query_sampling_weights

    def eval_samples(self, X, samples, samples_type):
        if self.nn_type == "num_tables":
            assert self.net is None
            assert self.optimizer is None
            all_preds = []
            if "train" in samples_type:
                nt_map = self.train_num_table_mapping
            else:
                nt_map = self.test_num_table_mapping

            for nt in nt_map:
                start,end = nt_map[nt]
                Xcur = X[start:end]
                net_map = self._map_num_tables(nt)
                pred = self.nets[net_map](Xcur).squeeze(1)
                all_preds.append(pred)

            pred = torch.cat(all_preds)
        else:
            pred = self.net(X).squeeze(1)
        return pred

    def _update_join_results(self, results, samples_type):
        if results is None:
            return
        jl_eval_start = time.time()
        # can be a blocking call
        (est_card_costs, opt_costs,_,_) = results.get()

        # TODO: do we need both these?
        join_losses = np.array(est_card_costs) - np.array(opt_costs)
        join_losses2 = np.array(est_card_costs) / np.array(opt_costs)
        jl1 = np.mean(join_losses)
        jl2 = np.mean(join_losses2)

        # FIXME: does this even happen?
        join_losses = np.maximum(join_losses, 0.00)

        self.stats[samples_type]["eval"]["join-loss"][self.num_iter] = jl1
        self.stats[samples_type]["eval"]["est_jl"][self.num_iter] = est_card_costs
        self.stats[samples_type]["eval"]["opt_jl"][0] = opt_costs

        # # TODO: add color to key values.
        print("""\n{}: {}, num samples: {}, jl1 {},jl2 {},time: {}""".format(
            samples_type, self.num_iter - self.eval_iter_jl, len(join_losses), jl1, jl2,
            time.time()-jl_eval_start))
        self.training_cache.dump()

        # 0 is just uniform priorities, as we had initialized
        # if self.sampling_priority_alpha > 0 and self.num_iter > 0:
            # query_sampling_weights = self._update_sampling_weights(join_losses2)
            # assert np.allclose(sum(query_sampling_weights), 1.0)
            # subquery_sampling_weights = []
            # for si, sample in enumerate(self.training_samples):
                # sq_weight = float(query_sampling_weights[si])
                # num_subq = len(sample["subset_graph"].nodes())
                # sq_weight /= num_subq
                # wts = [sq_weight]*(num_subq)
                # if not np.allclose(sum(wts), query_sampling_weights[si]):
                    # print("diff: ", sum(wts) - query_sampling_weights[si])
                    # pdb.set_trace()
                # # add lists
                # subquery_sampling_weights += wts
            # self.subquery_sampling_weights = subquery_sampling_weights

    def _periodic_eval(self, X, Y, samples, samples_type,
            join_loss_pool, env):
        pred = self.eval_samples(X, samples, samples_type)
        train_loss = self.loss(pred, Y, avg=False)
        loss_avg = train_loss.sum() / len(train_loss)
        print("""{}: {}, num samples: {}, qerr: {}""".format(
            samples_type, self.num_iter, len(X), loss_avg.item()))
        self.stats[samples_type]["eval"]["qerr"][self.num_iter] = loss_avg.item()
        if self.adaptive_lr and self.scheduler is not None:
            self.scheduler.step(loss_avg)

        # TODO: simplify this.
        if "train" in samples_type:
            nt_map = self.train_num_table_mapping
        else:
            nt_map = self.test_num_table_mapping

        for nt in nt_map:
            if nt not in self.stats[samples_type]["tables_eval"]["qerr"]:
                self.stats[samples_type]["tables_eval"]["qerr"][nt] = {}
                self.stats[samples_type]["tables_eval"]["qerr-all"][nt] = {}
            start,end = nt_map[nt]
            cur_loss = train_loss[start:end].detach().numpy()

            self.stats[samples_type]["tables_eval"]["qerr"][nt][self.num_iter] = \
                np.mean(cur_loss)
            self.stats[samples_type]["tables_eval"]["qerr-all"][nt][self.num_iter] = \
                cur_loss

        if (self.num_iter % self.eval_iter_jl == 0):
            jl_eval_start = time.time()
            assert self.jl_use_postgres

            # TODO: do we need this awkward loop. decompose?
            est_cardinalities = []
            true_cardinalities = []
            sqls = []
            for qrep in samples:
                sqls.append(qrep["sql"])
                ests = {}
                trues = {}
                for node, node_info in qrep["subset_graph"].nodes().items():
                    alias_key = ' '.join(node)
                    est_sel = pred[node_info["idx"]]
                    est_card = est_sel*node_info["cardinality"]["total"]
                    ests[alias_key] = int(est_card)
                    trues[alias_key] = node_info["cardinality"]["actual"]
                est_cardinalities.append(ests)
                true_cardinalities.append(trues)

            args = (sqls, true_cardinalities, est_cardinalities, env,
                    None, self.num_join_loss_processes)
            results = join_loss_pool.apply_async(join_loss_pg, args)
            return results
        return None

    def train_step(self, num_steps=1):
        if self.nn_type == "num_tables":
            assert self.net is None
            nt_map = self.train_num_table_mapping
            if self.single_threaded_nt:
                for nt in nt_map:
                    start,end = nt_map[nt]
                    net_map = self._map_num_tables(nt)
                    net = self.nets[net_map]
                    opt = self.optimizers[net_map]
                    Xcur = self.Xtrain[start:end]
                    Ycur = self.Ytrain[start:end]
                    for mini_iter in range(num_steps):
                        # TODO: reweight sampling weights here
                        idxs = np.random.choice(list(range(len(Xcur))),
                                self.mb_size)
                        xbatch = Xcur[idxs]
                        ybatch = Ycur[idxs]
                        pred = net(xbatch).squeeze(1)
                        assert pred.shape == ybatch.shape
                        loss = self.loss(pred, ybatch)
                        opt.zero_grad()
                        loss.backward()
                        if self.clip_gradient is not None:
                            clip_grad_norm_(net.parameters(), self.clip_gradient)
                        opt.step()
            else:
                par_args = []
                for i, nt in enumerate(nt_map):
                    start,end = nt_map[nt]
                    net_map = self._map_num_tables(nt)
                    net = self.nets[net_map]
                    opt = self.optimizers[net_map]
                    Xcur = self.Xtrain[start:end].cpu().detach().numpy()
                    Ycur = self.Ytrain[start:end].cpu().detach().numpy()
                    # TODO: make mb_size dependent on the level?
                    par_args.append((net, opt, Xcur, Ycur, num_steps,
                        self.mb_size, self.loss, self.clip_gradient))

                # launch single-threaded processes for each
                # TODO: might be better to launch pool of 4 + 2T each, so we
                # don't waste resources on levels that finish fast?
                num_processes = 4
                with Pool2(processes=num_processes) as pool:
                    pool.starmap(single_level_net_steps, par_args)
        else:
            # TODO: replace this with dataloader (...)
            for mini_iter in range(num_steps):
                # usual case
                idxs = np.random.choice(list(range(len(self.Xtrain))),
                        self.mb_size,
                        p=self.subquery_sampling_weights)

                xbatch = self.Xtrain[idxs]
                ybatch = self.Ytrain[idxs]
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

        self.db = db
        db.init_featurizer(num_tables_feature = self.num_tables_feature,
            max_discrete_featurizing_buckets = self.max_discrete_featurizing_buckets)
        # get one true source of X,Y feature vector pairs, which won't be
        # reused.
        start = time.time()
        self.training_samples = training_samples
        self.Xtrain, self.Ytrain, self.train_num_table_mapping = \
                self._get_feature_vectors(self.training_samples)
        # create a new park env, and close at the end.
        self.env = park.make('query_optimizer')

        self.test_samples = test_samples
        if test_samples is not None and len(test_samples) > 0:
            self.Xtest, self.Ytest, self.test_num_table_mapping = \
                self._get_feature_vectors(self.test_samples)
            print("{} training, {} test subqueries".format(len(self.Xtrain),
                len(self.Xtest)))
            self.test_env = park.make('query_optimizer')

        print("feature len: {}, generation time: {}".\
                format(len(self.Xtrain[0]), time.time()-start))

        # FIXME: multiple table version
        self.init_nets()
        self.num_iter = 0

        # start off uniformly
        self.subquery_sampling_weights = [1/len(self.Xtrain)]*len(self.Xtrain)

        prev_end = time.time()
        while True:
            if (self.num_iter % 100 == 0):
                # progress stuff
                it_time = time.time() - prev_end
                prev_end = time.time()
                print("MB: {}, T:{}, I:{} : {}".format(\
                        self.mb_size, self.num_threads, self.num_iter, it_time))
                sys.stdout.flush()

            if (self.num_iter % self.eval_iter == 0):
                # we will wait on these results when we reach this point in the
                # next iteration
                self._update_join_results(self.train_join_results, "train")
                self.train_join_results = self._periodic_eval(self.Xtrain, self.Ytrain,
                        self.training_samples, "train",
                        self.train_join_loss_pool, self.env)

                # TODO: handle reweighing schemes here

                if test_samples is not None:
                    self._update_join_results(self.test_join_results, "test")
                    self.test_join_results = self._periodic_eval(self.Xtest,
                            self.Ytest, self.test_samples, "test",
                            self.test_join_loss_pool, self.test_env)
                print("apply asyncs done, continuing to train...")

            if 1.00 - sum(self.subquery_sampling_weights) != 0.00:
                diff = 1.00 - sum(self.subquery_sampling_weights)
                random_idx = np.random.randint(0,len(self.subquery_sampling_weights))
                self.subquery_sampling_weights[random_idx] += diff

            self.train_step(self.eval_iter)
            self.num_iter += self.eval_iter

            if (self.num_iter >= self.max_iter):
                print("breaking because max iter done")
                break

    def test(self, test_samples):
        pass

    def __str__(self):
        cls = self.__class__.__name__
        name = cls + self.nn_type
        name += "lr-" + str(self.lr)
        name += "sp-" + str(self.sampling_priority_alpha)
        return name

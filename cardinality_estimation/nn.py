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
# from multiprocessing import Pool
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
import multiprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from .custom_linear import CustomLinearModel
from sqlalchemy import create_engine
from .algs import *
import sys
import gc

# sentinel value for NULLS
NULL_VALUE = "-1"

def get_all_num_table_queries(samples, num):
    '''
    @ret: all Query objects having @num tables
    '''
    ret = []
    for sample in samples:
        num_tables = len(sample.aliases)
        if num_tables == num:
            ret.append(sample)
        for subq in sample.subqueries:
            num_tables = len(subq.aliases)
            if num_tables == num:
                ret.append(subq)
    return ret

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

    def _init_nets(self):
        # TODO: num_tables version, need have multiple neural nets
        num_features = len(self.Xtrain[0])
        if self.net_name == "FCNN":
            # do training
            net = SimpleRegression(num_features,
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers)
            self.mb_size = 128
        elif self.net_name == "LinearRegression":
            net = LinearRegression(num_features,
                    1)
            self.mb_size = 128
        elif self.net_name == "Hydra":
            net = Hydra(num_features,
                    self.hidden_layer_multiple, 1,
                    len(db.aliases), False)
            self.mb_size = 512
        elif self.net_name == "FatHydra":
            net = FatHydra(num_features,
                    self.hidden_layer_multiple, 1,
                    len(db.aliases))
            self.mb_size = 512
            print("FatHydra created!")
        elif self.net_name == "HydraLinear":
            net = Hydra(num_features,
                    self.hidden_layer_multiple, 1,
                    len(db.aliases), True)
            self.mb_size = 512
            print("Hydra created!")
        else:
            assert False

        if self.optimizer_name == "ams":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=True)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(),
                    lr=self.lr, momentum=0.9)
        else:
            assert False

        return net, optimizer

    def _get_feature_vectors(self, samples, torch=True):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.

        @ret:
            X: feature vector for every query and its subquery. sorted by
            num_tables in each query: (0:N-1 --> 1 table, N:N2 --> 2 table, and
            so on)
            Y: corresponding selectivities.
            index_mapping: index : (query, aliases, total)
                - used to construct the true_cardinalities, est_cardinalities
                  dictionaries for computing join loss
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

    def _periodic_eval(self, X, Y, samples, samples_type):
        # TODO: do this step with multiple nets case too
        pred = self.net(X).squeeze(1)
        train_loss = self.loss(pred, Y)
        self.stats[samples_type]["eval"]["qerr"][self.num_iter] = train_loss.item()
        print("""\n{}: {}, num samples: {}, qerr: {}""".format(
            samples_type, self.num_iter, len(X), train_loss.item()))

        # if (self.num_iter % self.eval_iter_jl == 0 \
                # and self.num_iter != 0):
        if (self.num_iter % self.eval_iter_jl == 0):
            jl_eval_start = time.time()
            assert self.jl_use_postgres
            est_card_costs, opt_costs, _, _ = join_loss(pred,
                    samples, self.env, "EXHAUSTIVE", self.jl_use_postgres)

            join_losses = np.array(est_card_costs) - np.array(opt_costs)
            join_losses2 = np.array(est_card_costs) / np.array(opt_costs)

            jl1 = np.mean(join_losses)
            jl2 = np.mean(join_losses2)

            # FIXME: remove all negative values, so weighted_prob can work
            # fine. But there really shouldn't be any negative values here.
            join_losses = np.maximum(join_losses, 0.00)

            # TODO: add scheduler stuff here?

            # FIXME: better logging files.
            self.stats[samples_type]["eval"]["join-loss"][self.num_iter] = jl1
            self.stats[samples_type]["eval"]["est_jl"][self.num_iter] = est_card_costs
            self.stats[samples_type]["eval"]["opt_jl"][0] = opt_costs

            # TODO: add color to key values.
            print("""\n{}: {}, num samples: {}, loss: {}, jl1 {},jl2 {},time: {}""".format(
                samples_type, self.num_iter, len(X), train_loss.item(), jl1, jl2,
                time.time()-jl_eval_start))

            self.training_cache.dump()

            # 0 is just uniform priorities, as we had initialized
            if self.sampling_priority_alpha > 0 and self.num_iter > 0:
                query_sampling_weights = self._update_sampling_weights(join_losses2)
                assert np.allclose(sum(query_sampling_weights), 1.0)
                subquery_sampling_weights = []
                for si, sample in enumerate(self.training_samples):
                    sq_weight = float(query_sampling_weights[si])
                    num_subq = len(sample["subset_graph"].nodes())
                    sq_weight /= num_subq
                    wts = [sq_weight]*(num_subq)
                    if not np.allclose(sum(wts), query_sampling_weights[si]):
                        print("diff: ", sum(wts) - query_sampling_weights[si])
                        pdb.set_trace()
                    # add lists
                    subquery_sampling_weights += wts
                self.subquery_sampling_weights = subquery_sampling_weights

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
        self.test_samples = test_samples
        if test_samples is not None and len(test_samples) > 0:
            self.Xtest, self.Ytest, self.test_num_table_mapping = \
                self._get_feature_vectors(self.test_samples)
            print("{} training, {} test subqueries".format(len(self.Xtrain),
                len(self.Xtest)))
        print("feature len: {}, generation time: {}".\
                format(len(self.Xtrain[0]), time.time()-start))
        gc.collect()

        # FIXME: multiple table version
        self.net, self.optimizer = self._init_nets()
        self.num_iter = 0

        # start off uniformly
        self.subquery_sampling_weights = [1/len(self.Xtrain)]*len(self.Xtrain)

        # create a new park env, and close at the end.
        self.env = park.make('query_optimizer')

        while True:
            if (self.num_iter % 100 == 0):
                # progress stuff
                print(self.num_iter, end=",")
                sys.stdout.flush()

            if (self.num_iter % self.eval_iter == 0):
                self._periodic_eval(self.Xtrain, self.Ytrain,
                        self.training_samples, "train")
                if test_samples is not None:
                    self._periodic_eval(self.Xtest, self.Ytest,
                            self.test_samples, "test")

            if 1.00 - sum(self.subquery_sampling_weights) != 0.00:
                diff = 1.00 - sum(self.subquery_sampling_weights)
                random_idx = np.random.randint(0,len(self.subquery_sampling_weights))
                self.subquery_sampling_weights[random_idx] += diff

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

            self.num_iter += 1
            if (self.num_iter > self.max_iter):
                print("breaking because max iter done")
                break

    def test(self, test_samples):
        pass
        # X = []
        # for sample in test_samples:
            # X.append(self.db.get_features(sample))
        # # just pass each sample through net and done!
        # X = to_variable(X).float()
        # pred = self.net(X)
        # pred = pred.squeeze(1)
        # return pred.cpu().detach().numpy()

class NumTablesNN(CardinalityEstimationAlg):
    '''
    Will divide the queries AND subqueries based on the number of tables in it,
    and train a new neural network for each of those.

    TODO: computing join-loss for each subquery.
    '''

    # FIXME: common stuff b/w all neural network models should be decomposed
    def __init__(self, *args, **kwargs):

        self.reuse_env = kwargs["reuse_env"]
        self.models = {}
        self.optimizers = {}
        self.samples = {}
        # for all Xs, Ys from subqueries
        self.Xtrains = {}
        self.Ytrains = {}
        self.model_name = kwargs["num_tables_model"]
        # self.num_trees = kwargs["num_trees"]
        # self.eval_num_tables = kwargs["eval_num_tables"]
        self.eval_num_tables = True
        self.loss_stop_thresh = 1.00
        self.num_tables_train_qerr = {}
        self.group_models = kwargs["group_models"]
        self.jl_use_postgres = kwargs["jl_use_postgres"]
        self.loss_func = kwargs["loss_func"]

        # if kwargs["loss_func"] == "qloss":
            # self.loss_func = qloss_torch
        # else:
            # assert False

        # TODO: remove redundant crap.
        self.feature_len = None
        self.feat_type = "dict_encoding"

        # TODO: configure other variables
        self.max_iter = kwargs["max_iter"]
        self.jl_variant = kwargs["jl_variant"]
        if not self.jl_variant:
            # because we eval more frequently
            self.adaptive_lr_patience = 100
        else:
            self.adaptive_lr_patience = 5

        self.divide_mb_len = kwargs["divide_mb_len"]
        self.lr = kwargs["lr"]
        self.jl_start_iter = kwargs["jl_start_iter"]
        self.num_hidden_layers = kwargs["num_hidden_layers"]
        self.hidden_layer_multiple = kwargs["hidden_layer_multiple"]
        self.eval_iter = kwargs["eval_iter"]
        self.eval_iter_jl = kwargs["eval_iter_jl"]
        self.optimizer_name = kwargs["optimizer_name"]

        self.clip_gradient = kwargs["clip_gradient"]
        self.rel_qerr_loss = kwargs["rel_qerr_loss"]
        self.rel_jloss = kwargs["rel_jloss"]
        self.adaptive_lr = kwargs["adaptive_lr"]
        self.baseline = kwargs["baseline"]
        self.sampling = kwargs["sampling"]
        self.sampling_priority_method = kwargs["sampling_priority_method"]
        self.adaptive_priority_alpha = kwargs["adaptive_priority_alpha"]
        self.sampling_priority_alpha = kwargs["sampling_priority_alpha"]
        self.net_name = kwargs["net_name"]

        nn_cache_dir = kwargs["nn_cache_dir"]

        # caching related stuff
        self.training_cache = klepto.archives.dir_archive(nn_cache_dir,
                cached=True, serialized=True)
        # will keep storing all the training information in the cache / and
        # dump it at the end
        # self.key = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        dt = datetime.datetime.now()
        self.key = "{}-{}-{}-{}".format(dt.day, dt.hour, dt.minute, dt.second)
        self.key += "-" + str(deterministic_hash(str(kwargs)))[0:6]
        self.key += "gm-" + str(self.group_models)

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
        self.stats["train"]["eval"]["join-loss-all"] = {}

        self.stats["test"]["eval"] = {}
        self.stats["test"]["eval"]["qerr"] = {}
        self.stats["test"]["eval"]["join-loss"] = {}
        self.stats["test"]["eval"]["join-loss-all"] = {}

        self.stats["train"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["train"]["tables_eval"]["qerr"] = {}
        self.stats["train"]["tables_eval"]["qerr-all"] = {}

        self.stats["test"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["test"]["tables_eval"]["qerr"] = {}
        self.stats["test"]["tables_eval"]["qerr-all"] = {}

        self.stats["model_params"] = {}

    def map_num_tables(self, num_tables):

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

    def _get_loss_func(self, num_table, torch_version=True):
        if self.loss_func == "qloss":
            if torch_version:
                return qloss_torch
            else:
                return qloss
        elif self.loss_func == "mixed1":
            if num_table <= 4:
                if torch_version:
                    return qloss_torch
                else:
                    return qloss
            else:
                if torch_version:
                    return rel_loss_torch
                else:
                    return rel_loss
        elif self.loss_func == "mixed2":
            if num_table <= 4:
                if torch_version:
                    return qloss_torch
                else:
                    return qloss
            else:
                if torch_version:
                    return abs_loss_torch
                else:
                    return abs_loss
        else:
            assert False

    # same function for all the nns
    def _periodic_num_table_eval_nets(self, num_iter):
        for num_table in self.samples:
            x_table = self.table_x_train[num_table]
            y_table = self.table_y_train[num_table]
            if len(x_table) == 0:
                continue

            # if num_table in self.num_tables_train_qerr:
                # if self.num_tables_train_qerr[num_table] < self.loss_stop_thresh:
                    # continue

            net = self.models[num_table]
            pred_table = net(x_table)
            pred_table = pred_table.squeeze(1)
            pred_table = pred_table.data.numpy()

            loss_trains = self._get_loss_func(num_table, torch_version=False) \
                            (pred_table, y_table, avg=False)

            if num_table not in self.stats["train"]["tables_eval"]["qerr"]:
                self.stats["train"]["tables_eval"]["qerr"][num_table] = {}
                self.stats["train"]["tables_eval"]["qerr-all"][num_table] = {}

            self.stats["train"]["tables_eval"]["qerr"][num_table][num_iter] = \
                np.mean(loss_trains)
            self.stats["train"]["tables_eval"]["qerr-all"][num_table][num_iter] = \
                loss_trains

            self.num_tables_train_qerr[num_table] = np.mean(loss_trains)

            # do for test as well
            if num_table not in self.table_x_test:
                continue
            x_table = self.table_x_test[num_table]
            y_table = self.table_y_test[num_table]
            pred_table = net(x_table)
            pred_table = pred_table.squeeze(1)
            pred_table = pred_table.data.numpy()
            loss_test = self._get_loss_func(num_table, torch_version=False) \
                        (pred_table, y_table, avg=False)
            if num_table not in self.stats["test"]["tables_eval"]["qerr"]:
                self.stats["test"]["tables_eval"]["qerr"][num_table] = {}
                self.stats["test"]["tables_eval"]["qerr-all"][num_table] = {}

            self.stats["test"]["tables_eval"]["qerr"][num_table][num_iter] = \
                np.mean(loss_test)
            self.stats["test"]["tables_eval"]["qerr-all"][num_table][num_iter] = \
                loss_test

            print("num_tables: {}, train_qerr: {}, test_qerr: {}, size: {}".format(\
                    num_table, np.mean(loss_trains), np.mean(loss_test), len(y_table)))

    def _periodic_eval(self, samples, env, key,
            num_iter):
        '''
        this loss computation is not used for training, so we can just use
        qerror here.
        '''
        assert (num_iter % self.eval_iter == 0)
        Y = []
        pred = []

        # FIXME: optimize this
        # it is important to maintain the same order of traversal for the
        # join_loss compute function to work (ugh...)
        for sample in samples:
            Y.append(sample.true_sel)
            num_tables = self.map_num_tables(len(sample.froms))
            if num_tables == -1:
                # use true cardinality
                pred.append(sample.true_sel)
            else:
                pred.append(self.models[num_tables](sample.features).item())

            for subq in sample.subqueries:
                Y.append(subq.true_sel)
                num_tables = self.map_num_tables(len(subq.froms))
                if num_tables == -1:
                    pred.append(subq.true_sel)
                else:
                    pred.append(self.models[num_tables](subq.features).item())

        pred = np.array(pred)
        Y = np.array(Y)
        train_loss = qloss(pred, Y)

        self.stats[key]["eval"]["qerr"][num_iter] = train_loss

        print("""\n{}: {}, num samples: {}, loss: {}""".format(
            key, num_iter, len(Y), train_loss.item()))

        if (num_iter % self.eval_iter_jl == 0):
            jl_eval_start = time.time()
            est_card_costs, baseline_costs = join_loss(pred, samples, env,
                    "EXHAUSTIVE", self.jl_use_postgres)

            join_losses = np.array(est_card_costs) - np.array(baseline_costs)
            join_losses2 = np.array(est_card_costs) / np.array(baseline_costs)

            jl1 = np.mean(join_losses)
            jl2 = np.mean(join_losses2)

            # FIXME: remove all negative values, so weighted_prob can work
            # fine. But there really shouldn't be any negative values here.
            # join_losses = np.maximum(join_losses, 0.00)

            self.stats[key]["eval"]["join-loss"][num_iter] = jl1
            self.stats[key]["eval"]["join-loss-all"][num_iter] = join_losses

            # TODO: add color to key values.
            print("""\n{}: {}, num samples: {}, loss: {}, jl1 {},jl2 {},time: {}""".format(
                key, num_iter, len(Y), train_loss.item(), jl1, jl2,
                time.time()-jl_eval_start))

            self.training_cache.dump()
            return join_losses, join_losses2

        return None, None

    def _train_nn(self, db, training_samples, **kwargs):
        test_samples = kwargs["test_samples"]
        for sample in training_samples:
            features = db.get_features(sample)
            num_tables = self.map_num_tables(len(sample.froms))
            if num_tables != -1:
                if num_tables not in self.samples:
                    self.samples[num_tables] = []
                    self.Xtrains[num_tables] = []
                    self.Ytrains[num_tables] = []

                self.Xtrains[num_tables].append(features)
                self.Ytrains[num_tables].append(sample.true_sel)

                ## why convert to torch here and not there...
                features = to_variable(features).float()
                sample.features = features

                self.samples[num_tables].append(sample)

            for subq in sample.subqueries:
                num_tables = self.map_num_tables(len(subq.froms))
                if num_tables == -1:
                    continue
                if num_tables not in self.samples:
                    self.samples[num_tables] = []
                    self.Xtrains[num_tables] = []
                    self.Ytrains[num_tables] = []

                self.samples[num_tables].append(subq)
                subq_features = db.get_features(subq)

                self.Xtrains[num_tables].append(subq_features)
                self.Ytrains[num_tables].append(subq.true_sel)

                subq_features = to_variable(subq_features).float()
                subq.features = subq_features

        for num_tables in self.samples:
            X = self.Xtrains[num_tables]
            Y = self.Ytrains[num_tables]
            self.Xtrains[num_tables] = to_variable(X).float()
            self.Ytrains[num_tables] = to_variable(Y).float()
        # print("num tables in samples: ", len(self.samples))
        # TODO: summarize data in each table

        if test_samples:
            for sample in test_samples:
                features = db.get_features(sample)
                features = to_variable(features).float()
                sample.features = features
                for subq in sample.subqueries:
                    subq_features = db.get_features(subq)
                    subq_features = to_variable(subq_features).float()
                    subq.features = subq_features

        for num_tables in self.samples:
            print("setting up neural net for model index: ", num_tables)
            sample = self.samples[num_tables][0]
            features = db.get_features(sample)
            if self.net_name == "FCNN":
                # do training
                net = SimpleRegression(len(features),
                        self.hidden_layer_multiple, 1,
                        num_hidden_layers=self.num_hidden_layers)
            elif self.net_name == "LinearRegression":
                net = LinearRegression(len(features),
                        1)
            else:
                assert False

            self.models[num_tables] = net
            print("created net {} for {} tables".format(net, num_tables))

            if self.optimizer_name == "ams":
                optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                        amsgrad=True)
            elif self.optimizer_name == "adam":
                optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                        amsgrad=False)
            elif self.optimizer_name == "sgd":
                optimizer = torch.optim.SGD(net.parameters(), lr=self.lr,
                        momentum=0.9)
            else:
                assert False
            self.optimizers[num_tables] = optimizer

        num_iter = 0
        # create a new park env, and close at the end.
        if self.reuse_env:
            env = park.make('query_optimizer')
        else:
            env = None

        # now let us just train each of these separately. After every training
        # iteration, we will evaluate the join-loss, using ALL of them.
        # Train each net for N iterations, and then evaluate.
        start = time.time()
        try:

            while True:
                if (num_iter % 100 == 0):
                    # progress stuff
                    print(num_iter, end=",")
                    sys.stdout.flush()

                if (num_iter % self.eval_iter == 0
                        and num_iter != 0):
                # if (num_iter % self.eval_iter == 0):

                    if self.eval_num_tables:
                        self._periodic_num_table_eval_nets(num_iter)

                    # evaluation code
                    if (num_iter % self.eval_iter_jl == 0 \
                            and not self.reuse_env):
                        assert env is None
                        env = park.make('query_optimizer')

                    join_losses, join_losses_ratio = self._periodic_eval(training_samples,
                            env, "train", num_iter)
                    if test_samples:
                        self._periodic_eval(test_samples,
                                env,"test", num_iter)

                    if not self.reuse_env and env is not None:
                        env.clean()
                        env = None

                for num_tables, _ in self.samples.items():
                    # if num_tables in self.num_tables_train_qerr:
                        # if self.num_tables_train_qerr[num_tables] < self.loss_stop_thresh:
                            # # print("skipping training ", num_tables)
                            # continue

                    optimizer = self.optimizers[num_tables]
                    net = self.models[num_tables]
                    X = self.Xtrains[num_tables]
                    Y = self.Ytrains[num_tables]

                    MB_SIZE = 128
                    idxs = np.random.choice(list(range(len(X))), MB_SIZE)
                    xbatch = X[idxs]
                    ybatch = Y[idxs]

                    pred = net(xbatch)
                    pred = pred.squeeze(1)
                    loss = self._get_loss_func(num_tables)(pred, ybatch)

                    optimizer.zero_grad()
                    loss.backward()

                    if self.clip_gradient is not None:
                        clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()

                num_iter += 1
                if (num_iter > self.max_iter):
                    print("max iter done in: ", time.time() - start)
                    break

        except KeyboardInterrupt:
            print("keyboard interrupt")
        except park.envs.query_optimizer.query_optimizer.QueryOptError:
            print("park exception")

        self.training_cache.dump()

    def _train_rf(self):

        for num_tables in self.Xtrains:
            X = self.Xtrains[num_tables]
            Y = self.Ytrains[num_tables]
            # fit the model
            model = RandomForestRegressor(n_estimators=self.num_trees).fit(X, Y)

            self.models[num_tables] = model

            print("training random forest classifier done for ", num_tables)
            yhat = model.predict(X)
            train_loss = qloss(yhat, Y)
            print("train loss: ", train_loss)

    def train(self, db, training_samples, **kwargs):
        '''
        '''
        self.db = db
        self.num_tables = len(db.aliases)
        db.init_featurizer()
        test_samples = kwargs["test_samples"]
        # do common pre-processing part here
        # FIXME: decompose
        if self.eval_num_tables:
            self.table_x_train = defaultdict(list)
            self.table_x_test = defaultdict(list)
            self.table_y_train = defaultdict(list)
            self.table_y_test = defaultdict(list)
            num_real_tables = len(db.aliases)
            for i in range(1,num_real_tables+1):
                num_tables_map = self.map_num_tables(i)
                if num_tables_map == -1:
                    continue
                queries = get_all_num_table_queries(training_samples, i)
                print("mapping table {} to model index {}".format(i,
                    num_tables_map))
                for q in queries:
                    self.table_x_train[num_tables_map].append(db.get_features(q))
                    self.table_y_train[num_tables_map].append(q.true_sel)

                if test_samples:
                    queries = get_all_num_table_queries(test_samples, i)
                    for q in queries:
                        self.table_x_test[num_tables_map].append(db.get_features(q))
                        self.table_y_test[num_tables_map].append(q.true_sel)

            for i in range(len(self.table_x_train)):
                num_tables_map = i + 1  # starts from 1
                self.table_x_train[num_tables_map] = \
                    to_variable(self.table_x_train[num_tables_map]).float()
                # self.table_y_train[num_tables_map] = \
                    # to_variable(self.table_y_train[num_tables_map]).float()
                self.table_y_train[num_tables_map] = \
                    np.array(self.table_y_train[num_tables_map])
                self.table_x_test[num_tables_map] = \
                    to_variable(self.table_x_test[num_tables_map]).float()
                self.table_y_test[num_tables_map] = \
                    np.array(self.table_y_test[num_tables_map])
                # self.table_y_test[num_tables_map] = \
                    # to_variable(self.table_y_test[num_tables_map]).float()

        for sample in training_samples:
            features = db.get_features(sample)
            num_tables = self.map_num_tables(len(sample.froms))
            if num_tables == -1:
                continue
            if num_tables not in self.samples:
                self.samples[num_tables] = []
                self.Xtrains[num_tables] = []
                self.Ytrains[num_tables] = []

            self.Xtrains[num_tables].append(features)
            self.Ytrains[num_tables].append(sample.true_sel)
            for subq in sample.subqueries:
                num_tables = self.map_num_tables(len(subq.froms))
                if num_tables == -1:
                    continue
                if num_tables not in self.samples:
                    self.samples[num_tables] = []
                    self.Xtrains[num_tables] = []
                    self.Ytrains[num_tables] = []

                self.samples[num_tables].append(subq)
                subq_features = db.get_features(subq)
                self.Xtrains[num_tables].append(subq_features)
                self.Ytrains[num_tables].append(subq.true_sel)

        if self.model_name == "nn":
            self._train_nn(db, training_samples, **kwargs)
        elif self.model_name == "rf":
            self._train_rf()
        elif self.model_name == "linear":
            pdb.set_trace()

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: already includes subqueries, we just need to predict
        value for each.
        '''
        pred = []
        for sample in test_samples:
            num_tables = self.map_num_tables(len(sample.froms))
            if num_tables == -1:
                pred.append(sample.true_sel)
            else:
                pred.append(self.models[num_tables](sample.features).item())
        return pred

    def size(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        pass

    def __str__(self):
        return self.__class__.__name__
    def save_model(self, save_dir="./", suffix_name=""):
        pass

class XGBoost(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        pass

    def train(self, db, training_samples, **kwargs):
        test_samples = kwargs["test_samples"]
        X, Y = get_all_features(training_samples, db)
        Xtest, Ytest = get_all_features(test_samples, db)

        print("before training xgboost")
        gbm = xgb.XGBRegressor(max_depth=16, n_estimators=20,
                learning_rate=0.05, objective='reg:squarederror').fit(X, Y)
        print("training xgboost done!")
        yhat = gbm.predict(X)
        train_loss = qloss(yhat, Y)

        yhat = gbm.predict(Xtest)
        test_loss = qloss(yhat, Ytest)
        print("train loss: {}, test loss: {}".format(train_loss, test_loss))

        pdb.set_trace()

    def test(self, test_samples, **kwargs):
        pass
    def size(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        pass
    def __str__(self):
        return self.__class__.__name__
    def save_model(self, save_dir="./", suffix_name=""):
        pass

class RandomForest(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        # self.num_trees
        pass

    def train(self, db, training_samples, **kwargs):
        test_samples = kwargs["test_samples"]
        X, Y = get_all_features(training_samples, db)
        Xtest, Ytest = get_all_features(test_samples, db)

        model = RandomForestRegressor(n_estimators=128).fit(X, Y)
        print("training random forest classifier done")
        yhat = model.predict(X)
        train_loss = qloss(yhat, Y)

        yhat = model.predict(Xtest)
        test_loss = qloss(yhat, Ytest)
        print("train loss: {}, test loss: {}".format(train_loss, test_loss))

        pdb.set_trace()

    def test(self, test_samples, **kwargs):
        pass
    def size(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        pass
    def __str__(self):
        return self.__class__.__name__
    def save_model(self, save_dir="./", suffix_name=""):
        pass

class Linear(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        # self.num_trees
        pass

    def train(self, db, training_samples, **kwargs):
        test_samples = kwargs["test_samples"]
        X, Y = get_all_features(training_samples, db)
        Xtest, Ytest = get_all_features(test_samples, db)
        print("going to train custom linear model")
        model = CustomLinearModel(qloss, X=X, Y=Y)
        model.fit()
        print("training linear model done!")
        yhat = model.predict(X)
        train_loss = qloss(yhat, Y)

        yhat = model.predict(Xtest)
        test_loss = qloss(yhat, Ytest)
        print("train loss: {}, test loss: {}".format(train_loss, test_loss))

        pdb.set_trace()

    def test(self, test_samples, **kwargs):
        pass
    def size(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        pass
    def __str__(self):
        return self.__class__.__name__
    def save_model(self, save_dir="./", suffix_name=""):
        pass

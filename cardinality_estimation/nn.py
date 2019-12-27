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
        mb_size, loss_func, clip_gradient, sampling_weights):
    # TODO: make sure there are no wasteful copying
    torch.set_num_threads(1)
    start = time.time()
    Xcur = to_variable(Xcur).float()
    Ycur = to_variable(Ycur).float()
    for mini_iter in range(num_steps):
        # TODO: reweight sampling weights here
        idxs = np.random.choice(list(range(len(Xcur))),
                mb_size, p = sampling_weights)
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
        num_features = len(self.Xtrain[0])
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

    def _get_feature_vectors(self, samples, torch=True):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.

        @ret:
            X: feature vector for every query and its subquery. sorted by
            num_tables: (0:N-1 --> 1 table, N:N2 --> 2 table, and
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
                print("average priorities created!")
                priorities = self._normalize_priorities(new_priorities)

        return priorities

    def eval_samples(self, X, nt_map):
        if self.nn_type == "num_tables":
            assert self.net is None
            assert self.optimizer is None
            all_preds = []
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

    def _update_join_results(self, results, samples, samples_type,
            num_iter):
        if results is None:
            return
        jl_eval_start = time.time()
        # can be a blocking call
        (est_card_costs, opt_costs,_,_,_,_) = results.get()

        # TODO: do we need both these?
        join_losses = np.array(est_card_costs) - np.array(opt_costs)
        join_losses2 = np.array(est_card_costs) / np.array(opt_costs)
        join_losses = np.maximum(join_losses, 0.00)

        self.add_row(join_losses, "jerr", num_iter, "all",
                "all", samples_type)
        for sample in samples:
            template = sample["template_name"]

        jl_mean = round(np.mean(join_losses), 2)
        jl90 = round(np.percentile(join_losses, 90), 2)
        jl2 = round(np.mean(join_losses2), 2)

        print("""{}: {}, N: {}, jl90: {}, jl_mean: {}, jl2: {},time: {}""".format(
            samples_type, num_iter, len(join_losses), jl90, jl_mean, jl2,
            round(time.time()-jl_eval_start, 2)))
        sys.stdout.flush()
        return est_card_costs, opt_costs

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

    def _subquery_join_losses(self, samples, pred):
        print("subquery join losses!")

        for qrep in samples:
            # going to compute all the join losses for this qrep
            est_cardinalities = []
            true_cardinalities = []
            sqls = []
            subsets = qrep["subset_graph"].nodes()
            for node, node_info in subsets.items():
                ests = {}
                trues = {}
                subgraph = qrep["join_graph"].subgraph(node)
                sql = nx_graph_to_query(subgraph)
                descendants = nx.descendants(qrep["subset_graph"], node)
                for desc in descendants:
                    alias_key = ' '.join(desc)
                    node_info = subsets[desc]
                    est_sel = pred[node_info["idx"]]
                    est_card = est_sel*node_info["cardinality"]["total"]
                    ests[alias_key] = int(est_card)
                    trues[alias_key] = node_info["cardinality"]["actual"]

                sqls.append(sql)
                est_cardinalities.append(ests)
                true_cardinalities.append(trues)

            assert len(sqls) == len(est_cardinalities)
            # parallel
            # args = (sqls, true_cardinalities, est_cardinalities, self.env,
                    # None, 16)
            # results = self.train_join_loss_pool.apply(join_loss_pg, args)
            # print(results[0])

            # single threaded
            for sqli,sql in enumerate(sqls):
                jls = join_loss_pg([sqls[sqli]], [true_cardinalities[sqli]],
                        [est_cardinalities[sqli]], self.env, None, 16)
                # print(jls)


    def _periodic_eval(self, X, Y, samples, samples_type,
            join_loss_pool, env):
        if "train" in samples_type:
            pred = self.eval_samples(X, self.train_num_table_mapping)
        else:
            pred = self.eval_samples(X, self.test_num_table_mapping)

        train_loss = self.loss(pred, Y, avg=False).detach().numpy()
        loss_avg = round(np.sum(train_loss) / len(train_loss), 2)
        print("""{}: {}, N: {}, qerr: {}""".format(
            samples_type, self.num_iter, len(X), loss_avg))
        if self.adaptive_lr and self.scheduler is not None:
            self.scheduler.step(loss_avg)

        self.add_row(train_loss, "qerr", self.num_iter, "all",
                "all", samples_type)

        start_t = time.time()
        summary_data = defaultdict(list)
        for sample in samples:
            template = sample["template_name"]
            for node, node_info in sample["subset_graph"].nodes().items():
                loss = train_loss[node_info["idx"]]
                num_tables = len(node)
                summary_data["loss"].append(loss)
                summary_data["num_tables"].append(num_tables)
                summary_data["template"].append(template)

        df = pd.DataFrame(summary_data)
        for template in set(df["template"]):
            tvals = df[df["template"] == template]
            self.add_row(tvals["loss"].values, "qerr", self.num_iter,
                    template, "all", samples_type)
            for nt in set(tvals["num_tables"]):
                nt_losses = tvals[tvals["num_tables"] == nt]
                self.add_row(nt_losses["loss"].values, "qerr", self.num_iter, template,
                        str(nt), samples_type)

        # TODO: simplify this.
        if "train" in samples_type:
            nt_map = self.train_num_table_mapping
        else:
            nt_map = self.test_num_table_mapping

        for nt in nt_map:
            start,end = nt_map[nt]
            cur_losses = train_loss[start:end]
            self.add_row(cur_losses, "qerr", self.num_iter, "all", str(nt),
                    samples_type)

        if (self.num_iter % self.eval_iter_jl == 0):
            jl_eval_start = time.time()
            assert self.jl_use_postgres
            if self.sampling_priority_type == "subquery":
                self._subquery_join_losses(samples, pred)

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

            # simplify...
            if self.sampling_priority_alpha == 0.0 \
                    or self.sampling_priority_type != "query" \
                    or self.num_iter <= 100 \
                    or not "train" in samples_type:
                return results
            else:
                # do sampling update based on results
                print("update priorities..")
                est_costs, opt_costs = self._update_join_results(results,
                                    samples, "train", self.num_iter)
                jl_ratio = est_costs / opt_costs
                subquery_sampling_weights = \
                        np.zeros(len(self.subquery_sampling_weights))
                for si, sample in enumerate(self.training_samples):
                    sq_weight = float(jl_ratio[si])
                    for node, node_info in sample["subset_graph"].nodes().items():
                        subquery_sampling_weights[node_info["idx"]] = sq_weight
                assert 0.00 not in np.unique(subquery_sampling_weights)
                # print("before _update sampling weights")
                # print(max(subquery_sampling_weights))
                # print(min(subquery_sampling_weights))
                # print(np.unique(subquery_sampling_weights))
                # pdb.set_trace()
                self.subquery_sampling_weights = \
                        self._update_sampling_weights(subquery_sampling_weights)
                return None
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
                    sampling_wts = self.subquery_sampling_weights[start:end]
                    sampling_wts = self._normalize_priorities(sampling_wts)

                    # TODO: make mb_size dependent on the level?
                    par_args.append((net, opt, Xcur, Ycur, num_steps,
                        self.mb_size, self.loss, self.clip_gradient,
                        sampling_wts))

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
            # random.shuffle(test_samples)
            self.Xtest, self.Ytest, self.test_num_table_mapping = \
                self._get_feature_vectors(self.test_samples)
            print("{} training, {} test subqueries".format(len(self.Xtrain),
                len(self.Xtest)))
            self.test_env = park.make('query_optimizer')

        print("feature len: {}, generation time: {}".\
                format(len(self.Xtrain[0]), time.time()-start))

        # FIXME: multiple table version
        self.init_nets()
        model_size = self.num_parameters()
        print("model size: {} MB".format(model_size))
        self.num_iter = 0

        # start off uniformly
        self.subquery_sampling_weights = [1/len(self.Xtrain)]*len(self.Xtrain)

        prev_end = time.time()
        while True:
            if (self.num_iter % 100 == 0):
                pass
                # progress stuff
                it_time = time.time() - prev_end
                prev_end = time.time()
                print("MB: {}, T:{}, I:{} : {}".format(\
                        self.mb_size, self.num_threads, self.num_iter, it_time))
                sys.stdout.flush()

            if (self.num_iter % self.eval_iter == 0):
                # we will wait on these results when we reach this point in the
                # next iteration
                self._update_join_results(self.train_join_results,
                        self.training_samples, "train",
                        self.num_iter-self.eval_iter)
                self.train_join_results = self._periodic_eval(self.Xtrain,
                        self.Ytrain, self.training_samples, "train",
                        self.train_join_loss_pool, self.env)

                # TODO: handle reweighing schemes here

                if test_samples is not None:
                    self._update_join_results(self.test_join_results,
                            self.test_samples, "test",
                            self.num_iter-self.eval_iter)
                    self.test_join_results = self._periodic_eval(self.Xtest,
                            self.Ytest, self.test_samples, "test",
                            self.test_join_loss_pool, self.test_env)

            if self.num_iter % self.eval_iter == 0:
                self.save_stats()

            self.train_step(self.eval_iter)
            self.num_iter += self.eval_iter

            if (self.num_iter >= self.max_iter):
                print("breaking because max iter done")
                break

    def test(self, test_samples):
        # generate preds in the form needed
        X, _, nt_map = self._get_feature_vectors(test_samples)
        sel_preds = self.eval_samples(X, nt_map).detach().cpu().numpy()
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                pred = sel_preds[info["idx"]]
                pred_dict[(alias_key)] = info["cardinality"]["total"]*pred
            preds.append(pred_dict)
        return preds

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

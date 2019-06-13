import time
import numpy as np
import pdb
import math
from pomegranate import BayesianNetwork
from db_utils.utils import *
from utils.utils import *
import matplotlib.pyplot as plt
from utils.net import SimpleRegression
from cardinality_estimation.losses import *
import pandas as pd
import json
from multiprocessing import Pool

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        pass
    def train(self, db, training_samples, **kwargs):
        pass
    def test(self, db, test_samples, **kwargs):
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

class Postgres(CardinalityEstimationAlg):
    def test(self, db, test_samples):
        return np.array([(s.pg_count / float(s.total_count)) for s in test_samples])

class Random(CardinalityEstimationAlg):
    def test(self, db, test_samples):
        return np.array([random.random() for _ in test_samples])

class Independent(CardinalityEstimationAlg):
    '''
    independent assumption on true marginal values.
    '''
    def test(self, db, test_samples):
        return np.array([np.prod(np.array(s.marginal_sels)) \
                for s in test_samples])

class BN(CardinalityEstimationAlg):
    '''
    '''
    def __init__(self, *args, **kwargs):
        if "alg" in kwargs:
            self.alg = kwargs["alg"]
        else:
            self.alg = "chow-liu"
        if "num_bins" in kwargs:
            self.num_bins = kwargs["num_bins"]
        else:
            self.num_bins = 5

        if "avg_factor" in kwargs:
            self.avg_factor = kwargs["avg_factor"]
        else:
            self.avg_factor = 1

        self.model = None
        # non-persistent cast, just to avoid running same alg again
        self.test_cache = {}
        self.min_groupby = 2

    def train(self, db, training_samples, **kwargs):
        # generate the group-by over all the columns we care about.
        # FIXME: for now, just for one table.
        self.db = db
        if "synth" in db.db_name:
            self._load_synth_model(db, training_samples, **kwargs)
        elif "osm" in db.db_name:
            self._load_osm_model(db, training_samples, **kwargs)
        elif "imdb" in db.db_name:
            self._load_imdb_model(db, training_samples, **kwargs)
        elif "dmv" in db.db_name:
            self.model = self.load_model()
            if self.model is None:
                self._load_dmv_model(db, training_samples, **kwargs)
        else:
            assert False

        self.save_model()

    def _load_osm_model(self, db, training_samples, **kwargs):
        # load directly to numpy since should be much faster
        data = np.fromfile('/data/pari/osm.bin',
                dtype=np.int64).reshape(-1, 6)
        columns = list(db.column_stats.keys())
        # drop the index column

        # FIXME: temporarily, drop a bunch of data
        # data = data[1000:100000,1:6]
        data = data[:,1:6]
        self.column_discrete_bins = {}
        for i in range(data.shape[1]):
            if db.column_stats[columns[i]]["num_values"] < 1000:
                continue
            d0 = data[:, i]
            _, bins = pd.qcut(d0, self.num_bins, retbins=True, duplicates="drop")
            self.column_discrete_bins[columns[i]] = bins
            data[:, i] = np.digitize(d0, bins)

        self.model = self.load_model()
        if self.model is None:
            # now, data has become discretized, we can feed it directly into BN.
            self.model = BayesianNetwork.from_samples(data,
                    state_names=columns, algorithm=self.alg, n_jobs=-1)

    def _load_synth_model(self, db, training_samples, **kwargs):
        assert len(db.tables) == 1
        table = [t for t in db.tables][0]
        columns = list(db.column_stats.keys())
        columns_str = ",".join(columns)
        group_by = GROUPBY_TEMPLATE.format(COLS = columns_str, FROM_CLAUSE=table)
        group_by += " ORDER BY COUNT(*) DESC"
        groupby_output = db.execute(group_by)
        samples = []
        weights = []
        for i, sample in enumerate(groupby_output):
            samples.append([])
            for j in range(len(sample)-1):
                samples[i].append(sample[j])
            weights.append(sample[j+1])
        samples = np.array(samples)
        weights = np.array(weights)
        self.model = BayesianNetwork.from_samples(samples, weights=weights,
                state_names=columns, algorithm=self.alg, n_jobs=-1)

    def _load_dmv_model(self, db, training_samples, **kwargs):
        columns = list(db.column_stats.keys())
        columns_str = ",".join(columns)
        # sel_all = "SELECT {COLS} FROM dmv".format(COLS = columns_str)
        group_by = GROUPBY_TEMPLATE.format(COLS = columns_str, FROM_CLAUSE="dmv")
        # group_by += " ORDER BY COUNT(*) DESC"
        group_by += " HAVING COUNT(*) > {}".format(self.min_groupby)
        print(group_by)
        # TODO: use db_utils
        groupby_output = db.execute(group_by)
        print("DMV groupby output len: ", len(groupby_output))

        start = time.time()
        samples = []
        weights = []
        for i, sample in enumerate(groupby_output):
            samples.append([])
            for j in range(len(sample)-1):
                if sample[j] is None:
                    # FIXME: what should be the sentinel value?
                    samples[i].append("-1")
                else:
                    samples[i].append(sample[j])
            weights.append(sample[j+1])
        samples = np.array(samples)
        weights = np.array(weights)
        print("constructing samples took: ", time.time()-start)
        self.model = BayesianNetwork.from_samples(samples, weights=weights,
                state_names=columns, algorithm=self.alg, n_jobs=-1)

    def _load_imdb_model(self, db, training_samples, **kwargs):
        # tables = [t for t in db.tables]
        # columns = list(db.column_stats.keys())
        # columns_str = ",".join(columns)
        # group_by = GROUPBY_TEMPLATE.format(COLS = columns_str, FROM_CLAUSE=table)
        # group_by += " ORDER BY COUNT(*) DESC"
        # FIXME: temporary
        sql = training_samples[0].query
        froms, _, _ = extract_from_clause(sql)
        # FIXME: should be able to store this somewhere and not waste
        # re-executing it always
        from_clause = " , ".join(froms)
        joins = extract_join_clause(sql)
        join_clause = ' AND '.join(joins)
        if len(join_clause) > 0:
            from_clause += " WHERE " + join_clause
        from_clause += " AND production_year IS NOT NULL "
        pred_columns, _, _ = extract_predicates(sql)
        columns_str = ','.join(pred_columns)
        group_by = GROUPBY_TEMPLATE.format(COLS = columns_str,
                FROM_CLAUSE=from_clause)

        groupby_output = db.execute(group_by)
        samples = []
        weights = []
        for i, sample in enumerate(groupby_output):
            samples.append([])
            for j in range(len(sample)-1):
                samples[i].append(sample[j])
            weights.append(sample[j+1])
        samples = np.array(samples)
        weights = np.array(weights)
        if self.model is None:
            self.model = BayesianNetwork.from_samples(samples, weights=weights,
                    state_names=pred_columns, algorithm=self.alg, n_jobs=-1)

    def test(self, db, test_samples):
        def _query_to_sample(sample):
            '''
            takes in a Query object, and converts it to the representation to
            be fed into the pomegranate bayesian net model
            '''
            model_sample = []
            for state in self.model.states:
                # find the right column entry in sample
                val = None
                possible_vals = []
                for i, column in enumerate(sample.pred_column_names):
                    if column == state.name:
                        cmp_op = sample.cmp_ops[i]
                        val = sample.vals[i]
                        if cmp_op == "in":
                            if hasattr(sample.vals[i], "__len__"):
                                # dedup
                                val = set(val)
                            # possible_vals = [int(v.replace("'","")) for v in val]
                            ## FIXME:
                            all_vals = [str(v.replace("'","")) for v in val]
                            for v in all_vals:
                                if v != "tv mini series":
                                    possible_vals.append(v)
                        elif cmp_op == "lt":
                            assert len(val) == 2
                            if self.db.db_name == "imdb":
                                # FIXME: hardcoded for current query...
                                val = [int(v) for v in val]
                                for ival in range(val[0], val[1]):
                                    possible_vals.append(str(ival))
                            else:
                                # discretize first
                                bins = self.column_discrete_bins[column]
                                val = [float(v) for v in val]
                                try:
                                    disc_vals = np.digitize(val, bins)
                                    for ival in range(disc_vals[0], disc_vals[1]+1):
                                        # possible_vals.append(ival)
                                        possible_vals.append(str(ival))
                                except:
                                    print(val)
                                    pdb.set_trace()
                        elif cmp_op == "eq":
                            possible_vals.append(val)
                        else:
                            print(sample)
                            print(column)
                            print(cmp_op)
                            pdb.set_trace()
                if len(possible_vals) == 0:
                    assert "county" != state.name
                    for dv in self.model.marginal()[len(model_sample)].parameters:
                        for k in dv:
                            possible_vals.append(k)
                model_sample.append(possible_vals)
            return model_sample

        estimates = []
        for qi, query in enumerate(test_samples):
            hashed_query = deterministic_hash(query.query)
            if hashed_query in self.test_cache:
                estimates.append(self.test_cache[hashed_query])
                continue
            model_sample = _query_to_sample(query)
            # pdb.set_trace()
            # FIXME: generalize
            all_points = model_sample
            # FIXME: ugh
            if len(model_sample) == 11:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4], model_sample[5],
                    model_sample[6], model_sample[7], model_sample[8],
                    model_sample[9], model_sample[10])).T.reshape(-1,11)
            elif len(model_sample) == 10:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4], model_sample[5],
                    model_sample[6], model_sample[7],
                    model_sample[8], model_sample[9])).T.reshape(-1,10)

            elif len(model_sample) == 9:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4], model_sample[5],
                    model_sample[6], model_sample[7], model_sample[8])).T.reshape(-1,9)

            elif len(model_sample) == 9:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4], model_sample[5],
                    model_sample[6], model_sample[7], model_sample[8])).T.reshape(-1,9)
            elif len(model_sample) == 8:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4], model_sample[5],
                    model_sample[6], model_sample[7])).T.reshape(-1,8)
            elif len(model_sample) == 7:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4], model_sample[5],
                    model_sample[6])).T.reshape(-1,7)
            elif len(model_sample) == 6:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4], model_sample[5])).T.reshape(-1,6)
            elif len(model_sample) == 5:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3], model_sample[4])).T.reshape(-1,5)
            elif len(model_sample) == 4:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1], model_sample[2],
                    model_sample[3])).T.reshape(-1,4)
            elif len(model_sample) == 3:
                all_points = np.array(np.meshgrid(model_sample[0], model_sample[1],
                    model_sample[2])).T.reshape(-1,3)
            elif len(model_sample) == 2:
                all_points = np.array(np.meshgrid(model_sample[0],
                    model_sample[1])).T.reshape(-1,2)
            else:
                assert False
            # we shouldn't assume the order of column names in the trained model
            start = time.time()
            est_sel = 0.0
            if self.db.db_name == "imdb":
                for p in all_points:
                    try:
                        est_sel += self.model.probability(p)
                    except Exception as e:
                        # unknown key ...
                        # guest seems to be failing ...
                        continue
            elif self.db.db_name == "dmv":
                if self.avg_factor == 1:
                    for p in all_points:
                        try:
                            est_sel += self.model.probability(p)
                        except Exception as e:
                            # FIXME: add minimum amount.
                            # unknown key ...
                            total = self.db.column_stats["dmv.record_type"]["total_vals"]
                            est_sel += float(self.min_groupby) / total
                            continue
                else:
                    N = len(all_points)
                    # samples_to_use = max(1000, int(N/self.avg_factor))
                    samples_to_use = min(self.avg_factor, N)
                    # print("orig samples: {}, using: {}".format(N,
                        # samples_to_use))
                    np.random.shuffle(all_points)
                    est_samples = all_points[0:samples_to_use]
                    for p in est_samples:
                        try:
                            est_sel += self.model.probability(p)
                        except Exception as e:
                            # FIXME: add minimum amount.
                            # unknown key ...
                            total = self.db.column_stats["dmv.record_type"]["total_vals"]
                            est_sel += float(self.min_groupby) / total
                            continue
                    est_sel = N*(est_sel / len(est_samples))

            else:
                if self.avg_factor == 1:
                    # don't do any averaging
                    est_sel = np.sum(self.model.probability(all_points))
                else:
                    N = len(all_points)
                    samples_to_use = max(1000, int(N/self.avg_factor))
                    # print("orig samples: {}, using: {}".format(N,
                        # samples_to_use))
                    np.random.shuffle(all_points)
                    est_samples = all_points[0:samples_to_use]
                    est_sel = N*np.average(self.model.probability(est_samples))

            if qi % 100 == 0:
                print("test query: ", qi)
                print("evaluating {} points took {} seconds"\
                        .format(len(all_points), time.time()-start))
            self.test_cache[hashed_query] = est_sel
            estimates.append(est_sel)
        return np.array(estimates)

    def get_name(self, suffix_name):
        '''
        unique name.
        '''
        name = self.alg + str(self.num_bins) + self.db.db_name + suffix_name
        name += str(self.min_groupby)
        return name

    def save_model(self, save_dir="./models/", suffix_name=""):
        self.model.plot()
        if not os.path.exists(save_dir):
            make_dir(save_dir)
        unique_name = self.get_name(suffix_name)
        plt.savefig(save_dir + "/" + unique_name + ".pdf")
        with open(save_dir + "/" + unique_name + ".json", "w") as f:
            f.write(self.model.to_json())

    def load_model(self, save_dir="./models/", suffix_name=""):
        fn = save_dir + "/" + self.get_name(suffix_name) + ".json"
        model = None
        if os.path.exists(fn):
            with open(fn, "r") as f:
                model = BayesianNetwork.from_json(f.read())
        return model

    def size(self):
        pass
    def __str__(self):
        # FIXME: add parameters of the learning model etc.
        name = self.__class__.__name__ + "-" + self.alg
        # name += "-bins" + str(self.num_bins)
        name += "avg:" + str(self.avg_factor)
        return name

def rel_loss_torch(pred, ytrue):
    '''
    Loss function for neural network training. Should use the
    compute_relative_loss formula, but deal with appropriate pytorch types.
    '''
    # this part is the same for both rho_est, or directly selectivity
    # estimation cases
    assert len(pred) == len(ytrue)
    epsilons = to_variable([REL_LOSS_EPSILON]*len(pred)).float()
    errors = torch.abs(pred-ytrue) / (torch.max(epsilons, ytrue))
    error = (errors.sum() * 100.00) / len(pred)
    return error

def qloss_torch(yhat, ytrue):

    epsilons = to_variable([QERR_MIN_EPS]*len(yhat)).float()
    ytrue = torch.max(ytrue, epsilons)
    yhat = torch.max(yhat, epsilons)

    # TODO: check this
    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    error = errors.sum() / len(yhat)
    return error

class NN1(CardinalityEstimationAlg):
    '''
    Default implementation of various neural network based methods.
    '''
    def __init__(self, *args, **kwargs):

        # TODO: make these all configurable
        self.feature_len = None
        self.hidden_layer_multiple = 2
        self.feat_type = "dict_encoding"
        self.max_num_buckets = 10000

        # as in the dl papers (not sure if this is needed)
        self.log_transform = False

        # TODO: configure other variables
        self.max_iter = kwargs["max_iter"]

    def train(self, db, training_samples, use_subqueries=False):
        if use_subqueries:
            training_samples = get_all_subqueries(training_samples)
        db.init_featurizer()
        X = []
        Y = []
        for sample in training_samples:
            features = db.get_features(sample)
            assert len(features) == db.feature_len
            X.append(features)
            if self.log_transform:
                if sample.true_sel == 0.00:
                    sel = 0.000001
                else:
                    sel = sample.true_sel
                Y.append(abs(math.log10(sel)))
            else:
                Y.append(sample.true_sel)

        if self.log_transform:
            self.maxy = max(Y)
            self.miny = min(Y)
            for i, y in enumerate(Y):
                Y[i] = (y-self.miny) / (self.maxy-self.miny)

        # do training
        net = SimpleRegression(len(X[0]),
                len(X[0])*self.hidden_layer_multiple, 1)
        loss_func = qloss_torch
        # loss_func = rel_loss_torch
        train_nn(net, X, Y, loss_func=loss_func, max_iter=self.max_iter,
                tfboard_dir="./tf-logs", lr=0.001, adaptive_lr=True,
                loss_threshold=2.0)

        print("train done!")
        self.net = net

    def test(self, db, test_samples):
        X = []
        for sample in test_samples:
            X.append(db.get_features(sample))
        # just pass each sample through net and done!
        X = to_variable(X).float()

        if self.log_transform:
            pred = self.net(X)
            pred = pred.squeeze(1)
            pred = pred.detach().numpy()
            for i, p in enumerate(pred):
                pred[i] = (p*(self.maxy-self.miny)) + self.miny
                pred[i] = math.pow(10, -pred[i])
            return pred
        else:
            pred = self.net(X)
            pred = pred.squeeze(1)
        return pred.detach().numpy()

    def size(self):
        pass
    def __str__(self):
        # FIXME: add parameters of the neural network
        return self.__class__.__name__

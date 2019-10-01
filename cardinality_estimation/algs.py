import time
import numpy as np
import pdb
import math
from pomegranate import BayesianNetwork
from db_utils.utils import *
from utils.utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.net import *
from cardinality_estimation.losses import *
import pandas as pd
import json
# from multiprocessing import Pool
from pgm import PGM
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

# sentinel value for NULLS
NULL_VALUE = "-1"

def get_all_num_table_queries(samples, num):
    '''
    @ret: all Query objects having @num tables
    '''
    ret = []
    for sample in samples:
        num_tables = len(sample.froms)
        if num_tables == num:
            ret.append(sample)
        for subq in sample.subqueries:
            num_tables = len(subq.froms)
            if num_tables == num:
                ret.append(subq)
    return ret

def get_all_features(samples, db):
    '''
    @samples: Query objects, with subqueries.
    @ret:
        X:
        Y:
    '''
    X = []
    Y = []
    for sample in samples:
        X.append(db.get_features(sample))
        Y.append(sample.true_sel)
        for subq in sample.subqueries:
            X.append(db.get_features(subq))
            Y.append(subq.true_sel)

    return np.array(X),np.array(Y)

def get_possible_values(sample, db, column_bins=None):
    '''
    @sample: Query object.
    @db: DB class object.
    @column_bins: {column_name : bins}. Used if we want to discretize some of
    the columns.

    @ret: RV = Random Variable / Column
        [[RV1-1, RV1-2, ...], [RV2-1, RV2-2, ...] ...]
        Each index refers to a column in the database (or a random variable).
        The predicates in the sample query are used to get all possible values
        that the random variable will need to be evaluated on.
    '''
    all_possible_vals = []
    # loop over each column in db
    states = db.column_stats.keys()
    for state in states:
        # find the right column entry in sample
        # val = None
        possible_vals = []
        # Note: Query.vals / Query.pred_column_names aren't sorted, and if
        # there are no predicates on a column, then it will not have an entry
        # in Query.vals
        for i, column in enumerate(sample.pred_column_names):
            if column != state:
                continue
            cmp_op = sample.cmp_ops[i]
            val = sample.vals[i]
            # dedup
            if hasattr(sample.vals[i], "__len__"):
                val = set(val)

            if cmp_op == "in":
                # FIXME: something with the osm dataset
                possible_vals = [str(v.replace("'","")) for v in val]
                # possible_vals = [v for v in val]
            elif cmp_op == "lt":
                assert len(val) == 2
                if column not in column_bins:
                    # then select everything in the given range of
                    # integers.
                    val = [int(v) for v in val]
                    for ival in range(val[0], val[1]):
                        possible_vals.append(str(ival))
                else:
                    # discretize first
                    bins = column_bins[column]
                    vals = [float(v) for v in val]
                    for bi, bval in enumerate(bins):
                        if bval >= vals[0]:
                            # ntile's start from 1
                            possible_vals.append(bi+1)
                        elif bval < vals[0]:
                            continue
                        if bval > vals[1]:
                            break
                    # print(vals)
                    # print(possible_vals)
                    # pdb.set_trace()
            elif cmp_op == "eq":
                possible_vals.append(val)
            else:
                assert False
        all_possible_vals.append(possible_vals)
    return all_possible_vals

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass
    def train(self, db, training_samples, **kwargs):
        pass
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

class Postgres(CardinalityEstimationAlg):
    def test(self, test_samples):
        return np.array([(s.pg_count / float(s.total_count)) for s in test_samples])

class PostgresRegex(CardinalityEstimationAlg):
    def test(self, test_samples):
        # return np.array([(s.pg_count / float(s.total_count)) for s in test_samples])
        ret = []
        for s in test_samples:
            REGEX = False
            for cmp_op in s.cmp_ops:
                if "like" in cmp_op.lower():
                    REGEX = True
            if REGEX:
                sel = s.true_sel
            else:
                sel = s.pg_count / float(s.total_count)
            ret.append(sel)
        return ret

class Random(CardinalityEstimationAlg):
    def test(self, test_samples):
        return np.array([random.random() for _ in test_samples])

class Independent(CardinalityEstimationAlg):
    '''
    independent assumption on true marginal values.
    '''
    def test(self, test_samples):
        return np.array([np.prod(np.array(s.marginal_sels)) \
                for s in test_samples])

class OurPGM(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        self.min_groupby = 0
        self.kwargs = kwargs
        self.backend = kwargs["backend"]
        self.alg_name = kwargs["alg_name"]
        self.model = PGM(alg_name=self.alg_name, backend=self.backend)

        self.num_bins = 100
        self.test_cache = {}
        self.column_bins = {}
        self.DEBUG = True

    def __str__(self):
        name = self.alg_name
        return name

    def _load_training_data(self, db, continuous_cols):
        '''
        @ret:
            samples: 2d array. each row represents an output from the group by
            of postgres over the given columns.
            weights: count of that group by row
        '''
        start = time.time()
        assert len(db.tables) == 1
        table = [t for t in db.tables][0]
        columns = list(db.column_stats.keys())

        if not continuous_cols:
            FROM = table
            columns_str = ",".join(columns)
        else:
            # If some of the columns have continuous data, then we will divide them
            # into quantiles
            inner_from = []
            outer_columns = []
            for full_col_name,stats in db.column_stats.items():
                col  = full_col_name[full_col_name.find(".")+1:]
                outer_columns.append(col)
                if is_float(stats["max_value"]) and \
                        stats["num_values"] > 5000:
                    ntile = NTILE_CLAUSE.format(COLUMN = col,
                                                ALIAS  = col[col.find(".")+1:],
                                                BINS   = self.num_bins)
                    inner_from.append(ntile)
		    # SELECT MIN(model_year) FROM (SELECT model_year, ntile(5)
		    #OVER (order by model_year) AS ntile from dmv) AS tmp group by ntile order by
		    #ntile;
                    ntile = NTILE_CLAUSE.format(COLUMN = col,
                                                ALIAS  = "ntile",
                                                BINS   = self.num_bins)
                    bin_cmd = '''SELECT MIN({COL}) FROM (SELECT {COL}, {NTILE}
                    FROM {TABLE}) AS tmp group by ntile order
                    by ntile'''.format(COL = col,
                                       NTILE = ntile,
                                       TABLE = table)
                    result = db.execute(bin_cmd)
                    self.column_bins[full_col_name] = [r[0] for r in result]
                else:
                    inner_from.append(col)

            FROM = "(SELECT {COLS} FROM {TABLE}) AS tmp".format(\
                            COLS = ",".join(inner_from),
                            TABLE = table)
            columns_str = ",".join(outer_columns)

        group_by = GROUPBY_TEMPLATE.format(COLS = columns_str,
                FROM_CLAUSE=FROM)
        group_by += " HAVING COUNT(*) > {}".format(self.min_groupby)

        groupby_output = db.execute(group_by)

        samples = []
        weights = []
        # FIXME: we could potentially avoid the loop here?
        for i, sample in enumerate(groupby_output):
            samples.append([])
            for j in range(len(sample)-1):
                # FIXME: need to make this consistent throughout
                if sample[j] is None:
                    # FIXME: what should be the sentinel value?
                    print("adding NULL_VALUE")
                    samples[i].append(NULL_VALUE)
                else:
                    samples[i].append(sample[j])
            # last value of the output should be the count in the groupby
            # template
            weights.append(sample[j+1])
        samples = np.array(samples)
        weights = np.array(weights)
        print("training joint distribution  generation took {} seconds".format(time.time()-start))
        return samples, weights

    def train(self, db, training_samples, **kwargs):
        self.db = db

        if "synth" in db.db_name:
            samples, weights = self._load_training_data(db, False)
        elif "osm" in db.db_name:
            samples, weights = self._load_training_data(db, True)
        elif "imdb" in db.db_name:
            assert False
        elif "dmv" in db.db_name:
            samples, weights = self._load_training_data(db, False)
        else:
            assert False
        columns = list(db.column_stats.keys())

        # TODO: make this more robust
        model_key = deterministic_hash(str(columns) + str(self.kwargs))

        model = self.load_model(model_key)
        if model is None:
            self.model.train(samples, weights, state_names=columns)
            self.save_model(self.model, model_key)
        else:
            self.model = model


    def load_model(self, key):
        # TODO have a system to do this
        return None

    def save_model(self, model, key):
        pass

    def test(self, test_samples):
        estimates = []
        for qi, query in enumerate(test_samples):
            # TODO: add the cache for all PGM models etc.
            hashed_query = deterministic_hash(query.query)
            if hashed_query in self.test_cache:
                estimates.append(self.test_cache[hashed_query])
                continue
            model_sample = get_possible_values(query, self.db,
                    self.column_bins)

            # normal method
            est_sel = self.model.evaluate(model_sample)
            if self.DEBUG:
                true_sel = query.true_sel
                qerr = max(true_sel / est_sel, est_sel / true_sel)
                if qerr > 4.00:
                    print(query)
                    print("est sel: {}, true sel: {}, qerr: {}".format(est_sel,
                        true_sel, qerr))
                    pdb.set_trace()

            estimates.append(est_sel)
            self.test_cache[hashed_query] = est_sel
        return estimates

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

        if "gen_bn_dist" in kwargs:
            self.gen_bn_dist = kwargs["gen_bn_dist"]
            self.cur_est_sels = []
        else:
            self.gen_bn_dist = 0

        self.model = None
        # non-persistent cast, just to avoid running same alg again
        self.test_cache = {}
        self.min_groupby = 0
        self.column_bins = {}

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
        print("trained BN model!")

    def _load_osm_model(self, db, training_samples, **kwargs):
        # load directly to numpy since should be much faster
        data = np.fromfile('/data/pari/osm.bin',
                dtype=np.int64).reshape(-1, 6)
        columns = list(db.column_stats.keys())
        # drop the index column

        # FIXME: temporarily, drop a bunch of data
        # data = data[1000:100000,1:6]
        data = data[:,1:6]
        self.column_bins = {}
        for i in range(data.shape[1]):
            # these columns don't need to be discretized.
            # FIXME: use more general check here.
            if db.column_stats[columns[i]]["num_values"] < 1000:
                continue
            d0 = data[:, i]
            _, bins = pd.qcut(d0, self.num_bins, retbins=True, duplicates="drop")
            self.column_bins[columns[i]] = bins
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
        group_by += " HAVING COUNT(*) > {}".format(self.min_groupby)
        # TODO: use db_utils
        groupby_output = db.execute(group_by)

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

    def test(self, test_samples):
        if self.gen_bn_dist:
            self.est_dist_pdf = PdfPages("./bn_est_dist.pdf")
        db = self.db
        estimates = []
        for qi, query in enumerate(test_samples):
            if len(self.cur_est_sels) > 0:
                # write it to pdf, and reset
                x = pd.Series(self.cur_est_sels, name="Point Estimates")
                ax = sns.distplot(x, kde=False)
                plt.title("BN : " + str(qi))
                plt.tight_layout()
                self.est_dist_pdf.savefig()
                plt.clf()
                cur_est_sels = []

            hashed_query = deterministic_hash(query.query)
            if hashed_query in self.test_cache:
                estimates.append(self.test_cache[hashed_query])
                continue
            model_sample = get_possible_values(query, self.db,
                    self.column_bins)
            all_points = []
            for element in itertools.product(*model_sample):
                all_points.append(element)
            all_points = np.array(all_points)
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
                            if self.gen_bn_dist:
                                self.cur_est_sels.append(est_sel)
                        except Exception as e:
                            # FIXME: add minimum amount.
                            # unknown key ...
                            print("unknown key got!")
                            total = self.db.column_stats["dmv.record_type"]["total_values"]
                            est_sel += float(self.min_groupby) / total
                            print(e)
                            pdb.set_trace()
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
                pass
                # print("test query: ", qi)
                # print("evaluating {} points took {} seconds"\
                        # .format(len(all_points), time.time()-start))
            self.test_cache[hashed_query] = est_sel
            estimates.append(est_sel)

        if self.gen_bn_dist:
            self.est_dist_pdf.close()
        return np.array(estimates)

    def get_name(self, suffix_name):
        '''
        unique name.
        '''
        name = self.alg + str(self.num_bins) + self.db.db_name + suffix_name
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
        # name += "avg:" + str(self.avg_factor)
        return name

def weighted_loss(yhat, ytrue):
    loss1 = rel_loss_torch(yhat, ytrue)
    loss2 = qloss_torch(yhat, ytrue)
    return loss1 + loss2

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
    error = (errors.sum()) / len(pred)
    return error

def qloss(yhat, ytrue):

    epsilons = np.array([QERR_MIN_EPS]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    yhat = np.maximum(yhat, epsilons)

    # TODO: check this
    errors = np.maximum( (ytrue / yhat), (yhat / ytrue))
    error = np.sum(errors) / len(yhat)

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
        self.hidden_layer_multiple = 2.0
        self.feat_type = "dict_encoding"

        # as in the dl papers (not sure if this is needed)
        self.log_transform = False

        # TODO: configure other variables
        self.max_iter = kwargs["max_iter"]

    def train(self, db, training_samples, save_model=True,
            use_subqueries=False):
        self.db = db
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

        query_str = ""
        for s in training_samples:
            query_str += s.query

        # do training
        net = SimpleRegression(len(X[0]),
                int(len(X[0])*self.hidden_layer_multiple), 1)

        if save_model:
            make_dir("./models")
            model_path = "./models/" + "nn1" + str(deterministic_hash(query_str))[0:5]
            if os.path.exists(model_path):
                net.load_state_dict(torch.load(model_path))
                print("loaded trained model!")

        loss_func = qloss_torch
        # loss_func = rel_loss_torch
        # loss_func = weighted_loss
        print("feature len: ", len(X[0]))
        train_nn(net, X, Y, loss_func=loss_func, max_iter=self.max_iter,
                tfboard_dir=None, lr=0.0001, adaptive_lr=True,
                loss_threshold=2.0)

        self.net = net

        if save_model:
            print("saved model path")
            torch.save(net.state_dict(), model_path)

    def test(self, test_samples):
        X = []
        for sample in test_samples:
            X.append(self.db.get_features(sample))
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
        return pred.cpu().detach().numpy()

    def size(self):
        pass
    def __str__(self):
        # FIXME: add parameters of the neural network
        return self.__class__.__name__


class NN2(CardinalityEstimationAlg):
    '''
    Default implementation of various neural network based methods.
    '''
    def __init__(self, *args, **kwargs):

        # TODO: make these all configurable
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
        self.eval_num_tables = kwargs["eval_num_tables"]
        self.optimizer_name = kwargs["optimizer_name"]

        self.clip_gradient = kwargs["clip_gradient"]
        self.rel_qerr_loss = kwargs["rel_qerr_loss"]
        self.rel_jloss = kwargs["rel_jloss"]
        self.adaptive_lr = kwargs["adaptive_lr"]
        self.baseline = kwargs["baseline"]
        self.loss_func = kwargs["loss_func"]
        self.sampling = kwargs["sampling"]
        self.sampling_priority_method = kwargs["sampling_priority_method"]
        self.adaptive_priority_alpha = kwargs["adaptive_priority_alpha"]
        self.sampling_priority_alpha = kwargs["sampling_priority_alpha"]
        self.net_name = kwargs["net_name"]
        self.reuse_env = kwargs["reuse_env"]

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

        self.stats["test"]["eval"] = {}
        self.stats["test"]["eval"]["qerr"] = {}
        self.stats["test"]["eval"]["join-loss"] = {}

        self.stats["train"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["train"]["tables_eval"]["qerr"] = {}

        self.stats["test"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["test"]["tables_eval"]["qerr"] = {}

        # TODO: store these
        self.stats["model_params"] = {}

    def train(self, db, training_samples, use_subqueries=False,
            test_samples=None):
        self.db = db
        db.init_featurizer()

        if self.eval_num_tables:
            self.table_x_train = defaultdict(list)
            self.table_x_test = defaultdict(list)
            self.table_y_train = defaultdict(list)
            self.table_y_test = defaultdict(list)
            num_tables = len(db.aliases)
            # print("num tables: ", num_tables)
            for i in range(1,num_tables+1):
                queries = get_all_num_table_queries(training_samples, i)
                for q in queries:
                    self.table_x_train[i].append(db.get_features(q))
                    self.table_y_train[i].append(q.true_sel)

                self.table_x_train[i] = \
                    to_variable(self.table_x_train[i]).float()
                self.table_y_train[i] = \
                    to_variable(self.table_y_train[i]).float()
                if test_samples:
                    queries = get_all_num_table_queries(test_samples, i)
                    for q in queries:
                        self.table_x_test[i].append(db.get_features(q))
                        self.table_y_test[i].append(q.true_sel)
                    self.table_x_test[i] = \
                        to_variable(self.table_x_test[i]).float()
                    self.table_y_test[i] = \
                        to_variable(self.table_y_test[i]).float()

        ## FIXME: don't store features in query objects
        # initialize samples
        for sample in training_samples:
            features = db.get_features(sample)
            sample.features = features
            for subq in sample.subqueries:
                subq_features = db.get_features(subq)
                subq.features = subq_features

        if test_samples:
            for sample in test_samples:
                features = db.get_features(sample)
                sample.features = features
                for subq in sample.subqueries:
                    subq_features = db.get_features(subq)
                    subq.features = subq_features

        print("feature len: ", len(features))

        if self.net_name == "FCNN":
            # do training
            net = SimpleRegression(len(features),
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers)
            self.mb_size = 128
        elif self.net_name == "LinearRegression":
            net = LinearRegression(len(features),
                    1)
            self.mb_size = 128
        elif self.net_name == "Hydra":
            net = Hydra(len(features),
                    self.hidden_layer_multiple, 1,
                    len(db.aliases), False)
            self.mb_size = 512
        elif self.net_name == "FatHydra":
            net = FatHydra(len(features),
                    self.hidden_layer_multiple, 1,
                    len(db.aliases))
            self.mb_size = 512
            print("FatHydra created!")
        elif self.net_name == "HydraLinear":
            net = Hydra(len(features),
                    self.hidden_layer_multiple, 1,
                    len(db.aliases), True)
            self.mb_size = 512
            print("Hydra created!")
        else:
            assert False

        if self.loss_func == "qloss":
            loss_func = qloss_torch
        elif self.loss_func == "rel":
            loss_func = rel_loss_torch
        elif self.loss_func == "weighted":
            loss_func = weighted_loss
        else:
            assert False

        self.net = net
        try:
            self._train_nn_join_loss(self.net, training_samples, test_samples,
                    self.lr, self.jl_start_iter, self.reuse_env,
                    loss_func=loss_func,
                    max_iter=self.max_iter, tfboard_dir=None,
                    loss_threshold=2.0, jl_variant=self.jl_variant,
                    eval_iter_jl=self.eval_iter_jl,
                    clip_gradient=self.clip_gradient,
                    rel_qerr_loss=self.rel_qerr_loss,
                    adaptive_lr=self.adaptive_lr, rel_jloss=self.rel_jloss)
        except KeyboardInterrupt:
            print("keyboard interrupt")
        except park.envs.query_optimizer.query_optimizer.QueryOptError:
            print("park exception")

        self.training_cache.dump()

    # same function for all the nns
    def _periodic_num_table_eval(self, loss_func, net, num_iter):
        for num_table in self.table_x_train:
            x_table = self.table_x_train[num_table]
            y_table = self.table_y_train[num_table]
            if len(x_table) == 0:
                continue
            pred_table = net(x_table)
            try:
                pred_table = pred_table.squeeze(1)
            except:
                pass
            loss_train = loss_func(pred_table, y_table)
            if num_table not in self.stats["train"]["tables_eval"]["qerr"]:
                self.stats["train"]["tables_eval"]["qerr"][num_table] = {}

            self.stats["train"]["tables_eval"]["qerr"][num_table][num_iter] = loss_train.item()

            # do for test as well
            if num_table not in self.table_x_test:
                continue
            x_table = self.table_x_test[num_table]
            y_table = self.table_y_test[num_table]
            pred_table = net(x_table)
            try:
                pred_table = pred_table.squeeze(1)
            except:
                pass
            loss_test = loss_func(pred_table, y_table)

            if num_table not in self.stats["test"]["tables_eval"]["qerr"]:
                self.stats["test"]["tables_eval"]["qerr"][num_table] = {}

            self.stats["test"]["tables_eval"]["qerr"][num_table][num_iter] = loss_test.item()

            print("num_tables: {}, train_qerr: {}, test_qerr: {}, size: {}".format(\
                    num_table, loss_train, loss_test, len(y_table)))

    def _periodic_eval(self, net, samples, X, Y, env, key, loss_func, num_iter,
            scheduler):

        # evaluate qerr
        pred = net(X)
        try:
            pred = pred.squeeze(1)
        except:
            pass

        train_loss = loss_func(pred, Y)
        self.stats[key]["eval"]["qerr"][num_iter] = train_loss.item()
        print("""\n{}: {}, num samples: {}, qerr: {}""".format(
            key, num_iter, len(X), train_loss.item()))

        # FIXME: add scheduler loss ONLY for training cases
        if not self.jl_variant and self.adaptive_lr and key == "train":
            # FIXME: should we do this for minibatch / or for train loss?
            scheduler.step(train_loss)

        if (num_iter % self.eval_iter_jl == 0 \
                and num_iter != 0):
            jl_eval_start = time.time()
            est_card_costs, baseline_costs = join_loss(pred, samples, env,
                    baseline=self.baseline)

            join_losses = np.array(est_card_costs) - np.array(baseline_costs)
            join_losses2 = np.array(est_card_costs) / np.array(baseline_costs)

            jl1 = np.mean(join_losses)
            jl2 = np.mean(join_losses2)

            # FIXME: remove all negative values, so weighted_prob can work
            # fine. But there really shouldn't be any negative values here.
            join_losses = np.maximum(join_losses, 0.00)

            if self.jl_variant and self.adaptive_lr and key == "train":
                scheduler.step(jl1)

            self.stats[key]["eval"]["join-loss"][num_iter] = jl1

            # TODO: add color to key values.
            print("""\n{}: {}, num samples: {}, loss: {}, jl1 {},jl2 {},time: {}""".format(
                key, num_iter, len(X), train_loss.item(), jl1, jl2,
                time.time()-jl_eval_start))

            return join_losses, join_losses2

        return None, None

    def update_sampling_weights(self, priorities):
        '''
        refer to prioritized action replay
        '''
        priorities = np.power(priorities, self.sampling_priority_alpha)
        total = np.sum(priorities)
        query_sampling_weights = np.zeros(len(priorities))
        for i, priority in enumerate(priorities):
            query_sampling_weights[i] = priority / total

        return query_sampling_weights

    def _train_nn_join_loss(self, net, training_samples, test_samples,
            lr, jl_start_iter, reuse_env,
            max_iter=10000, eval_iter_jl=500,
            eval_iter_qerr=1000, mb_size=1,
            loss_func=None, tfboard_dir=None, adaptive_lr=True,
            min_lr=1e-17, loss_threshold=1.0, jl_variant=False,
            clip_gradient=10.00, rel_qerr_loss=False,
            jl_divide_constant=1, rel_jloss=False):
        '''
        TODO: explain and generalize.
        '''
        if loss_func is None:
            assert False
            loss_func = torch.nn.MSELoss()

        if self.optimizer_name == "ams":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                    amsgrad=True)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                    amsgrad=False)
        elif self.optimizer_name == "sgd":
            print(net)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        else:
            assert False

        # update learning rate
        if adaptive_lr:
            scheduler = ReduceLROnPlateau(optimizer, 'min',
                    patience=self.adaptive_lr_patience,
                            verbose=True, factor=0.1, eps=min_lr)
        else:
            scheduler = None

        num_iter = 0

        # create a new park env, and close at the end.
        env = park.make('query_optimizer')

        # FIXME: decompose

        X_all = []
        Y_all = []
        for si, sample in enumerate(training_samples):
            X_all.append(sample.features)
            Y_all.append(sample.true_sel)
            for sq in sample.subqueries:
                X_all.append(sq.features)
                Y_all.append(sq.true_sel)
        X = to_variable(X_all).float()
        Y = to_variable(Y_all).float()

        if test_samples:
            Xtest = []
            Ytest = []
            for si, sample in enumerate(test_samples):
                Xtest.append(sample.features)
                Ytest.append(sample.true_sel)
                for sq in sample.subqueries:
                    Xtest.append(sq.features)
                    Ytest.append(sq.true_sel)
            Xtest = to_variable(Xtest).float()
            Ytest = to_variable(Ytest).float()

        min_qerr = {}
        max_qerr = {}
        max_jloss = {}

        file_name = "./training-" + self.__str__() + ".dict"
        start = time.time()
        if self.sampling == "weighted_query":
            query_sampling_weights = [1 / len(training_samples)]*len(training_samples)
            # FIXME: decompose
            subquery_sampling_weights = []
            for si, sample in enumerate(training_samples):
                sq_weight = query_sampling_weights[si]
                sq_weight /= (len(sample.subqueries)+1)
                wts = [sq_weight]*(len(sample.subqueries)+1)
                # add lists
                subquery_sampling_weights += wts
        else:
            query_sampling_weights = None

        while True:
            if (num_iter % 100 == 0):
                # progress stuff
                print(num_iter, end=",")
                sys.stdout.flush()

            if (num_iter % self.eval_iter == 0):

                if self.eval_num_tables:
                    self._periodic_num_table_eval(loss_func, net, num_iter)

                join_losses, join_losses_ratio = self._periodic_eval(net, training_samples, X, Y,
                        env, "train", loss_func, num_iter, scheduler)
                if test_samples:
                    self._periodic_eval(net, test_samples, Xtest, Ytest,
                            env,"test", loss_func, num_iter, scheduler)

                if not reuse_env and (num_iter % self.eval_iter_jl == 0):
                    env.clean()
                    env = park.make('query_optimizer')

                # update query_sampling_wieghts if needed
                if query_sampling_weights is not None \
                        and join_losses is not None:
                    if self.adaptive_priority_alpha:
                        # temporary:
                        self.sampling_priority_alpha = num_iter / 4000
                        print("new priority alpha: ", self.sampling_priority_alpha)

                    if self.sampling_priority_method == "jl_ratio":
                        print("max: ", np.max(join_losses_ratio))
                        print("min: ", np.min(join_losses_ratio))
                        print("std: ", np.std(join_losses_ratio))
                        # total_join_loss = np.sum(np.array(join_losses_ratio))
                        # for wi, _ in enumerate(query_sampling_weights):
                            # wt = join_losses_ratio[wi] / total_join_loss
                            # query_sampling_weights[wi] = wt

                        query_sampling_weights = self.update_sampling_weights(join_losses_ratio)
                    elif self.sampling_priority_method == "jl_rank":
                        jl_ranks = np.argsort(join_losses_ratio)
                        jl_ranks += 1
                        jl_priorities = np.zeros(len(jl_ranks))
                        for ri, rank in enumerate(jl_ranks):
                            jl_priorities[ri] = 1.00 / float(rank)
                        query_sampling_weights = self.update_sampling_weights(jl_priorities)

                    elif self.sampling_priority_method == "jl_diff":
                        query_sampling_weights = self.update_sampling_weights(join_losses)

                    else:
                        print("sampling method: ", self.sampling_priority_method)
                        assert False

                    # print(query_sampling_weights)
                    assert (np.array(query_sampling_weights) >= 0).all()

                    assert query_sampling_weights is not None
                    subquery_sampling_weights = []
                    for si, sample in enumerate(training_samples):
                        sq_weight = query_sampling_weights[si]
                        sq_weight /= (len(sample.subqueries)+1)
                        wts = [sq_weight]*(len(sample.subqueries)+1)
                        # add lists
                        subquery_sampling_weights += wts

            # sampling function: do it at the granularity of Query or treat
            # all SubQueries the same?

            if self.sampling == "query":
                assert False
                mb_samples = []
                xbatch = []
                ybatch = []
                cur_samples = random.sample(training_samples, mb_size)
                for si, sample in enumerate(cur_samples):
                    mb_samples.append(sample)
                    xbatch.append(sample.features)
                    ybatch.append(sample.true_sel)
                    for sq in sample.subqueries:
                        xbatch.append(sq.features)
                        ybatch.append(sq.true_sel)

                xbatch = to_variable(xbatch).float()
                ybatch = to_variable(ybatch).float()
            elif self.sampling == "subquery":
                MB_SIZE = self.mb_size
                idxs = np.random.choice(list(range(len(X))), MB_SIZE)
                xbatch = X[idxs]
                ybatch = Y[idxs]
            elif self.sampling == "weighted_query":
                MB_SIZE = self.mb_size
                idxs = np.random.choice(list(range(len(X))), MB_SIZE,
                        p=subquery_sampling_weights)
                xbatch = X[idxs]
                ybatch = Y[idxs]

            pred = net(xbatch)
            try:
                pred = pred.squeeze(1)
            except:
                pass
            # pred = pred.squeeze(1)

            loss = loss_func(pred, ybatch)

            # how do we modify the loss variable based on different join loss
            # variants?
            if (num_iter > jl_start_iter and jl_variant):

                if jl_variant in [1, 2]:
                    est_card_costs, baseline_costs = join_loss(pred, mb_samples, self, env,
                            baseline=self.baseline)

                    ## TODO: first one is just too large a number to use (?)
                    # jl = np.array(est_card_costs)  - np.array(baseline_costs)
                    jl = np.array(est_card_costs)  / np.array(baseline_costs)

                    # set jl, one way or another.
                    if jl_variant == 1:
                        jl = torch.mean(to_variable(jl).float())
                    elif jl_variant == 2:
                        jl = torch.mean(to_variable(jl).float()) - 1.00
                    else:
                        assert False

                    if rel_qerr_loss:
                        sample_key = deterministic_hash(cur_samples[0].query)
                        loss = loss / to_variable(max_qerr[sample_key]).float()

                    if rel_jloss:
                        if sample_key not in max_jloss:
                            max_jloss[sample_key] = np.array(max(jl.item(), 1.00))
                        sample_key = deterministic_hash(cur_samples[0].query)
                        jl = jl / to_variable(max_jloss[sample_key]).float()

                    loss = loss*jl

                elif jl_variant == 4:
                    pred_sort, pred_idx = torch.sort(pred)
                    ybatch_sort, y_idx = torch.sort(ybatch)
                    diff_idx = pred_idx - y_idx
                    jl = torch.mean(torch.abs(diff_idx.float()))
                    loss = loss*jl
                elif jl_variant in [3]:
                    # order based loss, operating on pred and ybatch
                    pred_sort, idx = torch.sort(pred)
                    ybatch_sort, idx = torch.sort(ybatch)
                    assert pred_sort.requires_grad
                    assert ybatch_sort.requires_grad
                    loss2 = 0.2*loss_func(pred_sort, ybatch_sort)
                    assert loss2.requires_grad
                    loss1 = loss
                    loss = loss1 + loss2
                    # print("loss1: {}, loss 2: {}, loss: {}".format(loss1.item(), loss2.item(),
                                # loss.item()))
                else:
                    assert False

            # FIXME: temporary, to try and adjust for the variable batch size.
            # if self.divide_mb_len:
                # loss /= len(pred)

            if (num_iter > max_iter):
                print("breaking because max iter done")
                break

            optimizer.zero_grad()
            loss.backward()
            if clip_gradient is not None:
                clip_grad_norm_(net.parameters(), clip_gradient)

            optimizer.step()
            num_iter += 1

        print("training took: {} seconds".format(time.time()-start))
        env.clean()

    def test(self, test_samples):
        X = []
        for sample in test_samples:
            X.append(self.db.get_features(sample))
        # just pass each sample through net and done!
        X = to_variable(X).float()
        pred = self.net(X)
        pred = pred.squeeze(1)
        return pred.cpu().detach().numpy()

    def size(self):
        pass
    def __str__(self):
        name = "nn"
        if self.jl_variant:
            name += "-jl" + str(self.jl_variant)

        return name

def train_nn_par(net, optimizer, X, Y, loss_func, clip_gradient,
        num_iter):

    MB_SIZE = 128
    idxs = np.random.choice(list(range(len(X))), MB_SIZE)
    X = to_variable(X).float()
    Y = to_variable(Y).float()
    xbatch = X[idxs]
    ybatch = Y[idxs]

    pred = net(xbatch)
    pred = pred.squeeze(1)
    loss = loss_func(pred, ybatch)

    optimizer.zero_grad()
    loss.backward()
    print("loss: {}".format(loss.item()))

    if clip_gradient is not None:
        clip_grad_norm_(net.parameters(), clip_gradient)

    optimizer.step()

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

        if kwargs["loss_func"] == "qloss":
            self.loss_func = qloss_torch
        else:
            assert False

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
        # self.loss_func = kwargs["loss_func"]
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

        self.stats["test"]["eval"] = {}
        self.stats["test"]["eval"]["qerr"] = {}
        self.stats["test"]["eval"]["join-loss"] = {}

        self.stats["train"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["train"]["tables_eval"]["qerr"] = {}

        self.stats["test"]["tables_eval"] = {}
        # key will be int: num_table, and val: qerror
        self.stats["test"]["tables_eval"]["qerr"] = {}

        self.stats["model_params"] = {}

    # same function for all the nns
    def _periodic_num_table_eval_nets(self, loss_func, num_iter):
        for num_table in self.table_x_train:
            x_table = self.table_x_train[num_table]
            y_table = self.table_y_train[num_table]
            if len(x_table) == 0:
                continue

            if num_table in self.num_tables_train_qerr:
                if self.num_tables_train_qerr[num_table] < self.loss_stop_thresh:
                    continue

            net = self.models[num_table]
            pred_table = net(x_table)
            pred_table = pred_table.squeeze(1)
            loss_train = loss_func(pred_table, y_table)
            if num_table not in self.stats["train"]["tables_eval"]["qerr"]:
                self.stats["train"]["tables_eval"]["qerr"][num_table] = {}

            self.stats["train"]["tables_eval"]["qerr"][num_table][num_iter] = loss_train.item()
            self.num_tables_train_qerr[num_table] = loss_train.item()

            # do for test as well
            if num_table not in self.table_x_test:
                continue
            x_table = self.table_x_test[num_table]
            y_table = self.table_y_test[num_table]
            pred_table = net(x_table)
            pred_table = pred_table.squeeze(1)
            loss_test = loss_func(pred_table, y_table)
            if num_table not in self.stats["test"]["tables_eval"]["qerr"]:
                self.stats["test"]["tables_eval"]["qerr"][num_table] = {}

            self.stats["test"]["tables_eval"]["qerr"][num_table][num_iter] = loss_test.item()

            print("num_tables: {}, train_qerr: {}, test_qerr: {}, size: {}".format(\
                    num_table, loss_train, loss_test, len(y_table)))

    def _periodic_eval(self, samples, env, key, loss_func,
            num_iter):

        assert (num_iter % self.eval_iter == 0)
        Y = []
        pred = []

        # FIXME: optimize this
        # it is important to maintain the same order of traversal for the
        # join_loss compute function to work (ugh...)
        for sample in samples:
            Y.append(sample.true_sel)
            num_tables = len(sample.froms)
            pred.append(self.models[num_tables](sample.features).item())

            for subq in sample.subqueries:
                Y.append(subq.true_sel)
                num_tables = len(subq.froms)
                pred.append(self.models[num_tables](subq.features).item())

        pred = to_variable(pred).float()
        Y = to_variable(Y).float()
        train_loss = loss_func(pred, Y)

        self.stats[key]["eval"]["qerr"][num_iter] = train_loss.item()

        print("""\n{}: {}, num samples: {}, loss: {}""".format(
            key, num_iter, len(Y), train_loss.item()))

        if (num_iter % self.eval_iter_jl == 0):
            jl_eval_start = time.time()
            est_card_costs, baseline_costs = join_loss(pred, samples, env,
                    baseline=self.baseline)

            join_losses = np.array(est_card_costs) - np.array(baseline_costs)
            join_losses2 = np.array(est_card_costs) / np.array(baseline_costs)

            jl1 = np.mean(join_losses)
            jl2 = np.mean(join_losses2)

            # FIXME: remove all negative values, so weighted_prob can work
            # fine. But there really shouldn't be any negative values here.
            join_losses = np.maximum(join_losses, 0.00)

            self.stats[key]["eval"]["join-loss"][num_iter] = jl1

            # TODO: add color to key values.
            print("""\n{}: {}, num samples: {}, loss: {}, jl1 {},jl2 {},time: {}""".format(
                key, num_iter, len(Y), train_loss.item(), jl1, jl2,
                time.time()-jl_eval_start))

            return join_losses, join_losses2

        return None, None

    def _train_nn(self, db, training_samples, **kwargs):
        test_samples = kwargs["test_samples"]
        for sample in training_samples:
            features = db.get_features(sample)
            num_tables = len(sample.froms)
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
                num_tables = len(subq.froms)
                assert num_tables == len(subq.aliases)
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

                    if self.eval_num_tables:
                        self._periodic_num_table_eval_nets(self.loss_func, num_iter)

                    # evaluation code
                    if (num_iter % self.eval_iter_jl == 0 \
                            and not self.reuse_env):
                        env = park.make('query_optimizer')

                    join_losses, join_losses_ratio = self._periodic_eval(training_samples,
                            env, "train", self.loss_func, num_iter)
                    if test_samples:
                        self._periodic_eval(test_samples,
                                env,"test", self.loss_func, num_iter)

                    if not self.reuse_env and env is not None:
                        env.clean()
                        env = None

                for num_tables, _ in self.samples.items():

                    if num_tables in self.num_tables_train_qerr:
                        if self.num_tables_train_qerr[num_tables] < self.loss_stop_thresh:
                            # print("skipping training ", num_tables)
                            continue

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
                    loss = self.loss_func(pred, ybatch)

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
            num_tables = len(db.aliases)
            print("num tables: ", num_tables)
            for i in range(1,num_tables+1):
                queries = get_all_num_table_queries(training_samples, i)
                for q in queries:
                    self.table_x_train[i].append(db.get_features(q))
                    self.table_y_train[i].append(q.true_sel)

                self.table_x_train[i] = \
                    to_variable(self.table_x_train[i]).float()
                self.table_y_train[i] = \
                    to_variable(self.table_y_train[i]).float()
                if test_samples:
                    queries = get_all_num_table_queries(test_samples, i)
                    for q in queries:
                        self.table_x_test[i].append(db.get_features(q))
                        self.table_y_test[i].append(q.true_sel)
                    self.table_x_test[i] = \
                        to_variable(self.table_x_test[i]).float()
                    self.table_y_test[i] = \
                        to_variable(self.table_y_test[i]).float()

        for sample in training_samples:
            features = db.get_features(sample)
            num_tables = len(sample.froms)
            if num_tables not in self.samples:
                self.samples[num_tables] = []
                self.Xtrains[num_tables] = []
                self.Ytrains[num_tables] = []

            self.Xtrains[num_tables].append(features)
            self.Ytrains[num_tables].append(sample.true_sel)
            for subq in sample.subqueries:
                num_tables = len(subq.froms)
                assert num_tables == len(subq.aliases)
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
            num_tables = len(sample.froms)
            model = self.models[num_tables]
            pred.append(model.predict([self.db.get_features(sample)]))
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

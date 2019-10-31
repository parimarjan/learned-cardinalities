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
from sqlalchemy import create_engine

# sentinel value for NULLS
NULL_VALUE = "-1"
OSM_FILE = '/Users/pari/db_data/osm.bin'

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

def get_possible_values(sample, db, column_bins=None,
        column_bin_vals=None):
    '''
    @sample: Query object.
    @db: DB class object.
    @column_bins: {column_name : bins}. Used if we want to discretize some of
    the columns.

    @ret:
        @possible_vals = Random Variable / Column
        [[RV1-1, RV1-2, ...], [RV2-1, RV2-2, ...] ...]
        Each index refers to a column in the database (or a random variable).
        The predicates in the sample query are used to get all possible values
        that the random variable will need to be evaluated on.

        @weights: same dimensions as possible vals. Only really makes sense
        when dealing with continuous random variables which are binned. For the
        rv's which don't cover a complete bin, this will represent the weight
        given by: (end - val) / (end-start)
    '''
    all_possible_vals = []
    all_weights = []
    # loop over each column in db
    states = db.column_stats.keys()
    # avoid duplicates, in range queries, unforunately, Query object stores two
    # columns with same names
    seen = []
    for state in states:
        # find the right column entry in sample
        # val = None
        possible_vals = []
        weights = []
        # Note: Query.vals / Query.pred_column_names aren't sorted, and if
        # there are no predicates on a column, then it will not have an entry
        # in Query.vals
        for i, column in enumerate(sample.pred_column_names):
            if column != state or column in seen:
                continue
            seen.append(column)
            cmp_op = sample.cmp_ops[i]
            val = sample.vals[i]

            if cmp_op == "in":
                # dedup
                if hasattr(sample.vals[i], "__len__"):
                    val = set(val)
                possible_vals = [str(v.replace("'","")) for v in val]
                weights.append(1.00)
            elif cmp_op == "lt":
                if column not in column_bins:
                    # then select everything in the given range of
                    # integers.
                    val = [float(v) for v in val]
                    for v in db.column_stats[column]["unique_values"]:
                        v = v[0]
                        if v is None:
                            continue
                        if v >= val[0] and v <= val[1]:
                            possible_vals.append(v)
                            weights.append(1.00)
                else:
                    assert column_bins is not None
                    assert column_bin_vals is not None
                    # discretize first
                    bins = column_bins[column]
                    column_groupby = column_bin_vals[column]
                    vals = [float(v) for v in val]
                    binned_vals = np.digitize(vals, bins, right=True)
                    # FIXME: do something in between (collect stats like #rvs
                    # per bin etc.)
                    USE_PRECISE_WEIGHTS = False
                    if not USE_PRECISE_WEIGHTS:
                        for bi in range(binned_vals[0],binned_vals[1]+1):
                            possible_vals.append(bi)
                            weights.append(1.00)
                    else:
                        groupby_key = column_groupby.keys()[0]
                        if (binned_vals[0] == binned_vals[1]):
                            lower_lim = bins[binned_vals[0]-1]
                            upper_lim = bins[binned_vals[0]]

                            bin_groupby = \
                                column_groupby[column_groupby[groupby_key] >= lower_lim]
                            bin_groupby = \
                                    bin_groupby[bin_groupby[groupby_key] < upper_lim]
                            # FIXME: check edge conditions
                            total_val = bin_groupby[groupby_key].sum()
                            bin_groupby = \
                                bin_groupby[bin_groupby[groupby_key] > vals[0]]
                            selected_val = bin_groupby[groupby_key].sum()
                            weight = float(selected_val) / total_val
                            possible_vals.append(binned_vals[0])
                            weights.append(weight)
                        else:
                            # different bins, means the weight can be different for
                            # the endpoints, and 1.00 for all the middle ones

                            # because right=True when we use np.digitize
                            assert binned_vals[0] != 0
                            for bi in range(binned_vals[0],binned_vals[1]+1):
                                possible_vals.append(bi)
                                # if not an edge column then just add 1.00 weight
                                lower_lim = bins[bi-1]
                                upper_lim = bins[bi]
                                assert lower_lim <= vals[1]
                                assert upper_lim >= vals[0]
                                groupby_key = column_groupby.keys()[0]

                                if bi == binned_vals[0]:
                                    assert lower_lim <= vals[0]
                                    bin_groupby = \
                                        column_groupby[column_groupby[groupby_key] >= lower_lim]
                                    bin_groupby = \
                                            bin_groupby[bin_groupby[groupby_key] < upper_lim]
                                    # FIXME: check edge conditions
                                    total_val = bin_groupby[groupby_key].sum()
                                    bin_groupby = \
                                        bin_groupby[bin_groupby[groupby_key] > vals[0]]
                                    selected_val = bin_groupby[groupby_key].sum()
                                    weight = float(selected_val) / total_val
                                elif bi == binned_vals[1]:
                                    bin_groupby = \
                                        column_groupby[column_groupby[groupby_key] >= lower_lim]
                                    bin_groupby = \
                                            bin_groupby[bin_groupby[groupby_key] < upper_lim]
                                    # FIXME: check edge conditions
                                    total_val = bin_groupby[groupby_key].sum()
                                    bin_groupby = \
                                        bin_groupby[bin_groupby[groupby_key] < vals[1]]
                                    selected_val = bin_groupby[groupby_key].sum()
                                    weight = float(selected_val) / total_val
                                else:
                                    weight = 1.00

                                assert weight <= 1.00
                                weights.append(weight)

            elif cmp_op == "eq":
                possible_vals.append(val)
                weights.append(1.00)
            else:
                assert False

        all_possible_vals.append(possible_vals)
        all_weights.append(weights)

    return all_possible_vals, all_weights

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass
    def train(self, db, training_samples, **kwargs):
        pass
    def test(self, test_samples, **kwargs):
        pass
    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        return self.__class__.__name__
    def save_model(self, save_dir="./", suffix_name=""):
        pass

class Postgres(CardinalityEstimationAlg):
    def test(self, test_samples):
        return np.array([(s.pg_count / float(s.total_count)) for s in test_samples])

    # def test(self, test_samples):
        # # num tables based
        # ret = []
        # num_tables = defaultdict(list)
        # num_tables_true = defaultdict(list)
        # for sample in test_samples:
            # num_table = len(sample.froms)
            # true_sel = sample.true_sel
            # pg_sel = sample.pg_count / float(sample.total_count)
            # num_tables[num_table].append(pg_sel)
            # num_tables_true[num_table].append(true_sel)

            # if num_table <= 3:
                # ret.append(true_sel)
            # else:
                # ret.append(pg_sel)

        # for table in num_tables:
            # yhat = np.array(num_tables[table])
            # ytrue = np.array(num_tables_true[table])
            # qloss_val = qloss(yhat, ytrue)
            # print("{}: qerr: {}".format(table, qloss_val))

        # return ret


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

    def avg_eval_time(self):
        return 0.00001

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

class Sampling(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        self.sampling_percentage = kwargs["sampling_percentage"]
        self.eval_times = []
        self.gen_times = []

    def train(self, db, training_samples, **kwargs):
        # FIXME: this should be a utility function, also used in
        # _load_training_data2
        table = [t for t in db.tables][0]
        print(table)
        columns = list(db.column_stats.keys())
        FROM = table
        columns_str = ",".join(columns)
        # just select all of these columns from the given table
        select_all = "SELECT {COLS} FROM {TABLE} WHERE random() < {PERC}".format(
                COLS = columns_str,
                TABLE = table,
                PERC = str(self.sampling_percentage / 100.00))
        print(select_all)

        # TODO: add cache here.
        data_cache = klepto.archives.dir_archive("./misc_cache",
                cached=True, serialized=True)
        cache_key = select_all
        if cache_key in data_cache.archive:
            df = data_cache.archive[cache_key]
            print("loaded sampling data from the cache!")
        else:
            cmd = 'postgresql://{}:{}@localhost:5432/{}'.format(db.user,
                    db.pwd, db.db_name)
            engine = create_engine(cmd)
            df = pd.read_sql_query(select_all, engine)
            data_cache.archive[cache_key] = df

        print(df.keys())
        self.df = df
        print("len samples: ", len(self.df))
        print("training done!")
        # pdb.set_trace()
        self.test_cache = {}

        self.sampling_time = 10     #ms

    def test(self, test_samples, **kwargs):
        import functools
        def conjunction(*conditions):
            return functools.reduce(np.logical_and, conditions)
        predictions = []
        total = len(self.df)
        for si, sample in enumerate(test_samples):
            matching = 0
            # if si % 100 == 0:
                # print(si)

            hashed_query = deterministic_hash(sample.query)
            if hashed_query in self.test_cache:
                predictions.append(self.test_cache[hashed_query])
                continue
            # cur_df = self.df
            # go over every predicate in sample
            start = time.time()
            conditions = []
            for i, pred in enumerate(sample.pred_column_names):
                pred = pred[pred.find(".")+1:]
                cmp_op = sample.cmp_ops[i]
                vals = sample.vals[i]
                if cmp_op == "in":
                    # cur_df = cur_df[cur_df[pred].isin(vals)]
                    cond = self.df[pred].isin(vals)
                    conditions.append(cond)
                elif cmp_op == "lt":
                    lb = float(vals[0])
                    ub = float(vals[1])
                    assert lb <= ub
                    # cur_df = cur_df[cur_df[pred] >= lb]
                    # cur_df = cur_df[cur_df[pred] < ub]
                    cond1 = self.df[pred] >= lb
                    cond2 = self.df[pred] < ub
                    conditions.append(cond1)
                    conditions.append(cond2)
                else:
                    print(cmp_op)
                    assert False

            self.gen_times.append(time.time() - start)
            start = time.time()
            filtered = self.df[conjunction(*conditions)]
            matching = len(filtered)
            self.eval_times.append(time.time()-start)
            # print("filtering: {} sec".format(eval_times[-1])
            true_sel = matching / total

            predictions.append(true_sel)
            self.test_cache[hashed_query] = true_sel

        if len(self.eval_times) > 0:
            print("sampling eval time avg: ", np.mean(np.array(self.eval_times)))
            print("sampling gen cond time avg: ", np.mean(np.array(self.gen_times)))
            print("sampling eval time std: ", np.std(np.array(self.eval_times)))
        return np.array(predictions)

    def avg_eval_time(self):
        return np.mean(np.array(self.eval_times))

    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        if self.sampling_percentage >= 1.00:
            sp = int(self.sampling_percentage)
        else:
            sp = self.sampling_percentage
        return self.__class__.__name__ + str(sp)

    def save_model(self, save_dir="./", suffix_name=""):
        pass


class OurPGM(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        self.min_groupby = 0
        self.kwargs = kwargs
        self.backend = kwargs["backend"]
        self.alg_name = kwargs["alg_name"]
        self.use_svd = kwargs["use_svd"]
        self.num_singular_vals = kwargs["num_singular_vals"]

        self.num_bins = kwargs["num_bins"]
        self.recompute = kwargs["recompute"]
        self.test_cache = {}
        self.column_bins = {}
        self.DEBUG = True
        self.eval_times = []

        # key: column name
        # val: dataframe, which represents a sorted (ascending, based on column
        # values) group by on the given column name.
        # this will store ALL the values in the given bin, so we can compute
        # precisely which fraction of a range query that partially overlaps
        # with the bin covers.
        # TODO: it may be enough to assume uniformity and store each unique
        # value / or store some other stats etc.
        self.column_bin_vals = {}

        self.DEBUG = False
        self.param_count = -1

        self.model = PGM(alg_name=self.alg_name, backend=self.backend,
                use_svd=self.use_svd, num_singular_vals=self.num_singular_vals,
                recompute=self.recompute)

    def __str__(self):
        name = self.alg_name
        if self.recompute:
            name += "-recomp"
        if self.use_svd:
            name += str(self.num_singular_vals)
        return name

    def _load_osm_data(self, db):
        start = time.time()
        # load directly to numpy since should be much faster
        data = np.fromfile('/data/pari/osm.bin',
                dtype=np.int64).reshape(-1, 6)
        # data = np.fromfile(OSM_FILE,
                # dtype=np.int64).reshape(-1, 6)
        columns = list(db.column_stats.keys())
        # drop the index column
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
            data[:, i] = np.digitize(d0, bins, right=True)
            print(len(set(d0)))
            print(min(d0))

        end = time.time()
        df = pd.DataFrame(data)
        df = df.groupby([0,1,2,3,4]).size().\
                sort_values(ascending=False).\
                reset_index(name='count')

        # FIXME: should not be hardcoded
        samples = df.values[:,0:5]
        weights = np.array(df["count"])

        print("took : ", end-start)
        # pdb.set_trace()
        return samples, weights

    def _load_training_data2(self, db):
        '''
        Should be a general purpose function that works on all single table
        cases, with both discrete and continuous columns.

        TODO: add sanity check asserts to make sure this is doing sensible
        things.
        '''
        start = time.time()
        # assert len(db.tables) == 1
        table = [t for t in db.tables][0]
        print(table)
        columns = list(db.column_stats.keys())
        FROM = table
        columns_str = ",".join(columns)
        # just select all of these columns from the given table
        select_all = "SELECT {COLS} FROM {TABLE};".format(COLS = columns_str,
                                                         TABLE = table)
        # TODO: add cache here.
        data_cache = klepto.archives.dir_archive("./misc_cache",
                cached=True, serialized=True)
        cache_key = select_all
        if cache_key in data_cache.archive:
            df = data_cache.archive[cache_key]
        else:
            cmd = 'postgresql://{}:{}@localhost:5432/{}'.format(db.user,
                    db.pwd, db.db_name)
            engine = create_engine(cmd)
            df = pd.read_sql_query(select_all, engine)
            data_cache.archive[cache_key] = df

        # now, df should contain all the raw columns. If there are continuous
        # columns, then we will need to bin them.
        df_keys = list(df.keys())
        for i, column in enumerate(columns):
            if db.column_stats[column]["num_values"] < 1000:
                print("{} is treated as discrete column".format(column))
                # not continuous
                continue
            _, bins = pd.qcut(df[df_keys[i]].values, self.num_bins,
                    retbins=True, duplicates="drop")
            self.column_bins[column] = bins
            df2 = df.sort_values(df_keys[i], ascending=True)
            df2 = df2.groupby(df_keys[i]).size().reset_index(name='count')
            assert df2.values[-1,0] == db.column_stats[column]["max_value"]
            assert df2.values[0,0] == db.column_stats[column]["min_value"]

            self.column_bin_vals[column] = df2

            df[df_keys[i]] = np.digitize(df[df_keys[i]].values, bins, right=True)
            print("{}, num bins: {}, min value: {}".format(column,
                len(set(df.values[:,i])), min(df.values[:,i])))

        headers = [k for k in df.keys()]
        df = df.groupby(headers).size().\
                sort_values(ascending=False).\
                reset_index(name='count')

        samples = df.values[:,0:-1]
        weights = np.array(df["count"])

        print("_load_training_data took {} seconds".format(time.time()-start))
        print("samples shape: ", samples.shape)
        print("weights shape: ", weights.shape)
        return samples, weights

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
        print("continuous cols: ", continuous_cols)

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
                    print(bin_cmd)
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
        print(group_by)
        group_by += " HAVING COUNT(*) > {}".format(self.min_groupby)

        groupby_output = db.execute(group_by)
        print("len groupby output: ", len(groupby_output))

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
            # samples, weights = self._load_training_data(db, True)
            # samples, weights = self._load_osm_data(db)
            samples, weights = self._load_training_data2(db)
        elif "imdb" in db.db_name:
            assert False
        elif "dmv" in db.db_name:
            samples, weights = self._load_training_data(db, False)
        elif "higgs" in db.db_name:
            samples, weights = self._load_training_data2(db)
        elif "power" in db.db_name:
            samples, weights = self._load_training_data2(db)
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

        NUM_PARAMS_NOT_IMPL = False
        if NUM_PARAMS_NOT_IMPL:
            # calculate it using pomegranate model
            model = BayesianNetwork.from_samples(samples, weights=weights,
                    state_names=columns, algorithm="chow-liu", n_jobs=-1)
            self.param_count = 0
            for state in model.states:
                dist = state.distribution.parameters[0]
                if isinstance(dist, list):
                    self.param_count += len(dist)*3
                elif isinstance(dist, dict):
                    self.param_count += len(dist)*2

    def avg_eval_time(self):
        return np.mean(np.array(self.eval_times))

    def num_parameters(self):
        # approximate it based on the db stats
        alph_sizes = []
        num_columns = len(self.db.column_stats)
        print(self.db.column_stats.keys())
        for k,stats in self.db.column_stats.items():
            num_vals = stats["num_values"]
            if num_vals > 1000:
                print("{} treated as continuous binned for param est".format(k))
                alph_sizes.append(self.num_bins)
            else:
                alph_sizes.append(num_vals)

        # self.use_svd = kwargs["use_svd"]
        # self.num_singular_vals = kwargs["num_singular_vals"]

        # self.num_bins = kwargs["num_bins"]
        # self.recompute = kwargs["recompute"]
        avg_alph_size = np.mean(np.array(alph_sizes))
        ind_prob_sizes = num_columns * avg_alph_size
        if self.use_svd and self.recompute:
            k = self.num_singular_vals
            edge_sizes = ((num_columns-1)**2)*k*(avg_alph_size)
        elif self.use_svd:
            k = self.num_singular_vals
            edge_sizes = ((num_columns-1))*k*(avg_alph_size)
        elif self.recompute:
            edge_sizes = ((num_columns-1)**2)*(avg_alph_size**2)
        else:
            # no svd or recompute
            edge_sizes = ((num_columns-1))*(avg_alph_size**2)
        return ind_prob_sizes + edge_sizes

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

            possible_vals, weights = get_possible_values(query, self.db,
                    self.column_bins, self.column_bin_vals)
            # TODO: add assertion checks on possible_vals, weights shape etc.

            # if no binning, then don't need weights
            start = time.time()
            if len(self.column_bins) == 0:
                est_sel = self.model.evaluate(possible_vals)
            else:
                est_sel = self.model.evaluate(possible_vals, weights)
                # est_sel = self.model.evaluate(possible_vals)
            end = time.time()
            self.eval_times.append(end-start)

            if self.DEBUG:
                true_sel = query.true_sel
                pg_sel = query.pg_count / query.total_count
                qerr = max(true_sel / est_sel, est_sel / true_sel)
                pg_qerr = max(pg_sel / true_sel, true_sel / pg_sel)
                if qerr > 4.00:
                    print(query)
                    print("est sel: {}, true sel: {},pg sel: {}, qerr: {},pg_qerr: {}"\
                            .format(est_sel, true_sel, pg_sel, qerr, pg_qerr))
                    pdb.set_trace()

            estimates.append(est_sel)
            self.test_cache[hashed_query] = est_sel
        return estimates

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

def qloss(yhat, ytrue, avg=True):

    epsilons = np.array([QERR_MIN_EPS]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    yhat = np.maximum(yhat, epsilons)

    # TODO: check this
    errors = np.maximum( (ytrue / yhat), (yhat / ytrue))
    if avg:
        error = np.sum(errors) / len(yhat)
    else:
        return errors

    return error


def qloss_torch(yhat, ytrue, avg=True):

    epsilons = to_variable([QERR_MIN_EPS]*len(yhat)).float()
    ytrue = torch.max(ytrue, epsilons)
    yhat = torch.max(yhat, epsilons)

    # TODO: check this
    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    if avg:
        error = errors.sum() / len(yhat)
    else:
        return errors

    return error

class NN1(CardinalityEstimationAlg):
    '''
    Default implementation of various neural network based methods.
    '''
    def __init__(self, *args, **kwargs):

        # TODO: make these all configurable
        self.feature_len = None
        # self.hidden_layer_multiple = 2.0
        self.feat_type = "dict_encoding"
        self.num_hidden_layers = kwargs["num_hidden_layers"]
        self.hidden_layer_multiple = kwargs["hidden_layer_multiple"]

        # as in the dl papers (not sure if this is needed)
        self.log_transform = True

        # TODO: configure other variables
        self.max_iter = kwargs["max_iter"]
        self.lr = kwargs["lr"]
        self.eval_iter = kwargs["eval_iter"]

    def train(self, db, training_samples, save_model=True,
            use_subqueries=False):
        self.db = db
        if use_subqueries:
            training_samples = get_all_subqueries(training_samples)
        # db.init_featurizer()
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
        # net = SimpleRegression(len(X[0]),
                # int(len(X[0])*self.hidden_layer_multiple), 1)

        net = SimpleRegression(len(X[0]),
                self.hidden_layer_multiple, 1,
                num_hidden_layers=self.num_hidden_layers)

        if save_model:
            make_dir("./models")
            model_path = "./models/" + "nn1" + str(deterministic_hash(query_str))[0:5]
            if os.path.exists(model_path):
                net.load_state_dict(torch.load(model_path))
                print("loaded trained model!")

        loss_func = qloss_torch
        print("feature len: ", len(X[0]))
        train_nn(net, X, Y, loss_func=loss_func, max_iter=self.max_iter,
                tfboard_dir=None, lr=0.0001, adaptive_lr=True,
                loss_threshold=1.00, mb_size=128, eval_iter=self.eval_iter)

        self.net = net

        if save_model:
            print("saved model path")
            torch.save(net.state_dict(), model_path)

    def _test(self, test_samples):
        X = []
        for sample in test_samples:
            X.append(self.db.get_features(sample))
        # just pass each sample through net and done!
        X = to_variable(X).float()

        if self.log_transform:
            pred = self.net(X)
            pred = pred.squeeze(1)
            # pred = pred.detach().numpy()
            pred = pred.cpu().detach().numpy()
            for i, p in enumerate(pred):
                pred[i] = (p*(self.maxy-self.miny)) + self.miny
                pred[i] = math.pow(10, -pred[i])
            return pred
        else:
            pred = self.net(X)
            pred = pred.squeeze(1)
        return pred.cpu().detach().numpy()

    def test(self, test_samples):
        '''
        '''
        # TODO: evaluate in batches of MAX_TEST_SIZE in order to avoid
        # overwhelming the gpu memory
        preds = np.zeros(0)
        MAX_TRAINING_SIZE = 1024
        for i in range(0,len(test_samples),MAX_TRAINING_SIZE):
            batch = test_samples[i:i+MAX_TRAINING_SIZE]
            batch_preds = self._test(batch)
            preds = np.append(preds, batch_preds)
        return preds

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
        self.jl_use_postgres = kwargs["jl_use_postgres"]

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

        # TODO: store these
        self.stats["model_params"] = {}

    def train(self, db, training_samples, use_subqueries=False,
            test_samples=None):
        self.db = db
        # db.init_featurizer()

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
                # self.table_y_train[i] = \
                    # to_variable(self.table_y_train[i]).float()
                self.table_y_train[i] = \
                    np.array(self.table_y_train[i])
                if test_samples:
                    queries = get_all_num_table_queries(test_samples, i)
                    for q in queries:
                        self.table_x_test[i].append(db.get_features(q))
                        self.table_y_test[i].append(q.true_sel)
                    self.table_x_test[i] = \
                        to_variable(self.table_x_test[i]).float()
                    # self.table_y_test[i] = \
                        # to_variable(self.table_y_test[i]).float()
                    self.table_y_test[i] = \
                        np.array(self.table_y_test[i])

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
            pred_table = pred_table.data.numpy()
            loss_train = qloss(pred_table, y_table, avg=False)
            if num_table not in self.stats["train"]["tables_eval"]["qerr"]:
                self.stats["train"]["tables_eval"]["qerr"][num_table] = {}
                self.stats["train"]["tables_eval"]["qerr-all"][num_table] = {}

            self.stats["train"]["tables_eval"]["qerr"][num_table][num_iter] = \
                    np.mean(loss_train)
            self.stats["train"]["tables_eval"]["qerr-all"][num_table][num_iter] = \
                    loss_train

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
            pred_table = pred_table.data.numpy()
            loss_test = qloss(pred_table, y_table, avg=False)

            if num_table not in self.stats["test"]["tables_eval"]["qerr"]:
                self.stats["test"]["tables_eval"]["qerr"][num_table] = {}
                self.stats["test"]["tables_eval"]["qerr-all"][num_table] = {}

            self.stats["test"]["tables_eval"]["qerr"][num_table][num_iter] = \
                    np.mean(loss_test)
            self.stats["test"]["tables_eval"]["qerr-all"][num_table][num_iter] = loss_test

            print("num_tables: {}, train_qerr: {}, test_qerr: {}, size: {}".format(\
                    num_table, np.mean(loss_train), np.mean(loss_test), len(y_table)))

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
                    self.baseline, self.jl_use_postgres)

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
            self.stats[key]["eval"]["join-loss-all"][num_iter] = jl1

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
        if reuse_env:
            env = park.make('query_optimizer')
        else:
            env = None

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
        elif self.sampling == "num_tables_weight":
            query_sampling_weights = None
            subquery_sampling_weights = []
            for si, sample in enumerate(training_samples):
                subquery_sampling_weights.append(1.00 / len(sample.froms))
                for sq in sample.subqueries:
                    subquery_sampling_weights.append(1.00 / len(sq.froms))

            subquery_sampling_weights = \
                    self.update_sampling_weights(subquery_sampling_weights)
            assert len(subquery_sampling_weights) == len(X)
        else:
            query_sampling_weights = None

        while True:
            if (num_iter % 100 == 0):
                # progress stuff
                print(num_iter, end=",")
                sys.stdout.flush()

            if (num_iter % self.eval_iter == 0):

                if not reuse_env and (num_iter % self.eval_iter_jl == 0):
                    assert env is None
                    env = park.make('query_optimizer')

                if self.eval_num_tables:
                    self._periodic_num_table_eval(loss_func, net, num_iter)

                join_losses, join_losses_ratio = self._periodic_eval(net, training_samples, X, Y,
                        env, "train", loss_func, num_iter, scheduler)
                if test_samples:
                    self._periodic_eval(net, test_samples, Xtest, Ytest,
                            env,"test", loss_func, num_iter, scheduler)

                if not reuse_env and (num_iter % self.eval_iter_jl == 0):
                    env.clean()
                    env = None

                # update query_sampling_wieghts if needed
                if query_sampling_weights is not None \
                        and join_losses is not None:
                    if self.adaptive_priority_alpha:
                        # temporary:
                        self.sampling_priority_alpha = num_iter / 4000
                        print("new priority alpha: ", self.sampling_priority_alpha)

                    if self.sampling_priority_method == "jl_ratio":
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
            elif self.sampling == "weighted_subquery" \
                    or self.sampling == "num_tables_weight":
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
                            baseline, self.jl_use_postgres)

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
        self.group_models = kwargs["group_models"]

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

        elif self.group_models < 0:
            if tables <= abs(self.group_models):
                return -1
            else:
                return 1
        else:
            return tables

    # same function for all the nns
    def _periodic_num_table_eval_nets(self, loss_func, num_iter):
        for num_table in self.samples:
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
            pred_table = pred_table.data.numpy()

            loss_trains = qloss(pred_table, y_table, avg=False)

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
            loss_test = qloss(pred_table, y_table, avg=False)
            if num_table not in self.stats["test"]["tables_eval"]["qerr"]:
                self.stats["test"]["tables_eval"]["qerr"][num_table] = {}
                self.stats["test"]["tables_eval"]["qerr-all"][num_table] = {}

            self.stats["test"]["tables_eval"]["qerr"][num_table][num_iter] = \
                np.mean(loss_test)
            self.stats["test"]["tables_eval"]["qerr-all"][num_table][num_iter] = \
                loss_test

            print("num_tables: {}, train_qerr: {}, test_qerr: {}, size: {}".format(\
                    num_table, np.mean(loss_trains), np.mean(loss_test), len(y_table)))

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
                    baseline, self.jl_use_postgres)

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
            if num_tables == -1:
                continue

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
        print("num tables in samples: ", len(self.samples))

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

                # if (num_iter % self.eval_iter == 0
                        # and num_iter != 0):
                if (num_iter % self.eval_iter == 0):

                    if self.eval_num_tables:
                        self._periodic_num_table_eval_nets(self.loss_func, num_iter)

                    # evaluation code
                    if (num_iter % self.eval_iter_jl == 0 \
                            and not self.reuse_env):
                        assert env is None
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
            num_real_tables = len(db.aliases)
            for i in range(1,num_real_tables+1):
                queries = get_all_num_table_queries(training_samples, i)
                num_tables_map = self.map_num_tables(i)
                if num_tables_map == -1:
                    continue
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

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
from utils.net import SimpleRegression
from cardinality_estimation.losses import *
import pandas as pd
import json
from multiprocessing import Pool
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
from sqlalchemy import create_engine

# FIXME: temporary hack
NULL_VALUE = "-1"
OSM_FILE = '/Users/pari/db_data/osm.bin'

def _get_bin_length(bins, bin_num):
    if bin_num == 0:
        bin_length = 0
    else:
        bin_length = bins[bin_num] - bins[bin_num-1]

    return bin_length

def get_possible_values(sample, db, column_bins=None):
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
                    # print(column)
                    # print("was not in the binned columns")
                    # pdb.set_trace()
                    val = [float(v) for v in val]

                    # FIXME: this should only cover the discrete values in this
                    # column which are in between val[0] and val[1]
                    # for ival in range(val[0], val[1]):
                        # possible_vals.append(str(ival))
                        # weights.append(1.00)

                    for v in db.column_stats[column]["unique_values"]:
                        v = v[0]
                        if v >= val[0] and v <= val[1]:
                            possible_vals.append(v)
                            weights.append(1.00)
                else:
                    # discretize first
                    bins = column_bins[column]
                    vals = [float(v) for v in val]
                    binned_vals = np.digitize(vals, bins, right=True)

                    if (binned_vals[0] == binned_vals[1]):
                        possible_vals.append(binned_vals[0])
                        bin_length = _get_bin_length(bins, binned_vals[0])
                        if bin_length != 0.0:
                            weight = (vals[1] - vals[0]) / (bin_length)
                        else:
                            weight = 1.00
                        weights.append(weight)
                    else:
                        # different bins, means the weight can be different for
                        # the endpoints, and 1.00 for all the middle ones
                        for bi in range(binned_vals[0],binned_vals[1]+1):
                            possible_vals.append(bi)
                            bin_length = _get_bin_length(bins, bi)
                            if bin_length == 0.0:
                                weight = 1.00
                            elif bi == binned_vals[0]:
                                assert bi != 0
                                weight = (vals[0] - bins[bi-1]) / bin_length
                            elif bi == binned_vals[1]:
                                assert bi != 0
                                weight = (vals[1] - bins[bi-1]) / bin_length
                            else:
                                weight = 1.00
                            weights.append(weight)

                    # for bi in range(binned_vals[0],binned_vals[1]+1):
                        # possible_vals.append(bi)

                    # print(bins)
                    # print(vals)
                    # print(binned_vals)
                    # pdb.set_trace()

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
        self.use_svd = kwargs["use_svd"]
        self.num_singular_vals = kwargs["num_singular_vals"]
        self.model = PGM(alg_name=self.alg_name, backend=self.backend,
                use_svd=self.use_svd, num_singular_vals=self.num_singular_vals)

        self.num_bins = 100
        self.test_cache = {}
        self.column_bins = {}
        self.DEBUG = False
        self.param_count = -1

    def __str__(self):
        name = self.alg_name
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
        # data = data[0:100000,1:6]
        data = data[:,1:6]
        self.column_bins = {}
        for i in range(data.shape[1]):
            # these columns don't need to be discretized.
            # FIXME: use more general check here.
            if db.column_stats[columns[i]]["num_values"] < 1000:
                continue
            d0 = data[:, i]
            _, bins = pd.qcut(d0, self.num_bins, retbins=True, duplicates="drop")
            # _, bins = pd.qcut(d0, self.num_bins, retbins=True)
            self.column_bins[columns[i]] = bins
            data[:, i] = np.digitize(d0, bins, right=True)
            print(len(set(d0)))
            print(min(d0))
            # pdb.set_trace()

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
        '''
        start = time.time()
        assert len(db.tables) == 1
        table = [t for t in db.tables][0]
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
        # if False:
            df = data_cache.archive[cache_key]
            # samples = df.values[:,0:-1]
            # weights = np.array(df["count"])
            # print("loading training data from cache took {} seconds".format(\
                    # time.time() - start))
            # return samples, weights
        else:
            cmd = 'postgresql://{}:{}@localhost:5432/{}'.format(db.user,
                    db.pwd, db.db_name)
            engine = create_engine(cmd)
            df = pd.read_sql_query(select_all, engine)
            data_cache.archive[cache_key] = df

        # now, df should contain all the raw columns. If there are continuous
        # columns, then we will need to bin them.
        for i, column in enumerate(columns):
            if db.column_stats[column]["num_values"] < 1000:
                print("{} is treated as discrete column".format(column))
                # not continuous
                continue
            _, bins = pd.qcut(df.values[:,i], self.num_bins,
                    retbins=True, duplicates="drop")
            self.column_bins[column] = bins
            df.values[:,i] = np.digitize(df.values[:,i], bins, right=True)
            print("{}, bins: {}, min value: {}".format(column,
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
            samples, weights = self._load_osm_data(db)
            # samples, weights = self._load_training_data2(db)
        elif "imdb" in db.db_name:
            assert False
        elif "dmv" in db.db_name:
            samples, weights = self._load_training_data(db, False)
        elif "higgs" in db.db_name:
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

    def num_parameters(self):
        return self.model.num_parameters()

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
                    self.column_bins)
            # TODO: add assertion checks on possible_vals, weights shape etc.

            # print(possible_vals)
            # print(weights)
            # pdb.set_trace()

            # if no binning, then don't need weights
            if len(self.column_bins) == 0:
                est_sel = self.model.evaluate(possible_vals)
            else:
                est_sel = self.model.evaluate(possible_vals, weights)

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

    def get_possible_values(self, sample, db, column_bins=None):
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

                if cmp_op == "in":
                    # dedup
                    if hasattr(sample.vals[i], "__len__"):
                        val = set(val)
                    # FIXME: something with the osm dataset
                    possible_vals = [str(v.replace("'","")) for v in val]
                    # possible_vals = [v for v in val]
                    # possible_vals = [int(v) for v in val]
                elif cmp_op == "lt":
                    if column not in column_bins:
                        # then select everything in the given range of
                        # integers.
                        val = [int(v) for v in val]
                        for ival in range(val[0], val[1]):
                            possible_vals.append(str(ival))
                            # possible_vals.append(ival)
                    else:
                        # discretize first
                        bins = column_bins[column]
                        vals = [float(v) for v in val]
                        vals = np.digitize(vals, bins, right=True)
                        for bi in range(vals[0],vals[1]+1):
                            possible_vals.append(str(bi+1))
                            # possible_vals.append(bi)

                elif cmp_op == "eq":
                    possible_vals.append(val)
                else:
                    assert False
            all_possible_vals.append(possible_vals)
        return all_possible_vals

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
        print("load osm model!")
        data = np.fromfile(OSM_FILE,
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
            model_sample = self.get_possible_values(query, self.db,
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
                    print("going to evaluate")
                    # pdb.set_trace()
                    # est_sel = np.sum(self.model.probability(all_points))
                    est_sel = 0.00
                    for p in all_points:
                        try:
                            est_sel += self.model.probability(p)
                        except Exception as e:
                            # FIXME: add minimum amount.
                            print(e)
                            continue
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

    # tmp testing.
    # if random.random() < 0.5:
        # error * 3.0
    # else:
        # error * 0.1

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
        self.log_transform = False

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
        # loss_func = rel_loss_torch
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
            pred = pred.detach().numpy()
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
            self.adaptive_lr_patience = 10

        self.divide_mb_len = kwargs["divide_mb_len"]
        self.lr = kwargs["lr"]
        self.jl_start_iter = kwargs["jl_start_iter"]
        self.num_hidden_layers = kwargs["num_hidden_layers"]
        self.hidden_layer_multiple = kwargs["hidden_layer_multiple"]
        self.eval_iter = kwargs["eval_iter"]
        self.optimizer_name = kwargs["optimizer_name"]

        self.clip_gradient = kwargs["clip_gradient"]
        self.rel_qerr_loss = kwargs["rel_qerr_loss"]
        self.adaptive_lr = kwargs["adaptive_lr"]
        self.baseline = kwargs["baseline"]
        nn_cache_dir = kwargs["nn_cache_dir"]
        # caching related stuff
        self.training_cache = klepto.archives.dir_archive(nn_cache_dir,
                cached=True, serialized=True)
        # will keep storing all the training information in the cache / and
        # dump it at the end
        # self.key = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        dt = datetime.datetime.now()
        # self.key = str(dt.day) + str(dt.hour) + str(dt.minute) + str(dt.second)
        self.key = "{}-{}-{}-{}".format(dt.day, dt.hour, dt.minute, dt.second)

        self.stats = {}
        self.training_cache[self.key] = self.stats

        # all the configuration parameters are specified here
        self.stats["kwargs"] = kwargs

        # iteration : value
        self.stats["gradients"] = {}
        self.stats["lr"] = {}

        # iteration : value + additional stuff, like query-string : sql
        self.stats["mb-loss"] = {}

        # iteration: qerr: val, jloss: val
        self.stats["eval"] = {}
        self.stats["eval"]["qerr"] = {}
        self.stats["eval"]["join-loss"] = {}

        self.stats["model_params"] = {}

    def train(self, db, training_samples, use_subqueries=False):
        self.db = db
        # db.init_featurizer()

        # initialize samples
        for sample in training_samples:
            features = db.get_features(sample)
            sample.features = features
            for subq in sample.subqueries:
                subq_features = db.get_features(subq)
                subq.features = subq_features
        print("feature len: ", len(features))

        # do training
        net = SimpleRegression(len(features),
                self.hidden_layer_multiple, 1,
                num_hidden_layers=self.num_hidden_layers)
        loss_func = qloss_torch

        self.net = net
        try:
            self._train_nn_join_loss(self.net, training_samples, self.lr,
                    self.jl_start_iter,
                    loss_func=loss_func, max_iter=self.max_iter, tfboard_dir=None,
                    loss_threshold=2.0, jl_variant=self.jl_variant,
                    eval_iter_jl=self.eval_iter, clip_gradient=self.clip_gradient,
                    rel_qerr_loss=self.rel_qerr_loss,
                    adaptive_lr=self.adaptive_lr)
        except KeyboardInterrupt:
            print("keyboard interrupt")
        except park.envs.query_optimizer.query_optimizer.QueryOptError:
            print("park exception")

        self.training_cache.dump()
        # just continue as normal, go on to evaluate algorithm etc.

    def _train_nn_join_loss(self, net, training_samples,
            lr, jl_start_iter, max_iter=10000, eval_iter_jl=500,
            eval_iter_qerr=100, mb_size=1,
            loss_func=None, tfboard_dir=None, adaptive_lr=True,
            min_lr=1e-17, loss_threshold=1.0, jl_variant=False,
            clip_gradient=10.00, rel_qerr_loss=True):
        '''
        TODO: explain and generalize.
        '''
        if loss_func is None:
            assert False
            loss_func = torch.nn.MSELoss()

        # results = defaultdict(list)

        if self.optimizer_name == "ams":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                    amsgrad=True)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                    amsgrad=False)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        else:
            assert False

        # update learning rate
        if adaptive_lr:
            scheduler = ReduceLROnPlateau(optimizer, 'min',
                    patience=self.adaptive_lr_patience,
                            verbose=True, factor=0.1, eps=min_lr)

        num_iter = 0
        # create a new park env, and close at the end.
        env = park.make('query_optimizer')

        # TODO: figure out how to put everything on the gpu before starting

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

        min_qerr = {}
        max_qerr = {}

        file_name = "./training-" + self.__str__() + ".dict"
        start = time.time()
        while True:

            if (num_iter % 100 == 0):
                # progress stuff
                print(num_iter, end=",")
                sys.stdout.flush()

            if (num_iter % eval_iter_qerr == 0):
                # evaluate qerr

                pred = net(X)
                pred = pred.squeeze(1)
                train_loss = loss_func(pred, Y)
                self.stats["eval"]["qerr"][num_iter] = train_loss.item()
                # print("\nnum iter: {}, num samples: {}, loss: {}".format(
                    # num_iter, len(X), train_loss.item()))

                if not jl_variant and adaptive_lr:
                    # FIXME: should we do this for minibatch / or for train loss?
                    scheduler.step(train_loss)

                if (num_iter % eval_iter_jl == 0):
                    jl_eval_start = time.time()
                    jl = join_loss_nn(pred, training_samples, self, env,
                            baseline=self.baseline)
                    jl = np.mean(np.array(jl))

                    if jl_variant and adaptive_lr:
                        scheduler.step(jl)

                    self.stats["eval"]["join-loss"][num_iter] = jl

                    print("""\nnum iter: {}, num samples: {}, loss: {},join-loss {}, time: {}""".format(
                        num_iter, len(X), train_loss.item(), jl,
                        time.time()-jl_eval_start))

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

            pred = net(xbatch)
            pred = pred.squeeze(1)
            loss = loss_func(pred, ybatch)

            # FIXME: temporary, to try and adjust for the variable batch size.
            if self.divide_mb_len:
                loss /= len(pred)

            if (num_iter > jl_start_iter and \
                    rel_qerr_loss):
                # set the max qerrs for each query
                assert len(cur_samples) == 1
                sample_key = deterministic_hash(cur_samples[0].query)
                if sample_key not in max_qerr:
                    max_qerr[sample_key] = np.array(loss.item())


            if (num_iter > jl_start_iter and jl_variant):
                if jl_variant == 3:
                    use_pg_est = True
                else:
                    use_pg_est = False

                jl = join_loss_nn(pred, mb_samples, self, env,
                        baseline=self.baseline, use_pg_est=use_pg_est)

                if jl_variant == 1:
                    jl = torch.mean(to_variable(jl).float())
                elif jl_variant == 2:
                    jl = torch.mean(to_variable(jl).float()) - 1.00
                elif jl_variant == 3:
                    # using postgres values for join loss
                    jl = torch.mean(to_variable(jl).float())
                else:
                    assert False


                if rel_qerr_loss:
                    sample_key = deterministic_hash(cur_samples[0].query)
                    loss = loss / to_variable(max_qerr[sample_key]).float()

                loss = loss*jl

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


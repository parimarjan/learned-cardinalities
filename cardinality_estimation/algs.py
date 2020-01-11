import time
import numpy as np
import pdb
import math
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
import datetime

# sentinel value for NULLS
NULL_VALUE = "-1"

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass

    def train(self, db, training_samples, **kwargs):
        pass
    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subquery). Each key should be ' ' separated
        list of aliases / table names
        '''
        pass

    def get_exp_name(self):
        name = self.__str__()
        return name

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
    # def __init__(self, num_tables_true=0, regex_true=False):
        # pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                # alias_key = ' '.join(alias)
                pred_dict[(alias_key)] = info["cardinality"]["expected"]
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "postgres"

class TrueCardinalities(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                pred_dict[(alias_key)] = info["cardinality"]["actual"]
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true"

class TrueRank(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            all_cards = []
            for alias_key, info in sample["subset_graph"].nodes().items():
                # pred_dict[(alias_key)] = info["cardinality"]["actual"]
                card = info["cardinality"]["actual"]
                exp = info["cardinality"]["expected"]
                all_cards.append([alias_key, card, exp])
            all_cards.sort(key = lambda x : x[1])

            for i, (alias_key, true_est, pgest) in enumerate(all_cards):
                if i == 0:
                    pred_dict[(alias_key)] = pgest
                    continue
                prev_est = all_cards[i-1][2]
                prev_alias = all_cards[i-1][0]
                if pgest >= prev_est:
                    pred_dict[(alias_key)] = pgest
                else:
                    updated_est = prev_est
                    # updated_est = prev_est + 1000
                    # updated_est = true_est
                    all_cards[i][2] = updated_est
                    pred_dict[(alias_key)] = updated_est

            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_rank"

class TrueRankTables(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            all_cards_nt = defaultdict(list)
            for alias_key, info in sample["subset_graph"].nodes().items():
                # pred_dict[(alias_key)] = info["cardinality"]["actual"]
                card = info["cardinality"]["actual"]
                exp = info["cardinality"]["expected"]
                nt = len(alias_key)
                all_cards_nt[nt].append([alias_key,card,exp])

            for _,all_cards in all_cards_nt.items():
                all_cards.sort(key = lambda x : x[1])
                for i, (alias_key, true_est, pgest) in enumerate(all_cards):
                    if i == 0:
                        pred_dict[(alias_key)] = pgest
                        continue
                    prev_est = all_cards[i-1][2]
                    prev_alias = all_cards[i-1][0]
                    if pgest >= prev_est:
                        pred_dict[(alias_key)] = pgest
                    else:
                        updated_est = prev_est
                        # updated_est = prev_est + 1000
                        # updated_est = true_est
                        all_cards[i][2] = updated_est
                        pred_dict[(alias_key)] = updated_est

            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_rank_tables"

class Random(CardinalityEstimationAlg):
    def test(self, test_samples):
        # TODO: needs to go over all subqueries
        return np.array([random.random() for _ in test_samples])

class Sampling(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        self.sampling_percentage = kwargs["sampling_percentage"]
        self.eval_times = []
        self.gen_times = []

    def train(self, db, training_samples, **kwargs):
        # FIXME: this should be a utility function, also used in
        # _load_training_data_single_table
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
    return errors

def rel_loss(pred, ytrue, avg=True):
    '''
    Loss function for neural network training. Should use the
    compute_relative_loss formula, but deal with appropriate pytorch types.
    '''
    # this part is the same for both rho_est, or directly selectivity
    # estimation cases
    assert len(pred) == len(ytrue)
    epsilons = np.array([REL_LOSS_EPSILON]*len(pred))
    errors = np.abs(pred-ytrue) / (np.maximum(epsilons, ytrue))
    if avg:
        error = (np.sum(errors) / len(pred))
    else:
        error = errors
    return error

def qloss(yhat, ytrue, avg=True):
    '''
    numpy version.
    '''

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

def qloss_torch(yhat, ytrue):
    assert yhat.shape == ytrue.shape

    epsilons = to_variable([QERR_MIN_EPS]*len(yhat)).float()

    ytrue = torch.max(ytrue, epsilons)
    yhat = torch.max(yhat, epsilons)

    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    return errors

def abs_loss(yhat, ytrue, avg=True):
    '''
    numpy version.
    '''
    # ytrue = np.maximum(ytrue, epsilons)
    # yhat = np.maximum(yhat, epsilons)

    # TODO: check this
    errors = np.absolute(ytrue - yhat)
    if avg:
        error = np.sum(errors) / len(yhat)
    else:
        return errors

    return error

def abs_loss_torch(yhat, ytrue):
    '''
    numpy version.
    '''
    errors = torch.abs(ytrue - yhat)
    return errors

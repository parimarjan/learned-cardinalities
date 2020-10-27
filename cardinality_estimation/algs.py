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
# import park

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
from sklearn.ensemble import RandomForestRegressor
from .custom_linear import CustomLinearModel
from sqlalchemy import create_engine
import datetime

# from cardinality_estimation.query_dataset import QueryDataset
# from cardinality_estimation.nn import NN

# from scripts.get_query_cardinalities import CROSS_JOIN_CONSTANT

# TIMEOUT_COUNT_CONSTANT = 150001000001
# CROSS_JOIN_CONSTANT = 150001000000
# EXCEPTION_COUNT_CONSTANT = 150001000002

TIMEOUT_COUNT_CONSTANT = 150001000001
CROSS_JOIN_CONSTANT = 150001000000
EXCEPTION_COUNT_CONSTANT = 150001000002

# sentinel value for NULLS
NULL_VALUE = "-1"

WJ_TIMES = {}
WJ_TIMES["1a"] = 0.25
WJ_TIMES["2a"] = 0.5
WJ_TIMES["2b"] = 1.0
WJ_TIMES["2c"] = 0.5
WJ_TIMES["3a"] = 0.25
WJ_TIMES["4a"] = 0.1
WJ_TIMES["5a"] = 0.1
WJ_TIMES["6a"] = 1.0
WJ_TIMES["7a"] = 10.0
WJ_TIMES["8a"] = 5.0
WJ_TIMES["9a"] = 5.0
WJ_TIMES["9b"] = 5.0
WJ_TIMES["10a"] = 5.0
WJ_TIMES["11a"] = 5.0
WJ_TIMES["11b"] = 5.0
WJ_TIMES["3b"] = 5.0

WJ_TIMES0 = {}
WJ_TIMES0["1a"] = 0.12
WJ_TIMES0["2a"] = 0.25
WJ_TIMES0["2b"] = 0.5
WJ_TIMES0["2c"] = 0.25
WJ_TIMES0["3a"] = 0.12
WJ_TIMES0["4a"] = 0.05
WJ_TIMES0["5a"] = 0.05
WJ_TIMES0["6a"] = 0.5
WJ_TIMES0["7a"] = 5.0
WJ_TIMES0["8a"] = 2.5
WJ_TIMES0["9a"] = 5.0
WJ_TIMES0["9b"] = 5.0
WJ_TIMES0["10a"] = 5.0
WJ_TIMES0["11a"] = 5.0
WJ_TIMES0["11b"] = 5.0
WJ_TIMES0["3b"] = 5.0

def get_wj_times_dict(wj_key):
    if wj_key == "wanderjoin":
        return WJ_TIMES
    elif wj_key == "wanderjoin0.5":
        return WJ_TIMES0
    elif wj_key == "wanderjoin2":
        return None

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass

    def train(self, db, training_samples, **kwargs):
        if db.db_name == "so":
            global SOURCE_NODE
            SOURCE_NODE = tuple(["SOURCE"])

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

class SavedPreds(CardinalityEstimationAlg):

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        self.model_dir = kwargs["model_dir"]
        self.max_epochs = 0

    def train(self, db, training_samples, **kwargs):
        if db.db_name == "so":
            global SOURCE_NODE
            SOURCE_NODE = tuple(["SOURCE"])

        assert os.path.exists(self.model_dir)
        self.saved_preds = load_object_gzip(self.model_dir + "/preds.pkl")
        # self.saved_preds = load_object(self.model_dir + "/preds.pkl")

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subquery). Each key should be ' ' separated
        list of aliases / table names
        '''
        preds = []
        for sample in test_samples:
            # assert sample["name"] in self.saved_preds
            if sample["name"] not in self.saved_preds:
                print(sample["name"])
                pdb.set_trace()
            preds.append(self.saved_preds[sample["name"]])
        return preds

    def get_exp_name(self):
        old_name = os.path.basename(self.model_dir)
        name = "SavedRun-" + old_name
        return name

    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        return "SavedAlg"

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
            nodes = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            for alias_key in nodes:
                info = sample["subset_graph"].nodes()[alias_key]
                true_card = info["cardinality"]["actual"]
                est = info["cardinality"]["expected"]

                if "expected" not in info["cardinality"]:
                    print("no find expected :(")
                    pdb.set_trace()

                # if true_card >= CROSS_JOIN_CONSTANT:
                    # est = true_card
                # else:
                    # est = info["cardinality"]["expected"]

                # if est > EXCEPTION_COUNT_CONSTANT:
                    # pdb.set_trace()
                    # est = TIMEOUT_COUNT_CONSTANT

                # if est > EXCEPTION_COUNT_CONSTANT:
                    # print(est)
                    # print("postgres estimate > cj constant")
                    # pdb.set_trace()

                pred_dict[(alias_key)] = est
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "postgres"

class SamplingTables(CardinalityEstimationAlg):
    def __init__(self, sampling_key):
        self.sampling_key = sampling_key
        # dict with times, used if key is only wanderjoin
        # self.sampling_times = WJ_TIMES

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        bad_ests = 0
        total = 0
        preds = []
        if self.sampling_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
            wj_times = get_wj_times_dict(self.sampling_key)
        else:
            wj_times = None

        for sample in test_samples:
            pred_dict = {}
            if wj_times is not None:
                sk = "wanderjoin-" + str(wj_times[sample["template_name"]])
            else:
                sk = self.sampling_key

            for alias_key, info in sample["subset_graph"].nodes().items():
                if alias_key == SOURCE_NODE:
                    continue
                total += 1
                cards = info["cardinality"]
                if sk in cards:
                    cur_est = cards[sk]
                else:
                    print(sk)
                    print(cards.keys())
                    pdb.set_trace()
                    assert False

                # if "ci" in alias_key and "n" in alias_key and len(alias_key) == 2:
                    # print(alias_key, "est: " + str(cur_est), cards["actual"])
                    # pdb.set_trace()

                if cur_est == 0 or cur_est == 1:
                # if cur_est == 0:
                    bad_ests += 1
                    cur_est = cards["expected"]

                if cur_est == 0:
                    cur_est += 1
                pred_dict[(alias_key)] = cur_est
            preds.append(pred_dict)
        print("bad ests: {}, total: {}".format(bad_ests, total))
        # print("set failed ests to actual est")
        return preds

    def __str__(self):
        return self.sampling_key

class SamplingTablesOld(CardinalityEstimationAlg):
    def __init__(self, sampling_type, sampling_percentage):
        self.sampling_type = sampling_type
        self.sampling_percentage = sampling_percentage
        self.sampling_key = sampling_type + str(sampling_percentage) + "_" + "actual"

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                cards = info["cardinality"]
                if self.sampling_key in cards:
                    cur_est = cards[self.sampling_key]
                else:
                    print("key not found: ", self.sampling_key)
                    pdb.set_trace()
                    # cur_est = cards["actual"]
                if cur_est == 0:
                    cur_est += 1
                pred_dict[(alias_key)] = cur_est
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "sampling-tables"

class TrueCardinalities(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            nodes = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            for alias_key in nodes:
                info = sample["subset_graph"].nodes()[alias_key]
                pred_dict[(alias_key)] = info["cardinality"]["actual"]
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true"

class TrueRandom(CardinalityEstimationAlg):
    def __init__(self):
        # max percentage noise added / subtracted to true values
        self.max_noise = random.randint(1,500)

    def test(self, test_samples):
        # choose noise type

        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                true_card = info["cardinality"]["actual"]
                # add noise
                noise_perc = random.randint(1,self.max_noise)
                noise = (true_card * noise_perc) / 100.00
                if random.random() % 2 == 0:
                    updated_card = true_card + noise
                else:
                    updated_card = true_card - noise
                if updated_card <= 0:
                    updated_card = 1
                pred_dict[(alias_key)] = updated_card
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_random"

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
    # def test(self, test_samples):
        # # TODO: needs to go over all subqueries
        # return np.array([random.random() for _ in test_samples])

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                total = info["cardinality"]["total"]
                est = random.random()*total
                pred_dict[(alias_key)] = est
            preds.append(pred_dict)
        return preds

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

def qloss(yhat, ytrue):
    '''
    numpy version.
    '''

    epsilons = np.array([QERR_MIN_EPS]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    yhat = np.maximum(yhat, epsilons)

    # TODO: check this
    errors = np.maximum( (ytrue / yhat), (yhat / ytrue))
    return errors

def ll_scaled_norm_loss(preds, targets):
    # preds = preds.squeeze(1)
    assert preds.shape == targets.shape
    losses = torch.zeros_like(preds)
    mask = (targets > -1) & (targets < 1)
    losses[mask] = (preds[mask] - targets[mask]) ** 2
    factor = preds[~mask] / targets[~mask]
    losses[~mask] = 0.5 * (factor ** 2) - factor + .5
    return losses
    # return torch.mean(losses)

def qloss_torch(yhat, ytrue):
    assert yhat.shape == ytrue.shape

    epsilons = to_variable([QERR_MIN_EPS]*len(yhat)).float()

    ytrue = torch.max(ytrue, epsilons)
    yhat = torch.max(yhat, epsilons)

    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    return errors

# def mse_torch(yhat, ytrue, min_qerr=1.0):

    # mse_losses = torch.nn.MSELoss(reduction="none")(yhat, ytrue)
    # if min_qerr == 1.0:
        # return mse_losses
    # qerr =


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

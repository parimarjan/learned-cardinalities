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
from utils.transformer import *

from cardinality_estimation.losses import *
import pandas as pd
import json
import multiprocessing
# from torch.multiprocessing import Pool as Pool2
# from utils.tf_summaries import TensorboardSummaries
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from multiprocessing.pool import ThreadPool

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
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
from .algs import *
import sys
import gc
import psutil
import copy
import cvxpy as cp
import networkx as nx

# dataset
from cardinality_estimation.query_dataset import QueryDataset
from torch.utils import data
from cardinality_estimation.join_loss import JoinLoss, PlanError
from torch.multiprocessing import Pool as Pool2
import torch.multiprocessing as mp
# import torch.multiprocessing

# mp.set_sharing_strategy('file_system')
import resource

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn")
except:
    pass
# mp.set_start_method("spawn")

# torch.multiprocessing.set_start_method('spawn')

from cardinality_estimation.flow_loss import FlowLoss, get_optimization_variables

# def collate_fn(sample):
    # return sample[0]

DEBUG_BATCH=False
def collate_fn_set_batch(batch):
    collated = []
    for i in range(len(batch[0])):
        if i == 8:
            infos = []
            for b in batch:
                infos.append(b[8])
            collated.append(infos)
        # elif i == 7:
            # # Y
            # cur_batch = [b[i] for b in batch]
            # collated.append(torch.stack(cur_batch))
        else:
            cur_batch = [b[i] for b in batch]
            collated.append(torch.cat(cur_batch, dim=0))
            # try:
                # if not len(cur_batch[0]) == 0:
                    # collated.append(cur_batch)
                # else:
                    # collated.append(torch.cat(cur_batch, dim=0))
            # except:
                # print(cur_batch)
                # pdb.set_trace()

    # print(collated[0].shape, collated[1].shape, collated[2].shape)
    return collated

def collate_fn_combined(batch):
    collated = []
    for i in range(len(batch[0])):
        if i == 2:
            infos = []
            for b in batch:
                infos.append(b[2])
            collated.append(infos)
        else:
            cur_batch = [b[i] for b in batch]
            collated.append(torch.cat(cur_batch, dim=0))

    return collated

def collate_fn(batch):
    # print(batch)
    # y = [b[6] for b in batch]
    # pdb.set_trace()
    return tuple(zip(*batch))

def collate_fn_set(batch):
    batch = batch[0]

    pdb.set_trace()
    # return batch[0]
    # print("collate fn set!")
    # print(len(batch))
    # pdb.set_trace()
    # return tuple(zip(*batch))

# once we have stored them in archive, parallel just slows down stuff
UPDATE_TOLERANCES_PAR = False
USE_TOLERANCES = False

def update_samples(samples, flow_features, cost_model,
        debug_set, db_name, db_year):
    global SOURCE_NODE
    if db_name == "so":
        SOURCE_NODE = tuple(["SOURCE"])
    cardinality_key = str(db_year) + "cardinality"
    REGEN_COSTS = True
    subq_hash_opt_path = {}

    if REGEN_COSTS:
        print("going to regenerate {} estimates for all samples".format(cost_model))
    # FIXME: need to use correct cost_model here
    start = time.time()
    new_seen = False

    for sample in samples:
        subsetg = sample["subset_graph"]
        # if SOURCE_NODE not in subsetg.nodes():
            # print("SOURCE NODE {} not in graph".format(SOURCE_NODE))
        add_single_node_edges(subsetg, SOURCE_NODE)
        sample_edge = list(subsetg.edges())[0]
        # if (cost_model + "cost" in subsetg.edges()[sample_edge].keys() \
                # and not debug_set) and not REGEN_COSTS:
        if (cost_model + "cost" in subsetg.edges()[sample_edge].keys()) \
                and not REGEN_COSTS:
            continue
        else:
            # print(sample["name"])
            # print("new sample in update sample")
            # print(subsetg.edges()[sample_edge].keys())
            # pdb.set_trace()

            new_seen = True

            pg_total_cost = compute_costs(subsetg, cost_model,
                    cardinality_key,
                    cost_key="pg_cost", ests="expected")
            _ = compute_costs(subsetg, cost_model,
                    cardinality_key,
                    cost_key="cost",
                    ests=None)

            subsetg.graph[cost_model + "total_cost"] = pg_total_cost

            final_node = [n for n,d in subsetg.in_degree() if d==0][0]

            pg_path = nx.shortest_path(subsetg, final_node, SOURCE_NODE,
                    weight=cost_model+"pg_cost")

            # for node in pg_path:
                # subsetg.nodes()[node][cost_model + "pg_path"] = 1

            opt_path = nx.shortest_path(subsetg, final_node, SOURCE_NODE,
                    weight=cost_model+"cost")

            all_nodes = list(subsetg.nodes())
            for node in all_nodes:
                subsql_hash = deterministic_hash(sample["sql"] + str(node))
                if node in opt_path:
                    subsetg.nodes()[node][cost_model + "opt_path"] = 1
                    subq_hash_opt_path[subsql_hash] = 1
                else:
                    subsetg.nodes()[node][cost_model + "opt_path"] = 0
                    subq_hash_opt_path[subsql_hash] = 0

                if node in pg_path:
                    subsetg.nodes()[node][cost_model + "pg_path"] = 1
                else:
                    subsetg.nodes()[node][cost_model + "pg_path"] = 0

    if not new_seen:
        return

    save_or_update("subq_hash_opt_path.pkl", subq_hash_opt_path)
    num_proc = 16

    if USE_TOLERANCES:
        if UPDATE_TOLERANCES_PAR:
            par_args = []
            for i in range(len(samples)):
                par_args.append((samples[i], "expected", cost_model+"pg_cost"))

            with Pool(processes = num_proc) as pool:
                res = pool.starmap(get_subq_tolerances, par_args)
        else:
            res = []
            tcache = klepto.archives.dir_archive("./tolerances_cache",
                    cached=True, serialized=True)
            new_seen = False
            tcache.load()
            print("loaded tcache: ", time.time()-start)
            # key = deterministic_hash(card_key + cost_key + qrep["sql"])
            for qrep in samples:
                key = deterministic_hash("expected" + "pg_cost" + qrep["sql"])
                if key in tcache:
                    res.append(tcache[key])
                else:
                    print(qrep["template_name"])
                    print("!!sample not found in tolerances cache!!")
                    pdb.set_trace()
                    new_seen = True
                    res.append(get_subq_tolerances(qrep, "expected", "pg_cost"))
                    tcache[key] = res[-1]
            if new_seen:
                tcache.dump()

        for i in range(len(samples)):
            tolerances = res[i]
            subsetg = samples[i]["subset_graph"]
            nodes = list(subsetg.nodes())
            nodes.remove(SOURCE_NODE)
            nodes.sort()
            for j, node in enumerate(nodes):
                subsetg.nodes()[node]["tolerance"] = tolerances[j]

    # if not debug_set and not REGEN_COSTS:
    if not REGEN_COSTS:
        print("going to save sample!")
        # print(sample["name"])
        for sample in samples:
            save_sql_rep(sample["name"], sample)

    print("updated samples in", time.time()-start)

SUBQUERY_JERR_THRESHOLD = 100000
# PERCENTILES_TO_SAVE = [1,5,10,25, 50, 75, 90, 95, 99]
PERCENTILES_TO_SAVE = [1]
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f

DEBUG = False
SOURCE_NODE = tuple("s")

def get_subq_tolerances(qrep, card_key, cost_key):
    '''
    For each subquery, multiply cardinality by x, and see if shortest path
    changed.
    @ret: array of len node, with appropriate tolerances
    '''
    def set_costs(in_edges, cur_node, new_card, subsetg):
        for edge in in_edges:
            node1 = edge[1]
            assert node1 == cur_node
            diff = set(edge[0]) - set(edge[1])
            node2 = list(diff)
            node2.sort()
            node2 = tuple(node2)
            assert node2 in subsetg.nodes()
            card1 = new_card
            card2 = subsetg.nodes()[node2]["cardinality"][card_key]

            hash_join_cost = card1 + card2
            if len(node1) == 1:
                nilj_cost = card2 + NILJ_CONSTANT*card1
            elif len(node2) == 1:
                nilj_cost = card1 + NILJ_CONSTANT*card2
            else:
                nilj_cost = 10000000000
            cost = min(hash_join_cost, nilj_cost)
            assert cost != 0.0
            subsetg[edge[0]][edge[1]][cost_key] = cost

    tcache = klepto.archives.dir_archive("./tolerances_cache",
            cached=True, serialized=True)
    key = deterministic_hash(card_key + cost_key + qrep["sql"])
    if key in tcache.archive:
        return tcache.archive[key]
    tstart = time.time()
    nodes = list(qrep["subset_graph"].nodes())
    nodes.remove(SOURCE_NODE)
    nodes.sort()
    tolerances = np.zeros(len(nodes))
    subsetg = qrep["subset_graph"]
    final_node = [n for n,d in subsetg.in_degree() if d==0][0]
    source_node = tuple("s")

    opt_path = nx.shortest_path(subsetg, final_node, source_node,
            weight=cost_key)

    for i, node in enumerate(nodes):
        in_edges = subsetg.in_edges(node)
        actual = subsetg.nodes()[node]["cardinality"][card_key]
        for j in range(1, 5):
            err = 10**j
            updated_card = actual / err

            # updated_card can only change costs on the in_edges. So update
            # those here, won't need to change cardinality estimates
            # if it leads to changing plan, then break, and update tolerance

            set_costs(in_edges, node, updated_card, subsetg)
            cur_path = nx.shortest_path(subsetg, final_node, source_node,
                    weight=cost_key)
            if cur_path != opt_path:
                break

            updated_card2 = actual * err
            set_costs(in_edges, node, updated_card, subsetg)
            cur_path = nx.shortest_path(subsetg, final_node, source_node,
                    weight=cost_key)

            if cur_path != opt_path:
                break

        # reset to original cardinality, and change costs for in edges back
        tolerances[i] = 10**j
        set_costs(in_edges, node, actual, subsetg)

    # if time.time() - tstart > 5:
    tcache.archive[key] = tolerances
    return tolerances

EMBEDDING_OUTPUT = None
def embedding_hook(module, input_, output):
    global EMBEDDING_OUTPUT
    EMBEDDING_OUTPUT = output

def fcnn_loss(net, use_qloss=False):
    def f2(yhat,y):
        inp = torch.cat((yhat,y))
        jloss = net(inp)
        return jloss
        # if use_qloss:
            # qlosses = qloss_torch(yhat,y)
            # qloss = sum(qlosses) / len(qlosses)
            # # return qloss + jloss
            # return (qloss / 100.0) + jloss
        # else:
            # return jloss

    def f(yhat,y):
        net.layer1.register_forward_hook(embedding_hook)
        net(yhat)
        yhat_embedding = EMBEDDING_OUTPUT
        net(y)
        y_embedding = EMBEDDING_OUTPUT
        # jloss = torch.nn.MSELoss(reduction='mean')(y_embedding, yhat_embedding)
        jloss = qloss_torch(y_embedding, yhat_embedding)
        jloss = sum(jloss) / len(jloss)
        return jloss

    return f

def compute_subquery_priorities(qrep, true_cards, est_cards,
        explain, jerr, env, use_indexes=True):
    '''
    @return: subquery priorities, which must sum upto jerr.
    '''
    def get_sql(aliases):
        aliases = tuple(aliases)
        subgraph = qrep["join_graph"].subgraph(aliases)
        sql = nx_graph_to_query(subgraph)
        return sql, subgraph

    def handle_subtree(plan_tree, cur_node, cur_jerr):
        successors = list(plan_tree.successors(cur_node))
        # print(successors)
        if len(successors) == 0:
            return
        assert len(successors) == 2
        left = successors[0]
        right = successors[1]
        left_aliases = plan_tree.nodes()[left]["aliases"]
        right_aliases = plan_tree.nodes()[right]["aliases"]
        left_sql, left_sg = get_sql(left_aliases)
        right_sql, right_sg = get_sql(right_aliases)
        left_total_cost = plan_tree.nodes()[left]["Total Cost"]
        right_total_cost = plan_tree.nodes()[right]["Total Cost"]

        if len(left_aliases) >= 3:
            (left_est_costs, left_opt_costs,_,_,_,_) = \
                    join_loss_pg([left_sql], [left_sg], [true_cards],
                            [est_cards], env, use_indexes, None, 1)
            left_jerr = left_est_costs[0] - left_opt_costs[0]
        else:
            left_jerr = 0.0

        if len(right_aliases) >= 3:
            (right_est_costs, right_opt_costs,_,_,_,_) = \
                    join_loss_pg([right_sql], [right_sg], [true_cards],
                            [est_cards], env, use_indexes, None, 1)
            right_jerr = right_est_costs[0] - right_opt_costs[0]
        else:
            right_jerr = 0.00

        # -2.00 just added for close cases
        # assert left_jerr <= cur_jerr + 2.00
        # assert right_jerr <= cur_jerr + 2.00

        jerr_thresh = cur_jerr / 3.0
        if left_jerr < jerr_thresh \
                and right_jerr < jerr_thresh:
            # both the sides of the tree have low jerr, so take the complete
            # tree at this point and update the priorities of any subgraph
            # using the nodes in this tree (might not be in the current tree,
            # as those subqueries are important to reduce the jerr here as
            # well)
            cur_aliases = plan_tree.nodes()[left]["aliases"] + plan_tree.nodes()[right]["aliases"]
            node_list = list(qrep["subset_graph"].nodes())
            node_list.sort()
            for i, cur_nodes in enumerate(node_list):
                if set(cur_nodes) <= set(cur_aliases):
                    subpriorities[i] += cur_jerr
            return
        if left_jerr > jerr_thresh:
            handle_subtree(plan_tree, left, left_jerr)

        if right_jerr > jerr_thresh:
            handle_subtree(plan_tree, right, right_jerr)

    subpriorities = np.zeros(len(qrep["subset_graph"].nodes()))

    ## FIXME: what is a good initial priority?
    # give everyone the initial priority
    node_list = list(qrep["subset_graph"].nodes())
    node_list.sort()
    for i, _ in enumerate(node_list):
        subpriorities[i] = jerr

    # for i, _ in enumerate(qrep["subset_graph"].nodes()):
        # subpriorities[i] = 0

    # add sub-jerr priorities to the ones that are included in them - might at
    # most double it

    plan_tree = explain_to_nx(explain)
    root = [n for n,d in plan_tree.in_degree() if d==0]
    assert len(root) == 1
    handle_subtree(plan_tree, root[0], jerr)

    # TODO: normalize subpriorities
    subpriorities /= 1000000

    ## make sure it sums to jerr
    # subpriorities += 1
    # subpriorities /= np.sum(subpriorities)
    # subpriorities *= jerr

    return subpriorities

def compute_subquery_priorities_old(qrep, pred, env, use_indexes=True):
    priorities = np.ones(len(qrep["subset_graph"].nodes()))
    priority_dict = {}
    start = time.time()

    ests = {}
    trues = {}
    for i, node in enumerate(qrep["subset_graph"].nodes()):
        if len(node) > 4:
            continue
        alias_key = ' '.join(node)
        node_info = qrep["subset_graph"].nodes()[node]
        est_sel = pred[i]
        est_card = est_sel*node_info["cardinality"]["total"]
        ests[alias_key] = int(est_card)
        trues[alias_key] = node_info["cardinality"]["actual"]

    for i, node in enumerate(qrep["subset_graph"].nodes()):
        if len(node) != 4:
            continue
        node_info = qrep["subset_graph"].nodes()[node]
        # ests = {}
        # trues = {}
        subgraph = qrep["join_graph"].subgraph(node)
        sql = nx_graph_to_query(subgraph)
        # we need to go over descendants to get all the required cardinalities
        (est_costs, opt_costs,_,_,_,_) = \
                join_loss_pg([sql], [trues], [ests], env, use_indexes,
                        None, 1)

        # now, decide what to do next with these.
        assert len(est_costs) == 1
        jerr_diff = est_costs[0] - opt_costs[0]
        jerr_ratio = est_costs[0] / opt_costs[0]
        if jerr_diff < 20000:
            continue
        if jerr_ratio < 1.5:
            continue
        descendants = nx.descendants(qrep["subset_graph"], node)
        for desc in descendants:
            if desc not in priority_dict:
                priority_dict[desc] = 0.00
            priority_dict[desc] += jerr_ratio

    for i, node in enumerate(qrep["subset_graph"].nodes()):
        if node in priority_dict:
            priorities[i] += priority_dict[node]
    return priorities

def compute_subquery_priorities2(qrep, pred, env):
    '''
    Will use join error computation on the subqueries of @qrep to assign
    priorities to each subquery.

    Separate function to support calling this in parallel. Single threaded,
    because we don't want to iterate over all subqueries.
    @ret: priorities list for each subquery.
    '''
    pred_dict = {}
    for i, node in enumerate(qrep["subset_graph"].nodes()):
        pred_dict[node] = pred[i]

    nodes = list(qrep["subset_graph"].nodes())
    nodes.sort(key=lambda x: len(x), reverse=True)

    # sort by length, and go from longer to shorter. return array based on
    # original ordering.
    start = time.time()
    for node in nodes:
        if len(node) == 1:
            continue
        node_info = qrep["subset_graph"].nodes()[node]
        ests = {}
        trues = {}
        subgraph = qrep["join_graph"].subgraph(node)
        sql = nx_graph_to_query(subgraph)
        # we need to go over descendants to get all the required cardinalities
        descendants = nx.descendants(qrep["subset_graph"], node)
        for desc in descendants:
            alias_key = ' '.join(desc)
            node_info = qrep["subset_graph"].nodes()[desc]
            est_sel = pred_dict[desc]
            est_card = est_sel*node_info["cardinality"]["total"]
            ests[alias_key] = int(est_card)
            trues[alias_key] = node_info["cardinality"]["actual"]

        (est_costs, opt_costs,_,_,_,_) = \
                join_loss_pg([sql], [trues], [ests], env, use_indexes,
                        None, 1)

        # now, decide what to do next with these.
        assert len(est_costs) == 1
        jerr_diff = est_costs[0] - opt_costs[0]
        jerr_ratio = est_costs[0] / opt_costs[0]

    print("took: ", time.time() - start)
    pdb.set_trace()

    # convert back to the order we need

class NN(CardinalityEstimationAlg):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)
        if self.exp_prefix != "":
            self.exp_prefix += "-"
        if self.train_card_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
            self.wj_times = get_wj_times_dict(self.train_card_key)
        else:
            self.wj_times = None
        if self.load_query_together:
            assert self.num_groups == 1

        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekno = datetime.datetime.today().weekday()
        self.start_day = days[weekno]
        if self.loss_func == "qloss":
            self.loss = qloss_torch
        if self.loss_func == "ll_scaled_norm_loss":
            self.loss = self.scaled_norm_loss_wrapper
        elif self.loss_func == "flow_loss":
            self.loss = flow_loss
            self.load_query_together = True
        elif self.loss_func == "flow_loss2":
            self.loss = FlowLoss.apply
            self.load_query_together = True
            self.num_workers = 0

        elif self.loss_func == "rel":
            self.loss = rel_loss_torch
        elif self.loss_func == "weighted":
            self.loss = weighted_loss
        elif self.loss_func == "abs":
            self.loss = abs_loss_torch
        elif self.loss_func == "mse":
            # self.loss = torch.nn.MSELoss(reduction="none")
            self.loss = self.mse_with_min
        elif self.loss_func == "cm_fcnn":
            self.loss = qloss_torch
        else:
            assert False

        self.collate_fn = None
        if self.nn_type == "microsoft":
            self.featurization_scheme = "combined"
            # if self.net_name == "FCNN-Query":
                # self.load_query_together = True

        elif self.nn_type == "num_tables":
            self.featurization_scheme = "combined"
        elif self.nn_type == "mscn":
            self.featurization_scheme = "mscn"
        elif self.nn_type == "mscn_set":
            self.featurization_scheme = "set"
            if not self.use_set_padding:
                self.collate_fn = collate_fn
            else:
                if self.load_query_together:
                    self.collate_fn = collate_fn_set_batch

            # elif self.use_set_padding == 2:
                # self.collate_fn = collate_fn_set
            # if DEBUG_BATCH:


        elif self.nn_type == "transformer":
            self.featurization_scheme = "combined"
            self.load_query_together = True
        else:
            assert False

        if self.load_query_together:
            # if self.nn_type == "mscn_set":
                # self.mb_size = 8
            # else:
                # self.mb_size = 1
            if self.nn_type == "microsoft" and self.mb_size > 1:
                self.collate_fn = collate_fn_combined

            self.eval_batch_size = 1
        else:
            self.mb_size = 2500
            self.eval_batch_size = 10000

        self.nets = []
        self.optimizers = []
        self.schedulers = []

        # each element is a list of priorities
        self.past_priorities = []
        for i in range(self.num_groups):
            self.past_priorities.append([])

        # initialize stats collection stuff

        # will keep storing all the training information in the cache / and
        # dump it at the end
        # self.key = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # We want to only summarize and store the statistics we care about
        # header info:
        #   iter: every eval_epoch
        #   loss_type: qerr, join-loss etc.
        #   summary_type: mean, max, min, percentiles: 50,75th,90th,99th,25th
        #   template: all, OR only for specific template
        #   num_tables: all, OR t1,t2 etc.
        #   num_samples: in the given class, whose stats are being summarized.
        self.cur_stats = defaultdict(list)
        self.best_join_loss = 10000000
        self.best_model_dict = None
        # self.start_validation = 5

        # self.summary_funcs = [np.mean, np.max, np.min]
        # self.summary_types = ["mean", "max", "min"]
        self.summary_funcs = [np.mean]
        self.summary_types = ["mean"]

        for q in PERCENTILES_TO_SAVE:
            self.summary_funcs.append(percentile_help(q))
            self.summary_types.append("percentile:{}".format(str(q)))

        self.query_stats = defaultdict(list)
        self.query_qerr_stats = defaultdict(list)

        if self.eval_parallel:
            self.eval_sync = None
            self.eval_pool = ThreadPool(2)

    def scaled_norm_loss_wrapper(self, yhat, ytrue):
        # else convert them to preds / targets
        assert self.normalization_type == "mscn"
        # yhat = yhat.cpu().detach().cpu().numpy()
        # ytrue = ytrue.cpu().detach().cpu().numpy()
        yhat = torch.exp((yhat + \
            self.min_val)*(self.max_val-self.min_val))
        ytrue = torch.exp((ytrue + \
            self.min_val)*(self.max_val-self.min_val))
        return ll_scaled_norm_loss(yhat, ytrue)

    def mse_with_min(self, yhat, ytrue, min_qerr=1.0):
        mse_losses = torch.nn.MSELoss(reduction="none")(yhat, ytrue)
        if min_qerr == 1.0:
            return mse_losses
        # else convert them to preds / targets
        assert self.normalization_type == "mscn"
        yhat = yhat.cpu().detach().cpu().numpy()
        ytrue = ytrue.cpu().detach().cpu().numpy()
        yhat = np.exp((yhat + \
            self.min_val)*(self.max_val-self.min_val))
        ytrue = np.exp((ytrue + \
            self.min_val)*(self.max_val-self.min_val))
        qerrors = np.maximum( (ytrue / yhat), (yhat / ytrue))
        vals = []
        cur_min_mse_loss = 10000000
        for i, qerr in enumerate(qerrors):
            if qerr < min_qerr:
                vals.append(0.0)
            else:
                vals.append(1.0)
                if mse_losses[i] < cur_min_mse_loss:
                    cur_min_mse_loss = mse_losses[i]

        vals = to_variable(vals, requires_grad=True).float()
        assert vals.shape == ytrue.shape
        assert vals.shape == mse_losses.shape

        # mse_losses -= cur_min_mse_loss
        mse_losses *= vals
        mse_np = mse_losses.cpu().detach().cpu().numpy()
        assert ((mse_np >= 0).sum() == mse_np.size).astype(np.int)
        assert mse_losses.shape == ytrue.shape
        return mse_losses

    def train_transformer(self, net, optimizer, loader, loss_fn, loss_fn_name,
            clip_gradient, samples, normalization_type, min_val, max_val,
            load_query_together=True):
        torch.set_num_threads(1)

        if self.save_gradients:
            grads = []
            par_grads = defaultdict(list)
            grad_samples = []

        assert load_query_together
        for idx, (xbatch, ybatch,info) in enumerate(loader):

            start = time.time()
            assert ybatch.shape[0] == 1
            ybatch = ybatch.reshape(ybatch.shape[1])
            pred = net(xbatch)
            pred = pred.reshape(pred.shape[1])
            if len(pred) != len(ybatch):
                pred = pred[0:len(ybatch)]

            qidx = info[0]["query_idx"]
            assert qidx == info[1]["query_idx"]
            sample = samples[qidx]

            if "flow_loss" in loss_fn_name or \
                "flow_loss" in self.switch_loss_fn:
                assert load_query_together
                subsetg_vectors, trueC_vec, opt_loss = \
                        self.flow_training_info[qidx]

                assert len(subsetg_vectors) == 10

                losses = loss_fn(pred, ybatch.detach().cpu(),
                        normalization_type, min_val,
                        max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                        self.normalize_flow_loss,
                        self.join_loss_pool, self.cost_model)
            else:
                losses = loss_fn(pred, ybatch)

            try:
                loss = losses.sum() / len(losses)
            except:
                loss = losses

            if self.weighted_qloss != 0.0:
                qloss = qloss_torch(pred, ybatch)
                loss += self.weighted_qloss* (sum(qloss) / len(qloss))

            # if self.weighted_mse != 0.0:
                # mses = torch.nn.MSELoss(reduction="None")(pred,
                        # ybatch)
                # loss += self.weighted_mse * mse

            if self.weighted_mse != 0.0 and \
                "flow_loss" in loss_fn_name:
                pred = pred.to(device)
                ybatch = ybatch.to(device)
                mses = torch.nn.MSELoss(reduction="none")(pred,
                        ybatch)
                random.seed(1234)
                if self.num_mse_anchoring == -1 \
                        or len(mses) < self.num_mse_anchoring:
                    mse = torch.mean(mses)
                elif self.num_mse_anchoring in [-2, -3]:
                    mse_idxs = self.node_anchoring_idxs[qidx]
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)
                else:
                    mse_idxs = random.sample(range(0, len(mses)), self.num_mse_anchoring)
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)
                loss += self.weighted_mse * mse

            if self.save_gradients and "flow_loss" in loss_fn_name:
                optimizer.zero_grad()
                pred.retain_grad()
                loss.backward()
                # grads
                grads.append(np.mean(np.abs(pred.grad.detach().cpu().numpy())))
                if sample is not None:
                    grad_samples.append(sample)
                wt_grads = net.compute_grads()
                for wi, wt in enumerate(wt_grads):
                    par_grads[wi].append(wt)
            else:
                optimizer.zero_grad()
                loss.backward()

            if clip_gradient is not None:
                clip_grad_norm_(net.parameters(), clip_gradient)

            optimizer.step()

            idx_time = time.time() - start
            # if idx_time > 10:
                # print("train idx took: ", idx_time)

        if self.save_gradients and len(grad_samples) > 0 \
                and self.epoch % self.eval_epoch == 0:
            self.save_join_loss_stats(grads, None, grad_samples,
                    "train", loss_key="gradients")
            for k,v in par_grads.items():
                self.save_join_loss_stats(v, None, grad_samples,
                        "train", loss_key="param_gradients" + str(k))

    def train_combined_net(self, net, optimizer, loader, loss_fn, loss_fn_name,
            clip_gradient, samples, normalization_type, min_val, max_val,
            load_query_together=False):

        if "flow_loss" in loss_fn_name:
            torch.set_num_threads(1)
        if self.save_gradients:
            grads = []
            par_grads = defaultdict(list)
            grad_samples = []

        # FIXME: requires that each sample seen in training set
        if "flow_loss" in loss_fn_name:
            opt_flow_costs = []
            est_flow_costs = []

        for idx, (xbatch, ybatch,info) in enumerate(loader):
            start = time.time()
            # TODO: add handling for num_tables
            if load_query_together:
                # update the batches
                if self.mb_size > 1:
                    # TODO: should not need to be a separate thing
                    sample = None
                else:
                    xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                            xbatch.shape[2])
                    ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                    qidx = info[0]["query_idx"]
                    assert qidx == info[1]["query_idx"]
                    sample = None
            else:
                sample = None

            ybatch = ybatch.to(device, non_blocking=True)
            xbatch = xbatch.to(device, non_blocking=True)
            pred = net(xbatch).squeeze(1)
            if torch.isnan(pred).any():
                print("prediction is nan!")
                print(pred)
                pdb.set_trace()


            if "flow_loss" in loss_fn_name:
                assert load_query_together
                if self.mb_size > 1:
                    ybatch = ybatch.detach().cpu()
                    qstart = 0
                    losses = []
                    for cur_info in info:
                        qidx = cur_info[0]["query_idx"]
                        assert qidx == cur_info[1]["query_idx"]
                        subsetg_vectors, trueC_vec, opt_loss = \
                                self.flow_training_info[qidx]

                        assert len(subsetg_vectors) == 10

                        cur_loss = loss_fn(pred[qstart:qstart+len(cur_info)],
                                ybatch[qstart:qstart+len(cur_info)],
                                normalization_type, min_val,
                                max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                                self.normalize_flow_loss,
                                self.join_loss_pool, self.cost_model)
                        losses.append(cur_loss)
                        qstart += len(cur_info)
                    losses = torch.stack(losses)
                else:
                    subsetg_vectors, trueC_vec, opt_loss = \
                            self.flow_training_info[qidx]

                    assert len(subsetg_vectors) == 10

                    losses = loss_fn(pred, ybatch.detach().cpu(),
                            normalization_type, min_val,
                            max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                            self.normalize_flow_loss,
                            self.join_loss_pool, self.cost_model)
            else:
                losses = loss_fn(pred, ybatch)

            try:
                loss = losses.sum() / len(losses)
            except:
                loss = losses

            # print(loss)

            if self.weighted_qloss != 0.0:
                qloss = qloss_torch(pred, ybatch)
                loss += self.weighted_qloss* (sum(qloss) / len(qloss))

            if self.weighted_mse != 0.0 and \
                "flow_loss" in loss_fn_name:
                mses = torch.nn.MSELoss(reduction="none")(pred,
                        ybatch)
                random.seed(1234)
                if self.num_mse_anchoring == -1 \
                        or len(mses) < self.num_mse_anchoring:
                    mse = torch.mean(mses)
                elif self.num_mse_anchoring in [-2, -3]:
                    mse_idxs = self.node_anchoring_idxs[qidx]
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)
                else:
                    mse_idxs = random.sample(range(0, len(mses)), self.num_mse_anchoring)
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)

                # nodes = sample["subset_graph"].nodes()
                # if SOURCE_NODE in nodes:
                    # nodes.remove(SOURCE_NODE)
                # nodes.sort()
                # for ni, node in enumerate(nodes):
                    # print(sample["subset_graph"].nodes()[node].keys())
                    # pdb.set_trace()

                loss += self.weighted_mse * mse

            if self.save_gradients and "flow_loss" in loss_fn_name:
                optimizer.zero_grad()
                pred.retain_grad()
                loss.backward()
                # grads
                grads.append(np.mean(np.abs(pred.grad.detach().cpu().numpy())))
                if sample is not None:
                    grad_samples.append(sample)
                wt_grads = net.compute_grads()
                for wi, wt in enumerate(wt_grads):
                    par_grads[wi].append(wt)
            else:
                optimizer.zero_grad()
                loss.backward()

            if clip_gradient is not None:
                clip_grad_norm_(net.parameters(), clip_gradient)

            optimizer.step()
            idx_time = time.time() - start
            if idx_time > 10:
                print("train idx took: ", idx_time)

        if self.save_gradients and len(grad_samples) > 0 \
                and self.epoch % self.eval_epoch == 0:
            self.save_join_loss_stats(grads, None, grad_samples,
                    "train", loss_key="gradients")
            for k,v in par_grads.items():
                self.save_join_loss_stats(v, None, grad_samples,
                        "train", loss_key="param_gradients" + str(k))

    def train_mscn_set_padded(self, net, optimizer, loader, loss_fn, loss_fn_name,
            clip_gradient, samples, normalization_type, min_val, max_val,
            load_query_together=False):

        # because we use openmp to speed up flow-loss computations. Else, it is
        # good to have multiple threads
        if "flow_loss" in loss_fn_name:
            torch.set_num_threads(1)

        if self.save_gradients:
            grads = []
            par_grads = defaultdict(list)
            grad_samples = []

        # FIXME: requires that each sample seen in training set
        if "flow_loss" in loss_fn_name:
            opt_flow_costs = []
            est_flow_costs = []

        for idx, (tbatch, pbatch, jbatch,
                fbatch,tmask,pmask,jmask,ybatch,info) in enumerate(loader):
            ybatch = ybatch.to(device, non_blocking=True)
            # ybatch = torch.stack(ybatch)
            start = time.time()
            # print(tbatch.shape)
            # pdb.set_trace()
            if load_query_together:
                tbatch = tbatch.squeeze()
                pbatch = pbatch.squeeze()
                jbatch = jbatch.squeeze()

                if isinstance(fbatch, torch.Tensor):
                    fbatch = fbatch.squeeze()
                tmask = tmask.squeeze(0)
                pmask = pmask.squeeze(0)
                jmask = jmask.squeeze(0)

                ybatch = ybatch.squeeze()
                # sample = samples[qidx]
                sample = None

            else:
                sample = None

            pred = net(tbatch,pbatch,jbatch,fbatch,tmask,pmask,jmask).squeeze(1)

            if "flow_loss" in loss_fn_name:
                assert load_query_together
                # TODO: potentially can parallelize this, but would it even
                # give any benefit - since the flow-loss is already very
                # parallelized
                ybatch = ybatch.detach().cpu()

                qstart = 0
                losses = []
                for cur_info in info:
                    if "query_idx" not in cur_info[0]:
                        print(cur_info)
                        pdb.set_trace()
                    qidx = cur_info[0]["query_idx"]
                    assert qidx == cur_info[1]["query_idx"]
                    subsetg_vectors, trueC_vec, opt_loss = \
                            self.flow_training_info[qidx]

                    assert len(subsetg_vectors) == 10

                    cur_loss = loss_fn(pred[qstart:qstart+len(cur_info)],
                            ybatch[qstart:qstart+len(cur_info)],
                            normalization_type, min_val,
                            max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                            self.normalize_flow_loss,
                            self.join_loss_pool, self.cost_model)
                    losses.append(cur_loss)
                    qstart += len(cur_info)
                losses = torch.stack(losses)
            else:
                if self.unnormalized_mse:
                    assert self.normalization_type == "mscn"
                    pred = torch.exp((pred + self.min_val)*(self.max_val-self.min_val))
                    ybatch = torch.exp((ybatch + self.min_val)*(self.max_val-self.min_val))

                if self.loss_func == "mse_with_min":
                    losses = loss_fn(pred, ybatch, self.min_qerr)
                else:
                    losses = loss_fn(pred, ybatch)

                if self.flow_weighted_loss:
                    assert False
                    # which subqueries have we been using
                    subq_idxs = info["dataset_idx"]
                    loss_weights = np.ascontiguousarray(self.subq_imp["train"][subq_idxs])
                    loss_weights = to_variable(loss_weights).float()
                    assert losses.shape == loss_weights.shape
                    losses *= loss_weights

            try:
                loss = losses.sum() / len(losses)
            except:
                loss = losses

            if self.weighted_qloss != 0.0:
                qloss = qloss_torch(pred, ybatch)
                loss += self.weighted_qloss* (sum(qloss) / len(qloss))

            # if self.weighted_mse != 0.0:
            if self.weighted_mse != 0.0 and \
                "flow_loss" in loss_fn_name:
                if self.unnormalized_mse:
                    assert self.normalization_type == "mscn"
                    pred = torch.exp((pred + self.min_val)*(self.max_val-self.min_val))
                    ybatch = torch.exp((ybatch + self.min_val)*(self.max_val-self.min_val))

                pred = pred.to(device)
                ybatch = ybatch.to(device)
                mses = torch.nn.MSELoss(reduction="none")(pred,
                        ybatch)
                if self.num_mse_anchoring == -1 \
                        or len(mses) < self.num_mse_anchoring:
                    mse = torch.mean(mses)
                elif self.num_mse_anchoring in [-2, -3]:
                    mse_idxs = []
                    for cur_info in info:
                        qidx = cur_info[0]["query_idx"]
                        cur_mse_idxs = self.node_anchoring_idxs[qidx]
                        mse_idxs += cur_mse_idxs

                    # mse_idxs = self.node_anchoring_idxs[qidx]
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)
                else:
                    random.seed(1234)
                    mse_idxs = random.sample(range(0, len(mses)), self.num_mse_anchoring)
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)

                mse = mse.to(device)
                # self.weighted_mse = self.weighted_mse.to(device)
                loss += self.weighted_mse * mse
                loss = loss.to(device)

            if self.save_gradients and "flow_loss" in loss_fn_name:
                optimizer.zero_grad()
                pred.retain_grad()
                loss.backward()
                # grads
                grads.append(np.mean(np.abs(pred.grad.detach().cpu().numpy())))
                if sample is not None:
                    grad_samples.append(sample)
                wt_grads = net.compute_grads()
                for wi, wt in enumerate(wt_grads):
                    par_grads[wi].append(wt)
            else:
                optimizer.zero_grad()
                loss.backward()

            if clip_gradient is not None:
                clip_grad_norm_(net.parameters(), clip_gradient)

            optimizer.step()

            idx_time = time.time() - start
            if idx_time > 10:
                print("train idx took: ", idx_time)

        if self.save_gradients and len(grad_samples) > 0 \
                and self.epoch % self.eval_epoch == 0:
            self.save_join_loss_stats(grads, None, grad_samples,
                    "train", loss_key="gradients")
            for k,v in par_grads.items():
                self.save_join_loss_stats(v, None, grad_samples,
                        "train", loss_key="param_gradients" + str(k))

    def clean_memory(self):
        # TODO: add dataset cleaning here
        # self.flow_training_info[qidx]

        if hasattr(self, "flow_training_info"):
            # for k,v in self.flow_training_info.items():
            for v in self.flow_training_info:
                if isinstance(v, list) or isinstance(v, tuple):
                    for v0 in v:
                        if isinstance(v0, list) or isinstance(v0, tuple):
                            del(v0[:])
                        del(v0)
                else:
                    print(type(v))
                    del(v)
            gc.collect()

    def train_mscn_set(self, net, optimizer, loader, loss_fn, loss_fn_name,
            clip_gradient, samples, normalization_type, min_val, max_val,
            load_query_together=False):
        if "flow_loss" in loss_fn_name:
            torch.set_num_threads(1)
        if self.save_gradients:
            grads = []
            par_grads = defaultdict(list)
            grad_samples = []

        # FIXME: requires that each sample seen in training set
        if "flow_loss" in loss_fn_name:
            opt_flow_costs = []
            est_flow_costs = []

        for idx, (tbatch, pbatch, jbatch,fbatch, ybatch,info) in enumerate(loader):
            ybatch = torch.stack(ybatch)
            start = time.time()
            if load_query_together:
                assert tbatch.shape[0] <= self.mb_size
                tbatch = tbatch.reshape(tbatch.shape[0]*tbatch.shape[1],
                        tbatch.shape[2])
                pbatch = pbatch.reshape(pbatch.shape[0]*pbatch.shape[1],
                        pbatch.shape[2])
                jbatch = jbatch.reshape(jbatch.shape[0]*jbatch.shape[1],
                        jbatch.shape[2])
                fbatch = fbatch.reshape(fbatch.shape[0]*fbatch.shape[1],
                        fbatch.shape[2])
                ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                qidx = info[0]["query_idx"][0]
                # sample = samples[qidx]
                sample = None
            else:
                sample = None

            pred = net(tbatch,pbatch,jbatch,fbatch).squeeze(1)

            if "flow_loss" in loss_fn_name:
                assert load_query_together
                subsetg_vectors, trueC_vec, opt_loss = \
                        self.flow_training_info[qidx]

                assert len(subsetg_vectors) == 10

                losses = loss_fn(pred, ybatch.detach().cpu(),
                        normalization_type, min_val,
                        max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                        self.normalize_flow_loss,
                        self.join_loss_pool, self.cost_model)
            else:
                if self.unnormalized_mse:
                    assert self.normalization_type == "mscn"
                    pred = torch.exp((pred + self.min_val)*(self.max_val-self.min_val))
                    ybatch = torch.exp((ybatch + self.min_val)*(self.max_val-self.min_val))

                if self.loss_func == "mse_with_min":
                    losses = loss_fn(pred, ybatch, self.min_qerr)
                else:
                    losses = loss_fn(pred, ybatch)

                if self.flow_weighted_loss:
                    assert False
                    # which subqueries have we been using
                    subq_idxs = info["dataset_idx"]
                    loss_weights = np.ascontiguousarray(self.subq_imp["train"][subq_idxs])
                    loss_weights = to_variable(loss_weights).float()
                    assert losses.shape == loss_weights.shape
                    losses *= loss_weights

            try:
                loss = losses.sum() / len(losses)
            except:
                loss = losses

            if self.weighted_qloss != 0.0:
                qloss = qloss_torch(pred, ybatch)
                loss += self.weighted_qloss* (sum(qloss) / len(qloss))

            # if self.weighted_mse != 0.0:
            if self.weighted_mse != 0.0 and \
                "flow_loss" in loss_fn_name:
                if self.unnormalized_mse:
                    assert self.normalization_type == "mscn"
                    pred = torch.exp((pred + self.min_val)*(self.max_val-self.min_val))
                    ybatch = torch.exp((ybatch + self.min_val)*(self.max_val-self.min_val))

                mses = torch.nn.MSELoss(reduction="none")(pred,
                        ybatch)
                if self.num_mse_anchoring == -1 \
                        or len(mses) < self.num_mse_anchoring:
                    mse = torch.mean(mses)
                elif self.num_mse_anchoring in [-2, -3]:
                    mse_idxs = self.node_anchoring_idxs[qidx]
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)
                else:
                    random.seed(1234)
                    mse_idxs = random.sample(range(0, len(mses)), self.num_mse_anchoring)
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)

                loss += self.weighted_mse * mse


            if self.save_gradients and "flow_loss" in loss_fn_name:
                optimizer.zero_grad()
                pred.retain_grad()
                loss.backward()
                # grads
                grads.append(np.mean(np.abs(pred.grad.detach().cpu().numpy())))
                if sample is not None:
                    grad_samples.append(sample)
                wt_grads = net.compute_grads()
                for wi, wt in enumerate(wt_grads):
                    par_grads[wi].append(wt)
            else:
                optimizer.zero_grad()
                loss.backward()

            if clip_gradient is not None:
                clip_grad_norm_(net.parameters(), clip_gradient)

            optimizer.step()

            idx_time = time.time() - start
            if idx_time > 10:
                print("train idx took: ", idx_time)

        if self.save_gradients and len(grad_samples) > 0 \
                and self.epoch % self.eval_epoch == 0:
            self.save_join_loss_stats(grads, None, grad_samples,
                    "train", loss_key="gradients")
            for k,v in par_grads.items():
                self.save_join_loss_stats(v, None, grad_samples,
                        "train", loss_key="param_gradients" + str(k))


    def train_mscn(self, net, optimizer, loader, loss_fn, loss_fn_name,
            clip_gradient, samples, normalization_type, min_val, max_val,
            load_query_together=False):
        if "flow_loss" in loss_fn_name:
            torch.set_num_threads(1)

        if self.save_gradients:
            grads = []
            par_grads = defaultdict(list)
            grad_samples = []

        # FIXME: requires that each sample seen in training set
        if "flow_loss" in loss_fn_name:
            opt_flow_costs = []
            est_flow_costs = []

        for idx, (tbatch, pbatch, jbatch,fbatch, ybatch,info) in enumerate(loader):
            start = time.time()
            if load_query_together:
                assert tbatch.shape[0] <= self.mb_size
                tbatch = tbatch.reshape(tbatch.shape[0]*tbatch.shape[1],
                        tbatch.shape[2])
                pbatch = pbatch.reshape(pbatch.shape[0]*pbatch.shape[1],
                        pbatch.shape[2])
                jbatch = jbatch.reshape(jbatch.shape[0]*jbatch.shape[1],
                        jbatch.shape[2])
                fbatch = fbatch.reshape(fbatch.shape[0]*fbatch.shape[1],
                        fbatch.shape[2])
                ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                qidx = info[0]["query_idx"][0]
                # sample = samples[qidx]
                sample = None
            else:
                sample = None

            pred = net(tbatch,pbatch,jbatch,fbatch).squeeze(1)

            if "flow_loss" in loss_fn_name:
                assert load_query_together
                subsetg_vectors, trueC_vec, opt_loss = \
                        self.flow_training_info[qidx]

                assert len(subsetg_vectors) == 10

                losses = loss_fn(pred, ybatch.detach().cpu(),
                        normalization_type, min_val,
                        max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                        self.normalize_flow_loss,
                        self.join_loss_pool, self.cost_model)
            else:
                if self.unnormalized_mse:
                    assert self.normalization_type == "mscn"
                    pred = torch.exp((pred + self.min_val)*(self.max_val-self.min_val))
                    ybatch = torch.exp((ybatch + self.min_val)*(self.max_val-self.min_val))

                if self.loss_func == "mse_with_min":
                    losses = loss_fn(pred, ybatch, self.min_qerr)
                else:
                    losses = loss_fn(pred, ybatch)

                if self.flow_weighted_loss:
                    assert False
                    # which subqueries have we been using
                    subq_idxs = info["dataset_idx"]
                    loss_weights = np.ascontiguousarray(self.subq_imp["train"][subq_idxs])
                    loss_weights = to_variable(loss_weights).float()
                    assert losses.shape == loss_weights.shape
                    losses *= loss_weights

            try:
                loss = losses.sum() / len(losses)
            except:
                loss = losses

            if self.weighted_qloss != 0.0:
                qloss = qloss_torch(pred, ybatch)
                loss += self.weighted_qloss* (sum(qloss) / len(qloss))

            # if self.weighted_mse != 0.0:
            if self.weighted_mse != 0.0 and \
                "flow_loss" in loss_fn_name:
                if self.unnormalized_mse:
                    assert self.normalization_type == "mscn"
                    pred = torch.exp((pred + self.min_val)*(self.max_val-self.min_val))
                    ybatch = torch.exp((ybatch + self.min_val)*(self.max_val-self.min_val))

                mses = torch.nn.MSELoss(reduction="none")(pred,
                        ybatch)
                if self.num_mse_anchoring == -1 \
                        or len(mses) < self.num_mse_anchoring:
                    mse = torch.mean(mses)
                elif self.num_mse_anchoring in [-2, -3]:
                    mse_idxs = self.node_anchoring_idxs[qidx]
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)
                else:
                    random.seed(1234)
                    mse_idxs = random.sample(range(0, len(mses)), self.num_mse_anchoring)
                    mses = mses[mse_idxs]
                    mse = torch.mean(mses)

                loss += self.weighted_mse * mse


            if self.save_gradients and "flow_loss" in loss_fn_name:
                optimizer.zero_grad()
                pred.retain_grad()
                loss.backward()
                # grads
                grads.append(np.mean(np.abs(pred.grad.detach().cpu().numpy())))
                if sample is not None:
                    grad_samples.append(sample)
                wt_grads = net.compute_grads()
                for wi, wt in enumerate(wt_grads):
                    par_grads[wi].append(wt)
            else:
                optimizer.zero_grad()
                loss.backward()

            if clip_gradient is not None:
                clip_grad_norm_(net.parameters(), clip_gradient)

            optimizer.step()

            idx_time = time.time() - start
            if idx_time > 10:
                print("train idx took: ", idx_time)

        if self.save_gradients and len(grad_samples) > 0 \
                and self.epoch % self.eval_epoch == 0:
            self.save_join_loss_stats(grads, None, grad_samples,
                    "train", loss_key="gradients")
            for k,v in par_grads.items():
                self.save_join_loss_stats(v, None, grad_samples,
                        "train", loss_key="param_gradients" + str(k))

    def init_stats(self, samples):
        self.max_tables = 0
        self.max_val = 0
        self.min_val = 100000
        self.total_training_samples = 0

        for sample in samples:
            for node, info in sample["subset_graph"].nodes().items():
                if node == SOURCE_NODE:
                    continue
                self.total_training_samples += 1
                if len(node) > self.max_tables:
                    self.max_tables = len(node)
                card = info["cardinality"]["actual"]
                if card > self.max_val:
                    self.max_val = card
                if card < self.min_val:
                    self.min_val = card

    def init_groups(self, num_groups):
        groups = []
        for i in range(num_groups):
            groups.append([])

        tables_per_group = math.floor(self.max_tables / num_groups)
        for i in range(num_groups):
            start = i*tables_per_group
            for j in range(start,start+tables_per_group,1):
                groups[i].append(j+1)

        if j+1 < self.max_tables+1:
            for i in range(j+1,self.max_tables,1):
                print(i+1)
                groups[-1].append(i+1)

        print("nn groups: ", groups)
        return groups

    def _init_net(self, net_name, optimizer_name, sample):
        if net_name == "FCNN":
            # do training
            net = SimpleRegression(self.num_features,
                    0, 1,
                    num_hidden_layers=self.num_hidden_layers,
                    hidden_layer_size=self.hidden_layer_size,
                    use_batch_norm = self.use_batch_norm)
        elif net_name == "FCNN-Query":
            assert self.load_query_together
            net = SimpleRegression(self.num_features*self.max_subqs,
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers,
                    hidden_layer_size=self.hidden_layer_size)
        elif net_name == "Transformer":
            assert self.load_query_together
            # FIXME:
            net = RegressionTransformer(self.num_features, self.num_attention_heads,
                    self.num_hidden_layers, self.max_subqs,
                    self.max_subqs)

        elif net_name == "LinearRegression":
            net = LinearRegression(self.num_features,
                    1)
        elif net_name == "SetConv":
            if self.load_query_together:
                net = SetConv(len(sample[0][0]), len(sample[1][0]),
                        len(sample[2][0]), len(sample[3][0]),
                        self.hidden_layer_size, dropout= self.dropout,
                        max_hid = self.max_hid,
                        num_hidden_layers=self.num_hidden_layers)
            else:
                net = SetConv(len(sample[0]), len(sample[1]), len(sample[2]),
                        len(sample[3]),
                        self.hidden_layer_size, dropout=self.dropout,
                        max_hid = self.max_hid,
                        num_hidden_layers=self.num_hidden_layers)

        elif net_name == "MSCN":
            if self.use_set_padding:
                if self.load_query_together:
                    net = PaddedMSCN(len(sample[0][0][0]),
                            len(sample[1][0][0]), len(sample[2][0][0]),
                            len(sample[3][0]),
                            self.hidden_layer_size, dropout=self.dropout,
                            max_hid = self.max_hid,
                            num_hidden_layers=self.num_hidden_layers)
                else:
                    net = PaddedMSCN(len(sample[0][0]),
                            len(sample[1][0]), len(sample[2][0]),
                            len(sample[3]),
                            self.hidden_layer_size, dropout=self.dropout,
                            max_hid = self.max_hid,
                            num_hidden_layers=self.num_hidden_layers)

            else:
                if self.load_query_together:
                    assert False
                else:
                    # print(sample[2])
                    # pdb.set_trace()
                    net = MSCN(len(sample[0][0]), len(sample[1][0]),
                            len(sample[2][0]),
                            len(sample[3]),
                            self.hidden_layer_size, dropout=self.dropout,
                            max_hid = self.max_hid,
                            num_hidden_layers=self.num_hidden_layers)
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
                    amsgrad=True, weight_decay=self.weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(),
                    lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            assert False

        if self.adaptive_lr:
            scheduler = ReduceLROnPlateau(optimizer, 'min',
                    patience=self.adaptive_lr_patience,
                            verbose=True, factor=0.1, eps=self.lr)
        else:
            scheduler = None

        return net, optimizer, scheduler

    def init_nets(self, sample):
        for i in range(len(self.groups)):
            if self.nn_type == "mscn":
                net, optimizer, scheduler = \
                        self._init_net("SetConv", self.optimizer_name, sample)
            elif self.nn_type == "mscn_set":
                net, optimizer, scheduler = \
                        self._init_net("MSCN", self.optimizer_name, sample)

            elif self.nn_type == "transformer":
                net, optimizer, scheduler = \
                        self._init_net("Transformer", self.optimizer_name, sample)
            else:
                net, optimizer, scheduler = \
                        self._init_net(self.net_name, self.optimizer_name, sample)
            print(net)

            self.nets.append(net)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

        print("initialized {} nets for num_tables version".format(len(self.nets)))
        assert len(self.nets) == len(self.groups)

    def _eval_transformer(self, net, loader):
        tstart = time.time()
        # TODO: set num threads, gradient off etc.
        torch.set_grad_enabled(False)
        all_preds = []
        all_y = []
        all_idxs = []

        for idx, (xbatch, ybatch,info) in enumerate(loader):

            assert self.load_query_together
            # pred = net(xbatch).reshape(ybatch.shape[1])
            ybatch = ybatch.reshape(ybatch.shape[1])
            pred = net(xbatch)
            pred = pred.reshape(pred.shape[1])
            if len(pred) != len(ybatch):
                pred = pred[0:len(ybatch)]

            all_preds.append(pred)
            all_y.append(ybatch)

        pred = torch.cat(all_preds).detach().cpu().numpy()
        y = torch.cat(all_y).detach().cpu().numpy()

        if not self.load_query_together:
            all_idxs = torch.cat(all_idxs).detach().cpu().numpy()
        return pred, y, all_idxs

    def _eval_combined(self, net, loader):
        tstart = time.time()
        # TODO: set num threads, gradient off etc.
        torch.set_grad_enabled(False)
        all_preds = []
        all_y = []
        all_idxs = []

        for idx, (xbatch, ybatch,info) in enumerate(loader):
            if self.load_query_together:
                # update the batches
                if self.mb_size == 1:
                    xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                            xbatch.shape[2])
                    ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                    all_idxs.append(0)
                else:
                    assert self.eval_batch_size == 1
                    # FIXME: does it not matter at all?
                    all_idxs.append(0)
            else:
                all_idxs.append(info["dataset_idx"])

            ybatch = ybatch.to(device, non_blocking=True)
            xbatch = xbatch.to(device, non_blocking=True)
            pred = net(xbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)

        pred = torch.cat(all_preds).detach().cpu().numpy()
        y = torch.cat(all_y).detach().cpu().numpy()

        if not self.load_query_together:
            all_idxs = torch.cat(all_idxs).detach().cpu().numpy()
        return pred, y, all_idxs

    def _eval_mscn_set_padded(self, net, loader):
        tstart = time.time()
        # TODO: set num threads, gradient off etc.
        torch.set_grad_enabled(False)
        all_preds = []
        all_y = []
        all_idxs = []

        for idx, (tbatch, pbatch, jbatch,fbatch,
                tmask,pmask,jmask,ybatch,info) in enumerate(loader):
            if self.load_query_together:
                # update the batches
                tbatch = tbatch.squeeze()
                pbatch = pbatch.squeeze()
                jbatch = jbatch.squeeze()
                if isinstance(fbatch, torch.Tensor):
                    fbatch = fbatch.squeeze()
                # fbatch = fbatch.squeeze()
                tmask = tmask.squeeze(0)
                pmask = pmask.squeeze(0)
                jmask = jmask.squeeze(0)

                ybatch = ybatch.squeeze()

                all_idxs.append(0)
            else:
                all_idxs.append(info["dataset_idx"])

            pred = net(tbatch,pbatch,jbatch,fbatch,tmask,pmask,jmask).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)

        pred = torch.cat(all_preds).detach().cpu().numpy()
        y = torch.cat(all_y).detach().cpu().numpy()
        if not self.load_query_together:
            all_idxs = np.concatenate(all_idxs)
            # all_idxs = torch.cat(all_idxs).detach().cpu().numpy()
        return pred, y, all_idxs

    def _eval_mscn_set(self, net, loader):
        tstart = time.time()
        # TODO: set num threads, gradient off etc.
        torch.set_grad_enabled(False)
        all_preds = []
        all_y = []
        all_idxs = []

        for idx, (tbatch,pbatch,jbatch,fbatch, ybatch,info) in enumerate(loader):
            ybatch = torch.stack(ybatch)
            if self.load_query_together:
                # update the batches
                tbatch = tbatch.reshape(tbatch.shape[0]*tbatch.shape[1],
                        tbatch.shape[2])
                pbatch = pbatch.reshape(pbatch.shape[0]*pbatch.shape[1],
                        pbatch.shape[2])
                jbatch = jbatch.reshape(jbatch.shape[0]*jbatch.shape[1],
                        jbatch.shape[2])
                fbatch = fbatch.reshape(fbatch.shape[0]*fbatch.shape[1],
                        fbatch.shape[2])

                ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                all_idxs.append(0)
            else:
                didxs = np.array([cinfo["dataset_idx" ]for cinfo in info])
                all_idxs.append(didxs)

            pred = net(tbatch,pbatch,jbatch,fbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)

        pred = torch.cat(all_preds).detach().cpu().numpy()
        y = torch.cat(all_y).detach().cpu().numpy()
        if not self.load_query_together:
            all_idxs = np.concatenate(all_idxs)
            # all_idxs = torch.cat(all_idxs).detach().cpu().numpy()
        return pred, y, all_idxs

    def _eval_mscn(self, net, loader):
        tstart = time.time()
        # TODO: set num threads, gradient off etc.
        torch.set_grad_enabled(False)
        all_preds = []
        all_y = []
        all_idxs = []

        for idx, (tbatch,pbatch,jbatch,fbatch, ybatch,info) in enumerate(loader):
            if self.load_query_together:
                # update the batches
                tbatch = tbatch.reshape(tbatch.shape[0]*tbatch.shape[1],
                        tbatch.shape[2])
                pbatch = pbatch.reshape(pbatch.shape[0]*pbatch.shape[1],
                        pbatch.shape[2])
                jbatch = jbatch.reshape(jbatch.shape[0]*jbatch.shape[1],
                        jbatch.shape[2])
                fbatch = fbatch.reshape(fbatch.shape[0]*fbatch.shape[1],
                        fbatch.shape[2])

                ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                all_idxs.append(0)
            else:
                all_idxs.append(info["dataset_idx"])

            pred = net(tbatch,pbatch,jbatch,fbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)

        pred = torch.cat(all_preds).detach().cpu().numpy()
        y = torch.cat(all_y).detach().cpu().numpy()
        if not self.load_query_together:
            all_idxs = torch.cat(all_idxs).detach().cpu().numpy()
        return pred, y, all_idxs

    def _eval_loaders(self, loaders):
        if len(loaders) == 1:
            # could also reduce this to the second case, but we don't need to
            # reindex stuff here, so might as well avoid it
            net = self.nets[0]
            loader = loaders[0]
            if self.featurization_scheme == "combined":
                if self.nn_type == "transformer":
                    res = self._eval_transformer(net, loader)
                else:
                    res = self._eval_combined(net, loader)
            elif self.featurization_scheme == "mscn":
                res = self._eval_mscn(net, loader)
            elif self.featurization_scheme == "set":
                if self.use_set_padding:
                    res = self._eval_mscn_set_padded(net, loader)
                else:
                    res = self._eval_mscn_set(net, loader)
            else:
                assert False

            pred = to_variable(res[0]).float()
            y = to_variable(res[1]).float()
        else:
            # in this case, we need to fix indexes after the evaluation
            res = []
            for i, loader in enumerate(loaders):
                if self.featurization_scheme == "combined":
                    res.append(self._eval_combined(self.nets[i], loader))
                elif self.featurization_scheme == "mscn":
                    res.append(self._eval_mscn(self.nets[i], loader))

            # FIXME: this can be faster
            idxs = [r[2] for r in res]
            all_preds = [r[0] for r in res]
            all_ys = [r[1] for r in res]
            idxs = np.concatenate(idxs)
            all_preds = np.concatenate(all_preds)
            all_ys = np.concatenate(all_ys)
            max_idx = np.max(idxs)
            assert len(idxs) == max_idx+1

            pred = np.zeros(len(idxs))
            y = np.zeros(len(idxs))
            for i, val in enumerate(all_preds):
                pred[idxs[i]] = val
                y[idxs[i]] = all_ys[i]
            pred = to_variable(pred).float()
            y = to_variable(y).float()

        return pred,y

    def _eval_samples(self, loaders):
        '''
        @ret: numpy arrays
        '''
        # don't need to compute gradients, saves a lot of memory
        torch.set_grad_enabled(False)
        ret = self._eval_loaders(loaders)
        torch.set_grad_enabled(True)
        return ret

    def eval_samples(self, samples_type):
        loaders = self.eval_loaders[samples_type]
        return self._eval_samples(loaders)

    def add_row(self, losses, loss_type, epoch, template,
            num_tables, samples_type):

        for i, func in enumerate(self.summary_funcs):
            loss = func(losses)
            row = [epoch, loss_type, loss, self.summary_types[i],
                    template, num_tables, len(losses)]
            self.cur_stats["epoch"].append(epoch)
            self.cur_stats["loss_type"].append(loss_type)
            self.cur_stats["loss"].append(loss)
            self.cur_stats["summary_type"].append(self.summary_types[i])
            self.cur_stats["template"].append(template)
            self.cur_stats["num_tables"].append(num_tables)
            self.cur_stats["num_samples"].append(len(losses))
            self.cur_stats["samples_type"].append(samples_type)
            if self.summary_types[i] == "mean" and \
                    (template == "all" or num_tables == "all") \
                    and self.tfboard:
                stat_name = self.tf_stat_fmt.format(
                        samples_type = samples_type,
                        loss_type = loss_type,
                        num_tables = num_tables,
                        template = template)
                try:
                    from tensorflow import summary as tf_summary
                    with self.tf_summary_writer.as_default():
                        tf_summary.scalar(stat_name, loss, step=epoch)
                except:
                    pass

    def get_exp_name(self):
        '''
        '''
        time_hash = str(deterministic_hash(self.start_time))[0:6]
        name = "{PREFIX}{CM}-{NN}-{PRIORITY}-{PR_NORM}-D{DECAY}-{HASH}".format(\
                    PREFIX = self.exp_prefix,
                    NN = self.__str__(),
                    CM = self.cost_model,
                    PRIORITY = self.sampling_priority_alpha,
                    PR_NORM = self.priority_normalize_type,
                    DECAY = str(self.weight_decay),
                    HASH = time_hash)
        return name

    def save_model_dict(self):
        if not os.path.exists(self.result_dir):
            make_dir(self.result_dir)
        exp_name = self.get_exp_name()
        exp_dir = self.result_dir + "/" + exp_name
        if not os.path.exists(exp_dir):
            make_dir(exp_dir)
        fn = exp_dir + "/" + "model_weights.pt"
        torch.save(self.nets[0].state_dict(), fn)

    def save_stats(self):
        '''
        replaces the results file.
        '''
        # TODO: maybe reset cur_stats
        # if self.eval_epoch > 5:
            # return

        self.stats = pd.DataFrame(self.cur_stats)
        if not os.path.exists(self.result_dir):
            make_dir(self.result_dir)
        exp_name = self.get_exp_name()
        exp_dir = self.result_dir + "/" + exp_name
        if not os.path.exists(exp_dir):
            make_dir(exp_dir)

        fn = exp_dir + "/" + "nn.pkl"

        results = {}
        results["stats"] = self.stats
        results["config"] = self.kwargs
        results["name"] = self.__str__()
        results["query_stats"] = self.query_stats
        results["query_qerr_stats"] = self.query_qerr_stats

        with open(fn, 'wb') as fp:
            pickle.dump(results, fp,
                    protocol=4)

        # sfn = exp_dir + "/" + "subq_summary.pkl"
        # with open(sfn, 'wb') as fp:
            # pickle.dump(self.subquery_summary_data, fp,
                    # protocol=4)

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6
        num_params = 0
        for net in self.nets:
            num_params += _calc_size(net)

        return num_params

    def _single_train_combined_net_cm(self, net, optimizer, loader):
        assert self.load_query_together
        for idx, (xbatch, ybatch,_) in enumerate(loader):
            # TODO: add handling for num_tables
            xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                    xbatch.shape[2])
            ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
            pred = net(xbatch).squeeze(1)
            loss = self.cm_loss(pred, ybatch)
            assert len(loss) == 1
            optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()
            optimizer.step()

    def train_one_epoch(self):
        if self.loss_func == "cm_fcnn":
            assert len(self.nets) == 1
            self._single_train_combined_net_cm(self.nets[0], self.optimizers[0],
                    self.training_loaders[0])

        for i, net in enumerate(self.nets):
            opt = self.optimizers[i]
            loader = self.training_loaders[i]
            if self.featurization_scheme == "combined":
                if self.nn_type == "transformer":
                    self.train_transformer(net, opt, loader, self.loss,
                            self.loss_func, self.clip_gradient,
                            self.training_samples, self.normalization_type,
                            self.min_val, self.max_val,
                            self.load_query_together)
                else:
                    self.train_combined_net(net, opt, loader, self.loss,
                            self.loss_func, self.clip_gradient,
                            self.training_samples, self.normalization_type,
                            self.min_val, self.max_val,
                            self.load_query_together)
            elif self.featurization_scheme == "mscn":
                self.train_mscn(net, opt, loader, self.loss,
                        self.loss_func, self.clip_gradient,
                        self.training_samples, self.normalization_type,
                        self.min_val, self.max_val,
                        self.load_query_together)
            elif self.featurization_scheme == "set":
                if self.use_set_padding:
                    self.train_mscn_set_padded(net, opt, loader, self.loss,
                            self.loss_func, self.clip_gradient,
                            self.training_samples, self.normalization_type,
                            self.min_val, self.max_val,
                            self.load_query_together)
                else:
                    self.train_mscn_set(net, opt, loader, self.loss,
                            self.loss_func, self.clip_gradient,
                            self.training_samples, self.normalization_type,
                            self.min_val, self.max_val,
                            self.load_query_together)
            else:
                assert False

    def save_join_loss_stats(self, join_losses, est_plans, samples,
            samples_type, loss_key="jerr", epoch=None):
        if epoch == None:
            epoch = self.epoch
        self.add_row(join_losses, loss_key, epoch, "all",
                "all", samples_type)
        if "grad" not in loss_key:
            print("{}, {} mean: {}".format(samples_type, loss_key,
                    np.mean(join_losses)))

        summary_data = defaultdict(list)

        query_idx = 0
        for i, sample in enumerate(samples):
            template = sample["template_name"]
            summary_data["template"].append(template)
            summary_data["loss"].append(join_losses[i])

        df = pd.DataFrame(summary_data)
        for template in set(df["template"]):
            tvals = df[df["template"] == template]
            self.add_row(tvals["loss"].values, loss_key, epoch,
                    template, "all", samples_type)

        if loss_key == "jerr":
            for i, sample in enumerate(samples):
                self.query_stats["epoch"].append(epoch)
                self.query_stats["query_name"].append(sample["name"])
                # this is also equivalent to the priority, we can normalize it
                # later
                self.query_stats[loss_key].append(join_losses[i])
                self.query_stats["plan"].append(get_leading_hint(est_plans[i]))
                self.query_stats["explain"].append(est_plans[i])

    def _eval_wrapper(self, samples_type):
        if self.eval_parallel:
            if self.eval_sync is not None and \
                    samples_type == "train":
                self.eval_sync.wait()
            self.eval_sync = \
                    self.eval_pool.apply_async(self.periodic_eval,
                            args=(samples_type,self.epoch))
        else:
            self.periodic_eval(samples_type, self.epoch)

    def periodic_eval(self, samples_type, epoch):
        '''
        FIXME: samples type is really train and validation and not test
        '''
        start = time.time()
        self.nets[0].eval()
        pred, Y = self.eval_samples(samples_type)
        self.nets[0].train()

        yhat = copy.deepcopy(pred)
        Ytrue = copy.deepcopy(Y)

        for i in range(len(pred)):
            yhat[i] = torch.exp((pred[i] + \
                        self.min_val)*(self.max_val-self.min_val))
            Ytrue[i] = torch.exp((Y[i] + \
                        self.min_val)*(self.max_val-self.min_val))
        losses = qloss_torch(yhat, Ytrue).detach().cpu().numpy()

        # assert pred.shape == Y.shape
        # print("eval samples done at epoch: ", self.epoch)
        # if "flow_loss" not in self.loss_func:
            # losses = self.loss(pred, Y).detach().cpu().numpy()
        # else:
            # # FIXME: store all appropriate losses throughout...
            # if self.normalization_type == "mscn":
                # losses = torch.nn.MSELoss(reduction="none")(pred,
                        # Y).detach().cpu().numpy()
            # else:
                # losses = qloss_torch(pred, Y).detach().cpu().numpy()

        loss_avg = round(np.sum(losses) / len(losses), 6)

        print("""{}: {}, N: {}, qerr: {}""".format(
            samples_type, epoch, len(Y), loss_avg))

        if self.adaptive_lr and self.scheduler is not None:
            self.scheduler.step(loss_avg)

        self.add_row(losses, "qerr", epoch, "all",
                "all", samples_type)

        if samples_type not in self.samples:
            return

        samples = self.samples[samples_type]
        if samples is None:
            return
        summary_data = defaultdict(list)
        query_idx = 0
        # subq_imps = self.subq_imp[samples_type]
        for samplei, sample in enumerate(samples):
            template = sample["template_name"]
            sample_losses = []
            nodes = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            nodes.sort()
            for subq_idx, node in enumerate(nodes):
                num_tables = len(node)
                idx = query_idx + subq_idx
                loss = float(losses[idx])
                sample_losses.append(loss)
                summary_data["loss"].append(loss)
                summary_data["num_tables"].append(num_tables)
                summary_data["template"].append(template)
                # summary_data["subq_imp"].append(subq_imps[idx])
                sorted_node = list(node)
                sorted_node.sort()
                subq_id = deterministic_hash(str(sorted_node))
                summary_data["subq_id"].append(subq_id)

            query_idx += len(nodes)
            self.query_qerr_stats["epoch"].append(epoch)
            self.query_qerr_stats["query_name"].append(sample["name"])
            self.query_qerr_stats["qerr"].append(sum(sample_losses) / len(sample_losses))

        df = pd.DataFrame(summary_data)

        df["samples_type"] = samples_type
        df["epoch"] = self.epoch

        # if samples_type == "test":
            # self.subquery_summary_data = pd.concat([self.subquery_summary_data,
                # df])
        # else:
            # self.subquery_summary_data = df

        for template in set(df["template"]):
            tvals = df[df["template"] == template]
            self.add_row(tvals["loss"].values, "qerr", epoch,
                    template, "all", samples_type)

        for nt in set(df["num_tables"]):
            nt_losses = df[df["num_tables"] == nt]
            self.add_row(nt_losses["loss"].values, "qerr", epoch, "all",
                    str(nt), samples_type)

        jl_eval_start = time.time()
        assert self.jl_use_postgres

        sqls, jgs, true_cardinalities, est_cardinalities = \
                self.get_query_estimates(pred, samples)

        if self.eval_flow_loss and \
                epoch % self.eval_epoch_flow_err == 0:
            opt_flow_costs, est_flow_costs, _,_, _,_ = \
                    self.flow_loss_env.compute_loss(samples,
                            est_cardinalities, pool = self.join_loss_pool)
            opt_flow_losses = est_flow_costs - opt_flow_costs
            opt_flow_ratios = est_flow_costs / opt_flow_costs

            self.save_join_loss_stats(est_flow_costs, None, samples,
                    samples_type, loss_key="flow_cost")
            self.save_join_loss_stats(opt_flow_losses, None, samples,
                    samples_type, loss_key="flow_err")
            self.save_join_loss_stats(opt_flow_ratios, None, samples,
                    samples_type, loss_key="flow_ratio")

        opt_plan_pg_costs = None
        if self.cost_model_plan_err and \
                epoch % self.eval_epoch_plan_err == 0:
            opt_plan_costs, est_plan_costs, opt_plan_pg_costs, \
                    est_plan_pg_costs, _,_ = \
                    self.plan_err.compute_loss(samples,
                            est_cardinalities, pool = self.join_loss_pool,
                            true_cardinalities=true_cardinalities,
                            join_graphs=jgs)

            cm_plan_losses = est_plan_costs - opt_plan_costs
            cm_plan_losses_ratio = est_plan_costs / opt_plan_costs
            self.save_join_loss_stats(est_plan_costs, None, samples,
                    samples_type, loss_key="mm1_plan_cost")
            self.save_join_loss_stats(cm_plan_losses, None, samples,
                    samples_type, loss_key="mm1_plan_err")
            self.save_join_loss_stats(cm_plan_losses_ratio, None, samples,
                    samples_type, loss_key="mm1_plan_ratio")

            # if opt_plan_pg_costs is not None:
                # cm_plan_pg_losses = est_plan_pg_costs - opt_plan_pg_costs
                # cm_plan_pg_ratio = est_plan_pg_costs / opt_plan_pg_costs

                # self.save_join_loss_stats(cm_plan_pg_losses, None, samples,
                        # samples_type, loss_key="mm1_plan_pg_err")
                # self.save_join_loss_stats(cm_plan_pg_ratio, None, samples,
                        # samples_type, loss_key="mm1_plan_pg_ratio")

                # # if self.debug_set:
                # min_idx = np.argmin(cm_plan_pg_losses)
                # min_idx2 = np.argmin(cm_plan_pg_ratio)
                # print("min plan pg loss: {}, name: {}".format(
                    # cm_plan_pg_losses[min_idx], samples[min_idx]["name"]))
                # print("min plan pg ratio: {}, name: {}".format(
                    # cm_plan_pg_ratio[min_idx2], samples[min_idx2]["name"]))


        if not epoch % self.eval_epoch_jerr == 0:
            return

        if (samples_type == "train" and \
                self.sampling_priority_alpha > 0 and \
                epoch % self.reprioritize_epoch == 0):
            if self.train_card_key == "actual":
                return
        # if priority on, then stats will be saved when calculating
        # priority

        (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                jgs,
                true_cardinalities, est_cardinalities, self.env,
                self.jl_indexes, None,
                pool = self.join_loss_pool,
                join_loss_data_file = self.join_loss_data_file)

        join_losses = np.array(est_costs) - np.array(opt_costs)
        join_losses_ratio = np.array(est_costs) / np.array(opt_costs)

        # join_losses = np.maximum(join_losses, 0.00)

        self.save_join_loss_stats(est_costs, est_plans, samples,
                samples_type, epoch=epoch, loss_key="pp_cost")
        self.save_join_loss_stats(join_losses, est_plans, samples,
                samples_type, epoch=epoch)
        self.save_join_loss_stats(join_losses_ratio, est_plans, samples,
                samples_type, loss_key="jerr_ratio", epoch=epoch)

        print("periodic eval took: ", time.time()-start)
        # if opt_plan_pg_costs is not None and self.debug_set:
        if opt_plan_pg_costs is not None and False:
            cost_model_losses = opt_plan_pg_costs - opt_costs
            cost_model_ratio = opt_plan_pg_costs / opt_costs
            print("cost model losses: ")
            print(np.mean(cost_model_losses), np.mean(cost_model_ratio))
            print("mean: {}, median: {}, 95: {}, 99: {}".format(\
            np.round(np.mean(cost_model_losses),3),
            np.round(np.median(cost_model_losses),3),
            np.round(np.percentile(cost_model_losses,95),3),
            np.round(np.percentile(cost_model_losses,99),3)))

            # for p in [0.90
            print("mean: {}, median: {}, 95: {}, 99: {}".format(\
            np.round(np.mean(cost_model_ratio),3),
            np.round(np.median(cost_model_ratio),3),
            np.round(np.percentile(cost_model_ratio,95),3),
            np.round(np.percentile(cost_model_ratio,99),3)))

            # pdb.set_trace()

        # if np.mean(join_losses) < self.best_join_loss \
                # and epoch > self.start_validation \
                # and self.use_val_set:
            # self.best_join_loss = np.mean(join_losses)
            # self.best_model_dict = copy.deepcopy(self.nets[0].state_dict())
            # print("going to save best join error at epoch: ", epoch)
            # self.save_model_dict()

        # temporary
        if self.env2 is not None:
            (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                    jgs,
                    true_cardinalities, est_cardinalities, self.env2,
                    self.jl_indexes, None,
                    pool = self.join_loss_pool,
                    join_loss_data_file = self.join_loss_data_file)

            join_losses = np.array(est_costs) - np.array(opt_costs)
            join_losses_ratio = np.array(est_costs) / np.array(opt_costs)

            # join_losses = np.maximum(join_losses, 0.00)

            self.save_join_loss_stats(join_losses, est_plans, samples,
                    samples_type, epoch=epoch, loss_key="inl_jerr")
            self.save_join_loss_stats(join_losses_ratio, est_plans, samples,
                    samples_type, loss_key="inl_jerr_ratio", epoch=epoch)

    def _normalize_priorities(self, priorities):
        priorities = np.maximum(priorities, 0.0)
        total = np.float64(np.sum(priorities))
        norm_priorities = np.zeros(len(priorities))
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

        return priorities

    def get_query_estimates(self, pred, samples, true_card_key="actual"):
        '''
        @ret:
        '''
        if not isinstance(pred, np.ndarray):
            pred = pred.detach().cpu().numpy()

        sqls = []
        true_cardinalities = []
        est_cardinalities = []
        join_graphs = []
        query_idx = 0
        for sample in samples:
            if true_card_key != "actual":
                sql = "/* {} */ ".format(true_card_key) + sample["sql"]
            else:
                sql = sample["sql"]
            sqls.append(sql)
            join_graphs.append(sample["join_graph"])
            ests = {}
            trues = {}
            # we don't need to sort these as we are returning a dict here...

            node_keys = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in node_keys:
                node_keys.remove(SOURCE_NODE)
            node_keys.sort()
            for subq_idx, node in enumerate(node_keys):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = ' '.join(node)
                # alias_key = node
                idx = query_idx + subq_idx
                if self.normalization_type == "mscn":
                    sel_est = pred[idx]
                    # assert sel_est <= 1.0
                    est_card = np.exp((sel_est + \
                        self.min_val)*(self.max_val-self.min_val))
                    assert est_card >= 0
                else:
                    est_sel = pred[idx]
                    est_card = est_sel*cards["total"]
                ests[alias_key] = est_card
                if self.wj_times is not None:
                    ck = "wanderjoin-" + str(self.wj_times[sample["template_name"]])
                    true_val = cards[ck]
                    if true_val == 0:
                        true_val = cards["expected"]
                else:
                    true_val = cards[true_card_key]
                trues[alias_key] = true_val

            est_cardinalities.append(ests)
            true_cardinalities.append(trues)
            query_idx += len(node_keys)

        return sqls, join_graphs, true_cardinalities, est_cardinalities

    def initialize_tfboard(self):
        try:
            from tensorflow import summary as tf_summary
            name = self.get_exp_name()
            # name = self.__str__()
            log_dir = "tfboard_logs/" + name
            self.tf_summary_writer = tf_summary.create_file_writer(log_dir)
            self.tf_stat_fmt = "{samples_type}-{loss_type}-nt:{num_tables}-tmp:{template}"
        except:
            print("no tensorflow, so no tf-logging")

    def init_dataset(self, samples, shuffle, batch_size,
            db_year, weighted=False, testing=False):
        training_sets = []
        training_loaders = []
        if testing:
            use_padding = 2
        else:
            use_padding = self.use_set_padding

        for i in range(len(self.groups)):
            training_sets.append(QueryDataset(samples, self.db,
                    self.featurization_scheme, self.heuristic_features,
                    self.preload_features, self.normalization_type,
                    self.load_query_together, self.flow_features,
                    self.table_features, self.join_features,
                    self.pred_features,
                    min_val = self.min_val,
                    max_val = self.max_val,
                    card_key = self.train_card_key,
                    db_year = db_year,
                    use_set_padding = use_padding,
                    group = self.groups[i], max_sequence_len=self.max_subqs,
                    exp_name = self.get_exp_name()))
            if not weighted:
                training_loaders.append(data.DataLoader(training_sets[i],
                        batch_size=batch_size, shuffle=shuffle,
                        num_workers=self.num_workers,
                        pin_memory=True, collate_fn=self.collate_fn))
            else:
                weight = 1 / len(training_sets[i])
                weights = torch.DoubleTensor([weight]*len(training_sets[i]))
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                        num_samples=len(weights))
                training_loader = data.DataLoader(training_sets[i],
                        batch_size=self.mb_size, shuffle=False,
                        pin_memory=False,
                        num_workers=self.num_workers,
                        sampler = sampler, collate_fn=self.collate_fn)
                training_loaders.append(training_loader)
                # priority_loader = data.DataLoader(training_set,
                        # batch_size=25000, shuffle=False, num_workers=0)

        assert len(training_sets) == len(self.groups) == len(training_loaders)
        return training_sets, training_loaders

    def get_subq_imp(self, samples):
        # FIXME: avoid repetition
        subq_hash_imp = {}
        subq_hash_imp_avg = {}

        subq_hash_id = {}
        subq_hash_pred_id = {}

        imps = []
        for sample in samples:
            subsetg_vectors = list(get_subsetg_vectors(sample,
                self.cost_model))

            true_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)
            nodes = list(sample["subset_graph"].nodes())
            nodes.remove(SOURCE_NODE)
            nodes.sort()
            for i, node in enumerate(nodes):
                true_cards[i] = \
                    sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

            # right is the flow for each edge
            edge_dict = {}
            edges = list(sample["subset_graph"].edges())
            edges.sort()
            for i, edge in enumerate(edges):
                edge_dict[edge] = i

            trueC_vec, dgdxT, G, Q = \
                get_optimization_variables(true_cards,
                    subsetg_vectors[0], self.min_val,
                        self.max_val, self.normalization_type,
                        subsetg_vectors[4],
                        subsetg_vectors[5],
                        subsetg_vectors[3],
                        subsetg_vectors[1],
                        subsetg_vectors[2],
                        self.cost_model, subsetg_vectors[-1])

            Gv = to_variable(np.zeros(len(subsetg_vectors[0]))).float()
            Gv[subsetg_vectors[-2]] = 1.0
            G = to_variable(G).float()
            Q = to_variable(Q).float()

            invG = torch.inverse(G)
            v = invG @ Gv
            flows = Q @ (v)
            flows = flows.cpu().numpy()

            # want to calculate the importance of each node in the subquery
            # graph
            node_importances = []
            for i, node in enumerate(nodes):
                if node == SOURCE_NODE:
                    continue

                in_edges = sample["subset_graph"].in_edges(node)
                node_pr = 0.0
                node_pr_avg = 0.0
                for edge in in_edges:
                    node_pr += flows[edge_dict[edge]]
                    node_pr_avg += (flows[edge_dict[edge]] / len(in_edges))

                node_importances.append(node_pr)

                imps += node_importances

                # subsql_hash = deterministic_hash(sample["sql"] + str(nodes[ci]))
                subsql_hash = deterministic_hash(sample["sql"] + str(nodes[i]))
                subq_hash_imp[subsql_hash] = node_pr
                subq_hash_imp_avg[subsql_hash] = node_pr_avg

                sorted_node = list(node)
                sorted_node.sort()

                # subq_id = deterministic_hash(str(sorted_node))
                subq_id = ",".join(sorted_node)
                subq_hash_id[subsql_hash] = subq_id

                cur_pred_cols = []
                for table in sorted_node:
                    pred_cols = sample["join_graph"].nodes()[table]["pred_cols"]
                    pred_cols = list(set(pred_cols))
                    pred_cols.sort()
                    cur_pred_cols += pred_cols
                # pred_ids.append(deterministic_hash(str(cur_pred_cols)))
                pred_id = ",".join(cur_pred_cols)
                subq_hash_pred_id[subsql_hash] = pred_id

        # lets save things

        save_or_update("subq_hash_imp_avg.pkl", subq_hash_imp_avg)
        # save_or_update("subq_hash_imp.pkl", subq_hash_imp)
        # save_or_update("subq_hash_id.pkl", subq_hash_id)
        # save_or_update("subq_hash_pred_id.pkl", subq_hash_pred_id)

        return np.array(imps)

    def update_flow_training_info(self):
        print("precomputing flow loss info")
        # self.true_flows["train"] = []

        fstart = time.time()
        # precompute a whole bunch of training things
        self.flow_training_info = []
        # farchive = klepto.archives.dir_archive("./flow_info_archive",
                # cached=True, serialized=True)
        # farchive.load()
        new_seen = False
        for sample in self.training_samples:
            qkey = deterministic_hash(sample["sql"])
            # if qkey in farchive:
            if False:
                subsetg_vectors = farchive[qkey]
                assert len(subsetg_vectors) == 10
            else:
                new_seen = True
                subsetg_vectors = list(get_subsetg_vectors(sample,
                    self.cost_model))
                # farchive[qkey] = subsetg_vectors

            true_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)
            nodes = list(sample["subset_graph"].nodes())
            nodes.remove(SOURCE_NODE)
            nodes.sort()
            for i, node in enumerate(nodes):
                true_cards[i] = \
                    sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

            trueC_vec, dgdxT, G, Q = \
                get_optimization_variables(true_cards,
                    subsetg_vectors[0], self.min_val,
                        self.max_val, self.normalization_type,
                        subsetg_vectors[4],
                        subsetg_vectors[5],
                        subsetg_vectors[3],
                        subsetg_vectors[1],
                        subsetg_vectors[2],
                        subsetg_vectors[6],
                        subsetg_vectors[7],
                        self.cost_model, subsetg_vectors[-1])

            Gv = to_variable(np.zeros(len(subsetg_vectors[0]))).float()
            Gv[subsetg_vectors[-2]] = 1.0
            trueC_vec = to_variable(trueC_vec).float()
            dgdxT = to_variable(dgdxT).float()
            G = to_variable(G).float()
            Q = to_variable(Q).float()

            trueC = torch.eye(len(trueC_vec)).float().detach()
            for i, curC in enumerate(trueC_vec):
                trueC[i,i] = curC

            invG = torch.inverse(G)
            v = invG @ Gv
            left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
            right = Q @ (v)
            left = left.detach().cpu()
            right = right.detach().cpu()
            opt_flow_loss = left @ trueC @ right
            del trueC

            self.flow_training_info.append((subsetg_vectors, trueC_vec,
                    opt_flow_loss))

        print("precomputing flow info took: ", time.time()-fstart)
        # if new_seen:
            # farchive.dump()
        # del farchive

    def load_model(self, model_dir):
        # TODO: can model dir be reconstructed based on args?
        model_path = model_dir + "/model_weights.pt"
        assert os.path.exists(model_path)
        assert len(self.nets) == 1
        self.nets[0].load_state_dict(torch.load(model_path,
            map_location=device))
        # self.nets[0].eval()
        print(self.nets[0])
        # pdb.set_trace()
        print("*****loaded model*****")

    def train(self, db, training_samples, use_subqueries=False,
            val_samples=None, join_loss_pool = None,
            db_year=""):
        global SOURCE_NODE
        if db.db_name == "so":
            SOURCE_NODE = tuple(["SOURCE"])
        self.ckey = db_year + "cardinality"
        assert isinstance(training_samples[0], dict)
        # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        # resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

        self.join_loss_pool = join_loss_pool

        if self.tfboard:
            self.initialize_tfboard()
        # model is always small enough that it runs fast w/o using many cores
        # torch.set_num_threads(2)
        self.db = db
        if self.eval_epoch > self.max_epochs:
            self.no_eval = True
        else:
            self.no_eval = False

        self.training_samples = training_samples
        self.init_stats(training_samples)

        max_subqs = 0
        for sample in training_samples:
            num_subqs = len(sample["subset_graph"])
            if num_subqs > max_subqs:
                max_subqs = num_subqs

        if val_samples:
            for sample in val_samples:
                num_subqs = len(sample["subset_graph"])
                if num_subqs > max_subqs:
                    max_subqs = num_subqs

        # -1 because this calc would include the source node
        self.max_subqs = max_subqs-1

        self.groups = self.init_groups(self.num_groups)

        if self.num_mse_anchoring == -2:
            # for each training sample, select the nodes to anchor on
            self.node_anchoring_idxs = []
            for sample in training_samples:
                idxs_to_anchor = []
                subsetg = sample["subset_graph"]
                nodes = list(subsetg.nodes())
                if SOURCE_NODE in nodes:
                    nodes.remove(SOURCE_NODE)
                nodes.sort()

                for i, node in enumerate(nodes):
                    key = self.cost_model + "opt_path"
                    if subsetg.nodes()[node][key]:
                        idxs_to_anchor.append(i)

                assert len(idxs_to_anchor) == len(sample["join_graph"].nodes())
                self.node_anchoring_idxs.append(idxs_to_anchor)

        elif self.num_mse_anchoring == -3:
            self.node_anchoring_idxs = []
            for sample in training_samples:
                idxs_to_anchor = []
                subsetg = sample["subset_graph"]
                nodes = list(subsetg.nodes())
                if SOURCE_NODE in nodes:
                    nodes.remove(SOURCE_NODE)
                nodes.sort()

                for i, node in enumerate(nodes):
                    if len(node) == 1:
                        idxs_to_anchor.append(i)

                assert len(idxs_to_anchor) == len(sample["join_graph"].nodes())
                self.node_anchoring_idxs.append(idxs_to_anchor)

        if self.normalization_type == "mscn":
            y = np.array(get_all_cardinalities(training_samples, self.ckey))
            y = np.log(y)
            self.max_val = np.max(y)
            self.min_val = np.min(y)
            # if self.min_val == 0:
                # print("there was 0 as min val")
                # self.min_val += 1
            print("min val: ", self.min_val)
            print("max val: ", self.max_val)
        else:
            assert self.normalization_type == "pg_total_selectivity"
            self.min_val, self.max_val = None, None

        self.env = JoinLoss("cm1", self.db.user, self.db.pwd,
                self.db.db_host, self.db.port, self.db.db_name)

        if self.cost_model != "cm1":
            # self.env2 = JoinLoss(self.cost_model, self.db.user, self.db.pwd,
                    # self.db.db_host, self.db.port, self.db.db_name)
            self.env2 = None
        else:
            self.env2 = None

        self.plan_err = PlanError(self.cost_model, "plan-loss", self.db.user,
                self.db.pwd, self.db.db_host, self.db.port, self.db.db_name,
                compute_pg_costs=False)
        self.flow_loss_env = PlanError(self.cost_model, "flow-loss",
                compute_pg_costs=False)

        self.training_samples = training_samples
        if self.sampling_priority_alpha > 0.00:
            training_sets, self.training_loaders = self.init_dataset(training_samples,
                                    False, self.mb_size, db_year, weighted=True)
            self.training_sets = training_sets
            priority_loaders = []
            for i, ds in enumerate(training_sets):
                priority_loaders.append(data.DataLoader(ds,
                        batch_size=self.eval_batch_size, shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=self.collate_fn))
        else:
            training_sets, self.training_loaders = self.init_dataset(training_samples,
                                    True, self.mb_size, db_year, weighted=False)
            self.training_sets = training_sets

        assert len(self.training_loaders) == len(self.groups)

        if self.featurization_scheme == "combined":
            if self.load_query_together:
                self.num_features = len(training_sets[0][0][0][0])
            else:
                self.num_features = len(training_sets[0][0][0])
                if len(self.groups) == 1:
                    print(self.total_training_samples, len(training_sets[0]))
                    assert self.total_training_samples == len(training_sets[0])
        else:
            # FIXME: need to get accurate number for load_query_together
            if self.load_query_together:
                # print(len(training_sets[0][0][0]))
                # pdb.set_trace()
                self.num_features = len(training_sets[0][0][0][0]) + \
                        len(training_sets[0][0][1][0]) + \
                        len(training_sets[0][0][2][0]) + \
                        len(training_sets[0][0][3][0])
            else:
                # if self.featurization_scheme == "set":
                    # self.num_features = 0
                # else:
                self.num_features = len(training_sets[0][0][0]) + \
                        len(training_sets[0][0][1]) + \
                        len(training_sets[0][0][2]) + \
                        len(training_sets[0][0][3])

            print("num features are: ", self.num_features)

        # self.subq_imp = {}
        # self.subq_imp["train"] = self.get_subq_imp(self.training_samples)
        # self.subq_imp["test"] = self.get_subq_imp(val_samples)

        if "flow" in self.loss_func or \
                "flow" in self.switch_loss_fn:
            if not self.max_epochs == 0:
                self.update_flow_training_info()

        subquery_rel_weights = None
        if self.priority_normalize_type == "paths1":
            print("generating path probs")
            training_set = training_sets[0]
            subw_start = time.time()
            subquery_rel_weights = np.zeros(len(training_set))
            qidx = 0
            template_weights = {}
            for sample in training_samples:
                subsetg = sample["subset_graph"]
                node_list = list(subsetg.nodes())
                node_list.sort(key = lambda v: len(v))
                dest = node_list[-1]
                node_list.remove(SOURCE_NODE)
                node_list.sort()
                cur_weights = np.zeros(len(node_list))
                if sample["template_name"] in template_weights:
                    cur_weights = template_weights[sample["template_name"]]
                else:
                    print("sample template name: ", sample["template_name"])
                    for i, node in enumerate(node_list):
                        all_paths = nx.all_simple_paths(subsetg, dest, node)
                        num_paths = len(list(all_paths))
                        cur_weights[i] = num_paths
                    cur_weights = cur_weights / sum(cur_weights)
                    assert np.abs(sum(cur_weights) - 1.0) < 0.001
                    template_weights[sample["template_name"]] = cur_weights

                subquery_rel_weights[qidx:qidx+len(node_list)] = cur_weights
                qidx += len(node_list)
            print("subquery weights calculate in: ", time.time()-subw_start)
        elif "tolerance" in self.priority_normalize_type:
            subquery_rel_weights = np.zeros(len(training_sets[0]))
            tstart = time.time()
            num_proc = 60
            par_args = []
            for s in training_samples:
                par_args.append((s, None))

            with Pool(processes = num_proc) as pool:
                res = pool.starmap(get_subq_tolerances, par_args)

            qidx = 0
            for si, sample in enumerate(training_samples):
                subsetg = sample["subset_graph"]
                # node_list = list(subsetg.nodes())
                # node_list.sort()
                # will update the cur_weights for this sample now
                tolerances = res[si]
                cur_weights = 1.00 / np.log2(tolerances)
                cur_weights = cur_weights / sum(cur_weights)
                assert np.abs(sum(cur_weights) - 1.0) < 0.001

                num_nodes = len(subsetg.nodes())
                subquery_rel_weights[qidx:qidx+num_nodes] = cur_weights
                qidx += num_nodes

            print("generating tolerances took: ", time.time()-tstart)
            # pdb.set_trace()

        elif "flow" in self.priority_normalize_type:
            # sample = training_samples[0]
            subquery_rel_weights = np.zeros(len(training_sets[0]))
            fl_start = time.time()
            num_proc = 10
            par_args = []
            # training_samples_hash = deterministic_hash(str(training_samples))
            # for s in training_samples:
                # par_args.append((s, "cost"))

            # with Pool(processes = num_proc) as pool:
                # res = pool.starmap(get_subq_flows, par_args)

            if self.priority_normalize_type == "flow4":
                # count number at each level
                template_level_counts = {}

            qidx = 0
            subq_imps = self.subq_imp["train"]
            for si, sample in enumerate(training_samples):
                subsetg = sample["subset_graph"]
                node_list = list(subsetg.nodes())
                if SOURCE_NODE in node_list:
                    node_list.remove(SOURCE_NODE)
                node_list.sort()
                cur_weights = np.zeros(len(node_list))
                # will update the cur_weights for this sample now
                # flows, edge_dict = res[si]

                for i, node in enumerate(node_list):
                    subq_idx = idx = qidx + i
                    node_pr = subq_imps[subq_idx]

                    if self.priority_normalize_type in ["flow1", "flow2"]:
                        cur_weights[i] = node_pr
                    elif self.priority_normalize_type == "flow3":
                        cur_weights[i] = node_pr * (1 / len(node))
                    elif self.priority_normalize_type == "flow4":
                        # approximation for # at the right level
                        # combs = nCr(len(node_lis), len(node))
                        if sample["template_name"] in template_level_counts:
                            counts = template_level_counts[sample["template_name"]]
                        else:
                            counts = defaultdict(int)
                            for node2 in node_list:
                                counts[len(node2)] += 1
                            template_level_counts[sample["template_name"]] = counts
                        level_num_nodes = counts[len(node)]
                        cur_weights[i] = node_pr * level_num_nodes

                cur_weights = cur_weights / sum(cur_weights)
                assert np.abs(sum(cur_weights) - 1.0) < 0.001

                subquery_rel_weights[qidx:qidx+len(node_list)] = cur_weights
                qidx += len(node_list)

            print("generating flow values took: ", time.time()-fl_start)

        if self.loss_func == "cm_fcnn":
            inp_len = len(training_samples[0]["subset_graph"].nodes())
            # fcnn_net = SimpleRegression(inp_len*2, 2, 1,
                    # num_hidden_layers=1)
            fcnn_net = torch.load("./cm_fcnn.pt")
            self.cm_loss = fcnn_loss(fcnn_net)

        # evaluation set, smaller
        self.samples = {}
        self.eval_loaders = {}
        random.seed(2112)

        ## training_set already loaded
        if self.debug_set:
            eval_samples_size_divider = 1
        else:
            # eval_samples_size_divider = 10
            eval_samples_size_divider = 1

        if eval_samples_size_divider != 1:
            eval_training_samples = random.sample(training_samples,
                    int(len(training_samples) / eval_samples_size_divider))
            eval_train_sets, eval_train_loaders = \
                    self.init_dataset(eval_training_samples, False,
                            self.eval_batch_size, db_year, weighted=False)
            self.eval_loaders["train"] = eval_train_loaders

        elif self.eval_epoch < self.max_epochs or \
                self.eval_epoch_qerr < self.max_epochs:
            # eval_training_samples = training_samples
            assert len(training_sets) == 1
            # eval loader should maintain order of samples for periodic_eval to
            # collect stats correctly
            self.eval_loaders["train"] = [data.DataLoader(training_sets[0],
                    batch_size=self.eval_batch_size, shuffle=False,
                    num_workers=self.num_workers, collate_fn=self.collate_fn)]

            if self.eval_epoch < self.max_epochs:
                self.samples["train"] = training_samples

        # TODO: add separate dataset, dataloaders for evaluation
        if val_samples is not None and len(val_samples) > 0 \
                and not self.no_eval \
                and self.eval_epoch < self.max_epochs:
            assert eval_samples_size_divider == 1
            val_samples = val_samples
            self.samples["test"] = val_samples
            eval_test_sets, eval_test_loaders = \
                    self.init_dataset(val_samples, False, self.eval_batch_size,
                            db_year,
                            weighted=False)
            self.eval_test_sets = eval_test_sets
            self.eval_loaders["test"] = eval_test_loaders
        else:
            self.samples["test"] = None

        # TODO: initialize self.num_features
        self.init_nets(training_sets[0][0])

        model_size = self.num_parameters()
        print("""training samples: {}, feature length: {}, model size: {},
        max_discrete_buckets: {}, hidden_layer_size: {}""".\
                format(self.total_training_samples, self.num_features, model_size,
                    self.max_discrete_featurizing_buckets,
                    self.hidden_layer_size))

        # print(type(self.samples["train"]))

        if self.eval_epoch > self.max_epochs:
            if val_samples is not None:
                del(val_samples[:])
            val_samples = None

            if self.sampling_priority_alpha == 0.0:
                if self.preload_features < 4:
                    del(self.training_samples[:])
                    self.training_samples = None

                if self.preload_features < 3:
                    del(training_sets[0].db)

        if self.model_dir is not None:
            print("going to load model!")
            self.load_model(self.model_dir)
            print("loaded model!")

        if self.sampling_priority_alpha > 0:
            self.clean_memory()

        for self.epoch in range(0,self.max_epochs):
            if self.epoch == self.switch_loss_fn_epoch:
                print("*************************")
                print("SWITCHING LOSS FUNCTIONS")
                print("*************************")
                if self.switch_loss_fn == "flow_loss2":
                    self.loss = FlowLoss.apply
                    self.load_query_together = True
                    self.loss_func = self.switch_loss_fn
                else:
                    assert False

            # if self.epoch % self.eval_epoch == 0 and \
                    # self.eval_epoch < self.max_epochs:
            if ((self.epoch % self.eval_epoch == 0 or \
                    self.epoch % self.eval_epoch_qerr == 0)
                and self.epoch != 0):

                eval_start = time.time()
                self._eval_wrapper("train")
                if self.samples["test"] is not None:
                    self._eval_wrapper("test")

                self.save_stats()

            elif self.epoch > self.start_validation and self.use_val_set \
                    and self.epoch % self.validation_epoch == 0:
                if self.samples["test"] is not None:
                    # self.periodic_eval("test")
                    self._eval_wrapper("test")

            start = time.time()
            self.train_one_epoch()
            self.save_model_dict()
            print("one epoch train took: ", time.time()-start)

            if self.sampling_priority_alpha > 0 \
                    and (self.epoch % self.reprioritize_epoch == 0 \
                            or self.epoch == self.prioritize_epoch):
                print("going to update priorities")
                pred, _ = self._eval_samples(priority_loaders)
                pred = pred.detach().cpu().numpy()
                weights = np.zeros(self.total_training_samples)
                if self.sampling_priority_type == "query":
                    # TODO: decompose
                    pr_start = time.time()
                    sqls, jgs, true_cardinalities, est_cardinalities = \
                            self.get_query_estimates(pred,
                                    self.training_samples,
                                    true_card_key=self.train_card_key)
                    (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                            jgs, true_cardinalities, est_cardinalities, self.env,
                            self.jl_indexes, None,
                            pool = self.join_loss_pool,
                            join_loss_data_file = self.join_loss_data_file)

                    jerr_ratio = est_costs / opt_costs
                    jerr = est_costs - opt_costs
                    jerr = np.maximum(jerr, 0.00)
                    # don't do this if train_card_key is not actual, as this
                    # does not reflect real join loss
                    if self.train_card_key == "actual":
                        self.save_join_loss_stats(jerr, est_plans,
                                self.training_samples, "train")

                    print("epoch: {}, jerr_ratio: {}, jerr: {}, time: {}"\
                            .format(self.epoch,
                                np.round(np.mean(jerr_ratio), 2),
                                np.round(np.mean(jerr), 2),
                                time.time()-pr_start))
                    query_idx = 0
                    for si, sample in enumerate(self.training_samples):
                        if self.priority_err_type == "jerr":
                            sq_weight = jerr[si] / 1000000.00
                        else:
                            sq_weight = jerr_ratio[si]

                        if self.priority_normalize_type == "div":
                            sq_weight /= len(sample["subset_graph"].nodes())
                        elif self.priority_normalize_type == "paths1":
                            pass
                        elif self.priority_normalize_type == "flow1":
                            pass

                        # the order of iteration doesn't matter here since each
                        # is being given the same weight
                        num_nodes = len(sample["subset_graph"].nodes())-1
                        for subq_idx in range(num_nodes):
                            weights[query_idx+subq_idx] = sq_weight

                        query_idx += num_nodes

                    if subquery_rel_weights is not None:
                        weights *= subquery_rel_weights

                elif self.sampling_priority_type == "subquery":
                    pr_start = time.time()
                    sqls, jgs, true_cardinalities, est_cardinalities = \
                            self.get_query_estimates(pred,
                                    self.training_samples,
                                    true_card_key = self.train_card_key)
                    (est_costs, opt_costs,est_plans,_,_,_) = join_loss_pg(sqls,
                            jgs, true_cardinalities, est_cardinalities, self.env,
                            self.jl_indexes, None,
                            pool = self.join_loss_pool,
                            join_loss_data_file = self.join_loss_data_file)
                    jerrs = est_costs - opt_costs
                    jerr_ratio = est_costs / opt_costs
                    self.save_join_loss_stats(jerrs, est_plans,
                            self.training_samples, "train")

                    print("subq_pr, epoch: {}, jerr_ratio: {}, jerr: {}, time: {}"\
                            .format(self.epoch,
                                np.round(np.mean(jerr_ratio), 2),
                                np.round(np.mean(jerrs), 2),
                                time.time()-pr_start))
                    par_args = []
                    num_proc = 8
                    for si, sample in enumerate(self.training_samples):
                        par_args.append((sample,
                                    true_cardinalities[si], est_cardinalities[si],
                                    est_plans[si], jerrs[si], self.env,
                                    self.jl_indexes))
                    all_subps = self.join_loss_pool.starmap(compute_subquery_priorities,
                            par_args)

                    query_idx = 0
                    for si, sample in enumerate(self.training_samples):
                        subps = all_subps[si]
                        slen = len(sample["subset_graph"].nodes())
                        weights[query_idx:query_idx+slen] = subps
                        query_idx += slen

                elif self.sampling_priority_type == "subquery_old":
                    start = time.time()
                    par_args = []
                    query_idx = 0
                    for _, qrep in enumerate(self.training_samples):
                        sqs = len(qrep["subset_graph"].nodes())
                        par_args.append((qrep, pred[query_idx:query_idx+sqs],
                            self.env))
                        query_idx += sqs
                    with ThreadPool(processes=multiprocessing.cpu_count()) as pool:
                        all_priorities = pool.starmap(compute_subquery_priorities, par_args)
                    print("subquery sampling took: ", time.time() - start)
                    weights = np.concatenate(all_priorities)
                    assert len(weights) == len(training_set)
                else:
                    assert False

                group_weights = []
                for group in self.groups:
                    group_weights.append([])

                query_idx = 0
                for si, sample in enumerate(self.training_samples):
                    node_list = list(sample["subset_graph"].nodes())
                    node_list.remove(SOURCE_NODE)
                    node_list.sort()
                    for subq_idx, nodes in enumerate(node_list):
                        num_nodes = len(nodes)
                        wt = weights[query_idx+subq_idx]
                        for gi, group in enumerate(self.groups):
                            if num_nodes in group:
                                group_weights[gi].append(wt)
                    query_idx += len(node_list)

                for gi, gwts in enumerate(group_weights):
                    assert len(gwts) == len(training_sets[gi])
                    gwts = np.maximum(gwts, 0.0)
                    gwts = self._update_sampling_weights(gwts)

                    if self.avg_jl_priority:
                        self.past_priorities[gi].append(gwts)
                        if len(self.past_priorities[gi]) > 1:
                            new_priorities = np.zeros(len(gwts))
                            num_past = min(self.num_last,
                                    len(self.past_priorities[gi]))
                            for i in range(1,num_past+1):
                                new_priorities += self.past_priorities[gi][-i]
                            gwts = self._normalize_priorities(new_priorities)
                        else:
                            gwts = self._normalize_priorities(gwts)

                    gwts = torch.DoubleTensor(gwts)

                    sampler = torch.utils.data.sampler.WeightedRandomSampler(gwts,
                            num_samples=len(gwts))
                    tloader = data.DataLoader(training_sets[gi],
                            batch_size=self.eval_batch_size, shuffle=False,
                            num_workers=self.num_workers,
                            sampler = sampler, collate_fn=self.collate_fn)
                    self.training_loaders[gi] = tloader

        self.clean_memory()
        if self.best_model_dict is not None and self.use_best_val_model:
            print("""training done, will update our model based on validation set
            errors now""")
            self.nets[0].load_state_dict(self.best_model_dict)
            self.nets[0].eval()
        else:
            self.save_model_dict()


    def test(self, test_samples, test_year=""):
        '''
        @test_samples: [] sql_representation dicts
        '''
        datasets, loaders = \
                self.init_dataset(test_samples, False, self.eval_batch_size,
                        test_year, weighted=False, testing=True)
        self.nets[0].eval()
        # self.nets[0].eval()
        pred, y = self._eval_samples(loaders)

        if self.preload_features:
            for dataset in datasets:
                dataset.clean()
                del(dataset)
            for loader in loaders:
                del(loader)

        pred = pred.detach().cpu().numpy()
        all_ests = []
        query_idx = 0
        # FIXME: why can't we just use get_query_estimates here?
        for sample in test_samples:
            ests = {}
            node_keys = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in node_keys:
                node_keys.remove(SOURCE_NODE)
            node_keys.sort()
            # for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
            for subq_idx, node in enumerate(node_keys):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = node
                idx = query_idx + subq_idx
                if self.normalization_type == "mscn":
                    est_card = np.exp((pred[idx] + \
                        self.min_val)*(self.max_val-self.min_val))
                elif self.normalization_type == "pg_total_selectivity":
                    est_sel = pred[idx]
                    est_card = est_sel*cards["total"]
                else:
                    assert False

                # true_card = cards["actual"]
                # if true_card >= CROSS_JOIN_CONSTANT:
                    # est_card = true_card

                ests[alias_key] = est_card

            all_ests.append(ests)
            query_idx += len(node_keys)
        # assert query_idx == len(dataset)
        return all_ests

    def __str__(self):
        if self.nn_type == "microsoft":
            name = "msft"
        elif self.nn_type == "num_tables":
            name = "nt"
        elif self.nn_type == "mscn":
            name = "mscn"
        elif self.nn_type == "transformer":
            name = "transformer"
        else:
            name = self.__class__.__name__

        if self.max_discrete_featurizing_buckets:
            name += "-df:" + str(self.max_discrete_featurizing_buckets)
        if self.sampling_priority_alpha > 0.00:
            name += "-pr:" + str(self.sampling_priority_alpha)
        name += "-nn:" + str(self.num_hidden_layers) + ":" + str(self.hidden_layer_size)

        if self.sampling_priority_type != "query":
            name += "-spt:" + self.sampling_priority_type
        # if not self.priority_query_len_scale:
            # name += "-psqls:0"
        if not self.avg_jl_priority:
            name += "-no_pr_avg"

        if self.loss_func != "qloss":
            name += "-loss:" + self.loss_func
        if not self.heuristic_features:
            name += "-no_pg_ests"

        return name

class XGBoost(NN):

    def init_dataset(self, samples, shuffle, batch_size,
            weighted=False):
        ds = QueryDataset(samples, self.db,
                    "combined", self.heuristic_features,
                    self.preload_features, self.normalization_type,
                    self.load_query_together, self.flow_features,
                    self.table_features, self.join_features,
                    self.pred_features,
                    min_val = self.min_val,
                    max_val = self.max_val,
                    card_key = self.train_card_key,
                    group = None, max_sequence_len=self.max_subqs,
                    exp_name = self.get_exp_name(),
                    use_set_padding=False)

        X = ds.X.cpu().numpy()
        Y = ds.Y.cpu().numpy()
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)

        del(ds)
        return X, Y

    def load_model(self, model_dir):
        # TODO: can model dir be reconstructed based on args?
        # model_path = model_dir + "/model_weights.pt"
        # assert os.path.exists(model_path)
        # assert len(self.nets) == 1
        # self.nets[0].load_state_dict(torch.load(model_path))
        # self.nets[0].eval()
        model_path = model_dir + "/xgb_model.json"
        self.xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
        self.xgb_model.load_model(model_path)
        print("*****loaded model*****")
        # pdb.set_trace()

    def set_min_max(self, training_samples):
        y = np.array(get_all_cardinalities(training_samples, self.ckey))
        y = np.log(y)
        self.max_val = np.max(y)
        self.min_val = np.min(y)
        print("min val: ", self.min_val)
        print("max val: ", self.max_val)

    def train(self, db, training_samples, use_subqueries=False,
            val_samples=None, join_loss_pool = None):
        self.db = db
        global SOURCE_NODE
        if db.db_name == "so":
            SOURCE_NODE = tuple(["SOURCE"])

        self.training_samples = training_samples
        self.set_min_max(training_samples)

        max_subqs = 0
        for sample in training_samples:
            num_subqs = len(sample["subset_graph"])
            if num_subqs > max_subqs:
                max_subqs = num_subqs

        if val_samples:
            for sample in val_samples:
                num_subqs = len(sample["subset_graph"])
                if num_subqs > max_subqs:
                    max_subqs = num_subqs

        # -1 because this calc would include the source node
        self.max_subqs = max_subqs-1
        if self.max_epochs == 0:
            return

        X,Y = \
                self.init_dataset(training_samples, False, self.eval_batch_size,
                        weighted=False)

        del(self.training_samples[:])
        print("deleted training samples, test samples: ", type(val_samples))
        if val_samples is not None:
            print(len(val_samples))
            del(val_samples[:])
            print("deleted val samples")

        if self.grid_search:

            # parameters = {'learning_rate':(0.001, 0.0001),
                    # 'n_estimators':(100, 1000),
                    # 'reg_alpha':(0.0, 0.1, 1),
                    # 'max_depth':(3, 6, 10)}
            # xgb_model = xgb.XGBRegressor(objective="reg:squarederror",
                    # verbose=2, njobs=1)


            parameters = {'learning_rate':(0.001, 0.01),
                    'n_estimators':(100, 250, 500, 1000),
                    'loss': ['ls'],
                    'max_depth':(3, 6, 8, 10),
                    'subsample':(1.0, 0.8, 0.5)}

            xgb_model = GradientBoostingRegressor()

            self.xgb_model = RandomizedSearchCV(xgb_model, parameters, n_jobs=-1,
                    verbose=1)

            self.xgb_model.fit(X, Y)

            print("*******************BEST ESTIMATOR FOUND**************")
            print(self.xgb_model.best_estimator_)
            print("*******************BEST ESTIMATOR DONE**************")

        else:
            self.xgb_model = xgb.XGBRegressor(tree_method=self.tree_method,
                          objective="reg:squarederror",
                          verbosity=1,
                          scale_pos_weight=0,
                          learning_rate=self.lr,
                          colsample_bytree = 1.0,
                          subsample = self.subsample,
                          n_estimators=self.n_estimators,
                          reg_alpha = 0.0,
                          max_depth=self.max_depth,
                          gamma=0)

            print("going to call fit")
            self.xgb_model.fit(X,Y, verbose=1)

        print("model fit, going to save")
        exp_name = self.get_exp_name()
        exp_dir = self.result_dir + "/" + exp_name
        self.xgb_model.save_model(exp_dir + "/xgb_model.json")

        # TODO: gridsearch thingy
        # params = {'n_estimators': 100,
          # 'max_depth': 3,
          # 'min_samples_split': 5,
          # 'learning_rate': 0.001,
          # 'loss': 'ls',
          # 'verbose':1}

        # self.xgb_model = GradientBoostingRegressor(**params)
        # self.xgb_model.fit(X, Y)

    def test(self, test_samples):
        X,Y = \
                self.init_dataset(test_samples, False, self.eval_batch_size,
                        weighted=False)
        pred = self.xgb_model.predict(X)
        # pred /= self.scale_up
        # print(pred.shape)

        print(min(pred), max(pred), np.mean(pred))

        all_ests = []
        query_idx = 0
        # FIXME: why can't we just use get_query_estimates here?
        for sample in test_samples:
            ests = {}
            node_keys = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in node_keys:
                node_keys.remove(SOURCE_NODE)
            node_keys.sort()
            # for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
            for subq_idx, node in enumerate(node_keys):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = node
                idx = query_idx + subq_idx
                if self.normalization_type == "mscn":
                    est_card = np.exp((pred[idx] + \
                        self.min_val)*(self.max_val-self.min_val))

                elif self.normalization_type == "pg_total_selectivity":
                    est_sel = pred[idx]
                    est_card = est_sel*cards["total"]
                else:
                    est_card = pred[idx]

                assert est_card > 0
                assert est_card != np.inf

                true_card = cards["actual"]
                if true_card == CROSS_JOIN_CONSTANT:
                    est_card = true_card

                ests[alias_key] = est_card
            all_ests.append(ests)
            query_idx += len(node_keys)
        return all_ests

    def __str__(self):
        return "XGBoost"

class RandomForest(NN):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)
        if self.exp_prefix != "":
            self.exp_prefix += "-"

        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekno = datetime.datetime.today().weekday()
        self.start_day = days[weekno]

    def init_dataset(self, samples, shuffle, batch_size,
            weighted=False):
        ds = QueryDataset(samples, self.db,
                    "combined", True,
                    True, "mscn",
                    False, False,
                    True, True,
                    True,
                    min_val = self.min_val,
                    max_val = self.max_val,
                    card_key = "actual",
                    group = None, max_sequence_len=self.max_subqs,
                    exp_name = self.get_exp_name())

        X = ds.X.cpu().numpy()
        Y = ds.Y.cpu().numpy()
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)

        del(ds)
        return X, Y

    def set_min_max(self, training_samples):
        y = np.array(get_all_cardinalities(training_samples))
        y = np.log(y)
        self.max_val = np.max(y)
        self.min_val = np.min(y)
        print("min val: ", self.min_val)
        print("max val: ", self.max_val)

    def train(self, db, training_samples, use_subqueries=False,
            val_samples=None, join_loss_pool = None):
        self.db = db
        self.training_samples = training_samples
        self.set_min_max(training_samples)

        max_subqs = 0
        for sample in training_samples:
            num_subqs = len(sample["subset_graph"])
            if num_subqs > max_subqs:
                max_subqs = num_subqs

        if val_samples:
            for sample in val_samples:
                num_subqs = len(sample["subset_graph"])
                if num_subqs > max_subqs:
                    max_subqs = num_subqs

        # -1 because this calc would include the source node
        self.max_subqs = max_subqs-1

        X,Y = \
                self.init_dataset(training_samples, False, None,
                        weighted=False)

        if self.grid_search:
            parameters = {'n_estimators':(100, 250, 500, 1000),
                    'max_depth':(3, 6, 8, 10)}

            model = RandomForestRegressor()

            self.model = RandomizedSearchCV(model, parameters, n_jobs=-1,
                    verbose=1)

            self.model.fit(X, Y)

            print("*******************BEST ESTIMATOR FOUND**************")
            print(self.model.best_estimator_)
            print("*******************BEST ESTIMATOR DONE**************")

        else:
            params = {'n_estimators': self.n_estimators,
                      'max_depth': self.max_depth}

            del(self.training_samples[:])
            self.model = RandomForestRegressor(n_jobs=-1, verbose=2, **params)
            self.model.fit(X, Y)

    def test(self, test_samples):
        X,Y = \
                self.init_dataset(test_samples, False, None)
        pred = self.model.predict(X)

        print(min(pred), max(pred), np.mean(pred))

        all_ests = []
        query_idx = 0
        # FIXME: why can't we just use get_query_estimates here?
        for sample in test_samples:
            ests = {}
            node_keys = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in node_keys:
                node_keys.remove(SOURCE_NODE)
            node_keys.sort()
            for subq_idx, node in enumerate(node_keys):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = node
                idx = query_idx + subq_idx
                est_card = np.exp((pred[idx] + \
                    self.min_val)*(self.max_val-self.min_val))

                assert est_card > 0
                assert est_card != np.inf
                ests[alias_key] = est_card
            all_ests.append(ests)
            query_idx += len(node_keys)
        # assert query_idx == len(dataset)
        return all_ests

    def get_exp_name(self):
        '''
        '''
        time_hash = str(deterministic_hash(self.start_time))[0:3]
        name = "{PREFIX}-{NAME}-{HASH}".format(\
                    PREFIX = self.exp_prefix,
                    NAME = self.__str__(),
                    HASH = time_hash)
        return name

    def __str__(self):
        return "RF"


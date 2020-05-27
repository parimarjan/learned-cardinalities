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
# from torch.multiprocessing import Pool as Pool2
# import torch.multiprocessing as mp
# from utils.tf_summaries import TensorboardSummaries
from tensorflow import summary as tf_summary
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

try:
    mp.set_start_method("spawn")
except:
    pass

from cardinality_estimation.flow_loss import FlowLoss, get_optimization_variables

# def collate_fn(sample):
    # return sample[0]

# once we have stored them in archive, parallel just slows down stuff
UPDATE_TOLERANCES_PAR = True
USE_TOLERANCES = True

def update_samples(samples, flow_features, cost_model,
        debug_set):
    # FIXME: need to use correct cost_model here
    start = time.time()
    new_seen = False
    for sample in samples:
        if "subset_graph_paths" in sample:
            subsetg = sample["subset_graph_paths"]
        else:
            subsetg = copy.deepcopy(sample["subset_graph"])
            add_single_node_edges(subsetg)

        sample_edge = list(subsetg.edges())[0]
        if cost_model + "cost" in subsetg.edges()[sample_edge].keys() \
                and not debug_set:
            continue
        else:
            new_seen = True
            pg_total_cost = compute_costs(subsetg, cost_model,
                    cost_key="pg_cost", ests="expected")
            _ = compute_costs(subsetg, cost_model, cost_key="cost",
                    ests=None)

            subsetg.graph[cost_model + "total_cost"] = pg_total_cost

            sample["subset_graph_paths"] = subsetg

            final_node = [n for n,d in subsetg.in_degree() if d==0][0]
            pg_path = nx.shortest_path(subsetg, final_node, SOURCE_NODE,
                    weight="pg_cost")
            for node in pg_path:
                subsetg.nodes()[node][cost_model + "pg_path"] = 1

    # if not new_seen:
        # return
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
            subsetg = samples[i]["subset_graph_paths"]
            nodes = list(subsetg.nodes())
            nodes.remove(SOURCE_NODE)
            nodes.sort()
            for j, node in enumerate(nodes):
                subsetg.nodes()[node]["tolerance"] = tolerances[j]

    if not debug_set:
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
def add_single_node_edges(subset_graph):
    source = SOURCE_NODE
    subset_graph.add_node(source)
    subset_graph.nodes()[source]["cardinality"] = {}
    subset_graph.nodes()[source]["cardinality"]["actual"] = 1.0
    subset_graph.nodes()[source]["cardinality"]["total"] = 1.0

    for node in subset_graph.nodes():
        if len(node) != 1:
            continue
        if node[0] == source[0]:
            continue

        # print("going to add edge from source to node: ", node)
        # subset_graph.add_edge(node, source, cost=0.0)
        subset_graph.add_edge(node, source)
        in_edges = subset_graph.in_edges(node)
        out_edges = subset_graph.out_edges(node)
        # print("in edges: ", in_edges)
        # print("out edges: ", out_edges)

        # if we need to add edges between single table nodes and rest
        for node2 in subset_graph.nodes():
            if len(node2) != 2:
                continue
            if node[0] in node2:
                subset_graph.add_edge(node2, node)

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
    nodes.sort()
    tolerances = np.zeros(len(nodes))
    subsetg = qrep["subset_graph_paths"]
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

def single_train_combined_net(net, optimizer, loader, loss_fn, loss_fn_name,
        clip_gradient, samples, normalization_type, min_val, max_val,
        load_query_together=False):
    torch.set_num_threads(1)
    # torch.set_num_threads(4)
    for idx, (xbatch, ybatch,info) in enumerate(loader):
        start = time.time()
        # TODO: add handling for num_tables
        if load_query_together:
            # update the batches
            xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                    xbatch.shape[2])
            ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
            query_idx = info[0]["query_idx"]
            assert query_idx == info[1]["query_idx"]
            sample = samples[query_idx]
            if DEBUG:
                print(sample["template_name"])
        pred = net(xbatch).squeeze(1)
        if "flow_loss" in loss_fn_name:
            assert load_query_together
            losses = loss_fn(pred, ybatch, sample, normalization_type, min_val,
                    max_val)
        else:
            losses = loss_fn(pred, ybatch)

        try:
            loss = losses.sum() / len(losses)
        except:
            loss = losses

        opt_start = time.time()
        optimizer.zero_grad()
        loss.backward()

        if clip_gradient is not None:
            clip_grad_norm_(net.parameters(), clip_gradient)
        optimizer.step()

        ## debug step, with autograd
        # pred2 = net(xbatch).squeeze(1)
        # flow_loss_debug, predC = flow_loss(pred2, ybatch, sample, normalization_type,
                # min_val, max_val)
        # optimizer.zero_grad()
        # predC.retain_grad()
        # pred2.retain_grad()
        # flow_loss_debug.backward()
        # pred_grad2 = pred2.grad
        # print(pred_grad2)
        # print(predC.grad)
        # update_list("pred_grad.pkl", pred.grad.detach().numpy())
        # update_list("dCdg_autograd.pkl", predC.grad.detach().numpy())
        # update_list("pred_autograd.pkl", pred2.grad.detach().numpy())
        # pdb.set_trace()

        if DEBUG:
            print("optimizer things took: ", time.time()-opt_start)

        # print("optimizer things took: ", time.time()-opt_start)
        idx_time = time.time() - start

        if idx_time > 10:
            print("train idx took: ", idx_time)
            # pdb.set_trace()

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
        elif self.loss_func == "flow_loss":
            self.loss = flow_loss
            self.load_query_together = True
        elif self.loss_func == "flow_loss2":
            self.loss = FlowLoss.apply
            self.load_query_together = True

        elif self.loss_func == "rel":
            self.loss = rel_loss_torch
        elif self.loss_func == "weighted":
            self.loss = weighted_loss
        elif self.loss_func == "abs":
            self.loss = abs_loss_torch
        elif self.loss_func == "mse":
            self.loss = torch.nn.MSELoss(reduction="none")
        elif self.loss_func == "cm_fcnn":
            self.loss = qloss_torch
        else:
            assert False


        if self.load_query_together:
            self.mb_size = 1
            self.eval_batch_size = 1
        else:
            self.mb_size = 2500
            self.eval_batch_size = 10000

        if self.nn_type == "microsoft":
            self.featurization_scheme = "combined"
        elif self.nn_type == "num_tables":
            self.featurization_scheme = "combined"
        elif self.nn_type == "mscn":
            self.featurization_scheme = "mscn"
        else:
            assert False

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

    def train_mscn(self, net, optimizer, loader, loss_fn, loss_fn_name,
            clip_gradient, samples, normalization_type, min_val, max_val,
            load_query_together=False):
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
                sample = samples[qidx]
            else:
                sample = None

            pred = net(tbatch,pbatch,jbatch,fbatch).squeeze(1)

            if "flow_loss" in loss_fn_name:
                assert load_query_together
                subsetg_vectors, trueC_vec, opt_loss = \
                        self.flow_training_info[qidx]

                assert len(subsetg_vectors) == 7

                losses = loss_fn(pred, ybatch.detach(),
                        normalization_type, min_val,
                        max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                        self.normalize_flow_loss,
                        self.join_loss_pool, self.cost_model)
                assert len(subsetg_vectors) == 7
            else:
                losses = loss_fn(pred, ybatch)

            try:
                loss = losses.sum() / len(losses)
            except:
                loss = losses

            if self.weighted_qloss != 0.0:
                qloss = qloss_torch(pred, ybatch)
                loss += self.weighted_qloss* (sum(qloss) / len(qloss))

            if self.weighted_mse != 0.0:
                mse = torch.nn.MSELoss(reduction="mean")(pred,
                        ybatch)
                loss += self.weighted_mse * mse

            if self.save_gradients and "flow_loss" in loss_fn_name:
                optimizer.zero_grad()
                pred.retain_grad()
                loss.backward()
                # grads
                grads.append(np.mean(np.abs(pred.grad.detach().numpy())))
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
                    self.hidden_layer_multiple, 1,
                    num_hidden_layers=self.num_hidden_layers,
                    hidden_layer_size=self.hidden_layer_size)
        elif net_name == "LinearRegression":
            net = LinearRegression(self.num_features,
                    1)
        elif net_name == "SetConv":
            if self.load_query_together:
                net = SetConv(len(sample[0][0]), len(sample[1][0]),
                        len(sample[2][0]), len(sample[3][0]),
                        self.hidden_layer_size, dropout= self.dropout,
                        min_hid = self.min_hid)
            else:
                net = SetConv(len(sample[0]), len(sample[1]), len(sample[2]),
                        len(sample[3]),
                        self.hidden_layer_size, dropout=self.dropout,
                        min_hid = self.min_hid)
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
            else:
                net, optimizer, scheduler = \
                        self._init_net(self.net_name, self.optimizer_name, sample)
            self.nets.append(net)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

        print("initialized {} nets for num_tables version".format(len(self.nets)))
        assert len(self.nets) == len(self.groups)

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
                xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                        xbatch.shape[2])
                ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                all_idxs.append(0)
            else:
                all_idxs.append(info["dataset_idx"])

            pred = net(xbatch).squeeze(1)
            all_preds.append(pred)
            all_y.append(ybatch)

        pred = torch.cat(all_preds).detach().numpy()
        y = torch.cat(all_y).detach().numpy()

        if not self.load_query_together:
            all_idxs = torch.cat(all_idxs).detach().numpy()
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

        pred = torch.cat(all_preds).detach().numpy()
        y = torch.cat(all_y).detach().numpy()
        if not self.load_query_together:
            all_idxs = torch.cat(all_idxs).detach().numpy()
        return pred, y, all_idxs

    def _eval_loaders(self, loaders):
        if len(loaders) == 1:
            # could also reduce this to the second case, but we don't need to
            # reindex stuff here, so might as well avoid it
            net = self.nets[0]
            loader = loaders[0]
            if self.featurization_scheme == "combined":
                res = self._eval_combined(net, loader)
            elif self.featurization_scheme == "mscn":
                res = self._eval_mscn(net, loader)

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
                with self.tf_summary_writer.as_default():
                    tf_summary.scalar(stat_name, loss, step=epoch)

    def get_exp_name(self):
        '''
        '''
        time_hash = str(deterministic_hash(self.start_time))[0:3]
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
        if self.eval_epoch > 5:
            return

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
                    protocol=pickle.HIGHEST_PROTOCOL)

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
                single_train_combined_net(net, opt, loader, self.loss,
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
                # if self.load_query_together:
                    # self.train_mscn_query(net, opt, loader, self.loss,
                            # self.loss_func, self.clip_gradient,
                            # self.training_samples, self.normalization_type,
                            # self.min_val, self.max_val,
                            # self.load_query_together)
                # else:
                    # self.train_mscn(net, opt, loader, self.loss,
                            # self.loss_func, self.clip_gradient,
                            # self.training_samples, self.normalization_type,
                            # self.min_val, self.max_val,
                            # self.load_query_together)
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
        print("eval samples done at epoch: ", self.epoch)
        self.nets[0].train()
        if "flow_loss" not in self.loss_func:
            losses = self.loss(pred, Y).detach().numpy()
        else:
            # FIXME: store all appropriate losses throughout...
            if self.normalization_type == "mscn":
                losses = torch.nn.MSELoss(reduction="none")(pred,
                        Y).detach().numpy()
            else:
                losses = qloss_torch(pred, Y).detach().numpy()

        loss_avg = round(np.sum(losses) / len(losses), 6)
        print("""{}: {}, N: {}, qerr: {}""".format(
            samples_type, epoch, len(Y), loss_avg))
        if self.adaptive_lr and self.scheduler is not None:
            self.scheduler.step(loss_avg)

        self.add_row(losses, "qerr", epoch, "all",
                "all", samples_type)
        samples = self.samples[samples_type]
        summary_data = defaultdict(list)
        query_idx = 0
        # print(samples_type)
        # totals_test = [len(s["subset_graph"].nodes()) for s in samples]
        # print("total samples: ", sum(totals_test))
        # pdb.set_trace()

        for sample in samples:
            template = sample["template_name"]
            sample_losses = []
            nodes = list(sample["subset_graph"].nodes())
            nodes.sort()
            for subq_idx, node in enumerate(nodes):
                num_tables = len(node)
                idx = query_idx + subq_idx
                loss = float(losses[idx])
                sample_losses.append(loss)
                summary_data["loss"].append(loss)
                summary_data["num_tables"].append(num_tables)
                summary_data["template"].append(template)
            query_idx += len(sample["subset_graph"].nodes())
            self.query_qerr_stats["epoch"].append(epoch)
            self.query_qerr_stats["query_name"].append(sample["name"])
            self.query_qerr_stats["qerr"].append(sum(sample_losses) / len(sample_losses))

        df = pd.DataFrame(summary_data)
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
            opt_flow_costs, est_flow_costs, _,_ = \
                    self.flow_loss_env.compute_loss(samples,
                            est_cardinalities, pool = self.join_loss_pool)
            opt_flow_losses = est_flow_costs - opt_flow_costs
            opt_flow_ratios = est_flow_costs / opt_flow_costs
            self.save_join_loss_stats(opt_flow_losses, None, samples,
                    samples_type, loss_key="flow_err")
            self.save_join_loss_stats(opt_flow_ratios, None, samples,
                    samples_type, loss_key="flow_ratio")

        if self.cost_model_plan_err and \
                epoch % self.eval_epoch_plan_err == 0:
            opt_plan_costs, est_plan_costs, opt_plan_pg_costs, \
                    est_plan_pg_costs = \
                    self.plan_err.compute_loss(samples,
                            est_cardinalities, pool = self.join_loss_pool,
                            true_cardinalities=true_cardinalities,
                            join_graphs=jgs)

            cm_plan_losses = est_plan_costs - opt_plan_costs
            cm_plan_losses_ratio = est_plan_costs / opt_plan_costs
            self.save_join_loss_stats(cm_plan_losses, None, samples,
                    samples_type, loss_key="mm1_plan_err")
            self.save_join_loss_stats(cm_plan_losses_ratio, None, samples,
                    samples_type, loss_key="mm1_plan_ratio")

            if opt_plan_pg_costs is not None:
                cm_plan_pg_losses = est_plan_pg_costs - opt_plan_pg_costs
                cm_plan_pg_ratio = est_plan_pg_costs / opt_plan_pg_costs

                self.save_join_loss_stats(cm_plan_pg_losses, None, samples,
                        samples_type, loss_key="mm1_plan_pg_err")
                self.save_join_loss_stats(cm_plan_pg_ratio, None, samples,
                        samples_type, loss_key="mm1_plan_pg_ratio")

                if self.debug_set:
                    min_idx = np.argmin(cm_plan_pg_losses)
                    min_idx2 = np.argmin(cm_plan_pg_ratio)
                    print("min plan pg loss: {}, name: {}".format(
                        cm_plan_pg_losses[min_idx], samples[min_idx]["name"]))
                    print("min plan pg ratio: {}, name: {}".format(
                        cm_plan_pg_ratio[min_idx2], samples[min_idx2]["name"]))

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

        self.save_join_loss_stats(join_losses, est_plans, samples,
                samples_type, epoch=epoch)
        self.save_join_loss_stats(join_losses_ratio, est_plans, samples,
                samples_type, loss_key="jerr_ratio", epoch=epoch)

        print("periodic eval took: ", time.time()-start)
        if opt_plan_pg_costs is not None and self.debug_set:
            cost_model_losses = opt_plan_pg_costs - opt_costs
            cost_model_ratio = opt_plan_pg_costs / opt_costs
            print("cost model losses: ")
            print(np.mean(cost_model_losses), np.mean(cost_model_ratio))
            # pdb.set_trace()

        if np.mean(join_losses) < self.best_join_loss \
                and epoch > self.start_validation \
                and self.use_val_set:
            self.best_join_loss = np.mean(join_losses)
            self.best_model_dict = copy.deepcopy(self.nets[0].state_dict())
            print("going to save best join error at epoch: ", epoch)
            self.save_model_dict()

    def _normalize_priorities(self, priorities):
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
            pred = pred.detach().numpy()
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
            node_keys.sort()
            for subq_idx, node in enumerate(node_keys):
                cards = sample["subset_graph"].nodes()[node]["cardinality"]
                alias_key = ' '.join(node)
                # alias_key = node
                idx = query_idx + subq_idx
                if self.normalization_type == "mscn":
                    est_card = np.exp((pred[idx] + \
                        self.min_val)*(self.max_val-self.min_val))
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
            query_idx += len(sample["subset_graph"].nodes())

        return sqls, join_graphs, true_cardinalities, est_cardinalities

    def initialize_tfboard(self):
        name = self.get_exp_name()
        # name = self.__str__()
        log_dir = "tfboard_logs/" + name
        self.tf_summary_writer = tf_summary.create_file_writer(log_dir)
        self.tf_stat_fmt = "{samples_type}-{loss_type}-nt:{num_tables}-tmp:{template}"

    def init_dataset(self, samples, shuffle, batch_size,
            weighted=False):
        training_sets = []
        training_loaders = []
        for i in range(len(self.groups)):
            training_sets.append(QueryDataset(samples, self.db,
                    self.featurization_scheme, self.heuristic_features,
                    self.preload_features, self.normalization_type,
                    self.load_query_together, self.flow_features,
                    min_val = self.min_val,
                    max_val = self.max_val,
                    card_key = self.train_card_key,
                    group = self.groups[i]))
            if not weighted:
                training_loaders.append(data.DataLoader(training_sets[i],
                        batch_size=batch_size, shuffle=shuffle, num_workers=0))
            else:
                weight = 1 / len(training_sets[i])
                weights = torch.DoubleTensor([weight]*len(training_sets[i]))
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                        num_samples=len(weights))
                training_loader = data.DataLoader(training_sets[i],
                        batch_size=self.mb_size, shuffle=False, num_workers=0,
                        sampler = sampler)
                training_loaders.append(training_loader)
                # priority_loader = data.DataLoader(training_set,
                        # batch_size=25000, shuffle=False, num_workers=0)

        assert len(training_sets) == len(self.groups) == len(training_loaders)
        return training_sets, training_loaders

    def update_flow_training_info(self):
        print("precomputing flow loss info")
        fstart = time.time()
        # precompute a whole bunch of training things
        self.flow_training_info = []
        farchive = klepto.archives.dir_archive("./flow_info_archive",
                cached=True, serialized=True)
        farchive.load()
        new_seen = False
        for sample in self.training_samples:
            qkey = deterministic_hash(sample["sql"])
            if qkey in farchive:
                subsetg_vectors = farchive[qkey]
                assert len(subsetg_vectors) == 7
            else:
                new_seen = True
                subsetg_vectors = list(get_subsetg_vectors(sample))
                farchive[qkey] = subsetg_vectors

            true_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)
            nodes = list(sample["subset_graph"].nodes())
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
                        self.cost_model)

            Gv = to_variable(np.zeros(len(subsetg_vectors[0]))).float()
            Gv[subsetg_vectors[-1]] = 1.0
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
            opt_flow_loss = left @ trueC @ right
            del trueC

            self.flow_training_info.append((subsetg_vectors, trueC_vec,
                    opt_flow_loss))

        print("precomputing flow info took: ", time.time()-fstart)
        if new_seen:
            farchive.dump()
        del farchive

    def train(self, db, training_samples, use_subqueries=False,
            val_samples=None, join_loss_pool = None):
        assert isinstance(training_samples[0], dict)
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

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
        self.groups = self.init_groups(self.num_groups)
        if self.cost_model_plan_err or self.eval_flow_loss or \
                self.flow_features:
            update_samples(training_samples, self.flow_features,
                    self.cost_model, self.debug_set)
            if val_samples and not self.no_eval:
                update_samples(val_samples, self.flow_features,
                        self.cost_model, self.debug_set)

        if self.normalization_type == "mscn":
            y = np.array(get_all_cardinalities(training_samples))
            y = np.log(y)
            self.max_val = np.max(y)
            self.min_val = np.min(y)
            print("min val: ", self.min_val)
            print("max val: ", self.max_val)
        else:
            assert self.normalization_type == "pg_total_selectivity"
            self.min_val, self.max_val = None, None

        # create a new park env, and close at the end.
        self.env = JoinLoss(self.cost_model, self.db.user, self.db.pwd,
                self.db.db_host, self.db.port, self.db.db_name)
        self.plan_err = PlanError(self.cost_model, "plan-loss", self.db.user,
                self.db.pwd, self.db.db_host, self.db.port, self.db.db_name,
                compute_pg_costs=True)
        self.flow_loss_env = PlanError(self.cost_model, "flow-loss",
                compute_pg_costs=False)

        self.training_samples = training_samples
        if self.sampling_priority_alpha > 0.00:
            training_sets, self.training_loaders = self.init_dataset(training_samples,
                                    False, self.mb_size, weighted=True)
            priority_loaders = []
            for i, ds in enumerate(training_sets):
                priority_loaders.append(data.DataLoader(ds,
                        batch_size=self.eval_batch_size, shuffle=False, num_workers=0))
        else:
            training_sets, self.training_loaders = self.init_dataset(training_samples,
                                    True, self.mb_size, weighted=False)

        assert len(self.training_loaders) == len(self.groups)

        if self.featurization_scheme == "combined":
            if self.load_query_together:
                self.num_features = len(training_sets[0][0][0][0])
            else:
                self.num_features = len(training_sets[0][0][0])
                if len(self.groups) == 1:
                    assert self.total_training_samples == len(training_sets[0])
        else:
            self.num_features = len(training_sets[0][0][0]) + \
                    len(training_sets[0][0][1]) + len(training_sets[0][0][2])

        if "flow" in self.loss_func:
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
            for s in training_samples:
                par_args.append((s, "cost"))

            with Pool(processes = num_proc) as pool:
                res = pool.starmap(get_subq_flows, par_args)

            if self.priority_normalize_type == "flow4":
                # count number at each level
                template_level_counts = {}

            qidx = 0
            for si, sample in enumerate(training_samples):
                subsetg = sample["subset_graph"]
                node_list = list(subsetg.nodes())
                node_list.sort()
                cur_weights = np.zeros(len(node_list))
                # will update the cur_weights for this sample now
                flows, edge_dict = res[si]

                for i, node in enumerate(node_list):
                    # all_paths = nx.all_simple_paths(subsetg, dest, node)
                    # num_paths = len(list(all_paths))
                    in_edges = subsetg.in_edges(node)
                    node_pr = 0.0
                    for edge in in_edges:
                        node_pr += flows[edge_dict[edge]]

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
                            self.eval_batch_size, weighted=False)
            self.eval_loaders["train"] = eval_train_loaders
        else:
            eval_training_samples = training_samples
            assert len(training_sets) == 1
            # eval loader should maintain order of samples for periodic_eval to
            # collect stats correctly
            self.eval_loaders["train"] = [data.DataLoader(training_sets[0],
                    batch_size=self.eval_batch_size, shuffle=False,
                    num_workers=0)]
        self.samples["train"] = eval_training_samples

        # TODO: add separate dataset, dataloaders for evaluation
        if val_samples is not None and len(val_samples) > 0 \
                and not self.no_eval:
            # val_samples = random.sample(val_samples, int(len(val_samples) /
                    # eval_samples_size_divider))
            assert eval_samples_size_divider == 1
            val_samples = val_samples
            # totals_test = [len(s["subset_graph"].nodes()) for s in val_samples]
            # print("total test samples: ", sum(totals_test))
            # pdb.set_trace()

            self.samples["test"] = val_samples
            eval_test_sets, eval_test_loaders = \
                    self.init_dataset(val_samples, False, self.eval_batch_size,
                            weighted=False)
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

        for self.epoch in range(0,self.max_epochs):
            # if self.epoch % self.eval_epoch == 0:
            if self.epoch % self.eval_epoch == 0 and \
                    self.epoch != 0:
                eval_start = time.time()
                self._eval_wrapper("train")
                if self.samples["test"] is not None:
                    self._eval_wrapper("test")

                self.save_stats()

            elif self.epoch > self.start_validation and self.use_val_set:
                if self.samples["test"] is not None:
                    self.periodic_eval("test")

            start = time.time()
            self.train_one_epoch()
            print("one epoch train took: ", time.time()-start)

            if self.sampling_priority_alpha > 0 \
                    and (self.epoch % self.reprioritize_epoch == 0 \
                            or self.epoch == self.prioritize_epoch):
                pred, _ = self._eval_samples(priority_loaders)
                pred = pred.detach().numpy()
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
                        for subq_idx, _ in enumerate(sample["subset_graph"].nodes()):
                            weights[query_idx+subq_idx] = sq_weight

                        query_idx += len(sample["subset_graph"].nodes())

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

                    gwts = torch.DoubleTensor(gwts)
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(gwts,
                            num_samples=len(gwts))
                    tloader = data.DataLoader(training_sets[gi],
                            batch_size=self.eval_batch_size, shuffle=False, num_workers=0,
                            sampler = sampler)
                    self.training_loaders[gi] = tloader

        if self.best_model_dict is not None and self.use_best_val_model:
            print("""training done, will update our model based on validation set
            errors now""")
            self.nets[0].load_state_dict(self.best_model_dict)
            self.nets[0].eval()
        else:
            self.save_model_dict()

    def test(self, test_samples):
        '''
        @test_samples: [] sql_representation dicts
        '''
        self.nets[0].eval()
        datasets, loaders = \
                self.init_dataset(test_samples, False, self.eval_batch_size,
                        weighted=False)
        self.nets[0].eval()
        pred, y = self._eval_samples(loaders)
        if self.preload_features:
            for dataset in datasets:
                del(dataset.X)
                del(dataset.Y)
                del(dataset.info)
        # loss = self.loss(pred, y).detach().numpy()
        # print("loss after test: ", np.mean(loss))
        pred = pred.detach().numpy()
        all_ests = []
        query_idx = 0
        # FIXME: why can't we just use get_query_estimates here?
        for sample in test_samples:
            ests = {}
            node_keys = list(sample["subset_graph"].nodes())
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
                ests[alias_key] = est_card
            all_ests.append(ests)
            query_idx += len(sample["subset_graph"].nodes())
        # assert query_idx == len(dataset)
        return all_ests

    def __str__(self):
        if self.nn_type == "microsoft":
            name = "msft"
        elif self.nn_type == "num_tables":
            name = "nt"
        elif self.nn_type == "mscn":
            name = "mscn"
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

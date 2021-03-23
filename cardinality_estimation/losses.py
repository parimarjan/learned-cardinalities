import numpy as np
import pdb
# import park
from utils.utils import *
from cardinality_estimation.query import *
# from cardinality_estimation.join_loss import JoinLoss,PlanError,\
        # get_simple_shortest_path_cost

from cardinality_estimation.join_loss import *

import itertools
import multiprocessing
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import datetime
import pandas as pd
import networkx as nx
import inspect

import MySQLdb
import json

EPSILON = 0.0000000001
REL_LOSS_EPSILON = EPSILON
QERR_MIN_EPS = EPSILON
CROSS_JOIN_CARD = 1313136191

PERCENTILES_TO_SAVE = [25, 50, 75, 90, 99]
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f
summary_funcs = [np.mean, np.max, np.min]
summary_types = ["mean", "max", "min"]
for q in PERCENTILES_TO_SAVE:
    summary_funcs.append(percentile_help(q))
    summary_types.append("percentile:{}".format(str(q)))

RESULTS_DIR_TMP = "{RESULT_DIR}/{ALG}/"

def add_query_result_row(sql_key, samples_type, exec_sql, cost,
        loss, plan, template, cur_costs, costs, qfn, card_key):
    '''
    '''
    # FIXME: is this needed / what situation is it for?
    if sql_key in costs["sql_key"].values:
        return

    arg_names = inspect.getfullargspec(add_query_result_row).args
    for arg in arg_names:
        arg_val = locals()[arg]
        if arg in costs:
            cur_costs[arg].append(arg_val)
        else:
            if not (arg == "costs" or arg == "cur_costs"):
                assert arg_val is None

def join_op_stats(explains):
    num_joins_opt = defaultdict(int)
    for i, _ in enumerate(explains):
        explain = explains[i]
        node_types = extract_values(explain[0][0], "Node Type")
        for nt in node_types:
            if "Nest" in nt:
                num_joins_opt["nested"] += 1
            elif "Merge" in nt:
                num_joins_opt["merge"] += 1
            elif "Hash" in nt:
                num_joins_opt["hash"] += 1
    return num_joins_opt

def node_match(n1, n2):
    return n1 == n2

def add_row(losses, loss_type, epoch, template,
        num_tables, samples_type, stats, ck):
    for i, func in enumerate(summary_funcs):
        loss = func(losses)
        # row = [epoch, loss_type, loss, summary_types[i],
                # template, num_tables, len(losses)]
        stats["epoch"].append(epoch)
        stats["loss_type"].append(loss_type)
        stats["loss"].append(loss)
        stats["summary_type"].append(summary_types[i])
        stats["template"].append(template)
        stats["num_tables"].append(num_tables)
        stats["num_samples"].append(len(losses))
        stats["samples_type"].append(samples_type)
        stats["card_key"].append(ck)

def qerr_loss_stats(samples, losses, samples_type,
        epoch, ckey):
    '''
    @samples: [] qrep objects.
    @preds: selectivity predictions for each

    @ret: dataframe summarizing all the stats
    '''
    stats = defaultdict(list)
    # assert "ordered" in type(samples[0]["subset_graph"])
    assert isinstance(samples[0]["subset_graph"], nx.OrderedDiGraph)

    add_row(losses, "qerr", epoch, "all", "all", samples_type,
            stats, ckey)
    summary_data = defaultdict(list)
    query_idx = 0
    for sample in samples:
        template = sample["template_name"]
        nodes = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in nodes:
            nodes.remove(SOURCE_NODE)
        for subq_idx, node in enumerate(nodes):
            num_tables = len(node)
            idx = query_idx + subq_idx
            if idx >= len(losses):
                print("idx > losses")
                continue
            loss = losses[idx]
            summary_data["loss"].append(loss)
            summary_data["num_tables"].append(num_tables)
            summary_data["template"].append(template)
        query_idx += len(nodes)

    df = pd.DataFrame(summary_data)

    # create new df summarizing the relevant results
    for template in set(df["template"]):
        tvals = df[df["template"] == template]
        add_row(tvals["loss"].values, "qerr", epoch,
                template, "all", samples_type, stats, ckey)
        for nt in set(tvals["num_tables"]):
            nt_losses = tvals[tvals["num_tables"] == nt]
            add_row(nt_losses["loss"].values, "qerr", epoch, template, str(nt),
                    samples_type, stats, ckey)

    for nt in set(df["num_tables"]):
        nt_losses = df[df["num_tables"] == nt]
        add_row(nt_losses["loss"].values, "qerr", epoch, "all", str(nt),
                samples_type, stats, ckey)

    return pd.DataFrame(stats)

def get_loss(loss):
    if loss == "abs":
        return compute_abs_loss
    elif loss == "rel":
        return compute_relative_loss
    elif loss == "qerr":
        return compute_qerror
    elif loss == "join-loss":
        return compute_join_order_loss
    elif loss == "mysql-loss":
        return compute_join_order_loss_mysql
    elif loss == "mysql-cost-model":
        return compute_cost_model_loss_mysql
    elif loss == "plan-loss":
        return compute_plan_loss
    elif loss == "flow-loss":
        return compute_flow_loss
    else:
        assert False

def get_loss_name(loss_name):
    if "qerr" in loss_name:
        return "qerr"
    elif "join" in loss_name:
        return "join"
    elif "abs" in loss_name:
        return "abs"
    elif "rel" in loss_name:
        return "rel"
    elif "plan_loss" in loss_name:
        return "plan-loss"
    elif "flow" in loss_name:
        return "flow-loss"

def _get_all_cardinalities(queries, preds, cardinality_key="cardinality"):
    ytrue = []
    yhat = []
    cur_queries = []
    # totals = []
    for i, pred_subsets in enumerate(preds):
        # one of pred_subsets could be None --> if the alg does not have a
        # valid prediction for this query (e.g., when using old-db predictions)
        if pred_subsets is None:
            continue

        qrep = queries[i]["subset_graph"].nodes()
        query_used = True
        keys = list(pred_subsets.keys())
        keys.sort()

        # for alias, pred in pred_subsets.items():
        for alias in keys:
            assert alias != SOURCE_NODE
            pred = pred_subsets[alias]
            if cardinality_key not in qrep[alias].keys():
                query_used = False
                break
            cards = qrep[alias][cardinality_key]

            if "actual" not in cards:
                query_used = False
                break

            actual = cards["actual"]

            if actual == 0:
                actual += 1
            ytrue.append(float(actual))
            yhat.append(float(pred))

        if query_used:
            cur_queries.append(queries[i])

    return ytrue, yhat, cur_queries

# TODO: put the yhat, ytrue parts in db_utils
def compute_relative_loss(queries, preds, **kwargs):
    '''
    as in the quicksel paper.
    '''
    ytrue, yhat, _ = _get_all_cardinalities(queries, preds)
    epsilons = np.array([REL_LOSS_EPSILON]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    errors = np.abs(ytrue - yhat) / ytrue
    return errors

# def compute_abs_loss(queries, preds, **kwargs):
    # ytrue, yhat, totals = _get_all_cardinalities(queries, preds)
    # errors = np.abs(yhat_total - ytrue)
    # return errors

def compute_qerror(queries, preds, **kwargs):
    assert len(preds) == len(queries)
    assert isinstance(preds[0], dict)

    args = kwargs["args"]
    if args.db_name == "so":
        global SOURCE_NODE
        SOURCE_NODE = tuple(["SOURCE"])
    cardinality_key = kwargs["cardinality_key"]

    # here, we assume that the alg name is unique enough, for their results to
    # be grouped together
    exp_name = kwargs["exp_name"]
    samples_type = kwargs["samples_type"]

    rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                   ALG = exp_name)
    make_dir(rdir)

    pred_fn = rdir + "/" + "preds.pkl"
    all_preds = {}

    for i, q in enumerate(queries):
        all_preds[q["name"]] = preds[i]

    old_results = load_object_gzip(pred_fn)
    if old_results is not None:
        all_preds.update(old_results)

    save_object_gzip(pred_fn, all_preds)

    ytrue, yhat, cur_queries = _get_all_cardinalities(queries, preds, cardinality_key)
    ytrue = np.array(ytrue)
    yhat = np.array(yhat)
    assert len(ytrue) == len(yhat)
    try:
        assert 0.00 not in ytrue
        assert 0.00 not in yhat
    except Exception as e:
        print(e)
        pdb.set_trace()

    # errors = np.maximum((ytrue / yhat), (yhat / ytrue))
    errors = []
    for i,yt in enumerate(ytrue):
        if yt > yhat[i]:
            errors.append(-yt / yhat[i])
        else:
            errors.append(yhat[i] / yt)

    errors_all = copy.deepcopy(errors)
    errors = np.abs(np.array(errors))
    df = qerr_loss_stats(cur_queries, errors,
            samples_type, -1, cardinality_key)

    fn = rdir + "/" + "qerr.pkl"
    # args_fn = rdir + "/" + "args.pkl"
    # save_object(args_fn, args)

    # update the qerrors here
    old_results = load_object(fn)
    if old_results is not None:
        df = pd.concat([old_results, df], ignore_index=True)

    save_object(fn, df)

    all_qerr_losses = defaultdict(list)
    query_losses = defaultdict(list)
    query_idx = 0
    full_query_qerrs = defaultdict(list)

    all_hashes = []
    for si, sample in enumerate(cur_queries):
        nodes = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in nodes:
            nodes.remove(SOURCE_NODE)

        nodes.sort()
        max_len = len(sample["join_graph"].nodes())

        template = sample["template_name"]
        # cur_err = np.mean(errors[query_idx:query_idx+len(nodes)])
        cur_errs = errors[query_idx:query_idx+len(nodes)]
        assert len(cur_errs) == len(nodes)

        for ci, cerr in enumerate(cur_errs):
            if len(nodes[ci]) == max_len:
                full_query_qerrs["qerr"].append(cerr)
                full_query_qerrs["samples_type"].append(samples_type)
                full_query_qerrs["name"].append(sample["name"])
                full_query_qerrs["card_key"].append(cardinality_key)
                break

        for ci, cerr in enumerate(cur_errs):
            subsql_hash = deterministic_hash(sample["sql"] + str(nodes[ci]))
            all_hashes.append(subsql_hash)

        query_losses["name"].append(sample["name"])
        query_losses["qerr_mean"].append(np.mean(cur_errs))
        query_losses["qerr50"].append(np.median(cur_errs))
        query_losses["qerr90"].append(np.percentile(cur_errs,90))
        query_losses["qerr95"].append(np.percentile(cur_errs, 95))
        query_losses["qerr99"].append(np.percentile(cur_errs, 99))
        query_losses["samples_type"].append(samples_type)
        query_idx += len(nodes)

    # print("case: {}: alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
            # .format(args.db_name, args.algs, len(cur_queries),
                # "full query qerr",
                # np.round(np.mean(full_query_qerrs["qerr"]),3),
                # np.round(np.median(full_query_qerrs["qerr"]), 3),
                # np.round(np.percentile(full_query_qerrs["qerr"], 95), 3),
                # np.round(np.percentile(full_query_qerrs["qerr"], 99), 3),
                # ))

    qfn = rdir + "/" + "full_query_qerr.pkl"
    full_query_qerrs = pd.DataFrame(full_query_qerrs)
    old_results = load_object(qfn)
    if old_results is not None:
        df = pd.concat([old_results, full_query_qerrs], ignore_index=True)
    else:
        df = full_query_qerrs
    save_object(qfn, df)

    query_losses = pd.DataFrame(query_losses)
    qfn = rdir + "/" + "query_qerr.pkl"
    old_results = load_object(qfn)
    if old_results is not None:
        df = pd.concat([old_results, query_losses], ignore_index=True)
    else:
        df = query_losses
    save_object(qfn, df)

    # query_avg_loss =

    assert len(all_hashes) == len(errors_all)
    for ei, error in enumerate(errors_all):
        all_qerr_losses["loss"].append(error)
        all_qerr_losses["samples_type"].append(samples_type)
        all_qerr_losses["subq_hash"].append(all_hashes[ei])
        all_qerr_losses["card_key"].append(cardinality_key)

    all_qerr_losses = pd.DataFrame(all_qerr_losses)
    qfn = rdir + "/" + "all_qerr.pkl"
    old_results = load_object(qfn)
    if old_results is not None:
        df = pd.concat([old_results, all_qerr_losses], ignore_index=True)
    else:
        df = all_qerr_losses
    save_object(qfn, df)

    return errors

def fix_query(query):
    # FIXME: make this shit not be so dumb.

    # for calcite rules etc.
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")

    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    if "::float" in query:
        query = query.replace("::float", "")

    if "::int" in query:
        query = query.replace("::int", "")

    return query

def save_join_loss_training_data(sqls, est_cardinalities,
        est_costs, opt_costs, est_plans, jloss_fn):
    '''
    saves a file: join_loss_data.pkl
        defaultdict with:
            keys: sql_hash
            ests: np array, sorted by alias names
                [est_cardinalities...]
            jloss: double

    if file already exists, then just updates it by loading the prev one in
    memory.
    '''
    jlosses = {}
    jlosses["key"] = []
    jlosses["est"] = []
    jlosses["jloss"] = []
    jlosses["jratio"] = []
    jlosses["plan"] = []

    jerrs = est_costs - opt_costs
    jratios = est_costs / opt_costs
    for i, sql in enumerate(sqls):
        key = deterministic_hash(sql)
        est_keys = list(est_cardinalities[i].keys())
        est_keys.sort()
        ests = np.zeros(len(est_keys))
        for j, k in enumerate(est_keys):
            ests[j] = est_cardinalities[i][k]

        jlosses["key"].append(key)
        jlosses["est"].append(ests)
        jlosses["jloss"].append(jerrs[i])
        jlosses["jratio"].append(jratios[i])
        jlosses["plan"].append(get_leading_hint(est_plans[i]))

    jlosses_orig = load_object(jloss_fn)
    if jlosses_orig is not None:
        for k in jlosses.keys():
            jlosses[k] = jlosses_orig[k] + jlosses[k]

    save_object(jloss_fn, jlosses)

def join_loss_pg(sqls, join_graphs,
        true_cardinalities, est_cardinalities, env,
        use_indexes, pdf=None, num_processes=1, pool=None,
        join_loss_data_file=None, backend="postgres", fns=None):
    '''
    @sqls: [sql strings]
    @pdf: None, or open pdf file to which the plans and cardinalities will be
    plotted.

    @ret:
    '''
    start = time.time()
    for i,sql in enumerate(sqls):
        sqls[i] = fix_query(sql)

    est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                env.compute_join_order_loss(sqls, join_graphs,
                        true_cardinalities, est_cardinalities,
                        None, use_indexes,
                        num_processes=num_processes, backend=backend,
                        pool=pool,
                        fns = fns)

    assert isinstance(est_costs, np.ndarray)
    if join_loss_data_file:
        join_losses = est_costs - opt_costs
        save_join_loss_training_data(sqls, est_cardinalities, est_costs,
                opt_costs, est_plans, join_loss_data_file)

    # if pool is not None:
        # print("join_loss_pg took: ", time.time() - start)
    return est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls

def get_join_results_name(alg_name):
    join_dir = "./join_results"
    make_dir(join_dir)
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    weekno = datetime.datetime.today().weekday()
    day = days[weekno]
    time_hash = str(deterministic_hash(time.time()))[0:3]
    name = "{DIR}/{DAY}-{ALG}-{HASH}".format(\
                DIR = join_dir,
                DAY = day,
                ALG = alg_name,
                HASH = time_hash)
    return name


def compute_join_order_loss_mysql(queries, preds, **kwargs):
    '''
    TODO: also updates each query object with the relevant stats that we want
    to plot.
    @queries: list of qrep objects
    @preds: list of dicts

    @output: updates ./results/join_order_loss.pkl file
    '''
    def run_join_loss_exp(env, cost_model):
        use_indexes = args.jl_indexes
        exp_name = kwargs["exp_name"]
        samples_type = kwargs["samples_type"]
        # pool = kwargs["pool"]
        pool = None

        # here, we assume that the alg name is unique enough, for their results to
        # be grouped together
        rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                       ALG = exp_name)
        make_dir(rdir)
        costs_fn = rdir + cost_model + "_mysql_jerr.pkl"
        costs = load_object(costs_fn)
        if costs is None:
            columns = ["sql_key", "explain","plan","exec_sql","cost", "loss",
                    "postgresql_conf", "samples_type", "template", "qfn",
                    "card_key"]
            costs = pd.DataFrame(columns=columns)

        cur_costs = defaultdict(list)
        assert isinstance(costs, pd.DataFrame)

        est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                        join_loss_pg(sqls, join_graphs, true_cardinalities,
                                est_cardinalities, env, use_indexes, pdf=None,
                                pool = pool, join_loss_data_file =
                                args.join_loss_data_file, backend="mysql",
                                fns = fns)
        losses = est_costs - opt_costs
        for i, qrep in enumerate(eval_queries):
            sql_key = str(deterministic_hash(qrep["sql"]))
            if save_exec_sql:
                exec_sql = est_sqls[i]
            else:
                exec_sql = None
            add_query_result_row(sql_key, samples_type,
                    exec_sql, est_costs[i],
                    losses[i],
                    None,
                    # get_leading_hint(est_plans[i]),
                    qrep["template_name"], cur_costs, costs,
                    qrep["name"], cardinality_key)

        cur_df = pd.DataFrame(cur_costs)
        combined_df = pd.concat([costs, cur_df], ignore_index=True)
        save_object(costs_fn, combined_df)

        return est_costs, opt_costs

    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)

    # env = park.make('query_optimizer')
    args = kwargs["args"]
    save_exec_sql = args.save_exec_sql
    if args.db_name == "so":
        global SOURCE_NODE
        SOURCE_NODE = tuple(["SOURCE"])

    alg_name = kwargs["name"]
    cardinality_key = kwargs["cardinality_key"]

    env2 = JoinLoss("cm1", args.user, args.pwd, args.db_host,
            args.port, args.db_name)

    est_cardinalities = []
    true_cardinalities = []
    sqls = []
    join_graphs = []
    # FIXME: wasting a lot of memory
    eval_queries = []
    fns = []

    # TODO: save alg based predictions too
    for i, qrep in enumerate(queries):
        # TODO: check if this prediction is valid from both predictor / and for
        # the ground truth
        if preds[i] is None:
            print("preds None!")
            pdb.set_trace()
            continue

        ests = {}
        trues = {}
        predq = preds[i]
        no_ground_truth_data = False

        for node, node_info in qrep["subset_graph"].nodes().items():
            if node == SOURCE_NODE:
                continue
            est_card = predq[node]
            alias_key = ' '.join(node)
            if cardinality_key not in node_info \
                    or "actual" not in node_info[cardinality_key]:
                no_ground_truth_data = True
                break

            trues[alias_key] = node_info[cardinality_key]["actual"]
            if est_card == 0:
                print("bad est card")
                est_card += 1
            ests[alias_key] = est_card

        if no_ground_truth_data:
            continue

        eval_queries.append(qrep)
        sqls.append(qrep["sql"])
        join_graphs.append(qrep["join_graph"])
        est_cardinalities.append(ests)
        true_cardinalities.append(trues)
        fns.append(qrep["name"])

    est_costs2, opt_costs2 = run_join_loss_exp(env2, "cm1")
    losses2 = est_costs2
    print("case: {}: alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
            .format(args.db_name, alg_name, len(queries),
                "mysql plan cost",
                np.round(np.mean(losses2),3),
                np.round(np.median(losses2),3),
                np.round(np.percentile(losses2,95),3),
                np.round(np.percentile(losses2,99),3)))

    return np.array(est_costs2) - np.array(opt_costs2)


def compute_join_order_loss(queries, preds, **kwargs):
    '''
    TODO: also updates each query object with the relevant stats that we want
    to plot.
    @queries: list of qrep objects
    @preds: list of dicts

    @output: updates ./results/join_order_loss.pkl file
    '''
    def run_join_loss_exp(env, cost_model):
        use_indexes = args.jl_indexes
        exp_name = kwargs["exp_name"]
        samples_type = kwargs["samples_type"]
        pool = kwargs["pool"]

        # here, we assume that the alg name is unique enough, for their results to
        # be grouped together
        rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                       ALG = exp_name)
        make_dir(rdir)
        costs_fn = rdir + cost_model + "_jerr.pkl"
        costs = load_object(costs_fn)
        if costs is None:
            columns = ["sql_key", "explain","plan","exec_sql","cost", "loss",
                    "postgresql_conf", "samples_type", "template", "qfn",
                    "card_key"]
            costs = pd.DataFrame(columns=columns)

        cur_costs = defaultdict(list)
        assert isinstance(costs, pd.DataFrame)

        est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                        join_loss_pg(sqls, join_graphs, true_cardinalities,
                                est_cardinalities, env, use_indexes, pdf=None,
                                pool = pool, join_loss_data_file =
                                args.join_loss_data_file)
        losses = est_costs - opt_costs
        for i, qrep in enumerate(eval_queries):
            sql_key = str(deterministic_hash(qrep["sql"]))
            if save_exec_sql:
                exec_sql = est_sqls[i]
            else:
                exec_sql = None
            add_query_result_row(sql_key, samples_type,
                    exec_sql, est_costs[i],
                    losses[i],
                    get_leading_hint(est_plans[i]),
                    qrep["template_name"], cur_costs, costs,
                    qrep["name"], cardinality_key)

        cur_df = pd.DataFrame(cur_costs)
        combined_df = pd.concat([costs, cur_df], ignore_index=True)
        save_object(costs_fn, combined_df)

        return est_costs, opt_costs

    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)

    # env = park.make('query_optimizer')
    args = kwargs["args"]
    save_exec_sql = args.save_exec_sql
    if args.db_name == "so":
        global SOURCE_NODE
        SOURCE_NODE = tuple(["SOURCE"])

    alg_name = kwargs["name"]
    cardinality_key = kwargs["cardinality_key"]

    assert "nested" in args.cost_model
    env2 = JoinLoss("cm1", args.user, args.pwd, args.db_host,
            args.port, args.db_name)

    est_cardinalities = []
    true_cardinalities = []
    sqls = []
    join_graphs = []
    # FIXME: wasting a lot of memory
    eval_queries = []

    # TODO: save alg based predictions too
    for i, qrep in enumerate(queries):
        # TODO: check if this prediction is valid from both predictor / and for
        # the ground truth
        if preds[i] is None:
            print("preds None!")
            pdb.set_trace()
            continue

        ests = {}
        trues = {}
        predq = preds[i]
        no_ground_truth_data = False

        for node, node_info in qrep["subset_graph"].nodes().items():
            if node == SOURCE_NODE:
                continue
            est_card = predq[node]
            alias_key = ' '.join(node)
            if cardinality_key not in node_info \
                    or "actual" not in node_info[cardinality_key]:
                no_ground_truth_data = True
                break

            trues[alias_key] = node_info[cardinality_key]["actual"]
            if est_card == 0:
                print("bad est card")
                est_card += 1
            ests[alias_key] = est_card

        if no_ground_truth_data:
            # print("no ground truth data!!! " + cardinality_key)
            # pdb.set_trace()
            continue

        eval_queries.append(qrep)
        sqls.append(qrep["sql"])
        join_graphs.append(qrep["join_graph"])
        est_cardinalities.append(ests)
        true_cardinalities.append(trues)

    # FIXME: avoiding nested_loop_index
    # est_costs, opt_costs = run_join_loss_exp(env, args.cost_model)

    # if "nested" in args.cost_model:
    assert "nested" in args.cost_model
    est_costs2, opt_costs2 = run_join_loss_exp(env2, "cm1")
    losses2 = est_costs2
    # print("case: {}: alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
            # .format(args.db_name, alg_name, len(queries),
                # "mysql plan cost",
                # np.round(np.mean(losses2),3),
                # np.round(np.median(losses2),3),
                # np.round(np.percentile(losses2,95),3),
                # np.round(np.percentile(losses2,99),3)))

    # dummy = []
    # save_object("dummy.pkl", dummy)

    return np.array(est_costs2) - np.array(opt_costs2)

def compute_flow_loss(queries, preds, **kwargs):

    cardinality_key = "cardinality"
    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)
    args = kwargs["args"]
    if args.db_name == "so":
        global SOURCE_NODE
        SOURCE_NODE = tuple(["SOURCE"])

    env = PlanError(args.cost_model, "flow-loss")
    exp_name = kwargs["exp_name"]
    samples_type = kwargs["samples_type"]
    pool = kwargs["pool"]

    # here, we assume that the alg name is unique enough, for their results to
    # be grouped together
    rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                   ALG = exp_name)
    make_dir(rdir)
    costs_fn = rdir + "flow_err.pkl"
    costs = load_object(costs_fn)
    if costs is None:
        columns = ["sql_key", "plan","cost", "loss","samples_type", "template",
                "qfn", "card_key"]
        costs = pd.DataFrame(columns=columns)

    cur_costs = defaultdict(list)
    assert isinstance(costs, pd.DataFrame)

    opt_costs, est_costs,_,_,_,_ = env.compute_loss(queries, preds, pool=pool)
    losses = est_costs - opt_costs
    for i, qrep in enumerate(queries):
        sql_key = str(deterministic_hash(qrep["sql"]))
        assert qrep["name"] is not None
        add_query_result_row(sql_key, samples_type, None, est_costs[i],
                losses[i], None, qrep["template_name"], cur_costs, costs,
                qrep["name"], cardinality_key)

    cur_df = pd.DataFrame(cur_costs)
    combined_df = pd.concat([costs, cur_df], ignore_index=True)
    save_object(costs_fn, combined_df)

    return np.array(est_costs) - np.array(opt_costs)

def compute_plan_loss(queries, preds, **kwargs):
    '''
    FIXME: a lot of code repetition w flow_loss, join order loss etc.
    '''
    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)
    args = kwargs["args"]
    if args.db_name == "so":
        global SOURCE_NODE
        SOURCE_NODE = tuple(["SOURCE"])

    # FIXME: plan-loss, need to change cardinality_key everywhere for
    # dynamic-db
    cardinality_key = "cardinality"

    env = PlanError(args.cost_model, "plan-loss", args.user, args.pwd,
            args.db_host, args.port, args.db_name, compute_pg_costs=True)
    exp_name = kwargs["exp_name"]
    alg_name = kwargs["name"]
    samples_type = kwargs["samples_type"]
    pool = kwargs["pool"]

    # here, we assume that the alg name is unique enough, for their results to
    # be grouped together
    rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                   ALG = exp_name)
    make_dir(rdir)
    costs_fn = rdir + "plan_err.pkl"
    costs_fn_pg = rdir + "plan_pg_err.pkl"
    costs = load_object(costs_fn)
    costs_pg = load_object(costs_fn_pg)

    if costs is None:
        columns = ["sql_key", "plan","cost", "loss","samples_type", "template",
                "qfn", "card_key"]
        costs = pd.DataFrame(columns=columns)
    if costs_pg is None:
        columns2 = ["sql_key", "explain","plan","exec_sql","cost", "loss",
                "postgresql_conf", "samples_type", "template", "qfn", "card_key"]
        costs_pg = pd.DataFrame(columns=columns2)

    cur_costs = defaultdict(list)
    cur_costs_pg = defaultdict(list)
    assert isinstance(costs, pd.DataFrame)
    assert isinstance(costs_pg, pd.DataFrame)

    true_cardinalities = []
    est_cardinalities = []
    join_graphs = []
    for i, qrep in enumerate(queries):
        # sqls.append(qrep["sql"])
        join_graphs.append(qrep["join_graph"])
        ests = {}
        trues = {}
        predq = preds[i]
        for node, node_info in qrep["subset_graph"].nodes().items():
            if node == SOURCE_NODE:
                continue
            est_card = predq[node]
            alias_key = ' '.join(node)
            trues[alias_key] = node_info["cardinality"]["actual"]
            # ests[alias_key] = int(est_card)
            if est_card == 0:
                print("bad est card")
                est_card += 1
            ests[alias_key] = est_card
        est_cardinalities.append(ests)
        true_cardinalities.append(trues)

    opt_costs, est_costs,opt_costs_pg,est_costs_pg, exec_sqls_pg, explains_pg = \
                env.compute_loss(queries, preds, pool=pool,
                        true_cardinalities=true_cardinalities,
                        join_graphs=join_graphs)

    # save plan_pg_err results
    losses = est_costs - opt_costs
    losses_pg = est_costs_pg - opt_costs_pg

    print("case: {}: alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
            .format(args.db_name, alg_name, len(queries),
                "plan_loss_pg",
                np.round(np.mean(losses_pg),3),
                np.round(np.median(losses_pg),3),
                np.round(np.percentile(losses_pg,95),3),
                np.round(np.percentile(losses_pg,99),3)))

    for i, qrep in enumerate(queries):
        sql_key = str(deterministic_hash(qrep["sql"]))
        assert qrep["name"] is not None
        add_query_result_row(sql_key, samples_type, None, est_costs[i], losses[i],
                None, qrep["template_name"], cur_costs, costs,
                qrep["name"], cardinality_key)
        add_query_result_row(sql_key, samples_type, exec_sqls_pg[i],
                est_costs_pg[i], losses_pg[i],
                get_leading_hint(explains_pg[i]), qrep["template_name"],
                cur_costs_pg, costs_pg, qrep["name"], cardinality_key)

    cur_df = pd.DataFrame(cur_costs)
    combined_df = pd.concat([costs, cur_df], ignore_index=True)
    save_object(costs_fn, combined_df)

    ## FIXME: not using these anymore, but add flags
    # cur_df_pg = pd.DataFrame(cur_costs_pg)
    # combined_df_pg = pd.concat([costs_pg, cur_df_pg], ignore_index=True)
    # save_object(costs_fn_pg, combined_df_pg)

    return np.array(est_costs) - np.array(opt_costs)

def compute_cost_model_loss_mysql(queries, preds, **kwargs):
    '''
    TODO: also updates each query object with the relevant stats that we want
    to plot.
    @queries: list of qrep objects
    @preds: list of dicts

    @output: updates ./results/join_order_loss.pkl file
    '''
    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)
    def get_est_plan(cm_plan_order):
        new_from = []
        for alias in cm_plan_order:
            table = join_graph.nodes()[alias]["real_name"]
            new_from.append("{} AS {}".format(table, alias))

        from_clause = " STRAIGHT_JOIN ".join(new_from)
        est_sql = nx_graph_to_query(join_graph, from_clause)
        est_sql = preprocess_sql_mysql(est_sql)
        est_sql_exec = est_sql
        est_sql = "EXPLAIN FORMAT=json " + est_sql
        cursor.execute(est_sql)
        out = cursor.fetchall()
        plan_explain = json.loads(out[0][0])

        est_join_order_forced = get_join_order_mysql(plan_explain)
        est_plan_cost=float(plan_explain["query_block"]["cost_info"]["query_cost"])
        # print("est_join_order_forced: ", est_join_order_forced)
        assert str(est_join_order_forced) == str(cm_plan_order)

        return est_plan_cost, plan_explain

    def debug_plan_orders(order, explain):
        path = []
        cur_node = []
        for node in order:
            cur_node.append(node)
            cur_node.sort()
            path.append(tuple(cur_node))
        subsetg = qrep["subset_graph"]
        compute_costs(subsetg, args.cost_model, "cardinality",
                cost_key="tmp_cost",
                ests=preds[i],
                mdata=mdata)

        path = path[::-1]
        costs = []
        for pi in range(len(path)-1):
            try:
                costs.append(subsetg[path[pi]][path[pi+1]][args.cost_model+"tmp_cost"])
            except:
                costs.append(-1)
        costs = costs[::-1]

        evalcost = extract_values(explain, "eval_cost")
        readcost = extract_values(explain, "read_cost")

        # print("shortest path plan, plan-cost: ", cm_est_cost)
        print("plan, our-cost: ", np.sum(costs))
        print(order)
        print("our-costs: ", costs)
        print("mysql readcosts: ", readcost)
        print("mysql evalcosts: ", evalcost)


    alg_name = kwargs["name"]
    cardinality_key = kwargs["cardinality_key"]
    args = kwargs["args"]

    # env = JoinLoss(args.cost_model, args.user, args.pwd, args.db_host,
            # args.port, args.db_name)

    est_cardinalities = []
    true_cardinalities = []
    sqls = []
    join_graphs = []
    # FIXME: wasting a lot of memory
    eval_queries = []

    # TODO: save alg based predictions too
    for i, qrep in enumerate(queries):
        # TODO: check if this prediction is valid from both predictor / and for
        # the ground truth
        if preds[i] is None:
            print("preds None!")
            pdb.set_trace()
            continue

        ests = {}
        trues = {}
        predq = preds[i]
        no_ground_truth_data = False

        for node, node_info in qrep["subset_graph"].nodes().items():
            if node == SOURCE_NODE:
                continue
            est_card = predq[node]
            alias_key = ' '.join(node)
            if cardinality_key not in node_info \
                    or "actual" not in node_info[cardinality_key]:
                no_ground_truth_data = True
                break

            trues[alias_key] = node_info[cardinality_key]["actual"]
            if est_card == 0:
                print("bad est card")
                est_card += 1
            ests[alias_key] = est_card

        if no_ground_truth_data:
            continue

        eval_queries.append(qrep)
        sqls.append(qrep["sql"])
        join_graphs.append(qrep["join_graph"])
        est_cardinalities.append(ests)
        true_cardinalities.append(trues)

    # est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                    # join_loss_pg(sqls, join_graphs, true_cardinalities,
                            # est_cardinalities, env, True, pdf=None,
                            # pool = None, join_loss_data_file =
                            # args.join_loss_data_file, backend="mysql")

    db = MySQLdb.connect(db="imdb", passwd="", user="root",
            host="127.0.0.1")
    # db = MySQLdb.connect(db="imdb", passwd="1234", user="root",
            # host="127.0.0.1")
    cursor = db.cursor()
    cm_losses = []
    cm_ratios = []

    for i,qrep in enumerate(queries):
        print(i)
        # if i % 10 == 0:
            # print(i)
        sql = sqls[i]

        # mysqldb stuff
        cursor.execute("SET optimizer_prune_level=0;")
        opt_flags = MYSQL_OPT_TMP.format(FLAGS=MYSQL_OPT_FLAGS)
        cursor.execute(opt_flags)
        sql = preprocess_sql_mysql(sql)
        sql = "EXPLAIN FORMAT=json " + sql
        join_graph = join_graphs[i]
        cards = true_cardinalities[i]
        with open(MYSQL_CARD_FILE_NAME, 'w') as fp:
            json.dump(cards, fp)

        # TODO: don't need this if we can cache these results
        cursor.execute(sql)
        out = cursor.fetchall()
        opt_plan_explain = json.loads(out[0][0])
        opt_join_order = get_join_order_mysql(opt_plan_explain)
        opt_cost=float(opt_plan_explain["query_block"]["cost_info"]["query_cost"])

        fn = qrep["name"]
        fn = fn.replace("queries", "mysql_data")
        # fn = fn.replace("queries", "tmp_mysql_data2")

        if os.path.exists(fn):
            # mdata = None
            mdata = load_object(fn)
        else:
            # fn2 = fn.replace("tmp_mysql_data", "tmp_fetched_rows")
            # fn3 = fn.replace("tmp_mysql_data", "tmp_read_costs")
            # make_dir(os.path.dirname(fn2))
            # make_dir(os.path.dirname(fn3))
            # os.rename("/tmp/fetched_rows.json", fn2)
            # os.rename("/tmp/read_costs.json", fn3)

            fetched_rows = {}
            rc = {}
            for line in open('/tmp/fetched_rows.json', 'r'):
                data = json.loads(line)
                assert len(data.keys()) == 1
                for k,v in data.items():
                    fetched_rows[k] = v


            for line in open('/tmp/read_costs.json', 'r'):
                data = json.loads(line)
                assert len(data.keys()) == 1
                for k,v in data.items():
                    rc[k] = v

            mdata = {}
            mdata["rc"] = rc
            mdata["rf"] = fetched_rows
            # let's save this ftw
            make_dir(os.path.dirname(fn))
            save_object(fn, mdata)

        # FIXME:
        cm_losses.append(1.0)
        cm_ratios.append(2.0)
        continue
        cm_opt_cost,cm_est_cost,opt_path,est_path= \
                get_simple_shortest_path_cost(qrep, preds[i], preds[i],
                args.cost_model, True, mdata=mdata)
        plan_loss = cm_opt_cost-cm_est_cost

        cm_plan_order = []
        est_path = est_path[::-1]
        cur_node = tuple([])

        for node in est_path:
            for alias in node:
                if alias not in cur_node:
                    cm_plan_order.append(alias)
            cur_node = node

        import copy
        cm2 = copy.deepcopy(cm_plan_order)
        cm2[0] = cm_plan_order[1]
        cm2[1] = cm_plan_order[0]

        est_cost1, est_explain1 = get_est_plan(cm_plan_order)
        est_cost2, est_explain2 = get_est_plan(cm2)
        if est_cost1 < est_cost2:
            est_plan_cost = est_cost1
            plan_explain = est_explain1
        else:
            est_plan_cost = est_cost2
            plan_explain = est_explain2
            cm_plan_order = cm2

        if False:
            print("opt plan  order: ", cm_plan_order)
            print("opt mysql order: ", opt_join_order)
            print("CM Loss: {}, Plan Loss: {}".format(est_plan_cost-opt_cost,
                                                      plan_loss))
            # print(cm_est_cost, cm_opt_cost)
            import copy
            opt2 = copy.deepcopy(opt_join_order)
            opt_join_order[0] = opt2[1]
            opt_join_order[1] = opt2[0]

            print("""DEBUGGING, shortest path plan-cost v/s mysql-cost for optimal mysql plan""")
            print("opt mysql-cost: ", opt_cost)
            debug_plan_orders(opt_join_order, opt_plan_explain)
            print("""DEBUGGING, shortest path plan-cost v/s mysql-cost for optimal shortest path""")
            print("est mysql-cost: ", est_plan_cost)
            debug_plan_orders(cm_plan_order, plan_explain)

            est_explain = plan_explain["query_block"]["nested_loop"]
            try:
                opt_explain = opt_plan_explain["query_block"]["nested_loop"]
            except:
                print("opt explain keys: ", opt_plan_explain["query_block"].keys())

            pdb.set_trace()

        cm_losses.append(est_plan_cost-opt_cost)
        cm_ratios.append(est_plan_cost / float(opt_cost))

    print("cost model ratio: ", np.mean(cm_ratios))
    return cm_losses



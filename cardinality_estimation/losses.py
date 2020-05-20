import numpy as np
import pdb
# import park
from utils.utils import *
from cardinality_estimation.query import *
from cardinality_estimation.join_loss import JoinLoss,PlanError

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
        loss, plan, template, cur_costs, costs, qfn):
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
        num_tables, samples_type, stats):
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

def qerr_loss_stats(samples, losses, samples_type,
        epoch):
    '''
    @samples: [] qrep objects.
    @preds: selectivity predictions for each

    @ret: dataframe summarizing all the stats
    '''
    stats = defaultdict(list)
    # assert "ordered" in type(samples[0]["subset_graph"])
    assert isinstance(samples[0]["subset_graph"], nx.OrderedDiGraph)

    add_row(losses, "qerr", epoch, "all", "all", samples_type,
            stats)
    summary_data = defaultdict(list)
    query_idx = 0
    for sample in samples:
        template = sample["template_name"]
        for subq_idx, node in enumerate(sample["subset_graph"].nodes()):
            num_tables = len(node)
            idx = query_idx + subq_idx
            loss = losses[idx]
            summary_data["loss"].append(loss)
            summary_data["num_tables"].append(num_tables)
            summary_data["template"].append(template)
        query_idx += len(sample["subset_graph"].nodes())

    df = pd.DataFrame(summary_data)

    # create new df summarizing the relevant results
    for template in set(df["template"]):
        tvals = df[df["template"] == template]
        add_row(tvals["loss"].values, "qerr", epoch,
                template, "all", samples_type, stats)
        for nt in set(tvals["num_tables"]):
            nt_losses = tvals[tvals["num_tables"] == nt]
            add_row(nt_losses["loss"].values, "qerr", epoch, template, str(nt),
                    samples_type, stats)

    for nt in set(df["num_tables"]):
        nt_losses = df[df["num_tables"] == nt]
        add_row(nt_losses["loss"].values, "qerr", epoch, "all", str(nt),
                samples_type, stats)

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

def _get_all_cardinalities(queries, preds):
    ytrue = []
    yhat = []
    totals = []
    for i, pred_subsets in enumerate(preds):
        qrep = queries[i]["subset_graph"].nodes()
        for alias, pred in pred_subsets.items():
            actual = qrep[alias]["cardinality"]["actual"]
            total = qrep[alias]["cardinality"]["total"]
            totals.append(total)
            # ytrue.append(float(actual) / total)
            # yhat.append(float(pred) / total)
            ytrue.append(float(actual))
            yhat.append(float(pred))
    return ytrue, yhat, totals

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

def compute_abs_loss(queries, preds, **kwargs):
    ytrue, yhat, totals = _get_all_cardinalities(queries, preds)
    errors = np.abs(yhat_total - ytrue)
    return errors

def compute_qerror(queries, preds, **kwargs):
    assert len(preds) == len(queries)
    assert isinstance(preds[0], dict)

    args = kwargs["args"]
    exp_name = kwargs["exp_name"]
    samples_type = kwargs["samples_type"]

    # here, we assume that the alg name is unique enough, for their results to
    # be grouped together
    rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                   ALG = exp_name)
    make_dir(rdir)

    ytrue, yhat, _ = _get_all_cardinalities(queries, preds)
    ytrue = np.array(ytrue)
    yhat = np.array(yhat)
    assert len(ytrue) == len(yhat)
    try:
        assert 0.00 not in ytrue
        assert 0.00 not in yhat
    except Exception as e:
        print(e)
        pdb.set_trace()

    errors = np.maximum((ytrue / yhat), (yhat / ytrue))
    df = qerr_loss_stats(queries, errors,
            samples_type, -1)

    fn = rdir + "/" + "qerr.pkl"
    # args_fn = rdir + "/" + "args.pkl"
    # save_object(args_fn, args)

    # update the qerrors here
    old_results = load_object(fn)
    if old_results is not None:
        df = pd.concat([old_results, df], ignore_index=True)

    save_object(fn, df)

    query_losses = {}
    query_idx = 0
    for sample in queries:
        template = sample["template_name"]
        cur_err = np.mean(errors[query_idx:query_idx+len(sample["subset_graph"].nodes())])
        query_losses[sample["name"]] = cur_err
        query_idx += len(sample["subset_graph"].nodes())

    qfn = rdir + "/" + "query_qerr.pkl"
    save_object(qfn, query_losses)
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
        join_loss_data_file=None):
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
                        num_processes=num_processes, postgres=True, pool=pool)
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

def compute_join_order_loss(queries, preds, **kwargs):
    '''
    TODO: also updates each query object with the relevant stats that we want
    to plot.
    @queries: list of qrep objects
    @preds: list of dicts

    @output: updates ./results/join_order_loss.pkl file
    '''
    # FIXME: remove this
    def add_joinresult_row(sql_key, samples_type, exec_sql, cost,
            loss, plan, template, cur_costs, costs, qfn):
        '''
        '''
        # TODO: add postgresql conf details too in check?
        if sql_key in costs["sql_key"].values:
            return

        cur_costs["sql_key"].append(sql_key)
        cur_costs["plan"].append(plan)
        cur_costs["exec_sql"].append(exec_sql)
        cur_costs["cost"].append(cost)
        cur_costs["loss"].append(loss)
        cur_costs["postgresql_conf"].append(None)
        cur_costs["samples_type"].append(samples_type)
        cur_costs["template"].append(template)
        cur_costs["qfn"].append(qfn)

    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)

    # env = park.make('query_optimizer')
    args = kwargs["args"]
    env = JoinLoss(args.cost_model, args.user, args.pwd, args.db_host,
            args.port, args.db_name)
    use_indexes = args.jl_indexes
    exp_name = kwargs["exp_name"]
    samples_type = kwargs["samples_type"]
    pool = kwargs["pool"]

    # here, we assume that the alg name is unique enough, for their results to
    # be grouped together
    rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                   ALG = exp_name)
    make_dir(rdir)
    costs_fn = rdir + "jerr.pkl"
    costs = load_object(costs_fn)
    if costs is None:
        columns = ["sql_key", "explain","plan","exec_sql","cost", "loss",
                "postgresql_conf", "samples_type", "template", "qfn"]
        costs = pd.DataFrame(columns=columns)

    cur_costs = defaultdict(list)

    assert isinstance(costs, pd.DataFrame)

    est_cardinalities = []
    true_cardinalities = []
    sqls = []
    join_graphs = []

    # TODO: save alg based predictions too
    for i, qrep in enumerate(queries):
        sqls.append(qrep["sql"])
        join_graphs.append(qrep["join_graph"])
        ests = {}
        trues = {}
        predq = preds[i]
        for node, node_info in qrep["subset_graph"].nodes().items():
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

    if args.jl_use_postgres:
        est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                        join_loss_pg(sqls, join_graphs, true_cardinalities,
                                est_cardinalities, env, use_indexes, pdf=None,
                                pool = pool, join_loss_data_file =
                                args.join_loss_data_file)
        losses = est_costs - opt_costs
        for i, qrep in enumerate(queries):
            sql_key = str(deterministic_hash(qrep["sql"]))
            add_query_result_row(sql_key, samples_type,
                    est_sqls[i], est_costs[i],
                    losses[i],
                    get_leading_hint(est_plans[i]),
                    qrep["template_name"], cur_costs, costs,
                    qrep["name"])
    else:
        print("TODO: add calcite based cost model")
        assert False

    cur_df = pd.DataFrame(cur_costs)
    combined_df = pd.concat([costs, cur_df], ignore_index=True)
    save_object(costs_fn, combined_df)

    losses = np.array(est_costs) - np.array(opt_costs)
    return np.array(est_costs) - np.array(opt_costs)

def compute_flow_loss(queries, preds, **kwargs):
    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)
    args = kwargs["args"]
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
                "qfn"]
        costs = pd.DataFrame(columns=columns)

    cur_costs = defaultdict(list)
    assert isinstance(costs, pd.DataFrame)

    opt_costs, est_costs,_,_ = env.compute_loss(queries, preds, pool=pool)
    losses = est_costs - opt_costs
    for i, qrep in enumerate(queries):
        sql_key = str(deterministic_hash(qrep["sql"]))
        assert qrep["name"] is not None
        add_query_result_row(sql_key, samples_type, None, est_costs[i], losses[i],
                None, qrep["template_name"], cur_costs, costs,
                qrep["name"])

    cur_df = pd.DataFrame(cur_costs)
    combined_df = pd.concat([costs, cur_df], ignore_index=True)
    save_object(costs_fn, combined_df)

    return np.array(est_costs) - np.array(opt_costs)

def compute_plan_loss(queries, preds, **kwargs):
    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)
    args = kwargs["args"]
    env = PlanError(args.cost_model, "plan-loss", args.user, args.pwd,
            args.db_host, args.port, args.db_name, compute_pg_costs=True)
    exp_name = kwargs["exp_name"]
    samples_type = kwargs["samples_type"]
    pool = kwargs["pool"]

    # here, we assume that the alg name is unique enough, for their results to
    # be grouped together
    rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                   ALG = exp_name)
    make_dir(rdir)
    costs_fn = rdir + "plan_err.pkl"
    costs = load_object(costs_fn)
    if costs is None:
        columns = ["sql_key", "plan","cost", "loss","samples_type", "template",
                "qfn"]
        costs = pd.DataFrame(columns=columns)

    cur_costs = defaultdict(list)
    assert isinstance(costs, pd.DataFrame)

    # for pg_costs
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

    opt_costs, est_costs,pg_opt_costs,pg_est_costs = env.compute_loss(queries,
            preds, pool=pool,
            true_cardinalities=true_cardinalities, join_graphs=join_graphs)
    losses = est_costs - opt_costs
    for i, qrep in enumerate(queries):
        sql_key = str(deterministic_hash(qrep["sql"]))
        assert qrep["name"] is not None
        add_query_result_row(sql_key, samples_type, None, est_costs[i], losses[i],
                None, qrep["template_name"], cur_costs, costs,
                qrep["name"])

    cur_df = pd.DataFrame(cur_costs)
    combined_df = pd.concat([costs, cur_df], ignore_index=True)
    save_object(costs_fn, combined_df)

    return np.array(est_costs) - np.array(opt_costs)

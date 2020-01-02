import numpy as np
import pdb
import park
from utils.utils import *
from cardinality_estimation.query import *
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

EPSILON = 0.000000001
REL_LOSS_EPSILON = EPSILON
QERR_MIN_EPS = EPSILON
CROSS_JOIN_CARD = 1313136191

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

def get_loss(loss):
    if loss == "abs":
        return compute_abs_loss
    elif loss == "rel":
        return compute_relative_loss
    elif loss == "qerr":
        return compute_qerror
    elif loss == "join-loss":
        return compute_join_order_loss
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

def _get_sel_arrays(queries, preds):
    ytrue = []
    yhat = []
    totals = []
    for i, pred_subsets in enumerate(preds):
        qrep = queries[i]["subset_graph"].nodes()
        for alias, pred in pred_subsets.items():
            actual = qrep[alias]["cardinality"]["actual"]
            total = qrep[alias]["cardinality"]["total"]
            totals.append(total)
            ytrue.append(float(actual) / total)
            yhat.append(pred / total)
    return ytrue, yhat, totals

# TODO: put the yhat, ytrue parts in db_utils
def compute_relative_loss(queries, preds, **kwargs):
    '''
    as in the quicksel paper.
    '''
    ytrue, yhat, _ = _get_sel_arrays(queries, preds)
    epsilons = np.array([REL_LOSS_EPSILON]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    errors = np.abs(ytrue - yhat) / ytrue
    return errors

def compute_abs_loss(queries, preds, **kwargs):
    ytrue, yhat, totals = _get_sel_arrays(queries, preds)
    yhat_total = np.multiply(yhat, totals)
    errors = np.abs(yhat_total - ytrue)
    return errors

def compute_qerror(queries, preds, **kwargs):
    assert len(preds) == len(queries)
    assert isinstance(preds[0], dict)
    ytrue, yhat, _ = _get_sel_arrays(queries, preds)
    ytrue = np.array(ytrue)
    yhat = np.array(yhat)
    epsilons = np.array([QERR_MIN_EPS]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    yhat = np.maximum(yhat, epsilons)
    errors = np.maximum((ytrue / yhat), (yhat / ytrue))
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

def join_loss_pg(sqls, true_cardinalities, est_cardinalities, env,
        pdf=None, num_processes=1):
    '''
    @sqls: [sql strings]
    @pdf: None, or open pdf file to which the plans and cardinalities will be
    plotted.

    @ret:
    '''
    for i,sql in enumerate(sqls):
        sqls[i] = fix_query(sql)
    est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                env.compute_join_order_loss(sqls,
                        true_cardinalities, est_cardinalities,
                        None, num_processes=num_processes, postgres=True)
    assert isinstance(est_costs, np.ndarray)

    if est_plans and pdf:
        print("going to plot query results for join-loss")
        for i, _ in enumerate(opt_costs):
            opt_cost = opt_costs[i]
            est_cost = est_costs[i]
            # plot both optimal, and estimated plans
            explain = est_plans[i]
            leading = get_leading_hint(explain)
            title = "Estimator Plan: {}, estimator cost: {}, opt cost: {}".format(\
                i, est_cost, opt_cost)
            estG = plot_explain_join_order(explain, true_cardinalities[i],
                    cardinalities[i], pdf, title)
            opt_explain = opt_plans[i]
            opt_leading = get_leading_hint(opt_explain)
            title = "Optimal Plan: {}, estimator cost: {}, opt cost: {}".format(\
                i, est_cost, opt_cost)
            optG = plot_explain_join_order(opt_explain, true_cardinalities[k],
                    cardinalities[k], pdf, title)

        print("num opt plans: {}, num est plans: {}".format(\
                len(all_opt_plans), len(all_est_plans)))

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
    @queries: list of qrep objects.
    @preds: list of dicts

    @output: updates ./results/join_order_loss.pkl file
    '''
    def add_joinresult_row(sql_key, exec_sql, cost, explain,
            plan, template):
        '''
        '''
        # TODO: add postgresql conf details too in check?
        if sql_key in costs["sql_key"].values:
            return

        cur_costs["sql_key"].append(sql_key)
        cur_costs["explain"].append(explain)
        cur_costs["plan"].append(plan)
        cur_costs["exec_sql"].append(exec_sql)
        cur_costs["cost"].append(cost)
        cur_costs["postgresql_conf"].append(None)
        cur_costs["samples_type"].append(samples_type)
        cur_costs["template"].append(template)

    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)

    env = park.make('query_optimizer')
    # TODO: do pdf stuff here
    args = kwargs["args"]
    alg_name = kwargs["name"]
    samples_type = kwargs["samples_type"]
    costs_dir_tmp = "{RESULT_DIR}/{ALG}/"
    costs_dir = costs_dir_tmp.format(RESULT_DIR = args.result_dir,
                                   ALG = alg_name)
    make_dir(costs_dir)
    costs_fn = costs_dir + "costs.pkl"
    costs = load_object(costs_fn)
    if costs is None:
        columns = ["sql_key", "explain","plan","exec_sql","cost",
                "postgresql_conf", "samples_type", "template"]
        costs = pd.DataFrame(columns=columns)

    cur_costs = defaultdict(list)

    assert isinstance(costs, pd.DataFrame)

    est_cardinalities = []
    true_cardinalities = []
    sqls = []

    # TODO: save alg based predictions too
    for i, qrep in enumerate(queries):
        sqls.append(qrep["sql"])
        ests = {}
        trues = {}
        predq = preds[i]
        for node, node_info in qrep["subset_graph"].nodes().items():
            est_card = predq[node]
            alias_key = ' '.join(node)
            trues[alias_key] = node_info["cardinality"]["actual"]
            # ests[alias_key] = int(est_card)
            ests[alias_key] = est_card
        est_cardinalities.append(ests)
        true_cardinalities.append(trues)

    if args.jl_use_postgres:
        est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                        join_loss_pg(sqls, true_cardinalities,
                                est_cardinalities, env, pdf=None,
                                num_processes=multiprocessing.cpu_count())

        for i, qrep in enumerate(queries):
            sql_key = str(deterministic_hash(qrep["sql"]))
            add_joinresult_row(sql_key, est_sqls[i], est_costs[i],
                    est_plans[i], get_leading_hint(est_plans[i]),
                    qrep["template_name"])
    else:
        print("TODO: add calcite based cost model")
        assert False

    cur_df = pd.DataFrame(cur_costs)

    combined_df = pd.concat([costs, cur_df], ignore_index=True)
    save_object(costs_fn, combined_df)

    env.clean()

    losses = np.array(est_costs) - np.array(opt_costs)
    print(losses[np.argmax(losses)])
    # pdb.set_trace()
    return np.array(est_costs) - np.array(opt_costs)

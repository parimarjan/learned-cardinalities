import numpy as np
import pdb
import park
from utils.utils import *
from cardinality_estimation.query import *
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

EPSILON = 0.000001
REL_LOSS_EPSILON = EPSILON
QERR_MIN_EPS = EPSILON
CROSS_JOIN_CARD = 1313136191

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
    epsilons = np.array([QERR_MIN_EPS]*len(yhat))
    ytrue = np.maximum(ytrue, epsilons)
    yhat = np.maximum(yhat, epsilons)
    errors = np.maximum((ytrue / yhat), (yhat / ytrue))
    print("compute qerr done!")
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
        pdf=None):
    '''
    @sqls: [sql strings]
    @pdf: None, or open pdf file to which the plans and cardinalities will be
    plotted.
    '''
    for i,sql in enumerate(sqls):
        sqls[i] = fix_query(sql)

    est_costs, opt_costs, est_plans, opt_plans = \
                env.compute_join_order_loss(sqls,
                        true_cardinalities, est_cardinalities,
                        None, True)

    if est_plans and pdf:
        print("going to plot query results for join-loss")
        for i, _ in enumerate(opt_costs):
            opt_cost = opt_costs[i]
            est_cost = est_card_costs[i]
            # plot both optimal, and estimated plans
            explain = est_plans[i]
            leading = get_leading_hint(explain)
            title = "Estimator Plan: {}, estimator cost: {}, opt cost: {}".format(\
                i, est_cost, opt_cost)
            estG = plot_explain_join_order(explain, true_cardinalities[k],
                    cardinalities[k], pdf, title)
            opt_explain = opt_plans[i]
            opt_leading = get_leading_hint(opt_explain)
            title = "Optimal Plan: {}, estimator cost: {}, opt cost: {}".format(\
                i, est_cost, opt_cost)
            optG = plot_explain_join_order(opt_explain, true_cardinalities[k],
                    cardinalities[k], pdf, title)

        print("num opt plans: {}, num est plans: {}".format(\
                len(all_opt_plans), len(all_est_plans)))

    return est_costs, opt_costs, est_plans, opt_plans

def compute_join_order_loss(queries, preds, **kwargs):
    '''
    TODO: also updates each query object with the relevant stats that we want
    to plot.
    '''
    env = park.make('query_optimizer')
    # TODO: do pdf stuff here
    args = kwargs["args"]
    if args.viz_join_plans:
        pdf_fn = args.viz_fn + os.path.basename(args.template_dir) \
                    + alg.__str__() + ".pdf"
        print("writing out join plan visualizations to ", pdf_fn)
        join_viz_pdf = PdfPages(pdf_fn)
    else:
        join_viz_pdf = None

    est_cardinalities = []
    true_cardinalities = []
    sqls = []
    for i, qrep in enumerate(queries):
        assert isinstance(qrep, dict)
        sqls.append(qrep["sql"])
        ests = {}
        trues = {}
        predq = preds[i]
        for node, node_info in qrep["subset_graph"].nodes().items():
            est_card = predq[node]
            alias_key = ' '.join(node)
            trues[alias_key] = node_info["cardinality"]["actual"]
            ests[alias_key] = int(est_card)
        est_cardinalities.append(ests)
        true_cardinalities.append(trues)

    if args.jl_use_postgres:
        est_card_costs, opt_costs, est_plans, opt_plans = \
                        join_loss_pg(sqls, true_cardinalities, est_cardinalities, env,
                                pdf=join_viz_pdf)

        all_opt_plans = set()
        all_est_plans = set()
        for i, _ in enumerate(opt_costs):
            opt_cost = opt_costs[i]
            est_cost = est_card_costs[i]
            # plot both optimal, and estimated plans
            explain = est_plans[i]
            leading = get_leading_hint(explain)
            all_est_plans.add(leading)
            opt_explain = opt_plans[i]
            opt_leading = get_leading_hint(opt_explain)
            all_opt_plans.add(opt_leading)

        print("num opt plans: {}, num est plans: {}".format(\
                len(all_opt_plans), len(all_est_plans)))
    else:
        print("TODO: add calcite based cost model")
        assert False

    env.clean()
    return np.array(est_card_costs) - np.array(opt_costs)

import pandas as pd
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scripts.parse_results import *
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
import networkx as nx
from cardinality_estimation.algs import *
import numpy as np
from cardinality_estimation.losses import *
#from cardinality_estimation.join_loss import JoinLoss, get_join_cost_sql, get_leading_hint
from cardinality_estimation.join_loss import *
from cardinality_estimation.nn import update_samples

from sql_rep.utils import nodes_to_sql, path_to_join_order
import cvxpy as cp
import time
import copy
from multiprocessing import Pool, freeze_support

# QUAD_BETAS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,1.9,2.0]
# QUAD_BETAS = [1.1]
# QUAD_BETAS = [2.0]
QUAD_BETAS = [1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
FLOW_LOSS_BETAS = [1.0, 2.0]

def get_qerr(true, est):
    qerrs = []
    for k,y in true.items():
        yhat = est[k]
        qerrs.append(max( y / yhat, yhat / y))
    return sum(qerrs) / len(qerrs)

def eval_alg(sql, y, yhat, join_graph):
    env = JoinLoss(COST_MODEL, USER, "", "localhost", 5432, "imdb")

    print("qerr: ", get_qerr(y, yhat))
    est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
                env.compute_join_order_loss([sql], [join_graph],
                        [y], [yhat], None, True, num_processes=1, postgres=True, pool=None)
    print("jerr: {}".format(est_costs[0]-opt_costs[0]))
    plot_explain_join_order(est_plans[0], y, yhat, None, "Plan based on Estimates")
    plt.show()
    # FIXME:
    plot_explain_join_order(opt_plans[0], y, y, None, "Plan based on true values")
    plt.show()
    return opt_costs[0], opt_plans[0]

def get_all_errors(y, yhat, qrep):
    env = JoinLoss(COST_MODEL, USER, "", "localhost", 5432, "imdb")
    flow_env = PlanError(COST_MODEL, "flow-loss")
    plan_env = PlanError(COST_MODEL, "plan-loss", USER, "", "localhost", 5432, "imdb")
    qerr = get_qerr(y, yhat)
    sql = qrep["sql"]
    join_graph = qrep["join_graph"]


    flow_errs = []
    flow_plan_errs = []
    flow_plans = []
    for beta in FLOW_LOSS_BETAS:
        opt_flow_cost, est_flow_cost, opt_cost, est_cost,opt_plan,est_plan = get_flow_cost(qrep, yhat, y, COST_MODEL,
                                                                            flow_loss_power=beta)
        flow_errs.append(est_flow_cost-opt_flow_cost)
        flow_plan_errs.append(est_cost-opt_cost)
        #flow_plans.append(opt_plan==est_plan)
        flow_plans.append(est_plan)

    # opt_costs, est_costs, est_plans, opt_plans, est_sqls, opt_sqls = plan_env.compute_loss([qrep], [yhat], true_cardinalities=[y], join_graphs=[join_graph], pool=pool)
    # plan_err = est_costs[0] - opt_costs[0]
    # plan_err = 0.0

    est_costs, opt_costs, est_plans, opt_plans, est_sqls, opt_sqls = \
            env.compute_join_order_loss([sql], [join_graph],
            [y], [yhat], None, True, num_processes=1, postgres=True, pool=None)
    join_err = est_costs[0] - opt_costs[0]

    opt_cost, est_cost,_,_ = get_simple_shortest_path_cost(qrep, yhat, y, COST_MODEL,
            directed=True)
    plan_err = est_cost - opt_cost

    opt_cost, est_cost,_,_ = get_simple_shortest_path_cost(qrep, yhat, y, COST_MODEL,
            directed=False)
    undirected_plan_err = est_cost - opt_cost

    quad_costs = []
    quad_flow_costs = []

    for beta in QUAD_BETAS:
        opt_flow_cost, est_flow_cost, opt_cost, est_cost,_,_ = \
            get_quadratic_program_cost(qrep, yhat, y, COST_MODEL, beta=beta,
                    alpha=beta)
        quadratic_cost = est_cost - opt_cost
        quadratic_flow_cost = est_flow_cost - opt_flow_cost
        quad_costs.append(quadratic_cost)
        quad_flow_costs.append(quadratic_flow_cost)

    return qerr,flow_errs,plan_err,join_err,undirected_plan_err,quad_costs,quad_flow_costs,flow_plan_errs,flow_plans

def check_errors(y, subsets, qrep, width=100):
    yhat = copy.deepcopy(y)
    xs = []
    qerrs = []
    flow_errs = []
    flow_plan_errs = []
    flow_plans = []
    plan_errs = []
    pg_errs = []
    und_plan_errs = []
    quadratic_errs = []
    quadratic_flow_errs = []

    par_args = []
    STEP=5
    for i in range(1, width+2,STEP):
        err_factor = i
        ests = []
        y1 = copy.deepcopy(y)
        yhat1 = copy.deepcopy(yhat)

        for subset in subsets:
            est1 = y1[subset] * err_factor
            yhat1[subset] = est1
            ests.append(est1)

        xs.append(np.mean(np.array(ests)))
        par_args.append((y1, yhat1, qrep))

        y2 = copy.deepcopy(y)
        yhat2 = copy.deepcopy(yhat)
        ests2 = []
        for subset in subsets:
            est2 = y2[subset] / err_factor
            yhat2[subset] = est2
            ests2.append(est2)
        xs.append(np.mean(np.array(ests2)))
        par_args.append((y2, yhat2, qrep))

    all_errs = pool.starmap(get_all_errors, par_args)
    for errors in all_errs:
        qerrs.append(errors[0])
        flow_errs.append(errors[1])
        plan_errs.append(errors[2])
        pg_errs.append(errors[3])
        und_plan_errs.append(errors[4])
        quadratic_errs.append(errors[5])
        quadratic_flow_errs.append(errors[6])
        flow_plan_errs.append(errors[7])
        flow_plans.append(errors[8])

    return xs,qerrs,flow_errs,plan_errs,pg_errs,und_plan_errs,quadratic_errs,\
        quadratic_flow_errs,flow_plan_errs,flow_plans


ERROR_FUNCS = [np.divide, np.multiply]
STEP_SIZE = 1
def check_errors2(y, subsets, subsets2, qrep, width=10):
    yhat = copy.deepcopy(y)
    xs = []
    ys = []
    qerrs = []
    flow_errs = []
    plan_errs = []
    pg_errs = []
    #assert len(subsets) == 1
    #assert len(subsets2) == 1

    for i in range(0, width,STEP_SIZE):
        err_factor = i
        #xs.append(err_factor)
        #print(err_factor)
        err_factor = 2**i

        for ef1 in ERROR_FUNCS:
            est1s = []
            for subset in subsets:
                est1 = ef1(y[subset], err_factor)
                yhat[subset] = est1
                est1s.append(est1)

            for j in range(1, width,STEP_SIZE):
                for ef2 in ERROR_FUNCS:
                    est2s = []
                    err_factor2 = j
                    for subset in subsets2:
                        est2 = ef2(y[subset], err_factor2)
                        yhat[subset] = est2
                        est2s.append(est2)

                    errors = get_all_errors(y, yhat, qrep)
                    qerrs.append(errors[0])
                    flow_errs.append(errors[1])
                    plan_errs.append(errors[2])
                    pg_errs.append(errors[3])

                    xs.append(np.mean(np.array(est1s)))
                    ys.append(np.mean(np.array(est2s)))
                    #xs.append(err_factor)
                    #ys.append(err_factor2)


    return xs,ys,qerrs,flow_errs,plan_errs,pg_errs

#COST_MODEL = "nested_loop_index8"
COST_MODEL = "cm1"
COST_KEY = COST_MODEL + "cost"


QUERY_DIR = "./debug_sqls/"
# QUERY_NAME = "1.pkl"
QUERY_NAME = "2.pkl"

qfn = QUERY_DIR + QUERY_NAME
postgres = Postgres()
true_alg = TrueCardinalities()

USER = "pari"
PWD = ""
HOST = "localhost"
PORT = 5432
DB_NAME = "imdb"
SAVE_DIR = "./QueryLabPlots/"

OUTPUT_DIR = "./all_subquery_quad/{}".format(QUERY_NAME)
OUTPUT_NAME = OUTPUT_DIR+"/query_data.pkl"

pool = None
def main():
    def run_subset(subsets, data):
        add_single_node_edges(qrep["subset_graph"])
        xs,qerrs,flow_errs,plan_errs,pg_errs,und_plan_errs,\
            quadratic_errs,quad_flow_errs,flow_plan_errs,flow_plans = \
                check_errors(y[0],subsets, qrep, width=2000)

        subsets.sort()
        subsets = str(subsets)
        data[subsets] = (xs,qerrs,flow_errs,plan_errs,pg_errs,und_plan_errs,quadratic_errs,quad_flow_errs,flow_plan_errs,flow_plans)

    global pool
    pool = Pool(10)
    qrep = load_sql_rep(qfn)
    join_graph = qrep["join_graph"]
    subset_graph = qrep["subset_graph"]
    sql = qrep["sql"]
    print(sql)
    qrep["name"] = "whatever"
    update_samples([qrep], 0, COST_MODEL, 1)
    y = true_alg.test([qrep])

    data = {}

    make_dir(OUTPUT_DIR)
    # for node in qrep["subset_graph"]:
        # if node == SOURCE_NODE:
            # continue
        # run_subset([node], data)
        # print("****DONE****")
        # print(node)
        # print("****DONE****")
        # save_object(OUTPUT_NAME, data)

    SUBSETS_TO_PLOT = [tuple(['ci', 'kt', 'rt', 't'])]
    for node in SUBSETS_TO_PLOT:
        if node == SOURCE_NODE:
            continue
        run_subset([node], data)
        print("****DONE****")
        print(node)
        print("****DONE****")
        save_object(OUTPUT_NAME, data)


    #subsets = [("ci", "n")]
    #subsets = [("kt",)]
    # subsets = [("ci", "kt", "t")]
    # run_subset(subsets, data)
    # save_object(OUTPUT_NAME, data)

    print("All subsets run!")
    pool.close()
    exit(-1)

if __name__ == "__main__":
    freeze_support()
    main()

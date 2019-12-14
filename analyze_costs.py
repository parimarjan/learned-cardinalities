import pickle
import glob
import pdb
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from collections import defaultdict
import os
from utils.utils import *

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results1")
    parser.add_argument("--worst_rt_costs_dir", type=str, required=False,
            default=None)
    return parser.parse_args()

def output_worst_costs(wsql_keys):
    for alg_dir in os.listdir(args.results_dir):
        if alg_dir not in ["postgres", "true"]:
            continue
        costs_dir = args.results_dir + "/" + alg_dir
        costs_fn = costs_dir + "/" + "costs.pkl"
        costs = load_object(costs_fn)
        wdir = args.worst_rt_costs_dir + "/" + alg_dir
        make_dir(wdir)
        wfn = wdir + "/" + "costs.pkl"
        costs = costs[costs["sql_key"].isin(wsql_keys)]
        save_object(wfn, costs)

def main():
    result_dir = args.results_dir
    comb_rts = None
    all_wsql_keys = []
    for alg_dir in os.listdir(result_dir):
        if alg_dir not in ["postgres", "true"]:
            continue
        costs_fn = result_dir + "/" + alg_dir + "/" + "costs.pkl"
        rt_fn = result_dir + "/" + alg_dir + "/" + "runtimes.pkl"
        costs = load_object(costs_fn)
        # costs = costs[~costs["template"].isin(["2U3.toml"])]
        rts = load_object(rt_fn)
        if rts is None:
            continue
        # write out worst runtimes
        if args.worst_rt_costs_dir:
            wfn = args.worst_rt_costs_dir + "/" + alg_dir + "/" + "costs.pkl"
            wrts = rts.sort_values(by=["runtime"], ascending=False)
            # print(rts[0:10])
            print(wrts[0:10]["runtime"])
            wsql_keys = list(set(wrts[0:100]["sql_key"]))
            all_wsql_keys += wsql_keys

        rts = rts.merge(costs, on="sql_key")
        rts = rts[["sql_key", "runtime","template"]]

        rts = rts.rename(columns={"runtime":alg_dir})
        if comb_rts is None:
            comb_rts = rts
        else:
            comb_rts = comb_rts.merge(rts, on="sql_key")

    if args.worst_rt_costs_dir:
        output_worst_costs(all_wsql_keys)

    print(comb_rts.describe())
    comb_rts_true = comb_rts.sort_values(by=["true"], ascending=False)
    print(comb_rts_true)
    pdb.set_trace()
    comb_rts_pg = comb_rts.sort_values(by=["postgres"], ascending=False)
    print(comb_rts_pg)
    pdb.set_trace()

args = read_flags()
main()

### analyzing costs
# TODO: do this based on result logs
# all_opt_plans = defaultdict(list)
# all_est_plans = defaultdict(list)
# num_opt_plans = []
# num_est_plans = []
# for i, _ in enumerate(opt_costs):
    # opt_cost = opt_costs[i]
    # est_cost = est_costs[i]
    # # plot both optimal, and estimated plans
    # explain = est_plans[i]
    # leading = get_leading_hint(explain)
    # opt_explain = opt_plans[i]
    # opt_leading = get_leading_hint(opt_explain)
    # sql = queries[i]["sql"]
    # template = queries[i]["template_name"]

    # all_est_plans[leading].append((template, deterministic_hash(sql), sql))
    # all_opt_plans[opt_leading].append((template, deterministic_hash(sql), sql))

# print("num opt plans: {}, num est plans: {}".format(\
        # len(all_opt_plans), len(all_est_plans)))

## saving per bucket summaries
# print(all_opt_plans.keys())
# for k,v in all_opt_plans.items():
    # num_opt_plans.append(len(v))
# for k,v in all_est_plans.items():
    # num_est_plans.append(len(v))

# print(sorted(num_opt_plans, reverse=True)[0:10])
# print(sorted(num_est_plans, reverse=True)[0:10])

# with open("opt_plan_summaries.pkl", 'wb') as fp:
    # pickle.dump(all_opt_plans, fp, protocol=pickle.HIGHEST_PROTOCOL)

# with open("est_plan_summaries.pkl", 'wb') as fp:
    # pickle.dump(all_est_plans, fp, protocol=pickle.HIGHEST_PROTOCOL)

# pdb.set_trace()

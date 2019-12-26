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

PERCENTILES_TO_SAVE = [99]
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f

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

def rt_loss(df, pdf):
    sns.scatterplot(x="c_loss",y="rt_loss",data=df)
    pdf.savefig()
    plt.clf()

def cost_rt(df, pdf):
    df = df[df["runtime"] <= 200]
    df = df[df["cost"] <= 1e8]
    sns.scatterplot(x="cost",y="runtime",data=df)
    pdf.savefig()
    plt.clf()

def plot_summaries(df, samples_type, pdf, est,
        est_type, y):
    df = df[df["samples_type"] == samples_type]
    sns.barplot(x="alg", y = y, data=df,
            estimator=est)
    plt.title("{}: {} {}".format(samples_type, est_type, y))
    pdf.savefig()
    plt.clf()

def template_summaries(df, samples_type, pdf, y):
    df = df[df["samples_type"] == samples_type]
    df = df.groupby(["template", "alg"]).mean().reset_index()
    fg = sns.FacetGrid(df, col = "template", hue="alg",
            col_wrap=3, sharey=False, sharex=True)
    fg = fg.map(plt.bar, "alg", y)
    fg.axes[0].legend(loc='upper left')

    # TODO: set how many queries are there on each table
    # for i, ax in enumerate(fg.axes.flat):
        # tmp = templates[i]
        # sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        # title = tmp + " ,#subqueries: " + str(sqs)
        # ax.set_title(title)

    # title
    fg.fig.suptitle("{}: {}".format(samples_type, y),
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 40)

    fg.despine(left=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig()
    plt.clf()

def main():
    result_dir = args.results_dir
    all_dfs = []
    key_intersects = None
    for alg_dir in os.listdir(result_dir):
        if alg_dir not in ["true"]:
            continue
        costs_fn = result_dir + "/" + alg_dir + "/" + "costs.pkl"
        rt_fn = result_dir + "/" + alg_dir + "/" + "runtimes.pkl"
        costs = load_object(costs_fn)
        if "template" in costs:
            costs = costs[~costs["template"].isin(["2U3.toml"])]
        costs = costs.drop_duplicates(subset="sql_key")
        rts = load_object(rt_fn)
        if rts is None:
            continue
        rts = rts.drop_duplicates(subset="sql_key")
        if key_intersects is None:
            key_intersects = set(rts["sql_key"])
        else:
            key_intersects = key_intersects.intersection(set(rts["sql_key"]))
        costs = pd.merge(costs, rts, on="sql_key", how="left")
        print("{} queries without runtime: {}".format(alg_dir,
            len(costs[costs["runtime"].isna()])))
        # if "msft" in alg_dir:
            # if "pr" in alg_dir:
                # alg_dir = "priority"
            # else:
                # alg_dir = "microsoft"

        alg_col = [alg_dir]*len(costs)
        costs["alg"] = alg_col
        all_dfs.append(costs)

    df = pd.concat(all_dfs, ignore_index=True)
    orig_size = len(df)
    df = df[df["sql_key"].isin(key_intersects)]
    print("samples reduced from {} to {} because not all runtime there"\
            .format(orig_size, len(df)))

    pdb.set_trace()

    pdf = PdfPages("results.pdf")

    plot_summaries(df, "train", pdf, np.mean, "Mean", "runtime")
    plot_summaries(df, "test", pdf, np.mean, "Mean", "runtime")

    # for ptile in PERCENTILES_TO_SAVE:
        # est_type = "percentile: " + str(ptile)
        # plot_summaries(df, "train", pdf, percentile_help(ptile), est_type,
        # "runtime")
        # plot_summaries(df, "test", pdf, percentile_help(ptile), est_type,
        # "runtime")

    plot_summaries(df, "train", pdf, np.mean, "Mean", "cost")
    plot_summaries(df, "test", pdf, np.mean, "Mean", "cost")

    template_summaries(df, "train", pdf, "runtime")
    template_summaries(df, "test", pdf, "runtime")

    template_summaries(df, "train", pdf, "cost")
    template_summaries(df, "test", pdf, "cost")

    # bar-chart sorted by each alg

    pdf.close()

    true_df = df[df["alg"] == "true"]
    pdb.set_trace()

    # comb_rts_true = comb_rts.sort_values(by=["true"], ascending=False)
    # print(comb_rts_true)
    # pdb.set_trace()
    # comb_rts_pg = comb_rts.sort_values(by=["postgres"], ascending=False)
    # print(comb_rts_pg)
    # pdb.set_trace()

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

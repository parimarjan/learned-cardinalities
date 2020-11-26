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

import sys
sys.path.append(".")
from utils.utils import *
from db_utils.utils import *

DROP_TIMEOUTS = False

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
    df = df[df["runtime"] <= 50]
    df = df[df["cost"] <= 1e6]
    sns.scatterplot(x="cost",y="runtime",data=df, hue="alg")
    pdf.savefig()
    plt.clf()

def save_data(df):
    for alg in set(df["alg"]):
        data = df[df["alg"] == alg]["runtime"].to_numpy()
        fn = alg + "_runtimes.pkl"
        save_object(fn, data)

def plot_fig1_dist(df):
    df = df[df["samples_type"] == "test"]
    save_data(df)
    true_rts = df[df["alg"] == "true"]["runtime"]
    pg_rts = df[df["alg"] == "postgres"]["runtime"]
    mscn_rts = df[df["alg"] == "mscn"]["runtime"]
    mscn_pr_rts = df[df["alg"] == "mscn-priority"]["runtime"]

    microsoft_rts = df[df["alg"] == "microsoft"]["runtime"]
    microsoft_pr_rts = df[df["alg"] == "microsoft-priority"]["runtime"]

    # COLORS = ["red", "orange", "green", "cyan"]
    sns.distplot(mscn_rts, label="mscn", hist=False, color="red")
    sns.distplot(mscn_pr_rts, label="mscn-priority", hist=False, color="orange")
    sns.distplot(true_rts, label="true", hist=False, color="black")
    ax = sns.distplot(pg_rts, label="postgres", hist=False, color="blue")
    # plt.xlim([0,200])
    plt.xlim([0,100])

    ax.lines[2].set_linestyle("--")
    ax.lines[3].set_linestyle("--")
    ax.lines[2].set_alpha(0.5)
    ax.lines[3].set_alpha(0.5)

    plt.savefig("runtime_dist_mscn.pdf")
    plt.clf()

    sns.distplot(microsoft_rts, label="fcnn", hist=False, color="green")
    sns.distplot(microsoft_pr_rts, label="fcnn-priority", hist=False,
        color="cyan")
    ax = sns.distplot(true_rts, label="true", hist=False, color="black")
    ax = sns.distplot(pg_rts, label="postgres", hist=False,
        color="blue")

    ax.lines[2].set_linestyle("--")
    ax.lines[3].set_linestyle("--")
    ax.lines[2].set_alpha(0.5)
    ax.lines[3].set_alpha(0.5)

    plt.xlim([0,100])

    plt.savefig("runtime_dist_microsoft.pdf")
    plt.clf()

def plot_fig1(df):
    algs_order = ["postgres", "mscn", "mscn-priority", "microsoft",
        "microsoft-priority", "true"]
    df = df[df["samples_type"] == "test"]
    df = pd.melt(df, value_vars=["runtime", "cost"], id_vars="alg")
    # print(set(df["alg"]))

    plot_order = ["cost", "runtime"]
    fg = sns.catplot(x = "alg",
            y = "value",
            # row = "variable",
            col = "variable",
            data = df,
            hue = "alg", order = algs_order, hue_order = algs_order,
            sharey = False,
            # kind="bar",
            # kind="box",
            kind="boxen",
            # row_order=plot_order,
            col_order=plot_order,
            ci = None,
            legend = False,
            # legend_out = True
            )

    fg.set(yscale="log")
    fg.set(ylabel="")


    for i, ax in enumerate(fg.axes.flatten()):
        if i == 0:
            # ax.legend(loc="upper left")
            # ax.legend(bbox_to_anchor=(0.0,1.0), loc="upper left")
            ax.set_ylabel('cost', fontsize=8)
            ax.set_title("Query Costs")
        else:
            ax.set_ylabel('seconds', fontsize=8)
            ax.set_title("Query Runtimes")

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    # fg.add_legend()
    # plt.legend(loc='upper left')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    # fg.add_legend()

    # plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig("runtime_fig.png")

    plt.clf()

def plot_summaries(df, samples_type, pdf, est,
        est_type, y):
    ORDER = ["postgres", "mscn1-10", "mscn-P1-10", "microsoft1-10",
            "microsoft-P1-10", "true"]
    df = df[df["samples_type"] == samples_type]
    sns.barplot(x="alg", y = y, data=df,
            estimator=est, order = ORDER)
    plt.title("{}: {} {}".format(samples_type, est_type, y))
    plt.yscale("log")
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

def get_alg_name(exp_args):
    if exp_args["algs"] == "nn":
        name = exp_args["nn_type"]
        if name == "microsoft":
            name = "fcnn"
        if exp_args["sampling_priority_alpha"] > 0:
            name += "-priority"
    else:
        name = exp_args["algs"]

    return name

def main():
    result_dir = args.results_dir
    all_dfs = []
    key_intersects = None
    for alg_dir in os.listdir(result_dir):
        # if alg_dir not in ["true"]:
            # continue
        costs_fn = result_dir + "/" + alg_dir + "/" + "jerr.pkl"
        rt_fn = result_dir + "/" + alg_dir + "/" + "runtimes.pkl"
        costs = load_object(costs_fn)
        if costs is None:
            print(costs_fn)
            pdb.set_trace()
        costs = costs.drop_duplicates(subset="sql_key")
        rts = load_object(rt_fn)
        if rts is None:
            continue
        rts = rts.drop_duplicates(subset="sql_key")
        exp_args = load_object(result_dir + "/" + alg_dir + "/" + "args.pkl")
        exp_args = vars(exp_args)
        alg_name = get_alg_name(exp_args)
        if exp_args["sampling_priority_alpha"] == 2.0:
            rts["priority"] = "yes"
        else:
            rts["priority"] = "no"

        num_timeouts = len(rts[rts["runtime"] >= 909])
        print("{}, timed out queries: {}".format(alg_name, num_timeouts))
        if DROP_TIMEOUTS:
            orig_len = len(rts)
            rts = rts[rts["runtime"] < 909]
            print("dropped: {} queries because of timeout".format(len(rts) - orig_len))

        print("{}: {} runtimes".format(alg_name, len(rts)))
        if key_intersects is None:
            key_intersects = set(rts["sql_key"])
        else:
            key_intersects = key_intersects.intersection(set(rts["sql_key"]))
        costs = pd.merge(costs, rts, on="sql_key", how="left")
        print("{} queries without runtime: {}".format(alg_dir,
            len(costs[costs["runtime"].isna()])))

        costs["alg"] = alg_name
        all_dfs.append(costs)

    df = pd.concat(all_dfs, ignore_index=True)
    orig_size = len(df)
    df = df[df["sql_key"].isin(key_intersects)]
    print("samples reduced from {} to {} because not all runtime there"\
            .format(orig_size, len(df)))

    print(df.groupby("alg")["runtime"].describe())

    pdb.set_trace()

    pdf = PdfPages("runtimes.pdf")
    plot_summaries(df, "test", pdf, np.mean, "Mean", "runtime")
    plot_summaries(df, "test", pdf, np.mean, "Mean", "cost")


    # template_summaries(df, "train", pdf, "runtime")
    template_summaries(df, "test", pdf, "runtime")
    template_summaries(df, "test", pdf, "cost")

    # bar-chart sorted by each alg

    cost_rt(df, pdf)
    pdf.close()


args = read_flags()
main()

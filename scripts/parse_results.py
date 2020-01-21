import sys
sys.path.append(".")
import pickle
import glob
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
import pdb
from cardinality_estimation.losses import *

LOSS_COLUMNS = ["loss_type", "loss", "summary_type", "template", "num_samples",
        "samples_type"]
EXP_COLUMNS = ["num_hidden_layers", "hidden_layer_size",
        "sampling_priority_alpha", "max_discrete_featurizing_buckets",
        "heuristic_features", "alg"]
PLOT_SUMMARY_TYPES = ["mean"]
ALGS_ORDER = ["mscn", "mscn-priority", "microsoft", "microsoft-priority"]

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--only_test", type=int, required=False,
            default=1)
    return parser.parse_args()

def load_qerrs(exp_dir):
    qerrs = load_object(exp_dir + "/qerr.pkl")
    if qerrs is None:
        assert False
    return qerrs

def load_jerrs(exp_dir):
    jerrs = load_object(exp_dir + "/jerr.pkl")
    if jerrs is None:
        print("jerr not found for: ", exp_dir)
        return None

    stats = defaultdict(list)

    for samples_type in set(jerrs["samples_type"]):
        cur_jerrs = jerrs[jerrs["samples_type"] == samples_type]
        add_row(cur_jerrs["cost"].values, "jcost", -1, "all", "all", samples_type,
                stats)
        for template in set(cur_jerrs["template"]):
            tmp_jerrs = cur_jerrs[cur_jerrs["template"] == template]
            add_row(tmp_jerrs["cost"].values, "jcost", -1, template, "all",
                    samples_type, stats)

    return pd.DataFrame(stats)

def get_alg_name(exp_args):
    if exp_args["algs"] == "nn":
        name = exp_args["nn_type"]
        if exp_args["sampling_priority_alpha"] > 0:
            # name += "-" + str(exp_args["sampling_priority_alpha"])
            name += "-" + "priority"
        return name
    else:
        return exp_args["algs"]

def skip_exp(exp_args):
    if exp_args["sampling_priority_alpha"] > 2.00:
        return True
    if exp_args["max_discrete_featurizing_buckets"] > 10:
        return True

    return False

def analyze_joins():
    fns = os.listdir(args.results_dir)
    for fn in fns:
        cur_dir = args.results_dir + "/" + fn
        jerrs = load_object(cur_dir + "/jerr.pkl")
        if jerrs is None:
            continue
        jerrs = jerrs[["cost", "samples_type", "template"]]

def get_all_qerrs():
    all_dfs = []
    fns = os.listdir(args.results_dir)
    for fn in fns:
        # convert to same format as qerrs
        cur_dir = args.results_dir + "/" + fn
        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            continue
        exp_args = vars(exp_args)
        if skip_exp(exp_args):
            continue

        try:
            qerrs = load_object(cur_dir + "/qerr.pkl")
        except:
            print("skipping ", cur_dir)
            continue
        if qerrs is None:
            continue

        exp_args["alg"] = get_alg_name(exp_args)
        for exp_column in EXP_COLUMNS:
            qerrs[exp_column] = exp_args[exp_column]

        all_dfs.append(qerrs)

    df = pd.concat(all_dfs, ignore_index=True)
    if args.only_test:
        df = df[df["samples_type"] == "test"]
    return df

def get_all_jerrs():
    all_dfs = []
    fns = os.listdir(args.results_dir)
    for fn in fns:
        # convert to same format as qerrs
        cur_dir = args.results_dir + "/" + fn
        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            continue
        exp_args = vars(exp_args)
        if skip_exp(exp_args):
            continue

        try:
            jerrs = load_object(cur_dir + "/jerr.pkl")
        except:
            print("skipping ", cur_dir)
            continue
        if jerrs is None:
            continue

        exp_args["alg"] = get_alg_name(exp_args)
        for exp_column in EXP_COLUMNS:
            jerrs[exp_column] = exp_args[exp_column]

        all_dfs.append(jerrs)

    df = pd.concat(all_dfs, ignore_index=True)
    if args.only_test:
        df = df[df["samples_type"] == "test"]
    return df

def plot_join_summaries(pdf):
    df = get_all_jerrs()
    for st in set(df["samples_type"]):
        cur_df = df[df["samples_type"] == st]
        # plot it in pdf

        fg = sns.catplot(x="alg", y="cost",
                data=df, row="max_discrete_featurizing_buckets",
                col = "hidden_layer_size", hue="alg", kind="bar",
                order=ALGS_ORDER, hue_order=ALGS_ORDER, ci=95)
        fg.add_legend()

        title = "Join Error"
        fg.fig.suptitle(title,
                x=0.5, y=.99, horizontalalignment='center',
                verticalalignment='top', fontsize = 40)
        fg.despine(left=True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.clf()

def plot_join_summaries2(pdf):
    df = get_all_jerrs()
    HUES_ORDER = [0, 1]

    for st in set(df["samples_type"]):
        cur_df = df[df["samples_type"] == st]
        fg = sns.catplot(x="alg", y="cost",
                data=df, row="max_discrete_featurizing_buckets",
                col = "hidden_layer_size", hue="heuristic_features",
                join = False,
                kind="point",
                sharex=False,
                order=ALGS_ORDER, hue_order=HUES_ORDER, ci=95)
        fg.add_legend()

        title = "Join Error"
        fg.fig.suptitle(title,
                x=0.5, y=.99, horizontalalignment='center',
                verticalalignment='top', fontsize = 40)
        fg.despine(left=True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.clf()

def plot_qerr_summaries2(pdf):
    df = get_all_qerrs()
    df = df[df["summary_type"] == "mean"]
    HUES_ORDER = [0, 1]

    for st in set(df["samples_type"]):
        cur_df = df[df["samples_type"] == st]
        fg = sns.catplot(x="alg", y="loss",
                data=df, row="max_discrete_featurizing_buckets",
                col = "hidden_layer_size", hue="heuristic_features",
                join = False,
                sharex=False,
                kind="point",
                order=ALGS_ORDER, hue_order=HUES_ORDER, ci=95)
        fg.add_legend()

        title = "QError"
        fg.fig.suptitle(title,
                x=0.5, y=.99, horizontalalignment='center',
                verticalalignment='top', fontsize = 40)
        fg.despine(left=True)

        # fg.set(ylim=(1.0,1000.0))
        # plt.ylim([1.0,1000.0])

        fg.set(yscale="log")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.clf()

def get_summary_df():
    all_dfs = []
    fns = os.listdir(args.results_dir)
    for fn in fns:
        # convert to same format as qerrs
        cur_dir = args.results_dir + "/" + fn
        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            continue
        exp_args = vars(exp_args)
        exp_args["alg"] = get_alg_name(exp_args)
        if skip_exp(exp_args):
            continue

        try:
            qerrs = load_qerrs(cur_dir)
            jerrs = load_jerrs(cur_dir)
        except:
            print("skipping ", cur_dir)
            continue
        qerrs = qerrs[qerrs["num_tables"] == "all"]
        qerrs = qerrs[LOSS_COLUMNS]

        if jerrs is None:
            continue
        jerrs = jerrs[LOSS_COLUMNS]
        # TODO: add rts too, if it exists

        cur_df = pd.concat([qerrs, jerrs], ignore_index=True)

        for exp_column in EXP_COLUMNS:
            cur_df[exp_column] = exp_args[exp_column]

        all_dfs.append(cur_df)

    summary_df = pd.concat(all_dfs, ignore_index=True)
    return summary_df

def plot_summary(pdf, df, title):
    fg = sns.catplot(x="alg", y="loss",
            data=df, row="max_discrete_featurizing_buckets",
            col = "hidden_layer_size", hue="alg", kind="bar",
            order = ALGS_ORDER, hue_order = ALGS_ORDER)
    # fg.axes[0].legend(loc='upper left')
    fg.add_legend()

    # TODO: set how many queries are there on each table
    # for i, ax in enumerate(fg.axes.flat):
        # tmp = templates[i]
        # sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        # title = tmp + " ,#subqueries: " + str(sqs)
        # ax.set_title(title)

    # title
    fg.fig.suptitle(title,
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 40)

    # fg.set(ylim=(0,10.0))
    fg.despine(left=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig()
    plt.clf()

def main():
    pdf = PdfPages("results.pdf")
    plot_qerr_summaries2(pdf)
    plot_join_summaries2(pdf)

    # summary_df = get_summary_df()
    # SUMMARY_TITLE_FMT = "{ST}-{LT}-{SUMMARY}"
    # if args.only_test:
        # summary_df = summary_df[summary_df["samples_type"] == "test"]
    # for samples_type in set(summary_df["samples_type"]):
        # st_df = summary_df[summary_df["samples_type"] == samples_type]
        # for lt in set(st_df["loss_type"]):
            # lt_df = st_df[st_df["loss_type"] == lt]
            # for summary_type in PLOT_SUMMARY_TYPES:
                # plot_df = lt_df[lt_df["summary_type"] == summary_type]
                # plot_df = plot_df[plot_df["template"] == "all"]
                # # print(set(plot_df["alg"]))
                # title = SUMMARY_TITLE_FMT.format(ST = samples_type,
                                                 # LT = lt,
                                                 # SUMMARY = summary_type)
                # plot_summary(pdf, plot_df, title)

    pdf.close()

args = read_flags()
main()

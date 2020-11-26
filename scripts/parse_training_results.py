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
from matplotlib import gridspec
from matplotlib import pyplot as plt

LOSS_COLUMNS = ["loss_type", "loss", "summary_type", "template", "num_samples",
        "samples_type"]
EXP_COLUMNS = ["num_hidden_layers", "hidden_layer_size",
        "sampling_priority_alpha", "max_discrete_featurizing_buckets",
        "heuristic_features", "alg", "nn_type"]

PLOT_SUMMARY_TYPES = ["mean"]

ALGS_ORDER = ["mscn", "mscn-priority", "microsoft", "microsoft-priority"]

COLORS = ["red", "orange", "green", "cyan"]

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--only_test", type=int, required=False,
            default=1)
    return parser.parse_args()

def get_alg_name(exp_args):
    if exp_args["algs"] == "nn":
        name = exp_args["nn_type"]
        if exp_args["sampling_priority_alpha"] > 0:
            name += "-priority"
        return name
    else:
        return exp_args["algs"]

def skip_exp(exp_args):
    if exp_args["sampling_priority_alpha"] > 2.00:
        return True
    if exp_args["max_discrete_featurizing_buckets"] > 10:
        return True

    return False

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

def get_fixed_summaries():
    fns = os.listdir(args.results_dir + "/fixed/")
    # df = pd.DataFrame(columns=["alg", "loss", "loss_type", "samples_type"])
    res = defaultdict(list)
    for fn in fns:
        if fn not in ["true", "postgres"]:
            continue
        # convert to same format as qerrs
        cur_dir = args.results_dir + "/fixed/" + fn
        qerrs = load_object(cur_dir + "/qerr.pkl")
        qerrs = qerrs[qerrs["summary_type"] == "mean"]
        qerrs = qerrs[qerrs["template"] == "all"]
        qerrs = qerrs[qerrs["num_tables"] == "all"]

        jerrs = load_jerrs(cur_dir)
        jerrs = jerrs[jerrs["summary_type"] == "mean"]
        jerrs = jerrs[jerrs["template"] == "all"]
        alg = fn

        st = "test"
        jerrs = jerrs[jerrs["samples_type"] == st]
        qerrs = qerrs[qerrs["samples_type"] == st]

        res["alg"].append(alg)
        res["loss_type"].append("jerr")
        res["samples_type"].append(st)
        res["loss"].append(jerrs["loss"].data[0])

        res["alg"].append(alg)
        res["loss_type"].append("qerr")
        res["samples_type"].append(st)
        res["loss"].append(qerrs["loss"].data[0])


    return pd.DataFrame(res)

def get_all_nn_summaries():
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
        nns = load_object(cur_dir + "/nn.pkl")
        stats = nns["stats"]

        exp_args["alg"] = get_alg_name(exp_args)
        for exp_column in EXP_COLUMNS:
            stats[exp_column] = exp_args[exp_column]

        all_dfs.append(stats)

    df = pd.concat(all_dfs, ignore_index=True)

    if args.only_test:
        df = df[df["samples_type"] == "test"]

    return df


def plot_jerr_fig(df, fixed_df):

    fixed_df = fixed_df[fixed_df["loss_type"] == "jerr"]
    true_jerr = fixed_df[fixed_df["alg"] == "true"]["loss"].data[0]
    pg_jerr = fixed_df[fixed_df["alg"] == "postgres"]["loss"].data[0]

    df = df[df["num_tables"] == "all"]
    df = df[df["template"] == "all"]

    # todo: use outliers
    df = df[df["summary_type"] == "mean"]
    df = df[df["loss_type"] == "jerr"]
    df = df[df["samples_type"] == "test"]
    df["loss"] += true_jerr
    df = df[["epoch","loss", "alg"]]

    tmp = defaultdict(list)
    for ep in set(df["epoch"]):
        tmp["epoch"].append(ep)
        tmp["loss"].append(true_jerr)
        tmp["alg"].append("true")

        tmp["epoch"].append(ep)
        tmp["loss"].append(pg_jerr)
        tmp["alg"].append("postgres")

    tmp = pd.DataFrame(tmp)
    df = pd.concat([df, tmp], ignore_index=True)

    max_loss = max(df["loss"])
    min_loss = min(df["loss"])

    hue_order = ["true", "postgres",
            "mscn", "mscn-priority", "microsoft", "microsoft-priority"]
    colors = ["black", "blue"] + COLORS

    ax = sns.lineplot(x="epoch", y="loss", hue="alg",
            data=df, hue_order=hue_order, palette=colors, legend=False)

    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")

    ax.set_yscale("log", basey=2)
    ax.set_xscale("linear")
    # ax.legend.remove()
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, loc='upper right', fontsize=6,
            # markerscale=3)

    plt.tight_layout()
    ax.set_ylabel("cost model units")

    plt.savefig("jerr.pdf")

def plot_qerr_fig(df, fixed_df):
    fixed_df = fixed_df[fixed_df["loss_type"] == "qerr"]
    pg_jerr = fixed_df[fixed_df["alg"] == "postgres"]["loss"].data[0]
    true_jerr = fixed_df[fixed_df["alg"] == "true"]["loss"].data[0]

    df = df[df["num_tables"] == "all"]
    df = df[df["template"] == "all"]

    # todo: use outliers
    df = df[df["summary_type"] == "mean"]
    df = df[df["loss_type"] == "qerr"]
    df = df[df["samples_type"] == "test"]

    # simplify..
    df = df[["epoch","loss", "alg"]]

    tmp = defaultdict(list)
    for ep in set(df["epoch"]):
        tmp["epoch"].append(ep)
        tmp["loss"].append(true_jerr)
        tmp["alg"].append("true")

        tmp["epoch"].append(ep)
        tmp["loss"].append(pg_jerr)
        tmp["alg"].append("postgres")

    tmp = pd.DataFrame(tmp)
    df = pd.concat([df, tmp], ignore_index=True)

    hue_order = ["true", "postgres",
            "mscn", "mscn-priority", "microsoft", "microsoft-priority"]
    colors = ["black", "blue"] + COLORS

    ax = sns.lineplot(x="epoch", y="loss", hue="alg",
            data=df, hue_order=hue_order, palette=colors)

    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")

    ax.set_ylim(bottom=min_loss, top=max_loss)
    plt.yscale("log", basey=2)
    ax.set_xscale("linear")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper left', fontsize=8,
            markerscale=3)

    plt.tight_layout()
    plt.savefig("qerr.pdf")
    plt.clf()

def main():

    # pdf = PdfPages("results.pdf")
    df = get_all_nn_summaries()
    df = df[df["hidden_layer_size"] == 100]
    fixed_df = get_fixed_summaries()

    plot_qerr_fig(df, fixed_df)
    plot_jerr_fig(df, fixed_df)

    # pdf.close()

args = read_flags()
main()

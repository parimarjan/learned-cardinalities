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

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--only_test", type=int, required=False,
            default=0)
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
        if name == "microsoft":
            return "fcnn"
        return name
    else:
        return exp_args["algs"]

def skip_exp(exp_args):
    if exp_args["sampling_priority_alpha"] > 2.00:
        return True
    if exp_args["max_discrete_featurizing_buckets"] > 10:
        return True

    return False

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

        if exp_args["sampling_priority_alpha"] == 2.0:
            qerrs["priority"] = "yes"
        else:
            qerrs["priority"] = "no"

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

        if exp_args["sampling_priority_alpha"] == 2.0:
            jerrs["priority"] = "yes"
        else:
            jerrs["priority"] = "no"

        all_dfs.append(jerrs)

    df = pd.concat(all_dfs, ignore_index=True)
    if args.only_test:
        df = df[df["samples_type"] == "test"]
    return df

def plot_fig1(jerrs):
    sns.barplot(data=jerrs, x = "hidden_layer_size", y = "cost",
            hue = "priority", estimator=np.mean)

    # plt.yscale("log", basey=10)
    plt.savefig("model_complexity_jerr.pdf")
    plt.clf()

def plot_fig2(qerrs):
    sns.barplot(data=qerrs, x = "hidden_layer_size", y = "loss",
            hue = "priority", estimator=np.mean)

    plt.savefig("model_complexity_qerr.pdf")
    plt.clf()

def main():

    jerrs = get_all_jerrs()
    # jerrs = jerrs[jerrs["hidden_layer_size"] != 256]
    # jerrs = jerrs[jerrs["alg"] == "fcnn"]
    jerrs = jerrs[jerrs["alg"] == "mscn"]
    jerrs = jerrs[jerrs["max_discrete_featurizing_buckets"] == 1]
    jerrs = jerrs[jerrs["samples_type"] == "train"]

    jerrs = jerrs[["alg", "priority", "cost","hidden_layer_size"]]
    plot_fig1(jerrs)

    qerrs = get_all_qerrs()
    qerrs = qerrs[qerrs["template"] == "all"]
    # qerrs = qerrs[qerrs["hidden_layer_size"] != 256]
    qerrs = qerrs[qerrs["summary_type"] == "mean"]
    qerrs = qerrs[qerrs["num_tables"] == "all"]
    qerrs = qerrs[qerrs["samples_type"] == "train"]
    qerrs = qerrs[qerrs["alg"] == "mscn"]

    print(qerrs.keys())
    qerrs = qerrs[["alg", "priority", "loss", "hidden_layer_size"]]
    plot_fig2(qerrs)

    # save_object("qerrs100.pkl", qerrs)

    # pdb.set_trace()

    # save_object("jerrs100.pkl", jerrs)
    # qerrs = get_all_qerrs()
    # qerrs = qerrs[qerrs["template"] == "all"]
    # qerrs = qerrs[qerrs["summary_type"] == "mean"]
    # qerrs = qerrs[qerrs["num_tables"] == "all"]
    # qerrs = qerrs[qerrs["samples_type"] == "test"]
    # qerrs = qerrs[["alg", "priority", "loss"]]
    # save_object("qerrs100.pkl", qerrs)

args = read_flags()
main()

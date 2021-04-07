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
# EXP_COLUMNS = ["num_hidden_layers", "hidden_layer_size",
        # "sampling_priority_alpha", "max_discrete_featurizing_buckets",
        # "heuristic_features", "alg", "nn_type", "normalization_type"]
EXP_COLUMNS = ["num_hidden_layers", "hidden_layer_size",
        "sampling_priority_alpha", "max_discrete_featurizing_buckets",
        "heuristic_features", "alg", "nn_type", "loss_func",
        "flow_features", "weight_decay", "sample_bitmap", "test_size",
        "max_epochs", "debug_set", "lr", "diff_templates_seed",
        "test_diff_templates", "cost_model", "normalize_flow_loss"]

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

def load_jerrs(exp_dir, file_name, loss_key):
    jerrs = load_object(exp_dir + "/" + file_name)
    if jerrs is None:
        # print("jerr not found for: ", exp_dir)
        return None

    stats = defaultdict(list)

    for samples_type in set(jerrs["samples_type"]):
        cur_jerrs = jerrs[jerrs["samples_type"] == samples_type]
        add_row(cur_jerrs["cost"].values, loss_key, -1, "all", "all", samples_type,
                stats, "cardinality")
        # add_row(cur_jerrs["loss"].values, loss_key, -1, "all", "all", samples_type,
                # stats, "cardinality")
        for template in set(cur_jerrs["template"]):
            tmp_jerrs = cur_jerrs[cur_jerrs["template"] == template]
            add_row(tmp_jerrs["cost"].values, loss_key, -1, template, "all",
                    samples_type, stats, "cardinality")
            # add_row(tmp_jerrs["loss"].values, loss_key, -1, template, "all",
                    # samples_type, stats, "cardinality")

    return pd.DataFrame(stats)

def get_alg_name_old(exp_args):
    if exp_args["algs"] == "nn":
        name = exp_args["nn_type"]
        # if not exp_args["heuristic_features"]:
            # name += "-no_heuristic"
        return name
    else:
        return exp_args["algs"]

def get_alg_name(exp_args):
    if exp_args["algs"] == "nn":
        name = exp_args["nn_type"]
        return name
    elif exp_args["algs"] == "sampling":
        return exp_args["sampling_key"]
    else:
        return exp_args["algs"]

def skip_exp(exp_args):
    if exp_args["sampling_priority_alpha"] > 2.00:
        return True
    # if exp_args["max_discrete_featurizing_buckets"] > 10:
        # return True

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
        # exp_args["alg"] += exp_args["sampling_priority_alpha"]
        for exp_column in EXP_COLUMNS:
            qerrs[exp_column] = exp_args[exp_column]
            # if exp_column in exp_args:
                # qerrs[exp_column] = exp_args[exp_column]
            # else:
                # qerrs[exp_column] = None

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

def get_summary_df(results_dir):
    all_dfs = []
    fns = os.listdir(results_dir)
    for fn in fns:
        # convert to same format as qerrs
        cur_dir = results_dir + "/" + fn
        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            continue
            # mse_fn = fn.replace("flow_loss2", "mse")
            # tmp_dir = results_dir + "/" + mse_fn
            # if os.path.exists(tmp_dir):
                # exp_args = load_object(tmp_dir + "/args.pkl")
            # else:
                # print("exp args is None!")
                # continue

        # print(fn)
        # print(exp_args)
        exp_args = vars(exp_args)
        # if exp_args["max_discrete_featurizing_buckets"] != 10:
            # print("skipping non-10 buckets")
            # continue
        exp_args["alg"] = get_alg_name(exp_args)

        if exp_args["diff_templates_seed"] <= 10:
            pass
            # print(exp_args["diff_templates_seed"])
            # print(fn)
            # print("******")
        else:
            pass
            # print("SKIPPING PARTITION OVER 10")
            # continue

        # if exp_args["diff_templates_seed"] <= 5:
            # print(cur_dir)
            # print(exp_args["diff_templates_seed"])
            # print("*********")

        if skip_exp(exp_args):
            print("skip exp!")
            continue

        try:
            qerrs = load_qerrs(cur_dir)
            # cm1_jerrs = load_jerrs(cur_dir, "cm1_jerr.pkl", "cm1_jerr")
            cm1_jerrs = load_jerrs(cur_dir, "cm1_mysql_jerr.pkl", "cm1_jerr")
            # perrs = load_jerrs(cur_dir, "plan_err.pkl", "plan_err")
            # perrs_pg = load_jerrs(cur_dir, "plan_pg_err.pkl", "plan_pg_err")
            # ferrs = load_jerrs(cur_dir, "flow_err.pkl", "flow_err")
            # jerrs = load_jerrs(cur_dir, "nested_loop_index7_jerr.pkl", "jerr")
            jerrs = None
            perrs = None
            perrs_pg = None
            ferrs = None
        except Exception as e:
            print(e)
            print("skipping1 ", cur_dir)
            continue

        qerrs = qerrs[qerrs["num_tables"] == "all"]
        qerrs = qerrs[LOSS_COLUMNS]

        to_concat = []
        to_concat.append(qerrs)

        if jerrs is not None:
            jerrs = jerrs[LOSS_COLUMNS]
            to_concat.append(jerrs)

        if cm1_jerrs is not None:
            cm1_jerrs = cm1_jerrs[LOSS_COLUMNS]
            to_concat.append(cm1_jerrs)

        if perrs is not None:
            perrs = perrs[LOSS_COLUMNS]
            to_concat.append(perrs)

        if perrs_pg is not None:
            perrs_pg = perrs_pg[LOSS_COLUMNS]
            to_concat.append(perrs_pg)

        if ferrs is not None:
            ferrs = ferrs[LOSS_COLUMNS]
            to_concat.append(ferrs)

        # TODO: add rts too, if it exists

        if len(to_concat) == 0:
            print("skipping: to concat, len 0")
            continue
        cur_df = pd.concat(to_concat, ignore_index=True)

        # cur_df = cur_df.assign(**exp_args)

        for exp_column in EXP_COLUMNS:
            cur_df[exp_column] = exp_args[exp_column]

        args_hash = str(deterministic_hash(str(exp_args)))[0:5]
        exp_hash = str(deterministic_hash(str(exp_args)))[0:5]

        cur_df["exp_hash"] = exp_hash
        cur_df["fn"] = fn
        cur_df["fn_hash"] = str(deterministic_hash(fn))[0:5]

        if "nn" in exp_args["algs"]:
            if exp_args["sampling_priority_alpha"] > 0:
                cur_df["alg_name"] = "priority"
            else:
                cur_df["alg_name"] = exp_args["loss_func"]
        else:
            cur_df["alg_name"] = exp_args["algs"]

        # decide partition
        # if exp_args["test_diff"] != 1:
            # print("skipping non-test diff cases")
            # continue

        if exp_args["test_diff_templates"]:
            if exp_args["diff_templates_type"] == 1:
                partition = "X"
            elif exp_args["diff_templates_type"] == 2:
                partition = "Y"
            elif exp_args["diff_templates_type"] == 3:
                partition = exp_args["diff_templates_seed"]
        else:
            partition = 0

        cur_df["partition"] = partition
        cur_df["test_diff_templates"] = exp_args["test_diff_templates"]
        all_dfs.append(cur_df)

    if len(all_dfs) == 0:
        print("empty all dfs!")
        return pd.DataFrame()
    summary_df = pd.concat(all_dfs, ignore_index=True)
    return summary_df

def plot_summary(pdf, df, title):
    fg = sns.catplot(x="alg", y="loss",
            data=df, row="max_discrete_featurizing_buckets",
            col = "hidden_layer_size", hue="priority", kind="bar",
            hue_order=["no", "yes"])
            # order = ALGS_ORDER, hue_order = ALGS_ORDER)
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

    # get_all_jerrs()
    # plot_fig2()
    # plot_fig2_cp()

    pdf = PdfPages("results.pdf")
    # plot_join_summaries2(pdf)

    summary_df = get_summary_df()
    summary_df = summary_df[summary_df["heuristic_features"] == 1]
    summary_df = summary_df[summary_df["hidden_layer_size"] <= 100]
    SUMMARY_TITLE_FMT = "{ST}-{LT}-{SUMMARY}"

    if args.only_test:
        summary_df = summary_df[summary_df["samples_type"] == "test"]
    for samples_type in set(summary_df["samples_type"]):
        st_df = summary_df[summary_df["samples_type"] == samples_type]
        for lt in set(st_df["loss_type"]):
            lt_df = st_df[st_df["loss_type"] == lt]
            for summary_type in PLOT_SUMMARY_TYPES:
                plot_df = lt_df[lt_df["summary_type"] == summary_type]
                plot_df = plot_df[plot_df["template"] == "all"]
                # print(set(plot_df["alg"]))
                title = SUMMARY_TITLE_FMT.format(ST = samples_type,
                                                 LT = lt,
                                                 SUMMARY = summary_type)
                plot_summary(pdf, plot_df, title)

    # plot_qerr_summaries2(pdf)
    # plot_join_summaries2(pdf)
    pdf.close()

if __name__ == "__main__":
    args = read_flags()
    main()

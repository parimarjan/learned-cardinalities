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
        # if not exp_args["heuristic_features"]:
            # name += "-no_heuristic"
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

def plot_fig2_cp():
    df = get_summary_df()
    df = df[df["samples_type"] == "test"]
    df = df[df["summary_type"] == "mean"]
    df = df[df["template"] == "all"]
    df = df[df["nn_type"] == "mscn"]
    df = df[df["heuristic_features"] == 1]
    df = df[df["max_discrete_featurizing_buckets"] == 1]
    print(df)
    pdb.set_trace()

    priority_order = ["no", "yes"]
    fg = sns.catplot(x = "alg",
            y = "loss",
            row = "loss_type",
            col = "hidden_layer_size",
            data = df,
            hue = "priority",
            hue_order = priority_order,
            sharey = "row",
            kind="bar",
            sharex = False,
            # row_order=plot_order,
            ci = None,
            legend = False,
            # legend_out = True
            )

    # fg.set(yscale="log")
    # fg.set(ylabel="")

    for i, ax in enumerate(fg.axes.flatten()):
        if i == 0:
            ax.legend(loc="upper left")
            ax.set_ylabel('QError', fontsize=8)
        # if i < 3:
            # ax.set_ylabel('cost', fontsize=8)
            # ax.set_title("Query Costs")
        elif i == 4:
            ax.set_ylabel('Cost', fontsize=8)
            # ax.set_title("Query Runtimes")

    # plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("qerr_jerr_catplot.png")

def plot_fig2():
    df = get_summary_df()
    df = df[df["samples_type"] == "test"]
    df = df[df["summary_type"] == "mean"]
    df = df[df["template"] == "all"]
    df = df[df["nn_type"] == "mscn"]
    df = df[df["heuristic_features"] == 1]
    df = df[df["max_discrete_featurizing_buckets"] == 1]

    # hls_entries = [10, 100, 10, 100, 10, 100]
    # bin_entries = [1, 1, 5, 5, 10, 10]
    hls_entries = [10, 50, 100]
    f = plt.figure(figsize=(3, 9))

    gs = gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.1)
    priority_order = ["no", "yes"]

    for i in range(3):
        nested_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i],
                wspace=0.5)
        hls = hls_entries[i]
        block_df = df[df["hidden_layer_size"] == hls]
        print(set(block_df["alg"]))

        for j in range(2):
            ax = plt.Subplot(f, nested_gs[j])
            # add the plots on ax
            if j == 0:
                plot_df = block_df[block_df["loss_type"] == "qerr"]
                title = "QError"
            else:
                plot_df = block_df[block_df["loss_type"] == "jcost"]
                title = "Join-Error"

            bar = sns.barplot(x="alg", y="loss", hue="priority",
                    data=plot_df, ax=ax, ci=None,
                    hue_order = priority_order)

            # Define some hatches
            # hatches = ['-', '/', '*', '--']
            # # # Loop over the bars
            # for bari,thisbar in enumerate(bar.patches):
                # # Set a different hatch for each bar
                # thisbar.set_hatch(hatches[bari])

            ax.get_legend().remove()

            if j == 1:
                # just for jerr
                ax.ticklabel_format(axis="y", style="sci", scilimits=(6,6))
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(1,1))

            ax.set_ylabel("")
            ax.set_xlabel("")

            for tl in ax.get_xticklabels():
                if i < 4:
                    tl.set_visible(False)
                else:
                    tl.set_fontsize(8)

            ax.set_title(title)

            pad = 5
            # if j == 0:
                # text = "hidden layer size: " + str(hls)
                # ax.annotate(text, xy=(250, 400),
                            # xycoords='axes points',
                            # ha='center',
                            # fontsize = 8)

            ax.set_yscale("log")
            f.add_subplot(ax)

    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='upper left', fontsize=8,
            markerscale=3, title="Priority", title_fontsize=8)
    plt.savefig("qerr_jerr_fig2.png")
    plt.clf()

def plot_fig1(pdf):
    df = get_summary_df()
    if args.only_test:
        df = df[df["samples_type"] == "test"]
    df = df[df["summary_type"] == "mean"]
    df = df[df["template"] == "all"]

    # df = df[df["nn_type"] == "mscn"]
    df = df[df["heuristic_features"] == 1]
    hls_entries = [10, 100, 10, 100, 10, 100]
    bin_entries = [1, 1, 5, 5, 10, 10]
    # hls_entries = [10, 50, 100, 10, 50, 100, 10, 50, 100]
    # bin_entries = [1, 1, 1, 5, 5, 5, 10, 10, 10]

    f = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.1)
    # gs = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.1)
    # combined_title = "Comparison of QError and Join-Error for MSCN without heuristic features"
    # f.suptitle(combined_title, fontsize=32)
    priority_order = ["no", "yes"]

    for i in range(6):
    # for i in range(9):
        # find the hls, bucket size
        nested_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i],
                wspace=0.2)
        # nested_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i],
                # wspace=0.2)
        hls = hls_entries[i]
        bins = bin_entries[i]
        block_df = df[df["hidden_layer_size"] == hls]
        block_df = block_df[block_df["max_discrete_featurizing_buckets"] == bins]
        print(set(block_df["alg"]))

        for j in range(2):
            ax = plt.Subplot(f, nested_gs[j])
            # add the plots on ax
            if j == 0:
                plot_df = block_df[block_df["loss_type"] == "qerr"]
                title = "QError"
            else:
                plot_df = block_df[block_df["loss_type"] == "jcost"]
                title = "Join-Error"

            bar = sns.barplot(x="alg", y="loss", hue="priority",
                    data=plot_df, ax=ax, ci=None,
                    hue_order = priority_order,
                    order = ["mscn", "microsoft"])

            # Define some hatches
            hatches = ['-', '/', '*', '--']
            # # Loop over the bars
            for bari,thisbar in enumerate(bar.patches):
                # Set a different hatch for each bar
                thisbar.set_hatch(hatches[bari])

            ax.get_legend().remove()

            if j == 1:
                # just for jerr
                ax.ticklabel_format(axis="y", style="sci", scilimits=(6,6))
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(1,1))

            ax.set_ylabel("")
            ax.set_xlabel("")

            for tl in ax.get_xticklabels():
                if i < 4:
                    tl.set_visible(False)
                else:
                    tl.set_fontsize(8)
                    # ax.set_xlabel("classifier")

            if i < 2:
                ax.set_title(title)

            pad = 5
            if i in [0, 1] and j == 0:
                text = "hidden layer size: " + str(hls)
                ax.annotate(text, xy=(250, 400),
                            xycoords='axes points',
                            ha='center',
                            fontsize = 8)

            if i in [0, 2, 4] and j == 0:
                text = "bins: " + str(bins)
                ax.annotate(text,
                            xy=(-30, 150),
                            xycoords = 'axes points',
                            fontsize = 8,
                            ha='right',
                            va='center')

            f.add_subplot(ax)

    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='upper left', fontsize=8,
            markerscale=3, title="Priority", title_fontsize=8)
    plt.savefig("fig1.png")
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

        if exp_args["sampling_priority_alpha"] == 2.0:
            cur_df["priority"] = "yes"
        else:
            cur_df["priority"] = "no"

        all_dfs.append(cur_df)

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

    get_all_jerrs()
    # plot_fig2()
    # plot_fig2_cp()

    # pdf = PdfPages("results.pdf")

    # plot_join_summaries2(pdf)

    summary_df = get_summary_df()
    summary_df = summary_df[summary_df["heuristic_features"] == 1]
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

args = read_flags()
main()

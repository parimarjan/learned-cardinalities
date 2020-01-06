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
import numpy as np

MAX_ERRS = {"qerr": 10.00, "jerr":1500000}
SUMMARY_FUNCS = ["mean", "percentile:50"]

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./nn_results")
    return parser.parse_args()

def get_classifier_name(config, name):
    # TODO: add other stuff?
    return name

def plot_global_figs(df):
    # for qerr, and jerr
    pass

def exclude_exp(config, stats):
    return False

def plot_table_errors(df, lt, template, samples_type, summary_type, pdf):
    df = df[df["loss_type"] == lt]
    df = df[df["samples_type"] == samples_type]
    df = df[df["summary_type"] == summary_type]
    df = df[df["template"] == template]
    # print(df)
    # pdb.set_trace()
    # fg = sns.FacetGrid(df, col = "num_tables", hue="alg", col_wrap=3,
            # hue_order = ALG_ORDER)
    fg = sns.FacetGrid(df, col = "num_tables", hue="cl_name",
            col_wrap=3)
    fg = fg.map(plt.scatter, "num_iter", "loss")
    fg.axes[0].legend(loc='upper left')

    # TODO: set how many queries are there on each table
    # for i, ax in enumerate(fg.axes.flat):
        # tmp = templates[i]
        # sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        # title = tmp + " ,#subqueries: " + str(sqs)
        # ax.set_title(title)

    # title
    fg.fig.suptitle("{} QError for {}".format(samples_type, template),
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 40)

    fg.set(ylim=(0,10.0))
    fg.despine(left=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig()
    plt.clf()

def est(arr):
    return arr[-1]

def plot_feature_bins(df, num_tables, template,
        summary_type, loss_type, samples_type, pdf):

    df = df[df["num_tables"] == num_tables]
    df = df[df["template"] == template]
    df = df[df["summary_type"] == summary_type]
    df = df[df["loss_type"] == loss_type]
    df = df[df["samples_type"] == samples_type]
    print(df)
    cls = df["cl_name"]
    simple_cl = []
    for cl in cls:
        if "pr" in cl or "num_table" in cl:
            simple_cl.append("priority")
        else:
            simple_cl.append("microsoft")
    df["simple_cl"] = simple_cl
    if "train" in samples_type:
        est_func = est
    else:
        est_func = np.min

    sns.barplot(x="bins", y = "loss", data=df, hue = "simple_cl",
            estimator=est_func, ci=None)
    plt.title("{}: {}".\
            format(samples_type, loss_type))
    pdf.savefig()
    plt.clf()

def plot_generalization_fig(df, num_tables, template,
        summary_type, loss_type, pdf):
    '''
    plots training - test together.
    '''
    # apply filters
    df = df[df["num_tables"] == num_tables]
    df = df[df["template"] == template]

    # TODO: use outliers
    df = df[df["summary_type"] == summary_type]
    df = df[df["loss_type"] == loss_type]

    max_loss = min(MAX_ERRS[loss_type], max(df["loss"]))
    min_loss = min(df["loss"])

    df_train = df[df["samples_type"] == "train"]

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(2, 1, 1)
    ax = sns.lineplot(x="num_iter", y="loss", hue="cl_name",
            data=df_train)
    ax.set_ylim(bottom=min_loss, top=max_loss)
    plt.title("{}: {}".\
            format("train", loss_type))
    ax.get_legend().remove()

    df_test = df[df["samples_type"] == "test"]
    ax = fig.add_subplot(2, 1, 2)
    ax = sns.lineplot(x="num_iter", y="loss", hue="cl_name",
            data=df_test)
    ax.set_ylim(bottom=min_loss, top=max_loss)
    # TODO: add global stats here
    # num_samples, lr etc.
    plt.title("{}: {}".\
            format("test", loss_type))
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, loc='upper left')

    plt.tight_layout()
    pdf.savefig()

def main():
    fns = glob.glob(args.results_dir + "/*.pkl")
    all_stats = []
    for fn in fns:
        with open(fn, "rb") as f:
            try:
                exp_data = pickle.load(f)
            except:
                continue
        stats = exp_data["stats"]
        config = exp_data["config"]
        if exclude_exp(config, stats):
            continue
        cl_name = get_classifier_name(config, exp_data["name"])
        stats["cl_name"] = [cl_name]*len(stats["num_iter"])
        stats["bins"] = [config["max_discrete_featurizing_buckets"]]*len(stats["num_iter"])
        print(set(stats["bins"]))
        stats = pd.DataFrame(stats)
        all_stats.append(stats)

    df = pd.concat(all_stats, ignore_index=True)

    '''
    Plots we want:
        - TODO: common error bar code for all cases
        - qerr: train v test
        - jerr: train v test
            - mean {min+max}, median {95th+25th}
        - num_tables, qerr: train
        - num_tables, qerr: test
        - template_name, qerr: train
        - template_name, qerr: test
        - template_name, jerr: train
        - template_name, jerr: test
    '''

    # TODO: better name
    pdf = PdfPages("training_curves.pdf")
    # plot_feature_bins(df, "all", "all", "mean", "jerr", "train", pdf)
    # plot_feature_bins(df, "all", "all", "mean", "jerr", "test", pdf)

    plot_generalization_fig(df, "all", "all", "mean", "qerr", pdf)
    plot_generalization_fig(df, "all", "all", "mean", "jerr", pdf)

    # pdb.set_trace()
    # templates = df["template"]

    # plot_table_errors(df, "qerr", "all", "train", "mean", pdf)
    # plot_table_errors(df, "qerr", "all", "test", "mean", pdf)

    # plot_table_errors(df, "jerr", "all", "train", "mean", pdf)
    # plot_table_errors(df, "jerr", "all", "test", "mean", pdf)

    # for template in templates:
        # plot_table_errors(df, "qerr", template, "train", "mean", pdf)
        # plot_table_errors(df, "qerr", template, "test", "mean", pdf)

    pdf.close()

args = read_flags()
main()

from utils.utils import *
import pdb
import klepto
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import argparse
from collections import defaultdict

OPT_OBJS = ["cm2", "qerr-qloss-subquery", "qerr-qloss-weighted_query"]
LOSS_FUNCS = ["qloss"]
JL_VARIANTS = [0]
SAMPLES_TYPE = ["train", "test"]
SAMPLING_METHODS = ["jl_ratio", "jl_diff", "jl_rank"]
LOSS_TYPES = ["qerr", "join-loss"]
MAX_ERRS = {"qerr": 20.00, "join-loss":1000000}
HIDDEN_LAYERS = [1, 2, 3, 4]
# LRS = [0.0001, 0.001, 0.01]
LRS = [0.01, 0.0001]
OPTIMIZERS = ["sgd", "adam", "ams"]
# ALG_ORDER = ["FCNN", "Tables-FCNN", "Tables-LinearRegression"]
# ALG_ORDER = ["FCNN0.01", "FCNN0.0001", "Tables-FCNN", "Tables-LinearRegression"]
ALG_ORDER = ["Postgres", "FCNN0.01", "FCNN0.0001", "Tables-FCNN",
        "Tables-LinearRegression"]

PG_TRAIN_JL = 376236.00
PG_TEST_JL = 352099.00

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./nn_training_cache")
    return parser.parse_args()

def parse_results():
    '''
    type of the neural network is specified by:
        - optimizer_name
        - model_name (fully connnected v/s others)
        - lr
        - sampling_alpha
        - priority
    '''

    cache_dir = args.results_dir
    cache = klepto.archives.dir_archive(cache_dir,
            cached=True, serialized=True)
    cache.load()
    print("results cache loaded")

    all_data = defaultdict(list)
    tables_data = defaultdict(list)

    for exp_name in cache:
        print(exp_name)
        data = cache[exp_name]
        kwargs = data["kwargs"]
        eval_iter = kwargs["eval_iter"]
        optimizer_name = data["kwargs"]["optimizer_name"]
        jl_variant = data["kwargs"]["jl_variant"]
        loss_func = kwargs["loss_func"]
        lr = kwargs["lr"]
        alpha = kwargs["sampling_priority_alpha"]
        sampling_method = kwargs["sampling_priority_method"]
        # hidden_layers = 0
        if lr not in LRS:
            print("skipping {} because lr= {}".format(exp_name, lr))
            continue

        if data["name"] == "nn":
            name = ""
            opt_obj = name + data["kwargs"]["net_name"] + str(lr)
        elif data["name"] == "NumTablesNN":
            name = "Tables-"
            opt_obj = name + data["kwargs"]["net_name"]
        else:
            opt_obj = data["kwargs"]["net_name"]

        # num_tables version
        for samples_type in SAMPLES_TYPE:
            if "tables_eval" not in data[samples_type]:
                continue
            exp_eval = data[samples_type]["tables_eval"]
            print(exp_eval.keys())
            for loss_type, tables in exp_eval.items():
                for num_table, losses in tables.items():
                    for num_iter, loss in losses.items():
                        tables_data["iter"].append(num_iter)
                        tables_data["loss"].append(loss)
                        tables_data["loss_type"].append(loss_type)
                        tables_data["num_tables"].append(num_table)
                        tables_data["lr"].append(lr)
                        tables_data["samples_type"].append(samples_type)
                        tables_data["alg"].append(opt_obj)

        for samples_type in SAMPLES_TYPE:
            exp_eval = data[samples_type]["eval"]
            for loss_type, losses in exp_eval.items():
                for num_iter, loss in losses.items():
                    all_data["iter"].append(num_iter)
                    all_data["loss"].append(loss)
                    all_data["loss_type"].append(loss_type)
                    all_data["optimizer_name"].append(optimizer_name)
                    all_data["optimizer_obj"].append(opt_obj)
                    all_data["jl_variant"].append(jl_variant)
                    all_data["lr"].append(lr)
                    all_data["samples_type"].append(samples_type)
                    all_data["alpha"].append(alpha)

    df = pd.DataFrame(all_data)
    tables_df = pd.DataFrame(tables_data)
    # add postgres stuff too
    iters = set(df["iter"])
    for it in iters:
        all_data["iter"].append(it)
        all_data["loss"].append(PG_TRAIN_JL)
        all_data["loss_type"].append("join-loss")
        all_data["optimizer_name"].append("Postgres")
        all_data["optimizer_obj"].append("Postgres")
        all_data["jl_variant"].append(0)
        all_data["lr"].append(0)
        all_data["samples_type"].append("train")
        all_data["alpha"].append(0)

        all_data["iter"].append(it)
        all_data["loss"].append(PG_TEST_JL)
        all_data["loss_type"].append("join-loss")
        all_data["optimizer_name"].append("Postgres")
        all_data["optimizer_obj"].append("Postgres")
        all_data["jl_variant"].append(0)
        all_data["lr"].append(0)
        all_data["samples_type"].append("test")
        all_data["alpha"].append(0)

    df = pd.DataFrame(all_data)
    return df, tables_df

def plot_overfit_figures(df, hidden_layers):
    lrs = set(df["lr"])
    loss_types = LOSS_TYPES
    for lr in lrs:
        for loss_type in loss_types:
            df2 = df[df["lr"] == lr]
            df2 = df2[df2["loss_type"] == loss_type]
            if len(df2) == 0:
                print("No data for: {} {} combination".format(lr, loss_type))
                continue
            ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj",
                    style="optimizer_obj",
                    data=df2)
            max_loss = min(MAX_ERRS[loss_type], max(df2["loss"]))
            ax.set_ylim(bottom=0, top=max_loss)
            plt.title("Exp Type: Overfit, {}, lr: {}, layers: {}".\
                    format(loss_type, lr, hidden_layers))
            plt.tight_layout()
            pdf.savefig()
            plt.clf()

def plot_subplot(ax, df, loss_type, samples_type, lr):
    # filter out stuff
    df_lt = df[df["loss_type"] == loss_type]
    df_lt = df_lt[df_lt["samples_type"] == samples_type]
    print(loss_type, samples_type)
    print(max(df_lt["loss"]))
    ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj",
            style="optimizer_obj",
            data=df_lt, hue_order=ALG_ORDER)
    max_loss = min(MAX_ERRS[loss_type], max(df_lt["loss"]))
    ax.set_ylim(bottom=0, top=max_loss)
    plt.title("{}: {}, lr: {}".\
            format(samples_type, loss_type, lr))
    plt.tight_layout()

def plot_generalization_figs(df):
    # lrs = set(df["lr"])
    # for lr in lrs:
        # print(lr)

    # df_lr = df[df["lr"] == lr]

    lr = ""
    df_lr = df
    for lt in LOSS_TYPES:
        if lt not in set(df_lr["loss_type"]):
            continue

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.04, wspace=0.04)
        ax = fig.add_subplot(2, 1, 1)
        plot_subplot(ax, df_lr, lt, "train", lr)
        ax.get_legend().remove()

        ax = fig.add_subplot(2, 1, 2)
        plot_subplot(ax, df_lr, lt, "test", lr)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        ax.get_legend().remove()
        pdf.savefig()

def plot_table_errors(tables_df, lt, samples_type):
    print("plot table errors!")
    tables_df = tables_df[tables_df["loss_type"] == lt]
    tables_df = tables_df[tables_df["samples_type"] == samples_type]
    tables_df = tables_df[tables_df["num_tables"] > 1]
    fg = sns.FacetGrid(tables_df, col = "num_tables", hue="alg", col_wrap=3,
            hue_order = ALG_ORDER)
    fg = fg.map(plt.plot, "iter", "loss")
    # plt.legend(loc='upper left')
    fg.axes[0].legend(loc='upper left')

    # TODO: set how many queries are there on each table
    # for i, ax in enumerate(fg.axes.flat):
        # tmp = templates[i]
        # sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        # title = tmp + " ,#subqueries: " + str(sqs)
        # ax.set_title(title)

    fg.fig.suptitle("{} QError by #Tables".format(samples_type),
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 40)

    fg.set(ylim=(0,20.0))
    # fg.set(yscale="log")
    fg.despine(left=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig()
    plt.clf()


args = read_flags()
df, tables_df = parse_results()

# pdb.set_trace()
# skip the first entry, since it is too large
# print(df[df["iter"] == 0])
df = df[df["iter"] != 0]
df = df[df["iter"] <= 100000]

pdf = PdfPages("training_curves.pdf")
plot_generalization_figs(df)

plot_table_errors(tables_df, "qerr", "train")
plot_table_errors(tables_df, "qerr", "test")
pdf.close()


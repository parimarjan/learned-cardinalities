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
MAX_ERRS = {"qerr": 100.00, "join-loss":1000000}
HIDDEN_LAYERS = [1, 2, 3, 4]
# LRS = [0.0001, 0.001, 0.01]
LRS = [0.001]
OPTIMIZERS = ["sgd", "adam", "ams"]

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

        if lr not in LRS:
            print("Skipping {} because lr: {}".format(exp_name, lr))
            continue

        if eval_iter < 1000:
            print("Skipping {} because eval_iter: {}".format(exp_name, eval_iter))
            continue

        if loss_func not in LOSS_FUNCS:
            print("Skipping {} because loss_func: {}".format(exp_name,
                loss_func))
            continue

        if kwargs["sampling"] != "weighted_query":
            print("Skipping {} because sampling != {}".format(exp_name,
                "weighted_query"))
            continue

        if sampling_method not in SAMPLING_METHODS:
            print("Skipping {} because sampling_method != {}".format(exp_name,
                SAMPLING_METHODS))
            continue

        # FIXME: temporary hack
        if sampling_method == "jl_rank":
            if alpha != 0.00:
                print("Skipping jl_rank because alpha not 0.00")
                continue
            else:
                sampling_method = "jl_ratio"

        if jl_variant not in JL_VARIANTS:
            print("Skipping {} because jl_variant: {}".format(exp_name,
                jl_variant))
            continue

        opt_obj = sampling_method + str(alpha)

        if kwargs["hidden_layer_multiple"] != 0.5:
            print("Skipping {} because hidden layer multiple not 0.5".format(exp_name))
            continue
        hidden_layers = kwargs["num_hidden_layers"]

        if hidden_layers not in HIDDEN_LAYERS:
            print("Skipping {} because hidden_layers: {}".format(exp_name,
                hidden_layers))
            continue

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
                    all_data["hidden_layers"].append(hidden_layers)

    df = pd.DataFrame(all_data)
    return df

args = read_flags()
df = parse_results()

# skip the first entry, since it is too large
# print(df[df["iter"] == 0])
# df = df[df["iter"] != 0]

pdf = PdfPages("training_curves.pdf")

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

hidden_layers = [hl for hl in set(df["hidden_layers"])]
hidden_layers.sort()

# for hl in hidden_layers:
    # df2 = df[df["hidden_layers"] == hl]
    # plot_overfit_figures(df2, hl)

lrs = set(df["lr"])

def plot_subplot(ax, df, loss_type, samples_type):
    # filter out stuff
    df_lt = df[df["loss_type"] == loss_type]
    df_lt = df_lt[df_lt["samples_type"] == samples_type]
    print(loss_type, samples_type)
    print(max(df_lt["loss"]))
    # pdb.set_trace()
    ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj",
            style="optimizer_obj",
            data=df_lt)
    max_loss = min(MAX_ERRS[loss_type], max(df_lt["loss"]))
    ax.set_ylim(bottom=0, top=max_loss)
    plt.title("{}: {}, lr: {}".\
            format(samples_type, loss_type, lr))
    plt.tight_layout()

print(set(df["optimizer_name"]))
# pdb.set_trace()
for lr in lrs:
    df_lr = df[df["lr"] == lr]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(2, 2, 1)
    plot_subplot(ax, df_lr, "qerr", "train")
    ax.get_legend().remove()

    ax = fig.add_subplot(2, 2, 2)
    plot_subplot(ax, df_lr, "qerr", "test")
    ax.get_legend().remove()

    ax = fig.add_subplot(2, 2, 3)
    plot_subplot(ax, df_lr, "join-loss", "train")
    ax.get_legend().remove()

    ax = fig.add_subplot(2, 2, 4)
    plot_subplot(ax, df_lr, "join-loss", "test")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    ax.get_legend().remove()

    pdf.savefig()

# Plot figures for model capacity

pdf.close()


import sys
sys.path.append(".")
from utils.utils import *
import pickle
import glob
import argparse
import pandas as pd
from collections import defaultdict
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# from utils import *
import pdb
# from db_utils.utils import *
# from db_utils.query_storage import *

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--only_test", type=int, required=False,
            default=1)
    return parser.parse_args()

def qkey_map(query_dir):
    query_dir += "/"
    qtmps = os.listdir(query_dir)
    mapping = {}
    for qtmp in qtmps:
        qfns = os.listdir(query_dir + qtmp)
        for fn in qfns:
            if ".pkl" not in fn:
                continue
            qfn = query_dir + qtmp + "/" + fn
            # qrep = load_sql_rep(qfn)
            with open(qfn, "rb") as f:
                qrep = pickle.load(f)
            mapping[str(deterministic_hash(qrep["sql"]))] = qfn
    return mapping

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
    # if exp_args["max_discrete_featurizing_buckets"] > 10:
        # return True

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

def get_all_objects(results_dir, obj_name):
    all_dfs = []
    fns = os.listdir(results_dir)
    for fn in fns:
        cur_dir = results_dir + "/" + fn
        if os.path.exists(cur_dir + "/" + obj_name):
            df = load_object(cur_dir + "/" + obj_name)
        else:
            continue
        exp_args = load_object(cur_dir + "/args.pkl")
        exp_args = vars(exp_args)
        if "nn" in exp_args["algs"]:
            df["alg"] = exp_args["loss_func"]
        else:
            df["alg"] = exp_args["algs"]
        all_dfs.append(df)
    return pd.concat(all_dfs)

def get_all_runtimes(results_dir):
    all_dfs = []
    fns = os.listdir(results_dir)
    for fn in fns:
        cur_dir = results_dir + "/" + fn
        if os.path.exists(cur_dir + "/runtimes.pkl"):
            runtimes = load_object(cur_dir + "/runtimes.pkl")
        elif os.path.exists("runtimes_jerr.pkl"):
            continue
        else:
            continue
        exp_args = load_object(cur_dir + "/args.pkl")
        exp_args = vars(exp_args)
        perrs = load_object(cur_dir + "/plan_pg_err.pkl")
        # ferrs = load_object(cur_dir + "/flow_err.pkl")
        # qerrs = load_object(cur_dir + "/query_qerr.pkl")
        runtimes = runtimes.drop_duplicates("sql_key")
        rt_keys = set(runtimes["sql_key"])
        assert len(rt_keys) == len(runtimes)
        # combined_df = jerr_df.merge(true_rts, on="sql_key")
        runtimes = runtimes.merge(perrs[["sql_key", "template"]], on="sql_key")
        runtimes = runtimes.merge(perrs[["sql_key", "qfn"]], on="sql_key")
        # runtimes["loss_type"] = "runtime"
        # runtimes = runtimes.rename(columns={"runtime":"cost"})
        # perrs = perrs[perrs["loss_type"] == "jcost"]
        runtimes = runtimes.merge(perrs[["sql_key", "loss"]], on="sql_key")
        runtimes = runtimes.merge(perrs[["sql_key", "cost"]], on="sql_key")

        # df = pd.concat([runtimes, perrs])
        df = runtimes
        if "nn" in exp_args["algs"]:
            df["alg"] = exp_args["loss_func"]
        else:
            df["alg"] = exp_args["algs"]
        df = df.drop_duplicates(["alg", "sql_key"])
        all_dfs.append(df)
    return pd.concat(all_dfs)

def get_all_training_df(results_dir):
    all_dfs = []
    fns = os.listdir(results_dir)
    for fn in fns:
        try:
            # convert to same format as qerrs
            cur_dir = results_dir + "/" + fn
            exp_args = load_object(cur_dir + "/args.pkl")
            if exp_args is None:
                continue
            exp_args = vars(exp_args)
            if skip_exp(exp_args):
                print("skip exp!")
                continue

            if "feat_tolerance" not in exp_args:
                if "675" in fn:
                    exp_args["feat_tolerance"] = 1
                else:
                    exp_args["feat_tolerance"] = 0

            alg = get_alg_name(exp_args)
            start = time.time()
            nns = load_object(cur_dir + "/nn.pkl")
            args_hash = str(deterministic_hash(str(exp_args)))[0:5]
            df = nns["stats"]
            df = df.assign(**exp_args)
            df["exp_hash"] = args_hash

            if exp_args["weight_decay"] == 4.0 and "138" in cur_dir:
                # print("resetting buggy weight decay to 10")
                df["weight_decay"] = 10.0
            else:
                df["weight_decay"] = exp_args["weight_decay"]

            if exp_args["sampling_priority_alpha"] > 0:
                df["priority"] = True
            else:
                df["priority"] = False

            if "normalize_flow_loss" in exp_args:
                df["normalize_flow_loss"] = exp_args["normalize_flow_loss"]
            else:
                df["normalize_flow_loss"] = 1

            all_dfs.append(df)
        except Exception as e:
            print("exception caught!")
            print(e)
            continue

    return pd.concat(all_dfs)

def get_all_plans(results_dir):
    all_dfs = []
    fns = os.listdir(results_dir)
    for fn in fns:
        # convert to same format as qerrs
        cur_dir = results_dir + "/" + fn
        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            continue
        exp_args = vars(exp_args)
        if skip_exp(exp_args):
            continue
        alg = get_alg_name(exp_args)
        nns = load_object(cur_dir + "/nn.pkl")
        qdf = pd.DataFrame(nns["query_stats"])
        if "query_qerr_stats" in nns:
            qerr_df = pd.DataFrame(nns["query_qerr_stats"])
            qdf = qdf.merge(qerr_df, on=["query_name", "epoch"])

        qdf["alg"] = alg
        qdf["hls"] = exp_args["hidden_layer_size"]
        qdf["exp_name"] = fn
        # priority based on args
        if exp_args["sampling_priority_alpha"] > 0:
            qdf["priority"] = True
        else:
            qdf["priority"] = False

        # TODO: add training / test detail
        # TODO: add template detail
        # TODO: need map from query_name : test/train + template etc.

        all_dfs.append(qdf)

    return pd.concat(all_dfs)

### Helper plotting utilities for jupyter notebooks
def plot_summaries(df, loss_type, HUE_COLORS=None, order=None):
    #fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    cur_df = df[df["loss_type"] == loss_type]
    #sns.catplot(x="alg_name", y="loss", data=cur_df, col_wrap = 2, cols="samples_type", kind="bar")
    train_df = cur_df[cur_df["samples_type"] == "train"]
    sns.barplot(x="alg_name", y="loss", data=train_df, hue="alg_name",
            palette=HUE_COLORS, ax = ax1, order=order)
    test_df = cur_df[cur_df["samples_type"] == "test"]
    sns.barplot(x="alg_name", y="loss", data=test_df, hue="alg_name",
            palette=HUE_COLORS, ax = ax2, order=order)
    ax1.set_title("Train", fontsize=25)
    ax2.set_title("Test", fontsize=25)
    sup_title = ERROR_NAMES[loss_type]
    fig.suptitle(sup_title, fontsize=75)
    # fg.despine(left=True)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()
    plt.clf()


ERROR_NAMES = {}
ERROR_NAMES["qerr"] = "MSE"
ERROR_NAMES["flow_err"] = "Flow Loss"
ERROR_NAMES["flow_ratio"] = "Flow Ratio"
ERROR_NAMES["mm1_plan_err"] = "Simple Plan Error"
ERROR_NAMES["mm1_plan_ratio"] = "Simple Plan Ratio"

ERROR_NAMES["mm1_plan_pg_err"] = "Simple Plan Postgres Error"
ERROR_NAMES["mm1_plan_pg_ratio"] = "Simple Plan Postgres Ratio"

ERROR_NAMES["plan_err"] = "Simple Plan Error"
ERROR_NAMES["plan_ratio"] = "Simple Plan Ratio"

ERROR_NAMES["jerr"] = "Postgres Plan Error"
ERROR_NAMES["jerr_ratio"] = "Postgres Plan Ratio"

COST_MODEL_NAMES = {}
COST_MODEL_NAMES["nested_loop_index3"] = "cm1"
COST_MODEL_NAMES["nested_loop_index2"] = "cm2"
COST_MODEL_NAMES["nested_loop_index"] = "cm3a"
COST_MODEL_NAMES["nested_loop_index4"] = "cm3b"
COST_MODEL_NAMES["nested_loop_index8"] = "cm4b"
COST_MODEL_NAMES["nested_loop_index8b"] = "cm4c"
COST_MODEL_NAMES["nested_loop_index9"] = "cm4a"
COST_MODEL_NAMES["nested_loop_index14"] = "cm4d"
COST_MODEL_NAMES["nested_loop_index13"] = "cm5"

def plot_loss_summary(df, loss_type, samples_type, yscale, ax,
        HUE_COLORS=None, miny=None, maxy=None):
    title_fmt = "{}"
    loss_title = ERROR_NAMES[loss_type]
    title = title_fmt.format(loss_title)

    ax.set_title(title, fontsize=40)
    cur_df = df[df["samples_type"] == samples_type]
    cur_df = cur_df[cur_df["loss_type"] == loss_type]

    # scale_df = df[df["epoch"] >= 4]
    # scale_df = scale_df[scale_df["loss_type"] == loss_type]

    # if miny is None:
    # miny = min(cur_df["loss"])
    miny = min(cur_df["loss"])

    if maxy is None:
        # maxy = max(scale_df["loss"])
        maxy = cur_df["loss"].quantile(0.95)

    sns.lineplot(x="epoch", y="loss", hue="alg_name", data=cur_df,
            palette=HUE_COLORS, ci=95,
                 ax=ax, legend="full", linewidth=10)
    ax.set_ylim((miny,maxy))
    # plt.setp(g.ax.lines,linewidth=lw)  # set lw for all lines of g axes
    # ax.set_linewidth(10)
    # ax.spines[0].set_linewidth(0.5)

    ax.set_yscale(yscale)
    ax.get_legend().remove()
    ax.tick_params(labelsize=20)
    ax.xaxis.label.set_size(20)

    #plt.show()

def construct_summary(df, samples_type, title, summary_type, HUE_COLORS=None):
    fig, axs = plt.subplots(1, 5, figsize=(40,10))
    fig.suptitle(title, fontsize=50)
    if summary_type == "ratio":
        plot_loss_summary(df, "qerr", samples_type, "linear", axs[0],
                     HUE_COLORS=HUE_COLORS, miny=0.0)

        plot_loss_summary(df, "flow_ratio", samples_type, "linear", axs[1],
                    HUE_COLORS=HUE_COLORS, miny=1.0)
        plot_loss_summary(df, "mm1_plan_ratio", samples_type, "linear", axs[2],
                HUE_COLORS=HUE_COLORS, miny=1.0)

        plot_loss_summary(df, "mm1_plan_pg_ratio", samples_type, "linear", axs[3],
                HUE_COLORS=HUE_COLORS, miny = 1.0)
        plot_loss_summary(df, "jerr_ratio", samples_type, "linear", axs[4],
                HUE_COLORS=HUE_COLORS, miny = 1.0)
    else:
        plot_loss_summary(df, "qerr", samples_type, "linear", axs[0],
                     HUE_COLORS=HUE_COLORS, miny=0.0)
        plot_loss_summary(df, "flow_err", samples_type, "linear", axs[1],
                    HUE_COLORS=HUE_COLORS, miny=0.0)
        plot_loss_summary(df, "mm1_plan_err", samples_type, "linear", axs[2],
                HUE_COLORS=HUE_COLORS, miny=0.0)

        plot_loss_summary(df, "mm1_plan_pg_err", samples_type, "linear", axs[3],
                HUE_COLORS=HUE_COLORS, miny = 0.0)
        plot_loss_summary(df, "jerr", samples_type, "linear", axs[4],
                HUE_COLORS=HUE_COLORS, miny = 0.0)


    if samples_type == "train":
        plt.tight_layout(rect=[0, 0, 1, 0.70])
        handles, labels = axs[-1].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='upper left',
                prop={'size': 30})
        for line in leg.get_lines():
            line.set_linewidth(10.0)
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.90])

    plt.show()

def plot_alg_grad(par_grad_df, alg_name, HUE_COLORS=None):
    par_grad_df = par_grad_df[par_grad_df["alg_name"] == alg_name]
    fg = sns.relplot(x="epoch", y="loss", data=par_grad_df, col="loss_type", col_wrap=3,
            hue="alg_name", palette=HUE_COLORS, kind="line", facet_kws={'sharey': False, 'sharex': True},
               legend=False, linewidth=10.0)
    title = "Parameter Gradients, per layer: " + alg_name
    fg.fig.suptitle(title,
                x=0.5, y=.99, horizontalalignment='center',
                verticalalignment='top', fontsize = 40)
    fg.despine(left=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_tmp_grad(par_grad_df, alg_name, HUE_COLORS=None):
    par_grad_df = par_grad_df[par_grad_df["alg_name"] == alg_name]
    fg = sns.relplot(x="epoch", y="loss", data=par_grad_df, col="template", col_wrap=3,
            hue="alg_name", palette=HUE_COLORS, kind="line", facet_kws={'sharey': False, 'sharex': True},
               legend=False, linewidth=10.0)
    title = "Parameter Gradients for each template " + alg_name
    fg.fig.suptitle(title,
                x=0.5, y=.99, horizontalalignment='center',
                verticalalignment='top', fontsize = 40)
    fg.despine(left=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

title_fmt = "Mean-{}-{}"
def plot_loss(df, loss_type, samples_type, yscale):

    fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
    if loss_type == "qerr":
        title = title_fmt.format("MSE", samples_type)
    else:
        title = title_fmt.format(loss_type, samples_type)
    plt.title(title, fontsize=50)
    cur_df = df[df["samples_type"] == samples_type]
    cur_df = cur_df[cur_df["loss_type"] == loss_type]

    scale_df = df[df["epoch"] == 4]
    scale_df = scale_df[scale_df["loss_type"] == loss_type]
    maxy = max(scale_df["loss"])
    miny = min(cur_df["loss"])
    #print(cur_df)
    sns.lineplot(x="epoch", y="loss", hue="alg_name", data=cur_df, palette=HUE_COLORS, ci=None)
    plt.ylim((miny,maxy))
    #plt.ylim((0,10000))
    plt.yscale(yscale)
    plt.rc('legend', fontsize=10, loc="lower left")    # legend fontsize
    plt.show()


template_title_fmt = "All-Templates-{}-{}"
def plot_loss_template(df, loss_type, samples_type, yscale,
        HUE_COLORS):
    #if loss_type == "qerr":
        #title = template_title_fmt.format("MSE", samples_type)
    #else:

    title_name = ERROR_NAMES[loss_type]
    title = title_fmt.format(title_name, samples_type)

    loss_df = df[df["loss_type"] == loss_type]
    loss_df = loss_df[loss_df["samples_type"] == samples_type]
    scale_df = df[df["epoch"] == 4]
    scale_df = scale_df[scale_df["loss_type"] == loss_type]
    #maxy = max(scale_df["loss"])
    #miny = min(loss_df["loss"])
    #fig = plt.figure(figsize=(50, 50), dpi= 80, facecolor='w', edgecolor='k')
    fg = sns.relplot(x = "epoch", y = "loss", data=loss_df, col="template", col_wrap=3,
                 facet_kws={'sharex':True, 'sharey':False},
                 hue="alg_name", palette=HUE_COLORS, linewidth=10.0, kind="line")
    #fg.set(ylim=(miny,maxy))
    fg.set(yscale=yscale)
       # fg.fig.suptitle(title, fontsize=50)
    fg.fig.suptitle(title,
                x=0.5, y=.99, horizontalalignment='center',
                verticalalignment='top', fontsize = 40)
    fg.despine(left=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    query_dir = "./our_dataset/queries/"
    qkey_mapping = qkey_map(args.query_dir)
    plans = get_all_plans(args.results_dir)
    print(plans)
    pdb.set_trace()

if __name__ == "__main__":
    args = read_flags()
    main()



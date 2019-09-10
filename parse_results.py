import argparse
import os
import pandas
import time
from utils.utils import *
from db_utils.utils import *
import pdb
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc("pdf", fonttype=42)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.image as mpimg
from collections import defaultdict
import klepto

import warnings
warnings.filterwarnings("ignore")

DB_ORDER = ["dmv", "synthdb", "imdb", "osm2", "higgs", "power"]
EXP_KEYS = ["num_params", "eval_times", "train_times"]
# ALGS_ORDER = ["pg", "NN1", "S0.1", "S1",
        # "cl", "cl5", "clr","clr5"]
ALGS_ORDER = ["pg", "S0.1", "S1",
        "cl", "cl5", "clr","clr5"]

PALETTE ={"pg":"blue",
        "NN1":"green",
        "S0.1":"#FF8C00",
        "S1": "#FFA500",
        "cl":"#FF0000",
        "cl5":"#8B0000",
        "clr":"#800000",
        "clr5":"#FF6347"}
        # "cl":"blue",
        # "cl5":"blue",
        # "clr":"blue",
        # "clr5":"blue"}

COL_WRAP = 3
DB_COLUMNS = {"dmv":"11", "osm2":"5", "higgs":"8", "power":"7", "synthdb":"0",
        "imdb":"0"}

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=False,
            default="dmv")
    parser.add_argument("--pgm", type=int, required=False,
            default=1)
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results/")
    parser.add_argument("--output_dir", type=str, required=False,
            default="./pdfs/")
    parser.add_argument("--per_subquery", type=int, required=False,
            default=0)
    parser.add_argument("--per_query", type=int, required=False,
            default=0)
    parser.add_argument("--worst_query_joins", type=int, required=False,
            default=0)

    return parser.parse_args()

def gen_table_data(df, algs, loss_types, summary_type):
    # generate nd-array of values
    vals = np.zeros((len(algs), len(loss_types)))
    for i, alg in enumerate(algs):
        tmp_df = df[df["alg_name"] == alg]
        for j, loss in enumerate(loss_types):
            tmp_df2 = tmp_df[tmp_df["loss_type"] == loss]["loss"]
            if summary_type == "mean":
                vals[i][j] = round(tmp_df2.mean(), 2)
            elif summary_type == "median":
                vals[i][j] = round(tmp_df2.median(), 2)
            elif summary_type == "95":
                vals[i][j] = round(tmp_df2.quantile(0.95), 2)
            elif summary_type == "99":
                vals[i][j] = round(tmp_df2.quantile(0.99), 2)

    return vals

def parse_results(results_cache, trainining_queries=True):
    '''
    '''
    data = defaultdict(list)
    exp_data = defaultdict(list)

    for k, results in results_cache.items():
        print(k)
        result_args = results["args"]

        if trainining_queries:
            queries = results["training_queries"]
        else:
            queries = results["test_queries"]

        for k in EXP_KEYS:
            # if k is not in results:
                # continue
            for alg, val in results[k].items():
                print(alg)
                if alg == "chow-liu5":
                    alg = "cl5"
                elif alg == "chow-liu-recomp":
                    alg = "clr"
                elif alg == "chow-liu-recomp5":
                    alg = "clr5"
                elif alg == "chow-liu":
                    alg = "cl"
                elif alg == "Sampling1":
                    alg = "S1"
                elif alg == "Sampling0.1":
                    alg = "S0.1"
                elif alg == "Postgres":
                    alg = "pg"

                exp_data["alg_name"].append(alg)
                exp_data["stat_type"].append(k)
                exp_data["db_name"].append(result_args.db_name)
                exp_data["val"].append(val)

        # parses per query stuff
        for i, q in enumerate(queries):
            # selectivity prediction
            true_sel = q.true_sel
            template = q.template_name

            for alg, loss_types in q.losses.items():
                for lt, loss in loss_types.items():
                    if alg == "chow-liu5":
                        alg = "cl5"
                    elif alg == "chow-liu-recomp":
                        alg = "clr"
                    elif alg == "chow-liu-recomp5":
                        alg = "clr5"
                    elif alg == "chow-liu":
                        alg = "cl"
                    elif alg == "Sampling1":
                        alg = "S1"
                    elif alg == "Sampling0.1":
                        alg = "S0.1"
                    elif alg == "Postgres":
                        alg = "pg"

                    data["alg_name"].append(alg)
                    data["loss_type"].append(lt)
                    data["loss"].append(loss)
                    data["template"].append(template)
                    data["true_sel"].append(true_sel)
                    data["db_name"].append(result_args.db_name)
                    if hasattr(q, "subqueries"):
                        data["num_subqueries"].append(len(q.subqueries))
                    else:
                        data["num_subqueries"].append(0)

    df = pd.DataFrame(data)
    exp_df = pd.DataFrame(exp_data)
    return df, exp_df

def gen_data_summary(df, pdf, loss_type="qerr"):
    fg = sns.FacetGrid(df, col="db_name", col_wrap=COL_WRAP,
            col_order=DB_ORDER)
    fg.map(sns.distplot, "true_sel")

    for i, ax in enumerate(fg.axes.flat):
        db = DB_ORDER[i]
        tmp_df = df[df["db_name"] == db]
        num_queries = len(set(tmp_df["template"]))
        num_columns = DB_COLUMNS[db]
        # sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        title = db
        title += "\n#columns: " + str(num_columns)
        title += "\n#queries: " + str(num_queries)
        ax.set_title(title)

    plt.suptitle("True Selectivity Distributions",
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 20)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig()
    plt.clf()

def gen_exp_summary(df, pdf, stat_type):
    '''
    TODO: share code w/ gen_losses_summary
    '''
    df = df[df["stat_type"] == stat_type]
    ylabel = "time"
    ymax = None
    if stat_type == "num_params":
        df["val"] = df["val"]*4 / (1e6)
        ylabel = "MBs"
    elif stat_type == "eval_times":
        ylabel = "Milliseconds"
        ymax = 10
    elif stat_type == "train_times":
        ylabel = "Seconds"

    # sns.set(style="ticks", rc={"lines.linewidth": 10.0})
    fg = sns.catplot(x="alg_name", y="val", hue="alg_name", col="db_name",
            col_wrap=COL_WRAP, kind="bar", data=df, estimator=np.mean, ci="sd",
            legend_out=False, sharex=True,sharey=True, col_order=DB_ORDER,
            hue_order=ALGS_ORDER, order=ALGS_ORDER, palette=PALETTE)

    fg.set(ylabel=ylabel)
    # if ymax is not None:
        # fg.set(ylim=ymax)

    # fg.set(yscale="log")
    fg.despine(left=True)

    plt.suptitle("{}".format(stat_type),
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 20)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig()
    plt.clf()

def gen_losses_summary(df, pdf, loss_type="qerr"):
    '''
    Each database should get its own column.
    '''
    # only plot the given loss type
    df = df[df["loss_type"] == loss_type]

    # TODO: mean, median, tail info should be present
    fg = sns.catplot(x="alg_name", y="loss", hue="alg_name", col="db_name",
            col_wrap=COL_WRAP, kind="bar", data=df, estimator=np.mean, ci="sd",
            legend_out=False, sharex=False, sharey=False, col_order=DB_ORDER,
            hue_order=ALGS_ORDER, order=ALGS_ORDER, palette=PALETTE)

    # set individual titles for each column
    for i, ax in enumerate(fg.axes.flat):
        db = DB_ORDER[i]
        tmp_df = df[df["db_name"] == db]
        num_queries = len(set(tmp_df["template"]))
        # sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        title = db + " ,#queries: " + str(num_queries)
        ax.set_title(title)

    plt.suptitle("Losses: {}".format(loss_type),
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 20)

    fg.set(yscale="log")
    fg.despine(left=True)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig()
    plt.clf()

    # TODO: generate tables with tails etc. explicitly shown

def pgm_plots():

    results_cache = klepto.archives.dir_archive(args.results_dir)
    results_cache.load()
    # collect all the data in a large dataframe
    train_df, exp_df = parse_results(results_cache, True)
    summary_pdf = PdfPages(args.results_dir + "/summary.pdf")

    # TODO: potentially exclude some of the algorithms

    # page 1: data / samples summary
    gen_data_summary(train_df, summary_pdf)

    # page 2: losses summary
    gen_losses_summary(train_df, summary_pdf)

    # aggregate summaries
    for k in EXP_KEYS:
        gen_exp_summary(exp_df, summary_pdf, k)

    summary_pdf.close()

def main():
    if args.pgm:
        pgm_plots()
    else:
        print("need to merge with join branch")
        exit(-1)

args = read_flags()
main()

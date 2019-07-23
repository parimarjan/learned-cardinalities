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

BASELINE = "EXHAUSTIVE"
FIX_TEMPLATE = False

# ms
MAX_RUNTIME = 500000

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
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

def add_data_row(data, alg, lt, loss, template, true_sel, optimizer_name,
        jl_start_iter, q, baseline, cost):
    data["alg_name"].append(alg)
    data["loss_type"].append(lt)
    data["loss"].append(loss)
    data["template"].append(template)
    data["true_sel"].append(true_sel)
    data["optimizer_name"].append(optimizer_name)
    data["jl_start_iter"].append(jl_start_iter)
    data["baseline"].append(baseline)
    data["cost"].append(cost)
    if hasattr(q, "subqueries"):
        data["num_subqueries"].append(len(q.subqueries))
    else:
        data["num_subqueries"].append(0)


def plot_query(q0, pdf):
    jc = extract_join_clause(q0.query)
    pred_columns, pred_types, _ = extract_predicates(q0.query)
    pred_columns = [p[0:p.find(".")] for p in pred_columns]
    jg = get_join_graph(jc)
    other_tables = []
    pred_tables = []
    for table in jg.nodes():
        if table in pred_columns:
            pred_tables.append(table)
        else:
            other_tables.append(table)

    pos=nx.spring_layout(jg) # positions for all nodes
    nx.draw_networkx_nodes(jg , pos,
                           nodelist=pred_tables,
                           node_color='r',
                           node_size=2500,
                           alpha=0.3)
    nx.draw_networkx_nodes(jg,pos,
                           nodelist=other_tables,
                           node_color='b',
                           node_size=2500,
                           alpha=0.3)
    nx.draw_networkx_edges(jg, pos)
    nx.draw_networkx_labels(jg, pos)

    pdf.savefig()
    plt.close()

def gen_table_data(df, algs, loss_types, summary_type):
    # generate nd-array of values
    vals = np.zeros((len(algs), len(loss_types)))
    for i, alg in enumerate(algs):
        tmp_df = df[df["alg_name"] == alg]
        for j, loss in enumerate(loss_types):
            tmp_df2 = tmp_df[tmp_df["loss_type"] == loss]["loss"]
            if summary_type == "mean":
                vals[i][j] = round(tmp_df2.mean(), 1)
            elif summary_type == "median":
                vals[i][j] = round(tmp_df2.median(), 1)
            elif summary_type == "95":
                vals[i][j] = round(tmp_df2.quantile(0.95), 1)
            elif summary_type == "99":
                vals[i][j] = round(tmp_df2.quantile(0.99), 1)

    return vals

def parse_query_objs(results_cache, trainining_queries=True):
    '''
    '''
    query_data = defaultdict(list)
    data = defaultdict(list)
    # other things we care about?
    if FIX_TEMPLATE:
        qmap = {}
        fns = glob.glob("./templates/myjob/*.sql")
        for fn in fns:
            f = open(fn, "r")
            query = f.read()
            qmap[query] = os.path.basename(fn)
            f.close()

    for k, results in results_cache.items():
        print(k)
        if "args" in results:
            result_args = results["args"]
            # filter out stuff based on args
            if args.db_name != result_args.db_name:
                print("skipping: ", result_args.db_name)
                continue
            if hasattr(result_args, "optimizer_name"):
                optimizer_name = result_args.optimizer_name
            else:
                optimizer_name = "adam"

            if hasattr(result_args, "jl_start_iter"):
                jl_start_iter = result_args.jl_start_iter
            else:
                jl_start_iter = 200

        # else, just parse it

        if trainining_queries:
            queries = results["training_queries"]
        else:
            queries = results["test_queries"]

        for i, q in enumerate(queries):
            # add runtime data to same df
            # selectivity prediction
            true_sel = q.true_sel
            if FIX_TEMPLATE:
                if not q.query in qmap:
                    pdb.set_trace()
                template = qmap[q.query]
            else:
                template = q.template_name
            query_data[q.template_name].append(q)

            rt_algs = q.join_info.keys()
            for alg in rt_algs:
                ## FIXME: decompose this!! better schema for experiment results
                # current alg
                rts = q.join_info[alg]["dbmsAllRuntimes"]["RL"]
                cost = q.join_info[alg]["costs"]["RL"]
                for rt in rts:
                    if rt > MAX_RUNTIME:
                        print(rt)
                        continue
                    add_data_row(data, alg, "runtime", float(rt), template,
                            true_sel, optimizer_name, jl_start_iter, q,
                            BASELINE, cost)

                # other baselines
                rts = q.join_info[alg]["dbmsAllRuntimes"][BASELINE]
                cost = q.join_info[alg]["costs"][BASELINE]
                for rt in rts:
                    add_data_row(data, "true", "runtime", float(rt), template,
                            true_sel, optimizer_name, jl_start_iter, q,
                            BASELINE, cost)

            for alg, loss_types in q.losses.items():
                cost = q.join_info[alg]["costs"]["RL"]
                true_card_cost = q.join_info[alg]["costs"][BASELINE]
                for lt, loss in loss_types.items():
                    add_data_row(data, alg, lt, loss, template, true_sel,
                            optimizer_name, jl_start_iter, q, BASELINE, cost)
                    if lt == "qerr":
                        min_loss = 1.00
                    else:
                        min_loss = 0.00

                    add_data_row(data, "true", lt, min_loss, template,
                            true_sel, optimizer_name, jl_start_iter, q,
                            BASELINE, true_card_cost)

                    # TODO: add the predicted selectivity by this alg

    df = pd.DataFrame(data)
    return df, query_data

def gen_query_bar_graphs(df, pdf, sort_by_loss_type, sort_by_alg,
        alg_order):

    # algs = ["nn", "nn-jl1", "nn-jl2", "Postgres"]

    # first, only plot join losses
    sort_df = df[df["loss_type"] == sort_by_loss_type]
    sort_df = sort_df[sort_df["alg_name"] == sort_by_alg]
    # assert len(sort_df) >= 110
    # if len(sort_df) < 110:
        # print("skipping experiment as less than 110 queries")
        # return

    # sort_df = sort_df[sort_df["loss"] > 5.00]
    sort_df.sort_values("loss", ascending=False, inplace=True)
    templates = sort_df["template"].drop_duplicates()
    # templates = [t for t in templates]
    templates = templates.values[0:15]

    # will also select the qerrors
    to_plot = df[df["template"].isin(templates)]

    fg = sns.catplot(x="loss_type", y="loss", hue="alg_name", col="template",
            col_wrap=5, kind="bar", data=to_plot, estimator=np.median, ci=100,
            legend_out=False, col_order=templates, sharex=False, order=["join",
                "qerr"], hue_order=alg_order)


    for i, ax in enumerate(fg.axes.flat):
        tmp = templates[i]
        sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        title = tmp + " ,#subqueries: " + str(sqs)
        ax.set_title(title)

    # plt.subplots_adjust(top= 1 - 1/8.0)
    # fg.fig.suptitle('Worst Queries, sorted by {}'.format(sort_by_loss_type),
            # y= 1 - 1/16.0)

    fg.fig.suptitle("Sorted by worst {} for {}".format(sort_by_loss_type,
        sort_by_alg),
            x=0.5, y=.99, horizontalalignment='center',
            verticalalignment='top', fontsize = 40)

    # plt.gcf()

    # fg.set(ylim=(0,50.0))
    fg.set(yscale="log")
    fg.despine(left=True)

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pdf.savefig()
    plt.clf()

def gen_runtime_plots(df, pdf):
    '''
    Plot: x-axis: cost, y-axis = runtime, color = alg name
    '''
    # select only runtime rows
    df = df[df["loss_type"] == "runtime"]
    df["loss"] /= 1000
    ax = sns.scatterplot(x="cost", y="loss", hue="alg_name",
            data=df, estimator=np.mean, ci=99)

    fig = ax.get_figure()
    # ax.set_yscale("log")
    ax.set_ylim((0, 10**2))
    ax.set_ylabel("seconds")

    # plt.title(",".join(q0.table_names))
    plt.title("Cost Model Output v/s Runtime")
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

def gen_error_summaries(df, pdf, algs_to_plot=None,barcharts=False, tables=True):
    # firstPage = plt.figure()
    # firstPage.clf()
    # txt = "Summary Results \n"
    # txt += "Num Experiments: " + str(len(set(orig_df["train-time"]))) + "\n"
    # txt += "DB: " + str(set(orig_df["dbname"])) + "\n"
    # txt += "Algs: " + str(set(orig_df["alg_name"])) + "\n"
    # txt += "Num Test Samples: " + str(set(orig_df["num_vals"])) + "\n"

    # firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
    # pdf.savefig()
    # plt.close()

    if barcharts:
        non_join_df = df[df["loss_type"] != "join"]
        ax = sns.barplot(x="loss_type", y="loss", hue="alg_name",
                data=non_join_df, estimator=np.median, ci=99)

        fig = ax.get_figure()
        plt.title(",".join(q0.table_names))
        plt.tight_layout()
        pdf.savefig()
        plt.clf()

        if "join" in set(df["loss_type"]):
            join_df = df[df["loss_type"] == "join"]
            ax = sns.barplot(x="loss_type", y="loss", hue="alg_name",
                    data=join_df, estimator=np.median, ci=99)

            fig = ax.get_figure()
            plt.title(",".join(q0.table_names))
            plt.tight_layout()
            pdf.savefig()
            plt.clf()

    if tables:
        FONT_SIZE = 12
        COL_WIDTH = 0.25

        # columns
        loss_types = [l for l in set(df["loss_type"])]
        COL_WIDTHS = [COL_WIDTH for l in loss_types]
        all_algs = [l for l in set(df["alg_name"])]
        algs = []
        for alg in all_algs:
            if alg in algs_to_plot:
                algs.append(alg)

        mean_vals = gen_table_data(df, algs, loss_types, "mean")
        median_vals = gen_table_data(df, algs, loss_types, "median")
        tail1 = gen_table_data(df, algs, loss_types, "95")
        tail2 = gen_table_data(df, algs, loss_types, "99")

        fig, axs = plt.subplots(2,2)
        for i in range(2):
            for j in range(2):
                axs[i][j].axis("tight")
                axs[i][j].axis("off")

        def plot_table(vals, i, j, title):
            table = axs[i][j].table(cellText=vals,
                                  rowLabels=algs,
                                  # rowColours=colors,
                                  colLabels=loss_types,
                                  loc='center',
                                  fontsize=FONT_SIZE,
                                  colWidths=COL_WIDTHS)
            axs[i][j].set_title(title)
            table.set_fontsize(FONT_SIZE)


        # plot_table(mean_vals, 0,0,rowLabels, colLabels, FONT_SIZE, COL_WIDTHS)
        plot_table(mean_vals, 0,0, "Mean Losses")
        plot_table(median_vals, 0,1, "Median Losses")
        plot_table(tail1, 1,0, "95th Percentile")
        plot_table(tail2, 1,1, "99th Percentile")

        tmp_df = df[df["alg_name"] == "Postgres"]
        tmp_df = tmp_df[tmp_df["loss_type"] == "qerr"]
        plt.suptitle("Dataset: {}, Num Queries: {}".format(args.db_name,
            len(tmp_df)),
                x=0.5, y=.99, horizontalalignment='center',
                verticalalignment='top', fontsize = 10)

        plt.savefig("summary.png")
        pdf.savefig()
        plt.clf()

def plot_queries(query_data, pdf):
    for qname, queries in query_data.items():
        plot_single_query(qname, queries, pdf)

def plot_single_query(qname, queries, pdf):
    base_alg = [alg for alg in queries[0].join_info.keys()][0]
    # alg name: true, postgres, random etc.
    unique_join_orders = {}
    for q in queries:
        all_infos = q.join_info
        # For true values, just add the Exhaustive orders
        if "true" not in unique_join_orders:
            unique_join_orders["true"] = []
        tmp_info = all_infos[base_alg]
        unique_join_orders["true"].append(tmp_info["joinOrders"][BASELINE]["joinStr"])

        # Postgres, Random etc.
        for alg, info in all_infos.items():
            if alg not in unique_join_orders:
                unique_join_orders[alg] = []
            unique_join_orders[alg].append(info["joinOrders"]["RL"]["joinStr"])

    order_data = {}
    order_data["alg"] = []
    order_data["order"] = []
    order_data["order_hash"] = []

    for k,v in unique_join_orders.items():
        for o in v:
            order_data["alg"].append(k)
            order_data["order"].append(o)
            order_data["order_hash"].append(hash(o) % 100)

    order_df = pd.DataFrame(order_data)
    ax = sns.countplot(x="order_hash", hue="alg",
            data=order_df)
    plt.title("Join Order Distribution")
    pdf.savefig()
    plt.clf()

    # sort queries according to join-loss
    sorted_queries = sorted(queries, key=lambda q: \
            q.losses[base_alg]["join"], reverse=True)

    for q in sorted_queries:
        all_infos = q.join_info
        from park.envs.query_optimizer.qopt_utils import plot_join_order
        # write out the sql
        ## parameters
        firstPage = plt.figure()
        firstPage.clf()

        ## TODO: just paste all the args here?
        # txt = "Average Results \n"
        # txt += "Num Experiments: " + str(len(set(orig_df["train-time"]))) + "\n"
        # txt += "DB: " + str(set(orig_df["dbname"])) + "\n"
        # txt += "Algs: " + str(set(orig_df["alg_name"])) + "\n"
        # txt += "Num Test Samples: " + str(set(orig_df["num_vals"])) + "\n"

        txt = all_infos[base_alg]["sql"]
        firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
        pdf.savefig()
        plt.close()

        for alg, info in all_infos.items():
            alg_cards = q.subq_cards[alg]
            true_cards = q.subq_cards["true"]
            plot_join_order(info, pdf, single_plot=False,
                    python_alg_name=alg, est_cards=alg_cards,
                    true_cards=true_cards)

def main():
    results_cache = klepto.archives.dir_archive(args.results_dir)
    results_cache.load()
    # collect all the data in a large dataframe
    train_df, query_data = parse_query_objs(results_cache, True)
    # test_df = parse_query_objs(results_cache, False)

    summary_pdf = PdfPages(args.results_dir + "/summary.pdf")
    make_dir(args.output_dir)
    algs = ["nn", "nn-jl1", "nn-jl2", "Postgres", "ourpgm", "greg", "chow-liu",
            "true"]
    train_df = train_df[train_df["alg_name"].isin(algs)]
    print("going to generate summary pdf")
    gen_error_summaries(train_df, summary_pdf, algs_to_plot=algs)
    if "runtime" in set(train_df["loss_type"]):
        print("going to generate runtime summary")
        gen_runtime_plots(train_df, summary_pdf)
    else:
        assert False

    if args.worst_query_joins:
        for alg in algs:
            gen_query_bar_graphs(train_df, summary_pdf, "join", alg, algs)
            gen_query_bar_graphs(train_df, summary_pdf, "qerr", alg, algs)

    summary_pdf.close()

    if args.per_query:
        # FIXME: fix this
        queries_pdf = PdfPages(args.results_dir + "/training_queries.pdf")
        plot_queries(query_data, queries_pdf)
        queries_pdf.close()

args = read_flags()
main()

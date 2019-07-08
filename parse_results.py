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

# TODO: maybe will use this?
def init_result_row(result):
    result["dbname"].append(args.db_name)
    result["template_dir"].append(args.template_dir)
    result["seed"].append(args.random_seed)
    result["args"].append(args)
    result["num_bins"].append(args.num_bins)
    result["avg_factor"].append(args.avg_factor)
    result["num_columns"].append(len(db.column_stats))

    result["train-time"].append(train_time)
    result["eval-time"].append(eval_time)
    result["alg_name"].append(alg.__str__())
    result["loss-type"].append(loss_func.__name__)
    result["loss"].append(cur_loss)
    result["test-set"].append(0)
    result["num_vals"].append(len(train_queries))

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_len", type=int, required=False,
            default=10)
    parser.add_argument("--db_name", type=str, required=False,
            default="synth-gaussian")
    parser.add_argument("--plot", type=int, required=False,
            default=1)
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results/")
    parser.add_argument("--per_subquery", type=int, required=False,
            default=0)
    parser.add_argument("--per_query", type=int, required=False,
            default=0)
    parser.add_argument("--join_parse", type=int, required=False,
            default=1)

    return parser.parse_args()

def visualize_query_class(queries, pdf, barcharts=False):
    q0 = queries[0]
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
    true_sels = [q.true_sel for q in queries]
    x = pd.Series(true_sels, name="true selectivities")
    ax = sns.distplot(x, kde=False)
    total_count = q0.total_count
    plt.title("Selectivities, total: " + str(total_count))
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

    use_losses = ["qerr", "rel", "join"]
    # for each loss key, make a new plot
    all_losses = []
    for q in queries:
        for alg_name, losses in q.losses.items():
            for loss_type, loss in losses.items():
                if not loss_type in use_losses:
                    continue
                tmp = {}
                tmp["alg_name"] = alg_name
                tmp["loss_type"] = loss_type
                tmp["loss"] = loss
                all_losses.append(tmp)

    df = pd.DataFrame(all_losses)
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
    else:
        FONT_SIZE = 12
        COL_WIDTH = 0.25

        # columns
        loss_types = [l for l in set(df["loss_type"])]
        COL_WIDTHS = [COL_WIDTH for l in loss_types]
        # rows
        algs = [l for l in set(df["alg_name"])]
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

        pdf.savefig()
        plt.clf()

        # columns
        # loss_types = [l for l in set(df["loss_type"])]
        # COL_WIDTHS = [COL_WIDTH for l in loss_types]
        # # rows
        # algs = [l for l in set(df["alg_name"])]
        # # generate nd-array of values
        # mean_vals = np.zeros((len(loss_types), len(algs)))
        # for i, alg in enumerate(algs):
            # tmp_df = df[df["alg_name"] == alg]
            # for j, loss in enumerate(loss_types):
                # tmp_df2 = tmp_df[tmp_df["loss_type"] == loss]
                # mean_vals[i][j] = round(tmp_df2.mean()[0], 2)

        # median_vals = np.zeros((len(loss_types), len(algs)))
        # for i, alg in enumerate(algs):
            # tmp_df = df[df["alg_name"] == alg]
            # for j, loss in enumerate(loss_types):
                # tmp_df2 = tmp_df[tmp_df["loss_type"] == loss]
                # median_vals[i][j] = round(tmp_df2.median()[0], 2)

        # fig, axs = plt.subplots(2,1)
        # axs[0].axis('tight')
        # axs[1].axis('tight')
        # axs[0].axis("off")
        # axs[1].axis("off")

        # # Add a table at the bottom of the axes
        # mean_table = axs[0].table(cellText=mean_vals,
                              # rowLabels=algs,
                              # # rowColours=colors,
                              # colLabels=loss_types,
                              # loc='center',
                              # fontsize=FONT_SIZE,
                              # colWidths=COL_WIDTHS)
        # axs[0].set_title("Mean Losses")
        # mean_table.set_fontsize(FONT_SIZE)

        # median_table = axs[1].table(cellText=median_vals,
                              # rowLabels=algs,
                              # # rowColours=colors,
                              # colLabels=loss_types,
                              # loc='center',
                              # fontsize=FONT_SIZE,
                              # colWidths=COL_WIDTHS)
        # axs[1].set_title("Median Losses")
        # median_table.set_fontsize(FONT_SIZE)

        # pdf.savefig()
        # plt.clf()

def gen_table_data(df, algs, loss_types, summary_type):
    # generate nd-array of values
    vals = np.zeros((len(algs), len(loss_types)))
    for i, alg in enumerate(algs):
        tmp_df = df[df["alg_name"] == alg]
        for j, loss in enumerate(loss_types):
            tmp_df2 = tmp_df[tmp_df["loss_type"] == loss]
            if summary_type == "mean":
                vals[i][j] = round(tmp_df2.mean()[0], 2)
            elif summary_type == "median":
                vals[i][j] = round(tmp_df2.median()[0], 2)
            elif summary_type == "95":
                vals[i][j] = round(tmp_df2.quantile(0.95)[0], 2)
            elif summary_type == "99":
                vals[i][j] = round(tmp_df2.quantile(0.99)[0], 2)

    return vals

def parse_query_file_card(fn):
    print(fn)
    pdf_name = fn.replace(".pickle", ".pdf")
    pdf = PdfPages(pdf_name)
    queries = load_object(fn)

    # TODO: add average results page
    data = defaultdict(list)
    for q in queries:
        for alg, loss_types in q.losses.items():
            for lt, loss in loss_types.items():
                data["alg_name"].append(alg)
                data["loss_type"].append(lt)
                data["loss"].append(loss)


    df = pd.DataFrame(data)

    firstPage = plt.figure()
    firstPage.clf()
    ## TODO: just paste all the args here?
    # txt += "DB: " + str(set(df["dbname"])) + "\n"
    txt = "Experiment Details: \n"
    txt += "Algs: " + str(set(df["alg_name"])) + "\n"
    txt += "Num Test Samples: " + str(len(queries)) + "\n"

    firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
    pdf.savefig()
    plt.close()

    # pdb.set_trace()

    train_data = defaultdict(list)
    eval_data = defaultdict(list)
    train_time = queries[0].train_time
    eval_time = queries[0].eval_time

    for alg, t in train_time.items():
        if alg == "Postgres":
            continue
        train_data["alg_name"].append(alg)
        train_data["time"].append(t)

    for alg, t in eval_time.items():
        if alg == "Postgres":
            continue
        eval_data["alg_name"].append(alg)
        eval_data["time"].append(t)

    tdf = pd.DataFrame(train_data)
    edf = pd.DataFrame(eval_data)
    ax = sns.barplot(x="alg_name", y="time", hue="alg_name",
            data=tdf, estimator=np.mean, ci=75)
    fig = ax.get_figure()
    plt.title("Train Time")
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

    ax = sns.barplot(x="alg_name", y="time", hue="alg_name",
            data=edf, estimator=np.mean, ci=75)
    fig = ax.get_figure()
    plt.title("Evaluation Time")
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

    FONT_SIZE = 12
    COL_WIDTH = 0.25
    # columns
    loss_types = [l for l in set(df["loss_type"])]
    COL_WIDTHS = [COL_WIDTH for l in loss_types]
    # rows
    algs = [l for l in set(df["alg_name"])]
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

    pdf.savefig()
    plt.clf()

    pdf.close()

def parse_query_file_join(fn):
    '''
    Plot List:
        - plot query structure graph at first
        - selectivity hist for all main queries
        - selectivity buckets for each subquery CLASS
            - plot tables used as a graph
            - first bucket them by tables used
            - for each class, have separate qerror values
    TODO:
        - change use of EXHAUSTIVE to baseline everywhere etc.
        - handle errors better
    '''
    print(fn)
    pdf_name = fn.replace(".pickle", ".pdf")
    queries = load_object(fn)
    # TODO: add average results page

    pdf = PdfPages(pdf_name)
    visualize_query_class(queries, pdf)

    if hasattr(queries[0], "join_info") and args.per_query:
        # alg name: true, postgres, random etc.
        unique_join_orders = {}
        for q in queries:
            all_infos = q.join_info
            # For true values, just add the Exhaustive orders
            if "true" not in unique_join_orders:
                unique_join_orders["true"] = []
            tmp_info = all_infos["Postgres"]
            unique_join_orders["true"].append(tmp_info["joinOrders"]["EXHAUSTIVE"]["joinStr"])

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

            # join_order_ids = [hash(o) % 100 for o in v]
            # x = pd.Series(join_order_ids, name="join_orders")
            # ax = sns.distplot(x, kde=False)
            # plt.title(k + ": Join Order Distribution")
            # plt.tight_layout()
            # pdf.savefig()
            # plt.clf()

        order_df = pd.DataFrame(order_data)
        ax = sns.countplot(x="order_hash", hue="alg",
                data=order_df)
        plt.title("Join Order Distribution")
        pdf.savefig()
        plt.clf()

        # sort queries according to join-loss
        sorted_queries = sorted(queries, key=lambda q: \
                q.losses["Postgres"]["join"], reverse=True)

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

            txt = all_infos["Postgres"]["sql"]
            firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
            pdf.savefig()
            plt.close()

            for alg, info in all_infos.items():
                if info["queryName"] == "16":
                    continue
                alg_cards = q.subq_cards[alg]
                true_cards = q.subq_cards["true"]
                plot_join_order(info, pdf, single_plot=False,
                        python_alg_name=alg, est_cards=alg_cards,
                        true_cards=true_cards)


    if len(queries[0].subqueries) > 0 and args.per_subquery:
        all_subq_list = []
        for i in range(len(queries[0].subqueries)):
            # class of subqueries
            sq_sample = queries[0].subqueries[i]
            # only visualize these if it has some predicates
            if len(sq_sample.pred_column_names) > 0 \
                    and len(sq_sample.table_names) > 1:
                subqs = [q.subqueries[i] for q in queries]
                qerrs = [sq.losses["Postgres"]["qerr"] for sq in subqs]
                all_subq_list.append((subqs, np.mean(qerrs)))

        all_subq_list = sorted(all_subq_list, key=lambda x: x[1], reverse=True)
        for sqs in all_subq_list:
            visualize_query_class(sqs[0], pdf)

    pdf.close()

def parse_data():
    fns = glob.glob(args.result_dir + "/*train*.pickle")
    for fn in fns:
        if args.join_parse:
            parse_query_file_join(fn)
        else:
            parse_query_file_card(fn)

def main():
    parse_data()

args = read_flags()
main()

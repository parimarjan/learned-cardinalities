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

BASELINE = "LEFT_DEEP"

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results/")
    parser.add_argument("--per_subquery", type=int, required=False,
            default=0)
    parser.add_argument("--per_query", type=int, required=False,
            default=0)
    parser.add_argument("--join_parse", type=int, required=False,
            default=1)

    return parser.parse_args()

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
        - change use of BASELINE to baseline everywhere etc.
        - handle errors better
    '''
    print(fn)
    pdf_name = fn.replace(".pickle", ".pdf")
    queries = load_object(fn)
    # TODO: add average results page

    pdf = PdfPages(pdf_name)
    visualize_query_class(queries, pdf)

    if hasattr(queries[0], "join_info") and args.per_query:
        print("number of queries to plot: ", len(queries))
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
                # if info["queryName"] == "16":
                    # continue
                alg_cards = q.subq_cards[alg]
                true_cards = q.subq_cards["true"]
                # pdb.set_trace()
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

def parse_query_objs(results_cache, trainining_queries=True):
    '''
    '''
    # open relevant pdfs
    if args.per_query:
        pass

    data = defaultdict(list)
    # other things we care about?
    for k, results in results_cache.items():
        if "args" in results:
            result_args = results["args"]
            # filter out stuff based on args
            if args.db_name != result_args.db_name:
                print("skipping: ", result_args.db_name)
                continue
        # else, just parse it

        if trainining_queries:
            queries = results["training_queries"]
        else:
            queries = results["test_queries"]
        print(k)
        print(len(queries))

        # update the dictionaries using each query
        if args.per_query:
            pass
            # plot_queries(queries, result_args)

        for q in queries:
            # selectivity prediction
            true_sel = q.true_sel
            template = q.template_name

            for alg, loss_types in q.losses.items():
                for lt, loss in loss_types.items():
                    data["alg_name"].append(alg)
                    data["loss_type"].append(lt)
                    data["loss"].append(loss)
                    data["template"].append(template)
                    data["true_sel"].append(true_sel)
                    # TODO: add the predicted selectivity by this alg

    df = pd.DataFrame(data)
    return df

def main():
    results_cache = klepto.archives.dir_archive(args.results_dir)
    results_cache.load()
    # collect all the data in a large dataframe
    train_df = parse_query_objs(results_cache, True)
    # test_df = parse_query_objs(results_cache, False)

    # do stuff with this data. Bar graphs, summary tables et al.
    pdb.set_trace()

args = read_flags()
main()

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

    return parser.parse_args()

def visualize_query_class(queries, pdf):
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

def parse_query_file(fn):
    '''
    Plot List:
        - plot query structure graph at first
        - selectivity hist for all main queries
        - selectivity buckets for each subquery CLASS
            - plot tables used as a graph
            - first bucket them by tables used
            - for each class, have separate qerror values

    '''
    print(fn)
    pdf_name = fn.replace(".pickle", ".pdf")
    queries = load_object(fn)
    # TODO: add average results page

    pdf = PdfPages(pdf_name)
    visualize_query_class(queries, pdf)

    if hasattr(queries[0], "join_info") and args.per_query:
        # sort queries according to join-loss
        sorted_queries = sorted(queries, key=lambda q: \
                q.losses["Postgres"]["join"], reverse=True)

        for q in sorted_queries:
            all_infos = q.join_info
            from park.envs.query_optimizer.qopt_utils import plot_join_order
            for alg, info in all_infos.items():
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
        parse_query_file(fn)

def main():
    parse_data()

args = read_flags()
main()

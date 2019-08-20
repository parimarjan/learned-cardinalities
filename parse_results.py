import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import pandas
import time
from utils.utils import *
from db_utils.utils import *
from cardinality_estimation.query import *
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

BASELINE = "EXHAUSTIVE"
OLD_QUERY = True
DEBUG = False
MAX_QUERY = 1000

def read_flags():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--runtime_reps", type=int, required=False,
            default=0)
    parser.add_argument("--use_orig_query", type=int, required=False,
            default=0)
    parser.add_argument("--use_explain", type=int, required=False,
            default=1)
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="ubuntu")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--port", type=str, required=False,
            default=5433)

    return parser.parse_args()

def add_data_row(data, alg, lt, loss, template, true_sel, optimizer_name,
        jl_start_iter, q, baseline, cost, pg_cost):
    data["alg_name"].append(alg)
    data["loss_type"].append(lt)
    data["loss"].append(loss)
    data["template"].append(template)
    data["true_sel"].append(true_sel)
    data["optimizer_name"].append(optimizer_name)
    data["jl_start_iter"].append(jl_start_iter)
    data["baseline"].append(baseline)
    data["cost"].append(cost)
    data["pg_cost"].append(pg_cost)
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

def update_pg_costs(query):
    if not hasattr(query, "pg_costs"):
        query.pg_costs = {}

    for alg_name, v in query.explains.items():
        # ugh ugly structure
        explain = v[0][0][0][0]
        vals = extract_values(explain, "Total Cost")
        total_cost = vals[-1]
        query.pg_costs[alg_name] = total_cost

def update_runtimes(query, explain, use_orig_query=False,
        patch_pg_cards=True):

    if not hasattr(query, "runtimes"):
        query.runtimes = defaultdict(list)
    if explain and not hasattr(query, "explains"):
        query.explains = defaultdict(list)

    for i in range(args.runtime_reps):
        for alg in query.costs:
            print(alg)
            if use_orig_query:
                sql = query.query
                rt_alg = "pg_" + alg
                join_collapse_limit = 15
            else:
                sql = query.executed_sqls[alg][-1]
                # postgres should not be allowed to set join collapse limit
                join_collapse_limit = 1
                rt_alg = alg

            if len(query.runtimes[rt_alg]) > i:
                print("already have runtimes for: ", query.template_name)
                continue

            if patch_pg_cards:
                updated_cards = get_cardinalities(query, alg)
                # update the cardinalities file in $PG_DATA_DIR
                PG_DATA_DIR = os.environ["PG_DATA_DIR"]
                assert PG_DATA_DIR != ""
                fn = PG_DATA_DIR + "/cur_cardinalities.json"
                with open(fn, 'w') as f:
                    json.dump(updated_cards, f)

            if explain:
                sql = "EXPLAIN (ANALYZE, COSTS, TIMING, FORMAT JSON) " + sql
                ## for debugging:
                # sql = "EXPLAIN (COSTS, FORMAT JSON) " + sql
            try:
                output, exec_time = benchmark_sql(sql, args.user, args.db_host,
                        args.port, args.pwd, args.db_name, join_collapse_limit)
            except Exception as e:
                print("{} failed to execute ".format(query.template_name))
                print(e)
                exit(-1)

            print("iter: {}, {}: alg: {}, time: {}".format(i,
                query.template_name, alg, exec_time))
            query.runtimes[rt_alg].append(exec_time)
            if explain:
                query.explains[rt_alg].append(output)

def update_runtimes_old(query, explain, use_orig_query=False,
        patch_pg_cards=True):

    if not hasattr(query, "runtimes"):
        query.runtimes = defaultdict(list)
    if explain and not hasattr(query, "explains"):
        query.explains = defaultdict(list)

    for alg, sqls in query.executed_sqls.items():
        for i in range(args.runtime_reps):
            print(alg)
            if patch_pg_cards:
                updated_cards = get_cardinalities(query, alg)
                # pdb.set_trace()
                # update the cardinalities file in $PG_DATA_DIR
                PG_DATA_DIR = os.environ["PG_DATA_DIR"]
                assert PG_DATA_DIR != ""
                fn = PG_DATA_DIR + "/cur_cardinalities.json"
                with open(fn, 'w') as f:
                    json.dump(updated_cards, f)
                print("updated cardinalities")

            # FIXME: all sqls should be same, ensure that?
            assert len(sqls) == 1
            sql = list(sqls)[0]

            # FIXME: this only has effect if the join collapse limit paramter
            # has not been set, ideally, that should be handled here too.
            if use_orig_query:
                sql = query.query
                rt_alg = "pg_" + alg
                join_collapse_limit = 15
            else:
                # postgres should not be allowed to set join collapse limit
                join_collapse_limit = 1
                rt_alg = alg

            if len(query.runtimes[rt_alg]) > i:
                print("already have runtimes for: ", query.template_name)
                continue

            if explain:
                sql = "EXPLAIN (ANALYZE, COSTS, TIMING, FORMAT JSON) " + sql
                # sql = "EXPLAIN (COSTS, FORMAT JSON) " + sql

            try:
                output, exec_time = benchmark_sql(sql, args.user, args.db_host,
                        args.port, args.pwd, args.db_name, join_collapse_limit)
            except Exception as e:
                print("{} failed to execute ".format(query.template_name))
                print(e)
                exit(-1)

            print("iter: {}, {}: alg: {}, time: {}".format(i,
                query.template_name, alg, exec_time))
            query.runtimes[rt_alg].append(exec_time)
            if explain:
                query.explains[rt_alg].append(output)

def fix_query_structure(query):
    '''
    TODO: ideally, this structure for Query objects should be created from the
    start itself.
    '''
    if not hasattr(query, "executed_sqls"):
        query.executed_sqls = defaultdict(set)
    if not hasattr(query, "costs"):
        query.costs = {}

    for alg in query.join_info:
        try:
            exec_sql = query.join_info[alg]["executedSqls"]["RL"]
            assert exec_sql != ""
            query.executed_sqls[alg].add(exec_sql)
        except:
            pass

        cost = query.join_info[alg]["costs"]["RL"]
        query.costs[alg] = cost

        try:
            exec_sql = query.join_info[alg]["executedSqls"][BASELINE]
            assert exec_sql != ""
            query.executed_sqls["true"].add(exec_sql)
        except:
            pass

        cost = query.join_info[alg]["costs"][BASELINE]
        query.costs["true"] = cost

    # add true losses
    for alg, loss_types in query.losses.items():
        break
    loss_types = [k for k in loss_types]

    for lt in loss_types:
        if lt == "qerr":
            min_loss = 1.00
        else:
            min_loss = 0.00

        query.losses["true"][lt] = min_loss

def print_runtime_summary(results_cache):
    rts = {}
    for k, results in results_cache.items():
        queries = results["training_queries"]
        for i, q in enumerate(queries):
            for alg, rts in q.runtimes.items():
                for rt in rts:
                    rts[alg].append(rt)

    for k,v in rts.items():
        print(k, np.mean(np.array(v)))

def parse_query_objs(results_cache, trainining_queries=True):
    '''
    '''
    query_data = defaultdict(list)
    data = defaultdict(list)

    for k, results in results_cache.items():
        print(k)
        # FIXME: simplify this
        if "args" in results:
            result_args = results["args"]
            # filter out stuff based on args
            if args.db_name != result_args.db_name:
                print("skipping: ", result_args.db_name)
                continue
            if hasattr(result_args, "optimizer_name"):
                optimizer_name = result_args.optimizer_name
            else:
                optimizer_name = "unknown"

            if hasattr(result_args, "jl_start_iter"):
                jl_start_iter = result_args.jl_start_iter
            else:
                jl_start_iter = -1

        # else, just parse it

        if trainining_queries:
            queries = results["training_queries"]
        else:
            queries = results["test_queries"]

        if DEBUG:
            # QUERY_LIST = ["29c.sql", "19d.sql", "30c.sql", "20a.sql"]
            # QUERY_LIST = ["30c.sql"]
            RT_EPS = 3.00
            QUERY_LIST = []
            PG_BETTER = []
            TRUE_BETTER = []

        for i, q in enumerate(queries):
            if OLD_QUERY:
                fix_query_structure(q)
            if i >= MAX_QUERY:
                break

            # just testing stuff
            update_runtimes(q, args.use_explain,
                    use_orig_query=args.use_orig_query)
            if args.use_explain:
                update_pg_costs(q)

            print("dumping results cache")
            results_cache.dump()

            if DEBUG:
                cur_query = q
                if len(QUERY_LIST) > 0:
                    cont = True
                    for debug_query_name in QUERY_LIST:
                        if debug_query_name in q.template_name:
                            cont = False
                    if cont:
                        continue

                for k,v in q.runtimes.items():
                    v = np.array(v)
                    # print(k, np.mean(v), np.var(v))

                pg_rts = q.runtimes["Postgres"]
                true_rts = q.runtimes["true"]
                pg_rt = np.mean(pg_rts)
                true_rt = np.mean(true_rts)
                if pg_rt - true_rt > RT_EPS:
                    print("true better: ", pg_rt - true_rt)
                    TRUE_BETTER.append(q)
                elif true_rt - pg_rt > RT_EPS:
                    print("pg better: ", true_rt - pg_rt)
                    PG_BETTER.append(q)

                for k,v in q.explains.items():
                    explain1 = v[0][0][0]
                    # TODO: fix this
                    G = explain_to_nx(explain1)

                    fn = "./explains/" + q.template_name[0:3] + str(k)
                    plot_graph_explain(G, G.base_table_nodes, G.join_nodes,
                            fn+".png", q.template_name[0:3] + str(k))
                    # save analyze plan:
                    explain_str = json.dumps(explain1)
                    with open(fn + ".json", "w") as f:
                        f.write(explain_str)

                    with open(fn + ".sql", "w") as f:
                        execs = [val for val in q.executed_sqls[k]]
                        assert len(execs) == 1
                        f.write(execs[0])


            # add runtime data to same df
            # selectivity prediction
            true_sel = q.true_sel
            query_data[q.template_name].append(q)

            # multiple runtimes, while there is only a single loss, so treating
            # them separately for now.
            for alg, rts in q.runtimes.items():
                if alg in q.costs:
                    cost = q.costs[alg]
                else:
                    cost = 0.00
                try:
                    pg_cost = q.pg_costs[alg]
                except:
                    pg_cost = 0.00

                for rt in rts:
                    add_data_row(data, alg, "runtime", rt, q.template_name,
                            true_sel, optimizer_name, jl_start_iter, q,
                            BASELINE, cost, pg_cost)

            for alg, loss_types in q.losses.items():
                cost = q.costs[alg]
                try:
                    pg_cost = q.pg_costs[alg]
                except:
                    pg_cost = -1.00
                for lt, loss in loss_types.items():
                    add_data_row(data, alg, lt, loss, q.template_name, true_sel,
                            optimizer_name, jl_start_iter, q, BASELINE, cost,
                            pg_cost)

    if DEBUG:
        pg_templates = [q.template_name for q in PG_BETTER]
        true_templates = [q.template_name for q in TRUE_BETTER]
        print(pg_templates)
        print(true_templates)

        pdb.set_trace()

    df = pd.DataFrame(data)

    return df, query_data

def gen_query_bar_graphs(df, pdf, sort_by_loss_type, sort_by_alg,
        alg_order):

    # first, only plot join losses
    sort_df = df[df["loss_type"] == sort_by_loss_type]
    sort_df = sort_df[sort_df["alg_name"] == sort_by_alg]
    sort_df = sort_df[sort_df["loss"] > 2.00]
    sort_df.sort_values("loss", ascending=False, inplace=True)
    templates = sort_df["template"].drop_duplicates()
    if len(templates) == 0:
        return

    templates = templates.values[0:15]

    ## this was done to plot multiple error bars for each loss type in same
    ## figure
    to_plot = df[df["template"].isin(templates)]

    fg = sns.catplot(x="loss_type", y="loss", hue="alg_name", col="template",
            col_wrap=5, kind="bar", data=to_plot, estimator=np.median, ci=100,
            legend_out=False, col_order=templates, sharex=False, order=["join",
                "qerr", "runtime"], hue_order=alg_order)


    for i, ax in enumerate(fg.axes.flat):
        tmp = templates[i]
        sqs = sort_df[sort_df["template"] == tmp]["num_subqueries"].values[0]
        title = tmp + " ,#subqueries: " + str(sqs)
        ax.set_title(title)

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
    def _gen_plot(x_axis, plot_type, style, hue):
        if plot_type == "scatter":
            ax = sns.scatterplot(x=x_axis, y="loss", hue=hue, style=style,
                    data=df, estimator=np.mean, ci=99)
        if plot_type == "reg":
            fg = sns.lmplot(x=x_axis, y="loss", hue=hue,
                    data=df, ci=5)
            # fg.set(yscale="log")

        elif plot_type == "line":
            ax = sns.lineplot(x=x_axis, y="loss", hue="alg_name",
                    data=df, estimator=np.mean, ci=99)

        # fig = ax.get_figure()
        # ax.set_yscale("log")
        # maxy = min(10**2, max(df["loss"]))
        # ax.set_ylim((0, maxy))
        # ax.set_ylabel("seconds")

        plt.title("Cost Model Output v/s Runtime")
        plt.tight_layout()
        pdf.savefig()
        plt.clf()

    # select only runtime rows
    df = df[df["loss_type"] == "runtime"]
    _gen_plot("cost", "reg", "alg_name", "alg_name")

    # _gen_plot("pg_cost", "scatter")
    # _gen_plot("cost", "line")
    # _gen_plot("pg_cost", "scatter")
    # _gen_plot("pg_cost", "line")

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
        txt = qname
        firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
        pdf.savefig()
        plt.close()

        firstPage = plt.figure()
        firstPage.clf()

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
    results_cache.dump()

    summary_pdf = PdfPages(args.results_dir + "/summary.pdf")
    make_dir(args.output_dir)

    # take intersection of the algs below, and the algs in train_df
    # algs = ["nn", "nn-jl1", "nn-jl2", "Postgres", "ourpgm", "greg", "chow-liu",
            # "true"]
    # train_df = train_df[train_df["alg_name"].isin(algs)]
    algs = [alg for alg in set(train_df["alg_name"])]

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
            gen_query_bar_graphs(train_df, summary_pdf, "runtime", alg, algs)
            gen_query_bar_graphs(train_df, summary_pdf, "qerr", alg, algs)

    summary_pdf.close()

    if args.per_query:
        # FIXME: fix this
        queries_pdf = PdfPages(args.results_dir + "/training_queries.pdf")
        plot_queries(query_data, queries_pdf)
        queries_pdf.close()

args = read_flags()
main()

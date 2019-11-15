import argparse
from park.param import parser
import psycopg2 as pg
# from db_utils.utils import *
from sql_rep.utils import *
# from utils.utils import *
import pdb
import random
import klepto
from multiprocessing import Pool
import multiprocessing
import toml
from db_utils.query_storage import *
from utils.utils import *
import json
import pickle

def remove_doubles(query_strs):
    print("remove_doubles")
    newq = []
    seen_samples = set()
    for q in query_strs:
        if q in seen_samples:
            print(q)
            # pdb.set_trace()
            continue
        seen_samples.add(q)
        newq.append(q)
    return newq

def convert_to_sql_rep(query, subqueries):

    start = time.time()
    sql = query.query
    join_graph = extract_join_graph(sql)
    for node in join_graph:
        if "predicates" not in join_graph.nodes[node]:
            join_graph.nodes[node]["pred_cols"] = []
            join_graph.nodes[node]["pred_types"] = []
            join_graph.nodes[node]["pred_vals"] = []
            continue
        subg = join_graph.subgraph(node)
        node_sql = nx_graph_to_query(subg)
        pred_cols, pred_types, pred_vals = extract_predicates(node_sql)
        join_graph.nodes[node]["pred_cols"] = pred_cols
        join_graph.nodes[node]["pred_types"] = pred_types
        join_graph.nodes[node]["pred_vals"] = pred_vals

    subset_graph = generate_subset_graph(join_graph)

    print("query has",
          len(join_graph.nodes), "relations,",
          len(join_graph.edges), "joins, and",
          len(subset_graph), " possible subsets.",
          "took:", time.time() - start)

    ret = {}
    ret["sql"] = sql
    ret["join_graph"] = join_graph
    ret["subset_graph"] = subset_graph

    # need to fill in the ground truth data
    for sq in subqueries:
        aliases = [k for k in sq.aliases]
        if len(aliases) == 1:
            aliases_key = tuple(sorted(aliases))
            # subset_graph.nodes.add(aliases_key)
            subset_graph.add_node(aliases_key)
        else:
            aliases_key = tuple(sorted(aliases))
        assert aliases_key in subset_graph.nodes

        node_card = {}
        node_card["expected"] = sq.pg_count
        node_card["actual"] = sq.true_count
        node_card["total"] = sq.total_count
        subset_graph.nodes[aliases_key]["cardinality"] = node_card

    # json-ify the graphs
    ret["join_graph"] = nx.adjacency_data(ret["join_graph"])
    ret["subset_graph"] = nx.adjacency_data(ret["subset_graph"])
    return ret

def main():

    fns = list(glob.glob(args.template_dir+"/*"))
    for fn in fns:
        start = time.time()
        samples, all_subqueries = load_all_queries(args, fn, subqueries=True)
        print("{} took {} seconds to load data".format(fn, time.time()-start))
        num_processes = multiprocessing.cpu_count()
        with Pool(processes=num_processes) as pool:
            par_args = [(query, all_subqueries[i]) for
                i, query in enumerate(samples)]
            conv_queries = pool.starmap(convert_to_sql_rep, par_args)

        # conv_queries = []
        # for i, query in enumerate(samples):
            # assert len(all_subqueries[i]) > 0
            # conv_queries.append(convert_to_sql_rep(query, all_subqueries[i]))

        # save all of these
        dir_name = "./queries/" + os.path.basename(fn) + "/"
        print("going to write queries to ", dir_name)
        make_dir(dir_name)
        for i, convq in enumerate(conv_queries):
            qfn = dir_name + str(i) + ".pkl"
            # TODO: combine this with gzip as well?
            with open(qfn, 'wb') as fp:
                pickle.dump(convq, fp, protocol=pickle.HIGHEST_PROTOCOL)


def read_flags():
    # parser = argparse.ArgumentParser()
    parser.add_argument("--results_cache", type=str, required=False,
            default=None)
    parser.add_argument("--num_tables_model", type=str, required=False,
            default="nn")
    parser.add_argument("--reuse_env", type=int, required=False,
            default=1)
    parser.add_argument("--group_models", type=int, required=False,
            default=0)
    parser.add_argument("--gen_queries", type=int, required=False,
            default=0)
    parser.add_argument("--update_subq_cards", type=int, required=False,
            default=0)
    parser.add_argument("--update_subq_preds", type=int, required=False,
            default=0)
    parser.add_argument("--eval_num_tables", type=int, required=False,
            default=0)
    parser.add_argument("--rf_trees", type=int, required=False,
            default=128)
    parser.add_argument("--exp_name", type=str, required=False,
            default="card_exp")

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--template_dir", type=str, required=False,
            default=None)
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/cards/")
    parser.add_argument("-n", "--num_samples_per_template", type=int,
            required=False, default=1000)
    parser.add_argument("--max_iter", type=int,
            required=False, default=100000)
    parser.add_argument("--jl_start_iter", type=int,
            required=False, default=200)
    parser.add_argument("--eval_iter", type=int,
            required=False, default=1000)
    parser.add_argument("--eval_iter_jl", type=int,
            required=False, default=5000)
    parser.add_argument("--lr", type=float,
            required=False, default=0.001)
    parser.add_argument("--clip_gradient", type=float,
            required=False, default=10.0)
    parser.add_argument("--rel_qerr_loss", type=int,
            required=False, default=0)
    parser.add_argument("--rel_jloss", type=int,
            required=False, default=0)
    parser.add_argument("--eval_test_while_training", type=int,
            required=False, default=1)
    parser.add_argument("--jl_use_postgres", type=int,
            required=False, default=0)

    parser.add_argument("--adaptive_lr", type=int,
            required=False, default=1)
    parser.add_argument("--viz_join_plans", type=int,
            required=False, default=0)
    parser.add_argument("--viz_fn", type=str,
            required=False, default="./test")

    parser.add_argument("--nn_cache_dir", type=str, required=False,
            default="./nn_training_cache")
    parser.add_argument("--divide_mb_len", type=int, required=False,
            default=0)

    parser.add_argument("--optimizer_name", type=str, required=False,
            default="sgd")
    parser.add_argument("--net_name", type=str, required=False,
            default="FCNN")

    parser.add_argument("--num_hidden_layers", type=int,
            required=False, default=1)
    parser.add_argument("--hidden_layer_multiple", type=float,
            required=False, default=0.5)

    # synthetic data flags
    parser.add_argument("--gen_synth_data", type=int, required=False,
            default=0)
    parser.add_argument("--gen_bn_dist", type=int, required=False,
            default=0)
    parser.add_argument("--compute_runtime", type=int, required=False,
            default=0)
    parser.add_argument("--only_nonzero_samples", type=int, required=False,
            default=1)
    parser.add_argument("--use_subqueries", type=int, required=False,
            default=0)
    parser.add_argument("--synth_table", type=str, required=False,
            default="test")
    parser.add_argument("--synth_num_columns", type=int, required=False,
            default=2)
    parser.add_argument('--min_corr', help='delimited list correlations',
            type=float, required=False, default=0.0)
    parser.add_argument('--synth_period_len', help='delimited list correlations',
            type=int, required=False, default=10)
    parser.add_argument('--synth_num_vals', help='delimited list correlations',
            type=int, required=False, default=100000)
    parser.add_argument("--random_seed", type=int, required=False,
            default=2112)
    parser.add_argument("--test", type=int, required=False,
            default=1)
    parser.add_argument("--avg_factor", type=int, required=False,
            default=1)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    parser.add_argument("--losses", type=str, required=False,
            default="abs,rel,qerr", help="comma separated list of loss names")
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results/")
    parser.add_argument("--baseline_join_alg", type=str, required=False,
            default="EXHAUSTIVE")
    parser.add_argument("--db_file_name", type=str, required=False,
            default=None)
    parser.add_argument("--cache_dir", type=str, required=False,
            default="./caches/")
    parser.add_argument("--save_cur_cache_dir", type=str, required=False,
            default=None)
    parser.add_argument("--execution_cache_threshold", type=int, required=False,
            default=20)
    parser.add_argument("--jl_variant", type=int, required=False,
            default=0)
    parser.add_argument("--sampling", type=str, required=False,
            default="subquery", help="weighted_query: reprioritize, subquery: uniform \
            over all queries")
    parser.add_argument("--sampling_priority_method", type=str, required=False,
            default="jl_ratio", help="jl_ratio OR jl_diff or jl_rank")
    parser.add_argument("--sampling_priority_alpha", type=float, required=False,
            default=5.00, help="")
    parser.add_argument("--adaptive_priority_alpha", type=int, required=False,
            default=0)

    parser.add_argument("--loss_func", type=str, required=False,
            default="qloss")

    ## pgm flags
    parser.add_argument("--pgm_backend", type=str, required=False,
            default="ourpgm")
    parser.add_argument("--pgm_alg_name", type=str, required=False,
            default="chow-liu")
    parser.add_argument("--sampling_percentage", type=float, required=False,
            default=0.001)
    parser.add_argument("--use_svd", type=int, required=False,
            default=0)
    parser.add_argument("--num_singular_vals", type=int, required=False,
            default=5, help="-1 means all")
    parser.add_argument("--num_bins", type=int, required=False,
            default=100)
    parser.add_argument("--cl_recompute", type=int, required=False,
            default=0)
    parser.add_argument("--pgm_sampling_percentage", type=float, required=False,
            default=0.001)
    parser.add_argument("--pgm_merge_aliases", type=int, required=False,
            default=0)

    return parser.parse_args()

args = read_flags()
main()

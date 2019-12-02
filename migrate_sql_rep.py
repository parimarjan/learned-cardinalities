import argparse
import psycopg2 as pg
# from db_utils.utils import *
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
from sql_rep.query import parse_sql
from sql_rep.utils import *

def remove_doubles(query_strs):
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
        # node_sql = nx_graph_to_query(subg)
        for subq in subqueries:
            subq_alias = [a for a in subq.aliases][0]
            if subq_alias == node:
                join_graph.nodes[node]["pred_cols"] = subq.pred_column_names
                join_graph.nodes[node]["pred_types"] = subq.cmp_ops
                join_graph.nodes[node]["pred_vals"] = subq.vals
                break

        assert join_graph.nodes[node]["pred_cols"] is not None

    subset_graph = generate_subset_graph(join_graph)

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
        sqrep_queries = []
        if args.gen_sql_rep_queries:
            assert ".toml" in fn
            template = toml.load(fn)
            num_processes = multiprocessing.cpu_count()
            num_processes = min(num_processes, args.num_samples_per_template)
            num_processes = max(4, num_processes)
            num_per_p = int(args.num_samples_per_template / num_processes)

            query_strs = []
            with Pool(processes=num_processes) as pool:
                par_args = [(template, num_per_p, args)
                        for _ in range(num_processes)]
                comb_query_strs = pool.starmap(gen_queries, par_args)
            # need to flatten_the list
            for cqueries in comb_query_strs:
                query_strs += cqueries
            print("generated {} query sqls".format(len(query_strs)))

            # num_processes = int(num_processes / 2) + 1
            num_processes = 1
            with Pool(processes=num_processes) as pool:
                par_args = [(query, args.user, args.db_name, args.db_host,
                    args.port, args.pwd, False, True) for query in query_strs]
                sqrep_queries = pool.starmap(parse_sql, par_args)
        else:
            samples, all_subqueries = load_all_queries(args, fn, subqueries=True)
            print("{} took {} seconds to load data".format(fn, time.time()-start))
            # just fixes the missing fields etc. (like subq.aliases,
            # pred_column_names) so that we can convert it to the sql_rep
            # format
            update_subq_preds(samples, all_subqueries, args.cache_dir)
            print("update_subq_preds done")

            num_processes = multiprocessing.cpu_count()
            with Pool(processes=num_processes) as pool:
                par_args = [(query, all_subqueries[i]) for
                    i, query in enumerate(samples)]
                sqrep_queries = pool.starmap(convert_to_sql_rep, par_args)

        # save all of these
        dir_name = "./queries/" + os.path.basename(fn) + "/"
        print("going to write queries to ", dir_name)
        make_dir(dir_name)
        for i, convq in enumerate(sqrep_queries):
            query_hash = deterministic_hash(convq["sql"])
            qfn = dir_name + str(query_hash) + ".pkl"
            # TODO: combine this with gzip as well?
            with open(qfn, 'wb') as fp:
                pickle.dump(convq, fp, protocol=pickle.HIGHEST_PROTOCOL)

def read_flags():
    # FIXME: simplify this stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_queries", type=int, required=False,
            default=0)
    parser.add_argument("--gen_sql_rep_queries", type=int, required=False,
            default=0)
    parser.add_argument("--update_subq_cards", type=int, required=False,
            default=0)
    parser.add_argument("--update_subq_preds", type=int, required=False,
            default=0)


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

    # synthetic data flags
    parser.add_argument("--only_nonzero_samples", type=int, required=False,
            default=1)
    parser.add_argument("--use_subqueries", type=int, required=False,
            default=0)
    parser.add_argument("--random_seed", type=int, required=False,
            default=2112)
    parser.add_argument("--db_file_name", type=str, required=False,
            default=None)
    parser.add_argument("--cache_dir", type=str, required=False,
            default="./caches/")
    parser.add_argument("--save_cur_cache_dir", type=str, required=False,
            default=None)
    parser.add_argument("--execution_cache_threshold", type=int, required=False,
            default=20)

    return parser.parse_args()

args = read_flags()
main()

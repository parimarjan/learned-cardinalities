import sys
sys.path.append(".")
import argparse
import psycopg2 as pg
# from db_utils.utils import *
from utils.utils import *
from db_utils.query_storage import *
from utils.utils import *
import pdb
import random
import klepto
from multiprocessing import Pool, cpu_count
import json
import pickle
from sql_rep.utils import execute_query
from networkx.readwrite import json_graph
# from progressbar import progressbar as bar

TIMEOUT_COUNT_CONSTANT = 150001001
CACHE_TIMEOUT = 4
CACHE_CARD_TYPES = ["actual"]

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--card_cache_dir", type=str, required=False,
            default="./cardinality_cache")
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_queries", type=int,
            required=False, default=-1)
    parser.add_argument("--card_type", type=str, required=False,
            default=None)
    parser.add_argument("--key_name", type=str, required=False,
            default=None)
    parser.add_argument("--true_timeout", type=int,
            required=False, default=1800000)
    parser.add_argument("--pg_total", type=int,
            required=False, default=1)

    return parser.parse_args()

def update_bad_qrep(qrep):
    from sql_rep.query import parse_sql
    qrep = parse_sql(qrep["sql"], None, None, None, None, None,
            compute_ground_truth=False)
    qrep["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
    return qrep

def get_cardinality(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, true_timeout, pg_total, cache_dir, fn, idx):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    if key_name is None:
        key_name = card_type
    if idx % 10 == 0:
        print("query: ", idx)

    # load the cache for few types
    if card_type in CACHE_CARD_TYPES:
        sql_cache = klepto.archives.dir_archive(cache_dir,
                cached=True, serialized=True)
    found_in_cache = 0
    existing = 0
    num_timeout = 0

    for subset, info in qrep["subset_graph"].nodes().items():
        if "cardinality" not in info:
            info["cardinality"] = {}

        cards = info["cardinality"]
        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)

        if key_name in cards:
            existing += 1
            continue

        if card_type == "pg":
            subsql = "EXPLAIN " + subsql
            output = execute_query(subsql, user, db_host, port, pwd, db_name, [])
            card = pg_est_from_explain(output)

        elif card_type == "actual":
            hash_sql = deterministic_hash(subsql)
            if "count" not in subsql.lower():
                print("cardinality query does not have count")
                pdb.set_trace()
            if hash_sql in sql_cache.archive:
                card = sql_cache.archive[hash_sql]
                found_in_cache += 1
            else:
                start = time.time()
                pre_execs = ["SET statement_timeout = {}".format(true_timeout)]
                output = execute_query(subsql, user, db_host, port, pwd, db_name,
                                pre_execs)
                if isinstance(output, Exception):
                    print(output)
                    pdb.set_trace()
                elif output == "timeout":
                    print("timeout query: ")
                    print(subsql)
                    card = TIMEOUT_COUNT_CONSTANT
                else:
                    card = output[0][0]
                exec_time = time.time() - start
                if exec_time > CACHE_TIMEOUT:
                    print(exec_time)
                    num_timeout += 1
                    sql_cache.archive[hash_sql] = card

        elif card_type == "wanderjoin":
            assert False

        elif card_type == "total":
            exec_sql = get_total_count_query(subsql)
            if args.pg_total:
                exec_sql = "EXPLAIN " + exec_sql
            print("should handle the case where total < true")
            pdb.set_trace()
        else:
            assert False

        cards[key_name] = card
    if card_type == "actual":
        print("timeout: {}, existing: {}, found in cache: {}".format(\
                num_timeout, existing, found_in_cache))

    if fn is not None:
        save_sql_rep(fn, qrep)
    return qrep

def main():
    fns = list(glob.glob(args.query_dir + "/*"))
    par_args = []
    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        qrep = load_sql_rep(fn)
        par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                args.db_name, args.user, args.pwd, args.port,
                args.true_timeout, args.pg_total, args.card_cache_dir, fn, i))

        # TO debug:
        # get_cardinality(qrep, args.card_type, args.key_name, args.db_host,
                # args.db_name, args.user, args.pwd, args.port,
                # args.true_timeout, args.pg_total, args.card_cache_dir, fn, i)
        # pdb.set_trace()

    print("going to get cardinalities for {} queries".format(len(par_args)))
    start = time.time()
    num_proc = cpu_count()
    with Pool(processes = num_proc) as pool:
        qreps = pool.starmap(get_cardinality, par_args)
    print("updated all cardinalities in {} seconds".format(time.time()-start))

args = read_flags()
main()

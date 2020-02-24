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

# TIMEOUT_COUNT_CONSTANT = 150001001
# CROSS_JOIN_CONSTANT = 150001000
TIMEOUT_COUNT_CONSTANT = 15000100001
CROSS_JOIN_CONSTANT = 15000100000
EXCEPTION_COUNT_CONSTANT = 15000100002

CACHE_TIMEOUT = 4

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
            default="actual")
    parser.add_argument("--true_timeout", type=int,
            required=False, default=3600000)
    parser.add_argument("--pg_total", type=int,
            required=False, default=1)

    return parser.parse_args()

def update_bad_qrep(qrep):
    from sql_rep.query import parse_sql
    qrep = parse_sql(qrep["sql"], None, None, None, None, None,
            compute_ground_truth=False)
    qrep["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["join_graph"]))
    return qrep

def main():
    fns = list(glob.glob(args.query_dir + "/*"))
    qreps = []
    actual = 0
    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        qrep = load_sql_rep(fn)
        total_cards = len(qrep["subset_graph"].nodes())
        for subset, info in qrep["subset_graph"].nodes().items():
            if "cardinality" not in info:
                info["cardinality"] = {}
            cards = info["cardinality"]
            if args.key_name in cards:
                actual += 1
            break

            # print(cards)
    print("total: {}, subq: {}, actual: {}".format(len(fns), len(qrep["subset_graph"].nodes()),
        actual))

    # total_qs = len(fns)
    total = 0
    timeouts = 0
    cjs = 0
    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        qrep = load_sql_rep(fn)
        for subset, info in qrep["subset_graph"].nodes().items():
            if "cardinality" not in info:
                info["cardinality"] = {}
            cards = info["cardinality"]
            if args.key_name in cards:
                total += 1
                card = cards[args.key_name]
                if card == TIMEOUT_COUNT_CONSTANT:
                    timeouts += 1
                elif card == CROSS_JOIN_CONSTANT:
                    cjs += 1
                elif card == EXCEPTION_COUNT_CONSTANT:
                    timeouts += 1

    bad_percentage = float(timeouts + cjs) / total
    print("cjs: {}, timeout percentage: {}".format(float(cjs)/total, bad_percentage))

    # let us save them all
    # for i, _ in enumerate(qreps):
        # qrep = qreps[i]
        # fn = fns[i]
        # save the updated file
        # save_sql_rep(fn, qrep)

args = read_flags()
main()

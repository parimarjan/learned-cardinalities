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

OLD_TIMEOUT_COUNT_CONSTANT = 1500010001
OLD_CROSS_JOIN_CONSTANT = 1500010000
OLD_EXCEPTION_COUNT_CONSTANT = 1500010002

TIMEOUT_COUNT_CONSTANT = 150001000001
CROSS_JOIN_CONSTANT = 150001000000
EXCEPTION_COUNT_CONSTANT = 150001000002

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

    total = 0
    timeouts = 0
    cjs = 0
    over_timeout = 0
    mapping = {}
    mapping[tuple("s")] = tuple(["s2"])

    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        qrep = load_sql_rep(fn)
        qreps.append(qrep)
        qrep["subset_graph"] = nx.relabel_nodes(qrep["subset_graph"], mapping)
        qrep["join_graph"] = nx.relabel_nodes(qrep["subset_graph"], mapping)
        save_sql_rep(fn, qrep)

    # let us save them all
    # for i, _ in enumerate(qreps):
        # qrep = qreps[i]
        # fn = fns[i]
        # # save the updated file

args = read_flags()
main()

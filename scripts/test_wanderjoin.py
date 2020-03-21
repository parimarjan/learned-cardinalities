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
from sql_rep.query import parse_sql
from networkx.readwrite import json_graph
import re
# from progressbar import progressbar as bar

from wanderjoin import WanderJoin
# from sql_rep.query import parse_sql

TIMEOUT_COUNT_CONSTANT = 150001000001
CROSS_JOIN_CONSTANT = 150001000000
EXCEPTION_COUNT_CONSTANT = 150001000002

CACHE_TIMEOUT = 4
CACHE_CARD_TYPES = ["actual"]
WANDERJOIN_TIME_FMT = " WITHTIME {TIME} CONFIDENCE {CONF} REPORTINTERVAL {INT}"

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
    parser.add_argument("--walks_timeout", type=float, required=False,
            default=0.5)
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
    parser.add_argument("--num_proc", type=int,
            required=False, default=-1)
    parser.add_argument("--sampling_percentage", type=int,
            required=False, default=None)
    parser.add_argument("--sampling_type", type=str,
            required=False, default=None)

    return parser.parse_args()

def main():
    # fns = list(glob.glob(args.query_dir + "/*"))
    fns = list(glob.glob(args.query_dir + "/*"))
    wj = WanderJoin(args.user, args.pwd, args.db_host, args.port,
            args.db_name, cache_dir="./debug_cache", verbose=True,
            walks_timeout=args.walks_timeout, use_tries=True)

    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        qrep = load_sql_rep(fn)

        # with open(fn, "r") as f:
            # sql = f.read()
        # qrep = parse_sql(sql, None, None, None, None, None,
                # compute_ground_truth=False)
        # qrep["subset_graph"] = \
                # nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
        # qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])

        wj.get_counts(qrep)
        pdb.set_trace()

args = read_flags()
main()

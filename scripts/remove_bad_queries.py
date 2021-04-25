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
# TIMEOUT_COUNT_CONSTANT = 15000100001
# CROSS_JOIN_CONSTANT = 15000100000
# EXCEPTION_COUNT_CONSTANT = 15000100002

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
    # parser.add_argument("--card_cache_dir", type=str, required=False,
            # default="./cardinality_cache")
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--query_dir", type=str, required=False,
            default="./queries/imdb/")
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

def main():
    tfns = list(glob.glob(args.query_dir + "/*"))
    qreps = []
    actual = 0
    for qi,qdir in enumerate(tfns):
        # let's first select all the qfns we are going to load
        fns = list(glob.glob(qdir+"/*.pkl"))
        for i, fn in enumerate(fns):
            if ".pkl" not in fn:
                continue
            if i >= args.num_queries and args.num_queries != -1:
                break
            qrep = load_sql_rep(fn)
            bad = False
            total_cards = len(qrep["subset_graph"].nodes())
            for subset, info in qrep["subset_graph"].nodes().items():
                if subset == SOURCE_NODE:
                    continue
                if "cardinality" not in info:
                    info["cardinality"] = {}
                    bad = True
                    continue

                cards = info["cardinality"]
                if "actual" not in cards:
                    bad = True
                    continue
                if "expected" not in cards:
                    bad = True
                    continue
                if cards["actual"] == 0:
                    bad = True
                    continue
                if cards["actual"] == 1.1:
                    bad = True
                    continue
            if bad:
                # delete path
                os.remove(fn)
                continue

            # TODO: timeouts
            if SOURCE_NODE in list(qrep["subset_graph"].nodes()):
                qrep["subset_graph"].remove_node(SOURCE_NODE)
                save_sql_rep(fn, qrep)

args = read_flags()
main()

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
    noactuals = 0

    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        print(i)
        if "pkl" not in fn:
            qreps.append(None)
            continue

        qrep = load_sql_rep(fn)

        # remove node!
        if SOURCE_NODE in qrep["subset_graph"].nodes():
            qrep["subset_graph"].remove_node(SOURCE_NODE)
        SOURCE2 = ("SOURCE",)

        if SOURCE2 in qrep["subset_graph"].nodes():
            qrep["subset_graph"].remove_node(SOURCE2)

        for node in qrep["subset_graph"].nodes():
            keys = list(qrep["subset_graph"].nodes()[node].keys())
            for k in keys:
                if k not in ["cardinality", "sample_bitmap"]:
                    del qrep["subset_graph"].nodes()[node][k]

        qreps.append(qrep)
        for subset, info in qrep["subset_graph"].nodes().items():
            if "cardinality" not in info:
                info["cardinality"] = {}
            cards = info["cardinality"]

            # handle actuals
            if not "actual" in cards:
                # pdb.set_trace()
                noactuals += 1
                qreps[-1] = None
                os.remove(fn)
                break
            else:
                actual_card = cards["actual"]
                # total_card = cards["total"]

                if actual_card == OLD_CROSS_JOIN_CONSTANT:
                    actual_card = CROSS_JOIN_CONSTANT
                elif actual_card == OLD_TIMEOUT_COUNT_CONSTANT:
                    actual_card = TIMEOUT_COUNT_CONSTANT
                elif actual_card == OLD_EXCEPTION_COUNT_CONSTANT:
                    actual_card = EXCEPTION_COUNT_CONSTANT

                if actual_card != cards["actual"]:
                    print("updating actual card!")
                    cards["actual"] = actual_card

                # if total_card < actual_card:
                    # cards["total"] = actual_card

                # total += 1
                card = cards["actual"]
                if card > TIMEOUT_COUNT_CONSTANT + 1:
                    print(card / float(TIMEOUT_COUNT_CONSTANT))
                    assert card / float(TIMEOUT_COUNT_CONSTANT) <= 100
                    # print(card)
                    over_timeout += 1

                if card == TIMEOUT_COUNT_CONSTANT:
                    timeouts += 1

                elif card == CROSS_JOIN_CONSTANT:
                    cjs += 1

    print("over timeout: ", over_timeout)
    print("timeout: ", timeouts + cjs)
    print("noactuals: ", noactuals)

    # let us save them all
    for i, _ in enumerate(qreps):
        qrep = qreps[i]
        if qrep is None:
            continue
        fn = fns[i]
        # save the updated file
        save_sql_rep(fn, qrep)

args = read_flags()
main()

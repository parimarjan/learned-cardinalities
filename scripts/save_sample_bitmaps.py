import sys
sys.path.append(".")

import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import argparse
import psycopg2 as pg
#from db_utils.utils import *
from utils.utils import *
from db_utils.query_storage import *
from utils.utils import *
import pdb
import random
import klepto
from multiprocessing import Pool, cpu_count
import json
import pickle
# from sql_rep.utils import execute_query
from networkx.readwrite import json_graph
import pickle
# from progressbar import progressbar as bar

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
    parser.add_argument("--bitmap_dir", type=str, required=False,
            default=None)
    parser.add_argument("--bitmap_type", type=str, required=False,
            default="join_bitmap")

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
    make_dir(args.bitmap_dir)

    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        # print(i)
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

        bitmaps = {}
        allzero = True

        for node in qrep["subset_graph"].nodes():
            if len(node) > 1:
                continue
            bitmaps[node] = {}
            data = qrep["subset_graph"].nodes()[node]
            # if len(node) == 1:
                # print(data.keys())

            if args.bitmap_type not in data:
                if len(node) == 1:
                    print("not in data")
                print("bitmap type not in data")
                # print(data.keys())
                # pdb.set_trace()
                continue

            if len(data[args.bitmap_type]) == 0:
                print("len 0")
                continue

            # print(len(data[args.bitmap_type]))

            for sk,bvals in data[args.bitmap_type].items():
                if len(bvals) != 0:
                    allzero = False
                bitmaps[node][sk] = set(bvals)
                print(node, sk, len(bvals))

        if allzero:
            print("allzero! ", i)
            print(fn)
            # print(qrep["name"])
            # continue
            # pdb.set_trace()

        fn = os.path.basename(fn)
        bitmapfn = os.path.join(args.bitmap_dir, fn)

        with open(bitmapfn, "wb") as handle:
            pickle.dump(bitmaps, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

args = read_flags()
main()

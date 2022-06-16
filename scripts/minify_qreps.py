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

EDGE_KEYS = ["id"]
NODE_KEYS = ["cardinality"]
CARD_KEYS = ["expected", "total", "actual"]

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("--output_dir", type=str, required=False,
            default=None)

    parser.add_argument("-n", "--num_queries", type=int,
            required=False, default=-1)

    return parser.parse_args()

def main():
    qdirs = list(glob.glob(args.query_dir + "/*"))
    make_dir(args.output_dir)


    for qdir in qdirs:
        qreps = []
        output_qdir = qdir.replace(args.query_dir, args.output_dir)
        make_dir(output_qdir)

        fns = list(glob.glob(qdir + "/*.pkl"))
        for i, fn in enumerate(fns):
            if i >= args.num_queries and args.num_queries != -1:
                break
            qrep = load_sql_rep(fn)

            for edge, info in qrep["subset_graph"].edges().items():
                # for k,v in info.items():
                to_del = []
                for k in info:
                    if k not in EDGE_KEYS:
                        to_del.append(k)

                for k in to_del:
                    del(info[k])
                assert len(info.keys()) == 1

            for subset, info in qrep["subset_graph"].nodes().items():
                to_del = []
                for k in info:
                    if k not in NODE_KEYS:
                        to_del.append(k)
                for k in to_del:
                    del(info[k])

                cards = info["cardinality"]
                to_del = []
                for k in cards:
                    if k not in CARD_KEYS:
                        to_del.append(k)
                for k in to_del:
                    del(cards[k])

            qreps.append(qrep)

        # let us save them all
        for i, _ in enumerate(qreps):
            qrep = qreps[i]
            fn = fns[i]
            output_fn = fn.replace(args.query_dir, args.output_dir)
            print(fn)
            print(output_fn)
            assert "our_dataset" not in output_fn
            # save the updated file
            save_sql_rep(output_fn, qrep)

args = read_flags()
main()

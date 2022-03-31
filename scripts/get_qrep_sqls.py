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

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("--output_query_dir", type=str, required=False,
            default="./sqls/")
    return parser.parse_args()

def main():
    dirs = list(glob.glob(args.query_dir + "/*"))
    make_dir(args.output_query_dir)
    qreps = []

    print(dirs)
    for dirname in dirs:
        cur_name = os.path.basename(dirname)
        cur_output_dir = os.path.join(args.output_query_dir, cur_name)
        make_dir(cur_output_dir)
        fns = list(glob.glob(dirname + "/*"))
        print(cur_name)
        for i, fn in enumerate(fns):
            if ".pkl" not in fn:
                continue
            if i % 100 == 0:
                print(i)
            qrep = load_sql_rep(fn)
            cur_fn_name = os.path.basename(fn).replace(".pkl", "")
            output_dir = os.path.join(cur_output_dir, cur_fn_name)
            make_dir(output_dir)

            nodes = list(qrep["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            nodes.sort()
            for j, node in enumerate(nodes):
                sg = qrep["join_graph"].subgraph(node)
                subsql = nx_graph_to_query(sg)

                out_fn = os.path.join(output_dir, str(j) + ".sql")
                with open(out_fn, "w") as f:
                    f.write(subsql)

args = read_flags()
main()

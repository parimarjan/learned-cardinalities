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
# from progressbar import progressbar as bar
import os

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("--unique_query_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_queries", type=int,
            required=False, default=-1)

    return parser.parse_args()

def main():
    fns = list(glob.glob(args.query_dir + "/*"))
    uniquefns = list(glob.glob(args.unique_query_dir + "/*"))
    uniquefns_base = [os.path.basename(f) for f in uniquefns]

    qreps = []

    saved = 0
    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        if "pkl" not in fn:
            continue
        basename = os.path.basename(fn)
        if basename not in uniquefns_base:
            os.remove(fn)
        else:
            saved += 1
    print(args.query_dir)
    print("saved: ", saved)

args = read_flags()
main()

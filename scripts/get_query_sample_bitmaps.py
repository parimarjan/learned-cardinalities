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
# from db_utils.utils import *
from utils.utils import *
from db_utils.query_storage import load_sql_rep, save_sql_rep,update_qrep
from utils.utils import *
from db_utils.utils import *
import pdb
import random
import klepto
from multiprocessing import Pool, cpu_count
import json
import pickle
from sql_rep.utils import execute_query
from networkx.readwrite import json_graph
import re
from sql_rep.query import parse_sql
from wanderjoin import WanderJoin
import math
# from progressbar import progressbar as bar
import scipy.stats as st

OLD_TIMEOUT_COUNT_CONSTANT = 150001001
OLD_CROSS_JOIN_CONSTANT = 150001000
OLD_EXCEPTION_COUNT_CONSTANT = 150001002

TIMEOUT_COUNT_CONSTANT = 150001000001
CROSS_JOIN_CONSTANT = 150001000000
EXCEPTION_COUNT_CONSTANT = 150001000002

CACHE_TIMEOUT = 4
CACHE_CARD_TYPES = ["actual"]

DEBUG_CHECK_TIMES = False
CONF_ALPHA = 0.99

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="postgres")
    parser.add_argument("--pwd", type=str, required=False,
            default="postgres")
    parser.add_argument("--card_cache_dir", type=str, required=False,
            default="./cardinality_cache")
    parser.add_argument("--port", type=str, required=False,
            default=5431)
    parser.add_argument("--wj_walk_timeout", type=float, required=False,
            default=0.5)
    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_queries", type=int,
            required=False, default=-1)
    parser.add_argument("--use_tries", type=int,
            required=False, default=0)
    parser.add_argument("--no_parallel", type=int,
            required=False, default=0)
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
    parser.add_argument("--seed", type=int,
            required=False, default=1234)
    parser.add_argument("--sample_num", type=int,
            required=False, default=1000)
    parser.add_argument("--sampling_type", type=str,
            required=False, default="sb")

    return parser.parse_args()

def update_bad_qrep(qrep):
    qrep = parse_sql(qrep["sql"], None, None, None, None, None,
            compute_ground_truth=False)
    qrep["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
    return qrep

def is_cross_join(sg):
    '''
    enforces the constraint that the graph should be connected w/o the site
    node.
    '''
    if len(sg.nodes()) < 2:
        # FIXME: should be return False
        return False
    sg2 = nx.Graph(sg)
    to_remove = []
    for node, data in sg2.nodes(data=True):
        if data["real_name"] == "site":
            to_remove.append(node)
    for node in to_remove:
        sg2.remove_node(node)
    if nx.is_connected(sg2):
        return False
    return True

def get_sample_bitmaps(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, fn, idx, sample_num, sampling_type):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    if "imdb_id" in qrep["sql"]:
        return

    if key_name is None:
        key_name = card_type

    if sample_num is not None:
        key_name = str(sampling_type) + str(sample_num)

    print(user, pwd)
    con = pg.connect(user=user, host=db_host, port=port,
            password=pwd, database=db_name)
    cursor = con.cursor()
    cursor.execute("SET enable_hashjoin=false")

    if idx % 10 == 0:
        print("query: ", idx)

    node_list = list(qrep["subset_graph"].nodes())
    if SOURCE_NODE in node_list:
        node_list.remove(SOURCE_NODE)
    node_list.sort(reverse=False, key = lambda x: len(x))

    for subqi, subset in enumerate(node_list):
        if len(subset) > 1:
            break

        info = qrep["subset_graph"].nodes()[subset]

        if "sample_bitmap" not in info:
            info["sample_bitmap"] = {}

        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)

        if "postHistory" in subsql:
            subsql = subsql.replace("postHistory", "posthistory")
        if "postLinks" in subsql:
            subsql = subsql.replace("postLinks", "postlinks")

        for k,v in sg.nodes(data=True):
            if "real_name" not in v:
                return
            table = v["real_name"]
            sample_table = table + "_" + sampling_type + str(sample_num)

        # if table not in SAMPLE_TABLES:
            # print(subsql)
            # print("continuing coz no sample table")
            # continue

        # subsql = subsql.replace("COUNT(*)", "id")
        # subsql = subsql.replace(table, sample_table, 1)

        # subsql = subsql.replace("COUNT(*)", "\"Id\"")
        # subsql = subsql.replace("COUNT(*)", "\"id\"")

        ### ergast-f1
        subsql = subsql.replace("COUNT(*)", "\"index\"")

        subsql = subsql.replace(table, "\"" + sample_table + "\"" , 1)

        try:
            cursor.execute(subsql)
            outputs = cursor.fetchall()
            bitmaps = [int(o[0]) for o in outputs]
            print(len(bitmaps))
        except Exception as e:
            print(subsql)
            print(table)
            print(subset)
            print(e)
            continue
            # pdb.set_trace()

        # if "where" not in subsql.lower():
            # print(subsql)
            # print(bitmaps)

        info["sample_bitmap"][key_name] = bitmaps

    if fn is not None:
        update_qrep(qrep)
        save_sql_rep(fn, qrep)
        print("saved new qrep!")

    con.close()
    cursor.close()

    return qrep

def main():
    print("query dir: ", args.query_dir)
    fns = list(glob.glob(args.query_dir + "/*"))
    print("number of files: ", len(fns))
    fns.sort()
    par_args = []
    for i, fn in enumerate(fns):
        # print(fn)
        if i >= args.num_queries and args.num_queries != -1:
            break

        if (".pkl" not in fn and ".sql" not in fn):
            continue

        if ".pkl" in fn:
            qrep = load_sql_rep(fn)
        else:
            assert False

        if args.no_parallel:
            get_sample_bitmaps(qrep, args.card_type, args.key_name, args.db_host,
                    args.db_name, args.user, args.pwd, args.port,fn,
                     i, args.sample_num, args.sampling_type)
            continue

        par_func = get_sample_bitmaps
        par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                args.db_name, args.user, args.pwd, args.port,
                fn, i,
                args.sample_num,
                args.sampling_type))

    assert not args.no_parallel
    start = time.time()
    if args.num_proc == -1:
        num_proc = cpu_count()
    else:
        num_proc = args.num_proc
    with Pool(processes = num_proc) as pool:
        qreps = pool.starmap(par_func, par_args)
    print("updated all cardinalities in {} seconds".format(time.time()-start))

args = read_flags()
main()

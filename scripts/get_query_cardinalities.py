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
import re
# from progressbar import progressbar as bar

TIMEOUT_COUNT_CONSTANT = 150001001
CROSS_JOIN_CONSTANT = 150001000
EXCEPTION_COUNT_CONSTANT = 150001002

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
    parser.add_argument("--wj_time", type=int, required=False,
            default=1000)
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

def update_bad_qrep(qrep):
    from sql_rep.query import parse_sql
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

def get_cardinality(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, true_timeout, pg_total, cache_dir, fn, wj_time, idx,
        sampling_percentage, sampling_type):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    if key_name is None:
        key_name = card_type

    if sampling_percentage is not None:
        key_name = str(sampling_type) + str(sampling_percentage) + "_" + key_name

        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)
        cursor = con.cursor()

    if idx % 10 == 0:
        print("query: ", idx)

    # load the cache for few types
    if card_type in CACHE_CARD_TYPES:
        sql_cache = klepto.archives.dir_archive(cache_dir,
                cached=True, serialized=True)
    found_in_cache = 0
    existing = 0
    num_timeout = 0
    site_cj = 0
    query_exec_times = []

    for subset, info in qrep["subset_graph"].nodes().items():
        if "cardinality" not in info:
            info["cardinality"] = {}
        if "exec_time" not in info:
            info["exec_time"] = {}

        cards = info["cardinality"]
        execs = info["exec_time"]
        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)

        if sampling_percentage is not None:
            table_names = []
            for k,v in sg.nodes(data=True):
                table = v["real_name"]
                new_table_name = table + "_" + sampling_type + str(sampling_percentage)
                new_table_name += " "
                new_table_name = " " + new_table_name
                # TODO: check if table exists...
                cursor.execute("select * from information_schema.tables where table_name='{}'".format(new_table_name))
                if bool(cursor.rowcount):
                    subsql = re.sub(r"\b {} \b".format(table), new_table_name,
                            subsql)

        if key_name in cards:
            if not (sampling_percentage is not None and \
                    cards[key_name] >= TIMEOUT_COUNT_CONSTANT):
                existing += 1
                print("key name already in cards")
                continue

        if card_type == "pg":
            subsql = "EXPLAIN " + subsql
            output = execute_query(subsql, user, db_host, port, pwd, db_name, [])
            card = pg_est_from_explain(output)
            cards[key_name] = card

        elif card_type == "actual":
            hash_sql = deterministic_hash(subsql)
            if "count" not in subsql.lower():
                print("cardinality query does not have count")
                pdb.set_trace()
            if is_cross_join(sg):
                site_cj += 1
                card = CROSS_JOIN_CONSTANT
                cards[key_name] = card
                continue
            if hash_sql in sql_cache.archive:
                card = sql_cache.archive[hash_sql]
                found_in_cache += 1
                cards[key_name] = card
                continue

            start = time.time()
            pre_execs = ["SET statement_timeout = {}".format(true_timeout)]
            output = execute_query(subsql, user, db_host, port, pwd, db_name,
                            pre_execs)
            if isinstance(output, Exception):
                print(output)
                card = EXCEPTION_COUNT_CONSTANT
                num_timeout += 1
                # continue
                # pdb.set_trace()
            elif output == "timeout":
                print("timeout query: ")
                print(subsql)
                card = TIMEOUT_COUNT_CONSTANT
                num_timeout += 1
            else:
                card = output[0][0]

            exec_time = time.time() - start
            if exec_time > CACHE_TIMEOUT:
                print(exec_time)
                sql_cache.archive[hash_sql] = card
            cards[key_name] = card
            execs[key_name] = exec_time
            query_exec_times.append(exec_time)

        elif card_type == "wanderjoin":
            assert "SELECT" in subsql
            subsql = subsql.replace("SELECT", "SELECT ONLINE")
            subsql = subsql.replace(";","")
            subsql += WANDERJOIN_TIME_FMT.format(
                    TIME = wj_time,
                    CONF = 95,
                    INT = 1000)
            print(subsql)
            output = execute_query(subsql, user, db_host, port, pwd, db_name,
                            [])
            print(output)
            pdb.set_trace()
            assert False

        elif card_type == "total":
            exec_sql = get_total_count_query(subsql)
            if args.pg_total:
                exec_sql = "EXPLAIN " + exec_sql

            output = execute_query(exec_sql, user, db_host, port, pwd, db_name, [])
            card = pg_est_from_explain(output)
            cards[key_name] = card
        else:
            assert False

    if card_type == "actual":
        print("total: {}, timeout: {}, existing: {}, found in cache: {}".format(\
                len(qrep["subset_graph"].nodes()), num_timeout, existing, found_in_cache))
        # print("site cj: ", site_cj)
        if len(query_exec_times) != 0:
            print("avg exec time: ", sum(query_exec_times) / len(query_exec_times))

    if fn is not None:
        update_qrep(qrep)
        save_sql_rep(fn, qrep)
    return qrep

def main():
    fns = list(glob.glob(args.query_dir + "/*"))
    par_args = []
    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break
        qrep = load_sql_rep(fn)
        par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                args.db_name, args.user, args.pwd, args.port,
                args.true_timeout, args.pg_total, args.card_cache_dir, fn,
                args.wj_time, i, args.sampling_percentage,
                args.sampling_type))

        # TO debug:
        # get_cardinality(qrep, args.card_type, args.key_name, args.db_host,
                # args.db_name, args.user, args.pwd, args.port,
                # args.true_timeout, args.pg_total, args.card_cache_dir, fn,
                # args.wj_time, i, args.sampling_percentage, args.sampling_type)
        # pdb.set_trace()

    print("going to get cardinalities for {} queries".format(len(par_args)))
    start = time.time()
    if args.num_proc == -1:
        num_proc = cpu_count()
    else:
        num_proc = args.num_proc
    with Pool(processes = num_proc) as pool:
        qreps = pool.starmap(get_cardinality, par_args)
    print("updated all cardinalities in {} seconds".format(time.time()-start))

args = read_flags()
main()

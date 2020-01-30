import sys
sys.path.append(".")
import argparse
import psycopg2 as pg
from db_utils.utils import *
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

TIMEOUT_COUNT_CONSTANT = 150001001

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
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_samples_per_template", type=int,
            required=False, default=-1)
    parser.add_argument("--card_type", type=str, required=False,
            default=None)
    parser.add_argument("--key_name", type=str, required=False,
            default=None)
    parser.add_argument("--true_timeout", type=int,
            required=False, default=3600000)
    parser.add_argument("--pg_total", type=int,
            required=False, default=1)

    return parser.parse_args()

def get_cardinality(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, true_timeout, pg_total):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    if key_name is None:
        key_name = card_type

    for subset, info in qrep["subset_graph"].nodes().items():
        cards = info["cardinality"]
        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)

        if card_type == "pg":
            subsql = "EXPLAIN " + subsql
            output = execute_query(subsql, user, db_host, port, pwd, db_name, [])
            card = pg_est_from_explain(output)

        elif card_type == "true":
            if "count" not in subsql.lower():
                print("not handling non-count queries yet..")
                pdb.set_trace()

            pre_execs = ["SET statement_timeout = {}".format(true_timeout)]
            output = execute_query(subsql, user, db_host, port, pwd, db_name,
                            pre_execs)
            if isinstance(output, Exception):
                print(output)
                pdb.set_trace()
            elif output == "timeout":
                card = TIMEOUT_COUNT_CONSTANT
            else:
                card = output[0][0]
            pdb.set_trace()

        elif card_type == "wanderjoin":
            assert False
        elif card_type == "total":
            exec_sql = get_total_count_query(subsql)
            if args.pg_total:
                exec_sql = "EXPLAIN " + exec_sql
            print("should handle the case where total < true")
            pdb.set_trace()
        else:
            assert False

        cards[key_name] = card
    print(cards)

def main():
    fns = list(glob.glob(args.query_dir + "/*"))
    num_proc = cpu_count()
    par_args = []
    for fn in fns:
        qrep = load_sql_rep(fn)
        par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                args.db_name, args.user, args.pwd, args.port,
                args.true_timeout, args.pg_total))
        # TO debug:
        # get_cardinality(qrep, args.card_type, args.key_name, args.db_host,
                # args.db_name, args.user, args.pwd, args.port,
                # args.true_timeout, args.pg_total)

    with Pool(processes = num_proc) as pool:
        pool.starmap(get_cardinality, par_args)
    pdb.set_trace()

args = read_flags()
main()

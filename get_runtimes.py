import pickle
import argparse
import glob
import pdb
import psycopg2 as pg
import time
import subprocess as sp
import os
import pandas as pd
from collections import defaultdict
from utils.utils import *
import sys

TIMEOUT_CONSTANT = 3609

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    return parser.parse_args()

def execute_sql(sql, template="sql"):
    '''
    '''
    drop_cache_cmd = "./drop_cache.sh > /dev/null"
    p = sp.Popen(drop_cache_cmd, shell=True)
    p.wait()
    # time.sleep(2)

    sql = sql.replace("explain (format json)", "explain (analyze,costs, format json)")
    # FIXME: generalize
    con = pg.connect(port=5432,dbname="imdb",
            user="ubuntu",password="",host="localhost")

    # TODO: clear cache

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(16))
    cursor.execute("SET join_collapse_limit = {}".format(1))
    cursor.execute("SET from_collapse_limit = {}".format(1))
    cursor.execute("SET statement_timeout = {}".format(3600000))

    start = time.time()

    try:
        cursor.execute(sql)
    except Exception as e:
        cursor.execute("ROLLBACK")
        con.commit()
        cursor.close()
        con.close()
        if not "timeout" in str(e):
            print("failed to execute for reason other than timeout")
            print(e)
            print(sql)
            return None, TIMEOUT_CONSTANT
        else:
            print("failed because of timeout!")
            return None, TIMEOUT_CONSTANT

    explain = cursor.fetchall()
    end = time.time()
    print("{} took {} seconds".format(template, end-start))
    sys.stdout.flush()

    return explain, end-start

def main():

    def add_runtime_row(sql_key, rt, exp_analyze):
        cur_runtimes["sql_key"].append(sql_key)
        cur_runtimes["runtime"].append(rt)
        cur_runtimes["exp_analyze"].append(exp_analyze)

    for alg_dir in os.listdir(args.results_dir):
        costs_fn = args.results_dir + "/" + alg_dir + "/" + "costs.pkl"
        costs = load_object(costs_fn)
        assert isinstance(costs, pd.DataFrame)
        rt_fn = args.results_dir + "/" + alg_dir + "/" + "runtimes.pkl"
        # go in order and execute runtimes...
        runtimes = load_object(rt_fn)
        if runtimes is None:
            columns = ["sql_key", "runtime","exp_analyze"]
            runtimes = pd.DataFrame(columns=columns)
        cur_runtimes = defaultdict(list)

        for i,row in costs.iterrows():
            if row["sql_key"] in runtimes["sql_key"].values:
                print("skipping {} with stored runtime".format(row["sql_key"]))
                continue
            if row["sql_key"] in cur_runtimes["sql_key"]:
                print("should never have repeated for execution")
                continue
            if "template" in row:
                exp_analyze, rt = execute_sql(row["exec_sql"], row["template"])
            else:
                exp_analyze, rt = execute_sql(row["exec_sql"])
            add_runtime_row(row["sql_key"], rt, exp_analyze)

            rts = cur_runtimes["runtime"]
            print("Alg:{}, N:{}, AvgRt: {}".format(alg_dir, len(rts),
                sum(rts) / len(rts)))
            df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
            save_object(rt_fn, df)

        df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
        save_object(rt_fn, df)
        print("saved df")

args = read_flags()
main()

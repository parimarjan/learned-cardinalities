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
# from utils.utils import *
import sys
#import pdb
# from cardinality_estimation.join_loss import set_cost_model
import MySQLdb

TIMEOUT_CONSTANT = 909
RERUN_TIMEOUTS = True
# TIMEOUT_VAL = 900000
# MYSQL_OPT_FLAGS=""""""

MYSQL_OPT_TMP = "set optimizer_switch='{FLAGS}';"
MYSQL_OPT_FLAGS="""materialization=off,block_nested_loop=off,semijoin=off,subquery_materialization_cost_based=off,index_merge_union=off,index_merge_sort_union=off,prefer_ordering_index=off,loosescan=off,firstmatch=off,use_index_extensions=off"""

def save_object(file_name, data):
    with open(file_name, "wb") as f:
        res = f.write(pickle.dumps(data))

def load_object(file_name):
    res = None
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            res = pickle.loads(f.read())
    return res

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--results_fn", type=str, required=False,
            default="cm1_mysql_jerr.pkl")
    parser.add_argument("--cost_model", type=str, required=False,
            default=None)
    parser.add_argument("--explain", type=int, required=False,
            default=0)
    parser.add_argument("--timeout", type=int, required=False,
            default=1800000)
    parser.add_argument("--materialize", type=int, required=False,
            default=0)
    parser.add_argument("--drop_cache", type=int, required=False,
            default=1)
    parser.add_argument("--reps", type=int, required=False,
            default=3)
    parser.add_argument("--rerun_timeouts", type=int, required=False,
            default=0)
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--user", type=str, required=False,
            default="root")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--host", type=str, required=False,
            default="127.0.0.1")

    return parser.parse_args()

def execute_sql(db_name, sql, template="sql", cost_model="cm1",
        results_fn="jerr.pkl", explain=False, timeout=900000,
        drop_cache=True):
    '''
    '''
    start = time.time()
    if drop_cache:
        # drop_cache_cmd = "./drop_cache_mysql.sh > /dev/null"
        drop_cache_cmd = "./drop_cache_mysql.sh"
        p = sp.Popen(drop_cache_cmd, shell=True)
        p.wait()

    if explain:
        sql = "EXPLAIN FORMAT=json " + sql

    db = MySQLdb.connect(db=db_name, passwd=args.pwd, user=args.user,
            host=args.host)
    cursor = db.cursor()
    if timeout is not None and timeout != 0.0:
        #print("set timeout to: ", timeout)
        cursor.execute("SET SESSION MAX_EXECUTION_TIME={};".format(timeout))

    cursor.execute("SET optimizer_prune_level=0;")
    opt_flags = MYSQL_OPT_TMP.format(FLAGS=MYSQL_OPT_FLAGS)
    cursor.execute(opt_flags)

    try:
        cursor.execute(sql)
    except Exception as e:
        #cursor.execute("ROLLBACK")
        #con.commit()
        if not "time" in str(e):
            print("failed to execute for reason other than timeout")
            print(e)
            print(sql)
            cursor.close()
            return None, -1.0
        else:
            print("failed because of timeout!")
            end = time.time()
            print("{} took {} seconds".format(template, end-start))

            return None, (timeout/1000) + 9.0

    explain = None
    if explain:
        explain = cursor.fetchall()[0]
        print(explain)

    exec_time = time.time()-start
    print(exec_time)
    return explain, exec_time

def main():

    def add_runtime_row(sql_key, rt, exp_analyze):
        cur_runtimes["sql_key"].append(sql_key)
        cur_runtimes["runtime"].append(rt)
        cur_runtimes["exp_analyze"].append(exp_analyze)

    rt_dirs = os.listdir(args.results_dir)
    print("sorted runtime directories: ", rt_dirs)
    rt_dirs.sort()
    for alg_dir in rt_dirs:
        args_fn = args.results_dir + "/" + alg_dir + "/" + "args.pkl"
        exp_args = load_object(args_fn)
        exp_args = vars(exp_args)
        print("exp args cost model: ", exp_args["cost_model"])
        print("cur cost model: ", args.cost_model)
        if args.cost_model is None or args.cost_model == "":
            cost_model = exp_args["cost_model"]
        else:
            cost_model = args.cost_model

        costs_fn = args.results_dir + "/" + alg_dir + "/" + args.results_fn
        costs = load_object(costs_fn)
        if costs is None:
            continue
        assert isinstance(costs, pd.DataFrame)
        rt_fn = args.results_dir + "/" + alg_dir + "/" + "runtimes_" + args.results_fn
        rt_fn = rt_fn.replace(".pkl", ".csv")
        # go in order and execute runtimes...
        if os.path.exists(rt_fn):
            runtimes = pd.read_csv(rt_fn)
        else:
            runtimes = None

        if runtimes is None:
            columns = ["sql_key", "runtime","exp_analyze"]
            runtimes = pd.DataFrame(columns=columns)

        cur_runtimes = defaultdict(list)

        for i,row in costs.iterrows():
            if row["sql_key"] in runtimes["sql_key"].values:
                # what is the stored value for this key?
                rt_df = runtimes[runtimes["sql_key"] == row["sql_key"]]
                stored_rt = rt_df["runtime"].values[0]
                print(stored_rt)
                if stored_rt == TIMEOUT_CONSTANT and args.rerun_timeouts:
                    print("going to rerun timed out query")
                else:
                    print("skipping {} with stored runtime".format(row["sql_key"]))
                    continue
            if row["sql_key"] in cur_runtimes["sql_key"]:
                print("should never have repeated for execution")
                continue

            for rep in enumerate(range(args.reps)):
                exp_analyze, rt = execute_sql(args.db_name, row["exec_sql"],
                        cost_model=cost_model, results_fn=args.results_fn,
                        explain=args.explain,
                        timeout=args.timeout,
                        drop_cache=args.drop_cache)
                add_runtime_row(row["sql_key"], rt, exp_analyze)

            rts = cur_runtimes["runtime"]
            print("Alg:{}, N:{}, AvgRt: {}".format(alg_dir, len(rts),
                sum(rts) / len(rts)), flush=True)

            df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
            df.to_csv(rt_fn, index=False)

        df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
        #save_object(rt_fn, df)
        df.to_csv(rt_fn, index=False)
        print("DONE")
        sys.stdout.flush()

args = read_flags()
main()

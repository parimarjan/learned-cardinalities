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

TIMEOUT_CONSTANT = 909
RERUN_TIMEOUTS = False
TIMEOUT_VAL = 900000

def set_indexes(cursor, val):
    cursor.execute("SET enable_indexscan = {}".format(val))
    cursor.execute("SET enable_indexonlyscan = {}".format(val))
    cursor.execute("SET enable_bitmapscan = {}".format(val))
    cursor.execute("SET enable_tidscan = {}".format(val))

def set_cost_model(cursor, cost_model):
    # makes things easier to understand
    cursor.execute("SET enable_material = off")
    if cost_model == "hash_join":
        cursor.execute("SET enable_hashjoin = on")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = off")
        set_indexes(cursor, "off")
    elif cost_model == "nested_loop":
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        set_indexes(cursor, "off")
    elif "nested_loop_index" in cost_model:
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        set_indexes(cursor, "on")

    elif cost_model == "cm1" \
            or cost_model == "cm2":
        pass
    else:
        assert False

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
            default="plan_pg_err.pkl")
    return parser.parse_args()

def execute_sql(sql, template="sql", cost_model="cm1"):
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
    cursor.execute("SET geqo_threshold = {}".format(20))
    set_cost_model(cursor, cost_model)
    cursor.execute("SET join_collapse_limit = {}".format(1))
    cursor.execute("SET from_collapse_limit = {}".format(1))
    cursor.execute("SET statement_timeout = {}".format(TIMEOUT_VAL))

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
        # if alg_dir not in ["true", "postgres"]:
            # continue
        args_fn = args.results_dir + "/" + alg_dir + "/" + "args.pkl"
        exp_args = load_object(args_fn)
        exp_args = vars(exp_args)
        cost_model = exp_args["cost_model"]
        print("cost model: ", cost_model)

        costs_fn = args.results_dir + "/" + alg_dir + "/" + args.results_fn
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
                # what is the stored value for this key?
                rt_df = runtimes[runtimes["sql_key"] == row["sql_key"]]
                stored_rt = rt_df["runtime"].values[0]
                if stored_rt == TIMEOUT_CONSTANT and RERUN_TIMEOUTS:
                    print("going to rerun timed out query")
                else:
                    continue
            if row["sql_key"] in cur_runtimes["sql_key"]:
                print("should never have repeated for execution")
                continue
            if "template" in row:
                exp_analyze, rt = execute_sql(row["exec_sql"], row["template"],
                        cost_model)
            else:
                exp_analyze, rt = execute_sql(row["exec_sql"], cost_model)
            add_runtime_row(row["sql_key"], rt, exp_analyze)

            rts = cur_runtimes["runtime"]
            print("Alg:{}, N:{}, AvgRt: {}".format(alg_dir, len(rts),
                sum(rts) / len(rts)))
            df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
            save_object(rt_fn, df)

        df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
        save_object(rt_fn, df)
        print("DONE")
        sys.stdout.flush()

args = read_flags()
main()

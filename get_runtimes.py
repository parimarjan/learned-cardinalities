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

TIMEOUT_CONSTANT = 909
RERUN_TIMEOUTS = True
# TIMEOUT_VAL = 900000

def set_indexes(cursor, val):
    cursor.execute("SET enable_indexscan = {}".format(val))
    cursor.execute("SET enable_seqscan = {}".format("off"))
    cursor.execute("SET enable_indexonlyscan = {}".format("off"))
    cursor.execute("SET enable_bitmapscan = {}".format("off"))
    cursor.execute("SET enable_tidscan = {}".format("off"))

def set_cost_model(cursor, cost_model, materialize):
    # makes things easier to understand
    if not materialize:
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
    elif "nested_loop_index9" == cost_model:
        print("cost model: only index scan allowed")
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        cursor.execute("SET enable_indexscan = {}".format("on"))
        cursor.execute("SET enable_seqscan = {}".format("off"))
        cursor.execute("SET enable_indexonlyscan = {}".format("off"))
        cursor.execute("SET enable_bitmapscan = {}".format("off"))
        cursor.execute("SET enable_tidscan = {}".format("off"))
    elif "nested_loop_index8" in cost_model or \
            "nested_loop_index7" in cost_model:
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        cursor.execute("SET enable_indexscan = {}".format("on"))
        cursor.execute("SET enable_seqscan = {}".format("on"))

        # print("debug mode for nested loop index8")
        # cursor.execute("SET random_page_cost = 1.0")
        # cursor.execute("SET cpu_tuple_cost = 1.0")
        # cursor.execute("SET cpu_index_tuple_cost = 1.0")

        cursor.execute("SET enable_indexonlyscan = {}".format("off"))
        cursor.execute("SET enable_bitmapscan = {}".format("off"))
        cursor.execute("SET enable_tidscan = {}".format("off"))

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
    parser.add_argument("--cost_model", type=str, required=False,
            default=None)
    parser.add_argument("--explain", type=int, required=False,
            default=1)
    parser.add_argument("--timeout", type=int, required=False,
            default=900)
    parser.add_argument("--materialize", type=int, required=False,
            default=0)
    parser.add_argument("--rerun_timeouts", type=int, required=False,
            default=0)
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    return parser.parse_args()

def execute_sql(db_name, sql, template="sql", cost_model="cm1",
        results_fn="jerr.pkl", explain=False,
        materialize=True, timeout=900000):
    '''
    '''
    drop_cache_cmd = "./drop_cache.sh > /dev/null"
    p = sp.Popen(drop_cache_cmd, shell=True)
    p.wait()

    if explain:
        sql = sql.replace("explain (format json)", "explain (analyze,costs, format json)")
    else:
        sql = sql.replace("explain (format json)", "")

    # FIXME: generalize
    con = pg.connect(port=5432,dbname=db_name,
            user="ubuntu",password="",host="localhost")

    # TODO: clear cache

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(20))
    set_cost_model(cursor, cost_model, materialize)
    if "jerr.pkl" in results_fn:
        cursor.execute("SET join_collapse_limit = {}".format(17))
        cursor.execute("SET from_collapse_limit = {}".format(17))
    else:
        cursor.execute("SET join_collapse_limit = {}".format(1))
        cursor.execute("SET from_collapse_limit = {}".format(1))

    # TODO: comment this out and use 17
    # cursor.execute("SET join_collapse_limit = {}".format(1))
    # cursor.execute("SET from_collapse_limit = {}".format(1))

    cursor.execute("SET statement_timeout = {}".format(timeout))

    start = time.time()

    try:
        cursor.execute(sql)
    except Exception as e:
        cursor.execute("ROLLBACK")
        con.commit()
        if not "timeout" in str(e):
            print("failed to execute for reason other than timeout")
            print(e)
            print(sql)
            cursor.close()
            con.close()
            return None, timeout/1000 + 9.0
        else:
            print("failed because of timeout!")
            if explain:
                sql = sql.replace("explain (analyze,costs, format json)",
                "explain (format json)")
            else:
                sql = "explain (format json) " + sql

            set_cost_model(cursor, cost_model, materialize)
            cursor.execute("SET join_collapse_limit = {}".format(1))
            cursor.execute("SET from_collapse_limit = {}".format(1))
            cursor.execute(sql)
            explain_output = cursor.fetchall()
            cursor.close()
            con.close()
            return explain_output, timeout/1000 + 9.0

    explain_output = cursor.fetchall()
    end = time.time()
    print("{} took {} seconds".format(template, end-start))
    sys.stdout.flush()

    return explain_output, end-start

def main():

    def add_runtime_row(sql_key, rt, exp_analyze):
        cur_runtimes["sql_key"].append(sql_key)
        cur_runtimes["runtime"].append(rt)
        cur_runtimes["exp_analyze"].append(exp_analyze)

    rt_dirs = os.listdir(args.results_dir)
    rt_dirs.sort()
    print("sorted runtime directories: ", rt_dirs)
    for alg_dir in rt_dirs:
        # if alg_dir not in ["true", "postgres"]:
            # continue
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
        # runtimes = load_object(rt_fn)
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
                print("skipping {} with stored runtime".format(row["sql_key"]))
                # what is the stored value for this key?
                rt_df = runtimes[runtimes["sql_key"] == row["sql_key"]]
                stored_rt = rt_df["runtime"].values[0]
                if stored_rt == TIMEOUT_CONSTANT and args.rerun_timeouts:
                    print("going to rerun timed out query")
                else:
                    continue
            if row["sql_key"] in cur_runtimes["sql_key"]:
                print("should never have repeated for execution")
                continue
            if "template" in row:
                exp_analyze, rt = execute_sql(args.db_name, row["exec_sql"],
                        template=row["template"], cost_model=cost_model,
                        results_fn=args.results_fn, explain=args.explain,
                        materialize=args.materialize, timeout=args.timeout)
            else:
                exp_analyze, rt = execute_sql(args.db_name, row["exec_sql"],
                        cost_model=cost_model, results_fn=args.results_fn,
                        explain=args.explain, materialize=args.materialize,
                        timeout=args.timeout)

            add_runtime_row(row["sql_key"], rt, exp_analyze)

            rts = cur_runtimes["runtime"]
            print("Alg:{}, N:{}, AvgRt: {}".format(alg_dir, len(rts),
                sum(rts) / len(rts)))
            df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
            #save_object(rt_fn, df)
            df.to_csv(rt_fn, index=False)

        df = pd.concat([runtimes, pd.DataFrame(cur_runtimes)], ignore_index=True)
        #save_object(rt_fn, df)
        df.to_csv(rt_fn, index=False)
        print("DONE")
        sys.stdout.flush()

args = read_flags()
main()

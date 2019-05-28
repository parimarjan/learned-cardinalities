from cardinality_estimation.db import DB
from cardinality_estimation.cardinality_sample import CardinalitySample
from cardinality_estimation.query import Query
from cardinality_estimation.algs import *
import argparse
from park.param import parser
import psycopg2 as pg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras
from db_utils.utils import *
from utils.utils import *
from sklearn.model_selection import train_test_split
import glob
from collections import defaultdict
import pdb
import pandas as pd
import random
import itertools

def get_alg(alg):
    if alg == "independent":
        return Independent()
    elif alg == "postgres":
        return Postgres()
    elif alg == "chow":
        return BN(alg="chow-liu")
    elif alg == "bn-exact":
        return BN(alg="exact-dp")
    else:
        assert False

def get_loss(loss):
    if loss == "abs":
        return compute_abs_loss
    elif loss == "rel":
        return compute_relative_loss
    elif loss == "qerr":
        return compute_qerror
    else:
        assert False

def get_columns(num_columns, column_type = "varchar"):
    col_header = ""
    for i in range(num_columns):
        col_name = "col" + str(i)
        col_header += col_name + " " + column_type
        if i != num_columns-1:
            # add comma
            col_header += ", "
    return col_header

def get_table_name():
    return args.synth_table + str(args.synth_num_columns) + str(args.seed)

def get_gaussian_data_params():
    '''
    @ret: means, covariance matrix. This should depend on the random seed.
    '''
    # random seed used for generating the correlations
    random.seed(args.seed)
    RANGES = []
    for i in range(args.synth_num_columns):
        RANGES.append([i*args.synth_period_len, i*args.synth_period_len+args.synth_period_len])

    # corrs = [float(item) for item in args.synth_correlations.split(',')]

    # part 1: generate real data
    ranges = []
    means = []
    stds = []
    for r in RANGES:
        ranges.append(np.array(r))
    for r in ranges:
        means.append(r.mean())
        stds.append(r.std() / 3)
    covs = np.zeros((len(ranges), len(ranges)))
    for i in range(len(ranges)):
        for j in range(len(ranges)):
            if i == j:
                covs[i][j] = stds[i]**2
            elif i > j:
                continue
            else:
                # for the non-diagonal entries
                # uniformly choose the correlation between the elements
                corr = random.uniform(args.min_corr, 1.00)
                covs[i][j] = corr*stds[i]*stds[j]
                covs[j][i] = corr*stds[i]*stds[j]

            # when the correlation was being only added between the first and
            # nth element
            # elif i == 0 and j != 0:
                # # add the symmetric correlations according to corr variable
                # corr = corrs[j]
                # covs[i][j] = corr*stds[i]*stds[j]
                # covs[j][i] = corr*stds[i]*stds[j]
            # just leave it as 0.
    return means, covs

def gen_synth_data():
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    cur = con.cursor()
    table_name = get_table_name()
    exists = check_table_exists(cur, table_name)
    print("exists: ", exists)
    # if exists:
        # return
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    # always creates the table and fills it up with data.
    cur.execute("DROP TABLE IF EXISTS {TABLE}".format(TABLE=table_name))
    columns = get_columns(args.synth_num_columns)
    create_sql = CREATE_TABLE_TEMPLATE.format(name = table_name,
                                             columns = columns)
    cur.execute(create_sql)
    means, covs = get_gaussian_data_params()
    data = gen_gaussian_data(means, covs, args.synth_num_vals)
    # insert statement
    insert_sql = INSERT_TEMPLATE.format(name = table_name,
                                        columns = columns.replace("varchar",
                                            ""))
    pg.extras.execute_values(cur, insert_sql, data, template=None,
            page_size=100)
    con.commit()
    print("generate and inserted new data!")

    # let's run vacuum to update stats, annoyingly slow on large DB's.
    db_vacuum(con, cur)

def main():
    def init_result_row(result):
        means, covs = get_gaussian_data_params()
        result["dbname"].append(args.db_name)
        result["num_vals"].append(len(test_queries))
        result["template_dir"].append(args.template_dir)
        result["seed"].append(args.seed)
        result["means"].append(means)
        result["covs"].append(covs)

    if args.gen_synth_data:
        gen_synth_data()
    db = DB(args.user, args.pwd, args.db_host, args.port,
            args.db_name)

    samples = []
    query_templates = []
    if args.template_dir is None:
        # artificially generate the query templates based on synthetic data
        # generation stuff
        table_name = get_table_name()
        # add a select count(*) for every combination of columns
        meta_tmp = "SELECT COUNT(*) FROM {TABLE} WHERE {CONDS}"
        # TEST.col2 = 'col2'
        cond_meta_tmp = "{TABLE}.{COLUMN} = '{COLUMN}'"
        column_list = []
        combs = []
        for i in range(args.synth_num_columns):
            column_list.append("col" + str(i))

        for i in range(2, args.synth_num_columns+1):
            combs += itertools.combinations(column_list, i)
        print(combs)
        for all_cols in combs:
            # each set of columns should lead to a new query template.
            conditions = []
            for col in all_cols:
                conditions.append(cond_meta_tmp.format(TABLE  = table_name,
                                                       COLUMN = col))
            cond_str = " AND ".join(conditions)
            query_tmp = meta_tmp.format(TABLE = table_name,
                            CONDS = cond_str)
            query_templates.append(query_tmp)
    else:
        for fn in glob.glob(args.template_dir+"/*"):
            with open(fn, "r") as f:
                template = f.read()
                query_templates.append(template)

    for template in query_templates:
        samples += db.get_samples(template,
                num_samples=args.num_samples_per_template)

    print("len samples: " , len(samples))
    train_queries, test_queries = train_test_split(samples, test_size=args.test_size,
            random_state=args.seed)

    result = defaultdict(list)

    algorithms = []
    losses = []
    for alg_name in args.algs.split(","):
        algorithms.append(get_alg(alg_name))
    for loss_name in args.losses.split(","):
        losses.append(get_loss(loss_name))

    print("going to run algorithms: ", args.algs)
    # this is deterministic, so just using it to store this in the saved data.
    for alg in algorithms:
        start = time.time()
        alg.train(db, train_queries)
        train_time = round(time.time() - start, 2)

        start - time.time()
        yhat = alg.test(test_queries)
        test_time = round(time.time() - start, 2)

        for loss_func in losses:
            cur_loss = loss_func(yhat, test_queries)
            init_result_row(result)
            result["train-time"].append(train_time)
            result["test-time"].append(test_time)
            result["alg_name"].append(alg.__str__())
            result["loss-type"].append(loss_func.__name__)
            result["loss"].append(cur_loss)
            print("case: {}, alg: {}, samples: {}, test_time: {}, train_time: {}, {}: {}"\
                    .format(args.db_name, alg, len(yhat), train_time,
                        test_time, loss_func.__name__, cur_loss))

        ## generate bar plot with how errors are spread out.
        # ytrue = [t.true_count for t in test_queries]
        # yhat_abs = []
        # for i, y in enumerate(yhat):
            # yhat_abs.append(y*queries[i].total_count)
        # errors = np.abs(np.array(yhat_abs) - np.array(ytrue))

    df = pd.DataFrame(result)
    file_name = gen_results_name()
    save_or_update(file_name, df)
    db.save_cache()

    pdb.set_trace()

def gen_results_name():
    return "./results.pd"

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=False,
            default="card_est")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--template_dir", type=str, required=False,
            default=None)
    # parser.add_argument("--user", type=str, required=False,
            # default="card_est")
    # parser.add_argument("--pwd", type=str, required=False,
            # default="card_est")
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/cards/")
    parser.add_argument("--num_samples_per_template", type=int,
            required=False, default=1000)

    # synthetic data flags
    parser.add_argument("--gen_synth_data", type=int, required=False,
            default=0)
    parser.add_argument("--synth_table", type=str, required=False,
            default="test")
    parser.add_argument("--synth_num_columns", type=int, required=False,
            default=2)
    # parser.add_argument('--synth_correlations', help='delimited list correlations',
            # type=str, required=False, default="1.0,0.9")
    parser.add_argument('--min_corr', help='delimited list correlations',
            type=float, required=False, default=0.0)
    parser.add_argument('--synth_period_len', help='delimited list correlations',
            type=int, required=False, default=10)
    parser.add_argument('--synth_num_vals', help='delimited list correlations',
            type=int, required=False, default=100000)
    parser.add_argument("--seed", type=int, required=False,
            default=1234)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    parser.add_argument("--losses", type=str, required=False,
            default="abs,rel,qerr", help="comma separated list of loss names")
    parser.add_argument("--store_results", action="store_true")

    return parser.parse_args()

args = read_flags()
main()

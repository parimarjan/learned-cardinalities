from cardinality_estimation.db import DB
from cardinality_estimation.cardinality_sample import CardinalitySample
from cardinality_estimation.query import Query
from cardinality_estimation.algs import *
from cardinality_estimation.losses import *
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
import klepto

def get_loss_name(loss_name):
    if "qerr" in loss_name:
        return "qerr"
    elif "join" in loss_name:
        return "join"
    elif "abs" in loss_name:
        return "abs"
    elif "rel" in loss_name:
        return "rel"

def get_alg(alg):
    if alg == "independent":
        return Independent()
    elif alg == "postgres":
        return Postgres()
    elif alg == "chow":
        return BN(alg="chow-liu", num_bins=args.num_bins,
                        avg_factor=args.avg_factor)
    elif alg == "bn-exact":
        return BN(alg="exact-dp", num_bins=args.num_bins)
    elif alg == "nn1":
        return NN1(max_iter = args.max_iter)
    else:
        assert False

def get_loss(loss):
    if loss == "abs":
        return compute_abs_loss
    elif loss == "rel":
        return compute_relative_loss
    elif loss == "qerr":
        return compute_qerror
    elif loss == "join-loss":
        return compute_join_order_loss
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
    return args.synth_table + str(args.synth_num_columns) + str(args.random_seed)
    # return args.synth_table + gen_exp_hash()[0:5]

def get_gaussian_data_params():
    '''
    @ret: means, covariance matrix. This should depend on the random random_seed.
    '''
    # random random_seed used for generating the correlations
    random.seed(args.random_seed)
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
    # print("exists: ", exists)
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
    # drop_index = "DROP INDEX IF EXISTS myindex"
    # cur.execute(drop_index)
    # INDEX_TMP = "CREATE INDEX myindex ON {TABLE} ({COLUMNS});"
    # index_cmd = INDEX_TMP.format(TABLE=table_name,
                    # COLUMNS = columns.replace("varchar",""))
    # cur.execute(index_cmd)
    # con.commit()
    # print("created index")

    # let's run vacuum to update stats, annoyingly slow on large DB's.
    db_vacuum(con, cur)
    cur.close()
    con.close()

def remove_doubles(samples):
    new_samples = []
    seen_samples = set()
    for s in samples:
        if s.query in seen_samples:
            continue
        seen_samples.add(s.query)
        new_samples.append(s)
    return new_samples

def main():
    def init_result_row(result):
        means, covs = get_gaussian_data_params()
        result["dbname"].append(args.db_name)
        result["template_dir"].append(args.template_dir)
        result["seed"].append(args.random_seed)
        result["means"].append(means)
        result["covs"].append(covs)
        result["args"].append(args)
        result["num_bins"].append(args.num_bins)
        result["avg_factor"].append(args.avg_factor)
        result["num_columns"].append(len(db.column_stats))

    if args.gen_synth_data:
        gen_synth_data()
    elif "osm" in args.db_name:
        # if the table doesn't already exist, then load it in
        con = pg.connect(user=args.user, host=args.db_host, port=args.port,
                password=args.pwd, database=args.db_name)
        cur = con.cursor()
        table_name = args.db_name
        exists = check_table_exists(cur, table_name)
        if not exists:
            con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur.execute("DROP TABLE IF EXISTS {TABLE}".format(TABLE=table_name))
            column_header = ["c0 bigint", "c1 bigint", "c2 bigint",
                "d0 int", "d1 int"]
            column_header = ",".join(column_header)
            columns = ["c0", "c1", "c2",
                "d0", "d1"]
            columns = ",".join(columns)
            data = np.fromfile('osm.bin',
                    dtype=np.int64).reshape(-1, 6)
            # drop the index column
            data = data[:,1:6]
            from psycopg2.extensions import register_adapter, AsIs
            def addapt_numpy_float64(numpy_float64):
                    return AsIs(numpy_float64)
            def addapt_numpy_int64(numpy_int64):
                    return AsIs(numpy_int64)
            register_adapter(np.float64, addapt_numpy_float64)
            register_adapter(np.int64, addapt_numpy_int64)

            create_sql = CREATE_TABLE_TEMPLATE.format(name = table_name,
                                                     columns = column_header)
            cur.execute(create_sql)
            insert_sql = INSERT_TEMPLATE.format(name = table_name,
                                                columns = columns)
            pg.extras.execute_values(cur, insert_sql, data, template=None,
                    page_size=100)
            con.commit()
            print("inserted osm data!")
            pdb.set_trace()

            # let's run vacuum to update stats, annoyingly slow on large DB's.
            db_vacuum(con, cur)
            cur.close()
            con.close()

    elif "dmv" in args.db_name:
        # if the table doesn't already exist, then load it in
        print("going to setup dmv")
        con = pg.connect(user=args.user, host=args.db_host, port=args.port,
                password=args.pwd, database=args.db_name)
        cur = con.cursor()
        table_name = args.db_name
        exists = check_table_exists(cur, table_name)
        if not exists:
            from sqlalchemy import create_engine
            df = pd.read_csv("/data/pari/dmv.csv")
            no_space_column_names = []
            for k in df.keys():
                no_space_column_names.append(k.replace(" ", "_").lower())
            df.columns = no_space_column_names
            engine = create_engine('postgresql://pari@localhost:5432/dmv')
            df.to_sql(table_name, engine)

    db = DB(args.user, args.pwd, args.db_host, args.port,
            args.db_name)
    print("started using db: ", args.db_name)
    query_templates = []

    samples = []
    if args.template_dir is None:
        # artificially generate the query templates based on synthetic data
        # generation stuff
        table_name = get_table_name()
        # add a select count(*) for every combination of columns
        meta_tmp = "SELECT COUNT(*) FROM {TABLE} WHERE {CONDS}"
        # TEST.col2 = 'col2'
        cond_meta_tmp = "{TABLE}.{COLUMN} in (X{COLUMN})"
        column_list = []
        combs = []
        for i in range(args.synth_num_columns):
            column_list.append("col" + str(i))

        # for i in range(2, args.synth_num_columns+1):
            # combs += itertools.combinations(column_list, i)
        # for all_cols in combs:
            # # each set of columns should lead to a new query template.
            # conditions = []
        # for col in all_cols:
        conditions = []
        for col in column_list:
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

    print("len all samples: " , len(samples))
    if args.only_nonzero_samples:
        nonzero_samples = []
        for s in samples:
            if s.true_sel != 0.00:
                nonzero_samples.append(s)
            # else:
                # print("zero sample!")
                # print(s)
                # pdb.set_trace()
        print("len nonzero samples: ", len(nonzero_samples))
        samples = nonzero_samples

    if args.use_subqueries:
        for i, q in enumerate(samples):
            q.subqueries = db.gen_subqueries(q)
            if i % 10 == 0:
                print("{} subqueries generated for query {}".format(len(q.subqueries), i))

    samples = remove_doubles(samples)

    train_queries, test_queries = train_test_split(samples, test_size=args.test_size,
            random_state=args.random_seed)

    result = defaultdict(list)

    algorithms = []
    losses = []
    for alg_name in args.algs.split(","):
        algorithms.append(get_alg(alg_name))
    for loss_name in args.losses.split(","):
        losses.append(get_loss(loss_name))

    print("going to run algorithms: ", args.algs)
    print("num train queries: ", len(train_queries))
    print("num test queries: ", len(test_queries))
    # this is deterministic, so just using it to store this in the saved data.
    for alg in algorithms:
        start = time.time()
        alg.train(db, train_queries, use_subqueries=args.use_subqueries)
        alg.save_model(save_dir=args.result_dir, suffix_name=gen_exp_hash()[0:3])
        train_time = round(time.time() - start, 2)
        print("{}, train-time: {}".format(alg, train_time))

        eval_time = None
        for loss_func in losses:
            start = time.time()
            cur_loss = loss_func(alg, train_queries, db, args.use_subqueries,
                    baseline=args.baseline_join_alg)
            if eval_time is None:
                eval_time = round(time.time() - start, 2)
            init_result_row(result)
            result["train-time"].append(train_time)
            result["eval-time"].append(eval_time)
            result["alg_name"].append(alg.__str__())
            result["loss-type"].append(loss_func.__name__)
            result["loss"].append(cur_loss)
            result["test-set"].append(0)
            result["num_vals"].append(len(train_queries))
            # lname = get_loss_name(loss_func.__name__)
            print("case: {}: training-set, alg: {}, samples: {}, train_time: {}, {}: {}"\
                    .format(args.db_name, alg, len(train_queries), train_time,
                        get_loss_name(loss_func.__name__), round(cur_loss,3)))

        if args.test:
            start = time.time()
            eval_time = None
            for loss_func in losses:
                lname = get_loss_name(loss_func.__name__)
                cur_loss = loss_func(alg, test_queries, db,
                        args.use_subqueries, baseline=args.baseline_join_alg)
                if eval_time is None:
                    # since testing results are cached right now.
                    eval_time = time.time() - start
                init_result_row(result)
                result["train-time"].append(train_time)
                result["eval-time"].append(eval_time)
                result["alg_name"].append(alg.__str__())
                result["loss-type"].append(loss_func.__name__)
                result["loss"].append(cur_loss)
                result["test-set"].append(1)
                result["num_vals"].append(len(test_queries))
                print("case: {}: test-set, alg: {}, samples: {}, test_time: {}, {}: {}"\
                        .format(args.db_name, alg, len(test_queries),
                            eval_time, lname, round(cur_loss,3)))

    df = pd.DataFrame(result)
    file_name = gen_results_name()
    save_or_update(file_name, df)
    db.save_cache()

def gen_results_name():
    return args.result_dir + "/results" + gen_exp_hash()[0:3] + ".pd"

def gen_exp_hash():
    return str(deterministic_hash(str(args)))

def gen_samples_hash():
    string = ""
    sample_keys = ["db_name", "db_host", "user", "pwd", "template_dir",
            "gen_synth_data"]
    d = vars(args)
    for k in sample_keys:
        string += str(d[k])
    return deterministic_hash(string)

def read_flags():
    # parser = argparse.ArgumentParser()
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
    parser.add_argument("-n", "--num_samples_per_template", type=int,
            required=False, default=1000)
    parser.add_argument("--max_iter", type=int,
            required=False, default=100000)

    # synthetic data flags
    parser.add_argument("--gen_synth_data", type=int, required=False,
            default=0)
    parser.add_argument("--only_nonzero_samples", type=int, required=False,
            default=1)
    parser.add_argument("--use_subqueries", type=int, required=False,
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
    parser.add_argument("--random_seed", type=int, required=False,
            default=2112)
    parser.add_argument("--test", type=int, required=False,
            default=1)
    parser.add_argument("--num_bins", type=int, required=False,
            default=10)
    parser.add_argument("--avg_factor", type=int, required=False,
            default=1)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    parser.add_argument("--losses", type=str, required=False,
            default="abs,rel,qerr", help="comma separated list of loss names")
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results/")
    parser.add_argument("--baseline_join_alg", type=str, required=False,
            default="EXHAUSTIVE")

    return parser.parse_args()

args = read_flags()
main()

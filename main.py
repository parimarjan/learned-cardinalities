from cardinality_estimation.db_stats import DBStats
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
import pdb

def get_alg(alg):
    if alg == "independent":
        return Independent()
    elif alg == "postgres":
        return Postgres()
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

def get_gaussian_data_params():
    RANGES = []
    for i in range(args.synth_num_columns):
        RANGES.append([i*args.synth_period_len, i*args.synth_period_len+args.synth_period_len])

    corrs = [float(item) for item in args.synth_correlations.split(',')]

    # part 1: generate real data
    assert args.synth_num_columns == len(RANGES)
    # we needs means, ranges for each columns. corr for n-1 columns.
    # correlations for the first column with all others
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
            elif i == 0 and j != 0:
                # add the symmetric correlations according to corr variable
                corr = corrs[j]
                covs[i][j] = corr*stds[i]*stds[j]
                covs[j][i] = corr*stds[i]*stds[j]
            # just leave it as 0.
    return means, covs

def main():
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd)
    cur = con.cursor()
    exists = check_table_exists(cur, args.synth_table)
    if args.gen_synth_data or not exists:
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        means, covs = get_gaussian_data_params()
        # always creates the table and fills it up with data.
        cur.execute("DROP TABLE IF EXISTS {TABLE}".format(TABLE=args.synth_table))
        columns = get_columns(args.synth_num_columns)
        create_sql = CREATE_TABLE_TEMPLATE.format(name = args.synth_table,
                                                 columns = columns)
        cur.execute(create_sql)
        data = gen_gaussian_data(means, covs, args.synth_num_vals)
        # insert statement
        insert_sql = INSERT_TEMPLATE.format(name = args.synth_table,
                                            columns = columns.replace("varchar",
                                                ""))
        pg.extras.execute_values(cur, insert_sql, data, template=None,
                page_size=100)
        con.commit()
        print("generate and inserted new data!")

    # let's run vacuum to update stats, annoyingly slow on large DB's.
    # db_vacuum(con, cur)
    db_stats = DBStats(args.user, args.pwd, args.db_host, args.port,
            args.db_name)
    TEST_SAMPLE = "SELECT COUNT(*) FROM TEST WHERE col0 = 'col0' AND \
            col1 = 'col1'"
    samples = db_stats.get_samples(TEST_SAMPLE)
    print("len samples: " , len(samples))
    cur.close()
    train, test = train_test_split(samples, test_size=args.test_size,
            random_state=args.seed)

    result = {}
    result["dbname"] = args.db_name
    result["samples"] = len(test)

    algorithms = []
    for alg_name in args.algs.split(","):
        algorithms.append(get_alg(alg_name))

    for alg in algorithms:
        alg.train(db_stats, train)
        yhat = alg.test(test)
        print(yhat)
        # TODO: compute loss, store, plot etc.

    # df = pd.DataFrame([result])
    # if args.store_results:
        # file_name = gen_results_name()
        # # load file first
        # orig_df = load_object(file_name)
        # if orig_df is None:
            # new_df = df
        # else:
            # new_df = orig_df.append(df, ignore_index=True)
        # save_object(file_name, new_df)

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=False,
            default="card_est")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="card_est")
    parser.add_argument("--pwd", type=str, required=False,
            default="card_est")
    parser.add_argument("--port", type=str, required=False,
            default=5401)
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/cards/")
    parser.add_argument("--gen_synth_data", type=int, required=False,
            default=1)
    parser.add_argument("--synth_table", type=str, required=False,
            default="test")
    parser.add_argument("--synth_num_columns", type=int, required=False,
            default=2)
    parser.add_argument('--synth_correlations', help='delimited list correlations',
            type=str, required=False, default="1.0,0.9")
    parser.add_argument('--synth_period_len', help='delimited list correlations',
            type=int, required=False, default=10)
    parser.add_argument('--synth_num_vals', help='delimited list correlations',
            type=int, required=False, default=10000)
    parser.add_argument("--seed", type=int, required=False,
            default=1234)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    parser.add_argument("--store_results", action="store_true")

    return parser.parse_args()

args = read_flags()
main()

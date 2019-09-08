import psycopg2 as pg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras
from utils.utils import *
from db_utils.utils import *
import pandas as pd
import numpy as np
import random

def load_csv_data(args, header=0, sep=","):
    # if the table doesn't already exist, then load it in
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)

    # con = pg.connect(user=args.user, port=args.port,
            # password=args.pwd, database=args.db_name)

    cur = con.cursor()
    table_name = args.db_name
    exists = check_table_exists(cur, table_name)
    if not exists:
        from sqlalchemy import create_engine
        print("going to read csv!")
        df = pd.read_csv(args.db_file_name, header=header, sep=sep,
                low_memory=False)
        # let's try to make each column something nicer
        for k in df.keys():
            print(k)
            print(df[k].dtype)
            if (df[k].dtype != "O"):
                # seems fine as is
                continue
            # try to get objects into more meaningful types
            # pdb.set_trace()
            if "date" in k.lower():
                # convert to unix timestamps
                print("converting datetime to unix")
                df[k] = pd.to_datetime(df[k]).astype(np.int64)
            elif "time" in k.lower():
                df[k] = pd.to_timedelta(df[k]).astype(np.int64)
            else:
                # try to see if it is a float
                try:
                    float(df[k][0])
                    # convert to numbery things
                    df[k] = pd.to_numeric(df[k], errors="coerce")
                except:
                    # do nothing since it is not a float!
                    pass

        # pdb.set_trace()
        updated_cols = []
        for k in df.keys():
            try:
                int(k)
                k = "col" + str(k)
            except:
                # if it was not an int, do nothing
                pass
            k = k.replace(" ", "_")
            k = k.lower()
            k = k.encode("utf-8", "strict").decode("utf-8")
            updated_cols.append(k)

        df.columns = updated_cols
        cmd = 'postgresql://{}:{}@localhost:5432/{}'.format(args.user,
                args.pwd, args.db_name)
        engine = create_engine(cmd)
        print("going to load in db!")
        df.to_sql(table_name, engine)

def load_osm_data(args):
    '''
    loads data into the appropriate db instance, if it is not already there.
    '''
    # if the table doesn't already exist, then load it in
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    # con = pg.connect(user=args.user, port=args.port,
            # password=args.pwd, database=args.db_name)
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
        data = np.fromfile(args.db_file_name,
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

def get_synth_columns(num_columns, column_type = "varchar"):
    col_header = ""
    for i in range(num_columns):
        col_name = "col" + str(i)
        col_header += col_name + " " + column_type
        if i != num_columns-1:
            # add comma
            col_header += ", "
    return col_header

def get_gaussian_data_params(args):
    '''
    @ret: means, covariance matrix. This should depend on the random random_seed.
    '''
    # random random_seed used for generating the correlations
    random.seed(args.random_seed)
    RANGES = []
    for i in range(args.synth_num_columns):
        RANGES.append([i*args.synth_period_len, i*args.synth_period_len+args.synth_period_len])

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
    return means, covs

def get_table_name(args):
    return args.synth_table + str(args.synth_num_columns) + \
        str(args.synth_period_len) + str(args.random_seed)
        # + \
        # str(args.synth_num_vals)

def gen_synth_data(args):
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    cur = con.cursor()
    table_name = get_table_name(args)
    # table_name = args.synth_table + str(args.synth_num_columns)
    exists = check_table_exists(cur, table_name)
    print("exists: ", exists)
    if exists:
        return
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    # always creates the table and fills it up with data.
    cur.execute("DROP TABLE IF EXISTS {TABLE}".format(TABLE=table_name))
    columns = get_synth_columns(args.synth_num_columns)
    create_sql = CREATE_TABLE_TEMPLATE.format(name = table_name,
                                             columns = columns)
    cur.execute(create_sql)
    means, covs = get_gaussian_data_params(args)
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

def update_synth_templates(args, query_templates):
    # artificially generate the query templates based on synthetic data
    # generation stuff
    print("update synth templates!!")
    table_name = get_table_name(args)

    # TODO: add a select count(*) for every combination of columns
    meta_tmp = "SELECT COUNT(*) FROM {TABLE} WHERE {CONDS}"
    # TEST.col2 = 'col2'
    # cond_meta_tmp = "{TABLE}.{COLUMN} IN (X{COLUMN})"
    cond_meta_tmp = "{TABLE}.{COLUMN} IN ('SELECT {COLUMN} FROM {TABLE}')"
    combs = []
    column_ids = list(range(args.synth_num_columns))
    for i in range(2, args.synth_num_columns+1, 1):
        combs += (list(itertools.combinations(column_ids, i)))
    random.shuffle(combs)
    combs = combs[0:10]
    print(combs)

    for comb in combs:
        conditions = []
        column_list = []
        for j in comb:
            column_list.append("col" + str(j))

        for col in column_list:
            conditions.append(cond_meta_tmp.format(TABLE  = table_name,
                                                   COLUMN = col))
        cond_str = " AND ".join(conditions)
        query_tmp = meta_tmp.format(TABLE = table_name,
                        CONDS = cond_str)
        print(query_tmp)
        query_templates.append(query_tmp)

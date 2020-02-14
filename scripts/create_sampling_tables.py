import sys
sys.path.append(".")
import argparse
import psycopg2 as pg
from utils.utils import *
import pdb
import random
from multiprocessing import Pool, cpu_count
import json
import pickle
from collections import defaultdict

SEL_TEMPLATE = "SELECT {COLS} FROM {TABLE} WHERE random() < {FRAC}"
CREATE_TEMPLATE = "CREATE TABLE {TABLE_NAME} AS {SEL_SQL}"
DROP_TEMPLATE = "DROP TABLE IF EXISTS {TABLE_NAME}"
NEW_TABLE_TEMPLATE = "{TABLE}_{SS}{PERCENTAGE}"

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
    parser.add_argument("--sampling_percentage", type=float, required=False,
            default=10)
    parser.add_argument("--sampling_type", type=str, required=False,
            default="ss")

    return parser.parse_args()

def main():
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    cursor = con.cursor()

    # build table to id_columns
    table_to_ids = defaultdict(list)
    table_to_ids["title"].append("id")
    table_to_ids["name"].append("id")
    table_to_ids["keyword"].append("id")
    table_to_ids["company_name"].append("id")

    table_to_ids["movie_info"].append("id")
    table_to_ids["movie_info"].append("movie_id")
    table_to_ids["movie_keyword"].append("id")
    table_to_ids["movie_keyword"].append("keyword_id")
    table_to_ids["movie_keyword"].append("movie_id")
    table_to_ids["cast_info"].append("id")
    table_to_ids["cast_info"].append("movie_id")
    table_to_ids["cast_info"].append("person_id")
    table_to_ids["movie_companies"].append( "id")
    table_to_ids["movie_companies"].append( "movie_id")
    table_to_ids["movie_companies"].append("company_id")

    fkey_to_primary = {}
    fkey_to_primary["movie_id"] = "title"
    fkey_to_primary["company_id"] = "company_name"
    fkey_to_primary["keyword_id"] = "keyword"
    fkey_to_primary["person_id"] = "name"

    sel_ids = defaultdict(list)
    # build primary key allowed values
    for table, ids in table_to_ids.items():
        sql = SEL_TEMPLATE.format(COLS = ",".join(ids),
                            TABLE = table,
                            FRAC  = args.sampling_percentage / 100.00)
        cursor.execute(sql)
        colnames = [desc[0] for desc in cursor.description]
        output = cursor.fetchall()
        for i, col in enumerate(colnames):
            assert col in fkey_to_primary or col == "id"
            col_vals = [v[i] for v in output]
            if col == "id":
                sel_ids[table] += col_vals
            else:
                sel_ids[fkey_to_primary[col]] += col_vals

    for table,vals in sel_ids.items():
        vals = set(vals)
        print(table, len(vals))
        # make a new table with the given row entries
        new_table = NEW_TABLE_TEMPLATE.format(TABLE = table,
                            SS = args.sampling_type,
                            PERCENTAGE = str(args.sampling_percentage))
        new_table = new_table.replace(".","")
        vals_str = ""
        for i,val in enumerate(vals):
            vals_str += "'{}'".format(val)
            if i != len(vals)-1:
                vals_str += ","
        sel_sql = "SELECT * FROM {} WHERE id in ({})".format(table,
                vals_str)
        drop_sql = DROP_TEMPLATE.format(TABLE_NAME = new_table)
        cursor.execute(drop_sql)
        create_sql = CREATE_TEMPLATE.format(TABLE_NAME = new_table,
                                                SEL_SQL=sel_sql)
        print(create_sql)
        cursor.execute(create_sql)
        con.commit()

        index_sql = "SELECT * FROM pg_indexes WHERE tablename = '{}'".format(\
                table)
        cursor.execute(index_sql)
        output = cursor.fetchall()
        for line in output:
            index_cmd = line[4]
            index_cmd = index_cmd.replace(table, new_table)
            cursor.execute(index_cmd)

        con.commit()
        con2 = pg.connect(user=args.user, host=args.db_host, port=args.port,
                password=args.pwd, database=args.db_name)
        con2.set_isolation_level(pg.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor2 = con2.cursor()
        cursor2.execute("VACUUM {}".format(new_table))
        cursor2.close()
        con2.close()

        pdb.set_trace()
    pdb.set_trace()
    # sample N% from each table, and

args = read_flags()
main()

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
# from db_utils.import *

SEL_TEMPLATE = """SELECT {COLS} FROM "{TABLE}" WHERE random() < {FRAC}"""
CREATE_TEMPLATE = """CREATE TABLE "{TABLE_NAME}" AS {SEL_SQL}"""
INSERT_TEMPLATE = """
INSERT INTO {TABLE_NAME}
SELECT * FROM
{ORIG_TABLE}
WHERE {ORIG_TABLE}.id IN (SELECT {FK_ID} FROM {FK_TABLE}
WHERE random() < {PERCENTAGE} AND {FK_ID} NOT IN
(SELECT id FROM {TABLE_NAME}))
"""
DROP_TEMPLATE = """DROP TABLE IF EXISTS "{TABLE_NAME}" """
NEW_TABLE_TEMPLATE = "{TABLE}_{SS}{PERCENTAGE}"

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="ergastf1")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="ceb")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--sample_num", type=int, required=False,
            default=100)
    parser.add_argument("--sampling_type", type=str, required=False,
            default="sb")

    parser.add_argument("--input_bitmap_dir", type=str, required=False,
            default=None)

    return parser.parse_args()

def main():
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    cursor = con.cursor()

    # build table to id_columns
    # tables = ["title", "name", "aka_name", "keyword", "movie_info",
            # "movie_companies", "company_type", "kind_type", "info_type",
            # "role_type", "company_name"]

    # tables = ["cast_info", "char_name"]

    # tables = ["link_type", "movie_info_idx", "comp_cast_type"]
    # tables = ["movie_link"]
    # tables = ["complete_cast"]
    # tables = ["aka_title"]

    # tables = ["title", "name", "aka_name", "keyword", "movie_info",
            # "movie_companies", "company_type", "kind_type", "info_type",
            # "role_type", "company_name", "cast_info", "char_name",
            # "link_type", "movie_info_idx", "comp_cast_type",
            # "person_info",
            # "movie_link", "movie_keyword",
            # "aka_title", "complete_cast"
            # ]

    ## 5440 stats db
    # tables = ["badges", "comments", "posthistory", "postlinks", "posts",
            # "tags", "users","votes"]

    # tables = ["badges", "comments", "postHistory", "postLinks", "posts",
            # "tags", "users","votes"]

     # public | circuits             | table | ceb
	 # public | constructorResults   | table | ceb
	 # public | constructorStandings | table | ceb
	 # public | constructors         | table | ceb
	 # public | driverStandings      | table | ceb
	 # public | drivers              | table | ceb
	 # public | lapTimes             | table | ceb
	 # public | pitStops             | table | ceb
	 # public | qualifying           | table | ceb
	 # public | races                | table | ceb
	 # public | results              | table | ceb
	 # public | status               | table | ceb

    ## 5432, ergast
    tables = ["circuits", "constructorResults", "constructorStandings",
				"constructors", "driverStandings", "drivers",
				"lapTimes", "pitStops", "qualifying",
				"races", "results", "status"]

    # let's build all the tables on primary keys first
    for table in tables:
        new_table = NEW_TABLE_TEMPLATE.format(TABLE = table,
                            SS = args.sampling_type,
                            PERCENTAGE = str(args.sample_num))
        print("new table name: ", new_table)
        drop_sql = DROP_TEMPLATE.format(TABLE_NAME = new_table)
        cursor.execute(drop_sql)
        con.commit()

        if args.input_bitmap_dir is None:
            ## we are selecting randomly from original table
            # what is the count of this table?
            count_sql = "SELECT COUNT(*) FROM \"{}\"".format(table)
            cursor.execute(count_sql)
            count = int(cursor.fetchall()[0][0])
            sampling_frac = (float(args.sample_num)+100) / float(count)

            sel_sql = "SELECT * FROM \"{}\" WHERE random() < {}".format(\
                    table, str(sampling_frac))

            print(sel_sql)
            create_sql = CREATE_TEMPLATE.format(TABLE_NAME = new_table,
                                                    SEL_SQL=sel_sql)
            cursor.execute(create_sql)

            count_sql = "SELECT COUNT(*) FROM \"{}\"".format(new_table)
            cursor.execute(count_sql)
            count = int(cursor.fetchall()[0][0])
            print("new table's count is: ", count)
        else:
            # open the csv file
            inp_fn = os.path.join(args.input_bitmap_dir, new_table.lower()) + ".csv"
            os.path.exists(inp_fn)
            inpdf = pd.read_csv(inp_fn)
            vals = ["'{}'".format(v) for v in inpdf["Id"].values]
            vals = ",".join(vals)
            vals = "({})".format(vals)

            # sel_sql = "SELECT * FROM \"{}\" WHERE \"Id\" IN {}".format(\
                    # table, vals)
            sel_sql = "SELECT * FROM \"{}\" WHERE \"id\" IN {}".format(\
                    table, vals)
            # print(sel_sql)
            create_sql = CREATE_TEMPLATE.format(TABLE_NAME = new_table,
                                                    SEL_SQL=sel_sql)
            cursor.execute(create_sql)
            count_sql = "SELECT COUNT(*) FROM \"{}\"".format(new_table)
            cursor.execute(count_sql)
            count = int(cursor.fetchall()[0][0])
            print("new table's count is: ", count)

        con.commit()

args = read_flags()
main()

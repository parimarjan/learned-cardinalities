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
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import math
import time

CREATE_DB_TMP="""
CREATE DATABASE {DB_NEW}
WITH TEMPLATE {DB_OLD}
OWNER {USER};
"""

IDS_TO_REMOVE_TMP="""
DELETE from title
WHERE title.production_year > {YEAR};
"""

PK_TO_FK_REMOVE_TMP="""
DELETE FROM
{FK_TABLE}
WHERE {FK_TABLE}.{FK_ID} BETWEEN {MIN_BATCH} AND {MAX_BATCH}
AND {FK_TABLE}.{FK_ID} NOT IN
(SELECT id from {PK_TABLE}
WHERE {PK_TABLE}.id BETWEEN {MIN_BATCH} AND {MAX_BATCH});
"""

# PK_TO_FK_REMOVE_TMP="""
# DELETE FROM {FK_TABLE} fk1
# USING (
# SELECT fk2.id
# FROM {FK_TABLE} fk2
# LEFT JOIN {PK_TABLE} pk
# ON pk.id = fk2.{FK_ID}
# WHERE pk.id IS NULL
# ) sq
# WHERE sq.id = fk1.id;
# """

# DELETE FROM one o
# USING (
    # SELECT o2.id
    # FROM one o2
    # LEFT JOIN two t ON t.one_id = o2.id
    # WHERE t.one_id IS NULL
    # ) sq
# WHERE sq.id = o.id
    # ;


FK_TO_PK_REMOVE_TMP="""
DELETE FROM
{PK_TABLE}
WHERE {PK_TABLE}.id BETWEEN {MIN_BATCH} AND {MAX_BATCH}
AND {PK_TABLE}.id NOT IN
(SELECT DISTINCT {FK_ID} from {FK_TABLE}
WHERE {FK_ID} BETWEEN {MIN_BATCH} AND {MAX_BATCH}
);
"""

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
    parser.add_argument("--sampling_type", type=str, required=False,
            default="year")
    parser.add_argument("--max_year", type=int, required=False,
            default=1950)
    parser.add_argument("--create_db", type=int, required=False,
            default=0)

    return parser.parse_args()

NUM_SPLITS = 10
MIN_TITLE_ID = 86
MAX_TITLE_ID = 2528312

TITLE_FTABLES = ["movie_info", "movie_info_idx", "movie_companies",
        "movie_keyword", "cast_info"]
# TITLE_FTABLES = ["movie_keyword"]
# TITLE_FTABLES = ["movie_companies"]
FTABLE_PTABLES = {}

FTABLE_PTABLES["cast_info"] = ("name", "person_id")
FTABLE_PTABLES["movie_info"] = None
FTABLE_PTABLES["movie_info_idx"] = None
FTABLE_PTABLES["movie_keyword"] = ("keyword", "keyword_id")
FTABLE_PTABLES["movie_companies"] = ("company_name", "company_id")

NAME_FTABLES = ["person_info", "aka_name"]

# build table to id_columns
table_to_ids = defaultdict(list)
table_to_ids["title"].append("id")
table_to_ids["name"].append("id")
table_to_ids["keyword"].append("id")
table_to_ids["company_name"].append("id")

table_to_ids["movie_info"].append("id")
table_to_ids["movie_info"].append("movie_id")
table_to_ids["movie_info_idx"].append("id")
table_to_ids["movie_info_idx"].append("movie_id")

table_to_ids["movie_keyword"].append("id")
table_to_ids["movie_keyword"].append("keyword_id")
table_to_ids["movie_keyword"].append("movie_id")
table_to_ids["cast_info"].append("id")
table_to_ids["cast_info"].append("movie_id")
table_to_ids["cast_info"].append("person_id")

table_to_ids["person_info"].append("person_id")
table_to_ids["person_info"].append("id")

table_to_ids["movie_companies"].append( "id")
table_to_ids["movie_companies"].append( "movie_id")
table_to_ids["movie_companies"].append("company_id")

fkey_to_primary = {}
fkey_to_primary["movie_id"] = "title"
fkey_to_primary["company_id"] = "company_name"
fkey_to_primary["keyword_id"] = "keyword"
fkey_to_primary["person_id"] = "name"


def main():

    new_dbname = args.db_name + str(args.max_year)

    if args.create_db:
        con = pg.connect(user=args.user, host=args.db_host, port=args.port,
                password=args.pwd, database=args.db_name)

        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = con.cursor()

        dropdb = "DROP DATABASE IF EXISTS {}".format(new_dbname)
        cursor.execute(dropdb)
        print("deleted old db")

        create_sql = CREATE_DB_TMP.format(DB_NEW = new_dbname,
                                          DB_OLD = args.db_name,
                                          USER = args.user)
        print(create_sql)
        cursor.execute(create_sql)
        con.commit()
        cursor.close()
        con.close()

    print("getting new db connection")
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=new_dbname)
    cursor = con.cursor()

    # select all the movie_ids that need to be removed
    if args.sampling_type == "year":
        del_title_sql = IDS_TO_REMOVE_TMP.format(YEAR=args.max_year)
        cursor.execute(del_title_sql)
        con.commit()
        count_sql = "SELECT COUNT(*) FROM TITLE"
        cursor.execute(count_sql)
        print("number of rows in title after deleting: ", cursor.fetchone()[0])
    else:
        assert False, "not implemented"

    # tables directly connected to title
    for table in TITLE_FTABLES:
        delta = math.ceil(MAX_TITLE_ID / NUM_SPLITS)
        fk_id = "movie_id"
        for i in range(NUM_SPLITS):
            start = time.time()
            min_batch = i*delta
            max_batch = min_batch + delta

            del_sql = PK_TO_FK_REMOVE_TMP.format(FK_TABLE = table,
                                                    FK_ID = fk_id,
                                                    PK_TABLE = "title",
                                                    MIN_BATCH = min_batch,
                                                    MAX_BATCH = max_batch)
            print(del_sql)
            cursor.execute(del_sql)

            count_sql = "SELECT COUNT(*) FROM {}".format(table)
            cursor.execute(count_sql)
            print("number of rows in {} after deleting: {}".format(table,
                cursor.fetchone()[0]))
            print("Batch: {} took {} seconds".format(i, time.time()-start))

        count_sql = "SELECT COUNT(*) FROM {}".format(table)
        cursor.execute(count_sql)
        print("number of rows in {} after deleting: {}".format(table,
            cursor.fetchone()[0]))
        con.commit()

        pt = FTABLE_PTABLES[table]
        if pt is None:
            continue

        ptable = pt[0]
        ptable_fk_id = pt[1]
        ptable_id = "id"

        min_sql = "SELECT MIN({TAB}.id), MAX({TAB}.id) FROM {TAB}".format(TAB=ptable)
        cursor.execute(min_sql)
        res = cursor.fetchone()
        min_id = res[0]
        max_id = res[1]
        delta = math.ceil(max_id / NUM_SPLITS)
        start = time.time()
        for i in range(NUM_SPLITS):
            start = time.time()
            min_batch = i*delta
            max_batch = min_batch + delta

            del_sql = FK_TO_PK_REMOVE_TMP.format(PK_TABLE=ptable,
                                                 FK_ID = ptable_fk_id,
                                                 FK_TABLE = table,
                                                 MIN_BATCH = min_batch,
                                                 MAX_BATCH = max_batch)
            print(del_sql)
            cursor.execute(del_sql)
            con.commit()
            count_sql = "SELECT COUNT(*) FROM {}".format(ptable)
            cursor.execute(count_sql)
            print("BATCH: {}, number of rows in {} after deleting: {}".format(
                i,
                ptable,
                cursor.fetchone()[0]))
        print("all batches of {} deleted in {}".format(ptable,
            time.time()-start))


    for table in NAME_FTABLES:
        delta = math.ceil(MAX_TITLE_ID / NUM_SPLITS)
        fk_id = "person_id"
        for i in range(NUM_SPLITS):
            start = time.time()
            min_batch = i*delta
            max_batch = min_batch + delta

            del_sql = PK_TO_FK_REMOVE_TMP.format(FK_TABLE = table,
                                                    FK_ID = fk_id,
                                                    PK_TABLE = "name",
                                                    MIN_BATCH = min_batch,
                                                    MAX_BATCH = max_batch)
            print(del_sql)
            cursor.execute(del_sql)

            count_sql = "SELECT COUNT(*) FROM {}".format(table)
            cursor.execute(count_sql)
            print("number of rows in {} after deleting: {}".format(table,
                cursor.fetchone()[0]))
            print("Batch: {} took {} seconds".format(i, time.time()-start))

        count_sql = "SELECT COUNT(*) FROM {}".format(table)
        cursor.execute(count_sql)
        print("number of rows in {} after deleting: {}".format(table,
            cursor.fetchone()[0]))
        con.commit()

    for table in table_to_ids:
        # new_table = NEW_TABLE_TEMPLATE.format(TABLE = table,
                            # SS = args.sampling_type,
                            # PERCENTAGE = str(args.sampling_percentage))
        # new_table = new_table.replace(".","")

        # index_sql = "SELECT * FROM pg_indexes WHERE tablename = '{}'".format(\
                # table)
        # cursor.execute(index_sql)
        # output = cursor.fetchall()
        # for line in output:
            # index_cmd = line[4]
            # drop_idx_cmd = index_cmd[0:index_cmd.find("ON")-1]
            # drop_idx_cmd = drop_idx_cmd.replace("create", "drop")
            # drop_idx_cmd = drop_idx_cmd.replace("CREATE", "DROP")
            # drop_idx_cmd = drop_idx_cmd.replace("UNIQUE", "")
            # drop_idx_cmd = drop_idx_cmd.replace("unique", "")
            # cursor.execute(drop_idx_cmd)
            # cursor.execute(index_cmd)

        # con.commit()
        con2 = pg.connect(user=args.user, host=args.db_host, port=args.port,
                password=args.pwd, database=args.db_name)
        con2.set_isolation_level(pg.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor2 = con2.cursor()
        cursor2.execute("VACUUM {}".format(table))
        cursor2.close()
        con2.close()
        print("index + vacuuming done for: ", table)

if __name__ == "__main__":
    args = read_flags()
    main()

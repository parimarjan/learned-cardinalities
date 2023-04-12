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

SEL_TEMPLATE = "SELECT {COLS} FROM {TABLE} WHERE random() < {FRAC}"
CREATE_TEMPLATE = "CREATE TABLE {TABLE_NAME} AS {SEL_SQL}"
INSERT_TEMPLATE = """
INSERT INTO {TABLE_NAME}
SELECT * FROM
{ORIG_TABLE}
WHERE {ORIG_TABLE}.id IN (SELECT {FK_ID} FROM {FK_TABLE}
WHERE random() < {PERCENTAGE} AND {FK_ID} NOT IN
(SELECT id FROM {TABLE_NAME}))
"""
DROP_TEMPLATE = "DROP TABLE IF EXISTS {TABLE_NAME}"
NEW_TABLE_TEMPLATE = "{TABLE}_{SS}{PERCENTAGE}"
NEW_JOIN_TABLE_TEMPLATE = "{TABLE}_{JOINKEY}_{SS}{NUM}"

SMALL_PRIMARY_TABS = ["role_type", "info_type", "kind_type",
                        "company_type", "comp_cast_type",
                        "link_type"]

JOIN_COL_MAP = {}
JOIN_COL_MAP["title.id"] = "movie_id"
JOIN_COL_MAP["movie_info.movie_id"] = "movie_id"
JOIN_COL_MAP["cast_info.movie_id"] = "movie_id"
JOIN_COL_MAP["movie_keyword.movie_id"] = "movie_id"
JOIN_COL_MAP["movie_companies.movie_id"] = "movie_id"
JOIN_COL_MAP["movie_link.movie_id"] = "movie_id"
JOIN_COL_MAP["movie_info_idx.movie_id"] = "movie_id"
JOIN_COL_MAP["movie_link.linked_movie_id"] = "movie_id"
## TODO: handle it so same columns map to same table+col
# JOIN_COL_MAP["miidx.movie_id"] = "movie_id"
JOIN_COL_MAP["aka_title.movie_id"] = "movie_id"
JOIN_COL_MAP["complete_cast.movie_id"] = "movie_id"

JOIN_COL_MAP["movie_keyword.keyword_id"] = "keyword"
JOIN_COL_MAP["keyword.id"] = "keyword"

JOIN_COL_MAP["name.id"] = "person_id"
JOIN_COL_MAP["person_info.person_id"] = "person_id"
JOIN_COL_MAP["cast_info.person_id"] = "person_id"
JOIN_COL_MAP["aka_name.person_id"] = "person_id"
# TODO: handle cases
# JOIN_COL_MAP["a.person_id"] = "person_id"

JOIN_COL_MAP["title.kind_id"] = "kind_id"
JOIN_COL_MAP["kind_type.id"] = "kind_id"

JOIN_COL_MAP["cast_info.role_id"] = "role_id"
JOIN_COL_MAP["role_type.id"] = "role_id"

JOIN_COL_MAP["cast_info.person_role_id"] = "char_id"
JOIN_COL_MAP["char_name.id"] = "char_id"

JOIN_COL_MAP["movie_info.info_type_id"] = "info_id"
JOIN_COL_MAP["movie_info_idx.info_type_id"] = "info_id"
# JOIN_COL_MAP["mi_idx.info_type_id"] = "info_id"
# JOIN_COL_MAP["miidx.info_type_id"] = "info_id"

JOIN_COL_MAP["person_info.info_type_id"] = "info_id"
JOIN_COL_MAP["info_type.id"] = "info_id"

JOIN_COL_MAP["movie_companies.company_type_id"] = "company_type"
JOIN_COL_MAP["company_type.id"] = "company_type"

JOIN_COL_MAP["movie_companies.company_id"] = "company_id"
JOIN_COL_MAP["company_name.id"] = "company_id"

JOIN_COL_MAP["movie_link.link_type_id"] = "link_id"
JOIN_COL_MAP["link_type.id"] = "link_id"

JOIN_COL_MAP["complete_cast.status_id"] = "subject"
JOIN_COL_MAP["complete_cast.subject_id"] = "subject"
JOIN_COL_MAP["comp_cast_type.id"] = "subject"

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="ceb")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")
    parser.add_argument("--port", type=str, required=False,
            default=5431)
    parser.add_argument("--sample_num", type=int, required=False,
            default=1000)
    parser.add_argument("--sampling_type", type=str, required=False,
            default="sb")

    return parser.parse_args()

def main():
    con = pg.connect(user=args.user, host=args.db_host, port=args.port,
            password=args.pwd, database=args.db_name)
    cursor = con.cursor()

    # select primary key tables
    join_real_map = {}
    for join_key, join_real in JOIN_COL_MAP.items():
        if ".id" in join_key:
            jointab = join_key[0:join_key.find(".")]
            if jointab in SMALL_PRIMARY_TABS:
                continue
            join_real_map[join_real] = join_key

    print(join_real_map)

    for join_key, join_real in JOIN_COL_MAP.items():
        if ".id" in join_key:
            # skipping primary ids
            continue

        if join_real not in join_real_map:
            continue

        pk_id = join_real_map[join_real]
        pk_table = pk_id[0:pk_id.find(".")]
        pk_sample_id = pk_id.replace(pk_table, pk_table + "_sb{}".format(args.sample_num))
        pk_sample_table = pk_table + "_sb{}".format(args.sample_num)

        table = join_key[0:join_key.find(".")]

        new_table = NEW_JOIN_TABLE_TEMPLATE.format(TABLE = table,
                            JOINKEY = join_real,
                            SS = args.sampling_type,
                            NUM = str(args.sample_num))
        print("new table name: ", new_table)

        drop_sql = DROP_TEMPLATE.format(TABLE_NAME = new_table)
        cursor.execute(drop_sql)

        sel_sql = "SELECT * FROM {} WHERE {} IN (SELECT id from {});".format(\
                table, join_key, pk_sample_table)

        create_sql = CREATE_TEMPLATE.format(TABLE_NAME = new_table,
                                                SEL_SQL=sel_sql)
        cursor.execute(create_sql)

        count_sql = "SELECT COUNT(*) FROM {}".format(new_table)
        cursor.execute(count_sql)
        count = int(cursor.fetchall()[0][0])
        print("new table's count is: ", count)

        con.commit()

args = read_flags()
main()

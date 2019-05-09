import numpy as np
import glob
import string
from moz_sql_parser import parse
import json

import re

CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})"
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s"
GROUPBY_TEMPLATE = "SELECT {COLS}, COUNT(*) FROM {FROM_CLAUSE} GROUP BY {COLS}"
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"


def parse_explain(output):
    '''
    '''
    # FIXME: handle joins etc.
    est_vals = None
    for line in output:
        line = line[0]
        if "Seq Scan" in line:
            for w in line.split():
                if "rows" in w and est_vals is None:
                    est_vals = int(re.findall("\d+", w)[0])
    assert est_vals is not None
    return est_vals

def extract_predicate_columns(query):
    predicate_cols = ""
    parsed_query = parse(query)
    # FIXME: add more checks etc.
    ands = parsed_query["where"]["and"]
    for i, p in enumerate(ands):
        # predicates.append(p["eq"][0])
        predicate_cols += p["eq"][0]
        if (i != len(ands)-1):
            predicate_cols += ", "
    return predicate_cols

def extract_from_clause(query):
    from_str = ""
    parsed_query = parse(query)
    # FIXME: needs more handling
    return parsed_query["from"]

def check_table_exists(cur, table_name):
    cur.execute("select exists(select * from information_schema.tables where\
            table_name=%s)", (table_name,))
    return cur.fetchone()[0]

def db_vacuum(conn, cur):
    old_isolation_level = conn.isolation_level
    conn.set_isolation_level(0)
    query = "VACUUM ANALYZE"
    cur.execute(query)
    conn.set_isolation_level(old_isolation_level)
    conn.commit()

def to_bitset(num_attrs, arr):
    ret = [i for i, val in enumerate(arr) if val == 1.0]
    for i, r in enumerate(ret):
        ret[i] = r % num_attrs
    return ret

def bitset_to_features(bitset, num_attrs):
    '''
    @bitset set of ints, which are the index of elements that should be in 1.
    Converts this into an array of size self.attr_count, with the appropriate
    elements 1.
    '''
    features = []
    for i in range(num_attrs):
        if i in bitset:
            features.append(1.00)
        else:
            features.append(0.00)
    return features

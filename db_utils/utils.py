import numpy as np
import glob
import string
from moz_sql_parser import parse
import json
import pdb

import re

CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})"
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s"
GROUPBY_TEMPLATE = "SELECT {COLS}, COUNT(*) FROM {FROM_CLAUSE} GROUP BY {COLS}"
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"
ALIAS_FORMAT = "{TABLE} AS {ALIAS}"

REL_LOSS_EPSILON = 0.001
def compute_relative_loss(yhat, queries):
    '''
    as in the quicksel paper.
    '''
    ytrue = [s.true_sel for s in queries]
    error = 0.00
    for i, y in enumerate(ytrue):
        yh = yhat[i]
        error += abs(y - yh) / (max(REL_LOSS_EPSILON, y))
    error = error / len(yhat)
    return round(error * 100, 3)

def compute_abs_loss(yhat, queries):
    ytrue = [t.true_count for t in queries]
    yhat_abs = []
    for i, y in enumerate(yhat):
        yhat_abs.append(y*queries[i].total_count)
    errors = np.abs(np.array(yhat_abs) - np.array(ytrue))
    error = np.sum(errors)
    error = error / len(yhat)
    return round(error, 3)

def compute_qerror(yhat, queries):
    ytrue = [s.true_sel for s in queries]
    error = 0.00
    for i, y in enumerate(ytrue):
        yh = yhat[i]
        cur_error = max((yh / y), (y / yh))
        # if cur_error > 1.00:
            # print(cur_error, y, yh)
            # pdb.set_trace()
        error += cur_error
    error = error / len(yhat)
    return round(error, 3)

def pg_est_from_explain(output):
    '''
    '''
    est_vals = None
    for line in output:
        line = line[0]
        if "Seq Scan" in line or "Loop" in line or "Join" in line:
            for w in line.split():
                if "rows" in w and est_vals is None:
                    est_vals = int(re.findall("\d+", w)[0])
                    return est_vals

    assert False
    return None

def extract_join_clause(query):
    parsed_query = parse(query)
    ands = parsed_query["where"]["and"]
    join_clauses = []
    for i, pred in enumerate(ands):
        # FIXME: when will there be more than one key in pred?
        assert len(pred.keys()) == 1
        op_type = list(pred.keys())[0]
        columns = pred[op_type]
        if op_type != "eq":
            continue

        if not "." in columns[1]:
            continue

        join_clauses.append(columns[0] + " = " + columns[1])

    return join_clauses

def extract_predicate_columns(query):
    '''
    @ret: column names with predicate conditions in WHERE.
    Note: join conditions don't count as predicate conditions.
    '''
    predicate_cols = []
    parsed_query = parse(query)
    ands = parsed_query["where"]["and"]
    for i, pred in enumerate(ands):
        assert len(pred.keys()) == 1
        op_type = list(pred.keys())[0]
        columns = pred[op_type]
        if op_type != "eq":
            # maybe need to handle separately?
            print(pred)
            pdb.set_trace()

        if "." in columns[1]:
            # should be a join, skip this.
            continue

        if columns[0] in predicate_cols:
            # skip repeating columns
            continue
        predicate_cols.append(columns[0])

    return predicate_cols

def extract_from_clause(query):
    '''
    Extracts the from statement, and the relevant joins when there are multiple
    tables.
    '''
    froms = []
    parsed_query = parse(query)

    from_clause = parsed_query["from"]
    if isinstance(from_clause, str):
        # only one table.
        return [from_clause]

    assert isinstance(from_clause, list)
    for i, table in enumerate(from_clause):
        if isinstance(table, dict):
            alias = ALIAS_FORMAT.format(TABLE = table["value"],
                                ALIAS = table["name"])
            froms.append(alias)
        else:
            froms.append(table)

    return froms

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

import numpy as np
import glob
import string
from moz_sql_parser import parse
import json
import pdb
import re
import sqlparse
import itertools
import psycopg2 as pg
from utils.utils import *
import networkx as nx

CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})"
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s"
GROUPBY_TEMPLATE = "SELECT {COLS}, COUNT(*) FROM {FROM_CLAUSE} GROUP BY {COLS}"
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"

SELECT_ALL_COL_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL"
ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
MIN_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} ASC LIMIT 1"
MAX_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} DESC LIMIT 1"
UNIQUE_VALS_TEMPLATE = "SELECT COUNT(*) FROM (SELECT DISTINCT {COL} from {FROM_CLAUSE}) AS t"

def prepare_text(dat):
    cpy = BytesIO()
    for row in dat:
        cpy.write('\t'.join([repr(x) for x in row]) + '\n')
    return(cpy)

def prepare_binary(dat):
    pgcopy_dtype = [('num_fields','>i2')]
    for field, dtype in dat.dtype.descr:
        pgcopy_dtype += [(field + '_length', '>i4'),
                         (field, dtype.replace('<', '>'))]
    pgcopy = np.empty(dat.shape, pgcopy_dtype)
    pgcopy['num_fields'] = len(dat.dtype)
    for i in range(len(dat.dtype)):
        field = dat.dtype.names[i]
        pgcopy[field + '_length'] = dat.dtype[i].alignment
        pgcopy[field] = dat[field]
    cpy = BytesIO()
    cpy.write(pack('!11sii', b'PGCOPY\n\377\r\n\0', 0, 0))
    cpy.write(pgcopy.tostring())  # all rows
    cpy.write(pack('!h', -1))  # file trailer
    return(cpy)
'''
https://stackoverflow.com/questions/8144002/use-binary-copy-table-from-with-psycopg2/8150329#8150329

Need to actually figure out how to use it etc.
'''
def time_pgcopy(dat, table, binary):
    print('Processing copy object for ' + table)
    tstart = datetime.now()
    if binary:
        cpy = prepare_binary(dat)
    else:  # text
        cpy = prepare_text(dat)
    tendw = datetime.now()
    print('Copy object prepared in ' + str(tendw - tstart) + '; ' +
          str(cpy.tell()) + ' bytes; transfering to database')
    cpy.seek(0)
    if binary:
        curs.copy_expert('COPY ' + table + ' FROM STDIN WITH BINARY', cpy)
    else:  # text
        curs.copy_from(cpy, table)
    conn.commit()
    tend = datetime.now()
    print('Database copy time: ' + str(tend - tendw))
    print('        Total time: ' + str(tend - tstart))
    return

def pg_est_from_explain(output):
    '''
    '''
    est_vals = None
    for line in output:
        line = line[0]
        if "Seq Scan" in line or "Loop" in line or "Join" in line \
                or "Index Scan" in line or "Scan" in line:
            for w in line.split():
                if "rows" in w and est_vals is None:
                    est_vals = int(re.findall("\d+", w)[0])
                    return est_vals

    print("pg est failed!")
    print(output)
    pdb.set_trace()
    return 1.00

def extract_join_clause(query):
    parsed_query = parse(query)
    pred_vals = get_all_wheres(parsed_query)
    join_clauses = []
    for i, pred in enumerate(pred_vals):
        # FIXME: when will there be more than one key in pred?
        assert len(pred.keys()) == 1
        pred_type = list(pred.keys())[0]
        columns = pred[pred_type]
        if pred_type != "eq":
            continue

        if not "." in columns[1]:
            continue

        join_clauses.append(columns[0] + " = " + columns[1])

    return join_clauses

def get_all_wheres(parsed_query):
    pred_vals = []
    if "where" not in parsed_query:
        pass
    elif "and" not in parsed_query["where"]:
        # print(parsed_query)
        # print("and not in where!!!")
        # pdb.set_trace()
        pred_vals = [parsed_query["where"]]
    else:
        pred_vals = parsed_query["where"]["and"]
    return pred_vals

def extract_predicates(query):
    '''
    @ret:
        - column names with predicate conditions in WHERE.
        - predicate operator type (e.g., "in", "lte" etc.)
        - predicate value
    Note: join conditions don't count as predicate conditions.

    FIXME: temporary hack. For range queries, always returning key
    "lt", and vals for both the lower and upper bound
    '''
    def parse_lt_column(pred, cur_pred_type):
        # TODO: generalize more.
        for obj in pred[cur_pred_type]:
            if isinstance(obj, str):
                assert "." in obj
                column = obj
            elif isinstance(obj, dict):
                assert "literal" in obj
                val = obj["literal"]
            else:
                assert False
        assert column is not None
        assert val is not None
        return column, val

    predicate_cols = []
    predicate_types = []
    predicate_vals = []
    parsed_query = parse(query)
    pred_vals = get_all_wheres(parsed_query)

    for i, pred in enumerate(pred_vals):
        assert len(pred.keys()) == 1
        pred_type = list(pred.keys())[0]
        if pred_type == "eq":
            columns = pred[pred_type]
            if "." in columns[1]:
                # should be a join, skip this.
                continue

            if columns[0] in predicate_cols:
                # skip repeating columns
                continue
            predicate_types.append(pred_type)
            predicate_cols.append(columns[0])
            predicate_vals.append(columns[1])
        elif pred_type == "lte":
            continue
        elif pred_type == "lt":
            # this should technically work for both "lt", "lte", "gt" etc.
            column, val = parse_lt_column(pred, pred_type)

            # find the matching lte
            pred_lte = None
            # fml, shitty hacks.
            for pred2 in pred_vals:
                pred2_type = list(pred2.keys())[0]
                if pred2_type == "lte":
                    column2, val2 = parse_lt_column(pred2, pred2_type)
                    if column2 == column:
                        pred_lte = pred2
                        break
            assert pred_lte is not None
            # if pred_lte is None:
                # print(pred_vals)
                # pdb.set_trace()

            predicate_types.append(pred_type)
            predicate_cols.append(column)
            predicate_vals.append((val, val2))

        elif pred_type == "in":
            column = pred[pred_type][0]
            vals = pred[pred_type][1]
            # print(vals)
            # pdb.set_trace()
            if isinstance(vals, dict):
                vals = vals["literal"]
            if not isinstance(vals, list):
                vals = [vals]

            predicate_types.append(pred_type)
            predicate_cols.append(column)
            predicate_vals.append(vals)
        else:
            assert False, "unsupported predicate type"

    return predicate_cols, predicate_types, predicate_vals

def extract_from_clause(query):
    '''
    Extracts the from statement, and the relevant joins when there are multiple
    tables.
    '''
    def handle_table(table):
        if isinstance(table, dict):
            alias = ALIAS_FORMAT.format(TABLE = table["value"],
                                ALIAS = table["name"])
            froms.append(alias)
            aliases[table["name"]] = table["value"]
            tables.append(table["value"])
        else:
            froms.append(table)
            tables.append(table)

    froms = []
    # key: alias, val: table name
    aliases = {}
    # just table names
    tables = []

    parsed_query = parse(query)
    from_clause = parsed_query["from"]
    if isinstance(from_clause, str):
        # only one table.
        # return [from_clause]
        handle_table(from_clause)
    else:
        assert isinstance(from_clause, list)
        for i, table in enumerate(from_clause):
            handle_table(table)

    return froms, aliases, tables

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

def find_all_tables_till_keyword(token):
    tables = []
    # print("fattk: ", token)
    index = 0
    while (True):
        if (type(token) == sqlparse.sql.Comparison):
            left = token.left
            right = token.right
            if (type(left) == sqlparse.sql.Identifier):
                tables.append(left.get_parent_name())
            if (type(right) == sqlparse.sql.Identifier):
                tables.append(right.get_parent_name())
            break
        elif (type(token) == sqlparse.sql.Identifier):
            tables.append(token.get_parent_name())
            break
        try:
            index, token = token.token_next(index)
            if ("Literal" in str(token.ttype)) or token.is_keyword:
                break
        except:
            break

    return tables

def find_next_match(tables, wheres, index):
    '''
    ignore everything till next
    '''
    match = ""
    _, token = wheres.token_next(index)
    if token is None:
        return None, None
    # FIXME: is this right?
    if token.is_keyword:
        index, token = wheres.token_next(index)

    tables_in_pred = find_all_tables_till_keyword(token)
    assert len(tables_in_pred) <= 2

    token_list = sqlparse.sql.TokenList(wheres)

    while True:
        index, token = token_list.token_next(index)
        if token is None:
            break
        # print("token.value: ", token.value)
        if token.value == "AND":
            break

        match += " " + token.value

        if (token.value == "BETWEEN"):
            # ugh..
            index, a = token_list.token_next(index)
            index, AND = token_list.token_next(index)
            index, b = token_list.token_next(index)
            match += " " + a.value
            match += " " + AND.value
            match += " " + b.value
            break

    # print("tables: ", tables)
    # print("match: ", match)
    # print("tables in pred: ", tables_in_pred)
    for table in tables_in_pred:
        if table not in tables:
            # print(tables)
            # print(table)
            # pdb.set_trace()
            # print("returning index, None")
            return index, None

    if len(tables_in_pred) == 0:
        return index, None

    return index, match

def find_all_clauses(tables, wheres):
    matched = []
    # print(tables)
    index = 0
    while True:
        index, match = find_next_match(tables, wheres, index)
        # print("got index, match: ", index)
        # print(match)
        if match is not None:
            matched.append(match)
        if index is None:
            break

    # print("tables: ", tables)
    # print("matched: ", matched)
    # print("all possible matches: " )
    # for w in str(wheres).split("\n"):
        # for t in tables:
            # if t in w:
                # print(w)
    # print("where: ", wheres)
    # pdb.set_trace()
    # if len(tables) == 2:
        # if (tables[0] == "ct" and tables[1] == "mc"):
            # print(matched)
            # pdb.set_trace()

    return matched

def get_join_graph(joins, tables=None):
    join_graph = nx.Graph()
    for j in joins:
        j1 = j.split("=")[0]
        j2 = j.split("=")[1]
        t1 = j1[0:j1.find(".")].strip()
        t2 = j2[0:j2.find(".")].strip()
        if tables is not None:
            try:
                assert t1 in tables
                assert t2 in tables
            except:
                print(t1, t2)
                print(tables)
                print(joins)
                print("table not in tables!")
                pdb.set_trace()
        join_graph.add_edge(t1, t2)
    return join_graph

def _gen_subqueries(all_tables, wheres):
    '''
    my old shitty sqlparse code that should be updated...
    @tables: list
    @wheres: sqlparse object
    '''
    all_subqueries = []
    combs = []
    for i in range(1, len(all_tables)+1):
        combs += itertools.combinations(list(range(len(all_tables))), i)
    # print("num combs: ", len(combs))
    for comb in combs:
        cur_tables = []
        for i, idx in enumerate(comb):
            cur_tables.append(all_tables[idx])

        matches = find_all_clauses(cur_tables, wheres)
        # print("matches: ", matches)
        # pdb.set_trace()
        cond_string = " AND ".join(matches)
        if cond_string != "":
            cond_string = " WHERE " + cond_string

        # need to handle joins: if there are more than 1 table in tables, then
        # the predicates must include a join in between them
        if len(cur_tables) > 1:
            all_joins = True
            for ctable in cur_tables:
                joined = False
                for match in matches:
                    # FIXME: so hacky ugh. more band-aid
                    if match.count(".") == 2 \
                            and "=" in match:
                        if (" " + ctable + "." in " " + match):
                            joined = True
                if not joined:
                    all_joins = False
                    break
            if not all_joins:
                continue
        from_clause = " , ".join(cur_tables)
        query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause) + cond_string
        # final sanity checks
        joins = extract_join_clause(query)
        _,_, tables = extract_from_clause(query)

        # TODO: maybe this should be done somewhere earlier in the pipeline?
        join_graph = nx.Graph()
        for j in joins:
            j1 = j.split("=")[0]
            j2 = j.split("=")[1]
            t1 = j1[0:j1.find(".")].strip()
            t2 = j2[0:j2.find(".")].strip()
            try:
                assert t1 in tables
                assert t2 in tables
            except:
                print(t1, t2)
                print(tables)
                print(joins)
                print("table not in tables!")
                pdb.set_trace()
            join_graph.add_edge(t1, t2)
        if len(joins) > 0 and not nx.is_connected(join_graph):
            print("skipping query!")
            print(tables)
            print(joins)
            # pdb.set_trace()
            continue
        all_subqueries.append(query)
        # print("num subqueries: ", len(all_subqueries))

    print("num generated sql subqueries: ", len(all_subqueries))
    return all_subqueries

def gen_all_subqueries(query):
    '''
    @query: sql string.
    @ret: [sql strings], that represent all subqueries excluding cross-joins.
    FIXME: mix-match of moz_sql_parser AND sqlparse...
    '''
    # print("gen all subqueries!")
    # print(query)
    _,_,tables = extract_from_clause(query)
    parsed = sqlparse.parse(query)[0]
    # print(tables)
    # let us go over all the where clauses
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    assert where_clauses is not None
    # print(where_clauses)
    # pdb.set_trace()
    all_subqueries = _gen_subqueries(tables, where_clauses)
    return all_subqueries

def cached_execute_query(sql, user, db_host, port, pwd, db_name,
        execution_cache_threshold, sql_cache=None, timeout=120000):
    '''
    @timeout:
    executes the given sql on the DB, and caches the results in a
    persistent store if it took longer than self.execution_cache_threshold.
    '''
    hashed_sql = deterministic_hash(sql)
    if sql_cache is not None and hashed_sql in sql_cache:
        print("loaded {} from in memory cache".format(hashed_sql))
        return sql_cache[hashed_sql]

    # archive only considers the stuff stored in disk
    if sql_cache is not None and hashed_sql in sql_cache.archive:
        # load it and return
        print("loaded {} from cache".format(hashed_sql))
        # pdb.set_trace()
        return sql_cache.archive[hashed_sql]
    start = time.time()

    con = pg.connect(user=user, host=db_host, port=port,
            password=pwd, database=db_name)
    cursor = con.cursor()
    if timeout is not None:
        cursor.execute("SET statement_timeout = {}".format(timeout))
    try:
        cursor.execute(sql)
    except Exception as e:
        print("query failed to execute: ", sql)
        print(e)
        cursor.execute("ROLLBACK")
        con.commit()
        cursor.close()
        con.close()
        print("returning arbitrary large value for now")
        return [[10000000]]
        # return None
    exp_output = cursor.fetchall()
    cursor.close()
    con.close()
    end = time.time()
    if (end - start > execution_cache_threshold) \
            and sql_cache is not None:
        sql_cache[hashed_sql] = exp_output
    return exp_output

def _get_total_count_query(sql):
    '''
    @ret: sql query.
    '''
    froms, _, _ = extract_from_clause(sql)
    # FIXME: should be able to store this somewhere and not waste
    # re-executing it always
    from_clause = " , ".join(froms)
    joins = extract_join_clause(sql)
    join_clause = ' AND '.join(joins)
    if len(join_clause) > 0:
        from_clause += " WHERE " + join_clause
    count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    return count_query
    # total_count = self.execute(count_query, timeout=SUBQUERY_TIMEOUT)
    # if total_count is None:
        # return total_count
    # return total_count[0][0]

def sql_to_query_object(sql, user, db_host, port, pwd, db_name,
        total_count=None,execution_cache_threshold=None,
        sql_cache=None, timeout=None):
    '''
    @sql: string sql.
    @ret: Query object with all fields appropriately initialized.
          If it fails anywhere, then return None.
    '''
    if execution_cache_threshold is None:
        execution_cache_threshold = 60
    # print("sql to query sample")
    # print(sql)
    output = cached_execute_query(sql, user, db_host, port, pwd, db_name,
            execution_cache_threshold, sql_cache, timeout)
    if output is None:
        return None
    # from query string, to Query object
    true_val = output[0][0]
    # print("true_val: ", true_val)
    exp_query = "EXPLAIN " + sql
    exp_output = cached_execute_query(exp_query, user, db_host, port, pwd, db_name,
            execution_cache_threshold, sql_cache, timeout)
    if exp_output is None:
        return None
    pg_est = pg_est_from_explain(exp_output)
    # print("pg_est: ", pg_est)

    if total_count is None:
        total_count_query = _get_total_count_query(sql)
        exp_output = cached_execute_query(total_count_query, user, db_host, port, pwd, db_name,
                execution_cache_threshold, sql_cache, timeout)
        if exp_output is None:
            return None
        total_count = exp_output[0][0]

    # need to extract predicate columns, predicate operators, and predicate
    # values now.
    pred_columns, pred_types, pred_vals = extract_predicates(sql)

    from cardinality_estimation.query import Query
    query = Query(sql, pred_columns, pred_vals, pred_types,
            true_val, total_count, pg_est)
    return query
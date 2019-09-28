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
import klepto
import getpass
import os
import subprocess as sp
import time
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

import networkx as nx
from networkx.drawing.nx_agraph import write_dot,graphviz_layout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TIMEOUT_COUNT_CONSTANT = 150001001

CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})"
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s"

NTILE_CLAUSE = "ntile({BINS}) OVER (ORDER BY {COLUMN}) AS {ALIAS}"
GROUPBY_TEMPLATE = "SELECT {COLS}, COUNT(*) FROM {FROM_CLAUSE} GROUP BY {COLS}"
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"

SELECT_ALL_COL_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL"
ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
MIN_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} ASC LIMIT 1"
MAX_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} DESC LIMIT 1"
UNIQUE_VALS_TEMPLATE = "SELECT DISTINCT {COL} FROM {FROM_CLAUSE}"
UNIQUE_COUNT_TEMPLATE = "SELECT COUNT(*) FROM (SELECT DISTINCT {COL} from {FROM_CLAUSE}) AS t"

INDEX_LIST_CMD = """
select
    t.relname as table_name,
    a.attname as column_name,
    i.relname as index_name
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'
   -- and t.relname like 'mytable'
order by
    t.relname,
    i.relname;"""


RANGE_PREDS = ["gt", "gte", "lt", "lte"]

CREATE_INDEX_TMP = '''CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {TABLE} ({COLUMN});'''

def _find_all_tables(plan):
    '''
    '''
    # find all the scan nodes under the current level, and return those
    table_names = extract_values(plan, "Relation Name")
    table_names.sort()
    return table_names

def plot_graph_explain(G, base_table_nodes, join_nodes, fn, title="test"):
    NODE_SIZE = 300
    plt.title(title)
    pos = graphviz_layout(G, prog='dot')
    # first draw just the base tables
    nx.draw_networkx_nodes(G, pos,
               nodelist=base_table_nodes,
               node_color='b',
               node_size=NODE_SIZE,
               alpha=0.2)

    nx.draw_networkx_nodes(G, pos,
               nodelist=join_nodes,
               node_color='r',
               node_size=NODE_SIZE,
               alpha=0.2)


    node_labels = {}
    for n in G.nodes():
        if len(G.nodes[n]["tables"]) == 1:
            node_labels[n] = n
        else:
            node_labels[n] = n

        nx.draw_networkx_labels(G, pos, node_labels, font_size=8)

    nx.draw_networkx_edges(G,pos,width=1.0,
            alpha=0.5,with_labels=False)
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

def explain_to_nx(explain):
    '''
    '''
    # JOIN_KEYS = ["Hash Join", "Nested Loop", "Join"]
    base_table_nodes = []
    join_nodes = []

    def _get_node_name(tables):
        name = ""
        if len(tables) > 1:
            name = str(deterministic_hash(str(tables)))[0:5]
            join_nodes.append(name)
        else:
            name = tables[0]
            # shorten it
            name = "".join([n[0] for n in name.split("_")])
            if name in base_table_nodes:
                name = name + "2"
            base_table_nodes.append(name)
        return name

    def traverse(obj):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            if "Plans" in obj:
                if len(obj["Plans"]) == 2:
                    # these are all the joins
                    left_tables = _find_all_tables(obj["Plans"][0])
                    right_tables = _find_all_tables(obj["Plans"][1])
                    all_tables = left_tables + right_tables
                    node_type = obj["Node Type"]
                    all_tables.sort()

                    node0 = _get_node_name(left_tables)
                    node1 = _get_node_name(right_tables)
                    node_new = _get_node_name(all_tables)
                    # print(left_tables)
                    # print(right_tables)
                    # print(obj.keys())
                    # print(obj["Node Type"])
                    # print("cost: ", obj["Total Cost"])
                    # # print("time: ", obj["Actual Total Time"])
                    # # print("actual rows: ", obj["Actual Rows"])
                    # print("plan rows: ", obj["Plan Rows"])
                    # print("width: ", obj["Plan Width"])
                    # # print("Actual Loops: ", obj["Actual Loops"])
                    # print("parallel: ", obj["Parallel Aware"])
                    # if left_tables[0] == "movie_info" and right_tables[0] == "title":
                        # print("movie info and title")
                        # pdb.set_trace()
                    # print("left plan rows: ", obj["Plans"][0]["Plan Rows"])
                    # print("right plan rows: ", obj["Plans"][1]["Plan Rows"])
                    # print("left parallel: ", obj["Plans"][0]["Parallel Aware"])
                    # print("right plan rows: ", obj["Plans"][1]["Parallel Aware"])
                    # assert not obj["Plans"][0]["Parallel Aware"]
                    # assert not obj["Plans"][1]["Parallel Aware"]

                    # pdb.set_trace()

                    # update graph
                    # G.add_edge(node0, node1)
                    G.add_edge(node0, node_new)
                    G.add_edge(node1, node_new)
                    # add other parameters on the nodes
                    G.nodes[node0]["tables"] = left_tables
                    G.nodes[node1]["tables"] = right_tables
                    G.nodes[node_new]["tables"] = all_tables

            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    traverse(v)
                # print(k)

        elif isinstance(obj, list):
            for item in obj:
                traverse(item)

    G = nx.DiGraph()
    traverse(explain)
    G.base_table_nodes = base_table_nodes
    G.join_nodes = join_nodes
    return G

def benchmark_sql(sql, user, db_host, port, pwd, db_name,
        join_collapse_limit):
    '''
    TODO: should we be doing anything smarter?
    '''
    query_tmp = "SET join_collapse_limit={jcl}; {sql}"
    sql = query_tmp.format(jcl=join_collapse_limit, sql=sql)
    # first drop cache
    # FIXME: choose the right file automatically
    drop_cache_cmd = "./drop_cache.sh"
    p = sp.Popen(drop_cache_cmd, shell=True)
    p.wait()
    time.sleep(2)

    os_user = getpass.getuser()
    # con = pg.connect(user=user, port=port,
            # password=pwd, database=db_name, host=db_host)

    if os_user == "ubuntu":
        # for aws
        # con = pg.connect(user=user, port=port,
                # password=pwd, database=db_name)
        print(user, db_host, port, pwd, db_name)
        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)
    else:
        # for chunky
        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)

    cursor = con.cursor()
    start = time.time()
    cursor.execute(sql)

    exec_time = time.time() - start

    output = cursor.fetchall()
    cursor.close()
    con.close()

    return output, exec_time

def visualize_query_plan(sql, db_name, out_name_suffix):
    '''
    '''
    if "EXPLAIN" not in sql:
        sql = "EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON) " + sql
    # first drop cache
    drop_cache_cmd = "./drop_cache.sh"
    p = sp.Popen(drop_cache_cmd, shell=True)
    p.wait()
    time.sleep(2)

    tmp_fn = "./explain/test_" + out_name_suffix + ".sql"
    with open(tmp_fn, "w") as f:
        f.write(sql)
    json_out = "./explain/analyze_" + out_name_suffix + ".json"
    psql_cmd = "psql -d {} -qAt -f {} > {}".format(db_name, tmp_fn, json_out)

    p = sp.Popen(psql_cmd, shell=True)
    p.wait()

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
    '''
    FIXME: this can be optimized further / or made to handle more cases
    '''
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    start = time.time()
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    if where_clauses is None:
        return []
    join_clauses = []

    froms, aliases, table_names = extract_from_clause(query)
    if len(aliases) > 0:
        tables = [k for k in aliases]
    else:
        tables = table_names
    matches = find_all_clauses(tables, where_clauses)
    for match in matches:
        if "=" not in match:
            continue
        if "<=" in match or ">=" in match:
            continue

        match = match.replace(";", "")
        left, right = match.split("=")
        # ugh dumb hack
        if "." in right:
            # must be a join, so add it.
            join_clauses.append(left.strip() + " = " + right.strip())

    # print("extract join clauses took ", time.time() - start)
    return join_clauses

def get_all_wheres(parsed_query):
    pred_vals = []
    if "where" not in parsed_query:
        pass
    elif "and" not in parsed_query["where"]:
        pred_vals = [parsed_query["where"]]
    else:
        pred_vals = parsed_query["where"]["and"]
    return pred_vals

def extract_predicates2(query):
    '''
    @ret:
        - column names with predicate conditions in WHERE.
        - predicate operator type (e.g., "in", "lte" etc.)
        - predicate value
    Note: join conditions don't count as predicate conditions.

    FIXME: temporary hack. For range queries, always returning key
    "lt", and vals for both the lower and upper bound
    '''
    predicate_cols = []
    predicate_types = []
    predicate_vals = []
    if "::float" in query:
        query = query.replace("::float", "")
    elif "::int" in query:
        query = query.replace("::int", "")
    # really fucking dumb
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")

    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    start = time.time()
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    if where_clauses is None:
        assert False
        return [], [], []

    froms, aliases, table_names = extract_from_clause(query)
    if len(aliases) > 0:
        tables = [k for k in aliases]
    else:
        tables = table_names
    matches = find_all_clauses(tables, where_clauses)

    print(matches)
    print(where_clauses)
    pdb.set_trace()

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
    def parse_column(pred, cur_pred_type):
        '''
        gets the name of the column, and whether column location is on the left
        (0) or right (1)
        '''
        for i, obj in enumerate(pred[cur_pred_type]):
            assert i <= 1
            if isinstance(obj, str) and "." in obj:
                # assert "." in obj
                column = obj
            elif isinstance(obj, dict):
                assert "literal" in obj
                val = obj["literal"]
                val_loc = i
            else:
                val = obj
                val_loc = i

        assert column is not None
        assert val is not None
        return column, val_loc, val

    def _parse_predicate(pred, pred_type):
        if pred_type == "eq":
            columns = pred[pred_type]
            if len(columns) <= 1:
                return None
            # FIXME: more robust handling?
            if "." in str(columns[1]):
                # should be a join, skip this.
                # Note: joins only happen in "eq" predicates
                return None
            predicate_types.append(pred_type)
            predicate_cols.append(columns[0])
            predicate_vals.append(columns[1])

        elif pred_type in RANGE_PREDS:
            vals = [None, None]
            col_name, val_loc, val = parse_column(pred, pred_type)
            vals[val_loc] = val

            # this loop may find no matching predicate for the other side, in
            # which case, we just leave the val as None
            for pred2 in pred_vals:
                pred2_type = list(pred2.keys())[0]
                if pred2_type in RANGE_PREDS:
                    col_name2, val_loc2, val2 = parse_column(pred2, pred2_type)
                    if col_name2 == col_name:
                        # assert val_loc2 != val_loc
                        if val_loc2 == val_loc:
                            # same predicate as pred
                            continue
                        vals[val_loc2] = val2
                        break

            predicate_types.append("lt")
            predicate_cols.append(col_name)
            if "g" in pred_type:
                # reverse vals, since left hand side now means upper bound
                vals.reverse()
            predicate_vals.append(vals)

        elif pred_type == "between":
            # we just treat it as a range query
            col = pred[pred_type][0]
            val1 = pred[pred_type][1]
            val2 = pred[pred_type][2]
            vals = [val1, val2]
            predicate_types.append("lt")
            predicate_cols.append(col)
            predicate_vals.append(vals)
        elif pred_type == "in" \
                or "like" in pred_type:
            # includes preds like, ilike, nlike etc.
            column = pred[pred_type][0]
            # what if column has been seen before? Will just be added again to
            # the list of predicates, which is the correct behaviour
            vals = pred[pred_type][1]
            if isinstance(vals, dict):
                vals = vals["literal"]
            if not isinstance(vals, list):
                vals = [vals]
            predicate_types.append(pred_type)
            predicate_cols.append(column)
            predicate_vals.append(vals)
        # elif pred_type == "or":
            # _parse_predicate(pred, pred_type)
            # for pred2 in pred[pred_type]:
                # pdb.set_trace()

        else:
            # assert False
            # TODO: need to support "OR" statements
            return None
            # assert False, "unsupported predicate type"

    start = time.time()
    predicate_cols = []
    predicate_types = []
    predicate_vals = []
    if "::float" in query:
        query = query.replace("::float", "")
    elif "::int" in query:
        query = query.replace("::int", "")
    # really fucking dumb
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")

    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    try:
        parsed_query = parse(query)
    except:
        print(query)
        print("moz sql parser failed to parse this!")
        pdb.set_trace()
    pred_vals = get_all_wheres(parsed_query)

    for i, pred in enumerate(pred_vals):
        try:
            assert len(pred.keys()) == 1
        except:
            print(pred)
            pdb.set_trace()
        pred_type = list(pred.keys())[0]
        # print(pred_type)
        # pdb.set_trace()
        if pred == "or" or pred == "OR":
            continue
        _parse_predicate(pred, pred_type)

    # print("extract predicate cols done!")
    # print("extract predicates took ", time.time() - start)
    return predicate_cols, predicate_types, predicate_vals

def extract_from_clause(query):
    '''
    Optimized version using sqlparse.
    Extracts the from statement, and the relevant joins when there are multiple
    tables.
    @ret: froms:
          froms: [alias1, alias2, ...] OR [table1, table2,...]
          aliases:{alias1: table1, alias2: table2} (OR [] if no aliases present)
          tables: [table1, table2, ...]
    '''
    def handle_table(identifier):
        table_name = identifier.get_real_name()
        alias = identifier.get_alias()
        tables.append(table_name)
        if alias is not None:
            from_clause = ALIAS_FORMAT.format(TABLE = table_name,
                                ALIAS = alias)
            froms.append(from_clause)
            aliases[alias] = table_name
        else:
            froms.append(table_name)

    start = time.time()
    froms = []
    # key: alias, val: table name
    aliases = {}
    # just table names
    tables = []

    start = time.time()
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    from_token = None
    from_seen = False
    for token in parsed.tokens:
        # print(type(token))
        # print(token)
        if from_seen:
            if isinstance(token, IdentifierList) or isinstance(token,
                    Identifier):
                from_token = token
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True

    assert from_token is not None
    if isinstance(from_token, IdentifierList):
        for identifier in from_token.get_identifiers():
            handle_table(identifier)
    elif isinstance(from_token, Identifier):
        handle_table(from_token)
    else:
        assert False

    # print("extract froms parse took: ", time.time() - start)

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
            # ugh ugliness
            index, a = token_list.token_next(index)
            index, AND = token_list.token_next(index)
            index, b = token_list.token_next(index)
            match += " " + a.value
            match += " " + AND.value
            match += " " + b.value
            # Note: important not to break here! Will break when we hit the
            # "AND" in the next iteration.

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

def _gen_subqueries(all_tables, wheres, aliases):
    '''
    my old shitty sqlparse code that should be updated...
    @tables: list
    @wheres: sqlparse object
    '''
    # FIXME: nicer setup
    if len(aliases) > 0:
        all_tables = [a for a in aliases]

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
        if len(aliases) > 0:
            aliased_tables = [ALIAS_FORMAT.format(TABLE=aliases[a], ALIAS=a) for a in cur_tables]
            from_clause = " , ".join(aliased_tables)
            # print(from_clause)
        else:
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
                assert t1 in tables or t1 in aliases
                assert t2 in tables or t2 in aliases
            except:
                print(t1, t2)
                print(tables)
                print(joins)
                print("table not in tables!")
                pdb.set_trace()
            join_graph.add_edge(t1, t2)
        if len(joins) > 0 and not nx.is_connected(join_graph):
            # print("skipping query!")
            # print(tables)
            # print(joins)
            # pdb.set_trace()
            continue
        all_subqueries.append(query)

    return all_subqueries

def nx_graph_to_query(G):
    froms = []
    conds = []
    for nd in G.nodes(data=True):
        node = nd[0]
        data = nd[1]
        if "real_name" in data:
            froms.append(ALIAS_FORMAT.format(TABLE=data["real_name"],
                                             ALIAS=node))
        else:
            froms.append(node)

        for pred in data["predicates"]:
            conds.append(pred)

    for edge in G.edges(data=True):
        conds.append(edge[2]['join_condition'])

    # preserve order for caching
    froms.sort()
    conds.sort()
    from_clause = " , ".join(froms)
    if len(conds) > 0:
        wheres = ' AND '.join(conds)
        from_clause += " WHERE " + wheres
    count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    return count_query

def _gen_subqueries_nx(query):
    start = time.time()
    froms,aliases,tables = extract_from_clause(query)
    joins = extract_join_clause(query)
    pred_columns, pred_types, pred_vals = extract_predicates(query)
    join_graph = nx.Graph()
    for j in joins:
        j1 = j.split("=")[0]
        j2 = j.split("=")[1]
        t1 = j1[0:j1.find(".")].strip()
        t2 = j2[0:j2.find(".")].strip()
        try:
            assert t1 in tables or t1 in aliases
            assert t2 in tables or t2 in aliases
        except:
            print(t1, t2)
            print(tables)
            print(joins)
            print("table not in tables!")
            pdb.set_trace()

        join_graph.add_edge(t1, t2)
        join_graph[t1][t2]["join_condition"] = j
        if t1 in aliases:
            table1 = aliases[t1]
            table2 = aliases[t2]

            join_graph.nodes()[t1]["real_name"] = table1
            join_graph.nodes()[t2]["real_name"] = table2

    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    assert where_clauses is not None

    for t1 in join_graph.nodes():
        tables = [t1]
        matches = find_all_clauses(tables, where_clauses)
        join_graph.nodes()[t1]["predicates"] = matches

    # TODO: Next, need an efficient way to generate all connected subgraphs, and
    # then convert each of them to a sql queries
    all_subqueries = []

    # find all possible subsets of the nodes
    combs = []
    all_nodes = list(join_graph.nodes())
    for i in range(1, len(all_nodes)+1):
        combs += itertools.combinations(list(range(len(all_nodes))), i)

    for node_idxs in combs:
        nodes = [all_nodes[idx] for idx in node_idxs]
        subg = join_graph.subgraph(nodes)
        if nx.is_connected(subg):
            sql_str = nx_graph_to_query(subg)
            all_subqueries.append(sql_str)

    print("num subqueries: ", len(all_subqueries))
    print("took: ", time.time() - start)

    return all_subqueries

def gen_all_subqueries(query):
    '''
    @query: sql string.
    @ret: [sql strings], that represent all subqueries excluding cross-joins.
    FIXME: mix-match of moz_sql_parser AND sqlparse...
    '''
    start = time.time()
    all_subqueries = _gen_subqueries_nx(query)
    return all_subqueries

def cached_execute_query(sql, user, db_host, port, pwd, db_name,
        execution_cache_threshold, sql_cache_dir=None, timeout=120000):
    '''
    @timeout:
    @db_host: going to ignore it so default localhost is used.
    executes the given sql on the DB, and caches the results in a
    persistent store if it took longer than self.execution_cache_threshold.
    '''
    sql_cache = None
    if sql_cache_dir is not None:
        assert isinstance(sql_cache_dir, str)
        sql_cache = klepto.archives.dir_archive(sql_cache_dir,
                cached=True, serialized=True)

    hashed_sql = deterministic_hash(sql)

    # archive only considers the stuff stored in disk
    if sql_cache is not None and hashed_sql in sql_cache.archive:
        return sql_cache.archive[hashed_sql], False

    start = time.time()

    os_user = getpass.getuser()
    if os_user == "ubuntu":
        # for aws
        con = pg.connect(user=user, port=port,
                password=pwd, database=db_name)
    else:
        # for chunky
        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)

    cursor = con.cursor()
    if timeout is not None:
        cursor.execute("SET statement_timeout = {}".format(timeout))
    try:
        cursor.execute(sql)
    except Exception as e:
        # print("query failed to execute: ", sql)
        # FIXME: better way to do this.
        cursor.execute("ROLLBACK")
        con.commit()
        cursor.close()
        con.close()

        if not "timeout" in str(e):
            print("failed to execute for reason other than timeout")
            print(e)
            print(sql)
            pdb.set_trace()
        else:
            print("failed because of timeout!")
            return None, True

        return None, False

    exp_output = cursor.fetchall()
    cursor.close()
    con.close()
    end = time.time()
    if (end - start > execution_cache_threshold) \
            and sql_cache is not None:
        sql_cache.archive[hashed_sql] = exp_output
    return exp_output, False

def get_total_count_query(sql):
    '''
    @ret: sql query.
    '''
    froms, _, _ = extract_from_clause(sql)
    # FIXME: should be able to store this somewhere and not waste
    # re-executing it always
    from_clause = " , ".join(froms)
    joins = extract_join_clause(sql)
    if len(joins) < len(froms)-1:
        print("joins < len(froms)-1")
        print(sql)
        print(joins)
        print(len(joins))
        print(froms)
        print(len(froms))
        # pdb.set_trace()
    join_clause = ' AND '.join(joins)
    if len(join_clause) > 0:
        from_clause += " WHERE " + join_clause
    count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    # print("COUNT QUERY:\n", count_query)
    # pdb.set_trace()
    return count_query

def sql_to_query_object(sql, user, db_host, port, pwd, db_name,
        total_count=None,execution_cache_threshold=None,
        sql_cache=None, timeout=None, num_query=1):
    '''
    @sql: string sql.
    @ret: Query object with all fields appropriately initialized.
          If it fails anywhere, then return None.
    @execution_cache_threshold: In seconds, if query takes beyond this, then
    cache it.
    '''
    if num_query % 100 == 0:
        print("sql_to_query_object num query: ", num_query)

    if execution_cache_threshold is None:
        execution_cache_threshold = 60

    if "SELECT COUNT" not in sql:
        print("no SELECT COUNT in sql!")
        exit(-1)

    output, tout = cached_execute_query(sql, user, db_host, port, pwd, db_name,
            execution_cache_threshold, sql_cache, timeout)

    if tout:
        # just fix all vals to be same
        true_val = TIMEOUT_COUNT_CONSTANT
        pg_est = TIMEOUT_COUNT_CONSTANT
        total_count = TIMEOUT_COUNT_CONSTANT
        pred_columns, pred_types, pred_vals = extract_predicates(sql)

        from cardinality_estimation.query import Query
        query = Query(sql, pred_columns, pred_vals, pred_types,
                true_val, total_count, pg_est)
        return query
    else:
        if output is None:
            print("cached execute query returned None!!")
            exit(-1)
            # return None
        # from query string, to Query object
        true_val = output[0][0]

    exp_query = "EXPLAIN " + sql
    exp_output, tout  = cached_execute_query(exp_query, user, db_host, port, pwd, db_name,
            execution_cache_threshold, sql_cache, timeout)

    assert not tout

    if exp_output is None:
        return None
    pg_est = pg_est_from_explain(exp_output)

    # FIXME: start caching the true total count values
    if total_count is None:
        total_count_query = get_total_count_query(sql)

        # if we should just update value based on pg' estimate for total count
        # v/s finding true count
        TRUE_TOTAL_COUNT = False
        total_timeout = 180000
        if TRUE_TOTAL_COUNT:
            exp_output, _ = cached_execute_query(total_count_query, user, db_host, port, pwd, db_name,
                    execution_cache_threshold, sql_cache, total_timeout)
            if exp_output is None:
                # print("total count query timed out")
                # print(total_count_query)
                # execute it with explain
                exp_query = "EXPLAIN " + total_count_query
                exp_output, _ = cached_execute_query(exp_query, user, db_host, port, pwd, db_name,
                        execution_cache_threshold, sql_cache, total_timeout)
                if exp_output is None:
                    print("pg est was None for ")
                    print(exp_query)
                    pdb.set_trace()
                total_count = pg_est_from_explain(exp_output)
                print("pg total count est: ", total_count)
            else:
                total_count = exp_output[0][0]
                # print("total count: ", total_count)
        else:
            exp_query = "EXPLAIN " + total_count_query
            exp_output, _ = cached_execute_query(exp_query, user, db_host, port, pwd, db_name,
                    execution_cache_threshold, sql_cache, total_timeout)
            if exp_output is None:
                print("pg est was None for ")
                print(exp_query)
                pdb.set_trace()
            total_count = pg_est_from_explain(exp_output)
            # print("pg total count est: ", total_count)

    # need to extract predicate columns, predicate operators, and predicate
    # values now.
    pred_columns, pred_types, pred_vals = extract_predicates(sql)
    # pred_columns, pred_types, pred_vals = None, None, None

    from cardinality_estimation.query import Query
    query = Query(sql, pred_columns, pred_vals, pred_types,
            true_val, total_count, pg_est)
    return query

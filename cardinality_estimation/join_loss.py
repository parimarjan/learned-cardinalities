import psycopg2 as pg
import getpass
from db_utils.utils import *
from sql_rep.utils import extract_aliases, nodes_to_sql, path_to_join_order

import multiprocessing as mp
import math
from cardinality_estimation.flow_loss import *
# from cardinality_estimation.nn import get_subq_flows

import pdb
import klepto
import copy
import cvxpy as cp

import MySQLdb
import json

from park.envs.query_optimizer.query_optimizer import QueryOptEnv

system = platform.system()
if system == 'Linux':
    lib_file = "libflowloss.so"
    lib_dir = "./flow_loss_cpp"
    lib_file = lib_dir + "/" + lib_file
    fl_cpp = CDLL(lib_file, mode=RTLD_GLOBAL)
else:
    print("flow loss C library not being used as we are not on linux")
    # lib_file = "libflowloss.dylib"

DEBUG_SQL=""""SELECT COUNT(*) FROM keyword AS k STRAIGHT_JOIN movie_keyword AS
mk STRAIGHT_JOIN movie_companies AS mc STRAIGHT_JOIN company_name AS cn
STRAIGHT_JOIN title AS t STRAIGHT_JOIN kind_type AS kt STRAIGHT_JOIN cast_info
AS ci STRAIGHT_JOIN company_type AS ct STRAIGHT_JOIN role_type AS rt
STRAIGHT_JOIN name AS n WHERE  1950 < t.production_year AND  cn.country_code IN
('[ar]','[at]','[au]','[ca]','[de]','[es]','[fr]','[ie]','[in]','[jp]','[pl]','[tr]','[us]')
AND  ct.kind IN ('distributors','production companies') AND  k.keyword IN
('anal-sex','blood','dancing','female-frontal-nudity','female-nudity','lesbian','non-fiction','nudity','one-word-title','sex','suicide','tv-mini-series')
AND  kt.kind IN ('episode','movie','tv movie') AND  n.gender IN ('f','m') AND
rt.role IN ('cinematographer','production designer') AND  t.production_year <=
1990 AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND
ci.person_id = n.id AND ci.role_id = rt.id AND cn.id = mc.company_id AND ct.id
= mc.company_type_id AND k.id = mk.keyword_id AND kt.id = t.kind_id AND
mk.movie_id = mc.movie_id AND t.id = ci.movie_id AND t.id = mc.movie_id AND
t.id = mk.movie_id"""


PG_HINT_CMNT_TMP = '''/*+ {COMMENT} */'''
PG_HINT_JOIN_TMP = "{JOIN_TYPE} ({TABLES}) "
PG_HINT_CARD_TMP = "Rows ({TABLES} #{CARD}) "
PG_HINT_SCAN_TMP = "{SCAN_TYPE}({TABLE}) "
PG_HINT_LEADING_TMP = "Leading({JOIN_ORDER})"
PG_HINT_JOINS = {}
PG_HINT_JOINS["Nested Loop"] = "NestLoop"
PG_HINT_JOINS["Hash Join"] = "HashJoin"
PG_HINT_JOINS["Merge Join"] = "MergeJoin"

PG_HINT_SCANS = {}
PG_HINT_SCANS["Seq Scan"] = "SeqScan"
PG_HINT_SCANS["Index Scan"] = "IndexScan"
PG_HINT_SCANS["Index Only Scan"] = "IndexOnlyScan"
PG_HINT_SCANS["Bitmap Heap Scan"] = "BitmapScan"
PG_HINT_SCANS["Tid Scan"] = "TidScan"

MAX_JOINS = 16

MYSQL_CARD_FILE_NAME = "/tmp/query_cardinalities.json"
# MYSQL_CARD_FILE_NAME = "/tmp/query_cardinalities_no.json"

MYSQL_OPT_TMP = "set optimizer_switch='{FLAGS}';"
# MYSQL_OPT_FLAGS=""""""
MYSQL_OPT_FLAGS="""materialization=off,block_nested_loop=off,semijoin=off,subquery_materialization_cost_based=off,index_merge_union=off,index_merge_sort_union=off,prefer_ordering_index=off,loosescan=off,firstmatch=off,use_index_extensions=off"""

MYSQL_ROW_EVAL_COST_TMP = """UPDATE mysql.server_cost SET cost_value={val} WHERE cost_name='row_evaluate_cost';"""

# cursor.execute("SET materialization=off;")
# cursor.execute("SET block_nested_loop=off;")
# cursor.execute("SET semijoin=off;")
# cursor.execute("SET subquery_materialization_cost_based=off;")
# TMP = "set optimizer_switch='index_merge_union=off,index_merge_sort_union=off';

def set_indexes(cursor, val):
    cursor.execute("SET enable_indexscan = {}".format(val))
    cursor.execute("SET enable_seqscan = {}".format("on"))
    cursor.execute("SET enable_indexonlyscan = {}".format(val))
    cursor.execute("SET enable_bitmapscan = {}".format(val))
    cursor.execute("SET enable_tidscan = {}".format(val))

def set_cost_model(cursor, cost_model):
    # makes things easier to understand
    cursor.execute("SET enable_material = off")
    if cost_model == "hash_join":
        cursor.execute("SET enable_hashjoin = on")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = off")
        set_indexes(cursor, "off")
    elif cost_model == "nested_loop":
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        set_indexes(cursor, "off")
    elif "nested_loop_index8_debug" == cost_model:
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        cursor.execute("SET enable_indexscan = {}".format("on"))
        cursor.execute("SET enable_seqscan = {}".format("on"))

        # print("debug mode for nested loop index8")
        # cursor.execute("SET random_page_cost = 1.0")
        # cursor.execute("SET cpu_tuple_cost = 1.0")
        # cursor.execute("SET cpu_index_tuple_cost = 1.0")

        cursor.execute("SET enable_indexonlyscan = {}".format("off"))
        cursor.execute("SET enable_bitmapscan = {}".format("off"))
        cursor.execute("SET enable_tidscan = {}".format("off"))

    elif "nested_loop_index8" in cost_model:
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        cursor.execute("SET enable_indexscan = {}".format("on"))
        cursor.execute("SET enable_seqscan = {}".format("on"))

        # print("debug mode for nested loop index8")
        # cursor.execute("SET random_page_cost = 1.0")
        # cursor.execute("SET cpu_tuple_cost = 1.0")
        # cursor.execute("SET cpu_index_tuple_cost = 1.0")

        cursor.execute("SET enable_indexonlyscan = {}".format("off"))
        cursor.execute("SET enable_bitmapscan = {}".format("off"))
        cursor.execute("SET enable_tidscan = {}".format("off"))

    elif "nested_loop_index7" in cost_model:
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        cursor.execute("SET enable_indexscan = {}".format("on"))
        cursor.execute("SET enable_seqscan = {}".format("on"))

        cursor.execute("SET enable_indexonlyscan = {}".format("off"))
        cursor.execute("SET enable_bitmapscan = {}".format("off"))
        cursor.execute("SET enable_tidscan = {}".format("off"))

    elif "nested_loop_index" in cost_model:
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        set_indexes(cursor, "on")

    elif cost_model == "cm1" \
            or cost_model == "cm2":
        pass
    else:
        assert False

def get_pg_cost_from_sql(sql, cur):
    assert "explain" in sql
    # cur = con.cursor()
    cur.execute(sql)
    explain = cur.fetchall()
    all_costs = extract_values(explain[0][0][0], "Total Cost")
    mcost = max(all_costs)
    cost = explain[0][0][0]["Plan"]["Total Cost"]
    if cost != mcost:
        print("cost != mcost!")
        print(cost, mcost)
    return mcost, explain

def _gen_pg_hint_cards(cards):
    '''
    '''
    card_str = ""
    for aliases, card in cards.items():
        if isinstance(aliases, tuple):
            aliases = " ".join(aliases)
        card_line = PG_HINT_CARD_TMP.format(TABLES = aliases,
                                            CARD = card)
        card_str += card_line
    return card_str

def _gen_pg_hint_join(join_ops):
    '''
    '''
    join_str = ""
    for tables, join_op in join_ops.items():
        join_line = PG_HINT_JOIN_TMP.format(TABLES = tables,
                                            JOIN_TYPE = PG_HINT_JOINS[join_op])
        join_str += join_line
    return join_str

def _gen_pg_hint_scan(scan_ops):
    '''
    '''
    scan_str = ""
    for alias, scan_op in scan_ops.items():
        scan_line = PG_HINT_SCAN_TMP.format(TABLE = alias,
                                            SCAN_TYPE = PG_HINT_SCANS[scan_op])
        scan_str += scan_line
    return scan_str

def get_leading_hint(join_graph, explain):
    '''
    Ryan's implementation.
    '''
    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))

            if len(left) == 1 and len(right) == 1:
                left_alias = left[0][left[0].lower().find(" as ")+4:]
                right_alias = right[0][right[0].lower().find(" as ")+4:]
                return left_alias +  " " + right_alias

            if len(left) == 1:
                left_alias = left[0][left[0].lower().find(" as ")+4:]
                return left_alias + " (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                right_alias = right[0][right[0].lower().find(" as ")+4:]
                return "(" + __extract_jo(plan["Plans"][0]) + ") " + right_alias

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    jo = __extract_jo(explain[0][0][0]["Plan"])
    jo = "(" + jo + ")"
    return PG_HINT_LEADING_TMP.format(JOIN_ORDER = jo)

def get_pghint_modified_sql(sql, cardinalities, join_ops,
        leading_hint, scan_ops):
    '''
    @cardinalities: dict
    @join_ops: dict

    @ret: sql, augmented with appropriate comments.
    '''
    if "explain (format json)" not in sql:
        sql = " explain (format json) " + sql

    # sql = " explain (format json) " + sql

    comment_str = ""
    if cardinalities is not None:
        card_str = _gen_pg_hint_cards(cardinalities)
        # gen appropriate sql with comments etc.
        comment_str += card_str

    if join_ops is not None:
        join_str = _gen_pg_hint_join(join_ops)
        comment_str += join_str + " "
    if leading_hint is not None:
        comment_str += leading_hint + " "
    if scan_ops is not None:
        scan_str = _gen_pg_hint_scan(scan_ops)
        comment_str += scan_str + " "

    pg_hint_str = PG_HINT_CMNT_TMP.format(COMMENT=comment_str)
    sql = pg_hint_str + sql
    # sql = pg_hint_str + " " + sql
    return sql

def get_join_cost_sql(sql_order, est_cardinalities, true_cardinalities,
        join_graph, user, pwd, db_host, port, db_name, cost_model,
        scan_types):
    try:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd)
    except:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd, host=db_host)

    if len(scan_types) < 0:
        print("scan types < 0")
        print(scan_types)
        print(sql_order)

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    set_cost_model(cursor, cost_model)
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))

    # ALL this drama just so we can get leading_order... uglyugh
    cursor.execute("SET join_collapse_limit = {}".format(1))
    cursor.execute("SET from_collapse_limit = {}".format(1))
    sql_to_exec = " explain (format json) " + sql_order
    sql_to_exec = get_pghint_modified_sql(sql_to_exec, est_cardinalities,
            None, None, None)
    cursor.execute(sql_to_exec)
    explain = cursor.fetchall()

    # TODO: compare scan_ops w/ scan_types
    # print("going to call get_pg_join order from join_cost_sql")
    est_join_order_sql, est_join_ops, scan_ops = get_pg_join_order(join_graph,
            explain)
    leading_hint = get_leading_hint(join_graph, explain)
    est_opt_sql = nx_graph_to_query(join_graph,
            from_clause=est_join_order_sql)
    # TODO: compare scan_types

    # add the join ops etc. information
    cost_sql = get_pghint_modified_sql(est_opt_sql, true_cardinalities,
            est_join_ops, leading_hint, scan_types)

    est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)
    debug_leading = get_leading_hint(join_graph, est_explain)

    # print("going to call get_pg_join order from join_cost_sql2")
    _, cost_join_ops, cost_scan_ops = get_pg_join_order(join_graph,
            est_explain)

    # print("do join orders match: ", debug_leading == leading_hint)
    # for k,v in cost_join_ops.items():
        # if (v != est_join_ops[k]):
            # print(k, v, est_join_ops[k])

    # for k,v in cost_scan_ops.items():
        # assert v == scan_ops[k]
        # if k not in scan_types:
            # print("not in scan types: ", k, v, scan_ops[k])
        # elif (v != scan_types[k]):
            # print(k, v, scan_ops[k])
    # pdb.set_trace()

    exec_sql = get_pghint_modified_sql(est_opt_sql, est_cardinalities,
            est_join_ops, leading_hint, scan_types)

    cursor.close()
    con.close()
    return exec_sql, est_cost, est_explain

def get_cardinalities_join_cost(query, est_cardinalities, true_cardinalities,
        join_graph, cursor, sql_costs):

    est_card_sql = get_pghint_modified_sql(query, est_cardinalities, None,
            None, None)
    assert "explain" in est_card_sql.lower()
    # if "explain" not in est_card_sql.lower():
        # print(est_card_sql)

    cursor.execute(est_card_sql)
    explain = cursor.fetchall()

    # print("going to call get_pg_join order from join_cost")
    est_join_order_sql, est_join_ops, scan_ops = get_pg_join_order(join_graph,
            explain)
    leading_hint = get_leading_hint(join_graph, explain)
    assert "info" not in leading_hint

    est_opt_sql = nx_graph_to_query(join_graph,
            from_clause=est_join_order_sql)

    # add the join ops etc. information
    cost_sql = get_pghint_modified_sql(est_opt_sql, true_cardinalities,
            est_join_ops, leading_hint, scan_ops)

    # set this to sql to be executed, as pg_hint will enforce the estimated
    # cardinalities, and let postgres make decisions for join order and
    # everything about operators based on the estimated cardinalities
    exec_sql = get_pghint_modified_sql(est_opt_sql, est_cardinalities,
            None, None, None)

    # cost_sql will be seen often, as true_cardinalities remain fixed. so we
    # can cache the results for it.

    cost_sql_key = deterministic_hash(cost_sql)
    ## archived version
    if sql_costs is not None:
        if cost_sql_key in sql_costs.archive:
            try:
                est_cost, est_explain = sql_costs.archive[cost_sql_key]
            except:
                est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)
                sql_costs.archive[cost_sql_key] = (est_cost, est_explain)
        else:
            est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)
            sql_costs.archive[cost_sql_key] = (est_cost, est_explain)
    else:
        est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)

    # FIXME: turned off archive above, use once cost model is finalized
    # est_cost, est_explain = get_pg_cost_from_sql(cost_sql, cursor)

    debug_leading = get_leading_hint(join_graph, est_explain)
    if debug_leading != leading_hint:
        pass
        # print(est_opt_sql)
        # print("actual order:\n ", debug_leading)
        # print("wanted order:\n ", leading_hint)
        # print("est cost: ", est_cost)
        # pdb.set_trace()

    return exec_sql, est_cost, est_explain

def compute_join_order_loss_pg_single(queries, join_graphs, true_cardinalities,
        est_cardinalities, opt_costs, opt_explains, opt_sqls,
        use_indexes, user, pwd, db_host, port, db_name, use_archive,
        cost_model):
    '''
    @query: str
    @true_cardinalities:
        key:
            sort([table_1 / alias_1, ..., table_n / alias_n])
        val:
            float
    @est_cardinalities:
        key:
            sort([table_1 / alias_1, ..., table_n / alias_n])
        val:
            float
    '''
    try:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd)
    except:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd, host=db_host)

    if use_archive:
        if cost_model == "cm1":
            archive_fn = "/tmp/sql_costs"
        else:
            archive_fn = "/tmp/sql_costs_" + cost_model
        sql_costs_archive = klepto.archives.dir_archive(archive_fn,
                cached=True, serialized=True)
    else:
        sql_costs_archive = None

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    set_cost_model(cursor, cost_model)
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))
    cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))

    # if not use_indexes:
        # set_indexes(cursor, "off")
    # else:
        # set_indexes(cursor, "on")

    ret = []
    for i, query in enumerate(queries):
        join_graph = join_graphs[i]
        if "mii1.info " in query:
            query = query.replace("mii1.info ", "mii1.info::float")
        if "mii2.info " in query:
            query = query.replace("mii2.info ", "mii2.info::float")
        if "mii1.info)" in query:
            query = query.replace("mii1.info)", "mii1.info::float)")
        if "mii2.info)" in query:
            query = query.replace("mii2.info)", "mii2.info::float)")

        est_sql, est_cost, est_explain = get_cardinalities_join_cost(query,
                est_cardinalities[i], true_cardinalities[i], join_graphs[i],
                cursor, sql_costs_archive)

        if opt_costs[i] is None:
            opt_sql, opt_cost, opt_explain = get_cardinalities_join_cost(query,
                    true_cardinalities[i], true_cardinalities[i],
                    join_graphs[i], cursor, sql_costs_archive)
            opt_sqls[i] = opt_sql
            opt_costs[i] = opt_cost
            opt_explains[i] = opt_explain

        # FIXME: temporary
        if est_cost < opt_costs[i]:
            est_cost = opt_costs[i]

        ret.append((est_cost, opt_costs[i], est_explain, opt_explains[i],
            est_sql, opt_sqls[i]))

    cursor.close()
    con.close()
    return ret

def preprocess_sql_mysql(sql):
    if "ILIKE" in sql:
        sql = sql.replace("ILIKE", "LIKE")

    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in sql:
        sql = sql.replace(bad_str1, "")

    if bad_str2 in sql:
        sql = sql.replace(bad_str2, "")

    if "::float" in sql:
        sql = sql.replace("::float", "")

    if "::int" in sql:
        sql = sql.replace("::int", "")

    return sql

def get_join_order_mysql(explain_json):
    tables = []
    qb = explain_json["query_block"]
    if "ordering_operation" in qb:
        qb = qb["ordering_operation"]

    if "grouping_operation" in qb:
        tab_list = qb["grouping_operation"]["nested_loop"]
    else:
        tab_list = qb["nested_loop"]

    for tab in tab_list:
        info = tab["table"]
        tname = info["table_name"]
        tables.append(tname)
        rows_join = info["rows_produced_per_join"]
        read_cost = info["cost_info"]["read_cost"]
        prefix_cost = info["cost_info"]["prefix_cost"]
        # print("table: {}, join_rows: {}, read_cost: {}, prefix_cost: {}".format(\
                # tname, rows_join, read_cost, prefix_cost))
        # print(info["cost_info"])

    return tables


class JoinLoss():

    def __init__(self, cost_model, user, pwd, db_host, port, db_name):
        self.cost_model = cost_model
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name

        opt_archive_fn = "/tmp/opt_archive_" + cost_model
        self.opt_archive = klepto.archives.dir_archive(opt_archive_fn,
                cached=True, serialized=True)

    def compute_join_order_loss(self, sqls, join_graphs, true_cardinalities,
            est_cardinalities, baseline_join_alg, use_indexes,
            num_processes=8, backend="postgres", pool=None, fns=None):
        '''
        @query_dict: [sqls]
        @true_cardinalities / est_cardinalities: [{}]
                dictionary, specifying cardinality of each subquery
                key: sorted list of [table_1_key, table_2_key, ...table_n_key]
                val: cardinality (double)
                In order to handle aliases (this information is lost when
                processing in calcite), each table_key is table name +
                first predicate (if no predicate present on that table, then just "")
            FIXME: this does not handle many edge cases, and we need a better
            way to deal with aliases.

        @ret:
            TODO
        '''
        start = time.time()
        assert isinstance(sqls, list)
        assert isinstance(true_cardinalities, list)
        assert isinstance(est_cardinalities, list)
        assert len(sqls) == len(true_cardinalities) == len(est_cardinalities)

        if backend == "postgres":
            return self._compute_join_order_loss_pg(sqls, join_graphs,
                    true_cardinalities, est_cardinalities, num_processes,
                    use_indexes, pool)
        elif backend == "mysql":
            return self._compute_join_order_loss_mysql(sqls, join_graphs,
                    true_cardinalities, est_cardinalities, num_processes,
                    use_indexes, pool, fns)
        elif backend == "calcite":
            return self._compute_join_order_loss_calcite(sqls, join_graphs,
                    true_cardinalities, est_cardinalities, num_processes,
                    use_indexes, pool, fns)
        else:
            assert False

    def _compute_join_order_loss_calcite(self, sqls, join_graphs, true_cardinalities,
            est_cardinalities, num_processes, use_indexes, pool, fns):

    # def compute_join_order_loss(self, sqls, true_cardinalities,
            # est_cardinalities, baseline_join_alg, use_indexes,
            # num_processes=8, postgres=True, pool=None):

        env = QueryOptEnv()
        print("test hello!")
        env.compute_join_order_loss(sqls, true_cardinalities,
                est_cardinalities, "EXHAUSTIVE", True, 1, postgres=False,
                pool=None)

        pdb.set_trace()

    def _compute_join_order_loss_mysql(self, sqls, join_graphs, true_cardinalities,
            est_cardinalities, num_processes, use_indexes, pool, fns):

        def run_single_sql(sql):
            # db=MySQLdb.connect(passwd=self.pwd,db=self.db_name, user=self.user)
            db = MySQLdb.connect(db="imdb", passwd="", user="root",
                    host="127.0.0.1")
            # db = MySQLdb.connect(db="imdb", passwd="1234", user="root",
                    # host="127.0.0.1")
            cursor = db.cursor()
            # row_eval_cost = 0.1
            # row_eval_query = MYSQL_ROW_EVAL_COST_TMP.format(val=row_eval_cost)
            # cursor.execute(row_eval_query)

            cursor.execute("SET optimizer_prune_level=0;")

            # TMP = "set optimizer_switch='index_merge_union=off,index_merge_sort_union=off';
            opt_flags = MYSQL_OPT_TMP.format(FLAGS=MYSQL_OPT_FLAGS)
            cursor.execute(opt_flags)

            sql = preprocess_sql_mysql(sql)
            orig_sql = sql

            # start = time.time()
            # con.execute(orig_sql)
            # print("executing default took: ", time.time()-start)
            sql = "EXPLAIN FORMAT=json " + sql
            join_graph = join_graphs[i]

            cards = est_cardinalities[i]
            with open(MYSQL_CARD_FILE_NAME, 'w') as fp:
                json.dump(cards, fp)
            cursor.execute(sql)
            out = cursor.fetchall()
            plan_explain = json.loads(out[0][0])
            # print("**********Est Cost*****************")
            # print(plan_explain["query_block"]["cost_info"])
            # print("***************************")
            est_join_order = get_join_order_mysql(plan_explain)
            est_est_explain = plan_explain
            est_est_explains.append(est_est_explain)
            est_cost=float(plan_explain["query_block"]["cost_info"]["query_cost"])
            # print("est_join_order: ", est_join_order)
            os.remove(MYSQL_CARD_FILE_NAME)

            cards = true_cardinalities[i]
            with open(MYSQL_CARD_FILE_NAME, 'w') as fp:
                json.dump(cards, fp)

            # TODO: don't need this if we can cache these results
            cursor.execute(sql)
            out = cursor.fetchall()
            plan_explain = json.loads(out[0][0])
            # print("**********True Cost*****************")
            # print(plan_explain["query_block"]["cost_info"])
            # print("***************************")
            opt_join_order = get_join_order_mysql(plan_explain)
            # print("opt join order: ", opt_join_order)
            # pdb.set_trace()

            opt_cost=float(plan_explain["query_block"]["cost_info"]["query_cost"])
            new_from = []
            for alias in opt_join_order:
                table = join_graph.nodes()[alias]["real_name"]
                new_from.append("{} AS {}".format(table, alias))
            from_clause = " STRAIGHT_JOIN ".join(new_from)
            opt_sql = nx_graph_to_query(join_graph, from_clause)
            opt_sql = preprocess_sql_mysql(opt_sql)
            opt_sql_exec = opt_sql
            opt_explains.append(plan_explain)

            # TODO: remove
            # start = time.time()
            # cursor.execute(orig_sql)
            # print("executing true took: ", time.time()-start)

            # force join order
            new_from = []
            for alias in est_join_order:
                table = join_graph.nodes()[alias]["real_name"]
                new_from.append("{} AS {}".format(table, alias))

            from_clause = " STRAIGHT_JOIN ".join(new_from)
            est_sql = nx_graph_to_query(join_graph, from_clause)
            est_sql = preprocess_sql_mysql(est_sql)
            est_sql_exec = est_sql
            est_sql = "EXPLAIN FORMAT=json " + est_sql
            cursor.execute(est_sql)
            out = cursor.fetchall()
            plan_explain = json.loads(out[0][0])

            # print("**********True Cost*****************")
            # print(plan_explain["query_block"]["cost_info"])
            # print("***************************")
            est_join_order_forced = get_join_order_mysql(plan_explain)
            est_plan_cost=float(plan_explain["query_block"]["cost_info"]["query_cost"])
            # print("est_join_order_forced: ", est_join_order_forced)
            assert str(est_join_order_forced) == str(est_join_order)

            # add all things to lists here, so if it crashed before, things
            # won't be out of place

            est_explains.append(plan_explain)

            opt_costs.append(opt_cost)
            est_costs.append(est_plan_cost)

            opt_sqls.append(opt_sql)
            est_sqls.append(est_sql_exec)

            # print("**********MySQL Plan Loss*****************")
            # print(est_plan_cost - opt_cost)
            # print("***************************")

            start = time.time()
            os.remove(MYSQL_CARD_FILE_NAME)
            cursor.close()


        # open mysql conn
        est_costs = []
        opt_costs = []
        est_explains = []
        est_est_explains = []
        opt_explains = []
        est_sqls = []
        opt_sqls = []

        # step 1: set estimated cards and compute plan
        for i,sql in enumerate(sqls):
            try:
                run_single_sql(sql)
                if i % 200 == 0:
                    avg_cost = np.mean(np.array(est_costs) - \
                            np.array(opt_costs))
                    # print("{}: avg mysql cost: {}".format(i, avg_cost))
                    # print(fns[i])
                    # exp1 = est_explains[i]["query_block"]["nested_loop"]
                    # exp2 = est_est_explains[i]["query_block"]["nested_loop"]
                    # opt_exp = opt_explains[i]["query_block"]["nested_loop"]
                    # rc1 = extract_values(exp1, "read_cost")
                    # rc2 = extract_values(exp2, "read_cost")
                    # print(rc1)
                    # print(rc2)
                    # pdb.set_trace()
            except Exception as e:
                print("crash, restart mysql")
                print(e)
                est_costs.append(0.0)
                opt_costs.append(0.0)
                est_sqls.append(None)
                opt_sqls.append(None)
                est_explains.append(None)
                est_est_explains.append(None)
                opt_explains.append(None)
                pdb.set_trace()
                continue

        return np.array(est_costs), np.array(opt_costs), est_explains, \
    opt_explains, est_sqls, opt_sqls

    def _compute_join_order_loss_pg(self, sqls, join_graphs, true_cardinalities,
            est_cardinalities, num_processes, use_indexes, pool):
        start = time.time()
        est_costs = [None]*len(sqls)
        opt_costs = [None]*len(sqls)
        est_explains = [None]*len(sqls)
        opt_explains = [None]*len(sqls)
        est_sqls = [None]*len(sqls)
        opt_sqls = [None]*len(sqls)

        if use_indexes:
            use_indexes = 1
        else:
            use_indexes = 0

        for i, sql in enumerate(sqls):
            sql_key = deterministic_hash(sql)
            if sql_key in self.opt_archive.archive:
                (opt_costs[i], opt_explains[i], opt_sqls[i]) = \
                        self.opt_archive.archive[sql_key]

        if pool is None:
            # single threaded case, useful for debugging
            all_costs = [compute_join_order_loss_pg_single(sqls, join_graphs,
                    true_cardinalities, est_cardinalities, opt_costs,
                    opt_explains, opt_sqls, use_indexes, self.user,
                    self.pwd, self.db_host, self.port, self.db_name, False,
                    self.cost_model)]
            batch_size = len(sqls)

        else:
            num_processes = pool._processes
            batch_size = max(1, math.ceil(len(sqls) / num_processes))
            assert num_processes * batch_size >= len(sqls)
            par_args = []
            for proc_num in range(num_processes):
                start_idx = proc_num * batch_size
                end_idx = min(start_idx + batch_size, len(sqls))
                par_args.append((sqls[start_idx:end_idx],
                    join_graphs[start_idx:end_idx],
                    true_cardinalities[start_idx:end_idx],
                    est_cardinalities[start_idx:end_idx],
                    opt_costs[start_idx:end_idx],
                    opt_explains[start_idx:end_idx],
                    opt_sqls[start_idx:end_idx],
                    use_indexes, self.user, self.pwd, self.db_host,
                    self.port, self.db_name, True, self.cost_model))

            all_costs = pool.starmap(compute_join_order_loss_pg_single, par_args)

        new_seen = False
        for num_proc, costs in enumerate(all_costs):
            start_idx = int(num_proc * batch_size)
            for i, (est, opt, est_explain, opt_explain, est_sql, opt_sql) \
                        in enumerate(costs):
                sql = sqls[start_idx + i]
                sql_key = deterministic_hash(sql)

                est_costs[start_idx+i] = est
                est_explains[start_idx+i] = est_explain
                est_sqls[start_idx+i] = est_sql
                opt_costs[start_idx+i] = opt
                opt_explains[start_idx+i] = opt_explain
                opt_sqls[start_idx+i] = opt_sql

                # pool is None used when computing subquery priorities,
                # archiving all those is too expensive..
                if sql_key not in self.opt_archive.archive \
                        and pool is not None:
                    self.opt_archive.archive[sql_key] = (opt, opt_explain, opt_sql)

        # print("num explains: ", len(est_explains))
        # for exp in est_explains:
            # vals = extract_values(exp[0][0][0], "Node Type")
            # nl_count = 0
            # hj_count = 0
            # ind_count = 0
            # for v in vals:
                # if "Nested" in v:
                    # nl_count += 1
                # if "Hash Join" in v:
                    # hj_count += 1
                # if "Index" in v:
                    # ind_count += 1

            # print("nl: {}, hj: {}, index: {}".format(nl_count, hj_count,
                # ind_count))

        loss = np.mean(np.array(est_costs) - np.array(opt_costs))

        if loss < 0.0:
            print("negative loss for postgres join loss")
            print(loss)
            pdb.set_trace()

        # print("compute postgres join error took: ", time.time()-start)
        return np.array(est_costs), np.array(opt_costs), est_explains, \
    opt_explains, est_sqls, opt_sqls

def fl_cpp_get_flow_loss(samples, source_node, cost_key,
        all_ests, known_costs, cost_model, trueC_vecs):
    start = time.time()
    costs = []
    # farchive = klepto.archives.dir_archive("./flow_info_archive",
            # cached=True, serialized=True)
    new_seen = False
    debug_sql = False
    for i, sample in enumerate(samples):
        if known_costs and known_costs[i] is not None:
            costs.append(known_costs[i])
            continue

        qkey = deterministic_hash(sample["sql"])
        # if qkey in farchive.archive:
        if False:
            subsetg_vectors = farchive.archive[qkey]
            assert len(subsetg_vectors) == 8
            # totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                    # edges_cost_node2, final_node, edges_penalties = subsetg_vectors
        else:
            new_seen = True
            # this must be for true cards
            # assert all_ests is None
            subsetg_vectors = list(get_subsetg_vectors(sample, cost_model))
            assert len(subsetg_vectors) == 10

        totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                edges_cost_node2, edges_read_costs, edges_rows_fetched, \
                final_node, edges_penalties = subsetg_vectors
        nodes = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in nodes:
            nodes.remove(SOURCE_NODE)
        nodes.sort()

        if qkey in trueC_vecs:
            if debug_sql:
                print("found trueC_vec!")
            trueC_vec = trueC_vecs[qkey]
            # calculate other variables needed for optimization
            est_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)

            for ni, node in enumerate(nodes):
                if all_ests is not None:
                    if node in all_ests[i]:
                        est_cards[ni] = all_ests[i][node]
                    else:
                        est_cards[ni] = all_ests[i][" ".join(node)]
                # else:
                    # # est_cards are also true cardinalities here
                    # est_cards[ni] = \
                            # sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

            predC2, _, G2, Q2 = get_optimization_variables(est_cards, totals,
                    0.0, 24.0, None, edges_cost_node1,
                    edges_cost_node2, nilj, edges_head, edges_tail,
                    edges_read_costs, edges_rows_fetched,
                    cost_model,
                    edges_penalties)

            if debug_sql:
                print(predC2)
                pdb.set_trace()
        else:
            # print("going to compute true card based flow cost")
            # computing based on true cards
            assert all_ests is None
            true_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)

            for ni, node in enumerate(nodes):
                true_cards[ni] = \
                        sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

            trueC_vec, _, G2, Q2 = get_optimization_variables(true_cards, totals,
                    0.0, 24.0, None, edges_cost_node1,
                    edges_cost_node2, nilj, edges_head, edges_tail,
                    edges_read_costs, edges_rows_fetched,
                    cost_model,
                    edges_penalties)
            trueC_vecs[qkey] = trueC_vec

        if debug_sql:
            pdb.set_trace()

        Gv2 = np.zeros(len(totals), dtype=np.float32)
        Gv2[final_node] = 1.0
        Gv2 = to_variable(Gv2).float()
        # predC2 = to_variable(predC2).float()
        G2 = to_variable(G2).float()
        invG = torch.inverse(G2)
        v = invG @ Gv2 # vshape: Nx1
        v = v.detach().cpu().numpy()
        if debug_sql:
            print("before calling fl_cpp.get_qvtqv")
            pdb.set_trace()

        # if debug_sql:
            # f = Q2 @ v
            # print(f)
            # pdb.set_trace()

        # flows = Q2 @ v
        # if np.min(flows) < 0.0:
            # print("negative flows!")
            # pdb.set_trace()

        # TODO: we don't even need to compute the loss here if we don't want to
        loss2 = np.zeros(1, dtype=np.float32)
        assert Q2.dtype == np.float32
        assert v.dtype == np.float32
        if isinstance(trueC_vec, torch.Tensor):
            trueC_vec = trueC_vec.detach().cpu().numpy()
        assert trueC_vec.dtype == np.float32
        fl_cpp.get_qvtqv(
                c_int(len(edges_head)),
                c_int(len(v)),
                edges_head.ctypes.data_as(c_void_p),
                edges_tail.ctypes.data_as(c_void_p),
                Q2.ctypes.data_as(c_void_p),
                v.ctypes.data_as(c_void_p),
                trueC_vec.ctypes.data_as(c_void_p),
                loss2.ctypes.data_as(c_void_p)
                )

        costs.append(loss2[0])
        # print(loss2[0])
        # pdb.set_trace()
        # if new_seen:
            # farchive.archive[qkey] = subsetg_vectors
    return costs

def get_flow_cost(qrep, yhat, y,
        cost_model, flow_loss_power=2.0):
    def get_cost(sample, cost_key, ests,
            true_edge_costs=None):
        assert SOURCE_NODE in sample["subset_graph"].nodes()

        # compute_costs(subsetg, cost_model, cost_key=cost_key,
                # ests=ests)
        subsetg_vectors = list(get_subsetg_vectors(sample, cost_model))
        assert len(subsetg_vectors) == 8

        totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                edges_cost_node2, final_node, edges_penalties = subsetg_vectors
        nodes = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in nodes:
            nodes.remove(SOURCE_NODE)
        nodes.sort()

        if true_edge_costs is not None:
            # calculate other variables needed for optimization
            est_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)
            for ni, node in enumerate(nodes):
                if node in ests:
                    est_cards[ni] = ests[node]
                else:
                    est_cards[ni] = ests[" ".join(node)]
            predC2, _, G2, Q2 = get_optimization_variables(est_cards, totals,
                    0.0, 24.0, None, edges_cost_node1,
                    edges_cost_node2, nilj, edges_head, edges_tail, cost_model,
                    edges_penalties)

        else:
            true_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)
            for ni, node in enumerate(nodes):
                true_cards[ni] = \
                        sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

            true_edge_costs, _, G2, Q2 = get_optimization_variables(true_cards, totals,
                    0.0, 24.0, None, edges_cost_node1,
                    edges_cost_node2, nilj, edges_head, edges_tail, cost_model,
                    edges_penalties)

        Gv2 = np.zeros(len(totals), dtype=np.float32)
        Gv2[final_node] = 1.0
        Gv2 = to_variable(Gv2).float()
        G2 = to_variable(G2).float()
        invG = torch.inverse(G2)
        v = invG @ Gv2 # vshape: Nx1
        v = v.detach().cpu().numpy()

        # TODO: we don't even need to compute the loss here if we don't want to
        loss2 = np.zeros(1, dtype=np.float32)
        assert Q2.dtype == np.float32
        assert v.dtype == np.float32

        if isinstance(true_edge_costs, torch.Tensor):
            true_edge_costs = true_edge_costs.detach().cpu().numpy()

        # assert true_edge_costs.dtype == np.float32
        fl_cpp.get_qvtqv(
                c_int(len(edges_head)),
                c_int(len(v)),
                edges_head.ctypes.data_as(c_void_p),
                edges_tail.ctypes.data_as(c_void_p),
                Q2.ctypes.data_as(c_void_p),
                v.ctypes.data_as(c_void_p),
                true_edge_costs.ctypes.data_as(c_void_p),
                loss2.ctypes.data_as(c_void_p)
                )

        flows = Q2 @ v
        if isinstance(flows, torch.Tensor):
            flows = flows.detach().cpu().numpy()
        flow_cost = np.dot(true_edge_costs, np.power(flows, flow_loss_power))

        # just reuse cost key so source node edge costs are already there
        edges = []
        # nodes =
        assert len(flows) == len(edges_head) == len(edges_tail)
        # tmp_subsetg = copy.deepcopy(sample["subset_graph"])
        tmp_subsetg = sample["subset_graph"]
        for edgei, edge in enumerate(edges_head):
            head_node = nodes[edge]
            if edges_tail[edgei] > len(nodes):
                # source node
                tmp_subsetg[head_node][SOURCE_NODE][cost_model+cost_key] = -flows[edgei]
                continue

            tail_node = nodes[edges_tail[edgei]]
            tmp_subsetg[head_node][tail_node][cost_model+cost_key] = -flows[edgei]

        nodes = list(sample["subset_graph"].nodes())
        nodes.sort(key=lambda x: len(x))
        final_node = nodes[-1]

        path = nx.shortest_path(tmp_subsetg, final_node,
                SOURCE_NODE, weight=cost_model+cost_key)

        path = path[0:-1]
        plan_cost = 0.0
        for pi in range(len(path)-1):
            plan_cost += tmp_subsetg[path[pi]][path[pi+1]][cost_model+"cost"]

        return path, flow_cost, plan_cost, true_edge_costs

    qrep = copy.deepcopy(qrep)
    opt_path, flow_cost, plan_cost, true_edge_costs = get_cost(qrep, "cost", y)
    est_path, est_flow_cost, est_plan_cost,_ = get_cost(qrep, "est_cost", yhat, true_edge_costs)
    return flow_cost, est_flow_cost, plan_cost,est_plan_cost,opt_path,est_path

def get_quadratic_program_cost(qrep, yhat, y,
        cost_model, beta=2.0, alpha=2.0):
    def get_cost(subsetg, cost_key, ests,
            true_edge_costs=None):
        assert SOURCE_NODE in subsetg.nodes()

        compute_costs(subsetg, cost_model, cost_key=cost_key,
                ests=ests)
        edges, costs, A, b, G, h = construct_lp(subsetg,
                cost_key=cost_model+cost_key)
        costs = costs / 1e6

        n = len(edges)
        P = np.zeros((len(edges),len(edges)))
        for i,c in enumerate(costs):
            P[i,i] = c

        q = np.zeros(len(edges))
        x = cp.Variable(n)

        obj = cp.Minimize(1/2 * (costs @ (x**beta)))
        # obj = cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x)

        prob = cp.Problem(obj,
                        [G @ x <= h,
                         A @ x == b])

        try:
            prob.solve(verbose=False, solver=cp.CVXOPT)
        except:
            return None,None,None,None

        qsolx = np.array(x.value)
        qsolx = np.maximum(qsolx, 0.00)
        if true_edge_costs is None:
            quad_cost = np.dot(costs, qsolx**alpha)
        else:
            quad_cost = np.dot(true_edge_costs, qsolx**alpha)

        # just reuse cost key so source node edge costs are already there
        ## negative because flows would be higher when costs are cheaper
        for edgei, edge in enumerate(edges):
            subsetg[edge[0]][edge[1]][cost_model+cost_key] = -qsolx[edgei]

        nodes = list(subsetg.nodes())
        nodes.sort(key=lambda x: len(x))
        final_node = nodes[-1]

        path = nx.shortest_path(subsetg, final_node,
                SOURCE_NODE, weight=cost_model+cost_key)

        path = path[0:-1]

        plan_cost = 0.0
        for pi in range(len(path)-1):
            plan_cost += subsetg[path[pi]][path[pi+1]][cost_model+"cost"]
        assert plan_cost != 0.0

        return path, quad_cost, plan_cost, costs

    subsetg = copy.deepcopy(qrep["subset_graph"])

    opt_path, quad_cost, cost, true_edge_costs = get_cost(subsetg, "cost", y)
    est_path, est_quad_cost, est_cost,_ = get_cost(subsetg, "est_cost", yhat, true_edge_costs)

    if opt_path is None or est_path is None:
        return 0.0, -1.0, 0.0, -1.0, [],[]

    return quad_cost, est_quad_cost, cost,est_cost,opt_path,est_path

def get_simple_shortest_path_cost(qrep, yhat, y,
        cost_model, directed, mdata=None):
    def get_cost(subsetg, cost_key, ests):
        assert SOURCE_NODE in subsetg.nodes()
        cardinality_key = "cardinality"
        compute_costs(subsetg, cost_model, cardinality_key,
                cost_key=cost_key,
                ests=ests, mdata=mdata)
        nodes = list(subsetg.nodes())
        nodes.sort(key=lambda x: len(x))

        # if not directed:
            # subsetg = subsetg.to_undirected()

        final_node = nodes[-1]
        path = nx.shortest_path(subsetg, final_node,
                SOURCE_NODE, weight=cost_model+cost_key)
        path = path[0:-1]

        # need to cost the path using true cardinalities
        cost = 0.0
        for pi in range(len(path)-1):
            cost += subsetg[path[pi]][path[pi+1]][cost_model+"cost"]

        return cost, path

    subsetg = copy.deepcopy(qrep["subset_graph"])
    cost,opt_path = get_cost(subsetg, "cost", y)
    # subsetg = copy.deepcopy(qrep["subset_graph"])
    est_cost,est_path = get_cost(subsetg, "est_cost", yhat)
    return cost,est_cost,opt_path,est_path

def get_shortest_path_costs(samples, source_node, cost_key,
        all_ests, known_costs, cost_model, compute_pg_costs,
        user=None, pwd=None, db_host=None, port=None, db_name=None,
        true_cardinalities=None, join_graphs=None):
    '''
    @ret: cost of the given path in subsetg.
    '''
    costs = []
    pg_costs = []
    paths = []
    pg_sqls = []
    pg_explains = []

    assert true_cardinalities is not None
    assert all_ests is not None

    assert len(true_cardinalities) == len(all_ests)
    cardinality_key = "cardinality"

    for i in range(len(samples)):
        if known_costs and known_costs[i] is not None:
            costs.append(known_costs[i])
            continue
        # subsetg = samples[i]["subset_graph_paths"]
        ## TODO: we should not need to recompute the costs here
        subsetg = samples[i]["subset_graph"]
        assert SOURCE_NODE in subsetg.nodes()

        # this should already be pre-computed
        ests = all_ests[i]
        if "mysql" in cost_model:
            fn = samples[i]["name"]
            fn = fn.replace("queries", "mysql_data_all")
            mdata = load_object(fn)
        else:
            mdata = None

        compute_costs(subsetg, cost_model,
                cardinality_key,
                cost_key=cost_key,
                ests=ests, mdata=mdata)

        # print("compute costs done")

        # TODO: precompute..
        nodes = list(subsetg.nodes())
        nodes.sort(key=lambda x: len(x))
        final_node = nodes[-1]
        # if subsetg.is_directed():
            # subsetg = subsetg.to_undirected()

        path = nx.shortest_path(subsetg, final_node,
                source_node, weight=cost_model+cost_key)
        path = path[0:-1]
        paths.append(path)
        # print("path done")

        cost = 0.0
        scan_types = {}
        for pi in range(len(path)-1):
            cost_key = cost_model + "cost"
            scan_key = cost_key + "scan_type"
            cost += subsetg[path[pi]][path[pi+1]][cost_key]

            if scan_key in subsetg[path[pi]][path[pi+1]]:
                scan_types.update(subsetg[path[pi]][path[pi+1]][scan_key])

        # if len(scan_types) < 1:
            # print("scan types: ", scan_types)
        assert cost >= 1
        costs.append(cost)

        if False:
            join_order = [tuple(sorted(x)) for x in path_to_join_order(path)]
            # print("path_to_join_order done")
            join_order.reverse()
            # join_order2 = copy.deepcopy(join_order)
            # join_order[0] = join_order2[1]
            # join_order[1] = join_order2[0]

            sql_to_exec = nodes_to_sql(join_order, join_graphs[i])

            cur_ests = all_ests[i]
            exec_sql, est_cost, est_explain = get_join_cost_sql(sql_to_exec,
                    cur_ests, true_cardinalities[i],
                    join_graphs[i], user, pwd, db_host, port, db_name,
                    cost_model, scan_types)
            # print("get join cost sql done")
            pg_costs.append(est_cost)
            pg_sqls.append(exec_sql)
            pg_explains.append(est_explain)
        else:
            pg_costs.append(0.0)
            pg_sqls.append("")
            pg_explains.append(None)

    return costs, pg_costs, paths, pg_sqls, pg_explains

class PlanError():

    def __init__(self, cost_model, loss_type,
            user=None, pwd=None, db_host=None, port=None, db_name=None,
            compute_pg_costs=False):
        if db_name == "so":
            global SOURCE_NODE
            SOURCE_NODE = tuple(["SOURCE"])

        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name
        self.compute_pg_costs = compute_pg_costs

        self.cost_model = cost_model
        self.source_node = SOURCE_NODE
        self.loss_type = loss_type
        if loss_type == "plan-loss":
            self.loss_func = get_shortest_path_costs
            self.opt_pg_costs = {}
        elif loss_type == "flow-loss":
            self.loss_func = fl_cpp_get_flow_loss

        self.subsetgs = {}
        self.opt_costs = {}
        if self.loss_type == "flow-loss":
            self.trueC_vecs = {}

    def compute_loss(self, qreps, ests, pool=None, true_cardinalities=None,
            join_graphs=None):
        '''
        @ests: [dicts] of estimates
        '''
        start = time.time()
        subsetgs = []
        opt_costs = []
        if self.loss_type == "plan-loss":
            opt_pg_costs = []
        else:
            opt_pg_costs = None
            all_pg_costs = None
            all_pg_exec_sqls = None
            all_pg_explains = None

        new_opt_cost = False
        for qrep in qreps:
            qkey = deterministic_hash(qrep["sql"])
            if qkey in self.opt_costs:
                opt_costs.append(self.opt_costs[qkey])
                # if self.loss_type == "plan-loss":
                    # opt_pg_costs.append(self.opt_pg_costs[qkey])
            else:
                opt_costs.append(None)
                new_opt_cost = True
            nodes = list(qrep["subset_graph"].nodes())

        if pool is None or self.loss_type == "flow-loss":
            opt_costs = self.loss_func(qreps, self.source_node, "cost", None, opt_costs,
                    self.cost_model, self.trueC_vecs)
            all_costs = self.loss_func(qreps, self.source_node, "est_cost", ests, None,
                    self.cost_model, self.trueC_vecs)
            if new_opt_cost:
                for i, qrep in enumerate(qreps):
                    qkey = deterministic_hash(qrep["sql"])
                    self.opt_costs[qkey] = opt_costs[i]
        else:
            num_processes = pool._processes
            batch_size = max(1, math.ceil(len(qreps) / num_processes))
            assert num_processes * batch_size >= len(qreps)
            opt_par_args = []
            par_args = []

            for proc_num in range(num_processes):
                start_idx = proc_num * batch_size
                end_idx = min(start_idx + batch_size, len(qreps))
                if end_idx <= start_idx:
                    continue

                if new_opt_cost:
                    opt_par_args.append((qreps[start_idx:end_idx],
                        self.source_node, "cost",
                        true_cardinalities[start_idx:end_idx],
                        None, self.cost_model,
                        self.compute_pg_costs,
                        self.user, self.pwd, self.db_host, self.port,
                        self.db_name, true_cardinalities[start_idx:end_idx],
                        join_graphs[start_idx:end_idx]))
                par_args.append((qreps[start_idx:end_idx],
                    self.source_node, "est_cost", ests[start_idx:end_idx],
                    None, self.cost_model, self.compute_pg_costs,
                    self.user, self.pwd, self.db_host, self.port, self.db_name,
                    true_cardinalities[start_idx:end_idx],
                    join_graphs[start_idx:end_idx]))

            if new_opt_cost:
                opt_costs = []
                opt_pg_costs = []

                # print("sequential opt costs!")
                # opt_costs_batched = self.loss_func(qreps, self.source_node,
                        # "cost", None, None,
                        # self.cost_model, self.compute_pg_costs, self.user,
                        # self.pwd, self.db_host, self.port, self.db_name,
                        # true_cardinalities, join_graphs)
                # opt_costs_batched = [opt_costs_batched]

                opt_costs_batched = pool.starmap(self.loss_func,
                        opt_par_args)
                for c in opt_costs_batched:
                    opt_costs += c[0]
                    if self.compute_pg_costs:
                        opt_pg_costs += c[1]

                for i, qrep in enumerate(qreps):
                    qkey = deterministic_hash(qrep["sql"])
                    self.opt_costs[qkey] = opt_costs[i]
                    if self.compute_pg_costs:
                        self.opt_pg_costs[qkey] = opt_pg_costs[i]

                # let's get the opt costs for each path

            all_costs = []
            all_pg_costs = []
            all_pg_exec_sqls = []
            all_pg_explains = []
            # print("debug run, no parallel!")
            # all_costs_batched = [self.loss_func(qreps, self.source_node, "est_cost", ests, None,
                    # self.cost_model, self.compute_pg_costs, self.user,
                    # self.pwd, self.db_host, self.port, self.db_name,
                    # true_cardinalities, join_graphs)]

            # parallel version...
            all_costs_batched = pool.starmap(self.loss_func,
                    par_args)

            for c in all_costs_batched:
                all_costs += c[0]
                if self.compute_pg_costs:
                    all_pg_costs += c[1]
                    all_pg_exec_sqls += c[3]
                    all_pg_explains += c[4]
            assert len(all_pg_costs) == len(all_pg_exec_sqls) == \
                len(all_pg_explains)

        all_costs = np.array(all_costs)
        opt_costs = np.array(opt_costs)
        loss = np.mean(all_costs - opt_costs)

        if loss < 0.0:
            print("negative loss for ", self.loss_type)
            print(loss)
            # pdb.set_trace()
            # for i,c in enumerate(all_costs):
                # if c < opt_costs[i]:
                    # idx = i
                    # break
            # print(idx)
            # est_cards = ests[idx]
            # # qreps[idx]
            # cur_ests = []
            # trues = []
            # nodes = []
            # subsetg = qreps[idx]["subset_graph"]
            # for node, info in subsetg.nodes().items():
                # cur_ests.append(est_cards[" ".join(node)])
                # trues.append(info["cardinality"]["actual"])
                # nodes.append(node)
            # pdb.set_trace()

        print("compute {} took: {}".format(self.loss_type, time.time()-start))

        return np.array(opt_costs), np.array(all_costs), \
                np.array(opt_pg_costs), np.array(all_pg_costs), \
                all_pg_exec_sqls, all_pg_explains

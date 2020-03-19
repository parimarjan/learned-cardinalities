import psycopg2 as pg
import getpass
from db_utils.utils import *
from sql_rep.utils import extract_aliases
import multiprocessing as mp
import math

import pdb
import klepto

PG_HINT_CMNT_TMP = '''/*+ {COMMENT} */'''
PG_HINT_JOIN_TMP = "{JOIN_TYPE} ({TABLES}) "
PG_HINT_CARD_TMP = "Rows ({TABLES} #{CARD}) "
PG_HINT_SCAN_TMP = "{SCAN_TYPE}({TABLE}) "
PG_HINT_LEADING_TMP = "Leading ({JOIN_ORDER})"
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

def _get_cost(sql, cur):
    assert "explain" in sql
    # cur = con.cursor()
    cur.execute(sql)
    explain = cur.fetchall()
    all_costs = extract_values(explain[0][0][0], "Total Cost")
    mcost = max(all_costs)
    # cur.close()
    # cost = all_costs[-1]
    # pdb.set_trace()
    cost = explain[0][0][0]["Plan"]["Total Cost"]
    # if cost != mcost:
        # print(cost, mcost)
        # pdb.set_trace()
    return cost, explain

def _gen_pg_hint_cards(cards):
    '''
    '''
    card_str = ""
    for aliases, card in cards.items():
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

def _get_modified_sql(sql, cardinalities, join_ops,
        leading_hint, scan_ops):
    '''
    @cardinalities: dict
    @join_ops: dict

    @ret: sql, augmented with appropriate comments.
    '''
    # if "explain" not in sql:
    sql = " explain (format json) " + sql

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
    return sql

def get_cardinalities_join_cost(query, est_cardinalities, true_cardinalities,
        join_graph, cursor, sql_costs):

    est_card_sql = _get_modified_sql(query, est_cardinalities, None,
            None, None)
    # assert "explain" in est_card_sql.lower()
    cursor.execute(est_card_sql)
    explain = cursor.fetchall()
    est_join_order_sql, est_join_ops, scan_ops = get_pg_join_order(join_graph,
            explain)
    leading_hint = get_leading_hint(join_graph, explain)
    assert "info" not in leading_hint

    est_opt_sql = nx_graph_to_query(join_graph,
            from_clause=est_join_order_sql)

    # add the join ops etc. information
    cost_sql = _get_modified_sql(est_opt_sql, true_cardinalities,
            est_join_ops, leading_hint, scan_ops)

    # set this to sql to be executed, as pg_hint will enforce the estimated
    # cardinalities, and let postgres make decisions for join order and
    # everything about operators based on the estimated cardinalities
    exec_sql = _get_modified_sql(est_opt_sql, est_cardinalities,
            None, None, None)

    # cost_sql will be seen often, as true_cardinalities remain fixed. so we
    # can cache the results for it.

    cost_sql_key = deterministic_hash(cost_sql)
    if cost_sql_key in sql_costs.archive:
        est_cost, est_explain = sql_costs.archive[cost_sql_key]
    else:
        est_cost, est_explain = _get_cost(cost_sql, cursor)
        sql_costs.archive[cost_sql_key] = (est_cost, est_explain)

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
        use_indexes, user, pwd, db_host, port, db_name):
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
    # FIXME: do this earlier for each query...

    try:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd)
    except:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd, host=db_host)

    sql_costs_archive = klepto.archives.dir_archive("/tmp/sql_costs",
            cached=True, serialized=True)

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))
    cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))
    if not use_indexes:
        cursor.execute("SET enable_indexscan = off")
        cursor.execute("SET enable_indexonlyscan = off")
    else:
        cursor.execute("SET enable_indexscan = on")
        cursor.execute("SET enable_indexonlyscan = on")

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
                    join_graphs[i], cursor)
            opt_sqls[i] = opt_sql
            opt_costs[i] = opt_cost
            opt_explains[i] = opt_explain

        # FIXME: temporary
        if est_cost < opt_costs[i]:
            # print(est_cost, opt_cost, opt_cost - est_cost)
            # pdb.set_trace()
            est_cost = opt_costs[i]

        # est_sqls.append(est_sql)
        # est_costs.append(est_cost)
        # est_explains.append(est_explain)
        ret.append((est_cost, opt_costs[i], est_explain, opt_explains[i],
            est_sql, opt_sqls[i]))

    cursor.close()
    con.close()
    # assert len(est_costs) == len(opt_costs)
    # return est_costs, opt_costs, est_explains, opt_explains, est_sqls, opt_sqls
    return ret

class JoinLoss():

    def __init__(self, user, pwd, db_host, port, db_name):
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name

        # self.sql_cost_cache_fn = "/tmp/sql_cost.pkl"
        # if os.path.isfile(self.sql_cost_cache_fn):
            # with open(self.sql_cost_cache_fn, 'rb') as handle:
                # self.sql_cost_cache = pickle.load(handle)
        # else:
            # # key: sql_key, vals: (cost, explain)
            # self.sql_cost_cache = {}

        self.opt_cache_fn = "/tmp/opt_cache.pkl"
        if os.path.isfile(self.opt_cache_fn):
            with open(self.opt_cache_fn, 'rb') as handle:
                self.opt_cache = pickle.load(handle)
        else:
            # 0, 1 is for use_indexes or not
            self.opt_cache = {}
            self.opt_cache[0] = {}
            self.opt_cache[1] = {}
            self.opt_cache[0]["costs"] = {}
            self.opt_cache[0]["explains"] = {}
            self.opt_cache[0]["sqls"] = {}

            self.opt_cache[1]["costs"] = {}
            self.opt_cache[1]["explains"] = {}
            self.opt_cache[1]["sqls"] = {}

    def compute_join_order_loss(self, sqls, join_graphs, true_cardinalities,
            est_cardinalities, baseline_join_alg, use_indexes,
            num_processes=8, postgres=True, pool=None):
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

        if not postgres:
            assert False

        return self._compute_join_order_loss_pg(sqls, join_graphs,
                true_cardinalities, est_cardinalities, num_processes,
                use_indexes, pool)

    def _compute_join_order_loss_pg(self, sqls, join_graphs, true_cardinalities,
            est_cardinalities, num_processes, use_indexes, pool):

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
        opt_costs_cache = self.opt_cache[use_indexes]["costs"]
        opt_sqls_cache = self.opt_cache[use_indexes]["sqls"]
        opt_explains_cache = self.opt_cache[use_indexes]["explains"]

        for i, sql in enumerate(sqls):
            sql_key = deterministic_hash(sql)
            if sql_key in opt_costs_cache:
                opt_costs[i] = opt_costs_cache[sql_key]
                opt_explains[i] = opt_explains_cache[sql_key]
                opt_sqls[i] = opt_sqls_cache[sql_key]

        if pool is None:
            # single threaded case for debugging
            assert False
            costs = []
            for i, sql in enumerate(sqls):
                costs.append(compute_join_order_loss_pg_single(sql,
                    join_graphs[i], true_cardinalities[i], est_cardinalities[i],
                    None, None, None, use_indexes, self.user,
                    self.pwd, self.db_host, self.port,
                    self.db_name))
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
                    self.port, self.db_name))

            all_costs = pool.starmap(compute_join_order_loss_pg_single, par_args)

        new_seen = False
        for num_proc, costs in enumerate(all_costs):
            start_idx = int(num_proc * batch_size)
            for i, (est, opt, est_explain, opt_explain, est_sql, opt_sql) \
                        in enumerate(costs):
                sql = sqls[start_idx + i]
                sql_key = deterministic_hash(sql)
                # est_costs.append(est)
                # est_explains.append(est_explain)
                # est_sqls.append(est_sql)

                est_costs[start_idx+i] = est
                est_explains[start_idx+i] = est_explain
                est_sqls[start_idx+i] = est_sql
                opt_costs[start_idx+i] = opt
                opt_explains[start_idx+i] = opt_explains
                opt_sqls[start_idx+i] = opt_sql
                # opt_costs.append(opt)
                # opt_explains.append(opt_explain)
                # opt_sqls.append(opt_sql)

                if sql_key not in opt_costs_cache:
                    opt_costs_cache[sql_key] = opt
                    opt_explains_cache[sql_key] = opt_explain
                    opt_sqls_cache[sql_key] = opt_sql
                    new_seen = True

        if new_seen:
            with open(self.opt_cache_fn, 'wb') as handle:
                pickle.dump(self.opt_cache, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        return np.array(est_costs), np.array(opt_costs), est_explains, \
    opt_explains, est_sqls, opt_sqls

import psycopg2 as pg
import getpass
from db_utils.utils import *
from sql_rep.utils import extract_aliases
import multiprocessing as mp
import math
from cardinality_estimation.flow_loss import *

import pdb
import klepto

system = platform.system()
if system == 'Linux':
    lib_file = "libflowloss.so"
else:
    lib_file = "libflowloss.dylib"

lib_dir = "./flow_loss_cpp"
lib_file = lib_dir + "/" + lib_file
fl_cpp = CDLL(lib_file, mode=RTLD_GLOBAL)

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

def get_join_cost_sql(sql_order, true_cardinalities,
        join_graph, use_indexes, user, pwd, db_host, port, db_name):
    try:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd)
    except:
        con = pg.connect(port=port,dbname=db_name,
                user=user,password=pwd, host=db_host)

    cursor = con.cursor()
    cursor.execute("SET join_collapse_limit = {}".format(1))
    cursor.execute("SET from_collapse_limit = {}".format(1))
    # sql_order = "EXPLAIN " + sql_order
    sql_order = " explain (format json) " + sql_order

    cursor.execute(sql_order)
    explain = cursor.fetchall()
    est_join_order_sql, est_join_ops, scan_ops = get_pg_join_order(join_graph,
            explain)
    leading_hint = get_leading_hint(join_graph, explain)

    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))
    # cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    # cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))
    if not use_indexes:
        cursor.execute("SET enable_indexscan = off")
        cursor.execute("SET enable_indexonlyscan = off")
    else:
        cursor.execute("SET enable_indexscan = on")
        cursor.execute("SET enable_indexonlyscan = on")

    est_opt_sql = nx_graph_to_query(join_graph,
            from_clause=est_join_order_sql)

    # add the join ops etc. information
    cost_sql = _get_modified_sql(est_opt_sql, true_cardinalities,
            None, leading_hint, None)

    # exec_sql = _get_modified_sql(est_opt_sql, est_cardinalities,
            # None, None, None)

    est_cost, est_explain = _get_cost(cost_sql, cursor)
    # debug_leading = get_leading_hint(join_graph, est_explain)

    cursor.close()
    con.close()
    return cost_sql, est_cost, est_explain

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
    if sql_costs is not None:
        if cost_sql_key in sql_costs.archive:
            try:
                est_cost, est_explain = sql_costs.archive[cost_sql_key]
            except:
                est_cost, est_explain = _get_cost(cost_sql, cursor)
                sql_costs.archive[cost_sql_key] = (est_cost, est_explain)
        else:
            est_cost, est_explain = _get_cost(cost_sql, cursor)
            sql_costs.archive[cost_sql_key] = (est_cost, est_explain)
    else:
        est_cost, est_explain = _get_cost(cost_sql, cursor)

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
    def set_indexes(cursor, val):
        cursor.execute("SET enable_indexscan = {}".format(val))
        cursor.execute("SET enable_indexonlyscan = {}".format(val))
        cursor.execute("SET enable_bitmapscan = {}".format(val))
        cursor.execute("SET enable_tidscan = {}".format(val))

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
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))
    cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))
    if not use_indexes:
        set_indexes(cursor, "off")
    else:
        set_indexes(cursor, "on")

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
    elif cost_model == "nested_loop_index" \
            or cost_model == "nested_loop_index2" \
            or cost_model == "nested_loop_index3" \
            or cost_model == "nested_loop_index4" \
            or cost_model == "nested_loop_index5" \
            or cost_model == "nested_loop_index6":
        cursor.execute("SET enable_hashjoin = off")
        cursor.execute("SET enable_mergejoin = off")
        cursor.execute("SET enable_nestloop = on")
        set_indexes(cursor, "on")

    elif cost_model == "cm1" \
            or cost_model == "cm2":
        pass
    else:
        assert False

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

        print("compute postgres join error took: ", time.time()-start)
        return np.array(est_costs), np.array(opt_costs), est_explains, \
    opt_explains, est_sqls, opt_sqls

def fl_cpp_get_flow_loss(samples, source_node, cost_key,
        all_ests, known_costs, cost_model, trueC_vecs):
    start = time.time()
    costs = []
    farchive = klepto.archives.dir_archive("./flow_info_archive",
            cached=True, serialized=True)
    new_seen = False
    for i, sample in enumerate(samples):
        if known_costs and known_costs[i] is not None:
            costs.append(known_costs[i])
            continue
        qkey = deterministic_hash(sample["sql"])
        if qkey in farchive.archive:
            subsetg_vectors = farchive.archive[qkey]
            assert len(subsetg_vectors) == 7
            totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                    edges_cost_node2, final_node = subsetg_vectors
        else:
            new_seen = True
            # this must be for true cards
            assert all_ests is None
            subsetg_vectors = list(get_subsetg_vectors(sample))
            assert len(subsetg_vectors) == 7

        totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                edges_cost_node2, final_node = subsetg_vectors
        nodes = list(sample["subset_graph"].nodes())
        nodes.sort()

        if qkey in trueC_vecs:
            # print("found trueC_vec!")
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
                    edges_cost_node2, nilj, edges_head, edges_tail, cost_model)
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
                    edges_cost_node2, nilj, edges_head, edges_tail, cost_model)
            trueC_vecs[qkey] = trueC_vec

        Gv2 = np.zeros(len(totals), dtype=np.float32)
        Gv2[final_node] = 1.0
        Gv2 = to_variable(Gv2).float()
        # predC2 = to_variable(predC2).float()
        G2 = to_variable(G2).float()
        invG = torch.inverse(G2)
        v = invG @ Gv2 # vshape: Nx1
        v = v.detach().numpy()

        # TODO: we don't even need to compute the loss here if we don't want to
        loss2 = np.zeros(1, dtype=np.float32)
        assert Q2.dtype == np.float32
        assert v.dtype == np.float32
        if isinstance(trueC_vec, torch.Tensor):
            trueC_vec = trueC_vec.detach().numpy()
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
        if new_seen:
            # was for true cards
            farchive.archive[qkey] = subsetg_vectors

    return costs

def get_shortest_path_costs(samples, source_node, cost_key,
        all_ests, known_costs, cost_model):
    '''
    @ret: cost of the given path in subsetg.
    '''
    costs = []
    for i in range(len(samples)):
        if known_costs and known_costs[i] is not None:
            costs.append(known_costs[i])
            continue
        subsetg = samples[i]["subset_graph_paths"]

        # this should already be pre-computed
        if cost_key != "cost":
            ests = all_ests[i]
            compute_costs(subsetg, cost_model, cost_key=cost_key,
                    ests=ests)

        # TODO: precompute..
        nodes = list(subsetg.nodes())
        nodes.sort(key=lambda x: len(x))
        final_node = nodes[-1]
        path = nx.shortest_path(subsetg, final_node,
                source_node, weight=cost_model+cost_key)

        cost = 0.0
        for i in range(len(path)-1):
            cost += subsetg[path[i]][path[i+1]][cost_model+"cost"]
        assert cost >= 1
        costs.append(cost)

    return costs

class PlanError():

    def __init__(self, cost_model, loss_type,
            user=None, pwd=None, db_host=None, port=None, db_name=None):
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name

        self.cost_model = cost_model
        self.source_node = SOURCE_NODE
        self.loss_type = loss_type
        if loss_type == "plan-loss":
            self.loss_func = get_shortest_path_costs
        elif loss_type == "flow-loss":
            self.loss_func = fl_cpp_get_flow_loss

        self.subsetgs = {}
        self.opt_costs = {}
        if self.loss_type == "flow-loss":
            self.trueC_vecs = {}

    def compute_loss(self, qreps, ests, pool=None):
        '''
        @ests: [dicts] of estimates
        '''
        start = time.time()
        subsetgs = []
        opt_costs = []
        new_opt_cost = False
        for qrep in qreps:
            qkey = deterministic_hash(qrep["sql"])
            if qkey in self.opt_costs:
                opt_costs.append(self.opt_costs[qkey])
            else:
                opt_costs.append(None)
                new_opt_cost = True

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
                        self.source_node, "cost", None,
                        opt_costs[start_idx:end_idx], self.cost_model))
                par_args.append((qreps[start_idx:end_idx],
                    self.source_node, "est_cost", ests[start_idx:end_idx],
                    None, self.cost_model))

            if new_opt_cost:
                opt_costs = []
                opt_costs_batched = pool.starmap(self.loss_func,
                        opt_par_args)
                for c in opt_costs_batched:
                    opt_costs += c

                for i, qrep in enumerate(qreps):
                    qkey = deterministic_hash(qrep["sql"])
                    self.opt_costs[qkey] = opt_costs[i]

            all_costs = []
            all_costs_batched = pool.starmap(self.loss_func,
                    par_args)
            for c in all_costs_batched:
                all_costs += c

        loss = np.mean(np.array(all_costs) - np.array(opt_costs))
        if loss < 0.0:
            print("negative loss for ", self.loss_type)
            print(loss)
            # pdb.set_trace()
        print("compute {} took: {}".format(self.loss_type, time.time()-start))
        return np.array(opt_costs), np.array(all_costs)

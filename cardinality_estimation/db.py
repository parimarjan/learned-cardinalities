import psycopg2 as pg
import pdb
import time
import random
from db_utils.utils import *
from db_utils.query_generator import QueryGenerator
from utils.utils import *
from cardinality_estimation.query import Query
import klepto
import time
from collections import OrderedDict, defaultdict
from multiprocessing import Pool
import concurrent.futures
import re

SUBQUERY_TIMEOUT = 3*60000
class DB():

    def __init__(self, user, pwd, db_host, port, db_name,
            cache_dir="./sql_cache"):
        '''
        Creates a conn to the db, and then will continue to reuse that.
        Provides the following:
            - TODO
        '''
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name

        # In seconds. If query execution takes longer, then the results are
        # cached for the future
        self.execution_cache_threshold = 20

        # TODO: featurization for all elements of DB / or on per table basis
        # etc.?
        # TODO: collect other stats about each table in the DB etc.
        # get a random sample of values from each column so we can use it to
        # generate sensible queries
        self.sql_cache = klepto.archives.dir_archive(cache_dir)
        # self.sql_cache = klepto.archives.dir_archive()

        # stats on the used columns
        #   table_name : column_name : attribute : value
        #   e.g., stats["title"]["id"]["max_value"] = 1010112
        #         stats["title"]["id"]["type"] = int
        #         stats["title"]["id"]["num_values"] = x
        self.column_stats = OrderedDict()
        # self.max_discrete_featurizing_buckets = 100
        # self.max_discrete_featurizing_buckets = 20
        # self.max_discrete_featurizing_buckets = 10
        self.max_discrete_featurizing_buckets = None

        # generally, these would be range queries, but they can be "=", or "in"
        # queries as well, and we will support upto 10 such values
        self.continuous_feature_size = 2

        self.featurizer = None
        self.cmp_ops = set()
        self.tables = set()
        self.joins = set()
        self.aliases = {}
        self.cmp_ops_onehot = {}
        self.regex_cols = set()

        # for pgm stuff
        self.templates = []
        self.foreign_keys = {}    # table.key : table.key
        self.primary_keys = set() # table.key
        self.alias_to_keys = defaultdict(set)

        # self.flow_node_features = ["in_edges", "out_edges", "paths",
                # "tolerance", "pg_flow"]

        self.max_in_degree = 0
        self.max_out_degree = 0
        self.max_paths = 0
        self.feat_num_paths = False

        # the node-edge connectivities stay constant through templates
        # key: template_name
        # val: {}:
        #   key: node name
        #   val: tolerance in powers of 10
        self.template_info = {}

        # things like tolerances, flows need to be computed on a per query
        # basis (maybe we should not precompute these?)
        self.query_info = {}

    def get_entropies(self):
        '''
        pairwise entropies among all columns of the db?
        '''
        pass

    def execute(self, sql, timeout=None):
        '''
        executes the given sql on the DB, and caches the results in a
        persistent store if it took longer than self.execution_cache_threshold.
        '''
        hashed_sql = deterministic_hash(sql)
        if hashed_sql in self.sql_cache:
            print("loaded {} from in memory cache".format(hashed_sql))
            return self.sql_cache[hashed_sql]

        # archive only considers the stuff stored in disk
        if hashed_sql in self.sql_cache.archive:
            # load it and return
            print("loaded {} from cache".format(hashed_sql))
            # pdb.set_trace()
            return self.sql_cache.archive[hashed_sql]
        start = time.time()

        ## FIXME: get stuff that works on both places
        # works on aws
        # con = pg.connect(user=self.user, port=self.port,
                # password=self.pwd, database=self.db_name)

        # works on chunky
        con = pg.connect(user=self.user, host=self.db_host, port=self.port,
                password=self.pwd, database=self.db_name)
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
            return [[1000000]]
            # return None
        try:
            exp_output = cursor.fetchall()
        except Exception as e:
            print(e)
            exp_output = None

        cursor.close()
        con.close()
        end = time.time()
        if (end - start > self.execution_cache_threshold):
            # print(hashed_sql)
            # print(sql)
            # print(exp_output)
            # pdb.set_trace()
            self.sql_cache.archive[hashed_sql] = exp_output
        return exp_output

    def save_cache(self):
        print("saved cache to disk")
        self.sql_cache.dump()

    def init_featurizer(self, heuristic_features=True,
            num_tables_feature=False,
            max_discrete_featurizing_buckets=10,
            flow_features = True,
            feat_num_paths= False, feat_flows=False,
            feat_pg_costs = True, feat_tolerance=True,
            feat_template=False, feat_pg_path=True,
            feat_rel_pg_ests=True, feat_join_graph_neighbors=True,
            feat_rel_pg_ests_onehot=True,
            feat_pg_est_one_hot=True,
            cost_model=None):
        '''
        Sets up a transformation to 1d feature vectors based on the registered
        templates seen in get_samples.
        Generates a featurizer dict:
            {table_name: (idx, num_vals)}
            {join_key: (idx, num_vals)}
            {pred_column: (idx, num_vals)}
        where the idx refers to the elements position in the feature vector,
        and num_vals refers to the number of values it will occupy.
        E.g. TODO.
        '''
        indexes = self.execute(INDEX_LIST_CMD)
        self.cost_model = cost_model
        self.heuristic_features = heuristic_features
        self.flow_features = flow_features
        # let's figure out the feature len based on db.stats
        assert self.featurizer is None
        # only need to know the number of tables for table features
        self.table_featurizer = {}
        for i, table in enumerate(sorted(self.tables)):
            self.table_featurizer[table] = i
        self.join_featurizer = {}

        for i, join in enumerate(sorted(self.joins)):
            self.join_featurizer[join] = i

        self.max_discrete_featurizing_buckets = max_discrete_featurizing_buckets
        self.featurizer = {}
        self.num_cols = len(self.column_stats)

        self.pred_features_len = 0
        for i, cmp_op in enumerate(self.cmp_ops):
            self.cmp_ops_onehot[cmp_op] = i

        # to find the number of features, need to go over every column, and
        # choose how many spots to keep for them
        for col, info in self.column_stats.items():
            pred_len = 0
            # for operator type
            pred_len += len(self.cmp_ops)

            if heuristic_features:
                # for pg_est
                pred_len += 1

            if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    and info["num_values"] >= self.max_discrete_featurizing_buckets:
                # then use min-max normalization, no matter what
                # only support range-queries, so lower / and upper predicate
                pred_len += self.continuous_feature_size
                continuous = True
            else:
                # use 1-hot encoding
                num_buckets = min(self.max_discrete_featurizing_buckets,
                        info["num_values"])
                pred_len += num_buckets
                continuous = False
                if col in self.regex_cols:
                    # give it more space...
                    pred_len += 2

                    ## extra space for regex buckets
                    pred_len += num_buckets

            self.featurizer[col] = (self.pred_features_len, pred_len, continuous)
            self.pred_features_len += pred_len

        # for pg_est of all features combined
        if heuristic_features:
            self.pred_features_len += 1

        # for num_tables present
        if num_tables_feature:
            self.pred_features_len += 1
        # FIXME:
        self.pred_features_len += 1

        self.flow_features = flow_features
        self.feat_num_paths = feat_num_paths
        self.feat_flows = feat_flows
        self.feat_pg_costs = feat_pg_costs
        self.feat_tolerance = feat_tolerance
        self.feat_template = feat_template
        self.feat_pg_path = feat_pg_path
        self.feat_rel_pg_ests = feat_rel_pg_ests
        self.feat_rel_pg_ests_onehot = feat_rel_pg_ests_onehot
        self.feat_join_graph_neighbors = feat_join_graph_neighbors
        self.feat_pg_est_one_hot = feat_pg_est_one_hot

        self.PG_EST_BUCKETS = 7
        if flow_features:
            self.flow_features = flow_features
            # num flow features: concat of 1-hot vectors
            self.num_flow_features = 0
            self.num_flow_features += self.max_in_degree+1
            self.num_flow_features += self.max_out_degree+1

            self.num_flow_features += len(self.aliases)

            # for heuristic estimate for the node
            self.num_flow_features += 1

            # for normalized value of num_paths
            if self.feat_num_paths:
                self.num_flow_features += 1
            if self.feat_pg_costs:
                self.num_flow_features += 1
            if self.feat_tolerance:
                # 1-hot vector based on dividing/multiplying value by 10...10^4
                self.num_flow_features += 4
            if self.feat_flows:
                self.num_flow_features += 1

            if self.feat_template:
                self.num_flow_features += len(self.templates)

            if self.feat_pg_path:
                self.num_flow_features += 1

            if self.feat_rel_pg_ests:
                # current node size est, relative to total cost
                self.num_flow_features += 1

                # current node est, relative to all neighbors in the join graph
                # we will hard code the neighbor into a 1-hot vector
                self.num_flow_features += len(self.table_featurizer)

            if self.feat_rel_pg_ests_onehot:
                self.num_flow_features += self.PG_EST_BUCKETS
                # 2x because it can be smaller or larger
                self.num_flow_features += \
                    2*len(self.table_featurizer)*self.PG_EST_BUCKETS

            if self.feat_join_graph_neighbors:
                self.num_flow_features += len(self.table_featurizer)

            if self.feat_pg_est_one_hot:
                # upto 10^7
                self.num_flow_features += self.PG_EST_BUCKETS

            # pg est for the node
            self.num_flow_features += len(self.cmp_ops)
            self.num_flow_features += 1

    def get_onehot_bucket(self, num_buckets, base, val):
        assert val >= 1.0
        for i in range(num_buckets):
            if val > base**i and val < base**(i+1):
                return i

        return num_buckets

    def get_flow_features(self, node, subsetg,
            template_name, join_graph, cmp_op):
        assert node != SOURCE_NODE
        flow_features = np.zeros(self.num_flow_features)
        cur_idx = 0
        # incoming edges
        in_degree = subsetg.in_degree(node)
        flow_features[cur_idx + in_degree] = 1.0
        cur_idx += self.max_in_degree+1
        # outgoing edges
        out_degree = subsetg.out_degree(node)
        flow_features[cur_idx + out_degree] = 1.0
        cur_idx += self.max_out_degree+1
        # num tables
        max_tables = len(self.aliases)
        nt = len(node)
        assert nt <= max_tables
        flow_features[cur_idx + nt] = 1.0
        cur_idx += max_tables

        # precomputed based stuff
        if self.feat_num_paths:
            if node in self.template_info[template_name]:
                num_paths = self.template_info[template_name][node]["num_paths"]
            else:
                num_paths = 0

            # assuming min num_paths = 0, min-max normalization
            flow_features[cur_idx] = num_paths / self.max_paths
            cur_idx += 1

        if self.feat_pg_costs:
            in_edges = subsetg.in_edges(node)
            in_cost = 0.0
            for edge in in_edges:
                in_cost += subsetg[edge[0]][edge[1]][self.cost_model + "pg_cost"]
            # normalized pg cost
            flow_features[cur_idx] = in_cost / subsetg.graph[self.cost_model + "total_cost"]
            cur_idx += 1

        if self.feat_tolerance:
            tol = subsetg.nodes()[node]["tolerance"]
            tol_idx = int(np.log10(tol))
            assert tol_idx <= 4
            flow_features[cur_idx + tol_idx-1] = 1.0
            cur_idx += 4

        if self.feat_flows:
            in_edges = subsetg.in_edges(node)
            in_flows = 0.0
            for edge in in_edges:
                in_flows += subsetg[edge[0]][edge[1]]["pg_flow"]
            # normalized pg flow
            flow_features[cur_idx] = in_flows
            cur_idx += 1

        if self.feat_template:
            tidx = 0
            for i,t in enumerate(self.templates):
                if t == template_name:
                    tidx = i
            flow_features[cur_idx + tidx] = 1.0
            cur_idx += len(self.templates)

        if self.feat_pg_path:
            if "pg_path" in subsetg.nodes()[node]:
                flow_features[cur_idx] = 1.0

        if self.feat_join_graph_neighbors:
            neighbors = nx.node_boundary(join_graph, node)
            for al in neighbors:
                table = self.aliases[al]
                tidx = self.table_featurizer[table]
                flow_features[cur_idx + tidx] = 1.0
            cur_idx += len(self.table_featurizer)

        if self.feat_rel_pg_ests:
            total_cost = subsetg.graph[self.cost_model+"total_cost"]
            pg_est = subsetg.nodes()[node]["cardinality"]["expected"]
            flow_features[cur_idx] = pg_est / total_cost
            cur_idx += 1
            neighbors = nx.node_boundary(join_graph, node)

            # neighbors in join graph
            for al in neighbors:
                # aidx = self.aliases[al]
                table = self.aliases[al]
                tidx = self.table_featurizer[table]
                ncard = subsetg.nodes()[tuple([al])]["cardinality"]["expected"]
                # TODO: should this be normalized? how?
                flow_features[cur_idx + tidx] = pg_est / ncard
                flow_features[cur_idx + tidx] /= 1e5

            cur_idx += len(self.table_featurizer)

        if self.feat_rel_pg_ests_onehot:
            total_cost = subsetg.graph[self.cost_model+"total_cost"]
            pg_est = subsetg.nodes()[node]["cardinality"]["expected"]
            # flow_features[cur_idx] = pg_est / total_cost
            pg_ratio = total_cost / float(pg_est)

            bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10, pg_ratio)
            flow_features[cur_idx+bucket] = 1.0
            cur_idx += self.PG_EST_BUCKETS

            neighbors = nx.node_boundary(join_graph, node)

            # neighbors in join graph
            for al in neighbors:
                # aidx = self.aliases[al]
                table = self.aliases[al]
                tidx = self.table_featurizer[table]
                ncard = subsetg.nodes()[tuple([al])]["cardinality"]["expected"]
                # TODO: should this be normalized? how?
                # flow_features[cur_idx + tidx] = pg_est / ncard
                # flow_features[cur_idx + tidx] /= 1e5
                if pg_est > ncard:
                    # first self.PG_EST_BUCKETS
                    bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10,
                            pg_est / float(ncard))
                    flow_features[cur_idx+bucket] = 1.0
                else:
                    bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10,
                            float(ncard) / pg_est)
                    flow_features[cur_idx+self.PG_EST_BUCKETS+bucket] = 1.0

                cur_idx += 2*self.PG_EST_BUCKETS

        if self.feat_pg_est_one_hot:
            pg_est = subsetg.nodes()[node]["cardinality"]["expected"]

            for i in range(self.PG_EST_BUCKETS):
                if pg_est > 10**i and pg_est < 10**(i+1):
                    flow_features[cur_idx+i] = 1.0
                    break

            if pg_est > 10**self.PG_EST_BUCKETS:
                flow_features[cur_idx+self.PG_EST_BUCKETS] = 1.0
            cur_idx += self.PG_EST_BUCKETS

        # pg_est for node will be added in query_dataset..
        if cmp_op is not None:
            cmp_idx = self.cmp_ops_onehot[cmp_op]
            flow_features[cur_idx + cmp_idx] = 1.0
        cur_idx += len(self.cmp_ops)

        return flow_features

    def get_table_features(self, table):
        '''
        '''
        tables_vector = np.zeros(len(self.table_featurizer))
        tables_vector[self.table_featurizer[table]] = 1.00
        return tables_vector

    def get_join_features(self, join_str):
        # TODO: split, sort join
        keys = join_str.split("=")
        keys.sort()
        keys = ",".join(keys)
        joins_vector = np.zeros(len(self.join_featurizer))
        joins_vector[self.join_featurizer[keys]] = 1.00
        return joins_vector

    def get_pred_features(self, col, val, cmp_op,
            pred_est=None):

        if pred_est is not None:
            assert self.heuristic_features
        preds_vector = np.zeros(self.pred_features_len)

        # set comparison operator 1-hot value
        cmp_op_idx, num_vals, continuous = self.featurizer[col]
        cmp_idx = self.cmp_ops_onehot[cmp_op]
        preds_vector[cmp_op_idx+cmp_idx] = 1.00

        pred_idx_start = cmp_op_idx + len(self.cmp_ops)
        num_pred_vals = num_vals - len(self.cmp_ops)
        col_info = self.column_stats[col]
        # assert num_pred_vals >= 2

        # 1 additional value for pg_est feature
        # assert num_pred_vals <= col_info["num_values"] + 1

        if pred_est:
            preds_vector[pred_idx_start + num_pred_vals] = pred_est

        if not continuous:
            if "like" in cmp_op:
                assert len(val) == 1
                num_buckets = min(self.max_discrete_featurizing_buckets,
                        col_info["num_values"])
                # first half of spaces reserved for "IN" predicates
                pred_idx_start += num_buckets
                regex_val = val[0].replace("%","")
                pred_idx = deterministic_hash(regex_val) % num_buckets
                preds_vector[pred_idx_start+pred_idx] = 1.00
                for v in regex_val:
                    pred_idx = deterministic_hash(str(v)) % num_buckets
                    preds_vector[pred_idx_start+pred_idx] = 1.00

                REGEX_USE_BIGRAMS = True
                REGEX_USE_TRIGRAMS = True
                if REGEX_USE_BIGRAMS:
                    for i,v in enumerate(regex_val):
                        if i != len(regex_val)-1:
                            pred_idx = deterministic_hash(v+regex_val[i+1]) % num_buckets
                            preds_vector[pred_idx_start+pred_idx] = 1.00

                if REGEX_USE_TRIGRAMS:
                    for i,v in enumerate(regex_val):
                        if i < len(regex_val)-2:
                            pred_idx = deterministic_hash(v+regex_val[i+1]+ \
                                    regex_val[i+2]) % num_buckets
                            preds_vector[pred_idx_start+pred_idx] = 1.00

                preds_vector[pred_idx_start + num_buckets + 1] = len(regex_val)
                if bool(re.search(r'\d', regex_val)):
                    preds_vector[pred_idx_start + num_buckets + 1] = 1

            else:
                num_buckets = min(self.max_discrete_featurizing_buckets,
                        col_info["num_values"])
                for v in val:
                    pred_idx = deterministic_hash(str(v)) % num_buckets
                    preds_vector[pred_idx_start+pred_idx] = 1.00
        else:
            # do min-max stuff
            # assert len(val) == 2
            min_val = float(col_info["min_value"])
            max_val = float(col_info["max_value"])
            min_max = [min_val, max_val]
            if isinstance(val, int):
                cur_val = float(val)
                norm_val = (cur_val - min_val) / (max_val - min_val)
                norm_val = max(norm_val, 0.00)
                norm_val = min(norm_val, 1.00)
                preds_vector[pred_idx_start+0] = norm_val
                preds_vector[pred_idx_start+1] = norm_val
            else:
                for vi, v in enumerate(val):
                    if "literal" == v:
                        v = val["literal"]
                    # handling the case when one end of the range is
                    # missing
                    if v is None and vi == 0:
                        v = min_val
                    elif v is None and vi == 1:
                        v = max_val

                    if v == 'NULL' and vi == 0:
                        v = min_val
                    elif v == 'NULL' and vi == 1:
                        v = max_val

                    cur_val = float(v)
                    norm_val = (cur_val - min_val) / (max_val - min_val)
                    norm_val = max(norm_val, 0.00)
                    norm_val = min(norm_val, 1.00)
                    preds_vector[pred_idx_start+vi] = norm_val

        return preds_vector

    def get_features(self, subgraph, true_sel=None):
        '''
        @subgraph:
        '''
        tables_vector = np.zeros(len(self.table_featurizer))
        preds_vector = np.zeros(self.pred_features_len)

        for nd in subgraph.nodes(data=True):
            node = nd[0]
            data = nd[1]
            tables_vector[self.table_featurizer[data["real_name"]]] = 1.00

            for i, col in enumerate(data["pred_cols"]):
                # add pred related feature
                val = data["pred_vals"][i]
                cmp_op = data["pred_types"][i]
                cmp_op_idx, num_vals, continuous = self.featurizer[col]
                cmp_idx = self.cmp_ops_onehot[cmp_op]
                preds_vector[cmp_op_idx+cmp_idx] = 1.00

                pred_idx_start = cmp_op_idx + len(self.cmp_ops)
                num_pred_vals = num_vals - len(self.cmp_ops)
                col_info = self.column_stats[col]
                # assert num_pred_vals >= 2
                assert num_pred_vals <= col_info["num_values"]
                if cmp_op == "in" or \
                        "like" in cmp_op or \
                        cmp_op == "eq":

                    if continuous:
                        assert len(val) <= self.continuous_feature_size
                        min_val = float(col_info["min_value"])
                        max_val = float(col_info["max_value"])
                        for vi, v in enumerate(val):
                            v = float(v)
                            normalized_val = (v - min_val) / (max_val - min_val)
                            preds_vector[pred_idx_start+vi] = 1.00
                    else:
                        num_buckets = min(self.max_discrete_featurizing_buckets,
                                col_info["num_values"])
                        assert num_pred_vals == num_buckets
                        # turn to 1 all the qualifying indexes in the 1-hot vector
                        if "like" in cmp_op:
                            print(cmp_op)
                            pdb.set_trace()
                        for v in val:
                            pred_idx = deterministic_hash(v) % num_buckets
                            preds_vector[pred_idx_start+pred_idx] = 1.00

                elif cmp_op in RANGE_PREDS:
                    assert cmp_op == "lt"

                    # does not have to be MIN / MAX, if there are very few values
                    # OR if the predicate is on string data, e.g., BETWEEN 'A' and 'F'

                    if not continuous:
                        # FIXME: temporarily, just treat it as discrete data
                        num_buckets = min(self.max_discrete_featurizing_buckets,
                                col_info["num_values"])
                        for v in val:
                            pred_idx = deterministic_hash(str(v)) % num_buckets
                            preds_vector[pred_idx_start+pred_idx] = 1.00
                    else:
                        # do min-max stuff
                        assert len(val) == 2

                        min_val = float(col_info["min_value"])
                        max_val = float(col_info["max_value"])
                        min_max = [min_val, max_val]
                        for vi, v in enumerate(val):
                            # handling the case when one end of the range is
                            # missing
                            if v is None and vi == 0:
                                v = min_val
                            elif v is None and vi == 1:
                                v = max_val

                            cur_val = float(v)
                            norm_val = (cur_val - min_val) / (max_val - min_val)
                            norm_val = max(norm_val, 0.00)
                            norm_val = min(norm_val, 1.00)
                            preds_vector[pred_idx_start+vi] = norm_val

        # based on edges, add the join conditions

        # TODO: combine all vectors, or not.
        if true_sel is not None:
            preds_vector[-1] = true_sel

        return preds_vector

    def gen_subqueries(self, query):
        '''
        @query: Query object.
        @ret: [Query objects] corresponding to each subquery of query,
        excluding crossjoins.
        '''
        hashed_key = deterministic_hash("subquery: " + query.query)
        queries = []
        if hashed_key in self.sql_cache.archive:
            # print("loading hashed key")
            queries = self.sql_cache.archive[hashed_key]
            return queries
        start = time.time()
        # sql_subqueries = query.sql_subqueries
        sql_query = query.query
        sql_subqueries = gen_all_subqueries(sql_query)
        # TODO: create query objects for each subqueries
        queries = []

        with Pool(processes=8) as pool:
            args = [(cur_query, self.user, self.db_host, self.port,
                self.pwd, self.db_name, None,
                self.execution_cache_threshold, self.sql_cache) for
                cur_query in sql_subqueries]
            all_query_objs = pool.starmap(sql_to_query_object, args)
        for q in all_query_objs:
            queries.append(q)
        print("{} subqueries generated in {} seconds".format(
            len(queries), time.time() - start))
        self.sql_cache[hashed_key] = queries
        self.sql_cache.dump()
        return queries

    def update_db_stats(self, qrep, flow_features):
        '''
        @query: Query object
        '''
        if qrep["template_name"] not in self.templates:
            self.templates.append(qrep["template_name"])

        cur_columns = []
        for node, info in qrep["join_graph"].nodes(data=True):
            for i, cmp_op in enumerate(info["pred_types"]):
                self.cmp_ops.add(cmp_op)
                if "like" in cmp_op:
                    self.regex_cols.add(info["pred_cols"][i])

            if node not in self.aliases:
                self.aliases[node] = info["real_name"]
                self.tables.add(info["real_name"])
            for col in info["pred_cols"]:
                cur_columns.append(col)

        # FIXME: might not need to parse this again...
        joins = extract_join_clause(qrep["sql"])
        for join in joins:
            keys = join.split("=")
            keys.sort()
            keys = ",".join(keys)
            self.joins.add(keys)

        if flow_features:
            flow_start = time.time()
            # TODO: track max incoming / outgoing edges, so we can have a
            # one-hot vector of #incoming / #outgoing
            subsetg = qrep["subset_graph"]
            node_list = list(subsetg.nodes())
            node_list.sort(key = lambda v: len(v))
            dest = node_list[-1]
            node_list.sort()
            info = {}
            tmp_name = qrep["template_name"]

            for node in subsetg.nodes():
                in_degree = subsetg.in_degree(node)
                if in_degree > self.max_in_degree:
                    self.max_in_degree = in_degree

                out_degree = subsetg.out_degree(node)
                if out_degree > self.max_out_degree:
                    self.max_out_degree = out_degree

                # TODO: compute flow / tolerances
                if tmp_name in self.template_info:
                    continue
                info[node] = {}
                # paths from node -> dest, but edges are reversed in our
                # representation
                if self.feat_num_paths:
                    all_paths = nx.all_simple_paths(subsetg, dest, node)
                    num_paths = len(list(all_paths))
                    if num_paths > self.max_paths:
                        self.max_paths = num_paths
                    info[node]["num_paths"] = num_paths

            self.template_info[tmp_name] = info

            if time.time() - flow_start > 10:
                print("generated stats for flows in: ", time.time()-flow_start)

        updated_cols = []
        for column in cur_columns:
            if column in self.column_stats:
                continue
            # need to load it. first check if it is in the cache, else
            # regenerate it.
            hashed_stats = deterministic_hash(column)
            updated_cols.append(column)
            column_stats = {}
            table = column[0:column.find(".")]
            if table in self.aliases:
                table = ALIAS_FORMAT.format(TABLE = self.aliases[table],
                                    ALIAS = table)
            min_query = MIN_TEMPLATE.format(TABLE = table,
                                            COL   = column)
            max_query = MAX_TEMPLATE.format(TABLE = table,
                                            COL   = column)
            unique_count_query = UNIQUE_COUNT_TEMPLATE.format(FROM_CLAUSE = table,
                                                      COL = column)
            total_count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE = table)
            unique_vals_query = UNIQUE_VALS_TEMPLATE.format(FROM_CLAUSE = table,
                                                            COL = column)

            # TODO: move to using cached_execute
            column_stats[column] = {}
            column_stats[column]["min_value"] = self.execute(min_query)[0][0]
            column_stats[column]["max_value"] = self.execute(max_query)[0][0]
            column_stats[column]["num_values"] = \
                    self.execute(unique_count_query)[0][0]
            column_stats[column]["total_values"] = \
                    self.execute(total_count_query)[0][0]

            # only store all the values for tables with small alphabet
            # sizes (so we can use them for things like the PGM).
            # Otherwise, it bloats up the cache.
            if column_stats[column]["num_values"] <= 5000:
                column_stats[column]["unique_values"] = \
                        self.execute(unique_vals_query)
            else:
                column_stats[column]["unique_values"] = None

            self.sql_cache.archive[hashed_stats] = column_stats
            self.column_stats.update(column_stats)

        if len(updated_cols) > 0:
            print("generated statistics for:", ",".join(updated_cols))

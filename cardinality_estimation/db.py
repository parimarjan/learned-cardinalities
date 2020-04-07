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

    def init_featurizer(self, heuristic_features=True, num_tables_feature=False,
            max_discrete_featurizing_buckets=10):
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
        self.heuristic_features = heuristic_features
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

            self.featurizer[col] = (self.pred_features_len, pred_len, continuous)
            self.pred_features_len += pred_len

        # for pg_est of all features combined
        if heuristic_features:
            self.pred_features_len += 1

        # for num_tables present
        if num_tables_feature:
            self.pred_features_len += 1

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
                regex_val = val[0].replace("%","")
                # pred_idx = deterministic_hash(regex_val) % num_buckets
                # preds_vector[pred_idx_start+pred_idx] = 1.00
                for v in regex_val:
                    pred_idx = deterministic_hash(str(v)) % num_buckets
                    preds_vector[pred_idx_start+pred_idx] = 1.00

                REGEX_USE_BIGRAMS = True
                if REGEX_USE_BIGRAMS:
                    for i,v in enumerate(regex_val):
                        if i != len(regex_val)-1:
                            pred_idx = deterministic_hash(v+regex_val[i+1]) % num_buckets
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

    def update_db_stats(self, qrep):
        '''
        @query: Query object
        '''
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

    def update_db_stats_old(self, query):
        '''
        @query: Query object
        '''
        for cmp_op in query.cmp_ops:
            self.cmp_ops.add(cmp_op)

        joins = extract_join_clause(query.query)
        for join in joins:
            keys = join.split("=")
            keys.sort()
            keys = ",".join(keys)
            self.joins.add(keys)

        for i, alias in enumerate(query.aliases):
            if alias not in self.aliases:
                self.aliases[alias] = query.table_names[i]
                self.tables.add(query.table_names[i])

        all_columns = []
        for column in query.pred_column_names:
            if column in self.column_stats:
                continue
            # need to load it. first check if it is in the cache, else
            # regenerate it.
            hashed_stats = deterministic_hash(column)
            all_columns.append(column)
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

        if len(all_columns) > 0:
            print("generated statistics for:", ",".join(all_columns))

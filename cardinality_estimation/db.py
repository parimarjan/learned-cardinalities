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
from collections import OrderedDict
from multiprocessing import Pool
import concurrent.futures

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
        # self.max_discrete_feauturizing_buckets = 100
        self.max_discrete_feauturizing_buckets = 20
        # generally, these would be range queries, but they can be "=", or "in"
        # queries as well, and we will support upto 10 such values
        self.continuous_feature_size = 10

        self.featurizer = None
        self.cmp_ops = set()
        self.tables = set()
        self.aliases = {}
        self.cmp_ops_onehot = {}

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

    def init_featurizer(self, heuristic_features=True, num_tables_feature=True):
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
        # let's figure out the feature len based on db.stats
        self.featurizer = {}
        self.num_tables = len(self.tables)
        self.num_tables_feature = num_tables_feature
        self.num_cols = len(self.column_stats)
        self.feature_len = 0
        for table in self.tables:
            self.featurizer[table] = (self.feature_len, 1, False)
            self.feature_len += 1

        for i, cmp_op in enumerate(self.cmp_ops):
            self.cmp_ops_onehot[cmp_op] = i

        assert self.feature_len == self.num_tables
        # FIXME: add join keys
        # self.feature_len += self.num_cols

        # to find the number of features, need to go over every column, and
        # choose how many spots to keep for them
        for col, info in self.column_stats.items():
            pred_len = 0
            # for operator type
            pred_len += len(self.cmp_ops)
            if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    and info["num_values"] >= self.max_discrete_feauturizing_buckets:
                # then use min-max normalization, no matter what
                # only support range-queries, so lower / and upper predicate
                pred_len += self.continuous_feature_size
                continuous = True
            else:
                # use 1-hot encoding
                num_buckets = min(self.max_discrete_feauturizing_buckets, info["num_values"])
                pred_len += num_buckets
                continuous = False

            self.featurizer[col] = (self.feature_len, pred_len, continuous)
            self.feature_len += pred_len

            if heuristic_features:
                self.feature_len += 1

            # for num_tables present
            if self.num_tables_feature:
                self.feature_len += 1

    def get_features(self, query, heuristic_features=True):
        '''
        TODO: add different featurization options.
        @query: Query object
        '''
        if self.featurizer is None:
            self.init_featurizer(heuristic_features)
        feature_vector = np.zeros(self.feature_len)
        for table in query.table_names:
            idx, _, _ = self.featurizer[table]
            feature_vector[idx] = 1.00

        # TODO: add joins

        for i, col in enumerate(query.pred_column_names):
            cmp_op = query.cmp_ops[i]
            # turn the element corresponding to the comparison operator as 1
            try:
                cmp_op_idx, num_vals, continuous = self.featurizer[col]
            except:
                print(self.column_stats.keys())
                print(self.featurizer.keys())
                print(col)
                pdb.set_trace()
            cmp_idx = self.cmp_ops_onehot[cmp_op]
            feature_vector[cmp_op_idx+cmp_idx] = 1.00

            # now, for featurizing the query predicate value
            pred_idx_start = cmp_op_idx + len(self.cmp_ops)
            num_pred_vals = num_vals - len(self.cmp_ops)
            col_info = self.column_stats[col]
            # temporary assertions only valid for the cases we have considered
            # so far (lte/lt, in)
            assert num_pred_vals >= 2
            assert num_pred_vals <= col_info["num_values"]
            # predicate featurization will depend on the type of cmp_op
            val = query.vals[i]

            # TODO: we can't assume if it is continuous data OR discrete
            # data just based on the predicate

            if cmp_op == "in" or \
                    "like" in cmp_op or \
                    cmp_op == "eq":

                # bandaid...
                try:
                    if isinstance(val, dict):
                        val = [val["literal"]]
                    elif not hasattr(val, "__len__"):
                        val = [val]
                    elif isinstance(val[0], dict):
                        val = val[0]["literal"]
                    val = set(val)
                except Exception as e:
                    print(e)
                    print(val)
                    pdb.set_trace()

                if continuous:
                    assert len(val) <= self.continuous_feature_size
                    min_val = float(col_info["min_value"])
                    max_val = float(col_info["max_value"])
                    for vi, v in enumerate(val):
                        v = float(v)
                        normalized_val = (v - min_val) / (max_val - min_val)
                        feature_vector[pred_idx_start+vi] = 1.00
                else:
                    num_buckets = min(self.max_discrete_feauturizing_buckets,
                            col_info["num_values"])
                    assert num_pred_vals == num_buckets
                    # if num_pred_vals != num_buckets:
                        # print("num_pred_vals != num_buckets!")
                        # # print(query)
                        # print(num_pred_vals, num_buckets)
                        # pdb.set_trace()
                    ## FIXME! this assert fails sometimes
                    # assert num_pred_vals == num_buckets
                    # turn to 1 all the qualifying indexes in the 1-hot vector
                    # try:
                    for v in val:
                        pred_idx = deterministic_hash(v) % num_buckets
                        feature_vector[pred_idx_start+pred_idx] = 1.00
                    # except Exception as e:
                        # print(e)
                        # pdb.set_trace()
            elif cmp_op in RANGE_PREDS:
                assert cmp_op == "lt"

                # does not have to be MIN / MAX, if there are very few values
                # OR if the predicate is on string data, e.g., BETWEEN 'A' and 'F'

                if not continuous:
                    # FIXME: temporarily, just treat it as discrete data
                    num_buckets = min(self.max_discrete_feauturizing_buckets,
                            col_info["num_values"])
                    for v in val:
                        pred_idx = deterministic_hash(str(v)) % num_buckets
                        feature_vector[pred_idx_start+pred_idx] = 1.00
                else:
                    # do min-max stuff
                    assert len(val) == 2

                    min_val = float(col_info["min_value"])
                    max_val = float(col_info["max_value"])
                    min_max = [min_val, max_val]
                    for vi, v in enumerate(val):
                        if v is None and vi == 0:
                            v = min_val
                        elif v is None and vi == 1:
                            v = max_val

                        cur_val = float(v)
                        norm_val = (cur_val - min_val) / (max_val - min_val)
                        norm_val = max(norm_val, 0.00)
                        norm_val = min(norm_val, 1.00)
                        feature_vector[pred_idx_start+vi] = norm_val

            else:
                assert False

        pg_est = query.pg_count / query.total_count
        if heuristic_features and self.num_tables_feature:
            feature_vector[-2] = pg_est

        elif heuristic_features and not self.num_tables_feature:
            feature_vector[-1] = pg_est

        if self.num_tables_feature:
            feature_vector[-1] = len(query.froms)

        return feature_vector

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

    def update_db_stats(self, query_template,
            create_indexes=False):
        '''
        '''
        if "SELECT COUNT" not in query_template:
            print(query_template)
            assert False

        pred_columns, pred_types, pred_vals = extract_predicates(query_template)
        for cmp_op in pred_types:
            self.cmp_ops.add(cmp_op)
        from_clauses, aliases, tables = extract_from_clause(query_template)
        self.aliases.update(aliases)
        self.tables.update(tables)

        if create_indexes:
            for alias, table_name in self.aliases.items():
                print(alias, table_name)
                for pred_column in pred_columns:
                    if " " + alias in " " + pred_column:
                        print(pred_column)
                        column_name = pred_column.split(".")[1]
                        index_cmd = CREATE_INDEX_TMP.format(COLUMN =
                                column_name, TABLE = table_name,
                                INDEX_NAME = table_name + "_" + column_name)
                        print(index_cmd)
                        self.execute(index_cmd)
                        # pdb.set_trace()

            pdb.set_trace()

        # TODO: load sql cache in memory?
        DEBUG = True
        for column in pred_columns:
            if column in self.column_stats:
                continue
            # need to load it. first check if it is in the cache, else
            # regenerate it.
            hashed_stats = deterministic_hash(column)

            if hashed_stats in self.sql_cache.archive and not DEBUG:
                column_stats = self.sql_cache.archive[hashed_stats]
                print("loading column stats from cache: ", column_stats.keys())
                self.column_stats.update(column_stats)
            else:
                print("need to generate stuff for column: ", column)
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

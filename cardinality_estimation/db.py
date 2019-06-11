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
        # self.con = pg.connect(user=self.user, host=self.db_host, port=self.port,
                # password=self.pwd, database=self.db_name)
        self.sql_cache = klepto.archives.dir_archive(cache_dir)
        # self.sql_cache = klepto.archives.dir_archive()

        # stats on the used columns
        #   table_name : column_name : attribute : value
        #   e.g., stats["title"]["id"]["max_value"] = 1010112
        #         stats["title"]["id"]["type"] = int
        #         stats["title"]["id"]["num_values"] = x
        self.column_stats = OrderedDict()
        self.max_discrete_feauturizing_buckets = 1000
        self.featurizer = None
        self.cmp_ops = set()
        self.tables = set()
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
        exp_output = cursor.fetchall()
        cursor.close()
        con.close()
        end = time.time()
        if (end - start > self.execution_cache_threshold):
            # print(hashed_sql)
            # print(sql)
            # print(exp_output)
            # pdb.set_trace()
            self.sql_cache[hashed_sql] = exp_output
        return exp_output

    def save_cache(self):
        print("saved cache to disk")
        self.sql_cache.dump()

    def init_featurizer(self):
        '''
        Sets up a transformation to 1d feature vectors based on the registered
        templates seen in get_samples.
        Generates a featurizer dict:
            {table_name: (idx, num_vals)}
            {join_key: (idx, num_vals)}
            {pred: (idx, num_vals)}
        '''
        # let's figure out the feature len based on db.stats
        self.featurizer = {}
        self.num_tables = len(self.tables)
        self.num_cols = len(self.column_stats)
        self.feature_len = 0
        for table in self.tables:
            self.featurizer[table] = (self.feature_len, 1)
            self.feature_len += 1

        # FIXME: the one-hot order of cmp_ops
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
                pred_len += 2
            else:
                # use 1-hot encoding
                num_buckets = min(self.max_discrete_feauturizing_buckets, info["num_values"])
                pred_len += num_buckets

            self.featurizer[col] = (self.feature_len, pred_len)
            self.feature_len += pred_len

    def get_features(self, query):
        '''
        TODO: add different featurization options.
        @query: Query object
        '''
        if self.featurizer is None:
            self.init_featurizer()
        feature_vector = np.zeros(self.feature_len)
        for table in query.table_names:
            idx, _ = self.featurizer[table]
            feature_vector[idx] = 1.00

        for i, col in enumerate(query.pred_column_names):
            cmp_op = query.cmp_ops[i]
            cmp_op_idx, num_vals = self.featurizer[col]
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

            if cmp_op == "in":
                # bandaid...
                if isinstance(val[0], dict):
                    val = val[0]["literal"]
                val = set(val)
                # if len(val) <= num_pred_vals:
                    # print(query)
                    # print(len(val), num_pred_vals)
                    # pdb.set_trace()
                # assert len(val) <= num_pred_vals

                num_buckets = min(self.max_discrete_feauturizing_buckets,
                        col_info["num_values"])
                # if num_pred_vals != num_buckets:
                    # print(query)
                    # print(num_pred_vals, num_buckets)
                    # pdb.set_trace()
                ## FIXME! this assert fails sometimes
                # assert num_pred_vals == num_buckets
                # turn to 1 all the qualifying indexes in the 1-hot vector
                try:
                    for v in val:
                        pred_idx = deterministic_hash(v) % num_buckets
                        feature_vector[pred_idx_start+pred_idx] = 1.00
                except Exception as e:
                    print(e)
                    pdb.set_trace()
            elif cmp_op == "lte" or cmp_op == "lt":
                # do min-max stuff
                assert len(val) == 2
                ## don't do assert here.
                # try:
                    # assert num_pred_vals == 2
                # except:
                    # print(num_pred_vals)
                    # pdb.set_trace()
                lb = float(val[0])
                ub = float(val[1])
                assert lb <= ub
                min_val = float(col_info["min_value"])
                max_val = float(col_info["max_value"])

                lb_val = (lb - min_val) / (max_val - min_val)
                lb_val = max(lb_val, 0.00)
                lb_val = min(lb_val, 1.00)

                ub_val = (ub - min_val) / (max_val - min_val)
                ub_val = max(ub_val, 0.00)
                ub_val = min(ub_val, 1.00)
                assert lb_val <= ub_val
                feature_vector[pred_idx_start] = lb_val
                feature_vector[pred_idx_start+1] = ub_val
            elif cmp_op == "eq":
                # FIXME: !!
                continue
            else:
                assert False

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
            print("loading hashed key")
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

    def get_samples(self, query_template, num_samples=100, random_seed=1234):
        '''
        @query_template: string, with 'X' denoting a predicate value that needs to be filled in.
        TODO: try to do it more intelligently from the predicates joint prob
        distribution.

        @ret: a list of Query objects.
        '''
        ## not setting the seed because we do not want samples to be generated
        ## in a reproducible manner, since we are caching previously generated
        ## samples
        # random.seed(random_seed)
        start = time.time()
        pred_columns, pred_types, pred_vals = extract_predicates(query_template)
        for cmp_op in pred_types:
            self.cmp_ops.add(cmp_op)
        froms = extract_from_clause(query_template)
        for table in froms:
            self.tables.add(table)
        joins = extract_join_clause(query_template)

        for column in pred_columns:
            table = column[0:column.find(".")]
            min_query = MIN_TEMPLATE.format(TABLE = table,
                                            COL   = column)
            max_query = MAX_TEMPLATE.format(TABLE = table,
                                            COL   = column)
            count_query = UNIQUE_VALS_TEMPLATE.format(FROM_CLAUSE = table,
                                                      COL = column)
            if column not in self.column_stats:
                self.column_stats[column] = {}
                self.column_stats[column]["min_value"] = self.execute(min_query)[0][0]
                self.column_stats[column]["max_value"] = self.execute(max_query)[0][0]
                self.column_stats[column]["num_values"] = self.execute(count_query)[0][0]

        print("collected stats on all columns")
        # first, try and see if we have enough queries with the given template
        # in our cache.
        hashed_template = deterministic_hash(query_template)
        queries = []
        if hashed_template in self.sql_cache.archive:
            queries = self.sql_cache.archive[hashed_template]

        if len(queries) == num_samples:
            return queries
        elif len(queries) > num_samples:
            return queries[0:num_samples]
        else:
            # generate just the remaining queries
            num_samples -= len(queries)

        qg = QueryGenerator(query_template, self.user, self.db_host, self.port,
                self.pwd, self.db_name)
        all_query_strs = qg.gen_queries(num_samples)

        print("num queries: ", len(all_query_strs))

        # get total count
        from_clause = ','.join(froms)
        join_clause = ' AND '.join(joins)
        if len(join_clause) > 0:
            from_clause += " WHERE " + join_clause

        # total count, without any predicates being applied
        count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
        total_count = self.execute(count_query)[0][0]
        print("total count: ", total_count)
        print(all_query_strs[0])

        # TODO: clean up code, thread seems pointless.
        THREAD = False
        if THREAD:
            # We can use a with statement to ensure threads are cleaned up promptly
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Start the load operations and mark each future with its URL
                future_to_queries = {executor.submit(sql_to_query_object, cur_query,
                    self.user, self.db_host, self.port, self.pwd, self.db_name,
                    total_count, self.execution_cache_threshold, self.sql_cache):
                                            cur_query for cur_query in all_query_strs}
                for future in concurrent.futures.as_completed(future_to_queries):
                    cur_query = future_to_queries[future]
                    try:
                        query_obj = future.result()
                        queries.append(query_obj)
                    except Exception as exc:
                        print('%r generated an exception: %s' % (cur_query, exc))
                    else:
                        print('generated query object for: %s' % (query_obj))
        else:
            with Pool(processes=8) as pool:
                args = [(cur_query, self.user, self.db_host, self.port,
                    self.pwd, self.db_name, total_count,
                    self.execution_cache_threshold, self.sql_cache) for
                    cur_query in all_query_strs]
                all_query_objs = pool.starmap(sql_to_query_object, args)
            for q in all_query_objs:
                queries.append(q)

        print("generated {} samples in {} secs".format(len(queries),
            time.time()-start))

        self.sql_cache[hashed_template] = queries
        self.sql_cache.dump()
        return queries

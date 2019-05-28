import psycopg2 as pg
import pdb
import time
import random
from db_utils.utils import *
from utils.utils import *
from cardinality_estimation.query import Query
import klepto
import time

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
        self.execution_cache_threshold = 60

        # TODO: featurization for all elements of DB / or on per table basis
        # etc.?
        # TODO: collect other stats about each table in the DB etc.
        # get a random sample of values from each column so we can use it to
        # generate sensible queries
        self.con = pg.connect(user=self.user, host=self.db_host, port=self.port,
                password=self.pwd, database=self.db_name)
        self.sql_cache = klepto.archives.dir_archive(cache_dir)

        # stats on the used columns
        #   table_name : column_name : attribute : value
        #   e.g., stats["title"]["id"]["max_value"] = 1010112
        #         stats["title"]["id"]["type"] = int
        #         stats["title"]["id"]["num_values"] = x
        self.stats = {}

    def get_entropies(self):
        '''
        pairwise entropies among all columns of the db?
        '''
        pass

    def execute(self, sql):
        '''
        executes the given sql on the DB, and caches the results in a
        persistent store if it took longer than self.execution_cache_threshold.
        '''
        hashed_sql = deterministic_hash(sql)
        if hashed_sql in self.sql_cache.archive:
            # load it and return
            print("loaded {} from cache".format(sql))
            pdb.set_trace()
            return self.sql_cache.archive[hashed_sql]
        start = time.time()

        cursor = self.con.cursor()
        try:
            cursor.execute(sql)
        except Exception as e:
            print("query failed to execute: ", exp_new_query)
            print(e)
            pdb.set_trace()
            return None
        exp_output = cursor.fetchall()
        cursor.close()
        end = time.time()
        if (end - start > self.execution_cache_threshold):
            self.sql_cache[hashed_sql] = exp_output
            print("saved {} in cache".format(sql))
            self.sql_cache.dump()
        return exp_output

    def save_cache(self):
        print("saved cache to disk")
        self.sql_cache.dump()

    def get_samples(self, query_template, num_samples=100):
        '''
        @query_template: string, with 'X' denoting a predicate value that needs
        to be filled in.

        TODO: try to do it more intelligently from the predicates joint prob
        distribution.

        @ret: a list of Query objects.
        '''
        queries = []

        columns = extract_predicate_columns(query_template)
        froms = extract_from_clause(query_template)
        joins = extract_join_clause(query_template)

        # FIXME: not handling the cases with aliases
        for table in froms:
            if table not in self.stats:
                self.stats[table] = {}
            for column in columns:
                # is this column from this table?
                if table in column:
                    self.stats[table][column] = {}
                    self.stats[table][column]["name"] = column
                    # TODO: fill these in.
                    self.stats[table][column]["min_value"] = None
                    self.stats[table][column]["max_value"] = None
                    self.stats[table][column]["num_values"] = None
        columns = ','.join(columns)
        from_clause = ','.join(froms)
        join_clause = ' AND '.join(joins)
        if len(join_clause) > 0:
            from_clause += " WHERE " + join_clause

        group_by = GROUPBY_TEMPLATE.format(COLS = columns, FROM_CLAUSE=from_clause)
        group_by += " ORDER BY COUNT(*) DESC"
        # group_by += " LIMIT 10000"
        # use these to generate some values for the query templates
        groupby_output = self.execute(group_by)

        # total count, without any predicates being applied
        count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
        total_count = self.execute(count_query)[0][0]

        # TODO: more intelligent sampling so can select hard queries
        for i in range(num_samples):
            # FIXME: how to sample?
            # index = random.randint(0, len(groupby_output)-1)
            sample = groupby_output[i % len(groupby_output)]
            # construct a Query object
            pred_column_names = columns.split(",")
            vals = []
            cmp_ops = []
            new_query = query_template
            for j,col in enumerate(pred_column_names):
                vals.append(sample[j])
                # FIXME: generalize this to all ops
                col = col[col.find(".")+1:]
                cmp_ops.append("=")
                new_query = new_query.replace("'{}'".format(col.strip()),
                        "'{}'".format(sample[j]))
            true_val = sample[-1]
            exp_new_query = "EXPLAIN " + new_query
            exp_output = self.execute(exp_new_query)
            if exp_output is None:
                continue
            # FIXME: need to verify if this is correct in all cases
            pg_est = pg_est_from_explain(exp_output)
            query = Query(new_query, pred_column_names, vals, cmp_ops,
                    true_val, total_count, pg_est)
            queries.append(query)

        return queries

import psycopg2 as pg
import pdb
import time
from db_utils.utils import *
from cardinality_estimation.query import Query

class DBStats():

    def __init__(self, user, pwd, db_host, port, db_name):
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

        # TODO: featurization for all elements of DB / or on per table basis
        # etc.?
        # TODO: collect other stats about each table in the DB etc.

    def get_entropies(self):
        '''
        pairwise entropies among all columns of the db?
        '''
        pass

    def get_samples(self, query_template, num_samples=100):
        '''
        @query_template: string, with 'X' denoting a predicate value that needs
        to be filled in.

        TODO: try to do it more intelligently from the predicates joint prob
        distribution.

        @ret: a list of Query objects.
        '''
        queries = []
        # get a random sample of values from each column so we can use it to
        # generate sensible queries
        con = pg.connect(user=self.user, host=self.db_host, port=self.port,
                password=self.pwd)
        cursor = con.cursor()

        # can get column names from the cursor as well
        samples = []
        start = time.time()
        columns = extract_predicate_columns(query_template)
        from_clause = extract_from_clause(query_template)
        # new_query = query_template[0:query_template.find("WHERE")]
        group_by = GROUPBY_TEMPLATE.format(COLS = columns, FROM_CLAUSE=from_clause)
        # use these to generate some values for the query templates
        cursor.execute(group_by)
        groupby_output = cursor.fetchall()
        count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
        cursor.execute(count_query)
        total_count = cursor.fetchall()[0][0]

        # TODO: more intelligent sampling so can select hard queries
        for i in range(num_samples):
            sample = groupby_output[i % len(groupby_output)]
            # construct a Query object
            pred_column_names = columns.split(",")
            vals = []
            cmp_ops = []
            new_query = query_template
            for j,col in enumerate(pred_column_names):
                vals.append(sample[j])
                # FIXME: generalize this to all ops
                cmp_ops.append("=")
                new_query = new_query.replace("'{}'".format(col.strip()),
                        "'{}'".format(sample[j]))

            true_val = sample[-1]
            exp_new_query = "EXPLAIN " + new_query
            cursor.execute(exp_new_query)
            exp_output = cursor.fetchall()
            # FIXME: need to verify if this is correct in all cases
            pg_est = parse_explain(exp_output)
            query = Query(new_query, pred_column_names, vals, cmp_ops,
                    true_val, total_count, pg_est)
            queries.append(query)
        return queries

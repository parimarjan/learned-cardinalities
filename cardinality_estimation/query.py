import pdb
import psycopg2 as pg
from db_utils.utils import *

class Query():
    '''
    '''
    def __init__(self, query, pred_column_names, vals, cmp_ops, count,
            total_count, pg_count, pg_marginal_sels=None, marginal_sels=None):
        self.query = query

        # TODO: all this could be extracted just from the query str maybe?
        self.pred_column_names = pred_column_names
        self.vals = vals
        self.cmp_ops = cmp_ops
        self.true_count = count
        self.total_count = total_count
        self.true_sel = float(self.true_count) / self.total_count
        self.pg_count = pg_count

        # FIXME: handle this better
        _, _, self.table_names = extract_from_clause(query)
        self.joins = extract_join_clause(query)

        self.pg_marginal_sels = pg_marginal_sels
        self.marginal_sels = marginal_sels

    def eval_cost_model(self, env, card_samples):
        '''
        Evaluate a particular assignment to card_samples on the cost model.

        Steps:
            - Find the optimal policy based on the given cardinality samples
            - Test this policy AGAINST the optimal policy with true cardinality
              samples

        Note: Will have to implement this by either connecting to park's query
        optimizer OR plugging these into postgres' cardinality estimator etc.
        '''
        pass

    def __str__(self):
        # TODO: print more informative summary
        return self.query

    def gen_subqueries(self):
        '''
        Will generate Query objects for all the subqueries that do not
        include a cross-join.
        '''
        pass

    def get_features(self, featurization_scheme="onehot"):
        '''
        Some default featurization scheme, that each cardinality estimation
        algorithm may overwrite.
        '''
        pass

    def get_error(self, pred, error_type="qerror"):
        '''
        Types of errors:
            - abs error
            - qerror
            - relative error
            - something better?
        '''
        pass
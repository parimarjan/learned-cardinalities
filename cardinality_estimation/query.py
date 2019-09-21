import pdb
import psycopg2 as pg
from db_utils.utils import *

def get_alias(query, table):
    if hasattr(query, "aliases"):
        for a,v in query.aliases.items():
            if v == table:
                return a
    else:
        return None

def update_query_structure(query):
    # assert not hasattr(query, "froms")
    query.froms, query.aliases, query.table_names = extract_from_clause(query.query)

def get_cardinalities(query, alg):
    '''
    @query: Query object
    @alg: str, name of the algorithm. "true" needs to be handled separately for
    now.

    @ret: dict, keys are sorted table names in the format: " table1 table2 ...
    tableN ", and values are the cardinality estimates
    '''
    cards = {}

    for i, subq in enumerate(query.subqueries):
        if alg == "true":
            yhat = subq.true_sel
        else:
            yhat = subq.yhats[alg]
        est_count = subq.total_count * yhat
        # if alg == "true":
            # est_count = subq.true_count

        if not hasattr(subq, "froms"):
            subq.froms, subq.aliases, subq.table_names = extract_from_clause(subq.query)

        if len(subq.aliases) >= 1:
            tables = list(subq.aliases.keys())
        else:
            tables = subq.table_names
        tables.sort()
        table_key = " ".join(tables)
        cards[table_key] = int(est_count)

    return cards

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
        start = time.time()
        self.froms, self.aliases, self.table_names = extract_from_clause(query)
        self.joins = extract_join_clause(query)

        self.pg_marginal_sels = pg_marginal_sels
        self.marginal_sels = marginal_sels

    def __str__(self):
        # TODO: print more informative summary
        return self.query

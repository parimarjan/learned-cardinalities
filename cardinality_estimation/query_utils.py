
class QueryTemplate():
    '''
    '''
    def __init__(self, query_template, db_name="cards"):
        pass

    def gen_query_instances(self, num):
        '''
        Will generate @num queries based on the query template, and some
        measure of interestingness.
        @ret: bunch of Query objects.
        '''
        pass

class Query():
    '''
    '''
    def __init__(self, query):
        pass

    def gen_card_samples(self):
        '''
        Will generate CardinalitySamples for all the subqueries that do not
        include a cross-join.
        '''
        pass

    def eval_cost_model(self, env, card_samples):
        '''
        Evaluate a particular assignment to card_samples on the cost model.

        Steps:
            - Find the optimal policy based on the given cardinality samples
            - Test this policy AGAINST the optimal policy with true cardinality
              samples

        Note: This is the only place where we need to connect to the query_optimizer
        backend.
        '''
        pass

    def _gen_subqueries(self):
        pass

class CardinalitySample():
    '''
    the training / test algorithms should only need to deal with these guys.
    '''

    def __init__(self, query):
        self.query = query
        self.true_card = None
        self.pg_card = None

    def __str__(self):
        pass

    def get_marginals(self, true_marginals=True):
        pass

    def get_features(self, featurization_scheme="wordvec"):
        '''
        '''
        pass

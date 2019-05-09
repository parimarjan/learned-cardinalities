import pdb
import psycopg2 as pg

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

    def get_features(self, featurization_scheme="wordvec"):
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

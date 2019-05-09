import time
import numpy as np
import pdb
import math

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        pass
    def train(self, table_stats, training_samples):
        pass
    def test(self, test_samples):
        pass
    def size(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        pass
    def __str__(self):
        return self.__class__.__name__

class Postgres(CardinalityEstimationAlg):
    def test(self, test_samples):
        return np.array([s.pg_count for s in test_samples])

class Independent(CardinalityEstimationAlg):
    '''
    independent assumption on true marginal values.
    '''
    def test(self, test_samples):
        return np.array([np.prod(np.array(s.marginal_sels)) \
                for s in test_samples])

class NN1(CardinalityEstimationAlg):
    '''
    Default implementation of various neural network based methods.
    '''
    def __init__(self, *args, **kwargs):
        pass
    def train(self, table_stats, training_samples):
        pass
    def test(self, test_samples):
        pass
    def size(self):
        pass
    def __str__(self):
        # FIXME: add parameters of the neural network
        return self.__class__.__name__

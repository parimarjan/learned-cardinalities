import time
import numpy as np
import pdb
import math
from pomegranate import BayesianNetwork
from db_utils.utils import *
import matplotlib.pyplot as plt

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        pass
    def train(self, db, training_samples):
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
        return np.array([(s.pg_count / float(s.total_count)) for s in test_samples])

class Independent(CardinalityEstimationAlg):
    '''
    independent assumption on true marginal values.
    '''
    def test(self, test_samples):
        return np.array([np.prod(np.array(s.marginal_sels)) \
                for s in test_samples])

class BN(CardinalityEstimationAlg):
    '''
    Default implementation of various neural network based methods.
    '''
    def __init__(self, *args, **kwargs):
        if "alg" in kwargs:
            self.alg = kwargs["alg"]
        else:
            self.alg = "chow-liu"

    def train(self, db, training_samples):
        # generate the group-by over all the columns we care about.
        # FIXME: for now, just for one table.
        assert len(db.stats.keys()) == 1
        table = list(db.stats.keys())[0]
        columns = list(db.stats[table].keys())
        columns_str = ",".join(columns)
        group_by = GROUPBY_TEMPLATE.format(COLS = columns_str, FROM_CLAUSE=table)
        group_by += " ORDER BY COUNT(*) DESC"
        groupby_output = db.execute(group_by)
        samples = []
        weights = []
        for i, sample in enumerate(groupby_output):
            samples.append([])
            for j in range(len(sample)-1):
                samples[i].append(sample[j])
            weights.append(sample[j+1])
        samples = np.array(samples)
        weights = np.array(weights)
        print("going to train BayesianNetwork with ", self.alg)
        self.model = BayesianNetwork.from_samples(samples, weights=weights,
                state_names=columns, algorithm=self.alg, n_jobs=-1)
        self.model.plot()
        plt.savefig(self.alg + ".png")
    def test(self, test_samples):
        def _query_to_sample(sample):
            '''
            takes in a Query object, and converts it to the representation to
            be fed into the pomegranate bayesian net model
            '''
            model_sample = []
            for state in self.model.states:
                # find the right column entry in sample
                val = None
                for i, column in enumerate(sample.pred_column_names):
                    if column == state.name:
                        val = sample.vals[i]
                        break
                model_sample.append(val)
            return model_sample

        estimates = []
        for query in test_samples:
            # print(query)
            sample = _query_to_sample(query)
            # print(sample)
            # we shouldn't assume the order of column names in the trained model
            est_sel = self.get_selectivity(sample)
            # print(est_sel)
            # print(query.true_sel)
            # print(query.pg_count / query.total_count)
            # print("abs loss: ", query.count - (est_sel*query.total_count))
            # pdb.set_trace()
            estimates.append(est_sel)
        return np.array(estimates)

    def get_selectivity(self, sample):
        '''
        sample is in the form of samples for the trained bayesian net
        model: [val0, val1, ..., valN] for the N random values. values can be
        a known value, or None if it is unknown, and we will need to
        marginalize those out.
        '''
        def recursive_sel_cal(idx, pred_proba, cur_sample):
            '''
            Marginalizes over the unknown variables, and builds the selectivity
            estimate.
            Recursively builds cur_sample, and evaluates using the estimate for
            the joint probability distribution in the bayesian net self.model,
            for every possible value.

            @ret: calculated selecitivity
            '''
            cur_sel = 0.00
            if idx == len(pred_proba):
                # base case
                # print("final sample after recursion: ", cur_sample)
                return self.model.probability(cur_sample)

            pred_val = pred_proba[idx]
            # is it just an element, or a distribution of values?
            if "Discrete" in str(type(pred_val)):
                for val, prob in pred_val.items():
                    cur_sample[idx] = val
                    assert isinstance(prob, float)
                    cur_sel += prob * recursive_sel_cal(idx+1, pred_proba,
                            cur_sample)
            else:
                cur_sample[idx] = pred_val
                cur_sel += recursive_sel_cal(idx+1, pred_proba, cur_sample)
            return cur_sel

        pred_proba = self.model.predict_proba(sample, n_jobs=-1)
        cur_sample = [0]*len(pred_proba)
        final_sel = recursive_sel_cal(0, pred_proba, cur_sample)
        return final_sel

    def size(self):
        pass
    def __str__(self):
        # FIXME: add parameters of the learning model etc.
        return self.__class__.__name__ + "-" + self.alg

class NN1(CardinalityEstimationAlg):
    '''
    Default implementation of various neural network based methods.
    '''
    def __init__(self, *args, **kwargs):
        pass
    def train(self, db, training_samples):
        pass
    def test(self, test_samples):
        pass
    def size(self):
        pass
    def __str__(self):
        # FIXME: add parameters of the neural network
        return self.__class__.__name__

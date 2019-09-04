import numpy as np
import py.test
import random
from pgm import PGM
from pomegranate import BayesianNetwork
import itertools
import pandas as pd
import pdb
import random

SEEDS = [123]
NUM_COLUMNS = [5]
PERIOD_LEN = [100]
NUM_DATA_SAMPLES = 100000
NUM_TEST_SAMPLES = 10
NUM_RVS = 5
EPSILON = 0.01

# Generalized from:
#https://stackoverflow.com/questions/18683821/generating-random-correlated-x-and-y-points-using-numpy
def gen_gaussian_data_discrete(means, covs, num):
    vals = np.random.multivariate_normal(means, covs, num).T
    for i, v in enumerate(vals):
        vals[i] = [int(x) for x in v]
    return np.array(list(zip(*vals)))

def get_gaussian_data_params(seed, num_columns, period_len):
    '''
    @ret: means, covariance matrix. This should depend on the random random_seed.
    '''
    # random random_seed used for generating the correlations
    random.seed(seed)
    RANGES = []
    for i in range(num_columns):
        RANGES.append([i*period_len, i*period_len+period_len])

    # part 1: generate real data
    ranges = []
    means = []
    stds = []
    for r in RANGES:
        ranges.append(np.array(r))
    for r in ranges:
        means.append(r.mean())
        stds.append(r.std() / 3)
    covs = np.zeros((len(ranges), len(ranges)))
    for i in range(len(ranges)):
        for j in range(len(ranges)):
            if i == j:
                covs[i][j] = stds[i]**2
            elif i > j:
                continue
            else:
                # for the non-diagonal entries
                # uniformly choose the correlation between the elements
                corr = random.uniform(0.1, 1.00)
                covs[i][j] = corr*stds[i]*stds[j]
                covs[j][i] = corr*stds[i]*stds[j]
    return means, covs

def get_samples(data, num_test_samples):
    '''
    generates num_test_samples cases from df, and returns error of the passed
    in model.
    '''
    samples = []
    for i in range(num_test_samples):
        # generate a list of options from each column
        sample = []
        for col in range(data.shape[1]):
            cur_col = data[:,col]
            rvs = np.random.choice(cur_col, NUM_RVS)
            # get rid of doubles
            rvs = np.unique(rvs)
            sample.append(rvs)

        samples.append(sample)
    return samples

def get_true_sel(df, samples):
    true_sels = []

def test_simple():
    '''
    Just tests that things don't crash
    '''
    means, covs = get_gaussian_data_params(1234, 5, 10)
    data = gen_gaussian_data_discrete(means, covs, 10000)

    df = pd.DataFrame(data)
    column_list = list(range(num_columns))
    df = df.groupby(column_list).size().\
            sort_values(ascending=False).\
            reset_index(name='count')

    samples = df.values[:,0:num_columns]
    weights = np.array(df["count"])
    state_names = [str(s) for s in column_list]
    # create pgm model, and train it
    model = PGM(alg_name="chow-liu", backend="ourpgm", use_svd=False)
    model.train(samples, weights, state_names)
    test_samples = get_samples(data, NUM_TEST_SAMPLES)
    our_ests = []
    pom_ests = []
    for s in test_samples:
        our_ests.append(model.evaluate(s))

def test_discrete():
    '''
    '''
    cases = [SEEDS, NUM_COLUMNS, PERIOD_LEN]
    for (seed, num_columns, period_len) in itertools.product(*cases):
        print(seed, num_columns, period_len)
        means, covs = get_gaussian_data_params(seed, num_columns, period_len)
        data = gen_gaussian_data_discrete(means, covs, NUM_DATA_SAMPLES)

        df = pd.DataFrame(data)
        column_list = list(range(num_columns))
        df = df.groupby(column_list).size().\
                sort_values(ascending=False).\
                reset_index(name='count')

        samples = df.values[:,0:num_columns]
        weights = np.array(df["count"])
        state_names = [str(s) for s in column_list]
        # create pgm model, and train it
        model = PGM(alg_name="chow-liu", backend="ourpgm", use_svd=False)
        model.train(samples, weights, state_names)
        test_samples = get_samples(data, NUM_TEST_SAMPLES)
        our_ests = []
        pom_ests = []
        for s in test_samples:
            temp=model.evaluate(s)
            our_ests.append(temp)


        model = PGM(alg_name="chow-liu", backend="pomegranate", use_svd=False)
        model.train(samples, weights, state_names)
        for s in test_samples:
            temp=model.evaluate(s)
            pom_ests.append(temp)

        for i in range(0,len(pom_ests)):
            print("Query "+str(i)+": our pred -> "+str(our_ests[i])+" pom pred -> "+str(pom_ests[i]))
            


        our_ests = np.array(our_ests)
        pom_ests = np.array(pom_ests)
        diff = pom_ests - our_ests
        print("abs diff: ", np.sum(abs(diff)))
        # assert np.allclose(pom_ests, our_ests)
        our_avg = np.average(our_ests)
        pom_avg = np.average(pom_ests)
        if abs(our_avg - pom_avg) > EPSILON:
            assert False

test_discrete()

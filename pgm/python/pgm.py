import numpy as np
import pdb
import platform
from ctypes import *
import os
import copy
import pkg_resources
import csv

system = platform.system()
if system == 'Linux':
    lib_file = "libpgm.so"
else:
    lib_file = "libpgm.dylib"

pgm = CDLL(lib_file, mode=RTLD_GLOBAL)

class PGM():
    '''
    Serves as a wrapper class for the libpgm.so backend.
    '''
    def __init__(self):
        # index = random variable. For each random variable, map it to integers
        # 0...n-1 (where n is the size of that random variable)
        self.word2index = []
        self.state_names = []
        self.save_csv = True

    def train(self, samples, weights, state_names=None):
        assert state_names is not None
        self.state_names = state_names

        weights = np.array(weights, dtype=np.int32)
        for col in range(samples.shape[1]):
            self.word2index.append({})
            col_alphabets = np.unique(samples[:,col])
            for i, alph in enumerate(col_alphabets):
                self.word2index[col][alph] = i

        mapped_samples = np.zeros(samples.shape, dtype=np.int32)

        for i in range(mapped_samples.shape[0]):
            for j in range(mapped_samples.shape[1]):
                mapped_samples[i][j] = self.word2index[j][samples[i][j]]

        if self.save_csv:
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            np.savetxt("data.csv", mapped_samples, delimiter=",")
            np.savetxt("counts.csv", weights, delimiter=",")

        pgm.py_init(mapped_samples.ctypes.data_as(c_void_p),
                c_long(mapped_samples.shape[0]), c_long(mapped_samples.shape[1]),
                weights.ctypes.data_as(c_void_p), c_long(weights.shape[0]))
        pgm.py_train()

    def evaluate(self, all_points):
        '''
        TODO: describe this.
        @all_points: all valid points in the space of all points for ONE
        sample.
        '''
        assert len(all_points) == len(self.state_names)
        # convert to ints
        sample = []
        for col, col_points in enumerate(all_points):
            mapper = self.word2index[col]
            sample.append([])
            try:
                for p in col_points:
                    sample[col].append(mapper[p])
            except:
                # point hasn't been mapped before ...
                print("point has not been mapped before!!")
                print(col)
                print(p)
                pdb.set_trace()
                assert False

        assert len(sample) == len(self.state_names)

        if self.save_csv:
            sample_points = []
            for pts in sample:
                str_pts = [str(p) for p in pts]
                sample_points.append("-".join(str_pts))
            with open("samples.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(sample_points)

        entrylist = []
        lengths = []
        for sub_l in sample:
            entrylist.append((c_int*len(sub_l))(*sub_l))
            lengths.append(c_int(len(sub_l)))

        c_l = (POINTER(c_int) * len(entrylist))(*entrylist)
        c_lengths = (c_int * len(sample))(*lengths)
        pgm.py_eval.restype = c_double
        est = pgm.py_eval(c_l, c_lengths, len(sample), 0, c_double(1.0)) 
        if self.save_csv:
            with open("results.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([est])
        return est

# self.model = BayesianNetwork.from_samples(samples, weights=weights,
        # state_names=columns, algorithm=self.alg, n_jobs=-1)


import numpy as np
import pdb
import platform
from ctypes import *
import os
import copy
import pkg_resources
import csv
from pomegranate import BayesianNetwork
import math
import itertools
import time
import klepto
# from utils.utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

system = platform.system()
if system == 'Linux':
    lib_file = "libpgm.so"
else:
    lib_file = "libpgm.dylib"

pgm = CDLL(lib_file, mode=RTLD_GLOBAL)

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

class PGM():
    '''
    Serves as a wrapper class for the libpgm.so backend.
    '''
    def __init__(self, alg_name="chow-liu", backend="ourpgm", use_svd=True,
            num_singular_vals=10):
        # index = random variable. For each random variable, map it to integers
        # 0...n-1 (where n is the size of that random variable)
        self.word2index = []
        # string names for each of the random variables corresponding to
        # elements in word2index
        self.state_names = []
        self.save_csv = True
        self.backend = backend
        self.alg_name = alg_name
        self.use_svd = use_svd
        self.num_singular_vals = num_singular_vals
        print(self.backend, self.alg_name)

    def train(self, samples, weights, state_names=None):
        '''
        @samples: 2d array. Each row represents a unique point in the joint
        distribution, with each column representing a random variable.
        '''
        start = time.time()
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

        if self.backend == "ourpgm":
            pgm.py_init(mapped_samples.ctypes.data_as(c_void_p),
                    c_long(mapped_samples.shape[0]), c_long(mapped_samples.shape[1]),
                    weights.ctypes.data_as(c_void_p), c_long(weights.shape[0]),
                    self.use_svd, c_long(self.num_singular_vals))
            pgm.py_train()
        elif self.backend == "pomegranate":
            # TODO: cache the trained model, based on hash of mapped samples?
            # TODO: mapped samples should be extended to include all 0's.
            self.pom_model = BayesianNetwork.from_samples(mapped_samples, weights=weights,
                    state_names=self.state_names, algorithm="chow-liu", n_jobs=-1)
            print("pomegranate training done!")

            if self.alg_name == "greg":
                # compute all the appropriate SVD's
                self.edge_svds = {}

                # TODO: might want to store this globally
                state_to_idx = {}
                for i, s in enumerate(self.pom_model.states):
                    state_to_idx[s.name] = i

                # Expensive computation, so save it if possible
                misc_cache = klepto.archives.dir_archive("./misc_cache/edge_svds/")
                misc_cache.load()
                for edge in self.pom_model.edges:
                    node1 = state_to_idx[edge[0].name]
                    node2 = state_to_idx[edge[1].name]
                    edge_nodes = [node1, node2]
                    edge_nodes.sort()
                    edge_key = (edge_nodes[0], edge_nodes[1])

                    # FIXME: check
                    cond_dist = edge[1].distribution
                    assert "ConditionalProbabilityTable" in str(type(cond_dist))
                    marg1 = self.pom_model.marginal()[node1].values()
                    node1_vals = [k for k in self.pom_model.marginal()[node1].parameters[0].keys()]
                    dim1 = len(marg1)
                    marg2 = self.pom_model.marginal()[node2].values()
                    node2_vals = [k for k in self.pom_model.marginal()[node2].parameters[0].keys()]
                    dim2 = len(marg2)

                    svd_key = str(marg1) + str(marg2) + str(node1_vals) + str(node2_vals)
                    svd_key = deterministic_hash(svd_key)
                    if svd_key in misc_cache:
                        self.edge_svds[edge_key] = misc_cache[svd_key]
                        print("found edge key {} in cache".format(edge_key))
                        print(np.max(self.edge_svds[edge_key][1]))
                        continue
                    else:
                        print("did not find edge key {} in cache".format(edge_keedge_key))

                    joint_mat = np.zeros((dim1, dim2))
                    for i in range(dim1):
                        for j in range(dim2):
                            ind_term = marg1[i] * marg2[j]
                            # FIXME: assuming that these are state values
                            assert node1_vals[i] == i
                            assert node2_vals[j] == j
                            # FIXME: assuming marg1 is always the parent in the
                            # conditional dist
                            sample = [node1_vals[i], node2_vals[j]]
                            joint_term = cond_dist.probability(sample) * marg1[i]

                            joint_mat[i, j] = (joint_term - ind_term) / math.sqrt(ind_term)


                    # TODO: replace this by scipy.sparse svd's so can only
                    # compute for top-k values

                    uh, sv, vh = np.linalg.svd(joint_mat, full_matrices=False)
                    # print(np.max(sv))
                    assert np.max(sv) < 1.1
                    # pdb.set_trace()

                    # TODO: check if this computation is what we need
                    # compute the f and g vectors
                    for xi in range(dim1):
                        uh[xi,:] /= math.sqrt(marg1[xi])

                    for xj in range(dim2):
                        vh[:,xj] /= math.sqrt(marg2[xj])

                    assert edge_key not in self.edge_svds
                    self.edge_svds[edge_key] = (uh, sv, vh)

                    misc_cache[svd_key] = self.edge_svds[edge_key]
                misc_cache.dump()
                misc_cache.clear()
            elif self.alg_name == "chow-liu":
                # should not need to do anything here.
                print("trained chow-liu using pomegranate")
            else:
                assert False

        print("pgm model took {} seconds to train".format(time.time()-start))

    def evaluate(self, rv_values, weights=None, weights_method=1):
        '''
        @rv_values: Each element is a list. i-th element represents the i-th
        random variable, and the assignments for that random variable which
        we need to evaluate on. List can be empty if there are no constraints
        placed on that random variable. The assignments are in the original
        alphabet of the random variable, and need to be converted to the int
        representation specified in self.word2index.
        '''
        # print(self.backend, self.alg_name)
        assert len(rv_values) == len(self.state_names)

        if self.backend == "ourpgm":
            sample = self._get_sample(rv_values, True, weights)
            assert len(sample) == len(self.state_names)
            # TODO: update weights based on _get_sample
            if weights is not None:
                if weights_method == 1:
                    rv_weights = []
                    for cur_weights in weights:
                        cur_rv_weight = 0.00
                        for w in cur_weights:
                            cur_rv_weight += w
                        cur_rv_weight /= len(cur_weights)
                        rv_weights.append(cur_rv_weight)

                    combined_weight = np.product(np.array(rv_weights))
                    assert combined_weight <= 1.00
                    # print(rv_weights, combined_weight)
                    # pdb.set_trace()
                    est_val = self._eval_ourpgm(sample)
                    return est_val * combined_weight
                else:
                    assert False
            else:
                est_val = self._eval_ourpgm(sample)

            assert est_val <= 1.00
            return est_val

        elif self.backend == "pomegranate":
            assert self.pom_model is not None
            if self.alg_name == "chow-liu":
                sample = self._get_sample(rv_values, True)
                # print(sample)
                # pdb.set_trace()
                # TODO: add approximation option / flag
                all_points = []
                for p in itertools.product(*sample):
                    all_points.append(p)
                all_points = np.array(all_points)
                # we should be able to evaluate all these in parallel
                # print("going to call pomegrante's eval")
                est_vals = self.pom_model.probability(all_points)
                est_val = np.sum(est_vals)
                # assert est_val <= 1.00
                if est_val > 1.00:
                    print(est_val)
                    pdb.set_trace()
                return est_val

            elif self.alg_name == "greg":
                sample = self._get_sample(rv_values, False)
                assert len(sample) == len(self.state_names)
                # find the appropriate marginals, and nodes
                cond_nodes = []
                margs = []
                cur_node_vals = []
                for i,vals in enumerate(sample):
                    if len(vals) == 0:
                        continue
                    cond_nodes.append(i)
                    marg = self.pom_model.marginal()[i].values()
                    margs.append(marg)
                    cur_node_vals.append(vals)
                all_points = itertools.product(*cur_node_vals)

                # TODO: parallelize this
                est = 0.00
                for p in all_points:
                    assert len(p) == len(cond_nodes) == len(cur_node_vals)
                    est += self._eval_greg_pointwise(cond_nodes, margs, p)
                return est
            else:
                assert False

    def _get_sample(self, rv_values, fill_empty_rv, weights=None):
        # convert to word2index representation
        sample = []
        for col, col_points in enumerate(rv_values):
            mapper = self.word2index[col]
            sample.append([])
            # TODO: if col_points is empty, then append every possible value into it.
            if len(col_points) == 0 and fill_empty_rv:
                if weights is not None:
                    assert len(weights[col]) == 0
                for _, cur_val in mapper.items():
                    sample[col].append(cur_val)
                    if weights is not None:
                        weights[col].append(1.0)

            for p in col_points:
                # FIXME: dumb shiz
                try:
                    if p in mapper:
                        sample[col].append(mapper[p])
                    elif int(p) in mapper:
                        sample[col].append(mapper[int(p)])
                    elif str(p) in mapper:
                        sample[col].append(mapper[str(p)])
                    else:
                        # point hasn't been mapped before ...
                        print("point has not been mapped before!!")
                        print("col idx: ", col)
                        print("point: ", p)
                        pdb.set_trace()
                        assert False
                except Exception as e:
                    print(e)
                    pdb.set_trace()
        return sample

    def _eval_ourpgm(self, sample):
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
        est = pgm.py_eval(c_l, c_lengths, len(sample), 0, c_double(1.00))
        if self.save_csv:
            with open("results.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([est])
        return est

    def _eval_greg_pointwise(self, cond_nodes, margs, point):
        '''
        @cond_nodes: the nodes (0...n-1) which are being used in the current
        point.
        @margs: for the given nodes in cond_nodes, these are the marginal
        distributions
        @point: particular point, where point[i] is a value assignment to the
        random variable cond_nodes[i].
        '''
        assert len(cond_nodes) == len(margs) == len(point)
        # calculate the correction term, only for the edges
        correction_term = 0.00
        ind_prob = 1.00
        for i, p in enumerate(point):
            ind_prob = ind_prob*margs[i][p]

        for edge_nodes, svds in self.edge_svds.items():
            # if not (edge_nodes[0] in cond_nodes and edge_nodes[1] in cond_nodes):
                # continue
            # idxs into point / margs
            idx1 = None
            idx2 = None
            for nodei, node in enumerate(cond_nodes):
                if edge_nodes[0] == node:
                    idx1 = nodei
                elif edge_nodes[1] == node:
                    idx2 = nodei

            # skip if both the nodes in the edge nodes aren't in cond_nodes
            if idx1 is None or idx2 is None:
                continue

            uh = svds[0]
            sv = svds[1]
            vh = svds[2]

            try:
                xi = point[idx1]
                xj = point[idx2]
                assert type(xi) == int
                assert type(xj) == int
            except:
                print(edge_nodes)
                print(point)
                pdb.set_trace()

            fi = uh[xi,:]
            gi = vh[:,xj]
            # FIXME: should we be summing these for each xi,xj
            # combination
            correction_term += np.dot(np.multiply(fi, sv), gi)
            # print(correction_term)
            # pdb.set_trace()

        # print("ind prob: ", ind_prob)
        # print("correction term: ", correction_term)
        # pdb.set_trace()
        return ind_prob + ind_prob*correction_term


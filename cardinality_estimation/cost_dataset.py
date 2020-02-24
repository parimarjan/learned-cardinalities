import torch
from torch.utils import data
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
import numpy as np

class CostDataset(data.Dataset):
    def __init__(self, mapping, keys, estimates, costs, feat_type):
        '''
        '''
        self.mapping = mapping
        self.feat_type = feat_type
        self.X, self.Y = self.get_features(keys, estimates, costs)

    def get_features(self, keys, estimates, jlosses):
        '''
        '''
        queries = {}
        X = []
        Y = []
        for i, key in enumerate(keys):
            key = str(key)
            assert key in self.mapping
            if key in queries:
                qrep = queries[key]
            else:
                qfn = self.mapping[key]
                qrep = load_sql_rep(qfn)
                queries[key] = qrep

            ests = estimates[i]
            assert len(ests) == len(qrep["subset_graph"])

            cost = jlosses[i]

            if self.feat_type == "fcnn":
                x = np.zeros(len(ests)*2)
                node_keys = list(qrep["subset_graph"].nodes())
                node_keys.sort()
                for j, node in enumerate(node_keys):
                    info = qrep["subset_graph"].nodes()[node]["cardinality"]
                    idx = j*2
                    x[idx] = ests[j] / info["total"]
                    x[idx+1] = info["actual"] / info["total"]

                # X.append(to_variable(x).float())
                X.append(x)
                Y.append(cost)
            else:
                assert False

        Y = np.array(Y)
        Y = (Y + np.min(Y)) / (np.max(Y) - np.min(Y))
        return to_variable(X).float(), to_variable(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        '''
        '''
        return self.X[index], self.Y[index]

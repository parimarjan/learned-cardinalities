import torch
from torch.utils import data
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
import numpy as np

class CostDataset(data.Dataset):
    def __init__(self, mapping, keys, estimates, costs, cost_ratios,
            feat_type, cost_type="jcost_ratio",
            input_feat_type=1, input_norm_type=1, add_true=False):
        '''
        '''
        self.mapping = mapping
        self.feat_type = feat_type
        self.cost_type = cost_type
        self.add_true = add_true
        self.input_feat_type = input_feat_type
        self.input_norm_type = input_norm_type
        self.X, self.Y = self.get_features(keys, estimates, costs, cost_ratios)

    def get_features(self, keys, estimates, jlosses, jratios):
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
            # cost = jlosses[i]
            cost = jratios[i]
            assert jratios[i] != jlosses[i]

            if self.feat_type == "fcnn":
                if self.add_true:
                    if self.input_feat_type == 1:
                        x = np.zeros(len(ests)*2)
                        node_keys = list(qrep["subset_graph"].nodes())
                        node_keys.sort()
                        for j, node in enumerate(node_keys):
                            info = qrep["subset_graph"].nodes()[node]["cardinality"]
                            x[j] = ests[j] / info["total"]
                            x[len(ests)+j] = info["actual"] / info["total"]
                    else:
                        assert False
                    X.append(x)
                    Y.append(cost)
                else:
                    x = np.zeros(len(ests))
                    node_keys = list(qrep["subset_graph"].nodes())
                    node_keys.sort()
                    for j, node in enumerate(node_keys):
                        info = qrep["subset_graph"].nodes()[node]["cardinality"]
                        x[j] = ests[j] / info["total"]
                    X.append(x)
                    Y.append(cost)
        Y = np.array(Y)
        if self.input_norm_type == 1:
            pass
        elif self.input_norm_type == 2:
            X = np.log(X)
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
        elif self.input_norm_type == 3:
            X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        elif self.input_norm_type == 4:
            X = np.log(X)
            X = (X - np.mean(X)) / (np.std(X))
        elif self.input_norm_type == 5:
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0))

        # Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
        return to_variable(X).float(), to_variable(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        '''
        '''
        return self.X[index], self.Y[index]

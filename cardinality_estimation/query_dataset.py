import torch
from torch.utils import data
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *

class QueryDataset(data.Dataset):
    def __init__(self, qreps, db):
        '''
        @samples: [] sqlrep query dictionaries, which represent a query and all
        of its subqueries.
        The actual dataset consists of all the subqueries in all queries. Each
        index should uniquely map to a subquery.
        '''
        self.db = db
        self.qreps = qreps
        # TODO: we want to avoid this, and convert them on the fly. Just keep
        # some indexing information around.
        self.X, self.Y = self._get_feature_vectors(qreps)
        self.num_samples = len(self.X)

    def _get_feature_vectors(self, samples):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.
        '''
        start = time.time()
        X = []
        Y = []

        # FIXME: just need to do this once for each query
        for i, qrep in enumerate(samples):
            for nodes, info in qrep["subset_graph"].nodes().items():
                pg_sel = info["cardinality"]["expected"] / info["cardinality"]["total"]
                X.append(self.db.get_features(qrep["join_graph"].subgraph(nodes),
                    pg_sel))
                true_sel = info["cardinality"]["actual"] / info["cardinality"]["total"]
                Y.append(true_sel)

        X = to_variable(X, requires_grad=False).float()
        Y = to_variable(Y, requires_grad=False).float()
        return X,Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        '''
        return self.X[index], self.Y[index]

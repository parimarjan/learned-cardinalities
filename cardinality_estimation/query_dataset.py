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
        X = []
        Y = []

        for i, sample in enumerate(samples):
            # FIXME: do more efficient way without converting to Query
            query = convert_sql_rep_to_query_rep(sample)
            for subq in query.subqueries:
                features = self.db.get_features(subq)
                aliases = tuple(sorted(subq.aliases))
                assert aliases in sample["subset_graph"].nodes()
                X.append(features)
                Y.append(subq.true_sel)

        X = to_variable(X, requires_grad=False).float()
        Y = to_variable(Y, requires_grad=False).float()
        return X,Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        '''
        return self.X[index], self.Y[index]

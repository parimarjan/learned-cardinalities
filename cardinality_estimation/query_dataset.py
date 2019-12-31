import torch
from torch.utils import data
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *

class QueryDataset(data.Dataset):
    def __init__(self, qreps, db, featurization_type = "combined",
            heuristic_features=True):
        '''
        @samples: [] sqlrep query dictionaries, which represent a query and all
        of its subqueries.
        @featurization_type:
            - combined: generates a single vector combining all features
            - mscn: generates 3 vectors, as described in the mscn paper

        The actual dataset consists of all the subqueries in all queries. Each
        index should uniquely map to a subquery.
        '''
        self.db = db
        self.qreps = qreps
        self.heuristic_features = heuristic_features
        self.featurization_type = featurization_type
        # TODO: we want to avoid this, and convert them on the fly. Just keep
        # some indexing information around.
        self.X, self.Y = self._get_feature_vectors(qreps)
        self.num_samples = len(self.X)

    def _get_feature_vectors(self, samples):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.
        '''
        print("start _get_feature_vectors")
        start = time.time()
        X = []
        Y = []

        for i, qrep in enumerate(samples):
            node_data = qrep["join_graph"].nodes(data=True)
            # TODO: can also save these values and generate features when
            # needed, without wasting memory
            table_feat_dict = {}
            pred_feat_dict = {}
            edge_feat_dict = {}
            for node, info in node_data:
                table_features = self.db.get_table_features(info["real_name"])
                table_feat_dict[node] = table_features
                # TODO: pass in the cardinality as well.
                pg_est = None
                if self.heuristic_features:
                    node_key = tuple([node])
                    cards = qrep["subset_graph"].nodes()[node_key]["cardinality"]
                    pg_est = float(cards["expected"]) / cards["total"]
                if len(info["pred_cols"]) == 0:
                    pred_features = np.zeros(self.db.pred_features_len)
                else:
                    pred_features = self.db.get_pred_features(info["pred_cols"][0],
                            info["pred_vals"][0], info["pred_types"][0], pg_est)

                pred_feat_dict[node] = pred_features

            # edge_data = qrep["join_graph"].edges(data=True)
            # for edge in edge_data:
                # info = edge[2]
                # edge_features = self.db.get_join_features(info["join_condition"])
                # edge_key = info["join_condition"]
                # edge_feat_dict[edge_key] = edge_features

            for nodes, info in qrep["subset_graph"].nodes().items():
                pg_sel = float(info["cardinality"]["expected"]) / info["cardinality"]["total"]
                true_sel = float(info["cardinality"]["actual"]) / info["cardinality"]["total"]
                pred_features = np.zeros(self.db.pred_features_len)
                table_features = np.zeros(len(self.db.tables))
                join_features = np.zeros(len(self.db.joins))
                for node in nodes:
                    # no overlap between these arrays
                    pred_features += pred_feat_dict[node]
                    table_features += table_feat_dict[node]

                # sg = qrep["join_graph"].subgraph(nodes)
                # for edge in sg.edges(data=True):
                    # edge_key = edge[2]["join_condition"]
                    # join_features += edge_feat_dict[edge_key]

                if self.heuristic_features:
                    assert pred_features[-1] == 0.00
                    pred_features[-1] = pg_sel

                # now, store features
                X.append(np.concatenate((table_features, pred_features,
                    join_features)))
                Y.append(true_sel)

        print("get features took: ", time.time() - start)
        pdb.set_trace()

        # FIXME: just need to do this once for each query
        # for i, qrep in enumerate(samples):
            # for nodes, info in qrep["subset_graph"].nodes().items():
                # pg_sel = info["cardinality"]["expected"] / info["cardinality"]["total"]
                # X.append(self.db.get_features(qrep["join_graph"].subgraph(nodes),
                    # pg_sel))
                # true_sel = float(info["cardinality"]["actual"]) / info["cardinality"]["total"]
                # Y.append(true_sel)

        X = to_variable(X, requires_grad=False).float()
        Y = to_variable(Y, requires_grad=False).float()
        return X,Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        '''
        return self.X[index], self.Y[index]

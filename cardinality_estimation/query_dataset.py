import torch
from torch.utils import data
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
from collections import defaultdict

class QueryDataset(data.Dataset):
    def __init__(self, qreps, db, featurization_type,
            heuristic_features):
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
        self.X, self.Y, self.info = self._get_feature_vectors(qreps)
        self.num_samples = len(self.Y)

    def _get_feature_vectors(self, samples):
        '''
        @samples: sql_rep format representation for query and all its
        subqueries.
        '''
        start = time.time()
        if self.featurization_type == "combined":
            X = []
        else:
            X = defaultdict(list)

        Y = []
        sample_info = []

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

            edge_data = qrep["join_graph"].edges(data=True)
            for edge in edge_data:
                info = edge[2]
                edge_features = self.db.get_join_features(info["join_condition"])
                edge_key = (edge[0], edge[1])
                edge_feat_dict[edge_key] = edge_features

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
                if self.heuristic_features:
                    assert pred_features[-1] == 0.00
                    pred_features[-1] = pg_sel

                # TODO: optimize...
                for node1 in nodes:
                    for node2 in nodes:
                        if (node1, node2) in edge_feat_dict:
                            join_features += edge_feat_dict[(node1, node2)]

                # now, store features
                if self.featurization_type == "combined":
                    X.append(np.concatenate((table_features, join_features,
                        pred_features)))
                else:
                    X["table"].append(table_features)
                    X["join"].append(join_features)
                    X["pred"].append(pred_features)
                    # X.append((table_features, join_features, pred_features))
                Y.append(true_sel)
                cur_info = {}
                cur_info["num_tables"] = len(nodes)
                cur_info["total"] = info["cardinality"]["total"]
                sample_info.append(cur_info)

        print("get features took: ", time.time() - start)

        if self.featurization_type == "combined":
            X = to_variable(X, requires_grad=False).float()
        else:
            for k,v in X.items():
                X[k] = to_variable(v, requires_grad=False).float()

        Y = to_variable(Y, requires_grad=False).float()
        return X,Y,sample_info

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        '''
        if self.featurization_type == "combined":
            return self.X[index], self.Y[index], self.info[index]
        else:
            return (self.X["table"][index], self.X["pred"][index],
                    self.X["join"][index], self.Y[index], self.info[index])

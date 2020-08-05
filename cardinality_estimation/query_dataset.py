import torch
from torch.utils import data
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
from collections import defaultdict
import numpy as np
from cardinality_estimation.algs import get_wj_times_dict

class QueryDataset(data.Dataset):
    def __init__(self, samples, db, featurization_type,
            heuristic_features, preload_features,
            normalization_type, load_query_together,
            flow_features, table_features, join_features, pred_features,
            min_val=None, max_val=None, card_key="actual",
            group=None):
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
        self.samples = samples
        self.heuristic_features = heuristic_features
        self.featurization_type = featurization_type
        self.preload_features = preload_features
        self.normalization_type = normalization_type
        self.min_val = min_val
        self.max_val = max_val
        self.card_key = card_key
        self.group = None
        self.flow_features = flow_features
        self.table_features = table_features
        self.join_features = join_features
        self.pred_features = pred_features

        # -1 to ignore SOURCE_NODE
        total_nodes = [len(s["subset_graph"].nodes())-1 for s in samples]
        total_expected_samples = sum(total_nodes)

        if self.card_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
            self.wj_times = get_wj_times_dict(self.card_key)
        else:
            self.wj_times = None

        if self.normalization_type == "mscn":
            assert min_val is not None
            assert max_val is not None

        self.load_query_together = load_query_together

        # TODO: we want to avoid this, and convert them on the fly. Just keep
        # some indexing information around.
        if self.preload_features:
            self.X, self.Y, self.info = self._get_feature_vectors(samples)
            assert len(self.Y) == total_expected_samples
            if load_query_together:
                self.start_idxs, self.idx_lens = self._update_idxs(samples)
                self.num_samples = len(samples)
            else:
                self.num_samples = len(self.Y)
        else:
            self.subq_to_query_idx, self.qstart_idxs = self._update_idxs2(samples)
            self.num_samples = len(self.subq_to_query_idx)

    def _update_idxs(self, samples):
        qidx = 0
        idx_starts = []
        idx_lens = []
        for i, qrep in enumerate(samples):
            # TODO: can also save these values and generate features when
            # needed, without wasting memory
            idx_starts.append(qidx)
            nodes = list(qrep["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            idx_lens.append(len(nodes))
            qidx += len(nodes)
        return idx_starts, idx_lens

    def _update_idxs2(self, samples):
        qidx = 0
        idx_map = {}
        idx_starts = []
        for i, qrep in enumerate(samples):
            # TODO: can also save these values and generate features when
            # needed, without wasting memory
            idx_starts.append(qidx)
            for sidx, _ in enumerate(qrep["subset_graph"].nodes(data=True)):
                idx_map[qidx + sidx] = i

            qidx += len(qrep["subset_graph"].nodes())
        return idx_map, idx_starts

    def normalize_val(self, val, total):
        if self.normalization_type == "mscn":
            return (np.log(val) - self.min_val) / (self.max_val - self.min_val)
        else:
            return float(val) / total

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
        qidx = 0
        for i, qrep in enumerate(samples):
            node_data = qrep["join_graph"].nodes(data=True)
            # TODO: can also save these values and generate features when
            # needed, without wasting memory
            table_feat_dict = {}
            pred_feat_dict = {}
            edge_feat_dict = {}

            for node, info in node_data:
                if SOURCE_NODE in node_data:
                    continue

                cards = qrep["subset_graph"].nodes()[(node,)]
                if "sample_bitmap" in cards:
                    bitmap = cards["sample_bitmap"]
                else:
                    bitmap = None
                table_features = self.db.get_table_features(info["real_name"],
                        bitmap_dict=bitmap)

                # print(bitmap)
                # print("len of table features: ", len(table_features))
                # pdb.set_trace()

                table_feat_dict[node] = table_features
                # TODO: pass in the cardinality as well.
                heuristic_est = None
                if self.heuristic_features:
                    node_key = tuple([node])
                    cards = qrep["subset_graph"].nodes()[node_key]["cardinality"]
                    # heuristic_est = float(cards["expected"]) / cards["total"]
                    heuristic_est = self.normalize_val(cards["expected"],
                            cards["total"])

                if len(info["pred_cols"]) == 0:
                    pred_features = np.zeros(self.db.pred_features_len)
                else:
                    pred_features = self.db.get_pred_features(info["pred_cols"][0],
                            info["pred_vals"][0], info["pred_types"][0],
                            heuristic_est)
                pred_feat_dict[node] = pred_features

            edge_data = qrep["join_graph"].edges(data=True)
            for edge in edge_data:
                info = edge[2]
                edge_features = self.db.get_join_features(info["join_condition"])
                edge_key = (edge[0], edge[1])
                edge_feat_dict[edge_key] = edge_features

            # now, we will generate the actual feature vectors over all the
            # subqueries
            node_names = list(qrep["subset_graph"].nodes())
            node_names.remove(SOURCE_NODE)
            node_names.sort()
            for node_idx, nodes in enumerate(node_names):
                if self.group is not None:
                    assert False
                    if len(nodes) not in self.group:
                        continue

                info = qrep["subset_graph"].nodes()[nodes]
                pg_est = info["cardinality"]["expected"]
                if self.wj_times is not None:
                    ck = "wanderjoin-" + str(self.wj_times[qrep["template_name"]])
                    true_val = info["cardinality"][ck]
                    if true_val == 0 or true_val == 1:
                        true_val = info["cardinality"]["expected"]
                else:
                    ck = self.card_key
                    true_val = info["cardinality"][ck]

                total = info["cardinality"]["total"]

                pred_features = np.zeros(self.db.pred_features_len)
                table_features = np.zeros(self.db.table_features_len)
                join_features = np.zeros(len(self.db.joins))

                # these are base tables within a join, or node in the subset
                # graph
                for node in nodes:
                    # no overlap between these arrays
                    pred_features += pred_feat_dict[node]
                    table_features += table_feat_dict[node]
                if self.heuristic_features:
                    assert pred_features[-1] == 0.00
                    pred_features[-1] = self.normalize_val(pg_est, total)

                # TODO: optimize...
                for node1 in nodes:
                    for node2 in nodes:
                        if (node1, node2) in edge_feat_dict:
                            join_features += edge_feat_dict[(node1, node2)]

                if self.flow_features:
                    # use db to generate feature vec using nodes + qrep
                    # info2 = qrep["subset_graph"].nodes()[nodes]
                    if "pred_types" in info:
                        cmp_op = info["pred_types"][0]
                    else:
                        cmp_op = None
                    flow_features = self.db.get_flow_features(nodes,
                            qrep["subset_graph"], qrep["template_name"],
                            qrep["join_graph"], cmp_op)
                    # heuristic estimate for the cardinality of this node
                    flow_features[-1] = pred_features[-1]
                else:
                    flow_features = []

                # now, store features
                if self.featurization_type == "combined":
                    comb_feats = []
                    if self.table_features:
                        comb_feats.append(table_features)
                    if self.join_features:
                        comb_feats.append(join_features)
                    if self.pred_features:
                        comb_feats.append(pred_features)
                    if self.flow_features:
                        comb_feats.append(flow_features)
                    assert len(comb_feats) > 0
                    X.append(np.concatenate(comb_feats))
                else:
                    X["table"].append(table_features)
                    X["join"].append(join_features)
                    X["pred"].append(pred_features)
                    X["flow"].append(flow_features)

                Y.append(self.normalize_val(true_val, total))
                cur_info = {}
                cur_info["num_tables"] = len(nodes)
                cur_info["dataset_idx"] = qidx + node_idx
                cur_info["query_idx"] = i

                # FIXME:
                # cur_info["total"] = total
                cur_info["total"] = 0.00
                sample_info.append(cur_info)

            qidx += len(node_names)

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
        if self.preload_features:
            if self.load_query_together:
                start_idx = self.start_idxs[index]
                end_idx = start_idx + self.idx_lens[index]
                if self.featurization_type == "combined":
                    return self.X[start_idx:end_idx], self.Y[start_idx:end_idx], \
                            self.info[start_idx:end_idx]
                else:
                    return (self.X["table"][start_idx:end_idx],
                            self.X["pred"][start_idx:end_idx],
                            self.X["join"][start_idx:end_idx],
                            self.X["flow"][start_idx:end_idx],
                            self.Y[start_idx:end_idx],
                            self.info[start_idx:end_idx])
            else:
                # usual path
                if self.featurization_type == "combined":
                    return self.X[index], self.Y[index], self.info[index]
                else:
                    return (self.X["table"][index], self.X["pred"][index],
                            self.X["join"][index], self.X["flow"][index],
                            self.Y[index], self.info[index])
        else:
            assert False
            qidx = self.subq_to_query_idx[index]
            idx_start = self.qstart_idxs[qidx]
            subq_idx = index - idx_start
            qrep = self.samples[qidx]
            return self._get_feature_vector(qrep,
                    list(qrep["subset_graph"].nodes())[subq_idx])


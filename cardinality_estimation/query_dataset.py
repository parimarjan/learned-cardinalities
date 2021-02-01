import torch
from torch.utils import data
from utils.utils import *
from db_utils.utils import *
from db_utils.query_storage import *
from collections import defaultdict
import numpy as np
from cardinality_estimation.algs import get_wj_times_dict
import shutil
import psutil
import gc

class QueryDataset(data.Dataset):
    def __init__(self, samples, db, featurization_type,
            heuristic_features, preload_features,
            normalization_type, load_query_together,
            flow_features, table_features, join_features, pred_features,
            db_year = "",
            min_val=None, max_val=None, card_key="actual",
            use_set_padding=True,
            group=None, max_sequence_len=None, log_base=10,
            exp_name=""):
        '''
        @samples: [] sqlrep query dictionaries, which represent a query and all
        of its subqueries.
        @featurization_type:
            - combined: generates a single vector combining all features
            - mscn: as in the mscn paper.

        The actual dataset consists of all the subqueries in all queries. Each
        index should uniquely map to a subquery.
        '''
        if db.db_name == "so":
            global SOURCE_NODE
            SOURCE_NODE = tuple(["SOURCE"])

        self.db = db
        # self.samples = samples
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
        self.max_sequence_len = max_sequence_len
        self.log_base = log_base
        self.exp_name = exp_name
        self.use_set_padding = use_set_padding
        self.db_year = db_year
        self.cinfo_key = str(self.db_year) + "cardinality"
        print("FIXME: handle if cinfo key not in info")

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

        if self.featurization_type == "set":
            self.max_tables = self.db.max_tables
            self.max_joins = self.db.max_joins
            self.max_preds = self.db.max_preds

            # update stats about max-predicates, max-tables etc.
            # self.max_tables = 0
            # self.max_joins = 0
            # self.max_preds = 0

            # for i, qrep in enumerate(samples):
                # node_data = qrep["join_graph"].nodes(data=True)

                # num_tables = len(node_data)
                # if num_tables > self.max_tables:
                    # self.max_tables = num_tables

                # num_preds = 0
                # for node, info in node_data:
                    # num_preds += len(info["pred_cols"])

                # if num_preds > self.max_preds:
                    # self.max_preds = num_preds

                # num_joins = len(qrep["join_graph"].edges())
                # if num_joins > self.max_joins:
                    # self.max_joins = num_joins

            # # TODO: estimated upper bound, need to figure out a better way to calculate this
            # # self.max_joins = self.max_tables + 12
            # print(self.max_tables, self.max_joins, self.max_preds)

        if self.preload_features == 1:
            # 3, means we will try to load the padded sets and masks from
            # memory
            if self.use_set_padding > 2:
                # otherwise, no point in doing this
                assert self.use_set_padding == 3

                # self.feature_dir = "/flashrd/pari/saved_features/"
                # self.feature_dir = "/flashrd/pari/saved_features2/"
                self.feature_dir = "./saved_features/"
                self.feature_dir += self.db.db_key + "/"
                print("saved features directory: ")
                print(self.feature_dir)
                # will store each query feature dicts in there
                make_dir(self.feature_dir)

            self.X, self.Y, self.info = self._get_feature_vectors(samples)
            assert len(self.Y) == total_expected_samples
            if load_query_together:
                self.start_idxs, self.idx_lens = self._update_idxs(samples)
                self.num_samples = len(samples)
            else:
                self.num_samples = len(self.Y)

        elif self.preload_features == 2:
            # creating unique directory for just this experiment to store
            # feature vectors on disk / and load them for each iteration. will
            # delete them while cleaning up experiments
            # need to do it per experiment because we are constantly changing
            # things with which features are used, and it depends on a lot of
            # flags
            self.feature_dir = "./saved_features/"
            # self.feature_dir += self.exp_name + "/"
            # self.feature_dir += str(time.time())
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            self.feature_dir += start_time + "/"

            make_dir(self.feature_dir)
            # self.query_fn_hashes = [deterministic_hash(q["name"]) for q in samples]

            _, self.Y, self.info = self._get_feature_vectors(samples,
                    feature_dir=self.feature_dir)

            assert len(self.Y) == total_expected_samples
            if load_query_together:
                self.start_idxs, self.idx_lens = self._update_idxs(samples)
                self.num_samples = len(samples)
            else:
                self.num_samples = len(self.Y)

        elif self.preload_features == 3:
            # map from each query to qrep file name
            assert self.load_query_together
            self.start_idxs, self.idx_lens = self._update_idxs(samples)
            self.query_fns = [q["name"] for q in samples]
            self.num_samples = len(samples)
            assert os.path.exists(self.query_fns[0])

        elif self.preload_features == 4:
            # map from each query to qrep file name
            assert self.load_query_together
            self.start_idxs, self.idx_lens = self._update_idxs(samples)
            self.samples = samples
            # self.query_fns = [q["name"] for q in samples]
            # assert os.path.exists(self.query_fns[0])
            self.num_samples = len(samples)
        else:
            self.subq_to_query_idx, self.qstart_idxs = self._update_idxs2(samples)
            self.num_samples = len(self.subq_to_query_idx)

    def clean(self):
        # TODO: add for others too
        if self.preload_features == 1:
            if self.use_set_padding == 3:
                if isinstance(self.X, dict):
                    for k,v in self.X.items():
                        del(v)
                gc.collect()
                del(self.info[:])

            else:
                if isinstance(self.X, dict):
                    for k,v in self.X.items():
                        del(v)


            del(self.X)
            del(self.Y)
            del(self)

        elif self.preload_features == 2:
            assert os.path.exists(self.feature_dir)
            # os.path.remove(self.feature_dir)
            shutil.rmtree(self.feature_dir)

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
            if self.log_base == 2:
                return (np.log2(val) - self.min_val) / (self.max_val - self.min_val)
            else:
                return (np.log(float(val)) - self.min_val) / (self.max_val - self.min_val)
        else:
            return float(val) / total

    def _pad_sets(self, all_table_features, all_pred_features,
            all_join_features, tensors):

        tf = []
        pf = []
        jf = []
        tm = []
        pm = []
        jm = []

        for i in range(len(all_table_features)):
            table_features = all_table_features[i]
            pred_features = all_pred_features[i]
            join_features = all_join_features[i]

            pred_features = np.vstack(pred_features)
            num_pad = self.max_preds - pred_features.shape[0]
            if num_pad < 0:
                print(num_pad)
                pdb.set_trace()

            predicate_mask = np.ones_like(pred_features).mean(1, keepdims=True)
            pred_features = np.pad(pred_features, ((0, num_pad), (0, 0)), 'constant')
            predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
            pred_features = np.expand_dims(pred_features, 0)
            predicate_mask = np.expand_dims(predicate_mask, 0)

            # do same for table, and joins
            table_features = np.vstack(table_features)
            num_pad = self.max_tables - table_features.shape[0]

            if num_pad < 0:
                print(num_pad)
                pdb.set_trace()
            table_mask = np.ones_like(table_features).mean(1, keepdims=True)
            table_features = np.pad(table_features, ((0, num_pad), (0, 0)), 'constant')
            table_mask = np.pad(table_mask, ((0, num_pad), (0, 0)), 'constant')
            table_features = np.expand_dims(table_features, 0)
            table_mask = np.expand_dims(table_mask, 0)

            join_features = np.vstack(join_features)
            num_pad = self.max_joins - join_features.shape[0]
            # assert num_pad >= 0
            if num_pad < 0:
                print(num_pad)
                pdb.set_trace()
            join_mask = np.ones_like(join_features).mean(1, keepdims=True)
            join_features = np.pad(join_features, ((0, num_pad), (0, 0)), 'constant')
            join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
            join_features = np.expand_dims(join_features, 0)
            join_mask = np.expand_dims(join_mask, 0)

            tf.append(table_features)
            pf.append(pred_features)
            jf.append(join_features)
            tm.append(table_mask)
            pm.append(predicate_mask)
            jm.append(join_mask)

        if tensors:
            tf = to_variable(tf,
                    requires_grad=False).float().squeeze()
            pf = to_variable(pf,
                    requires_grad=False).float().squeeze()
            jf = to_variable(jf,
                    requires_grad=False).float().squeeze()
            extra_dim = len(jf.shape)-1
            tm = to_variable(tm,
                    requires_grad=False).float().squeeze().unsqueeze(extra_dim)
            pm = to_variable(pm,
                    requires_grad=False).float().squeeze().unsqueeze(extra_dim)
            jm = to_variable(jm,
                    requires_grad=False).float().squeeze().unsqueeze(extra_dim)

            # print(tf.shape, pf.shape, jf.shape)
            # print(tm.shape, pm.shape, jm.shape)
            # pdb.set_trace()

        return tf, pf, jf, tm, pm, jm

    def _get_query_features_set(self, qrep, dataset_qidx,
            query_idx, use_saved=True):

        if self.use_set_padding == 3:
            qkey = str(deterministic_hash(qrep["sql"]))
            qpathx = self.feature_dir + qkey + "x.pkl"
            qpathy = self.feature_dir + qkey + "y.pkl"
            qpathi = self.feature_dir + qkey + "i.pkl"

            if os.path.exists(qpathx) and use_saved:
                # load and all
                # X = load_object(qpathx)
                X = load_object_gzip(qpathx)
                Y = []
                sample_info = []

                node_names = list(qrep["subset_graph"].nodes())
                if SOURCE_NODE in node_names:
                    node_names.remove(SOURCE_NODE)
                node_names.sort()

                for node_idx, nodes in enumerate(node_names):
                    info = qrep["subset_graph"].nodes()[nodes]
                    if self.wj_times is not None:
                        ck = "wanderjoin-" + str(self.wj_times[qrep["template_name"]])
                        true_val = info["cardinality"][ck]
                        if true_val == 0 or true_val == 1:
                            # true_val = info["cardinality"]["expected"]
                            true_val = 1.0
                    else:
                        ck = self.card_key
                        true_val = info["cardinality"][ck]

                    if "total" in info["cardinality"]:
                        total = info["cardinality"]["total"]
                    else:
                        total = None

                    Y.append(self.normalize_val(true_val, total))

                    ## temporary
                    cur_info = {}
                    # cur_info["num_tables"] = len(nodes)
                    cur_info["dataset_idx"] = dataset_qidx + node_idx
                    cur_info["query_idx"] = query_idx
                    # cur_info["total"] = 0.00
                    # cur_info = []
                    sample_info.append(cur_info)

                return X, Y, sample_info

        X = defaultdict(list)
        Y = []
        sample_info = []

        node_data = qrep["join_graph"].nodes(data=True)

        table_feat_dict = {}
        pred_feat_dict = {}
        edge_feat_dict = {}

        # iteration order doesn't matter
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

            table_feat_dict[node] = table_features
            # TODO: pass in the cardinality as well.
            heuristic_est = None
            if self.heuristic_features:
                node_key = tuple([node])
                cards = qrep["subset_graph"].nodes()[node_key]["cardinality"]
                if "total" in cards:
                    total = cards["total"]
                else:
                    total = None
                heuristic_est = self.normalize_val(cards["expected"],
                        total)

            if len(info["pred_cols"]) == 0:
                # pred_features = np.zeros(self.db.max_pred_len)
                # skip featurization of zeros
                continue
            else:
                pred_features = self.db.get_pred_features(info["pred_cols"][0],
                        info["pred_vals"][0], info["pred_types"][0],
                        self.db_year,
                        pred_est=heuristic_est)

            pred_feat_dict[node] = pred_features

        edge_data = qrep["join_graph"].edges(data=True)
        for edge in edge_data:
            info = edge[2]
            edge_features = self.db.get_join_features(info["join_condition"])
            edge_key = (edge[0], edge[1])
            edge_feat_dict[edge_key] = edge_features

        # now, we will generate the actual feature vectors over all the
        # subqueries. Order matters - dataset idx will be specified based on
        # order.
        node_names = list(qrep["subset_graph"].nodes())
        if SOURCE_NODE in node_names:
            node_names.remove(SOURCE_NODE)
        node_names.sort()

        for node_idx, nodes in enumerate(node_names):
            if self.group is not None:
                assert False
                if len(nodes) not in self.group:
                    continue

            info = qrep["subset_graph"].nodes()[nodes]
            pg_est = info["cardinality"]["expected"]
            # pfeats[-2] = self.normalize_val(pg_est, total)
            total = 0.0
            sample_heuristic_est = self.normalize_val(pg_est, total)

            if self.wj_times is not None:
                ck = "wanderjoin-" + str(self.wj_times[qrep["template_name"]])
                true_val = info["cardinality"][ck]
                if true_val == 0 or true_val == 1:
                    # true_val = info["cardinality"]["expected"]
                    true_val = 1.0
            else:
                ck = self.card_key
                true_val = info["cardinality"][ck]

            if "total" in info["cardinality"]:
                total = info["cardinality"]["total"]
            else:
                total = None

            table_features = []
            pred_features = []
            join_features = []

            # these are base tables within a join (or node) in the subset
            # graph
            for node in nodes:
                table_features.append(table_feat_dict[node])
                if node not in pred_feat_dict:
                    continue
                pfeats = copy.deepcopy(pred_feat_dict[node])
                pred_features.append(pfeats)

            for node1 in nodes:
                for node2 in nodes:
                    if (node1, node2) in edge_feat_dict:
                        join_features.append(edge_feat_dict[(node1, node2)])

            if len(join_features) == 0:
                empty_feats = np.zeros(len(self.db.join_featurizer))
                join_features.append(empty_feats)

            if len(pred_features) == 0:
                empty_feats = np.zeros(self.db.max_pred_len)
                pred_features.append(empty_feats)

            if self.use_set_padding in [1,3]:
                # TODO: _pad_sets should also work, need to check lists / arrays

                pred_features = np.vstack(pred_features)
                num_pad = self.max_preds - pred_features.shape[0]
                if num_pad < 0:
                    print(num_pad)
                    pdb.set_trace()

                predicate_mask = np.ones_like(pred_features).mean(1, keepdims=True)
                pred_features = np.pad(pred_features, ((0, num_pad), (0, 0)), 'constant')
                predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
                pred_features = np.expand_dims(pred_features, 0)
                predicate_mask = np.expand_dims(predicate_mask, 0)

                # do same for table, and joins
                table_features = np.vstack(table_features)
                num_pad = self.max_tables - table_features.shape[0]

                if num_pad < 0:
                    print(num_pad)
                    pdb.set_trace()
                table_mask = np.ones_like(table_features).mean(1, keepdims=True)
                table_features = np.pad(table_features, ((0, num_pad), (0, 0)), 'constant')
                table_mask = np.pad(table_mask, ((0, num_pad), (0, 0)), 'constant')
                table_features = np.expand_dims(table_features, 0)
                table_mask = np.expand_dims(table_mask, 0)

                join_features = np.vstack(join_features)
                num_pad = self.max_joins - join_features.shape[0]
                # assert num_pad >= 0
                if num_pad < 0:
                    print(num_pad)
                    pdb.set_trace()
                join_mask = np.ones_like(join_features).mean(1, keepdims=True)
                join_features = np.pad(join_features, ((0, num_pad), (0, 0)), 'constant')
                join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
                join_features = np.expand_dims(join_features, 0)
                join_mask = np.expand_dims(join_mask, 0)

            # FIXME: convert to set syntax? no set syntax?
            if self.flow_features:
                # use db to generate feature vec using nodes + qrep
                # info2 = qrep["subset_graph"].nodes()[nodes]
                if "pred_types" in info:
                    cmp_op = info["pred_types"][0]
                else:
                    cmp_op = None
                flow_features = self.db.get_flow_features(nodes,
                        qrep["subset_graph"], qrep["template_name"],
                        qrep["join_graph"], cmp_op, self.db_year)
                # heuristic estimate for the cardinality of this node
                if self.heuristic_features:
                    assert flow_features[-1] == 0.0
                    flow_features[-1] = sample_heuristic_est
            else:
                flow_features = []

            # now, store features
            X["table"].append(table_features)
            X["join"].append(join_features)
            X["pred"].append(pred_features)
            X["flow"].append(flow_features)

            if self.use_set_padding in [1,3]:
                X["pred_mask"].append(predicate_mask)
                X["table_mask"].append(table_mask)
                X["join_mask"].append(join_mask)

            Y.append(self.normalize_val(true_val, total))
            cur_info = {}
            cur_info["num_tables"] = len(nodes)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx

            # FIXME:
            # cur_info["total"] = total
            cur_info["total"] = 0.00
            sample_info.append(cur_info)

        if self.use_set_padding == 3:
            assert len(X) == 7
            # save_object(qpathx, X)
            save_object_gzip(qpathx, X)
            # save_object(qpathy, Y)
            # save_object(qpathi, sample_info)

        assert len(Y) == len(sample_info) == len(X["table"])
        return X,Y,sample_info

    def _get_query_features(self, qrep, dataset_qidx,
            query_idx):
        '''
        @qrep: qrep dict.
        @ret: table_feats, join_feats, pred_feats, flow_feats
        '''
        if self.featurization_type == "combined":
            X = []
        else:
            X = defaultdict(list)

        Y = []
        sample_info = []

        # if self.featurization_type == "transformer":
            # cur_query_features = []
            # cur_y = []

        node_data = qrep["join_graph"].nodes(data=True)

        table_feat_dict = {}
        pred_feat_dict = {}
        edge_feat_dict = {}

        # iteration order doesn't matter
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

            table_feat_dict[node] = table_features
            # TODO: pass in the cardinality as well.
            heuristic_est = None
            if self.heuristic_features:
                node_key = tuple([node])
                cards = qrep["subset_graph"].nodes()[node_key][self.cinfo_key]
                if "total" in cards:
                    total = cards["total"]
                else:
                    total = None
                heuristic_est = self.normalize_val(cards["expected"],
                        total)

            if len(info["pred_cols"]) == 0:
                pred_features = np.zeros(self.db.pred_features_len)
            else:
                pred_features = self.db.get_pred_features(info["pred_cols"][0],
                        info["pred_vals"][0], info["pred_types"][0],
                        self.db_year,
                        pred_est=heuristic_est)
            assert len(pred_features) == self.db.pred_features_len
            pred_feat_dict[node] = pred_features

        edge_data = qrep["join_graph"].edges(data=True)
        for edge in edge_data:
            info = edge[2]
            edge_features = self.db.get_join_features(info["join_condition"])
            edge_key = (edge[0], edge[1])
            edge_feat_dict[edge_key] = edge_features

        # now, we will generate the actual feature vectors over all the
        # subqueries. Order matters - dataset idx will be specified based on
        # order.
        node_names = list(qrep["subset_graph"].nodes())
        if SOURCE_NODE in node_names:
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
                    # true_val = info["cardinality"]["expected"]
                    true_val = 1.0
            else:
                ck = self.card_key
                if ck == "actual" and ck not in info[self.cinfo_key]:
                    true_val = info[self.cinfo_key]["expected"]
                else:
                    true_val = info[self.cinfo_key][ck]

            if "total" in info["cardinality"]:
                total = info["cardinality"]["total"]
            else:
                total = None

            pred_features = np.zeros(self.db.pred_features_len)
            table_features = np.zeros(self.db.table_features_len)
            join_features = np.zeros(len(self.db.joins))

            # these are base tables within a join (or node) in the subset
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
                        qrep["join_graph"], cmp_op, self.db_year)
                # heuristic estimate for the cardinality of this node
                flow_features[-1] = pred_features[-1]
            else:
                flow_features = []

            # now, store features
            if self.featurization_type == "combined" or \
                    self.featurization_type == "transformer":
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
                # if self.featurization_type == "combined":
                    # X.append(np.concatenate(comb_feats))
                # elif self.featurization_type == "transformer":
                    # cur_query_features.append(np.concatenate(comb_feats))
                X.append(np.concatenate(comb_feats))
            else:
                X["table"].append(table_features)
                X["join"].append(join_features)
                X["pred"].append(pred_features)
                X["flow"].append(flow_features)

            # if self.featurization_type == "transformer":
                # cur_y.append(self.normalize_val(true_val, total))
            # else:
            Y.append(self.normalize_val(true_val, total))

            cur_info = {}
            cur_info["num_tables"] = len(nodes)
            cur_info["dataset_idx"] = dataset_qidx + node_idx
            cur_info["query_idx"] = query_idx

            # cur_info["query_name"] = qrep["name"]

            # FIXME:
            # cur_info["total"] = total
            cur_info["total"] = 0.00
            sample_info.append(cur_info)

        return X,Y,sample_info


    def _get_feature_vectors(self, samples, feature_dir=None):
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

        self.input_feature_len = 0
        self.input_feature_len += self.db.pred_features_len
        self.input_feature_len += self.db.table_features_len
        self.input_feature_len += len(self.db.joins)
        if self.flow_features:
            self.input_feature_len += self.db.num_flow_features

        for i, qrep in enumerate(samples):
            if self.featurization_type == "set":
                x,y,cur_info = self._get_query_features_set(qrep, qidx, i)
                # just checking that shape works
                if self.use_set_padding == 3:
                    try:
                        for k,v in x.items():
                            to_variable(v, requires_grad=False).float()
                    except Exception as e:
                        print(e)
                        print("going to try w/o using saved features now")
                        x,y,cur_info = self._get_query_features_set(qrep, qidx, i,
                                use_saved=False)

            else:
                x,y,cur_info = self._get_query_features(qrep, qidx, i)

            qidx += len(y)

            if self.preload_features == 1:
                if self.featurization_type == "combined":
                    X += x
                else:
                    for k,v in x.items():
                        X[k] += v

            elif self.preload_features == 2:
                # store at the appropriate index
                if self.load_query_together:
                    # store at i
                    xpath = feature_dir + str(i) + "x.pkl"
                    # save_object(xpath, x)
                    save_object_gzip(xpath, x)
                else:
                    # store each at qidx
                    if self.featurization_type == "combined":
                        for qi, xi in enumerate(x):
                            dataset_idx = cur_info[qi]["dataset_idx"]
                            xpath = feature_dir + str(dataset_idx) + "x.pkl"
                            # make_dir(subq_feature_dir)
                            # save_object(xpath, xi)
                            save_object_gzip(xpath, xi)
                    else:
                        for qi, qinfo in enumerate(cur_info):
                            dataset_idx = qinfo["dataset_idx"]
                            curx = {}
                            for k,v in x.items():
                                curx[k] = x[k][qi]
                            xpath = feature_dir + str(dataset_idx) + "x.pkl"
                            # save_object(xpath, curx)
                            save_object_gzip(xpath, curx)

            Y += y
            sample_info += cur_info

        print("get features took: ", time.time() - start)

        if self.preload_features == 1:
            if self.featurization_type == "combined":
                X = to_variable(X, requires_grad=False).float()
            elif self.featurization_type == "set":
                if not self.use_set_padding:
                    for k,v in X.items():
                        if "flow" in k:
                            X[k] = to_variable(v, requires_grad=False).float()
                        else:
                            for xi, x in enumerate(X[k]):
                                X[k][xi] = to_variable(x, requires_grad=False).float()
                else:
                    if self.use_set_padding in [1,3]:
                        try:
                            for k,v in X.items():
                                X[k] = to_variable(v, requires_grad=False).float()
                                X[k] = X[k].squeeze()
                                if "mask" in k:
                                    X[k] = X[k].unsqueeze(2)
                        except Exception as e:
                            print(e)
                            pdb.set_trace()

                    elif self.use_set_padding == 2:
                        # don't do anything, create arrays when accessing index
                        X["flow"] = to_variable(X["flow"],
                                requires_grad=False).float()
                        X["flow"] = X["flow"].squeeze()
                    else:
                        assert False

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
        # print(index)
        # pdb.set_trace()
        if self.preload_features == 1:
            if self.load_query_together:
                start_idx = self.start_idxs[index]
                end_idx = start_idx + self.idx_lens[index]
                if self.featurization_type == "combined":
                    return self.X[start_idx:end_idx], self.Y[start_idx:end_idx], \
                            self.info[start_idx:end_idx]

                elif self.featurization_type == "set":
                    if self.use_set_padding == 2:
                        tf, pf, jf, tm, pm, jm = self._pad_sets(self.X["table"][start_idx:end_idx],
                                        self.X["pred"][start_idx:end_idx],
                                        self.X["join"][start_idx:end_idx], tensors=True)

                        return tf, pf, jf, self.X["flow"][start_idx:end_idx], \
                                tm, pm, jm, self.Y[start_idx:end_idx], \
                                self.info[start_idx:end_idx]

                    elif self.use_set_padding in [1,3]:
                        return (self.X["table"][start_idx:end_idx],
                                self.X["pred"][start_idx:end_idx],
                                self.X["join"][start_idx:end_idx],
                                self.X["flow"][start_idx:end_idx],
                                self.X["table_mask"][start_idx:end_idx],
                                self.X["pred_mask"][start_idx:end_idx],
                                self.X["join_mask"][start_idx:end_idx],
                                self.Y[start_idx:end_idx],
                                self.info[start_idx:end_idx])
                    else:
                        assert False
                        # return (self.X["table"][index], self.X["pred"][index],
                                # self.X["join"][index], self.X["flow"][index],
                                # self.Y[index], self.info[index])

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
                elif self.featurization_type == "set":
                    if self.use_set_padding == 2:
                        # tf, pf, jf, tm, pm, jm = \
                            # self._pad_sets(self.X["table"][index],
                                           # self.X["pred"][index],
                                           # self.X["join"][index], tensors=True)
                        tf, pf, jf, tm, pm, jm = \
                            self._pad_sets([self.X["table"][index]],
                                           [self.X["pred"][index]],
                                           [self.X["join"][index]], tensors=True)

                        return tf, pf, jf, self.X["flow"][index], \
                                tm, pm, jm, self.Y[index], self.info[index]

                    elif self.use_set_padding in [1,3]:
                        # print(self.info[index])
                        # pdb.set_trace()
                        return (self.X["table"][index], self.X["pred"][index],
                                self.X["join"][index], self.X["flow"][index],
                                self.X["table_mask"][index],
                                self.X["pred_mask"][index],
                                self.X["join_mask"][index],
                                self.Y[index], self.info[index])
                    else:
                        return (self.X["table"][index], self.X["pred"][index],
                                self.X["join"][index], self.X["flow"][index],
                                self.Y[index], self.info[index])
                else:
                    return (self.X["table"][index], self.X["pred"][index],
                            self.X["join"][index], self.X["flow"][index],
                            self.Y[index], self.info[index])

        elif self.preload_features == 2:
            # TODO: just load them separately
            if self.load_query_together:
                xpath = self.feature_dir + str(index) + "x.pkl"
                assert os.path.exists(xpath)
                # x = load_object(xpath)
                x = load_object_gzip(xpath)
                start_idx = self.start_idxs[index]
                end_idx = start_idx + self.idx_lens[index]

                if self.featurization_type == "combined":
                    x = to_variable(x, requires_grad=False).float()
                else:
                    for k,v in x.items():
                        x[k] = to_variable(v, requires_grad=False).float()

                if self.featurization_type == "combined":
                    return x, self.Y[start_idx:end_idx], \
                            self.info[start_idx:end_idx]
                else:
                    return (x["table"],
                            x["pred"],
                            x["join"],
                            x["flow"],
                            self.Y[start_idx:end_idx],
                            self.info[start_idx:end_idx])
            else:
                xpath = self.feature_dir + str(index) + "x.pkl"
                assert os.path.exists(xpath)
                # x = load_object(xpath)
                x = load_object_gzip(xpath)
                if self.featurization_type == "combined":
                    x = to_variable(x, requires_grad=False).float()
                else:
                    for k,v in x.items():
                        x[k] = to_variable(v, requires_grad=False).float()

                if self.featurization_type == "combined":
                    return x, self.Y[index], self.info[index]
                else:
                    return (x["table"], x["pred"],
                            x["join"], x["flow"],
                            self.Y[index], self.info[index])

        elif self.preload_features in [3,4]:

            if self.preload_features == 3:
                qfn = self.query_fns[index]
                qrep = load_sql_rep(qfn)
            else:
                qrep = self.samples[index]

            start_idx = self.start_idxs[index]

            if self.featurization_type == "set":
                x,y,cur_info = self._get_query_features_set(qrep, start_idx,
                        index)
            else:
                x,y,cur_info = self._get_query_features(qrep, start_idx, index)

            assert self.load_query_together
            y = to_variable(y, requires_grad=False).float()

            if self.featurization_type == "combined":
                x = to_variable(x, requires_grad=False).float()
                return x,y,cur_info
            elif self.featurization_type == "set":

                if self.use_set_padding == 2:
                    print("going to call pad sets!")
                    pdb.set_trace()
                    tf, pf, jf, tm, pm, jm = self._pad_sets(x["table"],
                                    x["pred"],
                                    x["join"], tensors=True)

                    return tf, pf, jf, x["flow"], \
                            tm, pm, jm, y, \
                            cur_info

                elif self.use_set_padding in [1,3]:

                    for k,v in x.items():
                        x[k] = to_variable(v, requires_grad=False).float()
                        x[k] = x[k].squeeze()
                        if "mask" in k:
                            x[k] = x[k].unsqueeze(2)
                    # for k,v in x.items():
                        # x[k] = to_variable(v, requires_grad=False).float()
                    return (x["table"],
                            x["pred"],
                            x["join"],
                            x["flow"],
                            x["table_mask"],
                            x["pred_mask"],
                            x["join_mask"],
                            y,
                            cur_info)

            else:
                for k,v in x.items():
                    x[k] = to_variable(v, requires_grad=False).float()
                return x["table"], x["pred"], x["join"], x["flow"], \
                        y, cur_info

        else:
            assert False
            qidx = self.subq_to_query_idx[index]
            idx_start = self.qstart_idxs[qidx]
            subq_idx = index - idx_start
            qrep = self.samples[qidx]
            return self._get_feature_vector(qrep,
                    list(qrep["subset_graph"].nodes())[subq_idx])


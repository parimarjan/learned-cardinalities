import sys
sys.path.append(".")
import psycopg2 as pg
from utils.utils import *
from db_utils.query_storage import *
from utils.utils import *
import pdb
import random
import klepto
from multiprocessing import Pool, cpu_count
import json
import pickle
# from sql_rep.utils import execute_query
from sql_rep.utils import *
import re
from collections import defaultdict
import scipy.stats as st

MAX_WALKS = 1000000
CONF_ALPHA = 0.95

NEXT_HOP_TMP = '''SELECT {SELS} from {TABLE}
WHERE {FKEY_CONDS}'''

# NEXT_HOP_TMP = '''SELECT {SELS} from {TABLE}
# WHERE {FKEY_CONDS}
# order by random() LIMIT 1'''

# FIRST_HOP_TMP = '''SELECT {SELS} from {TABLE}
# {WHERE} LIMIT 1'''
FIRST_HOP_TMP = '''SELECT {SELS} from {TABLE} {WHERE}'''

CREATE_INDEX_TMP = '''CREATE INDEX IF NOT EXISTS {INDEX} ON {TABLE} ({COLS})'''
DROP_INDEX_TMP = '''DROP INDEX {INDEX}'''
CREATE_MULTICOL_INDEXES = False
DROP_MULTICOL_INDEXES = False

class WanderJoin():

    def __init__(self, user, pwd, db_host, port, db_name,
            verbose=False, cache_dir="./sql_cache", walks_timeout=0.5):
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name
        self.con = pg.connect(user=self.user, host=self.db_host, port=self.port,
                password=self.pwd, database=self.db_name)
        self.cursor = self.con.cursor()
        self.walks_timeout = walks_timeout
        self.verbose = verbose

    def find_path(self, nodes, node_selectivities, sg):
        sels = [node_selectivities[t] for t in nodes]
        path = []
        first_node = nodes[np.argmax(sels)]

        nodes.remove(first_node)
        path.append(first_node)

        for i in range(len(sg.nodes())-1):
            join_edges = list(nx.edge_boundary(sg, path, nodes))
            assert len(join_edges) != 0

            # use node_selectivities here...
            if len(join_edges) == 1:
                path.append(join_edges[0][1])
            else:
                options = [j[1] for j in join_edges]
                sels = [node_selectivities[t] for t in options]
                path.append(options[np.argmax(sels)])

            nodes.remove(path[-1])

        # hack, for small tables. can also do it based on expected count or so.
        if first_node in ["kt","it1","it2","it3","it4","rt","k"]:
            path[0] = path[1]
            path[1] = first_node
        return path

    def init_path_details(self, path, sg):
        # to return:
        path_execs = []
        path_join_keys = []

        first_node = path[0]
        node_info = sg.nodes()[first_node]
        table = ALIAS_FORMAT.format(TABLE = node_info["real_name"], ALIAS =
                first_node)
        nodes_seen = set()

        if first_node not in self.init_sels:
            sels = ",".join(node_info["sels"])
            where_clause = ""
            if len(node_info["predicates"]) > 0:
                preds = " AND ".join(node_info["predicates"])
                where_clause = "WHERE " + preds
            exec_sql = FIRST_HOP_TMP.format(SELS = sels,
                              TABLE = table,
                              WHERE = where_clause)
            self.cursor.execute(exec_sql)
            outputs = self.cursor.fetchall()
            self.init_sels[first_node] = outputs
            print("computed first hop outputs: ", len(outputs))

        nodes_seen.add(first_node)
        path_execs.append(None)
        path_join_keys.append(None)
        # for rest of the path, compute join statements

        for node_idx in range(1,len(path),1):
            created_index = False
            node = path[node_idx]
            node_info = sg.nodes()[node]
            table = ALIAS_FORMAT.format(TABLE = node_info["real_name"], ALIAS = node)
            sels = ",".join(node_info["sels"])
            join_edges = list(nx.edge_boundary(sg, nodes_seen, {node}))
            assert len(join_edges) != 0

            fkey_conds = []
            cur_join_cols = []
            index_cols = []

            for join in join_edges:
                assert node == join[1]
                # a value for this column would already have been selected
                other_col = sg[join[0]][join[1]][join[0]]
                cur_join_cols.append(other_col)
                # other_val = vals[other_col]
                cur_col = sg[join[0]][join[1]][join[1]]

                col_name = cur_col.split(".")[1]
                if col_name not in index_cols:
                    index_cols.append(col_name)
                other_col_key = "X" + other_col + "X"
                cond = cur_col + " = " + other_col_key
                fkey_conds.append(cond)

            path_join_keys.append(cur_join_cols)
            nodes_seen.add(node)
            assert len(fkey_conds) != 0

            # FIXME: check math
            fkey_conds += node_info["predicates"]
            fkey_cond = " AND ".join(fkey_conds)
            for col in node_info["pred_cols"]:
                col_name = col.split(".")[1]
                if col_name not in index_cols:
                    index_cols.append(col_name)

            if DROP_MULTICOL_INDEXES:
                index_name = "_".join(index_cols)
                index_cmd = DROP_INDEX_TMP.format(INDEX = index_name)

                try:
                    num_index_sql = "SELECT * FROM pg_indexes WHERE tablename = '{}'".format(\
                            node_info["real_name"])
                    self.cursor.execute(num_index_sql)
                    num_indexes = len(self.cursor.fetchall())
                    self.cursor.execute(index_cmd)
                    self.con.commit()
                    num_index_sql = "SELECT * FROM pg_indexes WHERE tablename = '{}'".format(\
                            node_info["real_name"])
                    self.cursor.execute(num_index_sql)
                    num_indexes_new = len(self.cursor.fetchall())
                    if num_indexes_new != num_indexes+1:
                        print("dropped index")
                        created_index = True
                    else:
                        print("index already existed")

                    if created_index:
                        print("going to vacuum")
                        con2 = pg.connect(user=self.user, host=self.db_host, port=self.port,
                                password=self.pwd, database=self.db_name)
                        con2.set_isolation_level(pg.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                        cursor2 = con2.cursor()
                        cursor2.execute("VACUUM {}".format(node_info["real_name"]))
                        cursor2.close()
                        con2.close()
                        print("index + vacuuming done for: ", node_info["real_name"])

                except Exception as e:
                    print(e)
                    self.cursor.execute("ROLLBACK")
                    self.con.commit()

            if CREATE_MULTICOL_INDEXES:
                index_name = "_".join(index_cols)
                index_cmd = CREATE_INDEX_TMP.format(TABLE = node_info["real_name"],
                                        INDEX = index_name,
                                        COLS = ",".join(index_cols))

                try:
                    num_index_sql = "SELECT * FROM pg_indexes WHERE tablename = '{}'".format(\
                            node_info["real_name"])
                    self.cursor.execute(num_index_sql)
                    num_indexes = len(self.cursor.fetchall())
                    self.cursor.execute(index_cmd)
                    self.con.commit()
                    num_index_sql = "SELECT * FROM pg_indexes WHERE tablename = '{}'".format(\
                            node_info["real_name"])
                    self.cursor.execute(num_index_sql)
                    num_indexes_new = len(self.cursor.fetchall())
                    if num_indexes_new == num_indexes+1:
                        print("created new index")
                        created_index = True
                    else:
                        print("index already existed")

                    if created_index:
                        print("going to vacuum")
                        con2 = pg.connect(user=self.user, host=self.db_host, port=self.port,
                                password=self.pwd, database=self.db_name)
                        con2.set_isolation_level(pg.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                        cursor2 = con2.cursor()
                        cursor2.execute("VACUUM {}".format(node_info["real_name"]))
                        cursor2.close()
                        con2.close()
                        print("index + vacuuming done for: ", node_info["real_name"])

                except Exception as e:
                    print(e)
                    self.cursor.execute("ROLLBACK")
                    self.con.commit()

            exec_sql = NEXT_HOP_TMP.format(FKEY_CONDS = fkey_cond,
                                            TABLE = table,
                                            SELS = sels)
            path_execs.append(exec_sql)


        return path_execs, path_join_keys

    def get_counts(self, qrep):
        '''
        @ret: count for each subquery
        '''
        self.init_sels = {}
        # generate a map of key : fkey pairs
        subset_graph = qrep["subset_graph"]
        join_graph = qrep["join_graph"]
        node_selectivities = {}

        for node, info in join_graph.nodes(data=True):
            sels = []
            # for col in info["pred_cols"]:
                # if col not in sels:
                    # sels.append(col)
            if len(info["predicates"]) == 0:
                node_sel = 0.00
            else:
                cards = subset_graph.nodes()[tuple([node])]["cardinality"]
                # TODO: we can also compute, and use true values here maybe?
                node_sel = float(cards["total"]) - cards["expected"]
            node_selectivities[node] = node_sel

            edges = join_graph.edges(node)
            for edge in edges:
                # edge_data = join_graph.get_edge_data(edge[0], edge[1])
                edge_data = join_graph[edge[0]][edge[1]]
                if "!" in edge_data["join_condition"]:
                    jconds = edge_data["join_condition"].split("!=")
                else:
                    jconds = edge_data["join_condition"].split("=")
                for jc in jconds:
                    jc = jc.strip()
                    if node == jc[0:len(node)]:
                        if jc not in sels:
                            sels.append(jc)
                    jc_node = jc.split(".")[0]
                    join_graph[edge[0]][edge[1]][jc_node] = jc

            join_graph.nodes()[node]["sels"] = sels

        # sort each table by selectivity

        subset_keys = list(subset_graph.nodes())
        subset_keys.sort(key = lambda v : len(v), reverse=True)

        card_ests = {}
        card_vars = {}
        card_samples = {}
        succ_walks = {}

        self.init_sels = {}
        exec_nodes = 0
        for node in subset_keys:
            if node in card_ests:
                print("skipping {} ".format(node))
                continue
            if len(node) == 1:
                # FIXME: temporary hack
                true = subset_graph.nodes()[node]["cardinality"]["actual"]
                card_ests[node] = true
                card_vars[node] = 0.0
                card_samples[node] = 1
                continue

            exec_nodes += 1

            # optimize node order, sort by their selectivities
            tables = list(node)
            sg = join_graph.subgraph(tables)
            path = self.find_path(tables, node_selectivities,
                    sg)
            # let us initialize all the pre-computed material we can at this
            # step
            path_execs, path_join_keys = self.init_path_details(path, sg)

            all_rts = []
            tot_succ = 0
            print(path)
            tot_duration = 0.00
            for i in range(MAX_WALKS):
                cur_duration, pis = self.run_path(path, sg, path_execs,
                        path_join_keys)
                all_rts.append(cur_duration)
                tot_duration += cur_duration

                cur_pi = 1
                for nodeidx, _ in enumerate(path):
                    nodes = path[0:nodeidx+1]
                    nodes.sort()
                    nodes = tuple(nodes)
                    if nodes not in card_samples:
                        card_samples[nodes] = 0.0
                        card_ests[nodes] = 0.0
                        card_vars[nodes] = 0.0
                        succ_walks[nodes] = 0.0

                    card_samples[nodes] += 1
                    fi = 0
                    if nodeidx < len(pis):
                        # this walk was successful...
                        cur_pi *= pis[nodeidx]
                        fi = cur_pi
                        card_ests[nodes] += fi
                        succ_walks[nodes] += 1

                    # computing variance
                    cur_var = (fi - (card_ests[nodes] / card_samples[nodes]))**2
                    card_vars[nodes] += cur_var

                if self.verbose:
                    if i % 500 == 0 and i != 0:
                        print("i: {}, total succ walks: {}, avg time: {}, total time: {}".format(
                            i, tot_succ, round(sum(all_rts) / len(all_rts), 5),
                            round(sum(all_rts), 2)))

                        for nodeidx, _ in enumerate(path):
                            nodes = path[0:nodeidx+1]
                            nodes.sort()
                            nodes = tuple(nodes)
                            est = round(card_ests[nodes] / card_samples[nodes], 2)
                            std = round(np.sqrt(card_vars[nodes] / float((card_samples[nodes]-1))), 2)
                            true = subset_graph.nodes()[nodes]["cardinality"]["actual"]
                            # st.norm.ppf(95+1)
                            alpha = st.norm.ppf((CONF_ALPHA+1)/2)
                            half_interval = std*alpha / np.sqrt(card_samples[nodes])
                            print("nodes: {}, succ walks: {}, true: {}, est: {}+/-{}, std: {}".format(
                                nodes, succ_walks[nodes], true, est, half_interval, std))

                # if tot_duration > self.walks_timeout and succ_walks[nodes] > 0:
                if tot_duration > self.walks_timeout:
                    print("duration exceeded, num walks: {}, num succ walks: {}".format(
                        i, succ_walks[nodes]))
                    if node not in card_ests:
                        assert node in card_samples
                        card_ests[node] = 0.0
                        card_vars[node] = 0.0
                    break

        wj_data = {}
        wj_data["card_ests_sum"] = card_ests
        wj_data["card_vars_sum"] = card_vars
        wj_data["card_samples"] = card_samples
        wj_data["succ_walks"] = succ_walks
        return wj_data

    def run_path(self, node_list, join_graph, path_execs, path_join_keys):
        pis = []
        # unique to the vals seen in this particular run
        vals = {}
        nodes_seen = set()
        start = time.time()
        for i, node in enumerate(node_list):
            node_info = join_graph.nodes()[node]
            if i == 0:
                output = random.choice(self.init_sels[node])
                for j,out in enumerate(output):
                    vals[node_info["sels"][j]] = out
                nodes_seen.add(node)
                pis.append(len(self.init_sels[node]))
                continue

            exec_sql = path_execs[i]
            for join_key in path_join_keys[i]:
                exec_sql = exec_sql.replace("X"+join_key+"X",
                        str(vals[join_key]))

            nodes_seen.add(node)
            self.cursor.execute(exec_sql)
            output = self.cursor.fetchall()

            if len(output) == 0:
                # print("returning False, because no matching foreign key")
                return time.time()-start, pis

            pis.append(len(output))
            cur_row = random.choice(output)
            for j,out in enumerate(cur_row):
                vals[node_info["sels"][j]] = out

        return time.time()-start, pis

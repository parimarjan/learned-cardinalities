import sys
import os
import pdb
sys.path.append(".")
from db_utils.utils import *
from db_utils.query_storage import *
import networkx as nx

def get_timeouts(qrep):
    timeouts = 0
    for subset, info in qrep["subset_graph"].nodes().items():
        cards = info["cardinality"]
        if cards["actual"] == TIMEOUT_COUNT_CONSTANT \
                or cards["expected"] == TIMEOUT_COUNT_CONSTANT \
                or cards["total"] == TIMEOUT_COUNT_CONSTANT:
            assert cards["actual"] == cards["expected"] == cards["total"]
            timeouts += 1

    return timeouts

def fix_qrep(qrep):
    # json-ify the graphs
    qrep["subset_graph"] = nx.adjacency_data(nx.OrderedDiGraph(qrep["subset_graph"]))
    for nd in qrep["join_graph"].nodes(data=True):
        data = nd[1]
        for i, col in enumerate(data["pred_cols"]):
            # add pred related feature
            cmp_op = data["pred_types"][i]
            if cmp_op == "in" or \
                    "like" in cmp_op or \
                    cmp_op == "eq":
                val = data["pred_vals"][i]
                if isinstance(val, dict):
                    val = [val["literal"]]
                elif not hasattr(val, "__len__"):
                    val = [val]
                elif isinstance(val[0], dict):
                    val = val[0]["literal"]
                val = set(val)
                data["pred_vals"][i] = val
    qrep["join_graph"] = nx.adjacency_data(qrep["join_graph"])

def get_samples_per_plan(df):
    keys = []
    plans = set(df["plan"])
    for plan in plans:
        cur_df = df[df["plan"] == plan]
        cur_df = cur_df.sample(1)
        keys.append(cur_df["sql_key"].values[0])

    return keys

inp_dir = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)
# os.makedirs(out_dir + "/queries/", exist_ok=True)

cur_qnum = defaultdict(int)

for tdir in os.listdir(inp_dir):

    # new_tmp = template_map[tdir]
    new_tmp = tdir
    os.makedirs(out_dir + "/queries/" + new_tmp, exist_ok=True)

    fns = os.listdir(inp_dir + "/" + tdir)
    print(tdir)

    for fn in fns:
        qrep = load_sql_rep(inp_dir + "/" + tdir + "/" + fn)

        fix_qrep(qrep)

        cur_qnum[new_tmp] += 1
        new_fn = new_tmp + str(cur_qnum[new_tmp]) + ".pkl"
        total_added += 1
        out_name = out_dir + sample_dir + new_tmp + "/" + new_fn
        save_sql_rep(out_name, qrep)


import pickle
import sys
sys.path.append(".")
import glob
from utils.utils import *
import pdb

query_dir = sys.argv[1]
print(query_dir)
fns = list(glob.glob(query_dir + "/wj_data/*"))

succ_walks = {}
node_walks = {}
num_samples = {}
exec_times = {}

for fn in fns:
    assert ".pkl" in fn
    data = load_object(fn)
    for wj_type, vals in data.items():
        max_nodes = max([len(v) for v in vals["succ_walks"].keys()])
        if wj_type not in succ_walks:
            succ_walks[wj_type] = np.zeros(max_nodes)
            node_walks[wj_type] = {}
            exec_times[wj_type] = 0.0
        if "exec_time" not in vals:
            continue
        exec_times[wj_type] += vals["exec_time"]

        walk_count = succ_walks[wj_type]
        nodes_count = node_walks[wj_type]
        for nodes, num in vals["succ_walks"].items():
            walk_count[len(nodes)-1] += num
            if nodes not in nodes_count:
                nodes_count[nodes] = 0
            nodes_count[nodes] += num

for k,v in exec_times.items():
    print(k)
    print(v / len(fns))
pdb.set_trace()
# for wj_type, vals in succ_walks.items():
    # print(wj_type)
    # for i,v in enumerate(vals):
        # print("{}: {}".format(i, round(v,2)))

# for wj_type, data in node_walks.items():
    # print(wj_type)
    # for k,v in data.items():
        # print(k, v)
    # pdb.set_trace()

# pdb.set_trace()


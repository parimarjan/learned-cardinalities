import sys
import os
import pdb
sys.path.append(".")
from db_utils.utils import *
from db_utils.query_storage import *

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

def get_samples_per_plan(df):
    keys = []
    plans = set(df["plan"])
    for plan in plans:
        cur_df = df[df["plan"] == plan]
        cur_df = cur_df.sample(1)
        keys.append(cur_df["sql_key"].values[0])

    return keys

template_map = {}
template_map["2.toml"] = "1a"
template_map["2b1.toml"] = "1a"
template_map["2b2.toml"] = "1a"
template_map["2b3.toml"] = "1a"
template_map["2b4.toml"] = "1a"
template_map["2U2.toml"] = "1a"
template_map["2d.toml"] = "2a"
template_map["2d2.toml"] = "2b"
template_map["2dtitle.toml"] = "2c"
template_map["4.toml"] = "3a"
template_map["5.toml"] = "4a"
template_map["6.toml"] = "5a"
template_map["7b.toml"] = "6a"
template_map["7.toml"] = "7a"
template_map["8.toml"] = "8a"
# template_map["2U3.toml"] = "9a"
# template_map["9.toml"] = "9a"

inp_dir = sys.argv[1]
out_dir = sys.argv[2]
results_dir = sys.argv[3]
costs_file = results_dir + "/true/costs.pkl"
costs = load_object(costs_file)
costs = costs.drop_duplicates(subset="sql_key")
# train_costs = costs[costs["samples_type"] == "train"]
# test_costs = costs[costs["samples_type"] == "test"]

# train_keys = set(train_costs["sql_key"])
# test_keys = set(test_costs["sql_key"])
# runtime_train_keys = get_samples_per_plan(train_costs)
# runtime_test_keys = get_samples_per_plan(test_costs)
keys = set(costs["sql_key"])
runtime_keys = get_samples_per_plan(costs)

os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + "/queries/", exist_ok=True)
os.makedirs(out_dir + "/runtime_queries/", exist_ok=True)

# os.makedirs(out_dir + "/train/", exist_ok=True)
# os.makedirs(out_dir + "/test/", exist_ok=True)
# os.makedirs(out_dir + "/runtime_train/", exist_ok=True)
# os.makedirs(out_dir + "/runtime_test/", exist_ok=True)

cur_qnum = {}
for k,v in template_map.items():
    cur_qnum[v] = 0

for tdir in os.listdir(inp_dir):
    if tdir not in template_map:
        continue
    new_tmp = template_map[tdir]
    fns = os.listdir(inp_dir + "/" + tdir)
    # num_samples = get_template_samples(tdir)
    # fns = fns[0:num_samples]
    print(tdir)

    total_timeouts = 0
    skipped_queries = 0
    total_added = 0
    num_rt = 0
    for fn in fns:
        qrep = load_sql_rep(inp_dir + "/" + tdir + "/" + fn)
        sql_key = str(deterministic_hash(qrep["sql"]))
        rt_dir = None
        if sql_key in keys:
            sample_dir = "/queries/"
            if sql_key in runtime_keys:
                rt_dir = "/runtime_queries/"
        else:
            skipped_queries += 1
            continue

        timeout = get_timeouts(qrep)
        total_timeouts += timeout
        # if timeout > 0:
            # print("timeout: ", timeout)
        if timeout > 10:
            skipped_queries += 1
            continue

        cur_qnum[new_tmp] += 1
        new_fn = new_tmp + str(cur_qnum[new_tmp]) + ".pkl"
        total_added += 1
        out_name = out_dir + sample_dir + new_fn
        with open(out_name, 'wb') as fp:
            pickle.dump(qrep, fp, protocol=pickle.HIGHEST_PROTOCOL)

        if rt_dir is not None:
            num_rt += 1
            out_name2 = out_dir + rt_dir + new_fn
            with open(out_name2, 'wb') as fp:
                pickle.dump(qrep, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print("{}: avg timeouts: {}".format(tdir, total_timeouts / len(fns)))
    print("num skipped queries: ", skipped_queries)
    print("num added queries: ", total_added)
    print("num rt queries: ", num_rt)
    # pdb.set_trace()

import sys
import os
import pdb
sys.path.append(".")
from db_utils.utils import *
from db_utils.query_storage import *

def analyze_query(fn):
    qrep = load_sql_rep(inp_dir + "/" + tdir + "/" + fn)
    timeouts = 0

    for subset, info in qrep["subset_graph"].nodes().items():
        cards = info["cardinality"]
        if cards["actual"] == TIMEOUT_COUNT_CONSTANT \
                or cards["expected"] == TIMEOUT_COUNT_CONSTANT \
                or cards["total"] == TIMEOUT_COUNT_CONSTANT:
            assert cards["actual"] == cards["expected"] == cards["total"]
            timeouts += 1
        # if cards["actual"] == cards["total"]:
            # print(subset)
            # print("total = actual = {}, expected = {}".format(cards["actual"],
                # cards["expected"]))

    return timeouts

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
template_map["8.toml"] = "7a"
template_map["7.toml"] = "8a"
template_map["2U3.toml"] = "9a"

inp_dir = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)

cur_qnum = {}
for k,v in template_map.items():
    cur_qnum[v] = 0

for tdir in os.listdir(inp_dir):
    if tdir not in template_map:
        continue
    new_tmp = template_map[tdir]
    fns = os.listdir(inp_dir + "/" + tdir)
    num_samples = get_template_samples(tdir)
    fns = fns[0:num_samples]
    print(tdir)

    qrep = load_sql_rep(inp_dir + "/" + tdir + "/" + fns[0])
    totals = get_all_totals(qrep)

    # analyze_query(fns[0])
    # analyze_query(fns[100])
    # total_timeouts = 0
    # for fn in fns:
        # timeout = analyze_query(fn)
        # total_timeouts += timeout
        # if timeout > 0:
            # print("timeout: ", timeout)

        # # cur_qnum[new_tmp] += 1
        # # new_fn = new_tmp + str(cur_qnum[new_tmp]) + ".pkl"
        # # out_name = out_dir + "/" + new_fn
        # # print(new_fn)
        # # pdb.set_trace()

    # print("{}: avg timeouts: {}".format(tdir, total_timeouts / len(fns)))
    pdb.set_trace()

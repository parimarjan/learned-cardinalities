import os
import pdb

# OUTPUT_DIR="./queries/stats_train/s1/"
# OUTPUT_DIR="./queries/imdb_train/all/"
OUTPUT_DIR="./queries/ergastf1_train/all/"
# INPUT_FN = "/flash1/ziniuw/data_transfer/ergastf1/workloads/simple_workload_100k.sql"
# INPUT_FN = "./queries/stats2/all.sql"

INPUT_FN = "/flash1/ziniuw/data_transfer/ErgastF1/workloads/workload_50k_s1.sql"
OUTPUT_FN_TMP = "{i}.sql"

with open(INPUT_FN, "r") as f:
    data = f.read()

data = data.replace("go", "")
queries = data.split(";")
curq = 0

for i, q in enumerate(queries):
    if "set showplan_xml on" in q:
        print(q)
        continue
    if "use tpcds" in q or "use tpch" in q:
        print(q)
        continue

    curq += 1
    ## for stats workload
    # q = q[q.find("||")+2:]
    q = q.strip()
    if q == "":
        continue
    # q = q.replace

    # pdb.set_trace()

    output_fn = OUTPUT_DIR + str(curq) + ".sql"
    with open(output_fn, "w") as f:
        f.write(q)

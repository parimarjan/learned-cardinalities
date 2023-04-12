import os
import pdb
import random
import errno

# OUTPUT_DIR="./queries/stats_train/s1/"
# OUTPUT_DIR="./queries/imdb_train/all/"

def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

WK_NAME = "ssb"
OUTPUT_DIR="./queries/{}/all/".format(WK_NAME)
make_dir(OUTPUT_DIR)

# INPUT_FN = "/flash1/ziniuw/data_transfer/ergastf1/workloads/simple_workload_100k.sql"

INPUT_FN = "/flash1/ziniuw/data_transfer/{}/workloads/complex_workload_50k_s1.sql".format(WK_NAME)

with open(INPUT_FN, "r") as f:
    data = f.read()

data = data.replace("go", "")
queries = data.split(";")
curq = 0

queries = random.sample(queries, 150)

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

    output_fn = OUTPUT_DIR + WK_NAME + str(curq) + ".sql"

    with open(output_fn, "w") as f:
        f.write(q)

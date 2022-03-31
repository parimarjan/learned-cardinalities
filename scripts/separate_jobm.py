import os
import pdb


# OUTPUT_DIR="./queries/tpcds/all/"
# INPUT_FN = "./queries/tpcds/all.sql"
# OUTPUT_FN_TMP = "{i}.sql"

# OUTPUT_DIR="./queries/tpch/all/"
# INPUT_FN = "./queries/tpch/all.sql"
# OUTPUT_FN_TMP = "{i}.sql"

# OUTPUT_DIR="./queries/tpcds1/all/"
# INPUT_FN = "./queries/tpcds1/all.sql"
# OUTPUT_FN_TMP = "{i}.sql"

OUTPUT_DIR="./queries/tpch1/all/"
INPUT_FN = "./queries/tpch1/all.sql"
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
    if q.strip() == "":
        continue
    if "select" not in q.lower():
        continue

    curq += 1

    print(q)
    print(curq)
    print("*******")
    # pdb.set_trace()

    output_fn = OUTPUT_DIR + str(curq) + ".sql"
    with open(output_fn, "w") as f:
        f.write(q)


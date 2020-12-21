import os
import pdb


OUTPUT_DIR="./job_queries/jobm/"
INPUT_FN = "./job_queries/jobm/all/all.sql"
OUTPUT_FN_TMP = "{i}.sql"

with open(INPUT_FN, "r") as f:
    data = f.read()

queries = data.split(";")
for i, q in enumerate(queries):
    output_fn = OUTPUT_DIR + str(i+1) + ".sql"
    with open(output_fn, "w") as f:
        f.write(q)


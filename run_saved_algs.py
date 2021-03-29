import subprocess as sp
import os
import glob
import pdb

BASE_DIR="""/Users/pari/prism-testbed/all_results/imdb/VLDB/MSCN/diff/"""
OUT_DIR="""all_results/vldb/mysql/myisam/diff_mscn_debug1/"""
CMD_TMP="""time python3 main.py --algs saved --debug_set 1 --query_template all -n -1 --eval_epoch 100 --losses mysql-loss,qerr --query_dir queries/imdb --model_dir {MODEL_DIR} --result_dir {RES_DIR}"""


model_dirs = os.listdir(BASE_DIR)
out_dirs = os.listdir(OUT_DIR)
for i,o in enumerate(out_dirs):
    out_dirs[i] = o.replace("SavedRun-", "")

for mdir in model_dirs:
    if "pr" in mdir:
        print("skipping priority model!")
        continue
    if mdir in out_dirs:
        print("skipping existing experiment!")
        continue

    cmd = CMD_TMP.format(MODEL_DIR = BASE_DIR + mdir,
                         RES_DIR = OUT_DIR)

    print(cmd)
    p = sp.Popen(cmd, shell=True)
    p.wait()

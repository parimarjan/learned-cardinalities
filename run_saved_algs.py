import subprocess as sp
import os
import glob
import pdb

# BASE_DIR="""/Users/pari/prism-testbed/all_results/imdb/VLDB/MSCN/diff/"""

# BASE_DIR="""/Users/pari/prism-testbed/all_results/imdb/VLDB/FCNN/ablation/default/"""
# OUT_DIR="""all_results/vldb/mysql_lambda01/ablation/"""

# BASE_DIR="""/Users/pari/prism-testbed/all_results/imdb/VLDB/FCNN/diff/"""
# OUT_DIR="""all_results/vldb/mysql_lambda1/fcnn_diff_debug/"""

BASE_DIR="""/Users/pari/prism-testbed/all_results/imdb/MYSQL_lambda1/FCNN/rc2/diff_all/"""
OUT_DIR="""debug_fcnn_lambda1"""

CMD_TMP="""time python3 main.py --algs saved --debug_set 1 --debug_ratio 10.0 --query_template all -n -1 --eval_epoch 100 --losses mysql-loss,qerr --query_dir queries/imdb --model_dir {MODEL_DIR} --result_dir {RES_DIR} --cost_model mysql_rc2 """

model_dirs = os.listdir(BASE_DIR)
out_dirs = os.listdir(OUT_DIR)
for i,o in enumerate(out_dirs):
    out_dirs[i] = o.replace("SavedRun-", "")

for mdir in model_dirs:
    if "pr" in mdir:
        print("skipping priority model!")
        continue
    # if "flow" in mdir:
        # print("skipping flow-loss model!")
        # continue
    # if mdir in out_dirs:
        # print("skipping existing experiment!")
        # continue

    cmd = CMD_TMP.format(MODEL_DIR = BASE_DIR + mdir,
                         RES_DIR = OUT_DIR)

    print(cmd)
    p = sp.Popen(cmd, shell=True)
    p.wait()

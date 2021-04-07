import subprocess as sp
import os
import glob
import pdb

# BASE_DIR="""/Users/pari/prism-testbed/all_results/imdb/VLDB/MSCN/diff/"""
# BASE_DIR="""all_results/mysql/mscn/default/"""
# OUT_DIR="""all_results/mysql/mscn/default_debug/"""

# BASE_DIR = """all_results/mysql/fcnn/diff/no_norm_allseeds/"""
# BASE_DIR = """all_results/mysql/fcnn/diff/norm_allseeds/"""
# OUT_DIR = """norm_allseeds_debug/"""

# BASE_DIR="""/home/ubuntu/learned-cardinalities/all_results/mysql/fcnn/diff"""

# BASE_DIR="""diff_fcnn/"""
# OUT_DIR="""diff_mse_debug05/"""

# BASE_DIR="""all_results/mysql/fcnn/diff/norm_678/"""
BASE_DIR="""all_results/mysql/fcnn/diff/norm_allseeds/"""
OUT_DIR="""diff_mse_debug05/"""

# BASE_DIR="""all_results/mysql/fcnn/diff/final6_all_other_seeds/"""
# OUT_DIR="""flow_loss_seed1/"""

CMD_TMP="""time python3 main.py --algs saved --debug_set 1 --debug_ratio 10 --query_template all -n -1 --eval_epoch 100 --losses qerr,mysql-loss --query_dir queries/imdb --model_dir {MODEL_DIR} --result_dir {RES_DIR}"""

model_dirs = os.listdir(BASE_DIR)
out_dirs = os.listdir(OUT_DIR)
for i,o in enumerate(out_dirs):
    out_dirs[i] = o.replace("SavedRun-", "")

for mdir in model_dirs:
    if "pr" in mdir:
        print("skipping priority model!")
        continue
    # if "flow" in mdir:
        # print("skipping flow model!")
        # continue

    if mdir in out_dirs:
        print("skipping existing experiment!")
        continue

    cmd = CMD_TMP.format(MODEL_DIR = BASE_DIR + mdir,
                         RES_DIR = OUT_DIR)

    print(cmd)
    p = sp.Popen(cmd, shell=True)
    p.wait()

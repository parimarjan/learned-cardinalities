import subprocess as sp
import os
import glob
import pdb

BASE_DIR="""/Users/pari/prism-testbed/all_results/imdb/VLDB/MSCN/diff/"""
CMD_TMP="""time python3 main.py --algs saved --debug_set 0 --query_template all -n -1 --eval_epoch 100 --losses mysql-loss,qerr --query_dir queries/imdb --model_dir {MODEL_DIR} --result_dir all_results/vldb/mysql/myisam/diff2/"""


model_dirs = os.listdir(BASE_DIR)

for mdir in model_dirs:
    cmd = CMD_TMP.format(MODEL_DIR = BASE_DIR + mdir)
    print(cmd)
    p = sp.Popen(cmd, shell=True)
    p.wait()

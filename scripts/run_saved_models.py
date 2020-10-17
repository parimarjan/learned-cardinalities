import sys
sys.path.append(".")
import argparse
from utils.utils import *
import pdb
import random
import os

RUN_TMP='''export CUDA_VISIBLE_DEVICES="";python3 main.py --algs nn \
--model_dir {MODEL_DIR} \
 --max_epochs 0 --eval_epoch 100 --losses {LOSSES} \
 --use_set_padding {PADDING} \
 --debug_set {DEBUG_SET} --eval_on_job {JOB} \
 --result_dir {RES_DIR} \
'''

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=False,
            default=None)
    parser.add_argument("--result_dir", type=str, required=False,
            default=None)
    parser.add_argument("--losses", type=str, required=False,
            default="qerr,join-loss,plan-loss")
    parser.add_argument("--debug_set", type=int, required=False,
            default=0)
    parser.add_argument("--eval_on_job", type=int, required=False,
            default=0)
    # parser.add_argument("--query_directory", type=int, required=False,
            # default=0)

    return parser.parse_args()

def main():
    model_dirs = list(glob.glob(args.base_dir + "/*"))

    for i, model_dir in enumerate(model_dirs):
        print(i, model_dir)
        if args.result_dir is None:
            res_dir = args.base_dir
        else:
            res_dir = args.result_dir

        if os.path.exists(model_dir + "/cm1_jerr.pkl"):
            continue

        if not os.path.exists(model_dir + "/model_weights.pt"):
            continue

        cmd = RUN_TMP.format(MODEL_DIR = model_dir,
                LOSSES = args.losses,
                DEBUG_SET = args.debug_set,
                PADDING = 3,
                JOB = args.eval_on_job,
                RES_DIR = res_dir)
        print(cmd)
        os.system(cmd)
        break

args = read_flags()
main()

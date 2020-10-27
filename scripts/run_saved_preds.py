import sys
sys.path.append(".")
import argparse
from utils.utils import *
import pdb
import random
import os
import subprocess as sp

RUN_TMP='''export CUDA_VISIBLE_DEVICES="";python3 main.py --algs saved \
--model_dir {MODEL_DIR} --result_dir {RES_DIR} --debug_set {DEBUG}
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
        if args.result_dir is None:
            res_dir = args.base_dir
        else:
            res_dir = args.result_dir

        # if os.path.exists(model_dir + "/done.pkl"):
            # print("continuing because done")
            # continue

        if os.path.exists(model_dir + "/cm1_jerr.pkl"):
            continue

        if not os.path.exists(model_dir + "/model_weights.pt"):
            print("no model weights")
            continue

        print(model_dir)
        cmd = RUN_TMP.format(MODEL_DIR = model_dir,
                             RES_DIR = res_dir,
                             DEBUG = args.debug_set)

        print(cmd)
        p = sp.Popen(cmd, shell=True)
        p.wait()

        done = []
        save_object(model_dir + "/done.pkl", done)

args = read_flags()
main()

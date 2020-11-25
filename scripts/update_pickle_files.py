import sys
sys.path.append(".")
import argparse
from utils.utils import *
import pdb
import random
import os
import subprocess as sp

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
        fns = list(glob.glob(model_dir + "/*.csv"))
        # fns = list(glob.glob(model_dir + "/*.pkl"))
        print(model_dir)
        for fn in fns:
            try:
                data = load_object(fn)
            except Exception as e:
                print(e)
                print(fn)
                pdb.set_trace()
                continue
            # data = pd.read_csv(fn, sep="|")
            csv_name = fn.replace(".csv", ".pkl")
            save_object(csv_name, data, use_csv=False)
            os.remove(fn)
            # save_object(fn, data)

args = read_flags()
main()

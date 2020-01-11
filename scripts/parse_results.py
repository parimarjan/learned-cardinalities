import sys
sys.path.append(".")
import pickle
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from collections import defaultdict
import os
from utils.utils import *
import pdb

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    return parser.parse_args()

def extract_exp_config(exp_dir):
    '''
    returns all the column values we should extract:
    '''
    # return nn_type, max_buckets
    pass

def load_qerrs(exp_dir):
    qerrs = load_object(exp_dir + "/qerr.pkl")
    if qerrs is not None:
        return qerrs

    print("deal with preds -> qerrs conversion")
    assert False

def load_jerrs(exp_dir):
    jerrs = load_object(exp_dir + "/jerr.pkl")
    if jerrs is not None:
        return jerrs

    assert False

def main():
    fns = os.listdir(args.results_dir)
    for fn in fns:
        print(fn)
        cur_dir = args.results_dir + "/" + fn
        qerrs = load_qerrs(cur_dir)
        jerrs = load_jerrs(cur_dir)
        exp_args = load_object(cur_dir + "/args.pkl")
        print(exp_args)
        pdb.set_trace()

    # print("going to parse results from {}, having {} files".format())

args = read_flags()
main()

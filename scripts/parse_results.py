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
from cardinality_estimation.losses import *

LOSS_COLUMNS = ["loss_type", "loss", "summary_type", "template", "num_samples",
        "samples_type", "template"]
EXP_COLUMNS = ["num_hidden_layers", "hidden_layer_size",
        "sampling_priority_alpha", "max_discrete_featurizing_buckets",
        "heuristic_features", "alg"]

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
    if qerrs is None:
        assert False
    return qerrs

def load_jerrs(exp_dir):
    jerrs = load_object(exp_dir + "/jerr.pkl")
    if jerrs is None:
        assert False

    stats = defaultdict(list)

    for samples_type in set(jerrs["samples_type"]):
        cur_jerrs = jerrs[jerrs["samples_type"] == samples_type]
        add_row(cur_jerrs["cost"].values, "jcost", -1, "all", "all", samples_type,
                stats)
        for template in set(cur_jerrs["template"]):
            tmp_jerrs = cur_jerrs[cur_jerrs["template"] == template]
            add_row(tmp_jerrs["cost"].values, "jcost", -1, template, "all",
                    samples_type, stats)

    return pd.DataFrame(stats)

def get_alg_name(exp_args):
    if exp_args["algs"] == "nn":
        name = exp_args["nn_type"]
        if exp_args["sampling_priority_alpha"] > 0:
            name += "-" + exp_args["sampling_priority_alpha"]
        return name
    else:
        return exp_args["algs"]

def get_summary_df():
    all_dfs = []
    fns = os.listdir(args.results_dir)
    for fn in fns:
        print(fn)
        cur_dir = args.results_dir + "/" + fn
        qerrs = load_qerrs(cur_dir)
        qerrs = qerrs[qerrs["num_tables"] == "all"]
        qerrs = qerrs[LOSS_COLUMNS]

        jerrs = load_jerrs(cur_dir)
        jerrs = jerrs[LOSS_COLUMNS]
        # TODO: add rts too, if it exists

        cur_df = pd.concat([qerrs, jerrs], ignore_index=True)

        # convert to same format as qerrs

        exp_args = load_object(cur_dir + "/args.pkl")
        exp_args = vars(exp_args)
        exp_args["alg"] = get_alg_name(exp_args)
        for exp_column in EXP_COLUMNS:
            cur_df[exp_column] = exp_args[exp_column]

        all_dfs.append(cur_df)

    summary_df = pd.concat(all_dfs, ignore_index=True)
    return summary_df

def main():
    summary_df = get_summary_df()
    print(summary_df)
    pdb.set_trace()

args = read_flags()
main()

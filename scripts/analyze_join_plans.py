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

def get_alg_name(exp_args):
    if exp_args["algs"] == "nn":
        name = exp_args["nn_type"]
        if exp_args["sampling_priority_alpha"] > 0:
            name += "-" + str(exp_args["sampling_priority_alpha"])
        return name
    else:
        return exp_args["algs"]

def get_all_jerrs():
    all_dfs = []
    fns = os.listdir(args.results_dir)
    for fn in fns:
        cur_dir = args.results_dir + "/" + fn
        try:
            jerrs = load_object(cur_dir + "/jerr.pkl")
        except:
            print("skipping ", cur_dir)
            continue
        if jerrs is None:
            continue
        exp_args = load_object(cur_dir + "/args.pkl")
        exp_args = vars(exp_args)
        exp_args["alg"] = get_alg_name(exp_args)

        all_dfs.append(jerrs)

    all_jerrs = pd.concat(all_dfs, ignore_index=True)
    print(all_jerrs.groupby(["sql_key", "plan"]).size())
    print(len(all_jerrs))
    pdb.set_trace()
    return summary_df

def main():
    jerrs = get_all_jerrs()
    # SUMMARY_TITLE_FMT = "{ST}-{LT}-{SUMMARY}"
    # pdf = PdfPages("results.pdf")
    # for samples_type in set(summary_df["samples_type"]):
        # st_df = summary_df[summary_df["samples_type"] == samples_type]
        # for lt in set(st_df["loss_type"]):
            # lt_df = st_df[st_df["loss_type"] == lt]
            # for summary_type in PLOT_SUMMARY_TYPES:
                # plot_df = lt_df[lt_df["summary_type"] == summary_type]
                # plot_df = plot_df[plot_df["template"] == "all"]
                # print(set(plot_df["alg"]))
                # title = SUMMARY_TITLE_FMT.format(ST = samples_type,
                                                 # LT = lt,
                                                 # SUMMARY = summary_type)
                # plot_summary(pdf, plot_df, title)

    # pdf.close()
    # pdb.set_trace()

args = read_flags()
main()

import argparse
import os
import pandas
import time
from utils.utils import *
import pdb
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc("pdf", fonttype=42)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.image as mpimg

# TODO: maybe will use this?
def init_result_row(result):
    result["dbname"].append(args.db_name)
    result["template_dir"].append(args.template_dir)
    result["seed"].append(args.random_seed)
    result["args"].append(args)
    result["num_bins"].append(args.num_bins)
    result["avg_factor"].append(args.avg_factor)
    result["num_columns"].append(len(db.column_stats))

    result["train-time"].append(train_time)
    result["eval-time"].append(eval_time)
    result["alg_name"].append(alg.__str__())
    result["loss-type"].append(loss_func.__name__)
    result["loss"].append(cur_loss)
    result["test-set"].append(0)
    result["num_vals"].append(len(train_queries))

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_len", type=int, required=False,
            default=10)
    parser.add_argument("--db_name", type=str, required=False,
            default="synth-gaussian")
    parser.add_argument("--plot", type=int, required=False,
            default=1)
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results/")
    return parser.parse_args()

def parse_query_file(fn):
    '''
    Plot List:
        - selectivity hist for all main queries
        - selectivity buckets for each subquery CLASS
            - plot tables used as a graph
            - first bucket them by tables used
            - for each class, have separate qerror values

    '''
    print(fn)
    pdf_name = fn.replace(".pickle", ".pdf")
    queries = load_object(fn)
    print(len(queries))

    pdf = PdfPages(pdf_name)
    firstPage = plt.figure()
    firstPage.clf()
    ## TODO: just paste all the args here?
    txt = ""
    txt += queries[0].query

    firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
    pdf.savefig()
    plt.close()

    true_sels = [q.true_sel for q in queries]
    x = pd.Series(true_sels, name="true selectivities")
    ax = sns.distplot(x)
    plt.title("Distribution of True Selectivities")
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

    pdf.close()

def parse_data():
    fns = glob.glob(args.result_dir + "/*train*.pickle")
    for fn in fns:
        parse_query_file(fn)

def main():
    parse_data()

args = read_flags()
main()

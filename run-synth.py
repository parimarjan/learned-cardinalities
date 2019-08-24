import subprocess as sp
import argparse
import os
import pandas
import time
from utils.utils import *
import pdb
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.image as mpimg
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# FIXME: should not need gen_data once it has been generated
SYNTH_EXP_TMP = '''python3 main.py --db_name {DB_NAME} --gen_synth_data 1 \
 --synth_num_columns {NUM_COLS} --synth_period_len {PERIOD_LEN} \
 --algs {ALGS} --synth_num_vals {NUM_VALS} --min_corr {MIN_CORR} --seed {SEED} \
 --num_samples_per_template {NUM_SAMPLES} --result_dir {RESULT_DIR}'''

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_len", type=int, required=False,
            default=10)
    parser.add_argument("--db_name", type=str, required=False,
            default="synth-gaussian")
    # parser.add_argument("--table_prefix", type=str, required=False,
            # default="synth")
    # parser.add_argument("--test_size", type=float, required=False,
            # default=0.70)
    # parser.add_argument("--max_iter", type=int, required=False,
            # default=1000000)
    parser.add_argument("--max_parallel", type=int, required=False,
            default=1)
    parser.add_argument("--run_all", type=int, required=False,
            default=0)
    parser.add_argument("--plot", type=int, required=False,
            default=0)
    # parser.add_argument("--gen_data", type=int, required=False,
            # default=0)
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results/")

    parser.add_argument("--algs", type=str, required=False,
            default="postgres,chow,bn-exact")
    return parser.parse_args()

def run_all():
    num_vals = args.period_len*100000
    num_samples = args.period_len*100

    processes = []
    log_files = []
    commands = []
    # SEEDS = [1234, 12345, 2944, 9030, 21221]
    SEEDS = [1234, 12345]
    MIN_CORR = [0.0, 0.2, 0.5]
    NUM_COLS = range(3, 8)
    # MIN_CORR = [0.0]
    # NUM_COLS = range(3, 5)
    for num_cols in NUM_COLS:
        for seed in SEEDS:
            for min_corr in MIN_CORR:
                print(num_cols, seed, min_corr)
                cmd = SYNTH_EXP_TMP.format(PERIOD_LEN = args.period_len,
                                      DB_NAME = args.db_name,
                                      MIN_CORR = min_corr,
                                      NUM_COLS = num_cols,
                                      NUM_VALS = num_vals,
                                      NUM_SAMPLES = num_samples,
                                      ALGS = args.algs,
                                      SEED = seed,
                                      RESULT_DIR = args.result_dir)
                commands.append(cmd)

    num_processes = 0
    for i, cmd in enumerate(commands):
        ## TODO: don't repeat commands
        # file_name = gen_results_name(table_name)
        # if os.path.exists(file_name):
            # print("skipping {}, as {} exists".format(table_name, file_name))
            # continue
        # TODO: add logging?
        num_processes += 1
        sp_log = open("./logs-run-all/" + str(i) + ".log", 'a')
        sp_log.write(cmd + "\n")
        log_files.append(sp_log)
        p = sp.Popen(cmd, stdout=sp_log, stderr=sp_log, shell=True)
        # p = sp.Popen(cmd, shell=True)
        processes.append(p)
        print(cmd)
        # FIXME: slowdown by 1 lagger process here.
        print("going to wait for {} processes to finish".format(args.max_parallel))
        while num_processes >= args.max_parallel:
            time.sleep(60)
            for pi, p in enumerate(processes):
                if p is None:
                    continue
                if p.poll() == None:
                    print("remaining processes: ", num_processes)
                    num_processes -= 1
                processes[pi] = None

    print("all processes started")
    for p in processes:
        if p is None:
            continue
        p.wait()

    for f in log_files:
        f.close()

    print("finished executing all!")

def plot_losses(df, pdf):
    ## plot losses, averaged across number of columns.
    loss_types = set(df["loss-type"])
    for i, ltype in enumerate(loss_types):
        cur_df = df[df["loss-type"] == ltype]
        if "synth" in args.db_name:
            ax = sns.barplot(x="num_columns", y="loss", hue="alg_name",
                    data=cur_df, estimator=np.mean, ci=75)
        else:
            ax = sns.barplot(x="alg_name", y="loss", hue="alg_name",
                    data=cur_df, estimator=np.mean, ci=75)

        fig = ax.get_figure()
        fig.get_axes()[0].set_yscale('log')
        plt.title(ltype)
        plt.tight_layout()
        pdf.savefig()
        plt.clf()

def plot_synth():
    pdf = PdfPages('exp_results.pdf')
    fns = glob.glob(args.result_dir + "/*.pd")
    orig_df = None
    for fn in fns:
        df = load_object(fn)
        if orig_df is None:
            orig_df = df
        else:
            orig_df = orig_df.append(df, ignore_index=True)

    if "num_columns" not in orig_df:
        orig_df["num_columns"] = orig_df.apply(lambda row: len(row["means"]),
                axis=1)
    orig_df = orig_df[orig_df["dbname"] == args.db_name]
    orig_df = orig_df[orig_df["test-set"] == 0]
    # temporary solutions
    # orig_df.loc[orig_df.alg_name == "BN-chow-liu-bins100-avg_factor1",
            # "alg_name"] = "BN-chow-liu"
    # orig_df.loc[orig_df.alg_name == "BN-exact-dp-bins100-avg_factor1",
            # "alg_name"] = "BN-exact-dp"

    print(orig_df.keys())
    ## groupby stuff:
    pdb.set_trace()
    gb = orig_df.groupby(["alg_name", "loss-type", "num_bins"]).mean()
    gb = gb.xs("compute_qerror", level=1)
    print(gb)
    pdb.set_trace()
    ## parameters
    firstPage = plt.figure()
    firstPage.clf()
    ## TODO: just paste all the args here?
    txt = "Average Results \n"
    txt += "Num Experiments: " + str(len(set(orig_df["train-time"]))) + "\n"
    txt += "DB: " + str(set(orig_df["dbname"])) + "\n"
    txt += "Algs: " + str(set(orig_df["alg_name"])) + "\n"
    txt += "Num Test Samples: " + str(set(orig_df["num_vals"])) + "\n"

    firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
    pdf.savefig()
    plt.close()

    ## training / eval-time

    if "synth" in args.db_name:
        ax = sns.barplot(x="num_columns", y="train-time", hue="alg_name",
                data=orig_df, estimator=np.mean, ci=75)
    else:
        ax = sns.barplot(x="alg_name", y="train-time", hue="alg_name",
                data=orig_df, estimator=np.mean, ci=75)

    fig = ax.get_figure()
    plt.title("Training Time (seconds)")
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

    if "synth" in args.db_name:
        ax = sns.barplot(x="num_columns", y="eval-time", hue="alg_name",
                data=orig_df, estimator=np.mean, ci=75)
    else:
        ax = sns.barplot(x="alg_name", y="eval-time", hue="alg_name",
                data=orig_df, estimator=np.mean, ci=75)

    fig = ax.get_figure()
    # fig.get_axes()[0].set_yscale('log')
    plt.title("Evaluaton Time (seconds)")
    plt.tight_layout()
    pdf.savefig()
    plt.clf()


    plot_losses(orig_df, pdf)

    for i, fn in enumerate(fns):
        df = load_object(fn)
        if "dbname" not in df:
            continue
        df = df[df["dbname"] == args.db_name]
        if len(df) == 0:
            continue

        exp_hash = extract_ints_from_string(fn)[-1]
        firstPage = plt.figure()
        firstPage.clf()
        txt = "Experiment: " + exp_hash + "\n"
        txt += "DB: " + str(set(df["dbname"])) + "\n"
        # txt += "means: \n"
        # txt += str(df["means"][0]) + "\n"
        # txt += "covs: \n"
        # txt += str(df["covs"][0]) + "\n"
        firstPage.text(0.5, 0, txt, transform=firstPage.transFigure, ha="center")
        pdf.savefig()
        plt.clf()

        ## find the relevant pngs
        bn_imgs = glob.glob(args.result_dir + "/*" + exp_hash + "*.png")
        for img_name in bn_imgs:
            img = mpimg.imread(img_name)
            imgplot = plt.imshow(img)
            plt.title(img_name)
            pdf.savefig()
            plt.clf()

        # if "num_columns" not in df:
            # df["num_columns"] = df.apply(lambda row: len(row["means"]),
                    # axis=1)
        # no point in plotting each individual loss ...
        # plot_losses(df, pdf)

    pdf.close()

def main():
    if args.run_all:
        run_all()

    # plotting time
    # go over all the .pd files in result, combine them, and generate a single
    # pdf
    if args.plot:
        plot_synth()

args = read_flags()
main()

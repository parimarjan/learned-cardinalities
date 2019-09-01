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
EXP_TMP = '''python3 main.py --db_name imdb --template_dir templates/toml2 --losses \
qerr,join-loss --qopt_java_output 0 --cache_dir /data/pari/caches -n {n} \
--use_subqueries 1 --test 1 --test_size 0.2 --qopt_exh 1 \
--qopt_scan_cost_factor 0.2 --results_cache {results_cache} \
--qopt_get_sql 0 --algs {algs} --jl_variant {jl_variant} --jl_start_iter {jl_start_iter} \
--max_iter {max_iter} --eval_iter 5000 --loss_func {loss_func} \
--sampling {sampling} --nn_cache_dir {nn_cache_dir} \
'''

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_parallel", type=int, required=False,
            default=1)
    parser.add_argument("--results_cache", type=str, required=False,
            default="./results/")
    parser.add_argument("--algs", type=str, required=False,
            default="nn2")
    parser.add_argument("-n", "--num_samples_per_template", type=int,
            required=False, default=1000)
    parser.add_argument("--max_iter", type=int,
            required=False, default=100000)
    parser.add_argument("--jl_start_iter", type=int,
            required=False, default=200)
    parser.add_argument("--eval_iter", type=int,
            required=False, default=200)
    parser.add_argument("--lr", type=float,
            required=False, default=0.001)
    parser.add_argument("--clip_gradient", type=float,
            required=False, default=10.0)
    parser.add_argument("--jl_variant", type=int, required=False,
            default=0)
    parser.add_argument("--sampling", type=str, required=False,
            default="query")
    parser.add_argument("--loss_func", type=str, required=False,
            default="qloss")
    parser.add_argument("--nn_cache_dir", type=str, required=False,
            default="./nn_training_cache")

    return parser.parse_args()

def run_all():
    processes = []
    log_files = []
    commands = []
    cmd = EXP_TMP.format(algs = args.algs, n = args.num_samples_per_template,
                         max_iter = args.max_iter, jl_start_iter =
                         args.jl_start_iter, eval_iter = args.eval_iter,
                         lr = args.lr, clip_gradient = args.clip_gradient,
                         jl_variant = args.jl_variant, sampling =
                         args.sampling, loss_func = args.loss_func,
                         nn_cache_dir = args.nn_cache_dir, results_cache =
                         args.results_cache)

    for i in range(args.num_parallel):
        commands.append(cmd)

    num_processes = 0
    for i, cmd in enumerate(commands):
        num_processes += 1
        sp_log = open("./logs-run-all/" + str(i) + ".log", 'a')
        sp_log.write(cmd + "\n")
        log_files.append(sp_log)
        print("going to start another process")
        p = sp.Popen(cmd, stdout=sp_log, stderr=sp_log, shell=True)
        # p = sp.Popen(cmd, shell=True)
        processes.append(p)
        print(cmd)
        time.sleep(5)
        # FIXME: slowdown by 1 lagger process here.
        # print("going to wait for {} processes to finish".format(args.num_parallel))
        # while num_processes >= args.num_parallel:
            # time.sleep(60)
            # for pi, p in enumerate(processes):
                # if p is None:
                    # continue
                # if p.poll() == None:
                    # print("remaining processes: ", num_processes)
                    # num_processes -= 1
                # processes[pi] = None

    print("all processes started")
    for p in processes:
        p.wait()

    for f in log_files:
        f.close()

    print("finished executing all!")

def main():
    run_all()

args = read_flags()
main()

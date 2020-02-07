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
from matplotlib import gridspec
from matplotlib import pyplot as plt
from db_utils.utils import *
from db_utils.query_storage import *

LOSS_COLUMNS = ["loss_type", "loss", "summary_type", "template", "num_samples",
        "samples_type"]
EXP_COLUMNS = ["num_hidden_layers", "hidden_layer_size",
        "sampling_priority_alpha", "max_discrete_featurizing_buckets",
        "heuristic_features", "alg", "nn_type"]
PLOT_SUMMARY_TYPES = ["mean"]
ALGS_ORDER = ["mscn", "mscn-priority", "microsoft", "microsoft-priority"]

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--only_test", type=int, required=False,
            default=1)
    return parser.parse_args()

def load_qerrs(exp_dir):
    qerrs = load_object(exp_dir + "/qerr.pkl")
    if qerrs is None:
        assert False
    return qerrs

def load_jerrs(exp_dir):
    jerrs = load_object(exp_dir + "/jerr.pkl")
    if jerrs is None:
        print("jerr not found for: ", exp_dir)
        return None

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
        if name == "microsoft":
            return "fcnn"
        return name
    else:
        return exp_args["algs"]

def skip_exp(exp_args):
    if exp_args["sampling_priority_alpha"] > 2.00:
        return True
    if exp_args["max_discrete_featurizing_buckets"] > 10:
        return True

    return False

def get_all_qerrs():
    all_dfs = []
    fns = os.listdir(args.results_dir)
    for fn in fns:
        # convert to same format as qerrs
        cur_dir = args.results_dir + "/" + fn
        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            continue
        exp_args = vars(exp_args)
        if skip_exp(exp_args):
            continue

        try:
            qerrs = load_object(cur_dir + "/qerr.pkl")
        except:
            print("skipping ", cur_dir)
            continue
        if qerrs is None:
            continue

        exp_args["alg"] = get_alg_name(exp_args)
        for exp_column in EXP_COLUMNS:
            qerrs[exp_column] = exp_args[exp_column]

        if exp_args["sampling_priority_alpha"] == 2.0:
            qerrs["priority"] = "yes"
        else:
            qerrs["priority"] = "no"

        all_dfs.append(qerrs)

    df = pd.concat(all_dfs, ignore_index=True)
    if args.only_test:
        df = df[df["samples_type"] == "test"]
    return df

def get_all_plans(results_dir):
    all_dfs = []
    fns = os.listdir(results_dir)
    for fn in fns:
        # convert to same format as qerrs
        cur_dir = results_dir + "/" + fn
        exp_args = load_object(cur_dir + "/args.pkl")
        if exp_args is None:
            continue
        exp_args = vars(exp_args)
        if skip_exp(exp_args):
            continue
        alg = get_alg_name(exp_args)
        print(alg)
        nns = load_object(cur_dir + "/nn.pkl")
        qdf = pd.DataFrame(nns["query_stats"])
        # qdf["alg"] = alg
        # qdf["hls"] = exp_args["hidden_layer_size"]
        qdf["exp_name"] = fn
        qdf["alg"] = alg

        print(qdf)
        pdb.set_trace()
        all_dfs.append(qdf)

    return pd.concat(all_dfs)

def qkey_map():
    query_dir = "./our_dataset/queries/"
    qtmps = os.listdir(query_dir)
    mapping = {}
    for qtmp in qtmps:
        qfns = os.listdir(query_dir + qtmp)
        print(len(qfns))
        for fn in qfns:
            qfn = query_dir + qtmp + "/" + fn
            # qrep = load_sql_rep(qfn)
            with open(qfn, "rb") as f:
                qrep = pickle.load(f)
            mapping[str(deterministic_hash(qrep["sql"]))] = qfn
    return mapping

def main():

    qkey_mapping = qkey_map()
    plans = get_all_plans(args.results_dir)
    print(plans)
    pdb.set_trace()

if __name__ == "__main__":
    args = read_flags()
    main()

from cardinality_estimation.db import DB
from cardinality_estimation.cardinality_sample import CardinalitySample
from cardinality_estimation.query import Query
from cardinality_estimation.algs import *
from cardinality_estimation.losses import *
from cardinality_estimation.data_loader import *
import argparse
from park.param import parser
import psycopg2 as pg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras
from db_utils.utils import *
from utils.utils import *
from sklearn.model_selection import train_test_split
import glob
from collections import defaultdict
import pdb
import pandas as pd
import random
import itertools
import klepto
from multiprocessing import Pool
import numpy as np

def get_alg(alg):
    if alg == "independent":
        return Independent()
    elif alg == "postgres":
        return Postgres()
    elif alg == "random":
        return Random()
    elif alg == "chow":
        return BN(alg="chow-liu", num_bins=args.num_bins,
                        avg_factor=args.avg_factor,
                        gen_bn_dist=args.gen_bn_dist)
    elif alg == "bn-exact":
        return BN(alg="exact-dp", num_bins=args.num_bins)
    elif alg == "nn1":
        return NN1(max_iter = args.max_iter)
    elif alg == "ourpgm":
        return OurPGM()
    else:
        assert False

def remove_doubles(samples):
    new_samples = []
    seen_samples = set()
    for s in samples:
        if s.query in seen_samples:
            continue
        seen_samples.add(s.query)
        new_samples.append(s)
    return new_samples

def eval_alg(alg, losses, queries, use_subqueries):
    '''
    Applies alg to each query, and measures loss using `loss_func`.
    Records each estimate, and loss in the query object.
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    if use_subqueries:
        all_queries = get_all_subqueries(queries)
    else:
        all_queries = queries
    # first, just evaluate them all, and save results in queries
    start = time.time()
    yhats = alg.test(all_queries)
    eval_time = round(time.time() - start, 2)
    for i, q in enumerate(all_queries):
        q.yhats[alg.__str__()] = yhats[i]

    for loss_func in losses:
        loss_name = get_loss_name(loss_func.__name__)
        if "join" in loss_name:
            losses = loss_func(alg, queries, args.use_subqueries,
                    baseline=args.baseline_join_alg)
            assert len(losses) == len(queries)
            # only used with queries, since subqueries don't have an associated join-loss
            for i, q in enumerate(queries):
                q.losses[alg.__str__()][loss_name] = losses[i]
        else:
            losses = loss_func(alg, queries, args.use_subqueries,
                    baseline=args.baseline_join_alg)
            assert len(losses) == len(all_queries)
            for i, q in enumerate(all_queries):
                q.losses[alg.__str__()][loss_name] = losses[i]

        # TODO: set global printoptions to round digits
        print("case: {}: alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
                .format(args.db_name, alg, len(queries),
                    get_loss_name(loss_func.__name__),
                    # np.mean(losses),
                    # np.median(losses),
                    # np.percentile(losses,95),
                    # np.percentile(losses,99)))
                    np.round(np.mean(losses),3),
                    np.round(np.median(losses),3),
                    np.round(np.percentile(losses,95),3),
                    np.round(np.percentile(losses,99),3)))

def main():
    file_name = gen_results_name()
    print(file_name)
    if args.gen_synth_data:
        gen_synth_data(args)
    elif "osm" in args.db_name:
        load_osm_data(args)
    elif "dmv" in args.db_name:
        load_dmv_data(args)

    db = DB(args.user, args.pwd, args.db_host, args.port,
            args.db_name)
    print("started using db: ", args.db_name)
    query_templates = []
    samples = []
    if args.template_dir is None:
        update_synth_templates(args, query_templates)
    else:
        for fn in glob.glob(args.template_dir+"/*"):
            with open(fn, "r") as f:
                template = f.read()
                query_templates.append(template)

    # FIXME: all this should happen together, and be cached together.
    # Steps: gen templates, filter out zeros and dups, gen subqueries.
    for template in query_templates:
        samples += db.get_samples(template,
                num_samples=args.num_samples_per_template)

    print("len all samples: " , len(samples))
    if args.only_nonzero_samples:
        nonzero_samples = []
        for s in samples:
            if s.true_sel != 0.00:
                nonzero_samples.append(s)
        print("len nonzero samples: ", len(nonzero_samples))
        samples = nonzero_samples

    if args.use_subqueries:
        # TODO: parallelize the generation of subqueries
        for i, q in enumerate(samples):
            q.subqueries = db.gen_subqueries(q)
            if i % 10 == 0:
                print("{} subqueries generated for query {}".format(len(q.subqueries), i))

    samples = remove_doubles(samples)

    all_queries = samples
    if args.use_subqueries:
        all_queries = get_all_subqueries(samples)
    for q in all_queries:
        q.yhats = {}
        q.losses = defaultdict(dict)

    if args.test:
        train_queries, test_queries = train_test_split(samples, test_size=args.test_size,
                random_state=args.random_seed)
    else:
        train_queries = samples
        test_queries = []

    if len(train_queries) == 0:
        # debugging
        train_queries = test_queries

    result = defaultdict(list)

    algorithms = []
    losses = []
    for alg_name in args.algs.split(","):
        algorithms.append(get_alg(alg_name))
    for loss_name in args.losses.split(","):
        losses.append(get_loss(loss_name))

    print("going to run algorithms: ", args.algs)
    print("num train queries: ", len(train_queries))
    print("num test queries: ", len(test_queries))
    # this is deterministic, so just using it to store this in the saved data.
    # TODO: should not need this if initialized properly.

    for alg in algorithms:
        start = time.time()
        alg.train(db, train_queries, use_subqueries=args.use_subqueries)
        alg.save_model(save_dir=args.result_dir, suffix_name=gen_exp_hash()[0:3])
        train_time = round(time.time() - start, 2)
        print("{}, train-time: {}".format(alg, train_time))
        eval_alg(alg, losses, train_queries, args.use_subqueries)

        if args.test:
            eval_alg(alg, losses, test_queries, args.use_subqueries)

    file_name = gen_results_name() + "_train" + ".pickle"
    save_or_update(file_name, train_queries)
    if args.test:
        file_name = gen_results_name() + "_test" + ".pickle"
        save_or_update(file_name, test_queries)

    # save global stuff in results
    df = pd.DataFrame(result)
    file_name = gen_results_name() + ".pd"
    save_or_update(file_name, df)
    db.save_cache()

def gen_results_name():
    return args.result_dir + "/results" + gen_exp_hash()[0:3]

def gen_exp_hash():
    return str(deterministic_hash(str(args)))

def gen_samples_hash():
    string = ""
    sample_keys = ["db_name", "db_host", "user", "pwd", "template_dir",
            "gen_synth_data"]
    d = vars(args)
    for k in sample_keys:
        string += str(d[k])
    return deterministic_hash(string)

def read_flags():
    # parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, required=False,
            default="card_est")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--template_dir", type=str, required=False,
            default=None)
    parser.add_argument("--port", type=str, required=False,
            default=5432)
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/cards/")
    parser.add_argument("-n", "--num_samples_per_template", type=int,
            required=False, default=1000)
    parser.add_argument("--max_iter", type=int,
            required=False, default=100000)

    # synthetic data flags
    parser.add_argument("--gen_synth_data", type=int, required=False,
            default=0)
    parser.add_argument("--gen_bn_dist", type=int, required=False,
            default=0)
    parser.add_argument("--only_nonzero_samples", type=int, required=False,
            default=1)
    parser.add_argument("--use_subqueries", type=int, required=False,
            default=0)
    parser.add_argument("--synth_table", type=str, required=False,
            default="test")
    parser.add_argument("--synth_num_columns", type=int, required=False,
            default=2)
    parser.add_argument('--min_corr', help='delimited list correlations',
            type=float, required=False, default=0.0)
    parser.add_argument('--synth_period_len', help='delimited list correlations',
            type=int, required=False, default=10)
    parser.add_argument('--synth_num_vals', help='delimited list correlations',
            type=int, required=False, default=100000)
    parser.add_argument("--random_seed", type=int, required=False,
            default=2112)
    parser.add_argument("--test", type=int, required=False,
            default=1)
    parser.add_argument("--num_bins", type=int, required=False,
            default=10)
    parser.add_argument("--avg_factor", type=int, required=False,
            default=1)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    parser.add_argument("--losses", type=str, required=False,
            default="abs,rel,qerr", help="comma separated list of loss names")
    parser.add_argument("--result_dir", type=str, required=False,
            default="./results/")
    parser.add_argument("--baseline_join_alg", type=str, required=False,
            default="EXHAUSTIVE")

    return parser.parse_args()

args = read_flags()
main()

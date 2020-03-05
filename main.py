from cardinality_estimation.db import DB

from cardinality_estimation.query import *
from cardinality_estimation.algs import *
try:
    from cardinality_estimation.bayesian import *
except:
    pass
from cardinality_estimation.nn import *
# from cardinality_estimation.nn_old import *

from cardinality_estimation.losses import *
from cardinality_estimation.data_loader import *
import argparse
# from park.param import parser
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

import numpy as np
from db_utils.query_generator import QueryGenerator
from db_utils.query_generator2 import QueryGenerator2
import toml
from db_utils.query_storage import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sql_rep.query
import multiprocessing as mp

def get_alg(alg):
    if alg == "independent":
        return Independent()
    elif alg == "xgboost":
        return XGBoost()
    elif alg == "randomforest":
        return RandomForest()
    elif alg == "linear":
        return Linear()
    elif alg == "postgres":
        return Postgres()
    elif alg == "true":
        return TrueCardinalities()
    elif alg == "true_rank":
        return TrueRank()
    elif alg == "true_rank_tables":
        return TrueRankTables()
    elif alg == "random":
        return Random()
    elif alg == "chow":
        return BN(alg="chow-liu", num_bins=args.num_bins,
                        avg_factor=args.avg_factor,
                        gen_bn_dist=args.gen_bn_dist)
    elif alg == "bn-exact":
        return BN(alg="exact-dp", num_bins=args.num_bins)
    elif alg == "nn":
        return NN(max_epochs = args.max_epochs, lr=args.lr,
                train_card_key = args.train_card_key,
                exp_prefix = args.exp_prefix,
                load_query_together = args.load_query_together,
                result_dir = args.result_dir,
                priority_err_type = args.priority_err_type,
                priority_err_divide_len = args.priority_err_divide_len,
                tfboard = args.tfboard,
                jl_indexes = args.jl_indexes,
                normalization_type = args.normalization_type,
                preload_features = args.preload_features,
                reprioritize_epoch = args.reprioritize_epoch,
                prioritize_epoch = args.prioritize_epoch,
                heuristic_features = args.heuristic_features,
                debug_set = args.debug_set,
                num_hidden_layers=args.num_hidden_layers,
                hidden_layer_multiple=args.hidden_layer_multiple,
                    eval_epoch = args.eval_epoch,
                    optimizer_name=args.optimizer_name,
                    adaptive_lr=args.adaptive_lr,
                    # rel_qerr_loss=args.rel_qerr_loss,
                    clip_gradient=args.clip_gradient,
                    # baseline=args.baseline_join_alg,
                    # nn_results_dir = args.nn_results_dir,
                    loss_func = args.loss_func,
                    sampling_priority_type = args.sampling_priority_type,
                    sampling_priority_alpha = args.sampling_priority_alpha,
                    priority_query_len_scale = args.priority_query_len_scale,
                    net_name = args.net_name,
                    eval_epoch_jerr = args.eval_epoch_jerr,
                    # eval_num_tables = args.eval_num_tables,
                    jl_use_postgres = args.jl_use_postgres,
                    num_tables_feature = args.num_tables_feature,
                    max_discrete_featurizing_buckets =
                            args.max_discrete_featurizing_buckets,
                    nn_type = args.nn_type,
                    group_models = args.group_models,
                    adaptive_lr_patience = args.adaptive_lr_patience,
                    # single_threaded_nt = args.single_threaded_nt,
                    nn_weights_init_pg = args.nn_weights_init_pg,
                    avg_jl_priority = args.avg_jl_priority,
                    hidden_layer_size = args.hidden_layer_size)
    elif alg == "ourpgm":
        if args.db_name == "imdb":
            return OurPGMMultiTable(alg_name = args.pgm_alg_name, backend = args.pgm_backend,
                    use_svd=args.use_svd, num_singular_vals=args.num_singular_vals,
                    num_bins = args.num_bins, recompute = args.cl_recompute,
                    pgm_sampling_percentage=args.pgm_sampling_percentage,
                    merge_aliases = args.pgm_merge_aliases)
        else:
            return OurPGM(alg_name = args.pgm_alg_name, backend = args.pgm_backend,
                    use_svd=args.use_svd, num_singular_vals=args.num_singular_vals,
                    num_bins = args.num_bins, recompute = args.cl_recompute)
    elif alg == "sampling":
        return SamplingTables(args.sampling_key)
    else:
        assert False

def remove_doubles(query_strs):
    print("remove_doubles")
    newq = []
    seen_samples = set()
    for q in query_strs:
        if q in seen_samples:
            print(q)
            # pdb.set_trace()
            continue
        seen_samples.add(q)
        newq.append(q)
    return newq

def eval_alg(alg, losses, queries, samples_type, join_loss_pool):
    '''
    Applies alg to each query, and measures loss using `loss_func`.
    Records each estimate, and loss in the query object.
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # first, just evaluate them all, and save results in queries
    start = time.time()
    yhats = alg.test(queries)
    # assert isinstance(yhats[0], dict)
    eval_time = round(time.time() - start, 2)
    print("evaluating alg took: {} seconds".format(eval_time))

    loss_start = time.time()
    alg_name = alg.__str__()
    exp_name = alg.get_exp_name()
    for loss_func in losses:
        losses = loss_func(queries, yhats, name=alg_name,
                args=args, samples_type=samples_type, exp_name = exp_name,
                pool = join_loss_pool)

        # TODO: set global printoptions to round digits
        print("case: {}: alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
                .format(args.db_name, alg, len(queries),
                    get_loss_name(loss_func.__name__),
                    np.round(np.mean(losses),3),
                    np.round(np.median(losses),3),
                    np.round(np.percentile(losses,95),3),
                    np.round(np.percentile(losses,99),3)))

    print("loss computations took: {} seconds".format(time.time()-loss_start))

def main():
    # TODO: stop using klepto
    misc_cache = klepto.archives.dir_archive("./misc_cache",
            cached=True, serialized=True)
    db_key = deterministic_hash("db-" + args.query_templates)
    found_db = db_key in misc_cache.archive
    # found_db = False
    if found_db:
        db = misc_cache.archive[db_key]
    else:
        # either load the db object from cache, or regenerate it.
        db = DB(args.user, args.pwd, args.db_host, args.port,
                args.db_name)
    train_queries = []
    test_queries = []
    query_templates = args.query_templates.split(",")

    fns = list(glob.glob(args.query_directory + "/*"))
    for qdir in fns:
        template_name = os.path.basename(qdir)
        if args.query_templates != "all":
            if template_name not in query_templates:
                print("skipping template ", template_name)
                continue

        # if "7a" in qdir:
            # print("skipping query 7a")
            # continue

        start = time.time()
        # loading, or generating samples
        samples = []
        qfns = list(glob.glob(qdir+"/*.pkl"))
        qfns.sort()
        if args.num_samples_per_template == -1:
            qfns = qfns
        elif args.num_samples_per_template == -2:
            assert False
            num_samples = get_template_samples(fn)
            qfns = qfns[0:num_samples]
        elif args.num_samples_per_template < len(qfns):
            qfns = qfns[0:args.num_samples_per_template]
        else:
            print("queries should be generated using appropriate script")
            assert False

        if args.debug_set:
            random.seed(args.random_seed)
            qfns = random.sample(qfns, int(len(qfns) / 10))

        if args.algs == "sampling":
            skey = args.sampling_type + str(args.sampling_percentage) + "_actual"

        skipped = 0
        for qfn in qfns:
            qrep = load_sql_rep(qfn)
            zero_query = False
            for _,info in qrep["subset_graph"].nodes().items():
                if args.train_card_key not in info["cardinality"]:
                    zero_query = True
                    break

                if "actual" not in info["cardinality"]:
                    zero_query = True
                    break

                elif info["cardinality"]["actual"] == 0:
                    zero_query = True
                    break

                if args.sampling_key is not None:
                    if args.sampling_key not in info["cardinality"]:
                        zero_query = True
                        break

            if zero_query:
                skipped += 1
                continue

            qrep["name"] = qfn
            qrep["template_name"] = template_name
            samples.append(qrep)

        if len(samples) == 0:
            continue

        print(("template: {}, zeros skipped: {}, subqueries: {}, queries: {}"
                ", loading time: {}").format( template_name, skipped,
                    len(samples[0]["subset_graph"].nodes()), len(samples),
                    time.time()-start))

        if not found_db:
            for sample in samples:
                # not all samples may share all predicates etc. so updating
                # them all. stats will not be recomputed for repeated columns
                # FIXME:
                # db.update_db_stats(convert_sql_rep_to_query_rep(sample))
                db.update_db_stats(sample)

        if args.test:
            cur_train_queries, cur_test_queries = train_test_split(samples,
                    test_size=args.test_size, random_state=args.random_seed)
        else:
            cur_train_queries = samples
            cur_test_queries = []

        train_queries += cur_train_queries
        test_queries += cur_test_queries

    # shuffle train, test queries so join loss computation can be parallelized
    # better: otherwise all queries from templates that take a long time would
    # go to same worker
    random.shuffle(train_queries)
    random.shuffle(test_queries)

    if not found_db:
        misc_cache.archive[db_key] = db

    if len(train_queries) == 0:
        # debugging, so doesn't crash
        train_queries = test_queries

    algorithms = []
    losses = []
    for alg_name in args.algs.split(","):
        algorithms.append(get_alg(alg_name))
    for loss_name in args.losses.split(","):
        losses.append(get_loss(loss_name))

    print("algs: {}, train queries: {}, test queries: {}".format(\
            args.algs, len(train_queries), len(test_queries)))

    train_times = {}
    eval_times = {}

    if "join-loss" in args.losses or \
            (args.sampling_priority_alpha > 0 and "nn" in args.algs):
        num_processes = int(mp.cpu_count())
        join_loss_pool = mp.Pool(num_processes)
    else:
        join_loss_pool = None

    for alg in algorithms:
        start = time.time()
        if args.eval_test_while_training:
            alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                    test_samples=test_queries, join_loss_pool=join_loss_pool)
        else:
            alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                    test_samples=None, join_loss_pool=join_loss_pool)

        train_times[alg.__str__()] = round(time.time() - start, 2)

        start = time.time()
        eval_alg(alg, losses, train_queries, "train", join_loss_pool)

        if args.test:
            eval_alg(alg, losses, test_queries, "test", join_loss_pool)

        eval_times[alg.__str__()] = round(time.time() - start, 2)

    if join_loss_pool is not None:
        join_loss_pool.close()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_directory", type=str, required=False,
            default="./our_dataset/queries")
    parser.add_argument("--exp_prefix", type=str, required=False,
            default="")
    parser.add_argument("--query_templates", type=str, required=False,
            default="all")
    parser.add_argument("--debug_set", type=int, required=False,
            default=0)
    parser.add_argument("--preload_features", type=int, required=False,
            default=1)
    parser.add_argument("--load_query_together", type=int, required=False,
            default=0)
    parser.add_argument("--normalization_type", type=str, required=False,
            default="pg_total_selectivity")

    parser.add_argument("--num_tables_model", type=str, required=False,
            default="nn")
    parser.add_argument("--nn_weights_init_pg", type=int, required=False,
            default=0)
    parser.add_argument("--single_threaded_nt", type=int, required=False,
            default=0)
    parser.add_argument("--reuse_env", type=int, required=False,
            default=1)
    parser.add_argument("--num_tables_feature", type=int, required=False,
            default=1)
    parser.add_argument("--max_discrete_featurizing_buckets", type=int, required=False,
            default=10)
    parser.add_argument("--heuristic_features", type=int, required=False,
            default=1)
    parser.add_argument("--group_models", type=int, required=False,
            default=0)
    parser.add_argument("--priority_err_divide_len", type=int, required=False,
            default=0)
    # parser.add_argument("--update_subq_cards", type=int, required=False,
            # default=0)
    # parser.add_argument("--update_subq_preds", type=int, required=False,
            # default=0)
    parser.add_argument("--eval_num_tables", type=int, required=False,
            default=0)
    parser.add_argument("--rf_trees", type=int, required=False,
            default=128)
    parser.add_argument("--exp_name", type=str, required=False,
            default="card_exp")

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
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
            required=False, default=-1)
    parser.add_argument("--max_epochs", type=int,
            required=False, default=20)
    parser.add_argument("--eval_epoch", type=int,
            required=False, default=1)
    parser.add_argument("--eval_epoch_jerr", type=int,
            required=False, default=1)
    parser.add_argument("--lr", type=float,
            required=False, default=0.001)
    parser.add_argument("--clip_gradient", type=float,
            required=False, default=10.0)
    parser.add_argument("--rel_qerr_loss", type=int,
            required=False, default=0)
    parser.add_argument("--rel_jloss", type=int,
            required=False, default=0)
    parser.add_argument("--eval_test_while_training", type=int,
            required=False, default=1)
    parser.add_argument("--jl_use_postgres", type=int,
            required=False, default=1)
    parser.add_argument("--nn_type", type=str,
            required=False, default="mscn")

    parser.add_argument("--priority_err_type", type=str, required=False,
            default = "jerr", help="jerr or jratio")
    parser.add_argument("--avg_jl_priority", type=int, required=False,
            default=1)
    parser.add_argument("--jl_indexes", type=int, required=False,
            default=1)
    parser.add_argument("--adaptive_lr", type=int,
            required=False, default=0)
    parser.add_argument("--adaptive_lr_patience", type=int,
            required=False, default=20)

    parser.add_argument("--viz_join_plans", type=int,
            required=False, default=0)
    parser.add_argument("--viz_fn", type=str,
            required=False, default="./test")

    # parser.add_argument("--nn_results_dir", type=str, required=False,
            # default="./nn_results")

    parser.add_argument("--optimizer_name", type=str, required=False,
            default="adam")
    parser.add_argument("--net_name", type=str, required=False,
            default="FCNN")

    parser.add_argument("--num_hidden_layers", type=int,
            required=False, default=1)
    parser.add_argument("--hidden_layer_multiple", type=float,
            required=False, default=None)
    parser.add_argument("--hidden_layer_size", type=int,
            required=False, default=100)

    # synthetic data flags
    parser.add_argument("--gen_synth_data", type=int, required=False,
            default=0)
    parser.add_argument("--gen_bn_dist", type=int, required=False,
            default=0)
    parser.add_argument("--compute_runtime", type=int, required=False,
            default=0)
    parser.add_argument("--only_nonzero_samples", type=int, required=False,
            default=1)
    parser.add_argument("--use_subqueries", type=int, required=False,
            default=1)
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
            default=0)
    parser.add_argument("--avg_factor", type=int, required=False,
            default=1)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    parser.add_argument("--losses", type=str, required=False,
            default="qerr,join-loss", help="comma separated list of loss names")
    parser.add_argument("--result_dir", type=str, required=False,
            default="./new_results/")
    parser.add_argument("--baseline_join_alg", type=str, required=False,
            default="EXHAUSTIVE")
    parser.add_argument("--db_file_name", type=str, required=False,
            default=None)
    parser.add_argument("--cache_dir", type=str, required=False,
            default="./caches/")
    parser.add_argument("--save_cur_cache_dir", type=str, required=False,
            default=None)
    parser.add_argument("--execution_cache_threshold", type=int, required=False,
            default=20)
    parser.add_argument("--jl_variant", type=int, required=False,
            default=0)
    parser.add_argument("--sampling_key", type=str, required=False,
            default=None, help="")
    parser.add_argument("--train_card_key", type=str, required=False,
            default="actual", help="")

    parser.add_argument("--sampling_priority_type", type=str, required=False,
            default="query", help="")
    parser.add_argument("--sampling_priority_alpha", type=float, required=False,
            default=0.00, help="")
    parser.add_argument("--prioritize_epoch", type=float, required=False,
            default=2, help="")
    parser.add_argument("--reprioritize_epoch", type=int, required=False,
            default=2, help="")

    parser.add_argument("--priority_query_len_scale", type=float, required=False,
            default=0, help="")
    parser.add_argument("--tfboard", type=float, required=False,
            default=1, help="")

    parser.add_argument("--loss_func", type=str, required=False,
            default="qloss")

    ## pgm flags
    parser.add_argument("--pgm_backend", type=str, required=False,
            default="ourpgm")
    parser.add_argument("--pgm_alg_name", type=str, required=False,
            default="chow-liu")
    parser.add_argument("--sampling_percentage", type=int, required=False,
            default=10)
    parser.add_argument("--sampling_type", type=str, required=False,
            default="ss")

    parser.add_argument("--use_svd", type=int, required=False,
            default=0)
    parser.add_argument("--num_singular_vals", type=int, required=False,
            default=5, help="-1 means all")
    parser.add_argument("--num_bins", type=int, required=False,
            default=100)
    parser.add_argument("--cl_recompute", type=int, required=False,
            default=0)
    parser.add_argument("--pgm_sampling_percentage", type=float, required=False,
            default=0.001)
    parser.add_argument("--pgm_merge_aliases", type=int, required=False,
            default=0)

    return parser.parse_args()

# need __name__ == "__main__" for torch multithreading haha
if __name__ == "__main__":
    args = read_flags()
    main()

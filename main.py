from cardinality_estimation.db import DB

from cardinality_estimation.query import *
from cardinality_estimation.algs import *
from cardinality_estimation.bayesian import *
from cardinality_estimation.nn import *

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
import multiprocessing
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
    elif alg == "random":
        return Random()
    elif alg == "chow":
        return BN(alg="chow-liu", num_bins=args.num_bins,
                        avg_factor=args.avg_factor,
                        gen_bn_dist=args.gen_bn_dist)
    elif alg == "bn-exact":
        return BN(alg="exact-dp", num_bins=args.num_bins)
    elif alg == "nn":
        return NN(max_iter = args.max_iter, lr=args.lr,
                num_hidden_layers=args.num_hidden_layers,
                hidden_layer_multiple=args.hidden_layer_multiple,
                    jl_start_iter=args.jl_start_iter,
                    eval_iter = args.eval_iter,
                    optimizer_name=args.optimizer_name,
                    adaptive_lr=args.adaptive_lr,
                    rel_qerr_loss=args.rel_qerr_loss,
                    clip_gradient=args.clip_gradient,
                    baseline=args.baseline_join_alg,
                    nn_cache_dir = args.nn_cache_dir,
                    loss_func = args.loss_func,
                    sampling=args.sampling,
                    sampling_priority_alpha = args.sampling_priority_alpha,
                    net_name = args.net_name,
                    reuse_env = args.reuse_env,
                    eval_iter_jl = args.eval_iter_jl,
                    eval_num_tables = args.eval_num_tables,
                    jl_use_postgres = args.jl_use_postgres,
                    num_tables_feature = args.num_tables_feature,
                    max_discrete_featurizing_buckets = args.max_discrete_featurizing_buckets)
    elif alg == "nn2":
        assert False
        # return NN2(max_iter = args.max_iter, jl_variant=args.jl_variant, lr=args.lr,
                # num_hidden_layers=args.num_hidden_layers,
                # hidden_layer_multiple=args.hidden_layer_multiple,
                    # jl_start_iter=args.jl_start_iter, eval_iter =
                    # args.eval_iter, optimizer_name=args.optimizer_name,
                    # adaptive_lr=args.adaptive_lr,
                    # rel_qerr_loss=args.rel_qerr_loss,
                    # clip_gradient=args.clip_gradient,
                    # baseline=args.baseline_join_alg,
                    # nn_cache_dir = args.nn_cache_dir,
                    # divide_mb_len = args.divide_mb_len,
                    # rel_jloss=args.rel_jloss,
                    # loss_func = args.loss_func,
                    # sampling=args.sampling,
                    # sampling_priority_method=args.sampling_priority_method,
                    # sampling_priority_alpha = args.sampling_priority_alpha,
                    # adaptive_priority_alpha = args.adaptive_priority_alpha,
                    # net_name = args.net_name,
                    # reuse_env = args.reuse_env,
                    # eval_iter_jl = args.eval_iter_jl,
                    # eval_num_tables = args.eval_num_tables,
                    # jl_use_postgres = args.jl_use_postgres)
    elif alg == "nn3":
        assert False
        # return NumTablesNN(max_iter = args.max_iter, jl_variant=args.jl_variant, lr=args.lr,
                # num_hidden_layers=args.num_hidden_layers,
                # hidden_layer_multiple=args.hidden_layer_multiple,
                    # jl_start_iter=args.jl_start_iter, eval_iter =
                    # args.eval_iter, optimizer_name=args.optimizer_name,
                    # adaptive_lr=args.adaptive_lr,
                    # rel_qerr_loss=args.rel_qerr_loss,
                    # clip_gradient=args.clip_gradient,
                    # baseline=args.baseline_join_alg,
                    # nn_cache_dir = args.nn_cache_dir,
                    # divide_mb_len = args.divide_mb_len,
                    # rel_jloss=args.rel_jloss,
                    # loss_func = args.loss_func,
                    # sampling=args.sampling,
                    # sampling_priority_method=args.sampling_priority_method,
                    # sampling_priority_alpha = args.sampling_priority_alpha,
                    # adaptive_priority_alpha = args.adaptive_priority_alpha,
                    # net_name = args.net_name,
                    # eval_iter_jl = args.eval_iter_jl,
                    # num_tables_model = args.num_tables_model,
                    # num_trees = args.rf_trees,
                    # reuse_env = args.reuse_env,
                    # eval_num_tables = args.eval_num_tables,
                    # group_models = args.group_models,
                    # jl_use_postgres = args.jl_use_postgres)
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
    print("evaluating alg took: {} seconds".format(eval_time))

    loss_start = time.time()
    for loss_func in losses:
        loss_name = get_loss_name(loss_func.__name__)
        if "join" in loss_name:
            if args.viz_join_plans:
                pdf_fn = args.viz_fn + os.path.basename(args.template_dir) \
                            + alg.__str__() + ".pdf"
                print("writing out join plan visualizations to ", pdf_fn)
                join_viz_pdf = PdfPages(pdf_fn)
            else:
                join_viz_pdf = None

            losses = loss_func(alg, queries, args.use_subqueries,
                    baseline=args.baseline_join_alg,
                    compute_runtime=args.compute_runtime,
                    use_postgres = args.jl_use_postgres,
                    pdf = join_viz_pdf)
            if join_viz_pdf:
                join_viz_pdf.close()

            assert len(losses) == len(queries)
        else:
            losses = loss_func(alg, queries, args.use_subqueries,
                    baseline=args.baseline_join_alg)
            assert len(losses) == len(all_queries)

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
    # TODO: separate out 1-table stuff
    if args.gen_synth_data:
        gen_synth_data(args)
    elif "osm" in args.db_name:
        load_osm_data(args)
    elif "dmv" in args.db_name:
        load_dmv_data(args)

    misc_cache = klepto.archives.dir_archive("./misc_cache",
            cached=True, serialized=True)
    db_key = deterministic_hash("db-" + args.template_dir)
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

    fns = list(glob.glob(args.template_dir+"/*"))
    for fn in fns:
        start = time.time()
        # loading, or generating samples
        samples = []
        qdir = args.query_directory + "/" + os.path.basename(fn)
        qfns = list(glob.glob(qdir+"/*"))
        if args.num_samples_per_template == -2:
            qfns = qfns
        elif args.num_samples_per_template == -1:
            num_samples = get_template_samples(fn)
            qfns = qfns[0:num_samples]
        elif args.num_samples_per_template < len(qfns):
            qfns = qfns[0:args.num_samples_per_template]
        else:
            print("queries should be generated using appropriate script")
            assert False

        for qfn in qfns:
            qrep = load_sql_rep(qfn)
            # FIXME: don't want to hardcode title here
            if "total" not in qrep["subset_graph"].nodes()[tuple("t")]["cardinality"]:
                # things to update: total, pred_cols etc.
                update_qrep(qrep, samples[0])
                # json-ify the graphs
                output = {}
                output["sql"] = qrep["sql"]
                output["join_graph"] = nx.adjacency_data(qrep["join_graph"])
                output["subset_graph"] = nx.adjacency_data(qrep["subset_graph"])
                # save it out to qfn
                with open(qfn, 'wb') as fp:
                    pickle.dump(output, fp, protocol=pickle.HIGHEST_PROTOCOL)
            samples.append(qrep)

        print("{} took {} seconds to load data".format(fn, time.time()-start))

        if not found_db:
            for sample in samples:
                # not all samples may share all predicates etc. so updating
                # them all. stats will not be recomputed for repeated columns
                ## FIXME:
                db.update_db_stats(convert_sql_rep_to_query_rep(sample))

        if args.test:
            cur_train_queries, cur_test_queries = train_test_split(samples,
                    test_size=args.test_size, random_state=args.random_seed)
        else:
            cur_train_queries = samples
            cur_test_queries = []

        train_queries += cur_train_queries
        test_queries += cur_test_queries

    print("train queries: {}, test queries: {}".format(len(train_queries),
        len(test_queries)))
    if not found_db:
        misc_cache.archive[db_key] = db

    if len(train_queries) == 0:
        # debugging, so doesn't crash
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

    train_times = {}
    eval_times = {}

    for alg in algorithms:
        start = time.time()
        if args.eval_test_while_training:
            alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                    test_samples=test_queries)
        else:
            alg.train(db, train_queries, use_subqueries=args.use_subqueries)

        train_times[alg.__str__()] = round(time.time() - start, 2)

        start = time.time()
        eval_alg(alg, losses, train_queries, args.use_subqueries)

        if args.test:
            eval_alg(alg, losses, test_queries, args.use_subqueries)
        eval_times[alg.__str__()] = round(time.time() - start, 2)

    if args.results_cache:
        results = {}
        results["training_queries"] = train_queries
        results["test_queries"] = test_queries
        results["args"] = args
        results["train_times"] = train_times
        results["eval_times"] = eval_times

        results_cache = klepto.archives.dir_archive(args.results_cache)
        dt = datetime.datetime.now()
        exp_name = args.exp_name + "-{}-{}-{}-{}".format(dt.day, dt.hour, dt.minute, dt.second)
        results_cache.archive[exp_name] = results

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
    parser.add_argument("--results_cache", type=str, required=False,
            default=None)
    parser.add_argument("--query_directory", type=str, required=False,
            default="./queries")
    parser.add_argument("--num_tables_model", type=str, required=False,
            default="nn")
    parser.add_argument("--reuse_env", type=int, required=False,
            default=1)
    parser.add_argument("--num_tables_feature", type=int, required=False,
            default=1)
    parser.add_argument("--max_discrete_featurizing_buckets", type=int, required=False,
            default=10)
    parser.add_argument("--group_models", type=int, required=False,
            default=0)
    parser.add_argument("--gen_queries", type=int, required=False,
            default=0)
    parser.add_argument("--update_subq_cards", type=int, required=False,
            default=0)
    parser.add_argument("--update_subq_preds", type=int, required=False,
            default=0)
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
            required=False, default=1000)
    parser.add_argument("--max_iter", type=int,
            required=False, default=100000)
    parser.add_argument("--jl_start_iter", type=int,
            required=False, default=200)
    parser.add_argument("--eval_iter", type=int,
            required=False, default=1000)
    parser.add_argument("--eval_iter_jl", type=int,
            required=False, default=5000)
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
            required=False, default=0)

    parser.add_argument("--adaptive_lr", type=int,
            required=False, default=1)
    parser.add_argument("--viz_join_plans", type=int,
            required=False, default=0)
    parser.add_argument("--viz_fn", type=str,
            required=False, default="./test")

    parser.add_argument("--nn_cache_dir", type=str, required=False,
            default="./nn_training_cache")
    parser.add_argument("--divide_mb_len", type=int, required=False,
            default=0)

    parser.add_argument("--optimizer_name", type=str, required=False,
            default="sgd")
    parser.add_argument("--net_name", type=str, required=False,
            default="FCNN")

    parser.add_argument("--num_hidden_layers", type=int,
            required=False, default=1)
    parser.add_argument("--hidden_layer_multiple", type=float,
            required=False, default=0.5)

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
            default=1)
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

    parser.add_argument("--sampling", type=str, required=False,
            default="subquery", help="weighted_query: reprioritize, subquery: uniform \
            over all queries")

    parser.add_argument("--sampling_priority_alpha", type=float, required=False,
            default=0.00, help="")

    parser.add_argument("--loss_func", type=str, required=False,
            default="qloss")

    ## pgm flags
    parser.add_argument("--pgm_backend", type=str, required=False,
            default="ourpgm")
    parser.add_argument("--pgm_alg_name", type=str, required=False,
            default="chow-liu")
    parser.add_argument("--sampling_percentage", type=float, required=False,
            default=0.001)
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

args = read_flags()
main()

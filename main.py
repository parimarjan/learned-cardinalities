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
import multiprocessing
import numpy as np
from db_utils.query_generator import QueryGenerator
from db_utils.query_generator2 import QueryGenerator2
import toml

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
        return NN1(max_iter = args.max_iter, lr=args.lr,
                num_hidden_layers=args.num_hidden_layers,
                hidden_layer_multiple=args.hidden_layer_multiple,
                eval_iter=args.eval_iter)
    elif alg == "nn2":
        return NN2(max_iter = args.max_iter, jl_variant=args.jl_variant, lr=args.lr,
                num_hidden_layers=args.num_hidden_layers,
                hidden_layer_multiple=args.hidden_layer_multiple,
                    jl_start_iter=args.jl_start_iter, eval_iter =
                    args.eval_iter, optimizer_name=args.optimizer_name,
                    adaptive_lr=args.adaptive_lr,
                    rel_qerr_loss=args.rel_qerr_loss,
                    clip_gradient=args.clip_gradient,
                    baseline=args.baseline_join_alg,
                    nn_cache_dir = args.nn_cache_dir,
                    divide_mb_len = args.divide_mb_len)
    elif alg == "ourpgm":
        return OurPGM(alg_name = args.pgm_alg_name, backend = args.pgm_backend,
                use_svd=args.use_svd, num_singular_vals=args.num_singular_vals)
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
    start = time.time()
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
                    np.round(np.mean(losses),3),
                    np.round(np.median(losses),3),
                    np.round(np.percentile(losses,95),3),
                    np.round(np.percentile(losses,99),3)))

    # eval_time = time.time() - start

    # FIXME: separate out global stats?
    # for q in queries:
        # q.eval_time[alg.__str__()] = eval_time
    print("evaluating alg took: {} seconds".format(eval_time))

def gen_query_strs(args, query_template, num_samples, sql_str_cache):
    '''
    @query_template: str OR dict.

    @ret: [Query, Query, ...]
    '''
    query_strs = []

    # TODO: change key to be based on file name?
    hashed_tmp = deterministic_hash(query_template)

    if hashed_tmp in sql_str_cache:
        query_strs = sql_str_cache[hashed_tmp]
        print("loaded {} query strings".format(len(query_strs)))

    if num_samples == -1:
        # select whatever we loaded
        query_strs = query_strs
    elif len(query_strs) > num_samples:
        query_strs = query_strs[0:num_samples]
    elif len(query_strs) < num_samples:
        # need to generate additional queries
        req_samples = num_samples - len(query_strs)
        if isinstance(query_template, dict):
            qg = QueryGenerator2(query_template, args.user, args.db_host, args.port,
                    args.pwd, args.db_name)
        elif isinstance(query_template, str):
            qg = QueryGenerator(query_template, args.user, args.db_host, args.port,
                    args.pwd, args.db_name)

        gen_sqls = qg.gen_queries(req_samples)
        query_strs += gen_sqls
        # save on the disk
        sql_str_cache.archive[hashed_tmp] = query_strs
    print("returning {} query strs".format(len(query_strs)))
    return query_strs

def gen_query_objs(args, query_strs, query_obj_cache):
    '''
    TODO: explain
    '''
    ret_queries = []
    unknown_query_strs = []

    # everything below this part is for query objects exclusively
    for sql in query_strs:
        hsql = deterministic_hash(sql)
        if hsql in query_obj_cache:
            ret_queries.append(query_obj_cache[hsql])
        else:
            unknown_query_strs.append(sql)

    # print("loaded {} query objects".format(len(ret_queries)))
    if len(unknown_query_strs) == 0:
        return ret_queries
    else:
        print("need to generate {} query objects".\
                format(len(unknown_query_strs)))

    sql_result_cache = args.cache_dir + "/sql_result"
    all_query_objs = []
    start = time.time()
    num_processes = int(min(len(unknown_query_strs),
        multiprocessing.cpu_count()))
    with Pool(processes=num_processes) as pool:
        args = [(cur_query, args.user, args.db_host, args.port,
            args.pwd, args.db_name, None,
            args.execution_cache_threshold, sql_result_cache) for
            cur_query in unknown_query_strs]
        all_query_objs = pool.starmap(sql_to_query_object, args)

    for i, q in enumerate(all_query_objs):
        ret_queries.append(q)
        hsql = deterministic_hash(unknown_query_strs[i])
        # save in memory, so potential repeat queries can be found in the
        # memory cache
        query_obj_cache[hsql] = q
        # save at the disk backend as well, without needing to dump all of
        # the cache
        query_obj_cache.archive[hsql] = q

    print("generated {} samples in {} secs".format(len(ret_queries),
        time.time()-start))
    return ret_queries

def main():
    if args.gen_synth_data:
        gen_synth_data(args)
    elif "osm" in args.db_name:
        load_osm_data(args)
    elif "dmv" in args.db_name:
        load_csv_data(args)
    elif "higgs" in args.db_name:
        load_csv_data(args)
    elif "power" in args.db_name:
        load_csv_data(args, 0, ";")

    # Steps: collect statistics, gen templates, filter out zeros and dups, gen
    # subqueries.
    query_templates = []
    if args.template_dir is None:
        update_synth_templates(args, query_templates)
    else:
        fns = list(glob.glob(args.template_dir+"/*"))
        for fn in fns:
            if ".sql" in fn:
                with open(fn, "r") as f:
                    template = f.read()
            elif ".toml" in fn:
                template = toml.load(fn)
            else:
                assert False
            query_templates.append(template)

    start = time.time()
    misc_cache = klepto.archives.dir_archive("./misc_cache",
            cached=True, serialized=True)
    db_key = deterministic_hash("db-" + str(args.template_dir))
    if args.template_dir is not None and db_key in misc_cache.archive:
    # if False:
        db = misc_cache.archive[db_key]
    else:
        # either load the db object from cache, or regenerate it.
        db = DB(args.user, args.pwd, args.db_host, args.port,
                args.db_name)
        for template in query_templates:
            print(template)
            if isinstance(template, dict):
                db.update_db_stats(template["base_sql"]["sql"])
            else:
                db.update_db_stats(template)
        misc_cache.archive[db_key] = db

    print("generating db object took {} seconds".format(\
            time.time() - start))

    # TODO: not sure if loading it into memory is a good idea or not.
    samples = []
    query_obj_cache = klepto.archives.dir_archive(args.cache_dir + "/query_obj",
            cached=True, serialized=True)
    query_obj_cache.load()
    sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/sql_str",
            cached=True, serialized=True)
    sql_str_cache.load()

    for i, template in enumerate(query_templates):
        # generate queries
        query_strs = gen_query_strs(args, template,
                args.num_samples_per_template, sql_str_cache)
        cur_samples = gen_query_objs(args, query_strs, query_obj_cache)
        for sample_id, q in enumerate(cur_samples):
            # if sample_id in [12]:
                # continue

            # q.template_sql = template
            if args.template_dir is None:
                q.template_name = "synth" + str(sample_id)
            else:
                q.template_name = os.path.basename(fns[i]) + str(sample_id)
            samples.append(q)

    # TODO: clear / dump the query_obj cache
    print("len all samples: " , len(samples))

    query_obj_cache.clear()
    sql_str_cache.clear()

    if args.only_nonzero_samples:
        nonzero_samples = []
        for s in samples:
            if s.true_sel != 0.00:
                nonzero_samples.append(s)
        print("len nonzero samples: ", len(nonzero_samples))
        samples = nonzero_samples

    if args.use_subqueries:
        start = time.time()
        sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_sql_str",
                cached=True, serialized=True)
        sql_str_cache.load()
        query_obj_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_query_obj",
                cached=True, serialized=True)
        query_obj_cache.load()

        # TODO: parallelize the generation of subqueries
        for i, q in enumerate(samples):
            hashed_key = deterministic_hash(q.query)
            if hashed_key in sql_str_cache:
            # if False:
                sql_subqueries = sql_str_cache[hashed_key]
            else:
                print("going to generate subqueries for query num ", i)
                sql_subqueries = gen_all_subqueries(q.query)
                # pdb.set_trace()
                # save it for the future!
                sql_str_cache.archive[hashed_key] = sql_subqueries

            loaded_queries = gen_query_objs(args, sql_subqueries, query_obj_cache)
            q.subqueries = loaded_queries

        print("subquery generation took {} seconds".format(time.time()-start))


    samples = remove_doubles(samples)
    all_queries = samples
    print("after removing doubles, len: ", len(samples))

    # FIXME: temporary, and slightly ugly hack -- need to initialize few fields
    # in all of the queries
    if args.use_subqueries:
        all_queries = get_all_subqueries(samples)

    for q in all_queries:
        q.yhats = {}
        q.losses = defaultdict(dict)
        q.eval_time = {}
        q.train_time = {}

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

    train_times = {}
    eval_times = {}
    num_params = {}

    for alg in algorithms:
        start = time.time()
        alg.train(db, train_queries, use_subqueries=args.use_subqueries)
        train_times[alg.__str__()] = round(time.time() - start, 2)

        start = time.time()
        eval_alg(alg, losses, train_queries, args.use_subqueries)

        if args.test:
            eval_alg(alg, losses, test_queries, args.use_subqueries)
        eval_times[alg.__str__()] = round(time.time() - start, 2)
        num_params[alg.__str__()] = alg.num_parameters()

    results = {}
    results["training_queries"] = train_queries
    results["test_queries"] = test_queries
    results["args"] = args
    results["train_times"] = train_times
    results["eval_times"] = eval_times
    results["num_params"] = num_params

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
            default="./results")
    parser.add_argument("--exp_name", type=str, required=False,
            default="card_exp")
    parser.add_argument("--pgm_backend", type=str, required=False,
            default="ourpgm")
    parser.add_argument("--pgm_alg_name", type=str, required=False,
            default="chow-liu")

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
            required=False, default=5000)
    parser.add_argument("--jl_start_iter", type=int,
            required=False, default=200)
    parser.add_argument("--eval_iter", type=int,
            required=False, default=200)
    parser.add_argument("--lr", type=float,
            required=False, default=0.001)
    parser.add_argument("--clip_gradient", type=float,
            required=False, default=10.0)
    parser.add_argument("--rel_qerr_loss", type=int,
            required=False, default=1)
    parser.add_argument("--adaptive_lr", type=int,
            required=False, default=1)
    parser.add_argument("--nn_cache_dir", type=str, required=False,
            default="./nn_training_cache")
    parser.add_argument("--divide_mb_len", type=int, required=False,
            default=1)
    parser.add_argument("--use_svd", type=int, required=False,
            default=0)
    parser.add_argument("--num_singular_vals", type=int, required=False,
            default=-1, help="-1 means all")

    parser.add_argument("--optimizer_name", type=str, required=False,
            default="ams")

    parser.add_argument("--num_hidden_layers", type=int,
            required=False, default=1)
    parser.add_argument("--hidden_layer_multiple", type=float,
            required=False, default=0.5)

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
            default="LEFT_DEEP")
    parser.add_argument("--db_file_name", type=str, required=False,
            default=None)
    parser.add_argument("--cache_dir", type=str, required=False,
            default="./pgm_caches/")
    parser.add_argument("--execution_cache_threshold", type=int, required=False,
            default=20)
    parser.add_argument("--jl_variant", type=int, required=False,
            default=0)

    return parser.parse_args()

args = read_flags()
main()

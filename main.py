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
from db_utils.query_generator import QueryGenerator

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
                    # np.mean(losses),
                    # np.median(losses),
                    # np.percentile(losses,95),
                    # np.percentile(losses,99)))
                    np.round(np.mean(losses),3),
                    np.round(np.median(losses),3),
                    np.round(np.percentile(losses,95),3),
                    np.round(np.percentile(losses,99),3)))

    print("evaluating alg took: {} seconds".format(time.time()-start))

def gen_query_strs(args, query_template, num_samples):
    '''
    @ret: [Query, Query, ...]
    '''
    # first, generate the query strings / or find it in the cache
    sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/sql_str",
            cached=True, serialized=True)

    query_strs = []

    # TODO: change key to be based on file name?
    hashed_tmp = deterministic_hash(query_template)
    if hashed_tmp in sql_str_cache.archive:
        query_strs = sql_str_cache.archive[hashed_tmp]
        print("loaded {} query strings".format(len(query_strs)))

    # FIXME: temporary
    if len(query_strs) == 0:
        return []

    if num_samples == -1:
        # select whatever we loaded
        query_strs = query_strs
    elif len(query_strs) > num_samples:
        query_strs = query_strs[0:num_samples]
    elif len(query_strs) < num_samples:
        # need to generate additional queries
        req_samples = num_samples - len(query_strs)
        qg = QueryGenerator(query_template, args.user, args.db_host, args.port,
                args.pwd, args.db_name)
        gen_sqls = qg.gen_queries(req_samples)
        query_strs += gen_sqls
        sql_str_cache.archive[hashed_tmp] = query_strs

    return query_strs

def gen_query_objs(args, query_strs, cache_name):

    query_obj_cache = klepto.archives.dir_archive(args.cache_dir + cache_name,
            cached=True, serialized=True)
    ret_queries = []
    unknown_query_strs = []

    # everything below this part is for query objects exclusively
    for sql in query_strs:
        hsql = deterministic_hash(sql)
        if hsql in query_obj_cache.archive:
            ret_queries.append(query_obj_cache.archive[hsql])
        else:
            unknown_query_strs.append(sql)

    print("loaded {} query objects".format(len(ret_queries)))
    print("need to generate {} query objects".format(len(unknown_query_strs)))

    if len(unknown_query_strs) == 0:
        return ret_queries

    pdb.set_trace()

    sql_result_cache = args.cache_dir + "/sql_result"
    all_query_objs = []
    with Pool(processes=8) as pool:
        args = [(cur_query, args.user, args.db_host, args.port,
            args.pwd, args.db_name, None,
            args.execution_cache_threshold, sql_result_cache) for
            cur_query in unknown_query_strs]
        all_query_objs = pool.starmap(sql_to_query_object, args)

    for q in all_query_objs:
        ret_queries.append(q)

    print("generated {} samples in {} secs".format(len(queries),
        time.time()-start))
    return ret_queries

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
    if args.template_dir is None:
        update_synth_templates(args, query_templates)
    else:
        for fn in glob.glob(args.template_dir+"/*"):
            with open(fn, "r") as f:
                template = f.read()
                query_templates.append(template)

    # FIXME: all this should happen together, and be cached together.
    # Steps: gen templates, filter out zeros and dups, gen subqueries.

    UPDATE_NEW_CACHE = False
    sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/sql_str", cached=True,
            serialized=True)
    samples = []
    for template in query_templates:
        # TODO: rename etc.
        # db.get_samples(template,
                # num_samples=args.num_samples_per_template)

        ## Test:
        # generate queries
        if not UPDATE_NEW_CACHE:
            query_strs = gen_query_strs(args, template, args.num_samples_per_template)
            samples += gen_query_objs(args, query_strs, "/query_obj")
        else:
            cur_samples = db.get_samples(template,
                    num_samples=args.num_samples_per_template)
            samples += cur_samples
            sql_queries = []
            for q in cur_samples:
                q.template = template
                sql_queries.append(q.query)
            hashed_tmp = deterministic_hash(template)
            sql_str_cache[hashed_tmp] = sql_queries

        sql_str_cache.dump()

    print("len all samples: " , len(samples))

    query_obj_cache = klepto.archives.dir_archive(args.cache_dir + "/query_obj",
            cached=True, serialized=True)

    if UPDATE_NEW_CACHE:
        # going to save all these in a new cache
        for q in samples:
            hashedq = deterministic_hash(q.query)
            query_obj_cache[hashedq] = q
        query_obj_cache.dump()

    start = time.time()
    loaded_queries = []
    for i, q in enumerate(samples):
        if (i % 1000) == 0:
            print(i)
        hashedq = deterministic_hash(q.query)
        if hashedq in query_obj_cache.archive:
            loadedq = query_obj_cache.archive[hashedq]
            loaded_queries.append(loadedq)

    print("took " , time.time() - start)
    print("len loaded queries: ", len(loaded_queries))
    query_obj_cache.clear()

    if args.only_nonzero_samples:
        nonzero_samples = []
        for s in samples:
            if s.true_sel != 0.00:
                nonzero_samples.append(s)
        print("len nonzero samples: ", len(nonzero_samples))
        samples = nonzero_samples

    if args.use_subqueries:
        sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_sql_str",
                cached=True, serialized=True)
        query_obj_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_query_obj",
                cached=True, serialized=True)

        # TODO: parallelize the generation of subqueries
        for i, q in enumerate(samples):
            # TODO: first, generate all subquery strings, and then generate
            # query objects based on those sql strings

            # FIXME: temporary to test the new caching mech
            if UPDATE_NEW_CACHE:
                pass
                # q.subqueries = db.gen_subqueries(q)
                # if i % 1 == 0:
                    # print("{} subqueries generated for query {}".format(len(q.subqueries), i))

            hashed_key = deterministic_hash(q.query)
            if hashed_key in sql_str_cache.archive:
                print("loading hashed key")
                sql_subqueries = sql_str_cache.archive[hashed_key]
            else:
                # FIXME: tmp.
                assert False
                sql_subqueries = gen_all_subqueries(q.query)

            if UPDATE_NEW_CACHE:
                sql_queries = []
                for subq in q.subqueries:
                    subq.template = ""
                    sql_queries.append(subq.query)
                hashed_tmp = deterministic_hash(q.query)
                sql_str_cache[hashed_tmp] = sql_queries
                # for each actual subquery, cache the query objects as well
                sql_str_cache.dump()
                # going to save all these in a new cache
                for subq in q.subqueries:
                    hashedq = deterministic_hash(subq.query)
                    query_obj_cache[hashedq] = subq
                query_obj_cache.dump()

            start = time.time()
            # loaded_queries = []
            # for i, subq in enumerate(q.subqueries):
                # hashedq = deterministic_hash(subq.query)
                # if hashedq in query_obj_cache.archive:
                    # loadedq = query_obj_cache.archive[hashedq]
                    # loaded_queries.append(loadedq)

            loaded_queries = gen_query_objs(args, sql_subqueries, "/subq_query_obj")
            q.subqueries = loaded_queries

    query_obj_cache.clear()

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
    parser.add_argument("--db_file_name", type=str, required=False,
            default=None)
    parser.add_argument("--cache_dir", type=str, required=False,
            default="./caches/")
    parser.add_argument("--execution_cache_threshold", type=int, required=False,
            default=20)

    return parser.parse_args()

args = read_flags()
main()

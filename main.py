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
import psutil

OVERLAP_DIR_TMP = "{RESULT_DIR}/{DIFF_TEMPLATES_TYPE}/"

def get_alg(alg):
    if alg == "independent":
        return Independent()
    elif alg == "saved":
        print("alg is saved!")
        assert args.model_dir is not None
        return SavedPreds(model_dir=args.model_dir)

    elif alg == "xgboost":
        return XGBoost(grid_search = args.grid_search,
                eval_epoch_qerr = args.eval_epoch_qerr,
                validation_epoch = args.validation_epoch,
                use_set_padding = args.use_set_padding,
                unnormalized_mse = args.unnormalized_mse,
                num_workers = args.num_workers,
                switch_loss_fn_epoch = args.switch_loss_fn_epoch,
                tree_method = args.xgb_tree_method,
                n_estimators = args.n_estimators,
                max_depth = args.max_depth,
                lr=args.xgb_lr,
                subsample=args.xgb_subsample,
                max_epochs = args.max_epochs,
                min_qerr = args.min_qerr,
                num_mse_anchoring = args.num_mse_anchoring,
                mat_sparse_features = args.mat_sparse_features,
                flow_weighted_loss = args.flow_weighted_loss,
                eval_parallel = args.eval_parallel,
                max_hid = args.max_hid,
                cost_model = args.cost_model,
                query_batch_size = args.query_batch_size,
                flow_features = args.flow_features,
                table_features = args.table_features,
                join_features = args.join_features,
                pred_features = args.pred_features,
                normalize_flow_loss = args.normalize_flow_loss,
                save_gradients = args.save_gradients,
                weighted_mse = args.weighted_mse,
                weighted_qloss = args.weighted_qloss,
                cost_model_plan_err = args.cost_model_plan_err,
                eval_flow_loss = args.eval_flow_loss,
                use_val_set = args.use_val_set,
                use_best_val_model = args.use_best_val_model,
                start_validation = args.start_validation,
                weight_decay = args.weight_decay,
                num_groups = args.num_groups,
                dropout = args.dropout,
                num_last = args.avg_jl_num_last,
                join_loss_data_file = args.join_loss_data_file,
                train_card_key = args.train_card_key,
                exp_prefix = args.exp_prefix,
                load_query_together = args.load_query_together,
                result_dir = args.result_dir,
                priority_err_type = args.priority_err_type,
                # priority_err_divide_len = args.priority_err_divide_len,
                priority_normalize_type = args.priority_normalize_type,
                tfboard = args.tfboard,
                jl_indexes = args.jl_indexes,
                normalization_type = args.normalization_type,
                preload_features = args.preload_features,
                reprioritize_epoch = args.reprioritize_epoch,
                prioritize_epoch = args.prioritize_epoch,
                heuristic_features = args.heuristic_features,
                debug_set = args.debug_set,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads = args.num_attention_heads,
                hidden_layer_multiple=args.hidden_layer_multiple,
                    eval_epoch = args.eval_epoch,
                    optimizer_name=args.optimizer_name,
                    adaptive_lr=args.adaptive_lr,
                    # rel_qerr_loss=args.rel_qerr_loss,
                    clip_gradient=args.clip_gradient,
                    loss_func = args.loss_func,
                    sampling_priority_type = args.sampling_priority_type,
                    sampling_priority_alpha = args.sampling_priority_alpha,
                    priority_query_len_scale = args.priority_query_len_scale,
                    net_name = args.net_name,
                    eval_epoch_jerr = args.eval_epoch_jerr,
                    eval_epoch_flow_err = args.eval_epoch_flow_err,
                    eval_epoch_plan_err = args.eval_epoch_plan_err,
                    # eval_num_tables = args.eval_num_tables,
                    jl_use_postgres = args.jl_use_postgres,
                    num_tables_feature = args.num_tables_feature,
                    max_discrete_featurizing_buckets =
                            args.max_discrete_featurizing_buckets,
                    nn_type = "microsoft",
                    group_models = args.group_models,
                    adaptive_lr_patience = args.adaptive_lr_patience,
                    # single_threaded_nt = args.single_threaded_nt,
                    nn_weights_init_pg = args.nn_weights_init_pg,
                    avg_jl_priority = args.avg_jl_priority,
                    hidden_layer_size = args.hidden_layer_size)

    elif alg == "rf":
        return RandomForest(grid_search = args.grid_search,
                n_estimators = args.n_estimators,
                max_depth = args.max_depth,
                lr = args.lr,
                exp_prefix = args.exp_prefix)
    elif alg == "linear":
        return Linear()
    elif alg == "postgres":
        return Postgres()
    elif alg == "true":
        return TrueCardinalities()
    elif alg == "true_rank":
        return TrueRank()
    elif alg == "true_random":
        return TrueRandom()
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
                use_batch_norm = args.use_batch_norm,
                mb_size = args.query_mb_size,
                eval_epoch_qerr = args.eval_epoch_qerr,
                validation_epoch = args.validation_epoch,
                use_set_padding = args.use_set_padding,
                unnormalized_mse = args.unnormalized_mse,
                num_workers = args.num_workers,
                switch_loss_fn_epoch = args.switch_loss_fn_epoch,
                switch_loss_fn = args.switch_loss_fn,
                model_dir = args.model_dir,
                min_qerr = args.min_qerr,
                num_mse_anchoring = args.num_mse_anchoring,
                mat_sparse_features = args.mat_sparse_features,
                flow_weighted_loss = args.flow_weighted_loss,
                eval_parallel = args.eval_parallel,
                max_hid = args.max_hid,
                cost_model = args.cost_model,
                query_batch_size = args.query_batch_size,
                flow_features = args.flow_features,
                table_features = args.table_features,
                join_features = args.join_features,
                pred_features = args.pred_features,
                normalize_flow_loss = args.normalize_flow_loss,
                save_gradients = args.save_gradients,
                weighted_mse = args.weighted_mse,
                weighted_qloss = args.weighted_qloss,
                cost_model_plan_err = args.cost_model_plan_err,
                eval_flow_loss = args.eval_flow_loss,
                use_val_set = args.use_val_set,
                use_best_val_model = args.use_best_val_model,
                start_validation = args.start_validation,
                weight_decay = args.weight_decay,
                num_groups = args.num_groups,
                dropout = args.dropout,
                num_last = args.avg_jl_num_last,
                join_loss_data_file = args.join_loss_data_file,
                train_card_key = args.train_card_key,
                exp_prefix = args.exp_prefix,
                load_query_together = args.load_query_together,
                result_dir = args.result_dir,
                priority_err_type = args.priority_err_type,
                # priority_err_divide_len = args.priority_err_divide_len,
                priority_normalize_type = args.priority_normalize_type,
                tfboard = args.tfboard,
                jl_indexes = args.jl_indexes,
                normalization_type = args.normalization_type,
                preload_features = args.preload_features,
                reprioritize_epoch = args.reprioritize_epoch,
                prioritize_epoch = args.prioritize_epoch,
                heuristic_features = args.heuristic_features,
                debug_set = args.debug_set,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads = args.num_attention_heads,
                hidden_layer_multiple=args.hidden_layer_multiple,
                    eval_epoch = args.eval_epoch,
                    optimizer_name=args.optimizer_name,
                    adaptive_lr=args.adaptive_lr,
                    # rel_qerr_loss=args.rel_qerr_loss,
                    clip_gradient=args.clip_gradient,
                    loss_func = args.loss_func,
                    sampling_priority_type = args.sampling_priority_type,
                    sampling_priority_alpha = args.sampling_priority_alpha,
                    priority_query_len_scale = args.priority_query_len_scale,
                    net_name = args.net_name,
                    eval_epoch_jerr = args.eval_epoch_jerr,
                    eval_epoch_flow_err = args.eval_epoch_flow_err,
                    eval_epoch_plan_err = args.eval_epoch_plan_err,
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

def compute_subq_ids(samples):

    node_ids = []
    pred_ids = []

    for samplei, sample in enumerate(samples):
        nodes = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in nodes:
            nodes.remove(SOURCE_NODE)
        nodes.sort()

        for subq_idx, node in enumerate(nodes):
            sorted_node = list(node)
            sorted_node.sort()
            # subq_id = deterministic_hash(str(sorted_node))
            subq_id = ",".join(sorted_node)
            node_ids.append(subq_id)
            cur_pred_cols = []
            for table in sorted_node:
                pred_cols = sample["join_graph"].nodes()[table]["pred_cols"]
                pred_cols = list(set(pred_cols))
                pred_cols.sort()
                cur_pred_cols += pred_cols
            # pred_ids.append(deterministic_hash(str(cur_pred_cols)))
            pred_ids.append(",".join(cur_pred_cols))

    node_ids = np.array(list(set(node_ids)))
    pred_ids = np.array(list(set(pred_ids)))

    print(len(node_ids), len(pred_ids))

    return node_ids, pred_ids

def clear_memory(ts):
    if hasattr(ts, "X"):
        if isinstance(ts.X, dict):
            for k,v in ts.X.items():
                del(v)

        del(ts.X)

        del(ts.Y)
        del(ts)

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

def eval_alg(alg, loss_funcs, queries, samples_type,
        join_loss_pool):
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
    for loss_func in loss_funcs:
        losses = loss_func(queries, yhats, name=alg_name,
                args=args, samples_type=samples_type, exp_name = exp_name,
                pool = join_loss_pool)

        # TODO: set global printoptions to round digits
        print("db: {}, samples_type: {}, alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
                .format(args.db_name, samples_type, alg, len(queries),
                    get_loss_name(loss_func.__name__),
                    np.round(np.mean(losses),3),
                    np.round(np.median(losses),3),
                    np.round(np.percentile(losses,95),3),
                    np.round(np.percentile(losses,99),3)))

    print("loss computations took: {} seconds".format(time.time()-loss_start))

def load_samples(qfns, db, found_db, template_name,
        skip_zero_queries=True, train_template=True, wj_times=None,
        pool=None):

    start = time.time()
    # loading, or generating samples
    samples = []

    if args.debug_set:
        random.seed(args.random_seed)
        qfns = random.sample(qfns, int(len(qfns) / args.debug_ratio))

    if len(qfns) == 0:
        return samples

    skipped = 0

    if pool is not None:
        par_args = []
    qreps = []
    for qfn in qfns:
        if ".pkl" not in qfn:
            print("skipping because qfn not .pkl file")
            continue

        if pool is None:
            qrep = load_sql_rep(qfn)
            qreps.append(qrep)
        else:
            par_args.append((qfn, None))

    if pool is not None:
        # print("going to call pool!")
        qreps = pool.starmap(load_sql_rep, par_args)

    for qi, qrep in enumerate(qreps):
        zero_query = False
        nodes = list(qrep["subset_graph"].nodes())
        # if train_template and "job" in template_name:
            # pdb.set_trace()

        if SOURCE_NODE in nodes:
            nodes.remove(SOURCE_NODE)
        for node in nodes:
            info = qrep["subset_graph"].nodes()[node]

            if "cardinality" not in info:
                print("cardinality not in qrep")
                zero_query = True
                break
            # print(info["cardinality"].keys())
            assert len(info["cardinality"]) > 1
            if "total" not in info["cardinality"]:
                print("total not in query ", qfn)
                zero_query = True
                pdb.set_trace()
                break

            if args.train_card_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
                if not "wanderjoin-" + str(wj_times[template_name]) in info["cardinality"]:
                    zero_query = True
                    break

            elif args.train_card_key not in info["cardinality"]:
                zero_query = True
                break

            if "actual" not in info["cardinality"]:
                print("actual not in card")
                zero_query = True
                break

            if "expected" not in info["cardinality"]:
                print("expected not in card")
                zero_query = True
                break

            # ugh FIXME
            elif info["cardinality"]["actual"] == 0 or \
                    info["cardinality"]["actual"] == 1.1:
                if skip_zero_queries:
                    zero_query = True
                    break
                else:
                    if info["cardinality"]["actual"] == 0:
                        info["cardinality"]["actual"] += 1.1

            if args.sampling_key is not None:
                if wj_times is None:
                    if not (args.sampling_key in info["cardinality"]):
                        zero_query = True
                        break
                else:
                    if not ("wanderjoin-" + str(wj_times[template_name])
                                in info["cardinality"]):
                        zero_query = True
                        break

            # just so everyone is forced to use the wj template queries
            if args.sampling_key is not None:
                if not "wanderjoin-" + str(wj_times[template_name]) in info["cardinality"]:
                    zero_query = True
                    break

        if zero_query:
            skipped += 1
            continue

        qrep["name"] = qfns[qi]
        qrep["template_name"] = template_name
        samples.append(qrep)

    if len(samples) != 0:
        print(("template: {}, zeros skipped: {}, edges: {}, subqueries: {}, queries: {}"
                ", loading time: {}").format(template_name, skipped,
                    len(samples[0]["subset_graph"].edges()),
                    len(samples[0]["subset_graph"].nodes()), len(samples),
                    time.time()-start))
    else:
        print(("template: {}, zero queries").format(template_name))

    if "job" in template_name:
        update_samples(samples, args.flow_features,
                args.cost_model, False, args.db_name)

    if not found_db:
        # print("not found db!!")
        # pdb.set_trace()
        if "job" in template_name and \
                not args.add_job_features:
            return samples

        # elif "job" in template_name and \
                # args.nn_type == "mscn_set":
            # return samples

        elif args.test_diff_templates and \
                not args.add_test_features and \
                not train_template:
                    return samples

        elif args.test_diff_templates and \
                args.nn_type == "mscn_set" and \
                not train_template:
                    return samples

        if db is not None:
            if "job" in template_name:
                print("updating db w/ job features!")
            for sample in samples:
                # not all samples may share all predicates etc. so updating
                # them all. stats will not be recomputed for repeated columns
                # FIXME:
                db.update_db_stats(sample, args.flow_features)
    # else:
        # print("found db :(")
        # pdb.set_trace()

    return samples

def load_all_qrep_data(load_job_queries,
        load_test_queries, load_db, load_train_queries, load_val_queries,
        pool=None):
    misc_cache = klepto.archives.dir_archive("./misc_cache",
            cached=True, serialized=True)

    print("loading qrep data from: ", args.query_directory)
    if load_db:
        if args.eval_on_jobm:
            db_key = deterministic_hash("db-" + args.query_directory + \
                        args.query_templates + str(args.eval_on_job) + \
                        str(args.eval_on_jobm) + \
                        args.nn_type)
        else:
            db_key = deterministic_hash("db-" + args.query_directory + \
                        args.query_templates + str(args.eval_on_job) + \
                        args.nn_type)

        found_db = db_key in misc_cache.archive and not args.regen_db
        # found_db = False
        if found_db:
            db = misc_cache.archive[db_key]
        else:
            # turned on by default so we can update the db stats
            load_train_queries = True
            load_job_queries = True
            load_test_queries = True
            load_val_queries = True
            db = DB(args.user, args.pwd, args.db_host, args.port,
                    args.db_name)
    else:
        db = None
        found_db = False

    train_queries = []
    test_queries = []
    val_queries = []
    query_templates = args.query_templates.split(",")

    fns = list(glob.glob(args.query_directory + "/*"))

    if args.no7a:
        for fn in fns:
            if "7a" in fn:
                fns.remove(fn)
                break

    if args.test_diff_templates:
        # get a sorted version
        if args.diff_templates_type == 3:
            sorted_fns = copy.deepcopy(fns)
            sorted_fns.sort()
            train_tmps, test_tmps = train_test_split(sorted_fns,
                    test_size=args.test_size, random_state=args.diff_templates_seed)
            print(train_tmps)
            print(test_tmps)

    if args.sampling_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
        wj_times = get_wj_times_dict(args.sampling_key)
    elif args.train_card_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
        wj_times = get_wj_times_dict(args.train_card_key)
    else:
        wj_times = get_wj_times_dict("wanderjoin")

    for qi,qdir in enumerate(fns):
        if not load_train_queries and not load_test_queries \
                and not load_val_queries:
            continue
        template_name = os.path.basename(qdir)
        if args.query_templates != "all":
            if template_name not in query_templates:
                print("skipping template ", template_name)
                continue

        if args.no7a:
            if "7a" in template_name:
                print("skipping template 7a")
                continue

        # let's first select all the qfns we are going to load
        qfns = list(glob.glob(qdir+"/*.pkl"))
        qfns.sort()

        if args.num_samples_per_template == -1 \
                or "job" in qdir:
            qfns = qfns
        elif args.num_samples_per_template < len(qfns):
            qfns = qfns[0:args.num_samples_per_template]
        else:
            print("queries should be generated using appropriate script")
            assert False
        # let's do the train-test split on the qfns itself
        cur_val_fns = []
        if args.test and args.use_val_set:
            cur_val_fns, qfns = train_test_split(qfns,
                    test_size=1-args.val_size,
                    random_state=args.random_seed_queries)
            cur_train_fns, cur_test_fns = train_test_split(qfns,
                    test_size=args.test_size,
                    random_state=args.random_seed_queries)

        elif args.test_diff_templates:
            # train template, else test
            if args.diff_templates_type == 1:
                if qi % 2 == 0:
                    cur_test_fns = qfns
                    cur_train_queries = []
                else:
                    cur_train_fns = qfns
                    cur_test_queries = []

            elif args.diff_templates_type == 3:
                if qdir in train_tmps:
                    cur_train_fns = qfns
                    cur_test_fns = []
                    cur_val_fns = []
                else:
                    assert qdir in test_tmps
                    cur_test_fns = qfns
                    cur_train_fns = []
                    cur_val_fns = []
            else:
                assert False

        elif args.test:
            cur_train_fns, cur_test_fns = train_test_split(qfns,
                    test_size=args.test_size,
                    random_state=args.random_seed_queries)
            cur_val_fns = []
        else:
            cur_train_fns = qfns
            cur_test_fns = []
            cur_val_fns = []

        if load_train_queries:
            cur_train_queries = load_samples(cur_train_fns, db, found_db,
                    template_name, skip_zero_queries=args.skip_zero_queries,
                    train_template=True,
                    wj_times=wj_times, pool=pool)
        else:
            cur_train_queries = []
        if load_val_queries:
            cur_val_queries = load_samples(cur_val_fns, db, found_db,
                    template_name, skip_zero_queries=args.skip_zero_queries,
                    train_template=True,
                    wj_times=wj_times, pool=pool)
            print("load val queries: ", len(cur_val_queries))
        else:
            cur_val_queries = []

        if load_test_queries:
            cur_test_queries = load_samples(cur_test_fns, db, found_db,
                    template_name, skip_zero_queries=args.skip_zero_queries,
                    train_template=True,
                    wj_times=wj_times, pool=pool)
        else:
            cur_test_queries = []

        # if len(samples) == 0:
            # print("skipping template {} because zero queries".format(template_name))
            # continue

        train_queries += cur_train_queries
        test_queries += cur_test_queries
        val_queries += cur_val_queries

    job_queries = []
    if args.eval_on_job and load_job_queries:
        job_fns = list(glob.glob(args.job_query_dir + "/*"))
        for qi,qdir in enumerate(job_fns):
            qfns = list(glob.glob(qdir+"/*.pkl"))
            qfns.sort()

            template_name = os.path.basename(qdir)

            samples = load_samples(qfns, db, found_db, template_name,
                    skip_zero_queries=args.job_skip_zero_queries,
                    train_template=False)
            job_queries += samples

    jobm_queries = []
    if args.eval_on_jobm and load_job_queries:
        job_fns = list(glob.glob(args.jobm_query_dir + "/*"))
        for qi,qdir in enumerate(job_fns):
            qfns = list(glob.glob(qdir+"/*.pkl"))
            qfns.sort()

            template_name = os.path.basename(qdir)

            samples = load_samples(qfns, db, found_db, template_name,
                    skip_zero_queries=args.job_skip_zero_queries,
                    train_template=False)
            jobm_queries += samples

    # shuffle train, test queries so join loss computation can be parallelized
    # better: otherwise all queries from templates that take a long time would
    # go to same worker
    random.seed(1234)
    random.shuffle(train_queries)
    random.shuffle(test_queries)
    if args.use_val_set:
        random.shuffle(val_queries)

    if not found_db and db is not None:
        misc_cache.archive[db_key] = db
    del(misc_cache)

    if args.nn_type == "mscn_set" and args.algs == "nn":
        feat_type = "set"
    else:
        feat_type = "combined"

    if db is not None:
        db.init_featurizer(num_tables_feature = args.num_tables_feature,
                separate_regex_bins = args.separate_regex_bins,
                separate_cont_bins = args.separate_cont_bins,
                featurization_type = feat_type,
                max_discrete_featurizing_buckets =
                args.max_discrete_featurizing_buckets,
                heuristic_features = args.heuristic_features,
                flow_features = args.flow_features,
                feat_pg_costs = args.feat_pg_costs,
                feat_pg_path = args.feat_pg_path,
                feat_rel_pg_ests = args.feat_rel_pg_ests,
                feat_rel_pg_ests_onehot = args.feat_rel_pg_ests_onehot,
                feat_pg_est_one_hot = args.feat_pg_est_one_hot,
                feat_tolerance = args.feat_tolerance,
                cost_model = args.cost_model, sample_bitmap=args.sample_bitmap,
                sample_bitmap_num=args.sample_bitmap_num,
                sample_bitmap_buckets=args.sample_bitmap_buckets,
                db_key = db_key)

    return train_queries, test_queries, val_queries, job_queries, jobm_queries, db

def compare_overlap(train_queries, test_queries, test_kind):
    def get_overlap_percentage(train_node_ids, test_node_ids):
        overlap_percentage = []
        for tid in test_node_ids:
            # what is the closest overlap in the training set for it
            best_so_far = 0.0
            if tid in node_common:
                overlap_percentage.append(1.0)
                continue
            tid = tid.split(",")
            for trid in train_node_ids:
                trid = trid.split(",")
                common = np.intersect1d(tid, trid)
                overlap = float(len(common)) / len(tid)
                assert overlap <= 1
                if overlap > best_so_far:
                    best_so_far = overlap
                if best_so_far > 0.98:
                    break
            overlap_percentage.append(best_so_far)

        return overlap_percentage

    train_node_ids, train_pred_ids = compute_subq_ids(train_queries)

    test_node_ids, test_pred_ids = compute_subq_ids(test_queries)
    # print stats
    node_common = np.intersect1d(train_node_ids, test_node_ids)
    pred_common = np.intersect1d(train_pred_ids, test_pred_ids)
    node_all = np.union1d(train_node_ids, test_node_ids)
    pred_all = np.union1d(train_pred_ids, test_pred_ids)


    train_tmps = set([s["template_name"] for s in train_queries])
    test_tmps = set([s["template_name"] for s in test_queries])

    print(train_tmps)
    print(test_tmps)
    overlap_results = defaultdict(list)

    node_exact = float(len(node_common)) / len(test_node_ids)
    pred_exact = float(len(pred_common)) / len(test_pred_ids)

    overlap_results["samples_type"].append(test_kind)
    overlap_results["seed"].append(args.diff_templates_seed)
    overlap_results["overlap_type"].append("node_exact")
    overlap_results["overlap_ratio"].append(node_exact)

    overlap_results["samples_type"].append(test_kind)
    overlap_results["seed"].append(args.diff_templates_seed)
    overlap_results["overlap_type"].append("pred_exact")
    overlap_results["overlap_ratio"].append(pred_exact)

    print("""train-test intersection; test type: {} , node#: {}, common node /all: {}, common
    node /test_nodes:
    {}, pred#: {}, common pred/all: {}, common pred / test_preds:
    {}""".format(test_kind, len(node_common),
        float(len(node_common)) / len(node_all), float(len(node_common)) /
            len(test_node_ids), len(pred_common),
        float(len(pred_common)) / len(pred_all), float(len(pred_common)) /
        len(test_pred_ids)))

    overlap_node_percentage = get_overlap_percentage(train_node_ids,
            test_node_ids)

    print("node overlap: ", test_kind, np.mean(overlap_node_percentage),
            np.median(overlap_node_percentage))

    overlap_results["samples_type"].append(test_kind)
    overlap_results["seed"].append(args.diff_templates_seed)
    overlap_results["overlap_type"].append("node_overlap_mean")
    overlap_results["overlap_ratio"].append(np.mean(overlap_node_percentage))

    overlap_results["samples_type"].append(test_kind)
    overlap_results["seed"].append(args.diff_templates_seed)
    overlap_results["overlap_type"].append("node_overlap_median")
    overlap_results["overlap_ratio"].append(np.median(overlap_node_percentage))

    overlap_pred_percentage = get_overlap_percentage(train_pred_ids,
            test_pred_ids)

    print("pred overlap: ", test_kind, np.mean(overlap_pred_percentage),
            np.median(overlap_pred_percentage))

    overlap_results["samples_type"].append(test_kind)
    overlap_results["seed"].append(args.diff_templates_seed)
    overlap_results["overlap_type"].append("node_overlap_mean")
    overlap_results["overlap_ratio"].append(np.mean(overlap_pred_percentage))

    overlap_results["samples_type"].append(test_kind)
    overlap_results["seed"].append(args.diff_templates_seed)
    overlap_results["overlap_type"].append("node_overlap_median")
    overlap_results["overlap_ratio"].append(np.median(overlap_pred_percentage))

    df = pd.DataFrame(overlap_results)
    rdir = OVERLAP_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                  DIFF_TEMPLATES_TYPE = args.diff_templates_type)
    print("going to overlap at: ", rdir)
    make_dir(rdir)
    fn = rdir + "/overlap_info.pkl"
    old_df = load_object(fn)
    if old_df is not None:
        df = pd.concat([old_df, df], ignore_index=True)

    save_object(fn, df)

def main():
    global args

    if args.db_name == "so":
        global SOURCE_NODE
        SOURCE_NODE = tuple(["SOURCE"])
        args.eval_on_job = False

    if "join-loss" in args.losses or \
            (args.sampling_priority_alpha > 0 and "nn" in args.algs):
        if args.join_loss_pool_num == -1:
            num_processes = int(mp.cpu_count())
        else:
            num_processes = args.join_loss_pool_num
        join_loss_pool = mp.Pool(num_processes)
    else:
        join_loss_pool = None

    if args.max_epochs < args.eval_epoch \
            or not args.eval_test_while_training:
        load_test_samples = False
    else:
        load_test_samples = True

    if args.model_dir is not None:
        old_args = load_object(args.model_dir + "/args.pkl")

        # going to keep old args for most params, except these:
        old_args.losses = args.losses
        old_args.eval_on_job = args.eval_on_job
        old_args.max_epochs = args.max_epochs
        old_args.debug_set = args.debug_set
        old_args.eval_epoch = args.eval_epoch
        old_args.result_dir = args.result_dir
        # so it can load the current model
        old_args.model_dir = args.model_dir
        old_args.query_directory = args.query_directory
        old_num_samples = args.num_samples_per_template
        old_args.use_set_padding = args.use_set_padding
        old_args.eval_on_jobm = args.eval_on_jobm
        old_args.jobm_query_dir = args.jobm_query_dir
        old_args.skip_zero_queries = args.skip_zero_queries

        # if args.max_epochs == 0:
            # # because we aren't actually training, this is just used for init
            # # nn etc.
            # if args.debug_set:
                # old_args.num_samples_per_template = 100
            # else:
                # old_args.num_samples_per_template = 10

        if args.algs == "saved":
            old_args.algs = args.algs

        args = old_args

    train_queries, test_queries, val_queries, job_queries, jobm_queries, db = \
            load_all_qrep_data(False, load_test_samples, True, True,
                    load_test_samples,
                    pool=join_loss_pool)

    update_samples(train_queries, args.flow_features,
            args.cost_model, args.debug_set, args.db_name)
    if len(test_queries) > 0:
        update_samples(test_queries, args.flow_features,
                args.cost_model, args.debug_set, args.db_name)
    if len(val_queries) > 0:
        update_samples(val_queries, args.flow_features,
                args.cost_model, args.debug_set, args.db_name)

    del(job_queries[:])

    if args.model_dir is not None and args.algs == "nn":
        args.num_samples_per_template = old_num_samples

    if args.only_compute_overlap:
        compare_overlap(train_queries, test_queries, "test")

    if len(job_queries) > 0:
        # job_node_ids, job_pred_ids = compute_subq_ids(job_queries)
        compare_overlap(train_queries, job_queries, "job")
    else:
        job_node_ids = []
        job_pred_ids = []

    # if args.only_compute_overlap:
        # exit(-1)

    if len(train_queries) == 0:
        # debugging, so doesn't crash
        train_queries = test_queries

    algorithms = []
    losses = []
    for alg_name in args.algs.split(","):
        algorithms.append(get_alg(alg_name))

    for loss_name in args.losses.split(","):
        losses.append(get_loss(loss_name))

    print("""algs: {}, train queries: {}, val queries: {}, test queries: {},
            job_queries: {}""".format(\
            args.algs, len(train_queries), len(val_queries), len(test_queries),
            len(job_queries)))

    # here, we assume that the alg name is unique enough, for their results to
    # be grouped together
    exp_name = algorithms[0].get_exp_name()
    rdir = RESULTS_DIR_TMP.format(RESULT_DIR = args.result_dir,
                                   ALG = exp_name)
    print("going to save results at: ", rdir)
    make_dir(rdir)
    args_fn = rdir + "/" + "args.pkl"
    save_object(args_fn, args)

    train_times = {}
    eval_times = {}

    for alg in algorithms:
        start = time.time()
        if args.model_dir is None:
            if args.use_val_set:
                alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                        val_samples=val_queries, join_loss_pool=join_loss_pool)
            elif args.eval_test_while_training:
                alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                        val_samples=test_queries, join_loss_pool=join_loss_pool)
            else:
                alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                        val_samples=None, join_loss_pool=join_loss_pool)

            train_times[alg.__str__()] = round(time.time() - start, 2)
            # print("before deleting training sets")
            # print(psutil.virtual_memory())
            # pdb.set_trace()

        else:
            # just used to initialize the fields in the alg
            if alg.max_epochs == 0:
                alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                        val_samples=val_queries, join_loss_pool=join_loss_pool)
                # load the model instead of training it!
                # alg.load_model(args.model_dir)
            else:
                # alg.load_model(args.model_dir)
                alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                        val_samples=test_queries, join_loss_pool=join_loss_pool)
                # alg.train(db, train_queries, use_subqueries=args.use_subqueries,
                        # val_samples=test_queries, join_loss_pool=join_loss_pool)

        if hasattr(alg, "training_sets"):
            ts = alg.training_sets[0]
            # clear_memory(alg.training_sets[0])
            alg.training_sets[0].clean()
            del(alg.training_loaders[0])
            del(alg.training_sets[0])

            if args.eval_epoch < args.max_epochs and len(alg.eval_test_sets) > 0:
                # clear_memory(alg.eval_test_sets[0])
                alg.eval_test_sets[0].clean()
                del(alg.eval_test_sets[0])

            for k,v in alg.eval_loaders.items():
                del(v)

        # may have deleted it to save space
        if len(train_queries) == 0:
            train_queries, _, _, _, _, _ = \
                    load_all_qrep_data(False, False, False, True, False,
                            pool=join_loss_pool)
            update_samples(train_queries, args.flow_features,
                    args.cost_model, args.debug_set, args.db_name)

        start = time.time()

        eval_alg(alg, losses, train_queries, "train", join_loss_pool)
        del(train_queries[:])

        if args.use_val_set:
            if len(val_queries) == 0:
                _, _, val_queries, _, _, _ = \
                        load_all_qrep_data(False, False, False, False, True,
                                pool=join_loss_pool)
            assert len(val_queries) > 0
            update_samples(val_queries, args.flow_features,
                    args.cost_model, args.debug_set, args.db_name)
            eval_alg(alg, losses, val_queries, "validation", join_loss_pool)
            del(val_queries[:])

        if len(test_queries) == 0:
            _, test_queries, _, _, _, _ = \
                    load_all_qrep_data(False, True,
                            False, False, False, pool=join_loss_pool)
            update_samples(test_queries, args.flow_features,
                    args.cost_model, args.debug_set, args.db_name)

        # if args.test:
            # size = int(len(test_queries) / 10)
            # for i in range(10):
                # idx = size*i
                # eval_alg(alg, losses, test_queries[idx:idx+size], "test", join_loss_pool)

        eval_alg(alg, losses, test_queries, "test", join_loss_pool)
        del(test_queries[:])

        if args.eval_on_job:
            _, _, _, job_queries, jobm_queries, _ = \
                    load_all_qrep_data(True, False, False, False, False)
            eval_alg(alg, losses, job_queries, "job", join_loss_pool)
            eval_alg(alg, losses, jobm_queries, "jobm", join_loss_pool)

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

    parser.add_argument("--regen_db", type=int, required=False,
            default=0)
    parser.add_argument("--query_mb_size", type=int, required=False,
            default=1)
    parser.add_argument("--skip_zero_queries", type=int, required=False,
            default=1)
    parser.add_argument("--grid_search", type=int, required=False,
            default=0)
    parser.add_argument("--separate_regex_bins", type=int, required=False,
            default=1)
    parser.add_argument("--separate_cont_bins", type=int, required=False,
            default=1)
    parser.add_argument("--n_estimators", type=int, required=False,
            default=500)
    parser.add_argument("--max_depth", type=int, required=False,
            default=10)
    parser.add_argument("--xgb_subsample", type=float, required=False,
            default=1.0)
    parser.add_argument("--xgb_tree_method", type=str, required=False,
            default="hist")

    parser.add_argument("--query_directory", type=str, required=False,
            default="./minified_dataset")
    parser.add_argument("--cost_model", type=str, required=False,
            default="nested_loop_index7")
    parser.add_argument("--join_loss_data_file", type=str, required=False,
            default=None)
    parser.add_argument("--exp_prefix", type=str, required=False,
            default="")
    parser.add_argument("--query_templates", type=str, required=False,
            default="all")
    parser.add_argument("--debug_set", type=int, required=False,
            default=0)
    parser.add_argument("--debug_ratio", type=float, required=False,
            default=10.0)
    parser.add_argument("--num_mse_anchoring", type=int, required=False,
            default=-1)
    parser.add_argument("--only_compute_overlap", type=int, required=False,
            default=0)
    parser.add_argument("--sample_bitmap", type=int, required=False,
            default=0)
    parser.add_argument("--sample_bitmap_num", type=int, required=False,
            default=1000)
    parser.add_argument("--sample_bitmap_buckets", type=int, required=False,
            default=1000)
    parser.add_argument("--mat_sparse_features", type=int, required=False,
            default=0)
    parser.add_argument("--eval_on_job", type=int, required=False,
            default=1)
    parser.add_argument("--eval_on_jobm", type=int, required=False,
            default=0)

    parser.add_argument("--add_job_features", type=int, required=False,
            default=1)
    parser.add_argument("--add_test_features", type=int, required=False,
            default=1)
    parser.add_argument("--job_skip_zero_queries", type=int, required=False,
            default=0)
    parser.add_argument("--flow_weighted_loss", type=int, required=False,
            default=0)
    parser.add_argument("--job_query_dir", type=str, required=False,
            default="./job_queries/")
    parser.add_argument("--jobm_query_dir", type=str, required=False,
            default="./jobm_queries/")
    parser.add_argument("--test_diff_templates", type=int, required=False,
            default=0)
    parser.add_argument("--diff_templates_type", type=int, required=False,
            default=3)
    parser.add_argument("--diff_templates_seed", type=int, required=False,
            default=1)
    parser.add_argument("--eval_parallel", type=int, required=False,
            default=0)
    parser.add_argument("--max_hid", type=int, required=False,
            default=512)
    parser.add_argument("--no7a", type=int, required=False,
            default=0)
    parser.add_argument("--feat_pg_costs", type=int, required=False,
            default=1)
    parser.add_argument("--feat_pg_path", type=int, required=False,
            default=1)
    parser.add_argument("--feat_rel_pg_ests", type=int, required=False,
            default=1)
    parser.add_argument("--feat_tolerance", type=int, required=False,
            default=0)
    parser.add_argument("--feat_pg_est_one_hot", type=int, required=False,
            default=1)
    parser.add_argument("--feat_rel_pg_ests_onehot", type=int, required=False,
            default=1)

    parser.add_argument("--cost_model_plan_err", type=int, required=False,
            default=1)
    parser.add_argument("--eval_flow_loss", type=int, required=False,
            default=1)
    parser.add_argument("--weighted_qloss", type=int, required=False,
            default=0)
    parser.add_argument("--weighted_mse", type=float, required=False,
            default=0.0)

    parser.add_argument("--unnormalized_mse", type=int, required=False,
            default=0)

    parser.add_argument("--avg_jl_num_last", type=int, required=False,
            default=5)
    parser.add_argument("--preload_features", type=int, required=False,
            default=1)
    parser.add_argument("--load_query_together", type=int, required=False,
            default=0)
    parser.add_argument("--query_batch_size", type=int, required=False,
            default=1)
    parser.add_argument("--normalization_type", type=str, required=False,
            default="mscn")
    parser.add_argument("--min_qerr", type=float, required=False,
            default=1.00)

    parser.add_argument("--nn_weights_init_pg", type=int, required=False,
            default=0)
    parser.add_argument("--single_threaded_nt", type=int, required=False,
            default=0)
    parser.add_argument("--num_tables_feature", type=int, required=False,
            default=1)
    parser.add_argument("--flow_features", type=int, required=False,
            default=1)
    parser.add_argument("--table_features", type=int, required=False,
            default=1)
    parser.add_argument("--join_features", type=int, required=False,
            default=1)
    parser.add_argument("--pred_features", type=int, required=False,
            default=1)
    parser.add_argument("--weight_decay", type=float, required=False,
            default=0.1)

    parser.add_argument("--max_discrete_featurizing_buckets", type=int, required=False,
            default=10)
    parser.add_argument("--heuristic_features", type=int, required=False,
            default=1)
    parser.add_argument("--join_loss_pool_num", type=int, required=False,
            default=10)
    parser.add_argument("--group_models", type=int, required=False,
            default=0)
    parser.add_argument("--priority_normalize_type", type=str, required=False,
            default="")
    parser.add_argument("--normalize_flow_loss", type=int, required=False,
            default=1)

    # parser.add_argument("--priority_err_divide_len", type=int, required=False,
            # default=0)
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
            required=False, default=10)
    parser.add_argument("--switch_loss_fn_epoch", type=int,
            required=False, default=100000)
    parser.add_argument("--switch_loss_fn", type=str,
            required=False, default="")
    parser.add_argument("--num_workers", type=int,
            required=False, default=0)
    parser.add_argument("--eval_epoch", type=int,
            required=False, default=1)
    parser.add_argument("--eval_epoch_qerr", type=int,
            required=False, default=100)
    parser.add_argument("--eval_epoch_jerr", type=int,
            required=False, default=1)
    parser.add_argument("--use_batch_norm", type=int,
            required=False, default=0)
    parser.add_argument("--eval_epoch_flow_err", type=int,
            required=False, default=1)
    parser.add_argument("--eval_epoch_plan_err", type=int,
            required=False, default=101)

    parser.add_argument("--lr", type=float,
            required=False, default=0.0001)
    parser.add_argument("--xgb_lr", type=float,
            required=False, default=0.01)

    parser.add_argument("--clip_gradient", type=float,
            required=False, default=20.0)
    parser.add_argument("--save_gradients", type=int,
            required=False, default=1)
    parser.add_argument("--dropout", type=float,
            required=False, default=0.0)
    parser.add_argument("--rel_qerr_loss", type=int,
            required=False, default=0)
    parser.add_argument("--rel_jloss", type=int,
            required=False, default=0)
    parser.add_argument("--use_val_set", type=int,
            required=False, default=0)
    parser.add_argument("--use_best_val_model", type=int,
            required=False, default=0)
    parser.add_argument("--start_validation", type=int,
            required=False, default=500)
    parser.add_argument("--validation_epoch", type=int,
            required=False, default=100)

    parser.add_argument("--eval_test_while_training", type=int,
            required=False, default=1)
    parser.add_argument("--jl_use_postgres", type=int,
            required=False, default=1)
    parser.add_argument("--nn_type", type=str,
            required=False, default="mscn_set")
    parser.add_argument("--use_set_padding", type=int,
            required=False, default=3)

    parser.add_argument("--num_groups", type=int, required=False,
            default=1, help="""number of groups we divide the input space in.
            If we have at most M tables in a query, and N groups, then each
            group will have samples with M/N tables. e.g., N = 2, M=14,
            samples with 1...7 tables will be in group 1, and rest in group 2.
            """)
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

    parser.add_argument("--optimizer_name", type=str, required=False,
            default="adamw")
    parser.add_argument("--net_name", type=str, required=False,
            default="FCNN")

    parser.add_argument("--num_hidden_layers", type=int,
            required=False, default=2)
    parser.add_argument("--num_attention_heads", type=int,
            required=False, default=1)
    parser.add_argument("--hidden_layer_multiple", type=float,
            required=False, default=None)
    parser.add_argument("--hidden_layer_size", type=int,
            required=False, default=128)

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
    parser.add_argument("--random_seed_queries", type=int, required=False,
            default=2112)
    parser.add_argument("--test", type=int, required=False,
            default=1)
    parser.add_argument("--avg_factor", type=int, required=False,
            default=1)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--val_size", type=float, required=False,
            default=0.2)
    parser.add_argument("--algs", type=str, required=False,
            default="postgres")
    # parser.add_argument("--losses", type=str, required=False,
            # default="qerr,join-loss,flow-loss,plan-loss",
            # help="comma separated list of loss names")
    parser.add_argument("--losses", type=str, required=False,
            default="qerr,join-loss",
            help="comma separated list of loss names")

    parser.add_argument("--result_dir", type=str, required=False,
            default="./results2/")

    parser.add_argument("--model_dir", type=str, required=False,
            default=None)

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
            default=1, help="")
    parser.add_argument("--reprioritize_epoch", type=int, required=False,
            default=1, help="")

    parser.add_argument("--priority_query_len_scale", type=float, required=False,
            default=0, help="")
    parser.add_argument("--tfboard", type=float, required=False,
            default=1, help="")

    parser.add_argument("--loss_func", type=str, required=False,
            default="mse")

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

if __name__ == "__main__":
    args = read_flags()
    main()

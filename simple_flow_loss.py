import argparse
import glob
import os
import klepto
from utils.utils import *
from db_utils.utils import *
from sklearn.model_selection import train_test_split

from utils.net import *
from db_utils.query_storage import *
from cardinality_estimation.db import DB
from cardinality_estimation.query import *
from cardinality_estimation.algs import *
from cardinality_estimation.query_dataset import QueryDataset
from cardinality_estimation.nn import update_samples
from cardinality_estimation.flow_loss import FlowLoss, get_optimization_variables

import torch
from torch.utils import data

import pdb

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

    max_edges = 0
    for qi, qrep in enumerate(qreps):
        if train_template and "job" in template_name:
            # if "29" in qfns[qi] and "flow_loss" in args.loss_func:
                # print("29 query, skipping for training set")
                # continue

            num_edges = len(qrep["subset_graph"].edges())
            if num_edges > max_edges:
                max_edges = num_edges

        zero_query = False
        nodes = list(qrep["subset_graph"].nodes())
        # if train_template and "job" in template_name:
            # pdb.set_trace()

        if SOURCE_NODE in nodes:
            nodes.remove(SOURCE_NODE)

        for node in nodes:
            info = qrep["subset_graph"].nodes()[node]
            cardinality_key = args.db_year_train + "cardinality"

            if cardinality_key not in info:
                # print("cardinality not in qrep")
                zero_query = True
                break

            # if args.db_year_train is not None:
                # db_card_key = str(args.db_year_train) + "cardinality"
                # if db_card_key not in info:
                    # # won't be able to use the old estimates for these, so skip
                    # zero_query = True
                    # break

            # assert len(info[cardinality_key]) > 1
            # if "total" not in info[cardinality_key]:
                # print("total not in query ", qfn)
                # zero_query = True
                # pdb.set_trace()
                # break

            if args.train_card_key in ["wanderjoin", "wanderjoin0.5", "wanderjoin2"]:
                if not "wanderjoin-" + str(wj_times[template_name]) in info[cardinality_key]:
                    zero_query = True
                    break

            elif args.train_card_key not in info[cardinality_key]:
                zero_query = True
                break

            if "actual" not in info[cardinality_key]:
                # print("actual not in card")
                zero_query = True
                break

            if "expected" not in info[cardinality_key]:
                print("expected not in card")
                zero_query = True
                break

            # ugh FIXME
            elif info[cardinality_key]["actual"] == 0 or \
                    info[cardinality_key]["actual"] == 1.1:
                if skip_zero_queries:
                    zero_query = True
                    break
                else:
                    if info[cardinality_key]["actual"] == 0:
                        info[cardinality_key]["actual"] += 1.1

            if args.sampling_key is not None:
                if wj_times is None:
                    if not (args.sampling_key in info[cardinality_key]):
                        zero_query = True
                        break
                else:
                    if not ("wanderjoin-" + str(wj_times[template_name])
                                in info[cardinality_key]):
                        zero_query = True
                        break

            # just so everyone is forced to use the wj template queries
            if args.sampling_key is not None:
                if not "wanderjoin-" + str(wj_times[template_name]) in info[cardinality_key]:
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
                args.cost_model, False, args.db_name, args.db_year_train)


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
            for sample in samples:
                db.update_db_stats(sample, args.flow_features)

    return samples

def collate_fn_combined(batch):
    collated = []
    for i in range(len(batch[0])):
        if i == 2:
            infos = []
            for b in batch:
                infos.append(b[2])
            collated.append(infos)
        else:
            cur_batch = [b[i] for b in batch]
            collated.append(torch.cat(cur_batch, dim=0))

    return collated

def init_dataset(samples, shuffle, batch_size,
        db, db_year, min_val, max_val, load_query_together):
    use_padding = args.use_set_padding

    collate_fn = None
    if load_query_together:
        if args.query_mb_size > 1:
            collate_fn = collate_fn_combined

    training_set = QueryDataset(samples, db,
            "combined", args.heuristic_features,
            args.preload_features, args.normalization_type,
            load_query_together, args.flow_features,
            args.table_features, args.join_features,
            args.pred_features,
            min_val = min_val,
            max_val = max_val,
            card_key = args.train_card_key,
            db_year = db_year,
            use_set_padding = use_padding,
            exp_name = "test")
    training_loader = data.DataLoader(training_set,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True, collate_fn=collate_fn)

    return training_set, training_loader


def load_all_qrep_data(pool=None):

    # turned on by default so we can update the db stats
    load_train_queries = True
    load_job_queries = True
    load_test_queries = True
    load_val_queries = True
    found_db = False
    db_key = ""

    db = DB(args.user, args.pwd, args.db_host, args.port,
            args.db_name, [""])

    train_queries = []
    test_queries = []
    val_queries = []
    query_templates = args.query_templates.split(",")

    fns = list(glob.glob(args.query_directory + "/*"))

    for qi,qdir in enumerate(fns):
        if not load_train_queries and not load_test_queries \
                and not load_val_queries:
            continue
        template_name = os.path.basename(qdir)
        if args.query_templates != "all":
            if template_name not in query_templates:
                print("skipping template ", template_name)
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

        # let's do the train-test split on the qfns itargs
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
                    wj_times=None, pool=pool)
        else:
            cur_train_queries = []
        if load_val_queries:
            cur_val_queries = load_samples(cur_val_fns, db, found_db,
                    template_name, skip_zero_queries=args.skip_zero_queries,
                    train_template=True,
                    wj_times=None, pool=pool)
            print("load val queries: ", len(cur_val_queries))
        else:
            cur_val_queries = []

        if load_test_queries:
            cur_test_queries = load_samples(cur_test_fns, db, found_db,
                    template_name, skip_zero_queries=args.skip_zero_queries,
                    train_template=True,
                    wj_times=None, pool=pool)
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

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_year_train", type=str, required=False,
            default="")
    parser.add_argument("--db_year_test", type=str, required=False,
            default=None, help="1950,1960,... OR all")
    parser.add_argument("--regen_db", type=int, required=False,
            default=0)
    parser.add_argument("--save_exec_sql", type=int, required=False,
            default=1)
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
    # parser.add_argument("--query_directory", type=str, required=False,
            # default="./our_dataset/queries")
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
            default=0)
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
            default=0)
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
    parser.add_argument("--no_join_loss_pool", type=int, required=False,
            default=0)
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
            required=False, default=1)

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

def precompute_flow_training_info(samples, min_val, max_val):
    print("precomputing flow loss info")

    fstart = time.time()
    # precompute a whole bunch of training things
    flow_training_info = []

    for sample in samples:
        qkey = deterministic_hash(sample["sql"])
        subsetg_vectors = list(get_subsetg_vectors(sample,
            args.cost_model))
        true_cards = np.zeros(len(subsetg_vectors[0]),
                dtype=np.float32)
        nodes = list(sample["subset_graph"].nodes())
        nodes.remove(SOURCE_NODE)
        nodes.sort()
        for i, node in enumerate(nodes):
            true_cards[i] = \
                sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

        trueC_vec, dgdxT, G, Q = \
            get_optimization_variables(true_cards,
                subsetg_vectors[0], min_val,
                    max_val, args.normalization_type,
                    subsetg_vectors[4],
                    subsetg_vectors[5],
                    subsetg_vectors[3],
                    subsetg_vectors[1],
                    subsetg_vectors[2],
                    args.cost_model, subsetg_vectors[-1])

        Gv = to_variable(np.zeros(len(subsetg_vectors[0]))).float()
        Gv[subsetg_vectors[-2]] = 1.0
        trueC_vec = to_variable(trueC_vec).float()
        dgdxT = to_variable(dgdxT).float()
        G = to_variable(G).float()
        Q = to_variable(Q).float()

        trueC = torch.eye(len(trueC_vec)).float().detach()
        for i, curC in enumerate(trueC_vec):
            trueC[i,i] = curC

        invG = torch.inverse(G)
        v = invG @ Gv
        left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
        right = Q @ (v)
        left = left.detach().cpu()
        right = right.detach().cpu()
        opt_flow_loss = left @ trueC @ right
        del trueC

        flow_training_info.append((subsetg_vectors, trueC_vec,
                opt_flow_loss))

    print("precomputing flow info took: ", time.time()-fstart)
    return flow_training_info

def main():

    train_queries, test_queries, val_queries, job_queries, jobm_queries, db = \
            load_all_qrep_data(pool=None)
    update_samples(train_queries, args.flow_features,
            args.cost_model, args.debug_set, args.db_name, args.db_year_train)
    if len(test_queries) > 0:
        update_samples(test_queries, args.flow_features,
                args.cost_model, args.debug_set, args.db_name,
                args.db_year_train)
    if len(val_queries) > 0:
        update_samples(val_queries, args.flow_features,
                args.cost_model, args.debug_set, args.db_name,
                args.db_year_train)

    # after applying the log to all cardinalities
    min_val = 0.0
    max_val = 25.0

    load_query_together = True
    training_set, training_loader = init_dataset(train_queries,
                            True, 1, db, "", min_val, max_val,
                            load_query_together)
    flow_training_info = precompute_flow_training_info(train_queries, min_val,
            max_val)

    if load_query_together:
        # load all sub-plans of a query in the same batch
        num_features = len(training_set[0][0][0])
    else:
        num_features = len(training_set[0][0])

    net = SimpleRegression(num_features,
            0, 1,
            num_hidden_layers=args.num_hidden_layers,
            hidden_layer_size=args.hidden_layer_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
            amsgrad=False, weight_decay=args.weight_decay)

    # flow-loss calculations are parallel as well, and multi-threaded torch
    # seems to cause overall slowdown
    torch.set_num_threads(1)

    # all the flow-loss computations are done here
    loss_fn = FlowLoss.apply

    # load query during a loss together

    for epoch in range(args.max_epochs):
        for idx, (xbatch, ybatch,info) in enumerate(training_loader):
            start = time.time()

            # we need this for flow-loss, for q-error it seems better to just load
            # random sub-plans together
            if load_query_together:
                # if mb_size is larger, then it is handled in the collate fn
                if args.query_mb_size == 1:
                    xbatch = xbatch.reshape(xbatch.shape[0]*xbatch.shape[1],
                            xbatch.shape[2])
                    ybatch = ybatch.reshape(ybatch.shape[0]*ybatch.shape[1])
                    qidx = info[0]["query_idx"]
                    assert qidx == info[1]["query_idx"]

            ybatch = ybatch.to(device, non_blocking=True)
            xbatch = xbatch.to(device, non_blocking=True)
            pred = net(xbatch).squeeze(1)

            if args.query_mb_size > 1:
                ybatch = ybatch.detach().cpu()
                qstart = 0
                losses = []
                for cur_info in info:
                    qidx = cur_info[0]["query_idx"]
                    assert qidx == cur_info[1]["query_idx"]
                    subsetg_vectors, trueC_vec, opt_loss = \
                            flow_training_info[qidx]

                    assert len(subsetg_vectors) == 8

                    cur_loss = loss_fn(pred[qstart:qstart+len(cur_info)],
                            ybatch[qstart:qstart+len(cur_info)],
                            normalization_type, min_val,
                            max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                            self.normalize_flow_loss,
                            self.join_loss_pool, self.cost_model)
                    losses.append(cur_loss)
                    qstart += len(cur_info)
                losses = torch.stack(losses)
            else:
                subsetg_vectors, trueC_vec, opt_loss = \
                        flow_training_info[qidx]

                assert len(subsetg_vectors) == 8

                losses = loss_fn(pred, ybatch.detach().cpu(),
                        args.normalization_type, min_val,
                        max_val, [(subsetg_vectors, trueC_vec, opt_loss)],
                        args.normalize_flow_loss,
                        None, args.cost_model)

                loss = losses.sum() / len(losses)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # TODO: evaluate the final model etc.

if __name__ == "__main__":
    args = read_flags()
    main()

import klepto
from .utils import *
from .query_generator import *
from .query_generator2 import *
from cardinality_estimation.query import *
from utils.utils import *
import toml
import multiprocessing
from multiprocessing import Pool
from cardinality_estimation.db import DB

'''
TODO: bring in the Query object format in here as well.
'''

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

def gen_query_strs(args, query_template, num_samples,
        sql_str_cache, save_cur_cache_dir=None):
    '''
    @query_template: str OR dict.

    @ret: [Query, Query, ...]
    '''
    query_strs = []

    # TODO: change key to be based on file name?
    if isinstance(query_template, str):
        hashed_tmp = deterministic_hash(query_template)
    elif isinstance(query_template, dict):
        # hashed_tmp_old = deterministic_hash(query_template)
        # hashed_tmp = deterministic_hash(query_template)
        hashed_tmp = deterministic_hash(query_template["base_sql"]["sql"])
        # if hashed_tmp_old in sql_str_cache.archive:
            # # load it and put it into the new one
            # sql_str_cache.archive[hashed_tmp] = sql_str_cache.archive[hashed_tmp_old]
    else:
        assert False

    if hashed_tmp in sql_str_cache.archive:
        query_strs = sql_str_cache.archive[hashed_tmp]
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

    return query_strs

def gen_query_objs(args, query_strs, query_obj_cache):
    '''
    Note: this must return query objects in the same order as query_strs.
    '''
    ret_queries = []
    unknown_query_strs = []
    idx_map = {}

    # everything below this part is for query objects exclusively
    for i, sql in enumerate(query_strs):
        assert i == len(ret_queries)
        hsql = deterministic_hash(sql)
        if hsql in query_obj_cache:
            curq = query_obj_cache[hsql]
            if not hasattr(curq, "froms"):
                print("NEED TO UPDATE QUERY STRUCT")
                update_query_structure(curq)
                query_obj_cache.archive[hsql] = curq
            assert hasattr(curq, "froms")
            # update the query structure as well if needed
            ret_queries.append(curq)
        elif hsql in query_obj_cache.archive:
            curq = query_obj_cache.archive[hsql]
            if not hasattr(curq, "froms"):
                print("NEED TO UPDATE QUERY STRUCT")
                update_query_structure(curq)
                query_obj_cache.archive[hsql] = curq
            assert hasattr(curq, "froms")
            # update the query structure as well if needed
            ret_queries.append(curq)
        else:
            idx_map[len(unknown_query_strs)] = i
            ret_queries.append(None)
            unknown_query_strs.append(sql)
            # store the appropriate index

    # print("loaded {} query objects".format(len(ret_queries)))
    if len(unknown_query_strs) == 0:
        return ret_queries
    else:
        print("need to generate {} query objects".\
                format(len(unknown_query_strs)))
        # assert False

    sql_result_cache = args.cache_dir + "/sql_result"
    all_query_objs = []
    start = time.time()
    num_processes = int(min(len(unknown_query_strs),
        multiprocessing.cpu_count()))
    # num_processes = 1
    with Pool(processes=num_processes) as pool:
        args = [(cur_query, args.user, args.db_host, args.port,
            args.pwd, args.db_name, None,
            args.execution_cache_threshold, sql_result_cache, None, i) for
            i, cur_query in enumerate(unknown_query_strs)]
        all_query_objs = pool.starmap(sql_to_query_object, args)

    for i, q in enumerate(all_query_objs):
        ret_queries[idx_map[i]] = q
        hsql = deterministic_hash(unknown_query_strs[i])
        # save in memory, so potential repeat queries can be found in the
        # memory cache
        query_obj_cache[hsql] = q
        # save at the disk backend as well, without needing to dump all of
        # the cache
        query_obj_cache.archive[hsql] = q

    print("generated {} samples in {} secs".format(len(unknown_query_strs),
        time.time()-start))

    assert len(ret_queries) == len(query_strs)

    # sanity check: commented out so we don't spend time here
    for i, query in enumerate(ret_queries):
        assert query.query == query_strs[i]

    for i, query in enumerate(ret_queries):
        ret_queries[i] = Query(query.query, query.pred_column_names,
                query.vals, query.cmp_ops, query.true_count, query.total_count,
                query.pg_count, query.pg_marginal_sels, query.marginal_sels)

    return ret_queries

def get_template_samples(fn):
    # number of samples to use from this template (fn)
    print(fn)
    if "2.toml" in fn:
        return 2000
    elif "2b1.toml" in fn:
        return 2000
    elif "2b2.toml" in fn:
        return 2000
    elif "2b3.toml" in fn:
        return 2000
    elif "2b4.toml" in fn:
        return 2000
    elif "3.toml" in fn:
        return 1000
    elif "4.toml" in fn:
        return 1000
    else:
        assert False

def load_all_queries(args, subqueries=True):
    '''
    iterates over every Query, and SubQuery object stored in this cache dir.
    '''

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

    # FIXME: clean up everything...
    query_ret = []
    subquery_ret = []

    query_templates = []
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

    # TODO: not sure if loading it into memory is a good idea or not.
    samples = []
    query_obj_cache = klepto.archives.dir_archive(args.cache_dir + "/query_obj",
            cached=True, serialized=True)
    sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/sql_str",
            cached=True, serialized=True)

    for i, template in enumerate(query_templates):
        # generate queries
        print(os.path.basename(fns[i]))
        # num_samples = get_template_samples(fns[i])
        num_samples = args.num_samples_per_template
        query_strs = gen_query_strs(args, template,
                num_samples, sql_str_cache)

        if not found_db:
            print("going to update db stats!")
            db.update_db_stats(query_strs[0])

        # deduplicate
        query_strs = remove_doubles(query_strs)

        cur_samples = gen_query_objs(args, query_strs, query_obj_cache)
        for sample_id, q in enumerate(cur_samples):
            q.template_name = os.path.basename(fns[i]) + str(sample_id)
            samples.append(q)

        if args.save_cur_cache_dir:
            backup_cache = klepto.archives.dir_archive(args.save_cur_cache_dir + "/sql_str",
                    cached=True, serialized=True)
            if isinstance(template, str):
                hashed_tmp = deterministic_hash(template)
            elif isinstance(template, dict):
                hashed_tmp = deterministic_hash(template["base_sql"]["sql"])
            backup_cache.archive[hashed_tmp] = query_strs

            backup_query_obj_cache = \
                        klepto.archives.dir_archive(args.save_cur_cache_dir + "/query_obj",
                    cached=True, serialized=True)
            for qi, sql in enumerate(query_strs):
                hsql = deterministic_hash(sql)
                backup_query_obj_cache.archive[hsql] = cur_samples[qi]

    print("len all samples: " , len(samples))

    if not found_db:
        misc_cache.archive[db_key] = db

    if args.only_nonzero_samples:
        nonzero_samples = []
        for s in samples:
            print("true count: ", s.true_count)
            if s.true_sel != 0.00:
                nonzero_samples.append(s)
            else:
                pass
                # print(s)
                # print("ZERO QUERY!")
                # pdb.set_trace()

            # if "NULL" in str(s):
                # print(q)
                # pdb.set_trace()

        print("len nonzero samples: ", len(nonzero_samples))
        samples = nonzero_samples

    query_ret = samples

    if args.use_subqueries:
        start = time.time()
        sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_sql_str",
                cached=True, serialized=True)
        query_obj_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_query_obj",
                cached=True, serialized=True)

        # TODO: parallelize the generation of subqueries
        all_sql_subqueries = []
        num_subq_per_query = []
        for i, q in enumerate(samples):
            hashed_key = deterministic_hash(q.query)
            if hashed_key in sql_str_cache:
                sql_subqueries = sql_str_cache[hashed_key]
            elif hashed_key in sql_str_cache.archive:
                sql_subqueries = sql_str_cache.archive[hashed_key]
            else:
                s1 = time.time()
                print("going to generate subqueries for query num ", i)
                sql_subqueries = gen_all_subqueries(q.query)
                # save it for the future!
                sql_str_cache.archive[hashed_key] = sql_subqueries
                print("generating + saving subqueries: ", time.time() - s1)
            all_sql_subqueries += sql_subqueries
            num_subq_per_query.append(len(sql_subqueries))

        # assert len(all_sql_subqueries) % num_subq_per_query == 0
        all_loaded_queries = gen_query_objs(args, all_sql_subqueries, query_obj_cache)
        assert len(all_loaded_queries) == len(all_sql_subqueries)

        for i in range(len(samples)):
            start_idx = i*num_subq_per_query[i]
            end_idx = start_idx + num_subq_per_query[i]
            subquery_ret.append(all_loaded_queries[start_idx:end_idx])

        if args.save_cur_cache_dir:
            backup_cache = \
                    klepto.archives.dir_archive(args.save_cur_cache_dir +
                    "/subq_sql_str", cached=True, serialized=True)
            backup_cache.archive[hashed_key] = sql_subqueries

            backup_query_obj_cache = \
                        klepto.archives.dir_archive(args.save_cur_cache_dir + "/subq_query_obj",
                    cached=True, serialized=True)
            for qi, sql in enumerate(sql_subqueries):
                hsql = deterministic_hash(sql)
                backup_query_obj_cache.archive[hsql] = loaded_queries[qi]

        print("subquery generation took {} seconds".format(time.time()-start))

        return db, query_ret, subquery_ret

def update_all_queries(args):

    assert args.save_cur_cache_dir
    update_sql_cache = klepto.archives.dir_archive(args.save_cur_cache_dir + "/sql_str",
            cached=True, serialized=True)
    update_query_cache = \
                        klepto.archives.dir_archive(args.save_cur_cache_dir + "/query_obj",
                    cached=True, serialized=True)

    update_subquery_sql_cache = \
            klepto.archives.dir_archive(args.save_cur_cache_dir +
            "/subq_sql_str", cached=True, serialized=True)
    update_subquery_cache = \
                klepto.archives.dir_archive(args.save_cur_cache_dir + "/subq_query_obj",
            cached=True, serialized=True)

    samples, subqueries = load_all_queries(args, subqueries=True)


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
    if "2.toml" in fn:
        num = 2000
    elif "2b1.toml" in fn:
        num = 1997
    elif "2b2.toml" in fn:
        num = 1800
    elif "2b3.toml" in fn:
        num = 2000
    elif "2b4.toml" in fn:
        num = 1500
    elif "4.toml" in fn:
        num = 1000
    elif "3.toml" in fn:
        num = 1000
    else:
        assert False

    return num

def _load_subquery_strs(args, queries, sql_str_cache,
        gen_subqueries=True):
    '''
    @ret:
    '''
    start = time.time()
    all_sql_subqueries = []
    new_queries = []

    for i, q in enumerate(queries):
        hashed_key = deterministic_hash(q.query)
        if hashed_key in sql_str_cache:
            sql_subqueries = sql_str_cache[hashed_key]
        elif hashed_key in sql_str_cache.archive:
            sql_subqueries = sql_str_cache.archive[hashed_key]
        else:
            if not gen_subqueries:
                continue
            else:
                s1 = time.time()
                print("going to generate subqueries for query num ", i)
                sql_subqueries = gen_all_subqueries(q.query)
                # save it for the future!
                sql_str_cache.archive[hashed_key] = sql_subqueries
                print("generating + saving subqueries: ", time.time() - s1)

        all_sql_subqueries += sql_subqueries
        new_queries.append(q)

    return new_queries, all_sql_subqueries

def _load_query_strs(args, cache_dir, template):
    sql_str_cache = klepto.archives.dir_archive(cache_dir + "/sql_str",
            cached=True, serialized=True)
    # find all the query strs associated with this template
    if args.num_samples_per_template == -1:
        num_samples = get_template_samples(fns[i])
    else:
        num_samples = args.num_samples_per_template
    query_strs = gen_query_strs(args, template,
            num_samples, sql_str_cache)
    return query_strs

def _remove_zero_samples(samples):
    nonzero_samples = []
    for i, s in enumerate(samples):
        if s.true_sel != 0.00:
            nonzero_samples.append(s)
        else:
            pass

    print("len nonzero samples: ", len(nonzero_samples))
    return nonzero_samples

def _load_subquery_objs(args, all_sql_subqueries, query_obj_cache):
    all_subqueries = gen_query_objs(args, all_sql_subqueries, query_obj_cache)
    assert len(all_subqueries) == len(all_sql_subqueries)
    return all_subqueries

def _save_subq_sqls(queries, subq_sqls, cache_dir):
    sql_cache = klepto.archives.dir_archive(cache_dir + "/subq_sql_str",
            cached=True, serialized=True)
    assert len(queries) == len(subq_sqls)
    for i, q in enumerate(queries):
        sql = q.query
        hkey = deterministic_hash(sql)
        sql_cache.archive[hkey] = subq_sqls[i]

def _save_sqls(template, sqls, cache_dir):
    sql_cache = klepto.archives.dir_archive(cache_dir + "/sql_str",
            cached=True, serialized=True)
    hashed_tmp = deterministic_hash(template["base_sql"]["sql"])
    sql_cache.archive[hashed_tmp] = sqls

def _save_subquery_objs(subqs, cache_dir):
    query_obj_cache = klepto.archives.dir_archive(cache_dir + "/subq_query_obj",
            cached=True, serialized=True)
    for query in subqs:
        hsql = deterministic_hash(query.query)
        query_obj_cache.archive[hsql] = query

def _save_query_objs(queries, cache_dir):
    query_obj_cache = klepto.archives.dir_archive(cache_dir + "/query_obj",
            cached=True, serialized=True)
    for query in queries:
        hsql = deterministic_hash(query.sql)
        query_obj_cache.archive[hsql] = query

def _load_query_objs(args, cache_dir, query_strs, template_name=None,
        gen_subqueries=True):
    query_obj_cache = klepto.archives.dir_archive(cache_dir + "/query_obj",
            cached=True, serialized=True)
    samples = gen_query_objs(args, query_strs, query_obj_cache)

    if args.only_nonzero_samples:
        samples = _remove_zero_samples(samples)

    for i, s in enumerate(samples):
        s.template_name = template_name

    return samples

def load_all_queries(args, subqueries=True):
    all_queries = []
    all_subqueries = []

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

    fns = list(glob.glob(args.template_dir+"/*"))

    subq_query_obj_cache = klepto.archives.dir_archive(args.cache_dir +
            "/subq_query_obj", cached=True, serialized=True)
    subq_sql_str_cache = klepto.archives.dir_archive(args.cache_dir + "/subq_sql_str",
            cached=True, serialized=True)

    for fn in fns:
        assert ".toml" in fn
        template = toml.load(fn)
        query_strs = _load_query_strs(args, args.cache_dir, template)
        # deduplicate
        query_strs = remove_doubles(query_strs)
        if not found_db:
            print("going to update db stats!")
            db.update_db_stats(query_strs[0])

        template_name = os.path.basename(fn)
        queries = _load_query_objs(args, args.cache_dir, query_strs,
                template_name)

        if not args.use_subqueries:
            continue

        queries, subq_strs = _load_subquery_strs(args, queries,
                subq_sql_str_cache, args.gen_queries)
        assert len(subq_strs) % len(queries) == 0
        num_subq_per_query = int(len(subq_strs) / len(queries))
        print("{}: queries: {}, subqueries: {}".format(template_name,
            len(queries), num_subq_per_query))

        subqueries = _load_subquery_objs(args, subq_strs,
                subq_query_obj_cache)

        start_idx = 0
        for i in range(len(queries)):
            end_idx = start_idx + num_subq_per_query
            all_subqueries.append(subqueries[start_idx:end_idx])
            all_queries.append(queries[i])
            if len(all_subqueries[-1]) == 0:
                print(i)
                print("found no subqueries")
                pdb.set_trace()
            start_idx += num_subq_per_query

    if not found_db:
        misc_cache.archive[db_key] = db

    return db, all_queries, all_subqueries

def update_subq_cards(all_subqueries, cache_dir):

    for subqueries in all_subqueries:
        wrong_count = 0
        for subq in subqueries:
            if subq.true_count > subq.total_count:
                subq.total_count = subq.true_count
                wrong_count += 1

        if wrong_count > 0:
            print("wrong counts: ", wrong_count)
            _save_subquery_objs(subqueries, cache_dir)


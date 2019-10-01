import klepto
import glob
import sys
import pdb
import os
# from cardinality_estimation.query import Query

def update_sql_str(new_cache_dir, old_cache_dir, cache_type):
    old_cache = klepto.archives.dir_archive(old_cache_dir + "/" +
            cache_type, cached=True, serialized=True)
    old_cache.load()
    new_cache = klepto.archives.dir_archive(new_cache_dir + "/" +
            cache_type, cached=True, serialized=True)
    new_cache.load()

    for k, sql_strs in old_cache.items():
        assert isinstance(sql_strs, list)
        if k in new_cache:
            new_cache_list = new_cache[k]
            # append the list
            # new_cache[k] += sql_strs
        else:
            new_cache_list = []

        # only append non-doubles
        for sql in sql_strs:
            if sql not in new_cache_list:
                new_cache_list.append(sql)
        new_cache[k] = new_cache_list

        print("new cache len: ", len(new_cache[k]))
    new_cache.dump()

def update_query_obj(new_cache_dir, old_cache_dir, cache_type):
    old_cache = klepto.archives.dir_archive(old_cache_dir + "/" +
            cache_type, cached=True, serialized=True)
    old_cache.load()
    new_cache = klepto.archives.dir_archive(new_cache_dir + "/" +
            cache_type, cached=True, serialized=True)
    new_cache.load()

    for k, qobj in old_cache.items():
        new_cache[k] = qobj

    new_cache.dump()

# goes over every directory, and generates a final single caches directory

all_dir = sys.argv[1]
out_dir = sys.argv[2]

print(all_dir)
print(out_dir)

for cache_dir in glob.glob(all_dir + "/*"):
    # if not os.path.isdir(cache_dir):
        # continue
    # print(cache_dir)
    update_sql_str(out_dir, cache_dir, "sql_str")
    # update_sql_str(out_dir, cache_dir, "subq_sql_str")
    # update_query_obj(out_dir, cache_dir, "query_obj")
    # update_query_obj(out_dir, cache_dir, "subq_query_obj")

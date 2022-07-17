import sys
sys.path.append(".")

import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import argparse
import psycopg2 as pg
# from db_utils.utils import *
from utils.utils import *
from db_utils.query_storage import load_sql_rep, save_sql_rep,update_qrep
from utils.utils import *
from db_utils.utils import *
import pdb
import random
import klepto
from multiprocessing import Pool, cpu_count
import json
import pickle
from sql_rep.utils import execute_query
from networkx.readwrite import json_graph
import re
from sql_rep.query import parse_sql
from wanderjoin import WanderJoin
import math
# from progressbar import progressbar as bar
import scipy.stats as st

OLD_TIMEOUT_COUNT_CONSTANT = 150001001
OLD_CROSS_JOIN_CONSTANT = 150001000
OLD_EXCEPTION_COUNT_CONSTANT = 150001002

TIMEOUT_COUNT_CONSTANT = 150001000001
CROSS_JOIN_CONSTANT = 150001000000
EXCEPTION_COUNT_CONSTANT = 150001000002

CACHE_TIMEOUT = 4
CACHE_CARD_TYPES = ["actual"]

DEBUG_CHECK_TIMES = False
CONF_ALPHA = 0.99

NEW_TABLE_TEMPLATE = "{TABLE}_{SS}{NUM}"
NEW_JOIN_TABLE_TEMPLATE = "{TABLE}_{JOINKEY}_{SS}{NUM}"

SMALL_PRIMARY_TABS = ["role_type", "info_type", "kind_type",
                        "company_type", "comp_cast_type",
                        "link_type"]

## TODO: maybe remove keyword_id from here
SMALL_FIDS = ["kind_id", "role_id", "person_role_id", "info_type_id",
                "company_type_id", "company_id", "link_id",
                "keyword_id", "linked_movie_id", "link_type_id",
                "status_id", "subject_id"]

JOIN_COL_MAP_IMDB = {}
JOIN_COL_MAP_IMDB["title.id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_info.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["cast_info.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_keyword.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_companies.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_link.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_info_idx.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_link.linked_movie_id"] = "movie_id"
## TODO: handle it so same columns map to same table+col
# JOIN_COL_MAP_IMDB["miidx.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["aka_title.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["complete_cast.movie_id"] = "movie_id"

JOIN_COL_MAP_IMDB["movie_keyword.keyword_id"] = "keyword"
JOIN_COL_MAP_IMDB["keyword.id"] = "keyword"

JOIN_COL_MAP_IMDB["name.id"] = "person_id"
JOIN_COL_MAP_IMDB["person_info.person_id"] = "person_id"
JOIN_COL_MAP_IMDB["cast_info.person_id"] = "person_id"
JOIN_COL_MAP_IMDB["aka_name.person_id"] = "person_id"
# TODO: handle cases
# JOIN_COL_MAP_IMDB["a.person_id"] = "person_id"

JOIN_COL_MAP_IMDB["title.kind_id"] = "kind_id"
JOIN_COL_MAP_IMDB["kind_type.id"] = "kind_id"

JOIN_COL_MAP_IMDB["cast_info.role_id"] = "role_id"
JOIN_COL_MAP_IMDB["role_type.id"] = "role_id"

JOIN_COL_MAP_IMDB["cast_info.person_role_id"] = "char_id"
JOIN_COL_MAP_IMDB["char_name.id"] = "char_id"

JOIN_COL_MAP_IMDB["movie_info.info_type_id"] = "info_id"
JOIN_COL_MAP_IMDB["movie_info_idx.info_type_id"] = "info_id"
# JOIN_COL_MAP_IMDB["mi_idx.info_type_id"] = "info_id"
# JOIN_COL_MAP_IMDB["miidx.info_type_id"] = "info_id"

JOIN_COL_MAP_IMDB["person_info.info_type_id"] = "info_id"
JOIN_COL_MAP_IMDB["info_type.id"] = "info_id"

JOIN_COL_MAP_IMDB["movie_companies.company_type_id"] = "company_type"
JOIN_COL_MAP_IMDB["company_type.id"] = "company_type"

JOIN_COL_MAP_IMDB["movie_companies.company_id"] = "company_id"
JOIN_COL_MAP_IMDB["company_name.id"] = "company_id"

JOIN_COL_MAP_IMDB["movie_link.link_type_id"] = "link_id"
JOIN_COL_MAP_IMDB["link_type.id"] = "link_id"

JOIN_COL_MAP_IMDB["complete_cast.status_id"] = "subject"
JOIN_COL_MAP_IMDB["complete_cast.subject_id"] = "subject"
JOIN_COL_MAP_IMDB["comp_cast_type.id"] = "subject"

#### port:5440 stuff;
JOIN_COL_MAP_STATS = {}
JOIN_COL_MAP_STATS["badges.id"] = "badge_id"
JOIN_COL_MAP_STATS["badges.userid"] = "user_id"

JOIN_COL_MAP_STATS["comments.userid"] = "user_id"
JOIN_COL_MAP_STATS["comments.postid"] = "post_id"
JOIN_COL_MAP_STATS["comments.id"] = "comment_id"

JOIN_COL_MAP_STATS["posthistory.id"] = "history_id"
JOIN_COL_MAP_STATS["posthistory.postid"] = "post_id"
JOIN_COL_MAP_STATS["posthistory.userid"] = "user_id"

JOIN_COL_MAP_STATS["postlinks.id"] = "link_id"
JOIN_COL_MAP_STATS["postlinks.postid"] = "post_id"
JOIN_COL_MAP_STATS["postlinks.relatedpostid"] = "post_id"
JOIN_COL_MAP_STATS["postlinks.linktypeid"] = "link_type_id"

JOIN_COL_MAP_STATS["posts.id"] = "post_id"
JOIN_COL_MAP_STATS["posts.posttypeid"] = "post_type_id"
JOIN_COL_MAP_STATS["posts.owneruserid"] = "user_id"
JOIN_COL_MAP_STATS["posts.lasteditoruserid"] = "user_id"

JOIN_COL_MAP_STATS["tags.id"] = "tag_id"
JOIN_COL_MAP_STATS["tags.excerptpostid"] = "post_id"

JOIN_COL_MAP_STATS["users.id"] = "user_id"

JOIN_COL_MAP_STATS["votes.id"] = "vote_id"
JOIN_COL_MAP_STATS["votes.postid"] = "post_id"
JOIN_COL_MAP_STATS["votes.userid"] = "user_id"

# JOIN_COL_MAP_STATS["badges.Id"] = "badge_id"
# JOIN_COL_MAP_STATS["badges.UserId"] = "user_id"

# JOIN_COL_MAP_STATS["comments.UserId"] = "user_id"
# JOIN_COL_MAP_STATS["comments.PostId"] = "post_id"

JOIN_COL_MAP_STATS["comments.id"] = "comment_id"
# JOIN_COL_MAP_STATS["postHistory.Id"] = "history_id"
# JOIN_COL_MAP_STATS["postHistory.PostId"] = "post_id"
# JOIN_COL_MAP_STATS["postHistory.UserId"] = "user_id"

# JOIN_COL_MAP_STATS["postLinks.Id"] = "link_id"
# JOIN_COL_MAP_STATS["postLinks.PostId"] = "post_id"
# JOIN_COL_MAP_STATS["postLinks.RelatedPostId"] = "post_id"
#JOIN_COL_MAP_STATS["postLinks.LinkTypeId"] = "link_type_id"

# JOIN_COL_MAP_STATS["posts.Id"] = "post_id"
# JOIN_COL_MAP_STATS["posts.PostTypeId"] = "post_type_id"
# JOIN_COL_MAP_STATS["posts.OwnerUserId"] = "user_id"
# JOIN_COL_MAP_STATS["posts.LastEditorUserId"] = "user_id"


# JOIN_COL_MAP_STATS["tags.Id"] = "tag_id"
# JOIN_COL_MAP_STATS["tags.ExcerptPostId"] = "post_id"
# JOIN_COL_MAP_STATS["users.Id"] = "user_id"
# JOIN_COL_MAP_STATS["votes.Id"] = "vote_id"
# JOIN_COL_MAP_STATS["votes.PostId"] = "post_id"
# JOIN_COL_MAP_STATS["votes.UserId"] = "user_id"

JOIN_COL_MAP = JOIN_COL_MAP_STATS
#JOIN_COL_MAP = JOIN_COL_MAP_IMDB

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="postgres")
    parser.add_argument("--pwd", type=str, required=False,
            default="postgres")
    parser.add_argument("--card_cache_dir", type=str, required=False,
            default="./cardinality_cache")
    parser.add_argument("--port", type=str, required=False,
            default=5431)
    parser.add_argument("--wj_walk_timeout", type=float, required=False,
            default=0.5)
    parser.add_argument("--query_dir", type=str, required=False,
            default=None)
    parser.add_argument("-n", "--num_queries", type=int,
            required=False, default=-1)
    parser.add_argument("--use_tries", type=int,
            required=False, default=0)
    parser.add_argument("--no_parallel", type=int,
            required=False, default=0)
    parser.add_argument("--card_type", type=str, required=False,
            default=None)
    parser.add_argument("--key_name", type=str, required=False,
            default=None)
    parser.add_argument("--true_timeout", type=int,
            required=False, default=1800000)
    parser.add_argument("--pg_total", type=int,
            required=False, default=1)
    parser.add_argument("--num_proc", type=int,
            required=False, default=-1)
    parser.add_argument("--seed", type=int,
            required=False, default=1234)
    parser.add_argument("--sample_num", type=int,
            required=False, default=1000)
    parser.add_argument("--sampling_type", type=str,
            required=False, default="sb")

    return parser.parse_args()

def update_bad_qrep(qrep):
    qrep = parse_sql(qrep["sql"], None, None, None, None, None,
            compute_ground_truth=False)
    qrep["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
    return qrep

def get_join_bitmaps(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, fn, idx, sample_num, sampling_type):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    if "imdb_id" in qrep["sql"]:
        return

    if key_name is None:
        key_name = card_type

    key_name = str(sampling_type) + str(sample_num)

    con = pg.connect(user=user, host=db_host, port=port,
            password=pwd, database=db_name)
    cursor = con.cursor()

    if idx % 10 == 0:
        print("query: ", idx)

    node_list = list(qrep["subset_graph"].nodes())
    if SOURCE_NODE in node_list:
        node_list.remove(SOURCE_NODE)
    node_list.sort(reverse=False, key = lambda x: len(x))

    jg = qrep["join_graph"]

    for subqi, subset in enumerate(node_list):
        info = qrep["subset_graph"].nodes()[subset]
        if len(subset) > 1:
            # if "join_bitmap" in info:
                # del(info["join_bitmap"])
            break

        if "join_bitmap" not in info:
            info["join_bitmap"] = {}

        ## for future
        # if len(info["join_bitmap"]) != 0:
            # continue

        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)

        if "postHistory" in subsql:
            subsql = subsql.replace("postHistory", "posthistory")
        if "postLinks" in subsql:
            subsql = subsql.replace("postLinks", "postlinks")

        assert len(sg.nodes()) == 1
        for k,v in sg.nodes(data=True):
            table = v["real_name"]
            break
            # sample_table = table + "_" + sampling_type + str(sample_num)

        sample_tables = []
        exec_tables = []
        sample_ids = []

        jinfo = jg.nodes()[subset[0]]
        for join_key, join_real in JOIN_COL_MAP.items():
            if ".id" in join_key.lower():
                continue
            if "posttypeid" in join_key.lower():
                continue

            join_key_col = join_key[join_key.find(".")+1:]
            join_tab = join_key[:join_key.find(".")]

            if table == join_tab:
                # if len(jinfo["predicates"]) > 0 and join_real not in ["movie_id",
                        # "person_id"]:
                    # print("Join Key: ", join_key)
                    # print(jinfo["predicates"])
                    # pdb.set_trace()

                newid = join_key[join_key.find(".")+1:]
                newtab = NEW_JOIN_TABLE_TEMPLATE.format(TABLE=table,
                                               # JOINKEY = join_key_col,
                                               JOINKEY = join_real,
                                               SS = "sb",
                                               NUM = "1000")
                exectab = newtab
                if newid in SMALL_FIDS:
                    exectab = NEW_TABLE_TEMPLATE.format(TABLE=table,
                            SS = "sb",
                            NUM = "1000")

                sample_tables.append(newtab)
                exec_tables.append(exectab)
                sample_ids.append(newid)

        if len(sample_tables) == 0:
            continue

        for si, sample_tab in enumerate(sample_tables):
            if sample_tab in info["join_bitmap"]:
                continue

            exec_tab = exec_tables[si]
            # adding extra space because table name may appear in predicates,
            # e.g., t.title
            execsql = subsql.replace(" " + table + " ", " \"" + exec_tab + "\" ")
            execsql = execsql.replace("COUNT(*)", "\"" + sample_ids[si] + "\"")

            try:
                cursor.execute(execsql)
                outputs = cursor.fetchall()
                bitmaps = [int(o[0]) for o in outputs if o[0] is not None]
                print(len(bitmaps))

            except Exception as e:
                print("Exception!")
                print(execsql)
                print(table)
                print(subset)
                print(e)
                pdb.set_trace()
                cursor.close()
                con.close()
                con = pg.connect(user=user, host=db_host, port=port,
                        password=pwd, database=db_name)
                cursor = con.cursor()

                continue

            info["join_bitmap"][sample_tab] = bitmaps

    if fn is not None:
        update_qrep(qrep)
        save_sql_rep(fn, qrep)
        print("saved new qrep!")

    con.close()
    cursor.close()

    return qrep

def main():
    print("query dir: ", args.query_dir)
    fns = list(glob.glob(args.query_dir + "/*"))
    print("number of files: ", len(fns))
    fns.sort()
    par_args = []
    for i, fn in enumerate(fns):
        if i >= args.num_queries and args.num_queries != -1:
            break

        if (".pkl" not in fn and ".sql" not in fn):
            continue

        if ".pkl" in fn:
            qrep = load_sql_rep(fn)
        else:
            assert False

        if args.no_parallel:
            get_join_bitmaps(qrep, args.card_type, args.key_name, args.db_host,
                    args.db_name, args.user, args.pwd, args.port,fn,
                     i, args.sample_num, args.sampling_type)
            continue

        par_func = get_join_bitmaps
        par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                args.db_name, args.user, args.pwd, args.port,
                fn, i,
                args.sample_num,
                args.sampling_type))

    assert not args.no_parallel
    start = time.time()
    if args.num_proc == -1:
        num_proc = cpu_count()
    else:
        num_proc = args.num_proc
    with Pool(processes = num_proc) as pool:
        qreps = pool.starmap(par_func, par_args)
    print("updated all cardinalities in {} seconds".format(time.time()-start))

args = read_flags()
main()

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

JOIN_COL_MAP = {}
JOIN_COL_MAP["t.id"] = "movie_id"
JOIN_COL_MAP["mi.movie_id"] = "movie_id"
JOIN_COL_MAP["ci.movie_id"] = "movie_id"
JOIN_COL_MAP["mk.movie_id"] = "movie_id"
JOIN_COL_MAP["mc.movie_id"] = "movie_id"
JOIN_COL_MAP["ml.movie_id"] = "movie_id"
JOIN_COL_MAP["mi_idx.movie_id"] = "movie_id"
JOIN_COL_MAP["mi_idx.movie_id"] = "movie_id"
JOIN_COL_MAP["ml.linked_movie_id"] = "movie_id"
## TODO: handle it so same columns map to same table+col
JOIN_COL_MAP["miidx.movie_id"] = "movie_id"
JOIN_COL_MAP["at.movie_id"] = "movie_id"
JOIN_COL_MAP["cc.movie_id"] = "movie_id"

JOIN_COL_MAP["mk.keyword_id"] = "keyword"
JOIN_COL_MAP["k.id"] = "keyword"

JOIN_COL_MAP["n.id"] = "person_id"
JOIN_COL_MAP["pi.person_id"] = "person_id"
JOIN_COL_MAP["ci.person_id"] = "person_id"
JOIN_COL_MAP["an.person_id"] = "person_id"
# TODO: handle cases
JOIN_COL_MAP["a.person_id"] = "person_id"

JOIN_COL_MAP["t.kind_id"] = "kind_id"
JOIN_COL_MAP["kt.id"] = "kind_id"

JOIN_COL_MAP["ci.role_id"] = "role_id"
JOIN_COL_MAP["rt.id"] = "role_id"

JOIN_COL_MAP["ci.person_role_id"] = "char_id"
JOIN_COL_MAP["chn.id"] = "char_id"

JOIN_COL_MAP["mi.info_type_id"] = "info_id"
JOIN_COL_MAP["mii.info_type_id"] = "info_id"
JOIN_COL_MAP["mi_idx.info_type_id"] = "info_id"
JOIN_COL_MAP["miidx.info_type_id"] = "info_id"

JOIN_COL_MAP["pi.info_type_id"] = "info_id"
JOIN_COL_MAP["it.id"] = "info_id"

JOIN_COL_MAP["mc.company_type_id"] = "company_type"
JOIN_COL_MAP["ct.id"] = "company_type"

JOIN_COL_MAP["mc.company_id"] = "company_id"
JOIN_COL_MAP["cn.id"] = "company_id"

JOIN_COL_MAP["ml.link_type_id"] = "link_id"
JOIN_COL_MAP["lt.id"] = "link_id"

## complete_cast
JOIN_COL_MAP["cc.status_id"] = "subject"
JOIN_COL_MAP["cc.subject_id"] = "subject"
JOIN_COL_MAP["cct.id"] = "subject"

def pg_est_aggr_from_explain(output):
    '''
    '''
    est_vals = None

    for line in output:
        line = line[0]
        # if "Seq Scan" in line or "Loop" in line or "Join" in line \
                # or "Index Scan" in line or "Scan" in line:
        for w in line.split():
            if "rows" in w and est_vals is None:
                est_vals = int(re.findall("\d+", w)[0])
                return est_vals

    print("pg est aggre failed!")
    print(output)
    pdb.set_trace()
    return 1.00

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user", type=str, required=False,
            default="ceb")
    parser.add_argument("--pwd", type=str, required=False,
            default="password")
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
            required=False, default=32)
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

def is_cross_join(sg):
    '''
    enforces the constraint that the graph should be connected w/o the site
    node.
    '''
    if len(sg.nodes()) < 2:
        # FIXME: should be return False
        return False
    sg2 = nx.Graph(sg)
    to_remove = []
    for node, data in sg2.nodes(data=True):
        if data["real_name"] == "site":
            to_remove.append(node)
    for node in to_remove:
        sg2.remove_node(node)
    if nx.is_connected(sg2):
        return False
    return True

def get_distinct_joinkeys(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, fn, idx, sample_num, sampling_type):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    if key_name is None:
        key_name = card_type

    # if sample_num is not None:
        # key_name = str(sampling_type) + str(sample_num)

    con = pg.connect(user=user, host=db_host, port=port,
            password=pwd, database=db_name)
    cursor = con.cursor()
    cursor.execute("SET enable_hashjoin = off")

    if idx % 10 == 0:
        print("query: ", idx)

    node_list = list(qrep["subset_graph"].nodes())
    if SOURCE_NODE in node_list:
        node_list.remove(SOURCE_NODE)
    node_list.sort(reverse=False, key = lambda x: len(x))

    # subset_edges = qrep["subset_graph"].edges()
    sg = qrep["subset_graph"]
    jg = qrep["join_graph"]

    # for si, subset_edge in enumerate(subset_edges):
        # if si % 100 == 0:
            # print(si)

    for ni,node in enumerate(node_list):

        info = qrep["subset_graph"].nodes()[node]
        node_card = info["cardinality"]["actual"]
        node_pg = info["cardinality"]["expected"]

        # edges
        joinedges = sg.in_edges(node)
        aggr_conds = []
        aggr_cols = []
        updated_edges = []

        for subset_edge in joinedges:
            u = subset_edge[0]
            subset = subset_edge[1]
            assert len(u) > len(subset)
            einfo = sg.edges()[subset_edge]

            if "join_key_cardinality" not in einfo or \
                    len(einfo["join_key_cardinality"]) == 0:
                einfo["join_key_cardinality"] = {}
                ecards = einfo["join_key_cardinality"]

                updated_edges.append(subset_edge)
                newtab = tuple(list(set(u) - set(subset)))[0]
                poss_edges = jg.edges(newtab)

                cur_edges = []
                cur_edge_cols = []

                for e in poss_edges:
                    for e0 in e:
                        if e0 == newtab:
                            continue
                        if e0 in subset:
                            cur_edges.append(e)
                            cur_edge_cols.append(e0)
                            break

                    if len(cur_edges) == 1:
                        break

                assert len(cur_edges) == 1
                for ei, e in enumerate(cur_edges):
                    jcondition = jg.edges()[e]["join_condition"]
                    ecol = cur_edge_cols[ei]
                    jcols = jcondition.split("=")
                    curattr = None
                    for jcol in jcols:
                        jcol_tmp = jcol.split(".")[0]
                        jcol_tmp = jcol_tmp.strip()
                        if ecol == jcol_tmp:
                            curattr = jcol
                            break

                    if curattr is None:
                        print(ecol)
                        print(jcols)
                        pdb.set_trace()

                    curattr = curattr.replace(" ", "")
                    assert curattr is not None
                    aggr_conds.append("COUNT(DISTINCT {})".format(curattr))
                    aggr_cols.append(curattr)

                    # TODO: need to add postgres sqls for all

        # if len(node) > 2:
            # print(aggr_conds)
            # print(updated_edges)
            # pdb.set_trace()

        if len(aggr_conds) == 0:
            continue

        # if len(node) < 3:
            # continue

        join_subg = qrep["join_graph"].subgraph(node)
        subsql = nx_graph_to_query(join_subg)

        aggr_cond = ",".join(aggr_conds)
        csql = subsql.replace("COUNT(*)", aggr_cond)

        try:
            cursor.execute(csql)
            outputs = cursor.fetchall()

        except Exception as e:
            print("Exception!")
            print(e)
            print(csql)
            pdb.set_trace()

        assert len(outputs[0]) == len(aggr_conds) == len(updated_edges)
        print(outputs)

        for ci, cure in enumerate(updated_edges):
            ecards = sg.edges()[cure]["join_key_cardinality"]
            card = outputs[0][ci]
            aggrcol = aggr_cols[ci]
            ecards[aggrcol] = {}
            ecards[aggrcol]["actual"] = card

        for ci, curattr in enumerate(aggr_cols):
            ecards = sg.edges()[updated_edges[ci]]["join_key_cardinality"]
            pgsql = subsql.replace("COUNT(*)", " DISTINCT {}".format(curattr))
            pgsql = "EXPLAIN " + pgsql
            try:
                cursor.execute(pgsql)
                output = cursor.fetchall()
            except Exception as e:
                print("Exception!")
                print(e)
                print(pgsql)
                pdb.set_trace()
                continue

            card = pg_est_aggr_from_explain(output)
            ecards[curattr]["expected"] = card

    if fn is not None:
        update_qrep(qrep)
        save_sql_rep(fn, qrep)
        print("saved new qrep!")

    cursor.close()
    con.close()
    return qrep

def get_sample_bitmaps(qrep, card_type, key_name, db_host, db_name, user, pwd,
        port, fn, idx, sample_num, sampling_type):
    '''
    updates qrep's fields with the needed cardinality estimates, and returns
    the qrep.
    '''
    if key_name is None:
        key_name = card_type

    if sample_num is not None:
        key_name = str(sampling_type) + str(sample_num)

        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)
        cursor = con.cursor()

    cursor.execute("SET enable_hashjoin = off")

    if idx % 10 == 0:
        print("query: ", idx)

    # node_list = list(qrep["subset_graph"].nodes())
    # if SOURCE_NODE in node_list:
        # node_list.remove(SOURCE_NODE)
    # node_list.sort(reverse=False, key = lambda x: len(x))
    subset_edges = qrep["subset_graph"].edges()
    jg = qrep["join_graph"]

    for si, subset_edge in enumerate(subset_edges):
        if si % 100 == 0:
            print(si)

        u = subset_edge[0]
        subset = subset_edge[1]
        if subset == SOURCE_NODE:
            continue

        assert len(u) > len(subset)

        einfo = qrep["subset_graph"].edges()[subset_edge]

        if "join_key_cardinality" not in einfo:
            # DEBUG:
            continue
            einfo["join_key_cardinality"] = {}
            ecards = einfo["join_key_cardinality"]
        else:
            # TODO:
            # continue
            ecards = einfo["join_key_cardinality"]
            if len(ecards) == 0:
                continue

        info = qrep["subset_graph"].nodes()[subset]
        node_card = info["cardinality"]["actual"]
        node_pg = info["cardinality"]["expected"]

        newtab = tuple(list(set(u) - set(subset)))[0]
        poss_edges = jg.edges(newtab)

        ## join_graph edges
        cur_edges = []
        cur_edge_cols = []

        for e in poss_edges:
            for e0 in e:
                if e0 == newtab:
                    continue
                if e0 in subset:
                    cur_edges.append(e)
                    cur_edge_cols.append(e0)
                    break

        sg = qrep["join_graph"].subgraph(subset)
        subsql = nx_graph_to_query(sg)

        execsqls = []
        pgsqls = []
        curattrs = []

        for ei, e in enumerate(cur_edges):
            jcondition = jg.edges()[e]["join_condition"]
            ecol = cur_edge_cols[ei]
            jcols = jcondition.split("=")
            curattr = None
            for jcol in jcols:
                jcol_tmp = jcol.split(".")[0]
                jcol_tmp = jcol_tmp.strip()
                if ecol == jcol_tmp:
                    curattr = jcol
                    break

            if curattr is None:
                print(ecol)
                print(jcols)
                pdb.set_trace()

            curattr = curattr.replace(" ", "")
            assert curattr is not None
            csql = subsql.replace("COUNT(*)", "COUNT(DISTINCT {})".format(curattr))
            # EXPLAIN SELECT DISTINCT mc.company_type_id from movie_companies as mc;
            pgsql = subsql.replace("COUNT(*)", " DISTINCT {}".format(curattr))
            pgsql = "EXPLAIN " + pgsql

            execsqls.append(csql)
            pgsqls.append(pgsql)
            curattrs.append(curattr)
            # TODO: confirm assumption
            # break

        ## DEBUG stuff:
        real_col_vals = []
        for curattr in curattrs:
            orig_attr = "".join([ca for ca in curattr if not ca.isdigit()])
            orig_attr = orig_attr.replace(" ", "")

            if orig_attr not in JOIN_COL_MAP:
                print(orig_attr + " not found in JOIN_COL_MAP")
                pdb.set_trace()
            else:
                real_col_vals.append(JOIN_COL_MAP[orig_attr])

        if len(set(real_col_vals)) != 1:
            print("Num real attributes: ", len(set(real_col_vals)))
            print(real_col_vals, curattrs)
            pdb.set_trace()

        # update ecards
        newecards = {}
        for ci, ce in enumerate(cur_edge_cols):
            if ce not in ecards:
                print(ce + " not found!")
                continue
            newecards[curattrs[ci]] = ecards[ce]

        if len(newecards) > 0:
            einfo["join_key_cardinality"] = newecards

        #continue

        for ei, esql in enumerate(execsqls):
            try:
                cursor.execute(esql)
                outputs = cursor.fetchall()
                ecards[curattrs[ei]] = {}
                actual_card = outputs[0][0]
                ecards[curattrs[ei]]["actual"] = actual_card

                esql2 = pgsqls[ei]
                cursor.execute(esql2)
                output = cursor.fetchall()
                card = pg_est_aggr_from_explain(output)
                # print("Explain: ", card)

                ecards[curattrs[ei]]["expected"] = card

                print(subset, node_card, node_pg, cur_edge_cols[ei],
                        actual_card, card)

            except Exception as e:
                print(e)
                print(esql)
                continue
                # pdb.set_trace()

    if fn is not None:
        update_qrep(qrep)
        save_sql_rep(fn, qrep)
        print("saved new qrep!")

    cursor.close()
    con.close()
    return qrep

def main():
    print("query dir: ", args.query_dir)
    fns = list(glob.glob(args.query_dir + "/*"))
    print("number of files: ", len(fns))
    fns.sort()
    par_args = []
    for i, fn in enumerate(fns):
        print(fn)
        if i >= args.num_queries and args.num_queries != -1:
            break

        if (".pkl" not in fn and ".sql" not in fn):
            continue

        if ".pkl" in fn:
            qrep = load_sql_rep(fn)
        else:
            assert False

        if args.no_parallel:
            get_distinct_joinkeys(qrep, args.card_type, args.key_name, args.db_host,
                    args.db_name, args.user, args.pwd, args.port,fn,
                     i, args.sample_num, args.sampling_type)
            continue

        par_func = get_distinct_joinkeys
        par_args.append((qrep, args.card_type, args.key_name, args.db_host,
                args.db_name, args.user, args.pwd, args.port,
                fn, i,
                args.sample_num,
                args.sampling_type))

    random.shuffle(par_args)

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

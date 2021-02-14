import sys
import os
import pdb
sys.path.append(".")
import pickle
from db_utils.utils import *
from db_utils.query_storage import *
import time

inp_dir = sys.argv[1]
USER = "ubuntu"
PWD = ""
DB = "imdb"
DB_HOST = "localhost"
PORT = 5432

# DB_KEYS=["1960", "1970", "1980", "1990"]
DB_KEYS=["1950",""]

for dbk in DB_KEYS:
    print(dbk)
    db_name = DB + str(dbk)
    ckey = dbk + "cardinality"
    for tdir in os.listdir(inp_dir):
        print(tdir)
        if "1a" != tdir:
            continue
        start = time.time()
        fns = os.listdir(inp_dir + "/" + tdir)
        qrep = load_sql_rep(inp_dir + "/" + tdir + "/" + fns[0])
        totals = get_all_totals(qrep, USER, PWD, db_name, DB_HOST, PORT,
                use_explain=True)

        updated_totals = False
        for fn in fns:
            if "1a766.pkl" in fn:
                print("found correct file!")
                pdb.set_trace()
            qfn = inp_dir + "/" + tdir + "/" + fn
            qrep = load_sql_rep(qfn)
            # update the total
            for subset in qrep["subset_graph"].nodes():
                if subset == SOURCE_NODE:
                    continue
                assert subset in totals
                if ckey not in qrep["subset_graph"].nodes()[subset]:
                    continue

                cards = qrep["subset_graph"].nodes()[subset][ckey]
                if "actual" not in cards:
                    qrep["subset_graph"].nodes()[subset][ckey]["total"] = \
                            totals[subset]
                    continue
                actual = cards["actual"]
                if actual > totals[subset]:
                    # print(subset)
                    # print("actual > totals!")
                    updated_totals = True
                    totals[subset] = actual

                qrep["subset_graph"].nodes()[subset][ckey]["total"] = \
                        totals[subset]

            # with open(qfn, 'wb') as fp:
                # pickle.dump(qrep, fp, protocol=pickle.HIGHEST_PROTOCOL)

            print("saving qrep!")
            save_sql_rep(qfn, qrep)

        if updated_totals:
            for fn in fns:
                qfn = inp_dir + "/" + tdir + "/" + fn
                qrep = load_sql_rep(qfn)
                # update the total
                for subset in qrep["subset_graph"].nodes():
                    if subset == SOURCE_NODE:
                        continue
                    assert subset in totals
                    if ckey not in qrep["subset_graph"].nodes()[subset]:
                        continue
                    cards = qrep["subset_graph"].nodes()[subset][ckey]
                    if "actual" not in cards:
                        continue
                    actual = cards["actual"]
                    if actual > totals[subset]:
                        print(subset, actual, totals[subset])
                        pdb.set_trace()
                    qrep["subset_graph"].nodes()[subset][ckey]["total"] = \
                            totals[subset]

                save_sql_rep(qfn, qrep)
                # with open(qfn, 'wb') as fp:
                    # pickle.dump(qrep, fp, protocol=pickle.HIGHEST_PROTOCOL)


        print("saved all the queries with updated totals for ", tdir)
        print("took: ", time.time() - start)

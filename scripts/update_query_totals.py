import sys
import os
import pdb
sys.path.append(".")
import pickle
from db_utils.utils import *
from db_utils.query_storage import *
import time

inp_dir = sys.argv[1]

for tdir in os.listdir(inp_dir):
    if "6" in tdir:
        continue
    start = time.time()
    fns = os.listdir(inp_dir + "/" + tdir)
    qrep = load_sql_rep(inp_dir + "/" + tdir + "/" + fns[0])
    totals = get_all_totals(qrep)

    for fn in fns:
        qfn = inp_dir + "/" + tdir + "/" + fn
        qrep = load_sql_rep(qfn)
        # update the total
        for subset in qrep["subset_graph"].nodes():
            assert subset in totals
            qrep["subset_graph"].nodes()[subset]["cardinality"]["total"] = \
                    totals[subset]

        with open(qfn, 'wb') as fp:
            pickle.dump(qrep, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print("saved all the queries with updated totals for ", tdir)
    print("took: ", time.time() - start)
    # pdb.set_trace()

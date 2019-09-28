from db_utils.utils import *

FN = "./templates/myjob/26c.sql"

def test(fn):
    with open(fn, "r") as f:
        sql = f.read()
    cols, pred_types, pred_vals = extract_predicates(sql)
    cols2, pred_types2, pred_vals2 = extract_predicates2(sql)
    pdb.set_trace()

test(FN)


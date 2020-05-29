import sys
import os
from utils.utils import *
import pdb

def get_samples_per_plan(df):
    keys = []
    plans = set(df["plan"])
    for plan in plans:
        cur_df = df[df["plan"] == plan]
        cur_df = cur_df.sample(1)
        keys.append(cur_df["sql_key"].values[0])

    return keys

res_dir = sys.argv[1]
print("results directory is: ", res_dir)
assert os.path.exists(res_dir)
fn = res_dir + "/plan_pg_err.pkl"
assert os.path.exists(fn)
df = load_object(fn)
df = df[df["samples_type"] == "test"]
keys = get_samples_per_plan(df)
print("num unique plans: ", len(keys))
output_fn = res_dir + "/unique_plan_keys.pkl"
save_object(output_fn, keys)

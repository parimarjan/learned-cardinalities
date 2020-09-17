import sys
import os
sys.path.append(".")
from utils.utils import *
import pdb

def get_samples_per_plan(df, existing_keys = None):
    keys = []
    plans = set(df["plan"])
    print("num unique plans: ", len(plans))
    for plan in plans:
        cur_df = df[df["plan"] == plan]
        # if existing_keys is not None:
            # found_key = False
            # for key in cur_df["sql_key"]:
                # if key in existing_keys:
                    # keys.append(key)
                    # found_key = True
                    # break
            # if found_key:
                # continue
        # cur_df = cur_df.sample(1)

        cur_df = cur_df.sort_values(by="cost", ascending=False)
        keys.append(cur_df["sql_key"].values[0])

    return keys

existing_keys = load_object("all_rt_keys.pkl")
if existing_keys is not None:
    print("num existing keys: ", len(existing_keys))
res_dir = sys.argv[1]

print("results directory is: ", res_dir)
assert os.path.exists(res_dir)
# fn = res_dir + "/nested_loop_index7_jerr.pkl"
fn = res_dir + "/cm1_jerr.pkl"
assert os.path.exists(fn)
df = load_object(fn)
# df = df[df["samples_type"] == "test"]

keys = get_samples_per_plan(df, existing_keys)

print("num unique plans: ", len(keys))
output_fn = res_dir + "/unique_plan_keys.pkl"
save_object(output_fn, keys)

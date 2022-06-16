from utils.utils import *
import pickle
import os
import glob
import sys
from pathlib import Path
import pdb

# FILES_TO_UPDATE = ["cm1_jerr.pkl", "nested_loop_index7_jerr.pkl"]
FILES_TO_UPDATE = ["plan_pg_err.pkl"]
base_dir = sys.argv[1]
pathlist = Path(base_dir).glob('**/*.pkl')

for i, path in enumerate(pathlist):
    if i % 10 == 0:
        print(i)
    file_name = os.path.basename(path)
    try:
        if file_name in FILES_TO_UPDATE:
            df = load_object(path)
            df["exec_sql"] = None
            save_object(path, df)
            print("updated path: ", path)
    except:
        print("failed for: ", path)
        pass

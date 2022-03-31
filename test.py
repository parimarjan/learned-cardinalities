import pickle
import os
import pdb

# for the sampling from the 1% version of the db
SPERCENTAGE = 1.0
QDIR = "./binned_cards/{}/job/all_job/".format(SPERCENTAGE)

fns = os.listdir(QDIR)
for fn in fns:
    fpath = os.path.join(QDIR, fn)
    print(fpath)
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    for i, alias in enumerate(data["all_aliases"]):
        column = data["all_columns"][i]
        cards = data["results"][i][0]
        print(alias, column, cards)

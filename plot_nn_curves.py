from utils.utils import *
import pdb
import klepto
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import argparse

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=False,
            default="./nn_training_cache")
    return parser.parse_args()

args = read_flags()
cache_dir = args.results_dir

cache = klepto.archives.dir_archive(cache_dir,
        cached=True, serialized=True)
cache.load()

# df = pd.DataFrame()
all_data = {}
all_data["iter"] = []
# all_data["qerr"] = []
# all_data["join-loss"] = []
all_data["loss_type"] = []
all_data["loss"] = []
all_data["jl_variant"] = []
all_data["optimizer_name"] = []
all_data["optimizer_obj"] = []

for k in cache:
    print(k)
    data = cache[k]
    print("eval iter: ", data["kwargs"]["eval_iter"])
    optimizer_name = data["kwargs"]["optimizer_name"]
    jl_variant = data["kwargs"]["jl_variant"]
    for loss_type, losses in data["eval"].items():
        for num_iter, loss in losses.items():
            if jl_variant == 0:
                opt_obj = "qerr-loss"
            elif jl_variant == 1:
                opt_obj = "join-loss-cm1"
            elif jl_variant == 2:
                opt_obj = "join-loss-cm2"
            elif jl_variant == 3:
                opt_obj = "join-loss-sort"
            elif jl_variant == 4:
                opt_obj = "join-loss-sort-indexes"
            else:
                continue

            all_data["iter"].append(num_iter)
            all_data["loss"].append(loss)
            all_data["loss_type"].append(loss_type)
            all_data["optimizer_name"].append(optimizer_name)
            all_data["optimizer_obj"].append(opt_obj)
            all_data["jl_variant"].append(jl_variant)

df = pd.DataFrame(all_data)
pdb.set_trace()

# skip the first entry, since it is too large
df = df[df["iter"] != 0]

pdf = PdfPages("test.pdf")

# fig, axs = plt.subplots(1,2)
jl_df = df[df["loss_type"] == "join-loss"]
qerr_df = df[df["loss_type"] == "qerr"]
max_loss = max(qerr_df["loss"])
ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj",
        style="optimizer_obj",
        data=qerr_df)

ax.set_ylim(bottom=0, top=max_loss)
plt.title("Q-Error")
plt.tight_layout()
pdf.savefig()
plt.clf()

max_loss = max(jl_df["loss"])
min_loss = min(jl_df["loss"])
print("max loss df: ", max_loss)

ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj", style="optimizer_obj",
        data=jl_df)
ax.set_ylim(bottom=min_loss, top=max_loss)
plt.title("Join-Loss")
plt.tight_layout()
pdf.savefig()
plt.clf()

opt_df = jl_df[jl_df["jl_variant"] != 0]
# going to do ams v/s adam plot
ax = sns.lineplot(x="iter", y="loss", hue="optimizer_name",
        style="optimizer_name",
        data=opt_df)
ax.set_ylim(bottom=min_loss, top=max_loss)
plt.title("Adam v/s AMSGrad")
plt.tight_layout()
pdf.savefig()
plt.clf()
pdf.close()


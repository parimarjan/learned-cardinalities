from utils.utils import *
import pdb
import klepto
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# cache_dir = "./nn_training_cache_backup"
cache_dir = "/data/pari/gpu_results/nn_training_cache"
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
    # print(k)
    data = cache[k]
    # print(data["kwargs"])
    optimizer_name = data["kwargs"]["optimizer_name"]
    jl_variant = data["kwargs"]["jl_variant"]
    for loss_type, losses in data["eval"].items():
        for num_iter, loss in losses.items():
            if jl_variant == 0:
                opt_obj = "qerr-loss"
            elif jl_variant == 1:
                opt_obj = "join-loss1"
            elif jl_variant == 2:
                opt_obj = "join-loss2"
            else:
                continue

            all_data["iter"].append(num_iter)
            all_data["loss"].append(loss)
            all_data["loss_type"].append(loss_type)
            all_data["optimizer_name"].append(optimizer_name)
            all_data["optimizer_obj"].append(opt_obj)
            all_data["jl_variant"].append(jl_variant)

df = pd.DataFrame(all_data)
# pdb.set_trace()

# skip the first entry, since it is too large
df = df[df["iter"] != 0]

pdf = PdfPages("test.pdf")

# fig, axs = plt.subplots(1,2)
jl_df = df[df["loss_type"] == "join-loss"]
qerr_df = df[df["loss_type"] == "qerr"]

ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj",
        style="optimizer_obj",
        data=qerr_df)

ax.set_ylim(bottom=0, top=100)
plt.title("Q-Error")
plt.tight_layout()
pdf.savefig()
plt.clf()

ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj", style="optimizer_obj",
        data=jl_df)
ax.set_ylim(bottom=1, top=100)
plt.title("Join-Loss")
plt.tight_layout()
pdf.savefig()
plt.clf()

opt_df = jl_df[jl_df["jl_variant"] != 0]
pdb.set_trace()
## going to do ams v/s adam plot
ax = sns.lineplot(x="iter", y="loss", hue="optimizer_name",
        style="optimizer_name",
        data=opt_df)
ax.set_ylim(bottom=1, top=100)
plt.title("Adam v/s AMSGrad")
plt.tight_layout()
pdf.savefig()
plt.clf()

pdf.close()


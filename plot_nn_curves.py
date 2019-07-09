from utils.utils import *
import pdb
import klepto
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

cache_dir = "./nn_training_cache_backup"
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
all_data["use_jl"] = []
all_data["optimizer_name"] = []
all_data["optimizer_obj"] = []

for k in cache:
    # print(k)
    data = cache[k]
    # print(data["kwargs"])
    optimizer_name = data["kwargs"]["optimizer_name"]
    use_jl = data["kwargs"]["use_jl"]
    for loss_type, losses in data["eval"].items():
        for num_iter, loss in losses.items():
            all_data["iter"].append(num_iter)
            all_data["loss"].append(loss)
            all_data["loss_type"].append(loss_type)
            all_data["optimizer_name"].append(optimizer_name)
            if use_jl:
                opt_obj = "join-loss"
            else:
                opt_obj = "qerr-loss"

            all_data["optimizer_obj"].append(opt_obj)

            all_data["use_jl"].append(use_jl)

df = pd.DataFrame(all_data)

# skip the first entry, since it is too large
df = df[df["iter"] != 0]

pdf = PdfPages("test.pdf")

# fig, axs = plt.subplots(1,2)
jl_df = df[df["loss_type"] == "join-loss"]
qerr_df = df[df["loss_type"] == "qerr"]

ax = sns.lineplot(x="iter", y="loss", hue="optimizer_obj", style="optimizer_obj",
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

pdf.close()

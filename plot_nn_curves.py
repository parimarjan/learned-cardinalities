from utils.utils import *
import pdb
import klepto

cache_dir = "./nn_training_cache"
cache = klepto.archives.dir_archive(cache_dir,
        cached=True, serialized=True)
cache.load()
for k in cache:
    print(k)
    data = cache[k]
    print(data["kwargs"])
    print(data["eval"])
    pdb.set_trace()

# fn = "training-NN2jl-0hl-1hlm-0.5.dict"
# data = load_object(fn)
# print(data.keys())
# pdb.set_trace()


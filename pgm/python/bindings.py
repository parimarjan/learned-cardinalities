from ctypes import *

import os
import platform
import copy
import pkg_resources
import numpy as np
from ctypes import c_long

system = platform.system()
if system == 'Linux':
    lib_file = "libpgm.so"
else:
    lib_file = "libpgm.dylib"

lib_file = pkg_resources.resource_filename(__name__, lib_file)

pgm = CDLL(lib_file, mode=RTLD_GLOBAL)
print("loaded library!")
pgm.test(5)

# TODO: for testing, use something like this to pass int ** to C code
# https://stackoverflow.com/questions/13436218/how-can-i-pass-an-array-of-integer-pointers-to-a-function-in-c-library-from-py
# pgm.init(mat)
# l = [[1],[1,2],[1,2,3]]
# entrylist = []
# lengths = []

# for sub_l in l:
    # entrylist.append((c_int*len(sub_l))(*sub_l))
    # lengths.append(c_int(len(sub_l)))

# c_l = (POINTER(c_int) * len(entrylist))(*entrylist)
# c_lengths = (c_int * len(l))(*lengths)
# print(c_l)
# # test this
# pgm.test_inference.restype = c_double
# test2 = pgm.test_inference(c_l, c_lengths, len(l))  #here we also pass the sizes of all the arrays
# print(test2)

a = [[0,1,2], [0,1,2]]
a = np.array(a, dtype=np.int32)
counts = [100, 20, 30]
counts = np.array(counts, dtype=np.int32)
# pgm.test_init(mat.ctypes.data_as(c_void_p), c_long(mat.shape[0]),
        # c_long(mat.shape[1]))
pgm.py_init(a.ctypes.data_as(c_void_p), c_long(a.shape[0]), c_long(a.shape[1]),
        counts.ctypes.data_as(c_void_p), c_long(counts.shape[0]))
print("py init call worked!")
pgm.py_train()
print("py_train worked!")

l = [[0,1,2],[1,2]]
entrylist = []
lengths = []

for sub_l in l:
    entrylist.append((c_int*len(sub_l))(*sub_l))
    lengths.append(c_int(len(sub_l)))

c_l = (POINTER(c_int) * len(entrylist))(*entrylist)
c_lengths = (c_int * len(l))(*lengths)
print(c_l)
# # test this
pgm.py_eval.restype = c_double
est = pgm.py_eval(c_l, c_lengths, len(l), 0, c_double(1.0))  #here we also pass the sizes of all the arrays
print(est)

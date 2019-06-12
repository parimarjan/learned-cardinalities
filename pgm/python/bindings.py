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
    assert False

lib_file = pkg_resources.resource_filename(__name__, lib_file)

pgm = CDLL(lib_file, mode=RTLD_GLOBAL)
print("loaded library!")
pgm.test(5)

mat = np.ones((5,7), dtype=np.int32)
pgm.test_init(mat.ctypes.data_as(c_void_p), c_long(mat.shape[0]),
        c_long(mat.shape[1]))
# pgm.init(mat)

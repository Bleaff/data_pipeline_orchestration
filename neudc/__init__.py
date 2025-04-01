__version__ = "0.0.1"

import os

from neudc.utils import NUMBA_DISABLE_JIT

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMBA_DISABLE_JIT"] = "1" if NUMBA_DISABLE_JIT else "0"

"""One BLAS/OpenMP thread per worker: enforced (not setdefault) and reported from the live thread pools."""

import ctypes
import glob
import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def force_env():
    """Call BEFORE importing numpy/torch. Overrides any inherited values."""
    for v in THREAD_VARS:
        os.environ[v] = "1"


def _openblas():
    import numpy
    libs = glob.glob(os.path.join(os.path.dirname(numpy.__file__), "..", "numpy.libs", "*openblas*"))
    if not libs:
        return None, None
    lib = ctypes.CDLL(libs[0])
    for get, put in (("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"),
                     ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
                     ("openblas_get_num_threads", "openblas_set_num_threads")):
        if hasattr(lib, get):
            return getattr(lib, get), getattr(lib, put)
    return None, None


def enforce_and_report():
    """Force one thread in the live pools (numpy OpenBLAS, torch) and return what they actually report."""
    get, put = _openblas()
    if put is not None:
        put(1)
    out = {"env": {v: os.environ.get(v) for v in THREAD_VARS},
           "numpy_openblas_threads": int(get()) if get is not None else None}
    try:
        import torch
        torch.set_num_threads(1)
        out["torch_threads"] = torch.get_num_threads()
    except ImportError:
        out["torch_threads"] = None
    return out

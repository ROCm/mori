# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import sys
import importlib.util
import ctypes
import threading

try:
    import torch  # noqa: F401
except ImportError:
    pass

cur_dir = os.path.dirname(os.path.abspath(__file__))
mori_lib_dir = os.path.abspath(os.path.join(cur_dir, ".."))

_preload_libs = ["libmori_application.so", "libmori_io.so"]
for _lib_name in _preload_libs:
    _lib_full_path = os.path.join(mori_lib_dir, _lib_name)
    if os.path.exists(_lib_full_path):
        ctypes.CDLL(_lib_full_path, mode=ctypes.RTLD_GLOBAL)

lib_path = os.path.abspath(os.path.join(cur_dir, "../libmori_pybinds.so"))

spec = importlib.util.spec_from_file_location("libmori_pybinds", lib_path)
module = importlib.util.module_from_spec(spec)
sys.modules["libmori_pybinds"] = module
spec.loader.exec_module(module)

from libmori_pybinds import *

# UMBP's bindings are a separate module because it links gRPC/Protobuf, which
# mori.ops/io/shmem must not need. A name not found above is looked up there,
# loading it on first use, so `from mori.cpp import UMBPClient` and UMBP's
# exported enum values still resolve. If it cannot load, the lookup is an
# AttributeError so hasattr()/getattr(default) probes stay safe; mori.umbp calls
# _load_umbp() itself to report why.
_UMBP_LIB = "libmori_umbp_pybinds"
_umbp = None
_umbp_error = None
_umbp_lock = threading.Lock()


def _import_umbp():
    path = os.path.join(mori_lib_dir, _UMBP_LIB + ".so")
    if not os.path.exists(path):
        raise ImportError(f"mori was built without UMBP ({_UMBP_LIB}.so is absent)")
    spec = importlib.util.spec_from_file_location(_UMBP_LIB, path)
    try:
        # module_from_spec is where an extension module is dlopen'ed.
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except ImportError as e:
        hint = ""
        if any(k in str(e) for k in ("grpc", "gpr", "protobuf")):
            hint = " UMBP needs the gRPC/Protobuf runtime (Ubuntu 22.04: apt install libgrpc++1)."
        raise ImportError(f"cannot load {_UMBP_LIB}: {e}.{hint}") from e
    sys.modules[_UMBP_LIB] = mod
    return mod


def _load_umbp():
    """UMBP's module, loaded once; a failure is cached and re-raised as ImportError."""
    global _umbp, _umbp_error
    with _umbp_lock:  # a pybind11 module must not be exec'd twice
        if _umbp is None and _umbp_error is None:
            try:
                _umbp = _import_umbp()
            except ImportError as e:
                _umbp_error = e
    if _umbp_error is not None:
        raise ImportError(str(_umbp_error)) from _umbp_error
    return _umbp


def __getattr__(name):
    if not name.startswith("_"):
        try:
            umbp = _load_umbp()
        except ImportError:
            umbp = None
        if umbp is not None and hasattr(umbp, name):
            return getattr(umbp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

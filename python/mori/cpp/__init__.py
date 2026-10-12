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

# UMBP's bindings are a separate module because it links gRPC/Protobuf. It loads
# on first access to a UMBP name, so mori.ops/io/shmem import without gRPC and
# `from mori.cpp import UMBPClient` still works.
_UMBP_LIB = "libmori_umbp_pybinds"
_umbp = None


def _is_umbp_name(name):
    return name.startswith("UMBP") or name == "CacheRemoteAdmission"


def _load_umbp():
    global _umbp
    if _umbp is not None:
        return _umbp
    path = os.path.join(mori_lib_dir, _UMBP_LIB + ".so")
    if not os.path.exists(path):
        raise ImportError(
            f"mori was built without UMBP ({_UMBP_LIB}.so is absent); rebuild with BUILD_UMBP=ON"
        )
    spec = importlib.util.spec_from_file_location(_UMBP_LIB, path)
    try:
        # module_from_spec is where an extension module is dlopen'ed.
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except ImportError as e:
        raise ImportError(
            f"mori.umbp needs the gRPC/Protobuf runtime (e.g. `apt install libgrpc++1`): {e}"
        ) from e
    sys.modules[_UMBP_LIB] = mod
    _umbp = mod
    return mod


def __getattr__(name):
    if _is_umbp_name(name):
        return getattr(_load_umbp(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

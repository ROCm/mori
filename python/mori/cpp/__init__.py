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

cur_dir = os.path.dirname(os.path.abspath(__file__))
mori_lib_dir = os.path.abspath(os.path.join(cur_dir, ".."))

_torch_lib = os.path.join(mori_lib_dir, "libmori_pybinds.so")
_core_lib = os.path.join(mori_lib_dir, "libmori_core_pybinds.so")

if os.path.exists(_torch_lib):
    # Torch-enabled build: must initialize libtorch before loading
    import torch  # noqa: F401
    lib_path = _torch_lib
    lib_name = "libmori_pybinds"
elif os.path.exists(_core_lib):
    # Core-only build: no Torch dependency
    lib_path = _core_lib
    lib_name = "libmori_core_pybinds"
else:
    raise ImportError(
        f"Cannot find mori pybind library. Looked for:\n"
        f"  {_torch_lib}\n"
        f"  {_core_lib}\n"
    )

spec = importlib.util.spec_from_file_location(lib_name, lib_path)
module = importlib.util.module_from_spec(spec)
sys.modules[lib_name] = module
spec.loader.exec_module(module)

from importlib import import_module
_m = import_module(lib_name)
globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith('_')})

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

# PyTorch must be imported first to initialize its runtime
# (libtorch, libc10, etc.) before loading mori's native extensions
import torch  # noqa: F401

cur_dir = os.path.dirname(os.path.abspath(__file__))
mori_lib_dir = os.path.abspath(os.path.join(cur_dir, ".."))

# Pre-load dependent shared libraries
# This is necessary because libmori_pybinds.so depends on libmori_application.so etc.
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

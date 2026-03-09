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

try:
  import jax  # noqa: F401
  from mori.cpp import mori_ep_handler, mori_ep_type_info, preload_kernels
  from .ops import *

  # note this used to work for jax-0.7.1
  # jax.ffi.register_ffi_type_id("mori_ep", mori_ep_type_id(), platform="ROCM")
  jax.ffi.register_ffi_type("mori_ep", mori_ep_type_info(), platform="ROCM")
  jax.ffi.register_ffi_target("mori_ep", mori_ep_handler(), platform="ROCM")
  
except ImportError as e:
  #print(f"Error importing mori.jax: {e}")
  pass

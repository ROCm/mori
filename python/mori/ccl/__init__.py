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
    from .collective import All2allSdma
    from .collective import AllgatherSdma
    from .collective import AllreduceSdma

    # NCCL/RCCL-style C++ AllGather-into-tensor dispatcher.  The class and its
    # DataType enum are implemented entirely in C++ (see
    # ``include/mori/collective/allgather/allgather_into_tensor.hpp`` and
    # ``src/collective/core/allgather_into_tensor.cpp``); we only re-export the
    # pybind11 symbols here so callers can do
    # ``from mori.ccl import AllGatherIntoTensor, DataType``.
    from mori import cpp as _mori_cpp

    AllGatherIntoTensor = _mori_cpp.AllGatherIntoTensor
    DataType = _mori_cpp.DataType
    size_of = _mori_cpp.size_of

    __all__ = [
        "All2allSdma",
        "AllgatherSdma",
        "AllreduceSdma",
        "AllGatherIntoTensor",
        "DataType",
        "size_of",
    ]
except (ImportError, AttributeError):
    __all__ = [
        "All2allSdma",
        "AllgatherSdma",
        "AllreduceSdma",
    ]

    def __getattr__(name: str):
        raise ImportError(f"mori.ccl.{name} is not available — not yet ported to JIT.")

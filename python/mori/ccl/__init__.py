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

# Host-proxy hierarchical all-gather (persistent CPU-posted transport). Pure
# Python (depends only on mori.io, imported lazily in its ctor), so it stays
# importable whether or not the C++ collective bindings below are available.
from .host_proxy_ag import HostProxyHierAllGather

try:
    from .collective import All2allSdma
    from .collective import AllgatherSdma
    from .collective import AllreduceSdma
    from .collective import InterNodeRingAllgather
    from .collective import IntraNodeSubGroupAllgatherSdma
    from .collective import IntraNodeSubGroupBroadcastSdma

    # Depends on ``.collective`` (the C++ bindings); imported inside the guard
    # so a missing .so does not break the whole ``mori.ccl`` package.
    from .hier_allgather import HierAllGather, hier_allgather_reference

    # C++ AllGather-into-tensor dispatcher. The class and its DataType enum are
    # implemented in C++ (allgather_into_tensor.hpp / .cpp); re-export the
    # pybind11 symbols so callers can
    # ``from mori.ccl import AllGatherIntoTensor, DataType``.
    from mori import cpp as _mori_cpp

    AllGatherIntoTensor = _mori_cpp.AllGatherIntoTensor
    DataType = _mori_cpp.DataType
    size_of = _mori_cpp.size_of

    __all__ = [
        "All2allSdma",
        "AllgatherSdma",
        "AllreduceSdma",
        "InterNodeRingAllgather",
        "IntraNodeSubGroupAllgatherSdma",
        "IntraNodeSubGroupBroadcastSdma",
        "AllGatherIntoTensor",
        "DataType",
        "size_of",
        "HierAllGather",
        "hier_allgather_reference",
        "HostProxyHierAllGather",
    ]
except (ImportError, AttributeError):
    # C++ bindings unavailable: only the pure-Python reference specs import.
    # Device classes are exposed via ``__getattr__`` so accessing them raises a
    # clear ImportError rather than AttributeError.
    from .hier_allgather import hier_allgather_reference, inter_node_ring_reference

    __all__ = [
        "hier_allgather_reference",
        "inter_node_ring_reference",
        "HostProxyHierAllGather",
    ]

    def __getattr__(name: str):
        raise ImportError(f"mori.ccl.{name} is not available — not yet ported to JIT.")

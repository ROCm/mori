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
#
# The dispatch/combine (v2) op. The base class and config live in
# dispatch_combine_op and import no kernel backend, so this package imports
# without FlyDSL: the flydsl backend (needs `pip install amd_mori[flydsl]`) and
# the hip backend are each imported lazily, only when selected. mori.ops still
# lazy-loads this submodule.
from .dispatch_combine_op import (
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
    EpDispatchRoutingHandle,
)

__all__ = [
    "EpDispatchCombineConfig",
    "EpDispatchCombineOp",
    "EpDispatchRoutingHandle",
]

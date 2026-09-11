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
# Fused fp8 GEMM + all-reduce over cco SDMA. See README.md for the design and
# the measurements.
#
# `layout` is pure arithmetic and imports nothing, so it stays eagerly
# available; everything else needs FlyDSL (`pip install amd_mori[flydsl]`) and
# is imported lazily, so `import mori.ops.gemm_ar` works without it and only
# touching the op or the kernels raises.
import importlib

from .layout import ArConfig, MAX_WORLD, select_stage
from ._shuffle import preshuffle_b

_LAZY = {
    "GemmAllReduceOp": "op",
    "counter_chunks": "op",
    "padded_m": "op",
    "supports": "op",
    "DEFAULT_BLOCK_M": "op",
    "DEFAULT_BLOCK_N": "op",
    "MAX_CHUNKS": "op",
    "SCALE_BLOCK_K": "op",
    "compile_fused_gemm_scatter": "kernels_fused",
    "build_sdma_phases": "kernels_sdma",
    "build_sdma_ar": "kernels_sdma",
    "build_lsa_ar": "kernels_lsa",
}

__all__ = [
    "ArConfig",
    "GemmAllReduceOp",
    "MAX_WORLD",
    "build_lsa_ar",
    "build_sdma_ar",
    "build_sdma_phases",
    "compile_fused_gemm_scatter",
    "counter_chunks",
    "padded_m",
    "preshuffle_b",
    "select_stage",
    "supports",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

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
# Fused fp8 GEMM + all-to-all over cco. See README.md for the design and the
# measurements.
#
# The GEMM itself is `gemm_ar`'s: this op differs in where C goes, not in how it
# is computed, so `_gemm_a8w8_8wave.py`, `_compat.py` and `_shuffle.py` are
# imported from there rather than duplicated.
#
# `layout` is pure arithmetic and imports nothing, so it stays eagerly
# available; everything else needs FlyDSL (`pip install amd_mori[flydsl]`) and
# is imported lazily, so `import mori.ops.gemm_a2a` works without it and only
# touching the kernels raises.
import importlib

from .layout import A2aConfig, MAX_WORLD, a2a_config, counter_chunks
from ..gemm_ar import preshuffle_b

_LAZY = {
    "build_lsa_a2a": "kernels_lsa",
    "build_lsa_barrier": "kernels_lsa",
    "build_sdma_phases": "kernels_sdma",
    "compile_fused_gemm_a2a": "kernels_fused",
    "compile_gemm_local": "kernels_fused",
}

__all__ = [
    "A2aConfig",
    "MAX_WORLD",
    "a2a_config",
    "build_lsa_a2a",
    "build_lsa_barrier",
    "build_sdma_phases",
    "compile_fused_gemm_a2a",
    "compile_gemm_local",
    "counter_chunks",
    "preshuffle_b",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

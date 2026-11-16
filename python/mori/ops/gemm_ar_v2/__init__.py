# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License (see the repository's LICENSE).
"""BF16 GEMM + all-reduce over cco SDMA, written for the gemm_ar_bf16 case.

``layout`` is pure arithmetic; the op and the kernels need FlyDSL and are
imported lazily.
"""

import importlib

from .layout import BLOCK_M, BLOCK_N, Layout, Shape, padded_m


def __getattr__(name):
    if name == "GemmArV2":
        return importlib.import_module(".op", __name__).GemmArV2
    raise AttributeError(name)

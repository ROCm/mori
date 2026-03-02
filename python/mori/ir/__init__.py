# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
mori.ir — Mori device IR layer.

Provides framework-agnostic access to the shmem device bitcode and
function ABI metadata, plus framework-specific integration sub-packages
(``mori.ir.triton``, and in the future ``mori.ir.flydsl``, etc.).

Quick start (no framework dependency)::

    from mori.ir import find_bitcode
    bc = find_bitcode()   # path to libmori_shmem_device.bc
"""

from .bitcode import find_bitcode, get_bitcode_path
from .ops import MORI_DEVICE_FUNCTIONS, SIGNAL_SET, SIGNAL_ADD

__all__ = [
    "find_bitcode",
    "get_bitcode_path",
    "MORI_DEVICE_FUNCTIONS",
    "SIGNAL_SET",
    "SIGNAL_ADD",
]

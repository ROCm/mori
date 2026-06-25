# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""cco device-side bindings for DSL kernels.

Sub-packages target a specific DSL:

  * :mod:`mori.cco.device.flydsl` — FlyDSL (``@flyc.kernel``) bindings.

All bindings link against the cco device bitcode located by
:mod:`mori.cco.device.bitcode`.
"""

from mori.cco.device.bitcode import find_cco_bitcode, get_bitcode_path

__all__ = ["find_cco_bitcode", "get_bitcode_path"]

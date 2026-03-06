# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
#
# Based on PR #173 by Chao Chen <cchen104@amd.com>
"""Mori JAX integration -- XLA FFI custom call handlers for MoE ops."""

from mori.jax._ffi_registry import register_ffi_targets  # noqa: F401
from mori.jax.ops import EpDispatchCombineOp  # noqa: F401

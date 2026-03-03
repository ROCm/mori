# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Mori JAX integration -- XLA FFI custom call handlers for MoE ops."""

from mori.jax._ffi_registry import register_ffi_targets  # noqa: F401
from mori.jax.ops import EpDispatchCombineOp  # noqa: F401

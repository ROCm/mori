# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Load libmori_xla_ffi_ops.so and register FFI targets with JAX."""

import ctypes
import os
import pathlib

_lib = None
_FFI_TARGET_NAMES = [
    "mori_ep_dispatch",
    "mori_ep_combine",
    "mori_ep_dispatch_recv",
    "mori_ep_combine_recv",
    "mori_ep_reset",
]


def _find_library() -> str:
    """Locate libmori_xla_ffi_ops.so relative to this package."""
    pkg_dir = pathlib.Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent / "libmori_xla_ffi_ops.so",
        pkg_dir.parent.parent / "lib" / "libmori_xla_ffi_ops.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    env_path = os.environ.get("MORI_XLA_FFI_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    raise FileNotFoundError(
        "Cannot find libmori_xla_ffi_ops.so. "
        "Set MORI_XLA_FFI_LIB or install mori with ENABLE_XLA_FFI=ON."
    )


def _load_dependencies():
    """Pre-load shared libraries that libmori_xla_ffi_ops.so depends on."""
    pkg_dir = pathlib.Path(__file__).resolve().parent.parent
    for name in ["libmori_application.so", "libmori_io.so"]:
        lib_path = pkg_dir / name
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


def _load_library():
    global _lib
    if _lib is not None:
        return _lib
    _load_dependencies()
    lib_path = _find_library()
    _lib = ctypes.CDLL(lib_path)

    _lib.mori_ffi_create_handle.restype = ctypes.c_int64
    _lib.mori_ffi_create_handle.argtypes = [
        ctypes.c_int,  # rank
        ctypes.c_int,  # world_size
        ctypes.c_int,  # hidden_dim
        ctypes.c_int,  # scale_dim
        ctypes.c_int,  # scale_type_size
        ctypes.c_int,  # max_token_type_size
        ctypes.c_int,  # max_num_inp_token_per_rank
        ctypes.c_int,  # num_experts_per_rank
        ctypes.c_int,  # num_experts_per_token
        ctypes.c_int,  # warp_num_per_block
        ctypes.c_int,  # block_num
        ctypes.c_int,  # kernel_type
        ctypes.c_int,  # gpu_per_node
        ctypes.c_int,  # rdma_block_num
        ctypes.c_int,  # num_qp_per_pe
    ]

    _lib.mori_ffi_destroy_handle.restype = None
    _lib.mori_ffi_destroy_handle.argtypes = [ctypes.c_int64]

    return _lib


def register_ffi_targets():
    """Register all mori FFI handler symbols with JAX.

    Call this once at import time. Requires jax to be importable.
    """
    try:
        from jax._src.interpreters import mlir as jax_mlir
        from jax._src.lib import _jax
    except ImportError:
        raise ImportError(
            "jax is required to register FFI targets. Install jax first."
        )

    lib = _load_library()
    for name in _FFI_TARGET_NAMES:
        fn_ptr = ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
        capsule = _jax.pjrt_plugin_xla_cpu_buffer_to_capsule(fn_ptr, name)
        _jax.register_custom_call_target(name, capsule, "rocm")


def create_handle(
    rank: int,
    world_size: int,
    hidden_dim: int,
    scale_dim: int = 0,
    scale_type_size: int = 0,
    max_token_type_size: int = 4,
    max_num_inp_token_per_rank: int = 128,
    num_experts_per_rank: int = 1,
    num_experts_per_token: int = 2,
    warp_num_per_block: int = 1,
    block_num: int = 1,
    kernel_type: int = 0,
    gpu_per_node: int = 8,
    rdma_block_num: int = 0,
    num_qp_per_pe: int = 1,
) -> int:
    """Create an EpDispatchCombineHandle and return its integer ID."""
    lib = _load_library()
    return lib.mori_ffi_create_handle(
        rank, world_size, hidden_dim,
        scale_dim, scale_type_size,
        max_token_type_size, max_num_inp_token_per_rank,
        num_experts_per_rank, num_experts_per_token,
        warp_num_per_block, block_num,
        kernel_type, gpu_per_node,
        rdma_block_num, num_qp_per_pe,
    )


def destroy_handle(handle_id: int) -> None:
    """Destroy an EpDispatchCombineHandle by its ID."""
    lib = _load_library()
    lib.mori_ffi_destroy_handle(handle_id)

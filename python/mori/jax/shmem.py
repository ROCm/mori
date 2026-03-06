# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""JAX-native shmem initialization and array interop helpers.

Allows using mori shmem from JAX without any torch dependency.

Usage::

    import jax
    jax.distributed.initialize()

    from mori.jax.shmem import shmem_jax_init, jax_data_ptr, shmem_ptr_to_jax
    shmem_jax_init()

    # Register a jax array for RDMA
    from mori.shmem import api as shmem
    shmem.shmem_buffer_register(jax_data_ptr(arr), arr.nbytes)

    # Wrap shmem-allocated memory as jax.Array
    import jax.numpy as jnp
    ptr = shmem.shmem_malloc(size)
    arr = shmem_ptr_to_jax(ptr, (M, N), jnp.float32)
"""

import ctypes

from mori.tensor_utils import GpuTensorView, kDLFloat, kDLInt, kDLUInt, kDLBfloat


def shmem_jax_init():
    """Initialize shmem from JAX distributed runtime.

    Requires ``jax.distributed.initialize()`` to have been called first.
    Broadcasts UniqueId from process 0 to all processes.

    Returns:
        Status code (0 for success).
    """
    import jax
    import numpy as np
    from mori.shmem.api import (
        _ensure_shmem_module,
        shmem_get_unique_id,
        shmem_init_attr,
        MORI_SHMEM_INIT_WITH_UNIQUEID,
    )

    _ensure_shmem_module()

    rank = jax.process_index()
    world_size = jax.process_count()

    if rank == 0:
        uid_bytes = shmem_get_unique_id()
        uid_arr = np.frombuffer(uid_bytes, dtype=np.uint8).copy()
    else:
        uid_arr = None

    from jax.experimental.multihost_utils import broadcast_one_to_all

    uid_arr = broadcast_one_to_all(uid_arr)
    uid_bytes = uid_arr.tobytes()

    return shmem_init_attr(MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world_size, uid_bytes)


def jax_data_ptr(arr) -> int:
    """Extract raw GPU device pointer from a single-device jax.Array.

    Args:
        arr: A single-device jax.Array on GPU.

    Returns:
        Integer device pointer address.
    """
    import jax

    if hasattr(arr, "addressable_data"):
        buf = arr.addressable_data(0)
    else:
        buf = arr

    if hasattr(buf, "unsafe_buffer_pointer"):
        return buf.unsafe_buffer_pointer()

    dl_capsule = jax.dlpack.to_dlpack(buf)
    managed = ctypes.cast(
        ctypes.pythonapi.PyCapsule_GetPointer(
            ctypes.py_object(dl_capsule), b"dltensor"
        ),
        ctypes.POINTER(_DLManagedTensorCompat),
    )
    return managed.contents.dl_tensor.data
    

_JAX_DTYPE_TO_DL = None


def _init_jax_dtype_map():
    global _JAX_DTYPE_TO_DL
    if _JAX_DTYPE_TO_DL is not None:
        return
    import jax.numpy as jnp

    _JAX_DTYPE_TO_DL = {
        jnp.float32: (kDLFloat, 32, "<f4"),
        jnp.float64: (kDLFloat, 64, "<f8"),
        jnp.float16: (kDLFloat, 16, "<f2"),
        jnp.bfloat16: (kDLBfloat, 16, "<u2"),
        jnp.int32: (kDLInt, 32, "<i4"),
        jnp.int64: (kDLInt, 64, "<i8"),
        jnp.int8: (kDLInt, 8, "<i1"),
        jnp.uint8: (kDLUInt, 8, "<u1"),
        jnp.uint32: (kDLUInt, 32, "<u4"),
    }


def shmem_ptr_to_jax(ptr, shape, dtype, device_id=0):
    """Wrap a shmem-allocated GPU pointer as a jax.Array (zero-copy via DLPack).

    Args:
        ptr: int64 device pointer address (e.g. from shmem_malloc).
        shape: tuple of dimensions.
        dtype: jax.numpy dtype (e.g. jnp.float32).
        device_id: GPU device ordinal (default 0).

    Returns:
        A jax.Array backed by the given device memory.
    """
    import jax

    _init_jax_dtype_map()
    info = _JAX_DTYPE_TO_DL.get(dtype)
    if info is None:
        raise ValueError(f"Unsupported JAX dtype: {dtype}")
    dl_code, dl_bits, typestr = info

    view = GpuTensorView(
        ptr, shape,
        dl_dtype=(dl_code, dl_bits),
        typestr=typestr,
        device_id=device_id,
    )
    return jax.dlpack.from_dlpack(view)


def shmem_register_jax_array(arr) -> int:
    """Register a jax.Array for shmem RDMA operations.

    Args:
        arr: A single-device jax.Array on GPU.

    Returns:
        Status code (0 for success).
    """
    from mori.shmem.api import shmem_buffer_register

    ptr = jax_data_ptr(arr)
    return shmem_buffer_register(ptr, arr.nbytes)


# Minimal DLPack structures for pointer extraction fallback
class _DLTensorCompat(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", ctypes.c_int32 * 2),
        ("ndim", ctypes.c_int32),
        ("dtype", ctypes.c_uint8 * 4),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensorCompat(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensorCompat),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]

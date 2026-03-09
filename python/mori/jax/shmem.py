# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""JAX-native shmem initialization and array interop helpers.

Fully torch-free: uses ctypes to call shmem functions from
libmori_xla_ffi_ops.so directly, bypassing the mori pybind module
and its torch dependency.

Usage::

    import jax
    jax.distributed.initialize()

    from mori.jax.shmem import shmem_jax_init, jax_data_ptr, shmem_ptr_to_jax
    shmem_jax_init()

    # Allocate symmetric memory
    ptr = shmem_malloc(size)
    arr = shmem_ptr_to_jax(ptr, (M, N), jnp.float32)

    # Register a jax array for RDMA
    shmem_register_jax_array(arr)
"""

import ctypes

from mori.jax._ffi_registry import _load_library

# ---------------------------------------------------------------------------
# DLPack constants and GpuTensorView (self-contained, no mori.tensor_utils)
# ---------------------------------------------------------------------------
kDLCUDA = 2
kDLROCM = 10
kDLFloat = 2
kDLInt = 0
kDLUInt = 1
kDLBfloat = 4

_DL_GPU_DEVICE_TYPE = None


def _get_dl_gpu_device_type():
    global _DL_GPU_DEVICE_TYPE
    if _DL_GPU_DEVICE_TYPE is not None:
        return _DL_GPU_DEVICE_TYPE
    import os
    _DL_GPU_DEVICE_TYPE = kDLROCM if os.path.isdir("/opt/rocm") else kDLCUDA
    return _DL_GPU_DEVICE_TYPE


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64),
    ]


_DL_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


@_DL_DELETER
def _dl_noop_deleter(_ptr):
    pass


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", _DL_DELETER),
    ]


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class GpuTensorView:
    """Non-owning GPU array view with DLPack + __cuda_array_interface__."""

    def __init__(self, ptr, shape, *, dl_dtype, typestr, device_id=0):
        self._ptr = ptr
        self._shape = tuple(shape)
        self._dl_code, self._dl_bits = dl_dtype
        self._device_id = device_id
        self.__cuda_array_interface__ = {
            "data": (ptr, False), "shape": self._shape, "typestr": typestr, "version": 2,
        }
        self._dlpack_refs = None

    def __dlpack__(self, *, stream=None):
        ndim = len(self._shape)
        shape_arr = (ctypes.c_int64 * ndim)(*self._shape)
        managed = _DLManagedTensor()
        managed.dl_tensor.data = ctypes.c_void_p(self._ptr)
        managed.dl_tensor.device = _DLDevice(_get_dl_gpu_device_type(), self._device_id)
        managed.dl_tensor.ndim = ndim
        managed.dl_tensor.dtype = _DLDataType(self._dl_code, self._dl_bits, 1)
        managed.dl_tensor.shape = shape_arr
        managed.dl_tensor.strides = None
        managed.dl_tensor.byte_offset = 0
        managed.manager_ctx = None
        managed.deleter = _dl_noop_deleter
        self._dlpack_refs = (managed, shape_arr)
        return _PyCapsule_New(ctypes.byref(managed), b"dltensor", None)

    def __dlpack_device__(self):
        return (_get_dl_gpu_device_type(), self._device_id)


# ---------------------------------------------------------------------------
# Shmem C API wrappers (via ctypes, torch-free)
# ---------------------------------------------------------------------------

_shmem_lib_setup_done = False
MORI_SHMEM_UNIQUE_ID_BYTES = 128


def _setup_shmem_ctypes():
    global _shmem_lib_setup_done
    if _shmem_lib_setup_done:
        return
    lib = _load_library()

    lib.mori_ffi_shmem_get_unique_id.restype = ctypes.c_int
    lib.mori_ffi_shmem_get_unique_id.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    lib.mori_ffi_shmem_init_attr.restype = ctypes.c_int64
    lib.mori_ffi_shmem_init_attr.argtypes = [
        ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
    ]

    lib.mori_ffi_shmem_finalize.restype = ctypes.c_int64
    lib.mori_ffi_shmem_finalize.argtypes = []

    lib.mori_ffi_load_shmem_module.restype = ctypes.c_int64
    lib.mori_ffi_load_shmem_module.argtypes = [ctypes.c_char_p]

    lib.mori_ffi_shmem_module_init.restype = ctypes.c_int64
    lib.mori_ffi_shmem_module_init.argtypes = [ctypes.c_uint64]

    lib.mori_ffi_shmem_mype.restype = ctypes.c_int
    lib.mori_ffi_shmem_npes.restype = ctypes.c_int
    lib.mori_ffi_shmem_barrier_all.restype = None
    lib.mori_ffi_shmem_barrier_on_stream.restype = None
    lib.mori_ffi_shmem_barrier_on_stream.argtypes = [ctypes.c_int64]

    lib.mori_ffi_shmem_malloc.restype = ctypes.c_uint64
    lib.mori_ffi_shmem_malloc.argtypes = [ctypes.c_uint64]
    lib.mori_ffi_shmem_free.restype = None
    lib.mori_ffi_shmem_free.argtypes = [ctypes.c_uint64]

    lib.mori_ffi_shmem_buffer_register.restype = ctypes.c_int64
    lib.mori_ffi_shmem_buffer_register.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    lib.mori_ffi_shmem_buffer_deregister.restype = ctypes.c_int64
    lib.mori_ffi_shmem_buffer_deregister.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    lib.mori_ffi_shmem_ptr_p2p.restype = ctypes.c_uint64
    lib.mori_ffi_shmem_ptr_p2p.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]

    lib.mori_ffi_shmem_num_qp_per_pe.restype = ctypes.c_int
    lib.mori_ffi_shmem_init_flag_uniqueid.restype = ctypes.c_uint

    _shmem_lib_setup_done = True


def shmem_get_unique_id() -> bytes:
    _setup_shmem_ctypes()
    lib = _load_library()
    buf = (ctypes.c_uint8 * MORI_SHMEM_UNIQUE_ID_BYTES)()
    size = ctypes.c_int(0)
    lib.mori_ffi_shmem_get_unique_id(buf, ctypes.byref(size))
    return bytes(buf[:size.value])


def shmem_init_attr(flags, rank, nranks, uid_bytes):
    _setup_shmem_ctypes()
    lib = _load_library()
    uid_buf = (ctypes.c_uint8 * len(uid_bytes))(*uid_bytes)
    return lib.mori_ffi_shmem_init_attr(flags, rank, nranks, uid_buf, len(uid_bytes))


def shmem_finalize():
    _setup_shmem_ctypes()
    return _load_library().mori_ffi_shmem_finalize()


def shmem_malloc(size):
    _setup_shmem_ctypes()
    return _load_library().mori_ffi_shmem_malloc(size)


def shmem_free(ptr):
    _setup_shmem_ctypes()
    _load_library().mori_ffi_shmem_free(ptr)


def shmem_buffer_register(ptr, size):
    _setup_shmem_ctypes()
    return _load_library().mori_ffi_shmem_buffer_register(ptr, size)


def shmem_buffer_deregister(ptr, size):
    _setup_shmem_ctypes()
    return _load_library().mori_ffi_shmem_buffer_deregister(ptr, size)


def shmem_barrier_all():
    _setup_shmem_ctypes()
    _load_library().mori_ffi_shmem_barrier_all()


def shmem_barrier_on_stream(stream=0):
    _setup_shmem_ctypes()
    _load_library().mori_ffi_shmem_barrier_on_stream(stream)


def shmem_ptr_p2p(dest_ptr, my_pe, dest_pe):
    _setup_shmem_ctypes()
    return _load_library().mori_ffi_shmem_ptr_p2p(dest_ptr, my_pe, dest_pe)


def shmem_mype():
    _setup_shmem_ctypes()
    return _load_library().mori_ffi_shmem_mype()


def shmem_npes():
    _setup_shmem_ctypes()
    return _load_library().mori_ffi_shmem_npes()


_shmem_module_loaded = False


def _ensure_shmem_module():
    """JIT-compile and load the shmem device module."""
    global _shmem_module_loaded
    if _shmem_module_loaded:
        return
    _setup_shmem_ctypes()
    from mori.jit.core import compile_genco
    hsaco = compile_genco("shmem_kernels")
    _load_library().mori_ffi_load_shmem_module(hsaco.encode())
    _shmem_module_loaded = True


# ---------------------------------------------------------------------------
# JAX-native shmem init
# ---------------------------------------------------------------------------


def shmem_jax_init():
    """Initialize shmem from JAX distributed runtime.

    Requires ``jax.distributed.initialize()`` to have been called first.

    Returns:
        Status code (0 for success).
    """
    import jax
    import numpy as np

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

    _setup_shmem_ctypes()
    flag = _load_library().mori_ffi_shmem_init_flag_uniqueid()
    return shmem_init_attr(flag, rank, world_size, uid_bytes)


# ---------------------------------------------------------------------------
# JAX array interop
# ---------------------------------------------------------------------------


def jax_data_ptr(arr) -> int:
    """Extract raw GPU device pointer from a single-device jax.Array."""
    if hasattr(arr, "addressable_data"):
        buf = arr.addressable_data(0)
    else:
        buf = arr
    if hasattr(buf, "unsafe_buffer_pointer"):
        return buf.unsafe_buffer_pointer()
    dl_capsule = arr.__dlpack__()
    managed = ctypes.cast(
        ctypes.pythonapi.PyCapsule_GetPointer(
            ctypes.py_object(dl_capsule), b"dltensor"
        ),
        ctypes.POINTER(_DLManagedTensor),
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


_dlpack_prevent_gc = []


def shmem_ptr_to_jax(ptr, shape, dtype, device_id=0):
    """Wrap a shmem-allocated GPU pointer as a jax.Array (zero-copy via DLPack).

    Args:
        ptr: int64 device pointer address.
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
        ptr, shape, dl_dtype=(dl_code, dl_bits), typestr=typestr, device_id=device_id,
    )
    _dlpack_prevent_gc.append(view)
    result = jax.dlpack.from_dlpack(view)
    if len(_dlpack_prevent_gc) > 64:
        del _dlpack_prevent_gc[:32]
    return result


def shmem_register_jax_array(arr) -> int:
    """Register a jax.Array for shmem RDMA operations."""
    ptr = jax_data_ptr(arr)
    return shmem_buffer_register(ptr, arr.nbytes)

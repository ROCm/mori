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
"""
Framework-agnostic GPU tensor utilities.

Wraps raw GPU device pointers into tensors for any framework (torch, jax, etc.)
via DLPack and __cuda_array_interface__. No C++ torch dependency.

Usage:
    # Framework-agnostic view (supports DLPack + __cuda_array_interface__)
    view = gpu_tensor_view(ptr, shape, dtype)

    # Consume with any framework:
    t = torch.from_dlpack(view)         # PyTorch
    a = jax.dlpack.from_dlpack(view)    # JAX
    t = torch.as_tensor(view)           # PyTorch via __cuda_array_interface__

    # Or use the convenience wrapper:
    t = from_gpu_ptr(ptr, shape, dtype, framework="torch")

Dtype int mapping (matches C++ IntToHipDataType in pybind_ops.cpp):
    0 = float32, 1 = bfloat16, 2 = float8_e4m3fn,
    3 = float8_e4m3fnuz, 4 = int32
"""
import ctypes
import math

# ---------------------------------------------------------------------------
# DLPack constants and ctypes structures
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
    try:
        import torch
        _DL_GPU_DEVICE_TYPE = kDLROCM if hasattr(torch.version, 'hip') and torch.version.hip else kDLCUDA
    except ImportError:
        import os
        _DL_GPU_DEVICE_TYPE = kDLROCM if os.path.isdir("/opt/rocm") else kDLCUDA
    return _DL_GPU_DEVICE_TYPE


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DL_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DL_DELETER),
    ]


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


# ---------------------------------------------------------------------------
# Dtype mapping tables (lazy-initialized)
# ---------------------------------------------------------------------------

_TORCH_DTYPE_TO_INT = None


def _init_torch_dtype_map():
    global _TORCH_DTYPE_TO_INT
    if _TORCH_DTYPE_TO_INT is not None:
        return
    import torch
    _TORCH_DTYPE_TO_INT = {
        torch.float32: 0,
        torch.bfloat16: 1,
        torch.int32: 4,
    }
    if hasattr(torch, 'float8_e4m3fn'):
        _TORCH_DTYPE_TO_INT[torch.float8_e4m3fn] = 2
    if hasattr(torch, 'float8_e4m3fnuz'):
        _TORCH_DTYPE_TO_INT[torch.float8_e4m3fnuz] = 3
    if hasattr(torch, 'float4_e2m1fn_x2'):
        _TORCH_DTYPE_TO_INT[torch.float4_e2m1fn_x2] = 5


def dtype_to_int(dtype):
    """Convert a torch dtype to the C++ dtype int."""
    _init_torch_dtype_map()
    result = _TORCH_DTYPE_TO_INT.get(dtype)
    if result is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return result


_DTYPE_INFO = None


def _init_dtype_info():
    """Build dtype metadata table.

    Each entry maps a torch.dtype to:
        (typestr, elem_size, reinterpret_torch, dl_code, dl_bits)

    - typestr: numpy-style type string for __cuda_array_interface__
    - elem_size: bytes per element
    - reinterpret_torch: if not None, torch needs a view() after as_tensor()
    - dl_code: DLPack DLDataTypeCode
    - dl_bits: DLPack bit width
    """
    global _DTYPE_INFO
    if _DTYPE_INFO is not None:
        return
    import torch
    _DTYPE_INFO = {
        #                   typestr  size  reinterpret     dl_code   dl_bits
        torch.float32:     ('<f4',   4,    None,           kDLFloat, 32),
        torch.float64:     ('<f8',   8,    None,           kDLFloat, 64),
        torch.float16:     ('<f2',   2,    None,           kDLFloat, 16),
        torch.int8:        ('<i1',   1,    None,           kDLInt,   8),
        torch.int16:       ('<i2',   2,    None,           kDLInt,   16),
        torch.int32:       ('<i4',   4,    None,           kDLInt,   32),
        torch.int64:       ('<i8',   8,    None,           kDLInt,   64),
        torch.uint8:       ('<u1',   1,    None,           kDLUInt,  8),
        torch.bfloat16:    ('<u2',   2,    torch.bfloat16, kDLBfloat, 16),
    }
    if hasattr(torch, 'float8_e4m3fnuz'):
        _DTYPE_INFO[torch.float8_e4m3fnuz] = ('<u1', 1, torch.float8_e4m3fnuz, kDLUInt, 8)
    if hasattr(torch, 'float8_e4m3fn'):
        _DTYPE_INFO[torch.float8_e4m3fn] = ('<u1', 1, torch.float8_e4m3fn, kDLUInt, 8)
    if hasattr(torch, 'float4_e2m1fn_x2'):
        _DTYPE_INFO[torch.float4_e2m1fn_x2] = ('<u1', 1, torch.float4_e2m1fn_x2, kDLUInt, 8)


# ---------------------------------------------------------------------------
# GpuTensorView — framework-agnostic GPU array wrapper
# ---------------------------------------------------------------------------

@_DL_DELETER
def _dl_noop_deleter(_ptr):
    pass


class GpuTensorView:
    """Framework-agnostic non-owning GPU array view.

    Implements both the DLPack protocol (__dlpack__ / __dlpack_device__)
    and the __cuda_array_interface__ protocol, so any major ML framework
    can consume it zero-copy:

        view = GpuTensorView(ptr, (M, N), dl_dtype=(kDLFloat, 32), typestr='<f4')
        t = torch.from_dlpack(view)
        a = jax.dlpack.from_dlpack(view)
        t = torch.as_tensor(view, device='cuda')
    """

    def __init__(self, ptr, shape, *, dl_dtype, typestr, device_id=0):
        """
        Args:
            ptr: raw GPU device pointer (int).
            shape: tuple of dimensions.
            dl_dtype: (code, bits) for DLPack DLDataType.
            typestr: numpy-style type string for __cuda_array_interface__.
            device_id: GPU device ordinal.
        """
        self._ptr = ptr
        self._shape = tuple(shape)
        self._dl_code, self._dl_bits = dl_dtype
        self._device_id = device_id

        self.__cuda_array_interface__ = {
            'data': (ptr, False),
            'shape': self._shape,
            'typestr': typestr,
            'version': 2,
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

        capsule = _PyCapsule_New(ctypes.byref(managed), b"dltensor", None)
        return capsule

    def __dlpack_device__(self):
        return (_get_dl_gpu_device_type(), self._device_id)


def gpu_tensor_view(ptr, shape, dtype, device_id=0):
    """Create a framework-agnostic GpuTensorView from a raw GPU pointer.

    Args:
        ptr: int64 device pointer address.
        shape: tuple/list of dimensions.
        dtype: torch.dtype.
        device_id: GPU device ordinal (default 0).

    Returns:
        GpuTensorView that can be consumed by torch.from_dlpack(),
        jax.dlpack.from_dlpack(), or torch.as_tensor().
    """
    _init_dtype_info()
    info = _DTYPE_INFO.get(dtype)
    if info is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    typestr, _elem_size, _reinterpret, dl_code, dl_bits = info
    return GpuTensorView(
        ptr, shape,
        dl_dtype=(dl_code, dl_bits),
        typestr=typestr,
        device_id=device_id,
    )


# ---------------------------------------------------------------------------
# from_gpu_ptr — convenience wrapper returning framework-native tensors
# ---------------------------------------------------------------------------

def from_gpu_ptr(ptr, shape, dtype, framework="torch"):
    """
    Wrap a raw CUDA/ROCm device pointer into a tensor (zero-copy).

    Args:
        ptr: int64 device pointer address.
        shape: tuple/list of dimensions.
        dtype: torch.dtype.
        framework: "torch" (default, production-ready) or
                   "jax" (experimental, requires jax[rocm]).

    Returns:
        A tensor backed by the given device memory.
    """
    if framework == "torch":
        return _torch_from_ptr(ptr, shape, dtype)
    elif framework == "jax":
        return _jax_from_ptr(ptr, shape, dtype)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def _torch_from_ptr(ptr, shape, dtype):
    """Create a torch.Tensor from a raw GPU pointer via __cuda_array_interface__.

    Uses __cuda_array_interface__ which is stable in multiprocessing environments.
    For cross-framework use (jax, etc.), use gpu_tensor_view() + from_dlpack().
    """
    import torch
    _init_dtype_info()

    info = _DTYPE_INFO.get(dtype)
    if info is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    typestr, elem_size, reinterpret_as, dl_code, dl_bits = info
    device_id = torch.cuda.current_device()

    if reinterpret_as is not None:
        total_elems = math.prod(shape)
        view = GpuTensorView(
            ptr, (total_elems * elem_size,),
            dl_dtype=(kDLUInt, 8), typestr='<u1',
            device_id=device_id,
        )
        raw = torch.as_tensor(view, device=f'cuda:{device_id}')
        return raw.view(reinterpret_as).reshape(shape)

    view = GpuTensorView(
        ptr, shape,
        dl_dtype=(dl_code, dl_bits), typestr=typestr,
        device_id=device_id,
    )
    return torch.as_tensor(view, device=f'cuda:{device_id}')


def _jax_from_ptr(ptr, shape, dtype):
    """Create a jax array from a raw GPU pointer via DLPack.

    **Experimental** — requires jax[rocm] (ROCm GPU build) to be installed.
    The standard ``pip install jax`` only provides CPU support and will raise
    ``RuntimeError: Unknown backend rocm``.

    Example::

        # Requires: pip install jax[rocm] (or AMD's ROCm JAX wheel)
        arr = from_gpu_ptr(ptr, (M, N), torch.float32, framework="jax")
    """
    import jax

    _init_dtype_info()
    info = _DTYPE_INFO.get(dtype)
    if info is None:
        raise ValueError(f"Unsupported dtype for jax: {dtype}")

    typestr, elem_size, reinterpret_as, dl_code, dl_bits = info

    if reinterpret_as is not None:
        total_elems = math.prod(shape)
        view = GpuTensorView(
            ptr, (total_elems * elem_size,),
            dl_dtype=(kDLUInt, 8), typestr='<u1',
        )
        raw = jax.dlpack.from_dlpack(view)
        return raw.view(reinterpret_as).reshape(shape)

    view = gpu_tensor_view(ptr, shape, dtype)
    return jax.dlpack.from_dlpack(view)

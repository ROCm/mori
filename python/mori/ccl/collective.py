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

import os
import time

import torch
from mori import cpp as mori_cpp
from typing import Optional


# ---------------------------------------------------------------------------
# JIT compilation for CCL kernels
# ---------------------------------------------------------------------------
_ccl_hip_module = None


def _ensure_ccl_jit():
    """JIT compile ccl_kernels.hip and load the HipModule (once)."""
    global _ccl_hip_module
    if _ccl_hip_module is not None:
        return
    from mori.jit.core import compile_genco
    from mori.ops._jit_loader import _compiled_hsaco, load_hip_module

    if "ccl_kernels" not in _compiled_hsaco:
        _compiled_hsaco["ccl_kernels"] = compile_genco(
            "ccl_kernels", source_dir="src/collective/kernels"
        )
    _ccl_hip_module = load_hip_module("ccl_kernels", init_shmem=True)


def _get_ccl_func(name: str):
    """Get a kernel function from the JIT-compiled CCL module."""
    return _ccl_hip_module.get_function(name)


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------


def _require_sdma_env(class_name: str) -> None:
    raw = os.environ.get("MORI_ENABLE_SDMA", "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        raise RuntimeError(
            f"{class_name} requires MORI_ENABLE_SDMA=1 in the process "
            f"environment (got MORI_ENABLE_SDMA={os.environ.get('MORI_ENABLE_SDMA')!r}). "
            f"Export it before launch on every rank."
        )


_TORCH_DTYPE_TO_NUMPY = {
    torch.uint32: "<u4",
    torch.int32: "<i4",
    torch.float16: "<f2",
    torch.bfloat16: "<u2",
    torch.float32: "<f4",
}


def _stream_to_int(stream) -> int:
    if stream is None:
        return 0
    if isinstance(stream, int):
        return stream
    return stream.cuda_stream


class _GpuBufferView:
    def __init__(self, ptr: int, shape: tuple, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 2,
        }


def _ptr_to_tensor(ptr: int, size_bytes: int, dtype=torch.uint32, device=None):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = size_bytes // elem_size
    typestr = _TORCH_DTYPE_TO_NUMPY.get(dtype, "<u4")
    buf = _GpuBufferView(ptr, (num_elements,), typestr)
    device = _normalize_cuda_device(device) or torch.device("cuda")
    if dtype == torch.bfloat16:
        raw = torch.as_tensor(buf, device=device).view(torch.bfloat16)
    else:
        raw = torch.as_tensor(buf, device=device)
    return raw


def _normalize_cuda_device(device):
    if device is None:
        return None
    if isinstance(device, torch.Tensor):
        device = device.device
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise TypeError("device must be a CUDA tensor, torch.device, int, str, or None")
    if device.type != "cuda":
        raise ValueError("device must refer to a CUDA device")
    return device


def _resolve_transit_view_args(dtype, device, fallback_dtype):
    if device is None and dtype is not None and not isinstance(dtype, torch.dtype):
        device = dtype
        dtype = None
    if isinstance(device, torch.Tensor) and dtype is None:
        dtype = device.dtype
    if dtype is None:
        dtype = fallback_dtype
    return dtype, _normalize_cuda_device(device)


# ---------------------------------------------------------------------------
# All2allSdma
# ---------------------------------------------------------------------------


class All2allSdma:
    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
    ):
        _require_sdma_env("All2allSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        handle_class = getattr(mori_cpp, "All2allSdmaHandle")

        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, input_buffer_size, output_buffer_size, copy_output_to_user
            )
        elif transit_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, transit_buffer_size, copy_output_to_user
            )
        else:
            self._handle = handle_class(
                my_pe, npes, 512 * 1024 * 1024, copy_output_to_user
            )

    def __call__(self, input_data, output_data, count: int, stream=None) -> float:
        s = _stream_to_int(stream)
        t0 = time.perf_counter()
        args = self._handle.prepare_sync(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        _get_ccl_func("OneShotAll2allSdmaKernel_u32").launch_struct(
            (1,), (64,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), count, s)
        return time.perf_counter() - t0

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_start(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        _get_ccl_func("OneShotAll2allSdmaAsyncPutKernel_u32").launch_struct(
            (1,), (64,), 0, s, args
        )
        self._handle.after_async_start()
        return True

    def wait_async(self, stream=None) -> float:
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_wait(s)
        _get_ccl_func("OneShotAll2allSdmaAsyncWaitKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        return self._handle.finish_async_wait(s)

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def get_output_transit_buffer(self, dtype=None, device=None):
        dtype, device = _resolve_transit_view_args(dtype, device, torch.uint32)
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype, device)


# ---------------------------------------------------------------------------
# AllgatherSdma
# ---------------------------------------------------------------------------


class AllgatherSdma:
    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
    ):
        _require_sdma_env("AllgatherSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        handle_class = getattr(mori_cpp, "AllgatherSdmaHandle")

        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, input_buffer_size, output_buffer_size, copy_output_to_user
            )
        elif transit_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, transit_buffer_size, copy_output_to_user
            )
        else:
            self._handle = handle_class(
                my_pe, npes, 512 * 1024 * 1024, copy_output_to_user
            )

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync(
            input_data.data_ptr(), output_data.data_ptr(), u32_count, s
        )
        _get_ccl_func("OneShotAllGatherSdmaKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), u32_count, s)
        return True

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_start(
            input_data.data_ptr(), output_data.data_ptr(), u32_count, s
        )
        _get_ccl_func("OneShotAllGatherSdmaAsyncPutKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.after_async_start()
        return True

    def wait_async(self, stream=None) -> float:
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_wait(s)
        _get_ccl_func("OneShotAllGatherSdmaAsyncWaitKernel_u32").launch_struct(
            (1,), (64,), 0, s, args
        )
        return self._handle.finish_async_wait(s)

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def get_output_transit_buffer(self, dtype=None, device=None):
        dtype, device = _resolve_transit_view_args(dtype, device, torch.uint32)
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype, device)

    def register_output_buffer(self, tensor):
        self._handle.register_output_buffer(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )

    def deregister_output_buffer(self, tensor):
        self._handle.deregister_output_buffer(tensor.data_ptr())

    def is_output_registered(self, tensor) -> bool:
        return self._handle.is_output_registered(tensor.data_ptr())


# ---------------------------------------------------------------------------
# AllreduceSdma
# ---------------------------------------------------------------------------

_DTYPE_TO_SUFFIX = {
    torch.uint32: "u32",
    torch.int32: "i32",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}

_HANDLE_MAP = {
    torch.uint32: "AllreduceSdmaHandle",
    torch.int32: "AllreduceSdmaHandleInt32",
    torch.float32: "AllreduceSdmaHandleFp32",
    torch.float16: "AllreduceSdmaHandleFp16",
    torch.bfloat16: "AllreduceSdmaHandleBf16",
}


class AllreduceSdma:
    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
        dtype: torch.dtype = torch.uint32,
        mode: str = "eager",
    ):
        _require_sdma_env("AllreduceSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.dtype = dtype
        self.mode = mode
        self._type_suffix = _DTYPE_TO_SUFFIX.get(dtype)
        if self._type_suffix is None:
            raise ValueError(
                f"Unsupported dtype {dtype}. Supported: {list(_DTYPE_TO_SUFFIX.keys())}"
            )

        handle_name = _HANDLE_MAP.get(dtype)
        if handle_name is None:
            raise ValueError(f"Unsupported dtype {dtype}")
        handle_class = getattr(mori_cpp, handle_name)

        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, input_buffer_size, output_buffer_size, copy_output_to_user
            )
        elif transit_buffer_size is not None:
            self._handle = handle_class(
                my_pe, npes, transit_buffer_size, copy_output_to_user
            )
        else:
            self._handle = handle_class(
                my_pe, npes, 512 * 1024 * 1024, copy_output_to_user
            )

    def _run_sync(
        self,
        input_data,
        output_data,
        count: int,
        stream=None,
        force_copy_output: bool = False,
    ) -> bool:
        s = _stream_to_int(stream)
        sfx = self._type_suffix
        # Step 1: ReduceScatter
        args = self._handle.prepare_reduce_scatter(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        blocks, threads = self._handle.get_reduce_scatter_grid(count)
        _get_ccl_func(f"SdmaReduceScatterKernel_{sfx}").launch_struct(
            (blocks,), (threads,), 0, s, args
        )
        # Step 2: AllGather
        args = self._handle.prepare_allgather(count, s)
        _get_ccl_func(f"AllGatherSdmaKernel_{sfx}").launch_struct(
            (1,), (512,), 0, s, args
        )
        # Sync + copy output
        self._handle.finish_sync(output_data.data_ptr(), count, s, force_copy_output)
        return True

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        return self._run_sync(input_data, output_data, count, stream)

    def allreduce_inplace(self, data, count: int, stream=None) -> bool:
        return self._run_sync(data, data, count, stream, force_copy_output=True)

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        s = _stream_to_int(stream)
        sfx = self._type_suffix
        # ReduceScatter
        args = self._handle.prepare_async_reduce_scatter(
            input_data.data_ptr(), output_data.data_ptr(), count, s
        )
        blocks, threads = self._handle.get_reduce_scatter_grid(count)
        _get_ccl_func(f"SdmaReduceScatterKernel_{sfx}").launch_struct(
            (blocks,), (threads,), 0, s, args
        )
        # AllGather PUT
        args = self._handle.prepare_async_allgather_put(count, s)
        _get_ccl_func(f"AllGatherAsyncPutKernel_{sfx}").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.after_async_start()
        return True

    def wait_async(self, stream=None) -> float:
        s = _stream_to_int(stream)
        sfx = self._type_suffix
        args = self._handle.prepare_async_wait(s)
        _get_ccl_func(f"AllGatherAsyncWaitKernel_{sfx}").launch_struct(
            (1,), (64,), 0, s, args
        )
        return self._handle.finish_async_wait(s)

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def get_output_transit_buffer(self, dtype=None, device=None):
        dtype, device = _resolve_transit_view_args(dtype, device, self.dtype)
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype, device)

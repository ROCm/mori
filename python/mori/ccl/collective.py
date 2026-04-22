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

import torch
from mori import cpp as mori_cpp
from typing import Optional


_TORCH_DTYPE_TO_NUMPY = {
    torch.uint32: "<u4",
    torch.int32: "<i4",
    torch.float16: "<f2",
    torch.bfloat16: "V2",
    torch.float32: "<f4",
}


def _stream_to_int(stream) -> int:
    """Convert a torch.cuda.Stream (or None / int) to an integer handle."""
    if stream is None:
        return 0
    if isinstance(stream, int):
        return stream
    return stream.cuda_stream


class _GpuBufferView:
    """Lightweight wrapper that exposes __cuda_array_interface__ so
    ``torch.as_tensor`` can wrap a raw GPU pointer without a copy."""

    def __init__(self, ptr: int, shape: tuple, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 2,
        }


def _ptr_to_tensor(ptr: int, size_bytes: int, dtype=torch.uint32):
    """Wrap a raw GPU pointer as a 1-D torch CUDA tensor (zero-copy view)."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = size_bytes // elem_size
    typestr = _TORCH_DTYPE_TO_NUMPY.get(dtype, "<u4")
    buf = _GpuBufferView(ptr, (num_elements,), typestr)
    if dtype == torch.bfloat16:
        raw = torch.as_tensor(buf, device="cuda").view(torch.bfloat16)
    else:
        raw = torch.as_tensor(buf, device="cuda")
    return raw


def _cpp_all2all_factory(entity_name: str):
    """Factory function to get C++ entities from mori_cpp module"""
    return getattr(mori_cpp, entity_name)


class All2allSdma:
    """Python wrapper for All2allSdma C++ class"""

    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
    ):
        self.my_pe = my_pe
        self.npes = npes
        handle_class = _cpp_all2all_factory("All2allSdmaHandle")

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
        """Execute All2All SDMA operation.

        Args:
            input_data: Input CUDA tensor (1D, GPU memory)
            output_data: Output CUDA tensor (1D, GPU memory)
            count: Number of elements per PE
            stream: Optional torch.cuda.Stream / int / None

        Returns:
            Execution time in seconds
        """
        return self._handle(
            input_data.data_ptr(), output_data.data_ptr(), count, _stream_to_int(stream)
        )

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        return self._handle.start_async(
            input_data.data_ptr(), output_data.data_ptr(), count, _stream_to_int(stream)
        )

    def wait_async(self, stream=None) -> float:
        return self._handle.wait_async(_stream_to_int(stream))

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def get_output_transit_buffer(self, dtype=torch.uint32):
        """Get output transit buffer as a PyTorch CUDA tensor (zero-copy view).

        Args:
            dtype: torch.dtype for the returned tensor view (default torch.uint32).
        """
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype)


def _cpp_allgather_factory(entity_name: str):
    return getattr(mori_cpp, entity_name)


class AllgatherSdma:
    """Python wrapper for AllgatherSdma C++ class"""

    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
    ):
        self.my_pe = my_pe
        self.npes = npes
        handle_class = _cpp_allgather_factory("AllgatherSdmaHandle")

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
        """Execute Allgather SDMA operation.

        ``count`` is in *elements* of the tensor dtype; the wrapper converts
        to a uint32-equivalent count so the SDMA kernel copies the right
        number of bytes.
        """
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        return self._handle(
            input_data.data_ptr(),
            output_data.data_ptr(),
            u32_count,
            _stream_to_int(stream),
        )

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        return self._handle.start_async(
            input_data.data_ptr(),
            output_data.data_ptr(),
            u32_count,
            _stream_to_int(stream),
        )

    def wait_async(self, stream=None) -> float:
        return self._handle.wait_async(_stream_to_int(stream))

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def get_output_transit_buffer(self, dtype=torch.uint32):
        """Get output transit buffer as a PyTorch CUDA tensor (zero-copy view)."""
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype)

    def register_output_buffer(self, tensor):
        """Register a CUDA tensor as direct SDMA output target (collective)."""
        self._handle.register_output_buffer(tensor.data_ptr(), tensor.nbytes())

    def deregister_output_buffer(self, tensor):
        """Deregister a previously registered output buffer (collective)."""
        self._handle.deregister_output_buffer(tensor.data_ptr())

    def is_output_registered(self, tensor) -> bool:
        """Check whether an output tensor is registered."""
        return self._handle.is_output_registered(tensor.data_ptr())


def _cpp_allreduce_factory(entity_name: str):
    return getattr(mori_cpp, entity_name)


class AllreduceSdma:
    """Python wrapper for AllreduceSdma C++ class.

    Performs AllReduce in two stages:
      Stage 1: ReduceScatter -- each rank reduces its shard across all peers
      Stage 2: AllGather -- broadcast reduced shards to all peers via SDMA

    Supported dtypes: torch.uint32, torch.int32, torch.float16, torch.bfloat16
    """

    _HANDLE_MAP = {
        torch.uint32: "AllreduceSdmaHandle",
        torch.int32: "AllreduceSdmaHandle",
        torch.float32: "AllreduceSdmaHandle",
        torch.float16: "AllreduceSdmaHandleFp16",
        torch.bfloat16: "AllreduceSdmaHandleBf16",
    }

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
        self.my_pe = my_pe
        self.npes = npes
        self.dtype = dtype
        self.mode = mode

        handle_name = self._HANDLE_MAP.get(dtype)
        if handle_name is None:
            raise ValueError(
                f"Unsupported dtype {dtype}. Supported: {list(self._HANDLE_MAP.keys())}"
            )
        handle_class = _cpp_allreduce_factory(handle_name)

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
        """Execute out-of-place AllReduce SDMA operation."""
        return self._handle(
            input_data.data_ptr(), output_data.data_ptr(), count,
            _stream_to_int(stream)
        )

    def allreduce_inplace(self, data, count: int, stream=None) -> bool:
        """Execute in-place AllReduce SDMA operation (result overwrites input)."""
        return self._handle.allreduce_inplace(
            data.data_ptr(), count, _stream_to_int(stream)
        )

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        return self._handle.start_async(
            input_data.data_ptr(), output_data.data_ptr(), count, _stream_to_int(stream)
        )

    def wait_async(self, stream=None) -> float:
        return self._handle.wait_async(_stream_to_int(stream))

    def is_async_in_progress(self) -> bool:
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        self._handle.cancel_async()

    def reset_flags(self):
        self._handle.reset_flags()

    def get_output_transit_buffer(self, dtype=None):
        """Get output transit buffer as a PyTorch CUDA tensor (zero-copy view).

        Args:
            dtype: torch.dtype for the view. Defaults to self.dtype.
        """
        if dtype is None:
            dtype = self.dtype
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype)

    # --- Phase-level timestamp instrumentation (diagnostic) -----------------
    def enable_phase_timing(self, on: bool):
        """Enable/disable per-phase timestamp recording for pipelined AR kernel.

        When enabled, block 0 thread 0 of PipelinedAllReduceSdmaKernel writes
        __builtin_amdgcn_s_memtime() at each phase boundary. Read back with
        get_phase_timestamps() after the stream has synchronized.
        """
        self._handle.enable_phase_timing(on)

    def get_phase_timestamps(self) -> list:
        """Return per-phase GPU timestamps (list of 32 uint64 cycles) from the
        most recent pipelined() call. Stream must be synchronized before call.

        Slot layout (see kernel header for details):
          0: kernel entry
          1: scatter submit done
          2 + 3*c + {0,1,2}: chunk c {compute-wait, cross-PE-barrier, AG-submit} done
          2 + 3*numChunks:   AG wait done (all peers)
          3 + 3*numChunks:   block 0 exit
        """
        return self._handle.get_phase_timestamps()

    def get_last_num_chunks(self) -> int:
        """Return numChunks used by the most recent pipelined() call.

        Needed to parse get_phase_timestamps()'s layout.
        """
        return self._handle.get_last_num_chunks()

    # --- Copy-path timing (baseline: single hipMemcpyAsync) ---------------
    def enable_copy_timing(self, on: bool):
        """Enable per-call copy-path timing (hipMemcpyAsync host us + gpu ms)."""
        self._handle.enable_copy_timing(on)

    def get_copy_timing_ms(self) -> list:
        """Return [host_us, gpu_ms] for the most recent copy_output_to_user()
        call. Baseline has a single hipMemcpyAsync so len() == 2.

        host_us: wall time spent inside the host hipMemcpyAsync() API call
        gpu_ms:  wall time between events recorded before/after the copy on
                 the stream (= actual GPU copy kernel runtime).

        Call AFTER the stream has been synchronized.
        """
        return list(self._handle.get_copy_timing_ms())

    def get_copy_timing_last_num_chunks(self) -> int:
        """Baseline always returns 1 (single copy). Reserved for path B to
        indicate number of chunked copies."""
        return self._handle.get_copy_timing_last_num_chunks()

    def enable_post_ag_wait(self, on: bool):
        """Stage 1 of E' prototype: compute blocks stay alive during AG wait.

        When enabled, AR kernel's compute blocks (block 1..N) do NOT exit
        after their reduce phase; instead they spin-wait on a device flag
        that block 0 sets once AG wait completes. This is a cost-measurement
        step — no in-kernel copy is performed yet. Compare wall time with
        and without to see if compute blocks holding CUs hurts overlap.
        """
        self._handle.enable_post_ag_wait(on)

    # --- D' fast path: lazy-register user output (see perf_history Entry 10)
    def enable_register_user_output(self, on: bool):
        """Enable the D' fast-path lookup in pipelined(). When on, if the
        caller's output tensor has been pre-registered (see
        register_user_output), AR kernel writes directly to it and the
        external hipMemcpyAsync is skipped. Zero effect if no entry is
        pre-registered (falls back to baseline transit+copy path)."""
        self._handle.enable_register_user_output(on)

    def register_user_output(self, ptr: int, size: int) -> bool:
        """Collective: pre-register a user output buffer as shmem symmetric
        memory. All ranks must call with the same size; ptr may differ per
        rank. Returns True on success. Cached for later lookup by pipelined().
        """
        return self._handle.register_user_output(ptr, size)

    # --- Direction θ: multi-qId parallel AG -----------------------------
    def enable_ag_multi_q(self, on: bool):
        """Enable multi-qId parallel AG in MULTI_CHUNK AR kernel.

        When on, chunk c's AG uses SDMA qId = 1+(c%(numQ-1)) so chunks
        transfer in parallel on separate queues. Must be called BEFORE
        the first pipelined() call on this object (signal counter
        semantics change). See docs/perf_history.md Entry 12.
        """
        self._handle.enable_ag_multi_q(on)

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

    def enqueue(self, input_data, output_data, count: int, stream=None) -> bool:
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

    def enqueue_param_contiguous(
        self,
        input_data,
        output_data,
        count: int,
        split_sizes,
        split_offsets,
        stream=None,
    ) -> bool:
        if split_sizes.numel() != split_offsets.numel():
            raise ValueError("split_sizes and split_offsets must have the same length")
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_param_contiguous(
            input_data.data_ptr(),
            output_data.data_ptr(),
            u32_count,
            split_sizes.data_ptr(),
            split_offsets.data_ptr(),
            split_sizes.numel(),
            s,
        )
        _get_ccl_func("OneShotAllGatherSdmaParamContiguousKernel_u32").launch_struct(
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

    def start_async_param_contiguous(
        self,
        input_data,
        output_data,
        count: int,
        split_sizes,
        split_offsets,
        stream=None,
    ) -> bool:
        if split_sizes.numel() != split_offsets.numel():
            raise ValueError("split_sizes and split_offsets must have the same length")
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_async_start_param_contiguous(
            input_data.data_ptr(),
            output_data.data_ptr(),
            u32_count,
            split_sizes.data_ptr(),
            split_offsets.data_ptr(),
            split_sizes.numel(),
            s,
        )
        _get_ccl_func(
            "OneShotAllGatherSdmaParamContiguousAsyncPutKernel_u32"
        ).launch_struct((1,), (512,), 0, s, args)
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
# InterNodeRingAllgather — inter-node RDMA ring
# ---------------------------------------------------------------------------


class InterNodeRingAllgather:
    """Inter-node AllGather over the shmem ring (P2P intra-node, RDMA inter-node).

    This is the inter-node phase of the hierarchical cross-node AllGather. Each
    participating PE contributes one ``count``-element chunk; after the ring
    every PE holds all ``npes`` chunks concatenated in PE order -- identical to
    ``torch.distributed.all_gather_into_tensor`` when every PE is a participant.

    The ring schedule (CPU-validated by ``inter_node_ring_reference``) is run on
    device by the JIT kernel ``InterNodeRingAllGatherKernel_u32``, which moves
    raw bytes so a single u32 kernel serves bf16/fp16/fp32/int32.

    ``shmem`` must already be initialized (e.g. via
    ``mori.shmem.shmem_torch_process_group_init``) with ``my_pe``/``npes``
    matching this handle.
    """

    def __init__(
        self,
        my_pe: int,
        npes: int,
        ring_buffer_bytes: Optional[int] = None,
        ring_size: int = -1,
        ring_pos: int = -1,
        pe_base: int = 0,
        pe_stride: int = 1,
        num_qp: int = 1,
        num_blocks: int = 1,
    ):
        # NOTE: the inter-node phase deliberately uses the RDMA/P2P shmem
        # transport (DESIGN: inter-node == RDMA), NOT the SDMA copy engines
        # (those drive the *intra*-node phase). So MORI_ENABLE_SDMA is not
        # required here; forcing it on would route same-node puts through the
        # SDMA multi-queue path, which is the intra-node optimization, not what
        # the ring needs.
        #
        # Sub-group ring (M2b, hierarchical inter-node phase): when ``ring_size``
        # >= 0 the ring runs over the arithmetic sub-group of global PEs
        # ``{pe_base, pe_base+pe_stride, ..., pe_base+(ring_size-1)*pe_stride}``
        # and this PE is at position ``ring_pos`` within it. The output holds the
        # ``ring_size`` chunks in ring order. The default (``ring_size=-1``) is
        # the flat whole-world ring (ring_size=npes, ring_pos=my_pe).
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.ring_size = ring_size if ring_size >= 0 else npes
        handle_class = getattr(mori_cpp, "InterNodeRingAllgatherHandle")
        if ring_buffer_bytes is None:
            ring_buffer_bytes = 512 * 1024 * 1024
        # num_qp>1 fans the per-round ring put across that many RDMA QPs (the
        # kernel applies it only to true cross-node neighbours; same-node P2P/SDMA
        # neighbours stay single-warp). Default 1 == unchanged single-QP put.
        self.num_qp = num_qp
        # M4: num_blocks>1 launches the ring as that many CTAs
        # ("channels"), each driving a disjoint chunk sub-range on its own QP
        # (RCCL-style). The kernel engages it only for true RDMA neighbours;
        # same-node sims fall back to a single working block. Default 1 ==
        # unchanged single-block ring.
        self.num_blocks = num_blocks if num_blocks and num_blocks >= 1 else 1
        self._handle = handle_class(
            my_pe, npes, ring_buffer_bytes, ring_size, ring_pos, pe_base, pe_stride, num_qp,
            self.num_blocks,
        )

    def __call__(
        self, input_data, output_data, count: int, stream=None, chunk_in_place: bool = False,
        out_in_place: bool = False, stream_ring: bool = False, defer_inter_fin: bool = False,
    ) -> bool:
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        # stream_ring=True uses the on-device
        # ShmemBarrierOnStream prepare/finish (no host hipStreamSynchronize / host
        # ShmemBarrierAll), keeping the whole op enqueued on the stream. Same byte
        # moves and global fencing as the host-synced path; removes 2 CPU<->GPU
        # round-trips per op.
        if chunk_in_place:
            # M4: this PE's chunk is already in its ring slot (the
            # upstream intra gather wrote there via ``slot_tensor``), so skip the
            # prepare_sync copy-IN.
            if stream_ring:
                args = self._handle.prepare_stream_in_place(u32_count, s)
            else:
                args = self._handle.prepare_sync_in_place(u32_count, s)
        else:
            if stream_ring:
                args = self._handle.prepare_stream(input_data.data_ptr(), u32_count, s)
            else:
                args = self._handle.prepare_sync(input_data.data_ptr(), u32_count, s)
        _get_ccl_func("InterNodeRingAllGatherKernel_u32").launch_struct(
            (self.num_blocks,), (512,), 0, s, args
        )
        if out_in_place:
            # M4: leave the gathered result in the ring buffer (read it
            # via ``full_tensor``) and skip the finish_sync copy-OUT. ``output_data``
            # is ignored in this mode.
            if stream_ring:
                self._handle.finish_stream_no_copy(s)
            else:
                self._handle.finish_sync_no_copy(s)
        else:
            if stream_ring:
                # defer the ring-reuse fence to the next op's
                # prepare_stream barrier (defer_inter_fin) -- the copy-OUT stays
                # stream-ordered so the collection is correct; only cross-PE ring
                # reuse needs the fence, which the successor op provides.
                self._handle.finish_stream(output_data.data_ptr(), u32_count, s,
                                           barrier=not defer_inter_fin)
            else:
                self._handle.finish_sync(output_data.data_ptr(), u32_count, s)
        return True

    def prepare_stream_only(self, input_data, count: int, stream=None):
        """issue ONLY the stream-ordered ring prepare (the
        global on-stream ShmemBarrierOnStream entry barrier + the per-PE copy-IN
        of ``input_data`` into the ring slot) and return ``(args, u32_count, s)``
        WITHOUT launching the ring kernel.

        This splits the monolithic ``__call__`` (prepare -> kernel -> finish) so
        the caller can interleave INDEPENDENT work (e.g. the slice path's local
        node-block SDMA reassembly gather, which reads only this rank's own input
        and has no ring dependency) on a SIDE stream between the entry barrier and
        the ring kernel -- the side work then overlaps the ring kernel while the
        ring's prepare barrier remains the SOLE global entry fence (the side work
        runs barrier-free, so only one global on-stream fence is ever in flight).
        Stream-ring only (the on-device barrier path).
        """
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_stream(input_data.data_ptr(), u32_count, s)
        return args, u32_count, s

    def launch_finish_stream(self, args, output_data, u32_count: int, s: int,
                             barrier: bool = True) -> bool:
        """launch the ring kernel for a previously-prepared op
        (``prepare_stream_only``) then run the stream-ordered finish copy-OUT into
        ``output_data``. ``barrier`` controls the finish ShmemBarrierOnStream
        (defer it like ``defer_inter_fin``). Pairs with ``prepare_stream_only``.
        """
        _get_ccl_func("InterNodeRingAllGatherKernel_u32").launch_struct(
            (self.num_blocks,), (512,), 0, s, args
        )
        self._handle.finish_stream(output_data.data_ptr(), u32_count, s,
                                   barrier=barrier)
        return True

    def finish_ring_stream(self, output_data, count: int, stream=None,
                           barrier: bool = True) -> bool:
        """stream-ordered ring finish copy-OUT ONLY -- the ring
        kernel was already launched ELSEWHERE (e.g. by the FUSED
        ``FusedRingLocalGatherKernel_u32`` that runs the ring concurrently with
        the local-block SDMA gather in one grid). This issues just the copy-OUT of
        the gathered ring buffer into ``output_data`` + the (optionally deferred)
        ShmemBarrierOnStream reuse fence. Pairs with ``prepare_stream_only`` +
        the external fused kernel launch. ``barrier`` mirrors ``defer_inter_fin``.
        """
        byte_count = count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        self._handle.finish_stream(output_data.data_ptr(), u32_count, s,
                                   barrier=barrier)
        return True

    def full_tensor(self, count: int, dtype, device=None):
        """A torch view of the FULL ring buffer (``ring_size * count`` elements).

        After ``__call__(..., out_in_place=True)`` the ring buffer holds the
        ``ring_size`` gathered chunks in ring order -- the full rank-major
        result for this sub-group. Reading from here avoids the finish_sync
        copy-OUT. ``count`` is the per-chunk element count (of ``dtype``);
        ``count*element_size`` must be a multiple of 4 (the u32 lane size).
        """
        byte_count = count * torch.tensor([], dtype=dtype).element_size()
        u32_count = (byte_count + 3) // 4
        total = u32_count * self.ring_size
        ptr = self._handle.buf_ptr()
        return _ptr_to_tensor(ptr, total * 4, dtype, device)[: count * self.ring_size]

    def slot_tensor(self, count: int, dtype, device=None):
        """A torch view of this PE's ring slot (``count`` elements of ``dtype``).

        Write this PE's chunk here (e.g. as the intra gather's output) and then
        call ``__call__(..., chunk_in_place=True)`` to run the ring without the
        prepare_sync copy-IN. ``count`` is in elements of ``dtype``; the slot is
        sized in u32 lanes, so ``count*element_size`` must be a multiple of 4.
        """
        byte_count = count * torch.tensor([], dtype=dtype).element_size()
        u32_count = (byte_count + 3) // 4
        ptr = self._handle.slot_ptr(u32_count)
        return _ptr_to_tensor(ptr, u32_count * 4, dtype, device)[:count]


# ---------------------------------------------------------------------------
# IntraNodeSubGroupAllgatherSdma — intra-node SDMA gather over a
# sub-group of local ranks (the intra-node phase of the hierarchical AllGather)
# ---------------------------------------------------------------------------


class IntraNodeSubGroupAllgatherSdma:
    """Intra-node SDMA AllGather over an arithmetic sub-group of local ranks.

    The ``group_size`` ranks ``{pe_base, pe_base+pe_stride, ...}`` (this PE at
    position ``group_pos``) gather their ``count``-element shards over the SDMA
    copy engines (XGMI); after the call every member holds the ``group_size``
    shards concatenated in group-position order -- its node's contiguous block.
    This is the SDMA-side building block of the hierarchical cross-node
    AllGather (DESIGN: intra-node == SDMA). The default (``group_size=-1``) is
    the flat whole-world SDMA gather (group_size=npes, group_pos=my_pe).

    ``shmem`` must already be initialized (e.g. via
    ``mori.shmem.shmem_torch_process_group_init``) with ``my_pe``/``npes``.
    Requires ``MORI_ENABLE_SDMA=1`` (the SDMA copy-engine path).
    """

    def __init__(
        self,
        my_pe: int,
        npes: int,
        out_buffer_bytes: Optional[int] = None,
        group_size: int = -1,
        group_pos: int = -1,
        pe_base: int = 0,
        pe_stride: int = 1,
    ):
        _require_sdma_env("IntraNodeSubGroupAllgatherSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.group_size = group_size if group_size >= 0 else npes
        handle_class = getattr(mori_cpp, "IntraNodeSubGroupAllgatherSdmaHandle")
        if out_buffer_bytes is None:
            out_buffer_bytes = 512 * 1024 * 1024
        self._handle = handle_class(
            my_pe, npes, out_buffer_bytes, group_size, group_pos, pe_base, pe_stride
        )

    def __call__(self, input_data, output_data, count: int, stream=None, barrier: bool = True,
                 prepare_barrier: bool = True) -> bool:
        # M4: ``barrier=False`` skips the trailing ShmemBarrierAll in
        # finish_sync. Safe only when an immediately-following global barrier
        # synchronizes all PEs before any remote read (the PUSH gather's
        # in-kernel flag-wait already makes this PE's node-block complete on
        # kernel return). Used by HierAllGather's fused-barrier path.
        # M4: ``prepare_barrier=False`` additionally skips the ENTRY
        # ShmemBarrierAll in prepare_sync. Safe only when the PREVIOUS pipeline
        # iteration ended with a global barrier (so every peer's out_ transit is
        # free) AND this is not the first call (out_ already registered). The
        # caller (HierAllGather) enforces both via a first-call guard.
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync(input_data.data_ptr(), u32_count, s, prepare_barrier)
        _get_ccl_func("OneShotAllGatherSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), u32_count, s, barrier)
        return True

    def gather_kernel(self, input_data, count: int, dst_base_offset: int = 0, stream=None,
                      prepare_barrier: bool = True, dst_slot_stride: int = 0) -> bool:
        # M5: launch ONLY the gather kernel (no copy-OUT), writing this
        # gather's groupSize-slot block into ``out_`` at element offset
        # ``dst_base_offset`` (of the input dtype). Used by the fused sliced path
        # to stack the N reassembly gathers into disjoint regions of one enlarged
        # transit; a single ``finish_batch`` then copies them all out at once.
        # M5: ``dst_slot_stride`` (in elements of the input dtype, 0 ==
        # contiguous) decouples the per-peer destination slot stride from the
        # copy ``count``, so a CHUNK of a slice can land at its strided position
        # inside a full-size block -- the chunked inter/intra pipeline enabler.
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        dst_base_offset_bytes = dst_base_offset * input_data.element_size()
        dst_slot_stride_bytes = dst_slot_stride * input_data.element_size()
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync(
            input_data.data_ptr(), u32_count, s, prepare_barrier, dst_base_offset_bytes,
            dst_slot_stride_bytes,
        )
        _get_ccl_func("OneShotAllGatherSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        return True

    def get_output_transit_buffer(self, dtype=None, device=None):
        dtype, device = _resolve_transit_view_args(dtype, device, torch.uint32)
        ptr, size_bytes = self._handle.get_output_transit_buffer()
        return _ptr_to_tensor(ptr, size_bytes, dtype, device)

    def finish_batch(self, output_data, total_count: int, stream=None,
                     barrier: bool = True) -> bool:
        # M5: one bulk copy-OUT of ``total_count`` elements (the full
        # N*groupSize*chunk stacked by ``gather_kernel``) from ``out_`` to the
        # user output, then one barrier. Replaces N per-gather finish copies.
        byte_count = total_count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        self._handle.finish_batch(output_data.data_ptr(), u32_count, s, barrier)
        return True

    def finish_batch_stream(self, output_data, total_count: int, stream=None,
                            barrier: bool = True) -> bool:
        # STREAM-ORDERED bulk copy-OUT. Same bytes as
        # finish_batch but the trailing global fence is an on-device
        # ShmemBarrierOnStream(stream) instead of host hipStreamSynchronize +
        # ShmemBarrierAll. Removes the last host CPU<->GPU round-trip in the
        # fused sliced Phase-B so the whole op stays enqueued on ``stream``.
        # Pairs with the Turn-10 stream-ordered inter ring.
        # ``barrier=False`` DEFERS the trailing fence to the
        # next op's inter-ring prepare ShmemBarrierOnStream (redundant back-to-
        # back across the op boundary). The copy-OUT stays stream-ordered so the
        # output is correct regardless.
        byte_count = total_count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        self._handle.finish_batch_stream(output_data.data_ptr(), u32_count, s, barrier)
        return True

    def register_output_buffer(self, tensor) -> bool:
        # register a user output tensor for DIRECT-TO-OUTPUT
        # gathers (collective; cached). No-op if already registered.
        self._handle.register_output_buffer(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )
        return True

    def deregister_output_buffer(self, tensor) -> bool:
        self._handle.deregister_output_buffer(tensor.data_ptr())
        return True

    def deregister_output_buffer_ptr(self, ptr: int) -> bool:
        # deregister by raw base pointer (the live-registration
        # tracker holds an int, not the original tensor, after the buffer was
        # freed). Collective -- must be called in lockstep on every PE; the C++
        # side looks up the stored extent so only the ptr is needed.
        self._handle.deregister_output_buffer(ptr)
        return True

    def is_output_registered(self, tensor) -> bool:
        return self._handle.is_output_registered(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )

    def gather_kernel_direct(self, input_data, output_data, count: int, dst_block_offset: int = 0,
                             stream=None, prepare_barrier: bool = True,
                             dst_slot_stride: int = 0) -> bool:
        # DIRECT gather -- SDMA-PUSH each member's slice straight
        # into the (registered) ``output_data`` at element offset
        # ``dst_block_offset`` (of the input dtype), no internal transit + no
        # copy-OUT. ``output_data`` MUST have been registered via
        # register_output_buffer. ``dst_slot_stride`` matches gather_kernel.
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        dst_block_offset_bytes = dst_block_offset * input_data.element_size()
        dst_slot_stride_bytes = dst_slot_stride * input_data.element_size()
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_direct(
            input_data.data_ptr(), u32_count, s, prepare_barrier, output_data.data_ptr(),
            dst_block_offset_bytes, dst_slot_stride_bytes,
        )
        _get_ccl_func("OneShotAllGatherSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        return True

    def gather_kernel_direct_param_contiguous(
        self, input_data, output_data, block_stride: int, num_blocks: int,
        world_size: int, split_sizes_u32, split_offsets_u32, stream=None,
        prepare_barrier: bool = True, first_block: int = 0) -> bool:
        # FUSED param-contiguous direct gather -- ONE launch scatters ALL node
        # blocks * param splits from the Phase-A ``input_data`` collection
        # straight into the (registered) ``output_data`` in PARAM-CONTIGUOUS
        # layout. Replaces the per-(block, param) gather_kernel_direct loop that
        # made HierAllGather.enqueue_param_contiguous slower than RCCL. All size
        # arguments are in u32-lane units. ``split_sizes_u32`` / ``split_offsets_
        # u32`` are int64 DEVICE tensors (E_s / O_s per param, u32 lanes).
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_direct_param_contiguous(
            input_data.data_ptr(), s, prepare_barrier, output_data.data_ptr(),
            block_stride, num_blocks, world_size,
            split_sizes_u32.data_ptr(), split_offsets_u32.data_ptr(),
            split_sizes_u32.numel(), 0, first_block,
        )
        _get_ccl_func("OneShotAllGatherSdmaSubGroupParamContiguousKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        return True

    def prepare_direct_only(self, input_data, output_data, count: int,
                            dst_block_offset: int = 0, stream=None,
                            prepare_barrier: bool = True,
                            dst_slot_stride: int = 0) -> int:
        # build the DIRECT-gather jit_args (prime the per-peer
        # flag slots + optional entry barrier + register the strided output dst)
        # and RETURN the int64 args pointer WITHOUT launching the kernel -- so the
        # FUSED ``FusedRingLocalGatherKernel_u32`` can run this gather as a single
        # designated block of a larger grid (blockLocal=true) concurrently with
        # the inter-node RDMA ring. Mirrors ``gather_kernel_direct`` minus the
        # launch_struct; the returned ptr is the handle's jit_args_ member (kept
        # alive until the next prepare on this handle, exactly as the launch path
        # relies on). ``output_data`` MUST already be registered.
        byte_count = count * input_data.element_size()
        u32_count = (byte_count + 3) // 4
        dst_block_offset_bytes = dst_block_offset * input_data.element_size()
        dst_slot_stride_bytes = dst_slot_stride * input_data.element_size()
        s = _stream_to_int(stream)
        args = self._handle.prepare_sync_direct(
            input_data.data_ptr(), u32_count, s, prepare_barrier,
            output_data.data_ptr(), dst_block_offset_bytes, dst_slot_stride_bytes,
        )
        return args

    def finish_direct_stream(self, stream=None, barrier: bool = True) -> bool:
        # completion fence for the DIRECT path (no copy-OUT;
        # gathers already pushed into the user output). On-device global fence.
        s = _stream_to_int(stream)
        self._handle.finish_direct_stream(s, barrier)
        return True


def launch_fused_ring_local_gather(ring_args: int, gather_args: int,
                                   ring_blocks: int, s: int) -> bool:
    """merge a prepared inter-node ring's jit_args with a
    prepared intra-node local-block direct-gather's jit_args (via the
    ``build_fused_ring_local_gather_args`` C++ glue) and launch the FUSED
    ``FusedRingLocalGatherKernel_u32`` ONCE on stream ``s`` with ``ring_blocks +
    1`` CTAs: blocks ``[0, ring_blocks)`` run the RDMA ring (Phase A, over the
    NIC) and the last block runs the local node-block SDMA reassembly gather
    (Phase B, m == node_id -- the half independent of the ring) over XGMI. This
    replaces the two serial kernel launches + host ``wait_stream`` merge of the
    overlap path with one concurrent grid (NIC ring || XGMI gather), the
    RCCL-parity lever this work proved out, adopted here.
    """
    rb = ring_blocks if ring_blocks and ring_blocks >= 1 else 1
    fused = mori_cpp.build_fused_ring_local_gather_args(ring_args, gather_args, rb)
    _get_ccl_func("FusedRingLocalGatherKernel_u32").launch_struct(
        (rb + 1,), (512,), 0, s, fused
    )
    return True


# ---------------------------------------------------------------------------
# IntraNodeSubGroupBroadcastSdma — intra-node SDMA broadcast over a
# sub-group of local ranks (the placement phase of the leader-only hierarchical
# AllGather: leader's full N*G output is fanned to its G local ranks over XGMI).
# ---------------------------------------------------------------------------


class IntraNodeSubGroupBroadcastSdma:
    """Intra-node SDMA broadcast over an arithmetic sub-group of local ranks.

    The ``group_size`` ranks ``{pe_base, pe_base+pe_stride, ...}`` (this PE at
    position ``group_pos``) receive the full ``count``-element buffer held by the
    root (``group_pos == 0``) over the SDMA copy engines (XGMI). After the call
    every member's output equals the root's input. This is the placement-side
    building block of the leader-only hierarchical cross-node AllGather (DESIGN:
    intra-node == SDMA). The default (``group_size=-1``) broadcasts from rank 0
    over the whole world.

    ``shmem`` must already be initialized (e.g. via
    ``mori.shmem.shmem_torch_process_group_init``) with ``my_pe``/``npes``.
    Requires ``MORI_ENABLE_SDMA=1`` (the SDMA copy-engine path).
    """

    def __init__(
        self,
        my_pe: int,
        npes: int,
        out_buffer_bytes: Optional[int] = None,
        group_size: int = -1,
        group_pos: int = -1,
        pe_base: int = 0,
        pe_stride: int = 1,
    ):
        _require_sdma_env("IntraNodeSubGroupBroadcastSdma")
        _ensure_ccl_jit()
        self.my_pe = my_pe
        self.npes = npes
        self.group_size = group_size if group_size >= 0 else npes
        handle_class = getattr(mori_cpp, "IntraNodeSubGroupBroadcastSdmaHandle")
        if out_buffer_bytes is None:
            out_buffer_bytes = 512 * 1024 * 1024
        self._handle = handle_class(
            my_pe, npes, out_buffer_bytes, group_size, group_pos, pe_base, pe_stride
        )

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        # ``count`` is the number of elements in the broadcast payload (the full
        # buffer), not a per-rank shard. On non-root members ``input_data`` is
        # ignored; pass the user output tensor (any same-dtype buffer) as input.
        byte_count = count * output_data.element_size()
        u32_count = (byte_count + 3) // 4
        s = _stream_to_int(stream)
        in_ptr = input_data.data_ptr() if input_data is not None else 0
        args = self._handle.prepare_sync(in_ptr, u32_count, s)
        _get_ccl_func("OneShotBroadcastSdmaSubGroupKernel_u32").launch_struct(
            (1,), (512,), 0, s, args
        )
        self._handle.finish_sync(output_data.data_ptr(), u32_count, s)
        return True


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

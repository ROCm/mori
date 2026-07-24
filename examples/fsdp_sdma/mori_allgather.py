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
"""Cross-node FSDP2 all-gather backend backed by mori.ccl.HierAllGather.

Intra-node traffic rides SDMA copy engines (XGMI); inter-node goes over RDMA.
Wired in via ``FSDPModule.set_custom_all_gather``. The all-gather result is
copied out rank-major into FSDP's output tensor.
"""

import importlib
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather


class _HierWork:
    def __init__(self, event: "torch.cuda.Event", device: torch.device) -> None:
        self._event = event
        self._device = device
        self._waited = False

    def wait(self) -> bool:
        if not self._waited:
            torch.cuda.current_stream(self._device).wait_event(self._event)
            self._waited = True
        return True


class _HostProxyDeferredWork(dist.distributed_c10d.Work):
    """c10d Work deferring the host-blocking landing fence of the host-proxy AG.

    ``_post`` (RDMA write + step-1 intra gather) already ran non-blocking;
    ``wait()`` runs ``_complete`` (wait_all + rail-pair barrier + step-3) at
    copy-out so the cross-node round trip overlaps the caller's backward GEMM.
    Must subclass c10d Work: FSDP calls ``.wait()`` only on a
    ``dist.distributed_c10d.Work``.
    """

    def __init__(self, collective, handle, drain=False) -> None:
        super().__init__()
        self._collective = collective
        self._handle = handle
        self._drain = drain
        self._done = False

    def wait(self, timeout=None) -> bool:  # noqa: ARG002
        if not self._done:
            self._collective._complete(self._handle)
            # ASYNC_DRAIN=1: host-drain the AG stream after _complete so step-3
            # (the remote-half broadcast) is landed+visible before copy-out reads
            # ``out``; else copy-out races the still-pending step-3 -> garbage
            # remote half -> NaN.
            if self._drain:
                self._handle["stream"].synchronize()
            self._collective._pending = None
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done


class _DeviceDeferredHostSyncWork(dist.distributed_c10d.Work):
    """c10d Work deferring the device-path host landing fence to copy-out.

    The device HierAllGather (fused fill) is issued non-blocking on
    all_gather_stream; the only bit-exact landing fence on MI300X/mlx5 is a host
    ``stream.synchronize()`` (drains both SDMA copy-engine HSA signals and RDMA
    CQEs, which on-device/on-stream fences do not). Deferred to ``wait()`` so it
    overlaps the backward GEMM. Must subclass c10d Work (FSDP calls ``.wait()``
    only on one). Bit-exact: the sync drains before the consumer reads.
    """

    def __init__(
        self,
        stream: "torch.cuda.Stream",
        collective=None,
        event=None,
    ) -> None:
        super().__init__()
        self._stream = stream
        self._collective = collective
        # Optional per-AG event: wait() host-blocks on THIS AG's completion event
        # instead of the whole ag_stream (avoids over-draining a FWD-prefetched
        # later AG on the same stream). Still a host fence -> bit-exact.
        self._event = event
        self._done = False

    def wait(self, timeout=None) -> bool:  # noqa: ARG002
        if not self._done:
            # Per-AG event fence (host-wait on this AG only) if provided,
            # else the full-stream drain.
            _fence = (
                self._event.synchronize
                if self._event is not None
                else self._stream.synchronize
            )
            _fence()
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done


class MoriAllGather(AllGather):
    """Unified MORI all-gather backend for FSDP2 (single-node and cross-node).
    Routes intra-node over SDMA (XGMI) and, when the PG spans nodes, inter-node
    over RDMA. Usage: ``model.set_custom_all_gather(MoriAllGather())``.
    """

    def __init__(self, ranks_per_node: int | None = None) -> None:
        self._ranks_per_node = ranks_per_node
        self._collective: Any | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
        self._cap_bytes = 0
        self._output_buffer: torch.Tensor | None = None
        # The auto-tuning defaults below are cross-node only (num_nodes>=2), so
        # derive num_nodes from the launch env (set before the collective exists).
        world = int(os.environ.get("WORLD_SIZE", "0") or "0")
        if world > 0:
            rpn = self._ranks_per_node_value(world)
            num_nodes = world // rpn if rpn else 1
            # Zero-tuning defaults: bit-exact, faster than the framework default
            # with no env tuning. Applied via setdefault() so any explicit MORI_*
            # still wins. Cross-node only (num_nodes>=2).
            if num_nodes >= 2:
                _sd = os.environ.setdefault
                # fused hierarchical fill: SDMA intra reassembly + RDMA inter ring
                # (CU-free path, frees compute units for the GEMMs).
                _sd("MORI_HIER_FUSE_LOCAL", "1")
                _sd("MORI_HIER_FUSE_REMOTE", "1")
                _sd("MORI_HIER_LOCAL_PUSHONLY", "1")
                # Pipe depth is topology-aware: at 8 GPU/node every SDMA engine is
                # already committed, so the deep pipe / extra channels add no
                # bandwidth and risk an init fault on the giant embed/lm_head AG;
                # <=4 GPU/node has spare engines and benefits.
                if rpn < 8:
                    _sd("MORI_HIER_DEEP_PIPE", "auto")
                    _sd("MORI_SDMA_NUM_CHANNELS", "8")
                else:
                    # rpn>=8 (8 GPU/node) correctness gate; bit-exact base =
                    # rank-major copy-out with the host-drain landing fence:
                    #   * DEBUG_SYNC is the host stream.synchronize() fence draining
                    #     cross-PE RDMA/SDMA completions (only bit-exact fence here)
                    #     and disables the poison-prone HIP-graph capture;
                    #   * CUDA_GRAPH=0 belt-and-suspenders.
                    _sd("MORI_HIER_DEBUG_SYNC", "1")
                    _sd("MORI_HIER_CUDA_GRAPH", "0")
                # Deferred host landing fence: issue non-blocking, drain the one
                # reliable host fence at copy-out so it overlaps the backward GEMM.
                # Bit-exact (drains before the consumer reads).
                _sd("MORI_FSDP_DEFER_HOSTSYNC", "1")
                _sd("MORI_FSDP_EVENT_FENCE", "1")
                _sd("MORI_FSDP_FWD_PREFETCH", "1")
        self._host_proxy = os.environ.get("MORI_FSDP_HOST_PROXY", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        # Deferred-completion overlap for the host-proxy path
        # (MORI_HOSTPROXY_ASYNC=1, default OFF): post RDMA write + step-1 now
        # (non-blocking), return a c10d Work whose wait() runs the landing fence
        # (wait_all + rail-pair barrier + step-3) -> round trip overlaps compute.
        self._hostproxy_async = os.environ.get("MORI_HOSTPROXY_ASYNC", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        # Async correctness defaults (bit-exact only with both):
        #   * ASYNC_DRAIN: host-drain the AG stream after _complete so step-3 is
        #     landed+visible before copy-out reads ``out`` (else garbage remote
        #     half -> NaN);
        #   * ASYNC_RING=2: double-buffer recv staging so the partner's next-op
        #     write cannot overtake this op's step-3 read.
        # setdefault => explicit overrides still win.
        if self._hostproxy_async:
            os.environ.setdefault("MORI_HOSTPROXY_ASYNC_DRAIN", "1")
            os.environ.setdefault("MORI_HOSTPROXY_ASYNC_RING", "2")
        self._hostproxy_async_drain = os.environ.get(
            "MORI_HOSTPROXY_ASYNC_DRAIN", ""
        ) not in ("", "0", "false", "False")
        # Deferred device-path host landing fence (MORI_FSDP_DEFER_HOSTSYNC=1,
        # default OFF): issue the fused device AG non-blocking, host
        # stream.synchronize() at copy-out so the reliable fence overlaps the
        # backward GEMM. Composes with FUSE_LOCAL/FUSE_REMOTE.
        self._defer_hostsync = os.environ.get("MORI_FSDP_DEFER_HOSTSYNC", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        # MORI_FSDP_EVENT_FENCE=1 (default OFF): under FWD_PREFETCH the next
        # layer's AG is already on the shared ag_stream at copy-out, so
        # ``stream.synchronize()`` over-drains (L's fence blocks on L+1's landing,
        # serializing tails). Instead host-wait on a per-AG event draining only up
        # to L. Still a host fence (bit-exact). Default OFF => byte-identical
        # stream fence.
        self._event_fence = os.environ.get("MORI_FSDP_EVENT_FENCE", "") not in (
            "",
            "0",
            "false",
            "False",
        )

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        numel = 1
        for dim in size:
            numel *= int(dim)
        if (
            self._output_buffer is not None
            and self._output_buffer.dtype == dtype
            and self._output_buffer.device == device
            and self._output_buffer.numel() >= numel
        ):
            return self._output_buffer.narrow(0, 0, numel)
        self._output_buffer = torch.empty(numel, dtype=dtype, device=device)
        return self._output_buffer

    def _ranks_per_node_value(self, world_size: int) -> int:
        if self._ranks_per_node is not None:
            return self._ranks_per_node
        env = os.environ.get("LOCAL_WORLD_SIZE")
        if env:
            return int(env)
        return min(torch.cuda.device_count(), world_size)

    def _get_collective(self, group: dist.ProcessGroup, per_rank_bytes: int) -> Any:
        rank, world_size = group.rank(), group.size()
        if (
            self._collective is not None
            and self._rank == rank
            and self._world_size == world_size
            and self._cap_bytes >= per_rank_bytes
        ):
            return self._collective

        # Host-proxy E2E path (default OFF): route the AG through the CPU-posted
        # HostProxyHierAllGather instead of the device IBGDA HierAllGather. Same
        # handle(inp, out, numel, stream) contract; host wait_all + rail-pair
        # barrier is the landing fence. Built once with a generous cap (no mid-run
        # rebuild).
        if os.environ.get("MORI_FSDP_HOST_PROXY", "") not in (
            "",
            "0",
            "false",
            "False",
        ):
            HostProxy = importlib.import_module("mori.ccl").HostProxyHierAllGather
            ranks_per_node = self._ranks_per_node_value(world_size)
            cap_floor = int(os.environ.get("MORI_FSDP_HOSTPROXY_CAP_MB", "160")) * (
                1 << 20
            )
            cap = max(per_rank_bytes, cap_floor, self._cap_bytes)
            if self._collective is not None:
                raise RuntimeError(
                    "HostProxyHierAllGather built with cap "
                    f"{self._cap_bytes} B but a {per_rank_bytes} B AG arrived; "
                    "raise MORI_FSDP_HOSTPROXY_CAP_MB"
                )
            self._collective = HostProxy(
                rank,
                world_size,
                ranks_per_node,
                output_buffer_size=cap * world_size,
            )
            self._rank = rank
            self._world_size = world_size
            self._cap_bytes = cap
            return self._collective

        shmem = importlib.import_module("mori.shmem")
        HierAllGather = importlib.import_module("mori.ccl").HierAllGather
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        if my_pe != rank or npes != world_size:
            raise RuntimeError(
                "MORI FSDP Hier allgather requires the FSDP process group to "
                f"match SHMEM PEs, got rank/world_size={rank}/{world_size} and "
                f"my_pe/npes={my_pe}/{npes}"
            )
        cap = max(per_rank_bytes, self._cap_bytes)
        ranks_per_node = self._ranks_per_node_value(world_size)
        self._collective = HierAllGather(
            my_pe,
            npes,
            input_buffer_size=cap,
            output_buffer_size=cap * world_size,
            copy_output_to_user=True,
            ranks_per_node=ranks_per_node,
        )
        self._rank = rank
        self._world_size = world_size
        self._cap_bytes = cap
        return self._collective

    def _validate(self, output_tensor, input_tensor, group) -> None:
        if not input_tensor.is_cuda or not output_tensor.is_cuda:
            raise RuntimeError("MORI FSDP Hier allgather requires CUDA tensors")
        if input_tensor.device != output_tensor.device:
            raise RuntimeError("MORI FSDP Hier allgather requires same device")
        if input_tensor.dtype != output_tensor.dtype:
            raise RuntimeError("MORI FSDP Hier allgather requires matching dtypes")
        expected = input_tensor.numel() * group.size()
        if output_tensor.numel() != expected:
            raise RuntimeError(
                f"MORI FSDP Hier allgather expected output numel {expected}, "
                f"got {output_tensor.numel()}"
            )
        if (input_tensor.numel() * input_tensor.element_size()) % 4 != 0:
            raise RuntimeError(
                "MORI FSDP Hier allgather requires 4-byte-aligned input bytes"
            )

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Any | None:
        self._validate(output_tensor, input_tensor, group)
        count = input_tensor.numel()
        per_rank_bytes = count * input_tensor.element_size()
        collective = self._get_collective(group, per_rank_bytes)
        device = input_tensor.device
        stream = torch.cuda.current_stream(device)
        # Cross-stream lifetime guard: FSDP issues this AG on a dedicated comm
        # stream, but the input was produced on / freed by reshard on the compute
        # stream. The allocator tracks only the input's compute stream, so it may
        # hand the storage to a later compute op while mori still reads it on the
        # comm stream -> corruption. record_stream defers reuse until this AG
        # completes (same for output). Sync-free; standard non-default-stream
        # collective safety contract.
        input_tensor.record_stream(stream)
        output_tensor.record_stream(stream)

        # Deferred-completion host-proxy path (HOST_PROXY + HOSTPROXY_ASYNC): post
        # RDMA write + step-1 now (non-blocking), return a c10d Work whose wait()
        # runs the landing fence at copy-out so the round trip overlaps compute.
        # One in-flight op; a still-pending op is defensively drained.
        if self._host_proxy and self._hostproxy_async:
            pend = getattr(collective, "_pending", None)
            if pend is not None:
                pend.wait()
            handle = collective.call_async(
                input_tensor, output_tensor, count, stream=stream
            )
            if handle is None:  # single-node degenerate path (already blocking)
                return None
            work = _HostProxyDeferredWork(
                collective, handle, drain=self._hostproxy_async_drain
            )
            collective._pending = work
            return work

        ok = collective(input_tensor, output_tensor, count, stream=stream)
        if not ok:
            raise RuntimeError("MORI HierAllGather call failed")
        # Deferred device-path host landing fence: the device AG was issued
        # non-blocking above; return a c10d Work whose wait() runs the host
        # stream.synchronize() at copy-out so the fence overlaps the backward GEMM.
        if self._defer_hostsync:
            _ev = None
            if self._event_fence:
                # Record L's completion on the ag_stream now (before any later
                # prefetched AG is queued) so wait() drains only L.
                _ev = torch.cuda.Event()
                _ev.record(stream)
            return _DeviceDeferredHostSyncWork(stream, collective, _ev)
        if async_op:
            event = torch.cuda.Event()
            event.record(stream)
            return _HierWork(event, device)
        return None

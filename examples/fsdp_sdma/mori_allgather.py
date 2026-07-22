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

Intra-node traffic rides the SDMA copy engines (XGMI); inter-node traffic goes
over RDMA (NIC). Wired into FSDP2 via ``FSDPModule.set_custom_all_gather``.

``HierAllGather`` exposes a param-contiguous zero-copy path
(``enqueue_param_contiguous``) for the cross-node (num_nodes>=2, slice_direct
over RDMA) case: the gathered result is pushed straight into FSDP's
``[param][rank]`` output, eliminating the rank-major -> param copy-out. On
single-node (num_nodes==1) the direct path is unavailable, so this backend
keeps the rank-major copy-out there.
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
    """c10d Work that defers the host-blocking landing fence of the host-proxy AG.

    The all-gather's ``_post`` (RDMA write + step-1 intra gather) already ran
    non-blocking; this Work's ``wait()`` runs ``_complete`` (wait_all + rail-pair
    barrier + step-3 intra gather). FSDP invokes ``.wait()`` at copy-out -- after
    it has issued the current layer's compute and prefetched the next unshard --
    so the cross-node RDMA round trip + the intra gather overlap the caller's
    backward GEMM instead of stalling the host mid-all-gather. Must subclass c10d
    Work: FSDP's foreach_all_gather_copy_out calls ``.wait()`` iff the returned
    object is a ``dist.distributed_c10d.Work``.
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
            # MORI_HOSTPROXY_ASYNC_DRAIN=1: host-drain the AG stream after
            # _complete so step-3 (the remote-half broadcast) is landed+visible
            # before FSDP's copy-out reads ``out``. _complete only enqueues step-3
            # on the captured AG stream; if the consumer stream is not ordered
            # after it, copy-out races the still-pending step-3 -> the remote half
            # stays garbage -> NaN.
            if self._drain:
                self._handle["stream"].synchronize()
            self._collective._pending = None
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done


class _DeviceDeferredHostSyncWork(dist.distributed_c10d.Work):
    """c10d Work that DEFERS the device-path host landing fence to copy-out.

    The device HierAllGather (fused fill, FUSE_LOCAL/FUSE_REMOTE) is issued
    non-blocking on the dedicated all_gather_stream; the only reliably bit-exact
    landing fence on this MI300X/mlx5 is a host ``stream.synchronize()`` (it drains
    both SDMA copy-engine HSA signals and RDMA CQEs, which on-device/on-stream
    fences do not). Doing that sync inline at issue time stalls the host
    mid-all-gather and caps throughput. Deferring it to ``wait()`` lets FSDP issue
    this layer's compute and prefetch the next unshard between the AG issue and the
    copy-out wait(), so the host sync overlaps the backward GEMM. FSDP's
    foreach_all_gather_copy_out calls ``.wait()`` iff the returned object subclasses
    ``dist.distributed_c10d.Work``. Bit-exact: the sync drains before the consumer
    reads.
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
        # Optional per-AG event -- when set, wait() host-blocks on THIS AG's
        # completion event instead of the whole ag_stream (avoids over-draining a
        # FWD-prefetched later AG queued on the same stream). event.synchronize()
        # is still a host fence, so it drains SDMA/RDMA HW completion (bit-exact).
        self._event = event
        self._done = False

    def wait(self, timeout=None) -> bool:  # noqa: ARG002
        if not self._done:
            # Async completion-ordering: if the composition ran the inter leg on
            # an off-thread async worker (HOSTPROXY_ASYNC), join it first so every
            # chunkReadyFlag is published+landed before this fence gates the
            # consumer. Without this the caller-stream synchronize can fire while
            # the worker is still publishing a later AG's flags -> reassembly reads
            # stale flags -> loss drifts. On the non-async path drain is a no-op.
            drain = getattr(self._collective, "drain_hostproxy", None)
            if drain is not None:
                drain()
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
    """Unified MORI all-gather backend for FSDP2 — the same class for single-node
    and cross-node. Internally routes intra-node traffic over SDMA (XGMI) and, when
    the process group spans multiple nodes, inter-node traffic over RDMA. User code
    is identical in both cases: ``model.set_custom_all_gather(MoriAllGather())``.
    """

    supports_param_contiguous_output = False

    def __init__(self, ranks_per_node: int | None = None) -> None:
        self._ranks_per_node = ranks_per_node
        self._collective: Any | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
        self._cap_bytes = 0
        self._output_buffer: torch.Tensor | None = None
        self._pc_split_sizes: list[int] | None = None
        self._pc_split_offsets: list[int] | None = None
        self._pc_input_numel: int | None = None
        # Param-contiguous zero-copy is only available cross-node (num_nodes>=2,
        # slice_direct over RDMA). FSDP reads this attribute before the collective
        # exists, so derive num_nodes from the launch env (torchrun sets both).
        world = int(os.environ.get("WORLD_SIZE", "0") or "0")
        if world > 0:
            rpn = self._ranks_per_node_value(world)
            num_nodes = world // rpn if rpn else 1
            self.supports_param_contiguous_output = num_nodes >= 2
            # Zero-tuning defaults (MORI_FSDP_AUTO=1, on by default): make
            # `model.set_custom_all_gather(MoriAllGather())` bit-exact and faster
            # than the framework default with no env tuning. Cross-node, turn on
            # the fused SDMA-intra + RDMA-inter overlap path and its deferred
            # (bit-exact) host landing fence, and pick the pipe depth from the
            # topology. Everything is applied with setdefault(), so any MORI_*
            # variable set explicitly still wins; MORI_FSDP_AUTO=0 restores the
            # fully-manual (all-off) behavior.
            if num_nodes >= 2 and os.environ.get("MORI_FSDP_AUTO", "1") not in (
                "0",
                "false",
                "False",
            ):
                _sd = os.environ.setdefault
                # fused hierarchical fill: SDMA intra-node reassembly + RDMA inter
                # ring (the CU-free path that frees compute units for the GEMMs).
                _sd("MORI_HIER_FUSE_LOCAL", "1")
                _sd("MORI_HIER_FUSE_REMOTE", "1")
                _sd("MORI_HIER_LOCAL_PUSHONLY", "1")
                # Pipe depth is topology-aware. On 8-GPU/node x >=2 nodes
                # (world>=16) the fused plain path already saturates the fabric and
                # temporal deep-pipelining adds no bandwidth while risking an init
                # fault on the giant embed/lm_head AG, so leave the deep pipe off
                # and rely on the fused host-drain path (bit-exact). Smaller
                # topologies (<=4 GPU/node) benefit, and use the per-size adaptive
                # depth. <=4 GPU/node also fans the reassembly across more SDMA
                # queues (spare engines exist); at 8 GPU/node every engine is
                # already committed, so raising the channel count over-subscribes
                # and is left at the library default.
                if rpn < 8:
                    _sd("MORI_HIER_DEEP_PIPE", "auto")
                    _sd("MORI_SDMA_NUM_CHANNELS", "8")
                else:
                    # rpn >= 8 (world>=16, 8 GPU/node) correctness gate.
                    # At 8 ranks/node the zero-copy param-contiguous scatter
                    # (enqueue_param_contiguous) produces uniform-garbage output
                    # (loss stuck at ln(vocab); works at rpn==4/w8), and the
                    # copy-out __call__'s HIP-graph capture poisons the HIP context
                    # -> a "Shmem state is not initialized" SIGABRT at the first
                    # cross-node AG. The bit-exact dense-node base is the copy-out
                    # path with the full per-op host-drain landing fence:
                    #   * MORI_FSDP_NO_ZERO_COPY=1 routes off the broken zero-copy
                    #     scatter to the rank-major copy-out __call__ path;
                    #   * MORI_HIER_DEBUG_SYNC=1 is the host stream.synchronize()
                    #     landing fence that drains the cross-PE RDMA/SDMA
                    #     completions (the only bit-exact fence at 8 ranks/node) and
                    #     disables the poison-prone HIP-graph capture (the __call__
                    #     graph path is gated `and not self._debug_sync`).
                    #   * MORI_HIER_CUDA_GRAPH=0 belt-and-suspenders (redundant once
                    #     DEBUG_SYNC is on, but explicit for anyone toggling sync).
                    # w8 (rpn==4) never enters this branch -> unchanged.
                    _sd("MORI_FSDP_NO_ZERO_COPY", "1")
                    _sd("MORI_HIER_DEBUG_SYNC", "1")
                    _sd("MORI_HIER_CUDA_GRAPH", "0")
                # Deferred host landing fence: issue the AG non-blocking and drain
                # the one reliable host fence at copy-out so it overlaps the
                # backward GEMM / forward prefetch instead of stalling inline.
                # Bit-exact by construction (drains before the consumer reads).
                _sd("MORI_FSDP_DEFER_HOSTSYNC", "1")
                _sd("MORI_FSDP_EVENT_FENCE", "1")
                _sd("MORI_FSDP_FWD_PREFETCH", "1")
                _sd("MORI_FSDP_FWD_PREFETCH_ROOT", "1")
        # MORI_FSDP_NO_ZERO_COPY=1 forces the copy-out __call__ path (which keeps
        # the fuse_local/slice_direct_overlap inter-ring overlap) instead of the
        # zero-copy scatter.
        if os.environ.get("MORI_FSDP_NO_ZERO_COPY", "") not in (
            "",
            "0",
            "false",
            "False",
        ):
            self.supports_param_contiguous_output = False
        # Host-proxy path (MORI_FSDP_HOST_PROXY=1): the CPU-posted collective has
        # no enqueue_param_contiguous method, so it must go through the rank-major
        # copy-out __call__ branch. Force param-contiguous off so FSDP allocates a
        # plain rank-major output and never calls the zero-copy scatter.
        self._host_proxy = os.environ.get("MORI_FSDP_HOST_PROXY", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        if self._host_proxy:
            self.supports_param_contiguous_output = False
        # Deferred-completion overlap for the host-proxy path
        # (MORI_HOSTPROXY_ASYNC=1, default OFF). When set, the host-proxy AG posts
        # the cross-node RDMA write + step-1 intra gather now (non-blocking) and
        # returns a c10d Work whose wait() runs the host-blocking landing fence
        # (wait_all + rail-pair barrier + step-3), so the cross-node round trip
        # overlaps the caller's compute instead of stalling the host mid-AG.
        self._hostproxy_async = os.environ.get("MORI_HOSTPROXY_ASYNC", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        # Async correctness defaults. The deferred-completion path is bit-exact
        # only with both of:
        #   * ASYNC_DRAIN: host-drain the AG stream after _complete so step-3 (the
        #     remote-half broadcast) is landed+visible before FSDP's copy-out reads
        #     ``out`` (else a completion-visibility race -> garbage remote half ->
        #     total NaN);
        #   * ASYNC_RING=2: double-buffer the recv staging so the partner's next-op
        #     RDMA write cannot overtake this op's step-3 read (else a residual
        #     nondeterministic drift in later windows).
        # Enabling ASYNC without these races, so turn them on by default whenever
        # ASYNC is requested (setdefault => explicit overrides still win).
        if self._hostproxy_async:
            os.environ.setdefault("MORI_HOSTPROXY_ASYNC_DRAIN", "1")
            os.environ.setdefault("MORI_HOSTPROXY_ASYNC_RING", "2")
        self._hostproxy_async_drain = os.environ.get(
            "MORI_HOSTPROXY_ASYNC_DRAIN", ""
        ) not in ("", "0", "false", "False")
        # Deferred device-path host landing fence (MORI_FSDP_DEFER_HOSTSYNC=1,
        # default OFF). For the device HierAllGather (fused fill), issue the AG
        # non-blocking and return a c10d Work whose wait() does the host
        # stream.synchronize() landing fence at copy-out -- so the reliable host
        # fence overlaps the backward GEMM instead of stalling inline. This is the
        # fused-fill counterpart of the host-proxy async path. Composes with
        # FUSE_LOCAL/FUSE_REMOTE (fast device fill).
        self._defer_hostsync = os.environ.get("MORI_FSDP_DEFER_HOSTSYNC", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        # MORI_FSDP_EVENT_FENCE=1 (default OFF). The deferred landing fence
        # normally does a full ``stream.synchronize()`` on the shared
        # all_gather_stream. Under FWD_PREFETCH=1 the next layer's AG is already
        # issued on that same stream by the time this layer's copy-out wait() fires,
        # so ``stream.synchronize()`` over-drains: layer L's fence blocks on L+1's
        # RDMA landing too, serializing tails that should overlap. Instead record a
        # per-AG CUDA event right after L's AG kernel and host-wait on that event
        # (event.synchronize()) -- a host-blocking landing fence that drains only up
        # to L's completion, letting L+1's landing overlap L's compute/consume.
        # Still a host fence (bit-exact, unlike on-device event.wait ordering);
        # does not reorder landing->consume for L. Default OFF => byte-identical
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

        # Host-proxy E2E path (MORI_FSDP_HOST_PROXY=1, default OFF). Route the
        # FSDP all-gather through the persistent CPU-posted hierarchical transport
        # (mori.ccl.HostProxyHierAllGather) instead of the device IBGDA
        # HierAllGather. Same handle(inp, out, numel, stream) contract, so the
        # __call__ ``else`` branch drives it unchanged. The host completion
        # (wait_all) + rail-pair barrier is the landing fence. Built once with a
        # generous per-rank cap (no mid-run rebuild -> no engine/port churn).
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

    def prepare_param_contiguous_output(
        self,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object | None:
        """Build per-param split metadata (in DTYPE elements) for the direct
        param-contiguous scatter. ``HierAllGather.enqueue_param_contiguous``
        writes param ``s`` (per-rank numel ``E_s`` at cumulative input offset
        ``O_s``) so global rank ``r``'s slice lands at ``O_s*W + r*E_s`` == the
        exact ``[param][rank]`` layout FSDP views in place.
        """
        self.clear_param_contiguous_output()
        if not self.supports_param_contiguous_output:
            return None
        if not all_gather_input_split_sizes:
            raise RuntimeError("MORI zero-copy allgather requires non-empty splits")
        if sum(all_gather_input_split_sizes) != all_gather_input_numel:
            raise RuntimeError(
                "MORI zero-copy allgather split sizes do not match input numel"
            )
        element_size = torch.empty((), dtype=dtype).element_size()
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for split_size in all_gather_input_split_sizes:
            e = int(split_size)
            # SDMA byte extents must be 4-byte aligned (both size and offset).
            if (e * element_size) % 4 != 0 or (offset * element_size) % 4 != 0:
                raise RuntimeError(
                    "MORI zero-copy allgather requires 4-byte-aligned splits"
                )
            sizes.append(e)
            offsets.append(offset)
            offset += e
        # Store Python lists (not GPU tensors): passing GPU tensors into
        # enqueue_param_contiguous forces a .tolist() D2H sync on every per-layer
        # all-gather, draining the async pipeline and destroying the AG<->backward
        # overlap. Lists are consumed sync-free; enqueue_param_contiguous caches
        # the u32 GPU tensors internally.
        self._pc_split_sizes = sizes
        self._pc_split_offsets = offsets
        self._pc_input_numel = all_gather_input_numel
        return (self._pc_split_sizes, self._pc_split_offsets)

    def clear_param_contiguous_output(self) -> None:
        self._pc_split_sizes = None
        self._pc_split_offsets = None
        self._pc_input_numel = None

    def _can_call_param_contiguous(self, input_tensor: torch.Tensor) -> bool:
        if self._pc_split_sizes is None or self._pc_split_offsets is None:
            return False
        # Compare against the cached python-int numel -- no .item()/.sum() D2H
        # sync (that would fire on every call and serialize the whole step).
        if self._pc_input_numel != input_tensor.numel():
            self.clear_param_contiguous_output()
            return False
        return True

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
        # Cross-stream lifetime guard: FSDP2 issues this all-gather on a dedicated
        # comm stream, but the sharded-param input was produced on -- and is
        # freed/recycled by reshard_after_forward on -- the compute stream. The
        # caching allocator only tracks the input's original (compute) stream, so
        # it may hand the input's storage to a later compute-stream op while mori
        # is still reading it on the comm stream -> the big embed/lm_head AGs read a
        # partially-overwritten input and diverge. record_stream tells the
        # allocator the input is in use on the AG stream, deferring reuse until this
        # AG completes. Same for the output (FSDP may recycle it too). Sync-free (no
        # host stall) and the standard non-default-stream collective safety
        # contract.
        input_tensor.record_stream(stream)
        output_tensor.record_stream(stream)

        # Deferred-completion host-proxy path (MORI_FSDP_HOST_PROXY=1 +
        # MORI_HOSTPROXY_ASYNC=1): post the cross-node RDMA write + step-1 intra
        # gather now (non-blocking) and return a c10d Work whose wait() runs the
        # landing fence at copy-out. FSDP issues this layer's compute + the next
        # unshard prefetch between the post and the wait, so the cross-node round
        # trip overlaps the backward GEMM. The staging heap holds one in-flight op;
        # FSDP's schedule guarantees copy-out(N) (which wait()s) precedes
        # all-gather(N+1). A still-pending op is defensively drained.
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

        if self._can_call_param_contiguous(input_tensor):
            ok = collective.enqueue_param_contiguous(
                input_tensor,
                output_tensor,
                count,
                self._pc_split_sizes,
                self._pc_split_offsets,
                stream=stream,
            )
            if not ok:
                # FSDP already committed to the [param][rank] layout; a rank-major
                # fallback would corrupt it. Fail loudly instead (the cross-node
                # slice_direct path is expected to be available on the target run).
                raise RuntimeError(
                    "MORI HierAllGather param-contiguous path unavailable "
                    "(slice_direct/RDMA required); refusing rank-major fallback"
                )
        else:
            ok = collective(input_tensor, output_tensor, count, stream=stream)
            if not ok:
                raise RuntimeError("MORI HierAllGather call failed")
        # Deferred device-path host landing fence: the fused/zero-copy device AG
        # was issued non-blocking above; return a c10d Work whose wait() runs the
        # host stream.synchronize() at copy-out so the reliable landing fence
        # overlaps the caller's backward GEMM on the fast fused fill.
        if self._defer_hostsync:
            _ev = None
            if self._event_fence:
                # Record L's completion on the ag_stream NOW (before any
                # later prefetched AG is queued) so wait() drains only L.
                _ev = torch.cuda.Event()
                _ev.record(stream)
            return _DeviceDeferredHostSyncWork(stream, collective, _ev)
        if async_op:
            event = torch.cuda.Event()
            event.record(stream)
            return _HierWork(event, device)
        return None


class MoriIntraSubGroupAllGather(AllGather):
    """MORI all-gather for HSDP / hybrid-shard, where the FSDP shard group is a
    single node's ranks (4 GPU). The all-gather then rides pure intra-node SDMA
    (XGMI copy engines) with no inter-node RDMA ring. The inter-node grad
    all-reduce (the replicate dim) stays on native for both native and mori modes.

    SHMEM is initialized globally (all PEs, over the "default" PG). This backend
    drives ``IntraNodeSubGroupAllgatherSdma`` over the arithmetic sub-group
    ``{pe_base .. pe_base+group_size-1}`` of global PEs (pe_base = (my_pe //
    group_size) * group_size), so no per-subgroup SHMEM re-init is needed.
    """

    supports_param_contiguous_output = True

    def __init__(self) -> None:
        self._collective = None
        self._cap_bytes = 0
        self._group_size = None
        self._my_pe = None
        self._output_buffer: torch.Tensor | None = None
        self._pc_split_sizes: torch.Tensor | None = None
        self._pc_split_offsets: torch.Tensor | None = None
        self._direct_reg_ptr = None
        if os.environ.get("MORI_FSDP_NO_ZERO_COPY", "") not in (
            "",
            "0",
            "false",
            "False",
        ):
            self.supports_param_contiguous_output = False

    def allocate(self, size, *, dtype, device):
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

    def _get_collective(self, group, per_rank_bytes):
        shmem = importlib.import_module("mori.shmem")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        gsize = group.size()
        if (
            self._collective is not None
            and self._group_size == gsize
            and self._my_pe == my_pe
            and self._cap_bytes >= per_rank_bytes
        ):
            return self._collective
        Cls = importlib.import_module("mori.ccl").IntraNodeSubGroupAllgatherSdma
        cap = max(per_rank_bytes, self._cap_bytes)
        group_pos = my_pe % gsize
        pe_base = (my_pe // gsize) * gsize
        self._collective = Cls(
            my_pe,
            npes,
            out_buffer_bytes=cap * gsize,
            group_size=gsize,
            group_pos=group_pos,
            pe_base=pe_base,
            pe_stride=1,
        )
        self._cap_bytes = cap
        self._group_size = gsize
        self._my_pe = my_pe
        self._direct_reg_ptr = None  # new handle -> re-register output
        return self._collective

    def prepare_param_contiguous_output(
        self,
        all_gather_input_split_sizes,
        all_gather_input_numel,
        world_size,
        dtype,
        device,
    ):
        self.clear_param_contiguous_output()
        if not self.supports_param_contiguous_output:
            return None
        if not all_gather_input_split_sizes:
            raise RuntimeError("MORI zero-copy allgather requires non-empty splits")
        if sum(all_gather_input_split_sizes) != all_gather_input_numel:
            raise RuntimeError("MORI zero-copy split sizes do not match input numel")
        element_size = torch.empty((), dtype=dtype).element_size()
        sizes, offsets, offset = [], [], 0
        for split_size in all_gather_input_split_sizes:
            e = int(split_size)
            if (e * element_size) % 4 != 0 or (offset * element_size) % 4 != 0:
                raise RuntimeError(
                    "MORI zero-copy allgather requires 4-byte-aligned splits"
                )
            sizes.append(e)
            offsets.append(offset)
            offset += e
        self._pc_split_sizes = torch.tensor(sizes, dtype=torch.int64, device=device)
        self._pc_split_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        return (self._pc_split_sizes, self._pc_split_offsets)

    def clear_param_contiguous_output(self):
        self._pc_split_sizes = None
        self._pc_split_offsets = None

    def _can_call_param_contiguous(self, input_tensor):
        if self._pc_split_sizes is None or self._pc_split_offsets is None:
            return False
        if int(self._pc_split_sizes.sum().item()) != input_tensor.numel():
            self.clear_param_contiguous_output()
            return False
        return True

    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        if not input_tensor.is_cuda or not output_tensor.is_cuda:
            raise RuntimeError("MORI intra allgather requires CUDA tensors")
        count = input_tensor.numel()
        elem = input_tensor.element_size()
        if (count * elem) % 4 != 0:
            raise RuntimeError(
                "MORI intra allgather requires 4-byte-aligned input bytes"
            )
        gsize = group.size()
        if output_tensor.numel() != count * gsize:
            raise RuntimeError("MORI intra allgather output numel mismatch")
        collective = self._get_collective(group, count * elem)
        device = input_tensor.device
        stream = torch.cuda.current_stream(device)
        if self._can_call_param_contiguous(input_tensor):
            out_ptr = output_tensor.data_ptr()
            if out_ptr != self._direct_reg_ptr:
                if self._direct_reg_ptr is not None:
                    collective.deregister_output_buffer_ptr(self._direct_reg_ptr)
                collective.register_output_buffer(output_tensor)
                self._direct_reg_ptr = out_ptr
            u32 = 4
            blk_stride_u32 = (count * elem) // u32
            ss = self._pc_split_sizes.tolist()
            so = self._pc_split_offsets.tolist()
            split_sizes_u32 = torch.tensor(
                [(E * elem) // u32 for E in ss], dtype=torch.int64, device=device
            )
            split_offsets_u32 = torch.tensor(
                [(off * elem) // u32 for off in so], dtype=torch.int64, device=device
            )
            collective.gather_kernel_direct_param_contiguous(
                input_tensor,
                output_tensor,
                blk_stride_u32,
                1,  # num_blocks = 1 (single node block; pure intra)
                gsize,  # world_size for the [param][rank] output stride
                split_sizes_u32,
                split_offsets_u32,
                stream=stream,
                prepare_barrier=True,
                first_block=0,
            )
            collective.finish_direct_stream(stream=stream, barrier=True)
        else:
            ok = collective(input_tensor, output_tensor, count, stream=stream)
            if not ok:
                raise RuntimeError("MORI intra allgather call failed")
        if async_op:
            event = torch.cuda.Event()
            event.record(stream)
            return _HierWork(event, device)
        return None


# Backward-compatible alias (old name).
MoriHierAllGather = MoriAllGather

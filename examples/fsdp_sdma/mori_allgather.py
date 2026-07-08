"""Cross-node FSDP2 all-gather backend backed by mori.ccl.HierAllGather.

Intra-node traffic rides the SDMA copy engines (XGMI); inter-node traffic goes
over RDMA (NIC). Wired into FSDP2 via ``FSDPModule.set_custom_all_gather``.

``HierAllGather`` now exposes a PARAM-CONTIGUOUS zero-copy path
(``enqueue_param_contiguous``) for the cross-node (num_nodes>=2, slice_direct
over RDMA) case: the gathered result is PUSHED straight into FSDP's
``[param][rank]`` output, eliminating the rank-major -> param copy-OUT that made
SDMA FSDP lose to RCCL. On single-node (num_nodes==1) the direct path is
unavailable, so this backend keeps the rank-major copy-out there.
"""

import importlib
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather


# --- lightweight per-call size histogram (MORI_FSDP_AG_PROFILE=1) ---------
# No CUDA sync -> zero perturbation; sizes are model-determined so they are
# identical for the RCCL and SDMA runs. Buckets by log2(per-rank bytes) to see
# which regime FSDP's all-gathers land in (SDMA loses <4MB, wins >=8MB).
_AG_PROFILE = os.environ.get("MORI_FSDP_AG_PROFILE", "") not in ("", "0", "false", "False")
_ag_hist: dict[int, int] = {}
_ag_calls = 0


def _ag_profile_record(nbytes: int) -> None:
    global _ag_calls
    _ag_calls += 1
    b = nbytes.bit_length() - 1 if nbytes > 0 else 0  # floor(log2)
    _ag_hist[b] = _ag_hist.get(b, 0) + 1
    if _ag_calls % 100 == 0:
        _ag_profile_dump()


def _ag_profile_dump() -> None:
    import sys
    total = sum(_ag_hist.values())
    lines = [f"AGHIST calls={total}"]
    for b in sorted(_ag_hist):
        lo = 1 << b
        lines.append(f"  [{lo/1e6:8.3f}MB..{2*lo/1e6:8.3f}MB) : {_ag_hist[b]:5d}")
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


# --- in-situ per-layer AG-output bit-exact verify (MORI_FSDP_AG_VERIFY=1) --
# Turn 9: the UT proves the AG kernel is bit-exact + deterministic under a
# SYNTHETIC overlap pattern, yet FSDP loss drifts (~-0.2%). This probe runs
# INSIDE a real 2-node step: right after the mori copy-out AG fills the
# rank-major output, it SNAPSHOTS that output at the EARLIEST stream point
# (output.clone() enqueued immediately after the AG, before any masking
# latency), then computes the RCCL truth (all_gather_into_tensor of the SAME
# input) and counts bit-exact mismatches. If mismatches > 0 in-situ (while the
# UT is clean), the FSDP-specific trigger (reshard buffer reuse / tight
# back-to-back scheduling) is confirmed as the completion-ordering exposure. If
# zero, the drift is downstream of the AG (reduce/accum). One host sync per dump.
_AG_VERIFY = os.environ.get("MORI_FSDP_AG_VERIFY", "") not in ("", "0", "false", "False")
_verify_calls = 0
_verify_mismatch_calls = 0
_verify_total_mismatch_elems = 0
_verify_max_abs_diff = 0.0
_verify_first_bad = None  # (call_idx, per_rank_numel, mismatch_elems, max_abs)
_verify_pending: list = []  # (snap, ref, call_idx, per_rank_numel, out_ptr)
_verify_ptr_prev: int = 0        # data_ptr of the immediately-preceding AG output
_verify_recycle_calls = 0        # #calls whose output reuses the prior call's addr
_verify_recycle_bad = 0          # #of those recycling calls that ALSO mismatched


def _ag_verify_flush() -> None:
    global _verify_mismatch_calls, _verify_total_mismatch_elems
    global _verify_max_abs_diff, _verify_first_bad
    global _verify_ptr_prev, _verify_recycle_calls, _verify_recycle_bad
    if not _verify_pending:
        return
    import torch
    torch.cuda.synchronize()
    for snap, ref, idx, prn, out_ptr in _verify_pending:
        recycled = out_ptr != 0 and out_ptr == _verify_ptr_prev
        if recycled:
            _verify_recycle_calls += 1
        _verify_ptr_prev = out_ptr
        ne = torch.ne(snap, ref)
        n_bad = int(ne.sum().item())
        if n_bad:
            d = (snap.to(torch.float32) - ref.to(torch.float32)).abs()
            mx = float(d.max().item())
            _verify_mismatch_calls += 1
            _verify_total_mismatch_elems += n_bad
            if recycled:
                _verify_recycle_bad += 1
            if mx > _verify_max_abs_diff:
                _verify_max_abs_diff = mx
            if _verify_first_bad is None:
                _verify_first_bad = (idx, prn, n_bad, mx, hex(out_ptr))
    _verify_pending.clear()


def _ag_verify_dump() -> None:
    import sys
    _ag_verify_flush()
    sys.stderr.write(
        f"AGVERIFY calls={_verify_calls} mismatch_calls={_verify_mismatch_calls} "
        f"total_mismatch_elems={_verify_total_mismatch_elems} "
        f"max_abs_diff={_verify_max_abs_diff:.6g} first_bad={_verify_first_bad} "
        f"recycle_calls={_verify_recycle_calls} recycle_bad={_verify_recycle_bad}\n"
    )
    sys.stderr.flush()


# --- per-call GPU-time timing (MORI_FSDP_AG_TIMING=1) ----------------------
# Measures the EFFECTIVE per-all-gather GPU bandwidth INSIDE the FSDP step, to
# compare against the standalone slice_direct number (141 GB/s @ 64MiB/rank,
# 1.06x behind RCCL). If the FSDP per-AG bandwidth ~= standalone, the ~13% FSDP
# gap is pure exposure/overlap; if it's much lower, per-call host/launch/finish
# overhead is the cause. Bucketed by log2(out bytes). One cuda sync per dump.
_AG_TIMING = os.environ.get("MORI_FSDP_AG_TIMING", "") not in ("", "0", "false", "False")
_ag_time_pending: list = []  # (start_event, end_event, out_bytes, bucket)
_ag_time_acc: dict = {}      # bucket -> [sum_ms, sum_out_bytes, n]


def _ag_timing_flush() -> None:
    import torch
    if not _ag_time_pending:
        return
    torch.cuda.synchronize()
    for s, e, ob, b in _ag_time_pending:
        ms = s.elapsed_time(e)
        acc = _ag_time_acc.setdefault(b, [0.0, 0, 0])
        acc[0] += ms
        acc[1] += ob
        acc[2] += 1
    _ag_time_pending.clear()


def _ag_timing_dump() -> None:
    import sys
    _ag_timing_flush()
    lines = ["AGTIME (effective per-AG GPU bandwidth = out_bytes/elapsed)"]
    for b in sorted(_ag_time_acc):
        sms, sob, n = _ag_time_acc[b]
        lo = 1 << b
        gbps = (sob / 1e9) / (sms / 1e3) if sms > 0 else 0.0
        lines.append(
            f"  [{lo/1e6:8.3f}MB..{2*lo/1e6:8.3f}MB) out: n={n:5d} "
            f"avg={sms/max(n,1):7.3f}ms  {gbps:7.1f} GB/s"
        )
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


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
    barrier + step-3 intra gather). FSDP invokes ``.wait()`` at copy-out -- AFTER
    it has issued the current layer's compute and prefetched the next unshard --
    so the ~1.5ms cross-node RDMA round trip + the intra gather overlap the
    caller's backward GEMM instead of stalling the host mid-all-gather (the
    overlap window native RCCL gets by returning a Work). Must subclass c10d Work:
    FSDP's foreach_all_gather_copy_out calls ``.wait()`` iff the returned object
    is a ``dist.distributed_c10d.Work``.
    """

    def __init__(self, collective, handle) -> None:
        super().__init__()
        self._collective = collective
        self._handle = handle
        self._done = False

    def wait(self, timeout=None) -> bool:  # noqa: ARG002
        if not self._done:
            self._collective._complete(self._handle)
            self._collective._pending = None
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done


class _DeviceDeferredHostSyncWork(dist.distributed_c10d.Work):
    """c10d Work that DEFERS the device-path host landing fence to copy-out.

    The DEVICE HierAllGather (fused fill, FUSE_LOCAL/FUSE_REMOTE) is issued
    non-blocking on the dedicated all_gather_stream; the ONLY reliably bit-exact
    landing fence on this MI300X/mlx5 is a host ``stream.synchronize()`` (drains
    both SDMA copy-engine HSA signals and RDMA CQEs -- on-device/on-stream fences
    are 26x-refuted). Doing that sync INLINE at issue time caps at ~112 TFLOPS
    (0.72x) because the host stalls mid-all-gather. Deferring it to ``wait()``
    means FSDP issues this layer's compute + prefetches the next unshard between
    the AG issue and the copy-out wait(), so the host sync overlaps the backward
    GEMM -- the same overlap that made the host-proxy async path bit-exact at
    118.44 (Turn 18), but on the FAST fused device fill (0.97x UT) instead of the
    IBGDA-capped host-proxy transport. FSDP's foreach_all_gather_copy_out calls
    ``.wait()`` iff the returned object subclasses ``dist.distributed_c10d.Work``.
    """

    def __init__(self, stream: "torch.cuda.Stream") -> None:
        super().__init__()
        self._stream = stream
        self._done = False

    def wait(self, timeout=None) -> bool:  # noqa: ARG002
        if not self._done:
            self._stream.synchronize()
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done


class MoriAllGather(AllGather):
    """Unified MORI all-gather backend for FSDP2 — the SAME class for single-node
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
        # PARAM-CONTIGUOUS zero-copy is only available cross-node (num_nodes>=2,
        # slice_direct over RDMA). FSDP reads this attribute BEFORE the collective
        # exists, so derive num_nodes from the launch env (torchrun sets both).
        world = int(os.environ.get("WORLD_SIZE", "0") or "0")
        if world > 0:
            rpn = self._ranks_per_node_value(world)
            num_nodes = world // rpn if rpn else 1
            self.supports_param_contiguous_output = num_nodes >= 2
        # A/B switch: MORI_FSDP_NO_ZERO_COPY=1 forces the copy-OUT __call__ path
        # (which keeps the fuse_local/slice_direct_overlap inter-ring overlap) so
        # the zero-copy scatter can be compared against it in the same session.
        if os.environ.get("MORI_FSDP_NO_ZERO_COPY", "") not in ("", "0", "false", "False"):
            self.supports_param_contiguous_output = False
        # HOST-PROXY path (MORI_FSDP_HOST_PROXY=1): the CPU-posted collective has
        # NO enqueue_param_contiguous method, so it must go through the rank-major
        # copy-out __call__ branch. Force param-contiguous OFF so FSDP allocates a
        # plain rank-major output and never calls the zero-copy scatter.
        self._host_proxy = os.environ.get(
            "MORI_FSDP_HOST_PROXY", "") not in ("", "0", "false", "False")
        if self._host_proxy:
            self.supports_param_contiguous_output = False
        # DEFERRED-COMPLETION overlap lever for the host-proxy path
        # (MORI_HOSTPROXY_ASYNC=1, default OFF). When set, the host-proxy AG posts
        # the cross-node RDMA write + step-1 intra gather now (non-blocking) and
        # returns a c10d Work whose wait() runs the host-blocking landing fence
        # (wait_all + rail-pair barrier + step-3), so the ~1.5ms cross-node round
        # trip overlaps the caller's compute instead of stalling the host mid-AG.
        self._hostproxy_async = os.environ.get(
            "MORI_HOSTPROXY_ASYNC", "") not in ("", "0", "false", "False")
        # DEFERRED device-path host landing fence (MORI_FSDP_DEFER_HOSTSYNC=1,
        # default OFF). For the DEVICE HierAllGather (fused fill), issue the AG
        # non-blocking and return a c10d Work whose wait() does the host
        # stream.synchronize() landing fence at copy-out -- so the reliable host
        # fence OVERLAPS the backward GEMM instead of stalling inline (which caps
        # at ~112). This is the fused-fill counterpart of the host-proxy async
        # path. Composes with FUSE_LOCAL/FUSE_REMOTE (fast device fill).
        self._defer_hostsync = os.environ.get(
            "MORI_FSDP_DEFER_HOSTSYNC", "") not in ("", "0", "false", "False")
        # TARGETED completion sync for the LARGE-band all-gathers only. AGVERIFY
        # localized the residual cross-node completion race to the ~29M-elem
        # embed/lm_head AG (call#3/#4): the inter-node ring's remote RDMA puts are
        # not globally visible to the copy-out drain by the time the recorded
        # event fires, but ONLY at that largest size (per-layer AGs are clean).
        # Every device-fence/env toggle failed; a host sync (DEBUG_SYNC on EVERY
        # call) fixes the loss but costs ~23% perf. This env host-syncs the
        # caller stream ONLY when per-rank bytes >= threshold, so just the ~2
        # offender AGs/step are forced to complete (negligible perf) while all
        # calls still ride SDMA cross-node. 0 = off.
        self._sync_big_bytes = int(os.environ.get("MORI_FSDP_SYNC_BIG_BYTES", "0") or "0")
        # T22 DIAGNOSTIC: restrict the big-AG host-sync to only the FORWARD or only
        # the BACKWARD all-gather phase, to localize whether the loss-stall (~11.0,
        # forward stale lm_head/embed) and the 16x-grad collapse (backward) are
        # separable loci. FSDP's backward re-unshard runs inside the autograd
        # engine with grad DISABLED; the forward unshard runs grad-ENABLED. So we
        # use torch.is_grad_enabled() to tell the phase apart. Values: "both"
        # (default), "fwd", "bwd".
        self._sync_phase = os.environ.get("MORI_FSDP_SYNC_PHASE", "both").strip().lower()
        # CHUNKING lever (T24). The T23 SYNC-threshold sweep localized the ENTIRE
        # cross-node completion race + the ENTIRE ~24% sync cost to a SINGLE giant
        # all-gather (per-rank size in (40,80]MB = the tied embed/lm_head at
        # VOCAB=32000); per-layer AGs (<40MB per-rank) are race-free. Hypothesis:
        # splitting that one giant AG into K sub-all-gathers, each landing in the
        # race-free per-layer band, avoids the timing race WITHOUT any host drain
        # -> bit-exact AT FULL SPEED. Each sub-AG is a copy-out into a temp
        # rank-major buffer, then a strided 2D scatter into the user output
        # (out[:, j*c:(j+1)*c] = tmp[:, :]). Only the big AG is chunked; small AGs
        # take the normal single-call path (chunk_bytes=0 disables entirely).
        self._chunk_big_bytes = int(os.environ.get("MORI_FSDP_CHUNK_BIG_BYTES", "0") or "0")
        self._chunk_k = int(os.environ.get("MORI_FSDP_CHUNK_K", "4") or "4")
        self._chunk_tmp: torch.Tensor | None = None

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

        # HOST-PROXY E2E path (MORI_FSDP_HOST_PROXY=1, default OFF). Route the
        # FSDP all-gather through the persistent CPU-posted hierarchical transport
        # (mori.ccl.HostProxyHierAllGather) instead of the device IBGDA
        # HierAllGather. Same handle(inp, out, numel, stream) contract, so the
        # __call__ ``else`` branch drives it unchanged. The host completion
        # (wait_all) + rail-pair barrier is the landing fence, so this is the
        # first E2E test of whether the host-proxy transport is bit-exact
        # (loss == GT) -- the open gap flagged by R28-R30. Built ONCE with a
        # generous per-rank cap (no mid-run rebuild -> no engine/port churn).
        if os.environ.get("MORI_FSDP_HOST_PROXY", "") not in ("", "0", "false", "False"):
            HostProxy = importlib.import_module("mori.ccl").HostProxyHierAllGather
            ranks_per_node = self._ranks_per_node_value(world_size)
            cap_floor = int(os.environ.get("MORI_FSDP_HOSTPROXY_CAP_MB", "160")) * (1 << 20)
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
        # Store PYTHON lists (not GPU tensors): passing GPU tensors into
        # enqueue_param_contiguous forced a .tolist() D2H sync on EVERY per-layer
        # all-gather, draining the async pipeline and destroying the AG<->backward
        # overlap (the whole cross-node FSDP gap). Lists are consumed sync-free;
        # enqueue_param_contiguous caches the u32 GPU tensors internally.
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
        # Compare against the CACHED python-int numel -- no .item()/.sum() D2H
        # sync (that fired on every call and serialized the whole step).
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
        if _AG_PROFILE:
            _ag_profile_record(per_rank_bytes)
        collective = self._get_collective(group, per_rank_bytes)
        device = input_tensor.device
        stream = torch.cuda.current_stream(device)
        # CROSS-STREAM lifetime guard (the FSDP accuracy race, localized by the
        # rapid-fire XSTREAM+FREE_INPUT probe): FSDP2 issues this all-gather on a
        # dedicated comm stream but the sharded-param INPUT was produced on -- and
        # is FREED/recycled by reshard_after_forward on -- the compute stream. The
        # caching allocator only tracks the input's ORIGINAL (compute) stream, so
        # it may hand the input's storage to a later compute-stream op while mori
        # is still reading it on the comm stream (a read on `stream` that the
        # allocator does not know about) -> the big embed/lm_head AGs read a
        # partially-overwritten input and diverge. record_stream tells the
        # allocator the input is in use on the AG stream, deferring reuse until
        # this AG completes. Same for the output (FSDP may recycle it too). This
        # is sync-free (no host stall) and is the standard non-default-stream
        # collective safety contract.
        input_tensor.record_stream(stream)
        output_tensor.record_stream(stream)

        # DEFERRED-COMPLETION host-proxy path (MORI_FSDP_HOST_PROXY=1 +
        # MORI_HOSTPROXY_ASYNC=1): post the cross-node RDMA write + step-1 intra
        # gather now (non-blocking) and return a c10d Work whose wait() runs the
        # landing fence at copy-out. FSDP issues this layer's compute + the next
        # unshard prefetch between the post and the wait, so the ~1.5ms cross-node
        # round trip overlaps the backward GEMM (the no-CU-contention dividend).
        # The staging heap holds ONE in-flight op; FSDP's schedule guarantees
        # copy-out(N) (which wait()s) precedes all-gather(N+1). A still-pending op
        # is defensively drained.
        if self._host_proxy and self._hostproxy_async:
            pend = getattr(collective, "_pending", None)
            if pend is not None:
                pend.wait()
            handle = collective.call_async(
                input_tensor, output_tensor, count, stream=stream)
            if handle is None:  # single-node degenerate path (already blocking)
                return None
            work = _HostProxyDeferredWork(collective, handle)
            collective._pending = work
            return work

        _t_start = None
        if _AG_TIMING:
            _t_start = torch.cuda.Event(enable_timing=True)
            _t_start.record(stream)
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
        elif (
            self._chunk_big_bytes
            and per_rank_bytes >= self._chunk_big_bytes
            and self._chunk_k >= 2
            and count % (self._chunk_k * max(1, 4 // input_tensor.element_size())) == 0
        ):
            # CHUNKED copy-out: split the one giant AG into K sub-AGs each in the
            # race-free size band, then strided-scatter into the user output.
            # Mathematically identical to a single rank-major AG (each sub-AG is
            # bit-exact; the scatter is exact) but each transfer is small enough
            # to physically land before the consumer reads -> no host sync needed.
            world = group.size()
            K = self._chunk_k
            c = count // K
            csz = c * input_tensor.element_size()
            sub = self._get_collective(group, max(csz, self._cap_bytes))
            need = world * c
            if (
                self._chunk_tmp is None
                or self._chunk_tmp.dtype != input_tensor.dtype
                or self._chunk_tmp.device != input_tensor.device
                or self._chunk_tmp.numel() < need
            ):
                self._chunk_tmp = torch.empty(
                    need, dtype=input_tensor.dtype, device=input_tensor.device
                )
            out2d = output_tensor.view(world, count)
            for j in range(K):
                o = j * c
                in_sub = input_tensor.narrow(0, o, c)
                tmp = self._chunk_tmp.narrow(0, 0, need)
                ok = sub(in_sub, tmp, c, stream=stream)
                if not ok:
                    raise RuntimeError("MORI HierAllGather chunk call failed")
                out2d[:, o:o + c].copy_(tmp.view(world, c))
        else:
            ok = collective(input_tensor, output_tensor, count, stream=stream)
            if not ok:
                raise RuntimeError("MORI HierAllGather call failed")
            if _AG_VERIFY:
                global _verify_calls
                _verify_calls += 1
                # SNAPSHOT the mori output at the EARLIEST stream point (right
                # after the AG kernel) so we capture any early-completion stale
                # bytes before a later op masks them.
                snap = output_tensor.clone()
                ref = torch.empty_like(output_tensor)
                work = dist.all_gather_into_tensor(
                    ref, input_tensor, group=group, async_op=True
                )
                work.wait()  # current stream waits on the RCCL truth
                _verify_pending.append(
                    (snap, ref.clone(), _verify_calls, count,
                     output_tensor.data_ptr())
                )
                if len(_verify_pending) >= 64:
                    _ag_verify_dump()
        if _AG_TIMING and _t_start is not None:
            _t_end = torch.cuda.Event(enable_timing=True)
            _t_end.record(stream)
            ob = output_tensor.numel() * output_tensor.element_size()
            b = ob.bit_length() - 1 if ob > 0 else 0
            _ag_time_pending.append((_t_start, _t_end, ob, b))
            if len(_ag_time_pending) >= 50:
                _ag_timing_dump()
        # DEFERRED device-path host landing fence: the fused/zero-copy device AG
        # was issued non-blocking above; return a c10d Work whose wait() runs the
        # host stream.synchronize() at copy-out so the reliable landing fence
        # overlaps the caller's backward GEMM (the async-overlap dividend on the
        # FAST fused fill). Skips the inline _sync_big_bytes host stall.
        if self._defer_hostsync:
            # Only the big embed/lm_head AGs race on this HW (Turn 17); small
            # per-layer AGs are fenced sufficiently by the fused kernel's
            # on-stream completion + FSDP's own recorded event. A host sync on
            # EVERY small AG is wasted host stall. When MORI_FSDP_SYNC_BIG_BYTES
            # is set, defer the host fence ONLY on big AGs; small AGs return None
            # (device-event path) so they overlap freely. Threshold 0 = fence all.
            if not self._sync_big_bytes or per_rank_bytes >= self._sync_big_bytes:
                return _DeviceDeferredHostSyncWork(stream)
            return None
        # Targeted large-band completion sync (see __init__): force the biggest
        # cross-node AGs (the localized race band) to fully land before returning,
        # so a consumer/next-op cannot read stale remote-half bytes. Only ~2
        # calls/step exceed the threshold, so perf impact is small.
        if self._sync_big_bytes and per_rank_bytes >= self._sync_big_bytes:
            _phase_ok = True
            if self._sync_phase == "fwd":
                _phase_ok = torch.is_grad_enabled()
            elif self._sync_phase == "bwd":
                _phase_ok = not torch.is_grad_enabled()
            if _phase_ok:
                stream.synchronize()
        if async_op:
            event = torch.cuda.Event()
            event.record(stream)
            return _HierWork(event, device)
        return None


class MoriIntraSubGroupAllGather(AllGather):
    """MORI all-gather for HSDP / hybrid-shard, where the FSDP shard group is a
    single node's ranks (4 GPU). The all-gather then rides PURE intra-node SDMA
    (XGMI copy engines) with NO inter-node RDMA ring -- exactly the regime where
    single-node SDMA+zero-copy beat RCCL by +24.6%. The inter-node grad
    all-reduce (the replicate dim) stays on RCCL for both native and mori modes,
    so this is a fair A/B on the all-gather path only.

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
        if os.environ.get("MORI_FSDP_NO_ZERO_COPY", "") not in ("", "0", "false", "False"):
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
        self, all_gather_input_split_sizes, all_gather_input_numel,
        world_size, dtype, device,
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
                raise RuntimeError("MORI zero-copy allgather requires 4-byte-aligned splits")
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
            raise RuntimeError("MORI intra allgather requires 4-byte-aligned input bytes")
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
                [(O * elem) // u32 for O in so], dtype=torch.int64, device=device
            )
            collective.gather_kernel_direct_param_contiguous(
                input_tensor,
                output_tensor,
                blk_stride_u32,
                1,            # num_blocks = 1 (single node block; pure intra)
                gsize,        # world_size for the [param][rank] output stride
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


class TimedDefaultAllGather(AllGather):
    """Diagnostic backend: performs the all-gather with torch's default RCCL
    collective (``all_gather_into_tensor``) but brackets it with the SAME
    cuda-event per-call timer used by ``MoriAllGather`` (MORI_FSDP_AG_TIMING).

    Purpose (Turn 12 disambiguation): measure RCCL's per-AG GPU time UNDER the
    identical FSDP overlap/prefetch schedule, apples-to-apples with the mori
    per-AG numbers. If RCCL per-AG ~= mori per-AG -> both hidden behind compute
    (benign overlap-sharing, gap is step scheduling); if RCCL per-AG is much
    smaller -> mori's AG is genuinely EXPOSED under FSDP.
    """

    def allocate(self, size, *, dtype, device):
        numel = 1
        for d in size:
            numel *= int(d)
        return torch.empty(numel, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        device = input_tensor.device
        stream = torch.cuda.current_stream(device)
        _t_start = None
        if _AG_TIMING:
            _t_start = torch.cuda.Event(enable_timing=True)
            _t_start.record(stream)
        work = dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=group, async_op=True
        )
        if _AG_TIMING and _t_start is not None:
            # RCCL runs on the PG's OWN stream; make the current stream WAIT on
            # the collective before recording the end event, so the [start,end]
            # interval on `stream` actually brackets the AG completion (as the
            # current stream perceives it) -- apples-to-apples exposure vs mori.
            work.wait()
            _t_end = torch.cuda.Event(enable_timing=True)
            _t_end.record(stream)
            ob = output_tensor.numel() * output_tensor.element_size()
            b = ob.bit_length() - 1 if ob > 0 else 0
            _ag_time_pending.append((_t_start, _t_end, ob, b))
            if len(_ag_time_pending) >= 50:
                _ag_timing_dump()
            return None
        if async_op:
            return work
        work.wait()
        return None


# Backward-compatible alias (old name).
MoriHierAllGather = MoriAllGather

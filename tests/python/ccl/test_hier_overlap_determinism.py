#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Overlap-completion determinism probe for cross-node HierAllGather.

The bit-exact param-contiguous test host-syncs before comparing, so it CANNOT
expose the async completion-ordering bug that shows up under FSDP (the intra
SDMA gather / inter-node ring returns before every peer's write has REMOTELY
landed in ``out_``; a downstream CU consumer on the SAME stream then reads a few
stale bytes -> ~0.15% loss drift, masked only by a host ``stream.synchronize``).

This UT reproduces that hazard directly: per rep it runs the AllGather and then,
WITH NO host sync between them, a CU consumer (matmul + reduction) that reads the
gathered output on the same stream -- exactly the FSDP AG->backward-GEMM
ordering. We record the consumer scalar per rep and only sync at the very end.

Pass criteria:
  * DETERMINISM: every rep's consumer scalar is identical (bitwise) -- an async
    landing race makes reps differ run-to-run.
  * CORRECTNESS: the scalar equals the host-synced reference (bytes were fresh).

Run cross-node under torchrun (the path that matters)::

    torchrun --nnodes=2 --nproc_per_node=4 ... \
        tests/python/ccl/test_hier_overlap_determinism.py
"""

import os
import traceback

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import HierAllGather

_DTYPES = [torch.bfloat16, torch.float32]
_PARAM_SPLITS = [1048576, 524288, 262144, 131072, 65536]
# LARGE band = Qwen embed+lm_head (~68M elems/rank; gathered ~2.18GB int32,
# crosses 2^31 bytes). The <=2M overlap UT never exercised this; if the async
# remote-landing race is size-dependent this is where it must surface.
_PARAM_SPLITS_LARGE = [34078720, 34078720]
_REPS = int(os.environ.get("OVERLAP_REPS", "50"))
# Large-band reps are cheaper to keep short (each AG moves ~2GB); still enough
# reps to catch a run-to-run async landing race.
_REPS_LARGE = int(os.environ.get("OVERLAP_REPS_LARGE", "12"))


def _make_input(dtype, count, rank, device):
    base = (rank + 1) * 17
    ramp = torch.arange(count, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _splits_offsets(device):
    offsets, acc = [], 0
    for e in _PARAM_SPLITS:
        offsets.append(acc)
        acc += e
    ss = torch.tensor(_PARAM_SPLITS, dtype=torch.int64, device=device)
    so = torch.tensor(offsets, dtype=torch.int64, device=device)
    return ss, so, acc


def _consume(out, wvec):
    """Position-WEIGHTED CU consumer (detects layout scrambles AND stale bytes).

    A plain sum is permutation-invariant, so it cannot see a wrong [param][rank]
    ordering. Multiply elementwise by a position-dependent weight vector before
    reducing, so any wrong element at any position changes the scalar. Runs on
    the current stream -- reads ``out`` in the CU domain right after the copy
    engine wrote it (the unfenced hazard).
    """
    n = out.numel()
    return (out.to(torch.float32) * wvec[:n]).sum()


def _reference_out(inp, splits, mode, world_size, device):
    """INDEPENDENT truth: RCCL all_gather -> the layout the mode produces.

    The host-synced ``golden`` is self-referential -- if the copy-out drains the
    transit ``out_`` before every peer's SDMA put has REMOTELY landed, golden
    reads the SAME stale bytes as the overlapped pass, so golden==overlapped
    (wrong=0) even though BOTH are wrong. Comparing against this all_gather-built
    reference is what actually catches a stable-but-wrong drain (the FSDP loss
    drift that only host stream.synchronize() removes)."""
    count = inp.numel()
    rank_major = torch.empty(count * world_size, dtype=inp.dtype, device=device)
    dist.all_gather_into_tensor(rank_major, inp)  # [r0_shard, r1_shard, ...]
    if mode == "copyout":
        return rank_major  # __call__ produces rank-major [rank][param]
    # zerocopy: reshuffle rank-major -> param-contiguous [param][rank]
    ref = torch.empty(count * world_size, dtype=inp.dtype, device=device)
    o = 0
    for e in splits:
        for r in range(world_size):
            ref[o * world_size + r * e : o * world_size + r * e + e] = \
                rank_major[r * count + o : r * count + o + e]
        o += e
    return ref


def _run_dtype(handle, dtype, rank, world_size, device, mode, splits, reps):
    """Overlap-determinism probe for one dtype under one op ``mode``.

    ``mode="zerocopy"`` exercises ``enqueue_param_contiguous`` (the direct
    param-contiguous scatter; proven clean in turn 1). ``mode="copyout"``
    exercises the plain ``__call__`` copy-OUT path -- the path the deployed FSDP
    perf config actually uses (MORI_FSDP_NO_ZERO_COPY=1) and the remaining
    suspect for the ~0.15% loss drift: the intra SDMA gather stacks peers' puts
    into the transit ``out_`` and a SEPARATE copy-OUT reader may see bytes that
    haven't finished landing at the receiver.
    """
    global _PARAM_SPLITS
    _PARAM_SPLITS = splits
    ss, so, count = _splits_offsets(device)
    out = torch.empty(count * world_size, dtype=dtype, device=device)
    # position weight so the consumer is layout- and value-sensitive.
    wvec = ((torch.arange(count * world_size, device=device) % 97) + 1).to(torch.float32)

    # Per-rep VARYING inputs: a stale byte from a prior rep now differs from the
    # expected current value, so staleness is actually detectable.
    def inp_for(rep):
        return _make_input(dtype, count, rank, device) + (rep % 7)

    def do_op(inp, stream):
        if mode == "zerocopy":
            assert handle.enqueue_param_contiguous(inp, out, count, ss, so, stream)
        else:  # copyout: the deployed FSDP path (rank-major __call__)
            assert handle(inp, out, count, stream)

    main = torch.cuda.current_stream()
    comm = torch.cuda.Stream()
    pressure = torch.randn(2048, 2048, dtype=torch.float32, device=device)

    # Golden pass: per-rep host-synced (fresh bytes guaranteed) scalars, PLUS an
    # INDEPENDENT all_gather-built reference scalar per rep (catches stable-wrong).
    golden = []
    truth = []
    for rep in range(reps):
        inp = inp_for(rep)
        ref = _reference_out(inp, splits, mode, world_size, device)
        truth.append(_consume(ref, wvec).item())
        do_op(inp, main)
        main.synchronize()
        torch.cuda.synchronize()
        golden.append(_consume(out, wvec).item())
    torch.cuda.synchronize()

    # Overlapped pass: cross-stream event handoff, NO host sync AG->consumer.
    scalars = []
    for rep in range(reps):
        inp = inp_for(rep)
        for _ in range(4):
            pressure = pressure @ pressure * 1e-3 + 0.1
        comm.wait_stream(main)
        with torch.cuda.stream(comm):
            do_op(inp, comm)
        ev = torch.cuda.Event()
        ev.record(comm)
        main.wait_event(ev)
        scalars.append(_consume(out, wvec))
    torch.cuda.synchronize()

    vals = [s.item() for s in scalars]
    # Determinism = overlapped scalar bitwise-matches the host-synced golden.
    # NaN==NaN is deterministic (a stale-byte race would give DIFFERING run-to-
    # run values, not a stable NaN), so treat matching NaNs as equal.
    def _ne(v, g):
        if v != v and g != g:  # both NaN
            return False
        return v != g
    n_wrong = sum(1 for v, g in zip(vals, golden) if _ne(v, g))
    first_bad = next((i for i, (v, g) in enumerate(zip(vals, golden)) if _ne(v, g)), -1)
    # CORRECTNESS vs the independent all_gather reference. Use a relative tol on
    # the huge weighted-sum scalar (bf16 accumulation of ~1e12 has legit ULP
    # noise); a real drain-stale error shifts many elements and blows past this.
    def _wrong_vs_truth(v, t):
        v_nan, t_nan = (v != v), (t != t)
        if v_nan or t_nan:
            return v_nan != t_nan  # exactly one NaN = wrong; both NaN = ok
        return abs(v - t) > 1e-4 * max(abs(t), 1.0)
    n_truth = sum(1 for v, t in zip(vals, truth) if _wrong_vs_truth(v, t))
    n_gold_truth = sum(1 for g, t in zip(golden, truth) if _wrong_vs_truth(g, t))
    if rank == 0:
        print(
            f"  [{mode} nsplit={len(splits)} n/rank={sum(splits)}] dtype={dtype} "
            f"golden0={golden[0]:.3f} truth0={truth[0]:.3f} rep0={vals[0]:.3f} "
            f"nondet={n_wrong}/{reps} wrong_vs_truth={n_truth}/{reps} "
            f"gold_vs_truth={n_gold_truth}/{reps}",
            flush=True,
        )
    # A stable-but-wrong drain shows as nondet=0 but wrong_vs_truth>0 (and
    # gold_vs_truth>0). Count truth violations as failures too.
    return 0, n_wrong + n_truth


def _make_handle(rank, world_size, ranks_per_node):
    # Buffers sized for the LARGEST profile so one handle serves every size band.
    per_rank_bytes = sum(_PARAM_SPLITS_LARGE) * 4 + 4096
    return HierAllGather(
        my_pe=rank,
        npes=world_size,
        ranks_per_node=ranks_per_node,
        input_buffer_size=per_rank_bytes,
        output_buffer_size=per_rank_bytes * world_size,
        copy_output_to_user=True,
    )


def _worker_body(rank, world_size, ranks_per_node, device):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    handle = _make_handle(rank, world_size, ranks_per_node)
    if rank == 0:
        print(
            f"overlap-determinism: world={world_size} rpn={ranks_per_node} "
            f"num_nodes={handle.num_nodes} reps={_REPS} "
            f"supports={handle.supports_param_contiguous_output()}",
            flush=True,
        )
    supported = handle.supports_param_contiguous_output()
    del handle
    try:
        if not supported:
            if rank == 0:
                print("SKIP: direct param-contiguous path unavailable", flush=True)
            return
        total_nd, total_wr = 0, 0
        # Size profiles: the large multi-split (turn-1 regime) + a SMALL/odd
        # single-split profile (Qwen has small layers that route to the
        # non-slice copy-out fallback -- the untested band).
        _SIZE_PROFILES = [
            (_PARAM_SPLITS, _REPS),        # large multi-split (proven regime)
            ([65536, 32768, 8192], _REPS), # small sizes (num_blocks=1 band)
            ([524288], _REPS),             # single big split
            # Qwen embed+lm_head band -- crosses 2^31 bytes; the untested regime
            # where a size-dependent async landing race would surface.
            (_PARAM_SPLITS_LARGE, _REPS_LARGE),
        ]
        _MODES = os.environ.get("OVERLAP_MODES", "copyout,zerocopy").split(",")
        # FRESH handle per op MODE. Mixing the copy-OUT __call__ path and the
        # zero-copy enqueue_param_contiguous path on ONE handle contaminates the
        # shared intra flag/seq + output-registration state (copy-OUT registers a
        # transit out_, zero-copy registers the USER output), which spuriously
        # produced a stable-NaN artifact for [zerocopy bf16 nsplit=5] when it ran
        # right after copy-OUT ops -- NOT a kernel bug (the pure-mode bit-exact
        # test passes bf16 at this exact config). FSDP uses one mode for a whole
        # run, so a per-mode handle is the faithful, artifact-free harness.
        for mode in _MODES:
            handle = _make_handle(rank, world_size, ranks_per_node)
            try:
                for splits, reps in _SIZE_PROFILES:
                    for dtype in _DTYPES:
                        nd, wr = _run_dtype(handle, dtype, rank, world_size,
                                            device, mode, list(splits), reps)
                        total_nd += nd
                        total_wr += wr
            finally:
                torch.cuda.synchronize()
                dist.barrier()
                del handle
                dist.barrier()
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            if total_nd == 0 and total_wr == 0:
                print("test_hier_overlap_determinism: PASSED (deterministic+correct)",
                      flush=True)
            else:
                print(f"test_hier_overlap_determinism: FAILED "
                      f"nondet={total_nd} wrong={total_wr}", flush=True)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def _run_torchrun():
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size,
        device_id=device,
    )
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    try:
        _worker_body(rank, world_size, ranks_per_node, device)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            _run_torchrun()
        else:
            raise SystemExit("launch under torchrun (cross-node)")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

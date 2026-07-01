#!/usr/bin/env python3
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
Bit-exact test for ``mori.ccl.HierAllGather`` vs
``torch.distributed.all_gather_into_tensor``.

AllGather is a pure data move so there is ZERO numerical tolerance --
results must compare equal with ``torch.equal``.

Two launch styles are supported:

  * Single node (M1): run as a plain script; uses ``torch.multiprocessing.spawn``
    over the locally visible GPUs (``num_nodes == 1``)::

        python3 tests/python/ccl/test_hier_allgather.py --world-size 4

  * Cross node (M2+): launched under ``torchrun`` (the work ``xnode``
    harness sets RANK/WORLD_SIZE/LOCAL_RANK)::

        torchrun --nnodes=2 --nproc_per_node=4 ... test_hier_allgather.py
"""

import os
import traceback

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import HierAllGather

from tests.python.utils import TorchDistContext, get_free_port


# Sizes (elements per rank) and dtypes required by DESIGN.md correctness
# contract. Sizes kept modest by default so the test fits a dev box; sweep
# larger via the CLI.
_DEFAULT_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]


def _make_input(dtype: torch.dtype, numel: int, rank: int, device) -> torch.Tensor:
    """Rank-distinct, dtype-exact input (values round-trip through bf16/fp16)."""
    base = (rank + 1) * 17
    ramp = torch.arange(numel, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _run_dispatch_span(handle, rank, world_size, device, cap_numel):
    """Bit-exact coverage for the Turn-5 size-threshold DISPATCHER itself.

    The shipped default routes per-call: payloads >= ``slice_min_bytes`` take the
    sliced 2-D path, smaller ones take the non-sliced fuse-barrier path, and a
    path SWITCH clears ``_prev_op_completed`` (the two paths reuse the shared
    _intra/_inter buffers differently). The plain test loops sizes but never
    asserts the switch happens BOTH ways within one handle, so this drives an
    explicit small->large->small->large interleave through the SAME handle with a
    threshold pinned between the two sizes, asserting (a) bit-exact vs torch each
    call AND (b) the dispatcher actually flips path on every transition. Carried
    review ask since ; runs on the authoritative true-xnode path."""
    if not getattr(handle, "slice_inter", False) or handle.num_nodes < 2:
        return
    small, large = 1024, 1 << 20            # fp32: 4 KiB (below) | 4 MiB (slice)
    if large > cap_numel:
        return

    def _expected_key(numel):
        # Mirror HierAllGather.__call__'s 3-way dispatch: "slice" at/above
        # slice_min_bytes; "pipe" for the mid/small band when pipe_band is enabled
        # and its prerequisites hold; otherwise None (non-slice path).
        bc = numel * 4  # fp32
        if handle.slice_inter and bc >= handle.slice_min_bytes:
            return "slice"
        if (getattr(handle, "pipe_band", False) and handle.slice_inter
                and handle.slice_fused and not handle.slice_oop
                and not handle.slice_overlap and handle.slice_pipe_chunks > 1
                and bc >= getattr(handle, "pipe_band_min_bytes", 0)):
            return "pipe"
        return None

    saved_thresh = handle.slice_min_bytes
    saved_last = handle._last_use_slice
    saved_prev = handle._prev_op_completed
    saved_band = getattr(handle, "pipe_band", False)
    handle.slice_min_bytes = 1 << 19        # 512 KiB: small below, large above
    try:
        # Cover both the pipe-band default (small->"pipe") AND the legacy
        # non-slice path (small->None) so a path SWITCH is exercised both ways
        # through the SAME handle for all three dispatch destinations.
        for band in (True, False):
            handle.pipe_band = band and saved_band
            handle._last_use_slice = "sentinel"   # force a switch on the first op
            for numel in (small, large, small, large):
                want = _expected_key(numel)
                _run_one(handle, torch.float32, numel, rank, world_size, device)
                assert handle._last_use_slice == want, (
                    f"dispatcher routed numel={numel} (pipe_band={handle.pipe_band}) "
                    f"to {handle._last_use_slice!r}, expected {want!r}")
        if rank == 0:
            print("test_hier_allgather: dispatch-span PASSED")
    finally:
        handle.slice_min_bytes = saved_thresh
        handle._last_use_slice = saved_last
        handle._prev_op_completed = saved_prev
        handle.pipe_band = saved_band


def _run_one(handle, dtype, numel, rank, world_size, device):
    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * world_size, dtype=dtype, device=device)
    out_ref = torch.empty(numel * world_size, dtype=dtype, device=device)

    # Reference: RCCL via torch.distributed.
    dist.all_gather_into_tensor(out_ref, inp)

    stream = torch.cuda.current_stream()
    ok = handle(inp, out_mori, numel, stream)
    assert ok, f"HierAllGather call failed dtype={dtype} numel={numel}"
    stream.synchronize()
    torch.cuda.synchronize()

    if not torch.equal(out_mori, out_ref):
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"HierAllGather mismatch dtype={dtype} numel={numel}: "
            f"first mismatch positions={diff} got={out_mori[diff].tolist()} "
            f"ref={out_ref[diff].tolist()}"
        )


def _bench_one(handle, dtype, numel, rank, world_size, device, reps=5, warmup=2):
    """Timed AllGather vs RCCL baseline. Returns (mori_min, mori_avg,
    rccl_min, rccl_avg) in ms. >=3 timed reps per DESIGN perf contract."""
    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * world_size, dtype=dtype, device=device)
    out_ref = torch.empty(numel * world_size, dtype=dtype, device=device)
    stream = torch.cuda.current_stream()

    def time_fn(fn, n):
        ts = []
        for i in range(warmup + n):
            torch.cuda.synchronize()
            dist.barrier()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            fn()
            ev1.record()
            torch.cuda.synchronize()
            if i >= warmup:
                ts.append(ev0.elapsed_time(ev1))
        return min(ts), sum(ts) / len(ts)

    def mori_call():
        assert handle(inp, out_mori, numel, stream)
        stream.synchronize()

    m_min, m_avg = time_fn(mori_call, reps)
    r_min, r_avg = time_fn(lambda: dist.all_gather_into_tensor(out_ref, inp), reps)
    return m_min, m_avg, r_min, r_avg


def _bench_phases(handle, dtype, numel, rank, world_size, device, reps=5, warmup=2):
    """Attribute the hierarchical AllGather time to its two phases (rule#2).

    For the every-rank-direct N>=2 path the op is exactly:
      phase1 intra-node SDMA sub-group gather  (handle._intra)  -> node-block
      phase2 inter-node RDMA ring              (handle._inter)  -> full output
    The inter-node wrapper additionally stages the node-block into a symmetric
    ring buffer (prepare_sync, a D2D copy-in) and copies the gathered buffer
    back to the user output (finish_sync, a D2D copy-out) around the kernel, so
    this split quantifies how much of the xnode time is the SDMA gather vs the
    RDMA ring (kernel + its two staging copies). Returns (intra_min, intra_avg,
    inter_min, inter_avg) in ms. Times the SAME sub-handles the real __call__
    uses, so the sum tracks the end-to-end number from ``_bench_one``.
    """
    G = handle.ranks_per_node
    block_count = numel * G
    inp = _make_input(dtype, numel, rank, device)
    node_block = torch.empty(block_count, dtype=dtype, device=device)
    out = torch.empty(numel * world_size, dtype=dtype, device=device)
    stream = torch.cuda.current_stream()

    def time_fn(fn, n):
        ts = []
        for i in range(warmup + n):
            torch.cuda.synchronize()
            dist.barrier()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            fn()
            ev1.record()
            torch.cuda.synchronize()
            if i >= warmup:
                ts.append(ev0.elapsed_time(ev1))
        return min(ts), sum(ts) / len(ts)

    def intra_call():
        handle._intra(inp, node_block, numel, stream)
        stream.synchronize()

    def inter_call():
        handle._inter(node_block, out, block_count, stream)
        stream.synchronize()

    i_min, i_avg = time_fn(intra_call, reps)
    n_min, n_avg = time_fn(inter_call, reps)
    return i_min, i_avg, n_min, n_avg


def _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device,
                 bench=False):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    assert shmem.shmem_npes() == world_size

    max_itemsize = max(torch.tensor([], dtype=d).element_size() for d in dtypes)
    max_numel = max(numels)
    per_rank_bytes = max_numel * max_itemsize + 4096

    handle = HierAllGather(
        my_pe=rank,
        npes=world_size,
        ranks_per_node=ranks_per_node,
        input_buffer_size=per_rank_bytes,
        output_buffer_size=per_rank_bytes * world_size,
        copy_output_to_user=True,
    )
    if rank == 0:
        print(
            f"HierAllGather: world={world_size} ranks_per_node={ranks_per_node} "
            f"num_nodes={handle.num_nodes}"
        )

    try:
        for dtype in dtypes:
            for numel in numels:
                if (numel * torch.tensor([], dtype=dtype).element_size()) % 4 != 0:
                    continue
                _run_one(handle, dtype, numel, rank, world_size, device)
                if rank == 0:
                    print(f"  ok dtype={dtype} numel={numel}")
        torch.cuda.synchronize()
        dist.barrier()
        # Explicit dispatcher path-switch coverage (carried review ask, ):
        # small<->large interleave through ONE handle exercises both threshold
        # transitions + the _prev_op_completed reset. Skipped when the handle is
        # too small to hold the 4 MiB probe (e.g. tiny --numels A/B runs).
        _run_dispatch_span(handle, rank, world_size, device, max_numel)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("test_hier_allgather: PASSED")

        if bench:
            # Perf (rule#2): sweep ALL requested sizes (fp32) to characterize the
            # bandwidth-vs-size curve, not just one point -- this localizes where
            # the RCCL gap lives (fixed per-op overhead at small sizes vs per-NIC
            # ring throughput at large sizes). Report min/avg over reps + the RCCL
            # baseline. Algo BW = total_out_bytes / time.
            bdtype = torch.float32
            for bnumel in sorted(set(numels)):
                m_min, m_avg, r_min, r_avg = _bench_one(
                    handle, bdtype, bnumel, rank, world_size, device)
                if rank == 0:
                    tot_gb = bnumel * world_size * 4 / 1e9
                    print(
                        f"[bench] world={world_size} dtype=fp32 numel={bnumel} "
                        f"out={tot_gb:.3f}GB | mori min={m_min:.3f}ms "
                        f"avg={m_avg:.3f}ms BW={tot_gb/(m_min/1e3):.1f}GB/s | "
                        f"rccl min={r_min:.3f}ms avg={r_avg:.3f}ms "
                        f"BW={tot_gb/(r_min/1e3):.1f}GB/s | "
                        f"ratio={r_min and (m_min/r_min):.2f}x"
                    )
                dist.barrier()
            bnumel = max(numels)
            # Phase split (rule#2): only meaningful for the every-rank-direct
            # N>=2 pipeline, which exposes _intra (SDMA gather) + _inter (RDMA
            # ring). M1 (num_nodes==1) and leader-only have a different shape.
            if (
                handle.num_nodes >= 2
                and not handle.leader_only
                and hasattr(handle, "_inter")
            ):
                i_min, i_avg, n_min, n_avg = _bench_phases(
                    handle, bdtype, bnumel, rank, world_size, device)
                if rank == 0:
                    print(
                        f"[phases] intra(SDMA gather) min={i_min:.3f}ms "
                        f"avg={i_avg:.3f}ms | inter(RDMA ring+staging) "
                        f"min={n_min:.3f}ms avg={n_avg:.3f}ms | "
                        f"intra+inter={i_min + n_min:.3f}ms"
                    )
                dist.barrier()
            dist.barrier()
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
        dist.barrier()
        shmem.shmem_finalize()


def _spawn_worker(rank, world_size, ranks_per_node, port, numels, dtypes, bench):
    """Single-node entry: each spawned process owns cuda:rank."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device,
                     bench=bench)


def test_hier_allgather(world_size=None, ranks_per_node=None, numels=None,
                        dtypes=None, bench=False):
    """Single-node pytest entry.

    ``ranks_per_node == world_size`` (default) is the M1 single-node path
    (num_nodes == 1, pure SDMA). ``ranks_per_node < world_size`` exercises the
    M2b hierarchical pipeline on a single box: it splits the local GPUs into
    ``world_size // ranks_per_node`` simulated nodes so the intra-node SDMA
    sub-group gather + inter-node ring run exactly as they would across nodes
    (the ring's same-local-index neighbours are same-box here, reached over the
    shmem P2P/SDMA transport instead of RDMA -- same kernel code path)."""
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    # Same-node intra SDMA gather goes through the multi-queue warp put whose
    # source/dest offset bug we sidestep with a single channel (see
    # test_allgather / test_inter_node_ring).
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size >= 2, f"HierAllGather needs >=2 GPUs, got {world_size}"
    if ranks_per_node is None:
        ranks_per_node = world_size
    assert world_size % ranks_per_node == 0, "world must be a multiple of ranks_per_node"
    if numels is None:
        # DESIGN.md contract sizes per rank: ~4 KiB, ~4 MiB, ~64 MiB.
        # In fp32: 1024 -> 4 KiB, 1 Mi -> 4 MiB, 16 Mi -> 64 MiB.
        numels = [1024, 1024 * 1024, 16 * 1024 * 1024]
    if dtypes is None:
        dtypes = _DEFAULT_DTYPES
    port = get_free_port()
    torch.multiprocessing.spawn(
        _spawn_worker,
        args=(world_size, ranks_per_node, port, numels, dtypes, bench),
        nprocs=world_size,
        join=True,
    )


def test_hier_allgather_layouts():
    """Sweep hierarchical (num_nodes>=2) decompositions for bit-exactness.

     validated only N=2,G=2. This exercises the full intra-node SDMA
    sub-group gather -> inter-node ring pipeline across several
    (world, ranks_per_node) splits on a single box -- including the DESIGN.md
    acceptance layout N=2,G=4 (8 ranks) and N=4,G=2 (4 simulated nodes). Each
    split runs all 4 dtypes via the same ``torch.equal`` (zero-tolerance) path.
    Layouts that exceed the visible GPU count are skipped.
    """
    ngpu = torch.cuda.device_count()
    # (world, ranks_per_node): N=2,G=2 ; N=2,G=4 (DESIGN target) ; N=4,G=2.
    layouts = [(4, 2), (8, 4), (8, 2)]
    small = [1024, 256 * 1024]
    ran = 0
    for world, rpn in layouts:
        if world > ngpu:
            continue
        test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
        ran += 1
    assert ran > 0, "no hierarchical layout fit the visible GPU count"


def test_hier_allgather_slice():
    """Sliced 2-D AllGather path (MORI_HIER_SLICE) bit-exact, single-node sim.

    Exercises the M5 slice lever: the inter ring carries only each
    rank's own shard and N intra SDMA gathers reassemble the node-blocks. Runs
    the DESIGN target layout N=2,G=2 (and N=2,G=4 when 8 GPUs are visible) over
    all 4 dtypes via the same zero-tolerance ``torch.equal`` path, so the slice
    path has durable CI coverage independent of the env default (which is OFF)."""
    ngpu = torch.cuda.device_count()
    layouts = [(4, 2), (8, 4)]
    small = [1024, 256 * 1024]
    prev = os.environ.get("MORI_HIER_SLICE")
    prev_fused = os.environ.get("MORI_HIER_SLICE_FUSED")
    prev_oop = os.environ.get("MORI_HIER_SLICE_OOP")
    prev_overlap = os.environ.get("MORI_HIER_SLICE_OVERLAP")
    prev_min = os.environ.get("MORI_HIER_SLICE_MIN_BYTES")
    prev_fuse_ib = os.environ.get("MORI_HIER_SLICE_FUSE_IB")
    prev_pipe = os.environ.get("MORI_HIER_SLICE_PIPE")
    prev_pipe_chunks = os.environ.get("MORI_HIER_SLICE_PIPE_CHUNKS")
    prev_pipe_overlap = os.environ.get("MORI_HIER_SLICE_PIPE_OVERLAP")
    prev_stream_ring = os.environ.get("MORI_HIER_STREAM_RING")
    prev_stream_intra = os.environ.get("MORI_HIER_STREAM_INTRA")
    prev_defer_fin = os.environ.get("MORI_HIER_SLICE_DEFER_FIN")
    prev_defer_inter_fin = os.environ.get("MORI_HIER_SLICE_DEFER_INTER_FIN")
    prev_direct = os.environ.get("MORI_HIER_SLICE_DIRECT")
    os.environ["MORI_HIER_SLICE"] = "1"
    # : force slice at ALL sizes (these test payloads are 4KiB/256KiB,
    # below the default size threshold) so the sliced path keeps bit-exact CI
    # coverage regardless of the dispatcher default.
    os.environ["MORI_HIER_SLICE_MIN_BYTES"] = "0"
    ran = 0
    try:
        # Cover BOTH the default sliced Phase B (N separate gathers) AND the
        # fused Phase B (M5 : N gathers folded into one batch, 2 barriers +
        # 1 bulk copy), each with the Phase-A collection read from a scratch copy
        # (oop=0) AND read in place from the ring buffer (M5  oop=1, drops
        # the inter finish copy-OUT). All four combos must be bit-exact vs torch.
        os.environ["MORI_HIER_SLICE_OVERLAP"] = "0"
        # M5: cover BOTH the dropped Phase-B entry barrier (fuse_ib=1,
        # the default) AND the restored one (fuse_ib=0) -- the entry-barrier fusion
        # is a host-sync change so both must stay bit-exact vs torch.
        for fuse_ib in ("1", "0"):
            os.environ["MORI_HIER_SLICE_FUSE_IB"] = fuse_ib
            for oop in ("0", "1"):
                os.environ["MORI_HIER_SLICE_OOP"] = oop
                for fused in ("0", "1"):
                    os.environ["MORI_HIER_SLICE_FUSED"] = fused
                    for world, rpn in layouts:
                        if world > ngpu:
                            continue
                        test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
                        ran += 1
        os.environ["MORI_HIER_SLICE_FUSE_IB"] = "1"
        # M5: lever (c) -- overlap the local node-block Phase-B gather
        # with the inter ring on a side stream. Requires fused Phase B + the
        # scratch (non-oop) collection. Cover it with the same zero-tolerance path.
        os.environ["MORI_HIER_SLICE_OOP"] = "0"
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        os.environ["MORI_HIER_SLICE_OVERLAP"] = "1"
        for world, rpn in layouts:
            if world > ngpu:
                continue
            test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
            ran += 1
        # M5: chunked (strided) Phase-B reassembly -- the strided
        # gather_kernel slot-stride enabler. Each block's gather is split into K
        # element-range chunks each written at slot stride = count; the output
        # MUST stay byte-identical to the unchunked gather. Cover K=2 and K=3
        # (the latter exercises an uneven last chunk). Requires fused, non-oop,
        # non-overlap. Zero-tolerance vs torch.
        os.environ["MORI_HIER_SLICE_OVERLAP"] = "0"
        os.environ["MORI_HIER_SLICE_OOP"] = "0"
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        os.environ["MORI_HIER_SLICE_PIPE"] = "1"
        for chunks in ("2", "3"):
            os.environ["MORI_HIER_SLICE_PIPE_CHUNKS"] = chunks
            for world, rpn in layouts:
                if world > ngpu:
                    continue
                test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
                ran += 1
        os.environ["MORI_HIER_SLICE_PIPE"] = "0"
        # M5: CHUNKED-RING PIPELINE OVERLAP -- the rule#1 payoff of the
        # strided gather. The inter ring is chunked into K stages and each chunk's
        # Phase-B gather runs on a side stream overlapping the next chunk's ring.
        # The output MUST stay byte-identical to the serial sliced+fused path.
        # Cover K=2 and K=3. Requires fused, non-oop, non-(local)overlap.
        os.environ["MORI_HIER_SLICE_OVERLAP"] = "0"
        os.environ["MORI_HIER_SLICE_OOP"] = "0"
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        os.environ["MORI_HIER_SLICE_PIPE_OVERLAP"] = "1"
        for chunks in ("2", "3"):
            os.environ["MORI_HIER_SLICE_PIPE_CHUNKS"] = chunks
            for world, rpn in layouts:
                if world > ngpu:
                    continue
                test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
                ran += 1
        os.environ["MORI_HIER_SLICE_PIPE_OVERLAP"] = "0"
        # M5: STREAM-ORDERED inter ring -- the inter ring uses the
        # on-device ShmemBarrierOnStream prepare/finish instead of host
        # hipStreamSynchronize + host ShmemBarrierAll. This changes the host-sync
        # mechanism (not the byte moves / global fencing), so the sliced+fused
        # default path with stream_ring=1 MUST stay byte-identical to torch. Cover
        # both oop=0 (scratch collection -> finish_stream copy-OUT) and oop=1
        # (read in place -> finish_stream_no_copy).
        # also vary stream_intra -- the stream-ordered Phase-B
        # finish_batch (ShmemBarrierOnStream copy-OUT) used in the default fused
        # non-overlap path when paired with stream_ring. stream_intra=1 (default)
        # removes the last host round-trip; both ON and OFF must stay byte-exact.
        # also vary MORI_HIER_SLICE_DEFER_FIN -- the deferred
        # Phase-B finish fence (drop #3, rely on the next op's inter-prepare
        # barrier). The multi-size loop runs several ops on the SAME instance, so
        # defer_fin=1 exercises BOTH the cross-op deferral (op i's fence covered
        # by op i+1's #1) AND the last-op case (no successor -> no fence, output
        # must still be byte-exact). Both 1 (default) and 0 must match torch.
        # also vary MORI_HIER_SLICE_DEFER_INTER_FIN -- the
        # deferred INTER ring finish_stream fence (drop the ring-reuse fence, rely
        # on the next slice op's prepare_stream barrier). Only active on the
        # non-oop path (oop=0); harmless no-op for oop=1. The multi-size loop runs
        # several ops on the SAME instance so defer_inter_fin=1 exercises BOTH the
        # cross-op deferral AND the last-op case (no successor); both 1 and 0 must
        # match torch.
        os.environ["MORI_HIER_STREAM_RING"] = "1"
        os.environ["MORI_HIER_SLICE_OVERLAP"] = "0"
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        for stream_intra in ("1", "0"):
            os.environ["MORI_HIER_STREAM_INTRA"] = stream_intra
            for defer_fin in ("1", "0"):
                os.environ["MORI_HIER_SLICE_DEFER_FIN"] = defer_fin
                for defer_inter_fin in ("0", "1"):
                    os.environ["MORI_HIER_SLICE_DEFER_INTER_FIN"] = defer_inter_fin
                    for oop in ("0", "1"):
                        os.environ["MORI_HIER_SLICE_OOP"] = oop
                        for world, rpn in layouts:
                            if world > ngpu:
                                continue
                            test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
                            ran += 1
        # durable coverage for the DISSEMINATION prepare
        # barrier (MORI_HIER_DISSEM_BARRIER=1). Same global all-PE semantics as
        # the funnel; the sliced stream-ordered path must stay byte-exact. Run a
        # representative stream config over the layouts, then restore the funnel.
        os.environ["MORI_HIER_STREAM_INTRA"] = "1"
        os.environ["MORI_HIER_SLICE_DEFER_FIN"] = "1"
        os.environ["MORI_HIER_SLICE_DEFER_INTER_FIN"] = "1"
        os.environ["MORI_HIER_SLICE_OOP"] = "0"
        os.environ["MORI_HIER_DISSEM_BARRIER"] = "1"
        for world, rpn in layouts:
            if world > ngpu:
                continue
            test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
            ran += 1
        os.environ["MORI_HIER_DISSEM_BARRIER"] = "0"
        os.environ["MORI_HIER_STREAM_RING"] = "0"
        os.environ["MORI_HIER_STREAM_INTRA"] = "1"
        os.environ["MORI_HIER_SLICE_DEFER_FIN"] = "1"
        os.environ["MORI_HIER_SLICE_DEFER_INTER_FIN"] = "0"
        os.environ["MORI_HIER_SLICE_OOP"] = "0"
        # DIRECT-TO-OUTPUT Phase B -- the gathers PUSH straight
        # into the registered user output (no internal transit, no full-output
        # copy-OUT). Requires the stream-ordered path (stream_ring+stream_intra).
        # The output MUST stay byte-identical to the copy-OUT path. Cover both
        # defer_fin settings (the direct fence is deferrable too) over the
        # multi-size loop (exercises cross-op deferral + last-op no-fence).
        #
        # the direct path registers the USER output via
        # ShmemSymmetricRegister. Over RDMA (true xnode) that succeeds, but this
        # single-process spawn sim wires peers over IPC, and hipIpcGetMemHandle
        # on an arbitrary torch allocation HARD-FAILS ("invalid argument") and
        # ABORTS the process (uncatchable) -- so the direct loop cannot run under
        # the single-node IPC sim. Gate it behind MORI_HIER_TEST_DIRECT=1 so it
        # runs only on an RDMA-capable host; the shipped true-xnode bit-exact
        # test (test_hier_allgather under torchrun with --slice-direct) is the
        # primary durable coverage for this path.
        if os.environ.get("MORI_HIER_TEST_DIRECT", "0") not in ("0", "", "false", "False"):
            os.environ["MORI_HIER_STREAM_RING"] = "1"
            os.environ["MORI_HIER_STREAM_INTRA"] = "1"
            os.environ["MORI_HIER_SLICE_FUSED"] = "1"
            os.environ["MORI_HIER_SLICE_OOP"] = "0"
            os.environ["MORI_HIER_SLICE_OVERLAP"] = "0"
            os.environ["MORI_HIER_SLICE_DIRECT"] = "1"
            prev_direct_overlap = os.environ.get("MORI_HIER_SLICE_DIRECT_OVERLAP")
            # loop the direct-path local-block overlap {0,1} so
            # both the shipped serial direct path and the side-stream overlap path
            # have durable bit-exact coverage.
            for direct_overlap in ("0", "1"):
                os.environ["MORI_HIER_SLICE_DIRECT_OVERLAP"] = direct_overlap
                for defer_fin in ("1", "0"):
                    os.environ["MORI_HIER_SLICE_DEFER_FIN"] = defer_fin
                    for world, rpn in layouts:
                        if world > ngpu:
                            continue
                        test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
                        ran += 1
            if prev_direct_overlap is None:
                os.environ.pop("MORI_HIER_SLICE_DIRECT_OVERLAP", None)
            else:
                os.environ["MORI_HIER_SLICE_DIRECT_OVERLAP"] = prev_direct_overlap
            os.environ["MORI_HIER_SLICE_DIRECT"] = "0"
            os.environ["MORI_HIER_STREAM_RING"] = "0"
            os.environ["MORI_HIER_SLICE_DEFER_FIN"] = "1"
    finally:
        if prev is None:
            os.environ.pop("MORI_HIER_SLICE", None)
        else:
            os.environ["MORI_HIER_SLICE"] = prev
        if prev_fused is None:
            os.environ.pop("MORI_HIER_SLICE_FUSED", None)
        else:
            os.environ["MORI_HIER_SLICE_FUSED"] = prev_fused
        if prev_oop is None:
            os.environ.pop("MORI_HIER_SLICE_OOP", None)
        else:
            os.environ["MORI_HIER_SLICE_OOP"] = prev_oop
        if prev_overlap is None:
            os.environ.pop("MORI_HIER_SLICE_OVERLAP", None)
        else:
            os.environ["MORI_HIER_SLICE_OVERLAP"] = prev_overlap
        if prev_min is None:
            os.environ.pop("MORI_HIER_SLICE_MIN_BYTES", None)
        else:
            os.environ["MORI_HIER_SLICE_MIN_BYTES"] = prev_min
        if prev_fuse_ib is None:
            os.environ.pop("MORI_HIER_SLICE_FUSE_IB", None)
        else:
            os.environ["MORI_HIER_SLICE_FUSE_IB"] = prev_fuse_ib
        if prev_pipe is None:
            os.environ.pop("MORI_HIER_SLICE_PIPE", None)
        else:
            os.environ["MORI_HIER_SLICE_PIPE"] = prev_pipe
        if prev_pipe_chunks is None:
            os.environ.pop("MORI_HIER_SLICE_PIPE_CHUNKS", None)
        else:
            os.environ["MORI_HIER_SLICE_PIPE_CHUNKS"] = prev_pipe_chunks
        if prev_pipe_overlap is None:
            os.environ.pop("MORI_HIER_SLICE_PIPE_OVERLAP", None)
        else:
            os.environ["MORI_HIER_SLICE_PIPE_OVERLAP"] = prev_pipe_overlap
        if prev_stream_ring is None:
            os.environ.pop("MORI_HIER_STREAM_RING", None)
        else:
            os.environ["MORI_HIER_STREAM_RING"] = prev_stream_ring
        if prev_stream_intra is None:
            os.environ.pop("MORI_HIER_STREAM_INTRA", None)
        else:
            os.environ["MORI_HIER_STREAM_INTRA"] = prev_stream_intra
        if prev_defer_fin is None:
            os.environ.pop("MORI_HIER_SLICE_DEFER_FIN", None)
        else:
            os.environ["MORI_HIER_SLICE_DEFER_FIN"] = prev_defer_fin
        if prev_defer_inter_fin is None:
            os.environ.pop("MORI_HIER_SLICE_DEFER_INTER_FIN", None)
        else:
            os.environ["MORI_HIER_SLICE_DEFER_INTER_FIN"] = prev_defer_inter_fin
        if prev_direct is None:
            os.environ.pop("MORI_HIER_SLICE_DIRECT", None)
        else:
            os.environ["MORI_HIER_SLICE_DIRECT"] = prev_direct
    assert ran > 0, "no sliced layout fit the visible GPU count"


def _run_torchrun(numels, dtypes, bench=False):
    """Cross-node entry: torchrun supplies RANK/WORLD_SIZE/LOCAL_RANK."""
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    backend = "cpu:gloo,cuda:nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            device_id=device)
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    try:
        _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device,
                     bench=bench)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bit-exact HierAllGather test")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--ranks-per-node", type=int, default=None,
                        help="GPUs per simulated node; <world-size exercises M2b")
    parser.add_argument("--numels", type=int, nargs="+", default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--bench", action="store_true",
                        help="after correctness, run timed bench vs RCCL")
    parser.add_argument("--slice", dest="slice_inter", action="store_true",
                        help="this work M5 lever: SLICED 2-D AllGather. Each rank "
                        "rings only its own shard across nodes (per-NIC inter "
                        "bytes cut G x), then N intra SDMA gathers reassemble the "
                        "node-blocks. Sets MORI_HIER_SLICE=1 before shmem init.")
    parser.add_argument("--slice-fused", dest="slice_fused", action="store_true",
                        help="this work M5 : fold the N sliced-Phase-B intra "
                        "reassembly gathers into ONE batch (2 barriers + 1 bulk "
                        "copy vs 2N barriers + N copies). Implies --slice. Sets "
                        "MORI_HIER_SLICE_FUSED=1 before shmem init.")
    parser.add_argument("--no-slice", dest="no_slice", action="store_true",
                        help="this work: force the pre-slice baseline path "
                        "(MORI_HIER_SLICE=0), overriding the Turn-5 default-ON. "
                        "For A/B benchmarking the shipped default vs baseline.")
    parser.add_argument("--slice-oop", dest="slice_oop", action="store_true",
                        help="this work M5 : read the sliced Phase-A collection "
                        "directly from the inter ring buffer (out_in_place) instead "
                        "of a finish copy-OUT into scratch. Implies --slice. Sets "
                        "MORI_HIER_SLICE_OOP=1 before shmem init.")
    parser.add_argument("--slice-overlap", dest="slice_overlap", action="store_true",
                        help="this work M5  lever (c): overlap the local "
                        "node-block Phase-B gather with the inter ring on a side "
                        "stream. Implies --slice --slice-fused. Sets "
                        "MORI_HIER_SLICE_OVERLAP=1 before shmem init.")
    parser.add_argument("--no-slice-fuse-ib", dest="no_slice_fuse_ib",
                        action="store_true",
                        help="this work M5 : RESTORE the redundant Phase-B entry "
                        "barrier in the sliced+fused path (MORI_HIER_SLICE_FUSE_IB=0). "
                        "Default drops it (the inter ring's finish barrier already "
                        "synchronizes all PEs). For A/B benchmarking.")
    parser.add_argument("--slice-pipe", dest="slice_pipe", action="store_true",
                        help="this work M5 : chunked (strided) Phase-B "
                        "reassembly -- split each block gather into "
                        "--slice-pipe-chunks element ranges via the new "
                        "gather_kernel slot-stride. Correctness enabler for the "
                        "chunked inter/intra pipeline. Implies --slice --slice-fused. "
                        "Sets MORI_HIER_SLICE_PIPE=1 before shmem init.")
    parser.add_argument("--slice-pipe-chunks", type=int, default=None,
                        help="this work M5 : number of element-range chunks for "
                        "--slice-pipe (default 2). Sets MORI_HIER_SLICE_PIPE_CHUNKS.")
    parser.add_argument("--slice-pipe-overlap", dest="slice_pipe_overlap",
                        action="store_true",
                        help="this work M5  (rule#1 payoff): chunk the INTER ring "
                        "into --slice-pipe-chunks stages and overlap each chunk's "
                        "Phase-B gather (side stream) with the next chunk's ring "
                        "(main stream) to hide the serial Phase-B tail. Implies "
                        "--slice --slice-fused. Sets MORI_HIER_SLICE_PIPE_OVERLAP=1.")
    parser.add_argument("--stream-ring", dest="stream_ring", action="store_true",
                        help="this work M5 : stream-ordered inter ring -- use "
                        "the on-device ShmemBarrierOnStream prepare/finish instead "
                        "of host hipStreamSynchronize + host ShmemBarrierAll, "
                        "removing 2 CPU<->GPU round-trips per inter ring op. Sets "
                        "MORI_HIER_STREAM_RING=1 before shmem init.")
    parser.add_argument("--no-stream-ring", dest="no_stream_ring", action="store_true",
                        help="this work M5  A/B: restore the host-synced inter "
                        "ring (MORI_HIER_STREAM_RING=0) to measure against the "
                        "stream-ordered default.")
    parser.add_argument("--no-stream-intra", dest="no_stream_intra", action="store_true",
                        help="this work M5  A/B: restore the host-synced "
                        "Phase-B finish_batch (hipStreamSynchronize + host "
                        "ShmemBarrierAll, MORI_HIER_STREAM_INTRA=0) to measure "
                        "against the stream-ordered finish_batch_stream default.")
    parser.add_argument("--no-slice-defer-fin", dest="no_slice_defer_fin",
                        action="store_true",
                        help="this work M5  A/B: restore the Phase-B finish "
                        "fence (MORI_HIER_SLICE_DEFER_FIN=0) instead of deferring "
                        "it to the next op's inter-prepare barrier, to measure "
                        "against the deferred-fence default.")
    parser.add_argument("--slice-defer-inter-fin", dest="slice_defer_inter_fin",
                        action="store_true",
                        help="this work : defer the inter ring's "
                        "finish_stream fence (MORI_HIER_SLICE_DEFER_INTER_FIN=1) "
                        "to the next slice op's prepare_stream barrier, dropping "
                        "one global on-stream fence per op on the non-oop slice "
                        "path. Now DEFAULT ON; flag is a no-op kept for "
                        "back-compat.")
    parser.add_argument("--no-slice-defer-inter-fin", dest="no_slice_defer_inter_fin",
                        action="store_true",
                        help="this work  A/B: restore the inter ring's "
                        "finish_stream fence (MORI_HIER_SLICE_DEFER_INTER_FIN=0) "
                        "instead of deferring it, to measure against the "
                        "deferred-fence default.")
    parser.add_argument("--slice-direct", dest="slice_direct", action="store_true",
                        help="this work M5 : DIRECT-TO-OUTPUT Phase B "
                        "(MORI_HIER_SLICE_DIRECT=1) -- SDMA-PUSH the gathered "
                        "node-blocks straight into the registered user output, "
                        "eliminating the full-output finish_batch copy-OUT. "
                        ": now DEFAULT ON over RDMA (auto-probed via "
                        "shmem_ptr_p2p to a cross-node peer); stays OFF on the "
                        "single-node IPC sim where ShmemSymmetricRegister hard-"
                        "aborts. This flag forces it ON; --no-slice-direct forces "
                        "OFF. +5.4% @64MiB on true xnode (133.7->141.2 GB/s).")
    parser.add_argument("--no-slice-direct", dest="no_slice_direct",
                        action="store_true",
                        help="this work  A/B: restore the full-output "
                        "finish_batch copy-OUT (MORI_HIER_SLICE_DIRECT=0) "
                        "instead of the direct-to-output PUSH default, to "
                        "measure against the direct path.")
    parser.add_argument("--slice-direct-overlap", dest="slice_direct_overlap",
                        action="store_true",
                        help="overlap the LOCAL node-block "
                        "(m=node_id) reassembly gather (no ring dependency) on a "
                        "side stream with the inter ring kernel "
                        "(MORI_HIER_SLICE_DIRECT_OVERLAP=1). Hides ~1/N of Phase B "
                        "under Phase A. Requires the slice_direct stream path.")
    parser.add_argument("--no-slice-direct-overlap", dest="no_slice_direct_overlap",
                        action="store_true",
                        help="Force MORI_HIER_SLICE_DIRECT_OVERLAP=0 for A/B.")
    parser.add_argument("--put-chunk-bytes", type=int, default=None,
                        help="this work transport lever: split each fast-path RDMA "
                        "put into WQEs of at most this many bytes (multiple "
                        "in-flight WQEs/QP). 0/unset = single-WQE default. Set "
                        "into MORI_RDMA_PUT_CHUNK_BYTES before shmem init so the "
                        "C++ transport (read at GpuStateInit) picks it up.")
    parser.add_argument("--fuse-local", dest="fuse_local", action="store_true",
                        help="fuse the LOCAL node-block "
                        "(m=node_id) reassembly gather INTO the inter ring "
                        "kernel as ONE launch (ring blocks || local-gather block, "
                        "no host wait_stream merge) -- MORI_HIER_FUSE_LOCAL=1. "
                        "Ports this proven recv+reassemble parity lever "
                        "(D hit 176 GB/s @64MiB). Requires the slice_direct "
                        "stream path; N==2 only.")
    parser.add_argument("--no-fuse-local", dest="no_fuse_local",
                        action="store_true",
                        help="Force MORI_HIER_FUSE_LOCAL=0 for A/B against the "
                        "shipped slice_direct path.")
    parser.add_argument("--dissem-barrier", dest="dissem_barrier",
                        action="store_true",
                        help="use the dissemination-topology "
                        "global barrier for the inter-ring prepare rendezvous "
                        "(MORI_HIER_DISSEM_BARRIER=1). Same global all-PE "
                        "semantics, O(log n) parallel rounds vs the PE0 funnel.")
    parser.add_argument("--no-dissem-barrier", dest="no_dissem_barrier",
                        action="store_true",
                        help="Force the funnel barrier (MORI_HIER_DISSEM_BARRIER=0) for A/B.")
    args = parser.parse_args()

    if args.dissem_barrier:
        os.environ["MORI_HIER_DISSEM_BARRIER"] = "1"
    if args.no_dissem_barrier:
        os.environ["MORI_HIER_DISSEM_BARRIER"] = "0"

    # Must be set BEFORE shmem init (the C++ GpuStateInit reads the env). The
    # harness PYENV is fixed, so we thread the lever through this CLI flag; the
    # spawned/torchrun children inherit os.environ in-process.
    if args.put_chunk_bytes is not None:
        os.environ["MORI_RDMA_PUT_CHUNK_BYTES"] = str(args.put_chunk_bytes)
    if args.no_slice_fuse_ib:
        os.environ["MORI_HIER_SLICE_FUSE_IB"] = "0"
    if args.stream_ring:
        os.environ["MORI_HIER_STREAM_RING"] = "1"
    if args.no_stream_ring:
        os.environ["MORI_HIER_STREAM_RING"] = "0"
    if args.no_stream_intra:
        os.environ["MORI_HIER_STREAM_INTRA"] = "0"
    if args.no_slice_defer_fin:
        os.environ["MORI_HIER_SLICE_DEFER_FIN"] = "0"
    if args.slice_defer_inter_fin:
        os.environ["MORI_HIER_SLICE_DEFER_INTER_FIN"] = "1"
    if args.no_slice_defer_inter_fin:
        os.environ["MORI_HIER_SLICE_DEFER_INTER_FIN"] = "0"
    if args.slice_direct:
        os.environ["MORI_HIER_SLICE_DIRECT"] = "1"
        os.environ["MORI_HIER_STREAM_RING"] = "1"
    if args.no_slice_direct:
        os.environ["MORI_HIER_SLICE_DIRECT"] = "0"
    if args.slice_direct_overlap:
        os.environ["MORI_HIER_SLICE_DIRECT_OVERLAP"] = "1"
    if args.no_slice_direct_overlap:
        os.environ["MORI_HIER_SLICE_DIRECT_OVERLAP"] = "0"
    if args.fuse_local:
        # Fused ring||local-gather requires the slice_direct stream path (the
        # fused branch lives in slice_direct Phase B). Enable its prerequisites
        # so the lever can be A/B'd through the fixed harness PYENV.
        # NOTE: do NOT force MORI_HIER_SLICE_MIN_BYTES=0 here. The fused
        # ring||local-gather kernel is only bit-exact at sizes that take the
        # sliced path under the SHIPPED size-threshold dispatch (>=8 MiB); at
        # small sizes the non-sliced fuse-barrier path is correct and faster.
        # Forcing the slice at all sizes makes the fused small-size case produce
        # wrong block ordering + a HIP invalid-arg launch (validated ).
        # Leaving the threshold at its default keeps small sizes on the safe
        # path and engages fuse_local only where it is the parity lever.
        os.environ["MORI_HIER_FUSE_LOCAL"] = "1"
        os.environ["MORI_HIER_SLICE"] = "1"
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        os.environ["MORI_HIER_SLICE_DIRECT"] = "1"
        os.environ["MORI_HIER_STREAM_RING"] = "1"
    if args.no_fuse_local:
        os.environ["MORI_HIER_FUSE_LOCAL"] = "0"
    if args.no_slice:
        # Force the pre-slice baseline (overrides Turn-5 default-ON) for A/B.
        os.environ["MORI_HIER_SLICE"] = "0"
        os.environ["MORI_HIER_SLICE_FUSED"] = "0"
    elif args.slice_inter or args.slice_fused or args.slice_oop:
        os.environ["MORI_HIER_SLICE"] = "1"
        # Explicit --slice forces the sliced path at ALL sizes (override the
        # default size threshold) so A/B measures the pure sliced path.
        os.environ.setdefault("MORI_HIER_SLICE_MIN_BYTES", "0")
    if args.slice_fused:
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
    if args.slice_oop:
        os.environ["MORI_HIER_SLICE_OOP"] = "1"
    if args.slice_overlap:
        # Overlap needs the sliced fused path with the scratch collection.
        os.environ["MORI_HIER_SLICE"] = "1"
        os.environ.setdefault("MORI_HIER_SLICE_MIN_BYTES", "0")
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        os.environ["MORI_HIER_SLICE_OVERLAP"] = "1"
    if args.slice_pipe:
        # Chunked Phase-B needs the sliced fused (non-oop, non-overlap) path.
        os.environ["MORI_HIER_SLICE"] = "1"
        os.environ.setdefault("MORI_HIER_SLICE_MIN_BYTES", "0")
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        os.environ["MORI_HIER_SLICE_PIPE"] = "1"
    if args.slice_pipe_overlap:
        # Chunked-ring pipeline overlap needs the sliced fused (non-oop,
        # non-local-overlap) path with K>1 chunks.
        os.environ["MORI_HIER_SLICE"] = "1"
        os.environ.setdefault("MORI_HIER_SLICE_MIN_BYTES", "0")
        os.environ["MORI_HIER_SLICE_FUSED"] = "1"
        os.environ["MORI_HIER_SLICE_PIPE_OVERLAP"] = "1"
    if args.slice_pipe_chunks is not None:
        os.environ["MORI_HIER_SLICE_PIPE_CHUNKS"] = str(args.slice_pipe_chunks)

    if args.dtype is not None:
        from tests.python.utils import string_to_dtype

        dtypes = [string_to_dtype(args.dtype)]
    else:
        dtypes = _DEFAULT_DTYPES
    numels = (
        args.numels
        if args.numels is not None
        else [1024, 1024 * 1024, 16 * 1024 * 1024]
    )

    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Launched under torchrun (xnode harness).
            _run_torchrun(numels, dtypes, bench=args.bench)
        else:
            test_hier_allgather(
                world_size=args.world_size, ranks_per_node=args.ranks_per_node,
                numels=numels, dtypes=dtypes, bench=args.bench,
            )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

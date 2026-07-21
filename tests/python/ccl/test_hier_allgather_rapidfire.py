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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Rapid-fire (inter-call) determinism probe for cross-node HierAllGather.

Reproduces FSDP's multi-layer regime: many AllGather calls per step issued
back-to-back on one reused handle, into distinct per-layer output buffers, with
no consume or sync between them (the backward GEMMs consume them later). Call
k+1's ring/intra reuses the handle's shared transit (``_slice_scratch`` / the
intra out_ / the inter ``collection``) while call k's output has not yet been
drained -- aliasing a single-call test cannot trigger.

Fires N AllGathers back-to-back into N distinct output buffers on one handle and
one stream, no consume/sync between calls, then consumes all N and compares each
to an independent all_gather-built reference. Mixed sizes per call (big
embed-band + small layers, interleaved) mimic FSDP walking layers of different
shapes through the same handle.

Pass criteria (per rep-of-the-whole-burst):
  * Determinism: each output's consumer scalar bitwise-matches a per-call
    host-synced golden (a stale/aliased transit gives run-to-run drift).
  * Correctness: each equals the all_gather reference (a stable-but-wrong
    aliased drain shows as nondet=0 but wrong_vs_truth>0).

Run cross-node under torchrun::

    torchrun --nnodes=2 --nproc_per_node=4 ... \
        tests/python/ccl/test_hier_allgather_rapidfire.py
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import HierAllGather

# Cross-node torchrun-only harness: SKIP under a plain `pytest` collection (no
# torchrun env) so the suite does not ERROR; runs only when torchrun sets
# RANK/WORLD_SIZE.
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ or "WORLD_SIZE" not in os.environ,
    reason="cross-node harness; launch under torchrun",
)

_DTYPES = [torch.bfloat16, torch.float32]
# Burst layout: a sequence of per-rank element counts fired back-to-back on one
# handle, no consume between. Interleave the big embed/lm_head band (~34M/rank)
# with small layers, like FSDP walking Qwen's layers. Distinct outputs so nothing
# is overwritten by a later call's output -- only the handle's shared internal
# transit can alias.
_BURST = [34078720, 65536, 34078720, 131072, 34078720, 32768]
_REPS = int(os.environ.get("RAPIDFIRE_REPS", "8"))
_MAX_COUNT = max(_BURST)


def _make_input(dtype, count, rank, device, salt):
    base = (rank + 1) * 17 + salt
    ramp = torch.arange(count, dtype=torch.int32, device=device) % 64
    return (ramp + base).to(dtype=dtype).contiguous()


def _consume(out, wvec):
    n = out.numel()
    return (out.to(torch.float32) * wvec[:n]).sum()


def _reference_out(inp, world_size, device):
    """Independent truth: RCCL all_gather -> rank-major (copy-OUT __call__ layout)."""
    count = inp.numel()
    rank_major = torch.empty(count * world_size, dtype=inp.dtype, device=device)
    dist.all_gather_into_tensor(rank_major, inp)
    return rank_major


def _run_dtype(handle, dtype, rank, world_size, device):
    # wvec big enough for the largest single output in the burst.
    wvec = ((torch.arange(_MAX_COUNT * world_size, device=device) % 97) + 1).to(
        torch.float32
    )

    # GOLDEN: run each burst call with a per-call host sync (fresh, un-aliased
    # bytes guaranteed) + the independent all_gather reference. One golden set
    # over all reps (inputs are deterministic in rep+idx).
    def inp_for(rep, idx):
        return _make_input(dtype, _BURST[idx], rank, device, salt=rep * 13 + idx)

    golden = []  # [rep][idx]
    truth = []  # [rep][idx]
    for rep in range(_REPS):
        g_row, t_row = [], []
        for idx, count in enumerate(_BURST):
            inp = inp_for(rep, idx)
            ref = _reference_out(inp, world_size, device)
            t_row.append(_consume(ref, wvec).item())
            out = torch.empty(count * world_size, dtype=dtype, device=device)
            assert handle(inp, out, count, torch.cuda.current_stream())
            torch.cuda.synchronize()
            g_row.append(_consume(out, wvec).item())
        golden.append(g_row)
        truth.append(t_row)
    torch.cuda.synchronize()
    dist.barrier()

    # Optional concurrent-compute contention (RAPIDFIRE_CONTEND=1): launch heavy
    # GEMMs on a side stream that run concurrently with the AG burst, mimicking
    # FSDP overlapping every AG with backward GEMMs. Exercises copy-engine (SDMA)
    # vs CU contention -- intra SDMA gather completion under GPU load
    # (flag-beats-data only shows under contention).
    contend = os.environ.get("RAPIDFIRE_CONTEND", "0") == "1"
    load_stream = torch.cuda.Stream() if contend else None
    load_a = (
        torch.randn(4096, 4096, dtype=torch.float32, device=device) if contend else None
    )

    # Input-lifetime probe (RAPIDFIRE_FREE_INPUT=1): FSDP2's
    # reshard_after_forward frees the sharded-param input storage right after
    # issuing the AllGather; the caching allocator then reallocates that block for
    # a later op and writes it (free + realloc-of-the-same-block, not an in-place
    # write to the live tensor). record_stream on the AG stream tells the
    # allocator to defer that reuse until the AG (which may run on a different
    # stream) finishes reading. Drop the input ref (free) and immediately allocate
    # a same-nbytes decoy and fill it (poison) to coax the allocator into reusing
    # the freed block. Without the AG-stream lifetime guard + a cross-stream AG
    # (XSTREAM), the decoy write races the AG read; the guard fixes it. Decoys are
    # kept alive so the reuse pressure persists across the burst.
    free_input = os.environ.get("RAPIDFIRE_FREE_INPUT", "0") == "1"
    decoys: list = []

    # Cross-stream handoff probe (RAPIDFIRE_XSTREAM=1): mirror FSDP2. FSDP issues
    # each AG on a dedicated comm stream, records an event on that comm stream, and
    # the compute stream waits on the event before the backward GEMM consumes the
    # output. If mori launches any AG work not captured by an event recorded on
    # the comm stream (e.g. an internal side-queue the caller stream doesn't join
    # before __call__ returns), the recorded event fires early, the compute stream
    # proceeds, and the consumer reads stale bytes.
    xstream = os.environ.get("RAPIDFIRE_XSTREAM", "0") == "1"
    comm_stream = torch.cuda.Stream() if xstream else None

    # Rapid-fire: fire the whole burst back-to-back into distinct outputs on one
    # stream, no consume/sync between calls -> consecutive calls reuse the
    # handle's shared transit while prior outputs are undrained. Then consume.
    n_nondet = 0
    n_wrong = 0
    first_bad = (-1, -1)
    main = torch.cuda.current_stream()
    for rep in range(_REPS):
        inps = [inp_for(rep, idx) for idx in range(len(_BURST))]
        outs = [
            torch.empty(_BURST[idx] * world_size, dtype=dtype, device=device)
            for idx in range(len(_BURST))
        ]
        if contend:
            # queue a stream of GEMMs that run alongside the AG burst
            with torch.cuda.stream(load_stream):
                for _ in range(24):
                    load_a = load_a @ load_a * 1e-3 + 0.1
        # back-to-back, no sync
        if xstream:
            # FSDP-like: AG on comm stream, per-call event, compute stream waits.
            comm_stream.wait_stream(main)
            evts = []
            for idx, count in enumerate(_BURST):
                with torch.cuda.stream(comm_stream):
                    assert handle(inps[idx], outs[idx], count, comm_stream)
                    e = torch.cuda.Event()
                    e.record(comm_stream)
                evts.append(e)
                if free_input:
                    # free the input, then realloc same-nbytes + poison on the
                    # compute (main) stream to force the allocator to reuse the
                    # freed block while the comm-stream AG may still read it.
                    inps[idx] = None
                    d = torch.empty(count, dtype=dtype, device=device)
                    d.fill_(float(-(rep * 13 + idx) - 999))
                    decoys.append(d)
            # consumer on the compute (main) stream: wait the AG's event, consume
            vals = []
            for idx in range(len(_BURST)):
                main.wait_event(evts[idx])
            with torch.cuda.stream(main):
                vals = [_consume(outs[idx], wvec) for idx in range(len(_BURST))]
            torch.cuda.synchronize()
        else:
            for idx, count in enumerate(_BURST):
                assert handle(inps[idx], outs[idx], count, main)
                if free_input:
                    # mimic reshard: free the input then realloc+poison the block
                    # on the caller stream (same stream -> ordered -> must pass).
                    inps[idx] = None
                    d = torch.empty(count, dtype=dtype, device=device)
                    d.fill_(float(-(rep * 13 + idx) - 999))
                    decoys.append(d)
            # only now drain + consume
            vals = [_consume(outs[idx], wvec) for idx in range(len(_BURST))]
            torch.cuda.synchronize()
        for idx in range(len(_BURST)):
            v = vals[idx].item()
            g = golden[rep][idx]
            t = truth[rep][idx]
            nd = (v != g) and not (v != v and g != g)
            wr = (
                (abs(v - t) > 1e-4 * max(abs(t), 1.0))
                if (v == v and t == t)
                else ((v != v) != (t != t))
            )
            if nd:
                n_nondet += 1
            if wr:
                n_wrong += 1
            if (nd or wr) and first_bad == (-1, -1):
                first_bad = (rep, idx)
    torch.cuda.synchronize()

    if rank == 0:
        print(
            f"  [rapidfire dtype={dtype} burst={_BURST} reps={_REPS}] "
            f"nondet={n_nondet}/{_REPS * len(_BURST)} "
            f"wrong_vs_truth={n_wrong}/{_REPS * len(_BURST)} first_bad={first_bad}",
            flush=True,
        )
    return n_nondet, n_wrong


def _make_handle(rank, world_size, ranks_per_node):
    per_rank_bytes = _MAX_COUNT * 4 + 4096
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
            f"rapidfire: world={world_size} rpn={ranks_per_node} "
            f"num_nodes={handle.num_nodes} reps={_REPS} burst={_BURST}",
            flush=True,
        )
    try:
        total_nd, total_wr = 0, 0
        for dtype in _DTYPES:
            nd, wr = _run_dtype(handle, dtype, rank, world_size, device)
            total_nd += nd
            total_wr += wr
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            if total_nd == 0 and total_wr == 0:
                print(
                    "test_hier_allgather_rapidfire: PASSED (deterministic+correct)",
                    flush=True,
                )
            else:
                print(
                    f"test_hier_allgather_rapidfire: FAILED "
                    f"nondet={total_nd} wrong={total_wr}",
                    flush=True,
                )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
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
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
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


def test_hier_allgather_rapidfire():
    """Pytest entry: runs only under torchrun (guarded by module pytestmark)."""
    _run_torchrun()


if __name__ == "__main__":
    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            _run_torchrun()
        else:
            raise SystemExit("launch under torchrun (cross-node)")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

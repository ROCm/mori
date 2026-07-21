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
Bit-exact test for ``mori.ccl.IntraNodeSubGroupAllgatherSdma`` (the *intra-node*
phase of the hierarchical cross-node AllGather).

The hierarchical AllGather first gathers, within each node, the ``G`` local
shards over the SDMA copy engines (XGMI). This is a sub-group AllGather: node
``n`` owns the contiguous global PEs ``{n*G, ..., n*G+G-1}``; every rank in the
node ends up holding ``concat(input[n*G], ..., input[n*G+G-1])`` -- its node's
contiguous G-shard block. Every node runs its own gather concurrently
(pe_base=n*G, pe_stride=1, group_size=G, group_pos=local_rank).

Single-node this validates the SDMA gather building block directly on device.
AllGather is a pure data move => zero tolerance (``torch.equal``).

  python3 tests/python/ccl/test_intra_subgroup_sdma.py --world-size 4 --ranks-per-node 2
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import IntraNodeSubGroupAllgatherSdma

from tests.python.utils import TorchDistContext, get_free_port

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >=2 GPUs"
)


_DEFAULT_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]


def _make_input(dtype: torch.dtype, numel: int, rank: int, device) -> torch.Tensor:
    base = (rank + 1) * 17
    ramp = torch.arange(numel, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _run_one(dtype, numel, rank, world_size, G, device):
    node = rank // G
    local = rank % G
    pe_base = node * G

    handle = IntraNodeSubGroupAllgatherSdma(
        my_pe=rank,
        npes=world_size,
        out_buffer_bytes=numel * torch.tensor([], dtype=dtype).element_size() * G
        + 4096,
        group_size=G,
        group_pos=local,
        pe_base=pe_base,
        pe_stride=1,
    )

    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * G, dtype=dtype, device=device)

    # Reference: this PE's node block is concat of the G local shards in
    # local-rank order. Inputs are deterministic in the global rank, so each
    # member's shard is reproducible locally -- no collective needed.
    out_ref = torch.empty(numel * G, dtype=dtype, device=device)
    for k in range(G):
        member_rank = pe_base + k
        out_ref[k * numel : (k + 1) * numel] = _make_input(
            dtype, numel, member_rank, device
        )

    stream = torch.cuda.current_stream()
    ok = handle(inp, out_mori, numel, stream)
    assert ok, f"sub-group SDMA gather call failed dtype={dtype} numel={numel}"
    stream.synchronize()
    torch.cuda.synchronize()
    del handle

    if not torch.equal(out_mori, out_ref):
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"sub-group SDMA gather mismatch dtype={dtype} numel={numel} rank={rank} "
            f"(local={local},node={node},G={G}): first mismatch positions={diff} "
            f"got={out_mori[diff].tolist()} ref={out_ref[diff].tolist()}"
        )


def _bench_one(dtype, numel, rank, world_size, G, device, reps=5, warmup=2):
    """Time the intra-node sub-group SDMA gather. Reports the per-rank gathered
    throughput (node-block bytes / time) on rank 0. Exercises whether the
    multi-channel (MORI_SDMA_NUM_CHANNELS>1) SdmaPutWarp path parallelizes the
    XGMI copy across SDMA engines vs the old single-queue path."""
    elem = torch.tensor([], dtype=dtype).element_size()
    handle = IntraNodeSubGroupAllgatherSdma(
        my_pe=rank,
        npes=world_size,
        out_buffer_bytes=numel * elem * G + 4096,
        group_size=G,
        group_pos=rank % G,
        pe_base=(rank // G) * G,
        pe_stride=1,
    )
    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * G, dtype=dtype, device=device)
    stream = torch.cuda.current_stream()

    for _ in range(warmup):
        handle(inp, out_mori, numel, stream)
    stream.synchronize()
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        handle(inp, out_mori, numel, stream)
        end.record(stream)
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)  # seconds
    del handle

    gathered_gb = numel * G * elem / 1e9
    tmin, tavg = min(times), sum(times) / len(times)
    if rank == 0:
        print(
            f"[bench] world={world_size} G={G} dtype={dtype} numel={numel} "
            f"block={gathered_gb:.3f}GB | min={tmin*1e3:.3f}ms avg={tavg*1e3:.3f}ms "
            f"BW={gathered_gb/tmin:.1f}GB/s (reps={reps})"
        )


def _worker_body(rank, world_size, G, numels, dtypes, device, bench=False):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    assert shmem.shmem_npes() == world_size

    if rank == 0:
        print(f"IntraSubGroupSDMA: world={world_size} G={G} N={world_size // G}")
    try:
        for dtype in dtypes:
            for numel in numels:
                if (numel * torch.tensor([], dtype=dtype).element_size()) % 4 != 0:
                    continue
                _run_one(dtype, numel, rank, world_size, G, device)
                if rank == 0:
                    print(f"  ok dtype={dtype} numel={numel}")
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("test_intra_subgroup_sdma: PASSED")
        if bench:
            for _mb in (32, 64, 128, 256):
                _bench_one(
                    torch.float32, _mb * 1024 * 1024 // 4, rank, world_size, G, device
                )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def _spawn_worker(rank, world_size, G, port, numels, dtypes, bench):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        _worker_body(rank, world_size, G, numels, dtypes, device, bench=bench)


def test_intra_subgroup_sdma(
    world_size=None, ranks_per_node=2, numels=None, dtypes=None, bench=False
):
    """Single-node pytest entry. MORI_SDMA_NUM_CHANNELS=1 sidesteps the SDMA
    multi-queue source/dest offset bug for same-node puts (see test_allgather /
    test_inter_node_ring)."""
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size >= 2, f"need >=2 GPUs, got {world_size}"
    assert (
        world_size % ranks_per_node == 0
    ), "world must be a multiple of ranks_per_node"
    if numels is None:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bit-exact sub-group intra SDMA gather test"
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--ranks-per-node", type=int, default=2)
    parser.add_argument("--numels", type=int, nargs="+", default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="after correctness, time the gather (per-rank block BW)",
    )
    args = parser.parse_args()

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
        test_intra_subgroup_sdma(
            world_size=args.world_size,
            ranks_per_node=args.ranks_per_node,
            numels=numels,
            dtypes=dtypes,
            bench=args.bench,
        )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

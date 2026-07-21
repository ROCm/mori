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
Bit-exact test for the *sub-group* form of ``mori.ccl.InterNodeRingAllgather``
.

The hierarchical cross-node AllGather runs the inter-node RDMA ring over an
arithmetic sub-group of PEs rather than the whole world: with ``G`` ranks/node
and ``N`` nodes, the ranks sharing local index ``g`` form a ring
``{g, g+G, ..., g+(N-1)*G}`` (one member per node). Every rank participates in
exactly one such sub-group, so all ``G`` sub-group rings run concurrently. After
the ring, rank ``(node, g)`` holds the ``N`` chunks of its sub-group in node
order -- i.e. ``concat(input[g], input[g+G], ..., input[g+(N-1)*G])``.

Single-node this exercises the *same* kernel code path that runs over RDMA
across nodes (sub-group neighbours are reached via shmem put; cross-node they go
over RDMA). AllGather is a pure data move => zero tolerance (``torch.equal``).

  python3 tests/python/ccl/test_inter_node_ring_subgroup.py --world-size 4 --ranks-per-node 2
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import InterNodeRingAllgather

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
    N = world_size // G
    g = rank % G
    node = rank // G

    handle = InterNodeRingAllgather(
        my_pe=rank,
        npes=world_size,
        ring_buffer_bytes=numel * torch.tensor([], dtype=dtype).element_size() * N
        + 4096,
        ring_size=N,
        ring_pos=node,
        pe_base=g,
        pe_stride=G,
    )

    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * N, dtype=dtype, device=device)

    # Reference: this PE's sub-group is {g, g+G, ..., g+(N-1)*G} in ring (node)
    # order. Inputs are deterministic in the global rank, so we can rebuild any
    # member's chunk locally -- no collective needed.
    out_ref = torch.empty(numel * N, dtype=dtype, device=device)
    for k in range(N):
        member_rank = g + k * G
        out_ref[k * numel : (k + 1) * numel] = _make_input(
            dtype, numel, member_rank, device
        )

    stream = torch.cuda.current_stream()
    ok = handle(inp, out_mori, numel, stream)
    assert ok, f"sub-group ring call failed dtype={dtype} numel={numel}"
    stream.synchronize()
    torch.cuda.synchronize()
    del handle

    if not torch.equal(out_mori, out_ref):
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"sub-group ring mismatch dtype={dtype} numel={numel} rank={rank} "
            f"(g={g},node={node},N={N}): first mismatch positions={diff} "
            f"got={out_mori[diff].tolist()} ref={out_ref[diff].tolist()}"
        )


def _worker_body(rank, world_size, G, numels, dtypes, device):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    assert shmem.shmem_npes() == world_size

    if rank == 0:
        print(f"InterNodeRing sub-group: world={world_size} G={G} N={world_size // G}")
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
            print("test_inter_node_ring_subgroup: PASSED")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def _spawn_worker(rank, world_size, G, port, numels, dtypes):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        _worker_body(rank, world_size, G, numels, dtypes, device)


def test_inter_node_ring_subgroup(
    world_size=None, ranks_per_node=2, numels=None, dtypes=None
):
    """Single-node pytest entry. See ``test_inter_node_ring`` for the
    MORI_SDMA_NUM_CHANNELS=1 single-node note (same-node puts route through the
    SDMA multi-queue path whose offset bug we sidestep)."""
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
        args=(world_size, ranks_per_node, port, numels, dtypes),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bit-exact sub-group InterNodeRing test"
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--ranks-per-node", type=int, default=2)
    parser.add_argument("--numels", type=int, nargs="+", default=None)
    parser.add_argument("--dtype", type=str, default=None)
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
        test_inter_node_ring_subgroup(
            world_size=args.world_size,
            ranks_per_node=args.ranks_per_node,
            numels=numels,
            dtypes=dtypes,
        )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

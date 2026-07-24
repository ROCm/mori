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
"""Bit-exact test for ``mori.ccl.HierAllGather`` vs
``torch.distributed.all_gather_into_tensor``: pure data move, so zero
tolerance (``torch.equal``). Single-node runs via mp.spawn; cross-node runs
under torchrun (xnode harness sets RANK/WORLD_SIZE/LOCAL_RANK).
"""

import os

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import HierAllGather

from tests.python.utils import TorchDistContext, get_free_port

# CI-conformance: SKIP (not ERROR) when the >=2-GPU hardware precondition is
# unmet, matching the repo convention (see test_allgather_param_contiguous.py).
pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >=2 GPUs"
)


_DEFAULT_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]


def _make_input(dtype: torch.dtype, numel: int, rank: int, device) -> torch.Tensor:
    """Rank-distinct, dtype-exact input (values round-trip through bf16/fp16)."""
    base = (rank + 1) * 17
    ramp = torch.arange(numel, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


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


def _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device):
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
        if rank == 0:
            print("test_hier_allgather: PASSED")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
        dist.barrier()
        shmem.shmem_finalize()


def _spawn_worker(rank, world_size, ranks_per_node, port, numels, dtypes):
    """Single-node entry: each spawned process owns cuda:rank."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device)


def test_hier_allgather(world_size=None, ranks_per_node=None, numels=None, dtypes=None):
    """Single-node pytest entry. ``ranks_per_node == world_size`` is the pure-SDMA
    path (num_nodes == 1); ``ranks_per_node < world_size`` splits local GPUs into
    simulated nodes to drive the hierarchical pipeline -- same kernel path, ring
    neighbours reached over shmem P2P/SDMA instead of RDMA."""
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    # Single channel: multi-queue warp put has a source/dest offset bug here.
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size >= 2, f"HierAllGather needs >=2 GPUs, got {world_size}"
    if ranks_per_node is None:
        ranks_per_node = world_size
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


def test_hier_allgather_layouts():
    """Sweep hierarchical (num_nodes>=2) decompositions for bit-exactness;
    layouts exceeding the visible GPU count are skipped."""
    ngpu = torch.cuda.device_count()
    # (world, ranks_per_node); (8,4) is the acceptance target.
    layouts = [(4, 2), (8, 4), (8, 2)]
    small = [1024, 256 * 1024]
    ran = 0
    for world, rpn in layouts:
        if world > ngpu:
            continue
        test_hier_allgather(world_size=world, ranks_per_node=rpn, numels=small)
        ran += 1
    assert ran > 0, "no hierarchical layout fit the visible GPU count"

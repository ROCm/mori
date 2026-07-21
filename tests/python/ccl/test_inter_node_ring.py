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
Bit-exact test for ``mori.ccl.InterNodeRingAllgather`` (inter-node RDMA ring
building block) vs ``torch.distributed.all_gather_into_tensor``.

The ring kernel moves data over the shmem transport -- P2P within a node, RDMA
across nodes -- using the exact ring schedule CPU-validated by
``inter_node_ring_reference``. Running it single-node (every GPU a ring
participant) exercises the *same* kernel code path that runs over RDMA across
nodes, so this is the on-device validation of the inter-node phase.

AllGather is a pure data move => zero numerical tolerance (``torch.equal``).

  Single node::

      python3 tests/python/ccl/test_inter_node_ring.py --world-size 4

  Cross node (xnode harness sets RANK/WORLD_SIZE/LOCAL_RANK)::

      torchrun --nnodes=2 --nproc_per_node=4 ... test_inter_node_ring.py
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


def _run_one(handle, dtype, numel, rank, world_size, device):
    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * world_size, dtype=dtype, device=device)
    out_ref = torch.empty(numel * world_size, dtype=dtype, device=device)

    dist.all_gather_into_tensor(out_ref, inp)

    stream = torch.cuda.current_stream()
    ok = handle(inp, out_mori, numel, stream)
    assert ok, f"InterNodeRingAllgather call failed dtype={dtype} numel={numel}"
    stream.synchronize()
    torch.cuda.synchronize()

    if not torch.equal(out_mori, out_ref):
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"InterNodeRing mismatch dtype={dtype} numel={numel}: "
            f"first mismatch positions={diff} got={out_mori[diff].tolist()} "
            f"ref={out_ref[diff].tolist()}"
        )


def _worker_body(rank, world_size, numels, dtypes, device):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    assert shmem.shmem_npes() == world_size

    max_itemsize = max(torch.tensor([], dtype=d).element_size() for d in dtypes)
    max_numel = max(numels)
    # Ring buffer holds world_size chunks of the largest message.
    ring_bytes = max_numel * max_itemsize * world_size + 4096

    handle = InterNodeRingAllgather(
        my_pe=rank, npes=world_size, ring_buffer_bytes=ring_bytes
    )
    if rank == 0:
        print(f"InterNodeRingAllgather: world={world_size}")

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
            print("test_inter_node_ring: PASSED")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
        dist.barrier()
        shmem.shmem_finalize()


def _spawn_worker(rank, world_size, port, numels, dtypes):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        _worker_body(rank, world_size, numels, dtypes, device)


def test_inter_node_ring(world_size=None, numels=None, dtypes=None):
    """Single-node pytest entry.

    The ring uses the shmem put transport. On a *single* node every peer is
    same-node, so the put is routed through the SDMA copy engines. The SDMA
    multi-queue warp put (``core::SdmaPutWarp``) has a per-queue source/dest
    offset bug that drops all but the first queue's slice, so we pin
    ``MORI_SDMA_NUM_CHANNELS=1`` here to exercise the (correct) single-queue
    path. This is purely a single-node validation artifact: the real
    *inter-node* target routes over RDMA (no SDMA queues involved), so the
    cross-node xnode run does not need and must not be forced to this setting
    (``setdefault`` leaves any harness-provided value intact).
    """
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size >= 2, f"InterNodeRing needs >=2 GPUs, got {world_size}"
    if numels is None:
        numels = [1024, 1024 * 1024, 16 * 1024 * 1024]
    if dtypes is None:
        dtypes = _DEFAULT_DTYPES
    port = get_free_port()
    torch.multiprocessing.spawn(
        _spawn_worker,
        args=(world_size, port, numels, dtypes),
        nprocs=world_size,
        join=True,
    )


def _run_torchrun(numels, dtypes):
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    backend = "cpu:gloo,cuda:nccl"
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size, device_id=device
    )
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    try:
        _worker_body(rank, world_size, numels, dtypes, device)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bit-exact InterNodeRing test")
    parser.add_argument("--world-size", type=int, default=None)
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
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            _run_torchrun(numels, dtypes)
        else:
            test_inter_node_ring(
                world_size=args.world_size, numels=numels, dtypes=dtypes
            )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

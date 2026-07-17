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
Bit-exact test for ``mori.ccl.IntraNodeSubGroupBroadcastSdma`` (this work, M4 --
the intra-node *placement* phase of the leader-only hierarchical cross-node
AllGather).

The leader-only variant of the hierarchical AllGather (DESIGN.md's primary
suggestion) gathers the full ``N*G`` output on each node's leader (local_rank 0)
via the inter-node RDMA ring, then *broadcasts* that full buffer to the node's
``G`` local ranks over the SDMA copy engines (XGMI) -- cutting NIC traffic ~G x
vs the every-rank-direct ring. This test validates that broadcast building block
in isolation: node ``n`` owns the contiguous global PEs ``{n*G, ..., n*G+G-1}``
with root at ``n*G`` (group_pos 0); after the call every member of node ``n``
holds the root's buffer exactly. Each node runs its own broadcast concurrently
(pe_base=n*G, pe_stride=1, group_size=G, group_pos=local_rank).

AllGather/broadcast is a pure data move => ZERO tolerance (``torch.equal``).

  python3 tests/python/ccl/test_intra_subgroup_broadcast.py --world-size 4 --ranks-per-node 2
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import IntraNodeSubGroupBroadcastSdma

from tests.python.utils import TorchDistContext, get_free_port

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >=2 GPUs"
)


_DEFAULT_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]


def _make_buffer(
    dtype: torch.dtype, numel: int, root_rank: int, device
) -> torch.Tensor:
    # Deterministic in the root's global rank so every member can reproduce the
    # expected broadcast payload locally -- no collective needed for the ref.
    base = (root_rank + 1) * 23
    ramp = torch.arange(numel, dtype=torch.int32) % 97
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _run_one(dtype, numel, rank, world_size, G, device):
    node = rank // G
    local = rank % G
    pe_base = node * G
    root_rank = pe_base  # group_pos 0

    elem = torch.tensor([], dtype=dtype).element_size()
    handle = IntraNodeSubGroupBroadcastSdma(
        my_pe=rank,
        npes=world_size,
        out_buffer_bytes=numel * elem + 4096,
        group_size=G,
        group_pos=local,
        pe_base=pe_base,
        pe_stride=1,
    )

    # The root holds the full payload; non-root members start with garbage.
    if local == 0:
        inp = _make_buffer(dtype, numel, root_rank, device)
    else:
        inp = torch.full((numel,), -1, dtype=dtype, device=device)
    out_mori = torch.empty(numel, dtype=dtype, device=device)

    # Reference: every member ends with the root's payload.
    out_ref = _make_buffer(dtype, numel, root_rank, device)

    stream = torch.cuda.current_stream()
    ok = handle(inp, out_mori, numel, stream)
    assert ok, f"sub-group SDMA broadcast call failed dtype={dtype} numel={numel}"
    stream.synchronize()
    torch.cuda.synchronize()
    del handle

    if not torch.equal(out_mori, out_ref):
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"sub-group SDMA broadcast mismatch dtype={dtype} numel={numel} rank={rank} "
            f"(local={local},node={node},G={G}): first mismatch positions={diff} "
            f"got={out_mori[diff].tolist()} ref={out_ref[diff].tolist()}"
        )


def _worker_body(rank, world_size, G, numels, dtypes, device):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    assert shmem.shmem_npes() == world_size

    if rank == 0:
        print(f"IntraSubGroupBroadcast: world={world_size} G={G} N={world_size // G}")
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
            print("test_intra_subgroup_broadcast: PASSED")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def _spawn_worker(rank, world_size, G, port, numels, dtypes):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        _worker_body(rank, world_size, G, numels, dtypes, device)


def test_intra_subgroup_broadcast(
    world_size=None, ranks_per_node=2, numels=None, dtypes=None
):
    """Single-node pytest entry. MORI_SDMA_NUM_CHANNELS=1 sidesteps the SDMA
    multi-queue source/dest offset bug for same-node puts (see test_allgather /
    test_intra_subgroup_sdma)."""
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
        description="Bit-exact sub-group intra SDMA broadcast test"
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
        test_intra_subgroup_broadcast(
            world_size=args.world_size,
            ranks_per_node=args.ranks_per_node,
            numels=numels,
            dtypes=dtypes,
        )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

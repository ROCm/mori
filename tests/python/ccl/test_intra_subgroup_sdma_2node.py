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
MULTI-NODE bit-exact regression test for the intra-node sub-group SDMA gather
(``mori.ccl.IntraNodeSubGroupAllgatherSdma``) at the topology that exposed the
``MORI_HOSTPROXY_SDMA_INTRA=1`` "Slow wait for sub-group pos" hang / torn-recv.

WHY A SEPARATE 2-NODE TEST: the single-node ``test_intra_subgroup_sdma.py`` runs
every rank with global pe < 8, so ``pe`` and ``pe % 8`` coincide everywhere and
the bug is invisible. The defect was a PUT/DRAIN index mismatch: the gather
PUSHes each peer's column indexing the per-PE SDMA signal arrays by GLOBAL pe
(``remotePe * nq``) while the completion drain went through
``shmem::ShmemQuietThread(pe, dest)`` which indexed by ``pe % 8``. On the 2nd
node (pe_base = 8) the drain read the ZEROED slots [0, 8) instead of the armed
slots [8, 16), returned without waiting for SDMA completion, and let the flag
AMO fire before the pushed bytes landed -> the receiver observed the flag but
read torn / stale data (NaN / loss-drift in E2E; the hang is the same race's
never-see-the-flag timing case). This ONLY reproduces when a rank has a
same-node SDMA peer with global pe >= 8, i.e. node 1 of a real 2-node job.

Launch (per node, SAME worktree path), G = ranks-per-node = 8, world = 16:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<ip> \
      --master_port=<p> tests/python/ccl/test_intra_subgroup_sdma_2node.py
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 ...  (on the worker)

Pass criterion: EVERY rank's gathered node-block equals the reference
(``torch.equal``, zero tolerance -- AllGather is a pure data move). Before the
fix, ranks 8..15 (node 1) mismatch non-deterministically; after the fix all
ranks pass on both nodes.
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import IntraNodeSubGroupAllgatherSdma


_DEFAULT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _make_input(dtype, numel, rank, device):
    base = (rank + 1) * 17
    ramp = torch.arange(numel, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _run_one(dtype, numel, rank, world_size, G, device):
    node = rank // G
    local = rank % G
    pe_base = node * G
    elem = torch.tensor([], dtype=dtype).element_size()

    handle = IntraNodeSubGroupAllgatherSdma(
        my_pe=rank,
        npes=world_size,
        out_buffer_bytes=numel * elem * G + 4096,
        group_size=G,
        group_pos=local,
        pe_base=pe_base,
        pe_stride=1,
    )

    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * G, dtype=dtype, device=device)

    # Reference node-block: concat of the G local shards in local-rank order.
    out_ref = torch.empty(numel * G, dtype=dtype, device=device)
    for k in range(G):
        out_ref[k * numel : (k + 1) * numel] = _make_input(dtype, numel, pe_base + k, device)

    stream = torch.cuda.current_stream()
    ok = handle(inp, out_mori, numel, stream)
    assert ok, f"gather call failed dtype={dtype} numel={numel} rank={rank}"
    stream.synchronize()
    torch.cuda.synchronize()
    del handle

    if not torch.equal(out_mori, out_ref):
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"sub-group SDMA gather mismatch dtype={dtype} numel={numel} rank={rank} "
            f"(local={local}, node={node}, pe_base={pe_base}, G={G}): first mismatch "
            f"positions={diff} got={out_mori[diff].tolist()} ref={out_ref[diff].tolist()}"
        )


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # ranks-per-node == sub-group size G (each node gathers its G local shards).
    G = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("RANKS_PER_NODE", "8")))

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)

    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank, (shmem.shmem_mype(), rank)
    assert shmem.shmem_npes() == world_size

    numels = [1024, 1024 * 1024, 8 * 1024 * 1024]
    rc = 0
    try:
        for dtype in _DEFAULT_DTYPES:
            for numel in numels:
                if (numel * torch.tensor([], dtype=dtype).element_size()) % 4 != 0:
                    continue
                # Repeat to shake out the non-deterministic torn-recv race.
                for _ in range(4):
                    _run_one(dtype, numel, rank, world_size, G, device)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"test_intra_subgroup_sdma_2node: PASSED (world={world_size} G={G})")
    except Exception:
        traceback.print_exc()
        rc = 1
    finally:
        try:
            torch.cuda.synchronize()
            dist.barrier()
        except Exception:
            pass
        shmem.shmem_finalize()

    # Any rank's nonzero exit fails the whole torchrun job (which is what we want:
    # pre-fix, ranks 8..15 mismatch -> the launcher reports failure).
    sys.exit(rc)


if __name__ == "__main__":
    main()

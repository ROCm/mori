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
"""Ranks for ``test_sdma_allgather_op.py``: rank 0 prints one ``RESULT_JSON`` line per
case. ``fault`` runs last: it leaves the op unusable."""

import json
import os
import time
from unittest import mock

import torch
import torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.sdma_allgather import SdmaAllGatherOp

CAPACITY = 16 << 20


def _fill(op, block_bytes, base, salt):
    blocks = op.region(block_bytes, base).view(torch.int32).view(op.world_size, -1)
    n = blocks.shape[1]
    blocks[op.rank].copy_(
        torch.arange(n, device="cuda", dtype=torch.int32) + salt * 1000 + op.rank * n
    )
    want = (
        torch.arange(op.world_size * n, device="cuda", dtype=torch.int32) + salt * 1000
    )
    return blocks, want.view(op.world_size, n)


def case_setup_fault(comm, agree):
    """create_dev_comm returns on every rank, then fails on the last one only."""
    real = SdmaAllGatherOp._create_dev_comm

    def fail(self):
        real(self)
        raise RuntimeError("injected")

    last = comm.rank == comm.nranks - 1
    with mock.patch.object(SdmaAllGatherOp, "_create_dev_comm", fail if last else real):
        try:
            SdmaAllGatherOp(comm, CAPACITY, agree=agree)
            return {"raised": False}
        except RuntimeError:
            return {"raised": True}


def case_regions(op):
    """Back to back with no host sync, a different region each call."""
    held = []
    for salt, (block_bytes, base) in enumerate(
        [(16, 32), (1 << 20, 4096), (CAPACITY // op.world_size, 0)]
    ):
        blocks, want = _fill(op, block_bytes, base, salt)
        op.start(block_bytes, base)
        op.wait()
        held.append((blocks.clone(), want))
    return {"ok": all(torch.equal(g, w) for g, w in held), "faulted": op.faulted}


def case_graph(op):
    """start / wait captured once, replayed on new data."""
    blocks = op.region(64 << 10).view(torch.int32).view(op.world_size, -1)
    src = torch.zeros_like(blocks[op.rank])
    out = torch.empty_like(blocks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        blocks[op.rank].copy_(src)
        op.start(64 << 10)
        op.wait()
        out.copy_(blocks)
    ok = True
    for salt in range(3):
        src.fill_(salt * 100 + op.rank)
        graph.replay()
        torch.cuda.synchronize()
        want = torch.arange(op.world_size, device="cuda", dtype=torch.int32)
        ok &= bool((out == want[:, None] + salt * 100).all())
    return {"ok": ok}


def case_fault(op):
    """The last rank never calls: the others give up after a bounded wait, the fault
    sticks, and their devices keep working."""
    missing = op.rank == op.world_size - 1
    res = {"missing": missing}
    if not missing:
        t0 = time.monotonic()
        op.start(4096)
        op.wait(spin_limit=1 << 20)
        res["faulted"] = op.faulted
        res["give_up_s"] = time.monotonic() - t0
        t0 = time.monotonic()
        op.start(4096)
        op.wait()
        torch.cuda.synchronize()
        res["sticky_s"] = time.monotonic() - t0
        x = torch.randn(1 << 20, device="cuda")
        res["device_ok"] = bool(torch.equal(x.cpu().cuda(), x))
    dist.barrier()
    return res


def print_case(name, result):
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, result)
    if dist.get_rank() == 0:
        print(
            "RESULT_JSON " + json.dumps({"case": name, "ranks": per_rank}), flush=True
        )


def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = Communicator.init(
        world,
        rank,
        UniqueId.from_bytes(uid[0]),
        per_rank_vmm=SdmaAllGatherOp.per_rank_vmm(CAPACITY),
    )

    def agree(ok):
        flag = torch.tensor([int(ok)])
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    print_case("setup_fault", case_setup_fault(comm, agree))
    op = SdmaAllGatherOp(comm, CAPACITY, agree=agree)
    op.self_test()
    for name, case in [
        ("regions", case_regions),
        ("graph", case_graph),
        ("fault", case_fault),
    ]:
        print_case(name, case(op))
    op.close()
    dist.barrier()
    comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

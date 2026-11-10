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
"""Ranks for ``test_a2a_gemm.py``'s band-count cases: what the expected signal
count has to survive.

``wrap``   every source's signal counter preset to just below 2^32 before the
           first call, then several calls whose bands carry it across.
``mixed``  one ``arm``, then calls alternating between chunk counts on the same
           window and the same signal slots.

Every call gets fresh operands and is validated against the fp32 reference, so
a band read before it landed shows up as the previous call's bytes. A hang is
the other failure mode; the test's timeout catches it.

Prints one ``RESULT_JSON`` line from rank 0 with every rank's verdicts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco import (
    GDA_CONNECTION_NONE,
    CCODevCommRequirements,
    Communicator,
    UniqueId,
)
from mori.cco.device.flydsl.signals import signal_ptr, signal_store_u32
from mori.ops.a2a_gemm import A2aGemmConfig, build_phases, compile_a2a_gemm
from mori.ops.gemm_ar import preshuffle_b
from mori.tensor_utils import from_gpu_ptr

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "benchmark/cco/flydsl/a2a_gemm")
)
import bench_a2a_gemm as B  # noqa: E402  (operands and reference only)

TOL = 3e-3


def _preset(world: int, rank: int, value: int):
    """A kernel setting this rank's signal counter for every source to
    ``value`` -- only legal with nothing in flight."""
    lo, hi = value & 0xFFFFFFFF, value >> 32

    @flyc.kernel(known_block_size=[64, 1, 1])
    def preset(dev_comm: Int64):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < world:
            if tid != rank:
                slot = fx.Int64(sdma.signal_slot(tid, fx.Int32(0)))
                signal_store_u32(signal_ptr(slot), fx.Int32(lo))
                signal_store_u32(signal_ptr(slot + fx.Int64(4)), fx.Int32(hi))

    @flyc.jit
    def go(dev_comm: Int64, stream=fx.Stream(None)):
        preset(dev_comm).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)

    return go


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("wrap", "mixed"), required=True)
    p.add_argument("--wait", choices=("segment", "tile"), default="segment")
    p.add_argument("--acquire", choices=("inv", "none"), default="inv")
    p.add_argument("-m", type=int, default=1024)
    p.add_argument("-n", type=int, default=1024)
    p.add_argument("--calls", type=int, default=6)
    a = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    uid = UniqueId.from_bytes(payload[0])
    k = world * 256
    m, n = a.m, a.n
    plan = [4] * a.calls if a.case == "wrap" else [4, 2] * (a.calls // 2)
    cfgs = {c: A2aGemmConfig(world, m, k, 256, c) for c in set(plan)}
    cap = max(c.window_bytes for c in cfgs.values())

    b, sb = B.weight(n, k)
    b_i8 = preshuffle_b(b).contiguous().view(torch.int8).view(-1)
    verdicts = []
    with Communicator.init(
        world, rank, uid, per_rank_vmm=2 * cap + (256 << 20)
    ) as comm:
        mem = comm.alloc_mem(cap)
        win = comm.register_window(mem.ptr, mem.size)
        from_gpu_ptr(mem.ptr, (cap,), torch.uint8).zero_()
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        dc = comm.create_dev_comm(reqs)
        stream = fx.Stream(torch.cuda.current_stream())
        if a.case == "wrap":
            _preset(world, rank, 0xFFFF_FFF0)(dc.ptr, stream=stream)
        torch.cuda.synchronize()
        comm.barrier()
        any_cfg = next(iter(cfgs.values()))
        build_phases(any_cfg, rank)["arm"](dc.ptr, win.handle, stream=stream)
        torch.cuda.synchronize()
        comm.barrier()

        kernels = {
            c: compile_a2a_gemm(cfg, rank, N=n, wait=a.wait, acquire=a.acquire)
            for c, cfg in cfgs.items()
        }
        closers = {c: build_phases(cfg, rank)["barrier1"] for c, cfg in cfgs.items()}
        # The window layout does not depend on the chunk count.
        cfg0 = next(iter(cfgs.values()))
        send = from_gpu_ptr(
            mem.ptr + cfg0.send_off, (world * m, cfg0.k_shard), torch.uint8
        )
        a_i8 = from_gpu_ptr(mem.ptr + cfg0.send_off, (cfg0.operand_bytes,), torch.int8)
        sa = torch.empty(m, device="cuda", dtype=torch.float32)
        c_out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        for i, chunks in enumerate(plan):
            salt = 20 + i
            send.copy_(
                B.head_shard(rank, world * m, cfg0.k_shard, salt).view(torch.uint8)
            )
            sa.copy_(B.token_scale(world * m, salt)[rank * m : (rank + 1) * m])
            torch.cuda.synchronize()
            comm.barrier()
            kernels[chunks](
                a_i8, b_i8, c_out.view(-1), sa, sb, dc.ptr, win.handle, stream=stream
            )
            closers[chunks](dc.ptr, win.handle, stream=stream)
            torch.cuda.synchronize()
            comm.barrier()
            ref = B.reference(world, rank, m, k, b, sb, salt)
            rel = ((c_out.float() - ref).norm() / ref.norm()).item()
            verdicts.append({"call": i, "chunks": chunks, "rel": rel, "ok": rel < TOL})
        win.close()
        mem.close()
    gathered = [None] * world
    dist.all_gather_object(gathered, verdicts)
    if rank == 0:
        print(
            "RESULT_JSON " + json.dumps({"case": a.case, "ranks": gathered}), flush=True
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

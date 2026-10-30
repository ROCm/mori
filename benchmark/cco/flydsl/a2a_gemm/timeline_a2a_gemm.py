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
"""When does each band land, with and without the GEMM running?

The fused kernel with its diagnostic timeline on (``_timeline=True``): block 0
stamps, on every rank, when every (source, band) has landed here, and every tile
stamps its start and end. Variants:

    idle      grid=1: block 0 alone (post, observe, its one tile) -- the
              transfer with nothing else on the GPU
    nowait    the full grid, no waits: the transfer under the GEMM
    tile      the real kernel, waiting once per tile
    segment   the real kernel, waiting per K segment

    MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 torchrun --standalone \\
        --nproc_per_node=8 timeline_a2a_gemm.py -m 4096 -n 8192 -k 8192 \\
        --chunks 1 8 --out timeline.json

Times are microseconds from block 0's first stamp on each rank, medians over
``--iters``. A band's time is when its *last* source landed: one wave watches
every source, so a lane that is done waits for the slowest before it stamps.
Operands are random and C is not checked; this is for timing only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import flydsl.expr as fx
import torch
import torch.distributed as dist

from mori.cco import (
    GDA_CONNECTION_NONE,
    CCODevCommRequirements,
    Communicator,
    UniqueId,
)
from mori.ops.a2a_gemm import A2aGemmConfig, build_phases, compile_a2a_gemm
from mori.ops.a2a_gemm.layout import MAX_CHUNKS
from mori.ops.gemm_ar import preshuffle_b
from mori.tensor_utils import from_gpu_ptr

VARIANTS = ("idle", "nowait", "tile", "segment")


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def _decode(raw, cfg, rank, world, tiles):
    t0 = raw[0]

    def us(v):
        return (v - t0) / 100.0 if v else None

    bands = [
        max(us(raw[3 + s * MAX_CHUNKS + c]) for s in range(world) if s != rank)
        for c in range(cfg.chunks)
    ]
    spans = [
        (us(raw[cfg.STAMP_FIXED + 2 * i]), us(raw[cfg.STAMP_FIXED + 2 * i + 1]))
        for i in range(tiles)
    ]
    spans = [(a, b) for a, b in spans if a is not None and b is not None]
    return {
        "posted": us(raw[1]),
        "quiet": us(raw[2]),
        "band_landed": bands,
        "tile_end": max(b for _, b in spans) if spans else None,
        "last_tile_start": max(a for a, _ in spans) if spans else None,
        "tile_us": statistics.median(b - a for a, b in spans) if spans else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", type=int, default=4096)
    ap.add_argument("-k", type=int, default=8192)
    ap.add_argument("-n", type=int, default=8192)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    uid = UniqueId.from_bytes(payload[0])

    cfgs = [
        A2aGemmConfig(world, args.m, args.k, 256, c, timeline=True) for c in args.chunks
    ]
    cap = max(c.window_bytes for c in cfgs)
    comm = Communicator.init(world, rank, uid, per_rank_vmm=2 * cap + (512 << 20))
    mem = comm.alloc_mem(cap)
    win = comm.register_window(mem.ptr, mem.size)
    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_NONE
    reqs.gda_signal_count = 0
    reqs.gda_counter_count = 0
    reqs.sdma_queue_count = 1
    dc = comm.create_dev_comm(reqs)

    b = (torch.randn(args.n, args.k, device="cuda") / 8).to(torch.float8_e4m3fn)
    b_i8 = preshuffle_b(b).contiguous().view(torch.int8).view(-1)
    sb = torch.rand(args.n, device="cuda") * 0.01 + 0.01
    sa = torch.rand(args.m, device="cuda") * 0.01 + 0.01
    c_out = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)
    results = []
    for cfg in cfgs:
        tiles = cfg.m_tiles * (args.n // 256)
        from_gpu_ptr(mem.ptr, (cfg.control_bytes,), torch.uint8).zero_()
        operand = from_gpu_ptr(
            mem.ptr + cfg.send_off, (cfg.operand_bytes,), torch.uint8
        )
        operand.copy_(
            (torch.randn(cfg.operand_bytes, device="cuda") / 8)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        a_i8 = operand.view(torch.int8)
        stamps = from_gpu_ptr(
            mem.ptr + cfg.stamp_off, (cfg.stamp_slots(tiles),), torch.int64
        )
        phases = build_phases(cfg, rank)
        torch.cuda.synchronize()
        comm.barrier()
        phases["arm"](dc.ptr, win.handle, stream=fx.Stream(torch.cuda.current_stream()))
        torch.cuda.synchronize()
        comm.barrier()
        for variant in args.variants:
            kernel = compile_a2a_gemm(
                cfg,
                rank,
                N=args.n,
                wait="none" if variant in ("idle", "nowait") else variant,
                acquire="none",
                _timeline=True,
                _grid=1 if variant == "idle" else 0,
            )
            runs = []
            for it in range(args.iters + 3):
                stamps.zero_()
                torch.cuda.synchronize()
                comm.barrier()
                stream = fx.Stream(torch.cuda.current_stream())
                kernel(
                    a_i8,
                    b_i8,
                    c_out.view(-1),
                    sa,
                    sb,
                    dc.ptr,
                    win.handle,
                    stream=stream,
                )
                phases["barrier1"](dc.ptr, win.handle, stream=stream)
                torch.cuda.synchronize()
                comm.barrier()
                if it >= 3:
                    runs.append(_decode(stamps.cpu().tolist(), cfg, rank, world, tiles))
            mine = {
                key: _median(r[key] for r in runs)
                for key in ("posted", "quiet", "tile_end", "last_tile_start", "tile_us")
            }
            mine["band_landed"] = [
                _median(r["band_landed"][c] for r in runs) for c in range(cfg.chunks)
            ]
            gathered = [None] * world
            dist.all_gather_object(gathered, mine)
            if rank == 0:
                row = dict(
                    world=world, m=args.m, n=args.n, k=args.k,
                    chunks=cfg.chunks, variant=variant, ranks=gathered,
                )  # fmt: skip
                results.append(row)
                print(
                    f"c{cfg.chunks} {variant:8s} "
                    f"last band landed {[g['band_landed'][-1] for g in gathered]} "
                    f"| tiles end {[g['tile_end'] for g in gathered]}",
                    flush=True,
                )
                print(
                    f"   rank 0: bands landed {gathered[0]['band_landed']}, "
                    f"tile {gathered[0]['tile_us']}us, "
                    f"last tile started {gathered[0]['last_tile_start']}",
                    flush=True,
                )
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(results, f)
    dist.barrier()
    win.close()
    mem.close()


if __name__ == "__main__":
    main()

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
"""All-gather + fp8 GEMM, fused against split, at a ColumnParallelLinear after SP.

    MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 \\
    torchrun --standalone --nproc_per_node=8 bench_ag_gemm.py \\
        --mode fused-sdma -m 512 -n 6400 -k 7168 --chunks 2

``-m`` is rows **per rank**: every rank ends with ``[world*m, n]``, its own
weight shard applied to every rank's tokens. Modes, all ending in that state:

    gemm-only    the GEMM over an already-gathered operand: the compute floor
    split-rccl   all_gather_into_tensor (rows, then scales) -> GEMM
    split-sdma   push -> barrier -> GEMM -> barrier
    fused-sdma   GEMM with the push in block 0 and per-band waits -> barrier

``split-*`` and ``gemm-only`` share one GEMM kernel (``fuse=False``); ``fused``
differs from it in the tile order, the waits and the poster only.

Validation reads every row of C against an fp32 host reference built from all
ranks' operands, twice with different operands: a remote tile that reads a band
before it lands returns the *previous* call's rows, which only a second,
different call can tell from the right answer.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import flydsl.expr as fx
import torch
import torch.distributed as dist

from mori.cco import (
    GDA_CONNECTION_NONE,
    CCODevCommRequirements,
    Communicator,
    UniqueId,
)
from mori.ops.ag_gemm import AgGemmConfig, build_phases, compile_ag_gemm
from mori.ops.gemm_ar import preshuffle_b
from mori.tensor_utils import from_gpu_ptr

MODES = ("gemm-only", "split-rccl", "split-sdma", "fused-sdma")
VMM_SLACK = 512 * 1024 * 1024


def _setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return local_rank, rank, world, UniqueId.from_bytes(payload[0])


def activation(rank: int, m: int, k: int, salt: int):
    """Rank ``rank``'s token shard and its per-token scale."""
    g = torch.Generator(device="cuda").manual_seed(1000 + rank + 7919 * salt)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sa = torch.rand(m, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    return a, sa


def weight(rank: int, n: int, k: int):
    """Rank ``rank``'s column shard of the weight and its per-channel scale."""
    g = torch.Generator(device="cuda").manual_seed(5000 + rank)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sb = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    return b, sb


def reference(world, m, k, b, sb, salt):
    a = torch.cat([activation(s, m, k, salt)[0] for s in range(world)]).float()
    sa = torch.cat([activation(s, m, k, salt)[1] for s in range(world)])
    return (a @ b.float().T) * sa[:, None] * sb[None, :]


def _median_us(fn, warmup, iters, *, graph=True):
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        # thread_local so RCCL's watchdog may query events from its own thread
        # while this one captures (gemm_ag's bench found the global mode aborts).
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            fn()
        torch.cuda.synchronize()
        return _median_us(g.replay, warmup, iters, graph=False)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def run(args) -> int:
    local_rank, rank, world, uid = _setup_distributed()
    cfg = AgGemmConfig(
        world_size=world,
        m_local=args.m,
        k=args.k,
        block_m=args.block_m,
        chunks=args.chunks,
    )
    cfg.validate()
    fused = args.mode == "fused-sdma"
    sdma = args.mode in ("split-sdma", "fused-sdma", "gemm-only")

    b, sb = weight(rank, args.n, args.k)
    b_i8 = preshuffle_b(b).contiguous().view(torch.int8).view(-1)
    c = torch.empty(cfg.m, args.n, device="cuda", dtype=torch.bfloat16)
    c_flat = c.view(-1)

    vmm = 2 * cfg.window_bytes + VMM_SLACK
    result = None
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        mem = comm.alloc_mem(cfg.window_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        # Flags and the epoch are compared before they are first written.
        from_gpu_ptr(mem.ptr, (cfg.window_bytes,), torch.uint8).zero_()
        recv = from_gpu_ptr(mem.ptr + cfg.recv_off, (cfg.m, args.k), torch.uint8)
        recv_scale = from_gpu_ptr(mem.ptr + cfg.scale_off, (cfg.m,), torch.float32)
        my_rows = recv[rank * args.m : (rank + 1) * args.m]
        my_scale = recv_scale[rank * args.m : (rank + 1) * args.m]
        a_i8 = recv.view(torch.int8).view(-1)
        torch.cuda.synchronize()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        if sdma:
            reqs.sdma_queue_count = 1
        dc = comm.create_dev_comm(reqs)

        gemm = compile_ag_gemm(
            cfg, rank, N=args.n, fuse=fused, waves_per_eu=args.waves_per_eu
        )
        phases = build_phases(cfg, rank) if sdma else None
        # RCCL gathers out of a tensor of its own; fp8 travels as bytes.
        local_a = torch.empty(args.m, args.k, device="cuda", dtype=torch.uint8)
        local_s = torch.empty(args.m, device="cuda", dtype=torch.float32)

        def load_operands(salt):
            a, sa = activation(rank, args.m, args.k, salt)
            my_rows.copy_(a.view(torch.uint8))
            my_scale.copy_(sa)
            local_a.copy_(a.view(torch.uint8))
            local_s.copy_(sa)

        def run_gemm(stream, out=c_flat):
            gemm(a_i8, b_i8, out, recv_scale, sb, dc.ptr, win.handle, stream=stream)

        def once(out=c_flat):
            stream = fx.Stream(torch.cuda.current_stream())
            if args.mode == "split-rccl":
                dist.all_gather_into_tensor(recv, local_a)
                dist.all_gather_into_tensor(recv_scale, local_s)
                run_gemm(stream, out)
            elif args.mode == "split-sdma":
                phases["push"](dc.ptr, win.handle, stream=stream)
                phases["barrier0"](dc.ptr, win.handle, stream=stream)
                run_gemm(stream, out)
                phases["barrier1"](dc.ptr, win.handle, stream=stream)
            elif args.mode == "fused-sdma":
                run_gemm(stream, out)
                phases["barrier1"](dc.ptr, win.handle, stream=stream)
            else:
                run_gemm(stream, out)

        def gather_once():
            """gemm-only times the GEMM alone, so gather for it up front."""
            stream = fx.Stream(torch.cuda.current_stream())
            phases["push"](dc.ptr, win.handle, stream=stream)
            phases["barrier0"](dc.ptr, win.handle, stream=stream)
            phases["barrier1"](dc.ptr, win.handle, stream=stream)

        rel = []
        for salt in (0, 1) if not args.skip_validation else (0,):
            load_operands(salt)
            comm.barrier()
            if args.mode == "gemm-only":
                gather_once()
            once()
            torch.cuda.synchronize()
            comm.barrier()
            if not args.skip_validation:
                ref = reference(world, args.m, args.k, b, sb, salt)
                got = c.float()
                rel.append(((got - ref).norm() / ref.norm()).item())
                per_src = [
                    (
                        (
                            got[s * args.m : (s + 1) * args.m]
                            - ref[s * args.m : (s + 1) * args.m]
                        ).norm()
                        / ref[s * args.m : (s + 1) * args.m].norm()
                    ).item()
                    for s in range(world)
                ]
                if rel[-1] >= args.tolerance:
                    print(
                        f"[rank {rank}] salt {salt} per-source relL2: {[f'{x:.2e}' for x in per_src]}",
                        flush=True,
                    )
                del ref, got
        if args.back_to_back and args.mode != "gemm-only":
            # Calls issued back to back with no host sync, each with different
            # operands into its own C, and one rank held on the GPU before each
            # (a different one every call). The ranks it is not holding run a
            # call ahead, so their posters refill slots a held rank may still
            # be reading -- what barrier1 exists to prevent. Every call is then
            # checked on its own.
            outs = [torch.empty_like(c) for _ in range(args.back_to_back)]
            stage = []
            for i in range(args.back_to_back):
                a, sa = activation(rank, args.m, args.k, 10 + i)
                stage.append((a.view(torch.uint8).clone(), sa.clone()))
            comm.barrier()
            for i, out in enumerate(outs):
                if i % world == rank:
                    torch.cuda._sleep(args.skew_cycles)
                my_rows.copy_(stage[i][0])
                my_scale.copy_(stage[i][1])
                local_a.copy_(stage[i][0])
                local_s.copy_(stage[i][1])
                once(out.view(-1))
            torch.cuda.synchronize()
            comm.barrier()
            for i, out in enumerate(outs):
                ref = reference(world, args.m, args.k, b, sb, 10 + i)
                r = ((out.float() - ref).norm() / ref.norm()).item()
                rel.append(r)
                if r >= args.tolerance:
                    print(
                        f"[rank {rank}] back-to-back call {i}: relL2 {r:.2e}",
                        flush=True,
                    )
                del ref
            del outs, stage
        rel_l2 = max(rel) if rel else float("nan")
        validated = (not rel) or rel_l2 < args.tolerance

        comm.barrier()
        us = _median_us(once, args.warmup, args.iters, graph=not args.eager)
        comm.barrier()

        gathered = [None] * world
        dist.all_gather_object(gathered, {"us": us, "ok": validated, "rel": rel_l2})
        if rank == 0:
            result = {
                "mode": args.mode,
                "world_size": world,
                "m_local": args.m,
                "n": args.n,
                "k": args.k,
                "chunks": args.chunks if fused else None,
                "block_m": args.block_m,
                "us": max(g["us"] for g in gathered),
                "per_rank_us": [g["us"] for g in gathered],
                "rel_l2": max(g["rel"] for g in gathered),
                "validated": all(g["ok"] for g in gathered),
                "remote_bytes_per_rank": cfg.remote_bytes_per_rank,
                "timing": "eager" if args.eager else "graph",
            }
            print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
            print(
                f"[ag_gemm] mode={args.mode} m_local={args.m} n={args.n} k={args.k} "
                f"chunks={result['chunks']} time={result['us']:.1f}us "
                f"relL2={result['rel_l2']:.2e} validated={result['validated']}",
                flush=True,
            )
        win.close()
        mem.close()
    dist.barrier()
    return 0 if (result is None or result["validated"]) else 1


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=MODES, default="fused-sdma")
    p.add_argument("-m", type=int, default=512, help="rows per rank")
    p.add_argument("--out-dim", "-N", dest="n", type=int, default=6400)
    p.add_argument("-k", type=int, default=7168)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--block-m", type=int, default=256)
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=51)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument(
        "--back-to-back",
        type=int,
        default=0,
        help="also run this many calls back to back, each with its own operands "
        "and output, one rank held before each; validate every one",
    )
    p.add_argument("--skew-cycles", type=int, default=50_000_000)
    #: fp8 ptpc lands at ~1.66e-3 against the fp32 reference; a stale or missing
    #: band lands near 1.
    p.add_argument("--tolerance", type=float, default=3e-3)
    return p


if __name__ == "__main__":
    try:
        sys.exit(run(build_parser().parse_args()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

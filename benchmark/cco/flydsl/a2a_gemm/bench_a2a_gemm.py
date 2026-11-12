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
"""All-to-all + fp8 GEMM, fused against split, at Ulysses' o-projection.

    MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 \\
    torchrun --standalone --nproc_per_node=4 bench_a2a_gemm.py \\
        -m 2048 4096 -n 6144 -k 6144 --chunks 1 2 4 8 --rounds 3 --out a2a.jsonl

``-m`` is tokens **per rank** after the exchange. Before it, every rank holds
all ``world*m`` tokens for its ``K/world`` columns (its heads); after it, its
own ``m`` tokens over all ``K``, times the replicated ``B [N, K]``. Modes, all
ending in ``C [m, N]``:

    gemm-only    the GEMM over an already-exchanged operand: the compute floor
    split-rccl   all_to_all_single -> GEMM
    split-sdma   push -> barrier -> GEMM -> barrier
    fused-sdma   GEMM with the push in block 0 and per-segment waits -> barrier

``split-*`` and ``gemm-only`` share one GEMM kernel (``fuse=False``); ``fused``
differs from it in the waits and the poster only.

Every case runs in this one process on one communicator, rounds interleaved,
each timed by launching it back to back with an event between launches for at
least ``--min-seconds`` (median launch gap, max over ranks): the GPUs never wait
on the host or on a process launch. Validation reads every row of C against an
fp32 host reference built from all ranks' operands, twice with different
operands -- a tile that reads a band before it lands returns the *previous*
call's bytes, which only a second, different call can tell from the right
answer -- and ``--back-to-back`` adds calls with no host sync between them, one
rank held on the GPU before each.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

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


def head_shard(src: int, tokens: int, k_shard: int, salt: int):
    """Source ``src``'s attention output: every token, its own heads, fp8."""
    g = torch.Generator(device="cuda").manual_seed(1000 + src + 7919 * salt)
    return (torch.randn(tokens, k_shard, generator=g, device="cuda") / 8).to(
        torch.float8_e4m3fn
    )


def token_scale(tokens: int, salt: int):
    """The per-token scale over all heads; the same on every rank (ATOM gathers
    the row maxima before quantizing, so every rank agrees on it)."""
    g = torch.Generator(device="cuda").manual_seed(3000 + 7919 * salt)
    return (
        torch.rand(tokens, generator=g, device="cuda", dtype=torch.float32) * 0.01
        + 0.01
    )


def weight(n: int, k: int):
    """The replicated o-projection weight and its per-channel scale."""
    g = torch.Generator(device="cuda").manual_seed(5000)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sb = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    return b, sb


def reference(world, rank, m, k, b, sb, salt):
    ks = k // world
    a = torch.cat(
        [
            head_shard(s, world * m, ks, salt)[rank * m : (rank + 1) * m]
            for s in range(world)
        ],
        dim=1,
    ).float()
    sa = token_scale(world * m, salt)[rank * m : (rank + 1) * m]
    return (a @ b.float().T) * sa[:, None] * sb[None, :]


def _median_us(fn, warmup, iters, min_seconds):
    """Graph-capture ``fn``, then replay it back to back with an event between
    launches for at least ``min_seconds``; the median launch gap."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # thread_local so RCCL's watchdog may query events from its own thread while
    # this one captures.
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    gaps, t0 = [], time.monotonic()
    while time.monotonic() - t0 < min_seconds or len(gaps) < iters:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(51)]
        ev[0].record()
        for i in range(50):
            g.replay()
            ev[i + 1].record()
        ev[-1].synchronize()
        gaps += [ev[i].elapsed_time(ev[i + 1]) * 1000.0 for i in range(50)]
    return statistics.median(gaps)


class Bench:
    """One communicator, window and dev-comm for every case of a sweep."""

    def __init__(self, args, world, rank, uid):
        self.args, self.world, self.rank = args, world, rank
        self.b, self.sb = weight(args.n, args.k)
        self.b_i8 = preshuffle_b(self.b).contiguous().view(torch.int8).view(-1)
        cap = max(
            A2aGemmConfig(world, m, args.k, args.block_m).window_bytes for m in args.m
        )
        self.comm = Communicator.init(
            world, rank, uid, per_rank_vmm=2 * cap + VMM_SLACK
        )
        self.mem = self.comm.alloc_mem(cap)
        self.win = self.comm.register_window(self.mem.ptr, self.mem.size)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self.dc = self.comm.create_dev_comm(reqs)
        self._kernels = {}

    def close(self):
        self.win.close()
        self.mem.close()
        self.comm.__exit__(None, None, None)

    def _gemm(self, cfg, fuse, self_from_send, wait="segment"):
        key = (
            cfg,
            fuse,
            self_from_send,
            wait,
            self.args.acquire,
            self.args.no_post,
        )
        if key not in self._kernels:
            self._kernels[key] = compile_a2a_gemm(
                cfg,
                self.rank,
                N=self.args.n,
                fuse=fuse,
                self_from_send=self_from_send,
                waves_per_eu=self.args.waves_per_eu,
                wait=wait,
                acquire=self.args.acquire,
                _post=not self.args.no_post,
            )
        return self._kernels[key]

    def _phases(self, cfg):
        key = ("phases", cfg)
        if key not in self._kernels:
            self._kernels[key] = build_phases(cfg, self.rank)
        return self._kernels[key]

    def case(self, mode, m, chunks, wait="segment"):
        args, world, rank = self.args, self.world, self.rank
        cfg = A2aGemmConfig(world, m, args.k, args.block_m, chunks)
        cfg.validate()
        ptr = self.mem.ptr
        # Flags, barriers and the epoch are compared before they are written; a
        # previous case's residue would satisfy a wait early.
        from_gpu_ptr(ptr, (cfg.control_bytes,), torch.uint8).zero_()
        send = from_gpu_ptr(ptr + cfg.send_off, (world * m, cfg.k_shard), torch.uint8)
        recv = from_gpu_ptr(ptr + cfg.recv_off, (world * m, cfg.k_shard), torch.uint8)
        a_i8 = from_gpu_ptr(ptr + cfg.send_off, (cfg.operand_bytes,), torch.int8)
        c = torch.empty(m, args.n, device="cuda", dtype=torch.bfloat16)
        sa = torch.empty(m, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        self.comm.barrier()

        fused = mode == "fused-sdma"
        gemm = self._gemm(cfg, fused, mode != "split-rccl", wait)
        phases = self._phases(cfg)
        dc, win = self.dc.ptr, self.win.handle
        # The copy engine's signal counters run on across cases; start this
        # one's count from where they stand, with nothing in flight.
        phases["arm"](dc, win, stream=fx.Stream(torch.cuda.current_stream()))
        torch.cuda.synchronize()
        self.comm.barrier()

        def load(salt):
            send.copy_(head_shard(rank, world * m, cfg.k_shard, salt).view(torch.uint8))
            sa.copy_(token_scale(world * m, salt)[rank * m : (rank + 1) * m])

        def once(out=c):
            stream = fx.Stream(torch.cuda.current_stream())

            def run():
                gemm(a_i8, self.b_i8, out.view(-1), sa, self.sb, dc, win, stream=stream)

            if mode == "split-rccl":
                dist.all_to_all_single(recv.view(-1), send.view(-1))
                run()
            elif mode == "split-sdma":
                phases["push"](dc, win, stream=stream)
                phases["barrier0"](dc, win, stream=stream)
                run()
                phases["barrier1"](dc, win, stream=stream)
            elif fused:
                run()
                phases["barrier1"](dc, win, stream=stream)
            else:
                run()

        def exchange():
            """gemm-only times the GEMM alone, so exchange for it up front."""
            stream = fx.Stream(torch.cuda.current_stream())
            phases["push"](dc, win, stream=stream)
            phases["barrier0"](dc, win, stream=stream)
            phases["barrier1"](dc, win, stream=stream)

        rel = []
        skip = args.skip_validation or wait == "none"
        salts = (0,) if skip else (0, 1)
        for salt in salts:
            load(salt)
            torch.cuda.synchronize()
            self.comm.barrier()
            if mode == "gemm-only":
                exchange()
            once()
            torch.cuda.synchronize()
            self.comm.barrier()
            if not skip:
                ref = reference(world, rank, m, args.k, self.b, self.sb, salt)
                rel.append(((c.float() - ref).norm() / ref.norm()).item())
                del ref
        if args.back_to_back and mode not in ("gemm-only",) and not skip:
            # Calls issued back to back with no host sync, each with different
            # operands into its own C, one rank held on the GPU before each (a
            # different one every call): the ranks it is not holding run a call
            # ahead, so their posters refill recv slots the held rank may still
            # be reading -- what barrier1 exists to prevent.
            outs = [torch.empty_like(c) for _ in range(args.back_to_back)]
            stage = [
                (
                    head_shard(rank, world * m, cfg.k_shard, 10 + i).view(torch.uint8),
                    token_scale(world * m, 10 + i)[rank * m : (rank + 1) * m].clone(),
                )
                for i in range(args.back_to_back)
            ]
            torch.cuda.synchronize()
            self.comm.barrier()
            for i, out in enumerate(outs):
                if i % world == rank:
                    torch.cuda._sleep(args.skew_cycles)
                send.copy_(stage[i][0])
                sa.copy_(stage[i][1])
                once(out)
            torch.cuda.synchronize()
            self.comm.barrier()
            for i, out in enumerate(outs):
                ref = reference(world, rank, m, args.k, self.b, self.sb, 10 + i)
                rel.append(((out.float() - ref).norm() / ref.norm()).item())
                del ref
            del outs, stage
        rel_l2 = max(rel) if rel else float("nan")
        ok = (not rel) or rel_l2 < args.tolerance
        if not ok:
            print(f"[rank {rank}] {mode} m={m} c={chunks}: relL2 {rel}", flush=True)

        self.comm.barrier()
        us = _median_us(once, args.warmup, args.iters, args.min_seconds)
        self.comm.barrier()
        gathered = [None] * world
        dist.all_gather_object(gathered, {"us": us, "ok": ok, "rel": rel_l2})
        return {
            "mode": mode,
            "world_size": world,
            "m_local": m,
            "n": args.n,
            "k": args.k,
            "chunks": chunks if fused else None,
            "wait": wait if fused else None,
            "acquire": args.acquire if fused else None,
            "block_m": args.block_m,
            "us": max(g["us"] for g in gathered),
            "per_rank_us": [round(g["us"], 2) for g in gathered],
            "rel_l2": max(g["rel"] for g in gathered),
            "validated": all(g["ok"] for g in gathered),
            "remote_bytes_per_rank": cfg.remote_bytes_per_rank,
        }


def _cases(args, world):
    out = []
    for m in args.m:
        for mode in args.modes:
            if mode == "fused-sdma":
                for c in args.chunks:
                    if m % (args.block_m * c) == 0:
                        out.extend((mode, m, c, w) for w in args.wait)
            else:
                out.append((mode, m, 1, None))
    return out


def run(args) -> int:
    local_rank, rank, world, uid = _setup_distributed()
    bench = Bench(args, world, rank, uid)
    cases = _cases(args, world)
    fh = open(args.out, "a") if (args.out and rank == 0) else None
    all_ok = True
    try:
        for rnd in range(args.rounds):
            for mode, m, c, w in cases if rnd % 2 == 0 else cases[::-1]:
                res = bench.case(mode, m, c, w or "segment")
                all_ok &= res["validated"]
                if rank == 0:
                    res["round"] = rnd
                    print("RESULT_JSON " + json.dumps(res, sort_keys=True), flush=True)
                    print(
                        f"r{rnd} {mode:<11} m={m:<6} c={str(res['chunks']):<4} "
                        f"{str(res['wait'] or ''):<8} "
                        f"{res['us']:9.1f}us relL2={res['rel_l2']:.2e} ok={res['validated']}",
                        flush=True,
                    )
                    if fh:
                        fh.write(json.dumps(res, sort_keys=True) + "\n")
                        fh.flush()
    finally:
        if fh:
            fh.close()
        bench.close()
    dist.barrier()
    return 0 if all_ok else 1


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("-m", type=int, nargs="+", default=[2048], help="tokens per rank")
    p.add_argument("-n", type=int, default=6144)
    p.add_argument("-k", type=int, default=6144)
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--block-m", type=int, default=256)
    p.add_argument(
        "--wait",
        nargs="+",
        choices=("segment", "tile", "none"),
        default=["segment"],
        help="'none' is diagnostic: no waits, so not validated",
    )
    p.add_argument("--acquire", choices=("inv", "none"), default="inv")
    p.add_argument(
        "--no-post",
        action="store_true",
        help="diagnostic, with --wait none: the fused kernel without its puts",
    )
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--min-seconds", type=float, default=3.0)
    p.add_argument("--out", default="")
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

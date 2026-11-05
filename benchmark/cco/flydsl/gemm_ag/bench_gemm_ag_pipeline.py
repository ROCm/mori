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
"""Benchmark M-chunk GEMM + SDMA with serial or two-stream scheduling."""
import argparse
import json
import os
from types import SimpleNamespace

from _bench_utils import measure, positive_int
import bench_gemm_ag as bench

import flydsl.expr as fx
import torch
import torch.distributed as dist

from mori.cco import CCODevCommRequirements, Communicator, GDA_CONNECTION_NONE, UniqueId
from mori.ops.gemm_ag import (
    ag_config,
    build_sdma_phases,
    build_sdma_chunk_post,
    compile_bf16_gemm_ag,
)
from mori.tensor_utils import from_gpu_ptr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-m", type=positive_int, default=2048)
    p.add_argument("--out-dim", dest="n", type=positive_int, default=2048)
    p.add_argument("-k", type=positive_int, default=7168)
    p.add_argument("--backend", choices=("mori", "splitk", "torch"), default="mori")
    p.add_argument("--chunks", type=int, choices=(1, 2, 4, 8, 16), default=4)
    p.add_argument("--split-k", type=int, default=4)
    p.add_argument("--schedule", choices=("overlap", "serial"), default="overlap")
    p.add_argument("--rounds", type=positive_int, default=5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=positive_int, default=101)
    args = p.parse_args()
    if args.warmup < 0:
        p.error("warmup must be nonnegative")
    if args.backend == "splitk" and args.split_k not in (2, 4, 8):
        p.error("--backend splitk requires --split-k 2, 4, or 8")
    if os.environ.get("MORI_ENABLE_SDMA") != "1":
        p.error("set MORI_ENABLE_SDMA=1 and use a MORI build with SDMA support")
    splits = args.split_k if args.backend == "splitk" else 1
    if args.m % args.chunks or (args.m // args.chunks) % 128 or args.n % 128:
        p.error("each M chunk and N must be divisible by 128")
    if args.k % (splits * 64) or args.k // (splits * 64) < 2:
        p.error("each K partition must contain at least two complete 64-element steps")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    m, n, k = args.m, args.n, args.k
    rows = m // args.chunks
    assert m % args.chunks == 0 and rows % 128 == 0
    cfg = ag_config(world_size=world, m=m, n=n, elem_bytes=4, block_m=128, block_n=128)
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    uid = UniqueId.from_bytes(payload[0])
    a, b, _, _ = bench.make_operands_bf16(rank, m, n, k)
    b_arg = b.view(torch.int16).view(-1)
    if args.backend != "torch":
        local_cfg = ag_config(
            world_size=world, m=rows, n=n, elem_bytes=4, block_m=128, block_n=128
        )
        gemm = compile_bf16_gemm_ag(
            local_cfg,
            rank,
            K=k,
            BLOCK_M=128,
            BLOCK_N=128,
            out_dtype="fp32",
            split_k=splits,
        )
    partials = (
        torch.empty((args.split_k, rows, n), dtype=torch.float32, device="cuda")
        if args.backend == "splitk"
        else None
    )
    partial_arg = partials.view(-1) if partials is not None else None
    with Communicator.init(
        world, rank, uid, per_rank_vmm=2 * cfg.window_bytes + (64 << 20)
    ) as comm:
        mem = comm.alloc_mem(cfg.window_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        from_gpu_ptr(mem.ptr, (cfg.window_bytes,), torch.uint8).zero_()
        recv = from_gpu_ptr(mem.ptr + cfg.recv_off, (world * m, n), torch.float32)
        req = CCODevCommRequirements()
        req.gda_connection_type = GDA_CONNECTION_NONE
        req.gda_signal_count = req.gda_counter_count = 0
        req.sdma_queue_count = 1
        dc = comm.create_dev_comm(req)
        c = recv[rank * m : (rank + 1) * m]
        aa = [a[i * rows : (i + 1) * rows] for i in range(args.chunks)]
        a_args = [v.view(torch.int16).view(-1) for v in aa]
        cc = [c[i * rows : (i + 1) * rows] for i in range(args.chunks)]
        c_args = [v.view(-1) for v in cc]
        post = build_sdma_chunk_post(cfg, rank)
        drain = build_sdma_phases(cfg, rank, queues=1)["drain"]
        tx = torch.cuda.Stream()
        ready = [torch.cuda.Event() for _ in range(args.chunks)]
        chunk_bytes = rows * n * 4

        def compute(i):
            if args.backend == "torch":
                torch.mm(aa[i], b.T, out_dtype=torch.float32, out=cc[i])
            elif args.backend == "splitk":
                gemm(
                    a_args[i],
                    b_arg,
                    partial_arg,
                    rows,
                    n,
                    0,
                    0,
                    stream=fx.Stream(torch.cuda.current_stream()),
                )
                torch.sum(partials, dim=0, out=cc[i])
            else:
                gemm(
                    a_args[i],
                    b_arg,
                    c_args[i],
                    rows,
                    n,
                    0,
                    0,
                    stream=fx.Stream(torch.cuda.current_stream()),
                )

        def once():
            stream = torch.cuda.current_stream()
            for i in range(args.chunks):
                compute(i)
                ready[i].record(stream)
                if args.schedule == "overlap":
                    with torch.cuda.stream(tx):
                        tx.wait_event(ready[i])
                        post(
                            dc.ptr,
                            win.handle,
                            i * chunk_bytes,
                            chunk_bytes,
                            stream=fx.Stream(tx),
                        )
            if args.schedule == "serial":
                with torch.cuda.stream(tx):
                    tx.wait_event(ready[-1])
                    for i in range(args.chunks):
                        post(
                            dc.ptr,
                            win.handle,
                            i * chunk_bytes,
                            chunk_bytes,
                            stream=fx.Stream(tx),
                        )
            stream.wait_stream(tx)
            drain(dc.ptr, win.handle, stream=fx.Stream(stream))

        validation_args = SimpleNamespace(
            in_dtype="bf16",
            out_dtype="fp32",
            quant="ptpc",
            m=m,
            n=n,
            k=k,
            tolerance=1e-5,
        )

        def validate(snapshot, sign):
            rel, ok = bench._validate_recv(
                snapshot, validation_args, rank, world, sign=sign
            )
            ok = ok and bool(torch.isfinite(snapshot).all().item())
            checks = [None] * world
            dist.all_gather_object(checks, {"ok": ok, "rel": rel})
            assert all(c["ok"] for c in checks), checks
            return max(c["rel"] for c in checks)

        torch.cuda.synchronize()
        dist.barrier()
        once()
        initial = recv.clone()
        torch.cuda.synchronize()
        errors = [validate(initial, 1)]
        us, timing, replay = measure(once, args.warmup, args.iters, args.rounds)
        for sign in (-1, 1):
            a.neg_()
            recv.fill_(float("nan"))
            if partials is not None:
                partials.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
            replay()
            snapshot = recv.clone()
            torch.cuda.synchronize()
            errors.append(validate(snapshot, sign))
        if rank == 0:
            print(
                "TIMING_JSON "
                + json.dumps(
                    dict(
                        **timing,
                        changed_input_rel_l2=errors,
                    )
                ),
                flush=True,
            )
            print(
                "RESULT_JSON "
                + json.dumps(
                    dict(
                        mode="pipeline-sdma",
                        world_size=world,
                        backend=args.backend,
                        chunks=args.chunks,
                        split_k=args.split_k if args.backend == "splitk" else 1,
                        schedule=args.schedule,
                        m=m,
                        n=n,
                        k=k,
                        us=us,
                        validated=True,
                        rel_l2=max(errors),
                    )
                ),
                flush=True,
            )
        del replay
        comm.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

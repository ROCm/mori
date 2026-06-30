#!/usr/bin/env python3
"""Unified perf benchmark for the cco-LSA dispatch + combine kernels (EP8).

Mirrors mori's tests/python/ops/bench_dispatch_combine.py: run one real
dispatch to get the dynamic recv-token count, then time dispatch and combine
separately. Each is timed in BOTH modes:
  * eager  - FlyDSL precompiled callable, one direct launch per iter (real
             launch overhead, the op's hot path; no @flyc.jit dispatch tax)
  * graph  - CUDA-graph capture + replay (pure kernel, launch removed)

Bandwidth (mori parity) = total_recv * hidden * elem_size / time, for both
dispatch (P2P scatter volume) and combine (Stage-1 writeback volume).

    torchrun --standalone --nproc_per_node=8 bench_dispatch_combine.py
    MODE=eager|graph|both  SWEEP=128,512,2048  HIDDEN=7168 TOPK=8 EPR=32
"""
import os
import sys

import numpy as np
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from mori.cco import Communicator
import mori.cco.device.flydsl as cco  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "examples", "cco", "python"))
from cco_example_common import set_device, sync, zero  # noqa: E402
from symm_arena import SymmArena  # noqa: E402
from dispatch_kernel import make_dispatch  # noqa: E402
from combine_kernel import make_combine  # noqa: E402
from dist_common import Dist  # noqa: E402

HIDDEN = int(os.environ.get("HIDDEN", 7168))
K = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 32))
# dispatch (remote writes) saturates with fewer blocks; combine (remote reads)
# is latency-bound and needs more warps -> use separate block counts.
DISP_BLOCK = int(os.environ.get("DISP_BLOCK", os.environ.get("BLOCK_NUM", 64)))
COMB_BLOCK = int(os.environ.get("COMB_BLOCK", os.environ.get("BLOCK_NUM", 128)))
WARP_NUM = int(os.environ.get("WARP_NUM", 16))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
MODE = os.environ.get("MODE", "both")            # eager | graph | both
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512,2048").split(",")]


def main():
    d = Dist()
    rank, npes = d.rank, d.world
    set_device(d.local_rank)
    dev = torch.device("cuda", d.local_rank)
    num_experts = npes * EPR
    max_tok = max(SWEEP)
    M = max_tok
    max_recv = npes * M

    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    inp = torch.randn(max_tok, HIDDEN, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev)
    idx = torch.randint(0, num_experts, (max_tok, K), generator=g, dtype=torch.int32).to(dev)
    wts = torch.rand(max_tok, K, generator=g, dtype=torch.float32).to(dev)
    tok_map = torch.full((max_tok * K,), -1, dtype=torch.int32, device=dev)
    dest_pe_ctr = torch.zeros(npes, dtype=torch.int32, device=dev)
    disp_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    total_recv = torch.zeros(1, dtype=torch.int32, device=dev)
    comb_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    xdb_flag = torch.ones(1, dtype=torch.int64, device=dev)
    comb_out = torch.zeros(max_tok * HIDDEN, dtype=torch.int16, device=dev)
    comb_out_wts = torch.zeros(max_tok * K, dtype=torch.float32, device=dev)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)

    win_bytes = max_recv * HIDDEN * 2 + npes * M * HIDDEN * 2 + (1 << 24)
    with Communicator.init(npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)) as comm:
        regions = [("tok_off", 4), ("recv_num", npes * 4), ("tis", max_recv * 4),
                   ("out_idx", max_recv * K * 4), ("out_wts", max_recv * K * 4),
                   ("out_tok", max_recv * HIDDEN * 2), ("xdb_mem", npes * 8)]
        arena = SymmArena(comm, regions)
        zero(arena.local_ptr("tok_off"), arena.total_bytes)

        dispatch = make_dispatch(
            rank=rank, npes=npes, experts_per_rank=EPR, experts_per_token=K,
            hidden_dim=HIDDEN, hidden_elem_size=2, max_tok_per_rank=M, max_recv=max_recv,
            block_num=DISP_BLOCK, warp_num_per_block=WARP_NUM,
            off_tok_off=arena.offset("tok_off"), off_recv_num=arena.offset("recv_num"),
            off_tis=arena.offset("tis"), off_out_idx=arena.offset("out_idx"),
            off_out_wts=arena.offset("out_wts"), off_out_tok=arena.offset("out_tok"))
        combine = make_combine(
            rank=rank, npes=npes, experts_per_token=K, hidden_dim=HIDDEN, hidden_elem_size=2,
            max_tok_per_rank=M, max_recv=max_recv, block_num=COMB_BLOCK, warp_num_per_block=WARP_NUM,
            off_out_tok=arena.offset("out_tok"), off_xdb_mem=arena.offset("xdb_mem"),
            off_out_wts=arena.offset("out_wts"), reset_total_recv=False,
            _s3_cache=int(os.environ.get("S3_CACHE", 2)),
            _unroll=int(os.environ.get("UNROLL", 2)))

        cur = torch.cuda.current_stream()
        dptrs = (arena.handle, inp.data_ptr(), idx.data_ptr(), wts.data_ptr(), tok_map.data_ptr(),
                 dest_pe_ctr.data_ptr(), disp_bar.data_ptr(), total_recv.data_ptr())
        cptrs = (arena.handle, tok_map.data_ptr(), comb_bar.data_ptr(),
                 xdb_flag.data_ptr(), total_recv.data_ptr(), comb_out.data_ptr(),
                 comb_out_wts.data_ptr())
        disp_c = flyc.compile(dispatch, *[fx.Int64(p) for p in dptrs], rank, max_tok, fx.Stream(cur))
        comb_c = flyc.compile(combine, *[fx.Int64(p) for p in cptrs], rank, max_tok, fx.Stream(cur))

        # Launch on the CURRENT stream each call: under torch.cuda.graph capture
        # that resolves to the capture stream, so the kernel is actually recorded.
        def run_disp(ct):
            disp_c(*dptrs, rank, ct, fx.Stream(torch.cuda.current_stream()))

        def run_comb(ct):
            comb_c(*cptrs, rank, ct, fx.Stream(torch.cuda.current_stream()))

        def time_eager(fn, ct):
            for _ in range(WARMUP):
                fn(ct)
            sync(); comm.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS):
                fn(ct)
            e.record(); torch.cuda.synchronize(); comm.barrier()
            return s.elapsed_time(e) / ITERS * 1000

        def time_graph(fn, ct):
            for _ in range(WARMUP):
                fn(ct)
            sync()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn(ct)
            torch.cuda.synchronize(); comm.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS):
                gr.replay()
            e.record(); torch.cuda.synchronize(); comm.barrier()
            return s.elapsed_time(e) / ITERS * 1000

        def bw(nbytes, us):
            return nbytes / (1000 ** 3) / (us / 1e6)

        def verify(ct):
            # End-to-end correctness: identity expert (out_tok IS the dispatched
            # token), so combine[t] == U[t] * input[t], where U[t] = #unique dest
            # PEs of local token t. This transitively checks dispatch routing +
            # payload AND combine gather/accumulate. U is local (no allgather).
            total_recv.zero_(); sync()
            run_disp(ct); sync(); comm.barrier()
            comb_out.zero_(); sync()
            run_comb(ct); sync(); comm.barrier()
            idx_c = idx[:ct].cpu().numpy()
            U = np.array([len({int(idx_c[t, j]) // EPR for j in range(K)}) for t in range(ct)])
            exp = (torch.from_numpy(U).view(ct, 1).float() * inp[:ct].float().cpu()).to(torch.bfloat16)
            got = comb_out[:ct * HIDDEN].cpu().view(torch.bfloat16).view(ct, HIDDEN)
            ok = torch.allclose(got, exp, atol=2e-2, rtol=2e-2)
            # weights: out_weights[t][e] == U[t] * wts[t][e] (sum over valid experts
            # of the identical forwarded weight vector; #valid == U[t]).
            exp_w = torch.from_numpy(U).view(ct, 1).float() * wts[:ct].float().cpu()
            got_w = comb_out_wts[:ct * K].cpu().view(ct, K)
            ok_w = torch.allclose(got_w, exp_w, atol=2e-3, rtol=2e-3)
            errs = d.allreduce_sum(0 if (ok and ok_w) else 1)
            if rank == 0:
                print(f"# correctness ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                      f"(hidden={'ok' if ok else 'BAD'} wts={'ok' if ok_w else 'BAD'}; "
                      f"U in [{U.min()},{U.max()}], {ct} tok/rank, identity expert)", flush=True)
            return errs == 0

        eager = MODE in ("eager", "both")
        graph = MODE in ("graph", "both")
        if rank == 0:
            print(f"# EP{npes} hidden={HIDDEN} topk={K} experts={num_experts} "
                  f"block disp={DISP_BLOCK} comb={COMB_BLOCK} x{WARP_NUM}w  iters={ITERS}", flush=True)
        verify(min(SWEEP))   # correctness pass during warmup

        for ct in SWEEP:
            # clean dispatch to set total_recv (combine reads it; not reset)
            total_recv.zero_(); sync()
            run_disp(ct); sync(); comm.barrier()
            recv = int(total_recv.cpu().item())
            payload = recv * HIDDEN * 2

            # combine first (needs total_recv == recv; combine doesn't reset it)
            cb_e = time_eager(run_comb, ct) if eager else 0.0
            cb_g = time_graph(run_comb, ct) if graph else 0.0
            # dispatch next (accumulates total_recv, but combine already timed)
            dp_e = time_eager(run_disp, ct) if eager else 0.0
            dp_g = time_graph(run_disp, ct) if graph else 0.0

            if rank == 0:
                parts = [f"tok/rank {ct:5d}  recv {recv:6d}  payload {payload/1e6:7.2f}MB"]
                if eager:
                    parts.append(f"| EAGER disp {dp_e:8.2f}us/{bw(payload,dp_e):6.1f}GB/s "
                                 f"comb {cb_e:8.2f}us/{bw(payload,cb_e):6.1f}GB/s")
                if graph:
                    parts.append(f"| GRAPH disp {dp_g:8.2f}us/{bw(payload,dp_g):6.1f}GB/s "
                                 f"comb {cb_g:8.2f}us/{bw(payload,cb_g):6.1f}GB/s")
                print("  ".join(parts), flush=True)
    d.shutdown()


if __name__ == "__main__":
    main()

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
from mori.tensor_utils import from_gpu_ptr
import mori.cco.device.flydsl as cco  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                     # tests/ (dist_common)
sys.path.insert(0, os.path.dirname(_HERE))    # parent v2/ (op + kernels)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "examples", "cco", "python"))
from cco_example_common import set_device, sync, zero  # noqa: E402
from dispatch_combine_op import SymmArena  # noqa: E402
from dispatch_kernel import make_dispatch  # noqa: E402
from combine_kernel import make_combine, make_combine_scatter  # noqa: E402
from stdmoe_kernel import (  # noqa: E402
    make_convert_dispatch_output,
    make_convert_combine_input,
)
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
STDMOE = int(os.environ.get("STDMOE", 0))        # 1 = run StdMoE convert pipeline
DTYPE = os.environ.get("DTYPE", "bf16")          # bf16 | f32
COMBINE = os.environ.get("COMBINE", "gather")    # gather | scatter
QUANT = os.environ.get("QUANT", "none")          # none | fp8_direct_cast (scatter only)
SCALE_DIM = int(os.environ.get("SCALE_DIM", 0))  # >0 = forward per-token scales
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512,2048").split(",")]

# (torch token dtype, elem bytes, comb_out storage dtype matching elem bytes)
_DT = {"bf16": (torch.bfloat16, 2, torch.int16), "f32": (torch.float32, 4, torch.int32)}
TOK_DT, ESZ, COMB_DT = _DT[DTYPE]


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
    inp = torch.randn(max_tok, HIDDEN, generator=g, dtype=torch.float32).to(TOK_DT).to(dev)
    idx = torch.randint(0, num_experts, (max_tok, K), generator=g, dtype=torch.int32).to(dev)
    wts = torch.rand(max_tok, K, generator=g, dtype=torch.float32).to(dev)
    tok_map = torch.full((max_tok * K,), -1, dtype=torch.int32, device=dev)
    dest_pe_ctr = torch.zeros(npes, dtype=torch.int32, device=dev)
    disp_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    total_recv = torch.zeros(1, dtype=torch.int32, device=dev)
    comb_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    xdb_flag = torch.ones(1, dtype=torch.int64, device=dev)
    comb_out = torch.zeros(max_tok * HIDDEN, dtype=COMB_DT, device=dev)
    comb_out_wts = torch.zeros(max_tok * K, dtype=torch.float32, device=dev)
    # per-token scales (int8 bytes): pattern = (rank*100003 + tok) per dword so
    # the recv side can verify the bijection (origin decoded from tis).
    _sc_n_i32 = (SCALE_DIM + 3) // 4
    if SCALE_DIM:
        scales = torch.empty(max_tok * _sc_n_i32, dtype=torch.int32, device=dev)
        scales.copy_(torch.arange(max_tok, device=dev).view(max_tok, 1).expand(max_tok, _sc_n_i32)
                     .contiguous().view(-1) + rank * 100003)
    else:
        scales = torch.zeros(1, dtype=torch.int32, device=dev)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)

    win_bytes = max_recv * HIDDEN * ESZ + npes * M * HIDDEN * ESZ + (1 << 24)
    with Communicator.init(npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)) as comm:
        _fp8 = QUANT == "fp8_direct_cast"
        _bw = QUANT == "fp8_blockwise"
        _wire_esz = 1 if (_fp8 or _bw) else ESZ
        bw_scale_dim = HIDDEN // 128 if _bw else 0    # block_elems = 128
        if _bw:
            inp.mul_(float(os.environ.get("BW_INSCALE", 200)))  # >448 exercises scaling
        regions = [("tok_off", 4), ("recv_num", npes * 4), ("tis", max_recv * 4),
                   ("out_idx", max_recv * K * 4), ("out_wts", max_recv * K * 4),
                   ("out_tok", max_recv * HIDDEN * ESZ), ("xdb_mem", npes * 8)]
        if COMBINE == "scatter":
            regions.append(("comb_inp", npes * M * HIDDEN * _wire_esz))
            regions.append(("comb_wts", npes * M * K * 4))
            if _bw:
                regions.append(("comb_scales", npes * M * bw_scale_dim * 4))
        if SCALE_DIM:
            regions.append(("out_scales", max_recv * _sc_n_i32 * 4))
        arena = SymmArena(comm, regions)
        zero(arena.local_ptr("tok_off"), arena.total_bytes)

        dispatch = make_dispatch(
            rank=rank, npes=npes, experts_per_rank=EPR, experts_per_token=K,
            hidden_dim=HIDDEN, hidden_elem_size=ESZ, max_tok_per_rank=M, max_recv=max_recv,
            block_num=DISP_BLOCK, warp_num_per_block=WARP_NUM,
            off_tok_off=arena.offset("tok_off"), off_recv_num=arena.offset("recv_num"),
            off_tis=arena.offset("tis"), off_out_idx=arena.offset("out_idx"),
            off_out_wts=arena.offset("out_wts"), off_out_tok=arena.offset("out_tok"),
            off_out_scales=arena.offset("out_scales") if SCALE_DIM else 0,
            scale_dim=SCALE_DIM, scale_type_size=1)
        if COMBINE == "scatter":
            combine = make_combine_scatter(
                rank=rank, npes=npes, experts_per_token=K, hidden_dim=HIDDEN, hidden_elem_size=ESZ,
                max_tok_per_rank=M, max_recv=max_recv, block_num=COMB_BLOCK,
                warp_num_per_block=WARP_NUM, off_out_tok=arena.offset("out_tok"),
                off_comb_inp=arena.offset("comb_inp"), off_tis=arena.offset("tis"),
                off_xdb_mem=arena.offset("xdb_mem"), off_out_wts=arena.offset("out_wts"),
                off_comb_wts=arena.offset("comb_wts"), enable_weights=True,
                fp8_direct_cast=_fp8, fp8_blockwise=_bw, scale_dim=bw_scale_dim,
                off_comb_scales=arena.offset("comb_scales") if _bw else 0,
                reset_total_recv=False, _s3_cache=int(os.environ.get("S3_CACHE", 2)))
        else:
            combine = make_combine(
                rank=rank, npes=npes, experts_per_token=K, hidden_dim=HIDDEN, hidden_elem_size=ESZ,
                max_tok_per_rank=M, max_recv=max_recv, block_num=COMB_BLOCK,
                warp_num_per_block=WARP_NUM, off_out_tok=arena.offset("out_tok"),
                off_xdb_mem=arena.offset("xdb_mem"), off_out_wts=arena.offset("out_wts"),
                reset_total_recv=False, _s3_cache=int(os.environ.get("S3_CACHE", 2)),
                _unroll=int(os.environ.get("UNROLL", 2)))

        cur = torch.cuda.current_stream()
        dptrs = (arena.handle, inp.data_ptr(), idx.data_ptr(), wts.data_ptr(), tok_map.data_ptr(),
                 dest_pe_ctr.data_ptr(), disp_bar.data_ptr(), total_recv.data_ptr(),
                 scales.data_ptr())
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
            exp = (torch.from_numpy(U).view(ct, 1).float() * inp[:ct].float().cpu()).to(TOK_DT)
            # fp8 wire dtype is lossy (e4m3 ~6-12% rel) -> loose tolerance.
            _atol, _rtol = (1.0, 1.5e-1) if (_fp8 or _bw) else (2e-2, 2e-2)
            got_dt = torch.bfloat16 if (_fp8 or _bw) else TOK_DT   # fp8 paths output bf16
            got = comb_out[:ct * HIDDEN].cpu().view(got_dt).view(ct, HIDDEN)
            ok = torch.allclose(got.float(), exp.float(), atol=_atol, rtol=_rtol)
            # weights: out_weights[t][e] == U[t] * wts[t][e] (gather and scatter
            # both reduce the U forwarded copies of the identical weight vector).
            exp_w = torch.from_numpy(U).view(ct, 1).float() * wts[:ct].float().cpu()
            got_w = comb_out_wts[:ct * K].cpu().view(ct, K)
            ok_w = torch.allclose(got_w, exp_w, atol=2e-3, rtol=2e-3)
            errs = d.allreduce_sum(0 if (ok and ok_w) else 1)
            if rank == 0:
                print(f"# correctness ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                      f"(hidden={'ok' if ok else 'BAD'} wts={'ok' if ok_w else 'BAD'}; "
                      f"U in [{U.min()},{U.max()}], {ct} tok/rank, identity expert)", flush=True)
            return errs == 0

        if STDMOE:
            assert DTYPE == "bf16", "StdMoE convert kernels are bf16-only for now"
            # Full StdMoE pipeline: dispatch -> ConvertDispatchOutput ->
            # (identity expert GEMM) -> ConvertCombineInput -> combine.
            # Identity expert => packed_x[slot] == dispatched token, so the
            # per-rank weighted reduce + cross-rank gather telescopes to
            #   comb_out[s] == (sum_k wts[s][k]) * input[s]
            # (independent of routing/dedup; the weights collapse the U-count).
            mtpe = npes * M
            packed_x = torch.zeros(EPR * mtpe * HIDDEN, dtype=torch.int16, device=dev)
            packed_cnt = torch.zeros(EPR, dtype=torch.int32, device=dev)
            packed_src = torch.zeros(EPR * mtpe, dtype=torch.int32, device=dev)
            slot_map = torch.full((max_recv * K,), -1, dtype=torch.int64, device=dev)
            cvt_disp = make_convert_dispatch_output(
                rank=rank, experts_per_rank=EPR, experts_per_token=K, hidden_dim=HIDDEN,
                hidden_elem_size=2, max_tok_per_expert=mtpe, block_num=DISP_BLOCK,
                warp_num_per_block=WARP_NUM)
            cvt_comb = make_convert_combine_input(
                rank=rank, experts_per_rank=EPR, experts_per_token=K, hidden_dim=HIDDEN,
                hidden_elem_size=2, max_tok_per_expert=mtpe, block_num=COMB_BLOCK,
                warp_num_per_block=WARP_NUM)
            cdptrs = (arena.local_ptr("out_tok"), arena.local_ptr("out_idx"),
                      arena.local_ptr("tis"), total_recv.data_ptr(), packed_x.data_ptr(),
                      packed_cnt.data_ptr(), packed_src.data_ptr(), slot_map.data_ptr())
            ccptrs = (arena.local_ptr("out_tok"), arena.local_ptr("out_wts"),
                      total_recv.data_ptr(), packed_x.data_ptr(), slot_map.data_ptr())
            # flyc.compile LAUNCHES the kernel once to specialize; zero
            # total_recv first so that compile-launch is a no-op (loops 0 iters)
            # and does not scribble into out_tok / packed_x.
            total_recv.zero_(); sync()
            cdisp_c = flyc.compile(cvt_disp, *[fx.Int64(p) for p in cdptrs], fx.Stream(cur))
            ccomb_c = flyc.compile(cvt_comb, *[fx.Int64(p) for p in ccptrs], fx.Stream(cur))

            def run_cdisp():
                cdisp_c(*cdptrs, fx.Stream(torch.cuda.current_stream()))

            def run_ccomb():
                ccomb_c(*ccptrs, fx.Stream(torch.cuda.current_stream()))

            if rank == 0:
                print(f"# EP{npes} STDMOE hidden={HIDDEN} topk={K} experts={num_experts}",
                      flush=True)
            for ct in SWEEP:
                total_recv.zero_(); packed_cnt.zero_(); slot_map.fill_(-1); sync()
                run_disp(ct); sync(); comm.barrier()
                run_cdisp(); sync(); comm.barrier()    # identity GEMM: packed_x = token
                run_ccomb(); sync(); comm.barrier()
                comb_out.zero_(); sync()
                run_comb(ct); sync(); comm.barrier()
                # identity expert telescopes to comb_out[s] == (sum_k wts[s][k]) * input[s]
                ws = wts[:ct].float().cpu().sum(dim=1, keepdim=True)             # (ct,1)
                exp = (ws * inp[:ct].float().cpu()).to(TOK_DT)
                got = comb_out[:ct * HIDDEN].cpu().view(TOK_DT).view(ct, HIDDEN)
                bad = ~torch.isclose(got, exp, atol=5e-2, rtol=5e-2)
                nbad = int(bad.sum().item())
                errs = d.allreduce_sum(nbad)
                if rank == 0:
                    print(f"# STDMOE ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                          f"(sum-weighted identity, total_bad={errs})", flush=True)
            d.shutdown()
            return

        if SCALE_DIM:
            # Verify per-token scales forwarding: each recv slot's scale block
            # must equal its origin token's pattern (src_pe*100003 + src_lid),
            # origin decoded from tis. Mirrors mori's dispatch scale copy.
            total_recv.zero_(); sync()
            run_disp(min(SWEEP)); sync(); comm.barrier()
            recv = int(total_recv.cpu().item())
            out_sc = from_gpu_ptr(arena.local_ptr("out_scales"),
                                  (max_recv, _sc_n_i32), torch.int32)[:recv].cpu()
            tis = from_gpu_ptr(arena.local_ptr("tis"), (max_recv,), torch.int32)[:recv].cpu()
            exp = ((tis // M) * 100003 + (tis % M)).view(recv, 1)
            ok = bool((out_sc == exp).all().item()) if recv > 0 else True
            errs = d.allreduce_sum(0 if ok else 1)
            if rank == 0:
                print(f"# SCALES: {'PASS' if errs == 0 else 'FAIL'} "
                      f"(recv={recv}, scale_dim={SCALE_DIM}, {_sc_n_i32} dwords/tok)", flush=True)
            d.shutdown()
            return

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
            payload = recv * HIDDEN * ESZ

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

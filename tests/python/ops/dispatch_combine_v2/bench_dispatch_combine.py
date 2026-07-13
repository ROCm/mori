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
import torch.distributed as dist

import flydsl.expr as fx
from mori.cco import Communicator
from mori.tensor_utils import from_gpu_ptr
import mori.cco.device.flydsl as cco  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(
    0, os.path.join(_ROOT, "python", "mori", "ops", "dispatch_combine_v2")
)  # op + kernels
sys.path.insert(
    0, os.path.join(_ROOT, "examples", "cco", "python")
)  # cco_example_common
from cco_example_common import set_device, sync  # noqa: E402
from dispatch_combine_op import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)


class Dist:
    """Minimal torchrun/gloo bootstrap: RANK/WORLD_SIZE/LOCAL_RANK, carry the cco
    unique-id (broadcast) and a test-only int allreduce. gloo (CPU) is just the
    courier for the uid + pass/fail counts; cco does the GPU comm."""

    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(self.local_rank)

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


HIDDEN = int(os.environ.get("HIDDEN", 7168))
K = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 32))
# dispatch and combine want different block/warp counts -> separate knobs.
DISP_BLOCK = int(os.environ.get("DISP_BLOCK", os.environ.get("BLOCK_NUM", 64)))
COMB_BLOCK = int(os.environ.get("COMB_BLOCK", os.environ.get("BLOCK_NUM", 80)))
WARP_NUM = int(os.environ.get("WARP_NUM", 16))
# combine's K-deep per-lane MLP saturates with few warps.
COMB_WARP = int(os.environ.get("COMB_WARP", WARP_NUM))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
MODE = os.environ.get("MODE", "both")  # eager | graph | both
STDMOE = int(os.environ.get("STDMOE", 0))  # 1 = run StdMoE convert pipeline
DTYPE = os.environ.get("DTYPE", "bf16")  # bf16 | f32
COMBINE = os.environ.get("COMBINE", "gather")  # gather | scatter
QUANT = os.environ.get("QUANT", "none")  # none | fp8_direct_cast (scatter only)
SCALE_DIM = int(os.environ.get("SCALE_DIM", 0))  # >0 = forward per-token scales
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512,2048").split(",")]

# fp8 flavor is arch-specific: OCP e4m3 on gfx950/gfx1250, fnuz on gfx942.
import tuning_configs as _tc  # noqa: E402

_FP8_DT = (
    torch.float8_e4m3fn
    if _tc._topology()[1] in (90500, 120500)
    else torch.float8_e4m3fnuz
)
# (torch token dtype, elem bytes). fp4 packs 2 e2m1/byte -> 0.5 B/elem;
# per-token bytes handled via TOK_NB below.
_DT = {
    "bf16": (torch.bfloat16, 2),
    "f32": (torch.float32, 4),
    "fp8": (_FP8_DT, 1),
    "fp4": (torch.float4_e2m1fn_x2, 1),
}
TOK_DT, ESZ = _DT[DTYPE]
_FP4 = DTYPE == "fp4"
TOK_NB = HIDDEN // 2 if _FP4 else HIDDEN * ESZ  # per-token transport bytes

# Asymmetric I/O: dispatch fp8 + combine bf16 (an expert op converts between the
# two). DISPATCH_DT/COMBINE_DT override DTYPE per-op; unset => symmetric (DTYPE).
DISPATCH_DT = os.environ.get("DISPATCH_DT")
COMBINE_DT = os.environ.get("COMBINE_DT")
_ASYM = (DISPATCH_DT is not None) or (COMBINE_DT is not None)
_disp_s, _comb_s = DISPATCH_DT or DTYPE, COMBINE_DT or DTYPE
DISP_DT, DISP_ESZ = _DT[_disp_s]
COMB_DT, COMB_ESZ = _DT[_comb_s]
DISP_NB = HIDDEN // 2 if _disp_s == "fp4" else HIDDEN * DISP_ESZ  # dispatch bytes/tok
COMB_NB = HIDDEN // 2 if _comb_s == "fp4" else HIDDEN * COMB_ESZ  # combine bytes/tok


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
    if (
        _FP4
    ):  # fp4 has no float cast path; make packed uint8 (2 e2m1/byte) and reinterpret
        inp = (
            torch.randint(
                0, 256, (max_tok, HIDDEN // 2), generator=g, dtype=torch.uint8
            )
            .view(torch.float4_e2m1fn_x2)
            .to(dev)
        )
    else:
        inp = (
            torch.randn(max_tok, HIDDEN, generator=g, dtype=torch.float32)
            .to(DISP_DT)
            .to(dev)
        )
    idx = torch.randint(
        0, num_experts, (max_tok, K), generator=g, dtype=torch.int32
    ).to(dev)
    wts = torch.rand(max_tok, K, generator=g, dtype=torch.float32).to(dev)
    # per-token scales (int8 bytes): pattern = (rank*100003 + tok) per dword so
    # the recv side can verify the bijection (origin decoded from recv_to_src_token).
    _sc_n_i32 = (SCALE_DIM + 3) // 4
    if SCALE_DIM:
        scales = torch.empty(max_tok * _sc_n_i32, dtype=torch.int32, device=dev)
        scales.copy_(
            torch.arange(max_tok, device=dev)
            .view(max_tok, 1)
            .expand(max_tok, _sc_n_i32)
            .contiguous()
            .view(-1)
            + rank * 100003
        )
    else:
        scales = torch.zeros(1, dtype=torch.int32, device=dev)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)

    _max_nb = max(DISP_NB, COMB_NB)  # out_tok sized to the larger (bf16) side
    win_bytes = max_recv * _max_nb + npes * M * _max_nb + (1 << 24)
    with Communicator.init(
        npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)
    ) as comm:
        _fp8 = QUANT == "fp8_direct_cast"
        _bw = QUANT == "fp8_blockwise"
        if _bw:
            inp.mul_(float(os.environ.get("BW_INSCALE", 200)))  # >448 exercises scaling
        # Build kernels + arena THROUGH the op-layer (single source of truth for
        # dtype/mode support — the bench can no longer test a config the op can't
        # express). schedule=None + explicit block/warp => the op precompiles
        # exactly the (DISP_BLOCK,WARP_NUM) / (COMB_BLOCK,COMB_WARP) variants we
        # sweep; we time those, reusing the op's arena + buffers.
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=npes,
            hidden_dim=HIDDEN,
            max_num_inp_token_per_rank=M,
            num_experts_per_rank=EPR,
            num_experts_per_token=K,
            data_type=TOK_DT,
            # asymmetric dispatch/combine dtype (all-or-none); None => symmetric
            dispatch_data_type=(DISP_DT if _ASYM else None),
            combine_data_type=(COMB_DT if _ASYM else None),
            combine_mode=COMBINE,
            quant_type=QUANT,
            dispatch_block_num=DISP_BLOCK,
            warp_num_per_block=WARP_NUM,
            combine_block_num=COMB_BLOCK,
            combine_warp_num_per_block=COMB_WARP,
            schedule=None,
            scale_dim=SCALE_DIM,
            scale_type_size=1 if SCALE_DIM else 0,
            enable_std_moe=bool(STDMOE),
        )
        op = EpDispatchCombineOp(cfg, comm)
        op.reset()
        arena = op.arena
        # aliases so the correctness / perf code below reads the op's own buffers
        total_recv = op.total_recv
        comb_out = op.combine_out
        comb_out_wts = op.combine_out_weights
        tok_map = op.token_dest_map
        disp_kern = op._dispatch_variants[(DISP_BLOCK, WARP_NUM)]
        comb_kern = op._combine_variants[(COMB_BLOCK, COMB_WARP)]

        # Launch on the CURRENT stream each call: under torch.cuda.graph capture
        # that resolves to the capture stream, so the kernel is actually recorded.
        def run_disp(ct):
            disp_kern(
                arena.handle,
                inp.data_ptr(),
                idx.data_ptr(),
                wts.data_ptr(),
                tok_map.data_ptr(),
                op.dest_pe_counter.data_ptr(),
                op.dispatch_barrier.data_ptr(),
                total_recv.data_ptr(),
                scales.data_ptr(),
                rank,
                ct,
                fx.Stream(torch.cuda.current_stream()),
            )

        def run_comb(ct):
            comb_kern(
                arena.handle,
                tok_map.data_ptr(),
                op.combine_barrier.data_ptr(),
                op.cross_device_flag.data_ptr(),
                total_recv.data_ptr(),
                comb_out.data_ptr(),
                comb_out_wts.data_ptr(),
                rank,
                ct,
                fx.Stream(torch.cuda.current_stream()),
            )

        def mock_fmoe():
            # expert-op stand-in: dequant dispatched tokens to the combine dtype in
            # out_tok. Via a scratch — in-place would alias (differing strides).
            scratch = op.recv_tokens().to(COMB_DT)
            op.combine_in_view().view(-1).copy_(scratch.view(-1))

        def time_eager(fn, ct):
            for _ in range(WARMUP):
                fn(ct)
            sync()
            comm.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            s.record()
            for _ in range(ITERS):
                fn(ct)
            e.record()
            torch.cuda.synchronize()
            comm.barrier()
            return s.elapsed_time(e) / ITERS * 1000

        def time_graph(fn, ct):
            for _ in range(WARMUP):
                fn(ct)
            sync()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn(ct)
            torch.cuda.synchronize()
            comm.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            s.record()
            for _ in range(ITERS):
                gr.replay()
            e.record()
            torch.cuda.synchronize()
            comm.barrier()
            return s.elapsed_time(e) / ITERS * 1000

        def bw(nbytes, us):
            return nbytes / (1000**3) / (us / 1e6)

        def verify(ct):
            # End-to-end correctness: identity expert (out_tok IS the dispatched
            # token), so combine[t] == U[t] * input[t], where U[t] = #unique dest
            # PEs of local token t. This transitively checks dispatch routing +
            # payload AND combine gather/accumulate. U is local (no allgather).
            total_recv.zero_()
            sync()
            run_disp(ct)
            sync()
            comm.barrier()
            if _ASYM:  # expert op converts dispatch dtype -> combine dtype
                mock_fmoe()
                sync()
            comb_out.zero_()
            sync()
            run_comb(ct)
            sync()
            comm.barrier()
            if _FP4:  # fp4 combine is too lossy for a numeric check (mirror mori v1)
                if rank == 0:
                    print(
                        f"# correctness ct={ct}: SKIP (fp4 combine not checked)",
                        flush=True,
                    )
                return True
            idx_c = idx[:ct].cpu().numpy()
            U = np.array(
                [len({int(idx_c[t, j]) // EPR for j in range(K)}) for t in range(ct)]
            )
            exp = (torch.from_numpy(U).view(ct, 1).float() * inp[:ct].float().cpu()).to(
                COMB_DT
            )
            # fp8 (quant wire, plain fp8 token, or asymmetric fp8 side) is lossy.
            lossy = _fp8 or _bw or "fp8" in (DTYPE, _disp_s, _comb_s)
            _atol, _rtol = (1.0, 1.5e-1) if lossy else (2e-2, 2e-2)
            got_dt = (
                torch.bfloat16 if (_fp8 or _bw) else COMB_DT
            )  # quant paths output bf16
            got = comb_out[: ct * HIDDEN].cpu().view(got_dt).view(ct, HIDDEN)
            ok = torch.allclose(got.float(), exp.float(), atol=_atol, rtol=_rtol)
            # weights: out_weights[t][e] == U[t] * wts[t][e] (gather and scatter
            # both reduce the U forwarded copies of the identical weight vector).
            exp_w = torch.from_numpy(U).view(ct, 1).float() * wts[:ct].float().cpu()
            got_w = comb_out_wts[: ct * K].cpu().view(ct, K)
            ok_w = torch.allclose(got_w, exp_w, atol=2e-3, rtol=2e-3)
            errs = d.allreduce_sum(0 if (ok and ok_w) else 1)
            if rank == 0:
                print(
                    f"# correctness ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                    f"(hidden={'ok' if ok else 'BAD'} wts={'ok' if ok_w else 'BAD'}; "
                    f"U in [{U.min()},{U.max()}], {ct} tok/rank, identity expert)",
                    flush=True,
                )
            return errs == 0

        if STDMOE:
            assert DTYPE == "bf16", "StdMoE convert kernels are bf16-only for now"

            # Full StdMoE pipeline: dispatch -> ConvertDispatchOutput ->
            # (identity expert GEMM) -> ConvertCombineInput -> combine.
            # Identity expert => packed_x[slot] == dispatched token, so the
            # per-rank weighted reduce + cross-rank gather telescopes to
            #   comb_out[s] == (sum_k wts[s][k]) * input[s]
            # (independent of routing/dedup; the weights collapse the U-count).
            # convert kernels + packed buffers come from the op (enable_std_moe).
            def run_cdisp():
                op._convert_dispatch(
                    arena.local_ptr("out_tok"),
                    arena.local_ptr("out_idx"),
                    arena.local_ptr("recv_to_src_token"),
                    total_recv.data_ptr(),
                    op.packed_x.data_ptr(),
                    op.packed_count.data_ptr(),
                    op.packed_src.data_ptr(),
                    op.slot_map.data_ptr(),
                    fx.Stream(torch.cuda.current_stream()),
                )

            def run_ccomb():
                op._convert_combine(
                    arena.local_ptr("out_tok"),
                    arena.local_ptr("out_wts"),
                    total_recv.data_ptr(),
                    op.packed_x.data_ptr(),
                    op.slot_map.data_ptr(),
                    fx.Stream(torch.cuda.current_stream()),
                )

            if rank == 0:
                print(
                    f"# EP{npes} STDMOE hidden={HIDDEN} topk={K} experts={num_experts}",
                    flush=True,
                )
            for ct in SWEEP:
                total_recv.zero_()
                op.packed_count.zero_()
                op.slot_map.fill_(-1)
                sync()
                run_disp(ct)
                sync()
                comm.barrier()
                run_cdisp()
                sync()
                comm.barrier()  # identity GEMM: packed_x = token
                run_ccomb()
                sync()
                comm.barrier()
                comb_out.zero_()
                sync()
                run_comb(ct)
                sync()
                comm.barrier()
                # identity expert telescopes to comb_out[s] == (sum_k wts[s][k]) * input[s]
                ws = wts[:ct].float().cpu().sum(dim=1, keepdim=True)  # (ct,1)
                exp = (ws * inp[:ct].float().cpu()).to(TOK_DT)
                got = comb_out[: ct * HIDDEN].cpu().view(TOK_DT).view(ct, HIDDEN)
                bad = ~torch.isclose(got, exp, atol=5e-2, rtol=5e-2)
                nbad = int(bad.sum().item())
                errs = d.allreduce_sum(nbad)
                if rank == 0:
                    print(
                        f"# STDMOE ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                        f"(sum-weighted identity, total_bad={errs})",
                        flush=True,
                    )
            d.shutdown()
            return

        if SCALE_DIM:
            # Verify per-token scales forwarding: each recv slot's scale block
            # must equal its origin token's pattern (src_pe*100003 + src_lid),
            # origin decoded from tis. Mirrors mori's dispatch scale copy.
            total_recv.zero_()
            sync()
            run_disp(min(SWEEP))
            sync()
            comm.barrier()
            recv = int(total_recv.cpu().item())
            out_sc = from_gpu_ptr(
                arena.local_ptr("out_scales"), (max_recv, _sc_n_i32), torch.int32
            )[:recv].cpu()
            tis = from_gpu_ptr(
                arena.local_ptr("recv_to_src_token"), (max_recv,), torch.int32
            )[:recv].cpu()
            exp = ((tis // M) * 100003 + (tis % M)).view(recv, 1)
            ok = bool((out_sc == exp).all().item()) if recv > 0 else True
            errs = d.allreduce_sum(0 if ok else 1)
            if rank == 0:
                print(
                    f"# SCALES: {'PASS' if errs == 0 else 'FAIL'} "
                    f"(recv={recv}, scale_dim={SCALE_DIM}, {_sc_n_i32} dwords/tok)",
                    flush=True,
                )
            d.shutdown()
            return

        eager = MODE in ("eager", "both")
        graph = MODE in ("graph", "both")
        if rank == 0:
            print(
                f"# EP{npes} hidden={HIDDEN} topk={K} experts={num_experts} "
                f"block disp={DISP_BLOCK} comb={COMB_BLOCK} x{WARP_NUM}w  iters={ITERS}",
                flush=True,
            )
        verify(min(SWEEP))  # correctness pass during warmup

        for ct in SWEEP:
            # clean dispatch to set total_recv (combine reads it; not reset)
            total_recv.zero_()
            sync()
            run_disp(ct)
            sync()
            comm.barrier()
            recv = int(total_recv.cpu().item())
            disp_payload, comb_payload = recv * DISP_NB, recv * COMB_NB

            if _ASYM:  # expert op converts dispatch dtype -> combine dtype
                mock_fmoe()
                sync()
            # combine first (needs total_recv == recv; combine doesn't reset it)
            cb_e = time_eager(run_comb, ct) if eager else 0.0
            cb_g = time_graph(run_comb, ct) if graph else 0.0
            # dispatch next (accumulates total_recv, but combine already timed)
            dp_e = time_eager(run_disp, ct) if eager else 0.0
            dp_g = time_graph(run_disp, ct) if graph else 0.0

            if rank == 0:
                # payload shown = disp/comb bytes (same unless asymmetric dtype)
                pl = (
                    f"payload {disp_payload/1e6:.2f}/{comb_payload/1e6:.2f}MB"
                    if _ASYM
                    else f"payload {comb_payload/1e6:7.2f}MB"
                )
                parts = [f"tok/rank {ct:5d}  recv {recv:6d}  {pl}"]
                if eager:
                    parts.append(
                        f"| EAGER disp {dp_e:8.2f}us/{bw(disp_payload,dp_e):6.1f}GB/s "
                        f"comb {cb_e:8.2f}us/{bw(comb_payload,cb_e):6.1f}GB/s"
                    )
                if graph:
                    parts.append(
                        f"| GRAPH disp {dp_g:8.2f}us/{bw(disp_payload,dp_g):6.1f}GB/s "
                        f"comb {cb_g:8.2f}us/{bw(comb_payload,cb_g):6.1f}GB/s"
                    )
                print("  ".join(parts), flush=True)
    d.shutdown()


if __name__ == "__main__":
    main()

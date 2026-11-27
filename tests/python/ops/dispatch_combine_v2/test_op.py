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
"""EP8 correctness test for the EpDispatchCombineOp host op-layer.

Identity expert (recv tokens are combined unchanged), so combine[t] == U[t] *
input[t] (U = #unique dest PEs) and out_weights[t] == U[t] * wts[t].

    torchrun --standalone --nproc_per_node=8 test_op.py
"""
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from mori.cco import Communicator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(
    0, os.path.join(_ROOT, "examples", "cco", "python")
)  # cco_example_common
from cco_example_common import set_device, sync  # noqa: E402
from mori.ops.dispatch_combine_v2 import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)
from mori.ops.dispatch_combine_v2.quant_types import is_gfx125x  # noqa: E402


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

    def allreduce_max(self, value):
        t = torch.tensor([value], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def allgather(self, t):
        """Every rank's CPU tensor; shapes may differ between ranks."""
        objs = [None] * self.world
        dist.all_gather_object(objs, t.cpu())
        return objs

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


# fp8 flavor is arch-specific: OCP e4m3 on gfx950/gfx1250, fnuz on gfx942.
from mori.ops.dispatch_combine_v2 import tuning_configs as _tc  # noqa: E402

_FP8_DT = (
    torch.float8_e4m3fn
    if _tc._topology()[1] in (90500, 120500)
    else torch.float8_e4m3fnuz
)


def _configure():
    """Read the knobs below from the environment (again for every CASES entry)."""
    global HIDDEN, K, EPR, DTYPE, _IS_FP4, _IS_FP8, STDMOE, SCALE_DIM, COMBINE
    global QUANT, INSCALE, SPREAD, TINY, SWEEP, RAGGED, NOSELF, FRESH, REROUTE, NOWTS
    global DROPPED
    HIDDEN = int(os.environ.get("HIDDEN", 7168))
    K = int(os.environ.get("TOPK", 8))
    EPR = int(os.environ.get("EPR", 32))
    DTYPE = {
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "fp8": _FP8_DT,
        "fp4": torch.float4_e2m1fn_x2,
    }[os.environ.get("DTYPE", "bf16")]
    _IS_FP4 = DTYPE == torch.float4_e2m1fn_x2
    _IS_FP8 = DTYPE == _FP8_DT
    STDMOE = int(os.environ.get("STDMOE", 0))
    SCALE_DIM = int(
        os.environ.get("SCALE_DIM", 0)
    )  # >0 = also verify per-token scales forwarding
    COMBINE = os.environ.get("COMBINE", "gather")  # gather | scatter
    QUANT = os.environ.get("QUANT", "none")  # none | fp{8,4}_blockwise[_1x32|_1x128]
    # The gfx125x push combine folds no weights and refuses them (it says "call
    # combine with weights=None"), so those runs pass none and skip the weight check.
    NOWTS = QUANT != "none" and is_gfx125x()
    # Input magnitude: 200 puts a block's amax past the fp8 max, 0.001 below its
    # normal range; a quantized wire must keep both within _QUANT_REL_L2.
    INSCALE = float(os.environ.get("INSCALE", 1))
    # SPREAD=n: each 32-element group times 2^u, u uniform in [-n, n], so a scale
    # applied to the wrong group shows up as a gross error.
    SPREAD = int(os.environ.get("SPREAD", 0))
    # TINY=1: every row's first 128 elements are ~1e-37 with a zero among them, under
    # a 1x128 scale's normal range, where the scale must be 1. Without that rule
    # max / amax overflows and the zero becomes 0 * inf; the hardware clamp happens
    # to turn that NaN into a finite value, so this guards the rule, not a crash.
    TINY = int(os.environ.get("TINY", 0))
    SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512").split(",")]
    # RAGGED=1: rank r sends ct * (r % 3) // 2 tokens of each SWEEP point -- none on
    # ranks 0 and 3, half on 1, all on 2 -- so one call mixes ranks that send nothing
    # with ranks the tuned schedule puts on different geometries.
    RAGGED = int(os.environ.get("RAGGED", 0))
    # NOSELF=1: no token picks an expert on its own rank, so no rank receives from
    # itself.
    NOSELF = int(os.environ.get("NOSELF", 0))
    # FRESH=1: copy the received rows and the reverse map in-stream right after
    # dispatch, and require them to equal what the views hold after the barrier.
    FRESH = int(os.environ.get("FRESH", 0))
    # REROUTE=N: N more rounds per SWEEP point with new routing, each reading
    # local_expert_count (and StdMoE's packing) straight after dispatch.
    REROUTE = int(os.environ.get("REROUTE", 0))
    # DROPPED=1: slots dispatch must drop, ids < 0 or >= npes * EPR: padding rows
    # (all -1, or -1 and out of range), -1 next to valid slots, out of range ahead of
    # them.
    DROPPED = int(os.environ.get("DROPPED", 0))


_configure()

# Relative L2 error bound of a quantized combine, by element; rounding alone is
# ~2.6% for fp8 e4m3 and ~11% for e2m1.
_QUANT_REL_L2 = {"fp8": 0.05, "fp4": 0.15}


def _window_bytes(npes):
    return npes * max(SWEEP) * HIDDEN * _DT() * 2 + (1 << 24)


def _run(d, comm):
    """One configuration (the knobs as _configure last read them), on comm."""
    rank, npes = d.rank, d.world
    dev = torch.device("cuda", d.local_rank)
    M = max(SWEEP)
    num_experts = npes * EPR

    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    if _IS_FP4:  # fp4: packed uint8 (2 e2m1/byte), reinterpret as float4_e2m1fn_x2
        inp = (
            torch.randint(0, 256, (M, HIDDEN // 2), generator=g, dtype=torch.uint8)
            .view(torch.float4_e2m1fn_x2)
            .to(dev)
        )
    else:
        inp = torch.randn(M, HIDDEN, generator=g, dtype=torch.float32)
        if INSCALE != 1.0:
            inp = inp * INSCALE
        if SPREAD:
            groups = inp.view(M, HIDDEN // 32, 32)
            u = torch.randint(-SPREAD, SPREAD + 1, (M, HIDDEN // 32, 1), generator=g)
            inp = (groups * torch.exp2(u.float())).view(M, HIDDEN)
        if TINY:
            inp[:, :128] *= 1e-37
            inp[:, 0] = 0.0
        inp = inp.to(DTYPE).to(dev)
    if NOSELF:
        idx = torch.randint(
            0, num_experts - EPR, (M, K), generator=g, dtype=torch.int32
        )
        idx += (idx >= rank * EPR).to(torch.int32) * EPR
    else:
        idx = torch.randint(0, num_experts, (M, K), generator=g, dtype=torch.int32)
    if DROPPED:
        rows, cols = torch.arange(M), torch.arange(K)
        oob = (num_experts + cols).to(torch.int32)
        idx[rows % 4 == 3] = torch.where(cols % 2 == 0, oob, -1).to(torch.int32)
        idx[(rows % 3 == 1).nonzero(as_tuple=True)[0].unsqueeze(1), cols[1::2]] = -1
        idx[rows % 5 == 2, 0] = num_experts
        idx[0] = -1
    idx = idx.to(dev)
    wts = torch.rand(M, K, generator=g, dtype=torch.float32).to(dev)

    # Per-token scales (int8 bytes): pattern = rank*100003 + tok per dword, so the
    # recv side can verify the bijection by decoding the origin from the routing
    # handle's reverse map (disp_tok_id_to_src_tok_id_local).
    sc_n_i32 = (SCALE_DIM + 3) // 4
    scales = None
    if SCALE_DIM:
        scales = (
            torch.arange(M, device=dev).view(M, 1).expand(M, sc_n_i32).contiguous()
            + rank * 100003
        ).to(torch.int32)

    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=npes,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=K,
        data_type=DTYPE,
        enable_std_moe=bool(STDMOE),
        scale_dim=SCALE_DIM,
        scale_type_size=1 if SCALE_DIM else 0,
        combine_mode=COMBINE,
        quant_type=QUANT,
        max_total_recv_tokens=int(os.environ.get("MAXRECV", 0)),
    )
    if int(os.environ.get("TUNED", 0)):
        # Each backend tunes its own kernels, so take the table of the one that will
        # run: a cco row carries no promise about the hip dispatch's geometry, and on
        # gfx1250 it is not even launchable there (its 32-warp rows predate the hip
        # dispatch's per-warp LDS slabs -- ROCm/mori#951).
        from mori.ops.dispatch_combine_v2.dispatch_combine_op import _default_backend

        _dt = "fp4" if _IS_FP4 else ("fp8" if _IS_FP8 else "bf16")
        if (os.environ.get("MORI_V2_KERNEL_BACKEND") or _default_backend()) == "hip":
            from mori.ops.dispatch_combine_v2.hip_tuning_configs import lookup

            cfg.schedule = lookup(
                npes, HIDDEN, K, dtype=_dt, experts_per_rank=EPR, quant_type=QUANT
            )["schedule"]
        else:
            from mori.ops.dispatch_combine_v2.tuning_configs import lookup

            cfg.schedule = lookup(npes, HIDDEN, K, dtype=_dt)["schedule"]
    with EpDispatchCombineOp(cfg, comm) as op:
        comm.barrier()

        cap = cfg.effective_max_recv if cfg.max_total_recv_tokens > 0 else None
        for ct_point in SWEEP:
            # This rank's token count; `ct` keeps naming the SWEEP point in the output.
            ct = ct_point * (rank % 3) // 2 if RAGGED else ct_point
            if RAGGED and rank == 0:
                print(
                    f"# RAGGED ct={ct_point}: per-rank tokens "
                    f"{[ct_point * (r % 3) // 2 for r in range(npes)]}",
                    flush=True,
                )
            if int(os.environ.get("RESET", 0)):
                op.reset()
                sync()
                comm.barrier()
            sc_in = scales[:ct] if SCALE_DIM else None
            recv_x, disp_w, out_s, disp_i, total_recv_t, routing = op.dispatch(
                inp[:ct], wts[:ct], sc_in, idx[:ct], return_routing=True
            )
            # Read straight after dispatch, no sync or barrier, as SGLang does.
            has_lec = "local_expert_count" in getattr(op, "capabilities", frozenset())
            lec_now = op.local_expert_count().clone() if has_lec else None
            packed = None
            if STDMOE:
                px, pc, ps = op.convert_dispatch_output()
                packed = (px, pc.clone(), ps.clone())
            if FRESH:
                # Copies queued right behind the dispatch kernel, before any host sync
                # or barrier, of every row peers wrote, the reverse map included.
                views = (recv_x, disp_w, disp_i, out_s, routing._reverse_src_view)
                views = [v for v in views if v is not None]
                early = [v.clone() for v in views]
            total_recv = int(total_recv_t.cpu().item())
            sync()
            comm.barrier()
            if FRESH:
                ok_f = all(
                    torch.equal(e[:total_recv], v[:total_recv])
                    for e, v in zip(early, views)
                )
                errs_f = d.allreduce_sum(0 if ok_f else 1)
                if rank == 0:
                    print(
                        f"# OP-FRESH ct={ct_point}: {'PASS' if errs_f == 0 else 'FAIL'} "
                        f"(rows read before the barrier == after; recv={total_recv})",
                        flush=True,
                    )
            # The PEs each token reaches: from the routing alone, or under a cap from
            # the dest map (a dup or dropped slot is null; gfx125x never caps, and its
            # LL128 map is not flat-encoded).
            idx_cpu = idx[:ct].cpu()
            valid = (idx_cpu >= 0) & (idx_cpu < npes * EPR)
            pes = torch.where(valid, idx_cpu // EPR, -1)
            reach = [{int(p) for p in row if p >= 0} for row in pes]
            stride = cfg.effective_max_recv
            if cap is not None:
                dmap = routing.disp_dest_tok_id_map[: ct * K].cpu().view(ct, K)
                kept = dmap < npes * stride
                reach = [
                    {int(f) // stride for f in row[k]} for row, k in zip(dmap, kept)
                ]
            # each reached row carries every k whose expert lives on its PE
            lec_mine = sum(
                sum(1 for p in pes[t].tolist() if p >= 0 and p in reach[t])
                for t in range(ct)
            )
            # Exact per-expert reference from every rank's routing (from the rows
            # kept, under a cap); the global sum counts the (token,k) assignments
            # whose PE kept the token. Capability-gated: a backend without it says so.
            if cap is None:
                lec_ref = _lec_ref(d, idx, ct_point)
            else:
                lec_ref = _bincount_local(disp_i[:total_recv].cpu(), rank)
            if has_lec:
                lec_late = op.local_expert_count().cpu()
                ok_lec = torch.equal(lec_now.cpu(), lec_ref) and torch.equal(
                    lec_late, lec_ref
                )
                lec_sum = int(lec_late.sum())
            else:
                ok_lec, lec_sum = True, 0
            sync()
            comm.barrier()
            lec_errs = d.allreduce_sum(0 if ok_lec else 1)
            lec_total = d.allreduce_sum(lec_sum)
            lec_exp = d.allreduce_sum(lec_mine)
            if rank == 0 and has_lec:
                ok_lec = lec_total == lec_exp and lec_errs == 0
                print(
                    f"# OP-LEC ct={ct_point}: {'PASS' if ok_lec else 'FAIL'} "
                    f"(sum={lec_total} exp={lec_exp}, per-expert mismatches on "
                    f"{lec_errs} ranks)",
                    flush=True,
                )
            if int(os.environ.get("REPLAY", 0)):
                # Wipe what replay must rewrite, so a replay that writes nothing fails.
                views = [v for v in (recv_x, disp_w, disp_i, out_s) if v is not None]
                refs = [v[:total_recv].clone() for v in views]
                for v in views:
                    v[:total_recv].zero_()
                sync()
                comm.barrier()
                r2 = op.dispatch(inp[:ct], wts[:ct], sc_in, idx[:ct], routing=routing)
                sync()
                comm.barrier()
                outs = [v for v in (r2[0], r2[1], r2[3], r2[2]) if v is not None]
                ok_r = int(r2[4].cpu().item()) == total_recv and all(
                    torch.equal(o[:total_recv], ref) for o, ref in zip(outs, refs)
                )
                errs_r = d.allreduce_sum(0 if ok_r else 1)
                if rank == 0:
                    print(
                        f"# OP-REPLAY ct={ct_point}: {'PASS' if errs_r == 0 else 'FAIL'} "
                        f"(rows, weights, ids and scales rewritten as dispatched)",
                        flush=True,
                    )
            if SCALE_DIM:
                # Verify per-token scales forwarding via the routing handle's
                # reverse map: recv slot s came from token disp_tok_id_to_src_tok_id
                # -> expected dword = (src_rank)*100003 + (src_tok).
                tis = routing.disp_tok_id_to_src_tok_id_local[:total_recv].cpu()
                exp = ((tis // M) * 100003 + (tis % M)).view(total_recv, 1)
                got = out_s[:total_recv].cpu()
                ok_sc = bool((got == exp.expand(total_recv, sc_n_i32)).all())
                errs = d.allreduce_sum(0 if ok_sc else 1)
                if rank == 0:
                    print(
                        f"# OP-SCALES ct={ct_point}: {'PASS' if errs == 0 else 'FAIL'} "
                        f"(recv={total_recv}, scale_dim={SCALE_DIM}, "
                        f"{sc_n_i32} dwords/tok)",
                        flush=True,
                    )
            if not STDMOE:
                # The two metadata arrays dispatch forwards next to the payload.
                # Unlike the scales, their contents cannot be derived from the recv
                # slot, so every rank's inputs get gathered to know what each slot
                # should hold. Checking them here rather than through combine is
                # the point: combine only folds the weights, and reads no indices
                # at all, so a kernel can corrupt both while payload and combine
                # still pass.
                # Whole arrays rather than [:ct]: under RAGGED the ranks' counts
                # differ, and the reverse map only indexes rows a rank really sent.
                srcs_i = [torch.empty_like(idx) for _ in range(npes)]
                srcs_w = [torch.empty_like(wts) for _ in range(npes)]
                dist.all_gather(srcs_i, idx.contiguous())
                dist.all_gather(srcs_w, wts.contiguous())
                all_i = torch.cat([s.cpu() for s in srcs_i])
                all_w = torch.cat([s.cpu() for s in srcs_w])
                tim = routing.disp_tok_id_to_src_tok_id_local[:total_recv].cpu().long()
                # The reverse map indexes the expected values, here and in the scale
                # check above, so it is range-checked before being used as an index.
                # An out-of-range entry is a failure in its own right, and indexing
                # with it raises IndexError, which would kill this rank and leave the
                # others in all_reduce -- a product bug reported as a harness crash.
                ok_map = bool(((tim >= 0) & (tim < npes * M)).all())
                if ok_map:
                    ok_i = torch.equal(disp_i[:total_recv].cpu(), all_i[tim])
                    ok_dw = torch.allclose(
                        disp_w[:total_recv].cpu(), all_w[tim], atol=2e-3, rtol=2e-3
                    )
                else:
                    ok_i = ok_dw = False
                errs = d.allreduce_sum(0 if (ok_map and ok_i and ok_dw) else 1)
                if rank == 0:
                    print(
                        f"# OP-META ct={ct_point}: {'PASS' if errs == 0 else 'FAIL'} "
                        f"(recv={total_recv}, topk={K}, map="
                        f"{'ok' if ok_map else 'BAD'}, idx={'ok' if ok_i else 'BAD'}, "
                        f"wts={'ok' if ok_dw else 'BAD'})",
                        flush=True,
                    )
            # Every rank receives min(demand, capacity): all of it without a cap.
            demand = sum(
                sum(
                    1
                    for row in other.tolist()
                    if rank in {e // EPR for e in row if e >= 0}
                )
                for other in d.allgather(idx_cpu)
            )
            ok_recv = total_recv == min(demand, stride)
            if cap is None:
                errs = d.allreduce_sum(0 if ok_recv else 1)
                if rank == 0:
                    print(
                        f"# OP-RECV ct={ct_point}: {'PASS' if errs == 0 else 'FAIL'} "
                        f"(recv={total_recv} == demand)",
                        flush=True,
                    )
            else:
                # Drops depend on atomic order, so check invariants, not which tokens.
                maps, revs = d.allgather(dmap), d.allgather(
                    routing.disp_tok_id_to_src_tok_id_local[:total_recv].cpu()
                )
                ok = ok_recv
                for src, m in enumerate(maps):
                    t, k = (m < npes * stride).nonzero(as_tuple=True)
                    pe, slot = m[t, k] // stride, m[t, k] % stride
                    for p in range(npes):
                        sel = pe == p
                        if not bool(sel.any()):
                            continue
                        s_ = slot[sel].long()
                        ok &= bool((s_ < len(revs[p])).all())
                        if ok:
                            got = revs[p][s_]
                            ok &= torch.equal(got, (src * M + t[sel]).to(got.dtype))
                    ok &= bool((pe < npes).all())
                sent = d.allreduce_sum(int(kept.sum()))
                ok &= d.allreduce_sum(total_recv) == sent
                errs = d.allreduce_sum(0 if ok else 1)
                if rank == 0:
                    print(
                        f"# OP-CAP ct={ct_point}: {'PASS' if errs == 0 else 'FAIL'} "
                        f"(recv={total_recv} = min(demand, cap={cap}); kept slots "
                        f"{sent} match the recv sum and the reverse maps)",
                        flush=True,
                    )
            rel = 0.0  # relative L2 error of a quantized combine
            if STDMOE:
                # Identity GEMM: packed_x keeps the tokens convert_dispatch_output put there.
                ok_pk = _check_packed(
                    packed, lec_ref, routing, recv_x, disp_i, total_recv
                )
                errs = d.allreduce_sum(0 if ok_pk else 1)
                if rank == 0:
                    print(
                        f"# OP-STDMOE-PACK ct={ct_point}: "
                        f"{'PASS' if errs == 0 else 'FAIL'} (count, src, rows; "
                        f"recv={total_recv})",
                        flush=True,
                    )
                op.convert_combine_input(routing)
                sync()
                comm.barrier()
                out, out_w = op.combine(recv_x, wts[:ct], routing=routing)
                sync()
                comm.barrier()
                # telescopes to (sum_k wts)*input, over the k whose PE kept the token
                # (a dropped slot keeps nothing, with or without a cap)
                keep = torch.tensor(
                    [
                        [p >= 0 and p in reach[t] for p in pes[t].tolist()]
                        for t in range(ct)
                    ],
                    dtype=torch.bool,
                ).view(ct, K)
                w_k = wts[:ct].float().cpu()
                w_k = torch.where(keep, w_k, torch.zeros_like(w_k))
                ws = w_k.sum(dim=1, keepdim=True)
                exp = (ws * inp[:ct].float().cpu()).to(DTYPE)
                ok = torch.allclose(
                    out.float().cpu(), exp.float(), atol=5e-2, rtol=5e-2
                )
                ok_w = True
                tag = "OP-STDMOE"
            else:
                # identity expert: recv_x already holds the dispatched tokens.
                # Weights passed because this test checks the weight fold; an
                # inference caller passes None and the kernel skips it.
                out, out_w = op.combine(
                    recv_x, None if NOWTS else wts[:ct], routing=routing
                )
                sync()
                comm.barrier()
                # unique destination PEs the token reached (all of them unless capped)
                U = np.array([len(r) for r in reach])
                exp_w = torch.from_numpy(U).view(ct, 1).float() * wts[:ct].float().cpu()
                if _IS_FP4:
                    # fp4 combine is too lossy for a numeric hidden check (mirror v1);
                    # routing is validated by OP-LEC and the weights check.
                    ok = True
                else:
                    exp = (
                        torch.from_numpy(U).view(ct, 1).float() * inp[:ct].float().cpu()
                    ).to(DTYPE)
                    # fp8 paths (quant or plain fp8 token) lose precision; relax.
                    lossy = QUANT != "none" or _IS_FP8
                    atol = 3e-1 if lossy else 2e-2
                    rtol = 1e-1 if lossy else 2e-2
                    # e2m1 is too coarse for an elementwise bound; its L2 bound stands
                    ok = QUANT.startswith("fp4") or torch.allclose(
                        out.float().cpu(), exp.float(), atol=atol, rtol=rtol
                    )
                    if QUANT != "none":
                        # atol=0.3 passes any small-input loss; the L2 error does not
                        ref = exp.float()
                        err = (out.float().cpu() - ref).norm()
                        rel = float(err / ref.norm()) if ref.norm() > 0 else 0.0
                        ok = ok and rel < _QUANT_REL_L2[QUANT[:3]]
                ok_w = NOWTS or torch.allclose(out_w.cpu(), exp_w, atol=2e-3, rtol=2e-3)
                tag = f"OP-{COMBINE}" + (f"-{QUANT}" if QUANT != "none" else "")
            errs = d.allreduce_sum(0 if (ok and ok_w) else 1)
            quant_err = ""
            if QUANT != "none":
                quant_err = f" rel_l2={d.allreduce_max(rel):.4f}"
            if rank == 0:
                print(
                    f"# {tag} ct={ct_point}: {'PASS' if errs == 0 else 'FAIL'} "
                    f"(hidden={'ok' if ok else 'BAD'} "
                    f"wts={'n/a' if NOWTS else ('ok' if ok_w else 'BAD')}; "
                    f"recv={total_recv}{quant_err})",
                    flush=True,
                )
            if REROUTE and has_lec:
                _reroute_rounds(
                    d,
                    comm,
                    op,
                    inp,
                    wts,
                    sc_in,
                    ct,
                    ct_point,
                    num_experts,
                    cap is not None,
                )


def _lec_ref(d, idx, ct_point):
    """This rank's expected local_expert_count: its experts' ids over every rank's
    routed tokens."""
    rank, npes = d.rank, d.world
    all_i = [torch.empty_like(idx.cpu()) for _ in range(npes)]
    dist.all_gather(all_i, idx.cpu().contiguous())
    ids = torch.cat(
        [
            all_i[r][: (ct_point * (r % 3) // 2 if RAGGED else ct_point)].flatten()
            for r in range(npes)
        ]
    ).long()
    return _bincount_local(ids, rank)


def _bincount_local(ids, rank):
    ids = ids.flatten().long() - rank * EPR
    ids = ids[(ids >= 0) & (ids < EPR)]
    return torch.bincount(ids, minlength=EPR).to(torch.int32)


def _check_packed(packed, lec_ref, routing, recv_x, recv_i, total_recv):
    """StdMoE packing: per-expert counts, the multiset of sources each bucket holds,
    and each packed row equal to the recv row of its source."""
    px, pc, ps = packed
    if not torch.equal(pc.cpu(), lec_ref):
        return False
    bucket = px.shape[1]
    tis = routing.disp_tok_id_to_src_tok_id_local[:total_recv].long()
    ids = recv_i[:total_recv].long() - int(os.environ["RANK"]) * EPR
    inv = torch.full((bucket,), -1, dtype=torch.long, device=tis.device)
    inv[tis] = torch.arange(total_recv, device=tis.device)
    for e in range(EPR):
        n = int(pc[e])
        src = ps[e * bucket : e * bucket + n].long()
        rows_exp = (ids == e).nonzero(as_tuple=True)[0]
        if not torch.equal(src.sort().values, tis[rows_exp].sort().values):
            return False
        rows = inv[src]
        if bool((rows < 0).any()) or not torch.equal(px[e, :n], recv_x[rows]):
            return False
    return True


def _reroute_rounds(d, comm, op, inp, wts, sc_in, ct, ct_point, num_experts, capped):
    """REROUTE rounds of new routing; every read is launched straight after dispatch."""
    dev = inp.device
    g = torch.Generator(device="cpu").manual_seed(4321 + d.rank * 1000 + ct_point)
    bad = 0
    for _ in range(REROUTE):
        idx = torch.randint(0, num_experts, (inp.shape[0], K), generator=g)
        idx = idx.to(torch.int32)
        out = op.dispatch(inp[:ct], wts[:ct], sc_in, idx[:ct].to(dev))
        lec_now = op.local_expert_count().clone()
        pc_now = op.convert_dispatch_output()[1].clone() if STDMOE else None
        # Every rank's reads are done before any rank's next dispatch writes.
        sync()
        # under a cap, from the rows this rank kept, read before peers move on
        kept = out[3][: int(out[4].cpu().item())].cpu() if capped else None
        comm.barrier()
        ref = _bincount_local(kept, d.rank) if capped else _lec_ref(d, idx, ct_point)
        ok = torch.equal(lec_now.cpu(), ref)
        if STDMOE:
            ok = ok and torch.equal(pc_now.cpu(), ref)
        bad += 0 if ok else 1
    errs = d.allreduce_sum(bad)
    if d.rank == 0:
        print(
            f"# OP-REROUTE ct={ct_point}: {'PASS' if errs == 0 else 'FAIL'} "
            f"({REROUTE} rounds, {errs} stale reads)",
            flush=True,
        )

        _check_traffic_stats(op, d, rank, npes)


def _check_traffic_stats(op, d, rank, npes):
    """With metrics on (MORI_METRICS_ENABLE=1), check the per-peer token counts
    across ranks. Prints nothing when the op has no statistics."""
    get = getattr(op, "get_traffic_stats", None)
    stats = get() if get is not None else None
    if stats is None:
        return

    def world_matrix(key):
        # Row `rank` is this rank's per-peer vector; summing over ranks fills it in.
        m = torch.zeros(npes, npes, dtype=torch.int64)
        m[rank] = stats[key].to(torch.int64).cpu()
        dist.all_reduce(m, op=dist.ReduceOp.SUM)
        return m

    # MORI_METRICS_SAMPLE_EP=<n>: one dispatch in n counts, n-fold. Checked is each
    # rank's own bookkeeping: REPLAY does not give every dispatch its combine.
    import mori.metrics

    period = mori.metrics.sampling()["ep"]
    if period != 1:
        calls = stats["dispatch_calls"]
        sent = int(stats["dispatch_sent"].sum())
        if calls == 0:
            # With one in n, a rank may have had none of this run's dispatches taken.
            ok = stats["combine_calls"] == 0 and sent == 0
        elif period == 0:
            ok = False
        else:
            ok = (
                calls % period == 0
                and sent > 0
                and bool(
                    (
                        stats["combine_sent"] * calls
                        == stats["dispatch_recv"] * stats["combine_calls"]
                    ).all()
                )
                and stats["dispatch_kernel_seconds"] > 0
                and stats["combine_kernel_seconds"] > 0
            )
        bad = d.allreduce_sum(0 if ok else 1)
        if rank == 0:
            print(
                f"# OP-STATS ct=all: {'PASS' if bad == 0 else 'FAIL'} "
                f"(sampled 1/{period}: calls={calls}/{stats['combine_calls']} "
                f"tokens={sent}{'' if bad == 0 else f' bad_ranks={bad}'})",
                flush=True,
            )
        return

    disp_tx, disp_rx = world_matrix("dispatch_sent"), world_matrix("dispatch_recv")
    comb_tx, comb_rx = world_matrix("combine_sent"), world_matrix("combine_recv")
    checks = {
        "disp_tx==rx": bool((disp_tx == disp_rx.t()).all()),
        "comb_tx==rx": bool((comb_tx == comb_rx.t()).all()),
        # A combine returns what ONE dispatch delivered; REPLAY runs each one twice.
        "comb==disp": bool(
            (
                comb_tx * stats["dispatch_calls"] == disp_rx * stats["combine_calls"]
            ).all()
        ),
        "counted": int(disp_tx.sum()) > 0,
        "calls": stats["dispatch_calls"] > 0 and stats["combine_calls"] > 0,
        "timed": stats["dispatch_kernel_seconds"] > 0
        and stats["combine_kernel_seconds"] > 0,
    }
    if rank == 0:
        bad = [k for k, ok in checks.items() if not ok]
        print(
            f"# OP-STATS ct=all: {'PASS' if not bad else 'FAIL'} "
            f"(tokens={int(disp_tx.sum())} local={int(disp_tx.diag().sum())} "
            f"calls={stats['dispatch_calls']}/{stats['combine_calls']} "
            f"kernel_us={stats['dispatch_kernel_seconds'] / stats['dispatch_calls'] * 1e6:.0f}/"
            f"{stats['combine_kernel_seconds'] / stats['combine_calls'] * 1e6:.0f}"
            f"{' bad=' + ','.join(bad) if bad else ''})",
            flush=True,
        )


@contextlib.contextmanager
def _case_env(case):
    """os.environ with one CASES entry applied, the knobs re-read from it."""
    saved = {k: os.environ.get(k) for k in case}
    os.environ.update({k: str(v) for k, v in case.items()})
    _configure()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _configure()


def main():
    d = Dist()
    set_device(d.local_rank)
    # CASES='[{"DTYPE": "f32"}, ...]': env overrides run one after another in this
    # launch, on one communicator. Each case closes its op before the next builds
    # one, so the arena only has to fit the largest.
    cases = json.loads(os.environ.get("CASES") or "null")
    uid = Communicator.get_unique_id() if d.rank == 0 else None
    uid = d.bcast_uid(uid)
    win_bytes = 0
    for case in cases or [{}]:
        with _case_env(case):
            win_bytes = max(win_bytes, _window_bytes(d.world))
    with Communicator.init(
        d.world, d.rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)
    ) as comm:
        if cases is None:
            _run(d, comm)
        for i, case in enumerate(cases or ()):
            with _case_env(case):
                if d.rank == 0:
                    print(f"# CASE {i} {json.dumps(case, sort_keys=True)}", flush=True)
                _run(d, comm)
                if d.rank == 0:
                    print(f"# CASE {i} DONE", flush=True)
    d.shutdown()


def _DT():
    return {
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float4_e2m1fn_x2: 1,
        torch.float8_e4m3fnuz: 1,
        torch.float8_e4m3fn: 1,
    }[DTYPE]


if __name__ == "__main__":
    main()

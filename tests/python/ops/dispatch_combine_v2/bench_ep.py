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
"""Latency of the EPv2 intranode op, at the op API rather than the raw kernels.

Every backend is driven through EpDispatchCombineOp, so one script covers all of
them and BACKENDS=flydsl,hip compares them in a single process on one input.

dispatch and combine ALTERNATE, one pair per iteration, because that is the order
a layer runs them in. Timing N dispatches and then N combines instead leaves
combine's staging copy unmeasured: combine reads totalRecvTokenNum to size that
copy and clears it on the way out (ep_intranode_kernel.hpp:343, and
intranode_kernels.py:1209 for flydsl), so with no dispatch in between only the
first combine of the loop copies anything.

dispatch is called with return_routing=True, the way a serving stack calls it, so
the routing handle is inside the measured window. Each leg gets its own cuda event
pair; the loop does not sync, since a synchronize costs 5-20 us against a 30 us
kernel. Means over ITERS, not percentiles -- at these sizes the tail is the
machine, and a mean over enough iterations is the number that composes.

Every point is correctness-gated first: an identity expert makes combine[t] equal
U[t]*input[t], where U[t] is how many distinct PEs token t routed to. A geometry
that computes garbage never gets a bandwidth number. That check is deliberately
one invariant, not a matrix -- test_op.py owns dtypes, quant, scatter, StdMoE,
scales and recv-cap, across both backends.

Two machine-readable outputs sit alongside the human table. A final JSON line in
aiter's print_json_table shape carries every point, one row per (backend, mode,
m), so a parent driver reads results instead of parsing columns. And with
MORI_SMI_MONITOR=1 each point also replays under amdsmi: the clocks a number was
measured at belong with the number, since a throttled part reports a different
time for the same kernel. That replay is a window of its own, AFTER warmup and
graph capture, because ITERS pairs are ~10 ms against a 50 ms sampling tick.

    torchrun --standalone --nproc_per_node=8 bench_ep.py
    BACKENDS=flydsl,hip SWEEP=512,4096 ITERS=200 torchrun ... bench_ep.py
    MORI_SMI_MONITOR=1 MORI_SMI_DURATION=1.0 torchrun ... bench_ep.py
"""

import math
import os
import sys
import time

import torch
import torch.distributed as dist

import mori.cco as cco
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

import _data
import _report
import _smi

HIDDEN = int(os.environ.get("HIDDEN", 7168))
TOPK = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 32))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
# Per-token scale row transported alongside the payload (the block-scale a quantized
# MoE hands to dispatch). scale_bytes = SCALE_DIM * SCALE_TS must be a multiple of 4,
# which hip_backend asserts. The effective SCALE_DIM is resolved once the dispatch
# dtype is known (see below): fp8/fp4 default scales ON, bf16 OFF. An explicit
# SCALE_DIM env value overrides that default (SCALE_DIM=0 forces scales off).
_SCALE_DIM_ENV = os.environ.get("SCALE_DIM")
SCALE_TS = int(os.environ.get("SCALE_TS", 1))
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512,4096").split(",")]
# Comma-separated; MORI_V2_KERNEL_BACKEND still works for a single backend.
BACKENDS = [
    b
    for b in os.environ.get(
        "BACKENDS", os.environ.get("MORI_V2_KERNEL_BACKEND", "hip")
    ).split(",")
    if b
]
MODES = os.environ.get("MODES", "eager,graph").split(",")
# "inplace": the expert already wrote into the staging view, so combine elides the
# copy -- what a real pipeline does. "staged": a separate buffer, copy included.
COMBINE_IN = os.environ.get("COMBINE_IN", "inplace")
# COMB_MODE=push: the push-send combine (combine_mode="scatter", gfx125x only). It
# writes each received row to landing[srcPe][recvSlot] on the owner -- quantized in the
# default build -- and the owner reduces from there, so CHECK compares what landed and,
# for a build that reduces, the output; which checks apply follows the build (below).
COMB_MODE = os.environ.get("COMB_MODE", "pull")
if COMB_MODE not in ("pull", "push"):
    raise ValueError(f"COMB_MODE={COMB_MODE!r}: want pull|push")
_PUSH = COMB_MODE == "push"
_PUSH_WIRE = HIDDEN * 2 // 4
# The knobs below are read as the kernel compiles them (push_knob): the -D flags in
# MORI_JIT_EXTRA_FLAGS, else the shipped defaults unless -DMORI_PUSH_DEFAULTS=0. The push
# kernel is the hip backend's alone, so a pull-only run does not import it.
if _PUSH:
    from mori.ops.dispatch_combine_v2.hip_backend import push_knob
else:

    def push_knob(name):
        return 0


# MORI_PUSH_SIGNAL=3: the kernel also reduces, so CHECK compares the first _PUSH_WIRE bytes
# of each output row, too.
_PUSH_REDUCE = _PUSH and push_knob("SIGNAL") == 3
# MORI_PUSH_QUANT=1|2: the kernel quantizes whole bf16 rows (e2m1 + one e8m0 per 32 | 128
# elements), so CHECK stages real bf16 rows and bounds the reduce's output by the rounding.
_PUSH_QGROUP = {1: 32, 2: 128}.get(push_knob("QUANT"), 0) if _PUSH else 0
_PUSH_QUANT = _PUSH_QGROUP > 0
if _PUSH_QUANT:
    _PUSH_WIRE = (HIDDEN // 2 + HIDDEN // _PUSH_QGROUP + 127) // 128 * 128
# The scale rule the kernel was built with, for the exact check (QUANT=1 only):
#   mx    MORI_PUSH_QMX=1: MXFP4 RoundUp, e8m0 = ceil(log2(amax / 6)) (aiter's default)
#   lean  MORI_PUSH_QLEAN=1: 2^(floor(log2 amax) - 1), exponent floor -126
#   old   the same, exponent floor -127 (the conversion itself stops at -126)
# PUSH_QRULE overrides it: a negative control, a rule the kernel was not built with must fail.
_PUSH_QRULE = os.environ.get("PUSH_QRULE") or (
    "mx" if push_knob("QMX") else "lean" if push_knob("QLEAN") else "old"
)
# PROFILE=1 (push only): after a point's timing, trace PROF_N combines through the
# kernel's IF_ENABLE_PROFILER hooks and print per-phase times. The kernel only
# traces when built with MORI_JIT_EXTRA_FLAGS=-DENABLE_PROFILER. PROF_OUT=<dir>
# also writes one Perfetto trace per rank.
PROFILE = int(os.environ.get("PROFILE", 0))
PROF_N = int(os.environ.get("PROF_N", 5))
PROF_GAP = float(os.environ.get("PROF_GAP", 0.2))
# PROF_MODE=graph (default): trace inside the timed loop's shape -- PROF_R (dispatch, combine)
# pairs in one warmed-up graph -- and analyse the last combine of each replay.
# PROF_MODE=cold: one traced combine per host-synced step; PROF_WARM picks what precedes it:
#   0 an idle gap, 1 an untraced launch of the same kernel, 2 its dispatch with no host sync.
PROF_MODE = os.environ.get("PROF_MODE", "graph")
PROF_R = int(os.environ.get("PROF_R", 20))
PROF_WARM = int(os.environ.get("PROF_WARM", 0))
PROF_OUT = os.environ.get("PROF_OUT", "")
CHECK = int(os.environ.get("CHECK", 1))
# CHECK_REPEAT=n (push, quantized reduce): after the checked pair, n more dispatch + combine pairs
# with the staged rows times 2, 4, ..., each checked, the landing never cleared in between.
CHECK_REPEAT = int(os.environ.get("CHECK_REPEAT", 0))
# PUSH_QCHECK_DEV=1: the quantized-reduce check also runs on the device (see check_push_quant).
PUSH_QCHECK_DEV = int(os.environ.get("PUSH_QCHECK_DEV", 0))
# Payload distribution and RNG seed: DATA_INIT=zero|constant|uniform|norm, SEED,
# CONST_VAL. Same names and meanings as aiter's test_common, so the two harnesses
# describe the same input. Defaults reproduce this file's previous behaviour.
INIT, SEED, CONST_VAL = _data.env_config()
# One machine-readable line per run, aiter's print_json_table format. JSON=0 to
# drop it; the human table above it is unchanged either way.
JSON = int(os.environ.get("JSON", 1))
# Clock/power telemetry, off unless asked for: it adds a replay window per point.
SMI_ON, SMI_INTERVAL, SMI_DURATION = _smi.env_config()
# What dispatch transports; combine is always bf16, so anything else is asymmetric.
_DISP = os.environ.get("DISP", "bf16")
_DISP_DT = {
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "fp4": torch.float4_e2m1fn_x2,
}[_DISP]
_DISP_NBYTES = {torch.bfloat16: 2, torch.float8_e4m3fn: 1}.get(_DISP_DT, 0.5)
_FP4 = _DISP_DT is torch.float4_e2m1fn_x2
# Resolve the per-token scale width now that the dispatch dtype is known. Quantized
# dispatch (fp8/fp4) carries a block-scale row, so default it on and exercise the
# scale-transport path a real quantized pipeline uses; bf16 has no scales. Both fp8
# and fp4 default to hidden/32 bytes/token (224 B at hidden=7168, E8M0-style,
# SCALE_TS=1). An explicit SCALE_DIM env value wins, including SCALE_DIM=0 (off).
if _SCALE_DIM_ENV is not None:
    SCALE_DIM = int(_SCALE_DIM_ENV)
elif _DISP_DT is torch.bfloat16:
    SCALE_DIM = 0
else:
    SCALE_DIM = HIDDEN // 32
# What the correctness gate covers, as one token. A bool cannot say it: fp4 and
# an all-zero payload verify the dispatch bytes but never compare combine's
# output, and a row claiming "verified" next to a combine_us nobody checked is
# the machine-readable half losing what the human summary spells out.
VERIFY_SCOPE = (
    "none"
    if not CHECK
    else (
        "dispatch_bytes"
        if _data.verifies_nothing(INIT) or (_FP4 and not _PUSH)
        else ("dispatch_bytes+push_bytes" if _PUSH else "dispatch_bytes+combine")
    )
)
# Geometry, same spelling as tools/ep_test.sh. Unset = the backend's tuned default.
_G = {
    k: (int(os.environ[k]) if os.environ.get(k) else None)
    for k in ("DBN", "DWPB", "CBN", "CWPB")
}


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    n_experts = world * EPR
    M = max(SWEEP)
    # Inputs before the communicator: comm_create leaves a latched HIP error that
    # the next torch call reports as its own.
    #
    # Payload and routing draw from SEPARATE streams. Sharing one made the routing
    # depend on how much randomness the payload happened to consume, so changing
    # SWEEP -- which changes M, which changes the payload's shape -- silently
    # resampled the routing and moved the measured time.
    gp = _data.make_generator(_data.seed_for(SEED, rank))
    gr = _data.make_generator(_data.seed_for(SEED, rank, routing=True))
    inp = _data.make_payload((M, HIDDEN), INIT, gp, _DISP_DT, constant=CONST_VAL).to(
        dev
    )
    wts = torch.rand(M, TOPK, generator=gr, dtype=torch.float32).to(dev)
    idx = (
        torch.stack([torch.randperm(n_experts, generator=gr)[:TOPK] for _ in range(M)])
        .to(torch.int32)
        .to(dev)
    )
    # Per-token scale rows, transported when SCALE_DIM>0 (fp8/fp4 by default). Shaped
    # exactly as repro_epv2_topk9.py: sc_n_i32 int32 lanes viewed as bytes, trimmed to
    # SCALE_DIM so scale_type_size=1 * scale_dim holds. Deterministic (arange + rank
    # offset) so a run reproduces; dispatch moves them verbatim (opaque bytes).
    scales = None
    if SCALE_DIM:
        sc_n_i32 = (SCALE_DIM + 3) // 4
        scales = (
            (torch.arange(M, dtype=torch.int32) + rank * 100003)
            .view(M, 1)
            .repeat(1, sc_n_i32)
            .view(torch.uint8)[:, :SCALE_DIM]
            .contiguous()
            .to(dev)
        )
    # Unique destination PEs per token: what an identity expert makes combine sum.
    U = (
        torch.zeros(M, world, dtype=torch.bool)
        .scatter_(1, (idx.cpu().long() // EPR), True)
        .sum(1)
    )

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    # Sized for bf16, the widest, and for one arena per backend.
    vmm = len(BACKENDS) * 2 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (512 << 20)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    def build(backend):
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=world,
            hidden_dim=HIDDEN,
            max_num_inp_token_per_rank=M,
            num_experts_per_rank=EPR,
            num_experts_per_token=TOPK,
            data_type=torch.bfloat16,
            dispatch_data_type=None if _DISP_DT is torch.bfloat16 else _DISP_DT,
            combine_data_type=None if _DISP_DT is torch.bfloat16 else torch.bfloat16,
            scale_dim=SCALE_DIM,
            scale_type_size=SCALE_TS,
            kernel_backend=backend,
            dispatch_block_num=_G["DBN"],
            warp_num_per_block=_G["DWPB"],
            combine_block_num=_G["CBN"],
            combine_warp_num_per_block=_G["CWPB"],
            combine_mode="scatter" if _PUSH else "gather",
        )
        return EpDispatchCombineOp(cfg, comm)

    ops = {b: build(b) for b in BACKENDS}

    if rank == 0:
        print(
            f"# EP{world} hidden={HIDDEN} topk={TOPK} epr={EPR} "
            f"init={INIT} seed={SEED} scale_dim={SCALE_DIM}x{SCALE_TS}B "
            f"disp={_DISP_DT} comb=bf16 backends={BACKENDS} modes={MODES} "
            f"iters={ITERS} combine_in={COMBINE_IN} check={CHECK} comb_mode={COMB_MODE}",
            flush=True,
        )

    def lockstep():
        torch.cuda.synchronize()
        dist.barrier()

    def check_dispatch(op, total, routing):
        """Every received row must equal, BYTE FOR BYTE, the source row it claims.

        dispatch only transports -- "mori does no quantizing here: fp8/fp4 payloads
        arrive already packed" (hip_backend.py) -- so the bytes that land must be
        the bytes that were sent, for every wire dtype. Comparing them needs no
        conversion and no arithmetic, which is what makes this the one check fp4
        can pass: torch has no fp4 cast kernel at all, and combine cannot take fp4
        anyway. It is also sharper than the end-to-end check, which sums U copies
        and can average a wrong byte away.

        Each rank's payload is a pure function of (SEED, its rank), so a receiver
        can regenerate any sender's tensor locally and look up the row that the
        reverse map says a slot came from. That determinism is what makes this
        possible; before the seed was configurable it was not.
        """
        if not CHECK or total == 0:
            return 0
        tis = routing.disp_tok_id_to_src_tok_id_local[:total].cpu()
        src_pe, src_tok = (tis // M).to(torch.int64), (tis % M).to(torch.int64)
        got = op.recv_tokens()[:total].cpu().view(torch.uint8)
        bad = 0
        for pe in src_pe.unique().tolist():  # one regeneration per source rank
            sel = src_pe == pe
            ref = _data.make_payload(
                (M, HIDDEN),
                INIT,
                _data.make_generator(_data.seed_for(SEED, int(pe))),
                _DISP_DT,
                constant=CONST_VAL,
            ).view(torch.uint8)
            bad += int((got[sel] != ref[src_tok[sel]]).any(dim=1).sum())
        n = torch.tensor([bad])
        dist.all_reduce(n)
        if rank == 0 and n.item():
            print(
                f"  [{op.backend_name}] DISPATCH BYTE MISMATCH: "
                f"{int(n.item())} rows across {world} ranks",
                flush=True,
            )
        return int(n.item())

    def check_push(op, ct, routing):
        """Every landed slot must hold the first _PUSH_WIRE bytes of this rank's own
        input token that went there, and every other slot must still be zero.

        prime() fills each staged row's wire quarter with the bytes that slot
        received in dispatch, i.e. the owner's input row. The owner's dest map
        says which (pe, slot) each of its tokens went to, and the push lands at
        landing[pe][slot] -- so the whole region is predictable locally.
        """
        cap = op.cfg.effective_max_recv
        land = op.push_landing()
        dm = routing.disp_dest_tok_id_map[: ct * TOPK].long()
        valid = (dm >= 0) & (dm < world * cap)
        flat = dm[valid]
        tok = torch.arange(ct * TOPK, device=dm.device)[valid] // TOPK
        pe, slot = flat // cap, flat % cap
        got = land[pe, slot, :_PUSH_WIRE]
        exp = inp.view(torch.uint8)[tok, :_PUSH_WIRE]
        hit = torch.zeros(world, cap, dtype=torch.bool, device=land.device)
        hit[pe, slot] = True
        n = torch.tensor(
            [
                int((got != exp).any(dim=1).sum()),
                int(land[~hit].any(dim=1).sum()),
                int(flat.numel()),
            ]
        )
        dist.all_reduce(n)
        bad, stray, landed = (int(v) for v in n.tolist())
        if rank == 0:
            print(
                f"  [PUSHCHECK] ct={ct} landed={landed} bad={bad} stray={stray}",
                flush=True,
            )
        return bad + stray

    def check_push_reduce(out, ct):
        """Each token's rows are all its own input's wire bytes (check_push), one per
        distinct destination rank, so the reduce must give U * row, rounded to bf16 once.
        The kernel adds in fp32 and U * a bf16 is exact in fp32, so the match is exact;
        wire bytes that decode to NaN must decode to NaN on both sides, and values below the
        smallest normal may be flushed to zero by either side."""
        n_el = _PUSH_WIRE // 2
        got = (
            out.view(torch.uint8)[:ct, :_PUSH_WIRE]
            .contiguous()
            .view(torch.bfloat16)
            .float()
        )
        row = (
            inp.view(torch.uint8)[:ct, :_PUSH_WIRE]
            .contiguous()
            .view(torch.bfloat16)
            .float()
        )
        exp = (
            (U[:ct].to(row.device).view(ct, 1).float() * row).to(torch.bfloat16).float()
        )
        tiny = torch.finfo(torch.float32).tiny
        same = (
            (got == exp)
            | (torch.isnan(got) & torch.isnan(exp))
            | ((got.abs() < tiny) & (exp.abs() < tiny))
        )
        assert same.shape == (ct, n_el), same.shape
        n = torch.tensor([int((~same).any(dim=1).sum()), ct])
        dist.all_reduce(n)
        bad, rows = (int(v) for v in n.tolist())
        if rank == 0:
            print(f"  [PUSHREDUCE] ct={ct} rows={rows} bad={bad}", flush=True)
        return bad

    def bf16_payload(pe):
        """Rank pe's payload as bf16: a pure function of (SEED, pe), like check_dispatch's."""
        return _data.make_payload(
            (M, HIDDEN),
            INIT,
            _data.make_generator(_data.seed_for(SEED, int(pe))),
            torch.bfloat16,
            constant=CONST_VAL,
        )

    def push_quant_rows(total, routing):
        """The combine input an identity expert would leave: recv slot i holds its source
        token's row, regenerated from the source rank's seed."""
        tis = routing.disp_tok_id_to_src_tok_id_local[:total].cpu()
        src_pe, src_tok = (tis // M).to(torch.int64), (tis % M).to(torch.int64)
        rows = torch.empty(total, HIDDEN, dtype=torch.bfloat16)
        for pe in src_pe.unique().tolist():
            sel = src_pe == pe
            rows[sel] = bf16_payload(pe)[src_tok[sel]]
        return rows.to(dev)

    _FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def fp4_rne(v):
        """|v| to the nearest e2m1 magnitude, ties to the even code, saturating at 6."""
        g = _FP4_GRID.to(v.device)
        v = v.clamp(max=6.0).contiguous()
        idx = torch.searchsorted(g, v).clamp(1, 7)
        lo, hi = g[idx - 1], g[idx]
        up = (hi - v < v - lo) | ((hi - v == v - lo) & (idx % 2 == 0))
        return torch.where(up, hi, lo)

    def push_quant_deq(x):
        """What the wire decodes to for bf16 values x [..., group], under _PUSH_QRULE: the e8m0
        byte from the group's |max| bits, the codes from x / 2^(conv - 127) rounded to nearest
        even, decoded as code * 2^(byte - 127). conv differs from byte only for the old rule's
        smallest groups, where the conversion stops at 2^-126 and the byte does not."""
        m15 = (x.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0x7FFF).amax(
            dim=-1, keepdim=True
        )
        if _PUSH_QRULE == "mx":
            byte = torch.clamp(((m15 + 0x3F) >> 7) - 2, min=1)
        else:
            byte = torch.clamp((m15 >> 7) - 1, min=1 if _PUSH_QRULE == "lean" else 0)
        conv = byte.clamp(min=1)
        sc_conv = torch.pow(2.0, (conv - 127).float())
        sc_byte = torch.pow(2.0, (byte - 127).float())
        return torch.sign(x) * fp4_rne(x.abs() / sc_conv) * sc_byte

    def push_quant_stats(x, u, got, t0=0):
        """[bad rows, inexact rows, non-finite outputs], max error in steps, and the first
        inexact element described, for got against U * x; [tokens, groups, group] tensors
        on one device, token t0 first. Per group the sender divides by step = 2^(e-128), e
        the biased exponent of the group's |max|, so a code is within half a step of the
        value under round-to-nearest and within one step under truncation; the bound is one
        step plus bf16 rounding of the sum. Reading zeros, a wrong scale, or a wrong nibble
        order breaks it at the group's largest element. Inexact: U copies of one decoded
        row, each a small multiple of a power of two, sum and round to bf16 without loss,
        so the output must equal the reference bit for bit."""
        grp = x.shape[-1]
        amax = x.abs().amax(dim=2, keepdim=True)
        _, ex = torch.frexp(amax)  # amax = m * 2^ex, m in [0.5, 1)
        step = torch.ldexp(torch.ones_like(amax), (ex - 2).clamp(min=-127))
        err = (got - u * x).abs()
        tol = u * step + (u * x).abs() * 2.0**-8 + 1e-30
        # Not err > tol: a NaN compares false to everything and would pass as in bounds.
        bad = int((~(err <= tol)).flatten(1).any(dim=1).sum())
        nonfinite = int((~torch.isfinite(got)).sum())
        max_err = float(torch.nan_to_num(err / (u * step), nan=float("inf")).max())
        inexact, first = 0, ""
        if grp == 32:
            ref = (u * push_quant_deq(x)).to(torch.bfloat16).float()
            miss = got != ref
            inexact = int(miss.flatten(1).any(dim=1).sum())
            if inexact:
                i = int(miss.flatten().nonzero()[0])
                t, g = i // (x.shape[1] * grp), (i // grp) % x.shape[1]
                first = (
                    f" first@rank{rank}:tok{t0 + t}:grp{g}:el{i % grp}"
                    f" x={float(x.flatten()[i]):.6g} got={float(got.flatten()[i]):.6g}"
                    f" ref={float(ref.flatten()[i]):.6g} u={float(u.flatten()[t]):g}"
                    f" amax={float(amax[t, g, 0]):.6g}"
                    f" row_miss={int(miss[t].sum())} grp_miss={int(miss[t, g].sum())}"
                    f" grps_miss={int(miss[t].any(dim=1).sum())}"
                )
        return [bad, inexact, nonfinite], max_err, first

    def check_push_quant(out, ct, scale=1.0):
        """out[t] must be U * dequant(quant(x[t])); see push_quant_stats. max_err is printed
        in steps: ~0.5 means round-to-nearest. scale: the power of two the staged rows were
        multiplied by, which the codes carry exactly.

        Computed on the host, a chunk of tokens at a time. PUSH_QCHECK_DEV=1 runs the same
        arithmetic on the device as well and prints how far its counts are from the host's
        (dev_diff)."""
        grp = _PUSH_QGROUP
        x_all = (bf16_payload(rank)[:ct].float() * scale).view(ct, HIDDEN // grp, grp)
        u_all = U[:ct].float().view(ct, 1, 1)
        got_all = out[:ct].cpu().float().view(ct, HIDDEN // grp, grp)
        tot, max_err, first = [0, 0, 0], 0.0, ""
        for t0 in range(0, ct, 1024):
            sl = slice(t0, min(ct, t0 + 1024))
            c, e, f = push_quant_stats(x_all[sl], u_all[sl], got_all[sl], t0)
            tot = [a + b for a, b in zip(tot, c)]
            max_err = max(max_err, e)
            first = first or f
        devdiff = 0
        if PUSH_QCHECK_DEV:
            c, _, _ = push_quant_stats(x_all.to(dev), u_all.to(dev), got_all.to(dev))
            devdiff = sum(abs(a - b) for a, b in zip(c, tot))
        n = torch.tensor([tot[0], ct, tot[1], tot[2], devdiff], dtype=torch.float64)
        dist.all_reduce(n)
        m = torch.tensor([max_err], dtype=torch.float64)
        dist.all_reduce(m, op=dist.ReduceOp.MAX)
        # Every rank's first mismatch, printed by rank 0 alone: the row that fails is
        # usually not rank 0's, and ranks printing at once tear each other's lines.
        firsts = [None] * world
        dist.all_gather_object(firsts, first)
        if rank == 0:
            print(
                f"  [PUSHQUANT] ct={ct} x{scale:g} rows={int(n[1])} bad={int(n[0])} "
                f"max_err={float(m[0]):.3f} steps rule={_PUSH_QRULE} exact_bad={int(n[2])}"
                f" nonfinite={int(n[3])}"
                + (f" dev_diff={int(n[4])}" if PUSH_QCHECK_DEV else "")
                + "".join(f for f in firsts if f),
                flush=True,
            )
        return int(n[0]) + int(n[2])

    def prime(op, ct, i_, w_, x_, s_=None):
        """One full pair, untimed. Reads total_recv for the host, builds the buffer
        the timed loop will reuse, and with CHECK verifies the result through that
        same buffer -- so the gate covers exactly what gets timed, staged copy
        included. Must be a PAIR: a bare dispatch would leave total_recv set, and
        dispatch accumulates into it while only combine clears it, so the next
        combine would stage twice the tokens and run past the arena.
        Returns (total_recv, buf, ok, checked)."""
        *_, total_t, r = op.dispatch(i_, w_, s_, x_, return_routing=True)
        lockstep()  # the reverse map is only valid after this barrier
        total = int(total_t.cpu().item())
        dispatch_bad = check_dispatch(op, total, r)
        stage = op.combine_in_view()[:total]
        # An all-zero payload reduces the identity-expert check to 0 == 0, which
        # holds however wrong the kernel is. fp4 cannot go through combine at all
        # (hip has no fp4 combine), so for it check_dispatch is the whole story.
        checked = bool(CHECK) and not _FP4 and not _data.verifies_nothing(INIT)
        push_checked = bool(CHECK) and _PUSH and not _data.verifies_nothing(INIT)
        if _PUSH:
            checked = False
        if checked:  # identity expert: stage the dispatched tokens unchanged
            stage.copy_(op.recv_tokens()[:total].to(stage.dtype))
        if push_checked:
            if _PUSH_QUANT:
                stage.copy_(push_quant_rows(total, r))
            else:
                recv_b = op.recv_tokens()[:total].view(torch.uint8)
                assert recv_b.shape[1] >= _PUSH_WIRE, (recv_b.shape, _PUSH_WIRE)
                stage.view(torch.uint8)[:, :_PUSH_WIRE].copy_(recv_b[:, :_PUSH_WIRE])
            op.push_landing().zero_()
            lockstep()
        if COMBINE_IN == "staged":
            buf = stage.clone()
        elif COMBINE_IN == "cached":
            # Ordinary (cached) device memory rather than the symmetric window, which cco maps
            # uncached; full max_recv rows, since the push kernel may read rows past total.
            buf = op.combine_in_view().clone()[:total]
        else:
            buf = stage
        out, _ = op.combine(buf, routing=r)
        lockstep()
        if dispatch_bad:
            return total, buf, False, True
        if push_checked and _PUSH_QUANT:
            # The landed bytes are quantized, so only the reduce's output can be checked.
            if not _PUSH_REDUCE:
                return total, buf, True, False
            ok = check_push_quant(out, ct) == 0
            for k in range(1, CHECK_REPEAT + 1):
                # Another pair with every staged row times 2^k and the landing left as the last
                # combine wrote it: a launch released before its own rows landed would reduce the
                # previous launch's rows, off by a factor of two.
                *_, total_t, r = op.dispatch(i_, w_, s_, x_, return_routing=True)
                lockstep()
                assert int(total_t.cpu().item()) == total, (
                    int(total_t.cpu().item()),
                    total,
                )
                buf.copy_(push_quant_rows(total, r) * float(2**k))
                lockstep()
                out, _ = op.combine(buf, routing=r)
                lockstep()
                ok = check_push_quant(out, ct, float(2**k)) == 0 and ok
            return total, buf, ok, True
        if push_checked:
            ok = check_push(op, ct, r) == 0
            if _PUSH_REDUCE:
                ok = check_push_reduce(out, ct) == 0 and ok
            return total, buf, ok, True
        if not checked:
            return total, buf, True, bool(CHECK)
        exp = U[:ct].view(ct, 1).float() * inp[:ct].float().cpu()
        lossy = _DISP_DT is torch.float8_e4m3fn
        atol, rtol = (1.0, 1.5e-1) if lossy else (2e-2, 2e-2)
        bad = torch.tensor(
            [0 if torch.allclose(out.float().cpu(), exp, atol=atol, rtol=rtol) else 1]
        )
        dist.all_reduce(bad)
        if rank == 0 and bad.item():
            print(
                f"  ct={ct:<5d} [{op.backend_name}] CHECK FAIL "
                f"({int(bad.item())}/{world} ranks, identity expert, U in "
                f"[{int(U[:ct].min())},{int(U[:ct].max())}])",
                flush=True,
            )
        return total, buf, bad.item() == 0, True

    def time_pairs(mode, one_pair, capture, split_legs):
        """ITERS (dispatch, combine) pairs; mean us per leg.

        Warmup is lock-stepped per iteration: at small token counts a rank that
        starts call N+1 before every rank finished N can overwrite an unconsumed
        cross-device barrier flag, and both ranks hang. The timed loop is not --
        the kernels' own barrier keeps the ranks within one iteration."""
        for _ in range(WARMUP):
            one_pair()
            lockstep()
        lockstep()

        if mode == "graph":
            gd, gc = capture()
            for _ in range(WARMUP):
                gd.replay()
                gc.replay()
            lockstep()
            run_d, run_c = gd.replay, gc.replay
        else:
            run_d, run_c = capture()

        ev = [
            [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
            for _ in range(3)
        ]
        for i in range(ITERS):
            ev[0][i].record()
            run_d()
            ev[1][i].record()
            run_c()
            ev[2][i].record()
        torch.cuda.synchronize()
        dist.barrier()
        hd = sum(ev[0][i].elapsed_time(ev[1][i]) for i in range(ITERS)) / ITERS * 1000
        hc = sum(ev[1][i].elapsed_time(ev[2][i]) for i in range(ITERS)) / ITERS * 1000

        e2e = slope = -1.0
        rep = int(os.environ.get("E2E_R", 10))

        def graph_of(npairs):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(npairs):
                    one_pair()
            lockstep()
            for _ in range(WARMUP):
                g.replay()
            lockstep()
            return g

        def replay_us(g, n):
            a = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            b = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            for k in range(n):
                a[k].record()
                g.replay()
                b[k].record()
            torch.cuda.synchronize()
            dist.barrier()
            return sum(a[k].elapsed_time(b[k]) for k in range(n)) / n * 1000

        if mode == "graph" and rep > 0:
            n_e2e = int(os.environ.get("E2E_N", 20))
            alt = int(os.environ.get("E2E_ALT", 4))
            g_r, g_2r = graph_of(rep), graph_of(2 * rep)
            t_r = t_2r = 0.0
            for _ in range(alt):
                t_r += replay_us(g_r, n_e2e)
                t_2r += replay_us(g_2r, n_e2e)
            t_r, t_2r = t_r / alt, t_2r / alt
            e2e = t_r / rep
            slope = (t_2r - t_r) / rep
        gd_us = gc_us = -1.0
        gd_lo = gd_hi = gc_lo = gc_hi = float("nan")
        evflags = 0
        gevr = int(os.environ.get("GEV_R", 20))
        hip = None
        if mode == "graph" and gevr > 0:
            import ctypes

            try:
                hip = ctypes.CDLL("libamdhip64.so")
            except OSError as exc:
                if rank == 0:
                    print(f"  [GEV] off, cannot load libamdhip64.so: {exc}", flush=True)
        if hip is not None:
            hip.hipEventRecordWithFlags.restype = ctypes.c_int
            hip.hipEventRecordWithFlags.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint,
            ]

            hip.hipEventCreateWithFlags.restype = ctypes.c_int
            hip.hipEventCreateWithFlags.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ]
            hip.hipEventElapsedTime.restype = ctypes.c_int
            hip.hipEventElapsedTime.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]

            evflags = int(os.environ.get("GEV_EVFLAGS", "0"), 0)

            def make_ev():
                e = ctypes.c_void_p()
                r = hip.hipEventCreateWithFlags(ctypes.byref(e), evflags)
                if r != 0:
                    raise RuntimeError(f"hipEventCreateWithFlags(0x{evflags:x}) rc={r}")
                return e

            rd, rc_leg = split_legs()
            gev = [[make_ev() for _ in range(3)] for _ in range(gevr)]

            def rec_ext(e):
                r = hip.hipEventRecordWithFlags(
                    e,
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                    1,
                )
                if r != 0:
                    raise RuntimeError(f"hipEventRecordWithFlags rc={r}")

            def el_us(a, b):
                ms = ctypes.c_float()
                r = hip.hipEventElapsedTime(ctypes.byref(ms), a, b)
                if r != 0:
                    raise RuntimeError(f"hipEventElapsedTime rc={r}")
                return ms.value * 1000.0

            gg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gg):
                for i in range(gevr):
                    rec_ext(gev[i][0])
                    rd()
                    rec_ext(gev[i][1])
                    rc_leg()
                    rec_ext(gev[i][2])
            lockstep()
            for _ in range(WARMUP):
                gg.replay()
            lockstep()
            n_gev = int(os.environ.get("GEV_N", 20))
            skip = 1 if gevr > 1 else 0
            npair = gevr - skip
            ds, cs = [], []
            for _ in range(n_gev):
                gg.replay()
                lockstep()
                ds.append(
                    sum(el_us(gev[i][0], gev[i][1]) for i in range(skip, gevr)) / npair
                )
                cs.append(
                    sum(el_us(gev[i][1], gev[i][2]) for i in range(skip, gevr)) / npair
                )
            gd_us, gc_us = sum(ds) / n_gev, sum(cs) / n_gev
            gd_lo, gd_hi = min(ds), max(ds)
            gc_lo, gc_hi = min(cs), max(cs)

            def quart(xs):
                s = sorted(xs)
                n = len(s)
                mid = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
                return mid, s[n // 4], s[(3 * n) // 4]

            gd_med, gd_p25, gd_p75 = quart(ds)
            gc_med, gc_p25, gc_p75 = quart(cs)
        d, c = (gd_us, gc_us) if gd_us > 0 else (hd, hc)
        if rank == 0 and e2e > 0:
            src = "gev" if gd_us > 0 else "periter"
            print(
                f"[E2E] mode={mode} src={src} d={d:.2f} c={c:.2f} sum={d + c:.2f} "
                f"hd={hd:.2f} hc={hc:.2f} hsum={hd + hc:.2f} "
                f"e2e={e2e:.2f} slope={slope:.2f} cost={hd + hc - e2e:.2f} "
                f"R={rep} iters={ITERS}",
                flush=True,
            )
        if rank == 0 and gd_us > 0:
            print(
                f"[GEV] mode={mode} gd={gd_us:.2f} gc={gc_us:.2f} "
                f"gsum={gd_us + gc_us:.2f} "
                f"over={gd_us + gc_us - e2e if e2e > 0 else float('nan'):.2f} "
                f"overslope={gd_us + gc_us - slope if slope > 0 else float('nan'):.2f} "
                f"gdlo={gd_lo:.2f} gdhi={gd_hi:.2f} gclo={gc_lo:.2f} gchi={gc_hi:.2f} "
                f"gdmed={gd_med:.2f} gdp25={gd_p25:.2f} gdp75={gd_p75:.2f} "
                f"gcmed={gc_med:.2f} gcp25={gc_p25:.2f} gcp75={gc_p75:.2f} "
                f"evflags=0x{evflags:x} R={gevr} N={n_gev}",
                flush=True,
            )
        return d, c, run_d, run_c

    def smi_window(label, pair_us, run_d, run_c):
        """Replay the pair under the GPU monitor and gather every rank's clocks.

        A window of its own, after the timed loop, because ITERS pairs are ~10 ms
        against a 50 ms sampling tick -- too short to sample even once. Warmup,
        graph capture and input generation are already behind us, which is what
        "do not include init in the telemetry" asks for. Nothing here is timed.

        The replay count must be IDENTICAL on every rank: these kernels barrier
        across devices, so a per-rank duration loop would leave one rank waiting
        on a peer that has stopped. It is computed on rank 0 and broadcast.
        Launches are batched between synchronizations so a 40 us pair still keeps
        the GPU busy across a tick instead of measuring the sync gap.
        """
        if not SMI_ON or pair_us <= 0:
            return None
        plan = torch.zeros(2, dtype=torch.int64)
        if rank == 0:
            batch = max(1, min(1024, int(SMI_INTERVAL * 1e6 / pair_us)))
            plan[0] = batch
            plan[1] = max(1, math.ceil(SMI_DURATION * 1e6 / (pair_us * batch)))
        dist.broadcast(plan, src=0)
        batch, rounds = int(plan[0]), int(plan[1])

        mon, err = None, None
        try:
            mon = _smi.GpuMonitor(torch.cuda.current_device(), interval_s=SMI_INTERVAL)
            mon.start()
        except Exception as e:  # noqa: BLE001 - telemetry never fails the bench
            err = f"rank {rank}: {type(e).__name__}: {e}"
        # Every rank must agree on whether the replay happens, or the ranks that
        # skip it deadlock the ones that do not.
        bad = torch.tensor([1 if err else 0])
        dist.all_reduce(bad)
        if bad.item():
            if mon is not None:
                mon.stop()
            if rank == 0:
                print(f"  [smi] disabled: {err or 'a peer rank failed'}", flush=True)
            return None

        lockstep()
        t0 = time.perf_counter()
        for _ in range(rounds):
            for _ in range(batch):
                run_d()
                run_c()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        mon.stop()
        dist.barrier()

        want = max(1, int(SMI_DURATION / SMI_INTERVAL))
        metrics = mon.summary(start_s=t0, end_s=t1)
        inside = [s for s in mon.samples if t0 <= s["timestamp_s"] <= t1]
        # Count samples that CARRY THE CLOCK, not samples that exist. A metric
        # query that raises is swallowed so one bad read cannot stop the polling
        # thread, but the sample is still appended with only its timestamp -- so
        # counting samples would report a healthy "ok" for a rank whose clock was
        # never readable at all, which is the rank this telemetry exists to find.
        n_clk = sum(1 for s in inside if s.get("gfx_clk_mhz") is not None)
        if n_clk == 0:
            status = "no_metrics"
        elif n_clk >= max(2, want // 2):
            status = "ok"
        else:
            status = "insufficient"
        local = {
            "label": label,
            "device": torch.cuda.current_device(),
            "rank": rank,
            "interval_s": SMI_INTERVAL,
            "duration_s": t1 - t0,
            "launches": rounds * batch,
            "samples": len(inside),
            "clock_samples": n_clk,
            "sample_status": status,
            "metrics": metrics,
        }
        every = [None] * world
        dist.all_gather_object(every, local)
        if rank != 0:
            return None
        for r in every:
            _smi.emit(r)
        return every

    def case(ct, name, mode):
        """The columns that identify a point: the case arguments, aiter's order."""
        return {
            "backend": name,
            "mode": mode,
            "ep": world,
            "m": ct,  # tokens per rank, aiter's row key for problem size
            "hidden": HIDDEN,
            "topk": TOPK,
            "experts_per_rank": EPR,
            "dispatch_dtype": _DISP,
            "scale_dim": SCALE_DIM,
            "scale_type_size": SCALE_TS,
            "combine_dtype": "bf16",
            "combine_mode": COMB_MODE,
            "combine_in": COMBINE_IN,
            "data_init": INIT,
            "seed": SEED,
            "iters": ITERS,
        }

    def clocks(records):
        """Cross-rank clock columns. The MIN over ranks is the point of these:
        one throttled GPU sets the pair time for all of them, and a mean hides it.

        A rank whose clock never read is EXCLUDED from that min, so the status
        column has to say so: silently narrowing the min to the readable ranks
        would report the healthiest GPUs as if they were all of them.
        """
        if not records:
            return {}
        med = [
            r["metrics"]["gfx_clk_mhz"]["median"]
            for r in records
            if "gfx_clk_mhz" in r["metrics"]
        ]
        pwr = [
            r["metrics"]["power_w"]["median"]
            for r in records
            if "power_w" in r["metrics"]
        ]
        out = {}
        if med:
            out["gfx_clk_mhz"] = round(sum(med) / len(med), 1)
            out["gfx_clk_mhz_min"] = round(min(med), 1)
        if pwr:
            out["power_w"] = round(sum(pwr) / len(pwr), 1)
        seen = {r["sample_status"] for r in records}
        # Worst rank wins, and how many ranks the numbers above came from.
        for worst in ("no_metrics", "insufficient", "ok"):
            if worst in seen:
                break
        out["smi_status"] = worst
        out["smi_ranks"] = f"{len(med)}/{len(records)}"
        return out

    # Sequential phases, then the INSTANTs: Entry at the kernel's first statement, and
    # kEpPushSmSlot0 (8192) + CU id, which the Perfetto export names Unknown_Slot_<n>.
    _PROF_SEQ = {
        0: "Prologue",
        1: "Meta",
        2: "Load",
        3: "Store",
        4: "StoreWait",
        6: "DataPoll",
        7: "SigPoll",
        8: "Reduce",
        9: "Signal",
        27: "GoPoll",
    }
    _PROF_SLOTS = {
        **_PROF_SEQ,
        5: "Entry",
        10: "Timeout",
        **{11 + s: f"SigSrc{s}" for s in range(8)},
        **{19 + s: f"DataSrc{s}" for s in range(8)},
    }

    def profile_push(op, ct, i_, w_, x_, s_, buf):
        """PROF_N traced push combines. PROF_MODE=graph traces the last combine of a
        warmed-up PROF_R-pair graph replay; PROF_MODE=cold traces one combine per
        host-synced step, the ranks lock-stepped in between.

        wall_clock64 is a constant-rate clock whose rate gfx1250 does not report
        (hipDeviceAttributeWallClockRate reads 0), so it is timed against the host:
        consecutive traced launches are PROF_GAP s of host sleep apart, and the
        rate is first-event ticks over host seconds between the first and the
        last launch. Launch jitter is microseconds against that spread.
        """
        import numpy as np

        from mori.ops.dispatch_combine_v2.hip_backend import PUSH_PROF_EVENTS

        tb, to = getattr(op, "prof_time_buf", None), getattr(
            op, "prof_time_offset", None
        )
        if tb is None:
            if rank == 0:
                print(
                    "  [PROF] off: MORI_JIT_EXTRA_FLAGS has no -DENABLE_PROFILER",
                    flush=True,
                )
            return
        traces, host_t, bents = [], [], []
        wpb = op._pick(ct)[1][1]

        def split_replay(a):
            """One replay's ring [warps, E, 2] -> (the last launch's events only, block entry
            ticks [launch, block]). Every warp opens every launch with an Entry INSTANT, and
            the offset starts at 0 with no wrap, so ring order is launch order."""
            last = np.zeros_like(a)
            nb = a.shape[0] // wpb
            bent = np.full((PROF_R, nb), np.iinfo(np.int64).max, dtype=np.int64)
            for w in range(a.shape[0]):
                n = int((a[w, :, 0] != 0).sum())
                if n == 0:
                    continue
                if n >= PUSH_PROF_EVENTS:
                    return None, None  # the ring may have wrapped: launches overlap
                meta = a[w, :n, 1]
                starts = np.nonzero(
                    ((meta & 0x3) == 2) & (((meta >> 2) & 0x3FFF) == 5)
                )[0]
                if len(starts) != PROF_R:
                    return None, None
                last[w, : n - starts[-1]] = a[w, starts[-1] : n]
                b = w // wpb
                bent[:, b] = np.minimum(bent[:, b], a[w, starts, 0])
            return last, bent

        if PROF_MODE == "graph":
            # The timed loop's own shape: PROF_R (dispatch, combine) pairs in one graph, warmed
            # up, then replayed with the rings zeroed first. The last combine of a replay is the
            # one analysed -- by then the pipeline is hot, as in the bench's timing.
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(PROF_R):
                    *_, rr = op.dispatch(i_, w_, s_, x_, return_routing=True)
                    op.combine(buf, routing=rr)
            lockstep()
            for _ in range(WARMUP):
                g.replay()
            lockstep()
            failed = False
            for k in range(PROF_N):
                lockstep()
                if k:
                    time.sleep(PROF_GAP)
                tb.zero_()
                to.zero_()
                lockstep()
                host_t.append(time.perf_counter())
                g.replay()
                torch.cuda.synchronize()
                a = tb.cpu().numpy().reshape(-1, PUSH_PROF_EVENTS, 2)
                last, bent = split_replay(a)
                if last is None:
                    failed, last, bent = True, a, np.zeros((PROF_R, 1), dtype=np.int64)
                traces.append(last)
                bents.append(bent)
                lockstep()
            # Every rank takes the same exit, or the ones that stay wait on the others forever.
            flag = torch.tensor([1 if failed else 0])
            dist.all_reduce(flag)
            if flag.item():
                if rank == 0:
                    print(
                        f"  [PROF] ABORT: a warp's ring holds other than {PROF_R} launches",
                        flush=True,
                    )
                return
        for k in range(PROF_N if PROF_MODE != "graph" else 0):
            if PROF_WARM == 2:
                # As in the pipeline: the traced combine follows its dispatch on the stream,
                # with no host sync or idle gap between them. The gap goes before the dispatch.
                lockstep()
                if k:
                    time.sleep(PROF_GAP)
                lockstep()
                host_t.append(time.perf_counter())
                *_, r = op.dispatch(i_, w_, s_, x_, return_routing=True)
                tb.zero_()
                to.zero_()
                op.combine(buf, routing=r)
                torch.cuda.synchronize()
                traces.append(tb.cpu().numpy().reshape(-1, PUSH_PROF_EVENTS, 2))
                lockstep()
                continue
            *_, r = op.dispatch(i_, w_, s_, x_, return_routing=True)
            lockstep()
            tb.zero_()
            to.zero_()
            torch.cuda.synchronize()
            if k:
                time.sleep(PROF_GAP)
            host_t.append(time.perf_counter())
            if PROF_WARM:
                # The same kernel right before, on the same stream: its code and arguments are
                # warm when the traced launch starts. Only the trace-buffer resets sit between.
                op.combine(buf, routing=r)
                tb.zero_()
                to.zero_()
            op.combine(buf, routing=r)
            torch.cuda.synchronize()
            traces.append(tb.cpu().numpy().reshape(-1, PUSH_PROF_EVENTS, 2))
            lockstep()

        def first_ts(a):
            ts = a[:, :, 0]
            return int(ts[ts != 0].min())

        rate = None
        if PROF_N >= 2:
            rate = (first_ts(traces[-1]) - first_ts(traces[0])) / (
                (host_t[-1] - host_t[0]) * 1e6
            )  # ticks per us

        def stats(a):
            """Per-warp BEGIN/END pairs -> (span, send, {slot: (mean, max)}) ticks.

            INSTANT events (Entry, the CU id) are dropped before pairing. A phase's
            mean is over the warps that ran it: a warp with no slots never reaches
            Meta, and counting it as 0 dilutes every phase after the prologue."""
            wl = []
            for w in range(a.shape[0]):
                used = a[w, :, 0] != 0
                if not used.any():
                    continue
                ts, meta = a[w, used, 0], a[w, used, 1]
                keep = (meta & 0x3) != 2
                wl.append(
                    (
                        ts[keep],
                        (meta[keep] >> 2) & 0x3FFF,
                        meta[keep] & 0x3,
                        int(used.sum()),
                    )
                )
            if not wl:
                return None
            # A sequential trace is B,E,B,E,... from slot Prologue; anything else
            # means the ring wrapped or an event was lost.
            ok = all(
                len(t) > 0
                and len(t) % 2 == 0
                and s[0] == 0
                and (y[0::2] == 0).all()
                and (y[1::2] == 1).all()
                for t, s, y, _ in wl
            )
            t0 = min(int(t[0]) for t, _, _, _ in wl)
            t_end = max(int(t.max()) for t, _, _, _ in wl)
            got = {sid: [] for sid in _PROF_SEQ}
            meta_begin = None
            for t, s, _, _ in wl:
                for i in range(0, len(t) - 1, 2):
                    sid = int(s[i])
                    if sid in got:
                        got[sid].append(int(t[i + 1] - t[i]))
                    if sid == 1:
                        meta_begin = (
                            int(t[i])
                            if meta_begin is None
                            else min(meta_begin, int(t[i]))
                        )
            per = {
                sid: (float(np.mean(v)), float(np.max(v))) if v else (0.0, 0.0)
                for sid, v in got.items()
            }
            send = t_end - meta_begin if meta_begin is not None else 0
            return t_end - t0, send, per, ok, len(wl), max(n for *_, n in wl)

        rows_ = [stats(a) for a in traces]
        # Graph mode: per launch of every replay, the blocks entering > 1 us after the median
        # block. Launch 1 follows the ring reset and the host's idle gap; 2..R are the hot ones.
        late = None
        if bents and rate:
            per, worst_hot = [], 0.0
            for bent in bents:
                row = []
                for j in range(bent.shape[0]):
                    e = bent[j][bent[j] != np.iinfo(np.int64).max]
                    med = float(np.median(e))
                    row.append(int((e > med + rate).sum()))
                    if j:
                        worst_hot = max(worst_hot, float((e.max() - med) / rate))
                per.append(row)
            hot = [c for row in per for c in row[1:]]
            late = dict(
                last=per[-1], hit=sum(1 for c in hot if c), n=len(hot), worst=worst_hot
            )
        mine = None
        if rate and all(x is not None for x in rows_):
            med = lambda xs: float(np.median(xs))  # noqa: E731
            mine = dict(
                rank=rank,
                rate=rate,
                span=med([x[0] for x in rows_]) / rate,
                send=med([x[1] for x in rows_]) / rate,
                per={
                    sid: (
                        med([x[2][sid][0] for x in rows_]) / rate,
                        med([x[2][sid][1] for x in rows_]) / rate,
                    )
                    for sid in _PROF_SEQ
                },
                ok=all(x[3] for x in rows_),
                cus=getattr(op, "_multi_processor_count", -1),
                late=late,
                warps=rows_[-1][4],
                maxev=max(x[5] for x in rows_),
                t0=first_ts(traces[-1]),
            )
        if PROF_OUT and rate:
            from mori.kernel_profiler import export_to_perfetto

            os.makedirs(PROF_OUT, exist_ok=True)
            export_to_perfetto(
                torch.from_numpy(traces[-1].reshape(-1).copy()),
                os.path.join(PROF_OUT, f"push_ct{ct}_rank{rank}.json"),
                slot_map=_PROF_SLOTS,
                gpu_freq_ghz=rate / 1000.0,
            )
        every = [None] * world
        dist.all_gather_object(every, mine)
        if rank != 0:
            return
        for m in every:
            if m is None:
                print(f"  [PROF] ct={ct} a rank produced no trace", flush=True)
                continue
            ph = " ".join(
                f"{_PROF_SLOTS[s]}={a:.2f}/{b:.2f}"
                for s, (a, b) in m["per"].items()
                if b > 0
            )
            print(
                f"  [PROF] ct={ct} rank={m['rank']} tick_mhz={m['rate']:.3f} "
                f"span={m['span']:.2f} send={m['send']:.2f} warps={m['warps']} "
                f"maxev={m['maxev']} seq_ok={int(m['ok'])} cus={m['cus']} | {ph} "
                f"(us; per-warp mean/max; median of {PROF_N}; mode={PROF_MODE})",
                flush=True,
            )
            lt = m.get("late")
            if lt:
                print(
                    f"  [PROFLATE] ct={ct} rank={m['rank']} hot launches with a block entering "
                    f">1us after the median: {lt['hit']}/{lt['n']} worst={lt['worst']:.2f}us | "
                    f"last replay, late blocks per launch 1..{PROF_R}: {lt['last']}",
                    flush=True,
                )
        # The traced launch's first event on each rank, against the first rank's. Only a time if
        # the ranks' wall clocks share an origin; the trace files each start at their own zero.
        got = [m for m in every if m is not None]
        if got:
            base = got[0]["t0"] / got[0]["rate"]
            skew = " ".join(
                f"r{m['rank']}={m['t0'] / m['rate'] - base:+.2f}" for m in got
            )
            print(
                f"  [PROFT0] ct={ct} first event minus rank {got[0]['rank']}'s, us: {skew}",
                flush=True,
            )

    rows = []
    failures = checked = points = 0
    for ct in SWEEP:
        i_, w_, x_ = inp[:ct], wts[:ct], idx[:ct]
        s_ = scales[:ct] if SCALE_DIM else None
        for name, op in ops.items():
            points += 1
            total, buf, ok, was_checked = prime(op, ct, i_, w_, x_, s_)
            checked += was_checked
            if not ok:
                failures += 1
                # A failed point stays in the table as a row with err_msg, so a
                # gap in the sweep cannot be mistaken for a tier nobody ran.
                rows += [
                    dict(case(ct, name, mode), err_msg="correctness check failed")
                    for mode in MODES
                ]
                continue  # never report bandwidth for a kernel computing garbage

            def one_pair():
                """A layer's two all2all legs, in order, nothing in between."""
                *_, r = op.dispatch(i_, w_, s_, x_, return_routing=True)
                op.combine(buf, routing=r)

            def capture_pair():
                """One graph per leg, so the pair still alternates on replay. The
                dispatch graph rewrites the same dest_map every time and the
                combine graph was captured against that handle."""
                gd = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gd):
                    *_, r_cap = op.dispatch(i_, w_, s_, x_, return_routing=True)
                lockstep()
                gc = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gc):
                    op.combine(buf, routing=r_cap)
                lockstep()
                return gd, gc

            held = [None]  # eager's combine needs the handle its dispatch produced

            def eager_d():
                *_, r = op.dispatch(i_, w_, s_, x_, return_routing=True)
                held[0] = r

            def eager_legs():
                return eager_d, lambda: op.combine(buf, routing=held[0])

            for mode in MODES:
                d_us, c_us, run_d, run_c = time_pairs(
                    mode,
                    one_pair,
                    capture_pair if mode == "graph" else eager_legs,
                    eager_legs,
                )
                got = torch.tensor([d_us, c_us, float(total)], dtype=torch.float64)
                dist.all_reduce(got)
                n = world
                d_us_m, c_us_m = float(got[0]) / n, float(got[1]) / n
                recv_m = float(got[2]) / n
                # Bytes off one rank over that leg's time, BOTH cross-rank means,
                # so the reported bandwidth follows from the recv_tokens and us
                # this same row reports. Mixing a local byte count with a mean
                # time gave a row whose columns did not agree with each other,
                # and recv counts vary between ranks with the routing. The legs
                # differ whenever dispatch is narrower than combine.
                d_bw = (
                    recv_m
                    * (HIDDEN * _DISP_NBYTES + SCALE_DIM * SCALE_TS)
                    / (1000**3)
                    / (d_us_m / 1e6)
                )
                c_row = _PUSH_WIRE if _PUSH else HIDDEN * 2
                c_bw = recv_m * c_row / (1000**3) / (c_us_m / 1e6)
                if rank == 0:
                    print(
                        f"  ct={ct:<5d} [{name}/{mode}] "
                        f"dispatch {got[0]/n:7.1f} us ({d_bw:6.1f} GB/s)  "
                        f"combine {got[1]/n:7.1f} us ({c_bw:6.1f} GB/s)  "
                        f"pair {(got[0]+got[1])/n:7.1f} us  recv~{got[2]/n:.0f}",
                        flush=True,
                    )
                lockstep()
                smi = smi_window(
                    f"bench_ep/dispatch_combine/backend={name}/mode={mode}"
                    f"/M={ct}/disp={_DISP}",
                    d_us_m + c_us_m,
                    run_d,
                    run_c,
                )
                lockstep()
                if rank == 0:
                    rows.append(
                        dict(
                            case(ct, name, mode),
                            recv_tokens=round(recv_m),
                            dispatch_us=round(d_us_m, 2),
                            combine_us=round(c_us_m, 2),
                            pair_us=round(d_us_m + c_us_m, 2),
                            dispatch_gbps=round(d_bw, 1),
                            combine_gbps=round(c_bw, 1),
                            verified=VERIFY_SCOPE,
                            **clocks(smi),
                        )
                    )
                if PROFILE and _PUSH and mode == MODES[-1]:
                    profile_push(op, ct, i_, w_, x_, s_, buf)
                    lockstep()

    if rank == 0:
        # Say how many points were verified, not just that none failed -- with
        # CHECK=0 or DISP=fp4 nothing is compared, and a skipped check is not a
        # passing one.
        # Say what was actually verified. fp4 skips the identity-expert check
        # (hip has no fp4 combine) but its dispatch bytes ARE compared.
        if _PUSH and CHECK and not _data.verifies_nothing(INIT):
            why = " (push: dispatch bytes + landed push bytes)"
        elif _FP4:
            why = " (fp4: dispatch bytes only, combine not compared)"
        elif not CHECK:
            why = " (CHECK=0)"
        elif _data.verifies_nothing(INIT):
            why = f" ({INIT} payload: dispatch bytes only, combine check is vacuous)"
        else:
            why = ""
        print(
            f"# {'FAIL' if failures else 'PASS'}: {failures} failed, "
            f"{checked}/{points} points verified{why}"
        )
        if JSON:
            _report.print_json_table("mori ep dispatch_combine_v2 summary", rows)
    for op in ops.values():
        op.close()
    comm.destroy()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

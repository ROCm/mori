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
CHECK = int(os.environ.get("CHECK", 1))
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
# What the correctness gate covers, as one token. A bool cannot say it: fp4 and
# an all-zero payload verify the dispatch bytes but never compare combine's
# output, and a row claiming "verified" next to a combine_us nobody checked is
# the machine-readable half losing what the human summary spells out.
VERIFY_SCOPE = (
    "none"
    if not CHECK
    else (
        "dispatch_bytes"
        if _FP4 or _data.verifies_nothing(INIT)
        else "dispatch_bytes+combine"
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
            kernel_backend=backend,
            dispatch_block_num=_G["DBN"],
            warp_num_per_block=_G["DWPB"],
            combine_block_num=_G["CBN"],
            combine_warp_num_per_block=_G["CWPB"],
        )
        return EpDispatchCombineOp(cfg, comm)

    ops = {b: build(b) for b in BACKENDS}

    if rank == 0:
        print(
            f"# EP{world} hidden={HIDDEN} topk={TOPK} epr={EPR} "
            f"init={INIT} seed={SEED} "
            f"disp={_DISP_DT} comb=bf16 backends={BACKENDS} modes={MODES} "
            f"iters={ITERS} combine_in={COMBINE_IN} check={CHECK}",
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

    def prime(op, ct, i_, w_, x_):
        """One full pair, untimed. Reads total_recv for the host, builds the buffer
        the timed loop will reuse, and with CHECK verifies the result through that
        same buffer -- so the gate covers exactly what gets timed, staged copy
        included. Must be a PAIR: a bare dispatch would leave total_recv set, and
        dispatch accumulates into it while only combine clears it, so the next
        combine would stage twice the tokens and run past the arena.
        Returns (total_recv, buf, ok, checked)."""
        *_, total_t, r = op.dispatch(i_, w_, None, x_, return_routing=True)
        lockstep()  # the reverse map is only valid after this barrier
        total = int(total_t.cpu().item())
        dispatch_bad = check_dispatch(op, total, r)
        stage = op.combine_in_view()[:total]
        # An all-zero payload reduces the identity-expert check to 0 == 0, which
        # holds however wrong the kernel is. fp4 cannot go through combine at all
        # (hip has no fp4 combine), so for it check_dispatch is the whole story.
        checked = bool(CHECK) and not _FP4 and not _data.verifies_nothing(INIT)
        if checked:  # identity expert: stage the dispatched tokens unchanged
            stage.copy_(op.recv_tokens()[:total].to(stage.dtype))
        buf = stage.clone() if COMBINE_IN == "staged" else stage
        out, _ = op.combine(buf, routing=r)
        lockstep()
        if dispatch_bad:
            return total, buf, False, True
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

        Timed three ways, and which one gets returned is on the [E2E] line as src=. Under a
        graph it is the reading whose event records live inside the graph, because recording
        them from the host every iteration adds 9.3 us to every pair -- see the comments
        further down for what each way costs and what is left in the one that is reported.

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

        # Three events per iteration, not four: the middle one ends dispatch and starts
        # combine. Recording two back to back there left the time the GPU spent on those
        # two packets out of both legs, so d + c came to less than the pair took.
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

        # The floor: the same pairs with nothing recorded between them. E2E_R of them go into one
        # graph, one event pair wraps a whole replay, so a pair carries 1/R of a replay and 2/R
        # of an event instead of two replays and three events. It cannot say what either leg
        # took, only what a pair costs when nobody is watching, which is what the per-leg
        # readings get measured against.
        #
        # What the per-iteration method adds is not small and is not a property of the kernel.
        # On gfx1250 EP4 at 512 tokens it is 9.1-9.3 us per pair at bf16, fp8 and fp4 alike, and
        # cutting the dispatch kernel short until it returns on entry -- fifty times less work --
        # moves it by under 1 us. Before #576 this file timed ITERS calls under a single event
        # pair and paid none of it; that pass also kept the legs apart, so alternating pairs and
        # cheap timing were never in conflict.
        #
        # The pairs go inside the capture, never into a host loop. Submitting them back to back
        # from the host deadlocks the full kernel for the reason this function's own docstring
        # gives: a rank that starts call N+1 before every rank finished N overwrites a barrier
        # flag its peer has not consumed. Inside a graph the order is fixed.
        e2e = slope = -1.0
        rep = int(os.environ.get("E2E_R", 10))

        def graph_of(npairs):
            """A warmed-up graph holding npairs back-to-back pairs."""
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
            """Mean wall time of one replay of g, in us.

            n replays, each under its own event pair, submitted back to back the way
            the timed loop above submits its iterations. One replay was a single
            sample of a quantity the per-iteration reading averages over ITERS, and
            it moved several us between rounds, which is enough to swamp what this is
            measuring."""
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
            # R and 2R alternate rather than one running after the other. Taken in
            # sequence, whatever the machine drifts over those hundreds of replays lands
            # entirely in the slope below, where dividing by R multiplies it: measured
            # that way the slope moved 6.5 us between rounds while the two readings it is
            # built from moved 1.8.
            for _ in range(alt):
                t_r += replay_us(g_r, n_e2e)
                t_2r += replay_us(g_2r, n_e2e)
            t_r, t_2r = t_r / alt, t_2r / alt
            e2e = t_r / rep
            # Dividing one replay by R leaves 1/R of the launch and 1/R of the event pair
            # inside every pair; the difference between R and 2R is R pairs and nothing
            # else, so its slope is what one more pair costs once it is nobody's first.
            # Both are reported: e2e is the convention the numbers already on record were
            # measured against, and it is the steadier of the two because it is one
            # measurement rather than a difference of two.
            slope = (t_2r - t_r) / rep
        # What this function reports, when the mode allows it: the same R pairs in one graph,
        # with the event records captured INTO the graph instead of issued from the host once
        # per iteration. The order is unchanged and the legs stay separable; what changes is
        # that a timestamp costs a graph node rather than a host API call and a queue packet.
        # Against the floor above, on a08-1 at 512 tokens across bf16, fp8 and fp4, over 15
        # rounds that each held the cards alone, this reads 5.3 us per pair where the
        # per-iteration method reads 9.7.
        #
        # Those 5 us are the event nodes themselves and do not go away by batching: taking R
        # from 10 to 40 left them at 5.5 -> 5.0 us per pair, within the round-to-round spread,
        # where a cost paid once per replay would have dropped to a quarter. Three nodes a pair
        # at roughly 1.6 us each. Below this needs a timestamp that never leaves the GPU, i.e.
        # wall_clock64 inside the kernels (MORI_EP_SEGTIME), which cannot see the launch at all
        # and which makes the correctness check fail on 17-23% of runs.
        #
        # Both legs pay it. Over 15 rounds with the cards to ourselves, combine's two readings
        # differ by 2.38-3.25 us and dispatch's by -0.05-4.31, and over the ten fp8 and fp4
        # rounds not one put dispatch near zero. An earlier twelve read 3 us on combine and
        # nothing on dispatch, which was dispatch's own spread: that spread is still wider
        # than the gap between the legs, so whether they pay the same is not answerable here.
        #
        # Amortizing per leg is not an option and this is why: dispatch accumulates into
        # total_recv and only combine clears it (see prime), so R dispatches captured back to
        # back would leave the next combine staging R times the tokens and running past the
        # arena. Whatever replaces the per-iteration probes has to keep the pairs paired.
        gd_us = gc_us = -1.0
        gd_lo = gd_hi = gc_lo = gc_hi = float("nan")
        evflags = 0
        gevr = int(os.environ.get("GEV_R", 20))
        hip = None
        if mode == "graph" and gevr > 0:
            import ctypes

            # An event recorded from inside a capture becomes an internal graph node, and an
            # internal node's timestamp is not readable from the host: elapsed_time on one
            # returns hipErrorInvalidHandle. The flag that makes it readable is
            # hipEventRecordExternal, which torch.cuda.Event.record does not expose, so record
            # through the HIP runtime instead. Where that library is not there, say so and fall
            # back -- a reading 5 us off is worth having, a silent switch between two of them is
            # not.
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

            # Recording an event writes back and invalidates cache at system scope, and that is
            # charged twice: once to the node, once to the kernel behind it that now starts cold.
            # hip_runtime_api.h says as much where it defines hipEventDisableSystemFence
            # (0x20000000) -- "can improve the accuracy of timing measurements by avoiding the
            # cost of cache writeback and invalidation, and the performance impact of those
            # actions on the execution of following work" -- with hipEventReleaseToDevice
            # (0x40000000) as the narrower version that keeps a device-scope release. These
            # events feed nothing but hipEventElapsedTime, which is the case both are documented
            # for. torch.cuda.Event does not expose either, hence building them here.
            #
            # 0 is what torch.cuda.Event(enable_timing=True) creates, so it is the default and
            # the arm to compare against. It also means switching to hand-built events on its
            # own changes no reading.
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
            # The events are rewritten by every replay, so each one has to be read back before
            # the next. That sync sits between replays, not between the legs, so it cannot land
            # inside either reading -- what it does cost is that every replay starts against an
            # empty queue instead of a full one.
            n_gev = int(os.environ.get("GEV_N", 20))
            # The first pair of every replay is dropped. That read-back has to synchronize,
            # so every replay starts against an empty queue and its first dispatch waits out
            # the graph launch, which the other R-1 do not. Averaged in at R=10 it put a
            # tenth of a launch into dispatch and nothing into combine, a bias on one leg
            # only. Dropping it took dispatch's round-to-round spread from 3.6 us to between
            # 0.44 and 2.4 depending on dtype.
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
            # Reported so a reader can tell a 2 us effect from a 2 us spread without
            # rerunning. Each sample here is already a mean over R-1 pairs, so this is the
            # spread of replay-to-replay drift, not of single pairs.
            gd_lo, gd_hi = min(ds), max(ds)
            gc_lo, gc_hi = min(cs), max(cs)

            # A mean over the replays lets a few disturbed ones carry the whole run. Measured
            # 2026-09-15 on an empty kernel: the clean runs hold every replay inside 0.10 us,
            # and the runs that read high have the same floor -- 3.98 against the clean 3.91 --
            # with a few replays up at 5.36 doing all the lifting. Across eight rounds that
            # turned into 3 us of run-to-run swing on a reading whose real spread is 0.1, and
            # any effect smaller than the swing was unmeasurable.
            #
            # Both are reported until one round has been run with both in hand, because
            # switching the number that gets quoted is not something to do on an argument.
            def quart(xs):
                s = sorted(xs)
                n = len(s)
                mid = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
                return mid, s[n // 4], s[(3 * n) // 4]

            gd_med, gd_p25, gd_p75 = quart(ds)
            gc_med, gc_p25, gc_p75 = quart(cs)
        # Report the graph-event reading where there is one, and say so on the diagnostic line
        # rather than silently: two numbers 3 us apart that are both labelled "combine" is
        # exactly the kind of thing that gets compared across runs months later.
        d, c = (gd_us, gc_us) if gd_us > 0 else (hd, hc)
        # One rank prints. Four ranks each writing a long line is how torn output gets spliced
        # into a reading that parses and is wrong.
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
                # On the line, not only in the script that set it: an arm whose flag never
                # reached the runtime prints a table of the same shape as one where it did.
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
            "combine_dtype": "bf16",
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

    rows = []
    failures = checked = points = 0
    for ct in SWEEP:
        i_, w_, x_ = inp[:ct], wts[:ct], idx[:ct]
        for name, op in ops.items():
            points += 1
            total, buf, ok, was_checked = prime(op, ct, i_, w_, x_)
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
                *_, r = op.dispatch(i_, w_, None, x_, return_routing=True)
                op.combine(buf, routing=r)

            def capture_pair():
                """One graph per leg, so the pair still alternates on replay. The
                dispatch graph rewrites the same dest_map every time and the
                combine graph was captured against that handle."""
                gd = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gd):
                    *_, r_cap = op.dispatch(i_, w_, None, x_, return_routing=True)
                lockstep()
                gc = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gc):
                    op.combine(buf, routing=r_cap)
                lockstep()
                return gd, gc

            held = [None]  # eager's combine needs the handle its dispatch produced

            def eager_d():
                *_, r = op.dispatch(i_, w_, None, x_, return_routing=True)
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
                d_bw = recv_m * HIDDEN * _DISP_NBYTES / (1000**3) / (d_us_m / 1e6)
                c_bw = recv_m * HIDDEN * 2 / (1000**3) / (c_us_m / 1e6)
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

    if rank == 0:
        # Say how many points were verified, not just that none failed -- with
        # CHECK=0 or DISP=fp4 nothing is compared, and a skipped check is not a
        # passing one.
        # Say what was actually verified. fp4 skips the identity-expert check
        # (hip has no fp4 combine) but its dispatch bytes ARE compared.
        if _FP4:
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

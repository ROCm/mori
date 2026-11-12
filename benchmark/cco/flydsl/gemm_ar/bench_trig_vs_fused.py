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
"""Host-armed triggered GEMM + all-reduce against mori's fused one, same mxfp8 GEMM.

One main loop -- ``compile_fused_gemm_scatter`` at mxfp8, 256x256 tiles, the
production store options -- under different epilogues and collectives
(``--methods``, all by default):

* ``fused``       ``GemmAllReduceOp(quant="mxfp8")`` as SGLang builds it: the
                  scatter fused into the GEMM's epilogue, then drain / reduce /
                  gather (bf16 gather over LSA, one SDMA queue)
* ``trig``        ``GemmTriggeredAllReduceOp`` driven by the same GEMM compiled
                  with ``gate_tail``: the host arms the copy-engine chains
                  (``arm()``, timed apart), then begin / GEMM / complete
* ``trig-agent``  the same with an agent-scope gate release
* ``trig-conc``   ``trig`` with ``reduce_mode="concurrent"``
* ``split``       the plain GEMM, then ``torch.distributed.all_reduce`` (RCCL)

and the GEMM alone, which prices the gate epilogue: ``gemm`` (plain, into an
ordinary tensor), ``gemm-win`` (plain, into an uncached cco window, as the ops
write), ``gemm-gate`` / ``gemm-gate-win`` (system-scope gate tail, into the
tensor / the window), ``gemm-gate-agent-win`` (agent-scope gate tail, window).

Inputs are exact: A has 32 entries of +-1 per row, B is +-1, and every ue8m0
scale is 2^0, so each rank's partial is an integer of magnitude <= 32 and the
sum over up to 8 ranks is exact in bf16. Every timed output is compared bit for
bit with the fp32 reference; the triggered op also audits every gate
(``op.check()``).

Timing: before each call a host barrier and a ~2 ms hold kernel, so every launch
is queued before the region starts; GPU events around the region; a call's time
is its slowest rank's. Per round, the median over ``--iters`` calls; methods
interleave round by round; reported: the median over rounds and each round.
``--breakdown N`` also times each kernel of ``fused`` and ``trig`` (after mode)
over N calls -- per segment, the slowest and the fastest rank.

    # DeepSeek wo_b, TP8
    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \\
        bench_trig_vs_fused.py --rows 2048,4096,8192,16384 --breakdown 20
    # DeepSeek-V4.1-Flash wo_b, TP4
    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=4 \\
        bench_trig_vs_fused.py --features 5120 --rows 2048,4096,8192,16384

``--reducer-blocks`` sets the triggered op's reducer work-groups (default 32);
in "after" mode 128-256 shrinks its gap at small M on MI355X.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import flydsl.expr as fx
import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ar import (
    GateProducerSpec,
    GemmAllReduceOp,
    GemmTriggeredAllReduceOp,
    preshuffle_a_scale,
    preshuffle_b,
)
from mori.ops.gemm_ar.gates import GATE_STRIDE
from mori.ops.gemm_ar.kernels_fused import compile_fused_gemm_scatter
from mori.ops.gemm_ar.layout import ArConfig
from mori.ops.gemm_ar.op import _PinnedLaunch
from mori.ops.gemm_ar.triggered import TriggeredArLayout
from mori.tensor_utils import from_gpu_ptr

BLOCK = 256  # the mxfp8 GEMM's tile, both ways
NNZ = 32  # +-1 entries per row of A
MX_BK = 32  # ue8m0 block along K
UNIT_EXP = 127  # ue8m0 byte for 2^0
METHODS = (
    "fused",
    "trig",
    "trig-agent",
    "trig-conc",
    "split",
    "gemm",
    "gemm-win",
    "gemm-gate",
    "gemm-gate-win",
    "gemm-gate-agent-win",
)


def _setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world, UniqueId.from_bytes(payload[0])


def _weight(rank: int, n: int, k: int) -> torch.Tensor:
    """This rank's +-1 weight shard [n, k], fp8."""
    g = torch.Generator(device="cuda").manual_seed(900_001 + 7 * rank)
    w = torch.randint(0, 2, (n, k), generator=g, device="cuda", dtype=torch.int8)
    return (w.float() * 2 - 1).to(torch.float8_e4m3fn)


def _activation(rank: int, m: int, k: int) -> torch.Tensor:
    """This rank's [m, k] fp8 activation: 32 entries of +-1 per row, rest 0."""
    g = torch.Generator(device="cuda").manual_seed(500_009 + 13 * rank + m)
    stride = k // NNZ
    rows = torch.arange(m, device="cuda")[:, None]
    cols = (rows + rank) % stride + stride * torch.arange(NNZ, device="cuda")[None, :]
    signs = torch.randint(0, 2, (m, NNZ), generator=g, device="cuda").float() * 2 - 1
    a = torch.zeros(m, k, device="cuda").scatter_(1, cols, signs)
    return a.to(torch.float8_e4m3fn)


def _scales(m: int, n: int, k: int):
    """Unit ue8m0 scales in the layouts the kernel reads (see bench_gemm_ar)."""
    sa = torch.full((m, k // MX_BK), UNIT_EXP, dtype=torch.uint8, device="cuda")
    sb = torch.full(
        (n // MX_BK, k // MX_BK), UNIT_EXP, dtype=torch.uint8, device="cuda"
    )
    return preshuffle_a_scale(sa), sb.to(torch.int32).t().reshape(-1).contiguous()


def _hold_cycles(ms: float) -> int:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    took, cycles = [], 1_000_000
    for _ in range(5):
        e0.record()
        torch.cuda._sleep(cycles)
        e1.record()
        torch.cuda.synchronize()
        took.append(e0.elapsed_time(e1))
    return int(cycles * ms / statistics.median(took))


def _gemm_launcher(n: int, k: int, *, gate: bool, scope: str = "system"):
    """The production mxfp8 GEMM (as GemmAllReduceOp compiles it), unfused,
    optionally ending each block with the trigger gate epilogue."""
    cfg = ArConfig(world_size=2, m=BLOCK, n=n)  # unused with fuse=False
    return _PinnedLaunch(
        compile_fused_gemm_scatter(
            cfg,
            0,
            K=k,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            b_preshuffled=True,
            fuse=False,
            quant="mxfp8",
            swap_ab=True,
            permlane=True,
            lane_transpose=True,
            gate_tail=gate,
            gate_scope=scope,
        )
    )


def _breakdown(
    args, ops, comm, m, n, k, hold, a_i8, b_i8, sa, sb, ref, run_gated, gated, stream
):
    """GPU time of each kernel of one call, for ``fused`` and ``trig`` (after mode).

    The calls are expanded into their launches exactly as the ops issue them,
    with an event between consecutive kernels; a barrier and the hold kernel go
    first, so the events time the kernels, not the host. Per segment: the median
    over calls of the slowest rank's and of the fastest rank's time -- a kernel
    that mostly waits for its peers shows a wide gap between the two."""
    from mori.ops.gemm_ar.op import _flatten_mxfp8_a_scale

    def timed(names, launch_all, out, prep=None):
        rows, exact = [], 0
        for it in range(args.warmup + args.breakdown):
            if prep is not None:
                prep()
            out.fill_(float("nan"))
            comm.barrier()
            torch.cuda._sleep(hold)
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
            evs[0].record()
            launch_all(evs)
            torch.cuda.synchronize()
            if it >= args.warmup:
                rows.append(
                    [evs[i].elapsed_time(evs[i + 1]) * 1e3 for i in range(len(names))]
                )
                exact += int(torch.equal(out, ref))
        t = torch.tensor(rows, dtype=torch.float64)
        hi, lo = t.clone(), t.clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        ok = torch.tensor([exact, len(rows)], dtype=torch.int64)
        dist.all_reduce(ok, op=dist.ReduceOp.SUM)
        segs = {}
        for i, name in enumerate(names):
            segs[name] = [
                round(statistics.median(hi[:, i].tolist()), 1),
                round(statistics.median(lo[:, i].tolist()), 1),
            ]
        total = [
            round(statistics.median(hi.sum(1).tolist()), 1),
            round(statistics.median(lo.sum(1).tolist()), 1),
        ]
        return {
            "segments_max_min_us": segs,
            "sum_max_min_us": total,
            "exact": f"{int(ok[0])}/{int(ok[1])}",
        }

    out = {}
    if "fused" in ops:
        op_f = ops["fused"]
        plan = op_f._compiled(m)
        a_scale_k = _flatten_mxfp8_a_scale(sa, m, k)
        b_scale_k = sb.reshape(-1)
        names = ["gemm+scatter"] + list(plan.parts[op_f._FUSED_ORDER])

        def launch_fused(evs):
            plan.gemm(
                a_i8,
                b_i8,
                plan.input.view(-1),
                a_scale_k,
                b_scale_k,
                m,
                n,
                op_f.dev_comm.ptr,
                op_f.win.handle,
                stream=stream(),
            )
            evs[1].record()
            for i, phase in enumerate(plan.tail):
                phase(op_f.dev_comm.ptr, op_f.win.handle, stream=stream())
                evs[i + 2].record()

        out["fused"] = timed(names, launch_fused, plan.output)
    if "trig" in ops:
        op_t = ops["trig"]
        gate_base = op_t.base + op_t.layout.rs_gate_off
        reduce, wait = op_t._compiled(m)

        def launch_trig(evs):
            run_gated(op_t.input_view(m), gate_base, gated)
            evs[1].record()
            # complete(m) in "after" mode, with an event between its two kernels.
            op_t._armed_m = None
            op_t._begun = False
            reduce(op_t.base, op_t.wait_polls, stream=stream())
            evs[2].record()
            wait(op_t.base, op_t.wait_polls, stream=stream())
            evs[3].record()
            op_t._err_host.copy_(op_t._err_view(), non_blocking=True)
            op_t._err_ready = torch.cuda.Event()
            op_t._err_ready.record(torch.cuda.current_stream())

        out["trig"] = timed(
            ["gate GEMM (+RS)", "reduce", "wait (AG)"],
            launch_trig,
            op_t.output_view(m),
            prep=lambda: op_t.arm(m),
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Not "--n": torchrun would take it for an abbreviation of its own --nnodes etc.
    p.add_argument("--features", type=int, default=7168, help="N per rank")
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--rows", default="2048,4096,8192,16384", help="M per rank")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--hold-ms", type=float, default=2.0)
    p.add_argument("--reducer-blocks", type=int, default=32)
    p.add_argument("--json", default="")
    p.add_argument(
        "--breakdown",
        type=int,
        default=0,
        help="also time each kernel of fused and trig over this many calls",
    )
    args = p.parse_args()

    rank, world, uid = _setup()
    n, k = args.features, args.k
    ms = [int(x) for x in args.rows.split(",")]
    methods = [x for x in args.methods.split(",") if x]
    m_max = max(ms)
    for m in ms:
        if m % (world * BLOCK):
            raise SystemExit(
                f"M={m} must be a multiple of world*{BLOCK}={world * BLOCK}"
            )

    producer = GateProducerSpec(mt0=BLOCK, mt1=BLOCK)
    lay_bytes = TriggeredArLayout(
        world, n, m_max, BLOCK, producer.publishers(n)
    ).window_bytes
    a_bytes = GemmAllReduceOp.window_bytes_for(
        world, m_max=m_max, n=n, k=k, quant="mxfp8"
    )
    c_bytes = (m_max * n * 2 + 4095) // 4096 * 4096  # an uncached C for the -win GEMMs
    g_bytes = m_max // BLOCK * GATE_STRIDE
    vmm = 2 * a_bytes + 2 * 2 * lay_bytes + 2 * (c_bytes + g_bytes) + (512 << 20)

    hold = _hold_cycles(args.hold_ms)
    b = _weight(rank, n, k)
    b_shuf = preshuffle_b(b)
    b_i8 = b_shuf.contiguous().view(torch.int8).view(-1)
    plain = _gemm_launcher(n, k, gate=False)
    gated = _gemm_launcher(n, k, gate=True)
    gated_agent = _gemm_launcher(n, k, gate=True, scope="agent")
    scratch_gates = torch.zeros(
        m_max // BLOCK * GATE_STRIDE // 8, dtype=torch.int64, device="cuda"
    )
    rows_out = []

    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        ops = {}
        if "fused" in methods:
            ops["fused"] = GemmAllReduceOp(comm, n=n, k=k, m_max=m_max, quant="mxfp8")
        for name, mode in (("trig", "after"), ("trig-conc", "concurrent")):
            if name in methods or (name == "trig" and "trig-agent" in methods):
                # A generous wait bound: FlyDSL compiles each M's kernels on first
                # use, at different times on different ranks, and a default bound
                # can expire while a peer is still compiling. A wait ends as soon
                # as its condition holds, so the bound does not change any timing.
                ops[name] = GemmTriggeredAllReduceOp(
                    comm,
                    n=n,
                    m_max=m_max,
                    producer=producer,
                    reducer_blocks=args.reducer_blocks,
                    reduce_mode=mode,
                    wait_polls=1 << 30,
                )
                ops[name].self_test()
        if "fused" in ops:
            ops["fused"].self_test()
        # Uncached C and gates (a cco window, like the ops' own) for the -win GEMMs.
        scratch = comm.alloc_mem(c_bytes + g_bytes)
        from_gpu_ptr(scratch.ptr, (c_bytes + g_bytes,), torch.uint8).zero_()
        win_gates = scratch.ptr + c_bytes
        # Compile both GEMMs on every rank before any of them drives a collective:
        # a peer still compiling its GEMM looks like a hung producer.
        warm_m = world * BLOCK
        wa = _activation(rank, warm_m, k).contiguous().view(torch.int8).view(-1)
        wsa, wsb = _scales(warm_m, n, k)
        wout = torch.empty(warm_m, n, dtype=torch.bfloat16, device="cuda")
        for fn, gate_base in (
            (plain, 0),
            (gated, scratch_gates.data_ptr()),
            (gated_agent, win_gates),
        ):
            fn(
                wa,
                b_i8,
                wout.view(-1),
                wsa,
                wsb,
                warm_m,
                n,
                0,
                gate_base,
                stream=fx.Stream(torch.cuda.current_stream()),
            )
        torch.cuda.synchronize()
        comm.barrier()
        del wa, wsa, wsb, wout
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )

        for m in ms:
            a = _activation(rank, m, k)
            a_i8 = a.contiguous().view(torch.int8).view(-1)
            sa, sb = _scales(m, n, k)
            mine = (a.float() @ b.float().t()).to(torch.bfloat16)
            ref = torch.zeros(m, n, device="cuda")
            for r in range(world):
                ref += _activation(r, m, k).float() @ _weight(r, n, k).float().t()
            ref = ref.to(torch.bfloat16)
            tmp = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            stream = lambda: fx.Stream(torch.cuda.current_stream())  # noqa: E731

            def run_plain(out=tmp):
                plain(a_i8, b_i8, out.view(-1), sa, sb, m, n, 0, 0, stream=stream())

            def run_gated(out, gate_base, fn=gated):
                fn(
                    a_i8,
                    b_i8,
                    out.view(-1),
                    sa,
                    sb,
                    m,
                    n,
                    0,
                    gate_base,
                    stream=stream(),
                )

            c_win = from_gpu_ptr(scratch.ptr, (m, n), torch.bfloat16)

            bodies = {}
            if "fused" in ops:
                op_f = ops["fused"]
                out_f = op_f(
                    a, b_shuf, sa, sb
                )  # the output view; it aliases the window
                bodies["fused"] = (lambda: op_f(a, b_shuf, sa, sb), out_f, ref, None)
            for name, op_name, fn in (
                ("trig", "trig", gated),
                ("trig-agent", "trig", gated_agent),
                ("trig-conc", "trig-conc", gated),
            ):
                if name in methods and op_name in ops:
                    op_t = ops[op_name]
                    gate_base = op_t.base + op_t.layout.rs_gate_off

                    def body(op_t=op_t, gate_base=gate_base, fn=fn):
                        op_t.begin(m)
                        run_gated(op_t.input_view(m), gate_base, fn)
                        op_t.complete(m)

                    def arm(op_t=op_t):
                        t0 = time.perf_counter_ns()
                        op_t.arm(m)
                        return (time.perf_counter_ns() - t0) / 1e3

                    bodies[name] = (body, op_t.output_view(m), ref, arm)
            if "split" in methods:
                bodies["split"] = (
                    lambda: (run_plain(), dist.all_reduce(tmp)),
                    tmp,
                    ref,
                    None,
                )
            if "gemm" in methods:
                bodies["gemm"] = (lambda: run_plain(), tmp, mine, None)
            if "gemm-gate" in methods:
                bodies["gemm-gate"] = (
                    lambda: run_gated(tmp, scratch_gates.data_ptr()),
                    tmp,
                    mine,
                    None,
                )
            if "gemm-win" in methods:
                bodies["gemm-win"] = (lambda: run_plain(c_win), c_win, mine, None)
            if "gemm-gate-win" in methods:
                bodies["gemm-gate-win"] = (
                    lambda: run_gated(c_win, win_gates),
                    c_win,
                    mine,
                    None,
                )
            if "gemm-gate-agent-win" in methods:
                bodies["gemm-gate-agent-win"] = (
                    lambda: run_gated(c_win, win_gates, gated_agent),
                    c_win,
                    mine,
                    None,
                )

            per_round = {x: [] for x in bodies}
            arm_us = {x: [] for x in bodies}
            exact = {x: [0, 0] for x in bodies}
            late = {x: 0 for x in bodies}
            for _ in range(args.rounds):
                for name, (body, out, expect, arm) in bodies.items():
                    calls = []
                    for it in range(args.warmup + args.iters):
                        t_arm = arm() if arm else None
                        out.fill_(float("nan"))
                        comm.barrier()  # host-side; no GPU work in the region's way
                        torch.cuda._sleep(hold)
                        e0.record()
                        body()
                        was_late = e0.query()
                        e1.record()
                        torch.cuda.synchronize()
                        if it < args.warmup:
                            continue
                        calls.append(e0.elapsed_time(e1) * 1e3)
                        ok = torch.equal(out, expect)
                        exact[name][0] += int(ok)
                        exact[name][1] += 1
                        late[name] += int(was_late)
                        if t_arm is not None:
                            arm_us[name].append(t_arm)
                    t = torch.tensor(calls, dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank per call
                    per_round[name].append(statistics.median(t.tolist()))
            if args.breakdown:
                bd = _breakdown(
                    args,
                    ops,
                    comm,
                    m,
                    n,
                    k,
                    hold,
                    a_i8,
                    b_i8,
                    sa,
                    sb,
                    ref,
                    run_gated,
                    gated,
                    stream,
                )
                if rank == 0:
                    for name, stats in bd.items():
                        print(
                            "BREAKDOWN "
                            + json.dumps({"m": m, "method": name, **stats}),
                            flush=True,
                        )
            for name in ("trig", "trig-conc"):
                if name in ops:
                    ops[name].check()

            tot = torch.tensor(
                [[exact[x][0], exact[x][1], late[x]] for x in bodies], dtype=torch.int64
            )
            dist.all_reduce(tot, op=dist.ReduceOp.SUM)
            row = {"m": m, "n": n, "k": k, "world": world}
            for i, name in enumerate(bodies):
                row[name] = {
                    "us": round(statistics.median(per_round[name]), 1),
                    "rounds": [round(v, 1) for v in per_round[name]],
                    "exact": f"{int(tot[i, 0])}/{int(tot[i, 1])}",
                    "late": int(tot[i, 2]),
                }
                if arm_us[name]:
                    row[name]["arm_us"] = round(statistics.median(arm_us[name]), 1)
            rows_out.append(row)
            if rank == 0:
                print("ROW " + json.dumps(row), flush=True)

        for op in ops.values():
            op.close()

    if rank == 0:
        hdr = f"{'M':>6} " + " ".join(f"{x:>11}" for x in methods)
        print("\nus per call (slowest rank, median over rounds); trig arm us in []")
        print(hdr)
        for row in rows_out:
            cells = []
            for x in methods:
                c = row.get(x)
                cells.append(f"{c['us']:>11.1f}" if c else f"{'-':>11}")
            arms = " ".join(
                f"[{x} arm {row[x]['arm_us']}]"
                for x in methods
                if row.get(x, {}).get("arm_us") is not None
            )
            print(f"{row['m']:>6} " + " ".join(cells) + "  " + arms)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows_out, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

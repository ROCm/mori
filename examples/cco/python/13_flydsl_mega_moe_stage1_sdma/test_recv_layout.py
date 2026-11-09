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
"""recv_layout: stage1 with NO aiter plan -- the receiver builds its own expert-major layout.

    new prepare -> GEMM1 (gated; builds the layout itself) -> aiter stage2 + combine
    (``--fuse-prepare``: the prepare runs inside the GEMM1 too: one launch per call)

Checks, then times:

* aiter's own output does not change when its runtime fanout pairs switch on (call 1 has
  none, call 2 has them): the layout here builds none, so this is what makes it comparable;
* the final layer output equals aiter's bit for bit: once with every row landed, then
  ``--stress`` times gated, with the staging area, rowmap and stage1 output poisoned;
* time: new prepare + gated GEMM1 against aiter's compact prepare + fused stage1.

Phase 1c: the prepare always owns its tokens, stages the routes first, swizzles the quant
stores through LDS and splits a token over 3 waves; the gated GEMM1 always builds the layout.
Switches left: --ready-first, --no-plan, --fuse-prepare, --chunk-rows, --early-chunks,
--head-rows, --sbm, --tile-n.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 test_recv_layout.py --bs 4096
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "11_flydsl_mega_moe_sdma")
)

import main as ex11  # noqa: E402  (sets the shmem env defaults, imports torch first)
import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from mori.cco import Communicator, UniqueId  # noqa: E402
from mori.tensor_utils import from_gpu_ptr  # noqa: E402

from gemm1_sdma import (  # noqa: E402
    TRACE_ITEMS,
    TRACE_SLOTS,
    TRACE_STRIDE,
    gemm1_kwargs,
    run_gemm1_sdma,
)
from stage1_sdma import PREP_SLOTS, Stage1SdmaPrepare  # noqa: E402

#: ``trace_report``: GEMM1's tile order is cut into this many equal slices
BUCKETS = 10


def prepare_step(prep, x, ids, wts, args):
    """The separate prepare kernel -- unless the GEMM1 runs the prepare itself (``--fuse-prepare``)."""
    if not args.fuse_prepare:
        prep.run(x, ids, wts)


def trace_report(
    args, prep, gtrace, gemm1, x, ids, wts, sbm, moe, chunks_from, rank, world
):
    """Gate wait by position in GEMM1's tile order (flat work index, in ``BUCKETS`` equal
    slices), and when each slice first got past its gate. Times in us from prepare's start.
    """
    num_cu, nb, B = moe._s1_num_cu, prep.quant_blocks + 1, BUCKETS
    fused_rows = []
    rows = []
    for it in range(args.trace + 2):
        torch.cuda.synchronize()
        dist.barrier()
        prep.arm(chunks_from)
        gtrace.zero_()
        prep.trace_buf.zero_()
        prepare_step(prep, x, ids, wts, args)
        gemm1(gated=True, trace=gtrace)
        torch.cuda.synchronize()
        if it < 2:
            continue
        g_real = gtrace.cpu().view(num_cu, TRACE_STRIDE).double()
        # the same GEMM1 again with every row landed (gates pass at once; WG0 re-puts the rest
        # chunks, waited for below with this call's signal base kept): its compute and clock
        if args.fuse_prepare or args.ready_first:
            # a gated GEMM1 is a whole call here (the prepare is inside it): it cannot be run
            # again without ``arm``, so there is no "all landed" pass -- reuse the real trace
            g_land = g_real.clone()
        else:
            dist.barrier()
            sig_base = prep.sig_base.clone()
            gtrace.zero_()
            gemm1(gated=True, trace=gtrace)
            prep.wait(
                [
                    max(c - prep.arena.early_chunks, 0) if s_ != rank else 0
                    for s_, c in enumerate(chunks_from)
                ],
                routes=False,
            )
            prep.sig_base.copy_(sig_base)
            torch.cuda.synchronize()
            g_land = gtrace.cpu().view(num_cu, TRACE_STRIDE).double()
        it_l = g_land[1:, 8 : 8 + TRACE_SLOTS * TRACE_ITEMS].view(
            num_cu - 1, TRACE_ITEMS, TRACE_SLOTS
        )
        pt, g = prep.trace_buf.cpu().double(), g_real
        P = PREP_SLOTS
        p0 = pt[0 : P * nb : P].min()
        prep_end = pt[3 : P * nb : P].max()

        # prepare's own phases, per block: plan (start -> [2]), staging ([4] -> [5]), quant
        # pass 1 ([2] -> [6]). A block that skipped a phase leaves its stamps at 0.
        def span(a_, b_):
            u, v = pt[a_ : P * nb : P], pt[b_ : P * nb : P]
            m = (u > 0) & (v > 0)
            return float((v - u)[m].max()) / 100.0 if bool(m.any()) else float("nan")

        prep_plan, prep_stage = span(0, 2), span(4, 5)
        prep_q1, prep_tot = span(2, 6), span(0, 3)
        prep_pub = span(7, 5)  # publish: done counter, scount, flags
        puts = pt[P * nb : P * nb + 16]
        puts = puts[puts > 0]
        cb = g[1:]

        # fused kernel phases per worker block, us from the first prepare stamp:
        # [0] start, [4] prepare_phase returned (fuse_prepare), [1] layout done, [2] tiles done
        def q(col):
            v = cb[:, col]
            v = v[v > 0]
            return (
                (
                    (v.min() - p0) / 100.0,
                    (v.median() - p0) / 100.0,
                    (v.max() - p0) / 100.0,
                )
                if len(v)
                else (float("nan"),) * 3
            )

        fused_rows.append(
            torch.tensor(
                [x_ for c_ in (0, 4, 1, 2) for x_ in q(c_)], dtype=torch.float64
            )
        )
        n_items = cb[:, 3].long()
        total = int(n_items.sum())
        assert (
            int(n_items.max()) <= TRACE_ITEMS
        ), f"raise TRACE_ITEMS: {int(n_items.max())}"
        it_ = cb[:, 8 : 8 + TRACE_SLOTS * TRACE_ITEMS].view(
            num_cu - 1, TRACE_ITEMS, TRACE_SLOTS
        )
        k = torch.arange(TRACE_ITEMS).view(1, -1)
        flat = torch.arange(num_cu - 1).view(-1, 1) + k * (num_cu - 1)
        valid = k < n_items.view(-1, 1)
        bucket = (flat * B // max(total, 1)).clamp(max=B - 1)
        gate = (it_[:, :, 1] - it_[:, :, 0]) / 100.0
        passed = (it_[:, :, 1] - p0) / 100.0
        r = [
            (prep_end - p0) / 100.0,
            ((puts.min() - p0) / 100.0) if len(puts) else float("nan"),
            ((puts.max() - p0) / 100.0) if len(puts) else float("nan"),
            (g[1:, 0].min() - p0) / 100.0,  # GEMM1 start
            ((cb[:, 1] - cb[:, 0]) / 100.0).mean(),  # start gate (flags)
            (g[0, 2] - p0) / 100.0,  # comm WG done posting the rest
            (g[1:, 2].max() - p0) / 100.0,  # GEMM1 end
        ]
        land = g[0, 8 : 8 + world]
        land = land[land > 0]
        r_land = [
            ((land.min() - p0) / 100.0) if len(land) else float("nan"),
            ((land.max() - p0) / 100.0) if len(land) else float("nan"),
        ]
        for b in range(B):
            m = valid & (bucket == b)
            r.append(float((gate * m).sum()) / (num_cu - 1))  # gate wait per WG, us
        for b in range(B):
            m = valid & (bucket == b)
            r.append(float(passed[m].min()) if m.any() else float("nan"))
        comp = (it_[:, :, 2] - it_[:, :, 1]) / 100.0
        for b in range(B):
            m = valid & (bucket == b)
            r.append(float((comp * m).sum()) / (num_cu - 1))  # compute per WG, us
        # core clock per slice: s_memtime ticks over 100 MHz ticks of the same tiles, in MHz
        mt = it_[:, :, 4] - it_[:, :, 3]
        rt = it_[:, :, 2] - it_[:, :, 1]
        clk = []
        for b in range(B):
            m = valid & (bucket == b)
            clk.append(float((mt * m).sum()) / max(float((rt * m).sum()), 1.0) * 100.0)
        mt_l = it_l[:, :, 4] - it_l[:, :, 3]
        rt_l = it_l[:, :, 2] - it_l[:, :, 1]
        comp_l, clk_l = [], []
        for b in range(B):
            m = valid & (bucket == b)
            comp_l.append(float((rt_l * m).sum()) / 100.0 / (num_cu - 1))
            clk_l.append(
                float((mt_l * m).sum()) / max(float((rt_l * m).sum()), 1.0) * 100.0
            )
        rows.append(
            torch.tensor(
                [
                    float(v)
                    for v in r
                    + r_land
                    + clk
                    + comp_l
                    + clk_l
                    + [prep_plan, prep_stage, prep_q1, prep_tot, prep_pub]
                ],
                dtype=torch.float64,
            )
        )
    if fused_rows and rank == 0:
        fm = torch.stack(fused_rows).median(dim=0).values.tolist()
        for i_, n_ in enumerate(
            ("start", "prepare_done", "layout_done(=tiles start)", "tiles_done")
        ):
            print(
                f"TRACE FUSED {n_:28s} min/median/max over workers (us): {fm[3*i_]:.1f} / {fm[3*i_+1]:.1f} / {fm[3*i_+2]:.1f}",
                flush=True,
            )
    med = torch.stack(rows).median(dim=0).values.to(prep.trace_buf.device)
    dist.all_reduce(med)
    med = (med / world).tolist()
    if rank == 0:
        names = [
            "prep_end",
            "first_put",
            "last_put",
            "gemm_start",
            "start_gate",
            "comm_done",
            "gemm_end",
        ]
        print(
            "TRACE " + " ".join(f"{n}={v:.1f}" for n, v in zip(names, med[:7])),
            flush=True,
        )
        print(
            "TRACE gate wait per WG by tile slice (us): "
            + " ".join(f"{v:.1f}" for v in med[7 : 7 + B])
            + f" | sum {sum(med[7 : 7 + B]):.1f}",
            flush=True,
        )
        print(
            "TRACE slice first passed (us): "
            + " ".join(f"{v:.0f}" for v in med[7 + B : 7 + 2 * B]),
            flush=True,
        )
        print(
            "TRACE compute per WG by tile slice (us): "
            + " ".join(f"{v:.0f}" for v in med[7 + 2 * B : 7 + 3 * B])
            + f" | sum {sum(med[7 + 2 * B : 7 + 3 * B]):.0f}",
            flush=True,
        )
        print(
            f"TRACE all chunks landed (first peer / last peer, us): {med[7 + 3 * B]:.0f} / "
            f"{med[7 + 3 * B + 1]:.0f}",
            flush=True,
        )
        print(
            "TRACE core clock by tile slice (MHz, s_memtime / s_memrealtime): "
            + " ".join(f"{v:.0f}" for v in med[9 + 3 * B : 9 + 4 * B]),
            flush=True,
        )
        print(
            "TRACE all landed: compute per WG by tile slice (us): "
            + " ".join(f"{v:.0f}" for v in med[9 + 4 * B : 9 + 5 * B])
            + f" | sum {sum(med[9 + 4 * B : 9 + 5 * B]):.0f}",
            flush=True,
        )
        print(
            "TRACE all landed: core clock by tile slice (MHz): "
            + " ".join(f"{v:.0f}" for v in med[9 + 5 * B : 9 + 6 * B]),
            flush=True,
        )
        pl, stg, q1, tot, pub = med[9 + 6 * B : 9 + 6 * B + 5]
        print(
            f"TRACE prepare breakdown (slowest block, us): plan {pl:.1f}, staging {stg:.1f} "
            f"(publish {pub:.1f}), quant pass1 {q1:.1f}, total {tot:.1f}",
            flush=True,
        )


def trace_report_rf(
    args, prep, gtrace, gemm1, x, ids, wts, sbm, moe, chunks_from, rank, world
):
    """Ready-first timeline, ``TRACE-RF`` lines. Every time is in us from the start of the call on
    that rank (the standalone prepare's first block, or the GEMM1's first block when the prepare is
    fused), the median over the ``--trace`` calls; each value is printed ``rank 0 | median over
    ranks`` and ``a/b/c`` is min / median / max over the blocks of that role. Stamps are written by
    one lane after the barrier that ends the phase, so they do not change what a block does.
    """
    num_cu, P = int(moe._s1_num_cu), PREP_SLOTS
    sb = 0  # the scheduler's block (comm, wave 0)
    worker_b = list(range(1, num_cu))  # every worker quantises, lays out and runs tiles
    nan = float("nan")
    iters = []
    for it in range(args.trace + 2):
        torch.cuda.synchronize()
        dist.barrier()
        prep.arm(chunks_from)
        gtrace.zero_()
        prep.trace_buf.zero_()
        prepare_step(prep, x, ids, wts, args)
        gemm1(gated=True, trace=gtrace)
        torch.cuda.synchronize()
        if it < 2:
            continue
        g = gtrace.cpu().view(num_cu, TRACE_STRIDE).double()
        pt = prep.trace_buf.cpu().double()
        starts = [g[:, 0][g[:, 0] > 0].min()]
        ps = pt[0 : P * num_cu : P]
        if bool((ps > 0).any()):
            starts.append(ps[ps > 0].min())
        t0 = float(min(starts))

        def one(v):
            return [float(v - t0) / 100.0 if float(v) > 0 else nan]

        def tri(vals):
            vals = torch.as_tensor(vals, dtype=torch.float64)
            vals = vals[vals > 0]
            if not len(vals):
                return [nan] * 3
            u = (vals - t0) / 100.0
            return [float(u.min()), float(u.median()), float(u.max())]

        def col(blocks, c):
            return g[blocks, c] if blocks else torch.zeros(0, dtype=torch.float64)

        def pcol(blocks, c):
            return (
                torch.stack([pt[b * P + c] for b in blocks])
                if blocks
                else torch.zeros(0, dtype=torch.float64)
            )

        def span(a_, b_):
            u, v = pt[a_ : P * num_cu : P], pt[b_ : P * num_cu : P]
            m = (u > 0) & (v > 0)
            return [float((v - u)[m].max()) / 100.0 if bool(m.any()) else nan]

        r = []
        r.append(("SCHED layout_done seen", one(g[sb, 8])))
        r.append(("SCHED tile table loaded", one(g[sb, 12])))
        r.append(("SCHED first m-tile published", one(g[sb, 9])))
        r.append(("SCHED last m-tile published", one(g[sb, 10])))
        r.append(("SCHED m-tiles (count, not us)", [float(g[sb, 11])]))
        for ci, nt_ in enumerate((28, 56, 112)):
            r.append((f"SCHED {nt_} m-tiles published", one(g[sb, 40 + ci])))
        r.append(("SCHED first round started", one(g[sb, 13])))
        r.append(("SCHED first round finished", one(g[sb, 14])))
        r.append(("SCHED chunk 0 seen landed, per peer", tri(g[sb, 24:32])))
        w16 = g[0, 16:24]
        r.append(
            (
                "COMM  rest chunks posted (last wave)",
                one(w16.max()) if bool((w16 > 0).any()) else [nan],
            )
        )
        r.append(("COMM  qdone reached", one(g[0, 5])))
        puts = pt[P * num_cu : P * num_cu + 16]
        puts = puts[puts > 0]
        r.append(
            (
                "COMM  early puts posted first/last",
                (
                    [float(puts.min() - t0) / 100.0, float(puts.max() - t0) / 100.0]
                    if len(puts)
                    else [nan, nan]
                ),
            )
        )
        r.append(("QUANT B1 passed", tri(pcol(worker_b, 1))))
        r.append(("QUANT pass 1 done (B3 reached)", tri(pcol(worker_b, 6))))
        r.append(("QUANT B3 passed", tri(pcol(worker_b, 8))))
        r.append(("QUANT prepare_phase returned", tri(pcol(worker_b, 3))))
        r.append(
            (
                "LAYOUT start (all workers)",
                tri(col(worker_b, 4)) if args.fuse_prepare else tri(col(worker_b, 0)),
            )
        )
        r.append(("LAYOUT end", tri(col(worker_b, 1))))
        r.append(("WORKERS first tile starts", tri(col(worker_b, 8))))
        r.append(("WORKERS loop ends", tri(col(worker_b, 2))))
        r.append(("WORKERS tiles run (sum, not us)", [float(col(worker_b, 3).sum())]))
        ran = [b for b in worker_b if float(g[b, 3]) > 0]

        def dur(c):
            v = (
                torch.stack([g[b, c] for b in ran]) / 100.0
                if ran
                else torch.zeros(0, dtype=torch.float64)
            )
            return (
                [float(v.min()), float(v.median()), float(v.max())]
                if len(v)
                else [nan] * 3
            )

        r.append(("WORKERS queue wait, tiles 1.. (us, sum per worker)", dur(5)))
        r.append(("WORKERS first queue wait (us)", dur(6)))
        r.append(("WORKERS pure tile time (us, sum per worker)", dur(7)))
        if not args.fuse_prepare:
            r.append(
                (
                    "PREP-KERNEL plan / staging / pass1 / total (slowest block, us)",
                    span(0, 2) + span(4, 5) + span(2, 6) + span(0, 3),
                )
            )
        iters.append(r)
    labels = [name for name, _ in iters[0]]
    widths = [len(v) for _, v in iters[0]]
    flat = torch.tensor(
        [[x_ for _, v in r for x_ in v] for r in iters], dtype=torch.float64
    )
    med = flat.nanmedian(dim=0).values.to(x.device)
    gl = [torch.zeros_like(med) for _ in range(world)]
    dist.all_gather(gl, med)
    allm = torch.stack(gl).cpu()
    over = allm.nanmedian(dim=0).values
    if rank == 0:

        def fmt(vs):
            return "/".join("-" if v != v else f"{v:.1f}" for v in vs)

        print(
            f"TRACE-RF bs={args.bs} fuse_prepare={int(args.fuse_prepare)} no_plan={int(args.no_plan)}: "
            f"us from call start, median of {args.trace} calls; "
            f"'rank0 | median over {world} ranks'; a/b/c = min/median/max over blocks",
            flush=True,
        )
        i = 0
        for lab, w in zip(labels, widths):
            print(
                f"TRACE-RF {lab:58s} {fmt(allm[0, i:i + w].tolist()):>22s} | {fmt(over[i:i + w].tolist())}",
                flush=True,
            )
            i += w


def graph_timing(
    args,
    prep,
    moe,
    config,
    sbm,
    x,
    ids,
    wts,
    sx,
    ss,
    gemm1,
    check,
    chunks_from,
    window,
    stage,
    my_out,
    rank,
    world,
    eager,
):
    """Both sides captured in a CUDA graph each, timed per replay like the eager loop (sync and
    barrier between calls). The SDMA graph holds arm() too, so every replay is one full call.
    """
    device = x.device
    chunks_t = torch.tensor(chunks_from, dtype=torch.int32, device=device)

    def sdma_call():
        prep.arm(chunks_t)
        prepare_step(prep, x, ids, wts, args)
        gemm1(gated=True)

    def aiter_call():
        fs = fx.Stream(torch.cuda.current_stream().cuda_stream)
        moe._run_compact_prepare(ids, args.bs, config.stage1, fs, quant_input=x)
        moe._run_fused_stage1(
            sx, wts, ss, ids, stream=fs, config=config.stage1, prepared=True
        )

    def capture(fn):
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        return g

    g_sdma = capture(sdma_call)
    # correctness under the graph, with the window, staging and output poisoned
    for it in range(3):
        window.fill_(0xAA)
        stage.fill_(-7)
        prep.rowmap.zero_()
        my_out.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        g_sdma.replay()
        torch.cuda.synchronize()
        check(f"graph call {it}")
    g_aiter = capture(aiter_call)

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def timed(g):
        out = []
        for it in range(args.time_iters + 3):
            torch.cuda.synchronize()
            dist.barrier()
            ev[0].record()
            g.replay()
            ev[1].record()
            torch.cuda.synchronize()
            if it >= 3:
                out.append(ev[0].elapsed_time(ev[1]))
        return torch.tensor(out).median().item()

    new_ms = timed(g_sdma)
    old_ms = timed(g_aiter)
    # and once more after aiter's graph rewrote the shared tables
    g_sdma.replay()
    torch.cuda.synchronize()
    check("graph after aiter graph")
    med = torch.tensor([new_ms, old_ms], dtype=torch.float64, device=device)
    dist.all_reduce(med)
    med /= world
    if rank == 0:
        n_, o_ = med.tolist()
        tail = ""
        if eager is not None:
            tail = f" | eager: SDMA {eager[2]:.3f}, aiter {eager[3]:.3f} ({eager[3] / eager[2]:.2f}x)"
        print(
            f"RESULT-GRAPH {args.network} bs={args.bs}: SDMA {n_:.3f} ms, aiter {o_:.3f} ms "
            f"({o_ / n_:.2f}x){tail}",
            flush=True,
        )
    del g_sdma, g_aiter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=ex11.NETWORKS, default="flash")
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument("--early-chunks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stress", type=int, default=20)
    ap.add_argument("--time-iters", type=int, default=30)
    ap.add_argument(
        "--trace",
        type=int,
        default=0,
        help="trace this many gated calls: where in GEMM1's tile order the gates wait "
        "(TRACE lines), or the ready-first timeline (TRACE-RF lines)",
    )
    ap.add_argument(
        "--graph",
        action="store_true",
        help="also time both sides captured in CUDA graphs (same process, after eager)",
    )
    ap.add_argument(
        "--graph-only",
        action="store_true",
        help="time ONLY in CUDA graphs (no eager timing at all); implies --graph",
    )
    ap.add_argument(
        "--fuse-prepare",
        action="store_true",
        help="GEMM1 also runs the prepare (one launch per call)",
    )
    ap.add_argument(
        "--ready-first",
        action="store_true",
        help="GEMM1: a scheduler wave (comm workgroup) publishes m-tiles in the order "
        "they become ready; workers take tickets and read the ready queue "
        "instead of gating each tile on the SDMA signals",
    )
    ap.add_argument(
        "--no-plan",
        action="store_true",
        help="prepare without the counting sort: a block reserves its rows of each run "
        "with one atomic and places them in any order, no global histogram / "
        "prefix sum / grid barrier B1; GEMM1 then waits for ALL of a source's "
        "chunks. Free when a run is one put (small batch).",
    )
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="skip the stage2 comparison (needed with --sbm: aiter's stage2 is "
        "compiled for its own sort_block_m). Timings only, NO correctness.",
    )
    ap.add_argument(
        "--sbm",
        type=int,
        default=0,
        help="override GEMM1's sort_block_m (0 = aiter's choice for this shape)",
    )
    ap.add_argument(
        "--tile-n",
        type=int,
        default=0,
        help="override GEMM1's tile_n (0 = aiter's choice); LDS needs "
        "sbm * tile_n / 2 * 4 bytes",
    )
    ap.add_argument(
        "--head-rows",
        type=int,
        default=0,
        help="rows of the FIRST chunk of a run (0 = off: all chunks are "
        "--chunk-rows); a short head starts the first put sooner without "
        "making every later chunk small",
    )
    args = ap.parse_args()
    if args.graph_only:
        args.graph = True

    rank, world, device = ex11._setup()
    net, moe = ex11._build_moe(args, args.bs, rank, world, device)
    hidden, topk = net["model_dim"], net["topk"]
    x, wts, ids = ex11._make_inputs(
        args.bs, hidden, net["experts"], topk, rank, args.seed, device
    )
    ref = moe(x, wts, ids)[: args.bs].clone()
    ref2 = moe(x, wts, ids)[: args.bs].clone()
    torch.cuda.synchronize()
    config = moe._active_config
    # ``--sbm``: the GEMM1 M tile. aiter picks it per shape; overriding it trades padding
    # (every expert's rows round up to a multiple) against weight traffic (each expert's
    # weights are re-read once per m-tile). Our layout and GEMM1 must use the SAME value.
    sbm = int(args.sbm) if args.sbm else int(config.stage1.sort_block_m)
    op = moe._s1_op
    errors = 0
    fan_ok = torch.tensor(
        [int(torch.equal(ref, ref2))], dtype=torch.int32, device=device
    )
    dist.all_reduce(fan_ok)
    if rank == 0:
        print(
            f"aiter call 1 (no fanout) == call 2 (fanout): {int(fan_ok)}/{world} ranks "
            f"(SBM={sbm}, aligned_pair={config.stage2.aligned_pair})",
            flush=True,
        )

    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    vmm = 4 * world * moe.mtpr * hidden * 2 + (1 << 30)
    with Communicator.init(
        world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
    ) as comm:
        prep = Stage1SdmaPrepare(
            comm,
            experts=net["experts"],
            topk=topk,
            hidden=hidden,
            max_tok=moe.mtpr,
            chunk_rows=args.chunk_rows,
            early_chunks=args.early_chunks,
            trace=args.trace > 0,
            head_rows=args.head_rows,
            sched=args.ready_first,
            no_plan=args.no_plan,
        )
        a = prep.arena
        window = from_gpu_ptr(prep.mem.ptr, (2 * a.n_rows, a.row_bytes), torch.uint8)
        stage = from_gpu_ptr(
            prep.mem.ptr + a.off_stage, (a.npes * a.stage_routes * 4,), torch.int32
        )
        prep.rowmap.zero_()
        torch.cuda.synchronize()
        dist.barrier()

        # The first call, every chunk put and waited for: how many chunks each source sends.
        prep.run(x, ids, wts)  # the setup always uses the separate kernel
        prep.post_rest()
        cnt_all = [torch.zeros(world, dtype=torch.int32) for _ in range(world)]
        dist.all_gather(cnt_all, prep.cnt.cpu())
        cnt_all = torch.stack(cnt_all)
        chunks_from = [
            prep.signals_for(int(cnt_all[s][rank])) if s != rank else 0
            for s in range(world)
        ]
        prep.wait(chunks_from)
        prep.layout(
            moe, sbm
        )  # the setup uses the standalone layout kernel (the ungated GEMM1 below reads it)
        torch.cuda.synchronize()

        my_out = torch.zeros_like(moe._s1_out)
        my_osd = torch.zeros_like(moe._s1_osd)
        kwargs = gemm1_kwargs(moe, config)
        kwargs["sort_block_m"] = sbm
        if args.tile_n:
            kwargs["tile_n"] = int(args.tile_n)

        gtrace = (
            torch.zeros(moe._s1_num_cu * TRACE_STRIDE, dtype=torch.int64, device=device)
            if args.trace
            else None
        )

        def fused_args():
            fz = prep.fused_layout_args(moe)
            if args.fuse_prepare:
                fz["prepare"] = prep.fused_prepare_args(x, ids, wts)
            return fz

        rf_kwargs = dict(ready_first=args.ready_first)

        def gemm1(gated, trace=None):
            run_gemm1_sdma(
                my_out,
                window.view(torch.float8_e4m3fn),
                moe._s1_w1,
                moe._s1_w1_scale,
                op.tile_row_base,
                moe._s1_tile_input_base,
                op.sorted_expert_ids,
                op.num_valid,
                my_osd,
                prep.rowmap,
                moe._s1_nvm,
                fx.Stream(
                    torch.cuda.current_stream().cuda_stream
                ),  # the capture stream too
                prep=prep if gated else None,
                trace=trace,
                fuse=(
                    fused_args() if gated else None
                ),  # the gated GEMM1 builds the layout
                **kwargs,
                **(rf_kwargs if gated else {}),
            )

        def stage2_and_check(label):
            nonlocal errors
            if args.no_check:
                # ``--sbm`` changes num_valid and the padding, but aiter's stage2 kernel is
                # compiled for ITS sort_block_m and faults on our layout. Stage1 timings are
                # still meaningful; correctness is NOT checked in this mode.
                return True
            moe._s1_out.copy_(my_out)
            moe._s1_osd.copy_(my_osd)
            res = moe._run_stage2(args.bs, None, True, config, None)[: args.bs]
            torch.cuda.synchronize()
            if not torch.equal(res, ref):
                errors += 1
                print(
                    f"[rank {rank}] {label}: final output differs from aiter's, "
                    f"max abs diff {float((res.float() - ref.float()).abs().max()):.3g}",
                    flush=True,
                )
                return False
            return True

        gemm1(gated=False)
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"num_valid={int(op.num_valid[0])} rows (padded to SBM={sbm})",
                flush=True,
            )
        stage2_and_check("landed")

        def call():
            prepare_step(prep, x, ids, wts, args)
            gemm1(gated=True)

        bad = 0
        for it in range(args.stress):
            window.fill_(0xAA)
            stage.fill_(-7)
            prep.rowmap.zero_()
            my_out.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            prep.arm(chunks_from)
            call()
            torch.cuda.synchronize()
            if not stage2_and_check(f"gated call {it}"):
                bad += 1
                if bad >= 3:
                    break
        if rank == 0:
            print(f"stress: {args.stress} poisoned gated calls per rank", flush=True)

        if args.trace:
            (trace_report_rf if args.ready_first else trace_report)(
                args,
                prep,
                gtrace,
                gemm1,
                x,
                ids,
                wts,
                sbm,
                moe,
                chunks_from,
                rank,
                world,
            )

        if args.graph_only and args.time_iters:
            sx, ss = moe._s1_quant_x[: args.bs], moe._s1_quant_scale[: args.bs]
            graph_timing(
                args,
                prep,
                moe,
                config,
                sbm,
                x,
                ids,
                wts,
                sx,
                ss,
                gemm1,
                stage2_and_check,
                chunks_from,
                window,
                stage,
                my_out,
                rank,
                world,
                None,
            )
            torch.cuda.synchronize()
            dist.barrier()
            prep.arm(chunks_from)
            call()
            torch.cuda.synchronize()
            stage2_and_check("after timing")
        if args.time_iters and not args.graph_only:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            parts, e2e, base = [], [], []
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                prep.arm(chunks_from)
                ev[0].record()
                prepare_step(prep, x, ids, wts, args)
                ev[1].record()
                gemm1(gated=True)
                ev[2].record()
                torch.cuda.synchronize()
                if it >= 3:
                    parts.append([ev[i].elapsed_time(ev[i + 1]) for i in range(2)])
                    e2e.append(ev[0].elapsed_time(ev[2]))
            torch.cuda.synchronize()
            dist.barrier()
            prep.arm(chunks_from)
            call()
            torch.cuda.synchronize()
            stage2_and_check("after the timed calls")
            # GEMM1 alone with every row already landed (same tables, ungated): the compute floor.
            ung = []
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                ev[0].record()
                gemm1(gated=False)
                ev[1].record()
                torch.cuda.synchronize()
                if it >= 3:
                    ung.append(ev[0].elapsed_time(ev[1]))
            # The gated kernel again, every row already landed: no gate waits, but WG0 re-puts
            # the rest chunks (same bytes, so the traffic is there). Then wait for those puts.
            rest_from = [
                max(c - prep.arena.early_chunks, 0) if s_ != rank else 0
                for s_, c in enumerate(chunks_from)
            ]
            landed = []
            sig_base = (
                prep.sig_base.clone()
            )  # wait() re-arms: the gates must keep this call's base
            # (not with --fuse-prepare: there a gated GEMM1 is a whole call, and repeating it
            # without ``arm`` leaves the signal counters ahead of the host's expected totals)
            for it in range(
                0 if (args.fuse_prepare or args.ready_first) else args.time_iters + 3
            ):
                torch.cuda.synchronize()
                dist.barrier()
                ev[0].record()
                gemm1(gated=True)
                ev[1].record()
                prep.wait(rest_from, routes=False)
                prep.sig_base.copy_(sig_base)
                torch.cuda.synchronize()
                if it >= 3:
                    landed.append(ev[0].elapsed_time(ev[1]))
            if not landed:
                landed = [float("nan")]
            sx, ss = moe._s1_quant_x[: args.bs], moe._s1_quant_scale[: args.bs]
            fstream = fx.Stream(torch.cuda.current_stream().cuda_stream)
            a_prep, a_s1 = [], []
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                ev[0].record()
                moe._run_compact_prepare(
                    ids, args.bs, config.stage1, fstream, quant_input=x
                )
                ev[2].record()
                moe._run_fused_stage1(
                    sx,
                    wts,
                    ss,
                    ids,
                    stream=fstream,
                    config=config.stage1,
                    prepared=True,
                )
                ev[1].record()
                torch.cuda.synchronize()
                if it >= 3:
                    base.append(ev[0].elapsed_time(ev[1]))
                    a_prep.append(ev[0].elapsed_time(ev[2]))
                    a_s1.append(ev[2].elapsed_time(ev[1]))
            # counts: routes / rows (one per token and rank) I send, to peers and in all
            g = ids.long().flatten()
            g = g[g >= 0]
            dst = g // prep.epr
            routes_remote = int((dst != rank).sum())
            rows_all = int(sum(int(c) for c in prep.cnt.cpu()))
            rows_remote = rows_all - int(prep.cnt[rank])
            cnts = torch.tensor(
                [
                    float(torch.tensor(a_prep).median()),
                    float(torch.tensor(a_s1).median()),
                    routes_remote,
                    rows_remote,
                    rows_all,
                    int(g.numel()),
                ],
                dtype=torch.float64,
                device=x.device,
            )
            dist.all_reduce(cnts)
            cnts /= world
            if rank == 0:
                ap_, as_, rr, wr, wa, ra = cnts.tolist()
                rb = prep.arena.row_bytes
                print(
                    f"SPLIT aiter: compact prepare {ap_:.3f} ms, fused stage1 {as_:.3f} ms | per rank: "
                    f"routes {ra:.0f} (remote {rr:.0f}), rows {wa:.0f} (remote {wr:.0f}), "
                    f"route/row {ra / max(wa, 1):.2f}, remote bytes SDMA {wr * rb / 1e6:.1f} MB "
                    f"vs per-route {rr * rb / 1e6:.1f} MB",
                    flush=True,
                )
            med = torch.tensor(
                torch.tensor(parts, dtype=torch.float64).median(dim=0).values.tolist()
                + [
                    torch.tensor(e2e).median().item(),
                    torch.tensor(base).median().item(),
                    torch.tensor(ung).median().item(),
                    torch.tensor(landed).median().item(),
                ],
                dtype=torch.float64,
            )
            dist.all_reduce(med)
            med /= world
            if rank == 0:
                p_, g_, new_ms, old_ms, u_, gl_ = med.tolist()
                print(
                    f"parts: prepare {p_:.3f} ms, gated GEMM1 (layout inside) {g_:.3f} ms "
                    f"(all landed: gated {gl_:.3f} ms, ungated {u_:.3f} ms)",
                    flush=True,
                )
                print(
                    f"RESULT {args.network} bs={args.bs}: new prepare + gated GEMM1 (layout inside) "
                    f"{new_ms:.3f} ms (no aiter plan); aiter prepare+stage1 {old_ms:.3f} ms "
                    f"({old_ms / new_ms:.2f}x)",
                    flush=True,
                )
            if args.graph:
                graph_timing(
                    args,
                    prep,
                    moe,
                    config,
                    sbm,
                    x,
                    ids,
                    wts,
                    sx,
                    ss,
                    gemm1,
                    stage2_and_check,
                    chunks_from,
                    window,
                    stage,
                    my_out,
                    rank,
                    world,
                    med.tolist(),
                )
            # aiter's prepare rewrote the tables and the fanout pairs: one more checked call.
            torch.cuda.synchronize()
            dist.barrier()
            prep.arm(chunks_from)
            call()
            torch.cuda.synchronize()
            stage2_and_check("after timing")
        prep.close()

    err = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(err)
    if rank == 0:
        print("SUCCESS" if int(err) == 0 else f"FAILED: {int(err)} mismatches")
    dist.destroy_process_group()
    return int(err) != 0


if __name__ == "__main__":
    raise SystemExit(main())

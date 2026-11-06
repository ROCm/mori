"""recv_layout: stage1 with NO aiter plan -- the receiver builds its own expert-major layout.

    new prepare (recv_layout) -> layout -> GEMM1 (gated) -> aiter stage2 + combine

Checks, then times:

* aiter's own output does not change when its runtime fanout pairs switch on (call 1 has
  none, call 2 has them): the layout here builds none, so this is what makes it comparable;
* the final layer output equals aiter's bit for bit: once with every row landed, then
  ``--stress`` times gated, with the staging area, rowmap and stage1 output poisoned;
* time: new prepare + layout + gated GEMM1 against aiter's compact prepare + fused stage1.

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

from gemm1_sdma import TRACE_ITEMS, TRACE_SLOTS, TRACE_STRIDE, run_gemm1_sdma  # noqa: E402
from stage1_sdma import ARR_IDX, PREP_SLOTS, SCH_ARR, Stage1SdmaPrepare  # noqa: E402
from test_gemm1 import gemm1_kwargs  # noqa: E402


def prepare_step(prep, x, ids, wts, args):
    """The separate prepare kernel -- unless the GEMM1 runs the prepare itself (``--fuse-prepare``)."""
    if not args.fuse_prepare:
        prep.run(x, ids, wts, recv_layout=True)


def layout_step(prep, moe, sbm, args):
    """The separate layout kernel -- unless the GEMM1 builds the layout itself (``--fuse-layout``)."""
    if not args.fuse_layout:
        prep.layout(moe, sbm, local_first=args.local_first)


def trace_report(args, prep, gtrace, gemm1, x, ids, wts, sbm, moe, chunks_from, rank, world):
    """Gate wait by position in GEMM1's tile order (flat work index, in ``--buckets`` equal
    slices), and when each slice first got past its gate. Times in us from prepare's start."""
    num_cu, nb, B = moe._s1_num_cu, prep.quant_blocks + 1, args.buckets
    fused_rows = []
    rows = []
    for it in range(args.trace + 2):
        torch.cuda.synchronize()
        dist.barrier()
        prep.arm(chunks_from)
        gtrace.zero_()
        prep.trace_buf.zero_()
        prepare_step(prep, x, ids, wts, args)
        layout_step(prep, moe, sbm, args)
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
            prep.wait([max(c - prep.arena.early_chunks, 0) if s_ != rank else 0
                       for s_, c in enumerate(chunks_from)], routes=False)
            prep.sig_base.copy_(sig_base)
            torch.cuda.synchronize()
            g_land = gtrace.cpu().view(num_cu, TRACE_STRIDE).double()
        it_l = g_land[1:, 8 : 8 + TRACE_SLOTS * TRACE_ITEMS].view(num_cu - 1, TRACE_ITEMS, TRACE_SLOTS)
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
        prep_pub = span(7, 5)   # publish: done counter, scount, flags
        puts = pt[P * nb : P * nb + 16]
        puts = puts[puts > 0]
        cb = g[1:]
        if args.fuse_layout:
            # fused kernel phases per worker block, us from the first prepare stamp:
            # [0] start, [4] prepare_phase returned (fuse_prepare), [1] layout done, [2] tiles done
            def q(col):
                v = cb[:, col]
                v = v[v > 0]
                return ((v.min() - p0) / 100.0, (v.median() - p0) / 100.0, (v.max() - p0) / 100.0) if len(v) else (float("nan"),) * 3
            fused_rows.append(torch.tensor([x_ for c_ in (0, 4, 1, 2) for x_ in q(c_)], dtype=torch.float64))
        n_items = cb[:, 3].long()
        total = int(n_items.sum())
        assert int(n_items.max()) <= TRACE_ITEMS, f"raise TRACE_ITEMS: {int(n_items.max())}"
        it_ = cb[:, 8 : 8 + TRACE_SLOTS * TRACE_ITEMS].view(num_cu - 1, TRACE_ITEMS, TRACE_SLOTS)
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
        r_land = [((land.min() - p0) / 100.0) if len(land) else float("nan"),
                  ((land.max() - p0) / 100.0) if len(land) else float("nan")]
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
            clk_l.append(float((mt_l * m).sum()) / max(float((rt_l * m).sum()), 1.0) * 100.0)
        rows.append(torch.tensor([float(v) for v in r + r_land + clk + comp_l + clk_l
                                  + [prep_plan, prep_stage, prep_q1, prep_tot, prep_pub]],
                                 dtype=torch.float64))
    if fused_rows and rank == 0:
        fm = torch.stack(fused_rows).median(dim=0).values.tolist()
        for i_, n_ in enumerate(("start", "prepare_done", "layout_done(=tiles start)", "tiles_done")):
            print(f"TRACE FUSED {n_:28s} min/median/max over workers (us): {fm[3*i_]:.1f} / {fm[3*i_+1]:.1f} / {fm[3*i_+2]:.1f}", flush=True)
    med = torch.stack(rows).median(dim=0).values.to(prep.trace_buf.device)
    dist.all_reduce(med)
    med = (med / world).tolist()
    if rank == 0:
        names = ["prep_end", "first_put", "last_put", "gemm_start", "start_gate", "comm_done", "gemm_end"]
        print("TRACE " + " ".join(f"{n}={v:.1f}" for n, v in zip(names, med[:7])), flush=True)
        print("TRACE gate wait per WG by tile slice (us): "
              + " ".join(f"{v:.1f}" for v in med[7 : 7 + B])
              + f" | sum {sum(med[7 : 7 + B]):.1f}", flush=True)
        print("TRACE slice first passed (us): "
              + " ".join(f"{v:.0f}" for v in med[7 + B : 7 + 2 * B]), flush=True)
        print("TRACE compute per WG by tile slice (us): "
              + " ".join(f"{v:.0f}" for v in med[7 + 2 * B : 7 + 3 * B])
              + f" | sum {sum(med[7 + 2 * B : 7 + 3 * B]):.0f}", flush=True)
        print(f"TRACE all chunks landed (first peer / last peer, us): {med[7 + 3 * B]:.0f} / "
              f"{med[7 + 3 * B + 1]:.0f}", flush=True)
        print("TRACE core clock by tile slice (MHz, s_memtime / s_memrealtime): "
              + " ".join(f"{v:.0f}" for v in med[9 + 3 * B : 9 + 4 * B]), flush=True)
        print("TRACE all landed: compute per WG by tile slice (us): "
              + " ".join(f"{v:.0f}" for v in med[9 + 4 * B : 9 + 5 * B])
              + f" | sum {sum(med[9 + 4 * B : 9 + 5 * B]):.0f}", flush=True)
        print("TRACE all landed: core clock by tile slice (MHz): "
              + " ".join(f"{v:.0f}" for v in med[9 + 5 * B : 9 + 6 * B]), flush=True)
        pl, stg, q1, tot, pub = med[9 + 6 * B : 9 + 6 * B + 5]
        print(f"TRACE prepare breakdown (slowest block, us): plan {pl:.1f}, staging {stg:.1f} "
              f"(publish {pub:.1f}), quant pass1 {q1:.1f}, total {tot:.1f}", flush=True)


def trace_report_rf(args, prep, gtrace, gemm1, x, ids, wts, sbm, moe, chunks_from, rank, world):
    """Ready-first timeline, ``TRACE-RF`` lines. Every time is in us from the start of the call on
    that rank (the standalone prepare's first block, or the GEMM1's first block when the prepare is
    fused), the median over the ``--trace`` calls; each value is printed ``rank 0 | median over
    ranks`` and ``a/b/c`` is min / median / max over the blocks of that role. Stamps are written by
    one lane after the barrier that ends the phase, so they do not change what a block does."""
    num_cu, P, LB = int(moe._s1_num_cu), PREP_SLOTS, int(args.layout_blocks)
    NW = num_cu - 1 - LB
    sb = 1 if args.sched_layout else 0                  # the scheduler's block
    first_w = 2 if args.sched_layout else 1             # first worker block
    quant_b = list(range(1, 1 + NW)) if LB else list(range(1, num_cu))
    layout_b = list(range(1 + NW, num_cu)) if LB else list(range(first_w, num_cu))
    worker_b = list(range(first_w, num_cu))
    nan = float("nan")
    iters = []
    for it in range(args.trace + 2):
        torch.cuda.synchronize()
        dist.barrier()
        prep.arm(chunks_from)
        gtrace.zero_()
        prep.trace_buf.zero_()
        prepare_step(prep, x, ids, wts, args)
        layout_step(prep, moe, sbm, args)
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
            return torch.stack([pt[b * P + c] for b in blocks]) if blocks else torch.zeros(0, dtype=torch.float64)

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
        r.append(("COMM  rest chunks posted (last wave)", one(w16.max()) if bool((w16 > 0).any()) else [nan]))
        r.append(("COMM  qdone reached", one(g[0, 5])))
        puts = pt[P * num_cu : P * num_cu + 16]
        puts = puts[puts > 0]
        r.append(("COMM  early puts posted first/last", [float(puts.min() - t0) / 100.0, float(puts.max() - t0) / 100.0]
                  if len(puts) else [nan, nan]))
        r.append(("QUANT B1 passed", tri(pcol(quant_b, 1))))
        r.append(("QUANT pass 1 done (B3 reached)", tri(pcol(quant_b, 6))))
        r.append(("QUANT B3 passed", tri(pcol(quant_b, 8))))
        r.append(("QUANT prepare_phase returned", tri(pcol(quant_b, 3))))
        if LB:
            r.append(("LAYOUT-BLOCKS B3 passed", tri(pcol(layout_b, 8))))
        r.append(("LAYOUT start" + (" (blocks %d..%d)" % (layout_b[0], layout_b[-1]) if LB else " (all workers)"),
                  tri(col(layout_b, 4)) if args.fuse_prepare else tri(col(layout_b, 0))))
        r.append(("LAYOUT end", tri(col(layout_b, 1))))
        r.append(("WORKERS first tile starts", tri(col(worker_b, 8))))
        r.append(("WORKERS loop ends", tri(col(worker_b, 2))))
        r.append(("WORKERS tiles run (sum, not us)", [float(col(worker_b, 3).sum())]))
        ran = [b for b in worker_b if float(g[b, 3]) > 0]

        def dur(c):
            v = torch.stack([g[b, c] for b in ran]) / 100.0 if ran else torch.zeros(0, dtype=torch.float64)
            return [float(v.min()), float(v.median()), float(v.max())] if len(v) else [nan] * 3

        r.append(("WORKERS queue wait, tiles 1.. (us, sum per worker)", dur(5)))
        r.append(("WORKERS first queue wait (us)", dur(6)))
        r.append(("WORKERS pure tile time (us, sum per worker)", dur(7)))
        if not args.fuse_prepare:
            r.append(("PREP-KERNEL plan / staging / pass1 / total (slowest block, us)",
                      span(0, 2) + span(4, 5) + span(2, 6) + span(0, 3)))
        iters.append(r)
        last_g, last_t0 = g, t0
    labels = [l for l, _ in iters[0]]
    widths = [len(v) for _, v in iters[0]]
    flat = torch.tensor([[x_ for _, v in r for x_ in v] for r in iters], dtype=torch.float64)
    med = flat.nanmedian(dim=0).values.to(x.device)
    gl = [torch.zeros_like(med) for _ in range(world)]
    dist.all_gather(gl, med)
    allm = torch.stack(gl).cpu()
    over = allm.nanmedian(dim=0).values
    if rank == 0:
        def fmt(vs):
            return "/".join("-" if v != v else f"{v:.1f}" for v in vs)

        print(f"TRACE-RF bs={args.bs} own_tokens={int(args.own_tokens)} fuse_prepare={int(args.fuse_prepare)} "
              f"quant_skip={int(args.quant_skip)} quant_lds_store={int(args.quant_lds_store)} "
              f"plan_blocks={int(args.plan_blocks)} layout_blocks={LB} local_first={int(args.local_first)} "
              f"sched_layout={int(args.sched_layout)}: us from call start, median of {args.trace} calls; "
              f"'rank0 | median over {world} ranks'; a/b/c = min/median/max over blocks", flush=True)
        i = 0
        for lab, w in zip(labels, widths):
            print(f"TRACE-RF {lab:58s} {fmt(allm[0, i:i + w].tolist()):>22s} | {fmt(over[i:i + w].tolist())}", flush=True)
            i += w
        if args.arrival_order:
            aw = prep.sched[SCH_ARR + ARR_IDX : SCH_ARR + ARR_IDX + world].cpu().tolist()
            order = sorted((a - 1, s_) for s_, a in enumerate(aw) if s_ != rank and a > 0)
            c0 = {s_: round((float(last_g[sb, 24 + s_]) - last_t0) / 100.0, 1)
                  for s_ in range(world) if s_ != rank}
            print(f"TRACE-RF ARRIVAL ORDER rank0 (last call; index 0 = local {rank}): "
                  f"{[rank] + [s_ for _, s_ in order]}; chunk0 seen at (us) {c0}", flush=True)


def graph_timing(args, prep, moe, config, sbm, x, ids, wts, sx, ss, gemm1, check,
                 chunks_from, window, stage, my_out, rank, world, eager):
    """Both sides captured in a CUDA graph each, timed per replay like the eager loop (sync and
    barrier between calls). The SDMA graph holds arm() too, so every replay is one full call."""
    device = x.device
    chunks_t = torch.tensor(chunks_from, dtype=torch.int32, device=device)

    def sdma_call():
        prep.arm(chunks_t)
        prepare_step(prep, x, ids, wts, args)
        layout_step(prep, moe, sbm, args)
        gemm1(gated=True)

    def aiter_call():
        fs = fx.Stream(torch.cuda.current_stream().cuda_stream)
        moe._run_compact_prepare(ids, args.bs, config.stage1, fs, quant_input=x)
        moe._run_fused_stage1(sx, wts, ss, ids, stream=fs, config=config.stage1, prepared=True)

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
            tail = f" | eager: SDMA {eager[3]:.3f}, aiter {eager[4]:.3f} ({eager[4] / eager[3]:.2f}x)"
        print(
            f"RESULT-GRAPH {args.network} bs={args.bs}: SDMA {n_:.3f} ms, aiter {o_:.3f} ms "
            f"({o_ / n_:.2f}x){tail}",
            flush=True,
        )
    del g_sdma, g_aiter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=ex11.NETWORKS, default="glm52")
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument("--early-chunks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stress", type=int, default=20)
    ap.add_argument("--time-iters", type=int, default=30)
    ap.add_argument("--trace", type=int, default=0,
                    help="trace this many gated calls: where in GEMM1's tile order the gates wait")
    ap.add_argument("--buckets", type=int, default=10)
    ap.add_argument("--graph", action="store_true",
                    help="also time both sides captured in CUDA graphs (same process, after eager)")
    ap.add_argument("--stream", action="store_true",
                    help="prepare puts every chunk as soon as its rows are in S (per-chunk ready "
                         "counters, no rendezvous); --early-chunks is then ignored")
    ap.add_argument("--split", type=int, default=1,
                    help="prepare quant: waves sharing one token (each a slice of its hidden)")
    ap.add_argument("--unroll", type=int, default=1,
                    help="prepare quant: work items a wave loads before it stores")
    ap.add_argument("--prep-block", type=int, default=512,
                    help="threads per prepare block (waves per CU; the grid is resident)")
    ap.add_argument("--big-rest", action="store_true",
                    help="everything after the early chunks goes out as one SDMA copy")
    ap.add_argument("--graph-only", action="store_true",
                    help="time ONLY in CUDA graphs (no eager timing at all); implies --graph")
    ap.add_argument("--fuse-layout", action="store_true",
                    help="GEMM1 builds the receive layout itself (no separate layout kernel)")
    ap.add_argument("--own-tokens", action="store_true",
                    help="prepare: each block places, quantises and stages its own tokens, which "
                         "drops the plan's second grid barrier")
    ap.add_argument("--fuse-prepare", action="store_true",
                    help="GEMM1 also runs the prepare (one launch per call); implies --fuse-layout")
    ap.add_argument("--ready-first", action="store_true",
                    help="GEMM1: a scheduler wave (comm workgroup) publishes m-tiles in the order "
                         "they become ready; workers take tickets and read the ready queue "
                         "instead of gating each tile on the SDMA signals")
    ap.add_argument("--local-first", action="store_true",
                    help="(needs --ready-first) layout orders each expert's rows by source, "
                         "local rank first, and a tile waits only for the sources it has rows of")
    ap.add_argument("--sched-layout", action="store_true",
                    help="(needs --ready-first) block 1 of the GEMM1 does the layout AND the "
                         "scheduling with block barriers only (no grid rendezvous); workers are "
                         "blocks 2..; implies --fuse-layout")
    ap.add_argument("--quant-skip", action="store_true",
                    help="prepare: a quant pass loads and quantises only the tokens it stores "
                         "(with --own-tokens: per-block LDS lists of the early / the rest tokens "
                         "built by the plan; without: a wave-uniform skip)")
    ap.add_argument("--quant-lds-store", action="store_true",
                    help="prepare: stage each wave's quantised round of 64 groups in LDS and "
                         "store it back so every lane writes 16 CONTIGUOUS bytes (the wave 1024 "
                         "bytes per instruction, e8m0 scales packed 4 to a dword) instead of two "
                         "half-line pieces 32 bytes apart; bit-identical bytes, 2304 LDS bytes "
                         "per wave")
    ap.add_argument("--plan-blocks", type=int, default=0,
                    help="(needs --own-tokens, not --quant-skip) run the prepare's plan on the "
                         "first P worker blocks only, one token per thread over P*prep_block "
                         "threads (0 = off: every worker block plans). More blocks do not shorten "
                         "a one-token-per-thread scan, they only multiply the hist/cursor atomics. "
                         "Rule: P = ceil(bs / prep_block); -1 computes that on the host. Any P is "
                         "correct whatever the token count, it only changes the speed. Costs the "
                         "grid barrier B2 that --own-tokens had removed.")
    ap.add_argument("--layout-blocks", type=int, default=0, choices=[0, 8, 16, 32],
                    help="(needs --fuse-prepare --own-tokens --ready-first) dedicate the LAST L "
                         "worker blocks to the layout: they own no tokens, run layout_phase "
                         "among themselves right after the prepare's B3 while the other workers "
                         "quantise (0 = off: every worker runs the layout)")
    ap.add_argument("--local-only", action="store_true",
                    help="reference run with NO dispatch traffic: every token's experts are replaced "
                         "by topk distinct experts of THIS rank, so every route is local (no put, no "
                         "signal). Both our path and aiter's then run quant + prepare + GEMM1 on "
                         "local data only (6*bs GEMM rows per rank, as before, but only bs staged "
                         "rows): the time left is what the dispatch costs on top. Outputs are still "
                         "compared with aiter's (same ids).")
    ap.add_argument("--hot-experts", type=int, default=0,
                    help="(with --local-only) draw every token's experts from only the first K "
                         "experts of this rank (K >= topk): the same GEMM FLOPs, but a tiny weight "
                         "working set that stays in L2/MALL. If the GEMM1 time drops a lot, it is "
                         "weight-fetch bound; if not, it is compute / latency bound.")
    ap.add_argument("--prep-blocks", type=int, default=0,
                    help="quantising workgroups of the STANDALONE prepare kernel (0 = num_cu-1). "
                         "The grid is resident, so more blocks = more waves per CU = more memory "
                         "parallelism for the quant; not allowed with --fuse-prepare (there the "
                         "prepare must use the GEMM1 grid).")
    ap.add_argument("--pair-k", action="store_true",
                    help="GEMM1 do_tile: two K steps per loop iteration (avoids the per-step vmcnt drain of the B prefetch)")
    ap.add_argument("--async-a", type=int, default=-1, help="override async_a_copy (0/1)")
    ap.add_argument("--amajor", type=int, default=-1, help="override mfma_amajor (0/1)")
    ap.add_argument("--sched-cache-need", action="store_true",
                    help="(needs --ready-first) scheduler reads need[s][e] once into LDS and skips "
                         "fully published slices, so a round costs ~one signal-level load")
    ap.add_argument("--arrival-order", action="store_true",
                    help="(needs --local-first) the layout orders each expert's rows by the ORDER "
                         "IN WHICH THE SOURCES' ROUTE FLAGS ARRIVED (local source first) instead of "
                         "the fixed rotation; the scheduler follows that order")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the stage2 comparison (needed with --sbm: aiter's stage2 is "
                         "compiled for its own sort_block_m). Timings only, NO correctness.")
    ap.add_argument("--sbm", type=int, default=0,
                    help="override GEMM1's sort_block_m (0 = aiter's choice for this shape)")
    ap.add_argument("--tile-n", type=int, default=0,
                    help="override GEMM1's tile_n (0 = aiter's choice); LDS needs "
                         "sbm * tile_n / 2 * 4 bytes")
    ap.add_argument("--head-rows", type=int, default=0,
                    help="rows of the FIRST chunk of a run (0 = off: all chunks are "
                         "--chunk-rows); a short head starts the first put sooner without "
                         "making every later chunk small")
    ap.add_argument("--stage-first", action="store_true",
                    help="prepare pushes the routes before the quant, so the peer's layout "
                         "overlaps with the rest of prepare")
    ap.add_argument("--quant-sweep", default="",
                    help="in one process, per SPLITxUNROLL config (e.g. 1x1,3x1,3x2): 3 poisoned "
                         "checked calls, then prepare / layout / gated GEMM1 timed")
    args = ap.parse_args()
    if args.graph_only:
        args.graph = True
    if args.fuse_prepare:
        args.fuse_layout = True
    if args.sched_layout:
        if not args.ready_first:
            ap.error("--sched-layout needs --ready-first")
        args.fuse_layout = True
    if args.layout_blocks:
        if not (args.fuse_prepare and args.own_tokens and args.ready_first):
            ap.error("--layout-blocks needs --fuse-prepare, --own-tokens and --ready-first")
        if args.sched_layout:
            ap.error("--layout-blocks and --sched-layout are exclusive")
    if args.sched_cache_need and not args.ready_first:
        ap.error("--sched-cache-need needs --ready-first")
    if args.arrival_order and not args.local_first:
        ap.error("--arrival-order needs --local-first")
    if args.prep_blocks and args.fuse_prepare:
        ap.error("--prep-blocks cannot be used with --fuse-prepare")
    if args.local_first and not args.ready_first:
        ap.error("--local-first needs --ready-first")
    if args.plan_blocks:
        if not args.own_tokens:
            ap.error("--plan-blocks needs --own-tokens")
        if args.quant_skip:
            ap.error("--plan-blocks and --quant-skip are exclusive (the quant_skip lists are "
                     "built by the block that places a token)")
        if args.plan_blocks == -1:
            # ``tokens`` is a kernel argument but P is a compile-time constant, so the rule
            # P = ceil(tokens / prep_block) is applied HERE, to this run's batch size.
            args.plan_blocks = max(1, -(-int(args.bs) // int(args.prep_block)))
        elif args.plan_blocks < 1:
            ap.error("--plan-blocks must be -1 (host rule) or >= 1")

    rank, world, device = ex11._setup()
    net, moe = ex11._build_moe(args, args.bs, rank, world, device)
    hidden, topk = net["model_dim"], net["topk"]
    x, wts, ids = ex11._make_inputs(
        args.bs, hidden, net["experts"], topk, rank, args.seed, device
    )
    if args.local_only:
        epr_l = net["experts"] // world
        g_l = torch.Generator(device=device)
        g_l.manual_seed(args.seed + 1000 + rank)
        k_l = int(args.hot_experts) if args.hot_experts else epr_l
        assert topk <= k_l <= epr_l, "--hot-experts must be in [topk, experts per rank]"
        ids = (torch.rand(args.bs, k_l, generator=g_l, device=device).topk(topk, dim=1).indices
               + rank * epr_l).to(ids.dtype)
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
    fan_ok = torch.tensor([int(torch.equal(ref, ref2))], dtype=torch.int32, device=device)
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
            stream=args.stream,
            split=args.split,
            unroll=args.unroll,
            prep_block=args.prep_block,
            stage_first=args.stage_first,
            quant_blocks=(args.prep_blocks or None),
            own_tokens=args.own_tokens,
            big_rest=args.big_rest,
            head_rows=args.head_rows,
            sched=args.ready_first,
            quant_skip=args.quant_skip,
            quant_lds_store=args.quant_lds_store,
            plan_blocks=args.plan_blocks,
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
        prep.run(x, ids, wts, recv_layout=True)  # the setup always uses the separate kernel
        prep.post_rest()
        cnt_all = [torch.zeros(world, dtype=torch.int32) for _ in range(world)]
        dist.all_gather(cnt_all, prep.cnt.cpu())
        cnt_all = torch.stack(cnt_all)
        chunks_from = [
            prep.signals_for(int(cnt_all[s][rank])) if s != rank else 0
            for s in range(world)
        ]
        prep.wait(chunks_from)
        prep.layout(moe, sbm)  # the setup always uses the separate kernel
        torch.cuda.synchronize()

        my_out = torch.zeros_like(moe._s1_out)
        my_osd = torch.zeros_like(moe._s1_osd)
        kwargs = gemm1_kwargs(moe, config)
        kwargs["sort_block_m"] = sbm
        if args.tile_n:
            kwargs["tile_n"] = int(args.tile_n)
        if args.pair_k:
            kwargs["pair_k"] = True
        if args.async_a >= 0:
            kwargs["async_a_copy"] = bool(args.async_a)
        if args.amajor >= 0:
            kwargs["mfma_amajor"] = bool(args.amajor)

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

        rf_kwargs = dict(ready_first=args.ready_first, local_first=args.local_first,
                         sched_layout=args.sched_layout, layout_blocks=args.layout_blocks,
                         sched_cache_need=args.sched_cache_need,
                         arrival_order=args.arrival_order)

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
                fx.Stream(torch.cuda.current_stream().cuda_stream),  # the capture stream too
                prep=prep if gated else None,
                trace=trace,
                fuse=fused_args() if (gated and args.fuse_layout) else None,
                **kwargs, **(rf_kwargs if gated else {}),
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
            print(f"num_valid={int(op.num_valid[0])} rows (padded to SBM={sbm})", flush=True)
        stage2_and_check("landed")

        def call():
            prepare_step(prep, x, ids, wts, args)
            layout_step(prep, moe, sbm, args)
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

        if args.quant_sweep:
            # One CUDA graph per config: 3 poisoned replays checked, then replays timed against
            # aiter's own graph. Eager numbers are not reported -- the real path is graphed.
            chunks_t = torch.tensor(chunks_from, dtype=torch.int32, device=device)
            sx, ss = moe._s1_quant_x[: args.bs], moe._s1_quant_scale[: args.bs]

            def capture(fn):
                torch.cuda.synchronize()
                dist.barrier()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                torch.cuda.synchronize()
                dist.barrier()
                return g

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

            def aiter_call():
                fs = fx.Stream(torch.cuda.current_stream().cuda_stream)
                moe._run_compact_prepare(ids, args.bs, config.stage1, fs, quant_input=x)
                moe._run_fused_stage1(sx, wts, ss, ids, stream=fs, config=config.stage1,
                                      prepared=True)

            g_aiter = capture(aiter_call)
            old_ms = timed(g_aiter)
            del g_aiter
            for spec in args.quant_sweep.split(","):
                v = [int(t) for t in spec.split("x")]
                sp, un = v[0], v[1]
                pb = v[2] if len(v) > 2 else args.prep_block
                sf = bool(v[3]) if len(v) > 3 else args.stage_first
                prep.split, prep.unroll, prep.prep_block = sp, un, pb
                prep.stage_first = sf
                prep._prepares.clear()
                prep.arm(chunks_from)
                call()  # warm up / compile this config before capture
                torch.cuda.synchronize()

                def sdma_call():
                    prep.arm(chunks_t)
                    prepare_step(prep, x, ids, wts, args)
                    layout_step(prep, moe, sbm, args)
                    gemm1(gated=True)

                g_sdma = capture(sdma_call)
                ok = 0
                for it in range(3):
                    window.fill_(0xAA)
                    stage.fill_(-7)
                    prep.rowmap.zero_()
                    my_out.zero_()
                    torch.cuda.synchronize()
                    dist.barrier()
                    g_sdma.replay()
                    torch.cuda.synchronize()
                    ok += int(stage2_and_check(f"quant-sweep {spec} call {it}"))
                new_ms = timed(g_sdma)
                del g_sdma
                med = torch.tensor([new_ms, old_ms, ok], dtype=torch.float64, device=device)
                dist.all_reduce(med)
                med /= world
                if rank == 0:
                    n_, o_, ok_ = med.tolist()
                    print(f"QSWEEP-GRAPH {args.network} bs={args.bs} split={sp} unroll={un} "
                          f"block={pb} sf={int(sf)}: SDMA {n_:.3f} ms, aiter {o_:.3f} ms "
                          f"({o_ / n_:.2f}x), "
                          f"checked {ok_:.2f}/3 exact", flush=True)
            prep.split, prep.unroll, prep.prep_block = args.split, args.unroll, args.prep_block
            prep.stage_first = args.stage_first
            prep._prepares.clear()
            prep.arm(chunks_from)
            call()
            torch.cuda.synchronize()
            stage2_and_check("after quant sweep")

        if args.trace:
            (trace_report_rf if args.ready_first else trace_report)(
                args, prep, gtrace, gemm1, x, ids, wts, sbm, moe, chunks_from, rank, world)

        if args.graph_only and args.time_iters and not args.quant_sweep:
            sx, ss = moe._s1_quant_x[: args.bs], moe._s1_quant_scale[: args.bs]
            graph_timing(args, prep, moe, config, sbm, x, ids, wts, sx, ss, gemm1,
                         stage2_and_check, chunks_from, window, stage, my_out, rank, world, None)
            torch.cuda.synchronize()
            dist.barrier()
            prep.arm(chunks_from)
            call()
            torch.cuda.synchronize()
            stage2_and_check("after timing")
        if args.time_iters and not args.quant_sweep and not args.graph_only:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            parts, e2e, base = [], [], []
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                prep.arm(chunks_from)
                ev[0].record()
                prepare_step(prep, x, ids, wts, args)
                ev[1].record()
                layout_step(prep, moe, sbm, args)
                ev[2].record()
                gemm1(gated=True)
                ev[3].record()
                torch.cuda.synchronize()
                if it >= 3:
                    parts.append([ev[i].elapsed_time(ev[i + 1]) for i in range(3)])
                    e2e.append(ev[0].elapsed_time(ev[3]))
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
            rest_from = [max(c - prep.arena.early_chunks, 0) if s_ != rank else 0
                         for s_, c in enumerate(chunks_from)]
            landed = []
            sig_base = prep.sig_base.clone()  # wait() re-arms: the gates must keep this call's base
            # (not with --fuse-prepare: there a gated GEMM1 is a whole call, and repeating it
            # without ``arm`` leaves the signal counters ahead of the host's expected totals)
            for it in range(0 if (args.fuse_prepare or args.ready_first) else args.time_iters + 3):
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
                moe._run_compact_prepare(ids, args.bs, config.stage1, fstream, quant_input=x)
                ev[2].record()
                moe._run_fused_stage1(
                    sx, wts, ss, ids, stream=fstream, config=config.stage1, prepared=True
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
            cnts = torch.tensor([float(torch.tensor(a_prep).median()), float(torch.tensor(a_s1).median()),
                                 routes_remote, rows_remote, rows_all, int(g.numel())],
                                dtype=torch.float64, device=x.device)
            dist.all_reduce(cnts)
            cnts /= world
            if rank == 0:
                ap_, as_, rr, wr, wa, ra = cnts.tolist()
                rb = prep.arena.row_bytes
                print(f"SPLIT aiter: compact prepare {ap_:.3f} ms, fused stage1 {as_:.3f} ms | per rank: "
                      f"routes {ra:.0f} (remote {rr:.0f}), rows {wa:.0f} (remote {wr:.0f}), "
                      f"route/row {ra / max(wa, 1):.2f}, remote bytes SDMA {wr * rb / 1e6:.1f} MB "
                      f"vs per-route {rr * rb / 1e6:.1f} MB", flush=True)
            med = torch.tensor(
                torch.tensor(parts, dtype=torch.float64).median(dim=0).values.tolist()
                + [torch.tensor(e2e).median().item(), torch.tensor(base).median().item(),
                   torch.tensor(ung).median().item(), torch.tensor(landed).median().item()],
                dtype=torch.float64,
            )
            dist.all_reduce(med)
            med /= world
            if rank == 0:
                p_, l_, g_, new_ms, old_ms, u_, gl_ = med.tolist()
                print(
                    f"parts: prepare {p_:.3f} ms, layout {l_:.3f} ms, gated GEMM1 {g_:.3f} ms "
                    f"(all landed: gated {gl_:.3f} ms, ungated {u_:.3f} ms)",
                    flush=True,
                )
                print(
                    f"RESULT {args.network} bs={args.bs}: new prepare + layout + gated GEMM1 "
                    f"{new_ms:.3f} ms (no aiter plan); aiter prepare+stage1 {old_ms:.3f} ms "
                    f"({old_ms / new_ms:.2f}x)",
                    flush=True,
                )
            if args.graph:
                graph_timing(args, prep, moe, config, sbm, x, ids, wts, sx, ss, gemm1,
                             stage2_and_check, chunks_from, window, stage, my_out, rank, world,
                             med.tolist())
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

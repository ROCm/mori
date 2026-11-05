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

from gemm1_sdma import TRACE_ITEMS, TRACE_STRIDE, run_gemm1_sdma  # noqa: E402
from stage1_sdma import Stage1SdmaPrepare  # noqa: E402
from test_gemm1 import gemm1_kwargs  # noqa: E402


def trace_report(args, prep, gtrace, gemm1, x, ids, wts, sbm, moe, chunks_from, rank, world):
    """Gate wait by position in GEMM1's tile order (flat work index, in ``--buckets`` equal
    slices), and when each slice first got past its gate. Times in us from prepare's start."""
    num_cu, nb, B = moe._s1_num_cu, prep.quant_blocks + 1, args.buckets
    rows = []
    for it in range(args.trace + 2):
        torch.cuda.synchronize()
        dist.barrier()
        prep.arm(chunks_from)
        gtrace.zero_()
        prep.trace_buf.zero_()
        prep.run(x, ids, wts, recv_layout=True)
        prep.layout(moe, sbm)
        gemm1(gated=True, trace=gtrace)
        torch.cuda.synchronize()
        if it < 2:
            continue
        pt, g = prep.trace_buf.cpu().double(), gtrace.cpu().view(num_cu, TRACE_STRIDE).double()
        p0 = pt[0 : 4 * nb : 4].min()
        prep_end = pt[3 : 4 * nb : 4].max()
        puts = pt[4 * nb : 4 * nb + 16]
        puts = puts[puts > 0]
        cb = g[1:]
        n_items = cb[:, 3].long()
        total = int(n_items.sum())
        assert int(n_items.max()) <= TRACE_ITEMS, f"raise TRACE_ITEMS: {int(n_items.max())}"
        it_ = cb[:, 8 : 8 + 3 * TRACE_ITEMS].view(num_cu - 1, TRACE_ITEMS, 3)
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
        rows.append(torch.tensor([float(v) for v in r], dtype=torch.float64))
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
              + " ".join(f"{v:.0f}" for v in med[7 + 2 * B :])
              + f" | sum {sum(med[7 + 2 * B :]):.0f}", flush=True)


def graph_timing(args, prep, moe, config, sbm, x, ids, wts, sx, ss, gemm1, check,
                 chunks_from, window, stage, my_out, rank, world, eager):
    """Both sides captured in a CUDA graph each, timed per replay like the eager loop (sync and
    barrier between calls). The SDMA graph holds arm() too, so every replay is one full call."""
    device = x.device
    chunks_t = torch.tensor(chunks_from, dtype=torch.int32, device=device)

    def sdma_call():
        prep.arm(chunks_t)
        prep.run(x, ids, wts, recv_layout=True)
        prep.layout(moe, sbm)
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
        e_new, e_old = eager[3], eager[4]
        print(
            f"RESULT-GRAPH {args.network} bs={args.bs}: SDMA {n_:.3f} ms, aiter {o_:.3f} ms "
            f"({o_ / n_:.2f}x) | eager: SDMA {e_new:.3f}, aiter {e_old:.3f} ({e_old / e_new:.2f}x)",
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
    ap.add_argument("--stream-group", type=int, default=0,
                    help="> 0: prepare puts every chunk, each group of this many chunks as soon "
                         "as its tokens are quantised (--early-chunks is then ignored)")
    args = ap.parse_args()

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
    sbm = int(config.stage1.sort_block_m)
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
            stream_group=args.stream_group,
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
        prep.run(x, ids, wts, recv_layout=True)
        prep.post_rest()
        cnt_all = [torch.zeros(world, dtype=torch.int32) for _ in range(world)]
        dist.all_gather(cnt_all, prep.cnt.cpu())
        cnt_all = torch.stack(cnt_all)
        chunks_from = [
            -(-int(cnt_all[s][rank]) // args.chunk_rows) if s != rank else 0
            for s in range(world)
        ]
        prep.wait(chunks_from)
        prep.layout(moe, sbm)
        torch.cuda.synchronize()

        my_out = torch.zeros_like(moe._s1_out)
        my_osd = torch.zeros_like(moe._s1_osd)
        kwargs = gemm1_kwargs(moe, config)

        gtrace = (
            torch.zeros(moe._s1_num_cu * TRACE_STRIDE, dtype=torch.int64, device=device)
            if args.trace
            else None
        )

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
                **kwargs,
            )

        def stage2_and_check(label):
            nonlocal errors
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
            prep.run(x, ids, wts, recv_layout=True)
            prep.layout(moe, sbm)
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
            trace_report(args, prep, gtrace, gemm1, x, ids, wts, sbm, moe, chunks_from, rank, world)

        if args.time_iters:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            parts, e2e, base = [], [], []
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                prep.arm(chunks_from)
                ev[0].record()
                prep.run(x, ids, wts, recv_layout=True)
                ev[1].record()
                prep.layout(moe, sbm)
                ev[2].record()
                gemm1(gated=True)
                ev[3].record()
                torch.cuda.synchronize()
                if it >= 3:
                    parts.append([ev[i].elapsed_time(ev[i + 1]) for i in range(3)])
                    e2e.append(ev[0].elapsed_time(ev[3]))
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
            for it in range(args.time_iters + 3):
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
            sx, ss = moe._s1_quant_x[: args.bs], moe._s1_quant_scale[: args.bs]
            fstream = fx.Stream(torch.cuda.current_stream().cuda_stream)
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                ev[0].record()
                moe._run_compact_prepare(ids, args.bs, config.stage1, fstream, quant_input=x)
                moe._run_fused_stage1(
                    sx, wts, ss, ids, stream=fstream, config=config.stage1, prepared=True
                )
                ev[1].record()
                torch.cuda.synchronize()
                if it >= 3:
                    base.append(ev[0].elapsed_time(ev[1]))
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

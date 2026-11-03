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
"""M2: GEMM1 on the rows the new prepare delivered, against aiter's stage1.

aiter runs once (prepare, stage1, stage2, combine); its stage1 output is kept. Then the new
prepare (plan, route push, quant, puts) runs, and ``gemm1_sdma`` computes stage1 from the
window ``[S | R]`` through ``rowmap``. Checked two ways:

* every expert-major row the rank received: the fp8 output row is the same bytes;
* stage2 + combine run on the new stage1 output give aiter's output bit for bit.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 test_gemm1.py --bs 4096
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

from gemm1_sdma import run_gemm1_sdma  # noqa: E402
from stage1_sdma import Stage1SdmaPrepare  # noqa: E402


def gemm1_kwargs(moe, config):
    s1 = config.stage1
    return dict(
        model_dim=moe.model_dim,
        inter_dim=moe.inter_dim,
        expert_offset=moe.rank * moe.epr,
        sort_block_m=s1.sort_block_m,
        tile_n=s1.tile_n,
        tile_k=s1.tile_k,
        num_waves=s1.num_waves,
        pipe_weights=s1.pipe_weights,
        mfma_amajor=s1.mfma_amajor,
        swizzle_a=s1.swizzle_a,
        async_a_copy=s1.async_a_copy,
        use_tile_resource=s1.use_tile_resource,
        waves_per_eu_hint=s1.waves_per_eu_hint,
        b_cache_modifier=max(int(s1.b_nt), 0),
        swiglu_limit=moe.swiglu_limit,
        num_cu=moe._s1_num_cu,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=ex11.NETWORKS, default="glm52")
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument(
        "--early-chunks",
        type=int,
        default=1,
        help="chunks per run that prepare itself puts",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--time-iters", type=int, default=0)
    ap.add_argument(
        "--stress",
        type=int,
        default=0,
        help="repeat the poisoned gated call this many times",
    )
    ap.add_argument("--stall-test", action="store_true")
    ap.add_argument(
        "--skip-gemm1",
        action="store_true",
        help="negative control: the test must then fail",
    )
    args = ap.parse_args()

    rank, world, device = ex11._setup()
    net, moe = ex11._build_moe(args, args.bs, rank, world, device)
    hidden, epr, topk = net["model_dim"], net["experts"] // world, net["topk"]
    x, wts, ids = ex11._make_inputs(
        args.bs, hidden, net["experts"], topk, rank, args.seed, device
    )
    ref = moe(x, wts, ids)[: args.bs].clone()
    torch.cuda.synchronize()
    config = moe._active_config
    ws, op = moe._s1_dispatch_workspace, moe._s1_op
    plan = (ws["pair_order"], ws["pair_base"], ws["local_hist"], ws["my_base"])
    ref_out = moe._s1_out.clone()

    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    vmm = 4 * world * moe.mtpr * hidden * 2 + (512 << 20)
    errors = 0
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
        )
        a = prep.arena
        window = from_gpu_ptr(prep.mem.ptr, (2 * a.n_rows, a.row_bytes), torch.uint8)
        prep.rowmap.zero_()  # a padding row reads row 0 of S
        prep.srcmap.fill_(-1)
        torch.cuda.synchronize()
        dist.barrier()

        cnt_all = None

        def dispatch():
            nonlocal cnt_all
            prep.run(x, ids, wts, plan)
            prep.post_rest()

        dispatch()
        cnt_all = [torch.zeros(world, dtype=torch.int32) for _ in range(world)]
        dist.all_gather(cnt_all, prep.cnt.cpu())
        cnt_all = torch.stack(cnt_all)
        chunks_from = [
            -(-int(cnt_all[s][rank]) // args.chunk_rows) if s != rank else 0
            for s in range(world)
        ]
        prep.wait(chunks_from)
        torch.cuda.synchronize()

        my_out = torch.zeros_like(moe._s1_out)
        my_osd = torch.zeros_like(moe._s1_osd)
        kwargs = gemm1_kwargs(moe, config)
        stream = fx.Stream(torch.cuda.current_stream().cuda_stream)

        def gemm1():
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
                stream,
                **kwargs,
            )

        if not args.skip_gemm1:
            gemm1()
        torch.cuda.synchronize()

        valid = (prep.srcmap != -1).nonzero().flatten()
        bad = (my_out[valid].view(torch.uint8) != ref_out[valid].view(torch.uint8)).any(
            dim=1
        )
        if int(bad.sum()):
            errors += 1
            print(
                f"[rank {rank}] stage1 out: {int(bad.sum())}/{valid.numel()} rows differ",
                flush=True,
            )

        # Stage2 + combine on the new stage1 output.
        moe._s1_out.copy_(my_out)
        moe._s1_osd.copy_(my_osd)
        res = moe._run_stage2(args.bs, None, True, config, None)[: args.bs]
        torch.cuda.synchronize()
        diff = (res.float() - ref.float()).abs().max()
        if not torch.equal(res, ref):
            errors += 1
            print(
                f"[rank {rank}] final output differs from aiter's: max abs diff {float(diff):.3g}",
                flush=True,
            )

        def check(label):
            """My stage1 output against aiter's, then stage2 + combine on it."""
            nonlocal errors
            torch.cuda.synchronize()
            bad = (
                my_out[valid].view(torch.uint8) != ref_out[valid].view(torch.uint8)
            ).any(dim=1)
            if int(bad.sum()):
                errors += 1
                print(
                    f"[rank {rank}] {label}: stage1 out {int(bad.sum())}/{valid.numel()} rows differ",
                    flush=True,
                )
            moe._s1_out.copy_(my_out)
            moe._s1_osd.copy_(my_osd)
            res = moe._run_stage2(args.bs, None, True, config, None)[: args.bs]
            torch.cuda.synchronize()
            if not torch.equal(res, ref):
                errors += 1
                print(
                    f"[rank {rank}] {label}: final output differs from aiter's, max abs diff {float((res.float() - ref.float()).abs().max()):.3g}",
                    flush=True,
                )

        def call_gated():
            prep.run(x, ids, wts, plan)
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
                stream,
                prep=prep,
                **kwargs,
            )

        # M3: no wait kernel; the GEMM waits for its chunks itself. Everything the call has
        # to receive is poisoned first, so stale data of the call above cannot pass.
        need = from_gpu_ptr(prep.mem.ptr + a.off_need, (world * epr,), torch.int32)
        window.fill_(0xAA)
        for t in (prep.rowmap, need):
            t.zero_()
        prep.srcmap.fill_(-1)
        my_out.zero_()
        my_osd.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        prep.arm(chunks_from)
        call_gated()
        check("gated")

        if args.stress:
            # Poison everything this call has to receive, run it, compare the stage1 rows with
            # aiter's: again and again, to catch a stale line that only shows now and then.
            bad_calls = 0
            for it in range(args.stress):
                window.fill_(0xAA)
                for t_ in (prep.rowmap, need):
                    t_.zero_()
                prep.srcmap.fill_(-1)
                my_out.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                prep.arm(chunks_from)
                call_gated()
                torch.cuda.synchronize()
                if bool(
                    (
                        my_out[valid].view(torch.uint8)
                        != ref_out[valid].view(torch.uint8)
                    ).any()
                ):
                    bad_calls += 1
            errors += bad_calls
            if bad_calls:
                print(
                    f"[rank {rank}] stress: {bad_calls}/{args.stress} calls differ",
                    flush=True,
                )
            if rank == 0:
                print(f"stress: {args.stress} poisoned calls per rank", flush=True)

        if args.stall_test:
            # GEMM1 twice in a row after one prepare. The first has data arriving while it runs;
            # when the second starts everything has landed, so it cannot wait for any chunk.
            # What the first costs more than the second is the most waiting can account for.
            # (The second GEMM1's comm workgroup puts the chunks again, so both have SDMA traffic
            # in flight; the extra signals are added to the running totals.)
            def gemm_args():
                return (
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
                    stream,
                )

            early = [min(c, args.early_chunks) for c in chunks_from]
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            rows = []
            for it in range(33):
                torch.cuda.synchronize()
                dist.barrier()
                prep.arm(chunks_from)
                evs[0].record()
                prep.run(x, ids, wts, plan)
                evs[1].record()
                run_gemm1_sdma(*gemm_args(), prep=prep, **kwargs)
                evs[2].record()
                run_gemm1_sdma(*gemm_args(), prep=prep, **kwargs)
                evs[3].record()
                torch.cuda.synchronize()
                # the second GEMM1 put its chunks again
                prep.expected[:world] += torch.tensor(
                    [c - e for c, e in zip(chunks_from, early)],
                    dtype=torch.int32,
                    device=device,
                )
                if it >= 3:
                    rows.append(
                        [evs[1].elapsed_time(evs[2]), evs[2].elapsed_time(evs[3])]
                    )
            m = torch.tensor(rows, dtype=torch.float64).median(dim=0).values
            dist.all_reduce(m)
            if rank == 0:
                print(
                    f"stall test bs={args.bs}: GEMM1 with data arriving {float(m[0]) / world:.3f} ms, GEMM1 right after, all landed {float(m[1]) / world:.3f} ms",
                    flush=True,
                )

        if args.time_iters:
            # Where the time goes, one call at a time: prepare, gated GEMM1.
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            parts_it = []
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                prep.arm(chunks_from)
                evs[0].record()
                prep.run(x, ids, wts, plan)
                evs[1].record()
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
                    stream,
                    prep=prep,
                    **kwargs,
                )
                evs[2].record()
                torch.cuda.synchronize()
                if it >= 3:
                    parts_it.append([evs[i].elapsed_time(evs[i + 1]) for i in range(2)])
            parts = (
                torch.tensor(parts_it, dtype=torch.float64).median(dim=0).values
            )  # median: the host is noisy
            dist.all_reduce(parts)
            if rank == 0:
                a_, b_ = (v / world for v in parts.tolist())
                print(
                    f"parts: prepare {a_:.3f} ms, gated GEMM1 {b_:.3f} ms", flush=True
                )
            # GEMM1 alone on data that has all landed, without the gate or the comm workgroup.
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for _ in range(3):
                gemm1()
            torch.cuda.synchronize()
            dist.barrier()
            ev[0].record()
            for _ in range(args.time_iters):
                gemm1()
            ev[1].record()
            torch.cuda.synchronize()
            g = torch.tensor(
                [ev[0].elapsed_time(ev[1]) / args.time_iters], dtype=torch.float64
            )
            dist.all_reduce(g)
            if rank == 0:
                print(
                    f"GEMM1 alone on landed data: {float(g) / world:.3f} ms", flush=True
                )
            acc_it = []
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                prep.arm(chunks_from)  # the running totals, before the timed part
                ev[0].record()
                call_gated()
                ev[1].record()
                torch.cuda.synchronize()
                if it >= 3:
                    acc_it.append(ev[0].elapsed_time(ev[1]))
            acc = torch.tensor([acc_it], dtype=torch.float64).median(dim=1).values
            # Today's path for the same inputs: aiter's compact prepare and its fused stage1.
            base_it = []
            sx, ss = moe._s1_quant_x[: args.bs], moe._s1_quant_scale[: args.bs]
            fstream = fx.Stream(torch.cuda.current_stream().cuda_stream)
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                ev[0].record()
                moe._run_compact_prepare(
                    ids, args.bs, config.stage1, fstream, quant_input=x
                )
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
                    base_it.append(ev[0].elapsed_time(ev[1]))
            base = torch.tensor([base_it], dtype=torch.float64).median(dim=1).values
            both = torch.cat([acc, base])
            dist.all_reduce(both)
            if rank == 0:
                new_ms, old_ms = (v / world for v in both.tolist())
                print(
                    f"bs={args.bs}: new prepare + gated GEMM1 {new_ms:.3f} ms; aiter prepare+stage1 {old_ms:.3f} ms ({old_ms / new_ms:.2f}x)",
                    flush=True,
                )
        prep.close()
    err = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(err)
    if rank == 0:
        print("SUCCESS" if int(err) == 0 else f"FAILED: {int(err)} mismatches")
    dist.destroy_process_group()
    return int(err) != 0


if __name__ == "__main__":
    raise SystemExit(main())

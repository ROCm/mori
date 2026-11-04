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

from gemm1_sdma import run_gemm1_sdma  # noqa: E402
from stage1_sdma import Stage1SdmaPrepare  # noqa: E402
from test_gemm1 import gemm1_kwargs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=ex11.NETWORKS, default="glm52")
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument("--early-chunks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stress", type=int, default=20)
    ap.add_argument("--time-iters", type=int, default=30)
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
        stream = fx.Stream(torch.cuda.current_stream().cuda_stream)

        def gemm1(gated):
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
                prep=prep if gated else None,
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
                + [torch.tensor(e2e).median().item(), torch.tensor(base).median().item()],
                dtype=torch.float64,
            )
            dist.all_reduce(med)
            med /= world
            if rank == 0:
                p_, l_, g_, new_ms, old_ms = med.tolist()
                print(
                    f"parts: prepare {p_:.3f} ms, layout {l_:.3f} ms, gated GEMM1 {g_:.3f} ms",
                    flush=True,
                )
                print(
                    f"RESULT {args.network} bs={args.bs}: new prepare + layout + gated GEMM1 "
                    f"{new_ms:.3f} ms (no aiter plan); aiter prepare+stage1 {old_ms:.3f} ms "
                    f"({old_ms / new_ms:.2f}x)",
                    flush=True,
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

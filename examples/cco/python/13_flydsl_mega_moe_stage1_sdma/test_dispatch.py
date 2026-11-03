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
"""M1 end to end: what the new prepare delivers against what aiter's stage1 dispatch delivers.

aiter runs once (``MegaMoEV2``: prepare, stage1 with its CU producers, stage2), which leaves
its dispatch state behind: the expert-major rows ``rx_em`` with their scales, ``wts_em`` and
``srcmap_em``, and the compact plan's tables. Then the new prepare runs on the same inputs
and routes: one kernel plans, quantises into ``S``, puts the chunks, and pushes the route
table built from aiter's plan. For every expert-major row ``r`` that this rank received::

    R[rowmap[r]]  ==  rx_em[r] (payload and e8m0 scale)
    wts[r]        ==  wts_em[r]          srcmap[r]  ==  srcmap_em[r]

and the number of rows is the number of routes the other ranks have for my experts.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 test_dispatch.py --bs 4096
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "11_flydsl_mega_moe_sdma")
)

import main as ex11  # noqa: E402  (sets the shmem env defaults, imports torch first)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from mori.cco import Communicator, UniqueId  # noqa: E402
from mori.tensor_utils import from_gpu_ptr  # noqa: E402

from stage1_sdma import Stage1SdmaPrepare  # noqa: E402


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
    args = ap.parse_args()

    rank, world, device = ex11._setup()
    net, moe = ex11._build_moe(args, args.bs, rank, world, device)
    hidden, epr, topk = net["model_dim"], net["experts"] // world, net["topk"]
    x, wts, ids = ex11._make_inputs(
        args.bs, hidden, net["experts"], topk, rank, args.seed, device
    )
    moe(x, wts, ids)  # aiter's dispatch state for these inputs
    torch.cuda.synchronize()
    ws, op = moe._s1_dispatch_workspace, moe._s1_op
    plan = (ws["pair_order"], ws["pair_base"], ws["local_hist"], ws["my_base"])

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
        SR = from_gpu_ptr(
            prep.mem.ptr, (2 * a.n_rows, a.row_bytes), torch.uint8
        )  # S then R
        for t in (prep.rowmap, prep.wts, prep.srcmap):
            t.fill_(-1)
        torch.cuda.synchronize()
        dist.barrier()

        prep.run(x, ids, wts, plan)
        prep.post_rest()
        cnt_all = [torch.zeros(world, dtype=torch.int32) for _ in range(world)]
        dist.all_gather(cnt_all, prep.cnt.cpu())
        cnt_all = torch.stack(cnt_all)  # [src, dst]
        prep.wait(
            [
                -(-int(cnt_all[s][rank]) // args.chunk_rows) if s != rank else 0
                for s in range(world)
            ]
        )
        torch.cuda.synchronize()

        # How many routes the ranks have for my experts: local_hist[ge] summed over sources.
        hist_all = [torch.zeros(world * epr, dtype=torch.int32) for _ in range(world)]
        dist.all_gather(hist_all, ws["local_hist"][: world * epr].cpu())
        expect_rows = int(torch.stack(hist_all)[:, rank * epr : (rank + 1) * epr].sum())

        srcmap = prep.srcmap
        got = (srcmap != -1).nonzero().flatten()
        if got.numel() != expect_rows:
            errors += 1
            print(
                f"[rank {rank}] {got.numel()} route rows, expected {expect_rows}",
                flush=True,
            )
        rows = prep.rowmap[got].long()
        rx = op._ll_views()["rx_em"][got].view(torch.uint8)
        scale = op._ll_views()["scale_em"][got][:, : hidden // 32]
        checks = {
            "srcmap": (srcmap[got] != op.srcmap_em[got]),
            "weight": (prep.wts[got] != op.wts_em[got].view(torch.int32)),
            "payload": (SR[rows, :hidden] != rx).any(dim=1),
            "scale": (SR[rows, hidden : hidden + hidden // 32] != scale).any(dim=1),
        }
        for name, bad in checks.items():
            nbad = int(bad.sum())
            if nbad:
                errors += 1
                print(
                    f"[rank {rank}] {name}: {nbad}/{got.numel()} rows differ",
                    flush=True,
                )
        if rank == 0:
            hist0 = hist_all[0]
            routes_remote = int(
                sum(
                    hist0[d * epr : (d + 1) * epr].sum() for d in range(world) if d != 0
                )
            )
            rows_remote = int(sum(cnt_all[0][d] for d in range(world) if d != 0))
            print(
                f"rank 0 sends {rows_remote} rows to other ranks for {routes_remote} remote routes "
                f"({rows_remote / routes_remote:.1%}); {args.bs} tokens x topk {topk} = {args.bs * topk} routes in all",
                flush=True,
            )
            print(
                "| dst | routes (token,expert) | rows sent (deduped) | saved |\n|---:|---:|---:|---:|",
                flush=True,
            )
            for d in range(world):
                rt, rw = int(hist0[d * epr : (d + 1) * epr].sum()), int(cnt_all[0][d])
                print(
                    f"| {d}{' (self, not sent)' if d == 0 else ''} | {rt} | {rw} | {1 - rw / rt:.1%} |",
                    flush=True,
                )
            print(f"rank 0 received {got.numel()} expert-major rows", flush=True)
        if args.time_iters:
            chunks_from = [
                -(-int(cnt_all[s][rank]) // args.chunk_rows) if s != rank else 0
                for s in range(world)
            ]
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            acc = torch.zeros(2, dtype=torch.float64)
            for it in range(args.time_iters + 3):
                torch.cuda.synchronize()
                dist.barrier()
                ev[0].record()
                prep.run(x, ids, wts, plan)
                ev[1].record()
                prep.post_rest()
                prep.wait(chunks_from)
                ev[2].record()
                torch.cuda.synchronize()
                if it >= 3:
                    acc += torch.tensor(
                        [ev[i].elapsed_time(ev[i + 1]) for i in range(2)],
                        dtype=torch.float64,
                    )
            acc /= args.time_iters
            dist.all_reduce(acc)
            if rank == 0:
                a_, b_ = (v / world for v in acc.tolist())
                print(
                    f"time bs={args.bs}: prepare (plan + route push + quant + puts) {a_:.3f} ms, then wait {b_:.3f} ms (all landed {a_ + b_:.3f} ms)",
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

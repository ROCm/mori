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
"""Check payload, metadata, reverse mapping and sparse routing at a pinned geometry."""
import argparse
import os
import torch
import torch.distributed as dist
from mori.cco import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp


def worker(local_rank, node, args):
    rank = node * 8 + local_rank
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="16",
        LOCAL_RANK=str(local_rank),
        LOCAL_WORLD_SIZE="8",
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    uid = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    wire = torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fnuz
    scale_dim = 0 if args.dtype == "bf16" else args.hidden // 128
    with Communicator.init(16, rank, uid[0], per_rank_vmm=2**31) as comm:
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=16,
            hidden_dim=args.hidden,
            max_num_inp_token_per_rank=128,
            num_experts_per_rank=args.epr,
            num_experts_per_token=args.topk,
            gpu_per_node=8,
            num_qp_per_pe=args.num_qp,
            data_type=torch.bfloat16,
            dispatch_data_type=wire,
            combine_data_type=torch.bfloat16,
            scale_dim=scale_dim,
            scale_type_size=4 if scale_dim else 0,
            kernel_backend="hip",
            internode_kernel=args.family,
        )
        op = EpDispatchCombineOp(cfg, comm)
        errors = 0
        for iteration in range(4):
            n = 65 + rank % 3
            if iteration == 3 and rank % 3 == 0:
                n = 0
            rng = torch.Generator().manual_seed(7733 + rank + iteration * 100)
            x = torch.randint(-2, 3, (n, args.hidden), generator=rng).float()
            # Small integers are exact in both wire formats; token IDs use base 16.
            x[:, 0] = rank
            x[:, 1] = torch.arange(n) % 16
            x[:, 2] = torch.arange(n) // 16
            idx = (
                torch.stack(
                    [
                        torch.randperm(16 * args.epr, generator=rng)[: args.topk]
                        for _ in range(n)
                    ]
                ).int()
                if n
                else torch.empty(0, args.topk, dtype=torch.int32)
            )
            if iteration == 1:
                idx = (idx % (8 * args.epr)) + (rank // 8) * (8 * args.epr)
            if iteration == 2:
                idx[::2, 1::2] = -1
                idx[::5] = -1
            w = torch.randint(0, 8, (n, args.topk), generator=rng).float()
            scales = torch.randint(0, 10000, (n, scale_dim), generator=rng).float()
            gathered = [None] * 16
            dist.all_gather_object(gathered, (x, idx, w, scales))
            inp = x.cuda().to(wire)
            ids = idx.cuda()
            weights = w.cuda()
            sc = scales.cuda() if scale_dim else None
            rx, rw, rs, ri, count, routing = op.dispatch(
                inp, weights, sc, ids, return_routing=True
            )
            torch.cuda.synchronize()
            comm.barrier()
            nr = int(count.item())
            want = []
            for source, (sx, si, sw, ss) in enumerate(gathered):
                mask = ((si >= 0) & (si // args.epr == rank)).any(1)
                want.extend((source, t) for t in torch.where(mask)[0].tolist())
            gotx = rx[:nr].float().cpu()
            gotw = rw[:nr].cpu()
            goti = ri[:nr].cpu()
            gots = rs[:nr].cpu() if scale_dim else None
            keys = [(int(row[0]), int(row[1]) + 16 * int(row[2])) for row in gotx]
            good = sorted(keys) == sorted(want)
            for j, (source, token) in enumerate(keys):
                if source not in range(16) or token not in range(
                    len(gathered[source][0])
                ):
                    good = False
                    continue
                sx, si, sw, ss = gathered[source]
                good = (
                    good
                    and torch.equal(gotx[j], sx[token])
                    and torch.equal(goti[j], si[token])
                    and torch.equal(gotw[j], sw[token])
                )
                # recv_scales exposes opaque dwords, not float32 scale values.
                if scale_dim:
                    good = good and torch.equal(gots[j], ss[token].view(torch.int32))
            # Different destination transforms keep a wrong reverse map from cancelling.
            post = rx.to(torch.bfloat16) * (rank + 1)
            result, result_w = op.combine(post, rw, routing=routing)
            torch.cuda.synchronize()
            peers = [set((row[row >= 0] // args.epr).tolist()) for row in idx]
            factors = torch.tensor([sum(p + 1 for p in ps) for ps in peers]).view(n, 1)
            multiplicity = torch.tensor([len(ps) for ps in peers]).view(n, 1)
            expected = x * factors
            bound = 0.008 * expected.abs().clamp(min=1)
            good = (
                good
                and bool(((result.float().cpu() - expected).abs() <= bound).all())
                and torch.equal(result_w.cpu(), w * multiplicity)
            )
            bad = torch.tensor([int(not good)])
            dist.all_reduce(bad)
            errors += int(bad.item())
            if rank == 0:
                print(
                    "STRICT",
                    args.family,
                    args.dtype,
                    "iteration",
                    iteration,
                    "bad_ranks",
                    bad.item(),
                    "PASS" if bad.item() == 0 else "FAIL",
                    flush=True,
                )
            comm.barrier()
        op.close()
    dist.destroy_process_group()
    if errors:
        raise RuntimeError(f"{errors} failing rank-rounds")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("v2", "v2_ll"), required=True)
    p.add_argument("--dtype", choices=("bf16", "fp8"), required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--epr", type=int, required=True)
    p.add_argument("--num-qp", type=int, default=2)
    args = p.parse_args()
    if (
        int(os.environ.get("WORLD_SIZE", "0")) != 2
        or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 1
    ):
        p.error(
            "requires torchrun --nnodes=2 --nproc_per_node=1 on two eight-GPU nodes"
        )
    if args.hidden < 3 or args.epr < 1 or not 1 <= args.topk <= 16 * args.epr:
        p.error("requires hidden>=3, experts/rank>0 and 1<=topk<=total experts")
    if args.num_qp < 1:
        p.error("--num-qp must be positive")
    torch.multiprocessing.spawn(worker, args=(int(os.environ["RANK"]), args), nprocs=8)

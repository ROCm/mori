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
"""M1: the new prepare on 8 GPUs -- every row a peer sends me lands in ``R`` as aiter's quant makes it.

For every source ``s`` the receiver regenerates ``s``'s inputs from the seed, quantises them
with aiter's ``per_1x32_mx_quant`` and compares row by row with ``R`` (and, for my own run,
``S``): payload bytes and e8m0 scales. Run twice, to see the second call's counters work.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 test_prepare.py --bs 4096
"""

import argparse
import os

import torch
import torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.tensor_utils import from_gpu_ptr

from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

from stage1_sdma import Stage1SdmaPrepare

NETWORKS = {
    "glm52": dict(hidden=6144, experts=256, topk=8),
    "v4_pro": dict(hidden=7168, experts=384, topk=6),
}


def make_inputs(rank, seed, tokens, hidden, experts, topk, device):
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device, generator=g)
    scores = torch.randn((tokens, experts), device=device, generator=g)
    ids = torch.topk(scores, topk, dim=-1).indices.to(torch.int32).contiguous()
    return x, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=NETWORKS, default="glm52")
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument(
        "--early-chunks",
        type=int,
        default=1,
        help="chunks per run that prepare itself puts",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument(
        "--time-iters", type=int, default=0, help="also time this many calls"
    )
    args = ap.parse_args()
    net = NETWORKS[args.network]

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", local)
    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)

    max_tok = 1 << (args.bs - 1).bit_length()
    row_bytes = net["hidden"] + net["hidden"] // 32
    vmm = 2 * 2 * world * max_tok * row_bytes + (256 << 20)
    errors = 0
    with Communicator.init(
        world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
    ) as comm:
        prep = Stage1SdmaPrepare(
            comm,
            max_tok=max_tok,
            chunk_rows=args.chunk_rows,
            early_chunks=args.early_chunks,
            **net,
        )
        a = prep.arena
        S = from_gpu_ptr(prep.mem.ptr + a.off_s, (a.n_rows, a.row_bytes), torch.uint8)
        R = from_gpu_ptr(prep.mem.ptr + a.off_r, (a.n_rows, a.row_bytes), torch.uint8)
        hidden = net["hidden"]

        x, ids = make_inputs(rank, args.seed, args.bs, device=dev, **net)
        for call in range(args.calls):
            S.fill_(0xAA)
            R.fill_(0xAA)  # poison: a row that never lands cannot match
            torch.cuda.synchronize()
            dist.barrier()
            prep.run(x, ids)
            prep.post_rest()
            torch.cuda.synchronize()
            # What every rank sent to every rank (rows), and which token each row is.
            cnt_all = [torch.zeros(world, dtype=torch.int32) for _ in range(world)]
            dist.all_gather(cnt_all, prep.cnt.cpu())
            tok_all = [
                torch.zeros(prep.row_tok.numel(), dtype=torch.int32)
                for _ in range(world)
            ]
            dist.all_gather(tok_all, prep.row_tok.cpu())
            cnt_all = torch.stack(cnt_all)  # [src, dst]
            chunks_from = [
                -(-int(cnt_all[s][rank]) // args.chunk_rows) if s != rank else 0
                for s in range(world)
            ]
            prep.wait(chunks_from, routes=False)
            torch.cuda.synchronize()

            for src in range(world):
                n = int(cnt_all[src][rank])
                xs, _ = make_inputs(src, args.seed, args.bs, device=dev, **net)
                ref_q, ref_s = per_1x32_mx_quant(xs, quant_mode="fp8")
                ref_q = ref_q.view(torch.uint8)
                tok = tok_all[src][rank * max_tok : rank * max_tok + n].long().to(dev)
                if src == rank:
                    rows = S[rank * max_tok : rank * max_tok + n]
                else:
                    rows = R[src * max_tok : src * max_tok + n]
                bad_q = (rows[:, :hidden] != ref_q[tok]).any(dim=1)
                bad_s = (rows[:, hidden : hidden + hidden // 32] != ref_s[tok]).any(
                    dim=1
                )
                nbad = int((bad_q | bad_s).sum())
                if nbad:
                    errors += 1
                    print(
                        f"[rank {rank}] call {call} src {src}: {nbad}/{n} rows differ",
                        flush=True,
                    )
            if rank == 0:
                print(
                    f"call {call}: rows sent by rank 0 -> "
                    + " ".join(str(int(c)) for c in cnt_all[0]),
                    flush=True,
                )
        if args.time_iters:
            # One call at a time, ranks lined up: prepare alone, and until every row landed.
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
                prep.run(x, ids)
                ev[1].record()  # prepare alone: plan, quant, chunk 0
                prep.post_rest()
                prep.wait(chunks_from, routes=False)
                ev[2].record()
                torch.cuda.synchronize()
                if it >= 3:
                    acc += torch.tensor(
                        [ev[0].elapsed_time(ev[1]), ev[0].elapsed_time(ev[2])],
                        dtype=torch.float64,
                    )
            acc /= args.time_iters
            dist.all_reduce(acc)
            if rank == 0:
                print(
                    f"time bs={args.bs} chunk_rows={args.chunk_rows}: prepare (plan + quant + chunk 0) {acc[0] / world:.3f} ms, until landed {acc[1] / world:.3f} ms",
                    flush=True,
                )
        prep.close()
    err = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(err)
    if rank == 0:
        print(
            "SUCCESS"
            if int(err) == 0
            else f"FAILED: {int(err)} mismatching (rank, src) pairs"
        )
    dist.destroy_process_group()
    return int(err) != 0


if __name__ == "__main__":
    raise SystemExit(main())

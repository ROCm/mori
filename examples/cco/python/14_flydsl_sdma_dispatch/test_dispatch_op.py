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
"""``SdmaDispatch`` against a torch reference, on 8 GPUs.

Every rank regenerates every other rank's inputs from the seed, works out which of their
tokens route to it, and checks that each received row carries that token's quantised payload,
its e8m0, its expert ids, its weights and its source id -- byte for byte against
``per_1x32_mx_quant``. The window is poisoned before each call, so a row that never lands
cannot pass.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 test_dispatch_op.py --bs 4096
"""

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "20G")
os.environ.setdefault("MORI_ENABLE_SDMA", "1")

import torch  # noqa: E402  (must come before anything that pulls in mori.cco)
import torch.distributed as dist  # noqa: E402
from mori.cco import Communicator, UniqueId  # noqa: E402

from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant  # noqa: E402

from sdma_dispatch import SdmaDispatch  # noqa: E402

NETWORKS = {
    "glm52": dict(hidden=6144, experts=256, topk=8),
    "v4_pro": dict(hidden=7168, experts=384, topk=6),
    "flash": dict(hidden=5120, experts=384, topk=6),
}


def make_inputs(rank, seed, tokens, hidden, experts, topk, device):
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device, generator=g)
    scores = torch.randn((tokens, experts), device=device, generator=g)
    v, ids = torch.topk(scores, topk, dim=-1)
    return x.contiguous(), ids.to(torch.int32).contiguous(), v.softmax(-1).contiguous()


def check_call(op, args, net, rank, world, epr, dev, label):
    """Every delivered row against the reference, found through its source id: the row says
    which rank and which token it came from, so there is nothing to guess about the order.
    """
    hidden, experts, topk = net["hidden"], net["experts"], net["topk"]
    cap = op.max_tok
    nrecv = int(op.nrecv()[0].item())
    src = op.src()[:nrecv]
    valid = (src >= 0).nonzero().flatten()
    pay = op.payload()[:nrecv].view(torch.uint8)
    sc, got_ids, got_wts = op.scales()[:nrecv], op.ids()[:nrecv], op.weights()[:nrecv]

    bad = 0
    seen = 0
    for s_rank in range(world):
        xs, ids_s, wts_s = make_inputs(
            s_rank, args.seed, args.bs, hidden, experts, topk, dev
        )
        want = (ids_s // epr == rank).any(dim=1).nonzero().flatten()
        here = valid[(src[valid] // cap) == s_rank]
        if here.numel() != want.numel():
            print(
                f"[rank {rank}] {label} src {s_rank}: {here.numel()} rows, "
                f"expected {want.numel()}",
                flush=True,
            )
            bad += 1
            continue
        seen += here.numel()
        tok = (src[here] % cap).long()
        if not torch.equal(tok.sort().values, want.sort().values):
            print(
                f"[rank {rank}] {label} src {s_rank}: wrong set of tokens", flush=True
            )
            bad += 1
            continue
        q, qs = per_1x32_mx_quant(xs, quant_mode="fp8")
        for name, got, ref in (
            ("payload", pay[here], q.view(torch.uint8)[tok]),
            ("e8m0", sc[here], qs[tok]),
            ("ids", got_ids[here], ids_s[tok]),
            ("weights", got_wts[here], wts_s[tok]),
        ):
            n = int((got != ref).any(dim=1).sum())
            if n:
                print(
                    f"[rank {rank}] {label} src {s_rank} {name}: {n}/{here.numel()} differ",
                    flush=True,
                )
                bad += 1
    # a padding row must be marked on both arrays
    padm = src[:nrecv] < 0
    if int(padm.sum()) and int((got_ids[padm] != -1).sum()):
        print(f"[rank {rank}] {label}: a padding row has a live expert id", flush=True)
        bad += 1
    if seen + int(padm.sum()) != nrecv:
        print(
            f"[rank {rank}] {label}: {nrecv} rows, {seen} routed + {int(padm.sum())} padding",
            flush=True,
        )
        bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=NETWORKS, default="flash")
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--chunk-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument(
        "--graph", type=int, default=0, help="capture the call and replay it N times"
    )
    args = ap.parse_args()
    net = NETWORKS[args.network]
    hidden, experts, topk = net["hidden"], net["experts"], net["topk"]

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", local)
    epr = experts // world
    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)

    max_tok = 1 << (args.bs - 1).bit_length()
    vmm = 8 * world * max_tok * (hidden + hidden // 32 + 64) + (1 << 30)
    errors = 0
    with Communicator.init(
        world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
    ) as comm:
        op = SdmaDispatch(
            comm,
            experts=experts,
            topk=topk,
            hidden=hidden,
            max_tok=max_tok,
            chunk_rows=args.chunk_rows,
        )
        x, ids, wts = make_inputs(rank, args.seed, args.bs, hidden, experts, topk, dev)

        for call in range(args.calls):
            op.payload().view(torch.uint8).fill_(0xAA)
            op.ids().fill_(-7)
            torch.cuda.synchronize()
            dist.barrier()
            op.dispatch(x, ids, wts)
            torch.cuda.synchronize()
            # the allocator itself: every source's claim at a destination must be disjoint
            mine = torch.stack(
                [
                    op.peer_base.cpu() // args.chunk_rows,
                    (op.cnt.cpu() + args.chunk_rows - 1) // args.chunk_rows,
                ]
            )
            allb = [torch.zeros_like(mine) for _ in range(world)]
            dist.all_gather(allb, mine)
            allb = torch.stack(allb)  # [src, 2, dst]
            for d in range(world):
                iv = sorted(
                    (int(allb[s][0][d]), int(allb[s][1][d]), s) for s in range(world)
                )
                for (b0, n0, s0), (b1, _, s1) in zip(iv, iv[1:]):
                    if b0 + n0 > b1:
                        if rank == 0:
                            print(
                                f"call {call}: at dst {d}, src {s0} took chunks "
                                f"[{b0},{b0 + n0}) and src {s1} starts at {b1}",
                                flush=True,
                            )
                        errors += 1
            errors += check_call(op, args, net, rank, world, epr, dev, f"call {call}")
            if rank == 0:
                import sys as _s

                print(
                    f"call {call}: nrecv={int(op.nrecv()[0].item())} "
                    f"cnt={op.cnt.tolist()} base={(op.peer_base // args.chunk_rows).tolist()} "
                    f"call={int(op.callno[0].item())}",
                    flush=True,
                    file=_s.stderr,
                )

        if args.graph:
            op.dispatch(x, ids, wts)  # warm up: FlyDSL JITs on first call
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=torch.cuda.Stream()):
                op.dispatch(x, ids, wts)
            torch.cuda.synchronize()
            dist.barrier()
            op.payload().view(torch.uint8).fill_(0xAA)
            op.ids().fill_(-7)
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(args.graph):
                g.replay()
            torch.cuda.synchronize()
            errors += check_call(
                op, args, net, rank, world, epr, dev, f"graph x{args.graph}"
            )
            if rank == 0:
                print(f"graph: {args.graph} replays ok", flush=True)
        op.close()

    err = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(err)
    if rank == 0:
        print("SUCCESS" if int(err) == 0 else f"FAILED: {int(err)} mismatches")
    dist.destroy_process_group()
    return int(err) != 0


if __name__ == "__main__":
    raise SystemExit(main())

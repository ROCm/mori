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
"""Attribute #475 combine corruption to the transport a token actually used.

The reporter's probe uses uniform random routing, under which almost every token
has destinations on both nodes -- so it cannot say whether corruption follows the
RDMA path or hits tokens that never left the node. This probe *constructs* the
routing so each token belongs to a known class:

    local   all topk experts owned by ranks on MY node   -> XGMI/P2P only, no RDMA
    remote  all topk experts owned by ranks on the OTHER node -> every partial via RDMA
    mixed   half and half                                -> both

Expectation is unchanged (`input * unique_pes`, same as mori's own
check_combine_result), and the payload is a bf16-exact small integer per token so
the realized multiplier is recovered exactly as combined[t,0] / x_t.

Reading the result: a nonzero rate in `local` does NOT mean the intra-node path is
broken -- local tokens share the combine kernel, its chunk bookkeeping and its
buffers with remote traffic, so they can be collateral damage. What the three rates
together separate is whether corruption *tracks the wire* (only remote/mixed hit)
or is a property of the kernel's bookkeeping (all three hit).
"""
import argparse
import os
import sys

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import mori  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_475_combine_probe import build_config  # noqa: E402

# `uniform` reproduces the reporter's own routing (topk over all experts) and acts
# as the in-run control: it is the only class whose corruption rate is already known
# (~1% at this scale). If the constructed classes come out orders of magnitude worse
# while `uniform` matches the known rate, the skew -- not the probe -- is the trigger.
CLASSES = tuple(
    (os.environ.get("PATH_CLASSES") or "local,mixed,remote,uniform").split(",")
)


def sample_from_pool(T, pool, k, gen):
    """Pick k distinct experts per token out of `pool` (a 1-D tensor of expert ids)."""
    sel = torch.rand((T, pool.numel()), generator=gen).topk(k, dim=1).indices
    return pool[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384, help="per rank, split 3 ways")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--gpu-per-node", type=int, default=8)
    ap.add_argument("--blocks", default="96,64,8,8")
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()

    block_num, rdma_block_num, dwarps, cwarps = (int(v) for v in args.blocks.split(","))
    NCLS = len(CLASSES)
    T = args.tokens - args.tokens % NCLS
    per = T // NCLS

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group("cpu:gloo,cuda:nccl")
    epr = args.experts // world_size
    gpn = args.gpu_per_node
    my_node = rank // gpn
    experts_per_node = gpn * epr

    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("default")

    config = build_config(
        "InterNodeV1",
        rank,
        world_size,
        args.hidden,
        args.topk,
        epr,
        max(512, T),
        gpn,
        (block_num, rdma_block_num, dwarps, cwarps),
    )
    os.environ["MORI_EP_LAUNCH_CONFIG_MODE"] = "MANUAL"
    op = mori.ops.EpDispatchCombineOp(config)

    if rank == 0:
        print(
            f"[path] world={world_size} gpu_per_node={gpn} T={T} "
            f"({per}/class) rounds={args.rounds}",
            flush=True,
        )

    # Expert ids owned by this node's ranks vs the other node's.
    all_experts = torch.arange(args.experts)
    node_of_expert = all_experts // experts_per_node
    local_pool = all_experts[node_of_expert == my_node]
    remote_pool = all_experts[node_of_expert != my_node]

    # class id per token: 0=local, 1=mixed, 2=remote
    cls = torch.cat([torch.full((per,), i, dtype=torch.long) for i in range(NCLS)]).to(
        device
    )

    tally = torch.zeros((NCLS, 2), device=device)  # [class] x [n_tokens, n_bad]
    # For mixed/remote, how the failure looks: remote partial gone vs polluted.
    shape = torch.zeros(
        (NCLS, 3), device=device
    )  # [class] x [dropped, polluted, other]

    for r in range(args.rounds):
        g = torch.Generator().manual_seed(args.seed * 7919 + r * 131 + rank)
        half = args.topk // 2
        blocks_by_class = {
            "local": lambda: sample_from_pool(per, local_pool, args.topk, g),
            "mixed": lambda: torch.cat(
                [
                    sample_from_pool(per, local_pool, half, g),
                    sample_from_pool(per, remote_pool, args.topk - half, g),
                ],
                dim=1,
            ),
            "remote": lambda: sample_from_pool(per, remote_pool, args.topk, g),
            # control: the reporter's own uniform topk-over-all-experts routing
            "uniform": lambda: torch.rand((per, args.experts), generator=g)
            .topk(args.topk, dim=1)
            .indices,
        }
        idx = (
            torch.cat([blocks_by_class[c]() for c in CLASSES])
            .to(torch.int32)
            .to(device)
        )

        w = torch.rand((T, args.topk), generator=g).to(torch.float32)
        w = (w / w.sum(dim=1, keepdim=True)).to(device)
        vals = ((torch.arange(T) + rank * T + r) % 15 + 1).to(torch.bfloat16)
        x = vals.unsqueeze(1).expand(T, args.hidden).contiguous().to(device)
        scales = torch.empty((T, 0), dtype=torch.uint8, device=device)

        dest = idx.long() // epr
        one_hot = torch.zeros((T, world_size), device=device)
        one_hot.scatter_(1, dest, 1.0)
        m = one_hot.sum(dim=1)
        node_of_rank = torch.arange(world_size, device=device) // gpn
        local_only = (one_hot * (node_of_rank == my_node).float().unsqueeze(0)).sum(
            dim=1
        )

        # Dispatch oracle: with constructed routing the load is far more skewed
        # than under uniform routing, so verify dispatch itself before trusting
        # any combine number -- otherwise a dispatch-side overflow would be
        # misread as combine corruption.
        gathered = [torch.empty_like(idx) for _ in range(world_size)]
        dist.all_gather(gathered, idx)
        gdest = torch.cat(gathered).long() // epr
        expected_recv = int((gdest == rank).any(dim=1).sum().item())

        out, _ow, _sc, out_idx, recv_num = op.dispatch(
            x,
            w,
            scales,
            idx,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=dwarps,
        )
        got_recv = int(recv_num[0].item())
        if got_recv != expected_recv:
            print(
                f"[path][rank {rank}] round {r} DISPATCH MISMATCH "
                f"recv={got_recv} expected={expected_recv} -- combine numbers "
                f"for this round are not meaningful",
                flush=True,
            )
        combined = op.combine(
            out,
            None,
            out_idx,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=cwarps,
        )
        if isinstance(combined, (tuple, list)):
            combined = combined[0]
        torch.cuda.synchronize()
        combined = combined[:T].float()

        c_q = ((combined[:, 0] / vals.to(device).float()) * 4).round() / 4
        bad = c_q != m
        # Same buckets as the reporter's probe, plus the residual it leaves out.
        is_drop = bad & (c_q == local_only)
        is_poll = bad & (c_q != c_q.round())
        is_other = bad & ~is_drop & ~is_poll

        for ci in range(NCLS):
            sel = cls == ci
            tally[ci, 0] += int(sel.sum())
            tally[ci, 1] += int((bad & sel).sum())
            shape[ci, 0] += int((is_drop & sel).sum())
            shape[ci, 1] += int((is_poll & sel).sum())
            shape[ci, 2] += int((is_other & sel).sum())

        dist.barrier()

    dist.all_reduce(tally)
    dist.all_reduce(shape)
    if rank == 0:
        print(
            "[path] --- corruption by the transport a token's partials used ---",
            flush=True,
        )
        print(
            f"[path] {'class':8s} {'tokens':>9s} {'bad':>7s} {'rate':>9s}   "
            f"{'dropped':>8s} {'polluted':>9s} {'other':>7s}",
            flush=True,
        )
        for ci, name in enumerate(CLASSES):
            n, b = int(tally[ci, 0]), int(tally[ci, 1])
            rate = (100.0 * b / n) if n else 0.0
            print(
                f"[path] {name:8s} {n:9d} {b:7d} {rate:8.3f}%   "
                f"{int(shape[ci,0]):8d} {int(shape[ci,1]):9d} {int(shape[ci,2]):7d}",
                flush=True,
            )
        loc_rate = tally[0, 1] / tally[0, 0] if tally[0, 0] else 0
        rem_rate = tally[2, 1] / tally[2, 0] if tally[2, 0] else 0
        print(
            f"[path] VERDICT: local={'CLEAN' if loc_rate == 0 else 'HIT'} "
            f"remote={'CLEAN' if rem_rate == 0 else 'HIT'}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

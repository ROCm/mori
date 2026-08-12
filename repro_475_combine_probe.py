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
"""Standalone repro for ROCm/mori#475 - InterNodeV1 cross-node combine corruption.

Self-contained: needs only torch + mori. No benchmark framework.

    # 2 PHYSICAL nodes, 8 GPUs each (world=16). Node 0:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
             --master_addr=<node0> --master_port=29500 mori_ep16_combine_repro.py
    # Node 1: same with --node_rank=1

    # Single-node control (expected CLEAN):
    torchrun --standalone --nproc_per_node=8 mori_ep16_combine_repro.py

How it works
------------
Every token is a *constant* row whose value is a small integer, so the combine
multiplier actually realized by the kernel is recoverable exactly:

    c_t = combined[t, 0] / x_t

For mori's unweighted rank-sum combine the expected multiplier is the number of
DISTINCT destination ranks of token t (same expectation as upstream's own
tests/python/ops/dispatch_combine_test_utils.py::check_combine_result, which
uses `input * unique_pes`). Dispatch is checked independently: the number of
tokens each rank receives is compared against an all-gathered routing oracle.

Failure signature we observe (2 nodes, world=16, gfx950 + Pensando/ionic RoCE):
  * dispatch is always exact (recv counts match, payload rows never torn)
  * a small fraction of tokens come back with the WRONG combine multiplier:
      - "dropped":  c_t == (number of distinct destinations on the LOCAL node)
                    i.e. the remote node's partial is missing entirely
      - "polluted": c_t > m_t and often non-integer, i.e. a foreign token's
                    payload was summed in
  * rate grows with tokens/rank: ~0 at T<=16, sparse at T=32..128,
    ~1-3 tokens/rank at T=256, ~4-8 tokens/rank at T=512
  * single-node (world=8) is CLEAN at every T we tried

Prints a PASS/FAIL summary. Pass --strict to also exit non-zero on corruption
(off by default so torchrun does not print a ChildFailedError traceback over
the result).
"""
import argparse
import dataclasses
import os
import sys

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import mori  # noqa: E402


def build_config(
    kernel, rank, world_size, hidden, topk, experts_per_rank, cap, gpu_per_node, blocks
):
    """Build EpDispatchCombineConfig, tolerating field drift across releases."""
    block_num, rdma_block_num, dwarps, _cwarps = blocks
    fields = {f.name for f in dataclasses.fields(mori.ops.EpDispatchCombineConfig)}
    kwargs = dict(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=2,  # bf16
        max_num_inp_token_per_rank=cap,
        num_experts_per_rank=experts_per_rank,
        num_experts_per_token=topk,
        use_external_inp_buf=True,  # InterNodeV1 consumes dispatch output directly
        kernel_type=getattr(mori.ops.EpDispatchCombineKernelType, kernel),
        block_num=block_num,
        warp_num_per_block=dwarps,
        gpu_per_node=gpu_per_node,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=1,
    )
    if "quant_type" in fields:
        kwargs["quant_type"] = "none"
    return mori.ops.EpDispatchCombineConfig(
        **{k: v for k, v in kwargs.items() if k in fields}
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokens", default="16 32 128 256 512", help="tokens per rank to sweep"
    )
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--gpu-per-node", type=int, default=8)
    ap.add_argument("--kernel", default="InterNodeV1")
    ap.add_argument(
        "--blocks",
        default="96,64,8,8",
        help="block_num,rdma_block_num,dispatch_warps,combine_warps",
    )
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when corruption is detected",
    )
    args = ap.parse_args()

    ladder = [int(t) for t in args.tokens.split()]
    block_num, rdma_block_num, dwarps, cwarps = (int(v) for v in args.blocks.split(","))
    cap = max(512, max(ladder))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # gloo alongside nccl: some mori builds call getBackend(cpu) during shmem init.
    dist.init_process_group("cpu:gloo,cuda:nccl")
    experts_per_rank = args.experts // world_size
    my_node = rank // args.gpu_per_node

    if rank == 0:
        props = torch.cuda.get_device_properties(device)
        print(
            f"[repro] world={world_size} gpu_per_node={args.gpu_per_node} "
            f"arch={props.gcnArchName} torch={torch.__version__}",
            flush=True,
        )

    torch._C._distributed_c10d._register_process_group(
        "default", torch.distributed.group.WORLD
    )
    mori.shmem.shmem_torch_process_group_init("default")

    config = build_config(
        args.kernel,
        rank,
        world_size,
        args.hidden,
        args.topk,
        experts_per_rank,
        cap,
        args.gpu_per_node,
        (block_num, rdma_block_num, dwarps, cwarps),
    )
    os.environ["MORI_EP_LAUNCH_CONFIG_MODE"] = "MANUAL"
    op = mori.ops.EpDispatchCombineOp(config)
    if rank == 0:
        print(
            f"[repro] kernel={args.kernel} blocks={args.blocks} cap={cap} "
            f"qp_per_pe={int(mori.shmem.shmem_num_qp_per_pe())}",
            flush=True,
        )

    any_bad = 0
    for T in ladder:
        g = torch.Generator().manual_seed(args.seed * 100003 + T * 101 + rank)
        idx = (
            torch.rand((T, args.experts), generator=g)
            .topk(args.topk, dim=1)
            .indices.to(torch.int32)
            .to(device)
        )
        w = torch.rand((T, args.topk), generator=g).to(torch.float32)
        w = (w / w.sum(dim=1, keepdim=True)).to(device)
        # Constant, bf16-exact small-integer payload per token.
        vals = ((torch.arange(T) + rank * T) % 15 + 1).to(torch.bfloat16)
        x = vals.unsqueeze(1).expand(T, args.hidden).contiguous().to(device)
        scales = torch.empty((T, 0), dtype=torch.uint8, device=device)

        # Expected multipliers: distinct destination ranks (total / local-node).
        dest = idx.long() // experts_per_rank
        one_hot = torch.zeros((T, world_size), device=device)
        one_hot.scatter_(1, dest, 1.0)
        m = one_hot.sum(dim=1)
        node_of = torch.arange(world_size, device=device) // args.gpu_per_node
        local_only = (one_hot * (node_of == my_node).float().unsqueeze(0)).sum(dim=1)

        # Expected receive count from the global routing trace.
        gathered = [torch.empty_like(idx) for _ in range(world_size)]
        dist.all_gather(gathered, idx)
        gdest = torch.cat(gathered).long() // experts_per_rank
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
        rows = int(recv_num[0].item())

        combined = op.combine(
            out,
            None,
            idx if os.environ.get("COMBINE_IDX", "dispatch") == "orig" else out_idx,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=cwarps,
        )
        if isinstance(combined, (tuple, list)):
            combined = combined[0]
        torch.cuda.synchronize()
        combined = combined[:T].float()

        c = combined[:, 0] / vals.to(device).float()
        c_q = (c * 4).round() / 4
        bad = c_q != m
        n_bad = int(bad.sum().item())
        dropped = int((bad & (c_q == local_only)).sum().item())
        polluted = int((bad & (c_q != c_q.round())).sum().item())
        any_bad |= n_bad

        print(
            f"[repro][rank {rank}] T={T}: recv={rows} expected_recv={expected_recv} "
            f"dispatch_exact={rows == expected_recv} "
            f"wrong_combine={n_bad}/{T} (dropped={dropped} polluted={polluted})",
            flush=True,
        )

        tot = torch.tensor([float(n_bad)], device=device)
        dist.all_reduce(tot)
        if rank == 0:
            v = int(tot.item())
            print(
                f"[repro] T={T} GLOBAL: {'CORRUPTION' if v else 'clean'} "
                f"({v} wrong tokens across {world_size} ranks)",
                flush=True,
            )
        dist.barrier()

    flag = torch.tensor([float(any_bad > 0)], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(
            f"[repro] RESULT: {'FAIL (combine corruption)' if flag.item() else 'PASS'}",
            flush=True,
        )
    sys.exit(1 if (flag.item() and args.strict) else 0)


if __name__ == "__main__":
    main()

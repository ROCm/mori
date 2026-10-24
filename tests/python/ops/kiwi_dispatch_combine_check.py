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
"""Correctness check for EpDispatchCombineKiwiOp. Requires a BUILD_OPS_KIWI build.

    torchrun --nproc_per_node=8 tests/python/ops/kiwi_dispatch_combine_check.py \
        --rcomp-backend rdma-only

For multi-node runs launch one torchrun per node with the usual --nnodes /
--node_rank / --master_addr arguments.
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch.distributed as dist


def load_kiwi_module():
    try:
        from mori.ops import dispatch_combine_kiwi

        return dispatch_combine_kiwi
    except ImportError:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../python/mori/ops/dispatch_combine_kiwi.py",
        )
        spec = importlib.util.spec_from_file_location("dispatch_combine_kiwi", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def gather_cpu(tensor, world_size):
    out = [None] * world_size
    dist.all_gather_object(out, tensor.cpu())
    return out


def make_inputs(rank, round_idx, args, num_experts, device):
    gen = torch.Generator(device="cpu").manual_seed(1234 + 1000 * round_idx + rank)
    n = args.num_tokens
    tokens = torch.randn((n, args.hidden_dim), generator=gen).to(torch.bfloat16)
    indices = torch.stack(
        [torch.randperm(num_experts, generator=gen)[: args.topk] for _ in range(n)]
    ).to(torch.int32)
    weights = torch.rand((n, args.topk), generator=gen, dtype=torch.float32)
    scales = torch.rand((n, args.scale_dim), generator=gen, dtype=torch.float32)
    return tokens.to(device), indices.to(device), weights.to(device), scales.to(device)


def check_round(op, args, round_idx, rank, world_size, device):
    num_experts = args.num_experts_per_rank * world_size
    tokens, indices, weights, scales = make_inputs(rank, round_idx, args, num_experts, device)
    all_tokens = gather_cpu(tokens, world_size)
    all_indices = gather_cpu(indices, world_size)
    all_weights = gather_cpu(weights, world_size)
    all_scales = gather_cpu(scales, world_size)

    out, out_w, out_s, out_idx, total_recv = op.dispatch(tokens, weights, scales, indices)
    torch.cuda.synchronize()
    count = int(total_recv.item())
    src_pos = op.get_dispatch_src_token_pos().cpu()
    assert src_pos.numel() == count

    stride = op.max_num_tokens_to_send()
    expected = 0
    for src in range(world_size):
        dest_ranks = all_indices[src] // args.num_experts_per_rank
        expected += int((dest_ranks == rank).any(dim=1).sum())
    assert count == expected, f"rank {rank}: received {count} rows, expected {expected}"

    out_cpu, out_w_cpu = out[:count].cpu(), out_w[:count].cpu()
    out_idx_cpu, out_s_cpu = out_idx[:count].cpu(), out_s[:count].cpu()
    seen = set()
    for row in range(count):
        src_rank, src_tok = divmod(int(src_pos[row]), stride)
        assert (src_rank, src_tok) not in seen, f"duplicate row for {(src_rank, src_tok)}"
        seen.add((src_rank, src_tok))
        assert torch.equal(out_cpu[row], all_tokens[src_rank][src_tok]), (
            f"rank {rank}: hidden mismatch for row {row} from {(src_rank, src_tok)}"
        )
        assert torch.equal(out_idx_cpu[row], all_indices[src_rank][src_tok])
        assert torch.equal(out_w_cpu[row], all_weights[src_rank][src_tok])
        assert torch.equal(out_s_cpu[row], all_scales[src_rank][src_tok])

    expert_out = op.get_registered_combine_input_buffer(torch.bfloat16)
    expert_out[:count].copy_(out[:count])
    expert_w = out_w.clone()
    combined, combined_w = op.combine(expert_out, expert_w, indices)
    torch.cuda.synchronize()

    dest_ranks = (indices // args.num_experts_per_rank).cpu()
    num_unique = torch.tensor(
        [len(set(r.tolist())) for r in dest_ranks], dtype=torch.float32
    ).unsqueeze(1)
    ref = tokens.float().cpu() * num_unique
    got = combined[: args.num_tokens].float().cpu()
    torch.testing.assert_close(got, ref, atol=1e-1, rtol=2e-2)
    ref_w = weights.cpu() * num_unique
    torch.testing.assert_close(combined_w[: args.num_tokens].cpu(), ref_w, atol=1e-5, rtol=1e-5)
    op.reset()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--max-num-inp-token-per-rank", type=int, default=128)
    parser.add_argument("--num-experts-per-rank", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale-dim", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rcomp-backend", default="auto")
    parser.add_argument("--num-blocks", type=int, default=0)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)

    kiwi = load_kiwi_module()
    config = kiwi.EpDispatchCombineKiwiConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=args.hidden_dim,
        scale_dim=args.scale_dim,
        scale_type_size=4,
        max_token_type_size=2,
        max_num_inp_token_per_rank=args.max_num_inp_token_per_rank,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=args.topk,
    )
    op = kiwi.EpDispatchCombineKiwiOp(
        config, rcomp_backend=args.rcomp_backend, num_blocks=args.num_blocks
    )
    for i in range(args.rounds):
        count = check_round(op, args, i, rank, world_size, device)
        if rank == 0:
            print(f"round {i}: ok (rank 0 received {count} rows)", flush=True)
    del op
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Two-rank eager/CUDA-graph smoke test for adaptive MORI/Kiwi EP."""

import os

import kiwi_ep_ext
import mori
import torch
import torch.distributed as dist
from mori.ops import (
    AdaptiveEpDispatchCombineOp,
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
)


def env_int(*names, default=0):
    for name in names:
        if name in os.environ:
            return int(os.environ[name])
    return default


def make_inputs(rank, world_size, num_tokens, hidden_dim, device):
    tokens = (
        torch.arange(num_tokens * hidden_dim, device=device, dtype=torch.float32)
        .reshape(num_tokens, hidden_dim)
        .add_(rank * 1000)
        .to(torch.bfloat16)
    )
    weights = torch.ones((num_tokens, 1), device=device, dtype=torch.float32)
    destinations = (torch.arange(num_tokens, device=device) + rank) % world_size
    indices = destinations.to(torch.int32).reshape(num_tokens, 1)
    return tokens, weights, indices


def run_pair(op, rank, world_size, num_tokens, hidden_dim, device):
    tokens, weights, indices = make_inputs(
        rank, world_size, num_tokens, hidden_dim, device
    )
    dispatched, _, _, dispatched_indices, _ = op.dispatch(
        tokens, weights, None, indices
    )
    combined, _ = op.combine(dispatched, None, dispatched_indices)
    torch.cuda.synchronize()
    torch.testing.assert_close(combined[:num_tokens], tokens, rtol=0, atol=0)
    return dispatched_indices, combined


def capture_pair(op, rank, world_size, num_tokens, hidden_dim, device):
    tokens, weights, indices = make_inputs(
        rank, world_size, num_tokens, hidden_dim, device
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dispatched, _, _, dispatched_indices, _ = op.dispatch(
            tokens, weights, None, indices
        )
        combined, _ = op.combine(dispatched, None, dispatched_indices)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(combined[:num_tokens], tokens, rtol=0, atol=0)
    return graph, tokens, combined


def main():
    rank = env_int("OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK")
    world_size = env_int("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE", default=1)
    local_rank = env_int("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK", default=rank)
    if world_size != 2:
        raise RuntimeError(
            f"this smoke test requires exactly 2 ranks, got {world_size}"
        )

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch._C._distributed_c10d._register_process_group(
        "adaptive_ep_test", dist.group.WORLD
    )
    mori.shmem.shmem_torch_process_group_init("adaptive_ep_test")

    hidden_dim = 64
    config = EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=64,
        num_experts_per_rank=1,
        num_experts_per_token=1,
        warp_num_per_block=16,
        block_num=32,
        use_external_inp_buf=True,
        kernel_type=EpDispatchCombineKernelType.InterNodeV1,
        gpu_per_node=world_size,
        rdma_block_num=16,
    )
    device_name = kiwi_ep_ext.get_hca_name_for_current_gpu()
    op = AdaptiveEpDispatchCombineOp(
        config,
        kiwi_max_num_tokens=16,
        dispatch_dtype=torch.bfloat16,
        combine_dtype=torch.bfloat16,
        group_name="adaptive_ep_test",
        num_blocks=4,
        num_proxy_threads=2,
        device_name=device_name,
        lci_master_addr=os.environ.get("LCI_TEST_MASTER_ADDR"),
        lci_master_port=env_int("LCI_TEST_MASTER_PORT", default=0) or None,
    )

    run_pair(op, rank, world_size, 8, hidden_dim, torch.device("cuda", local_rank))
    if op.last_backend != "kiwi":
        raise AssertionError(f"8-token pair selected {op.last_backend}, expected kiwi")
    run_pair(op, rank, world_size, 32, hidden_dim, torch.device("cuda", local_rank))
    if op.last_backend != "mori":
        raise AssertionError(f"32-token pair selected {op.last_backend}, expected mori")

    kiwi_graph, _, _ = capture_pair(
        op, rank, world_size, 8, hidden_dim, torch.device("cuda", local_rank)
    )
    if op.last_backend != "kiwi":
        raise AssertionError("8-token graph did not capture Kiwi")
    mori_graph, _, _ = capture_pair(
        op, rank, world_size, 32, hidden_dim, torch.device("cuda", local_rank)
    )
    if op.last_backend != "mori":
        raise AssertionError("32-token graph did not capture MORI")

    kiwi_graph.replay()
    mori_graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("PASS adaptive MORI/Kiwi eager and CUDA-graph bucket switching")


if __name__ == "__main__":
    main()

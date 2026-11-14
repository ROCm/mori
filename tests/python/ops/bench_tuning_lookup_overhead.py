"""Reproducer: tuning config lookup overhead (Python + e2e).

Usage:
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nproc_per_node=8 tests/python/ops/bench_tuning_lookup_overhead.py

Measures:
  1. Pure Python lookup latency (no GPU, no distributed)
  2. E2E dispatch latency with tuning lookup vs hardcoded bn/wpb
"""

import os
import time

import torch
import torch.distributed as dist

import mori.ops
import mori.shmem
from mori.ops.dispatch_combine import EpDispatchCombineKernelType
from mori.ops.tuning_config import TuningConfigManager

rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "8"))

torch.cuda.set_device(rank)
device = torch.device("cuda", rank)
dist.init_process_group(
    backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
)
torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

# ── Part 1: Pure Python lookup latency ──────────────────────────────
if rank == 0:
    print("=" * 65)
    print("Part 1: Pure Python lookup latency (100k calls)")
    print("=" * 65)

    mgr = TuningConfigManager.get_instance("gfx950", "IntraNodeLL", 8, "mi350x")
    N = 100000

    for label, rules, nt, extra in [
        ("dispatch 1t", mgr.dispatch_rules, 1, {}),
        ("dispatch 1024t", mgr.dispatch_rules, 1024, {}),
        ("combine 1t", mgr.combine_rules, 1, {"zero_copy": True, "quant_type": "none"}),
        ("combine 1024t", mgr.combine_rules, 1024, {"zero_copy": True, "quant_type": "none"}),
    ]:
        for _ in range(1000):
            TuningConfigManager.lookup(rules, torch.bfloat16, nt, 7168, **extra)
        start = time.perf_counter()
        for _ in range(N):
            TuningConfigManager.lookup(rules, torch.bfloat16, nt, 7168, **extra)
        elapsed = time.perf_counter() - start
        print(f"  {label:>20}: {elapsed / N * 1e6:.2f} us/call")

# ── Part 2: E2E dispatch with lookup vs hardcoded ───────────────────
if rank == 0:
    print()
    print("=" * 65)
    print("Part 2: E2E dispatch — tuning lookup vs hardcoded params")
    print("=" * 65)
    print(f"  {'tokens':>6} {'with_lookup':>12} {'hardcoded':>10} {'diff':>8} {'diff%':>6}")
    print("  " + "-" * 45)

TOKEN_COUNTS = [1, 32, 128, 1024, 4096]
WARMUP = 10
ITERS = 200
MAX_TOKENS = max(TOKEN_COUNTS)

config = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16,
    rank=rank,
    world_size=world_size,
    hidden_dim=7168,
    scale_dim=0,
    scale_type_size=4,
    max_token_type_size=2,
    max_num_inp_token_per_rank=MAX_TOKENS,
    num_experts_per_rank=32,
    num_experts_per_token=8,
    warp_num_per_block=16,
    block_num=80,
    use_external_inp_buf=True,
    gpu_per_node=world_size,
    quant_type="none",
    kernel_type=EpDispatchCombineKernelType.IntraNodeLL,
)

mori.shmem.shmem_torch_process_group_init("default")
op = mori.ops.EpDispatchCombineOp(config)

mgr = TuningConfigManager.get_instance("gfx950", "IntraNodeLL", 8, "mi350x")

for max_tokens in TOKEN_COUNTS:
    # Get the correct per-token config (same as what the lookup returns at runtime)
    params = TuningConfigManager.lookup(
        mgr.dispatch_rules, torch.bfloat16, max_tokens, 7168
    )
    tuned_bn, tuned_wpb = params.block_num, params.warp_per_block

    inp = torch.randn(max_tokens, 7168, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(max_tokens, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(max_tokens, 8, dtype=torch.int64, device="cuda")
    for i in range(max_tokens):
        indices[i] = torch.randperm(32)[:8]

    # With lookup (normal path — _resolve_launch_params queries tuning rules)
    for _ in range(WARMUP):
        op.dispatch(inp, weights, None, indices)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        op.dispatch(inp, weights, None, indices)
    e.record()
    torch.cuda.synchronize()
    with_us = s.elapsed_time(e) * 1000 / ITERS

    # Without lookup (clear rules so _resolve_launch_params skips the scan)
    saved_rules = op._dispatch_rules
    saved_bn = op.auto_block_num
    saved_rbn = op.auto_rdma_block_num
    saved_wpb = op.auto_warp_per_block
    op._dispatch_rules = []
    op.auto_block_num = tuned_bn
    op.auto_rdma_block_num = 0
    op.auto_warp_per_block = tuned_wpb
    for _ in range(WARMUP):
        op.dispatch(inp, weights, None, indices)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        op.dispatch(inp, weights, None, indices)
    e.record()
    torch.cuda.synchronize()
    without_us = s.elapsed_time(e) * 1000 / ITERS
    op._dispatch_rules = saved_rules
    op.auto_block_num = saved_bn
    op.auto_rdma_block_num = saved_rbn
    op.auto_warp_per_block = saved_wpb

    if rank == 0:
        diff = with_us - without_us
        pct = diff / with_us * 100 if with_us > 0 else 0
        print(
            f"  {max_tokens:>6} (bn={tuned_bn},wpb={tuned_wpb})"
            f" {with_us:>9.1f} us {without_us:>8.1f} us {diff:>+7.1f} {pct:>+5.1f}%"
        )

dist.barrier()
dist.destroy_process_group()

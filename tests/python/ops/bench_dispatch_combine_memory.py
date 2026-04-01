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
"""Benchmark HBM memory consumption of dispatch/combine op with different configurations.

Runs on a single GPU by initializing shmem with nranks=1 while creating the op
with the actual target world_size.  This measures the real buffer allocation
footprint without needing multiple GPUs.

Usage:
    python -m tests.python.ops.bench_dispatch_combine_memory
    python -m tests.python.ops.bench_dispatch_combine_memory --world-size 8 16 --max-tokens 128 4096
    python -m tests.python.ops.bench_dispatch_combine_memory --csv results.csv
"""

import argparse
import gc
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "64G")
os.environ["MORI_SHMEM_MODE"] = "isolation"

import torch
import mori
from mori.ops.dispatch_combine import EpDispatchCombineKernelType
from mori.shmem.api import _ensure_shmem_module

_KERNEL_TYPE_MAP = {
    "intranode": EpDispatchCombineKernelType.IntraNode,
    "internode": EpDispatchCombineKernelType.InterNode,
    "internode_v1": EpDispatchCombineKernelType.InterNodeV1,
    "internode_v1ll": EpDispatchCombineKernelType.InterNodeV1LL,
    "async_ll": EpDispatchCombineKernelType.AsyncLL,
}

_DATA_TYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
    "fp8_e4m3": torch.float8_e4m3fn,
}
if hasattr(torch, "float4_e2m1fn_x2"):
    _DATA_TYPE_MAP["fp4"] = torch.float4_e2m1fn_x2


def _measure_hbm():
    """Return (free, total) HBM in bytes via torch.cuda.mem_get_info."""
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure_op_memory(config):
    """Create an EpDispatchCombineOp with *config* and return HBM consumed in bytes.

    Returns (hbm_bytes, error_string).  On success error_string is None.
    """
    _cleanup()
    free_before, _ = _measure_hbm()

    try:
        op = mori.ops.EpDispatchCombineOp(config)
    except Exception as e:
        return 0, str(e)

    torch.cuda.synchronize()
    free_after, _ = _measure_hbm()
    hbm_used = free_before - free_after

    del op
    _cleanup()

    return hbm_used, None


def run_sweep(args):
    # --- shmem init with 1 rank (single GPU) ---
    torch.cuda.set_device(0)
    _ensure_shmem_module()
    uid = mori.shmem.shmem_get_unique_id()

    mori.shmem.shmem_init_attr(mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, 0, 1, uid)

    results = []
    header = [
        "kernel_type",
        "world_size",
        "max_tokens",
        "max_total_recv_tokens",
        "hidden_dim",
        "topk",
        "dtype",
        "gpu_per_node",
        "hbm_GB",
        "error",
    ]

    total_configs = (
        len(args.kernel_type)
        * len(args.world_size)
        * len(args.max_tokens)
        * len(args.max_total_recv_tokens)
        * len(args.hidden_dim)
        * len(args.num_experts_per_token)
        * len(args.dtype)
    )
    print(f"Sweeping {total_configs} configurations...\n")
    print(
        f"{'kernel_type':<16} {'world_size':>10} {'max_tokens':>10} {'max_recv':>10} "
        f"{'hidden_dim':>10} {'topk':>4} {'dtype':>16} {'gpu_per_node':>12} {'HBM(GB)':>10} {'error'}"
    )
    print("-" * 104)

    idx = 0
    for kernel_type_str in args.kernel_type:
        kernel_type = _KERNEL_TYPE_MAP[kernel_type_str]
        for world_size in args.world_size:
            if kernel_type == EpDispatchCombineKernelType.IntraNode and world_size > 8:
                print(
                    f"{'[SKIP]':<16} intranode kernel max world_size is 8, got {world_size}"
                )
                continue
            for max_tokens in args.max_tokens:
                for max_total_recv_tokens in args.max_total_recv_tokens:
                    for hidden_dim in args.hidden_dim:
                        for topk in args.num_experts_per_token:
                            for dtype_str in args.dtype:
                                idx += 1
                                data_type = _DATA_TYPE_MAP[dtype_str]

                                gpu_per_node = min(args.gpu_per_node, world_size)

                                num_experts_per_rank = max(
                                    1, args.total_experts // world_size
                                )

                                # max_token_type_size: largest dtype we might dispatch
                                max_token_type_size = 2  # bf16 = 2 bytes
                                if data_type in (
                                    torch.float8_e4m3fnuz,
                                    torch.float8_e4m3fn,
                                ):
                                    max_token_type_size = 2  # op still uses 2 for max
                                if (
                                    hasattr(torch, "float4_e2m1fn_x2")
                                    and data_type is torch.float4_e2m1fn_x2
                                ):
                                    max_token_type_size = 2

                                config = mori.ops.EpDispatchCombineConfig(
                                    data_type=data_type,
                                    rank=0,
                                    world_size=world_size,
                                    hidden_dim=hidden_dim,
                                    scale_dim=0,
                                    scale_type_size=0,
                                    max_token_type_size=max_token_type_size,
                                    max_num_inp_token_per_rank=max_tokens,
                                    num_experts_per_rank=num_experts_per_rank,
                                    num_experts_per_token=topk,
                                    max_total_recv_tokens=max_total_recv_tokens,
                                    warp_num_per_block=16,
                                    block_num=80,
                                    use_external_inp_buf=True,
                                    kernel_type=kernel_type,
                                    gpu_per_node=gpu_per_node,
                                )

                                hbm_bytes, err = measure_op_memory(config)
                                hbm_gb = hbm_bytes / (1024**3)

                                err_str = err if err else ""
                                print(
                                    f"{kernel_type_str:<16} {world_size:>10} {max_tokens:>10} "
                                    f"{max_total_recv_tokens:>10} {hidden_dim:>10} {topk:>4} "
                                    f"{dtype_str:>16} {gpu_per_node:>12} {hbm_gb:>10.1f} {err_str}"
                                )

                                results.append(
                                    {
                                        "kernel_type": kernel_type_str,
                                        "world_size": world_size,
                                        "max_tokens": max_tokens,
                                        "max_total_recv_tokens": max_total_recv_tokens,
                                        "hidden_dim": hidden_dim,
                                        "topk": topk,
                                        "dtype": dtype_str,
                                        "gpu_per_node": gpu_per_node,
                                        "hbm_GB": round(hbm_gb, 2),
                                        "error": err_str,
                                    }
                                )

    mori.shmem.shmem_finalize()

    # --- CSV output ---
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to {args.csv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HBM memory consumption of dispatch/combine ops"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="World sizes (EP degrees) to sweep (default: 8 16 32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
        help="Max input tokens per rank to sweep (default: 4096 8192 16384)",
    )
    parser.add_argument(
        "--max-total-recv-tokens",
        type=int,
        nargs="+",
        default=[0],
        help="Max total received tokens across ranks to sweep (default: 0)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=[7168],
        help="Hidden dimensions to sweep (default: 7168)",
    )
    parser.add_argument(
        "--num-experts-per-token",
        type=int,
        nargs="+",
        default=[8],
        help="Top-k experts per token to sweep (default: 8)",
    )
    parser.add_argument(
        "--total-experts",
        type=int,
        default=256,
        help="Total number of experts (divided by world_size to get per-rank, default: 256)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="+",
        default=["bf16"],
        choices=list(_DATA_TYPE_MAP.keys()),
        help="Data types to sweep (default: bf16)",
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        nargs="+",
        default=[
            "intranode",
            "internode",
            "internode_v1",
            "internode_v1ll",
            "async_ll",
        ],
        choices=list(_KERNEL_TYPE_MAP.keys()),
        help="Kernel types to sweep (default: all)",
    )
    parser.add_argument(
        "--gpu-per-node",
        type=int,
        default=8,
        help="GPUs per node for inter-node kernel types (default: 8)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to write CSV results",
    )
    parser.add_argument(
        "--heap-size",
        type=str,
        default=None,
        help="Override MORI_SHMEM_HEAP_SIZE (e.g. '64G'). Default: 64G",
    )
    args = parser.parse_args()

    if args.heap_size:
        os.environ["MORI_SHMEM_HEAP_SIZE"] = args.heap_size

    print("=" * 100)
    print("Dispatch/Combine HBM Memory Benchmark (Single GPU)")
    print("=" * 100)
    print("  shmem_mode:    isolation")
    print(f"  MORI_SHMEM_HEAP_SIZE = {os.environ.get('MORI_SHMEM_HEAP_SIZE', 'unset')}")
    print(f"  world_size:    {args.world_size}")
    print(f"  max_tokens:    {args.max_tokens}")
    print(f"  max_recv:      {args.max_total_recv_tokens}")
    print(f"  hidden_dim:    {args.hidden_dim}")
    print(f"  topk:          {args.num_experts_per_token}")
    print(f"  total_experts: {args.total_experts}")
    print(f"  dtype:         {args.dtype}")
    print(f"  kernel_type:   {args.kernel_type}")
    print(f"  gpu_per_node:  {args.gpu_per_node}")
    free, total = torch.cuda.mem_get_info()
    print(
        f"  GPU HBM:       {total / (1024**3):.1f} GB total, {free / (1024**3):.1f} GB free"
    )
    print("=" * 100)
    print()

    run_sweep(args)


if __name__ == "__main__":
    main()

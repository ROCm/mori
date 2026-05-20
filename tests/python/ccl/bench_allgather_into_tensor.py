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
"""DeepSpeed-style benchmark for mori SDMA all_gather_into_tensor.

Measures end-to-end per-call latency and throughput for three variants
of the same allgather, using the contiguous-output pattern that ZeRO-3
prefetch buckets actually use (i.e. a single ``torch.empty`` of size
``world_size * per_rank_numel``):

  1. ``mori_cpp.AllGatherIntoTensor``  - 1 SDMA kernel + 1 D2D memcpy
  2. ``dist.all_gather_into_tensor``   - RCCL into a single contig tensor
  3. ``dist.all_gather``               - RCCL into a list of N separate tensors
                                        (forces N D2D copies internally)

Run with the same shapes DeepSpeed uses for ZeRO-3 fetch buckets
(``stage3_prefetch_bucket_size`` ranges 1MB..100MB in practice).

Example:
    python3 -m tests.python.ccl.bench_allgather_into_tensor \\
        --world-size 8 --dtype bfloat16 --iters 200 \\
        --sizes-mb 0.5,1,4,10,40,100
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import AllGatherIntoTensor, DataType

from tests.python.utils import TorchDistContext, get_free_port


_TORCH_TO_MORI = {
    torch.bfloat16: DataType.BFloat16,
    torch.float16: DataType.Float16,
    torch.float32: DataType.Float32,
}


def _bench_loop(launch_fn, *, warmup: int, iters: int):
    """Time ``launch_fn`` ``iters`` times on the current CUDA stream.

    Uses CUDA events on each iteration so the timer captures actual
    GPU-side execution (kernel + memcpy + RCCL collective), not host
    queue latency.
    """
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        launch_fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _stats(times_ms, *, bytes_per_call):
    mean = statistics.mean(times_ms)
    p50 = statistics.median(times_ms)
    p99 = sorted(times_ms)[max(0, int(0.99 * len(times_ms)) - 1)]
    bw = bytes_per_call / (mean / 1000.0) / 1e9
    return mean, p50, p99, bw


def _worker(
    rank: int,
    world_size: int,
    port: int,
    sizes_bytes,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        max_bytes = max(sizes_bytes)
        max_per_rank_bytes = max_bytes // world_size  # per-rank shard
        # round up to 4-byte alignment (SDMA kernel walks uint32 lanes)
        max_per_rank_bytes = (max_per_rank_bytes + 3) & ~0x3
        ag_mori = AllGatherIntoTensor(
            my_pe=rank,
            npes=world_size,
            input_buffer_size=max_per_rank_bytes + 4096,
            output_buffer_size=(max_per_rank_bytes + 4096) * world_size,
            copy_output_to_user=True,
        )
        mori_dtype = _TORCH_TO_MORI[dtype]
        stream = torch.cuda.current_stream()

        if rank == 0:
            print(
                f"=== bench_allgather_into_tensor "
                f"world_size={world_size}  dtype={dtype}  "
                f"iters={iters} (warmup={warmup}) ==="
            )
            print(
                f"{'output MB':>10}  {'variant':22}"
                f"{'mean ms':>10}{'p50 ms':>10}{'p99 ms':>10}{'GB/s':>10}"
            )
            print("-" * 72)

        for total_bytes in sizes_bytes:
            # Free any cached PyTorch tensors from the previous iteration
            # so big sizes don't compound the caching-allocator footprint.
            torch.cuda.empty_cache()
            per_rank_bytes = (
                (total_bytes // world_size) // dtype.itemsize * dtype.itemsize
            )
            if per_rank_bytes < 4 or (per_rank_bytes % 4) != 0:
                continue
            per_rank_numel = per_rank_bytes // dtype.itemsize
            total_numel = per_rank_numel * world_size

            inp = torch.randn(per_rank_numel, device=device).to(dtype).contiguous()
            out_flat = torch.empty(total_numel, dtype=dtype, device=device)
            # list-form: N independent allocations (the case that costs N memcpys)
            out_list = [
                torch.empty(per_rank_numel, dtype=dtype, device=device)
                for _ in range(world_size)
            ]

            def call_mori_into():
                # AllGatherIntoTensor: 1 SDMA kernel + 1 D2D memcpy total.
                ag_mori(
                    inp.data_ptr(),
                    out_flat.data_ptr(),
                    per_rank_numel,
                    mori_dtype,
                    stream.cuda_stream,
                )

            def call_mori_list():
                # Same SDMA kernel, but emulate the all_gather (list) API:
                # after the kernel writes the gathered shards into the flat
                # transit/staging buffer, scatter them out into N separately
                # allocated user tensors with N hipMemcpyAsync copies — this
                # is the cost a hypothetical mori ``AllGather`` (list form)
                # wrapper would have to pay every call.
                ag_mori(
                    inp.data_ptr(),
                    out_flat.data_ptr(),
                    per_rank_numel,
                    mori_dtype,
                    stream.cuda_stream,
                )
                for i in range(world_size):
                    out_list[i].copy_(
                        out_flat[i * per_rank_numel : (i + 1) * per_rank_numel],
                        non_blocking=True,
                    )

            # End-to-end equivalent of "list-form allgather + the consumer
            # actually wanting a contiguous tensor" — i.e. what DS / GEMM
            # would need.  Pays N scatter copies + one final torch.cat
            # (another ~world_size_bytes worth of D2D copy traffic).
            cat_target = torch.empty(total_numel, dtype=dtype, device=device)

            def call_mori_list_then_cat():
                ag_mori(
                    inp.data_ptr(),
                    out_flat.data_ptr(),
                    per_rank_numel,
                    mori_dtype,
                    stream.cuda_stream,
                )
                for i in range(world_size):
                    out_list[i].copy_(
                        out_flat[i * per_rank_numel : (i + 1) * per_rank_numel],
                        non_blocking=True,
                    )
                torch.cat(out_list, dim=0, out=cat_target)

            def call_rccl_into():
                dist.all_gather_into_tensor(out_flat, inp)

            def call_rccl_list():
                dist.all_gather(out_list, inp)

            # Sanity check: every mori variant must agree byte-for-byte
            # with the RCCL reference on the very first call.
            torch.cuda.synchronize()
            dist.barrier()
            call_rccl_into()
            torch.cuda.synchronize()
            ref = out_flat.clone()

            # Correctness checks: every variant must reproduce the
            # full contiguous result byte-for-byte.
            out_flat.zero_()
            call_mori_into()
            torch.cuda.synchronize()
            if not torch.equal(out_flat, ref):
                raise AssertionError(f"rank {rank} mori AG_into_tensor mismatch")

            for t in out_list:
                t.zero_()
            call_mori_list()
            torch.cuda.synchronize()
            for i in range(world_size):
                slice_ref = ref[i * per_rank_numel : (i + 1) * per_rank_numel]
                if not torch.equal(out_list[i], slice_ref):
                    raise AssertionError(f"rank {rank} mori AG_list slot {i} mismatch")

            cat_target.zero_()
            call_mori_list_then_cat()
            torch.cuda.synchronize()
            if not torch.equal(cat_target, ref):
                raise AssertionError(f"rank {rank} mori AG_list+cat final mismatch")

            t_mori_into = _bench_loop(call_mori_into, warmup=warmup, iters=iters)
            t_mori_list = _bench_loop(call_mori_list, warmup=warmup, iters=iters)
            t_mori_list_cat = _bench_loop(
                call_mori_list_then_cat, warmup=warmup, iters=iters
            )

            if rank == 0:
                mb = total_numel * dtype.itemsize / 1e6
                bytes_per = total_numel * dtype.itemsize
                for label, ts in (
                    ("mori AG_into_tensor", t_mori_into),
                    ("mori AG_list (1+N)", t_mori_list),
                    ("mori AG_list+cat (1+N+1)", t_mori_list_cat),
                ):
                    m, p50, p99, bw = _stats(ts, bytes_per_call=bytes_per)
                    print(
                        f"{mb:>10.2f}  {label:26}"
                        f"{m:>10.3f}{p50:>10.3f}{p99:>10.3f}{bw:>10.2f}"
                    )
                print()

        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--sizes-mb",
        type=str,
        default="1,4,10,40,100",
        help="Comma-separated list of total output sizes in MB "
        "(matches DS prefetch_bucket_size range).",
    )
    args = p.parse_args()

    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    assert args.world_size >= 2

    sizes_bytes = [int(float(s) * 1024 * 1024) for s in args.sizes_mb.split(",")]
    dtype = getattr(torch, args.dtype)

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.world_size, port, sizes_bytes, dtype, args.warmup, args.iters),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

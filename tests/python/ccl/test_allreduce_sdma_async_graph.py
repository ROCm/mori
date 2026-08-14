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
"""Regression tests for graph-captured and cross-stream asynchronous SDMA AllReduce."""

import argparse
import os

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


def _parse_size(value: str) -> int:
    value = value.strip().upper()
    scale = 1
    if value.endswith("K"):
        scale, value = 1024, value[:-1]
    elif value.endswith("M"):
        scale, value = 1024**2, value[:-1]
    return int(value) * scale


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    sizes: list[int],
    replays: int,
    expect_baseline_failure: bool,
    cross_stream: bool,
) -> None:
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dtype = torch.bfloat16
        element_size = torch.empty((), dtype=dtype).element_size()
        allreduce = AllreduceSdma(
            rank,
            world_size,
            input_buffer_size=max(sizes),
            output_buffer_size=max(sizes) + world_size * 128,
            copy_output_to_user=True,
            dtype=dtype,
        )

        captures = []
        start_stream = torch.cuda.Stream(device=device) if cross_stream else None
        capture_error = None
        try:
            for size_bytes in sizes:
                elems = size_bytes // element_size
                inp = torch.empty(elems, dtype=dtype, device=device)
                out = torch.empty_like(inp)
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize(device)
                dist.barrier()
                with torch.cuda.graph(graph):
                    wait_stream = torch.cuda.current_stream(device)
                    if start_stream is not None:
                        start_stream.wait_stream(wait_stream)
                        allreduce.start_async(inp, out, elems, start_stream)
                    else:
                        allreduce.start_async(inp, out, elems, wait_stream)
                    allreduce.wait_async(wait_stream)
                captures.append((size_bytes, inp, out, graph))
        except RuntimeError as exc:
            capture_error = str(exc)

        failed = torch.tensor(
            [int(capture_error is not None)], dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if expect_baseline_failure:
            if not failed.item():
                raise AssertionError("baseline unexpectedly captured async start+wait")
            if rank == 0:
                print(f"EXPECTED_BASELINE_CAPTURE_FAILURE: {capture_error}", flush=True)
            os._exit(0)
        if failed.item():
            raise AssertionError(capture_error or "another rank failed graph capture")

        for replay in range(replays):
            capture_index = replay % len(captures)
            size_bytes, inp, out, graph = captures[capture_index]
            value = rank + replay + 2
            inp.fill_(value)
            expected_value = sum(pe + replay + 2 for pe in range(world_size))
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            mismatch_count = int(torch.count_nonzero(out != expected_value).item())
            local_failed = torch.tensor(
                [int(mismatch_count != 0)], dtype=torch.int32, device="cpu"
            )
            dist.all_reduce(local_failed, op=dist.ReduceOp.MAX)
            if local_failed.item():
                raise AssertionError(
                    f"rank={rank}, bytes={size_bytes}, replay={replay}, "
                    f"mismatches={mismatch_count}/{out.numel()}"
                )

        torch.cuda.synchronize(device)
        dist.barrier()
        del captures
        del allreduce
        shmem.shmem_finalize()
        if rank == 0:
            print(
                "PASSED: MORI async graph replay "
                f"TP={world_size}, sizes={sizes}, replays={replays}, "
                f"cross_stream={cross_stream}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--world-size", type=int, default=4)
    parser.add_argument("--sizes", default="128M,32M,128M,1M,16K,16M,64M")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--expect-baseline-failure", action="store_true")
    parser.add_argument(
        "--cross-stream",
        action="store_true",
        help="Launch start_async on a side stream and wait_async on the capture stream.",
    )
    args = parser.parse_args()
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    sizes = [_parse_size(value) for value in args.sizes.split(",")]
    torch.multiprocessing.spawn(
        _run_rank,
        args=(
            args.world_size,
            get_free_port(),
            sizes,
            args.replays,
            args.expect_baseline_failure,
            args.cross_stream,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

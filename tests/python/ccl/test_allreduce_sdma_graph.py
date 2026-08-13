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

"""Reproduce MORI SDMA AllReduce corruption under HIP graph replay."""

import argparse
import os

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


def _parse_size(value: str) -> int:
    suffixes = {"K": 1 << 10, "M": 1 << 20}
    suffix = value[-1].upper()
    if suffix in suffixes:
        return int(value[:-1]) * suffixes[suffix]
    return int(value)


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    sizes: list[int],
    replays: int,
    external_reuse_barrier: bool,
) -> None:
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
        max_bytes = max(sizes)
        max_elems = max_bytes // element_size
        output_buffer_size = world_size * (max_elems // world_size + 64) * element_size
        allreduce = AllreduceSdma(
            rank,
            world_size,
            input_buffer_size=max_bytes,
            output_buffer_size=output_buffer_size,
            copy_output_to_user=True,
            dtype=torch.bfloat16,
        )

        failure = None
        global_failure = False
        captures = []
        for size_bytes in sizes:
            elems = size_bytes // element_size
            inp = torch.empty(elems, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            graph = torch.cuda.CUDAGraph()

            torch.cuda.synchronize(device)
            dist.barrier()
            with torch.cuda.graph(graph):
                capture_stream = torch.cuda.current_stream(device)
                allreduce(inp, out, elems, capture_stream)
                if external_reuse_barrier:
                    shmem.shmem_barrier_on_stream(capture_stream.cuda_stream)
            torch.cuda.synchronize(device)
            dist.barrier()
            captures.append((size_bytes, elems, inp, out, graph))

        for replay in range(replays):
            for capture_index, (size_bytes, elems, inp, out, graph) in enumerate(
                captures
            ):
                operation = replay * len(captures) + capture_index
                value_offset = operation % 32
                inp.fill_(rank + 2 + value_offset)
                expected_value = sum(
                    pe + 2 + value_offset for pe in range(world_size)
                )
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)

                mismatch_count = int(
                    torch.count_nonzero(out != expected_value).item()
                )
                if mismatch_count:
                    shard_elems = elems // world_size
                    shard_mismatches = [
                        int(
                            torch.count_nonzero(
                                out[pe * shard_elems : (pe + 1) * shard_elems]
                                != expected_value
                            ).item()
                        )
                        for pe in range(world_size)
                    ]
                    shard_first = [
                        float(out[pe * shard_elems].float().item())
                        for pe in range(world_size)
                    ]
                    max_abs = float(
                        (out.float() - expected_value).abs().max().item()
                    )
                    failure = (
                        f"rank={rank}, bytes={size_bytes}, replay={replay}, "
                        f"capture_index={capture_index}, "
                        f"mismatches={mismatch_count}/{elems}, max_abs={max_abs}, "
                        f"shard_mismatches={shard_mismatches}, "
                        f"shard_first={shard_first}, expected={expected_value}"
                    )
                failed = torch.tensor(
                    [int(failure is not None)], dtype=torch.int32, device="cpu"
                )
                dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                if failed.item():
                    global_failure = True
                    if failure is not None:
                        print(f"GRAPH_REPLAY_CORRUPTION: {failure}", flush=True)
                    break
            if global_failure:
                break

        torch.cuda.synchronize(device)
        dist.barrier()
        del captures
        del allreduce
        shmem.shmem_finalize()
        if global_failure:
            raise AssertionError(failure or "another rank detected graph corruption")
        if rank == 0:
            print(
                "PASSED: MORI SDMA graph replay "
                f"TP={world_size}, sizes={sizes}, replays={replays}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--world-size", type=int, default=4)
    parser.add_argument("--sizes", default="128M,32M,128M,1M,16K,16M,64M")
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument(
        "--external-reuse-barrier",
        action="store_true",
        help="Append an additional cross-PE barrier after the MORI call.",
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
            args.external_reuse_barrier,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

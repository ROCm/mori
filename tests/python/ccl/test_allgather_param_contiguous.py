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
"""Correctness test for param-contiguous SDMA allgather output.

The param-contiguous layout stores each local input split contiguously across
all ranks:

  output[split_offset * world_size + rank * split_size : ...]

This is the layout FSDP zero-copy allgather consumes.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import AllgatherSdma

try:
    from tests.python.utils import TorchDistContext, get_free_port
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from tests.python.utils import TorchDistContext, get_free_port


pytestmark = pytest.mark.skipif(
    os.environ.get("MORI_ENABLE_SDMA", "").strip().lower()
    not in ("1", "true", "yes", "on"),
    reason="requires MORI_ENABLE_SDMA=1",
)


def _expected_param_contiguous(
    world_size: int,
    total_count: int,
    split_sizes: torch.Tensor,
    split_offsets: torch.Tensor,
    device: torch.device,
    base_offset: int,
) -> torch.Tensor:
    expected = torch.empty(total_count * world_size, dtype=torch.uint32, device=device)
    for src_rank in range(world_size):
        src = torch.arange(total_count, dtype=torch.uint32, device=device) + (
            base_offset + src_rank * 1000
        )
        for split_size, split_offset in zip(split_sizes.tolist(), split_offsets.tolist()):
            out_start = split_offset * world_size + src_rank * split_size
            out_end = out_start + split_size
            expected[out_start:out_end] = src[split_offset : split_offset + split_size]
    return expected


def _assert_matches_param_contiguous(
    got: torch.Tensor,
    world_size: int,
    total_count: int,
    split_sizes: torch.Tensor,
    split_offsets: torch.Tensor,
    device: torch.device,
    base_offset: int,
    label: str,
) -> None:
    expected = _expected_param_contiguous(
        world_size, total_count, split_sizes, split_offsets, device, base_offset
    )
    if not torch.equal(got, expected):
        diff = (got != expected).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"{label} param-contiguous allgather mismatch: "
            f"first mismatching positions={diff}, "
            f"got={got[diff].tolist()}, expected={expected[diff].tolist()}"
        )


def _worker(rank: int, world_size: int, port: int) -> None:
    total_count = 10
    split_sizes_host = torch.tensor([3, 5, 2], dtype=torch.uint64)
    split_offsets_host = torch.tensor([0, 3, 8], dtype=torch.uint64)
    total_bytes = total_count * torch.tensor([], dtype=torch.uint32).element_size()

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        handle = AllgatherSdma(
            my_pe=rank,
            npes=world_size,
            input_buffer_size=total_bytes,
            output_buffer_size=total_bytes * world_size,
            copy_output_to_user=False,
        )

        stream = torch.cuda.Stream(device=device)
        split_sizes = split_sizes_host.to(device=device)
        split_offsets = split_offsets_host.to(device=device)

        try:
            sync_base = 100
            sync_input = (
                torch.arange(total_count, dtype=torch.uint32, device=device)
                + sync_base
                + rank * 1000
            )
            sync_output = torch.empty(total_count * world_size, dtype=torch.uint32, device=device)

            dist.barrier()
            with torch.cuda.stream(stream):
                ok = handle.enqueue_param_contiguous(
                    sync_input,
                    sync_output,
                    total_count,
                    split_sizes,
                    split_offsets,
                    stream,
                )
            assert ok, "enqueue_param_contiguous returned false"
            stream.synchronize()

            sync_got = handle.get_output_transit_buffer(dtype=torch.uint32, device=device)[
                : total_count * world_size
            ].clone()
            _assert_matches_param_contiguous(
                sync_got,
                world_size,
                total_count,
                split_sizes_host,
                split_offsets_host,
                device,
                sync_base,
                "sync",
            )

            async_base = 10000
            async_input = (
                torch.arange(total_count, dtype=torch.uint32, device=device)
                + async_base
                + rank * 1000
            )
            async_output = torch.empty(total_count * world_size, dtype=torch.uint32, device=device)

            dist.barrier()
            with torch.cuda.stream(stream):
                ok = handle.start_async_param_contiguous(
                    async_input,
                    async_output,
                    total_count,
                    split_sizes,
                    split_offsets,
                    stream,
                )
            assert ok, "start_async_param_contiguous returned false"
            with torch.cuda.stream(stream):
                elapsed = handle.wait_async(stream)
            assert elapsed >= 0, "wait_async returned a negative duration"
            stream.synchronize()

            async_got = handle.get_output_transit_buffer(dtype=torch.uint32, device=device)[
                : total_count * world_size
            ].clone()
            _assert_matches_param_contiguous(
                async_got,
                world_size,
                total_count,
                split_sizes_host,
                split_offsets_host,
                device,
                async_base,
                "async",
            )

            torch.cuda.synchronize()
            dist.barrier()
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            del handle
            dist.barrier()
            shmem.shmem_finalize()


def test_allgather_param_contiguous_sdma() -> None:
    world_size = min(2, torch.cuda.device_count())
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    port = get_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )

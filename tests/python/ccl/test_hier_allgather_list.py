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
"""Bit-exact test for the traditional list-based ``HierAllGather.all_gather``
(matches ``torch.distributed.all_gather``)."""
import os
import traceback

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import HierAllGather

from tests.python.utils import TorchDistContext, get_free_port

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >=2 GPUs"
)

_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]


def _make_input(dtype, numel, rank, device):
    base = (rank + 1) * 17
    ramp = torch.arange(numel, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device)


def _worker(rank, world_size, ranks_per_node, port, numel):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        shmem.shmem_torch_process_group_init("default")
        per_rank_bytes = numel * 4 + 4096
        # No ranks_per_node -> auto-detected (drop-in with the flat AllgatherSdma).
        handle = HierAllGather(
            my_pe=rank,
            npes=world_size,
            input_buffer_size=per_rank_bytes,
            output_buffer_size=per_rank_bytes * world_size,
            copy_output_to_user=True,
        )
        try:
            for dtype in _DTYPES:
                inp = _make_input(dtype, numel, rank, device)
                out_list = [
                    torch.empty(numel, dtype=dtype, device=device)
                    for _ in range(world_size)
                ]
                ref_list = [
                    torch.empty(numel, dtype=dtype, device=device)
                    for _ in range(world_size)
                ]
                dist.all_gather(ref_list, inp)
                assert handle.all_gather(out_list, inp, torch.cuda.current_stream())
                torch.cuda.synchronize()
                for i in range(world_size):
                    if not torch.equal(out_list[i], ref_list[i]):
                        raise AssertionError(
                            f"list all_gather mismatch dtype={dtype} slot={i}"
                        )
                if rank == 0:
                    print(f"  ok dtype={dtype}")
            dist.barrier()
            if rank == 0:
                print("test_hier_allgather_list: PASSED")
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            shmem.shmem_finalize()


def test_hier_allgather_list(world_size=None, ranks_per_node=None, numel=4096):
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size >= 2
    if ranks_per_node is None:
        ranks_per_node = world_size
    port = get_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, ranks_per_node, port, numel),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=None)
    p.add_argument("--ranks-per-node", type=int, default=None)
    p.add_argument("--numel", type=int, default=4096)
    a = p.parse_args()
    try:
        test_hier_allgather_list(a.world_size, a.ranks_per_node, a.numel)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

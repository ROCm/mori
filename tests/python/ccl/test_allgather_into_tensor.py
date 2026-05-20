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
"""
Unit test for mori::collective::AllGatherIntoTensor (the NCCL/RCCL-style
C++ dispatcher exposed as ``mori.ccl.AllGatherIntoTensor``).

For each supported dtype we

  1. compute the reference output via ``torch.distributed.all_gather_into_tensor``
  2. run the mori SDMA path (synchronous and async two-phase)
  3. assert exact equality with the reference (no tolerance — allgather is
     a pure data move, no arithmetic).

A small smoke test of the alignment-error path is also exercised
(int8 with an odd count must raise from C++ because the SDMA kernel
walks the buffer in uint32 lanes).
"""

import os
import traceback

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import AllGatherIntoTensor, DataType

from tests.python.utils import TorchDistContext, get_free_port


# torch.dtype -> mori_cpp.DataType.
_TORCH_TO_MORI = {
    torch.uint8: DataType.Uint8,
    torch.int8: DataType.Int8,
    torch.int16: DataType.Int16,
    torch.int32: DataType.Int32,
    torch.int64: DataType.Int64,
    torch.float16: DataType.Float16,
    torch.bfloat16: DataType.BFloat16,
    torch.float32: DataType.Float32,
    torch.float64: DataType.Float64,
}


def _make_input(dtype: torch.dtype, numel: int, rank: int, device) -> torch.Tensor:
    """Rank-distinct, dtype-friendly input tensor.

    Values must be exactly representable in every dtype we test (the
    correctness check is bit-exact), so we stick to small-magnitude
    integers that round-trip through fp16/bf16 without loss.
    """
    base = (rank + 1) * 17  # arbitrary, small, dtype-safe
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        ramp = torch.arange(numel, dtype=torch.int32) % 64
        t = (ramp + base).to(dtype=dtype)
    else:
        ramp = torch.arange(numel, dtype=torch.int64) % 64
        t = (ramp + base) % (256 if dtype in (torch.uint8, torch.int8) else 1024)
        t = t.to(dtype=dtype)
        # int8 / uint8 may have wrapped; that's fine — the reference uses the
        # same input so equality still holds.
    return t.contiguous().to(device=device)


def _run_one(
    handle: AllGatherIntoTensor,
    dtype: torch.dtype,
    numel: int,
    rank: int,
    world_size: int,
    device,
    async_mode: bool,
):
    """Run a single (dtype, mode) round and assert against torch reference."""
    mori_dtype = _TORCH_TO_MORI[dtype]

    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * world_size, dtype=dtype, device=device)
    out_ref = torch.empty(numel * world_size, dtype=dtype, device=device)

    # Reference path — RCCL via torch.distributed.
    dist.all_gather_into_tensor(out_ref, inp)

    # Mori SDMA path on the current compute stream, then make the verifying
    # code path wait on the same stream (mirrors what DeepSpeed does).
    stream = torch.cuda.current_stream()
    if async_mode:
        ok = handle.start_async(
            inp.data_ptr(), out_mori.data_ptr(), numel, mori_dtype, stream.cuda_stream
        )
        assert ok, f"start_async failed for dtype={dtype}, async={async_mode}"
        elapsed = handle.wait_async(stream.cuda_stream)
        assert elapsed >= 0, f"wait_async failed for dtype={dtype}, async={async_mode}"
    else:
        ok = handle(
            inp.data_ptr(), out_mori.data_ptr(), numel, mori_dtype, stream.cuda_stream
        )
        assert ok, f"sync call failed for dtype={dtype}, async={async_mode}"
    stream.synchronize()

    if not torch.equal(out_mori, out_ref):
        # Surface a useful diff so the test message is debuggable.
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"mori AllGatherIntoTensor mismatch for dtype={dtype} "
            f"async={async_mode}: first mismatching positions={diff} "
            f"got={out_mori[diff].tolist()} ref={out_ref[diff].tolist()}"
        )


def _check_alignment_error(handle: AllGatherIntoTensor, device):
    """Per-rank byte length must be a multiple of 4; otherwise C++ raises."""
    inp = torch.zeros(3, dtype=torch.int8, device=device)  # 3 bytes
    out = torch.zeros(3 * 8, dtype=torch.int8, device=device)
    raised = False
    try:
        handle(inp.data_ptr(), out.data_ptr(), 3, DataType.Int8, 0)
    except RuntimeError:
        raised = True
    assert raised, "Expected RuntimeError for non-uint32-aligned input bytes"


def _worker(
    rank: int,
    world_size: int,
    port: int,
    numel: int,
    dtypes: list,
    async_modes: list,
    auto_register: bool,
):
    """Body executed in each spawned process."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        # Size the transit buffers to the largest dtype × numel we'll see.
        max_dtype_bytes = max(t.itemsize for t in dtypes)
        # Pad up so shape changes don't trigger reallocation inside mori.
        per_rank_bytes = numel * max_dtype_bytes + 4096
        handle = AllGatherIntoTensor(
            my_pe=rank,
            npes=world_size,
            input_buffer_size=per_rank_bytes,
            output_buffer_size=per_rank_bytes * world_size,
            copy_output_to_user=True,
            auto_register=auto_register,
        )
        if rank == 0:
            print(f"  auto_register={auto_register}")

        try:
            for dtype in dtypes:
                # Skip dtypes whose total per-rank byte count isn't 4-byte
                # aligned: the SDMA kernel walks uint32 lanes and the C++
                # entry rejects them on purpose.
                if (numel * dtype.itemsize) % 4 != 0:
                    continue
                for async_mode in async_modes:
                    _run_one(handle, dtype, numel, rank, world_size, device, async_mode)
                    if rank == 0:
                        print(
                            f"  ok dtype={dtype}, async={async_mode}, " f"numel={numel}"
                        )

            _check_alignment_error(handle, device)
            if rank == 0:
                print("  ok alignment-error path")

            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                print("test_allgather_into_tensor: PASSED")
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            del handle
            dist.barrier()
            shmem.shmem_finalize()


def test_allgather_into_tensor(
    numel: int = 1024,
    world_size: int = None,
    dtypes=None,
    async_modes=None,
    auto_register: bool = False,
):
    """Pytest-friendly entry point.

    Defaults to a small problem (numel=1024) and the local visible GPU
    count so the test fits a CI box; tweak via CLI for perf sweeps.

    ``auto_register`` defaults to False to match the C++ default — direct
    SDMA writes into a registered user buffer only produce correct values
    when the buffer is uncached device memory.  PyTorch's caching
    allocator is cached, so the test allocates output via ``torch.empty``
    and would otherwise read stale (all-zero) data from L2.  Pass True
    only when ``out_mori`` is allocated through ``mori.shmem.shmem_malloc``
    (or another uncached allocator).
    """
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size >= 2, f"AllGatherIntoTensor needs >=2 GPUs, got {world_size}"
    if dtypes is None:
        dtypes = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int64,
        ]
    if async_modes is None:
        async_modes = [False, True]

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, port, numel, dtypes, async_modes, auto_register),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unit test for mori::collective::AllGatherIntoTensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--numel", type=int, default=1024, help="Per-rank element count"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of ranks (default: torch.cuda.device_count())",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Restrict to a single torch dtype, e.g. bfloat16",
    )
    parser.add_argument(
        "--auto-register",
        action="store_true",
        help="Enable experimental zero-copy direct-write path "
        "(requires uncached recv buffer; cached PyTorch "
        "tensors will read back zeros).",
    )
    args = parser.parse_args()

    if args.dtype is not None:
        from tests.python.utils import string_to_dtype

        dtypes = [string_to_dtype(args.dtype)]
    else:
        dtypes = None

    try:
        test_allgather_into_tensor(
            numel=args.numel,
            world_size=args.world_size,
            dtypes=dtypes,
            auto_register=args.auto_register,
        )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

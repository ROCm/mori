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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Bit-exact test for ``HierAllGather.enqueue_param_contiguous`` (param-
contiguous zero-copy) vs a ``torch.distributed.all_gather_into_tensor``
reference reshuffled into the FSDP ``[param][rank]`` layout.

The zero-copy path pushes straight into the ``[param][rank]`` output instead of
gathering rank-major and reshuffling; it must be byte-identical (AllGather is a
pure data move).

Cross node -- launch under torchrun::

    torchrun --nnodes=2 --nproc_per_node=4 ... \
        tests/python/ccl/test_hier_allgather_param_contiguous.py
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import HierAllGather

# Cross-node torchrun-only harness: SKIP under a plain `pytest` collection (no
# torchrun env) so the suite does not ERROR; runs only when torchrun sets
# RANK/WORLD_SIZE.
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ or "WORLD_SIZE" not in os.environ,
    reason="cross-node harness; launch under torchrun",
)

_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]

# Per-rank per-param element counts (packed shard = concat of these). Sizes are
# LARGE (multi-MiB total) so the fresh output allocation lands on its own caching-
# allocator segment base -- required for the direct path's ShmemSymmetricRegister
# / hipIpcGetMemHandle of the intra-node IPC peers (a sub-allocation aborts). All
# even -> 4-byte aligned byte extents for bf16/fp16.
_PARAM_SPLITS = [1048576, 524288, 262144, 131072, 65536]

# Large profile: matches Qwen-7B's biggest FSDP all-gathers (embed + lm_head,
# ~544M elems gathered = ~1.09GB/rank bf16). Per-rank shard ~= 544M/8 = 68M.
# Two ~34M splits so the gathered total per param crosses the u32 byte-offset
# regime: int32 gathered = 8*68M*4 ~= 2.14GB ~= 2^31 bytes -> probes u32
# byte-index overflow in the scatter/ring kernels. bf16 gathered ~= 1.07GB/rank.
_PARAM_SPLITS_LARGE = [34078720, 34078720]  # 2 * ~2^25.02; sum = 68157440/rank


def _make_input(dtype, count, rank, device):
    base = (rank + 1) * 17
    ramp = torch.arange(count, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _expected_param_contiguous(inp, splits, world_size, rank, device):
    """Reference: RCCL rank-major gather -> reshuffle to [param][rank]."""
    count = inp.numel()
    rank_major = torch.empty(count * world_size, dtype=inp.dtype, device=device)
    dist.all_gather_into_tensor(rank_major, inp)  # [r0_shard, r1_shard, ...]
    expected = torch.empty(count * world_size, dtype=inp.dtype, device=device)
    o = 0
    for e in splits:
        for r in range(world_size):
            src = rank_major[r * count + o : r * count + o + e]
            dst_start = o * world_size + r * e
            expected[dst_start : dst_start + e] = src
        o += e
    return expected


def _run_one(handle, dtype, rank, world_size, device, splits):
    count = sum(splits)
    inp = _make_input(dtype, count, rank, device)
    out = torch.empty(count * world_size, dtype=dtype, device=device)

    offsets = []
    acc = 0
    for e in splits:
        offsets.append(acc)
        acc += e
    ss = torch.tensor(splits, dtype=torch.int64, device=device)
    so = torch.tensor(offsets, dtype=torch.int64, device=device)

    ref = _expected_param_contiguous(inp, splits, world_size, rank, device)

    stream = torch.cuda.current_stream()
    ok = handle.enqueue_param_contiguous(inp, out, count, ss, so, stream)
    assert ok, (
        f"enqueue_param_contiguous returned False dtype={dtype} "
        f"(supports={handle.supports_param_contiguous_output()})"
    )
    stream.synchronize()
    torch.cuda.synchronize()

    if not torch.equal(out, ref):
        diff = (out != ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"param-contiguous mismatch dtype={dtype}: positions={diff} "
            f"got={out[diff].tolist()} ref={ref[diff].tolist()}"
        )


def _run_profile(name, splits, dtypes, reps, rank, world_size, ranks_per_node, device):
    """Build a fresh handle for this size profile and validate bit-exact."""
    count = sum(splits)
    per_rank_bytes = count * 4 + 4096
    handle = HierAllGather(
        my_pe=rank,
        npes=world_size,
        ranks_per_node=ranks_per_node,
        input_buffer_size=per_rank_bytes,
        output_buffer_size=per_rank_bytes * world_size,
        copy_output_to_user=True,
    )
    if rank == 0:
        print(
            f"[{name}] world={world_size} rpn={ranks_per_node} "
            f"num_nodes={handle.num_nodes} count/rank={count} "
            f"gathered_int32_bytes={count * world_size * 4} "
            f"supports={handle.supports_param_contiguous_output()}"
        )
    try:
        if not handle.supports_param_contiguous_output():
            if rank == 0:
                print(f"[{name}] SKIP: direct param-contiguous path unavailable")
            return True
        for _rep in range(reps):
            for dtype in dtypes:
                _run_one(handle, dtype, rank, world_size, device, splits)
                if rank == 0 and _rep == 0:
                    print(f"[{name}]   ok dtype={dtype}")
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"[{name}] PASSED")
        return True
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
        dist.barrier()


def _worker_body(rank, world_size, ranks_per_node, device):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    try:
        # Small profile: all dtypes, 3 reps (flag-recycle coverage).
        _run_profile(
            "small", _PARAM_SPLITS, _DTYPES, 3, rank, world_size, ranks_per_node, device
        )
        # Large profile: the Qwen embed+lm_head band. bf16/fp32 (FSDP dtypes) +
        # int32 (probes 2^31 byte-offset overflow in the scatter/ring). Fewer
        # reps -- each is ~2GB buffers.
        _run_profile(
            "large",
            _PARAM_SPLITS_LARGE,
            [torch.bfloat16, torch.float32, torch.int32],
            2,
            rank,
            world_size,
            ranks_per_node,
            device,
        )
        if rank == 0:
            print("test_hier_allgather_param_contiguous: PASSED")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def _run_torchrun():
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    try:
        _worker_body(rank, world_size, ranks_per_node, device)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def test_hier_allgather_param_contiguous():
    """Pytest entry: runs only under torchrun (guarded by module pytestmark)."""
    _run_torchrun()


if __name__ == "__main__":
    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            _run_torchrun()
        else:
            raise SystemExit("launch under torchrun (cross-node)")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

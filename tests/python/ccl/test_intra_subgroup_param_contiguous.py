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
"""Bit-exact test for the HSDP winning path:
``IntraNodeSubGroupAllgatherSdma.gather_kernel_direct_param_contiguous`` (the
intra-node PARAM-CONTIGUOUS zero-copy direct scatter) vs a
``torch.distributed.all_gather_into_tensor`` reference over the per-node
sub-group, reshuffled into the FSDP ``[param][rank]`` layout.

This param-contiguous zero-copy kernel path has its own standalone bit-exact
test here (the plain gather is covered in test_intra_subgroup_sdma.py). AllGather
is a pure data move => zero tolerance (``torch.equal``). This mirrors exactly the
call the adapter ``MoriIntraSubGroupAllGather.__call__`` makes: num_blocks=1,
first_block=0, world_size=group_size, register_output_buffer +
finish_direct_stream.

Cross node (2 nodes) -- launch under torchrun::

    torchrun --nnodes=2 --nproc_per_node=4 ... \
        tests/python/ccl/test_intra_subgroup_param_contiguous.py
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import IntraNodeSubGroupAllgatherSdma

# Cross-node torchrun-only harness: SKIP under a plain `pytest` collection (no
# torchrun env) so the suite does not ERROR; runs only when torchrun sets
# RANK/WORLD_SIZE.
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ or "WORLD_SIZE" not in os.environ,
    reason="cross-node harness; launch under torchrun",
)

_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]

# Per-rank per-param element counts (packed shard = concat of these). Must be
# large enough that the fresh output allocation lands on its own caching-allocator
# segment base (output_ptr == the registered segment base). The direct path's
# ShmemSymmetricRegister / hipIpcGetMemHandle registers the containing allocation;
# the scatter writes to peerPtrs[remotePe] + dstBaseOffset assuming peerPtr == the
# buffer base. If the output is a torch sub-allocation (small buffer packed inside
# a larger pool segment) the peer pointer resolves to the segment base, not the
# buffer, and the scatter silently corrupts (writes land at the wrong slots).
# bf16 (2 bytes/elem) is the smallest tested dtype => its output (sum*G*2 bytes)
# is the one most at risk of sub-allocation, so size for it: sum ~= 8.1M elems ->
# bf16 output ~= 65 MB, comfortably above the pool's own-segment threshold.
# All even -> 4-byte aligned.
_PARAM_SPLITS = [4194304, 2097152, 1048576, 524288, 262144]

_REPS = 12  # many reps to catch a flag-recycle / stale-read race


def _make_input(dtype, count, rank, device):
    base = (rank + 1) * 17
    ramp = torch.arange(count, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _expected_param_contiguous(inp, splits, G, subgroup, device):
    """Reference: rank-major all_gather over the per-node sub-group ->
    reshuffle to [param][rank] (the adapter's world_size == group_size G)."""
    count = inp.numel()
    rank_major = torch.empty(count * G, dtype=inp.dtype, device=device)
    dist.all_gather_into_tensor(rank_major, inp, group=subgroup)  # [g0, g1, ...]
    expected = torch.empty(count * G, dtype=inp.dtype, device=device)
    o = 0
    for e in splits:
        for g in range(G):
            src = rank_major[g * count + o : g * count + o + e]
            dst_start = o * G + g * e
            expected[dst_start : dst_start + e] = src
        o += e
    return expected


def _gather_once(handle, inp, out, ss_u32, so_u32, blk_stride_u32, G):
    """One param-contiguous direct gather into an ALREADY-registered ``out``
    (matches the adapter's register-once + repeated-call contract)."""
    stream = torch.cuda.current_stream()
    ok = handle.gather_kernel_direct_param_contiguous(
        inp,
        out,
        blk_stride_u32,
        1,  # num_blocks = 1 (single node block; pure intra)
        G,  # world_size for the [param][rank] output stride
        ss_u32,
        so_u32,
        stream=stream,
        prepare_barrier=True,
        first_block=0,
    )
    assert ok, "gather_kernel_direct_param_contiguous returned False"
    handle.finish_direct_stream(stream=stream, barrier=True)
    stream.synchronize()
    torch.cuda.synchronize()


def _run_dtype(handle, dtype, rank, G, subgroup, device, reps):
    """Register the output ONCE, then gather ``reps`` times into it -- exactly
    the FSDP adapter pattern (persistent registered output, repeated AG). A race
    in the completion/quiet path shows up as a run-to-run mismatch here."""
    splits = _PARAM_SPLITS
    count = sum(splits)
    inp = _make_input(dtype, count, rank, device)
    out = torch.empty(count * G, dtype=dtype, device=device)
    ref = _expected_param_contiguous(inp, splits, G, subgroup, device)

    elem = inp.element_size()
    offsets, acc = [], 0
    for e in splits:
        offsets.append(acc)
        acc += e
    ss_u32 = torch.tensor(
        [(e * elem) // 4 for e in splits], dtype=torch.int64, device=device
    )
    so_u32 = torch.tensor(
        [(o * elem) // 4 for o in offsets], dtype=torch.int64, device=device
    )
    blk_stride_u32 = (count * elem) // 4

    # register ONCE (collective symmetric op; barrier so every peer's IPC handles
    # are exchanged before any kernel dereferences peerPtrs).
    handle.register_output_buffer(out)
    torch.cuda.synchronize()
    dist.barrier()
    try:
        for _rep in range(reps):
            out.zero_()  # ensure a stale-read race can't be masked by prior data
            torch.cuda.synchronize()
            dist.barrier()
            _gather_once(handle, inp, out, ss_u32, so_u32, blk_stride_u32, G)
            if rank == 0 and _rep == 0 and dtype == torch.bfloat16:
                E0 = splits[0]
                slotvals = [int(out[g * E0].item()) for g in range(G)]
                bases = [int((g + 1) * 17) for g in range(G)]
                print(
                    f"RECVDUMP rank0 param0 slot bases got={slotvals} "
                    f"(expect r-th slot = base {bases}) inp0={int(inp[0].item())}"
                )
            if not torch.equal(out, ref):
                diff = (out != ref).nonzero(as_tuple=False).flatten()[:8].tolist()
                raise AssertionError(
                    f"intra param-contiguous mismatch dtype={dtype} rank={rank} "
                    f"rep={_rep}: positions={diff} got={out[diff].tolist()} "
                    f"ref={ref[diff].tolist()}"
                )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        handle.deregister_output_buffer(out)


def _worker_body(rank, world_size, G, device):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank

    # Per-node arithmetic sub-groups {n*G .. n*G+G-1}; every rank must build all.
    node_groups = []
    for n in range(world_size // G):
        ranks = list(range(n * G, n * G + G))
        node_groups.append(dist.new_group(ranks=ranks))
    subgroup = node_groups[rank // G]

    count = sum(_PARAM_SPLITS)
    handle = IntraNodeSubGroupAllgatherSdma(
        my_pe=rank,
        npes=world_size,
        out_buffer_bytes=count * 4 * G + 4096,
        group_size=G,
        group_pos=rank % G,
        pe_base=(rank // G) * G,
        pe_stride=1,
    )
    if rank == 0:
        print(
            f"intra param-contig: world={world_size} G={G} "
            f"num_nodes={world_size // G} reps={_REPS}"
        )
    try:
        for dtype in _DTYPES:
            _run_dtype(handle, dtype, rank, G, subgroup, device, _REPS)
            if rank == 0:
                print(f"  ok dtype={dtype} ({_REPS} reps)")
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("test_intra_subgroup_param_contiguous: PASSED")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
        dist.barrier()
        shmem.shmem_finalize()


def _run_torchrun():
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    G = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
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
        _worker_body(rank, world_size, G, device)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def test_intra_subgroup_param_contiguous():
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

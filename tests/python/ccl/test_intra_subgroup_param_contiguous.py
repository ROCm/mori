#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Bit-exact test for the HSDP winning path:
``IntraNodeSubGroupAllgatherSdma.gather_kernel_direct_param_contiguous`` (the
intra-node PARAM-CONTIGUOUS zero-copy direct scatter) vs a
``torch.distributed.all_gather_into_tensor`` reference over the per-node
sub-group, reshuffled into the FSDP ``[param][rank]`` layout.

Motivation: HSDP FSDP2 beats RCCL by +17.6% using the intra-only SDMA AG with
this param-contiguous zero-copy write, but the loss varied run-to-run because
this exact kernel path had NO standalone bit-exact test (only the plain gather
was covered in test_intra_subgroup_sdma.py). AllGather is a pure data move =>
ZERO tolerance (``torch.equal``). This mirrors EXACTLY the call the adapter
``MoriIntraSubGroupAllGather.__call__`` makes: num_blocks=1, first_block=0,
world_size=group_size, register_output_buffer + finish_direct_stream.

Cross node (2 nodes) -- launch under torchrun::

    torchrun --nnodes=2 --nproc_per_node=4 ... \
        tests/python/ccl/test_intra_subgroup_param_contiguous.py
"""

import os
import traceback

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import IntraNodeSubGroupAllgatherSdma

_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]

# Per-rank per-param element counts (packed shard = concat of these). LARGE so
# the fresh output allocation lands on its own caching-allocator segment base --
# required for the direct path's ShmemSymmetricRegister / hipIpcGetMemHandle of
# the intra-node IPC peers (a sub-allocation aborts). All even -> 4-byte aligned.
_PARAM_SPLITS = [1048576, 524288, 262144, 131072, 65536]

_REPS = 12  # many reps to catch a flag-recycle / stale-read race (loss varied)


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


def _run_one(handle, dtype, rank, G, subgroup, device, direct_reg):
    splits = _PARAM_SPLITS
    count = sum(splits)
    inp = _make_input(dtype, count, rank, device)
    out = torch.empty(count * G, dtype=dtype, device=device)

    ref = _expected_param_contiguous(inp, splits, G, subgroup, device)

    # Build the u32-lane split sizes/offsets exactly like the adapter does.
    elem = inp.element_size()
    offsets, acc = [], 0
    for e in splits:
        offsets.append(acc)
        acc += e
    ss_u32 = torch.tensor([(e * elem) // 4 for e in splits], dtype=torch.int64, device=device)
    so_u32 = torch.tensor([(o * elem) // 4 for o in offsets], dtype=torch.int64, device=device)
    blk_stride_u32 = (count * elem) // 4

    stream = torch.cuda.current_stream()
    # register the (large) output for the direct IPC scatter. Registration is a
    # collective symmetric op; barrier so every peer's IPC handles are exchanged
    # before any kernel dereferences peerPtrs (matches the adapter's persistent
    # register-once + prepare-barrier contract).
    handle.register_output_buffer(out)
    torch.cuda.synchronize()
    dist.barrier()
    ok = handle.gather_kernel_direct_param_contiguous(
        inp, out, blk_stride_u32,
        1,       # num_blocks = 1 (single node block; pure intra)
        G,       # world_size for the [param][rank] output stride
        ss_u32, so_u32,
        stream=stream, prepare_barrier=True, first_block=0,
    )
    assert ok, f"gather_kernel_direct_param_contiguous returned False dtype={dtype}"
    handle.finish_direct_stream(stream=stream, barrier=True)
    stream.synchronize()
    torch.cuda.synchronize()
    handle.deregister_output_buffer(out)

    if not torch.equal(out, ref):
        diff = (out != ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"intra param-contiguous mismatch dtype={dtype} rank={rank}: "
            f"positions={diff} got={out[diff].tolist()} ref={ref[diff].tolist()}"
        )


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
        print(f"intra param-contig: world={world_size} G={G} "
              f"num_nodes={world_size // G} reps={_REPS}")
    try:
        for _rep in range(_REPS):
            for dtype in _DTYPES:
                _run_one(handle, dtype, rank, G, subgroup, device, None)
                if rank == 0 and _rep == 0:
                    print(f"  ok dtype={dtype}")
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
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size,
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


if __name__ == "__main__":
    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            _run_torchrun()
        else:
            raise SystemExit("launch under torchrun (cross-node)")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

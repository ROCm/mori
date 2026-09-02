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
"""One-shot all-to-all over mori's torch SymmetricMemory backend, in HIP.

    torchrun --nnodes=1 --nproc_per_node=<gpus> all2all_hip.py --chunk-kib 256

The kernel below is JIT-built on first run, so there is nothing to build first. All it
needs from the backend is the peer pointer array torch publishes: the chunk rank r owes
rank p lands at ``peers[p] + r*chunk_bytes``. No mori shmem, no cco.

``all2all_triton.py`` is the same example in Triton.
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mori.allocator import handle_type  # importing registers the "MORI" backend
from torch.utils.cpp_extension import load_inline

HIP_SOURCE = r"""
#include <c10/hip/HIPStream.h>

#include <algorithm>

constexpr int kThreads = 256;

// Grid flattens (peer, slice of chunk). One block per peer would idle all but world_size
// CUs and keep too few writes in flight to cover interconnect latency.
__global__ void All2AllPushPtrs(const uint4* __restrict__ send, void** __restrict__ peers,
                                size_t chunk_bytes, int rank_id, int world_size,
                                int blocks_per_peer) {
  const size_t n_vec = chunk_bytes / sizeof(uint4);
  const int peer = blockIdx.x / blocks_per_peer;
  const int sub = blockIdx.x % blocks_per_peer;
  if (peer >= world_size) return;

  auto* dst = reinterpret_cast<uint4*>(static_cast<char*>(peers[peer]) +
                                       static_cast<size_t>(rank_id) * chunk_bytes);
  const uint4* src = send + static_cast<size_t>(peer) * n_vec;

  const size_t span = static_cast<size_t>(blocks_per_peer) * blockDim.x;
  for (size_t i = static_cast<size_t>(sub) * blockDim.x + threadIdx.x; i < n_vec; i += span) {
    dst[i] = src[i];
  }
}

// Two blocks per CU across peers, capped by how much there is to slice.
static int BlocksPerPeer(int64_t chunk_bytes, int64_t world_size) {
  static const int cus = [] {
    int v = 64, dev = 0;
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev);
    return v;
  }();
  const int64_t n_vec = chunk_bytes / static_cast<int64_t>(sizeof(uint4));
  const int64_t bpp = std::max<int64_t>(1, cus * 2 / std::max<int64_t>(1, world_size));
  return static_cast<int>(std::min<int64_t>(bpp, std::max<int64_t>(1, n_vec / kThreads)));
}

// peers_dev is hdl.buffer_ptrs_dev; send is local, world_size*chunk_bytes. Push semantics:
// no remote reads and no per-peer handshake, so the caller barriers before reading.
void all2all_push_ptrs(const at::Tensor& send, int64_t peers_dev, int64_t chunk_bytes,
                       int64_t rank_id, int64_t world_size) {
  TORCH_CHECK(send.is_contiguous(), "send must be contiguous");
  TORCH_CHECK(chunk_bytes % sizeof(uint4) == 0, "chunk_bytes must be a multiple of 16, got ",
              chunk_bytes);
  TORCH_CHECK(send.nbytes() == static_cast<size_t>(world_size * chunk_bytes), "send holds ",
              send.nbytes(), " bytes, expected ", world_size * chunk_bytes);

  const int bpp = BlocksPerPeer(chunk_bytes, world_size);
  hipLaunchKernelGGL(All2AllPushPtrs, dim3(world_size * bpp), dim3(kThreads), 0,
                     c10::hip::getCurrentHIPStream(),
                     reinterpret_cast<const uint4*>(send.data_ptr()),
                     reinterpret_cast<void**>(peers_dev), static_cast<size_t>(chunk_bytes),
                     static_cast<int>(rank_id), static_cast<int>(world_size), bpp);
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "all2all_push_ptrs launch failed: ", hipGetErrorString(err));
}
"""


def build():
    """JIT the kernel. Concurrent torchrun ranks share one build: torch holds a file lock."""
    return load_inline(
        name="all2all_hip_kernel",
        cpp_sources=(
            "void all2all_push_ptrs(const at::Tensor&, int64_t, int64_t, int64_t, int64_t);"
        ),
        cuda_sources=HIP_SOURCE,
        functions=["all2all_push_ptrs"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chunk-kib", type=int, default=256, help="bytes pushed to each peer"
    )
    p.add_argument("--iters", type=int, default=20, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    push = build().all2all_push_ptrs

    # Defaults so a bare `python3 all2all_*.py` is a 1-rank smoke test; torchrun sets them.
    for key, val in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        os.environ.setdefault(key, val)
    rank_id = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group_name = dist.group.WORLD.group_name

    symm_mem.set_backend("MORI")

    chunk_bytes = args.chunk_kib * 1024
    elems = chunk_bytes // 4

    # Receive window: one chunk per source rank, writable by peers.
    recv = symm_mem.empty(world_size * elems, dtype=torch.int32, device=device)
    recv.zero_()
    # Ordinary local memory. Chunk p carries a value identifying (me -> p).
    send = torch.empty(world_size * elems, dtype=torch.int32, device=device)
    for p in range(world_size):
        send[p * elems : (p + 1) * elems] = rank_id * 1000 + p
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(recv, group_name)
    peers_dev = hdl.buffer_ptrs_dev

    if rank_id == 0:
        print(
            f"kernel=HIP  world={world_size}  handle={handle_type(local_rank)}  "
            f"chunk={args.chunk_kib} KiB"
        )
        print(f"peers: {' '.join(hex(p) for p in hdl.buffer_ptrs)}")

    def run_once():
        push(send, peers_dev, chunk_bytes, rank_id, world_size)
        torch.cuda.synchronize()
        dist.barrier()  # no device-side barrier in the backend yet

    # Peers write into this window, so everyone must finish clearing before anyone pushes.
    dist.barrier()
    run_once()

    errors = 0
    for r in range(world_size):
        got, want = recv[r * elems].item(), r * 1000 + rank_id
        if got != want:
            errors += 1
            print(f"[rank {rank_id}] chunk {r}: got {got}, want {want}", flush=True)
    failed = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if rank_id == 0 and failed.item() == 0:
        print(f"correctness: OK ({world_size}x{world_size} chunks)")

    for _ in range(args.warmup):
        run_once()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    start.record()
    for _ in range(args.iters):
        push(send, peers_dev, chunk_bytes, rank_id, world_size)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters

    # Only (world_size-1) chunks leave the device; the self chunk stays local.
    gbps = torch.tensor(
        [(world_size - 1) * chunk_bytes / (ms / 1e3) / 1e9], dtype=torch.float64
    )
    dist.all_reduce(gbps, op=dist.ReduceOp.SUM)
    if rank_id == 0:
        print(f"{ms * 1e3:7.1f} us/iter, {gbps.item():8.1f} GB/s aggregate")
        print("SUCCESS" if failed.item() == 0 else "FAILED")

    dist.barrier()
    del hdl, recv
    gc.collect()
    dist.destroy_process_group()
    return 0 if failed.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

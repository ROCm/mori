// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Device-side bandwidth test helpers shared across all bw benchmarks.
// Include this header only from device (.cpp/.hip) translation units, NOT from
// util.cpp, to avoid duplicate __global__ kernel symbols at link time.

#pragma once

#include "mori/shmem/shmem.hpp"

namespace mori::shmem::benchmark {

// Intra-block linear thread index.
__device__ inline int linear_tid() {
  return threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
}

// Cross-block barrier: all nblocks rendezvous at the end of iteration i.
//
// counter_d layout:
//   [0]  arrival counter  — monotonically incremented by each block leader via atomicInc
//   [1]  phase counter    — incremented by the last-arriving block to signal round completion
//
// Must be called by ALL threads of ALL blocks with the same (counter_d, nblocks, i).
__device__ inline void bw_cross_block_barrier_round(volatile unsigned int* counter_d, int nblocks,
                                                    int i) {
  // __syncthreads() #1 — intra-block fence.
  // Ensures every thread in this block has finished issuing its NBI ops
  // for iteration i before thread 0 announces the block's arrival.
  // Without this, thread 0 could increment the arrival counter while other
  // threads are still posting NBI operations, making the barrier count meaningless.
  __syncthreads();

  if (!linear_tid()) {
    // __threadfence() — inter-CU memory ordering fence.
    // After __syncthreads() above, all NBI ops from this block are issued.
    // This fence ensures those writes are globally visible to every CU before
    // we increment the arrival counter.  Without it, a block that exits the
    // barrier first could observe stale remote data from a previous iteration.
    __threadfence();

    // atomicInc: each block leader increments counter_d[0] atomically.
    // The return value c is unique per block per round; the block that receives
    // the last slot (c == nblocks*(i+1)-1) is responsible for advancing the phase.
    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (i + 1) - 1)) {
      // Single writer: only this block reaches here, so no atomic needed.
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(i + 1)) {
    }
  }

  // __syncthreads() #2 — intra-block fence.
  // Non-leader threads (tid != 0) have been waiting here since __syncthreads() #1.
  // The leader (thread 0) is spinning in the while loop above; this sync
  // prevents non-leaders from racing into the next iteration before the leader
  // confirms that all blocks have arrived and the round is complete.
  __syncthreads();
}

// Final cross-block barrier + quiet: same arrival protocol as above, but also
// drains all in-flight NBI ops so the host stop-event captures only completed work.
__device__ inline void bw_final_barrier_and_quiet(volatile unsigned int* counter_d, int nblocks,
                                                  int iter) {
  // __syncthreads() #1 — same role as in bw_cross_block_barrier_round:
  // wait for all threads in this block to finish issuing their last NBI ops.
  __syncthreads();

  if (!linear_tid()) {
    // __threadfence() — same role: make this block's NBI writes globally
    // visible before counting arrival.
    __threadfence();

    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (iter + 1) - 1)) {
      // Winner block: drain in-flight ops before signalling completion.
      // This ensures the phase counter is set only after the winner's own
      // RDMA ops are fully retired, providing a clean ordering point.
      mori::shmem::ShmemQuietThread();
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(iter + 1)) {
    }

    // Every block leader drains its own block's remaining in-flight NBI ops.
    // Each block issues NBI ops independently, so each must quiet independently.
    // Without this, blocks could exit the kernel with in-flight RDMA operations,
    // causing hipDeviceSynchronize() to return before all transfers complete.
    mori::shmem::ShmemQuietThread();
  }

  // __syncthreads() #2 — intra-block fence.
  // Prevents non-leader threads from exiting the kernel while thread 0 is
  // still inside ShmemQuietThread().  hipDeviceSynchronize() on the host
  // returns only after every thread in every block has exited, so this sync
  // is what ties quiet completion to the host-side stop event.
  __syncthreads();
}

}  // namespace mori::shmem::benchmark

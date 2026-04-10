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

// N-PE ring put bandwidth: same launch pattern as p2p_put_bw (default block, -c/-t/-s),
// but each PE puts to (my_pe+1)%npes. Aggregate BW = npes*size*iters / MPI_MAX(local time).

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device_utils.hpp"
#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"
#include "util.hpp"

using namespace mori::application;
using namespace mori::shmem;
using namespace mori::perftest;

namespace {

__global__ void ring_bw_block(double* data_d, volatile unsigned int* counter_d, size_t len,
                              int my_pe, int npes, int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int peer = (my_pe + 1) % npes;

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * (len / static_cast<size_t>(nblocks));
    size_t chunk_doubles = len / static_cast<size_t>(nblocks);
    ShmemPutMemNbiBlock(slice, slice, chunk_doubles * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

__global__ void ring_bw_warp(double* data_d, volatile unsigned int* counter_d, size_t len,
                             int my_pe, int npes, int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int tid = bw_tid();
  int nwarps_per_block = (blockDim.x * blockDim.y * blockDim.z) / warpSize;
  int warpid = tid / warpSize;
  const int peer = (my_pe + 1) % npes;

  size_t put_per_block = len / static_cast<size_t>(nblocks);
  size_t put_per_warp = put_per_block / static_cast<size_t>(nwarps_per_block);

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * put_per_block + static_cast<size_t>(warpid) * put_per_warp;
    ShmemPutMemNbiWarp(slice, slice, put_per_warp * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

__global__ void ring_bw_thread(double* data_d, volatile unsigned int* counter_d, size_t len,
                               int my_pe, int npes, int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int tid = bw_tid();
  int nthreads = blockDim.x * blockDim.y * blockDim.z;
  const int peer = (my_pe + 1) % npes;

  size_t put_per_block = len / static_cast<size_t>(nblocks);
  size_t put_per_thread = put_per_block / static_cast<size_t>(nthreads);

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * put_per_block + static_cast<size_t>(tid) * put_per_thread;
    ShmemPutMemNbiThread(slice, slice, put_per_thread * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

void launch_ring_bw(PutScope scope, dim3 grid, dim3 block, double* data_d, unsigned int* counter_d,
                    size_t len_doubles, int my_pe, int npes, int count) {
  switch (scope) {
    case PutScope::kBlock:
      hipLaunchKernelGGL(ring_bw_block, grid, block, 0, 0, data_d, counter_d, len_doubles, my_pe,
                         npes, count);
      break;
    case PutScope::kWarp:
      hipLaunchKernelGGL(ring_bw_warp, grid, block, 0, 0, data_d, counter_d, len_doubles, my_pe,
                         npes, count);
      break;
    case PutScope::kThread:
      hipLaunchKernelGGL(ring_bw_thread, grid, block, 0, 0, data_d, counter_d, len_doubles, my_pe,
                         npes, count);
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
#ifndef MORI_WITH_MPI
  std::fprintf(
      stderr,
      "mori_shmem_ring_put_bw requires MORI_WITH_MPI (enable WITH_MPI / BUILD_EXAMPLES).\n");
  return 1;
#else

  PerfContext ctx;
  PerfArgs& args = ctx.args;

  assert(PerfInit(argc, argv, &ctx) == 0);

  if (ctx.npes < 2) {
    if (ctx.my_pe == 0) {
      std::fprintf(stderr, "mori_shmem_ring_put_bw requires at least 2 PEs (npes=%d)\n", ctx.npes);
    }
    PerfFinalize(&ctx);
    return 1;
  }

  const PutScope scope = args.put_scope;
  const char* scope_name = ScopeToChar(scope);
  dim3 grid(args.nblocks, 1, 1);
  dim3 block(args.threads_per_block, 1, 1);

  ShmemBarrierAll();
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  PerfRes res;
  PerfResAlloc(&res);

  void* symm = ShmemMalloc(args.max_size);
  if (!symm) {
    std::fprintf(stderr, "ShmemMalloc failed\n");
    PerfResFree(&res);
    PerfFinalize(&ctx);
    return 1;
  }
  auto* data_d = static_cast<double*>(symm);
  HIP_RUNTIME_CHECK(hipMemset(data_d, 0, args.max_size));

  std::vector<PerfTableRow> bandwidth_table;
  if (ctx.my_pe == 0) {
    bandwidth_table.reserve(64);
  }

  for (size_t size_bytes = args.min_size; size_bytes <= args.max_size;
       size_bytes *= args.step_factor) {
    if (size_bytes % sizeof(double) != 0) {
      continue;
    }
    const size_t len_doubles = size_bytes / sizeof(double);
    if (len_doubles == 0) {
      continue;
    }

    if (!size_ok(scope, size_bytes, args.nblocks, args.threads_per_block, ctx.device_warp_size)) {
      if (ctx.my_pe == 0) {
        bandwidth_table.push_back(PerfTableRow{size_bytes, true, 0.0});
      }
      ShmemBarrierAll();
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
      continue;
    }

    ShmemBarrierAll();
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    auto launch = [&](int count) {
      launch_ring_bw(scope, grid, block, data_d, res.counter_d, len_doubles, ctx.my_pe, ctx.npes,
                     count);
    };
    const float ms_local = RunWarmupAndTimed(res, args.warmup, args.iters, launch);
    double ms_max = 0.0;
    const double ms_d = static_cast<double>(ms_local);
    MPI_Reduce(&ms_d, &ms_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (ctx.my_pe == 0) {
      const double total_bytes = static_cast<double>(ctx.npes) * static_cast<double>(size_bytes) *
                                 static_cast<double>(args.iters);
      const double gbps = total_bytes / (ms_max * (kBToGb / static_cast<double>(kMsToS)));
      bandwidth_table.push_back(PerfTableRow{size_bytes, false, gbps});
    }

    ShmemBarrierAll();
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  }

  if (ctx.my_pe == 0) {
    std::printf(
        "# npes=%d ring -> (pe+1)%%np ; aggregate BW = sum PE puts / max PE time ; "
        "defaults match p2p_put_bw (block, grid=-c, threads=-t)\n",
        ctx.npes);
    PrintPerfTable("shmem_ring_put_bw", scope_name, args.nblocks, args.threads_per_block,
                   ctx.device_warp_size, args.iters, args.warmup, PerfTableMetric::kBandwidthGbps,
                   bandwidth_table);
  }

  ShmemBarrierAll();
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  PerfResFree(&res);
  ShmemFree(symm);
  PerfFinalize(&ctx);
  return 0;
#endif
}

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

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "device_utils.hpp"
#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"
#include "util.hpp"

namespace mori::shmem::perftest {

// Block-scope get: each block cooperatively gets its slice from the peer.
__global__ void bw_block(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                         int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int peer = !pe;

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * (len / static_cast<size_t>(nblocks));
    size_t chunk_doubles = len / static_cast<size_t>(nblocks);
    ShmemGetMemNbiBlock(slice, slice, chunk_doubles * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

// Warp-scope get: each warp cooperatively gets its slice from the peer.
__global__ void bw_warp(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                        int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int tid = linear_tid();
  int nwarps_per_block = (blockDim.x * blockDim.y * blockDim.z) / warpSize;
  int warpid = tid / warpSize;
  const int peer = !pe;

  size_t get_per_block = len / static_cast<size_t>(nblocks);
  size_t get_per_warp = get_per_block / static_cast<size_t>(nwarps_per_block);

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * get_per_block + static_cast<size_t>(warpid) * get_per_warp;
    ShmemGetMemNbiWarp(slice, slice, get_per_warp * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

// Thread-scope get: each thread independently gets its slice from the peer.
__global__ void bw_thread(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                          int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int tid = linear_tid();
  int nthreads = blockDim.x * blockDim.y * blockDim.z;
  const int peer = !pe;

  size_t get_per_block = len / static_cast<size_t>(nblocks);
  size_t get_per_thread = get_per_block / static_cast<size_t>(nthreads);

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * get_per_block + static_cast<size_t>(tid) * get_per_thread;
    ShmemGetMemNbiThread(slice, slice, get_per_thread * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

void launch_bw(PutScope scope, dim3 grid, dim3 block, double* data_d, unsigned int* counter_d,
               size_t len_doubles, int my_pe, int count) {
  switch (scope) {
    case PutScope::kBlock:
      hipLaunchKernelGGL(bw_block, grid, block, 0, 0, data_d, counter_d, len_doubles, my_pe, count);
      break;
    case PutScope::kWarp:
      hipLaunchKernelGGL(bw_warp, grid, block, 0, 0, data_d, counter_d, len_doubles, my_pe, count);
      break;
    case PutScope::kThread:
      hipLaunchKernelGGL(bw_thread, grid, block, 0, 0, data_d, counter_d, len_doubles, my_pe,
                         count);
      break;
  }
}

}  // namespace mori::shmem::perftest

int main(int argc, char** argv) {
#ifndef MORI_WITH_MPI
  std::fprintf(stderr,
               "mori_shmem_get_bw requires MORI_WITH_MPI (enable WITH_MPI / BUILD_EXAMPLES).\n");
  return 1;
#else

  using namespace mori::application;
  using namespace mori::shmem;
  using namespace mori::shmem::perftest;

  PerfContext ctx{};
  const int init_rc = PerfInit(argc, argv, &ctx);
  if (init_rc != 0) {
    return init_rc;
  }
  PerfArgs& args = ctx.args;

  const int my_pe = ctx.my_pe;
  const int npes = ctx.npes;
  if (npes != 2) {
    if (my_pe == 0) {
      std::fprintf(stderr, "mori_shmem_get_bw requires exactly 2 PEs (npes=%d)\n", npes);
    }
    PerfFinalize(&ctx);
    return 1;
  }

  PutScope scope = args.put_scope;

  // In unidirectional mode only PE 0 issues gets (PE 1 is the data source).
  // In bidirectional mode both PEs issue gets simultaneously.
  const bool run_kernels = args.bidirectional || my_pe == 0;

  const char* scope_name = ScopeToChar(scope);

  dim3 grid(args.nblocks, 1, 1);
  dim3 block(args.threads_per_block, 1, 1);

  PerfRes res;
  if (run_kernels) {
    PerfResAlloc(&res);
  }

  void* symm = ShmemMalloc(args.max_size);
  if (!symm) {
    std::fprintf(stderr, "ShmemMalloc failed\n");
    if (run_kernels) {
      PerfResFree(&res);
    }
    PerfFinalize(&ctx);
    return 1;
  }
  double* data_d = static_cast<double*>(symm);
  HIP_RUNTIME_CHECK(hipMemset(data_d, 0, args.max_size));

  std::vector<PerfTableRow> bandwidth_table;
  if (my_pe == 0) {
    bandwidth_table.reserve(64);
  }

  for (size_t size_bytes = args.min_size; size_bytes <= args.max_size;
       size_bytes *= args.step_factor) {
    if (size_bytes % sizeof(double) != 0) {
      continue;
    }
    size_t len_doubles = size_bytes / sizeof(double);
    if (!size_ok(scope, size_bytes, args.nblocks, args.threads_per_block, ctx.device_warp_size)) {
      if (my_pe == 0) {
        bandwidth_table.push_back(PerfTableRow{size_bytes, true, 0.0});
      }
      ShmemBarrierAll();
      continue;
    }

    if (run_kernels) {
      const float ms = RunWarmupAndTimed(res, args.warmup, args.iters, [&](int count) {
        launch_bw(scope, grid, block, data_d, res.counter_d, len_doubles, my_pe, count);
      });

      const double gbps = static_cast<double>(size_bytes) /
                          (static_cast<double>(ms) * (kBToGb / (args.iters * kMsToS)));

      if (args.bidirectional) {
        double gbps_local = gbps;
        double gbps_sum = 0.0;
        MPI_Reduce(&gbps_local, &gbps_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (my_pe == 0) {
          bandwidth_table.push_back(PerfTableRow{size_bytes, false, gbps_sum});
        }
      } else {
        if (my_pe == 0) {
          bandwidth_table.push_back(PerfTableRow{size_bytes, false, gbps});
        }
      }
    }

    ShmemBarrierAll();
  }

  ShmemBarrierAll();
  if (my_pe == 0) {
    const char* test_name = args.bidirectional ? "p2p_get_bw bidirection" : "p2p_get_bw unidirection";
    PrintPerfTable(test_name, scope_name, args.nblocks, args.threads_per_block,
                   ctx.device_warp_size, args.iters, args.warmup, PerfTableMetric::kBandwidthGbps,
                   bandwidth_table);
  }

  if (run_kernels) {
    PerfResFree(&res);
  }
  ShmemFree(symm);
  PerfFinalize(&ctx);
  return 0;
#endif
}

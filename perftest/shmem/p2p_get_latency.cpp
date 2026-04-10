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
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"
#include "util.hpp"

using namespace mori::application;
using namespace mori::shmem;
using namespace mori::perftest;

namespace {

__device__ inline int lat_tid() {
  return threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
}

// NVSHMEM latency_kern: single thread, put_nbi + quiet each iteration.
__global__ void lat_thread(double* data_d, size_t len_doubles, int pe, int iter) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int peer = !pe;
  for (int i = 0; i < iter; i++) {
    ShmemGetMemNbiThread(data_d, data_d, len_doubles * sizeof(double), peer, kDefaultQpId);
    ShmemQuietThread();
  }
}

// NVSHMEM latency_kern_warp: cooperative warp put; leader quiet.
__global__ void lat_warp(double* data_d, size_t len_doubles, int pe, int iter) {
  if (blockIdx.x != 0) {
    return;
  }
  const int tid = lat_tid();
  const int peer = !pe;
  for (int i = 0; i < iter; i++) {
    ShmemGetMemNbiWarp(data_d, data_d, len_doubles * sizeof(double), peer, kDefaultQpId);
    __syncthreads();
    if (!tid) {
      ShmemQuietThread();
    }
    __syncthreads();
  }
}

// NVSHMEM latency_kern_block: cooperative block put; leader quiet.
__global__ void lat_block(double* data_d, size_t len_doubles, int pe, int iter) {
  if (blockIdx.x != 0) {
    return;
  }
  const int tid = lat_tid();
  const int peer = !pe;
  for (int i = 0; i < iter; i++) {
    ShmemGetMemNbiBlock(data_d, data_d, len_doubles * sizeof(double), peer, kDefaultQpId);
    __syncthreads();
    if (!tid) {
      ShmemQuietThread();
    }
    __syncthreads();
  }
}

bool latency_size_ok(PutScope scope, size_t len_doubles, int threads_per_block,
                     int device_warp_size) {
  if (len_doubles == 0) {
    return false;
  }
  if (scope == PutScope::kThread || scope == PutScope::kWarp) {
    return true;
  }
  if (threads_per_block % device_warp_size != 0) {
    return false;
  }
  const int nw = threads_per_block / device_warp_size;
  return len_doubles % static_cast<size_t>(nw) == 0;
}

void launch_latency(PutScope scope, double* data_d, size_t len_doubles, int my_pe, int count,
                    int threads_per_block, int device_warp_size) {
  switch (scope) {
    case PutScope::kThread:
      hipLaunchKernelGGL(lat_thread, dim3(1), dim3(1), 0, 0, data_d, len_doubles, my_pe, count);
      break;
    case PutScope::kWarp:
      hipLaunchKernelGGL(lat_warp, dim3(1), dim3(device_warp_size, 1, 1), 0, 0, data_d, len_doubles,
                         my_pe, count);
      break;
    case PutScope::kBlock:
      hipLaunchKernelGGL(lat_block, dim3(1), dim3(threads_per_block, 1, 1), 0, 0, data_d,
                         len_doubles, my_pe, count);
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
#ifndef MORI_WITH_MPI
  std::fprintf(
      stderr,
      "mori_shmem_put_latency requires MORI_WITH_MPI (enable WITH_MPI / BUILD_EXAMPLES).\n");
  return 1;
#else

  PerfContext ctx{};
  const int init_rc = PerfInit(argc, argv, &ctx);
  if (init_rc != 0) {
    return init_rc;
  }
  PerfArgs& args = ctx.args;
  const int my_pe = ctx.my_pe;
  const int npes = ctx.npes;
  const int device_warp_size = ctx.device_warp_size;

  if (npes != 2) {
    if (my_pe == 0) {
      std::fprintf(stderr, "mori_shmem_put_latency requires exactly 2 PEs (npes=%d)\n", npes);
    }
    PerfFinalize(&ctx);
    return 1;
  }

  ShmemBarrierAll();
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  void* symm = ShmemMalloc(args.max_size);
  if (!symm) {
    std::fprintf(stderr, "ShmemMalloc failed\n");
    PerfFinalize(&ctx);
    return 1;
  }
  double* data_d = static_cast<double*>(symm);
  HIP_RUNTIME_CHECK(hipMemset(data_d, 0, args.max_size));

  hipEvent_t start{};
  hipEvent_t stop{};
  if (my_pe == 0) {
    HIP_RUNTIME_CHECK(hipEventCreate(&start));
    HIP_RUNTIME_CHECK(hipEventCreate(&stop));
  }

  const PutScope phase = args.put_scope;

  ShmemBarrierAll();
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  std::vector<PerfTableRow> lat_table;
  if (my_pe == 0) {
    lat_table.reserve(64);
  }

  for (size_t size_bytes = args.min_size; size_bytes <= args.max_size;
       size_bytes *= args.step_factor) {
    if (size_bytes % sizeof(double) != 0) {
      continue;
    }
    const size_t len_doubles = size_bytes / sizeof(double);
    if (!latency_size_ok(phase, len_doubles, args.threads_per_block, device_warp_size)) {
      if (my_pe == 0) {
        lat_table.push_back(PerfTableRow{size_bytes, true, 0.0});
      }
      ShmemBarrierAll();
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
      continue;
    }

    if (my_pe == 0) {
      launch_latency(phase, data_d, len_doubles, my_pe, static_cast<int>(args.warmup),
                     args.threads_per_block, device_warp_size);
      HIP_RUNTIME_CHECK(hipGetLastError());
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());

      HIP_RUNTIME_CHECK(hipEventRecord(start, nullptr));
      launch_latency(phase, data_d, len_doubles, my_pe, static_cast<int>(args.iters),
                     args.threads_per_block, device_warp_size);
      HIP_RUNTIME_CHECK(hipGetLastError());
      HIP_RUNTIME_CHECK(hipEventRecord(stop, nullptr));
      HIP_RUNTIME_CHECK(hipEventSynchronize(stop));

      float ms = 0.f;
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&ms, start, stop));
      const double latency_us = (static_cast<double>(ms) * static_cast<double>(kMsToUs)) /
                                static_cast<double>(args.iters);
      lat_table.push_back(PerfTableRow{size_bytes, false, latency_us});
    }

    ShmemBarrierAll();
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  }

  if (my_pe == 0) {
    int block_threads = 1;
    if (phase == PutScope::kWarp) {
      block_threads = device_warp_size;
    } else if (phase == PutScope::kBlock) {
      block_threads = args.threads_per_block;
    }
    PrintPerfTable("shmem_get_latency_uni", ScopeToChar(phase), 1, block_threads, device_warp_size,
                   args.iters, args.warmup, PerfTableMetric::kLatencyUs, lat_table);
  }
  ShmemBarrierAll();
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (my_pe == 0) {
    HIP_RUNTIME_CHECK(hipEventDestroy(start));
    HIP_RUNTIME_CHECK(hipEventDestroy(stop));
  }
  ShmemFree(symm);
  PerfFinalize(&ctx);
  return 0;
#endif
}

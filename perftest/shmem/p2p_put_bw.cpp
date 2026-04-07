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

#include "util.hpp"

#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::application;
using namespace mori::shmem;
using namespace mori::perftest;

namespace {

constexpr int kDefaultQpId = 1;
constexpr float kMsToS = 1000.0f;
constexpr double kBToGb = 1e9;


// NVSHMEM shmem_put_bw.cu: intra-block linear index; only !tid participates in cross-block atomics.
__device__ inline int bw_tid() {
  return threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
}

// After each timed/warmup iteration: all blocks rendezvous; counter_d[1] becomes i+1.
__device__ inline void bw_cross_block_barrier_round(volatile unsigned int* counter_d, int nblocks,
                                                    int i) {
  __syncthreads();
  const int tid = bw_tid();
  if (!tid) {
    __threadfence();
    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (i + 1) - 1)) {
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(i + 1)) {
    }
  }
  __syncthreads();
}

// After main loop: last global round bumps phase, quiet, then every block leader quiet again.
__device__ inline void bw_final_barrier_and_quiet(volatile unsigned int* counter_d, int nblocks,
                                                  int iter) {
  __syncthreads();
  const int tid = bw_tid();
  if (!tid) {
    __threadfence();
    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (iter + 1) - 1)) {
      ShmemQuietThread();
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(iter + 1)) {
    }
    ShmemQuietThread();
  }
  __syncthreads();
}

__global__ void bw_block(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                         int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int peer = !pe;

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * (len / static_cast<size_t>(nblocks));
    size_t chunk_doubles = len / static_cast<size_t>(nblocks);
    ShmemPutMemNbiBlock(slice, slice, chunk_doubles * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

__global__ void bw_warp(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                        int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int tid = bw_tid();
  int nwarps_per_block = (blockDim.x * blockDim.y * blockDim.z) / warpSize;
  int warpid = tid / warpSize;
  const int peer = !pe;

  size_t put_per_block = len / static_cast<size_t>(nblocks);
  size_t put_per_warp = put_per_block / static_cast<size_t>(nwarps_per_block);

  for (int i = 0; i < iter; i++) {
    double* slice =
        data_d + bid * put_per_block + static_cast<size_t>(warpid) * put_per_warp;
    ShmemPutMemNbiWarp(slice, slice, put_per_warp * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}

__global__ void bw_thread(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                          int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int tid = bw_tid();
  int nthreads = blockDim.x * blockDim.y * blockDim.z;
  const int peer = !pe;

  size_t put_per_block = len / static_cast<size_t>(nblocks);
  size_t put_per_thread = put_per_block / static_cast<size_t>(nthreads);

  for (int i = 0; i < iter; i++) {
    double* slice =
        data_d + bid * put_per_block + static_cast<size_t>(tid) * put_per_thread;
    ShmemPutMemNbiThread(slice, slice, put_per_thread * sizeof(double), peer, kDefaultQpId);

    bw_cross_block_barrier_round(counter_d, nblocks, i);
  }

  bw_final_barrier_and_quiet(counter_d, nblocks, iter);
}


bool size_ok(PutScope scope, size_t len_doubles, int nblocks, int threads_per_block,
             int device_warp_size) {
  if (len_doubles == 0 || (len_doubles % static_cast<size_t>(nblocks)) != 0) {
    return false;
  }
  size_t per_block = len_doubles / static_cast<size_t>(nblocks);
  if (scope == PutScope::kThread) {
    return per_block % static_cast<size_t>(threads_per_block) == 0;
  }
  if (scope == PutScope::kWarp) {
    if (threads_per_block % device_warp_size != 0) {
      return false;
    }
    int nw = threads_per_block / device_warp_size;
    if (nw <= 0) {
      return false;
    }
    return per_block % static_cast<size_t>(nw) == 0;
  }
  return true;
}

void launch_bw(PutScope scope, dim3 grid, dim3 block, double* data_d, unsigned int* counter_d,
               size_t len_doubles, int my_pe, int count) {
  switch (scope) {
    case PutScope::kBlock:
      hipLaunchKernelGGL(bw_block, grid, block, 0, 0, data_d, counter_d, len_doubles, my_pe,
                         count);
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

}  // namespace

int main(int argc, char** argv) {
#ifndef MORI_WITH_MPI
  std::fprintf(stderr, "mori_shmem_put_bw requires MORI_WITH_MPI (enable WITH_MPI / BUILD_EXAMPLES).\n");
  return 1;
#else

  MPI_Init(&argc, &argv);
  int mpi_world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);

  mori::perftest::PerfArgs args{};
  const int parse_rc = mori::perftest::ParseArgs(argc, argv, &args);
  if (parse_rc != 0) {
    if (mpi_world_rank == 0) {
      mori::perftest::PrintUsage(argv[0]);
    }
    MPI_Finalize();
    return parse_rc == 2 ? 0 : 1;
  }

  if (args.min_size > args.max_size || args.step_factor < 2 || args.nblocks < 1 ||
      args.threads_per_block < 1) {
    if (mpi_world_rank == 0) {
      std::fprintf(stderr, "Invalid arguments.\n");
    }
    MPI_Finalize();
    return 1;
  }

  PutScope scope = args.put_scope;

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank = 0;
  MPI_Comm_rank(local_comm, &local_rank);

  int device_count = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&device_count));
  int device_id = local_rank % device_count;
  HIP_RUNTIME_CHECK(hipSetDevice(device_id));

  int device_warp_size = 0;
  HIP_RUNTIME_CHECK(
      hipDeviceGetAttribute(&device_warp_size, hipDeviceAttributeWarpSize, device_id));

  int rc = ShmemMpiInit(MPI_COMM_WORLD);
  if (rc != 0) {
    std::fprintf(stderr, "ShmemMpiInit failed: %d\n", rc);
    MPI_Comm_free(&local_comm);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    return 1;
  }

  int my_pe = ShmemMyPe();
  int npes = ShmemNPes();
  if (npes != 2) {
    if (my_pe == 0) {
      std::fprintf(stderr, "mori_shmem_put_bw requires exactly 2 PEs (npes=%d)\n", npes);
    }
    MPI_Comm_free(&local_comm);
    ShmemFinalize();
    return 1;
  }

  const bool run_kernels = args.bidirectional || my_pe == 0;

  const char* scope_name = ScopeToChar(scope);

  dim3 grid(args.nblocks, 1, 1);
  dim3 block(args.threads_per_block, 1, 1);

  unsigned int* counter_d = nullptr;
  hipEvent_t start{};
  hipEvent_t stop{};
  if (run_kernels) {
    HIP_RUNTIME_CHECK(hipMalloc(&counter_d, 2 * sizeof(unsigned int)));
    HIP_RUNTIME_CHECK(hipMemset(counter_d, 0, 2 * sizeof(unsigned int)));
    HIP_RUNTIME_CHECK(hipEventCreate(&start));
    HIP_RUNTIME_CHECK(hipEventCreate(&stop));
  }

  void* symm = ShmemMalloc(args.max_size);
  if (!symm) {
    std::fprintf(stderr, "ShmemMalloc failed\n");
    if (run_kernels) {
      HIP_RUNTIME_CHECK(hipFree(counter_d));
    }
    MPI_Comm_free(&local_comm);
    ShmemFinalize();
    return 1;
  }
  double* data_d = static_cast<double*>(symm);
  HIP_RUNTIME_CHECK(hipMemset(data_d, 0, args.max_size));

  std::vector<BandwidthSample> bandwidth_table;
  if (my_pe == 0) {
    bandwidth_table.reserve(64);
  }

  for (size_t size_bytes = args.min_size; size_bytes <= args.max_size;
       size_bytes *= args.step_factor) {
    if (size_bytes % sizeof(double) != 0) {
      continue;
    }
    size_t len_doubles = size_bytes / sizeof(double);
    if (!size_ok(scope, len_doubles, args.nblocks, args.threads_per_block, device_warp_size)) {
      if (my_pe == 0) {
        bandwidth_table.push_back(BandwidthSample{size_bytes, true, 0.0});
      }
      ShmemBarrierAll();
      continue;
    }

    if (run_kernels) {
      HIP_RUNTIME_CHECK(hipMemset(counter_d, 0, 2 * sizeof(unsigned int)));
      launch_bw(scope, grid, block, data_d, counter_d, len_doubles, my_pe,
                static_cast<int>(args.warmup));
      HIP_RUNTIME_CHECK(hipGetLastError());
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());

      HIP_RUNTIME_CHECK(hipMemset(counter_d, 0, 2 * sizeof(unsigned int)));
      HIP_RUNTIME_CHECK(hipEventRecord(start, nullptr));
      launch_bw(scope, grid, block, data_d, counter_d, len_doubles, my_pe,
                static_cast<int>(args.iters));
      HIP_RUNTIME_CHECK(hipGetLastError());
      HIP_RUNTIME_CHECK(hipEventRecord(stop, nullptr));
      HIP_RUNTIME_CHECK(hipEventSynchronize(stop));

      float ms = 0.f;
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&ms, start, stop));

      const double gbps = static_cast<double>(size_bytes) /
                          (static_cast<double>(ms) * (kBToGb / (args.iters * kMsToS)));

      if (args.bidirectional) {
        double gbps_local = gbps;
        double gbps_sum = 0.0;
        MPI_Reduce(&gbps_local, &gbps_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (my_pe == 0) {
          bandwidth_table.push_back(BandwidthSample{size_bytes, false, gbps_sum});
        }
      } else {
        if (my_pe == 0) {
          bandwidth_table.push_back(BandwidthSample{size_bytes, false, gbps});
        }
      }
    }

    ShmemBarrierAll();
  }

  ShmemBarrierAll();
  if (my_pe == 0) {
    const char* test_name = args.bidirectional ? "shmem_put_bw_bidi" : "shmem_put_bw_uni";
    PrintTable(test_name, scope_name, args.nblocks, args.threads_per_block, device_warp_size,
               args.iters, args.warmup, bandwidth_table);
  }

  if (run_kernels) {
    HIP_RUNTIME_CHECK(hipEventDestroy(start));
    HIP_RUNTIME_CHECK(hipEventDestroy(stop));
    HIP_RUNTIME_CHECK(hipFree(counter_d));
  }
  ShmemFree(symm);
  MPI_Comm_free(&local_comm);
  ShmemFinalize();
  return 0;
#endif
}

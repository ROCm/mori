// Mori shmem device put bandwidth microbenchmark (modeled on NVSHMEM shmem_put_bw.cu).
// Requires: 2 MPI ranks, MORI_WITH_MPI, HIP.
//
// Run (from build dir): mpirun -np 2 ./perftest/p2p_put_bw [opts]
//   or from build/perftest: mpirun -np 2 ./p2p_put_bw [opts]
// Options: -b -e -f -n -w -c -t -s thread|warp|block
// Note: MPI_Comm_free(local_comm) must run before ShmemFinalize() (Mori calls MPI_Finalize inside).

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::application;
using namespace mori::shmem;

namespace {

constexpr int kDefaultQpId = 1;
constexpr float kMsToS = 1000.0f;
constexpr double kBToGb = 1e9;

enum class PutScope { kThread, kWarp, kBlock };

__device__ inline bool block_leader_1d() { return threadIdx.x == 0; }

__global__ void bw_block(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                         int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  const int peer = 1 - pe;

  for (int i = 0; i < iter; i++) {
    double* slice = data_d + bid * (len / static_cast<size_t>(nblocks));
    size_t chunk_doubles = len / static_cast<size_t>(nblocks);
    if (block_leader_1d()) {
      ShmemPutMemNbiBlock(slice, slice, chunk_doubles * sizeof(double), peer, kDefaultQpId);
    }

    __syncthreads();
    if (block_leader_1d()) {
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

  __syncthreads();
  if (block_leader_1d()) {
    __threadfence();
    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (iter + 1) - 1)) {
      ShmemQuietThread(peer);
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(iter + 1)) {
    }
    ShmemQuietThread(peer);
  }
  __syncthreads();
}

__global__ void bw_warp(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                        int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  int nwarps_per_block = (blockDim.x * blockDim.y * blockDim.z) / warpSize;
  int warpid = threadIdx.x / warpSize;
  const int peer = 1 - pe;

  size_t put_per_block = len / static_cast<size_t>(nblocks);
  size_t put_per_warp = put_per_block / static_cast<size_t>(nwarps_per_block);

  for (int i = 0; i < iter; i++) {
    double* slice =
        data_d + bid * put_per_block + static_cast<size_t>(warpid) * put_per_warp;
    if ((threadIdx.x % warpSize) == 0) {
      ShmemPutMemNbiWarp(slice, slice, put_per_warp * sizeof(double), peer, kDefaultQpId);
    }

    __syncthreads();
    if (block_leader_1d()) {
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

  __syncthreads();
  if (block_leader_1d()) {
    __threadfence();
    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (iter + 1) - 1)) {
      ShmemQuietThread(peer);
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(iter + 1)) {
    }
    ShmemQuietThread(peer);
  }
  __syncthreads();
}

__global__ void bw_thread(double* data_d, volatile unsigned int* counter_d, size_t len, int pe,
                          int iter) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  int nthreads = blockDim.x * blockDim.y * blockDim.z;
  const int peer = 1 - pe;

  size_t put_per_block = len / static_cast<size_t>(nblocks);
  size_t put_per_thread = put_per_block / static_cast<size_t>(nthreads);

  for (int i = 0; i < iter; i++) {
    double* slice =
        data_d + bid * put_per_block + static_cast<size_t>(threadIdx.x) * put_per_thread;
    ShmemPutMemNbiThread(slice, slice, put_per_thread * sizeof(double), peer, kDefaultQpId);

    __syncthreads();
    if (block_leader_1d()) {
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

  __syncthreads();
  if (block_leader_1d()) {
    __threadfence();
    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (iter + 1) - 1)) {
      ShmemQuietThread(peer);
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(iter + 1)) {
    }
    ShmemQuietThread(peer);
  }
  __syncthreads();
}

void print_usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [-b min_bytes] [-e max_bytes] [-f step_factor] [-n iters] [-w warmup] "
               "[-c grid_x] [-t threads_per_block] [-s thread|warp|block]\n",
               argv0);
}

bool parse_scope(const char* s, PutScope* out) {
  if (std::strcmp(s, "thread") == 0) {
    *out = PutScope::kThread;
    return true;
  }
  if (std::strcmp(s, "warp") == 0) {
    *out = PutScope::kWarp;
    return true;
  }
  if (std::strcmp(s, "block") == 0) {
    *out = PutScope::kBlock;
    return true;
  }
  return false;
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
  size_t min_size = 4;
  size_t max_size = 1024 * 1024 * 1024;
  size_t step_factor = 2;
  size_t iters = 10;
  size_t warmup = 5;
  int nblocks = 32;
  int threads_per_block = 256;
  PutScope scope = PutScope::kBlock;

  int opt = 0;
  while ((opt = getopt(argc, argv, "hb:e:f:n:w:c:t:s:")) != -1) {
    switch (opt) {
      case 'h':
        print_usage(argv[0]);
        return 0;
      case 'b':
        min_size = static_cast<size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'e':
        max_size = static_cast<size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'f':
        step_factor = static_cast<size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'n':
        iters = static_cast<size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'w':
        warmup = static_cast<size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'c':
        nblocks = std::atoi(optarg);
        break;
      case 't':
        threads_per_block = std::atoi(optarg);
        break;
      case 's':
        if (!parse_scope(optarg, &scope)) {
          std::fprintf(stderr, "Invalid -s (use thread|warp|block)\n");
          return 1;
        }
        break;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  if (min_size > max_size || step_factor < 2 || nblocks < 1 || threads_per_block < 1) {
    std::fprintf(stderr, "Invalid arguments.\n");
    return 1;
  }

  MPI_Init(&argc, &argv);

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

  const char* scope_name = "block";
  switch (scope) {
    case PutScope::kThread:
      scope_name = "thread";
      break;
    case PutScope::kWarp:
      scope_name = "warp";
      break;
    case PutScope::kBlock:
      scope_name = "block";
      break;
  }

  dim3 grid(nblocks, 1, 1);
  dim3 block(threads_per_block, 1, 1);

  unsigned int* counter_d = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&counter_d, 2 * sizeof(unsigned int)));
  HIP_RUNTIME_CHECK(hipMemset(counter_d, 0, 2 * sizeof(unsigned int)));

  void* symm = ShmemMalloc(max_size);
  if (!symm) {
    std::fprintf(stderr, "ShmemMalloc failed\n");
    HIP_RUNTIME_CHECK(hipFree(counter_d));
    MPI_Comm_free(&local_comm);
    ShmemFinalize();
    return 1;
  }
  double* data_d = static_cast<double*>(symm);
  HIP_RUNTIME_CHECK(hipMemset(data_d, 0, max_size));

  hipEvent_t start{}, stop{};
  HIP_RUNTIME_CHECK(hipEventCreate(&start));
  HIP_RUNTIME_CHECK(hipEventCreate(&stop));

  if (my_pe == 0) {
    std::printf("# mori_shmem_put_bw_uni scope=%s grid=%d block=%d warpSize=%d iters=%zu warmup=%zu\n",
                scope_name, nblocks, threads_per_block, device_warp_size, iters, warmup);
    std::printf("# size(B)\tBW(GB/s)\n");
  }

  for (size_t size_bytes = min_size; size_bytes <= max_size; size_bytes *= step_factor) {
    if (size_bytes % sizeof(double) != 0) {
      continue;
    }
    size_t len_doubles = size_bytes / sizeof(double);
    if (!size_ok(scope, len_doubles, nblocks, threads_per_block, device_warp_size)) {
      if (my_pe == 0) {
        std::printf("# skip size=%zu (not divisible for grid/scope)\n", size_bytes);
      }
      ShmemBarrierAll();
      continue;
    }

    HIP_RUNTIME_CHECK(hipMemset(counter_d, 0, 2 * sizeof(unsigned int)));
    launch_bw(scope, grid, block, data_d, counter_d, len_doubles, my_pe, static_cast<int>(warmup));
    HIP_RUNTIME_CHECK(hipGetLastError());
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    ShmemBarrierAll();

    HIP_RUNTIME_CHECK(hipMemset(counter_d, 0, 2 * sizeof(unsigned int)));
    HIP_RUNTIME_CHECK(hipEventRecord(start, nullptr));
    launch_bw(scope, grid, block, data_d, counter_d, len_doubles, my_pe, static_cast<int>(iters));
    HIP_RUNTIME_CHECK(hipGetLastError());
    HIP_RUNTIME_CHECK(hipEventRecord(stop, nullptr));
    HIP_RUNTIME_CHECK(hipEventSynchronize(stop));

    float ms = 0.f;
    HIP_RUNTIME_CHECK(hipEventElapsedTime(&ms, start, stop));
    ShmemBarrierAll();

    if (my_pe == 0) {
      double gbps =
          static_cast<double>(size_bytes) / (static_cast<double>(ms) * (kBToGb / (iters * kMsToS)));
      std::printf("%zu\t%.6f\n", size_bytes, gbps);
    }
  }

  HIP_RUNTIME_CHECK(hipEventDestroy(start));
  HIP_RUNTIME_CHECK(hipEventDestroy(stop));
  HIP_RUNTIME_CHECK(hipFree(counter_d));
  ShmemFree(symm);
  MPI_Comm_free(&local_comm);
  ShmemFinalize();
  return 0;
#endif
}

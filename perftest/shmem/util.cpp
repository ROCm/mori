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
#include "util.hpp"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem_api.hpp"

namespace mori::shmem::perftest {

void PrintUsage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s [options]\n"
      "  -b min_bytes   minimum message size\n"
      "  -e max_bytes   maximum message size\n"
      "  -f step        multiply size by this factor each step\n"
      "  -n iters       timed iterations\n"
      "  -w warmup      warmup iterations\n"
      "  -c grid_x      CUDA/HIP grid x (blocks)\n"
      "  -t threads     threads per block\n"
      "  -s scope       thread | warp | block (put_bw/ring: default block; put_latency: block)\n"
      "  -q num_qps     number of QPs to use concurrently (default 1, max MORI_NUM_QP_PER_PE)\n"
      "  -B             bidirectional (p2p_put_bw only; ignored elsewhere)\n"
      "  -h             this help\n",
      program != nullptr ? program : "program");
}

int ParseArgs(int argc, char** argv, PerfArgs* out_args) {
  if (out_args == nullptr) {
    return 1;
  }

  *out_args = PerfArgs{};

  int opt = 0;
  while ((opt = getopt(argc, argv, "hBb:e:f:n:w:c:t:s:q:")) != -1) {
    switch (opt) {
      case 'h':
        return 2;
      case 'B':
        out_args->bidirectional = true;
        break;
      case 'b':
        out_args->min_size = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'e':
        out_args->max_size = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'f':
        out_args->step_factor = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'n':
        out_args->iters = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'w':
        out_args->warmup = static_cast<std::size_t>(std::strtoul(optarg, nullptr, 0));
        break;
      case 'c':
        out_args->nblocks = std::atoi(optarg);
        break;
      case 't':
        out_args->threads_per_block = std::atoi(optarg);
        break;
      case 's':
        if (std::strcmp(optarg, "thread") == 0) {
          out_args->put_scope = PutScope::kThread;
        } else if (std::strcmp(optarg, "warp") == 0) {
          out_args->put_scope = PutScope::kWarp;
        } else if (std::strcmp(optarg, "block") == 0) {
          out_args->put_scope = PutScope::kBlock;
        } else {
          return 1;
        }
        break;
      case 'q':
        out_args->num_qps = std::atoi(optarg);
        break;
      default:
        return 1;
    }
  }

  return 0;
}

static std::string fmt_size(std::size_t bytes) {
  if (bytes >= (1ULL << 30)) return std::to_string(bytes >> 30) + " GB";
  if (bytes >= (1ULL << 20)) return std::to_string(bytes >> 20) + " MB";
  if (bytes >= (1ULL << 10)) return std::to_string(bytes >> 10) + " KB";
  return std::to_string(bytes) + " B";
}

void PrintPerfTable(const char* test_name, const char* scope_name, int grid_x, int block_threads,
                    int warp_size, std::size_t iters, std::size_t warmup, PerfTableMetric metric,
                    const std::vector<PerfTableRow>& rows, int num_qps) {
  const char* scope_col = (scope_name != nullptr && scope_name[0] != '\0') ? scope_name : "None";
  const char* default_tag =
      (metric == PerfTableMetric::kBandwidthGbps) ? "p2p_put_bw" : "p2p_put_latency";
  const char* tag = (test_name != nullptr && test_name[0] != '\0') ? test_name : default_tag;

  std::printf("# %s scope=%s grid=%d block=%d warpSize=%d iters=%zu warmup=%zu qps=%d\n", tag,
              scope_col, grid_x, block_threads, warp_size, iters, warmup, num_qps);

  constexpr int kWSize = 10;
  constexpr int kWScope = 8;
  constexpr int kVal = 16;

  const bool is_bw = (metric == PerfTableMetric::kBandwidthGbps);
  const char* val_header = is_bw ? "Bandwidth (GB/s)" : "latency";

  std::printf("%-*s %-*s %*s\n", kWSize, "size", kWScope, "scope", kVal, val_header);

  for (const PerfTableRow& r : rows) {
    std::string sz = fmt_size(r.size_bytes);
    if (r.skipped) {
      std::printf("%-*s %-*s %*s\n", kWSize, sz.c_str(), kWScope, scope_col, kVal, "skip");
    } else if (is_bw) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.3f GB/s", r.value);
      std::printf("%-*s %-*s %*s\n", kWSize, sz.c_str(), kWScope, scope_col, kVal, buf);
    } else {
      // Auto-scale latency: use ms when >= 1000 µs, else µs.
      char buf[32];
      if (r.value >= 1000.0) {
        std::snprintf(buf, sizeof(buf), "%.3f ms", r.value / 1000.0);
      } else {
        std::snprintf(buf, sizeof(buf), "%.3f µs", r.value);
      }
      std::printf("%-*s %-*s %*s\n", kWSize, sz.c_str(), kWScope, scope_col, kVal, buf);
    }
  }
  std::fflush(stdout);
}

int PerfInit(int argc, char** argv, struct PerfContext* ctx) {
  int rc;
  int finalized = 0;

  memset(ctx, 0, sizeof(struct PerfContext));
  ctx->args = PerfArgs{};
  PerfArgs& args = ctx->args;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx->world_rank);

  rc = ParseArgs(argc, argv, &args);
  if (rc) {
    if (ctx->world_rank == 0) {
      PrintUsage(argv[0]);
    }
    MPI_Finalize();
    return rc == 2 ? 0 : 1;
  }

  if (args.min_size > args.max_size || args.step_factor < 2 || args.iters < 1 || args.nblocks < 1 ||
      args.threads_per_block < 1 || args.num_qps < 1) {
    if (ctx->world_rank == 0) {
      std::fprintf(stderr, "Invalid arguments (need iters >= 1, nblocks/threads/qps >= 1).\n");
    }
    MPI_Finalize();
    return 1;
  }

  if (args.min_size % sizeof(double) != 0) {
    args.min_size = (args.min_size + sizeof(double) - 1) / sizeof(double) * sizeof(double);
  }

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &ctx->local_comm);

  MPI_Comm_rank(ctx->local_comm, &ctx->local_rank);

  HIP_RUNTIME_CHECK(hipGetDeviceCount(&ctx->device_count));

  assert(ctx->device_count);

  const int device_id = ctx->local_rank % ctx->device_count;
  HIP_RUNTIME_CHECK(hipSetDevice(device_id));

  HIP_RUNTIME_CHECK(
      hipDeviceGetAttribute(&ctx->device_warp_size, hipDeviceAttributeWarpSize, device_id));

  rc = ShmemMpiInit(MPI_COMM_WORLD);
  if (rc) {
    std::fprintf(stderr, "ShmemMpiInit failed: %d\n", rc);
    MPI_Comm_free(&ctx->local_comm);
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    return 1;
  }

  ctx->my_pe = ShmemMyPe();
  ctx->npes = ShmemNPes();
  return 0;
}

void PerfFinalize(struct PerfContext* ctx) {
  MPI_Comm_free(&ctx->local_comm);
  ShmemFinalize();
}

void PerfResAlloc(PerfRes* res) {
  HIP_RUNTIME_CHECK(hipMalloc(&res->counter_d, 2 * sizeof(unsigned int)));
  HIP_RUNTIME_CHECK(hipEventCreate(&res->start));
  HIP_RUNTIME_CHECK(hipEventCreate(&res->stop));
}

void PerfResFree(PerfRes* res) {
  HIP_RUNTIME_CHECK(hipEventDestroy(res->start));
  HIP_RUNTIME_CHECK(hipEventDestroy(res->stop));
  HIP_RUNTIME_CHECK(hipFree(res->counter_d));
}

float RunWarmupAndTimed(PerfRes& res, size_t warmup, size_t iters, LaunchFn launch) {
  HIP_RUNTIME_CHECK(hipMemset(res.counter_d, 0, 2 * sizeof(unsigned int)));
  launch(static_cast<int>(warmup));
  HIP_RUNTIME_CHECK(hipGetLastError());
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  HIP_RUNTIME_CHECK(hipMemset(res.counter_d, 0, 2 * sizeof(unsigned int)));
  HIP_RUNTIME_CHECK(hipEventRecord(res.start, nullptr));
  launch(static_cast<int>(iters));
  HIP_RUNTIME_CHECK(hipGetLastError());
  HIP_RUNTIME_CHECK(hipEventRecord(res.stop, nullptr));
  HIP_RUNTIME_CHECK(hipEventSynchronize(res.stop));

  float ms = 0.f;
  HIP_RUNTIME_CHECK(hipEventElapsedTime(&ms, res.start, res.stop));
  return ms;
}

}  // namespace mori::shmem::perftest

#include "util.hpp"

#include <cstdlib>
#include <cstring>
#include <functional>

#include "hip/hip_runtime.h"
#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem_api.hpp"

using namespace mori::shmem;

namespace mori::perftest {

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
      "  -B             bidirectional (p2p_put_bw only; ignored elsewhere)\n"
      "  -h             this help\n"
      "  ring_put_bw: N-PE ring; same -c/-t/-s as p2p_put_bw, peer=(pe+1)%%np.\n",
      program != nullptr ? program : "program");
}

int ParseArgs(int argc, char** argv, PerfArgs* out_args) {
  if (out_args == nullptr) {
    return 1;
  }

  *out_args = PerfArgs{};

  int opt = 0;
  while ((opt = getopt(argc, argv, "hBb:e:f:n:w:c:t:s:")) != -1) {
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
      default:
        return 1;
    }
  }

  return 0;
}

void PrintPerfTable(const char* test_name, const char* scope_name, int grid_x, int block_threads,
                    int warp_size, std::size_t iters, std::size_t warmup, PerfTableMetric metric,
                    const std::vector<PerfTableRow>& rows) {
  const char* scope_col = (scope_name != nullptr && scope_name[0] != '\0') ? scope_name : "None";
  const char* default_tag =
      (metric == PerfTableMetric::kBandwidthGbps) ? "shmem_put_bw" : "shmem_put_latency";
  const char* tag = (test_name != nullptr && test_name[0] != '\0') ? test_name : default_tag;

  std::printf("# %s scope=%s grid=%d block=%d warpSize=%d iters=%zu warmup=%zu\n", tag, scope_col,
              grid_x, block_threads, warp_size, iters, warmup);

  constexpr int kWSize = 14;
  constexpr int kWScope = 12;
  constexpr int kVal = 18;
  const char* val_header =
      (metric == PerfTableMetric::kBandwidthGbps) ? "Bandwidth (GB)" : "latency(us)";
  const int prec = 6;

  std::printf("%-*s %-*s %-*s\n", kWSize, "size(B)", kWScope, "scope", kVal, val_header);
  for (const PerfTableRow& r : rows) {
    if (r.skipped) {
      std::printf("%-*zu %-*s %-*s\n", kWSize, r.size_bytes, kWScope, scope_col, kVal, "skip");
    } else {
      std::printf("%-*zu %-*s %-*.*f\n", kWSize, r.size_bytes, kWScope, scope_col, kVal, prec,
                  r.value);
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
      args.threads_per_block < 1) {
    if (ctx->world_rank == 0) {
      std::fprintf(stderr, "Invalid arguments (need iters >= 1, nblocks/threads >= 1).\n");
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

}  // namespace mori::perftest

#include "util.hpp"

#include <cstdlib>
#include <cstring>

namespace mori::perftest {

void PrintUsage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "  -b min_bytes   minimum message size\n"
               "  -e max_bytes   maximum message size\n"
               "  -f step        multiply size by this factor each step\n"
               "  -n iters       timed iterations\n"
               "  -w warmup      warmup iterations\n"
               "  -c grid_x      CUDA/HIP grid x (blocks)\n"
               "  -t threads     threads per block\n"
               "  -s scope       thread | warp | block (put_bw: which put; put_latency: default block)\n"
               "  -B             bidirectional (p2p_put_bw only; ignored by p2p_put_latency)\n"
               "  -h             this help\n",
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
  const char* scope_col =
      (scope_name != nullptr && scope_name[0] != '\0') ? scope_name : "None";
  const char* default_tag = (metric == PerfTableMetric::kBandwidthGbps) ? "shmem_put_bw"
                                                                        : "shmem_put_latency";
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

}  // namespace mori::perftest

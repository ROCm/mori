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
               "  -s scope       thread | warp | block\n"
               "  -B             bidirectional (both PEs run kernels; BW summed on PE 0)\n"
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

void PrintTable(const char* test_name, const char* scope_name, int nblocks, int threads_per_block,
                int warp_size, std::size_t iters, std::size_t warmup,
                const std::vector<BandwidthSample>& rows) {
  const char* scope_col = scope_name;
  const char* tag = (test_name != nullptr && test_name[0] != '\0') ? test_name : "shmem_put_bw";

  std::printf("# %s scope=%s grid=%d block=%d warpSize=%d iters=%zu warmup=%zu\n", tag, scope_col,
              nblocks, threads_per_block, warp_size, iters, warmup);

  // Fixed widths; size / bandwidth numeric column right-aligned (bulk put_bw: no MPPS like scalar p_bw).
  constexpr int kWSize = 14;
  constexpr int kWScope = 12;
  constexpr int kBandwidth = 18;
  std::printf("%-*s %-*s %-*s\n", kWSize, "size(B)", kWScope, "scope", kBandwidth, "Bandwidth (GB)");
  for (const BandwidthSample& r : rows) {
    if (r.skipped) {
      std::printf("%-*zu %-*s %-*s\n", kWSize, r.size_bytes, kWScope, scope_col, kBandwidth, "skip");
    } else {
      std::printf("%-*zu %-*s %-*.6f\n", kWSize, r.size_bytes, kWScope, scope_col, kBandwidth,
                  r.gbps);
    }
  }
  std::fflush(stdout);
}

}  // namespace mori::perftest


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

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <unistd.h>

namespace mori::perftest {

enum class PutScope { kThread, kWarp, kBlock };

inline constexpr std::size_t kDefaultMinSize = 4;
inline constexpr std::size_t kDefaultMaxSize = 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kDefaultStepFactor = 2;
inline constexpr std::size_t kDefaultIters = 10;
inline constexpr std::size_t kDefaultWarmup = 5;
inline constexpr int kDefaultNumBlocks = 32;
inline constexpr int kDefaultThreadsPerBlock = 256;

struct PerfArgs {
  std::size_t min_size = kDefaultMinSize;
  std::size_t max_size = kDefaultMaxSize;
  std::size_t step_factor = kDefaultStepFactor;
  std::size_t iters = kDefaultIters;
  std::size_t warmup = kDefaultWarmup;
  int nblocks = kDefaultNumBlocks;
  int threads_per_block = kDefaultThreadsPerBlock;
  PutScope put_scope = PutScope::kBlock;
  bool bidirectional = false;
};

// Returns 0 on success, 1 on invalid option, 2 if -h was passed (exit main with 0).
// Does not print; caller should print help (e.g. only MPI rank 0).
int ParseArgs(int argc, char** argv, PerfArgs* out_args);

void PrintUsage(const char* program);

struct BandwidthSample {
  std::size_t size_bytes{};
  bool skipped{};  // if true, print skip line instead of gbps
  double gbps{}; // no need to print mpps, because we don't put 1/4/8 bytes at a time
};

// PE 0 only. test_name: e.g. shmem_put_bw_uni / shmem_put_bw_bidi (NVSHMEM device put_bw).
void PrintTable(const char* test_name, const char* scope_name, int nblocks, int threads_per_block,
                int warp_size, std::size_t iters, std::size_t warmup,
                const std::vector<BandwidthSample>& rows);


inline const char* ScopeToChar(PutScope scope) {
  switch (scope) {
    case PutScope::kThread:
      return "thread";
    case PutScope::kWarp:
      return "warp";
    case PutScope::kBlock:
      return "block";
    default:
      printf("Invalid scope: %d\n", scope);
      assert(0);
  }
  return "None";
}

}  // namespace mori::perftest

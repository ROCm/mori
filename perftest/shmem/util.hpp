
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

#include <mpi.h>
#include <unistd.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "hip/hip_runtime.h"

namespace mori::perftest {

enum class PutScope { kThread, kWarp, kBlock };

inline constexpr std::size_t kDefaultMinSize = 4;
inline constexpr std::size_t kDefaultMaxSize = 64ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kDefaultStepFactor = 2;
inline constexpr std::size_t kDefaultIters = 10;
inline constexpr std::size_t kDefaultWarmup = 5;
inline constexpr int kDefaultNumBlocks = 32;
inline constexpr int kDefaultThreadsPerBlock = 256;

inline constexpr int kDefaultQpId = 1;
inline constexpr float kMsToS = 1000.0f;
inline constexpr float kMsToUs = 1000.0f;
inline constexpr double kBToGb = 1e9;

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

struct PerfContext {
  MPI_Comm local_comm;
  int local_rank;
  int world_rank;
  PerfArgs args;
  int device_count;
  int device_warp_size;
  int my_pe;
  int npes;
};

int PerfInit(int argc, char** argv, struct PerfContext* ctx);
void PerfFinalize(struct PerfContext* ctx);

using LaunchFn = std::function<void(int /*count*/)>;

struct PerfRes {
  hipEvent_t start{};
  hipEvent_t stop{};
  unsigned int* counter_d = nullptr;
};

void PerfResAlloc(PerfRes* res);
void PerfResFree(PerfRes* res);
float RunWarmupAndTimed(PerfRes& res, size_t warmup, size_t iters, LaunchFn launch);

int ParseArgs(int argc, char** argv, PerfArgs* out_args);
void PrintUsage(const char* program);

enum class PerfTableMetric { kBandwidthGbps, kLatencyUs };

// One row: value is GB/s (put_bw) or µs/iter (put_latency) depending on metric.
struct PerfTableRow {
  std::size_t size_bytes{};
  bool skipped{};
  double value{};
};

// PE 0 only. test_name e.g. shmem_put_bw_uni / shmem_put_latency_uni; value column from metric.
void PrintPerfTable(const char* test_name, const char* scope_name, int grid_x, int block_threads,
                    int warp_size, std::size_t iters, std::size_t warmup, PerfTableMetric metric,
                    const std::vector<PerfTableRow>& rows);

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

inline bool size_ok(PutScope scope, size_t size_bytes, int nblocks, int threads_per_block,
                    int device_warp_size) {
  if (size_bytes == 0 || size_bytes % sizeof(double) != 0) {
    return false;
  }
  const size_t len = size_bytes / sizeof(double);
  if (len % static_cast<size_t>(nblocks) != 0) {
    return false;
  }
  const size_t per_block = len / static_cast<size_t>(nblocks);
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

}  // namespace mori::perftest

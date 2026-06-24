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
// Host control-plane types only (ccoComm / ccoDevComm / ccoWindow_t). The GDA
// device layer (ccoGda) lives in cco_scale_out.hpp and is included directly by
// the kernel TUs that need it — util.cpp is host CXX and must not pull it in.
#include "mori/cco/cco.hpp"

namespace mori::cco::benchmark {

// Cooperation granularity of the kernel transfer loop / RDMA op.
// kThreadAgg: thread scope with ccoGdaThreadAggregate (same-peer lanes post as one
// batch instead of per-peer grouping). Bandwidth kernels only; latency uses thread.
enum class PutScope { kThread, kWarp, kBlock, kThreadAgg };

// Which CCO p2p transport to exercise. Selected by the MORI_DISABLE_P2P env
// var (consistent with the rest of mori): unset or enabled → kIbgda (P2P
// disabled, use RDMA); explicitly disabled (0/false/off/no) → kLsa.
//   kLsa   — intra-node flat-VA load/store (no NIC), requires same-node peer.
//   kIbgda — cross-node one-sided RDMA via ccoGda<PrvdType>.
enum class Transport { kLsa, kIbgda };

inline constexpr std::size_t kDefaultMinSize = 8;
inline constexpr std::size_t kDefaultMaxSize = 64ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kDefaultStepFactor = 2;
inline constexpr std::size_t kDefaultIters = 10;
inline constexpr std::size_t kDefaultWarmup = 5;
inline constexpr int kDefaultNumBlocks = 32;
inline constexpr int kDefaultThreadsPerBlock = 256;

inline constexpr float kMsToS = 1000.0f;
inline constexpr float kMsToUs = 1000.0f;
inline constexpr double kBToGb = 1e9;

// Per-rank VMM reservation for the CCO flat address space. 0 lets CCO pick a
// default sized from the registered windows; we pass an explicit size so two
// max_size windows always fit.
inline constexpr std::size_t kVmmSlack = 64ULL * 1024ULL * 1024ULL;

struct PerfArgs {
  std::size_t min_size = kDefaultMinSize;
  std::size_t max_size = kDefaultMaxSize;
  std::size_t step_factor = kDefaultStepFactor;
  std::size_t iters = kDefaultIters;
  std::size_t warmup = kDefaultWarmup;
  int nblocks = kDefaultNumBlocks;
  int threads_per_block = kDefaultThreadsPerBlock;
  PutScope put_scope = PutScope::kBlock;
  // Default IBGDA; overridden in PerfInit from MORI_DISABLE_P2P.
  Transport transport = Transport::kIbgda;
};

// Holds MPI + CCO state for the lifetime of a benchmark run. PerfInit registers
// two windows (send/recv) of max_size and a DevComm matching the transport.
struct PerfContext {
  // MPI / topology.
  int world_rank = 0;
  int local_rank = 0;
  MPI_Comm local_comm = MPI_COMM_NULL;
  int device_count = 0;
  int device_warp_size = 0;
  int my_pe = 0;  // CCO rank
  int npes = 0;   // CCO world size

  PerfArgs args;

  // CCO handles (owned; released by PerfFinalize).
  ccoComm* comm = nullptr;
  ccoDevComm devComm{};
  ccoWindow_t send_win = nullptr;
  ccoWindow_t recv_win = nullptr;
  void* send_buf = nullptr;  // local pointer into send window
  void* recv_buf = nullptr;  // local pointer into recv window

  // Peer's LSA rank (the other PE on the node). Valid only when lsaSize >= 2.
  int peer_lsa_rank = 0;
};

// Returns 0 on success, 1 on bad args / setup failure, 2 when help was shown
// (caller should exit 0 without calling PerfFinalize).
int PerfInit(int argc, char** argv, PerfContext* ctx);
void PerfFinalize(PerfContext* ctx);

using LaunchFn = std::function<void(int /*count*/)>;

struct PerfRes {
  hipEvent_t start{};
  hipEvent_t stop{};
  // [0]=arrival counter, [1]=phase counter for the LSA bw cross-block barrier
  // (apples-to-apples with shmem). Zeroed before each warmup/timed launch.
  unsigned int* counter_d = nullptr;
};

void PerfResAlloc(PerfRes* res);
void PerfResFree(PerfRes* res);
float RunWarmupAndTimed(PerfRes& res, std::size_t warmup, std::size_t iters, LaunchFn launch);

int ParseArgs(int argc, char** argv, PerfArgs* out_args);
void PrintUsage(const char* program);

enum class PerfTableMetric { kBandwidthGbps, kLatencyUs };

struct PerfTableRow {
  std::size_t size_bytes{};
  bool skipped{};
  double value{};
};

void PrintPerfTable(const char* test_name, const char* transport_name, const char* scope_name,
                    int grid_x, int block_threads, int warp_size, std::size_t iters,
                    std::size_t warmup, PerfTableMetric metric,
                    const std::vector<PerfTableRow>& rows);

inline const char* ScopeToChar(PutScope scope) {
  switch (scope) {
    case PutScope::kThread:
      return "thread";
    case PutScope::kWarp:
      return "warp";
    case PutScope::kBlock:
      return "block";
    case PutScope::kThreadAgg:
      return "thread_agg";
  }
  return "none";
}

inline const char* TransportToChar(Transport t) {
  switch (t) {
    case Transport::kLsa:
      return "lsa";
    case Transport::kIbgda:
      return "ibgda";
  }
  return "none";
}

// Block-scope bandwidth kernels split the buffer into nblocks chunks, and warp/
// thread scope split each chunk further; require even divisibility so every
// lane gets equal work.
inline bool size_ok(PutScope scope, std::size_t size_bytes, int nblocks, int threads_per_block,
                    int device_warp_size) {
  if (size_bytes == 0 || size_bytes % sizeof(double) != 0) {
    return false;
  }
  const std::size_t len = size_bytes / sizeof(double);
  if (len % static_cast<std::size_t>(nblocks) != 0) {
    return false;
  }
  const std::size_t per_block = len / static_cast<std::size_t>(nblocks);
  if (scope == PutScope::kThread || scope == PutScope::kThreadAgg) {
    return per_block % static_cast<std::size_t>(threads_per_block) == 0;
  }
  if (scope == PutScope::kWarp) {
    if (threads_per_block % device_warp_size != 0) {
      return false;
    }
    const int nw = threads_per_block / device_warp_size;
    if (nw <= 0) {
      return false;
    }
    return per_block % static_cast<std::size_t>(nw) == 0;
  }
  return true;  // block scope: per_block already integral
}

// Latency kernels issue a single op for the whole buffer — no per-lane split.
inline bool latency_size_ok(std::size_t len_doubles) { return len_doubles > 0; }

// Latency block width by scope (mirrors shmem util).
inline int LatencyBlockThreads(PutScope scope, int threads_per_block, int device_warp_size) {
  if (scope == PutScope::kWarp) return device_warp_size;
  if (scope == PutScope::kBlock) return threads_per_block;
  return 1;
}

}  // namespace mori::cco::benchmark

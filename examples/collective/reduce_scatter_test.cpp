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

// Reduce-scatter test using a SINGLE fused SDMA kernel (single-process,
// multi-threaded: one thread per GPU/PE, no MPI, no file-based bootstrap).
//
// Reduce-scatter semantics: every PE owns the full vector of N = npes * chunk
// elements. After the collective, PE q holds the reduction (over all PEs) of
// shard q:
//
//   output_q[j] = REDUCE_p( input_p[ q*chunkElems + j ] )   for j in [0, chunkElems)
//
// Algorithm (all three steps fused into ONE kernel launch):
//   Phase 1 (block 0 only): SDMA "push" scatter. For every destination peer p,
//     PE myPe sends input[p*chunkElems ..] into peer p's staging slot myPe,
//     split across SDMA queues for bandwidth. Fire-and-forget: the completion
//     atomic rides the same queue as the copy and targets the *receiver's*
//     signalPtrs, so it lands only after the data does. No local quiet.
//   Phase 2 (all blocks): receiver-side wait. Each PE spins on its own signalPtrs
//     until every (sender, queue) slot reaches the launch generation `gen`,
//     i.e. all peers finished writing our staging. This replaces both the local
//     SDMA quiet and the cross-PE barrier, and -- since the signals are global --
//     needs no block-0 -> all-blocks flag handoff, so every block is independent.
//   Phase 3 (all blocks): grid-strided vectorized reduction of the npes staging
//     slots into the output shard. Templated by element type T and reduction Op.
//
// Because there is no co-resident flag handoff, the reduce grid is NOT capped by
// multiProcessorCount -- push uses full SM occupancy, like the pull kernel.
//
// This file also contains a small, self-contained COOPERATIVE-LAUNCH demo
// kernel (CoopGridSyncDemoKernel) that uses cooperative_groups::grid_group and
// grid.sync() for study/comparison. It is unrelated to shmem and runs locally.
//
// Usage: ./reduce_scatter_test [num_gpus] [num_elems]
//   num_gpus  : number of GPUs/PEs to run in this process.
//   num_elems : TOTAL input element count (matches XLA's num_elems). The per-rank
//               output shard is chunkElems = num_elems / num_gpus; its byte size
//               (chunkElems * sizeof(ElemT)) must be a multiple of 16.

// reduce_scatter_test: 4 PEs, 262144 bytes/shard (65536 elems), 1048576 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=8 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.023 ms (45.856 GB/s) min 0.021 ms (49.180 GB/s) max 0.025 ms
// --------------------
// reduce_scatter_test: 4 PEs, 524288 bytes/shard (131072 elems), 2097152 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=16 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.027 ms (76.369 GB/s) min 0.025 ms (83.485 GB/s) max 0.033 ms
// --------------------
// reduce_scatter_test: 4 PEs, 1048576 bytes/shard (262144 elems), 4194304 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=32 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.039 ms (108.664 GB/s) min 0.036 ms (117.550 GB/s) max 0.045 ms
// --------------------
// reduce_scatter_test: 4 PEs, 2097152 bytes/shard (524288 elems), 8388608 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=64 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.058 ms (145.213 GB/s) min 0.054 ms (154.310 GB/s) max 0.061 ms
// --------------------
// reduce_scatter_test: 4 PEs, 4194304 bytes/shard (1048576 elems), 16777216 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=128 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.098 ms (171.584 GB/s) min 0.095 ms (176.971 GB/s) max 0.101 ms
// --------------------
// reduce_scatter_test: 4 PEs, 8388608 bytes/shard (2097152 elems), 33554432 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=256 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.181 ms (185.193 GB/s) min 0.178 ms (188.164 GB/s) max 0.185 ms
// --------------------
// reduce_scatter_test: 4 PEs, 16777216 bytes/shard (4194304 elems), 67108864 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=256 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.334 ms (200.882 GB/s) min 0.330 ms (203.182 GB/s) max 0.338 ms
// --------------------
// reduce_scatter_test: 4 PEs, 33554432 bytes/shard (8388608 elems), 134217728 bytes input/PE
// reduce_scatter_test: mode=PUSH numQ=1 blocks=256 (SMs=256)
// Rank 0: PASS | 5 warmup + 20 runs | avg 0.638 ms (210.275 GB/s) min 0.632 ms (212.216 GB/s) max 0.652 ms
// --------------------

// RCCL:
// I0000 00:00:1781190820.849456  195820 gpu_collectives_test.cc:798] bytes: 4194304 ms: 0.034509 alg_bw: 121.542 GB/s  bus_bw: 91.1567 GB/s
// I0000 00:00:1781190820.850156  195820 gpu_collectives_test.cc:798] bytes: 8388608 ms: 0.0528935 alg_bw: 158.594 GB/s  bus_bw: 118.946 GB/s
// I0000 00:00:1781190820.851236  195820 gpu_collectives_test.cc:798] bytes: 16777216 ms: 0.0879905 alg_bw: 190.671 GB/s  bus_bw: 143.003 GB/s
// I0000 00:00:1781190820.853203  195820 gpu_collectives_test.cc:798] bytes: 33554432 ms: 0.160136 alg_bw: 209.536 GB/s  bus_bw: 157.152 GB/s
// I0000 00:00:1781190820.856929  195820 gpu_collectives_test.cc:798] bytes: 67108864 ms: 0.307485 alg_bw: 218.251 GB/s  bus_bw: 163.688 GB/s
// I0000 00:00:1781190820.864209  195820 gpu_collectives_test.cc:798] bytes: 134217728 ms: 0.605257 alg_bw: 221.753 GB/s  bus_bw: 166.315 GB/s

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

#include <hip/hip_bfloat16.h>

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"  // load<N>/store<N>
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

using ElemT = hip_bfloat16;  // element type used by the test instantiation
#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

// Device/template kernels (push, pull) and the shared streaming
// load/store + reduction helpers live in this header.
#include "mori/collective/reduce_scatter_kernels.hpp"

// ---------------------------------------------------------------------------
// Fill / verify kernels
// ---------------------------------------------------------------------------
// input[k] = (myPe + 1) + (k % 8). All small integers (exact in float). With
// this pattern the reduce-scatter SUM result on shard owned by PE q is:
//   output_q[j] = sum_p[(p+1) + ((q*chunkElems + j) % 8)]
//              = npes*(npes+1)/2 + npes * ((q*chunkElems + j) % 8)
__global__ void FillPatternKernel(ElemT* buf, size_t numElements, int myPe) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = static_cast<ElemT>(static_cast<float>((myPe + 1) + static_cast<int>(i % 8)));
  }
}

__global__ void VerifyKernel(const ElemT* output, size_t chunkElems, int myPe, int npes,
                             uint32_t* errorCount) {
  const float base = static_cast<float>(npes) * (npes + 1) / 2.0f;
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < chunkElems;
       j += (size_t)gridDim.x * blockDim.x) {
    size_t globalIdx = static_cast<size_t>(myPe) * chunkElems + j;
    float expected = base + static_cast<float>(npes) * static_cast<float>(globalIdx % 8);
    if (fabsf(static_cast<float>(output[j]) - expected) > 1e-3f) {
      atomicAdd(errorCount, 1u);
    }
  }
}
struct ThreadInfo {
  int rank{-1};
  int worldSize{-1};
  int deviceId{-1};
  int ret_code{-1};
};

// ---------------------------------------------------------------------------
// Test body (runs after ShmemInit)
// ---------------------------------------------------------------------------
static void RunReduceScatterThreadedTest(size_t numElems, const UniqueId& uid, ThreadInfo& info) {
  HIP_RUNTIME_CHECK(hipSetDevice(info.deviceId));

  auto* bootstrap = new SocketBootstrapNetwork(uid, info.rank, info.worldSize);
  int status = ShmemInit(bootstrap);
  if (status != 0) {
    XPUT("ERROR: ShmemInit failed (ret=%d)", status);
    info.ret_code = status;
    return;
  }

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // numElems is the TOTAL input element count (matches XLA's num_elems); the
  // per-rank output shard is chunkElems = numElems / npes.
  const size_t chunkElems = numElems / npes;
  const size_t chunkBytes = chunkElems * sizeof(ElemT);
  const size_t N = static_cast<size_t>(npes) * chunkElems;  // input/staging element count
  const size_t inBytes = N * sizeof(ElemT);
  const size_t stagingBytes = N * sizeof(ElemT);
  const size_t outBytes = chunkBytes;
  const size_t totalBytes = inBytes + stagingBytes + outBytes;
  ShmemBarrierAll();

  if (info.deviceId == 0) {
    XPUT("reduce_scatter_test: %d PEs, %zu bytes/shard (%zu elems), %zu bytes input/PE", npes,
         chunkBytes, chunkElems, inBytes);
  }

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  // Single symmetric-heap allocation: [ input(N) | staging(N) | output(chunk) ].
  void* baseBuf = ShmemMalloc(totalBytes);
  if (baseBuf == nullptr) {
    XPUT("ERROR: ShmemMalloc(%zu) failed", totalBytes);
    info.ret_code = -1;
    return;
  }
  SymmMemObjPtr baseObj = ShmemQueryMemObjPtr(baseBuf);
  assert(baseObj.IsValid());

  ElemT* input = reinterpret_cast<ElemT*>(baseBuf);
  ElemT* staging = input + N;
  ElemT* output = staging + N;

  HIP_RUNTIME_CHECK(hipMemsetAsync(baseBuf, 0, totalBytes, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  // Fill input with the per-PE pattern.
  constexpr int kThreads = 256;
  int fillBlocks = static_cast<int>(std::min<size_t>(1024, (N + kThreads - 1) / kThreads));
  FillPatternKernel<<<fillBlocks, kThreads, 0, stream>>>(input, N, myPe);
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  ShmemBarrierAll();

  // Channel sizing. With the receiver-side completion signal (Phase 2) the push
  // kernel no longer has a cross-block flag handoff or co-residency requirement,
  // so BOTH push and pull use full SM occupancy.
  const int numQ = static_cast<int>(std::max(1u, baseObj->sdmaNumQueue));
  hipDeviceProp_t prop;
  HIP_RUNTIME_CHECK(hipGetDeviceProperties(&prop, info.deviceId));
  // The reduce kernels run on a packed compute type (two bf16 per 4-byte element)
  // so vecSize stays at fp32 parity and the accumulator tile does not spill. Data
  // buffers are physically ElemT (bf16); counts passed to the kernels are in packs.
  using ComputeT = BF16Pack;
  constexpr size_t kPack = sizeof(ComputeT) / sizeof(ElemT);  // bf16 lanes per pack
  const size_t chunkElemsC = chunkElems / kPack;              // chunk size in packs
  constexpr int VecBytes = 16,  NumVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ComputeT);
  size_t totalVecs = chunkElemsC / (VecSize * NumVecs);
  int wantBlocks = static_cast<int>(std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  int blocks = std::min(wantBlocks, std::max(1, prop.multiProcessorCount));

  // Push slicing: RS_PUSH_SLICES (default 1) rounded DOWN to a power of two and
  // clamped to [1,8]; also clamped so each slice has at least one vector. logS =
  // log2(S) in [0,3]. pushBlocks is rounded to a multiple of S (>= S) so each of
  // the S groups gets >= 1 block (G = pushBlocks/S).
  int pushSlices = 1;
  if (const char* s = std::getenv("RS_PUSH_SLICES")) {
    int req = std::atoi(s);
    if (req > 1) {
      int p = 1;
      while ((p << 1) <= req && p < 8) p <<= 1;
      pushSlices = p;
    }
  }
  {
    const size_t maxSlicesByData = std::max<size_t>(1, chunkElemsC / VecSize);
    while (pushSlices > 1 && static_cast<size_t>(pushSlices) > maxSlicesByData) pushSlices >>= 1;
  }
  int logS = 0;
  while ((1 << logS) < pushSlices) logS++;
  int pushBlocks = std::max(pushSlices, (blocks / pushSlices) * pushSlices);

  // Local-only per-group block counters for the push reset (never peer-written).
  uint32_t* groupCounters = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&groupCounters, 8 * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(groupCounters, 0, 8 * sizeof(uint32_t)));

  // Mode selection: RS_MODE = push|pull (default push). RS_PULL=1 is kept as a
  // back-compat alias for pull.
  enum class RsMode { kPush, kPull };
  const RsMode mode = [] {
    const char* m = std::getenv("RS_MODE");
    if (m != nullptr) {
      if (std::strcmp(m, "pull") == 0) return RsMode::kPull;
      if (std::strcmp(m, "push") == 0) return RsMode::kPush;
    }
    if (const char* p = std::getenv("RS_PULL")) {
      if (std::atoi(p) != 0) return RsMode::kPull;
    }
    return RsMode::kPush;
  }();

  // Pull uses the all-peers-up-front reduction; NPES is a compile-time template
  // arg, so the host dispatches on the real npes (supported for npes in [1, 8],
  // the single-node GPU count).
  if (mode == RsMode::kPull && (npes < 1 || npes > 8)) {
    XPUT("ERROR: pull mode supports npes in [1,8], got %d", npes);
    info.ret_code = -1;
    return;
  }

  if (info.deviceId == 0) {
    if (mode == RsMode::kPush) {
      XPUT("reduce_scatter_test: mode=PUSH slices=%d blocks=%d (SMs=%d)", pushSlices,
           pushBlocks, prop.multiProcessorCount);
    } else {
      XPUT("reduce_scatter_test: mode=PULL blocks=%d (SMs=%d)", blocks, prop.multiProcessorCount);
    }
  }
  ShmemBarrierAll();

  // --- Benchmark ---
  constexpr int nWarmup = 2;
  constexpr int nRuns = 5;
  hipEvent_t tStart, tStop;
  HIP_RUNTIME_CHECK(hipEventCreate(&tStart));
  HIP_RUNTIME_CHECK(hipEventCreate(&tStop));

  float totalMs = 0, minMs = 1e9f, maxMs = 0;
  for (int iter = 0; iter < nWarmup + nRuns; iter++) {
    ShmemBarrierAll();
    HIP_RUNTIME_CHECK(hipEventRecord(tStart, stream));
    if (mode == RsMode::kPull) {
      // input lives at offset 0 of baseBuf, so baseObj->peerPtrs[pe] is peer pe's
      // input base. output is the local shard buffer. NPES is compile-time so the
      // all-peers register tile stays in VGPRs; dispatch on the real npes.
      auto launch = [&](auto NPES_c) {
        ReduceScatterPullKernel<VecBytes, NumVecs, decltype(NPES_c)::value, ComputeT, SumOp>
            <<<blocks, kThreads, 0, stream>>>(myPe, baseObj,
                                              reinterpret_cast<ComputeT*>(output), chunkElemsC);
      };
      switch (npes) {
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        default: launch(std::integral_constant<int, 8>{}); break;
      }
    } else {
      // Sliced push: S = 1<<logS slices, each with its own per-receiver bitmask
      // flag; the grid is partitioned into S groups, each reset by its last block.
      ReduceScatterPushKernel<VecBytes, NumVecs, ComputeT, SumOp><<<pushBlocks, kThreads, 0, stream>>>(
          myPe, npes, logS, reinterpret_cast<const ComputeT*>(input),
          reinterpret_cast<ComputeT*>(staging), reinterpret_cast<ComputeT*>(output),
          groupCounters, chunkElemsC);
    }
    HIP_RUNTIME_CHECK(hipEventRecord(tStop, stream));
    HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

    float iterMs = 0;
    HIP_RUNTIME_CHECK(hipEventElapsedTime(&iterMs, tStart, tStop));
    if (iter >= nWarmup) {
      totalMs += iterMs;
      minMs = std::min(minMs, iterMs);
      maxMs = std::max(maxMs, iterMs);
    }
  }

  // After every PE's last kernel has been stream-synced above, each PE has
  // observed all incoming completion signals -> every outgoing DMA (incoming to
  // some peer) has landed. This host barrier makes that global before teardown,
  // so no in-flight peer DMA can target our staging after we free it.
  ShmemBarrierAll();

  float avgMs = totalMs / nRuns;
  // Bytes scattered over the network per PE ~ input bytes (N elements).
  double avgBw = (inBytes / 1e9) / (avgMs / 1e3);
  double maxBw = (inBytes / 1e9) / (minMs / 1e3);

  // --- Verify (last iteration result) ---
  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));
  int vBlocks =
      static_cast<int>(std::min<size_t>(1024, (chunkElems + kThreads - 1) / kThreads));
  VerifyKernel<<<vBlocks, kThreads, 0, stream>>>(output, chunkElems, myPe, npes, dErrors);
  uint32_t hErrors = 0;
  HIP_RUNTIME_CHECK(hipMemcpyAsync(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  HIP_RUNTIME_CHECK(hipFree(dErrors));

  if (hErrors != 0 || myPe == 0) {
    XPUT("Rank %d: %s | %d warmup + %d runs | avg %.3f ms (%.3f GB/s) "
       "min %.3f ms (%.3f GB/s) max %.3f ms\n--------------------",
       myPe, hErrors == 0 ? "PASS" : "FAIL", nWarmup, nRuns, avgMs, avgBw, minMs, maxBw,
       maxMs);
  }

  HIP_RUNTIME_CHECK(hipEventDestroy(tStart));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStop));
  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  HIP_RUNTIME_CHECK(hipFree(groupCounters));
  ShmemFree(baseBuf);
  ShmemFinalize();
  info.ret_code = 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  int deviceCount = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));
  if (argc < 3) {
    XPUT("Usage: %s [num_gpus] [num_elems]\n", argv[0]);
    return 1;
  }
  int numGpus = std::atoi(argv[1]);
  if (numGpus < 1 || numGpus > deviceCount) {
    XPUT("Usage: %s [num_gpus] [num_elems]   (num_gpus in 1..%d)\n", argv[0],
            deviceCount);
    return 1;
  }

  // num_elems is the TOTAL input element count (matches XLA's num_elems). The
  // per-rank output shard is chunkElems = num_elems / num_gpus.
  size_t numElems = std::atol(argv[2]);
  assert(numElems % numGpus == 0 && "num_elems must be divisible by num_gpus");
  size_t chunkElems = numElems / numGpus;
  size_t chunkBytes = chunkElems * sizeof(ElemT);
  assert(chunkBytes >= 16 && (chunkBytes % 16) == 0 &&
         "per-shard bytes (num_elems/num_gpus * sizeof) must be a multiple of 16");

  // Single in-process UniqueId shared by all threads (no file/MPI needed).
  mori_shmem_uniqueid_t uid_bytes{};
  int ret = ShmemGetUniqueId(&uid_bytes);
  if (ret != 0) {
    XPUT("ERROR: ShmemGetUniqueId failed (ret=%d)", ret);
    return 1;
  }
  UniqueId uid;
  static_assert(sizeof(uid) == sizeof(uid_bytes), "UniqueId size mismatch");
  std::memcpy(&uid, uid_bytes.data(), sizeof(uid));

  std::vector<std::thread> threads;
  std::vector<ThreadInfo> infos(numGpus);
  threads.reserve(numGpus);
  for (int i = 0; i < numGpus; i++) {
    infos[i].rank = i;
    infos[i].worldSize = numGpus;
    infos[i].deviceId = i;
    threads.emplace_back(RunReduceScatterThreadedTest, numElems, std::cref(uid),
                         std::ref(infos[i]));
  }
  for (auto& t : threads) t.join();

  for (const auto& inf : infos) {
    if (inf.ret_code != 0) {
      XPUT("ERROR: Rank %d returned non-zero ret_code %d", inf.rank, inf.ret_code);
      return 1;
    }
  }
  return 0;
}

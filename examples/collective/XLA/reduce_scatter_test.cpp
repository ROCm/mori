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

using ElemT = float;  // element type used by the test instantiation
//using ElemT = hip_bfloat16;  // element type used by the test instantiation
#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

// Central facade that owns staging/counters and launches the collective kernels
// (push/pull), hiding all host-side sizing/mode logic.
// This standalone example is the single TU that compiles the facade Run*
// definitions, so it opts in to the device-only impl section.
#define MORI_KERNELS_IMPL
#include "mori/collective/collectives_facade.hpp"

static_assert(std::is_same<ElemT, float>::value || std::is_same<ElemT, hip_bfloat16>::value,
              "reduce_scatter_test supports only float and hip_bfloat16");

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
  const size_t N = static_cast<size_t>(npes) * chunkElems;  // input element count
  const size_t inBytes = N * sizeof(ElemT);
  const size_t outBytes = chunkBytes;
  const size_t totalBytes = inBytes + outBytes;  // [ input(N) | output(chunk) ]
  // Facade-owned staging holds only the npes-1 non-self peer contributions
  // (dense, no self hole); the receiver's own shard is reduced from local input.
  const size_t stagingBytes = static_cast<size_t>(npes - 1) * chunkBytes;
  ShmemBarrierAll();

  if (info.deviceId == 0) {
    XPUT("reduce_scatter_test: %d PEs, %zu bytes/shard (%zu elems), %zu bytes input/PE", npes,
         chunkBytes, chunkElems, inBytes);
  }

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  // Single symmetric-heap allocation: [ input(N) | output(chunk) ]. The staging
  // buffer is owned by the facade (allocated in Init below).
  void* baseBuf = ShmemMalloc(totalBytes);
  if (baseBuf == nullptr) {
    XPUT("ERROR: ShmemMalloc(%zu) failed", totalBytes);
    info.ret_code = -1;
    return;
  }
  SymmMemObjPtr baseObj = ShmemQueryMemObjPtr(baseBuf);
  assert(baseObj.IsValid());

  ElemT* input = reinterpret_cast<ElemT*>(baseBuf);
  ElemT* output = input + N;

  HIP_RUNTIME_CHECK(hipMemsetAsync(baseBuf, 0, totalBytes, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  // Fill input with the per-PE pattern.
  constexpr int kThreads = 256;
  int fillBlocks = static_cast<int>(std::min<size_t>(1024, (N + kThreads - 1) / kThreads));
  FillPatternKernel<<<fillBlocks, kThreads, 0, stream>>>(input, N, myPe);
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  ShmemBarrierAll();

  // The facade owns the staging buffer + per-slice group counters; size staging
  // to this run's (npes-1)*chunkBytes. All host-side sizing (blocks, slices,
  // RS_MODE / RS_LOG_PUSH_SLICES) now lives inside RunReduceScatter.
  auto facade = mori::collective::CollectivesFacade::Create(myPe, npes, stagingBytes);
  if (facade == nullptr) {
    XPUT("ERROR: Failed to create CollectivesFacade");
    info.ret_code = -1;
    return;
  }

  if (info.deviceId == 0) {
    const char* m = std::getenv("RS_MODE");
    XPUT("reduce_scatter_test: mode=%s", (m != nullptr && std::strcmp(m, "pull") == 0) ? "PULL"
                                                                                       : "PUSH");
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
    // Reduce-scatter over the full input into this PE's shard; the facade picks
    // push vs pull (RS_MODE) and all block/slice sizing internally. `count` is
    // the per-rank shard element count (full input = chunkElems * npes).
    facade->RunReduceScatter(input, output, chunkElems, mori::collective::DataType::F32,
                             mori::collective::ReduceOpKind::SUM, stream);
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
  facade.reset();
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
  // assert(chunkBytes >= 16 && (chunkBytes % 16) == 0 &&
  //        "per-shard bytes (num_elems/num_gpus * sizeof) must be a multiple of 16");

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

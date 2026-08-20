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
#include <hip/hip_fp16.h>

#define MORI_KERNELS_IMPL

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"  // load<N>/store<N>
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/collective/collectives_facade.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;
using mori::collective::ReduceOpKind;
using mori::collective::DataType;
using mori::collective::CollectivesFacade;

using ElemT = __half;  // element type used by the test instantiation
// using ElemT = hip_bfloat16;
// using ElemT = float;
// using ElemT = int32_t;
// using ElemT = int64_t;
#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)


static_assert(std::is_same<ElemT, float>::value || std::is_same<ElemT, hip_bfloat16>::value ||
                  std::is_same<ElemT, __half>::value || std::is_same<ElemT, int32_t>::value ||
                  std::is_same<ElemT, int64_t>::value,
              "reduce_scatter_test supports float, hip_bfloat16, __half, int32_t, int64_t");

constexpr DataType kDataType = []() {
  if constexpr (std::is_same<ElemT, float>::value) return DataType::F32;
  if constexpr (std::is_same<ElemT, hip_bfloat16>::value) return DataType::BF16;
  if constexpr (std::is_same<ElemT, __half>::value) return DataType::F16;
  if constexpr (std::is_same<ElemT, int32_t>::value) return DataType::S32;
  if constexpr (std::is_same<ElemT, int64_t>::value) return DataType::S64;
  return DataType::F32;
}();

const char *DataTypeName(DataType dt) {
#define ITEM(x) case DataType::x: return #x;
  switch (dt) {
    DATA_TYPE_LIST(ITEM)
    default:
      return "?";
  }
#undef ITEM
}

static const char* ReduceOpName(ReduceOpKind op) {
  switch (op) {
    case ReduceOpKind::SUM:
      return "SUM";
    case ReduceOpKind::PRODUCT:
      return "PRODUCT";
    case ReduceOpKind::MIN:
      return "MIN";
    case ReduceOpKind::MAX:
      return "MAX";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Fill / verify kernels
// ---------------------------------------------------------------------------
// input[k] = (myPe + 1) + (k % 8). All small integers (exact in float/bf16
// storage). Expected output is a sequential fold over peers (seed PE 0, then
// 1..npes-1) of that term, with a round-trip through ElemT after each op so
// packed per-op bf16 rounding matches the kernel.
__global__ void FillPatternKernel(ElemT* buf, size_t numElements, int myPe) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = static_cast<ElemT>(static_cast<float>((myPe + 1) + static_cast<int>(i % 8)));
  }
}

__global__ void VerifyKernel(const ElemT* output, size_t chunkElems, int myPe, int npes,
                             ReduceOpKind op, uint32_t* errorCount) {
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < chunkElems;
       j += (size_t)gridDim.x * blockDim.x) {
    size_t globalIdx = static_cast<size_t>(myPe) * chunkElems + j;
    float acc = static_cast<float>(1 + static_cast<int>(globalIdx % 8));
    for (int p = 1; p < npes; p++) {
      float term = static_cast<float>((p + 1) + static_cast<int>(globalIdx % 8));
      switch (op) {
        case ReduceOpKind::SUM:
          acc += term;
          break;
        case ReduceOpKind::PRODUCT:
          acc *= term;
          break;
        case ReduceOpKind::MIN:
          acc = fminf(acc, term);
          break;
        case ReduceOpKind::MAX:
          acc = fmaxf(acc, term);
          break;
      }
      acc = static_cast<float>(static_cast<ElemT>(acc));
    }
    if (fabsf(static_cast<float>(output[j]) - acc) > 1e-3f) {
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
    XPUT("reduce_scatter_test: %d PEs, %zu bytes/shard (%zu elems), %zu bytes input/PE, dtype=%s",
         npes, chunkBytes, chunkElems, inBytes, DataTypeName(kDataType));
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

  constexpr ReduceOpKind kOps[] = {
      ReduceOpKind::SUM, ReduceOpKind::PRODUCT,
      ReduceOpKind::MIN, ReduceOpKind::MAX};

  constexpr int nWarmup = 2;
  constexpr int nRuns = 5;
  hipEvent_t tStart, tStop;
  HIP_RUNTIME_CHECK(hipEventCreate(&tStart));
  HIP_RUNTIME_CHECK(hipEventCreate(&tStop));

  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  int vBlocks =
      static_cast<int>(std::min<size_t>(1024, (chunkElems + kThreads - 1) / kThreads));

  int nFailed = 0;
  for (auto op : kOps) {
    float totalMs = 0, minMs = 1e9f, maxMs = 0;
    hipError_t launchErr = hipSuccess;
    for (int iter = 0; iter < nWarmup + nRuns; iter++) {
      ShmemBarrierAll();
      HIP_RUNTIME_CHECK(hipEventRecord(tStart, stream));
      launchErr = facade->RunReduceScatter(input, output, chunkElems, kDataType, op, stream);
      HIP_RUNTIME_CHECK(hipEventRecord(tStop, stream));
      HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
      if (launchErr != hipSuccess) break;

      float iterMs = 0;
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&iterMs, tStart, tStop));
      if (iter >= nWarmup) {
        totalMs += iterMs;
        minMs = std::min(minMs, iterMs);
        maxMs = std::max(maxMs, iterMs);
      }
    }

    ShmemBarrierAll();

    uint32_t hErrors = 0;
    if (launchErr != hipSuccess) {
      hErrors = 1;
      if (myPe == 0) {
        XPUT("Rank %d: FAIL op=%s launch=%s", myPe, ReduceOpName(op), hipGetErrorString(launchErr));
      }
    } else {
      float avgMs = totalMs / nRuns;
      double avgBw = (inBytes / 1e9) / (avgMs / 1e3);
      double maxBw = (inBytes / 1e9) / (minMs / 1e3);

      HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));
      VerifyKernel<<<vBlocks, kThreads, 0, stream>>>(output, chunkElems, myPe, npes, op, dErrors);
      HIP_RUNTIME_CHECK(hipMemcpyAsync(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost,
                                       stream));
      HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

      if (hErrors != 0 || myPe == 0) {
        XPUT("Rank %d: %s op=%s | %d warmup + %d runs | avg %.3f ms (%.3f GB/s) "
             "min %.3f ms (%.3f GB/s) max %.3f ms",
             myPe, hErrors == 0 ? "PASS" : "FAIL", ReduceOpName(op), nWarmup, nRuns, avgMs, avgBw,
             minMs, maxBw, maxMs);
      }
    }
    if (hErrors != 0) nFailed++;
  }

  ShmemBarrierAll();

  HIP_RUNTIME_CHECK(hipFree(dErrors));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStart));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStop));
  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  facade.reset();
  ShmemFree(baseBuf);
  ShmemFinalize();
  info.ret_code = nFailed == 0 ? 0 : -1;
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

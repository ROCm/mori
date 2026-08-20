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

// All-reduce test using a SINGLE fused SDMA kernel (single-process,
// multi-threaded: one thread per GPU/PE, no MPI, no file-based bootstrap).
//
// All-reduce semantics: every PE owns the full vector of N = npes * chunk
// elements. After the collective every PE holds the reduction (over all PEs) of
// the FULL vector:
//
//   output_q[g] = REDUCE_p( input_p[g] )   for g in [0, N)  (same on every PE q)
//
// Algorithm (all fused into ONE kernel launch, see AllReducePushKernel):
//   Phase 1-3: reduce-scatter push -- block 0 SDMA-scatters each peer's shard
//     slice into that peer's staging slot, every block waits on its per-slice
//     completion counter, then the grid reduces the npes contributions of MY
//     shard into output[myPe] (= myShard), sliced into S slice-groups.
//   Phase 4 (pipelined broadcast): the moment a slice-group finishes reducing
//     its slice, its last block SDMA-broadcasts that slice into every peer's
//     output[myPe] slot and waits for every peer's copy of the slice to land.
//     Each slice-group runs independently (no device-wide barrier), overlapping
//     the broadcast with the reduce tail of the other slices.
//
// Usage: ./all_reduce_test [num_gpus] [num_elems]
//   num_gpus  : number of GPUs/PEs to run in this process.
//   num_elems : TOTAL element count (matches XLA's num_elems). The per-rank
//               shard is chunkElems = num_elems / num_gpus; its byte size
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

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"  // load<N>/store<N>
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"

#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

// Central facade that owns staging/counters and launches the collective kernels,
// hiding all host-side sizing logic.
// This standalone example is the single TU that compiles the facade Run*
// definitions, so it opts in to the device-only impl section.
#define MORI_KERNELS_IMPL
#include "mori/collective/collectives_facade.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;
using mori::collective::ReduceOpKind;
using mori::collective::DataType;
using mori::collective::CollectivesFacade;

using ElemT = float;  // element type used by the test instantiation
// using ElemT = __half;
// using ElemT = hip_bfloat16;
// using ElemT = int32_t;
// using ElemT = int64_t;

static_assert(std::is_same<ElemT, float>::value || std::is_same<ElemT, hip_bfloat16>::value ||
                  std::is_same<ElemT, __half>::value || std::is_same<ElemT, int32_t>::value ||
                  std::is_same<ElemT, int64_t>::value,
              "all_reduce_test supports float, hip_bfloat16, __half, int32_t, int64_t");

constexpr DataType kDataType = []() {
  if constexpr (std::is_same<ElemT, float>::value) return DataType::F32;
  if constexpr (std::is_same<ElemT, hip_bfloat16>::value) return DataType::BF16;
  if constexpr (std::is_same<ElemT, __half>::value) return DataType::F16;
  if constexpr (std::is_same<ElemT, int32_t>::value) return DataType::S32;
  if constexpr (std::is_same<ElemT, int64_t>::value) return DataType::S64;
  return DataType::F32;
}();

const char* DataTypeName(DataType dt) {
#define ITEM(x) case DataType::x: return #x;
  switch (dt) {
    DATA_TYPE_LIST(ITEM)
    default:
      return "?";
  }
#undef ITEM
}

// ---------------------------------------------------------------------------
// Fill / verify kernels
// ---------------------------------------------------------------------------
// input[k] = (myPe + 1) + (k % 8). All small integers (exact in float). With
// this pattern the all-reduce SUM result at global index g is:
//   output[g] = sum_p[(p+1) + (g % 8)]
//            = npes*(npes+1)/2 + npes * (g % 8)   (identical on every PE)
__global__ void FillPatternKernel(ElemT* buf, size_t numElements, int myPe) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = static_cast<ElemT>(static_cast<float>((myPe + 1) + static_cast<int>(i % 8)));
  }
}

__global__ void VerifyKernel(const ElemT* output, size_t N, int npes, uint32_t* errorCount) {
  const float base = static_cast<float>(npes) * (npes + 1) / 2.0f;
  for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < N;
       g += (size_t)gridDim.x * blockDim.x) {
    float expected = base + static_cast<float>(npes) * static_cast<float>(g % 8);
    if (fabsf(static_cast<float>(output[g]) - expected) > 1e-3f) {
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
static void RunAllReduceThreadedTest(size_t numElems, const UniqueId& uid, ThreadInfo& info) {
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

  // The SDMA fast path / bitmask completion flag are limited to npes <= 8.
  if (npes < 1 || npes > 8) {
    XPUT("ERROR: all_reduce_test supports npes in [1,8], got %d", npes);
    info.ret_code = -1;
    ShmemFinalize();
    return;
  }

  // numElems is the TOTAL element count (matches XLA's num_elems); the per-rank
  // shard is chunkElems = numElems / npes.
  const size_t chunkElems = numElems / npes;
  const size_t chunkBytes = chunkElems * sizeof(ElemT);
  const size_t N = static_cast<size_t>(npes) * chunkElems;  // input/output element count
  const size_t inBytes = N * sizeof(ElemT);
  const size_t outBytes = N * sizeof(ElemT);       // full reduced vector
  const size_t totalBytes = inBytes + outBytes;    // [ input(N) | output(N) ]
  // Facade-owned staging holds only the npes-1 non-self peer contributions
  // (dense, no self hole); the receiver's own shard is reduced from local input.
  const size_t stagingBytes = static_cast<size_t>(npes - 1) * chunkBytes;
  ShmemBarrierAll();

  if (info.deviceId == 0) {
    XPUT("all_reduce_test: %d PEs, %zu bytes/shard (%zu elems), %zu bytes input/PE, dtype=%s",
         npes, chunkBytes, chunkElems, inBytes, DataTypeName(kDataType));
  }

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  // Single symmetric-heap allocation: [ input(N) | output(N) ]. The staging
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
  // RS_LOG_PUSH_SLICES) now lives inside RunAllReduce.
  auto facade = mori::collective::CollectivesFacade::Create(myPe, npes, stagingBytes);
  if (facade == nullptr) {
    XPUT("ERROR: Failed to create CollectivesFacade");
    info.ret_code = -1;
    return;
  }

  if (info.deviceId == 0) {
    XPUT("all_reduce_test: mode=PUSH");
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
    // Fused all-reduce (reduce-scatter push + pipelined per-slice broadcast); the
    // facade handles block/slice sizing internally.
    facade->RunAllReduce(input, output, N, kDataType, ReduceOpKind::SUM, stream);
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
  // observed all incoming completion signals -> every outgoing DMA has landed.
  // This host barrier makes that global before teardown.
  ShmemBarrierAll();

  float avgMs = totalMs / nRuns;
  // Bytes scattered over the network per PE ~ input bytes (N elements).
  double avgBw = (inBytes / 1e9) / (avgMs / 1e3);
  double maxBw = (inBytes / 1e9) / (minMs / 1e3);

  // --- Verify (last iteration result): check the FULL reduced output over N. ---
  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));
  int vBlocks = static_cast<int>(std::min<size_t>(1024, (N + kThreads - 1) / kThreads));
  VerifyKernel<<<vBlocks, kThreads, 0, stream>>>(output, N, npes, dErrors);
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
  info.ret_code = (hErrors == 0) ? 0 : -1;
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
    XPUT("Usage: %s [num_gpus] [num_elems]   (num_gpus in 1..%d)\n", argv[0], deviceCount);
    return 1;
  }

  // num_elems is the TOTAL element count (matches XLA's num_elems). The per-rank
  // shard is chunkElems = num_elems / num_gpus.
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
    threads.emplace_back(RunAllReduceThreadedTest, numElems, std::cref(uid), std::ref(infos[i]));
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

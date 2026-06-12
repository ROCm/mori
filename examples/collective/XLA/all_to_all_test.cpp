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

// All-to-all test using a SINGLE compute-free fused SDMA kernel (single-process,
// multi-threaded: one thread per GPU/PE, no MPI, no file-based bootstrap).
//
// All-to-all semantics (matching RCCL send/recv vectors): PE myPe sends its chunk
// destined for peer p (send slot p) to peer p, and peer p stores it in its recv
// slot for sender myPe. Self (p==myPe) is included via an SDMA self-copy. After
// the collective:
//
//   recv_q[s*chunkElems + j] = send_s[q*chunkElems + j]   (chunk s sent to me)
//
// Algorithm (a single kernel launch, ONE block, no GPU compute):
//   Phase 1 (block 0 only): SDMA "push". PE myPe pushes each send slot p into
//     peer p's recv slot for sender myPe (self included). Each copy is trailed by
//     a completion ADD64 of (1<<myPe) into the *receiver's* signalPtrs[0].
//   Phase 2: receiver-side wait. Thread 0 spins on signalPtrs[0] until every
//     sender's bit is set (full mask), then an acquire fence makes the peer SDMA
//     writes to recv visible.
//   No Phase 3: recv is already fully written by SDMA.
//
// Usage: ./all_to_all_test [num_gpus] [num_elems]
//   num_gpus  : number of GPUs/PEs to run in this process.
//   num_elems : TOTAL element count per buffer (= npes * chunk). The per-peer
//               chunk is chunkElems = num_elems / num_gpus; its byte size
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
#include "mori/core/transport/p2p/device_primitives.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

using ElemT = float;  // element type used by the test instantiation
#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

// Central facade that launches the collective kernels (all-to-all is compute-free
// and needs no staging, so no Init is required).
// This standalone example is the single TU that compiles the facade Run*
// definitions, so it opts in to the device-only impl section.
#define MORI_KERNELS_IMPL
#include "mori/collective/collectives_facade.hpp"

static_assert(std::is_same<ElemT, float>::value || std::is_same<ElemT, hip_bfloat16>::value,
              "all_to_all_test supports only float and hip_bfloat16");

// ---------------------------------------------------------------------------
// Fill / verify kernels
// ---------------------------------------------------------------------------
// My send buffer: slot p (chunk destined to peer p) holds a value encoding
// (sender=myPe, dest=p, j). send[g] = myPe*100 + (g/chunkElems)*10 + (g%chunkElems)%8.
// float-exact for myPe,dest < 10.
__global__ void FillPatternKernel(ElemT* send, size_t chunkElems, int npes, int myPe) {
  const size_t N = static_cast<size_t>(npes) * chunkElems;
  for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < N;
       g += (size_t)gridDim.x * blockDim.x) {
    const int dest = static_cast<int>(g / chunkElems);
    const size_t j = g % chunkElems;
    send[g] = static_cast<ElemT>(
        static_cast<float>(myPe * 100 + dest * 10 + static_cast<int>(j % 8)));
  }
}

// After all-to-all, my recv slot s holds the chunk sender s destined to me, i.e.
// sender s's send slot myPe: recv[s*chunkElems + j] == s*100 + myPe*10 + j%8.
__global__ void VerifyKernel(const ElemT* recv, size_t chunkElems, int npes, int myPe,
                             uint32_t* errorCount) {
  const size_t N = static_cast<size_t>(npes) * chunkElems;
  for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < N;
       g += (size_t)gridDim.x * blockDim.x) {
    const int s = static_cast<int>(g / chunkElems);  // sender
    const size_t j = g % chunkElems;                 // element within chunk
    float expected = static_cast<float>(s * 100 + myPe * 10 + static_cast<int>(j % 8));
    if (fabsf(static_cast<float>(recv[g]) - expected) > 1e-3f) {
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
static void RunAllToAllThreadedTest(size_t numElems, const UniqueId& uid, ThreadInfo& info) {
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
    XPUT("ERROR: all_to_all_test supports npes in [1,8], got %d", npes);
    info.ret_code = -1;
    ShmemFinalize();
    return;
  }

  // All-to-all is compute-free (no staging needed) but the facade still needs the
  // rank identity, so Init with maxStagingBytes = 0.
  auto facade = mori::collective::CollectivesFacade::Create(myPe, npes, 0);
  if (facade == nullptr) {
    XPUT("ERROR: Failed to create CollectivesFacade");
    info.ret_code = -1;
    ShmemFinalize();
    return;
  }

  // numElems is the per-buffer element count (= npes * chunk); each PE exchanges
  // chunkElems = numElems / npes with every peer.
  const size_t chunkElems = numElems / npes;
  const size_t chunkBytes = chunkElems * sizeof(ElemT);
  const size_t N = static_cast<size_t>(npes) * chunkElems;  // per-buffer element count
  const size_t bufBytes = N * sizeof(ElemT);
  const size_t totalBytes = 2 * bufBytes;  // [ send(N) | recv(N) ]
  ShmemBarrierAll();

  if (info.deviceId == 0) {
    XPUT("all_to_all_test: %d PEs, %zu bytes/chunk (%zu elems), %zu bytes send/recv per PE", npes,
         chunkBytes, chunkElems, bufBytes);
  }

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  // Single symmetric-heap allocation: [ send(N) | recv(N) ]. Both live on the
  // heap so SDMA can read the send chunks and write every peer's recv slot.
  void* baseBuf = ShmemMalloc(totalBytes);
  if (baseBuf == nullptr) {
    XPUT("ERROR: ShmemMalloc(%zu) failed", totalBytes);
    info.ret_code = -1;
    return;
  }
  SymmMemObjPtr baseObj = ShmemQueryMemObjPtr(baseBuf);
  assert(baseObj.IsValid());

  ElemT* send = reinterpret_cast<ElemT*>(baseBuf);
  ElemT* recv = send + N;

  HIP_RUNTIME_CHECK(hipMemsetAsync(baseBuf, 0, totalBytes, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  // Per-peer endpoints (host AddressVector); the facade copies these into its
  // pinned buffer on each launch:
  //   src[p] = send + p*chunkElems  (chunk destined to peer p)
  //   dst[p] = recv + p*chunkElems  (recv slot for sender p)
  CollectivesFacade::AddressVector addrs(npes);
  for (int p = 0; p < npes; p++) {
    addrs[p] = {send + static_cast<size_t>(p) * chunkElems,
                recv + static_cast<size_t>(p) * chunkElems};
  }

  // Fill my send buffer with the per-(sender,dest) pattern.
  constexpr int kThreads = 256;
  int fillBlocks = static_cast<int>(std::min<size_t>(1024, (N + kThreads - 1) / kThreads));
  FillPatternKernel<<<fillBlocks, kThreads, 0, stream>>>(send, chunkElems, npes, myPe);
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  ShmemBarrierAll();

  if (info.deviceId == 0) {
    XPUT("all_to_all_test: mode=PUSH single-block (npes=%d)", npes);
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
    // Single block: SDMA pushes each send slot p to peer p's recv slot (+ self).
    facade->RunAllToAll(addrs, chunkBytes, stream);
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

  // After every PE's last kernel has been stream-synced, each PE has observed all
  // incoming completion signals -> every outgoing DMA has landed. This host
  // barrier makes that global before teardown so no in-flight peer DMA targets
  // our recv after we free it.
  ShmemBarrierAll();

  float avgMs = totalMs / nRuns;
  // Bytes received over the network per PE = (npes-1) chunks (self is local).
  const double rxBytes = static_cast<double>(npes - 1) * chunkBytes;
  double avgBw = (rxBytes / 1e9) / (avgMs / 1e3);
  double maxBw = (rxBytes / 1e9) / (minMs / 1e3);

  // --- Verify (last iteration result): check the FULL recv buffer. ---
  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));
  int vBlocks = static_cast<int>(std::min<size_t>(1024, (N + kThreads - 1) / kThreads));
  VerifyKernel<<<vBlocks, kThreads, 0, stream>>>(recv, chunkElems, npes, myPe, dErrors);
  uint32_t hErrors = 0;
  HIP_RUNTIME_CHECK(hipMemcpyAsync(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  HIP_RUNTIME_CHECK(hipFree(dErrors));

  if (hErrors != 0 || myPe == 0) {
    XPUT("Rank %d: %s | %d warmup + %d runs | avg %.3f ms (%.3f GB/s) "
         "min %.3f ms (%.3f GB/s) max %.3f ms\n--------------------",
         myPe, hErrors == 0 ? "PASS" : "FAIL", nWarmup, nRuns, avgMs, avgBw, minMs, maxBw, maxMs);
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

  // num_elems is the per-buffer element count (= npes * chunk). The per-peer
  // chunk is chunkElems = num_elems / num_gpus.
  size_t numElems = std::atol(argv[2]);
  assert(numElems % numGpus == 0 && "num_elems must be divisible by num_gpus");
  size_t chunkElems = numElems / numGpus;
  size_t chunkBytes = chunkElems * sizeof(ElemT);
  assert(chunkBytes >= 16 && (chunkBytes % 16) == 0 &&
         "per-chunk bytes (num_elems/num_gpus * sizeof) must be a multiple of 16");

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
    threads.emplace_back(RunAllToAllThreadedTest, numElems, std::cref(uid), std::ref(infos[i]));
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

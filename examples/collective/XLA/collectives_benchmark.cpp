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

// Unified benchmark for the XLA push/pull collectives (single-process,
// multi-threaded: one thread per GPU/PE, no MPI, no file-based bootstrap).
// Consolidates the former reduce_scatter/all_reduce/all_gather/all_to_all/
// collective_permute tests into one runtime-configurable executable so a
// collective and all of its tuning knobs can be selected without recompiling.
//
// Usage:
//   ./collectives_benchmark --coll <name> --npes <n> --size <num_elems> [opts]
//
//   --coll   reduce_scatter|all_reduce|all_gather|all_to_all|collective_permute
//            (aliases: rs|ar|ag|a2a|cp)                                [required]
//   --npes   number of GPUs/PEs to run in this process (1..8)          [required]
//   --size   TOTAL element count. For the sharded collectives the per-rank shard
//            is size/npes; for collective_permute size is the per-rank buffer.
//                                                                      [required]
//   --dtype  f32|bf16|f16|s32|s64   (reduction collectives)            [f32]
//   --op     sum|prod|min|max       (reduction collectives)            [sum]
//   --mode   push|pull              (reduce_scatter)                   [push]
//   --logS   push slice count log2                                     [0]
//   --warmup <n>                                                       [2]
//   --iters  <n>                                                       [5]
//
// Note: dtype/op beyond f32/sum require a build with FACADE_REDUCE_USE_ALL_TYPES=1.

#include <algorithm>
#include <cassert>
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

#include "mori/cco/cco.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"
#include "mori/collective/collectives_facade.hpp"

using namespace mori::cco;
using namespace mori::collective;
using RsMode = CollectivesFacade::RsMode;

// Headroom added on top of the facade heap when sizing the CCO comm's per-rank
// flat-VA slot: CCO carves its own internal windows (the DevComm resource
// window) out of the same slot, so the heap must not consume all of it.
static constexpr size_t kVmmSlack = 512ULL * 1024 * 1024;
// The flat VA is reserved in 4 GiB units (stride4G = perRankSize >> 32).
static constexpr size_t kVmmGrain = 4ULL * 1024 * 1024 * 1024;
// Slack on top of the benchmark's own buffers, absorbing the facade's 256B
// per-allocation alignment and its internal counters.
static constexpr size_t kHeapSlack = 64ULL * 1024 * 1024;

#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

static size_t AlignUp(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

enum class Collective { kReduceScatter, kAllReduce, kAllGather, kAllToAll, kCollectivePermute };

struct Config {
  Collective coll{};
  int npes{0};
  size_t numElems{0};
  DataType dt{DataType::F32};
  ReduceOpKind op{ReduceOpKind::SUM};
  RsMode mode{RsMode::kPush};
  int logS{0};
  int warmup{2};
  int iters{5};
  // Derived in main() by SizeHeap(): the facade's symmetric heap window, the
  // push-path staging carved from it, and the comm's flat-VA slot.
  size_t heapBytes{0};
  size_t stagingBytes{0};
  size_t perRankVmm{0};
};

// ---------------------------------------------------------------------------
// Fill / verify kernels (templated on the element type)
// ---------------------------------------------------------------------------
// Common per-PE pattern: buf[i] = (myPe + 1) + (i % 8). Small integers, exact in
// float/bf16 storage. Used by reduce-scatter/all-reduce (input), all-gather (my
// shard) and collective-permute (send buffer).
template <class ElemT>
__global__ void FillCommonKernel(ElemT* buf, size_t numElements, int myPe) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = static_cast<ElemT>(static_cast<float>((myPe + 1) + static_cast<int>(i % 8)));
  }
}

// All-to-all send pattern: slot dest holds (sender=myPe, dest, j) encoded as
// myPe*100 + dest*10 + j%8 (float-exact for myPe,dest < 10).
template <class ElemT>
__global__ void FillA2AKernel(ElemT* send, size_t chunkElems, int npes, int myPe) {
  const size_t N = static_cast<size_t>(npes) * chunkElems;
  for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < N;
       g += (size_t)gridDim.x * blockDim.x) {
    const int dest = static_cast<int>(g / chunkElems);
    const size_t j = g % chunkElems;
    send[g] = static_cast<ElemT>(
        static_cast<float>(myPe * 100 + dest * 10 + static_cast<int>(j % 8)));
  }
}

// Reduce-scatter / all-reduce verify: expected[i] is the fold (seed PE 0, then
// 1..npes-1) of ((p+1) + (globalIdx % 8)) at globalIdx = globalOffset + i, with a
// round-trip through ElemT after each op (matches packed per-op rounding).
template <class ElemT>
__global__ void VerifyReduceKernel(const ElemT* output, size_t count, size_t globalOffset,
                                   int npes, ReduceOpKind op, uint32_t* errorCount) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t globalIdx = globalOffset + i;
    float acc = static_cast<float>(1 + static_cast<int>(globalIdx % 8));  // PE 0 seed
    for (int p = 1; p < npes; p++) {
      float term = static_cast<float>((p + 1) + static_cast<int>(globalIdx % 8));
      auto err = detail::DispatchReduceOp<ElemT>(op, [&acc, term](auto reduceOp) {
        acc = reduceOp(acc, term);
        return hipSuccess;
      });
      if (err != hipSuccess) { atomicAdd(errorCount, 1u); break; }
      acc = static_cast<float>(static_cast<ElemT>(acc));
    }
    if (fabsf(static_cast<float>(output[i]) - acc) > 1e-3f) {
      atomicAdd(errorCount, 1u);
    }
  }
}

// All-gather verify: output[p*chunk + j] == (p + 1) + (j % 8).
template <class ElemT>
__global__ void VerifyGatherKernel(const ElemT* output, size_t chunkElems, int npes,
                                   uint32_t* errorCount) {
  const size_t N = static_cast<size_t>(npes) * chunkElems;
  for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < N;
       g += (size_t)gridDim.x * blockDim.x) {
    const int p = static_cast<int>(g / chunkElems);
    const size_t j = g % chunkElems;
    float expected = static_cast<float>((p + 1) + static_cast<int>(j % 8));
    if (fabsf(static_cast<float>(output[g]) - expected) > 1e-3f) {
      atomicAdd(errorCount, 1u);
    }
  }
}

// All-to-all verify: recv[s*chunk + j] == s*100 + myPe*10 + j%8.
template <class ElemT>
__global__ void VerifyA2AKernel(const ElemT* recv, size_t chunkElems, int npes, int myPe,
                                uint32_t* errorCount) {
  const size_t N = static_cast<size_t>(npes) * chunkElems;
  for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < N;
       g += (size_t)gridDim.x * blockDim.x) {
    const int s = static_cast<int>(g / chunkElems);
    const size_t j = g % chunkElems;
    float expected = static_cast<float>(s * 100 + myPe * 10 + static_cast<int>(j % 8));
    if (fabsf(static_cast<float>(recv[g]) - expected) > 1e-3f) {
      atomicAdd(errorCount, 1u);
    }
  }
}

// Collective-permute (ring) verify: recv[i] == (srcPe + 1) + (i % 8).
template <class ElemT>
__global__ void VerifyPermuteKernel(const ElemT* recv, size_t numElements, int srcPe,
                                    uint32_t* errorCount) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    float expected = static_cast<float>((srcPe + 1) + static_cast<int>(i % 8));
    if (fabsf(static_cast<float>(recv[i]) - expected) > 1e-3f) {
      atomicAdd(errorCount, 1u);
    }
  }
}

// ---------------------------------------------------------------------------
// Per-kernel launcher wrappers: the only place that needs the concrete element
// type. Each wraps one templated kernel in detail::DispatchReduceType(dt) so the
// benchmark body can stay type-agnostic and operate on raw void* buffers (the
// facade Run* methods already dispatch dtype/op internally).
// ---------------------------------------------------------------------------
static void LaunchFillCommon(DataType dt, void* buf, size_t n, int myPe, int blocks, int threads,
                             hipStream_t s) {
  (void)detail::DispatchReduceType(dt, [&](auto tag) {
    using ElemT = decltype(tag);
    FillCommonKernel<ElemT><<<blocks, threads, 0, s>>>(static_cast<ElemT*>(buf), n, myPe);
    return hipSuccess;
  });
}

static void LaunchFillA2A(DataType dt, void* send, size_t chunkElems, int npes, int myPe,
                          int blocks, int threads, hipStream_t s) {
  (void)detail::DispatchReduceType(dt, [&](auto tag) {
    using ElemT = decltype(tag);
    FillA2AKernel<ElemT><<<blocks, threads, 0, s>>>(static_cast<ElemT*>(send), chunkElems, npes,
                                                    myPe);
    return hipSuccess;
  });
}

static void LaunchVerifyReduce(DataType dt, const void* output, size_t count, size_t globalOffset,
                               int npes, ReduceOpKind op, uint32_t* errorCount, int blocks,
                               int threads, hipStream_t s) {
  (void)detail::DispatchReduceType(dt, [&](auto tag) {
    using ElemT = decltype(tag);
    VerifyReduceKernel<ElemT><<<blocks, threads, 0, s>>>(static_cast<const ElemT*>(output), count,
                                                         globalOffset, npes, op, errorCount);
    return hipSuccess;
  });
}

static void LaunchVerifyGather(DataType dt, const void* output, size_t chunkElems, int npes,
                               uint32_t* errorCount, int blocks, int threads, hipStream_t s) {
  (void)detail::DispatchReduceType(dt, [&](auto tag) {
    using ElemT = decltype(tag);
    VerifyGatherKernel<ElemT><<<blocks, threads, 0, s>>>(static_cast<const ElemT*>(output),
                                                         chunkElems, npes, errorCount);
    return hipSuccess;
  });
}

static void LaunchVerifyA2A(DataType dt, const void* recv, size_t chunkElems, int npes, int myPe,
                            uint32_t* errorCount, int blocks, int threads, hipStream_t s) {
  (void)detail::DispatchReduceType(dt, [&](auto tag) {
    using ElemT = decltype(tag);
    VerifyA2AKernel<ElemT><<<blocks, threads, 0, s>>>(static_cast<const ElemT*>(recv), chunkElems,
                                                      npes, myPe, errorCount);
    return hipSuccess;
  });
}

static void LaunchVerifyPermute(DataType dt, const void* recv, size_t numElements, int srcPe,
                                uint32_t* errorCount, int blocks, int threads, hipStream_t s) {
  (void)detail::DispatchReduceType(dt, [&](auto tag) {
    using ElemT = decltype(tag);
    VerifyPermuteKernel<ElemT><<<blocks, threads, 0, s>>>(static_cast<const ElemT*>(recv),
                                                          numElements, srcPe, errorCount);
    return hipSuccess;
  });
}

struct ThreadInfo {
  int rank{-1};
  int worldSize{-1};
  int deviceId{-1};
  int ret_code{-1};
};

// ---------------------------------------------------------------------------
// Shared benchmark + verify loop (identical across collectives).
//   runOnce()      -> issues the collective on `stream`, returns hipError_t
//   launchVerify(d)-> launches the verify kernel accumulating into d (uint32*)
// ---------------------------------------------------------------------------
template <class RunOnceFn, class LaunchVerifyFn>
static int BenchAndVerify(const Config& cfg, ccoComm* comm, int myPe, hipStream_t stream,
                          double rxBytes, const char* opLabel, RunOnceFn runOnce,
                          LaunchVerifyFn launchVerify) {
  const int nWarmup = cfg.warmup, nRuns = cfg.iters;
  hipEvent_t tStart, tStop;
  HIP_RUNTIME_CHECK(hipEventCreate(&tStart));
  HIP_RUNTIME_CHECK(hipEventCreate(&tStop));

  float totalMs = 0, minMs = 1e9f, maxMs = 0;
  hipError_t launchErr = hipSuccess;
  for (int iter = 0; iter < nWarmup + nRuns; iter++) {
    ccoBarrierAll(comm);
    HIP_RUNTIME_CHECK(hipEventRecord(tStart, stream));
    launchErr = runOnce();
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

  // After every PE's last kernel has been stream-synced, each PE has observed all
  // incoming completion signals -> every outgoing DMA has landed. This host
  // barrier makes that global before teardown.
  ccoBarrierAll(comm);

  uint32_t hErrors = 0;
  if (launchErr != hipSuccess) {
    hErrors = 1;
    if (myPe == 0) XPUT("Rank %d: FAIL launch=%s", myPe, hipGetErrorString(launchErr));
  } else {
    uint32_t* dErrors;
    HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));
    launchVerify(dErrors);
    HIP_RUNTIME_CHECK(
        hipMemcpyAsync(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
    HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
    HIP_RUNTIME_CHECK(hipFree(dErrors));

    float avgMs = totalMs / nRuns;
    double avgBw = (rxBytes / 1e9) / (avgMs / 1e3);
    double maxBw = (rxBytes / 1e9) / (minMs / 1e3);
    if (hErrors != 0 || myPe == 0) {
      if (opLabel != nullptr) {
        XPUT("Rank %d: %s op=%s | %d warmup + %d runs | avg %.3f ms (%.3f GB/s) "
             "min %.3f ms (%.3f GB/s) max %.3f ms\n--------------------",
             myPe, hErrors == 0 ? "PASS" : "FAIL", opLabel, nWarmup, nRuns, avgMs, avgBw, minMs,
             maxBw, maxMs);
      } else {
        XPUT("Rank %d: %s | %d warmup + %d runs | avg %.3f ms (%.3f GB/s) "
             "min %.3f ms (%.3f GB/s) max %.3f ms\n--------------------",
             myPe, hErrors == 0 ? "PASS" : "FAIL", nWarmup, nRuns, avgMs, avgBw, minMs, maxBw,
             maxMs);
      }
    }
  }

  HIP_RUNTIME_CHECK(hipEventDestroy(tStart));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStop));
  return hErrors == 0 ? 0 : 1;
}

static size_t DtypeSize(DataType dt);  // defined below (CLI helpers)

// ---------------------------------------------------------------------------
// Per-collective body: allocate symmetric buffers, fill, benchmark + verify.
// Type-agnostic: buffers are raw bytes (the facade Run* methods dispatch dtype/op
// internally); only the local fill/verify launches need the element type, which
// they resolve via the Launch* wrappers above.
// ---------------------------------------------------------------------------
static int RunCollective(const Config& cfg, CollectivesFacade* facade, ccoComm* comm, int myPe,
                         int npes, hipStream_t stream) {
  constexpr int kThreads = 256;
  const size_t size = cfg.numElems;
  const size_t elemSize = DtypeSize(cfg.dt);
  const DataType dt = cfg.dt;
  auto allocBytes = [&](size_t nElems) -> void* {
    return CollectivesFacade::Allocate(nElems * elemSize);
  };
  auto blocksFor = [&](size_t n) {
    return static_cast<int>(std::min<size_t>(1024, (n + kThreads - 1) / kThreads));
  };

  switch (cfg.coll) {
    case Collective::kReduceScatter: {
      const size_t chunk = size / npes, N = size;
      void* in = allocBytes(N);
      void* out = allocBytes(chunk);
      if (in == nullptr || out == nullptr) { XPUT("ERROR: Allocate failed"); return 1; }
      HIP_RUNTIME_CHECK(hipMemsetAsync(in, 0, N * elemSize, stream));
      HIP_RUNTIME_CHECK(hipMemsetAsync(out, 0, chunk * elemSize, stream));
      LaunchFillCommon(dt, in, N, myPe, blocksFor(N), kThreads, stream);
      HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      return BenchAndVerify(
          cfg, comm, myPe, stream, static_cast<double>(N) * elemSize,
          detail::ReduceOpName(cfg.op),
          [&] { return facade->RunReduceScatter(in, out, chunk, cfg.dt, cfg.op, stream); },
          [&](uint32_t* e) {
            LaunchVerifyReduce(dt, out, chunk, static_cast<size_t>(myPe) * chunk, npes, cfg.op, e,
                               blocksFor(chunk), kThreads, stream);
          });
    }
    case Collective::kAllReduce: {
      const size_t N = size;
      void* in = allocBytes(N);
      void* out = allocBytes(N);
      if (in == nullptr || out == nullptr) { XPUT("ERROR: Allocate failed"); return 1; }
      HIP_RUNTIME_CHECK(hipMemsetAsync(in, 0, N * elemSize, stream));
      HIP_RUNTIME_CHECK(hipMemsetAsync(out, 0, N * elemSize, stream));
      LaunchFillCommon(dt, in, N, myPe, blocksFor(N), kThreads, stream);
      HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      return BenchAndVerify(
          cfg, comm, myPe, stream, static_cast<double>(N) * elemSize,
          detail::ReduceOpName(cfg.op),
          [&] { return facade->RunAllReduce(in, out, N, cfg.dt, cfg.op, stream); },
          [&](uint32_t* e) {
            LaunchVerifyReduce(dt, out, N, 0, npes, cfg.op, e, blocksFor(N), kThreads, stream);
          });
    }
    case Collective::kAllGather: {
      const size_t chunk = size / npes, N = size;
      const size_t chunkBytes = chunk * elemSize;
      void* in = allocBytes(chunk);
      void* out = allocBytes(N);
      if (in == nullptr || out == nullptr) { XPUT("ERROR: Allocate failed"); return 1; }
      HIP_RUNTIME_CHECK(hipMemsetAsync(in, 0, chunkBytes, stream));
      HIP_RUNTIME_CHECK(hipMemsetAsync(out, 0, N * elemSize, stream));
      LaunchFillCommon(dt, in, chunk, myPe, blocksFor(chunk), kThreads, stream);
      HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      return BenchAndVerify(
          cfg, comm, myPe, stream, static_cast<double>(npes - 1) * chunkBytes, nullptr,
          [&] { return facade->RunAllGather(in, out, chunkBytes, stream); },
          [&](uint32_t* e) {
            LaunchVerifyGather(dt, out, chunk, npes, e, blocksFor(N), kThreads, stream);
          });
    }
    case Collective::kAllToAll: {
      const size_t chunk = size / npes, N = size;
      const size_t chunkBytes = chunk * elemSize;
      void* send = allocBytes(N);
      void* recv = allocBytes(N);
      if (send == nullptr || recv == nullptr) { XPUT("ERROR: Allocate failed"); return 1; }
      HIP_RUNTIME_CHECK(hipMemsetAsync(send, 0, N * elemSize, stream));
      HIP_RUNTIME_CHECK(hipMemsetAsync(recv, 0, N * elemSize, stream));
      LaunchFillA2A(dt, send, chunk, npes, myPe, blocksFor(N), kThreads, stream);
      HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      auto* sendBytes = static_cast<uint8_t*>(send);
      auto* recvBytes = static_cast<uint8_t*>(recv);
      CollectivesFacade::AddressVector addrs(npes);
      for (int p = 0; p < npes; p++) {
        addrs[p] = {sendBytes + static_cast<size_t>(p) * chunkBytes,
                    recvBytes + static_cast<size_t>(p) * chunkBytes};
      }
      return BenchAndVerify(
          cfg, comm, myPe, stream, static_cast<double>(npes - 1) * chunkBytes, nullptr,
          [&] { return facade->RunAllToAll(addrs, chunkBytes, stream); },
          [&](uint32_t* e) {
            LaunchVerifyA2A(dt, recv, chunk, npes, myPe, e, blocksFor(N), kThreads, stream);
          });
    }
    case Collective::kCollectivePermute: {
      // Ring: send whole per-rank buffer to (myPe+1)%npes, recv from (myPe-1).
      const size_t n = size;  // per-rank element count
      const size_t numBytes = n * elemSize;
      const int dstPe = (myPe + 1) % npes;
      const int srcPe = (myPe + npes - 1) % npes;
      void* send = allocBytes(n);
      void* recv = allocBytes(n);
      if (send == nullptr || recv == nullptr) { XPUT("ERROR: Allocate failed"); return 1; }
      HIP_RUNTIME_CHECK(hipMemsetAsync(send, 0, numBytes, stream));
      HIP_RUNTIME_CHECK(hipMemsetAsync(recv, 0, numBytes, stream));
      LaunchFillCommon(dt, send, n, myPe, blocksFor(n), kThreads, stream);
      HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
      ccoBarrierAll(comm);
      return BenchAndVerify(
          cfg, comm, myPe, stream, static_cast<double>(numBytes), nullptr,
          [&] {
            return facade->RunCollectivePermute(send, recv, numBytes, srcPe, {dstPe}, stream);
          },
          [&](uint32_t* e) {
            LaunchVerifyPermute(dt, recv, n, srcPe, e, blocksFor(n), kThreads, stream);
          });
    }
  }
  return 1;
}

// ---------------------------------------------------------------------------
// Per-rank test body: one thread per GPU. Creates the comm, registers this
// device's facade, applies the runtime tuning and runs the collective. Teardown
// is deliberately left to main() -- see the note at the end of the function.
// ---------------------------------------------------------------------------
static void RunThreaded(const Config& cfg, const ccoUniqueId& uid, ThreadInfo& info) {
  HIP_RUNTIME_CHECK(hipSetDevice(info.deviceId));

  ccoComm* comm = nullptr;
  int status = ccoCommCreate(uid, info.worldSize, info.rank, cfg.perRankVmm, &comm);
  if (status != 0 || comm == nullptr) {
    XPUT("ERROR: ccoCommCreate failed (ret=%d)", status);
    info.ret_code = status != 0 ? status : -1;
    return;
  }

  const int myPe = info.rank, npes = info.worldSize;
  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  int rc = 1;
  if (CollectivesFacade::Create(comm, myPe, npes, cfg.heapBytes, cfg.stagingBytes) != 0) {
    XPUT("ERROR: Failed to create CollectivesFacade");
  } else {
    auto* facade = CollectivesFacade::Get();
    if (facade == nullptr) {
      XPUT("ERROR: No CollectivesFacade for device %d", info.deviceId);
    } else if (facade->SetReduceMode(cfg.mode) && facade->SetPushLogSlices(cfg.logS)) {
      rc = RunCollective(cfg, facade, comm, myPe, npes, stream);
    }
  }

  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  // No ccoCommDestroy here: Create hands `comm` to the per-device facade registry
  // (the facade dtor destroys it), and TearDown() clears every device at once, so
  // tearing down from this thread would destroy the sibling threads' comms too.
  // main() calls TearDown() once after joining.
  info.ret_code = rc;
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------
static bool ParseColl(const char* s, Collective& c) {
  if (!std::strcmp(s, "reduce_scatter") || !std::strcmp(s, "rs")) c = Collective::kReduceScatter;
  else if (!std::strcmp(s, "all_reduce") || !std::strcmp(s, "ar")) c = Collective::kAllReduce;
  else if (!std::strcmp(s, "all_gather") || !std::strcmp(s, "ag")) c = Collective::kAllGather;
  else if (!std::strcmp(s, "all_to_all") || !std::strcmp(s, "a2a")) c = Collective::kAllToAll;
  else if (!std::strcmp(s, "collective_permute") || !std::strcmp(s, "cp"))
    c = Collective::kCollectivePermute;
  else return false;
  return true;
}
static const char* CollName(Collective c) {
  switch (c) {
    case Collective::kReduceScatter: return "reduce_scatter";
    case Collective::kAllReduce: return "all_reduce";
    case Collective::kAllGather: return "all_gather";
    case Collective::kAllToAll: return "all_to_all";
    case Collective::kCollectivePermute: return "collective_permute";
  }
  return "?";
}
static bool ParseDtype(const char* s, DataType& dt) {
  if (!std::strcmp(s, "f32")) dt = DataType::F32;
  else if (!std::strcmp(s, "bf16")) dt = DataType::BF16;
  else if (!std::strcmp(s, "f16")) dt = DataType::F16;
  else if (!std::strcmp(s, "s32")) dt = DataType::S32;
  else if (!std::strcmp(s, "s64")) dt = DataType::S64;
  else return false;
  return true;
}
static size_t DtypeSize(DataType dt) {
#define ITEM(name, type) case DataType::name: return sizeof(type);
  switch (dt) {
    COLLECTIVE_DATA_TYPE_LIST(ITEM)
    default:
      XPUT("ERROR: bad dtype %s", detail::DataTypeName(dt));
      return 0;
  }
#undef ITEM
}

static bool ParseOp(const char* s, ReduceOpKind& op) {
  if (!std::strcmp(s, "sum")) op = ReduceOpKind::SUM;
  else if (!std::strcmp(s, "prod") || !std::strcmp(s, "product")) op = ReduceOpKind::PRODUCT;
  else if (!std::strcmp(s, "min")) op = ReduceOpKind::MIN;
  else if (!std::strcmp(s, "max")) op = ReduceOpKind::MAX;
  else return false;
  return true;
}
static bool ParseMode(const char* s, RsMode& m) {
  if (!std::strcmp(s, "push")) m = RsMode::kPush;
  else if (!std::strcmp(s, "pull")) m = RsMode::kPull;
  else return false;
  return true;
}

static void Usage(const char* prog) {
  XPUT("Usage: %s --coll <name> --npes <n> --size <num_elems> "
       "[--dtype f32|bf16|f16|s32|s64] [--op sum|prod|min|max] [--mode push|pull] "
       "[--logS <n>] [--warmup <n>] [--iters <n>]\n"
       "  coll: reduce_scatter|all_reduce|all_gather|all_to_all|collective_permute "
       "(rs|ar|ag|a2a|cp)",
       prog);
}

// ---------------------------------------------------------------------------
// Heap / VA sizing. The facade heap is a bump allocator that never reclaims, so
// it has to cover every buffer RunCollective allocates for the selected
// collective plus the push path's staging region (which Create carves as the
// first heap allocation). The comm's flat-VA slot must in turn hold the heap
// plus CCO's own internal windows.
// ---------------------------------------------------------------------------
static void SizeHeap(Config& cfg) {
  const size_t elemSize = DtypeSize(cfg.dt);
  const size_t npes = static_cast<size_t>(cfg.npes);
  const size_t N = cfg.numElems * elemSize;
  // For collective_permute --size is already the per-rank count, so there is no
  // sharding and the whole buffer is the unit that moves.
  const bool isPermute = (cfg.coll == Collective::kCollectivePermute);
  const size_t chunkBytes = isPermute ? N : N / npes;

  size_t bufBytes = 0;
  switch (cfg.coll) {
    case Collective::kReduceScatter: bufBytes = N + chunkBytes; break;
    case Collective::kAllGather: bufBytes = chunkBytes + N; break;
    case Collective::kAllReduce:
    case Collective::kAllToAll:
    case Collective::kCollectivePermute: bufBytes = 2 * N; break;
  }

  // Only the reduce paths stage through peer-writable scratch. The others still
  // get a token region: Create reads 0 as "use heapBytes / 4", so asking for
  // nothing would silently burn a quarter of the heap.
  const bool isReduce =
      (cfg.coll == Collective::kReduceScatter || cfg.coll == Collective::kAllReduce);
  const size_t staging = isReduce ? (npes - 1) * chunkBytes : 0;
  cfg.stagingBytes = std::max<size_t>(staging, CollectivesFacade::kDefAlign);

  cfg.heapBytes = AlignUp(bufBytes + cfg.stagingBytes + kHeapSlack, CollectivesFacade::kDefAlign);
  cfg.perRankVmm = AlignUp(cfg.heapBytes + kVmmSlack, kVmmGrain);
}

int main(int argc, char* argv[]) {
  int deviceCount = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));

  Config cfg;
  bool haveColl = false, haveNpes = false, haveSize = false;
  for (int i = 1; i < argc; i++) {
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { XPUT("ERROR: %s needs a value", name); std::exit(1); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--coll")) {
      if (!ParseColl(need("--coll"), cfg.coll)) { XPUT("ERROR: bad --coll"); return 1; }
      haveColl = true;
    } else if (!std::strcmp(argv[i], "--npes")) {
      cfg.npes = std::atoi(need("--npes"));
      haveNpes = true;
    } else if (!std::strcmp(argv[i], "--size")) {
      cfg.numElems = static_cast<size_t>(std::atoll(need("--size")));
      haveSize = true;
    } else if (!std::strcmp(argv[i], "--dtype")) {
      if (!ParseDtype(need("--dtype"), cfg.dt)) { XPUT("ERROR: bad --dtype"); return 1; }
    } else if (!std::strcmp(argv[i], "--op")) {
      if (!ParseOp(need("--op"), cfg.op)) { XPUT("ERROR: bad --op"); return 1; }
    } else if (!std::strcmp(argv[i], "--mode")) {
      if (!ParseMode(need("--mode"), cfg.mode)) { XPUT("ERROR: bad --mode"); return 1; }
    } else if (!std::strcmp(argv[i], "--logS")) {
      cfg.logS = std::atoi(need("--logS"));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      cfg.warmup = std::atoi(need("--warmup"));
    } else if (!std::strcmp(argv[i], "--iters")) {
      cfg.iters = std::atoi(need("--iters"));
    } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      Usage(argv[0]);
      return 0;
    } else {
      XPUT("ERROR: unknown arg '%s'", argv[i]);
      Usage(argv[0]);
      return 1;
    }
  }

  if (!haveColl || !haveNpes || !haveSize) {
    XPUT("ERROR: --coll, --npes and --size are required");
    Usage(argv[0]);
    return 1;
  }
  if (cfg.npes < 1 || cfg.npes > 8 || cfg.npes > deviceCount) {
    XPUT("ERROR: --npes must be in 1..%d (SDMA fast path caps at 8)", deviceCount);
    return 1;
  }

  // Buffer-size validation (per-collective element size from dtype).
  const size_t elemSize = DtypeSize(cfg.dt);
  if (cfg.coll == Collective::kCollectivePermute) {
    const size_t nb = cfg.numElems * elemSize;
    if (nb < 16 || (nb % 16) != 0) {
      XPUT("ERROR: per-rank bytes (size * sizeof) must be a multiple of 16");
      return 1;
    }
  } else {
    if (cfg.numElems % static_cast<size_t>(cfg.npes) != 0) {
      XPUT("ERROR: --size must be divisible by --npes");
      return 1;
    }
    const size_t chunkBytes = (cfg.numElems / cfg.npes) * elemSize;
    if (chunkBytes < 16 || (chunkBytes % 16) != 0) {
      XPUT("ERROR: per-shard bytes (size/npes * sizeof) must be a multiple of 16");
      return 1;
    }
  }

  // The Launch* wrappers dispatch dtype internally and would silently no-op on a
  // type this build does not instantiate, so probe once up front (rank-independent)
  // and emit the actionable diagnostic instead of reporting a bogus verify failure.
  if (detail::DispatchReduceType(cfg.dt, [](auto) { return hipSuccess; }) == hipErrorNotSupported) {
    XPUT("ERROR: dtype %s not supported by this build "
         "(rebuild with FACADE_REDUCE_USE_ALL_TYPES=1)",
         detail::DataTypeName(cfg.dt));
    return 1;
  }

  SizeHeap(cfg);

  ccoUniqueId uid;
  int ret = ccoGetUniqueId(&uid);
  if (ret != 0) {
    XPUT("ERROR: ccoGetUniqueId failed (ret=%d) -- set MORI_SOCKET_IFNAME=<iface>", ret);
    return 1;
  }

  XPUT("collectives_benchmark: coll=%s npes=%d size=%zu dtype=%s op=%s mode=%s logS=%d",
       CollName(cfg.coll), cfg.npes, cfg.numElems, detail::DataTypeName(cfg.dt),
       detail::ReduceOpName(cfg.op), cfg.mode == RsMode::kPull ? "pull" : "push", cfg.logS);
  XPUT("collectives_benchmark: heap=%zu staging=%zu perRankVmm=%zu", cfg.heapBytes,
       cfg.stagingBytes, cfg.perRankVmm);

  std::vector<std::thread> threads;
  std::vector<ThreadInfo> infos(cfg.npes);
  threads.reserve(cfg.npes);
  for (int i = 0; i < cfg.npes; i++) {
    infos[i].rank = i;
    infos[i].worldSize = cfg.npes;
    infos[i].deviceId = i;
    threads.emplace_back(RunThreaded, std::cref(cfg), std::cref(uid), std::ref(infos[i]));
  }
  for (auto& t : threads) t.join();

  // Registry-wide: destroys every device's heap window, SDMA DevComm and ccoComm.
  // Must happen after the join, and exactly once.
  CollectivesFacade::TearDown();

  for (const auto& inf : infos) {
    if (inf.ret_code != 0) {
      XPUT("ERROR: Rank %d returned non-zero ret_code %d", inf.rank, inf.ret_code);
      return 1;
    }
  }
  return 0;
}

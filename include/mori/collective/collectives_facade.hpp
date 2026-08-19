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

// ===========================================================================
// collectives_facade.hpp
//
// Header-only central facade for the XLA push/pull collectives. It owns the
// per-device staging buffer (symmetric heap) and the per-slice group counters,
// and exposes Run* entry points that hoist all the host-side sizing/launch
// logic (block count, slice count, RS_MODE/RS_LOG_PUSH_SLICES env handling)
// that used to be duplicated in each test.
//
// CollectivesFacade is a per-device singleton, mirroring ShmemStatesSingleton
// (one slot per GPU, indexed by hipGetDevice(); the SPMT contract is one thread
// per GPU so each slot is accessed serially by its owning thread -- no lock).
//
// Fully header-only on purpose: all MORI device code (the templated __global__
// kernels) is compiled only by the translation unit that includes this header,
// so no MORI .cpp ever needs device compilation (keeps the Bazel/XLA wiring
// simple -- headers are enough).
// ===========================================================================
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include "mori/application/utils/check.hpp"  // HIP_RUNTIME_CHECK
#include "mori/shmem/shmem_api.hpp"          // host-safe shmem API (ShmemMalloc, ...)

#include "mori/collective/XLA/collectives_common.hpp"

// The device kernels and the reduction-op templates (SumOp/MaxOp/...) live in
// these headers. They are needed only by the single TU that compiles the Run*
// definitions -- the one that defines MORI_KERNELS_IMPL (the XLA device TU
// mori_kernels.cu.cc, or each standalone MORI example binary). Host TUs see only
// the declarations below and link against that TU's symbols.
#if defined(MORI_KERNELS_IMPL)
#include <hip/hip_bfloat16.h>
#include "mori/collective/XLA/all_gather_kernels.hpp"
#include "mori/collective/XLA/all_reduce_kernels.hpp"
#include "mori/collective/XLA/all_to_all_kernels.hpp"
#include "mori/collective/XLA/collective_permute_kernels.hpp"
#include "mori/collective/XLA/reduce_scatter_kernels.hpp"
#endif  // MORI_KERNELS_IMPL

#define FACADE_PRINTF(fmt, ...) std::fprintf(stderr, fmt"\n", ##__VA_ARGS__)
namespace mori {
namespace collective {

// Numeric element type of the reduce collectives, expressed as a plain enum so
// the facade public API stays non-templated. The type/op -> kernel dispatch
// happens inside the (device) Run* definitions.
enum class DataType {
  F8E5M2, F8E4M3FN, F16, BF16, S8, U8, S32, U32, S64, U64, F32, F64
};

// Reduction operation for the reduce collectives.
enum class ReduceOpKind { SUM, PRODUCT, MIN, MAX };

class CollectivesFacade {

  CollectivesFacade() = default;
 public:
  // Mode selection: RS_MODE = push|pull (default push). RS_PULL=1 back-compat.
  enum class RsMode { kPush, kPull };

  // Per-peer all-to-all endpoints: addrs[p] = {chunk sent to peer p, recv slot
  // for sender p}. The facade copies these into its pinned AddressPair buffer.
  using AddressVector = std::vector<std::pair<const void*, void*>>;

  CollectivesFacade(const CollectivesFacade&) = delete;
  CollectivesFacade& operator=(const CollectivesFacade&) = delete;
  
  // Record this device's rank identity (myPe, nPes) and allocate
  // the per-device staging buffer (symmetric heap, so the address-based SDMA put
  // can translate local->peer) and the per-slice group counters. Staging must be
  // sized to the largest use: maxStagingBytes >= (nPes-1) * maxChunkBytes for the
  // reduce-scatter / all-reduce push paths; pass 0 for the compute-free
  // collectives (all-gather / all-to-all) which need no staging. Returns nullptr
  // on failure. Host-only (no kernel launches), so it stays on the unconditional
  // path and is safe to call from a host TU. Must be called after ShmemInit.
  static std::unique_ptr<CollectivesFacade> Create(int myPe, int nPes,
                                                    size_t maxStagingBytes) {
    assert(mori::shmem::ShmemIsInitialized() && "Create requires ShmemInit first");

    auto facade = std::unique_ptr<CollectivesFacade>(new CollectivesFacade());
    facade->myPe_ = myPe;
    facade->nPes_ = nPes;

    hipDeviceProp_t prop;
    HIP_RUNTIME_CHECK(hipGetDeviceProperties(&prop, myPe));
    facade->MP_count_ = prop.multiProcessorCount;

    if (const char* s = std::getenv("RS_LOG_PUSH_SLICES")) {
      facade->logS_ = std::max(0, std::atoi(s));
    }
    if ((1 << facade->logS_) > kRSPushMaxSlices || facade->nPes_ > kRSPushMaxPeers) {
      FACADE_PRINTF("CollectivesFacade: logS too large or npes too large");
      return {};
    }

    if (const char* m = std::getenv("RS_MODE")) {
      if (std::strcmp(m, "pull") == 0)
        facade->mode_ = RsMode::kPull;
      else if (std::strcmp(m, "push") == 0)
        facade->mode_ = RsMode::kPush;
    }
    if (facade->mode_ == RsMode::kPull && facade->nPes_ > 8) {
      FACADE_PRINTF("CollectivesFacade: pull mode supports npes in [1,8]");
      return {};
    }

    if (maxStagingBytes > 0) {
      facade->staging_ = mori::shmem::ShmemMalloc(maxStagingBytes);
      if (facade->staging_ == nullptr) {
        FACADE_PRINTF("CollectivesFacade: failed to allocate staging buffer");
        return {};
      }
      facade->stagingBytes_ = maxStagingBytes;
    }
    HIP_RUNTIME_CHECK(hipMalloc(reinterpret_cast<void**>(&facade->groupCounters_),
                                kRSPushMaxSlices * sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipMemset(facade->groupCounters_, 0, kRSPushMaxSlices * sizeof(uint32_t)));
    // Preallocate the pinned (host-visible, device-readable) all-to-all pointer
    // buffer once, so each RunAllToAll is a plain host fill + launch (no per-call
    // device alloc/copy). Sized for the max supported peer count.
    HIP_RUNTIME_CHECK(hipHostMalloc(reinterpret_cast<void**>(&facade->pinnedPairs_),
                                    kRSPushMaxPeers * sizeof(AddressPair), 
                                    hipHostMallocDefault));
    return facade;
  }

  // Release this device's staging + counters. Must run before ShmemFinalize
  // (staging lives on the symmetric heap).
  ~CollectivesFacade() {
    if (staging_ != nullptr) mori::shmem::ShmemFree(staging_);
    if (groupCounters_ != nullptr) HIP_RUNTIME_CHECK(hipFree(groupCounters_));
    if (pinnedPairs_ != nullptr) HIP_RUNTIME_CHECK(hipHostFree(pinnedPairs_));
  }

  // Reduce-scatter: input is the full N = npes*chunk vector (symmetric heap),
  // output is this PE's reduced shard. `count` is the per-rank shard element
  // count (chunk = count); the full input is npes*count. RS_MODE=push|pull
  // selects the algorithm; RS_LOG_PUSH_SLICES tunes the push slice count. `dt`
  // and `op` select the element type and reduction; the definition dispatches to
  // the matching kernel (currently only F32/kSum is implemented).
  hipError_t RunReduceScatter(const void* input, void* output, size_t count,
                              DataType dt, ReduceOpKind op, hipStream_t stream);

  // All-reduce: input is the full N = npes*chunk vector (symmetric heap), output
  // is the full reduced N vector (symmetric heap). numElems is the TOTAL element
  // count (chunk = numElems / npes). Push-only (npes in [1,8]). `dt`/`op` select
  // the element type and reduction (currently only F32/kSum is implemented).
  hipError_t RunAllReduce(const void* input, void* output, size_t numElems,
                          DataType dt, ReduceOpKind op, hipStream_t stream);

  // All-gather: input is this PE's single shard (symmetric heap), output is the
  // gathered npes*chunk buffer (symmetric heap). Pure byte movement (element type
  // irrelevant); `chunkBytes` is this PE's per-rank shard size in bytes (the
  // gathered output is npes*chunkBytes).
  hipError_t RunAllGather(const void* input, void* output, size_t chunkBytes, hipStream_t stream);

  // All-to-all: addrs[p] = {chunk sent to peer p, recv slot for sender p}, one
  // pair per peer (addrs.size() == npes). Pure byte movement, so chunkBytes is
  // the per-peer chunk byte count.
  hipError_t RunAllToAll(const AddressVector& addrs, size_t chunkBytes, hipStream_t stream);

  // Device-wide barrier across all PEs on the given stream.
  hipError_t RunBarrier(hipStream_t stream);

  // Point-to-point send: push numBytes from sendBuf to PE `peer`. Dummy no-op.
  hipError_t RunSend(const void* sendBuf, size_t numBytes, int peer, hipStream_t stream);

  // Point-to-point recv: receive numBytes into recvBuf from PE `peer`. Dummy no-op.
  hipError_t RunRecv(void* recvBuf, size_t numBytes, int peer, hipStream_t stream);

  // Collective-permute: send my buffer to each target PE, recv from srcPe
  // (srcPe < 0 means no source). v1: dstPes.size() == 1, SDMA push.
  hipError_t RunCollectivePermute(const void* sendBuf, void* recvBuf, size_t numBytes,
                                  int srcPe, const std::vector<int>& dstPes,
                                  hipStream_t stream);

  // Drain outstanding operations on `stream`. Dummy no-op.
  hipError_t RunQuiet(hipStream_t stream);

  // Memory fence across the device. Dummy no-op.
  hipError_t RunFence();

 private:
  // Real reduce implementations, keyed by element type T and reduction Op. Defined
  // only in the MORI_KERNELS_IMPL TU; the enum Run* entry points dispatch to them.
  template <class T, class Op>
  hipError_t reduceScatterImpl(const void* input, void* output, size_t count, hipStream_t stream);
  template <class T, class Op>
  hipError_t allReduceImpl(const void* input, void* output, size_t numElems, hipStream_t stream);

  void* staging_{nullptr};
  size_t stagingBytes_{0};
  uint32_t* groupCounters_{nullptr};
  AddressPair* pinnedPairs_{nullptr};  // host-pinned, device-readable
  int myPe_{0};
  int nPes_{0};
  int logS_{0};
  int MP_count_{0};
  RsMode mode_{RsMode::kPush};
};

// ---------------------------------------------------------------------------
// Run* definitions. These contain kernel-launch syntax and pull in the
// __global__ kernel headers, so they are compiled only by the single TU that
// defines MORI_KERNELS_IMPL (a HIP compiler). Host TUs see only the declarations
// above and link against that TU's non-template symbols.
// ---------------------------------------------------------------------------
#if defined(MORI_KERNELS_IMPL)

namespace detail {
// Device-wide barrier: one thread issues the cross-PE shmem barrier.
__global__ void BarrierKernel() { mori::shmem::ShmemBarrierAllThread(); }
}  // namespace detail

template <class T, class Op>
hipError_t CollectivesFacade::reduceScatterImpl(const void* input_v, void* output_v, size_t count,
                                                hipStream_t stream) {
  const T* input = static_cast<const T*>(input_v);
  T* output = static_cast<T*>(output_v);
  using ComputeT = typename ReduceComputeType<T>::type;
  using ReduceOp = Op;

  const size_t chunkElems = count;
  const size_t chunkBytes = chunkElems * sizeof(T);
  if ((nPes_ - 1) * chunkBytes > stagingBytes_) {
    FACADE_PRINTF("ReduceScatter: staging too small; increase maxStagingBytes");
    return hipErrorInvalidConfiguration;
  }
  T* staging = reinterpret_cast<T*>(staging_);

  // The reduce runs on a compute type derived from T (float reduces as float,
  // bf16 reduces as a packed pair). Data buffers stay physically T; counts
  // passed to the kernels are in packs (kPack = sizeof(ComputeT)/sizeof(T)).
  constexpr size_t kPack = sizeof(ComputeT) / sizeof(T);
  const size_t chunkElemsC = chunkElems / kPack;
  constexpr int NumPushVecs = 8, NumPullVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ComputeT);
  constexpr int kThreads = 256;
  size_t totalVecs = chunkElemsC / (VecSize * NumPushVecs);
  int wantBlocks = static_cast<int>(std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  int blocks = std::min(wantBlocks, std::max(1, MP_count_));

  int logS = logS_;
  const size_t maxSlicesByData = std::max<size_t>(1, chunkElemsC / VecSize);
  while (logS > 0 && (1ULL << logS) > maxSlicesByData) logS--;

  int pushSlices = 1 << logS;
  int pushBlocks = std::max(pushSlices, (blocks / pushSlices) * pushSlices);

  if (mode_ == RsMode::kPull) {
    // input is its own symmetric allocation, so peerPtrs[pe] is peer pe's input
    // base. NPES is a compile-time template arg; dispatch on the real npes.
    mori::application::SymmMemObjPtr srcObj =
        mori::shmem::ShmemQueryMemObjPtr(const_cast<T*>(input));
    auto launch = [&](auto NPES_c) {
      ReduceScatterPullKernel<NumPullVecs, decltype(NPES_c)::value, ReduceOp>
          <<<blocks, kThreads, 0, stream>>>(myPe_, srcObj, reinterpret_cast<ComputeT*>(output),
                                            chunkElemsC);
    };
    switch (nPes_) {
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 4: launch(std::integral_constant<int, 4>{}); break;
      default: launch(std::integral_constant<int, 8>{}); break;
    }
  } else {
    ReduceScatterPushKernel<NumPushVecs, ReduceOp><<<pushBlocks, kThreads, 0, stream>>>(
        myPe_, nPes_, logS, reinterpret_cast<const ComputeT*>(input),
        reinterpret_cast<ComputeT*>(staging), reinterpret_cast<ComputeT*>(output),
        groupCounters_, chunkElemsC);
  }
  return hipGetLastError();
}

// Enum entry point: dispatch (dt, op) to the implemented reduceScatterImpl. Only
// F32/kSum is wired up today; add cases here as more kernels land.
hipError_t CollectivesFacade::RunReduceScatter(const void* input, void* output, size_t count,
                                               DataType dt, ReduceOpKind op, hipStream_t stream) {
  if (dt == DataType::F32 && op == ReduceOpKind::SUM)
    return reduceScatterImpl<float, ::SumOp<float>>(input, output, count, stream);
  return hipErrorNotSupported;
}

template <class T, class Op>
hipError_t CollectivesFacade::allReduceImpl(const void* input_v, void* output_v, size_t numElems,
                                            hipStream_t stream) {
  const T* input = static_cast<const T*>(input_v);
  T* output = static_cast<T*>(output_v);
  using ComputeT = typename ReduceComputeType<T>::type;
  using ReduceOp = Op;

  const size_t chunkElems = numElems / static_cast<size_t>(nPes_);
  const size_t chunkBytes = chunkElems * sizeof(T);
  if ((nPes_ - 1) * chunkBytes > stagingBytes_) {
    FACADE_PRINTF("AllReduce: staging too small; increase maxStagingBytes");
    return hipErrorInvalidConfiguration;
  }

  T* staging = reinterpret_cast<T*>(staging_);
  constexpr size_t kPack = sizeof(ComputeT) / sizeof(T);
  const size_t chunkElemsC = chunkElems / kPack;
  constexpr int NumPushVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ComputeT);
  constexpr int kThreads = 256;
  size_t totalVecs = chunkElemsC / (VecSize * NumPushVecs);
  int wantBlocks = static_cast<int>(std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  // Cap the grid to the SM count so all blocks are co-resident (required for the
  // multi-producer broadcast submitPacket ordering).
  int blocks = std::min(wantBlocks, std::max(1, MP_count_));

  int logS = logS_;
  const size_t maxSlicesByData = std::max<size_t>(1, chunkElemsC / VecSize);
  while (logS > 0 && (1ULL << logS) > maxSlicesByData) logS--;

  int pushSlices = 1 << logS;
  int pushBlocks = std::max(pushSlices, (blocks / pushSlices) * pushSlices);

  // Sliced push reduce-scatter + pipelined per-slice broadcast, all in one kernel.
  AllReducePushKernel<NumPushVecs, ReduceOp><<<pushBlocks, kThreads, 0, stream>>>(
        myPe_, nPes_, logS, reinterpret_cast<const ComputeT*>(input),
        reinterpret_cast<ComputeT*>(staging), reinterpret_cast<ComputeT*>(output), groupCounters_,
        chunkElemsC);
  return hipGetLastError();
}

// Enum entry point: dispatch (dt, op) to the implemented allReduceImpl. Only
// F32/kSum is wired up today; add cases here as more kernels land.
hipError_t CollectivesFacade::RunAllReduce(const void* input, void* output, size_t numElems,
                                           DataType dt, ReduceOpKind op, hipStream_t stream) {
  if (dt == DataType::F32 && op == ReduceOpKind::SUM)
    return allReduceImpl<float, ::SumOp<float>>(input, output, numElems, stream);
  return hipErrorNotSupported;
}

hipError_t CollectivesFacade::RunAllGather(const void* input, void* output, size_t chunkBytes,
                                           hipStream_t stream) {
  constexpr int kThreads = 256;
  // Single block: SDMA pushes my shard to every peer's output[myPe] slot (+ self).
  AllGatherPushKernel<<<1, kThreads, 0, stream>>>(myPe_, nPes_, input, output, chunkBytes);
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunAllToAll(const AddressVector& addrs, size_t chunkBytes,
                                          hipStream_t stream) {
  assert(static_cast<int>(addrs.size()) == nPes_ && "addrs size must equal npes");
  // Fill the preallocated pinned (host-visible, device-readable) AddressPair
  // buffer with a plain host write, then launch -- no per-call device copy.
  for (int p = 0; p < nPes_; ++p) {
    pinnedPairs_[p].source = addrs[p].first;
    pinnedPairs_[p].dest = addrs[p].second;
  }
  constexpr int kThreads = 256;
  // Single block: SDMA pushes each send slot p to peer p's recv slot (+ self).
  AllToAllPushKernel<<<1, kThreads, 0, stream>>>(myPe_, nPes_, pinnedPairs_, chunkBytes);
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunBarrier(hipStream_t stream) {
  detail::BarrierKernel<<<1, 1, 0, stream>>>();
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunSend(const void* /*sendBuf*/, size_t /*numBytes*/,
                                      int /*peer*/, hipStream_t /*stream*/) {
  // TODO: dummy placeholder; real P2P send push kernel goes here.
  return hipErrorNotSupported;
}

hipError_t CollectivesFacade::RunRecv(void* /*recvBuf*/, size_t /*numBytes*/,
                                      int /*peer*/, hipStream_t /*stream*/) {
  // TODO: dummy placeholder; real P2P recv kernel goes here.
  return hipErrorNotSupported;
}

hipError_t CollectivesFacade::RunCollectivePermute(const void* sendBuf, void* recvBuf,
                                                   size_t numBytes, int srcPe,
                                                   const std::vector<int>& dstPes,
                                                   hipStream_t stream) {
  if (dstPes.size() != 1) {
    FACADE_PRINTF("CollectivePermute: only dstPes.size()==1 is implemented");
    return hipErrorNotSupported;
  }
  const int dstPe = dstPes[0];
  if (dstPe < 0 || dstPe >= nPes_ || srcPe < -1 || srcPe >= nPes_) {
    FACADE_PRINTF("CollectivePermute: dstPe=%d srcPe=%d out of range (nPes=%d)", dstPe, srcPe,
                  nPes_);
    return hipErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  CollectivePermutePushKernel<<<1, kThreads, 0, stream>>>(nPes_, dstPe, srcPe, sendBuf, recvBuf,
                                                          numBytes);
  return hipGetLastError();
}

hipError_t CollectivesFacade::RunQuiet(hipStream_t /*stream*/) {
  // TODO: dummy placeholder; real quiet/drain goes here.
  return hipSuccess;
}

hipError_t CollectivesFacade::RunFence() {
  // TODO: dummy placeholder; real device fence goes here.
  return hipSuccess;
}

#endif  // MORI_KERNELS_IMPL

}  // namespace collective
}  // namespace mori

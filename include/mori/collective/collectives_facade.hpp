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
// logic (block count, slice count, push/pull mode) that used to be duplicated
// in each test.
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
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include "mori/cco/cco.hpp"                  // ccoComm / ccoWindow_t
#include "mori/application/utils/check.hpp"  // HIP_RUNTIME_CHECK
#include "mori/core/utils/utils.hpp"         // MORI_UNLIKELY

#include "mori/collective/XLA/reduce_ops.hpp"
#include "mori/collective/XLA/collectives_common.hpp"

// The device kernels and the reduction-op templates (SumOp/MaxOp/...) live in
// these headers. They are needed only by the single TU that compiles the Run*
// definitions -- the one that defines MORI_KERNELS_IMPL (the XLA device TU
// mori_kernels.cu.cc, or each standalone MORI example binary). Host TUs see only
// the declarations below and link against that TU's symbols.
//
// Device compilation is restricted to gfx942 / gfx950 / gfx1250. Other offload
// arches skip the kernel headers (and get Run* stubs below) so fatbin builds
// do not pull SDMA/b128 device code.
#if defined(MORI_KERNELS_IMPL)
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx942__) || defined(__gfx950__) || \
    defined(__gfx1250__)
#define MORI_FACADE_COLLECTIVES_ARCH 1
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include "mori/collective/XLA/all_gather_kernels.hpp"
#include "mori/collective/XLA/all_reduce_kernels.hpp"
#include "mori/collective/XLA/all_to_all_kernels.hpp"
#include "mori/collective/XLA/collective_permute_kernels.hpp"
#include "mori/collective/XLA/reduce_scatter_kernels.hpp"
#endif
#endif  // MORI_KERNELS_IMPL

#define FACADE_PRINTF(fmt, ...) std::fprintf(stderr, fmt"\n", ##__VA_ARGS__)
namespace mori {
namespace collective {

class CollectivesFacade {

  CollectivesFacade() = default;
 public:
  // Reduce-scatter algorithm selection (default push). Set at runtime via
  // SetReduceMode(); the push slice count via SetPushLogSlices().
  enum class RsMode { kPush, kPull };

  static constexpr size_t kDefAlign = 256;

  // Per-peer all-to-all endpoints: addrs[p] = {chunk sent to peer p, recv slot
  // for sender p}. The facade copies these into its pinned AddressPair buffer.
  using AddressVector = std::vector<std::pair<const void*, void*>>;

  CollectivesFacade(const CollectivesFacade&) = delete;
  CollectivesFacade& operator=(const CollectivesFacade&) = delete;

  static CollectivesFacade *Get() {
    int dev = 0;
    HIP_RUNTIME_CHECK(hipGetDevice(&dev));
    std::lock_guard<std::mutex> lock(GetMutex());
    if (MORI_UNLIKELY(GetInstances().size() <= static_cast<size_t>(dev))) {
      return nullptr;
    }
    return GetInstances()[dev].get();
  }

  // CCO-backed factory. Records this device's rank identity (myPe, nPes) and the
  // comm, and registers ONE big symmetric CCO "heap" window based on CCO comm
  // size. Need to be called by every rank in parallel.
  static int Create(mori::cco::ccoComm* comm, int myPe, int nPes, 
                    size_t heapBytes, size_t maxStagingBytes = 0) {
    if (comm == nullptr) {
      FACADE_PRINTF("CollectivesFacade: null comm for CCO create");
      return -1;
    }
    
    if (maxStagingBytes == 0) {
      maxStagingBytes = heapBytes / 4;
    }
    maxStagingBytes = (maxStagingBytes + kDefAlign - 1) & ~(kDefAlign - 1);
    if (heapBytes == 0 || maxStagingBytes >= heapBytes) {
      FACADE_PRINTF("CollectivesFacade: wrong heap size or maxStagingBytes");
      return -1;
    }

    int dev = 0;
    HIP_RUNTIME_CHECK(hipGetDevice(&dev));
    auto& facade = CreateInternal(dev);
    facade.myPe_ = myPe;
    facade.nPes_ = nPes;
    facade.ccoComm_ = comm;

    {
      int rc = mori::cco::ccoWindowRegister(comm, heapBytes, &facade.heapWin_,
                                            reinterpret_cast<void**>(&facade.heapBase_));
      if (rc != 0 || facade.heapBase_ == nullptr) {
        FACADE_PRINTF("CollectivesFacade: failed to register heap window (%zu bytes, ret=%d)",
                      heapBytes, rc);
        return -1;
      }
      facade.heapSize_ = heapBytes;
      facade.heapUsed_ = 0;
    }

    hipDeviceProp_t prop;
    HIP_RUNTIME_CHECK(hipGetDeviceProperties(&prop, dev));
    facade.MP_count_ = prop.multiProcessorCount;

    // Tuning (reduce-scatter mode / push slice count) defaults to push, logS=0.
    // Callers override at runtime via SetReduceMode() / SetPushLogSlices().
    if (facade.nPes_ > kRSPushMaxPeers) {
      FACADE_PRINTF("CollectivesFacade: npes too large (max %d)", kRSPushMaxPeers);
      return -1;
    }

    // Staging is the push path's peer-writable scratch (SDMA scatter target). It
    // lives in the shared heap window as the facade's FIRST internal allocation,
    // so its heap offset is identical on every rank (symmetric) since every rank
    // runs the same Create. User Allocate calls follow it. Pull needs none.
    if (maxStagingBytes > 0) {
      facade.staging_ = facade.Allocate(maxStagingBytes);
      if (facade.staging_ == nullptr) {
        FACADE_PRINTF("CollectivesFacade: failed to carve staging from heap (%zu bytes)",
                      maxStagingBytes);
        return -1;
      }
      facade.stagingBytes_ = maxStagingBytes;
    }
    HIP_RUNTIME_CHECK(hipMalloc(reinterpret_cast<void**>(&facade.groupCounters_),
                                kRSPushMaxSlices * sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipMemset(facade.groupCounters_, 0, kRSPushMaxSlices * sizeof(uint32_t)));
    // Preallocate the pinned (host-visible, device-readable) all-to-all pointer
    // buffer once, so each RunAllToAll is a plain host fill + launch (no per-call
    // device alloc/copy). Sized for the max supported peer count.
    HIP_RUNTIME_CHECK(hipHostMalloc(reinterpret_cast<void**>(&facade.pinnedPairs_),
                                    kRSPushMaxPeers * sizeof(AddressPair), 
                                    hipHostMallocDefault));

    // SDMA device comm for the push path (passed by value into the push kernel).
    mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
    reqs.gdaContextCount = 0;
    reqs.gdaSignalCount = 0;
    reqs.gdaCounterCount = 0;
    reqs.sdmaQueueCount = 0; // Use the context's SDMA queue count
    int ret = mori::cco::ccoDevCommCreate(comm, &reqs, &facade.devComm_);
    if (ret != 0 || facade.devComm_.sdma.sdmaNumQueue == 0) {
      FACADE_PRINTF("CollectivesFacade: ccoDevCommCreate failed or "
                    "no SDMA queues allocated.");
      return -1;
    }
    return 0;
  }

  // Release this device's heap window, counters, and SDMA comm. Staging is a
  // sub-region of heapWin_, so it needs no separate deregistration.
  ~CollectivesFacade() {
    if (heapWin_ != nullptr) mori::cco::ccoWindowDeregister(ccoComm_, heapWin_);
    if (ccoComm_ != nullptr) {
      mori::cco::ccoDevCommDestroy(ccoComm_, &devComm_);
      mori::cco::ccoCommDestroy(ccoComm_);
    }
    if (groupCounters_ != nullptr) HIP_RUNTIME_CHECK(hipFree(groupCounters_));
    if (pinnedPairs_ != nullptr) HIP_RUNTIME_CHECK(hipHostFree(pinnedPairs_));
  }

  // Tear down all collectives instances (should be safe to be called from 
  // a single thread).
  static void TearDown() {
    std::lock_guard<std::mutex> lock(GetMutex());
    GetInstances().clear();
  }

  // Bump-allocate a peer-visible buffer from the static heap window. Returns a
  // raw local pointer (heapBase_ + offset). Every rank MUST issue the same
  // Allocate sequence (same sizes/order) so a given buffer sits at the same heap
  // offset on all ranks (symmetric addressing). Returns nullptr if the heap is
  // exhausted. `align` defaults to 256B (SDMA/b128 friendly).
  static void* Allocate(size_t bytes) {
    auto *facade = Get();
    if (MORI_UNLIKELY(facade == nullptr)) {
      FACADE_PRINTF("CollectivesFacade::Allocate: no instance");
      return nullptr;
    }
    const size_t off = (facade->heapUsed_ + (kDefAlign - 1)) & ~(kDefAlign - 1);
    if (MORI_UNLIKELY(off + bytes > facade->heapSize_)) {
      FACADE_PRINTF("CollectivesFacade::Allocate: heap exhausted (need %zu at off %zu, size %zu)",
                    bytes, off, facade->heapSize_);
      return nullptr;
    }
    facade->heapUsed_ = off + bytes;
    return facade->heapBase_ + off;
  }

  // Bump allocator: individual frees are a no-op. Use Reset() to reclaim the
  // whole heap between phases (all outstanding buffers become invalid).
  static int Deallocate(void* ptr) {
    auto *facade = Get();
    if (MORI_UNLIKELY(facade == nullptr)) {
      FACADE_PRINTF("CollectivesFacade::Deallocate: no instance");
      return -1;
    }
    return 0;
  }

  static void Reset() {
    auto *facade = Get();
    if (MORI_UNLIKELY(facade == nullptr)) {
      FACADE_PRINTF("CollectivesFacade::Reset: no instance");
      return;
    }
    facade->heapUsed_ = 0;
  }

  // Runtime tuning "backdoors" (replace the former RS_MODE / RS_LOG_PUSH_SLICES
  // env vars). Call after Create, before the Run*. Both return false and leave
  // the current value unchanged on an invalid request.
  bool SetReduceMode(RsMode mode) {
    if (mode == RsMode::kPull && nPes_ > 8) {
      FACADE_PRINTF("CollectivesFacade: pull mode supports npes in [1,8]");
      return false;
    }
    mode_ = mode;
    return true;
  }
  bool SetPushLogSlices(int logS) {
    if (logS < 0 || (1 << logS) > kRSPushMaxSlices) {
      FACADE_PRINTF("CollectivesFacade: logS out of range (0..%d)",
                    kRSPushMaxSlices > 0 ? 31 : 0);
      return false;
    }
    logS_ = logS;
    return true;
  }
  RsMode GetReduceMode() const { return mode_; }

  // Reduce-scatter: input is the full N = npes*chunk vector (symmetric heap),
  // output is this PE's reduced shard. `count` is the per-rank shard element
  // count (chunk = count); the full input is npes*count. SetReduceMode() selects
  // push|pull; SetPushLogSlices() tunes the push slice count. `dt` and `op`
  // select the element type and reduction; F32/BF16/F16/S32/S64 support all four
  // ops (SUM/PRODUCT/MIN/MAX).
  hipError_t RunReduceScatter(const void* input, void* output, size_t count,
                              DataType dt, ReduceOpKind op, hipStream_t stream);

  // All-reduce: input is the full N = npes*chunk vector (symmetric heap), output
  // is the full reduced N vector (symmetric heap). numElems is the TOTAL element
  // count (chunk = numElems / npes). Push-only (npes in [1,8]). `dt`/`op` select
  // the element type and reduction; F32/BF16/F16/S32/S64 support all four ops.
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

  using FacadeMap = std::deque< std::unique_ptr<CollectivesFacade> >;
  static std::mutex& GetMutex() {
    static std::mutex mutex;
    return mutex;
  }

  static FacadeMap& GetInstances() {
    static FacadeMap instances(8);
    return instances;
  }

  static CollectivesFacade& CreateInternal(uint32_t ordinal) {
    std::lock_guard<std::mutex> lock(GetMutex());
    if (ordinal >= GetInstances().size()) {
      GetInstances().resize(ordinal + 1);
    }
    auto& inst = GetInstances()[ordinal];
    if (inst == nullptr) {
      inst = std::unique_ptr<CollectivesFacade>(new CollectivesFacade());
    }
    return *inst;
  }

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
  mori::cco::ccoComm* ccoComm_{nullptr};
  // Static heap: one symmetric window backing all user buffers (bump allocator).
  mori::cco::ccoWindow_t heapWin_{nullptr};
  uint8_t* heapBase_{nullptr};
  size_t heapSize_{0};
  size_t heapUsed_{0};
  // CCO-backed push: staging_ (above) is a sub-region of heapWin_ (peer-writable
  // SDMA target); devComm_ is the SDMA device comm.
  mori::cco::ccoDevComm devComm_{};
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
//
// Host compile and gfx942/gfx950/gfx1250 device compile get the real launches.
// Other device compiles of this TU never include the kernel headers and stub
// every Run* to hipErrorNotSupported.
// ---------------------------------------------------------------------------
#if defined(MORI_KERNELS_IMPL) && defined(MORI_FACADE_COLLECTIVES_ARCH)

namespace detail {
// Device-wide barrier: dummy no-op. The old shmem device barrier
// (ShmemBarrierAllThread) assumed shmem init, which is incompatible with the
// CCO-backed facade. Callers currently barrier host-side via ccoBarrierAll; a
// device-side CCO barrier can be wired here later if RunBarrier is needed.
__global__ void BarrierKernel() {}

}  // namespace detail

template <class T, class Op>
hipError_t CollectivesFacade::reduceScatterImpl(const void* input_v, void* output_v, size_t count,
                                                hipStream_t stream) {
  const T* input = static_cast<const T*>(input_v);
  T* output = static_cast<T*>(output_v);
  using ComputeT = typename detail::ReduceComputeType<T>::type;
  using ReduceOp = Op;

  const size_t chunkElems = count;
  const size_t chunkBytes = chunkElems * sizeof(T);
  // Staging is only needed by the push path; pull reads peers directly (no staging).
  if (mode_ != RsMode::kPull && (nPes_ - 1) * chunkBytes > stagingBytes_) {
    FACADE_PRINTF("ReduceScatter: staging too small; increase maxStagingBytes");
    return hipErrorInvalidConfiguration;
  }

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

  if (mode_ == RsMode::kPull) {
    // The kernel resolves peer pe's copy of `input` device-side by the same
    // flat-VA rank delta the push path uses: input + (pe - myPe)*stride4G<<32,
    // with stride4G read from heapWin_. NPES is a compile-time template arg;
    // dispatch on the real npes.
    auto launch = [&](auto NPES_c) {
      ReduceScatterPullKernel<NumPullVecs, decltype(NPES_c)::value, ReduceOp>
          <<<blocks, kThreads, 0, stream>>>(myPe_, heapWin_,
                                  reinterpret_cast<const ComputeT*>(input),
                                  reinterpret_cast<ComputeT*>(output), chunkElemsC);
    };
    switch (nPes_) {
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 4: launch(std::integral_constant<int, 4>{}); break;
      default: launch(std::integral_constant<int, 8>{}); break;
    }
  } else {
    // Every block loops over all S slices (no block-to-slice mapping), so the
    // grid needs no rounding to a multiple of S.
    ReduceScatterPushKernel<NumPushVecs, ReduceOp><<<blocks, kThreads, 0, stream>>>(
        myPe_, nPes_, logS, reinterpret_cast<ComputeT*>(output),
        groupCounters_, chunkElemsC, devComm_, heapWin_,
        reinterpret_cast<const ComputeT*>(input),
        reinterpret_cast<ComputeT*>(staging_));
  }
  return hipGetLastError();
}

// Enum entry point: dispatch (dt, op) to reduceScatterImpl for F32/BF16/F16/S32/S64
// and all four reduction ops.
hipError_t CollectivesFacade::RunReduceScatter(const void* input, void* output, size_t count,
                                               DataType dt, ReduceOpKind op, hipStream_t stream) {
  auto go = [=](auto typeTag) {
    using T = decltype(typeTag);
    return detail::DispatchReduceOp<T>(op, [=](auto reduceOp) {
      using Op = decltype(reduceOp);
      return this->reduceScatterImpl<T, Op>(input, output, count, stream);
    });
  };
  return detail::DispatchReduceType(dt, go);
}

template <class T, class Op>
hipError_t CollectivesFacade::allReduceImpl(const void* input_v, void* output_v, size_t numElems,
                                            hipStream_t stream) {
  const T* input = static_cast<const T*>(input_v);
  T* output = static_cast<T*>(output_v);
  using ComputeT = typename detail::ReduceComputeType<T>::type;
  using ReduceOp = Op;

  const size_t chunkElems = numElems / static_cast<size_t>(nPes_);
  const size_t chunkBytes = chunkElems * sizeof(T);
  if ((nPes_ - 1) * chunkBytes > stagingBytes_) {
    FACADE_PRINTF("AllReduce: staging too small; increase maxStagingBytes");
    return hipErrorInvalidConfiguration;
  }

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

  // Sliced push reduce-scatter + pipelined per-slice broadcast, all in one kernel.
  // Every block loops over all S slices (no block-to-slice mapping), so the grid
  // needs no rounding to a multiple of S.
  // input/output/staging all live in the static heap (heapWin_); the kernel
  // derives each intra-heap offset from the raw pointer via ccoGetLocalPtr.
  AllReducePushKernel<NumPushVecs, ReduceOp><<<blocks, kThreads, 0, stream>>>(
        myPe_, nPes_, logS, reinterpret_cast<const ComputeT*>(input),
        reinterpret_cast<ComputeT*>(output), groupCounters_, chunkElemsC, devComm_,
        reinterpret_cast<ComputeT*>(staging_), heapWin_);
  return hipGetLastError();
}

// Enum entry point: dispatch (dt, op) to allReduceImpl for F32/BF16/F16/S32/S64
// and all four reduction ops.
hipError_t CollectivesFacade::RunAllReduce(const void* input, void* output, size_t numElems,
                                           DataType dt, ReduceOpKind op, hipStream_t stream) {
  auto go = [&](auto typeTag) {
    using T = decltype(typeTag);
    return detail::DispatchReduceOp<T>(op, [&](auto reduceOp) {
      using Op = decltype(reduceOp);
      return this->allReduceImpl<T, Op>(input, output, numElems, stream);
    });
  };
  return detail::DispatchReduceType(dt, go);
}

hipError_t CollectivesFacade::RunAllGather(const void* input, void* output, size_t chunkBytes,
                                           hipStream_t stream) {
  constexpr int kThreads = 256;
  // Single block: SDMA pushes my shard to every peer's output[myPe] slot (+ self).
  AllGatherPushKernel<<<1, kThreads, 0, stream>>>(myPe_, nPes_, input, output, chunkBytes,
                                                  devComm_, heapWin_);
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
  AllToAllPushKernel<<<1, kThreads, 0, stream>>>(myPe_, nPes_, pinnedPairs_, chunkBytes, devComm_,
                                                 heapWin_);
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
                                                          numBytes, devComm_, heapWin_);
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

#elif defined(MORI_KERNELS_IMPL)
hipError_t CollectivesFacade::RunReduceScatter(const void*, void*, size_t, DataType, ReduceOpKind,
                                               hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunAllReduce(const void*, void*, size_t, DataType, ReduceOpKind,
                                           hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunAllGather(const void*, void*, size_t, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunAllToAll(const AddressVector&, size_t, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunBarrier(hipStream_t) { return hipErrorNotSupported; }
hipError_t CollectivesFacade::RunSend(const void*, size_t, int, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunRecv(void*, size_t, int, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunCollectivePermute(const void*, void*, size_t, int,
                                                   const std::vector<int>&, hipStream_t) {
  return hipErrorNotSupported;
}
hipError_t CollectivesFacade::RunQuiet(hipStream_t) { return hipErrorNotSupported; }
hipError_t CollectivesFacade::RunFence() { return hipErrorNotSupported; }

#endif  // MORI_KERNELS_IMPL

}  // namespace collective
}  // namespace mori

#undef MORI_FACADE_COLLECTIVES_ARCH

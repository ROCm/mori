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

#define FACADE_PRINTF(fmt, ...) std::fprintf(stderr, fmt"\n", ##__VA_ARGS__)
namespace mori {
namespace collective {

class CollectivesFacade {

  CollectivesFacade() = default;
 public:
  // Reduce algorithm selection for reduce-scatter and all-reduce (default
  // auto: pull or push chosen per call by message size). Env overrides read in
  // Create: MORI_CCL_FORCE_PUSH_N_SLICES=<S> (a power of two in {1,2,4,8})
  // forces push mode and pins the slice count, <=0/unset means no preference,
  // anything else is an error; MORI_CCL_FORCE_PULL!=0 forces pull (npes 4 or 8).
  // Set at runtime via SetReduceMode(); the push slice count via
  // SetPushLogSlices() (-1 = auto). MORI_CCL_MAX_NBLOCKS caps the launch grid
  // (<= 0/unset = no cap).
  enum class RsMode { kPush, kPull, kAuto };

  static constexpr size_t kDefAlign = 256;
  // Auto tuning: pull wins below kAutoPullMaxBytes of total input, except
  // 8-PE reduce-scatter (kAutoPullMaxBytesRs) and 4-PE all-reduce
  // (kAutoPullMaxBytesAr4), which cross over earlier (pull is only
  // instantiated for npes 4 and 8); push reduce-scatter keeps each slice
  // >= kAutoMinSliceBytes with at most 1<<kAutoMaxLogS slices; push all-reduce
  // always uses 1<<kAutoAllReduceLogS slices.
  static constexpr size_t kAutoPullMaxBytes = 32ull << 20;
  static constexpr size_t kAutoPullMaxBytesRs = 24ull << 20;
  static constexpr size_t kAutoPullMaxBytesAr4 = 16ull << 20;
  static constexpr size_t kAutoMinSliceBytes = 8ull << 20;
  static constexpr int kAutoMaxLogS = 2;
  static constexpr int kAutoAllReduceLogS = 1;

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

  // This device's MORI PE id and the total PE count the facade was created with.
  int MyPe() const { return myPe_; }
  int NumPes() const { return nPes_; }

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

    // Tuning (reduce-scatter mode / push slice count) defaults to auto for both;
    // MORI_CCL_FORCE_PUSH_N_SLICES / MORI_CCL_FORCE_PULL replace the mode default.
    // Callers override at runtime via SetReduceMode() / SetPushLogSlices().
    // Reset here: a device's facade is reused across Create/TearDown.
    //
    // MORI_CCL_FORCE_PUSH_N_SLICES is the actual push slice count S: a
    // power-of-two value in {1, 2, ..., kRSPushMaxSlices} forces push mode AND
    // pins the slice count (logS_ = log2(S)); a value <= 0 (or unset) means no
    // preference; any other value is an error.
    facade.mode_ = RsMode::kAuto;
    facade.logS_ = -1;
    bool pushOk = true;
    const long forcePushSlices = EnvInt("MORI_CCL_FORCE_PUSH_N_SLICES", 0, &pushOk);
    const bool forcePull = EnvInt("MORI_CCL_FORCE_PULL", 0) != 0;
    if (!pushOk ||
        (forcePushSlices > 0 &&
         (forcePushSlices > kRSPushMaxSlices || (forcePushSlices & (forcePushSlices - 1)) != 0))) {
      FACADE_PRINTF("CollectivesFacade: MORI_CCL_FORCE_PUSH_N_SLICES must be a power of two in "
                    "[1,%d] (or <=0/unset for no preference)", kRSPushMaxSlices);
      return -1;
    }
    if (forcePushSlices > 0) {
      facade.mode_ = RsMode::kPush;
      facade.logS_ = __builtin_ctz(static_cast<unsigned>(forcePushSlices));  // log2(S)
    } else if (forcePull) {
      if (facade.nPes_ == 2 || facade.nPes_ == 4 || facade.nPes_ == 8) {
        facade.mode_ = RsMode::kPull;
      } else {
        FACADE_PRINTF("CollectivesFacade: MORI_CCL_FORCE_PULL needs npes 2, 4 or 8; using auto");
      }
    }
    // Optional hard cap on the launch grid of the reduce-scatter/all-reduce
    // kernels (tightens, never raises, the CU-count cap). <= 0 / unset = no cap.
    facade.maxNBlocks_ = static_cast<int>(EnvInt("MORI_CCL_MAX_NBLOCKS", 0));

    if (facade.nPes_ > kRSPushMaxPeers) {
      FACADE_PRINTF("CollectivesFacade: npes too large (max %d)", kRSPushMaxPeers);
      return -1;
    }

    // The facade's internal buffers are carved from the shared heap window
    // BEFORE any user Allocate, so their heap offsets are identical on every
    // rank (symmetric) since every rank runs the same Create.
    //
    // Entry/exit-barrier flags of the push and pull collectives (see
    // PushEntryBarrier): slot p of every copy is written only by PE p with its
    // monotonic epoch; never reset after this.
    facade.barrierCtr_ =
        static_cast<uint64_t*>(facade.Allocate(kRSPushMaxPeers * sizeof(uint64_t)));
    if (facade.barrierCtr_ == nullptr) {
      FACADE_PRINTF("CollectivesFacade: failed to carve barrier flags from heap");
      return -1;
    }
    HIP_RUNTIME_CHECK(hipMemset(facade.barrierCtr_, 0, kRSPushMaxPeers * sizeof(uint64_t)));
    // Collective-permute ready tokens. Slot p of my copy is written only by PE p
    // ("p has entered the permute and may receive from me"); I clear it on use.
    facade.permuteReady_ =
        static_cast<uint64_t*>(facade.Allocate(kRSPushMaxPeers * sizeof(uint64_t)));
    if (facade.permuteReady_ == nullptr) {
      FACADE_PRINTF("CollectivesFacade: failed to carve permute ready tokens from heap");
      return -1;
    }
    HIP_RUNTIME_CHECK(hipMemset(facade.permuteReady_, 0, kRSPushMaxPeers * sizeof(uint64_t)));

    // Staging is the push path's peer-writable scratch (SDMA scatter target).
    // Pull needs none.
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
                                kGroupCounterCount * sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipMemset(facade.groupCounters_, 0, kGroupCounterCount * sizeof(uint32_t)));
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
    // The push collectives use signal slots [0, kSdmaSignalSlotsUsed) of the
    // lsaSize * sdmaNumQueue pool.
    const size_t signalSlots =
        static_cast<size_t>(facade.devComm_.lsaSize) * facade.devComm_.sdma.sdmaNumQueue;
    if (signalSlots < static_cast<size_t>(kSdmaSignalSlotsUsed)) {
      FACADE_PRINTF("CollectivesFacade: SDMA signal pool has %zu slots, need %d "
                    "(lsaSize=%d, sdmaNumQueue=%u)",
                    signalSlots, kSdmaSignalSlotsUsed, facade.devComm_.lsaSize,
                    facade.devComm_.sdma.sdmaNumQueue);
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
  // logS < 0 selects the slice count automatically per call.
  bool SetPushLogSlices(int logS) {
    if (logS >= 0 && (1 << logS) > kRSPushMaxSlices) {
      FACADE_PRINTF("CollectivesFacade: logS out of range (-1..%d)",
                    kRSPushMaxSlices > 0 ? 31 : 0);
      return false;
    }
    logS_ = logS < 0 ? -1 : logS;
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
  // count (chunk = numElems / npes). npes in [1,8]. SetReduceMode() selects
  // push (sliced SDMA reduce-scatter + pipelined broadcast) or pull (two-shot:
  // reduce-scatter pull then all-gather pull, no SDMA). `dt`/`op` select the
  // element type and reduction; F32/BF16/F16/S32/S64 support all four ops.
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
  
  // Parse an integer env var. Returns `defaultVal` (and sets *ok=true) when the
  // var is unset or empty; parses the whole string as a base-10 long otherwise.
  // A non-numeric or partially-numeric value sets *ok=false and returns
  // `defaultVal`. `ok` may be null if the caller does not care about validity.
  static long EnvInt(const char* name, long defaultVal, bool* ok = nullptr) {
    if (ok != nullptr) *ok = true;
    const char* e = std::getenv(name);
    if (e == nullptr || *e == '\0') return defaultVal;
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || *end != '\0') {
      if (ok != nullptr) *ok = false;
      return defaultVal;
    }
    return v;
  }

  RsMode resolveMode(size_t totalBytes, bool allReduce) const;
  int resolveLogS(size_t chunkBytes, bool allReduce) const;

  // Real reduce implementations, keyed by element type T and reduction Op. Defined
  // only in the MORI_KERNELS_IMPL TU; the enum Run* entry points dispatch to them.
  template <class T, class Op>
  hipError_t reduceScatterImpl(const void* input, void* output, size_t count, hipStream_t stream);
  template <class T, class Op>
  hipError_t allReduceImpl(const void* input, void* output, size_t numElems, hipStream_t stream);

private:
  void* staging_{nullptr};
  size_t stagingBytes_{0};
  uint32_t* groupCounters_{nullptr};
  uint64_t* barrierCtr_{nullptr};  // symmetric, entry-barrier flags (slot p written by PE p)
  uint64_t* permuteReady_{nullptr};  // symmetric, permute ready tokens (slot p written by PE p)
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
  int myPe_{0}, nPes_{0}, logS_{-1}, MP_count_{0};
  int maxNBlocks_{0};  // MORI_CCL_MAX_NBLOCKS grid cap; <= 0 means no cap
  RsMode mode_{RsMode::kAuto};
};

}  // namespace collective
}  // namespace mori

#if defined(MORI_KERNELS_IMPL)
#include "mori/collective/XLA/collectives_facade_impl.hpp"
#endif // MORI_KERNELS_IMPL

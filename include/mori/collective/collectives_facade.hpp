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

#define FACADE_PRINTF(fmt, ...) \
      std::fprintf(stderr, "CollectivesFacade: " fmt "\n", ##__VA_ARGS__)
namespace mori {
namespace collective {

class CollectivesFacade {

 public:
  // Push/pull algorithm selection for every collective (reduce-scatter,
  // all-reduce, all-gather, all-to-all, collective-permute; default auto: pull
  // or push chosen per call by message size). Env overrides read in
  // Create: MORI_CCL_FORCE_PUSH_N_SLICES=<S> (a power of two in {1,2,4,8})
  // forces push mode and pins the slice count, <=0/unset means no preference,
  // anything else is an error; MORI_CCL_FORCE_PULL!=0 forces pull (npes 4 or 8).
  // Set at runtime via SetReduceMode(); the push slice count via
  // SetPushLogSlices() (-1 = auto). MORI_CCL_MAX_NBLOCKS caps the launch grid
  // (default 128; <= 0 = no cap, use the full CU count).
  enum class RsMode { kPush, kPull, kAuto };

  static constexpr size_t kDefAlign = 128;
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
  // Auto tuning of the copy collectives: pull below these total byte counts
  // (all-gather/all-to-all: npes*chunkBytes, permute: numBytes).
  static constexpr size_t kAutoPullMaxBytesAg = 4ull << 20;
  static constexpr size_t kAutoPullMaxBytesA2a = 4ull << 20;
  static constexpr size_t kAutoPullMaxBytesPermute = 1536ull << 10;

  // Per-peer all-to-all endpoints: addrs[p] = {chunk sent to peer p, recv slot
  // for sender p}. The facade copies these into its pinned AddressPair buffer.
  using AddressVector = std::vector<std::pair<const void*, void*>>;

  CollectivesFacade(const CollectivesFacade&) = delete;
  CollectivesFacade& operator=(const CollectivesFacade&) = delete;

  // This device's rank within the (sub-)communicator, the communicator size,
  // and the global PE of rank 0 (rank r is PE PeBase() + r).
  int MyPe() const { return myPe_; }
  int NumPes() const { return nPes_; }
  int PeBase() const { return peBase_; }

  static bool PullNpesSupported(int n) { return n == 2 || n == 4 || n == 8; }

  // Creates a facade for rank `myPe` of a (sub-)communicator of `nPes` ranks,
  // where rank r maps to global PE rank_to_pe[r]. Only contiguous mappings
  // (rank_to_pe[r] == rank_to_pe[0] + r) within this node's LSA team are
  // supported. Collectives of different communicators on the same device may
  // interleave but must not run concurrently: they share staging and SDMA queues.
  // Gen-barrier and completion flags are per PE range (kCommCtlWords); permute
  // flags are pairwise and global.
  static std::unique_ptr< CollectivesFacade > Get(int myPe, int nPes,
                                                const int *rank_to_pe) {
    auto *state = GetState();
    if (MORI_UNLIKELY(state == nullptr)) {
      FACADE_PRINTF("Get: no instance");
      return nullptr;
    }
    if (nPes <= 0 || nPes > kCCLMaxPeers) {
      FACADE_PRINTF("nPes must be in [1, %d], got %d", kCCLMaxPeers, nPes);
      return nullptr;
    }
    if (myPe < 0 || myPe >= nPes || rank_to_pe == nullptr) {
      FACADE_PRINTF("invalid rank %d of %d or null rank_to_pe", myPe, nPes);
      return nullptr;
    }
    const int peBase = rank_to_pe[0];
    for (int i = 0; i < nPes; ++i) {
      if (rank_to_pe[i] != peBase + i) {
        FACADE_PRINTF("rank_to_pe must be contiguous: rank_to_pe[%d]=%d, expected %d",
                      i, rank_to_pe[i], peBase + i);
        return nullptr;
      }
    }
    const int lsaSize = state->devComm.lsaSize;
    if (peBase < 0 || peBase + nPes > lsaSize) {
      FACADE_PRINTF("PE range [%d, %d) exceeds the LSA team size %d", peBase,
                    peBase + nPes, lsaSize);
      return nullptr;
    }
    if (peBase + myPe != state->devComm.lsaRank) {
      FACADE_PRINTF("rank %d maps to PE %d, but this device is PE %d", myPe,
                    peBase + myPe, state->devComm.lsaRank);
      return nullptr;
    }

    // Tuning (reduce-scatter mode / push slice count) defaults to auto for both;
    // MORI_CCL_FORCE_PUSH_N_SLICES / MORI_CCL_FORCE_PULL replace the mode default.
    // Callers override at runtime via SetReduceMode() / SetPushLogSlices().
    // Reset here: a device's facade is reused across Create/TearDown.
    //
    // MORI_CCL_FORCE_PUSH_N_SLICES is the actual push slice count S: a
    // power-of-two value in {1, 2, ..., kPushMaxSlices} forces push mode AND
    // pins the slice count (logS_ = log2(S)); a value <= 0 (or unset) means no
    // preference; any other value is an error.
    RsMode mode = RsMode::kAuto;
    int logS = -1;
    bool pushOk = true;
    const long forcePushS = EnvInt("MORI_CCL_FORCE_PUSH_N_SLICES", 0, &pushOk);
    bool forcePull = EnvInt("MORI_CCL_FORCE_PULL", 0) != 0;
    
    if (forcePushS <= 0) pushOk = false;
    if (pushOk && (forcePushS > kPushMaxSlices || 
                  (forcePushS & (forcePushS - 1)) != 0)) {
      FACADE_PRINTF("MORI_CCL_FORCE_PUSH_N_SLICES must be a power of two in range [1,%d]",
                    kPushMaxSlices);
      pushOk = false;
    }
    if (pushOk) {
      mode = RsMode::kPush;
      logS = __builtin_ctz(static_cast<uint32_t>(forcePushS));  // log2(S)
    } else if (forcePull) {
      if (PullNpesSupported(nPes)) {
        mode = RsMode::kPull;
      } else {
        FACADE_PRINTF("MORI_CCL_FORCE_PULL needs nPes 2, 4 or 8; using auto");
      }
    }
    return std::unique_ptr<CollectivesFacade>(
                   new CollectivesFacade(*state, myPe, nPes, peBase, logS, mode));
  }

  // CCO-backed factory. Records the device's CCO comm, and registers ONE big 
  // symmetric CCO "heap" window based on CCO comm size.
  // Need to be called by every rank in parallel.
  static int Create(cco::ccoComm* comm, size_t heapBytes, 
                                                 size_t maxStagingBytes = 0) {
    if (comm == nullptr) {
      FACADE_PRINTF("NULL comm for CCO create");
      return -1;
    }
    if (maxStagingBytes == 0) {
      maxStagingBytes = heapBytes / 4;
    }
    maxStagingBytes = (maxStagingBytes + kDefAlign - 1) & ~(kDefAlign - 1);
    if (heapBytes == 0 || maxStagingBytes >= heapBytes) {
      FACADE_PRINTF("wrong heap size or maxStagingBytes");
      return -1;
    }

    int dev = 0;
    HIP_RUNTIME_CHECK(hipGetDevice(&dev));
    auto& state = GetOrCreateState(dev);
    state.ccoComm = comm;
    int ret = cco::ccoWindowRegister(comm, heapBytes, &state.heapWin,
                                     reinterpret_cast<void**>(&state.heapBase));
    if (ret != 0 || state.heapBase == nullptr) {
      FACADE_PRINTF("failed to register heap window (%zu bytes, ret=%d)",
                    heapBytes, ret);
      return -1;
    }
    state.heapSize = heapBytes;
    state.heapUsed = 0;

    hipDeviceProp_t prop;
    HIP_RUNTIME_CHECK(hipGetDeviceProperties(&prop, dev));
    state.MP_count = prop.multiProcessorCount;

    // Hard cap on the launch grid of the reduce-scatter/all-reduce kernels
    // (tightens, never raises, the CU-count cap). Defaults to 128: large-message
    // push/pull saturate XGMI bandwidth well below the full CU count, and a
    // smaller co-resident grid cuts per-block barrier/atomic contention (measured
    // ~1-4% faster than the full 256-CU grid on gfx950, never slower). Override
    // with MORI_CCL_MAX_NBLOCKS; <= 0 disables the cap (use the full CU count).
    state.maxNBlocks = static_cast<int>(EnvInt("MORI_CCL_MAX_NBLOCKS", 128));

    size_t controlBytes =
        kCCLMaxPeers * sizeof(uint64_t) * kPermuteFlagSlots + kCommCtlBytes;
    size_t totalBytes = maxStagingBytes + controlBytes;
    state.staging = Allocate(totalBytes);
    if (state.staging == nullptr) {
      FACADE_PRINTF("failed to carve staging and control flags from heap (%zu bytes)",
                      totalBytes);
      return -1;
    }
    state.stagingBytes = maxStagingBytes;

    // Collective-permute handshake flags (see PermuteFlagIdx / CollectivePermute*
    // kernels), consolidated into one [kPermuteFlagSlots][kCCLMaxPeers] buffer.
    // Slot [idx][p] of my copy is written only by PE p; never reset.
    state.permuteCtr = reinterpret_cast<uint64_t*>(
             static_cast<uint8_t*>(state.staging) + maxStagingBytes);
    // Per-communicator gen-barrier flags and SDMA completion slots, one slot per
    // contiguous PE range (see kCommCtlWords); never reset after this.
    state.commCtl = state.permuteCtr + kPermuteFlagSlots * kCCLMaxPeers;
    HIP_RUNTIME_CHECK(hipMemset(state.permuteCtr, 0, controlBytes));

    // One local, zeroed-once buffer: the uint64 per-peer permute counts first
    // (see CollectivePermutePushKernel: [p] sends to p, [kCCLMaxPeers + p]
    // receives from p), then the uint32 groupCtr.
    const size_t permuteNumBytes = 2 * kCCLMaxPeers * sizeof(uint64_t);
    const size_t localCtrBytes = permuteNumBytes + kGroupCtrTotal * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMalloc(reinterpret_cast<void**>(&state.permuteNum), localCtrBytes));
    HIP_RUNTIME_CHECK(hipMemset(state.permuteNum, 0, localCtrBytes));
    state.groupCtr = reinterpret_cast<uint32_t*>(state.permuteNum + 2 * kCCLMaxPeers);    // Preallocate the pinned (host-visible, device-readable) all-to-all pointer
    // buffer once, so each RunAllToAll is a plain host fill + launch (no per-call
    // device alloc/copy). Sized for the max supported peer count.
    HIP_RUNTIME_CHECK(hipHostMalloc(reinterpret_cast<void**>(&state.pinnedPairs),
                                    kCCLMaxPeers * sizeof(AddressPair), 
                                    hipHostMallocDefault));

    // SDMA device comm for the push path (passed by value into the push kernel).
    cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = cco::CCO_GDA_CONNECTION_NONE;
    reqs.gdaContextCount = 0;
    reqs.gdaSignalCount = 0;
    reqs.gdaCounterCount = 0;
    reqs.sdmaQueueCount = 1; // The push/scatter kernels only ever use queue 0;
                             // 0 means "all of the context's queues" and just
                             // slows ccoDevCommCreate (more queues to set up).
    
    ret = cco::ccoDevCommCreate(comm, &reqs, &state.devComm);
    if (ret != 0 || state.devComm.sdma.sdmaNumQueue == 0) {
      FACADE_PRINTF("ccoDevCommCreate failed or no SDMA queues allocated.");
      return -1;
    }
    // The kernels index deviceHandles by peer alone (queue 0 of each peer).
    if (state.devComm.sdma.sdmaNumQueue != 1) {
      FACADE_PRINTF("expected exactly one SDMA queue per peer, got %u",
                    state.devComm.sdma.sdmaNumQueue);
      return -1;
    }
    return 0;
  }

  // Tear down all collectives instances (should be safe to be called from 
  // a single thread).
  static void TearDown() {
    std::lock_guard<std::mutex> lock(GetMutex());
    GetStates().clear();
  }

  // Bump-allocate a peer-visible buffer from the static heap window. Returns a
  // raw local pointer (heapBase_ + offset). Every rank MUST issue the same
  // Allocate sequence (same sizes/order) so a given buffer sits at the same heap
  // offset on all ranks (symmetric addressing). Returns nullptr if the heap is
  // exhausted. `align` defaults to 128B (SDMA/b128 friendly).
  static void* Allocate(size_t bytes, size_t align = kDefAlign) {
    auto *state = GetState();
    if (MORI_UNLIKELY(state == nullptr)) {
      FACADE_PRINTF("Allocate: no instance");
      return nullptr;
    }
    const size_t off = (state->heapUsed + (align - 1)) & ~(align - 1);
    if (MORI_UNLIKELY(off + bytes > state->heapSize)) {
      FACADE_PRINTF("Allocate: heap exhausted (need %zu at off %zu, size %zu)",
                    bytes, off, state->heapSize);
      return nullptr;
    }
    state->heapUsed = off + bytes;
    return state->heapBase + off;
  }

  // Bump allocator: individual frees are a no-op. Use Reset() to reclaim the
  // whole heap between phases (all outstanding buffers become invalid).
  static int Deallocate(void* ptr) {
    auto *state = GetState();
    if (MORI_UNLIKELY(state == nullptr)) {
      FACADE_PRINTF("Deallocate: no instance");
      return -1;
    }
    return 0;
  }

  static void Reset() {
    auto *state = GetState();
    if (MORI_UNLIKELY(state == nullptr)) {
      FACADE_PRINTF("Reset: no instance");
      return;
    }
    state->heapUsed = 0;
  }

  // Runtime tuning "backdoors" (replace the former RS_MODE / RS_LOG_PUSH_SLICES
  // env vars). Call after Create, before the Run*. Both return false and leave
  // the current value unchanged on an invalid request.
  bool SetReduceMode(RsMode mode) {
    if (mode == RsMode::kPull && nPes_ > 8) {
      FACADE_PRINTF("pull mode supports npes in [1,8]");
      return false;
    }
    mode_ = mode;
    return true;
  }
  // logS < 0 selects the slice count automatically per call.
  bool SetPushLogSlices(int logS) {
    if (logS >= 0 && (1 << logS) > kPushMaxSlices) {
      FACADE_PRINTF("logS out of range!");
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
  // gathered output is npes*chunkBytes). SetReduceMode() selects push (SDMA, one
  // block) or pull (every PE reads the peers' shards over XGMI, multi-block).
  hipError_t RunAllGather(const void* input, void* output, size_t chunkBytes, hipStream_t stream);

  // All-to-all: addrs[p] = {chunk sent to peer p, recv slot for sender p}, one
  // pair per peer (addrs.size() == npes). Pure byte movement, so chunkBytes is
  // the per-peer chunk byte count. Push needs the recv slots in the symmetric
  // heap at the same offset on every rank; pull needs the same of the send slots.
  hipError_t RunAllToAll(const AddressVector& addrs, size_t chunkBytes, hipStream_t stream);

  // Device-wide barrier across all PEs on the given stream.
  hipError_t RunBarrier(hipStream_t stream);

  // Point-to-point send: push numBytes from sendBuf to PE `peer`. Dummy no-op.
  hipError_t RunSend(const void* sendBuf, size_t numBytes, int peer, hipStream_t stream);

  // Point-to-point recv: receive numBytes into recvBuf from PE `peer`. Dummy no-op.
  hipError_t RunRecv(void* recvBuf, size_t numBytes, int peer, hipStream_t stream);

  // Collective-permute: send my buffer to each target PE, recv from srcPe
  // (srcPe < 0 means no source). v1: dstPes.size() <= 1. SetReduceMode() selects
  // push (SDMA) or pull (the receiver reads srcPe's sendBuf over XGMI).
  hipError_t RunCollectivePermute(const void* sendBuf, void* recvBuf, size_t numBytes,
                                  int srcPe, const std::vector<int>& dstPes,
                                  hipStream_t stream);

  // Drain outstanding operations on `stream`. Dummy no-op.
  hipError_t RunQuiet(hipStream_t stream);

  // Memory fence across the device. Dummy no-op.
  hipError_t RunFence();

 private:
   struct SharedState {
    void* staging{nullptr};
    size_t stagingBytes{0};
    uint64_t* permuteNum{nullptr};    // local, per-peer permute send/recv counts (owns the buffer)
    uint32_t* groupCtr{nullptr};  // local, sub-region of the permuteNum buffer
    uint64_t* permuteCtr{nullptr};  // symmetric, consolidated permute handshake flags
                                      // [kPermuteFlagSlots][kCCLMaxPeers] (slot [idx][p] written by PE p)
    uint64_t* commCtl{nullptr};  // symmetric, [kCommCtlSlots][kCommCtlWords] per-range
                                 // gen-barrier flags + SDMA completion slots
    AddressPair* pinnedPairs{nullptr};  // host-pinned, device-readable
    cco::ccoComm* ccoComm{nullptr};
    // Static heap: one symmetric window backing all user buffers (bump allocator).
    cco::ccoWindow_t heapWin{nullptr};
    uint8_t* heapBase{nullptr};
    size_t heapSize{0};
    size_t heapUsed{0};
    // CCO-backed push: staging (above) is a sub-region of heapWin (peer-writable
    // SDMA target); devComm is the SDMA device comm.
    cco::ccoDevComm devComm{};
    int MP_count{0};    // Number of CUs per device
    int maxNBlocks{0};  // MORI_CCL_MAX_NBLOCKS grid cap (default 128); <= 0 means no cap

    // Release this device's heap window, counters, and SDMA comm. Staging is a
    // sub-region of heapWin, so it needs no separate deregistration.
    ~SharedState() {
      if (heapWin != nullptr) {
        cco::ccoWindowDeregister(ccoComm, heapWin);
      }
      if (ccoComm != nullptr) {
        cco::ccoDevCommDestroy(ccoComm, &devComm);
        cco::ccoCommDestroy(ccoComm);
      }
      if (permuteNum != nullptr) {  // also backs groupCtr
        HIP_RUNTIME_CHECK(hipFree(permuteNum));
      }
      if (pinnedPairs != nullptr) {
        HIP_RUNTIME_CHECK(hipHostFree(pinnedPairs));
      }
    }
  };

  using StateMap = std::deque< std::unique_ptr<SharedState> >;
  static std::mutex& GetMutex() {
    static std::mutex mutex;
    return mutex;
  }

  static StateMap& GetStates() {
    static StateMap states(8);
    return states;
  }

  static SharedState *GetState() {
    int dev = 0;
    HIP_RUNTIME_CHECK(hipGetDevice(&dev));
    std::lock_guard<std::mutex> lock(GetMutex());
    if (MORI_UNLIKELY(GetStates().size() <= static_cast<size_t>(dev))) {
      return nullptr;
    }
    return GetStates()[dev].get();
  }

  static SharedState& GetOrCreateState(uint32_t ordinal) {
    std::lock_guard<std::mutex> lock(GetMutex());
    if (MORI_UNLIKELY(ordinal >= GetStates().size())) {
      GetStates().resize(ordinal + 1);
    }
    auto& state = GetStates()[ordinal];
    if (state == nullptr) {
      state = std::make_unique<SharedState>();
    }
    return *state;
  }
  
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

  // The kernels run in the communicator's local rank space [0, nPes): the flat-VA
  // peer delta (peer - myPe) is unchanged by the PE base, the SDMA queue table is
  // shifted so local rank r addresses global PE peBase + r, and the barrier and
  // completion slots come from this range's own control slot.
  CollectivesFacade(SharedState& state, int myPe, int nPes, int peBase, int logS,
                    RsMode mode) :
           state_(state), myPe_(myPe), nPes_(nPes), peBase_(peBase),
           logS_(logS), mode_(mode), devComm_(state.devComm),
           barrierCtr_(state.commCtl +
                       (static_cast<size_t>(peBase) * kCCLMaxPeers + nPes - 1) * kCommCtlWords),
           arrived_(barrierCtr_ + kCCLMaxPeers) {
    devComm_.sdma.deviceHandles += peBase;
  }

  RsMode resolveMode(size_t totalBytes, bool allReduce) const;
  RsMode resolveCopyMode(size_t totalBytes, size_t pullMaxBytes) const;
  int resolveLogS(size_t chunkBytes, bool allReduce) const;

  // Real reduce implementations, keyed by element type T and reduction Op. Defined
  // only in the MORI_KERNELS_IMPL TU; the enum Run* entry points dispatch to them.
  template <class T, class Op>
  hipError_t reduceScatterImpl(const void* input, void* output, size_t count, hipStream_t stream);
  template <class T, class Op>
  hipError_t allReduceImpl(const void* input, void* output, size_t numElems, hipStream_t stream);

private:
  SharedState& state_;
  int myPe_, nPes_, peBase_, logS_;
  RsMode mode_;
  cco::ccoDevComm devComm_;  // state_.devComm with the SDMA tables shifted by peBase_
  uint64_t* barrierCtr_;     // this range's control slot in state_.commCtl
  uint64_t* arrived_;        // barrierCtr_ + kCCLMaxPeers
};

}  // namespace collective
}  // namespace mori

#if defined(MORI_KERNELS_IMPL)
#include "mori/collective/XLA/collectives_facade_impl.hpp"
#endif // MORI_KERNELS_IMPL

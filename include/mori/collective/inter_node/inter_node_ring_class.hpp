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

// Inter-node RDMA ring AllGather host handle.
//
// This is the inter-node phase of the hierarchical cross-node AllGather: it
// moves one chunk per PE over the shmem transport (P2P within a node, RDMA
// across nodes) using the ring schedule in
// ``inter_node/kernels/all_gather.hpp`` (validated bit-exactly on CPU by
// ``inter_node_ring_reference`` in the Python layer). The host handle owns a
// fixed-size symmetric ring buffer + flags, builds the JIT kernel args, and is
// driven from Python exactly like ``AllgatherSdma`` (prepare -> launch kernel
// -> finish).

#ifndef INTER_NODE_RING_CLASS_HPP
#define INTER_NODE_RING_CLASS_HPP

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "mori/collective/ccl_kernel_args.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// Select the dissemination barrier for the inter-ring prepare rendezvous when
// MORI_HIER_DISSEM_BARRIER!=0. Read once (env is fixed for the process). The
// dissem barrier has identical global all-PE semantics but an O(log n) parallel
// critical path instead of the PE0 funnel.
inline bool HierDissemBarrierEnabled() {
  static const bool enabled = []() {
    const char* e = std::getenv("MORI_HIER_DISSEM_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return enabled;
}

// Hierarchical 2-level entry barrier (env MORI_HIER_BARRIER_HIER=<rpn>, default
// 0=OFF). Value = ranks_per_node. When >0 the kernel's entry rendezvous routes
// through ShmemBarrierOnStreamHier instead of the flat funnel/dissem barrier:
// local PEs signal their node coordinator over XGMI, the per-node coordinators
// exchange once over RDMA, then coordinators release their locals over XGMI.
// This crosses the RDMA node boundary exactly twice (vs the funnel's serial
// per-rank cross-node ops). Same global all-PE semantics; a monotonic generation
// counter keeps it graph-replay-safe. Bit-exact; targets the op-to-op serialized
// regime (back-to-back all-gathers) where the barrier is exposed.
inline int HierBarrierHierRanksPerNode() {
  static const int rpn = []() {
    const char* e = std::getenv("MORI_HIER_BARRIER_HIER");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  return rpn;
}

// Opt-in (MORI_HIER_NO_ENTRY_BARRIER; off by default): skip the ring's prepare
// entry rendezvous barrier. NOT correctness-safe (it drops the cross-PE fence
// that orders every PE's op-end flag-reset + own-chunk staging before any peer's
// next-op atomic increment); benchmarking lever only. See the generation-counter
// barrier-free ring (prepare_stream flag-reset invariant note) for the
// correctness-safe alternative.
inline bool HierEntryBarrierDisabled() {
  static const bool disabled = []() {
    const char* e = std::getenv("MORI_HIER_NO_ENTRY_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return disabled;
}

// Opt-in master switch (MORI_HIER_NO_ALL_BARRIER; off by default): skip every
// cross-PE barrier (entry ShmemBarrierOnStream, the deferred finish fences, and
// the host ShmemBarrierAll rendezvous in both the inter-node ring and the
// intra-node subgroup gather). NOT correctness-safe; benchmarking lever only.
inline bool HierAllBarrierDisabled() {
  static const bool disabled = []() {
    const char* e = std::getenv("MORI_HIER_NO_ALL_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return disabled;
}

inline void HierPrepareBarrierOnStream(hipStream_t stream) {
  if (HierEntryBarrierDisabled() || HierAllBarrierDisabled()) {
    return;
  }
  const int hierRpn = HierBarrierHierRanksPerNode();
  if (hierRpn > 0) {
    // Topology-aware 2-level barrier (crosses the node boundary only via per-node
    // coordinators). Falls back to the funnel inside the launcher if the loaded
    // module lacks the hier kernel.
    shmem::ShmemBarrierOnStreamHier(stream, hierRpn);
  } else if (HierDissemBarrierEnabled()) {
    shmem::ShmemBarrierOnStreamDissem(stream);
  } else {
    shmem::ShmemBarrierOnStream(stream);
  }
}

// The ring finish / cross-PE reuse fence. By default the plain
// ShmemBarrierOnStream (PE0-funnel rendezvous). The dissemination barrier has
// identical global all-PE ordering semantics (a full rendezvous: every PE waits
// for every other PE), so routing the finish fence through it does not reorder the
// NIC-landing -> reassembly-consume dependency the correct path relies on -- the
// byte image and the completion ordering are unchanged. It only replaces the PE0
// funnel critical path with a parallel log-depth one. Gated on the same
// MORI_HIER_DISSEM_BARRIER env so the default path stays byte-for-byte identical.
inline void HierFinishBarrierOnStream(hipStream_t stream) {
  if (HierAllBarrierDisabled()) {
    return;
  }
  if (HierDissemBarrierEnabled()) {
    shmem::ShmemBarrierOnStreamDissem(stream);
  } else {
    shmem::ShmemBarrierOnStream(stream);
  }
}

class InterNodeRingAllgather {
 private:
  int myPe_;
  int npes_;

  // Sub-group descriptor. The ring runs over the arithmetic sub-group of
  // global PEs {peBase_, peBase_+peStride_, ..., peBase_+(ringSize_-1)*peStride_};
  // this PE is at position ringPos_. The whole-world ring is the flat default
  // peBase_=0, peStride_=1, ringSize_=npes_, ringPos_=myPe_.
  int ringPos_;
  int ringSize_;
  int peBase_;
  int peStride_;
  // RDMA QP fan-out degree for the per-round ring put. 1 keeps the single-QP put;
  // >1 fans the chunk across QPs (RDMA neighbours only, gated at runtime in the
  // kernel). The hierarchical inter-node ring passes >1.
  int numQp_;

  // Number of CTAs ("channels") the ring kernel is launched with. 1 keeps the
  // single-block ring. >1 partitions each chunk into numBlocks_ disjoint
  // sub-ranges, one CTA each (qpId=bid). Used only to size the per-block flag
  // regions (numBlocks_*ringSize slots); the kernel reads the actual block count
  // from gridDim.x at launch.
  int numBlocks_;

  // Symmetric ring buffer: holds ringSize_ contiguous chunks. Sized once for the
  // largest message; the kernel only touches the first ringSize_*chunkBytes.
  void* ring_;
  size_t ringBytes_;
  application::SymmMemObjPtr ringObj_;

  // Per-PE arrival flags (one uint64 per PE).
  void* flags_;
  application::SymmMemObjPtr flagsObj_;

  // Kept alive between prepare and the Python-side kernel launch.
  CclInterNodeRingArgs jit_args_;

  // GEN-RING (env MORI_HIER_GEN_RING): monotonic per-op generation counter and
  // the enable flag. When on, prepare skips the entry barrier + flag reset and
  // stamps jit_args_.opGen = ++ringOpGen_ so the kernel waits for the flag to
  // reach this generation (flags accumulate; never reset). Default off.
  bool genRing_ = false;
  uint64_t ringOpGen_ = 0;

  // Device-side gen-ring (env MORI_HIER_GEN_RING_DEV). The host gen-ring above is
  // graph-incompatible: prepare_stream runs once at capture so opGen freezes while
  // the accumulating flags advance every replay, desyncing the receiver's gate.
  // This variant instead exposes a device counter (one uint64 per ring
  // block/channel, numBlocks_ entries) that the ring kernel increments itself each
  // execution (eager or graph replay), so the per-op generation stays in lockstep
  // with the sender's per-op flag AMO_ADD(1) under graph replay. When on,
  // prepare_stream drops the entry barrier and publishes opGenCounter into
  // jit_args_ (kernel picks up the gen device-side). Default off.
  bool genRingDev_ = false;
  void* opGenCounter_ = nullptr;

  // Fuse-entry-barrier (env MORI_HIER_FUSE_ENTRY_BARRIER, requires the default
  // single-increment path; mutually exclusive with genRing*). When on,
  // prepare_stream skips the separate host-launched entry ShmemBarrierOnStream and
  // instead the fused kernel performs the same cross-PE rendezvous device-side at
  // its entry prologue (block 0 -> ShmemBarrierAllBlock), gated by a manual
  // grid-arrival barrier over this 2-word device scratch: gridArrival_[0] = arrival
  // counter (returns to 0 each op), gridArrival_[1] = monotonic release generation
  // (never reset -> graph-replay-safe). Per-PE local (never remote), plain HBM.
  bool fuseEntryBarrier_ = false;
  void* gridArrival_ = nullptr;

  // Cheap-correct reuse rendezvous (env MORI_HIER_GEN_RING_DISSEM, requires
  // genRingDev_). The barrier-free device gen-ring has a residual single-buffer
  // ring-reuse slip at op boundaries (op N+1's peer push overwrites a slot op N's
  // reassembly still reads) because the entry barrier was dropped. This restores a
  // true cross-PE rendezvous -- immune to graph-replay desync by construction --
  // routed through the O(log n) dissem barrier (ShmemBarrierOnStreamDissem) instead
  // of the PE0 funnel: a correct all-PE entry rendezvous that orders every PE's
  // op-N reassembly-read before any peer's op-N+1 push, at log-depth latency.
  // Composes with the accumulating-flag / device-gen graph-safety (genRingDev_).
  // Default off.
  bool genRingDissem_ = false;

  // Device double-buffered ring (env MORI_HIER_GEN_RING_DBL, requires genRingDev_).
  // A second symmetric ring buffer alternated with ring_ by op parity so the
  // barrier-free gen-ring's cross-PE reuse race is closed: op N+1's peer pushes
  // land in the other half than op N's still-reassembling half. parityCounter_ is a
  // device uint64 bumped once per op by the parity-bump kernel (captured on-stream,
  // graph-safe). Default off => single-buffer path.
  bool genRingDbl_ = false;
  void* ring2_ = nullptr;
  application::SymmMemObjPtr ring2Obj_;
  void* parityCounter_ = nullptr;

  InterNodeRingAllgather(const InterNodeRingAllgather&) = delete;
  InterNodeRingAllgather& operator=(const InterNodeRingAllgather&) = delete;

 public:
  // ringSize<0 selects the flat whole-world ring (ringSize=npes, ringPos=myPe,
  // peBase=0, peStride=1). Otherwise an explicit sub-group is used.
  InterNodeRingAllgather(int myPe, int npes, size_t ring_buffer_bytes, int ringSize = -1,
                         int ringPos = -1, int peBase = 0, int peStride = 1, int numQp = 1,
                         int numBlocks = 1)
      : myPe_(myPe),
        npes_(npes),
        numQp_(numQp < 1 ? 1 : numQp),
        numBlocks_(numBlocks < 1 ? 1 : numBlocks),
        ringBytes_(ring_buffer_bytes),
        ring_(nullptr),
        flags_(nullptr) {
    if (ringSize < 0) {
      ringSize_ = npes_;
      ringPos_ = myPe_;
      peBase_ = 0;
      peStride_ = 1;
    } else {
      if (ringSize < 1 || ringPos < 0 || ringPos >= ringSize || peStride < 1) {
        throw std::runtime_error("InterNodeRingAllgather: invalid sub-group descriptor");
      }
      ringSize_ = ringSize;
      ringPos_ = ringPos;
      peBase_ = peBase;
      peStride_ = peStride;
    }

    ring_ = shmem::ShmemMalloc(ringBytes_);
    if (ring_ == nullptr)
      throw std::runtime_error("InterNodeRingAllgather: ring ShmemMalloc failed");
    ringObj_ = shmem::ShmemQueryMemObjPtr(ring_);

    // Flags: one uint64 per ring slot PER BLOCK. Allocate numBlocks_*npes
    // (>= numBlocks_*ringSize_) so each CTA channel gets its own flag region and
    // the buffer is large enough for any sub-group on this PE set.
    size_t flagsBytes = static_cast<size_t>(numBlocks_) * npes_ * sizeof(uint64_t);
    flags_ = shmem::ShmemMalloc(flagsBytes);
    if (flags_ == nullptr)
      throw std::runtime_error("InterNodeRingAllgather: flags ShmemMalloc failed");
    (void)hipMemset(flags_, 0, flagsBytes);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_);

    // Transport-level flag-can't-beat-data completion protocol (env
    // MORI_HIER_RING_PUT_SIGNAL; off by default). When on, the single-warp RDMA
    // ring send fuses the data WRITE and the completion-flag AMO into one
    // ShmemPutMemNbiSignal (same QP, signal strictly AFTER data on the RC-ordered
    // QP) so the receiver never observes the flag before the remote data lands.
    // This removes the separate post-put ShmemQuietThread full-CQ drain that the
    // receiver's flag-spin otherwise waits behind. Default off keeps the
    // standalone bytes unchanged. The fused FSDP builders gate it independently
    // (HierRingPutSignalExplicitlyOn) so it never leaks into E2E.
    {
      const char* e = std::getenv("MORI_HIER_RING_PUT_SIGNAL");
      jit_args_.usePutSignal = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    // WRITE_WITH_IMM (env MORI_HIER_RING_WRITE_IMM, default off). When
    // set, the single-warp cross-node ring send emits RDMA_WRITE_WITH_IMM and the
    // receiver consumes the recv-CQ completion instead of the flag spin -- the
    // recv-CQE cannot precede the payload landing, closing the remote-landing race
    // with no host stall. Set once here; persists across calls like usePutSignal.
    {
      const char* e = std::getenv("MORI_HIER_RING_WRITE_IMM");
      jit_args_.useWriteImm = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    // RDMA-READ (PULL) ring (env MORI_HIER_RING_READ, default OFF). When set, the
    // single-round (ringSize==2) all-RDMA inter-node phase PULLS the peer's own
    // chunk with an RDMA READ instead of the peer PUSHing it -- the READ
    // completion (our own quiet) is the consumer-side landing fence, no flag AMO
    // / receiver spin / remote-landing race. Set once here; persists across calls.
    {
      const char* e = std::getenv("MORI_HIER_RING_READ");
      jit_args_.useRead = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    // WRITE-PUSH (SEND-CQ) landing fence (env MORI_HIER_RING_WRITE, default OFF).
    // The WRITE-side counterpart of MORI_HIER_RING_READ. On the giant
    // multiBlock AG each channel pushes its sub-range as a fused put-with-signal
    // then drains its own SEND CQE; keeps RDMA-WRITE fill where READ underfilled.
    // Set once here; persists across calls like usePutSignal / useRead.
    {
      const char* e = std::getenv("MORI_HIER_RING_WRITE");
      jit_args_.useWriteFence = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    // GENERATION-COUNTER barrier-free ring (env MORI_HIER_GEN_RING, default OFF).
    // When set, prepare_stream stops resetting the flags (the kernel accumulates
    // the +1 per op) and DROPS the entry ShmemBarrierOnStream -- one of the two
    // global barriers per ring round the plateau pays. Only engages on the
    // classic single-increment path (no put-signal / write-imm), so it is
    // mutually exclusive with those; if either is also set, gen-ring stays off.
    {
      const char* e = std::getenv("MORI_HIER_GEN_RING");
      genRing_ = (e != nullptr && e[0] != '\0' && e[0] != '0') && jit_args_.usePutSignal == 0 &&
                 jit_args_.useWriteImm == 0;
    }
    // Device-side gen-ring (env MORI_HIER_GEN_RING_DEV, default OFF). The
    // graph-compatible variant of GEN_RING: the DEVICE increments the per-op
    // generation so it advances under HIP-graph replay (host GEN_RING freezes at
    // capture). Same mutual-exclusion as GEN_RING (classic single-increment flag
    // path only). Allocate one device uint64 per ring block/channel, zeroed once;
    // the kernel does opGenCounter[bx] += 1 at entry. Uses plain HBM (hipMalloc):
    // the counter is per-PE local (never remotely accessed), so it need not be
    // symmetric. Publish the pointer into jit_args_ so BuildFusedRingLocalGatherArgs
    // (and the plain ring kernel) can pick it up; prepare_stream drops the entry
    // barrier when it is engaged.
    {
      const char* e = std::getenv("MORI_HIER_GEN_RING_DEV");
      genRingDev_ = (e != nullptr && e[0] != '\0' && e[0] != '0') && jit_args_.usePutSignal == 0 &&
                    jit_args_.useWriteImm == 0 && !genRing_;
    }
    if (genRingDev_) {
      size_t genBytes = static_cast<size_t>(numBlocks_) * sizeof(uint64_t);
      if (hipMalloc(&opGenCounter_, genBytes) != hipSuccess || opGenCounter_ == nullptr) {
        throw std::runtime_error("InterNodeRingAllgather: opGenCounter hipMalloc failed");
      }
      (void)hipMemset(opGenCounter_, 0, genBytes);
    }
    // Fuse-entry-barrier: engage only on the default single-increment path
    // (no put-signal / write-imm) and never together with the barrier-DROP gen-ring
    // variants (those already remove the entry rendezvous). Allocate the 2-word grid
    // arrival scratch (zeroed once; release generation accumulates across ops).
    {
      const char* e = std::getenv("MORI_HIER_FUSE_ENTRY_BARRIER");
      fuseEntryBarrier_ = (e != nullptr && e[0] != '\0' && e[0] != '0') &&
                          jit_args_.usePutSignal == 0 && jit_args_.useWriteImm == 0 && !genRing_ &&
                          !genRingDev_;
    }
    if (fuseEntryBarrier_) {
      if (hipMalloc(&gridArrival_, 2 * sizeof(unsigned int)) != hipSuccess ||
          gridArrival_ == nullptr) {
        throw std::runtime_error("InterNodeRingAllgather: gridArrival hipMalloc failed");
      }
      (void)hipMemset(gridArrival_, 0, 2 * sizeof(unsigned int));
    }
    // Cheap-correct reuse rendezvous (env MORI_HIER_GEN_RING_DISSEM, requires
    // genRingDev_). Restores a true cross-PE entry rendezvous on the barrier-free
    // gen-ring via the O(log n) dissem barrier (not the funnel), closing the
    // DEV-alone ring-reuse E2E drift without the double-buffer.
    {
      const char* e = std::getenv("MORI_HIER_GEN_RING_DISSEM");
      genRingDissem_ = (e != nullptr && e[0] != '\0' && e[0] != '0') && genRingDev_;
    }
    // Device double-buffered ring (env MORI_HIER_GEN_RING_DBL, requires
    // genRingDev_): allocate the SECOND symmetric ring buffer (same size, so it
    // has its own cross-node rkeys/peer pointers) + the device parity counter.
    {
      const char* e = std::getenv("MORI_HIER_GEN_RING_DBL");
      genRingDbl_ = (e != nullptr && e[0] != '\0' && e[0] != '0') && genRingDev_;
    }
    if (genRingDbl_) {
      ring2_ = shmem::ShmemMalloc(ringBytes_);
      if (ring2_ == nullptr)
        throw std::runtime_error("InterNodeRingAllgather: ring2 ShmemMalloc failed");
      ring2Obj_ = shmem::ShmemQueryMemObjPtr(ring2_);
      if (hipMalloc(&parityCounter_, sizeof(uint64_t)) != hipSuccess || parityCounter_ == nullptr) {
        throw std::runtime_error("InterNodeRingAllgather: parityCounter hipMalloc failed");
      }
      (void)hipMemset(parityCounter_, 0, sizeof(uint64_t));
    }
  }

  ~InterNodeRingAllgather() {
    if (ring_) shmem::ShmemFree(ring_);
    if (flags_) shmem::ShmemFree(flags_);
    if (opGenCounter_) (void)hipFree(opGenCounter_);
    if (gridArrival_) (void)hipFree(gridArrival_);
    if (ring2_) shmem::ShmemFree(ring2_);
    if (parityCounter_) (void)hipFree(parityCounter_);
  }

  // Place this PE's input chunk into its ring slot, clear the flags, barrier so
  // every PE is primed before any remote put/atomic lands, then return a host
  // pointer to the kernel args (consumed by the Python launch_struct call).
  int64_t prepare_sync(uintptr_t input, size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }
    size_t flagsBytes = static_cast<size_t>(numBlocks_) * npes_ * sizeof(uint64_t);

    (void)hipMemsetAsync(flags_, 0, flagsBytes, stream);
    // Stage this PE's chunk into its ring-position slot (not its global PE).
    char* myChunk = reinterpret_cast<char*>(ring_) + static_cast<size_t>(ringPos_) * chunkBytes;
    (void)hipMemcpyAsync(myChunk, reinterpret_cast<void*>(input), chunkBytes,
                         hipMemcpyDeviceToDevice, stream);
    (void)hipStreamSynchronize(stream);

    // Global barrier: all PEs have cleared flags + staged their own chunk
    // before the ring (with its cross-PE atomic increments) begins. (All PEs
    // call this op -- each participates in exactly one sub-group.)
    if (!HierAllBarrierDisabled()) shmem::ShmemBarrierAll();

    jit_args_.myPe = myPe_;
    jit_args_.npes = npes_;
    jit_args_.ringPos = ringPos_;
    jit_args_.ringSize = ringSize_;
    jit_args_.peBase = peBase_;
    jit_args_.peStride = peStride_;
    jit_args_.memObj = ringObj_;
    jit_args_.flagsObj = flagsObj_;
    jit_args_.chunkBytes = chunkBytes;
    jit_args_.numQp = numQp_;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // STREAM-ORDERED prepare. Identical byte moves to
  // prepare_sync, but the cross-PE rendezvous uses the on-device
  // ShmemBarrierOnStream(stream) instead of a host-blocking
  // hipStreamSynchronize(stream) + host bootNet ShmemBarrierAll(). This keeps
  // the whole op enqueued on ``stream`` (no host round-trip), so consecutive
  // hier-AllGather phases pipeline without two CPU<->GPU stalls per op. The
  // device barrier still globally fences (all PEs) before any remote put/atomic
  // lands, so correctness is preserved.
  int64_t prepare_stream(uintptr_t input, size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }

    // NO flag memset here. The ring kernel
    // (AllGatherRingSubGroupKernelBody) resets every USED flag slot to 0 at the
    // END of each op (the `for idx<ringSize ... flagsArray[flagBase+idx]=0` loop
    // + a trailing __threadfence_system), and the constructor zeroes the whole
    // buffer once at init. So the flags buffer is ALWAYS 0 on entry to prepare,
    // making this hipMemsetAsync redundant -- it only re-clears already-zero
    // memory while adding one async op (and a tail-RAW dependency the receiver's
    // predecessor would observe) on the critical stream right before the
    // rendezvous barrier. The ShmemBarrierOnStream below still provides the
    // cross-PE ordering (every PE's op-end reset is globally visible before any
    // peer's next-op atomic increment). This documents the kernel-reset invariant
    // the flag-gated (generation-counter, barrier-free) ring lever will build on.
    char* myChunk = reinterpret_cast<char*>(ring_) + static_cast<size_t>(ringPos_) * chunkBytes;
    (void)hipMemcpyAsync(myChunk, reinterpret_cast<void*>(input), chunkBytes,
                         hipMemcpyDeviceToDevice, stream);
    // Double-buffer: stage this PE's own chunk into both ring halves each op.
    // The host copy-IN address is FROZEN under HIP-graph capture, so we cannot pick
    // the parity-selected half on the host; staging into both (two fixed copy-engine
    // memcpys, graph-safe) guarantees the kernel's device-parity-selected half always
    // holds this PE's chunk to send. The extra memcpy is one per-PE shard (cheap vs
    // the whole AG); the kernel reads/receives from exactly one half via parity.
    if (genRingDbl_ && ring2_ != nullptr) {
      char* myChunk2 = reinterpret_cast<char*>(ring2_) + static_cast<size_t>(ringPos_) * chunkBytes;
      (void)hipMemcpyAsync(myChunk2, reinterpret_cast<void*>(input), chunkBytes,
                           hipMemcpyDeviceToDevice, stream);
    }
    // On-device global barrier (no host sync): all PEs have cleared flags +
    // staged their own chunk (stream-ordered before this barrier) before the
    // ring's cross-PE atomic increments begin. dissemination
    // topology when MORI_HIER_DISSEM_BARRIER=1 (same global semantics).
    // GEN-RING: with accumulating (never-reset) flags there is no reset for a
    // peer's next-op increment to race, so this entry barrier is redundant and
    // is skipped. The copy-IN above is stream-ordered before this op's ring
    // kernel on the same stream, and the trailing finish reuse barrier still
    // orders ring-buffer reuse -- so dropping ONLY the entry fence is safe.
    if (genRing_) {
      jit_args_.opGen = ++ringOpGen_;
    } else if (genRingDev_) {
      // DEVICE gen-ring: publish the device counter; the kernel bumps it per
      // execution (so it advances under graph replay). Entry barrier dropped:
      // the accumulating (never-reset) flags mean there is no per-op flag reset
      // for a peer's next-op increment to race, so the entry rendezvous is
      // redundant (same argument as host GEN_RING) -- and made graph-safe by the
      // device-side generation.
      jit_args_.opGen = 0;
      jit_args_.opGenCounter = reinterpret_cast<uint64_t*>(opGenCounter_);
      // Optional cheap-correct reuse rendezvous. A true all-PE dissem barrier orders
      // every PE's op-N reassembly-read before any peer's op-N+1 push -- immune to the
      // graph-replay parity desync seen with the double-buffer -- at O(log n) latency
      // instead of the funnel.
      if (genRingDissem_) shmem::ShmemBarrierOnStreamDissem(stream);
    } else if (fuseEntryBarrier_) {
      // Fuse-entry-barrier: do not launch the separate entry barrier kernel.
      // The fused kernel performs the identical cross-PE rendezvous device-side at
      // its entry prologue (block 0 -> ShmemBarrierAllBlock), gated by the grid
      // arrival scratch. Publish the flag + scratch so the kernel engages it. The
      // copy-IN memcpy above is stream-ordered before the kernel, so this PE's chunk
      // is staged before the in-kernel rendezvous releases the ring push -- same
      // ordering the separate barrier gave, so the byte image is unchanged.
      jit_args_.fuseEntryBarrier = 1;
      jit_args_.gridArrival = reinterpret_cast<unsigned int*>(gridArrival_);
    } else {
      HierPrepareBarrierOnStream(stream);
    }
    // Double-buffer: publish the second ring + parity counter and bump the parity
    // on-stream before the fused kernel (captured => advances per replay).
    PublishDoubleBuffer(stream);

    jit_args_.myPe = myPe_;
    jit_args_.npes = npes_;
    jit_args_.ringPos = ringPos_;
    jit_args_.ringSize = ringSize_;
    jit_args_.peBase = peBase_;
    jit_args_.peStride = peStride_;
    jit_args_.memObj = ringObj_;
    jit_args_.flagsObj = flagsObj_;
    jit_args_.chunkBytes = chunkBytes;
    jit_args_.numQp = numQp_;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // Double-buffer helper: publish ringMemObjB + parityCounter into jit_args_.
  // The per-op parity BUMP is a JIT kernel (RingParityBumpKernel_u32) launched from
  // Python on the launch stream BEFORE the fused kernel each op (captured => advances
  // under graph replay, every kernel block reads one stable post-bump value). This TU
  // (pybind) is HOST-compiled, so it cannot launch a device kernel here. No-op unless
  // MORI_HIER_GEN_RING_DBL is engaged (single-buffer path byte-identical).
  void PublishDoubleBuffer(hipStream_t /*stream*/) {
    if (!genRingDbl_ || parityCounter_ == nullptr) return;
    jit_args_.ringMemObjB = ring2Obj_;
    jit_args_.parityCounter = reinterpret_cast<uint64_t*>(parityCounter_);
  }

 public:
  // Device pointer of the per-op parity counter (0 unless GEN_RING_DBL on),
  // so the Python launcher can fire the captured RingParityBumpKernel_u32 on
  // the ring stream before each fused-kernel launch.
  uintptr_t parity_counter_ptr() const { return reinterpret_cast<uintptr_t>(parityCounter_); }

  // Stream-ordered counterpart of prepare_sync_in_place (chunk already in slot).
  int64_t prepare_stream_in_place(size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }

    // redundant flag memset removed (see prepare_stream): the
    // ring kernel resets all used flag slots to 0 at op end + the constructor
    // zeroes once, so flags are always 0 on entry. The barrier still orders the
    // cross-PE reset-vs-next-op-increment. dissemination
    // topology when MORI_HIER_DISSEM_BARRIER=1 (same global semantics).
    // GEN-RING: accumulating flags -> no reset to order -> entry barrier dropped
    // (see prepare_stream); the in-place chunk is already staged by the caller.
    if (genRing_) {
      jit_args_.opGen = ++ringOpGen_;
    } else if (genRingDev_) {
      jit_args_.opGen = 0;
      jit_args_.opGenCounter = reinterpret_cast<uint64_t*>(opGenCounter_);
      // Cheap-correct reuse rendezvous (see prepare_stream).
      if (genRingDissem_) shmem::ShmemBarrierOnStreamDissem(stream);
    } else if (fuseEntryBarrier_) {
      // Fuse-entry-barrier: do not launch the separate entry barrier kernel.
      // The fused kernel performs the identical cross-PE rendezvous device-side at
      // its entry prologue (block 0 -> ShmemBarrierAllBlock), gated by the grid
      // arrival scratch. Publish the flag + scratch so the kernel engages it. The
      // copy-IN memcpy above is stream-ordered before the kernel, so this PE's chunk
      // is staged before the in-kernel rendezvous releases the ring push -- same
      // ordering the separate barrier gave, so the byte image is unchanged.
      jit_args_.fuseEntryBarrier = 1;
      jit_args_.gridArrival = reinterpret_cast<unsigned int*>(gridArrival_);
    } else {
      HierPrepareBarrierOnStream(stream);
    }
    // Double-buffer: in the fuse_copyin path the in-kernel stage already targets the
    // parity-selected half, so only publish + bump is needed here.
    PublishDoubleBuffer(stream);

    jit_args_.myPe = myPe_;
    jit_args_.npes = npes_;
    jit_args_.ringPos = ringPos_;
    jit_args_.ringSize = ringSize_;
    jit_args_.peBase = peBase_;
    jit_args_.peStride = peStride_;
    jit_args_.memObj = ringObj_;
    jit_args_.flagsObj = flagsObj_;
    jit_args_.chunkBytes = chunkBytes;
    jit_args_.numQp = numQp_;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // Stream-ordered counterpart of finish_sync: copy-OUT enqueued on the stream,
  // then an on-device ShmemBarrierOnStream (no host sync / host barrier) so no
  // PE reuses the ring buffer while a peer still reads it. Stays on-stream.
  //
  // ``barrier`` (default true) gates that trailing fence. The
  // fence's ONLY job is cross-PE ring-buffer reuse: it ensures every PE has
  // finished its copy-OUT (reading its LOCAL ring_) before any peer's NEXT-op
  // RDMA put overwrites that ring_. Those peer puts happen inside the next op's
  // ring kernel, which is itself preceded by that op's prepare_stream
  // ShmemBarrierOnStream (a global on-stream fence after every PE's flag-clear +
  // slot-stage). So for ANY op that has a successor through the same handle, the
  // successor's prepare fence ALREADY provides the required global ordering ->
  // this finish fence is redundant and can be deferred (barrier=false), exactly
  // mirroring the deferred Phase-B finish fence (slice_defer_fin). The
  // copy-OUT stays stream-ordered so the result is correct regardless; only the
  // cross-PE reuse fence is deferred. Callers that have no guaranteed successor
  // (last op / path switch) must keep barrier=true.
  double finish_stream(uintptr_t output, size_t count_u32, hipStream_t stream,
                       bool barrier = true) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    size_t total = static_cast<size_t>(ringSize_) * chunkBytes;
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), ring_, total, hipMemcpyDeviceToDevice,
                         stream);
    if (barrier) HierFinishBarrierOnStream(stream);
    return 0.0;
  }

  // Stream-ordered counterpart of finish_sync_no_copy (result left in ring buf).
  double finish_stream_no_copy(hipStream_t stream) {
    HierFinishBarrierOnStream(stream);
    return 0.0;
  }

  // Device pointer to THIS PE's ring slot for a given message size, so an
  // upstream producer (the intra-node SDMA gather) can write its node-block
  // DIRECTLY into the ring buffer, eliminating the prepare_sync copy-IN. The slot
  // lives at ringPos_*chunkBytes -- exactly where prepare_sync would otherwise
  // stage it.
  uintptr_t slot_ptr(size_t count_u32) const {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    return reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(ring_) +
                                       static_cast<size_t>(ringPos_) * chunkBytes);
  }

  // Like prepare_sync but WITHOUT the copy-IN: the caller has already written
  // this PE's chunk into slot_ptr(count_u32) (e.g. the intra gather targeted the
  // ring slot). We only clear the flags, barrier so every PE is primed before
  // any remote put/atomic lands, and build the kernel args. Saves one full
  // chunk D2D copy per call.
  int64_t prepare_sync_in_place(size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }
    size_t flagsBytes = static_cast<size_t>(numBlocks_) * npes_ * sizeof(uint64_t);

    (void)hipMemsetAsync(flags_, 0, flagsBytes, stream);
    (void)hipStreamSynchronize(stream);

    // Global barrier: all PEs have cleared flags + staged their own chunk (the
    // upstream gather already wrote into slot_ptr) before the ring begins.
    if (!HierAllBarrierDisabled()) shmem::ShmemBarrierAll();

    jit_args_.myPe = myPe_;
    jit_args_.npes = npes_;
    jit_args_.ringPos = ringPos_;
    jit_args_.ringSize = ringSize_;
    jit_args_.peBase = peBase_;
    jit_args_.peStride = peStride_;
    jit_args_.memObj = ringObj_;
    jit_args_.flagsObj = flagsObj_;
    jit_args_.chunkBytes = chunkBytes;
    jit_args_.numQp = numQp_;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // After the ring kernel completes, copy the full ringSize*chunk result out to
  // the user output buffer (ring order) and synchronize.
  double finish_sync(uintptr_t output, size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    size_t total = static_cast<size_t>(ringSize_) * chunkBytes;
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), ring_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    // Barrier so no PE frees/reuses the ring buffer while a peer is still
    // reading from it in a subsequent op.
    if (!HierAllBarrierDisabled()) shmem::ShmemBarrierAll();
    return 0.0;
  }

  // Base device pointer of the full ring buffer. After the ring kernel completes,
  // ring_ already holds the ringSize_ chunks in ring order = the full rank-major
  // result. A consumer that reads its output DIRECTLY from here (via ``buf_ptr``
  // + ``finish_sync_no_copy``) avoids the finish_sync copy-OUT (a ringSize*chunk
  // D2D copy).
  uintptr_t buf_ptr() const { return reinterpret_cast<uintptr_t>(ring_); }

  // Like finish_sync but WITHOUT the copy-OUT: the gathered result is left in
  // the ring buffer (read it via ``buf_ptr``). Only synchronizes the stream and
  // barriers so no PE frees/reuses the ring buffer while a peer is still
  // reading from it. Saves one full ringSize*chunk D2D copy per call.
  //
  // Asymmetry vs the copy-IN elimination: the ring kernel already writes every
  // received chunk into the UNCACHED symmetric ring_ buffer, so removing the
  // copy-OUT is a pure saving -- there is no offsetting uncached write the way
  // copy-IN incurred. The only cost shifted to the consumer is reading its result
  // from uncached memory, which is outside the timed AllGather.
  double finish_sync_no_copy(hipStream_t stream) {
    (void)hipStreamSynchronize(stream);
    if (!HierAllBarrierDisabled()) shmem::ShmemBarrierAll();
    return 0.0;
  }

  int npes() const { return npes_; }
  int num_blocks() const { return numBlocks_; }
};

}  // namespace collective
}  // namespace mori

#endif  // INTER_NODE_RING_CLASS_HPP

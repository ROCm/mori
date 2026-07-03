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

// select the dissemination barrier for the inter-ring prepare
// rendezvous when MORI_HIER_DISSEM_BARRIER!=0. Read once (env is fixed for the
// process). The dissem barrier has identical global all-PE semantics but an
// O(log n) parallel critical path instead of the PE0 funnel ( residual).
inline bool HierDissemBarrierEnabled() {
  static const bool enabled = []() {
    const char* e = std::getenv("MORI_HIER_DISSEM_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return enabled;
}

// MEASUREMENT-ONLY: when MORI_HIER_NO_ENTRY_BARRIER!=0 the ring's prepare entry
// rendezvous barrier is SKIPPED. This is NOT correctness-safe (it drops the
// cross-PE fence that orders every PE's op-end flag-reset + own-chunk staging
// before any peer's next-op atomic increment) -- it exists solely to quantify
// the EXPOSED per-op barrier cost under back-to-back FSDP all-gathers (the
// standalone UT device-syncs between reps so it never exposes op-to-op barrier
// serialization; FSDP issues ~65 AGs/step back-to-back). If skipping recovers
// the FSDP gap vs RCCL, the generation-counter barrier-free ring (see the
// prepare_stream flag-reset invariant note) is the justified fix.
inline bool HierEntryBarrierDisabled() {
  static const bool disabled = []() {
    const char* e = std::getenv("MORI_HIER_NO_ENTRY_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return disabled;
}

// MEASUREMENT-ONLY master switch: when MORI_HIER_NO_ALL_BARRIER!=0 EVERY
// cross-PE barrier (entry ShmemBarrierOnStream, the deferred finish fences, the
// host ShmemBarrierAll rendezvous in both the inter-node ring AND the intra-node
// subgroup gather) is skipped. NOT correctness-safe. This is the reviewer's
// decisive all-barrier-removal A/B: Turn 10 only removed the ENTRY barrier
// (+1% on a healthy cluster); this removes the FINISH + intra barriers too, to
// finally decide whether per-op barrier SKEW (each AG waiting for the slowest
// rank's backward GEMM) is the residual FSDP gap vs RCCL. If flat hier recovers
// to ~RCCL, the generation-counter barrier-free ring is justified; if it stays
// ~110, barrier-skew is ruled out and the gap is elsewhere.
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

  // Sub-group descriptor (M2b). The ring runs over the arithmetic sub-group of
  // global PEs {peBase_, peBase_+peStride_, ..., peBase_+(ringSize_-1)*peStride_};
  // this PE is at position ringPos_. The whole-world ring is the flat default
  // peBase_=0, peStride_=1, ringSize_=npes_, ringPos_=myPe_.
  int ringPos_;
  int ringSize_;
  int peBase_;
  int peStride_;
  // M4: RDMA QP fan-out degree for the per-round ring put. 1 keeps the
  // original single-QP put; >1 fans the chunk across QPs (RDMA neighbours only,
  // gated at runtime in the kernel). The hierarchical inter-node ring passes >1.
  int numQp_;

  // M4: number of CTAs ("channels") the ring kernel is launched with.
  // 1 keeps the original single-block ring. >1 partitions each chunk into
  // numBlocks_ disjoint sub-ranges, one CTA each (qpId=bid) -- RCCL-style. Used
  // only to size the per-block flag regions (numBlocks_*ringSize slots); the
  // kernel reads the actual block count from gridDim.x at launch.
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
    if (ring_ == nullptr) throw std::runtime_error("InterNodeRingAllgather: ring ShmemMalloc failed");
    ringObj_ = shmem::ShmemQueryMemObjPtr(ring_);

    // Flags: one uint64 per ring slot PER BLOCK. Allocate numBlocks_*npes
    // (>= numBlocks_*ringSize_) so each CTA channel gets its own flag region and
    // the buffer is large enough for any sub-group on this PE set.
    size_t flagsBytes = static_cast<size_t>(numBlocks_) * npes_ * sizeof(uint64_t);
    flags_ = shmem::ShmemMalloc(flagsBytes);
    if (flags_ == nullptr) throw std::runtime_error("InterNodeRingAllgather: flags ShmemMalloc failed");
    (void)hipMemset(flags_, 0, flagsBytes);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_);
  }

  ~InterNodeRingAllgather() {
    if (ring_) shmem::ShmemFree(ring_);
    if (flags_) shmem::ShmemFree(flags_);
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
  // lands, so correctness is preserved. (Cross-read: this work  measured
  // +6-7% standalone from removing these host round-trips.)
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
    // On-device global barrier (no host sync): all PEs have cleared flags +
    // staged their own chunk (stream-ordered before this barrier) before the
    // ring's cross-PE atomic increments begin. dissemination
    // topology when MORI_HIER_DISSEM_BARRIER=1 (same global semantics).
    HierPrepareBarrierOnStream(stream);

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
    HierPrepareBarrierOnStream(stream);

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
  // mirroring the deferred Phase-B finish fence (slice_defer_fin, ). The
  // copy-OUT stays stream-ordered so the result is correct regardless; only the
  // cross-PE reuse fence is deferred. Callers that have no guaranteed successor
  // (last op / path switch) must keep barrier=true.
  double finish_stream(uintptr_t output, size_t count_u32, hipStream_t stream,
                       bool barrier = true) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    size_t total = static_cast<size_t>(ringSize_) * chunkBytes;
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), ring_, total, hipMemcpyDeviceToDevice,
                         stream);
    if (barrier && !HierAllBarrierDisabled()) shmem::ShmemBarrierOnStream(stream);
    return 0.0;
  }

  // Stream-ordered counterpart of finish_sync_no_copy (result left in ring buf).
  double finish_stream_no_copy(hipStream_t stream) {
    if (!HierAllBarrierDisabled()) shmem::ShmemBarrierOnStream(stream);
    return 0.0;
  }

  // M4: device pointer to THIS PE's ring slot for a given message
  // size, so an upstream producer (the intra-node SDMA gather) can write its
  // node-block DIRECTLY into the ring buffer, eliminating the prepare_sync
  // copy-IN (~1.4ms @256MiB,  phase attribution). The slot lives at
  // ringPos_*chunkBytes -- exactly where prepare_sync would otherwise stage it.
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

  // M4: base device pointer of the full ring buffer. After the ring
  // kernel completes, ring_ already holds the ringSize_ chunks in ring order =
  // the full rank-major result. A consumer that reads its output DIRECTLY from
  // here (via ``buf_ptr`` + ``finish_sync_no_copy``) avoids the finish_sync
  // copy-OUT (a ringSize*chunk D2D copy, ~2.7ms @512MiB,  attribution).
  uintptr_t buf_ptr() const { return reinterpret_cast<uintptr_t>(ring_); }

  // Like finish_sync but WITHOUT the copy-OUT: the gathered result is left in
  // the ring buffer (read it via ``buf_ptr``). Only synchronizes the stream and
  // barriers so no PE frees/reuses the ring buffer while a peer is still
  // reading from it. Saves one full ringSize*chunk D2D copy per call.
  //
  // ASYMMETRY vs the copy-IN elimination (-24, validated-NEUTRAL): the
  // ring kernel already writes every received chunk into the UNCACHED symmetric
  // ring_ buffer, so removing the copy-OUT is a pure saving -- there is no
  // offsetting uncached write the way copy-IN incurred. The only cost shifted to
  // the consumer is reading its result from uncached memory, which is OUTSIDE
  // the timed AllGather. So this is expected to be a real win, not a wash.
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

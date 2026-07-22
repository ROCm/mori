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

// Inter-node RDMA ring AllGather host handle: one chunk per PE over the shmem
// transport (P2P intra-node, RDMA inter-node) using the ring schedule in
// inter_node/kernels/all_gather.hpp. Owns the symmetric ring buffer + flags and
// builds the JIT kernel args; driven from Python (prepare -> kernel -> finish).

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

// MORI_HIER_DISSEM_BARRIER!=0 selects the dissemination barrier (read once).
inline bool HierDissemBarrierEnabled() {
  static const bool enabled = []() {
    const char* e = std::getenv("MORI_HIER_DISSEM_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return enabled;
}

// Ring entry rendezvous. Funnel by default; dissem variant has identical global
// all-PE ordering, so the default path is byte-identical.
inline void HierPrepareBarrierOnStream(hipStream_t stream) {
  if (HierDissemBarrierEnabled()) {
    shmem::ShmemBarrierOnStreamDissem(stream);
  } else {
    shmem::ShmemBarrierOnStream(stream);
  }
}

// Ring finish / cross-PE reuse fence. Same full-rendezvous semantics as the
// prepare barrier: the dissem variant preserves the NIC-landing -> consume
// ordering, so the default path stays byte-for-byte identical.
inline void HierFinishBarrierOnStream(hipStream_t stream) {
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

  // Sub-group descriptor: the ring runs over PEs {peBase_ + i*peStride_ :
  // i<ringSize_}, this PE at ringPos_. Flat default = 0/1/npes_/myPe_.
  int ringPos_;
  int ringSize_;
  int peBase_;
  int peStride_;
  // RDMA QP fan-out for the per-round put (>1 fans across QPs, RDMA neighbours
  // only, gated in the kernel).
  int numQp_;

  // CTA ("channel") count. >1 partitions each chunk into numBlocks_ disjoint
  // sub-ranges, one CTA each (qpId=bid). Only sizes the per-block flag regions
  // (numBlocks_*ringSize slots); the kernel reads the block count from gridDim.x.
  int numBlocks_;

  // Symmetric ring buffer of ringSize_ contiguous chunks. Sized for the largest
  // message; the kernel only touches the first ringSize_*chunkBytes.
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
    if (ring_ == nullptr)
      throw std::runtime_error("InterNodeRingAllgather: ring ShmemMalloc failed");
    ringObj_ = shmem::ShmemQueryMemObjPtr(ring_);

    // One uint64 per ring slot per block: numBlocks_*npes (>= numBlocks_*ringSize_)
    // gives each CTA channel its own flag region, sized for any sub-group here.
    size_t flagsBytes = static_cast<size_t>(numBlocks_) * npes_ * sizeof(uint64_t);
    flags_ = shmem::ShmemMalloc(flagsBytes);
    if (flags_ == nullptr)
      throw std::runtime_error("InterNodeRingAllgather: flags ShmemMalloc failed");
    (void)hipMemset(flags_, 0, flagsBytes);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_);

    // WRITE-PUSH (SEND-CQ) landing fence: each channel drains its own SEND CQE
    // after a fused put-with-signal (env MORI_HIER_RING_WRITE, default OFF).
    {
      const char* e = std::getenv("MORI_HIER_RING_WRITE");
      jit_args_.useWriteFence = (e != nullptr && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
  }

  ~InterNodeRingAllgather() {
    if (ring_) shmem::ShmemFree(ring_);
    if (flags_) shmem::ShmemFree(flags_);
  }

  // Stage this PE's chunk, clear flags, barrier so every PE is primed before any
  // remote put/atomic lands, then return the host pointer to the kernel args.
  int64_t prepare_sync(uintptr_t input, size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }
    size_t flagsBytes = static_cast<size_t>(numBlocks_) * npes_ * sizeof(uint64_t);

    (void)hipMemsetAsync(flags_, 0, flagsBytes, stream);
    // Stage into ring-position slot (not global PE).
    char* myChunk = reinterpret_cast<char*>(ring_) + static_cast<size_t>(ringPos_) * chunkBytes;
    (void)hipMemcpyAsync(myChunk, reinterpret_cast<void*>(input), chunkBytes,
                         hipMemcpyDeviceToDevice, stream);
    (void)hipStreamSynchronize(stream);

    shmem::ShmemBarrierAll();

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

  // Stream-ordered prepare_sync: rendezvous via on-device ShmemBarrierOnStream
  // (no host sync) so the op stays on ``stream``. Still globally fences all PEs
  // before any remote put/atomic lands.
  int64_t prepare_stream(uintptr_t input, size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }

    // No flag memset: the ring kernel resets every used slot to 0 at op end
    // (+ trailing __threadfence_system) and the constructor zeroes once, so the
    // buffer is always 0 on entry. The barrier below still orders each PE's
    // op-end reset before any peer's next-op atomic increment.
    char* myChunk = reinterpret_cast<char*>(ring_) + static_cast<size_t>(ringPos_) * chunkBytes;
    (void)hipMemcpyAsync(myChunk, reinterpret_cast<void*>(input), chunkBytes,
                         hipMemcpyDeviceToDevice, stream);
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

 public:
  // Stream-ordered counterpart of prepare_sync_in_place (chunk already in slot).
  int64_t prepare_stream_in_place(size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }

    // No flag memset (see prepare_stream): flags are always 0 on entry.
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

  // Stream-ordered finish_sync. ``barrier`` (default true) gates the trailing
  // fence whose only job is cross-PE ring reuse; it may be deferred (false) when
  // a successor op's prepare barrier already provides that ordering. Last op /
  // path switch must keep barrier=true.
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

  // Device pointer to this PE's ring slot (at ringPos_*chunkBytes), so an
  // upstream producer can write directly into the ring, eliminating the copy-IN.
  uintptr_t slot_ptr(size_t count_u32) const {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    return reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(ring_) +
                                       static_cast<size_t>(ringPos_) * chunkBytes);
  }

  // prepare_sync without the copy-IN: caller already wrote its chunk into
  // slot_ptr(count_u32). Clear flags, barrier, build args.
  int64_t prepare_sync_in_place(size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    if (static_cast<size_t>(ringSize_) * chunkBytes > ringBytes_) {
      throw std::runtime_error("InterNodeRingAllgather: message exceeds ring buffer capacity");
    }
    size_t flagsBytes = static_cast<size_t>(numBlocks_) * npes_ * sizeof(uint64_t);

    (void)hipMemsetAsync(flags_, 0, flagsBytes, stream);
    (void)hipStreamSynchronize(stream);

    shmem::ShmemBarrierAll();

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

  // Copy the full ringSize*chunk result (ring order) out to the user buffer.
  double finish_sync(uintptr_t output, size_t count_u32, hipStream_t stream) {
    size_t chunkBytes = count_u32 * sizeof(uint32_t);
    size_t total = static_cast<size_t>(ringSize_) * chunkBytes;
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), ring_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    // Barrier so no PE reuses the ring buffer while a peer still reads it.
    shmem::ShmemBarrierAll();
    return 0.0;
  }

  // Base pointer of the ring buffer, which after the kernel holds the ringSize_
  // chunks in ring order (the rank-major result). Read via buf_ptr +
  // finish_sync_no_copy to avoid the finish_sync copy-OUT.
  uintptr_t buf_ptr() const { return reinterpret_cast<uintptr_t>(ring_); }

  // finish_sync without the copy-OUT: result left in the ring buffer (read via
  // buf_ptr). Only syncs the stream and barriers against cross-PE reuse.
  double finish_sync_no_copy(hipStream_t stream) {
    (void)hipStreamSynchronize(stream);
    shmem::ShmemBarrierAll();
    return 0.0;
  }

};

}  // namespace collective
}  // namespace mori

#endif  // INTER_NODE_RING_CLASS_HPP

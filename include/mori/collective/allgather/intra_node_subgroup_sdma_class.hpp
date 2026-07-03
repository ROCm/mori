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

// Sub-group intra-node SDMA AllGather host handle.
//
// This is the intra-node phase of the hierarchical cross-node AllGather: the
// ``G`` local ranks of one node gather their ``G`` shards over the SDMA copy
// engines (XGMI), so every local rank ends up holding its node's contiguous
// G-shard block. It is the SDMA analogue of ``InterNodeRingAllgather`` and is
// driven from Python with the same prepare -> launch kernel -> finish pattern
// as ``AllgatherSdma``. The destination is a symmetric transit buffer holding
// ``groupSize`` contiguous shard slots; the kernel SDMA-writes each member's
// shard into its group-position slot on every member.

#ifndef INTRA_NODE_SUBGROUP_SDMA_CLASS_HPP
#define INTRA_NODE_SUBGROUP_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <utility>

#include "mori/collective/ccl_kernel_args.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// MEASUREMENT-ONLY (mirror of inter_node_ring_class HierAllBarrierDisabled): when
// MORI_HIER_NO_ALL_BARRIER!=0 the intra-node subgroup gather's cross-PE barriers
// are skipped too, so the reviewer's all-barrier-removal A/B covers BOTH phases
// (inter ring + intra gather). NOT correctness-safe.
inline bool IntraHierAllBarrierDisabled() {
  static const bool disabled = []() {
    const char* e = std::getenv("MORI_HIER_NO_ALL_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return disabled;
}

class IntraNodeSubGroupAllgatherSdma {
 private:
  int myPe_;
  int npes_;

  // Sub-group descriptor. The gather runs over the arithmetic sub-group of
  // global PEs {peBase_, peBase_+peStride_, ..., peBase_+(groupSize_-1)*peStride_};
  // this PE is at position groupPos_. The flat whole-world gather is the
  // default groupSize_=npes_, groupPos_=myPe_, peBase_=0, peStride_=1.
  int groupSize_;
  int groupPos_;
  int peBase_;
  int peStride_;

  // Symmetric output transit buffer: holds groupSize_ contiguous shard slots.
  // Registered (not just malloc'd) so the SDMA queue handles are populated.
  void* out_;
  size_t outBytes_;
  application::SymmMemObjPtr outObj_;

  // Per-position arrival flags (one uint64 per group position; npes_ allocated
  // so any sub-group on this PE set fits). Monotonic generation token avoids a
  // per-call reset.
  void* flags_;
  application::SymmMemObjPtr flagsObj_;
  uint64_t seq_;

  CclAllgatherSubGroupArgs<uint32_t> jit_args_;
  CclAllgatherSubGroupParamContiguousArgs<uint32_t> jit_args_pc_;

  // DIRECT-TO-OUTPUT registration. Maps a user output buffer
  // base address -> its symmetric mem object + size. When the fused sliced
  // Phase-B gathers PUSH directly into the (registered) user output instead of
  // the internal ``out_`` transit, the full-output ``finish_batch`` copy-OUT
  // (N*block bytes of pure D2D HBM traffic on the critical path) is eliminated.
  // Mirrors AllgatherSdma::registered_output_buffers_ (oneshot_allgather_sdma_
  // class.cpp). Registration is COLLECTIVE (ShmemSymmetricRegister all-gathers
  // peer pointers + opens IPC handles), so it is cached and only re-run when the
  // user passes a not-yet-seen output pointer. Under SPMD all PEs register in
  // lockstep, so the cache state stays symmetric (no barrier divergence).
  struct RegEntry {
    application::SymmMemObjPtr obj;
    size_t size;
  };
  std::map<uintptr_t, RegEntry> registered_outputs_;

  std::pair<application::SymmMemObjPtr, size_t> find_registered(uintptr_t ptr) const {
    if (ptr == 0) return {application::SymmMemObjPtr{}, 0};
    auto it = registered_outputs_.upper_bound(ptr);
    if (it != registered_outputs_.begin()) {
      --it;
      uintptr_t base = it->first;
      if (ptr >= base && ptr < base + it->second.size) {
        return {it->second.obj, ptr - base};
      }
    }
    return {application::SymmMemObjPtr{}, 0};
  }

  // EXACT-base lookup for the direct path. The hierarchical
  // direct gather always passes the user output's BASE pointer as ``output_ptr``
  // (per-node-block offsets are added via dst_block_offset, not via a sub-buffer
  // base), so the safe, unambiguous key is an exact base match -- byteOffset is
  // always 0. Range-containment (find_registered) is unsafe under torch's
  // caching allocator: a FRESH output can be carved INSIDE the address range of
  // a previously-registered-but-since-freed segment, so a range hit returns a
  // STALE SymmMemObj (peer IPC pointers gathered for the old allocation) ->
  // off-by-rank corruption (observed got=ref+17). Exact match + stale
  // eviction (register_output_buffer) guarantees the only live entry for a base
  // is the current registration.
  application::SymmMemObjPtr find_exact(uintptr_t ptr) const {
    auto it = registered_outputs_.find(ptr);
    if (it == registered_outputs_.end()) return application::SymmMemObjPtr{};
    return it->second.obj;
  }

  IntraNodeSubGroupAllgatherSdma(const IntraNodeSubGroupAllgatherSdma&) = delete;
  IntraNodeSubGroupAllgatherSdma& operator=(const IntraNodeSubGroupAllgatherSdma&) = delete;

 public:
  // groupSize<0 selects the flat whole-world gather (groupSize=npes,
  // groupPos=myPe, peBase=0, peStride=1).
  IntraNodeSubGroupAllgatherSdma(int myPe, int npes, size_t out_buffer_bytes, int groupSize = -1,
                                 int groupPos = -1, int peBase = 0, int peStride = 1)
      : myPe_(myPe),
        npes_(npes),
        out_(nullptr),
        outBytes_(out_buffer_bytes),
        flags_(nullptr),
        seq_(0) {
    if (groupSize < 0) {
      groupSize_ = npes_;
      groupPos_ = myPe_;
      peBase_ = 0;
      peStride_ = 1;
    } else {
      if (groupSize < 1 || groupPos < 0 || groupPos >= groupSize || peStride < 1) {
        throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: invalid sub-group descriptor");
      }
      groupSize_ = groupSize;
      groupPos_ = groupPos;
      peBase_ = peBase;
      peStride_ = peStride;
    }

    out_ = shmem::ShmemMalloc(outBytes_);
    if (out_ == nullptr)
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: out ShmemMalloc failed");
    outObj_ = shmem::ShmemSymmetricRegister(out_, outBytes_);
    if (!outObj_.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: out register failed");

    size_t flagsBytes = static_cast<size_t>(npes_) * sizeof(uint64_t);
    flags_ = shmem::ShmemMalloc(flagsBytes);
    if (flags_ == nullptr)
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: flags ShmemMalloc failed");
    (void)hipMemset(flags_, 0, flagsBytes);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_);
    if (!flagsObj_.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: flags query failed");
  }

  ~IntraNodeSubGroupAllgatherSdma() {
    if (out_) shmem::ShmemFree(out_);
    if (flags_) shmem::ShmemFree(flags_);
  }

  // Barrier so all members are primed (flags already monotonic), then build the
  // kernel args. ``input`` is this PE's shard (device ptr, count_u32 u32 lanes).
  //
  // M4: ``barrier`` lets a caller SKIP this entry ShmemBarrierAll.
  // The barrier's only role is to ensure every member's ``out_`` transit buffer
  // is free (no peer still reading it) before any peer SDMA-pushes into it, and
  // that all members have registered ``out_``. When this gather is the FIRST
  // phase of a pipeline whose PREVIOUS iteration ended with a global
  // ShmemBarrierAll (e.g. the inter-node ring's finish_sync barrier in the
  // hierarchical AllGather), that prior barrier already provides the same
  // guarantee: every peer is past it, and its ``out_`` was last read in the
  // prior iteration's finish_sync (well before the prior barrier), so it is
  // free. Flags are monotonic (per-call token, no reset) so there is no
  // cross-call flag hazard either. The caller MUST keep ``barrier=true`` on the
  // FIRST call (no prior global barrier exists post-construction to cover the
  // out_ registration / freshness). Default ``barrier=true`` keeps the
  // standalone contract byte-for-byte unchanged.
  // M5: ``dst_base_offset_bytes`` places this gather's groupSize-slot
  // block at a non-zero byte offset inside ``out_``. Used by the fused sliced
  // path: each of the N reassembly gathers writes its node-block into a DISJOINT
  // region [m*block_bytes, (m+1)*block_bytes) of an enlarged transit, so they
  // never overlap and the per-gather finish barrier + per-gather copy-OUT can be
  // dropped (replaced by ONE bulk copy in ``finish_batch``). Default 0 keeps the
  // single-block contract byte-for-byte unchanged.
  // M5: ``dst_slot_stride_bytes`` decouples the per-peer destination
  // slot stride from the copy size (count_u32 u32 lanes). Default 0 packs slots
  // contiguously (stride == copy size), byte-for-byte unchanged. A non-zero
  // stride lets a chunk land at its strided position inside a full-size block --
  // the chunked inter/intra reassembly pipeline enabler. The last slot ends at
  // dst_base_offset + (groupSize-1)*slotStride + copyBytes, which must fit out_.
  int64_t prepare_sync(uintptr_t input, size_t count_u32, hipStream_t stream, bool barrier = true,
                       size_t dst_base_offset_bytes = 0, size_t dst_slot_stride_bytes = 0) {
    size_t copy_bytes = count_u32 * sizeof(uint32_t);
    size_t slot_stride = dst_slot_stride_bytes != 0 ? dst_slot_stride_bytes : copy_bytes;
    size_t last_slot_end =
        dst_base_offset_bytes + static_cast<size_t>(groupSize_ - 1) * slot_stride + copy_bytes;
    if (last_slot_end > outBytes_) {
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: message exceeds out capacity");
    }
    (void)stream;
    uint64_t flag_token = ++seq_;

    // All members enter the gather together (flags are monotonic; the token
    // distinguishes this call from the previous one without a reset).
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierAll();

    jit_args_.myPe = myPe_;
    jit_args_.npes = npes_;
    jit_args_.groupSize = groupSize_;
    jit_args_.groupPos = groupPos_;
    jit_args_.peBase = peBase_;
    jit_args_.peStride = peStride_;
    jit_args_.input = reinterpret_cast<uint32_t*>(input);
    jit_args_.dstMemObj = outObj_;
    jit_args_.flagsMemObj = flagsObj_;
    jit_args_.elementCount = count_u32;
    jit_args_.dstBaseOffset = dst_base_offset_bytes;
    jit_args_.dstSlotStrideBytes = dst_slot_stride_bytes;
    jit_args_.flagVal = flag_token;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // M5: bulk copy-OUT for the fused sliced path. After N gathers have
  // each written a disjoint block into ``out_`` (via ``dst_base_offset_bytes``),
  // this copies ``total_count_u32`` contiguous u32 lanes (= N*groupSize*chunk)
  // straight to the user output in ONE memcpy + ONE stream sync, then one
  // barrier -- replacing the N per-gather finish_sync copies/syncs/barriers.
  double finish_batch(uintptr_t output, size_t total_count_u32, hipStream_t stream,
                      bool barrier = true) {
    if (total_count_u32 * sizeof(uint32_t) > outBytes_) {
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: batch exceeds out capacity");
    }
    size_t total = total_count_u32 * sizeof(uint32_t);
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierAll();
    return 0.0;
  }

  // STREAM-ORDERED counterpart of finish_batch. Identical
  // bulk copy-OUT, but the cross-PE rendezvous uses the on-device
  // ShmemBarrierOnStream(stream) instead of a host-blocking
  // hipStreamSynchronize(stream) + host bootNet ShmemBarrierAll(). This removes
  // the LAST host CPU<->GPU round-trip in the fused sliced Phase-B, so the whole
  // hier-AllGather op (stream-ordered inter ring + Phase-B gathers + this
  // copy-OUT) stays enqueued on ``stream`` with NO host stall. The device
  // barrier still globally fences (all PEs) so no peer reuses ``out_`` while a
  // peer still reads it in a subsequent op -- the same guarantee the
  // ShmemBarrierAll provided, just stream-ordered. Pairs with the Turn-10
  // stream-ordered inter ring. (Same lever family as InterNodeRing::finish_stream.)
  //
  // ``barrier`` lets a caller DEFER the trailing
  // ShmemBarrierOnStream. In the steady-state fused sliced op this finish fence
  // is back-to-back (across the op boundary) with the NEXT op's inter-ring
  // prepare ShmemBarrierOnStream, which already globally fences (all PEs) AFTER
  // this op's copy-OUT and BEFORE any peer reuses the shared transit/ring
  // buffers -- so this fence is redundant for every op that is followed by
  // another hier op. The copy-OUT is still stream-ordered, so THIS PE's output
  // is correct without the fence; only cross-PE buffer REUSE needs it, and the
  // next op's prepare barrier provides exactly that. Pass barrier=false to drop
  // it; the LAST op (no successor) leaves the result correct anyway.
  double finish_batch_stream(uintptr_t output, size_t total_count_u32, hipStream_t stream,
                             bool barrier = true) {
    if (total_count_u32 * sizeof(uint32_t) > outBytes_) {
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: batch exceeds out capacity");
    }
    size_t total = total_count_u32 * sizeof(uint32_t);
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierOnStream(stream);
    return 0.0;
  }

  // register a user output buffer for DIRECT-TO-OUTPUT
  // gathers. COLLECTIVE (ShmemSymmetricRegister all-gathers peer pointers +
  // opens same-node IPC handles), so every PE must call it in lockstep. Cached:
  // a no-op if ``ptr`` is already covered by a prior registration.
  void register_output_buffer(uintptr_t ptr, size_t size) {
    // Cache hit ONLY on an exact base with the same extent -- same physical
    // allocation (torch caching allocator reuses an address => same pages =>
    // peer IPC pointers still valid). Re-registering would be wasted collective
    // work.
    auto exact = registered_outputs_.find(ptr);
    if (exact != registered_outputs_.end() && exact->second.size == size) return;
    // Evict EVERY stale entry whose range overlaps [ptr, ptr+size) (including an
    // exact base with a different size). Under the torch caching allocator a new
    // output reuses/splits a previously-registered-then-freed segment, so any
    // overlapping prior registration is stale and must be deregistered before
    // re-registering the new extent. Deregistration is collective, but the alloc
    // sequence is identical across PEs (SPMD, deterministic allocator) so the
    // eviction set stays symmetric -> no collective divergence.
    for (auto it = registered_outputs_.begin(); it != registered_outputs_.end();) {
      uintptr_t b = it->first;
      size_t s = it->second.size;
      bool overlap = (ptr < b + s) && (b < ptr + size);
      if (overlap) {
        shmem::ShmemSymmetricDeregister(reinterpret_cast<void*>(b), s);
        it = registered_outputs_.erase(it);
      } else {
        ++it;
      }
    }
    auto obj = shmem::ShmemSymmetricRegister(reinterpret_cast<void*>(ptr), size);
    if (!obj.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: output register failed");
    registered_outputs_[ptr] = {obj, size};
  }

  void deregister_output_buffer(uintptr_t ptr) {
    auto it = registered_outputs_.find(ptr);
    if (it == registered_outputs_.end()) return;
    shmem::ShmemSymmetricDeregister(reinterpret_cast<void*>(ptr), it->second.size);
    registered_outputs_.erase(it);
  }

  // Exact-base + same-extent so a torch realloc at the same address with a
  // different size (dispatch path-switch) forces re-registration.
  bool is_output_registered(uintptr_t ptr, size_t size) const {
    auto it = registered_outputs_.find(ptr);
    return it != registered_outputs_.end() && it->second.size == size;
  }

  // DIRECT gather -- build args that SDMA-PUSH each member's
  // slice straight into the (registered) user output, no internal transit.
  // ``output_ptr`` must lie inside a previously-registered buffer; the gather's
  // groupSize-slot block is placed at byte offset (output_ptr - regBase) +
  // ``dst_block_offset_bytes`` within that buffer, with the same slot-stride
  // semantics as prepare_sync. The caller then needs only a global fence (no
  // copy-OUT) to complete the op. Throws if ``output_ptr`` is not registered.
  int64_t prepare_sync_direct(uintptr_t input, size_t count_u32, hipStream_t stream, bool barrier,
                              uintptr_t output_ptr, size_t dst_block_offset_bytes = 0,
                              size_t dst_slot_stride_bytes = 0) {
    auto regObj = find_exact(output_ptr);
    if (!regObj.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: output not registered for direct");
    size_t copy_bytes = count_u32 * sizeof(uint32_t);
    size_t slot_stride = dst_slot_stride_bytes != 0 ? dst_slot_stride_bytes : copy_bytes;
    size_t base_off = dst_block_offset_bytes;
    size_t last_slot_end = base_off + static_cast<size_t>(groupSize_ - 1) * slot_stride + copy_bytes;
    if (last_slot_end > regObj->size) {
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: direct gather exceeds output");
    }
    uint64_t flag_token = ++seq_;
    // keep the entry fence STREAM-ORDERED (no host CPU<->GPU
    // round-trip) to match the rest of the direct path, which is fully
    // on-stream (stream_ring + stream_intra). ShmemBarrierOnStream globally
    // fences all PEs (same guarantee as the host ShmemBarrierAll) but enqueued
    // on ``stream`` so it never stalls the launch thread. In the shipped config
    // (slice_fuse_ib ON) this barrier is skipped entirely (barrier=false); it
    // only fires in the fuse_ib-OFF A/B path, where it must stay on-stream to
    // avoid mixing a host barrier into the stream-ordered sequence.
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierOnStream(stream);
    jit_args_.myPe = myPe_;
    jit_args_.npes = npes_;
    jit_args_.groupSize = groupSize_;
    jit_args_.groupPos = groupPos_;
    jit_args_.peBase = peBase_;
    jit_args_.peStride = peStride_;
    jit_args_.input = reinterpret_cast<uint32_t*>(input);
    jit_args_.dstMemObj = regObj;
    jit_args_.flagsMemObj = flagsObj_;
    jit_args_.elementCount = count_u32;
    jit_args_.dstBaseOffset = base_off;
    jit_args_.dstSlotStrideBytes = dst_slot_stride_bytes;
    jit_args_.flagVal = flag_token;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // FUSED param-contiguous direct gather: build the jit_args for the single
  // OneShotAllGatherSdmaSubGroupParamContiguousKernel launch that scatters ALL
  // node blocks * param splits into the (registered) user output in one launch,
  // replacing the per-(block,param) prepare_sync_direct loop. ``split_*_ptr``
  // are DEVICE pointers to size_t arrays in u32-lane units (shared across
  // blocks). ``block_stride_u32`` is the per-node-block stride in the Phase-A
  // input collection; ``world_size`` == npes.
  int64_t prepare_sync_direct_param_contiguous(uintptr_t input, hipStream_t stream, bool barrier,
                                               uintptr_t output_ptr, size_t block_stride_u32,
                                               int num_blocks, size_t world_size,
                                               uintptr_t split_sizes_ptr, uintptr_t split_offsets_ptr,
                                               size_t split_count, size_t dst_block_offset_bytes = 0,
                                               int first_block = 0) {
    auto regObj = find_exact(output_ptr);
    if (!regObj.IsValid())
      throw std::runtime_error(
          "IntraNodeSubGroupAllgatherSdma: output not registered for direct param-contiguous");
    uint64_t flag_token = ++seq_;
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierOnStream(stream);
    jit_args_pc_.myPe = myPe_;
    jit_args_pc_.npes = npes_;
    jit_args_pc_.groupSize = groupSize_;
    jit_args_pc_.groupPos = groupPos_;
    jit_args_pc_.peBase = peBase_;
    jit_args_pc_.peStride = peStride_;
    jit_args_pc_.numBlocks = num_blocks;
    jit_args_pc_.firstBlock = first_block;
    jit_args_pc_.input = reinterpret_cast<uint32_t*>(input);
    jit_args_pc_.dstMemObj = regObj;
    jit_args_pc_.flagsMemObj = flagsObj_;
    jit_args_pc_.blockStrideElems = block_stride_u32;
    jit_args_pc_.worldSize = world_size;
    jit_args_pc_.dstBaseOffset = dst_block_offset_bytes;
    jit_args_pc_.flagVal = flag_token;
    jit_args_pc_.splitSizes = reinterpret_cast<const size_t*>(split_sizes_ptr);
    jit_args_pc_.splitOffsets = reinterpret_cast<const size_t*>(split_offsets_ptr);
    jit_args_pc_.splitCount = split_count;
    return reinterpret_cast<int64_t>(&jit_args_pc_);
  }

  // completion fence for the DIRECT path. The gathers already
  // PUSHED into the user output (data is in place when the kernels return), so
  // there is NO copy-OUT -- only the cross-PE on-stream fence so no peer reuses
  // its output / the flags region before all peers have finished pushing. Pairs
  // with prepare_sync_direct + the stream-ordered inter ring. ``barrier=false``
  // defers the fence to the next op's inter-prepare barrier (same rationale as
  // finish_batch_stream's deferral).
  double finish_direct_stream(hipStream_t stream, bool barrier = true) {
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierOnStream(stream);
    return 0.0;
  }

  // Copy the full groupSize*chunk node-block out to the user buffer (group
  // order) and synchronize, then (optionally) barrier so no peer reuses the
  // buffer early.
  //
  // M4: ``barrier`` lets a caller SKIP the trailing ShmemBarrierAll.
  // The PUSH gather already guarantees this PE's ``out_`` is complete when the
  // kernel returns: every member spins in-kernel (oneshot_sdma_kernel.hpp:196)
  // until all peers have pushed their shard AND quieted, so after the stream
  // sync the node-block is fully populated WITHOUT a host barrier. Flags are
  // monotonic (per-call token, no reset), so there is no cross-call flag hazard
  // requiring the barrier either. The barrier is therefore redundant when this
  // gather is immediately followed by ANOTHER global ShmemBarrierAll that
  // synchronizes all PEs before the next phase reads remote state -- exactly the
  // case in the hierarchical pipeline (the inter-node ring's prepare_sync
  // barrier follows). Default ``barrier=true`` keeps the standalone contract
  // (e.g. test_intra_subgroup_sdma) byte-for-byte unchanged.
  double finish_sync(uintptr_t output, size_t count_u32, hipStream_t stream, bool barrier = true) {
    size_t total = static_cast<size_t>(groupSize_) * count_u32 * sizeof(uint32_t);
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierAll();
    return 0.0;
  }

  int npes() const { return npes_; }
};

}  // namespace collective
}  // namespace mori

#endif  // INTRA_NODE_SUBGROUP_SDMA_CLASS_HPP

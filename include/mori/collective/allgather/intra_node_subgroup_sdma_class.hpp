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

// Sub-group intra-node SDMA AllGather host handle: the intra-node phase of the
// hierarchical cross-node AllGather. The G local ranks of a node SDMA-gather
// their G shards over XGMI into a symmetric transit of groupSize contiguous
// shard slots. SDMA analogue of InterNodeRingAllgather; driven prepare ->
// launch kernel -> finish, as AllgatherSdma.

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

// EVENT-SCOPED finish_sync copy-OUT (MORI_INTRA_EVENT_SYNC, default on): scopes
// the host stall to the SDMA copy (private stream + copy-done event) instead of
// draining the whole caller stream. =0 restores the full-stream drain.
inline bool IntraEventSync() {
  static const bool on = []() {
    const char* e = std::getenv("MORI_INTRA_EVENT_SYNC");
    if (e == nullptr) return true;
    return std::atoi(e) != 0;
  }();
  return on;
}

class IntraNodeSubGroupAllgatherSdma {
 private:
  int myPe_;
  int npes_;

  // Arithmetic sub-group over global PEs {peBase_ + k*peStride_ : k in
  // [0,groupSize_)}; this PE is at groupPos_. Flat whole-world gather is the
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

  // Per-position arrival flags (one uint64 per group position; npes_ allocated).
  // Monotonic generation token avoids a per-call reset.
  void* flags_;
  application::SymmMemObjPtr flagsObj_;
  uint64_t seq_;

  CclAllgatherSubGroupArgs<uint32_t> jit_args_;
  CclAllgatherSubGroupParamContiguousArgs<uint32_t> jit_args_pc_;

  // MORI_INTRA_EVENT_SYNC scratch (lazily created, null in the default path):
  // copy_stream_ carries the copy-OUT, gather_ev_ orders it after the gather,
  // copy_ev_ is what the host waits on.
  hipStream_t copy_stream_ = nullptr;
  hipEvent_t gather_ev_ = nullptr;
  hipEvent_t copy_ev_ = nullptr;

  void ensure_event_sync_scratch() {
    if (copy_stream_ == nullptr) {
      (void)hipStreamCreateWithFlags(&copy_stream_, hipStreamNonBlocking);
      (void)hipEventCreateWithFlags(&gather_ev_, hipEventDisableTiming);
      (void)hipEventCreateWithFlags(&copy_ev_, hipEventDisableTiming);
    }
  }

  // DIRECT-TO-OUTPUT registration: user output base -> symmetric mem object.
  // Direct gathers PUSH straight into it (no transit copy-OUT). COLLECTIVE
  // (ShmemSymmetricRegister), cached on unseen ptr; SPMD lockstep keeps it symmetric.
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

  // EXACT-base lookup for the direct path (byteOffset always 0). Range-containment
  // (find_registered) is unsafe under torch's caching allocator: a fresh output
  // can be carved INSIDE a since-freed registered segment, so a range hit returns
  // a STALE SymmMemObj (peer IPC pointers for the old allocation) -> off-by-rank
  // corruption. Exact match + stale eviction guarantees the only live entry.
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
    // out_ is the intra-node node-block: gathered/consumed ONLY via P2P/SDMA
    // (XGMI), never an RDMA src/dst. Register P2P/SDMA-only (rdmaRegister=false)
    // to dodge the ionic single-MR limit at large _max_bytes.
    outObj_ = shmem::ShmemSymmetricRegister(out_, outBytes_, /*rdmaRegister=*/false);
    if (!outObj_.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: out register failed");

    // Sized for up to kMaxReassemblyBlocks concurrent reassembly blocks, each at a
    // DISJOINT flagBase = (j*numNodes + m)*groupSize so gathers never race on flag
    // slots; flagBase=0 (single gather) owns [0, groupSize). Symmetric across PEs.
    constexpr size_t kMaxReassemblyBlocks = 32;
    size_t flagsSlots = static_cast<size_t>(npes_) * (kMaxReassemblyBlocks + 1);
    size_t flagsBytes = flagsSlots * sizeof(uint64_t);
    flags_ = shmem::ShmemMalloc(flagsBytes);
    if (flags_ == nullptr)
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: flags ShmemMalloc failed");
    (void)hipMemset(flags_, 0, flagsBytes);
    flagsObj_ = shmem::ShmemQueryMemObjPtr(flags_);
    if (!flagsObj_.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: flags query failed");
  }

  ~IntraNodeSubGroupAllgatherSdma() {
    // TEARDOWN-ORDERING GUARD: Python GC may destroy this handle AFTER
    // shmem_finalize(); once finalized every ShmemFree asserts -> SIGABRT. Skipping
    // the free is safe (heap already reclaimed; at worst a benign leak on exit).
    if (copy_ev_) (void)hipEventDestroy(copy_ev_);
    if (gather_ev_) (void)hipEventDestroy(gather_ev_);
    if (copy_stream_) (void)hipStreamDestroy(copy_stream_);
    if (!shmem::ShmemIsInitialized()) return;
    if (out_) shmem::ShmemFree(out_);
    if (flags_) shmem::ShmemFree(flags_);
  }

  // Enter the gather (entry barrier), then build kernel args. ``input`` is this
  // PE's shard (device ptr, count_u32 u32 lanes).
  //
  // ``barrier`` guards that every member's out_ transit is free and registered
  // before any peer SDMA-pushes into it; skippable after a prior global
  // ShmemBarrierAll, but MUST stay true on the FIRST call. Default true.
  // ``dst_base_offset_bytes`` offsets this gather's block in out_ (fused sliced
  // path: N disjoint node-blocks -> one bulk finish_batch copy).
  // ``dst_slot_stride_bytes`` decouples per-peer slot stride from copy size (0
  // packs contiguously). Last slot must fit: base + (groupSize-1)*stride + copy.
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

    if (barrier) shmem::ShmemBarrierAll();

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
    // transit path: single gather, base 0.
    jit_args_.flagBase = 0;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // Bulk copy-OUT for the fused sliced path: after N gathers wrote disjoint
  // blocks into out_, copy all total_count_u32 contiguous lanes to the user
  // output in ONE memcpy + sync + barrier, replacing N per-gather finishes.
  double finish_batch(uintptr_t output, size_t total_count_u32, hipStream_t stream,
                      bool barrier = true) {
    if (total_count_u32 * sizeof(uint32_t) > outBytes_) {
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: batch exceeds out capacity");
    }
    size_t total = total_count_u32 * sizeof(uint32_t);
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    if (barrier) shmem::ShmemBarrierAll();
    return 0.0;
  }

  // STREAM-ORDERED finish_batch: bulk copy-OUT with an on-stream rendezvous (no
  // host round-trip). The device barrier still globally fences reuse of out_.
  //
  // DEFERRABLE reuse fence: barrier=false drops the trailing fence, relying on the
  // next op's inter-ring prepare barrier to fence reuse; the LAST op is correct
  // anyway (copy-OUT is stream-ordered). Later finish_* methods re-use this rule.
  double finish_batch_stream(uintptr_t output, size_t total_count_u32, hipStream_t stream,
                             bool barrier = true) {
    if (total_count_u32 * sizeof(uint32_t) > outBytes_) {
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: batch exceeds out capacity");
    }
    size_t total = total_count_u32 * sizeof(uint32_t);
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    if (barrier) shmem::ShmemBarrierOnStream(stream);
    return 0.0;
  }

  // Register a user output for DIRECT-TO-OUTPUT gathers. COLLECTIVE
  // (ShmemSymmetricRegister) so every PE must call in lockstep. Cached: no-op if
  // ``ptr`` is already covered.
  void register_output_buffer(uintptr_t ptr, size_t size) {
    // Cache hit ONLY on an exact base + same extent -- same physical allocation
    // (torch reuses an address => same pages => peer IPC pointers still valid).
    auto exact = registered_outputs_.find(ptr);
    if (exact != registered_outputs_.end() && exact->second.size == size) return;
    // Evict every stale entry overlapping [ptr, ptr+size) (torch reuses/splits a
    // freed segment). Deregistration is collective but the alloc sequence is
    // identical across PEs (SPMD), so the eviction set stays symmetric.
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

  // DIRECT gather: SDMA-PUSH each member's slice straight into the registered user
  // output (no transit, no copy-OUT). ``output_ptr`` must be inside a registered
  // buffer; block lands at (output_ptr - regBase) + dst_block_offset_bytes.
  int64_t prepare_sync_direct(uintptr_t input, size_t count_u32, hipStream_t stream, bool barrier,
                              uintptr_t output_ptr, size_t dst_block_offset_bytes = 0,
                              size_t dst_slot_stride_bytes = 0, size_t flag_slot_base = 0) {
    auto regObj = find_exact(output_ptr);
    if (!regObj.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: output not registered for direct");
    size_t copy_bytes = count_u32 * sizeof(uint32_t);
    size_t slot_stride = dst_slot_stride_bytes != 0 ? dst_slot_stride_bytes : copy_bytes;
    size_t base_off = dst_block_offset_bytes;
    size_t last_slot_end =
        base_off + static_cast<size_t>(groupSize_ - 1) * slot_stride + copy_bytes;
    if (last_slot_end > regObj->size) {
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: direct gather exceeds output");
    }
    // Race-free concurrent direct gathers: lane j passes a disjoint
    // flag_slot_base = j*groupSize. Guard against OOB on the flags buffer.
    constexpr size_t kMaxReassemblyBlocks = 32;
    size_t flags_slot_cap = static_cast<size_t>(npes_) * (kMaxReassemblyBlocks + 1);
    if (flag_slot_base + static_cast<size_t>(groupSize_) > flags_slot_cap) {
      throw std::runtime_error(
          "IntraNodeSubGroupAllgatherSdma: direct flag_slot_base exceeds flag capacity");
    }
    uint64_t flag_token = ++seq_;
    // Entry fence is STREAM-ORDERED (ShmemBarrierOnStream) to keep the direct path
    // fully on-stream; skipped (barrier=false) on the default slice_fuse_ib path.
    if (barrier) shmem::ShmemBarrierOnStream(stream);
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
    jit_args_.flagBase = flag_slot_base;
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // FUSED param-contiguous direct gather: one kernel launch scatters ALL node
  // blocks * param splits into the registered output (replaces the per-(block,
  // param) loop). ``split_*_ptr`` are DEVICE size_t arrays (u32 lanes); world==npes.
  int64_t prepare_sync_direct_param_contiguous(uintptr_t input, hipStream_t stream, bool barrier,
                                               uintptr_t output_ptr, size_t block_stride_u32,
                                               int num_blocks, size_t world_size,
                                               uintptr_t split_sizes_ptr,
                                               uintptr_t split_offsets_ptr, size_t split_count,
                                               size_t dst_block_offset_bytes = 0,
                                               int first_block = 0) {
    auto regObj = find_exact(output_ptr);
    if (!regObj.IsValid())
      throw std::runtime_error(
          "IntraNodeSubGroupAllgatherSdma: output not registered for direct param-contiguous");
    uint64_t flag_token = ++seq_;
    if (barrier) shmem::ShmemBarrierOnStream(stream);
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

  // Completion fence for the DIRECT path: gathers already PUSHED into the user
  // output, so no copy-OUT -- only the cross-PE on-stream fence so no peer reuses
  // its output/flags before all peers finish pushing. barrier=false = deferrable.
  double finish_direct_stream(hipStream_t stream, bool barrier = true) {
    if (barrier) shmem::ShmemBarrierOnStream(stream);
    return 0.0;
  }

  // Copy the full groupSize*chunk node-block to the user buffer (group order),
  // sync, then optionally barrier so no peer reuses the buffer early.
  //
  // ``barrier`` skippable: the PUSH gather spins in-kernel until all peers pushed
  // AND quieted, so out_ is complete on kernel return -- correct without it, and
  // deferrable to the next inter-ring prepare_sync barrier. Default true.
  double finish_sync(uintptr_t output, size_t count_u32, hipStream_t stream, bool barrier = true) {
    size_t total = static_cast<size_t>(groupSize_) * count_u32 * sizeof(uint32_t);
    if (IntraEventSync()) {
      // EVENT-SCOPED path: copy-OUT on a private stream ordered after the gather
      // (gather_ev_); host waits only on the copy-done event, scoping the stall to
      // gather+copy, not the whole caller stream.
      ensure_event_sync_scratch();
      (void)hipEventRecord(gather_ev_, stream);
      (void)hipStreamWaitEvent(copy_stream_, gather_ev_, 0);
      (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                           copy_stream_);
      (void)hipEventRecord(copy_ev_, copy_stream_);
      (void)hipStreamWaitEvent(stream, copy_ev_, 0);
      // Host-wait only when a host ShmemBarrierAll follows (barrier=true): the
      // copy must land before the cross-PE rendezvous. Hot path skips it.
      if (barrier) {
        (void)hipEventSynchronize(copy_ev_);
      }
      if (barrier) shmem::ShmemBarrierAll();
      return 0.0;
    }
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    if (barrier) shmem::ShmemBarrierAll();
    return 0.0;
  }

  // Expose out_ so a caller can do the Phase-B copy-OUT with a COMPUTE-UNIT
  // kernel (torch elementwise) instead of the copy engine. The SDMA receiver's
  // __threadfence_system() after acquiring each peer's flag makes the gathered
  // bytes coherent to a subsequent CU read but NOT to the separate copy engine
  // (whose out_ read is unfenced against the raw-SDMA writes, absent a host
  // stream sync). A CU copy-OUT reads out_ in the coherent CU domain, closing the
  // gap without a host stall.
  uintptr_t out_ptr() const { return reinterpret_cast<uintptr_t>(out_); }
  size_t out_bytes() const { return outBytes_; }
};

}  // namespace collective
}  // namespace mori

#endif  // INTRA_NODE_SUBGROUP_SDMA_CLASS_HPP

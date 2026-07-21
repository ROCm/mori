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

// Opt-in (MORI_HIER_NO_ALL_BARRIER; off by default; mirror of
// inter_node_ring_class HierAllBarrierDisabled): skip the intra-node subgroup
// gather's cross-PE barriers so both phases (inter ring + intra gather) run
// barrier-free. NOT correctness-safe; benchmarking lever only.
inline bool IntraHierAllBarrierDisabled() {
  static const bool disabled = []() {
    const char* e = std::getenv("MORI_HIER_NO_ALL_BARRIER");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return disabled;
}

// Opt-in (MORI_INTRA_MQ; off by default): drive all sdmaNumQueue SDMA queues
// (both recommended XGMI engines per peer link) for each intra all-to-all column
// instead of only queue 0. Default off is byte-identical. See
// oneshot_sdma_kernel.hpp body.
inline int IntraMultiQueue() {
  static const int mq = []() {
    const char* e = std::getenv("MORI_INTRA_MQ");
    return (e != nullptr) ? std::atoi(e) : 0;
  }();
  return mq;
}

// EVENT-SCOPED finish_sync copy-OUT (MORI_INTRA_EVENT_SYNC). The default
// finish_sync host-blocks on hipStreamSynchronize(stream) -- but ``stream`` is
// the CALLER's compute stream (host_proxy_ag runs the whole AG inside
// ``with torch.cuda.stream(stream)``), so the drain waits for the ENTIRE stream
// to reach the copy-OUT, not just the SDMA copy. When set, the copy-OUT is
// issued on a PRIVATE stream that only waits (via event) on the gather kernel,
// then the host waits on a copy-done event -- scoping the host stall to
// gather+copy instead of the full caller stream. Default on;
// MORI_INTRA_EVENT_SYNC=0 restores the path where the copy-OUT stays on
// ``stream`` and the host does a full hipStreamSynchronize. See finish_sync.
inline bool IntraEventSync() {
  static const bool on = []() {
    const char* e = std::getenv("MORI_INTRA_EVENT_SYNC");
    // Default on: event-scoped copy-OUT. Set MORI_INTRA_EVENT_SYNC=0 to force
    // the full-stream-drain path.
    if (e == nullptr) return true;
    return std::atoi(e) != 0;
  }();
  return on;
}

// EVENT-SCOPED finish_sync WITHOUT the host event wait (MORI_INTRA_EVENT_NOSYNC).
// The EVENT_SYNC path already GPU-orders the caller stream after the copy-OUT via
// hipStreamWaitEvent(stream, copy_ev_): every downstream consumer AND the deferred
// node-local dist.barrier(intra_group) (enqueued on the caller stream) are gated on
// the landed copy-OUT on the DEVICE. On the hot path (barrier=false) no host-side
// ShmemBarrierAll follows finish_sync, so the trailing host hipEventSynchronize
// blocks the LAUNCH thread for a completion that nothing on the host actually needs
// -- it only prevents the host from racing ahead to post the next AG. Skipping it
// (when barrier=false only) lets the host launch the next op's RDMA + gather while
// this op's copy-OUT is still draining on the copy engine, without weakening any
// correctness fence (device ordering is intact). Default on; layers on top of
// EVENT_SYNC. The standalone UT (barrier=true) always keeps the host wait. Set
// MORI_INTRA_EVENT_NOSYNC=0 to restore the per-AG host event wait on the hot path.
inline bool IntraEventNoSync() {
  static const bool on = []() {
    const char* e = std::getenv("MORI_INTRA_EVENT_NOSYNC");
    if (e == nullptr) return true;
    return std::atoi(e) != 0;
  }();
  return on;
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

  // MORI_INTRA_EVENT_SYNC scratch (lazily created on first use, never in the
  // default path). ``copy_stream_`` carries only the copy-OUT; ``gather_ev_``
  // orders it after the gather on the caller stream; ``copy_ev_`` is what the
  // host waits on -- scoping the stall to gather+copy, not the whole caller
  // stream. All null unless the env lever is on.
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
  // off-by-rank corruption. Exact match + stale eviction
  // (register_output_buffer) guarantees the only live entry for a base is the
  // current registration.
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
    // The out_ transit is the intra-node node-block: gathered and consumed ONLY
    // via P2P/SDMA (XGMI) — never an RDMA src/dst (the inter ring uses its own
    // RDMA-registered buffer). Register P2P/SDMA-only (rdmaRegister=false) so the
    // node-block buffer dodges the ionic single-MR limit at large _max_bytes.
    outObj_ = shmem::ShmemSymmetricRegister(out_, outBytes_, /*rdmaRegister=*/false);
    if (!outObj_.IsValid())
      throw std::runtime_error("IntraNodeSubGroupAllgatherSdma: out register failed");

    // Parallel remote reassembly: the fused ring+remote-gather kernel
    // (FusedRingRemoteGatherKernel) runs ``reasm`` concurrent reassembly blocks,
    // each of which launches a blockLocal subgroup gather at a DISJOINT flagBase
    // = (j*numNodes + m)*groupSize so the concurrent gathers never race on the
    // shared flag slots. The max flag slot used is reasm*numNodes*groupSize ==
    // reasm*npes. The single-block (reasm=1) path only needs ``npes`` slots, but
    // sizing the buffer for up to ``kMaxReassemblyBlocks`` reassembly blocks lets
    // MORI_HIER_REASSEM_BLOCKS>1 parallelise the XGMI reassembly without an OOB
    // flag write. Cost is a few KiB of the symmetric heap, identical across PEs
    // (npes is collective), so the
    // symmetric layout stays consistent. flagBase=0 (the classic single gather
    // and the fused local block) still occupies [0, groupSize) unchanged.
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
    // TEARDOWN-ORDERING GUARD. This handle is owned (via HostProxyHierAllGather /
    // MoriAllGather) by Python, whose GC destroys it at interpreter shutdown --
    // which, in the FSDP bench, runs AFTER bench.py's shmem_finalize(). Once the
    // runtime is finalized every ShmemFree hits CheckStatusValid()'s assert(false)
    // -> SIGABRT on all ranks (observed only on the SDMA_INTRA path, the only one
    // that constructs this handle; the non-SDMA host-proxy path and device-ibgda
    // never build it). If shmem is already gone the symmetric heap it owned has
    // been reclaimed by finalize, so skipping the free is correct (at worst a
    // benign leak in a process that is exiting anyway). This is TEARDOWN-ONLY:
    // during operation shmem is always initialized, so the free path is
    // unchanged -- no effect on any live gather or on device-ibgda.
    if (copy_ev_) (void)hipEventDestroy(copy_ev_);
    if (gather_ev_) (void)hipEventDestroy(gather_ev_);
    if (copy_stream_) (void)hipStreamDestroy(copy_stream_);
    if (!shmem::ShmemIsInitialized()) return;
    if (out_) shmem::ShmemFree(out_);
    if (flags_) shmem::ShmemFree(flags_);
  }

  // Barrier so all members are primed (flags already monotonic), then build the
  // kernel args. ``input`` is this PE's shard (device ptr, count_u32 u32 lanes).
  //
  // ``barrier`` lets a caller skip this entry ShmemBarrierAll.
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
  // ``dst_base_offset_bytes`` places this gather's groupSize-slot
  // block at a non-zero byte offset inside ``out_``. Used by the fused sliced
  // path: each of the N reassembly gathers writes its node-block into a DISJOINT
  // region [m*block_bytes, (m+1)*block_bytes) of an enlarged transit, so they
  // never overlap and the per-gather finish barrier + per-gather copy-OUT can be
  // dropped (replaced by ONE bulk copy in ``finish_batch``). Default 0 keeps the
  // single-block contract byte-for-byte unchanged.
  // ``dst_slot_stride_bytes`` decouples the per-peer destination
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
    // classic prepare_sync (transit path): always base 0 (single gather).
    jit_args_.flagBase = 0;
    jit_args_.multiQueue = IntraMultiQueue();
    return reinterpret_cast<int64_t>(&jit_args_);
  }

  // Bulk copy-OUT for the fused sliced path. After N gathers have
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
  // ShmemBarrierAll provided, just stream-ordered. Pairs with the stream-ordered
  // inter ring. (Same lever family as InterNodeRing::finish_stream.)
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
    // Race-free concurrent direct gathers. Lane j passes a disjoint
    // flag_slot_base = j*groupSize so simultaneous launches (REASM_STREAMS)
    // never collide on the shared flag slots. The flags buffer is sized for
    // npes*(kMaxReassemblyBlocks+1) slots (see ctor); guard against OOB.
    constexpr size_t kMaxReassemblyBlocks = 32;
    size_t flags_slot_cap = static_cast<size_t>(npes_) * (kMaxReassemblyBlocks + 1);
    if (flag_slot_base + static_cast<size_t>(groupSize_) > flags_slot_cap) {
      throw std::runtime_error(
          "IntraNodeSubGroupAllgatherSdma: direct flag_slot_base exceeds flag capacity");
    }
    uint64_t flag_token = ++seq_;
    // keep the entry fence STREAM-ORDERED (no host CPU<->GPU
    // round-trip) to match the rest of the direct path, which is fully
    // on-stream (stream_ring + stream_intra). ShmemBarrierOnStream globally
    // fences all PEs (same guarantee as the host ShmemBarrierAll) but enqueued
    // on ``stream`` so it never stalls the launch thread. In the default config
    // (slice_fuse_ib on) this barrier is skipped entirely (barrier=false); it
    // only fires in the fuse_ib-off path, where it must stay on-stream to avoid
    // mixing a host barrier into the stream-ordered sequence.
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
    jit_args_.flagBase = flag_slot_base;
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
                                               uintptr_t split_sizes_ptr,
                                               uintptr_t split_offsets_ptr, size_t split_count,
                                               size_t dst_block_offset_bytes = 0,
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
  // ``barrier`` lets a caller skip the trailing ShmemBarrierAll.
  // The PUSH gather already guarantees this PE's ``out_`` is complete when the
  // kernel returns: every member spins in-kernel until all peers have pushed
  // their shard AND quieted, so after the stream
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
    if (IntraEventSync()) {
      // EVENT-SCOPED path: run the copy-OUT on a private stream ordered AFTER
      // the gather (via gather_ev_ on the caller stream), then host-wait ONLY on
      // the copy-done event. The gather kernel already spun in-kernel until all
      // peers pushed (out_ is complete on kernel return), so ordering the copy
      // after the gather is sufficient; the host stall then covers gather+copy
      // instead of every op queued on the caller's compute stream.
      ensure_event_sync_scratch();
      (void)hipEventRecord(gather_ev_, stream);
      (void)hipStreamWaitEvent(copy_stream_, gather_ev_, 0);
      (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                           copy_stream_);
      (void)hipEventRecord(copy_ev_, copy_stream_);
      // Make the caller stream observe the copy-OUT (downstream consumers gate on
      // it) without a host drain of the caller stream.
      (void)hipStreamWaitEvent(stream, copy_ev_, 0);
      // Host-wait on the copy-OUT ONLY when a host-side ShmemBarrierAll follows
      // (barrier=true, the standalone contract) -- there the host must know the
      // copy landed before entering the cross-PE rendezvous. On the hot path
      // (barrier=false) nothing on the host consumes the copy: the caller stream
      // is already device-ordered after it (hipStreamWaitEvent above), so the
      // MORI_INTRA_EVENT_NOSYNC lever lets the launch thread skip the host wait
      // and race ahead to post the next AG while this copy drains on the copy
      // engine.
      if (barrier || !IntraEventNoSync()) {
        (void)hipEventSynchronize(copy_ev_);
      }
      if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierAll();
      return 0.0;
    }
    (void)hipMemcpyAsync(reinterpret_cast<void*>(output), out_, total, hipMemcpyDeviceToDevice,
                         stream);
    (void)hipStreamSynchronize(stream);
    if (barrier && !IntraHierAllBarrierDisabled()) shmem::ShmemBarrierAll();
    return 0.0;
  }

  int npes() const { return npes_; }

  // Expose the internal transit ``out_`` so a caller can perform the Phase-B
  // copy-OUT with a COMPUTE-UNIT kernel (torch elementwise) instead of the
  // copy-engine hipMemcpyAsync. The SDMA gather's receiver does a
  // __threadfence_system() after acquiring each peer's completion flag, which
  // makes the gathered bytes coherently visible to a subsequent CU read -- but
  // NOT to the separate copy engine, whose read of ``out_`` is not fenced
  // against the raw-SDMA writes (only a host stream.synchronize otherwise drains
  // it). A CU copy-OUT reads ``out_`` in
  // the fenced/coherent CU domain and writes the user output in the CU/L2 domain
  // the consumer GEMM reads, closing the gap WITHOUT a host stall.
  uintptr_t out_ptr() const { return reinterpret_cast<uintptr_t>(out_); }
  size_t out_bytes() const { return outBytes_; }
};

}  // namespace collective
}  // namespace mori

#endif  // INTRA_NODE_SUBGROUP_SDMA_CLASS_HPP

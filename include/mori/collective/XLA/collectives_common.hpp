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
#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__HIPCC__) || defined(__HIP__)
#include "mori/core/transport/sdma/anvil_device.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"
#include "mori/shmem/internal.hpp"  // GetGlobalGpuStatesPtr / heapObj for StartSdmaScatter
#endif  // __HIPCC__ || __HIP__

#include "mori/core/transport/sdma/sdma_pkt_struct.h"

// ---------------------------------------------------------------------------
// Shared building blocks for collective kernels: cache-bypassing vector
// load/store (StreamLoad/StreamStore), a 128-bit SDMA ring-store helper, and the
// warp-cooperative fused copy+atomic SDMA put used by the push collectives.
// ---------------------------------------------------------------------------
namespace mori {
namespace collective {

// A copy-linear packet immediately followed by its completion atomic, then one
// trailing zero dword. Both SDMA packet structs are made entirely of 4-byte
// fields, so copy (dwords 0..6) and atomic (7..14) land in the ring as two
// adjacent packets; the `nop` dword (15, value 0) is a single-dword SDMA NOP the
// engine skips. The padding rounds the packet to 64 bytes and `alignas(16)` makes
// each ring slot 16-byte aligned, so the body can be written with b128 stores.
struct alignas(16) SDMA_PKT_COPY_WITH_ATOMIC {
  SDMA_PKT_COPY_LINEAR copy;  // 28B
  SDMA_PKT_ATOMIC atomic;     // 32B
  uint32_t nop;               // 4B trailing single-dword NOP (must be 0)
};
static_assert(sizeof(SDMA_PKT_COPY_WITH_ATOMIC) == 64,
              "fused copy+atomic packet must be 64B (16B-aligned) for b128 stores");

static constexpr int kRSPushMaxPeers = 16;
static constexpr int kRSPushMaxSlices = 8;
static constexpr int kRSPushPktDwords = sizeof(SDMA_PKT_COPY_WITH_ATOMIC)/4;
static constexpr int kRSPushSlotDwords = kRSPushPktDwords + 1; // padded stride

// Per-peer all-to-all endpoints: chunk sent from `source` to peer p / received
// into `dest` from peer p. Host-fillable, device-readable (host-pinned buffer).
struct AddressPair {
  const void* source;
  void* dest;
};

#if defined(__HIPCC__) || defined(__HIP__)

#define USE_NONTEMPORAL_LOAD 0
#define GLOBAL_SPACE __attribute__((address_space(1)))
#define BREAK_ON_RETRIES 1

// Streaming (cache-bypassing) 16-byte load/store.
#if (defined(__gfx942__) || defined(__gfx950__)) &&     \
    __has_builtin(__builtin_amdgcn_global_load_b128) && \
    __has_builtin(__builtin_amdgcn_global_store_b128)
#elif defined(__HIP_DEVICE_COMPILE__)
#error "Global b128 load/store not supported on this architecture"
#endif

constexpr uint32_t VecBytes = 16;

using V128 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using V128_GLOBAL = GLOBAL_SPACE V128*;
template <int TVecBytes>
using TVecType = std::conditional_t<
    TVecBytes == 1, uint8_t,
        std::conditional_t<TVecBytes == 2, uint16_t,
        std::conditional_t<TVecBytes == 4, uint32_t,
        std::conditional_t<TVecBytes == 8, uint64_t,
        std::conditional_t<TVecBytes == 16, V128, void>>>>>;

template <typename T>
__device__ __host__ inline static T* Tglobal(T* ptr) {
  return (T*)(GLOBAL_SPACE T*)reinterpret_cast<uintptr_t>(ptr);
}

template <typename T>
__device__ __host__ inline static GLOBAL_SPACE T* MemSpace(T* ptr) {
  uintptr_t u = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<GLOBAL_SPACE T*>(u);
}

enum StreamScope {
  ESystemScope = 0,
  EAgentScope = 1,
};

template <StreamScope Scope, int Bytes = VecBytes>
__device__ __forceinline__ TVecType<Bytes> StreamLoad(const void* p) {
  static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8 || Bytes == 16,
                "StreamLoad supports 1/2/4/8/16 byte accesses");
  auto ptr = reinterpret_cast<const TVecType<Bytes>*>(p); 
#if USE_NONTEMPORAL_LOAD
  return __builtin_nontemporal_load(MemSpace(ptr));
#else
  if constexpr (Bytes == 16) {
    if constexpr (Scope == ESystemScope) {
      return __builtin_amdgcn_global_load_b128((V128_GLOBAL)p, "");
    } else {
      return __builtin_amdgcn_global_load_b128((V128_GLOBAL)p, "agent");
    }
  } else {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED,
              Scope == ESystemScope ? __HIP_MEMORY_SCOPE_SYSTEM 
                                    : __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

template <StreamScope Scope, int Bytes = VecBytes>
__device__ __forceinline__ void StreamStore(void* p, TVecType<Bytes> v) {
  static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8 || Bytes == 16,
                "StreamStore supports 1/2/4/8/16 byte accesses");
  auto ptr = reinterpret_cast<TVecType<Bytes>*>(p);
#if USE_NONTEMPORAL_LOAD
  __builtin_nontemporal_store(v, MemSpace(ptr));
#else
  if constexpr (Bytes == 16) {
    if constexpr (Scope == ESystemScope) {
      __builtin_amdgcn_global_store_b128((V128_GLOBAL)p, v, "");
    } else {
      __builtin_amdgcn_global_store_b128((V128_GLOBAL)p, v, "agent");
    }
  } else {
    __hip_atomic_store(ptr, v, __ATOMIC_RELAXED,
                       Scope == ESystemScope ? __HIP_MEMORY_SCOPE_SYSTEM 
                                             : __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

#define FORCE_SGPR(x) __builtin_amdgcn_readfirstlane(x)

// ---------------------------------------------------------------------------
// Range-checked raw-buffer 128-bit load/store.
//
// A buffer resource (V#) is built by hand as a uniform int32x4: word0..1 = 64-bit
// base, word2 = num_records (valid byte extent, stride 0 => bytes), word3 = the
// raw-buffer config. Building the descriptor explicitly (rather than via
// __builtin_amdgcn_make_buffer_rsrc) keeps it a scalar/uniform value in SGPRs and
// avoids the compiler emitting a per-lane "waterfall" around the buffer op.
//
// Out-of-range accesses are hardware range-checked: buffer_load/store_dwordx4
// check PER 32-bit COMPONENT (CDNA4 ISA 9.1.5 note 4) -- OOB reads return 0, OOB
// writes write nothing. That lets a reduce loop cover the final partial vector
// with no scalar tail: the store's per-component check drops any OOB output dword
// regardless of the reduction op.
//
// RS_BUF_AUX is the cache-policy immediate (the intrinsic's last operand),
// separate from word3 and from memory scope; it carries the non-temporal /
// streaming hint. Verified on gfx950: bit0 (0x1) -> `sc0`, bit1 (0x2) -> `nt`.
// Default 0x2 = non-temporal streaming, CU-scope caching (no sc bits): correct
// for the push reduce, whose local-HBM sources are already made visible by the
// caller's system-scope acquire fence before Phase 3.
// ---------------------------------------------------------------------------
#ifndef RS_BUF_AUX
#define RS_BUF_AUX 2
#endif

// word3 config for a plain (non-format) raw buffer on gfx9xx / CDNA. make_buffer_rsrc
// packs its `flags` arg straight into word3 without auto-setting DATA_FORMAT, so on
// gfx9 flags=0 => BUF_DATA_FORMAT_INVALID and buffer_load_dwordx4 returns garbage;
// the descriptor MUST carry this (DATA_FORMAT=32) constant.
#define RS_BUF_RSRC_WORD3 0x00020000
using BufRsrc = int32_t __attribute__((ext_vector_type(4)));

__device__ BufRsrc llvm_amdgcn_raw_buffer_load_v4i32(BufRsrc rsrc, int voffset, int soffset,
                                                     int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ void llvm_amdgcn_raw_buffer_store_v4i32(BufRsrc vdata, BufRsrc rsrc, int voffset,
                                                   int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ __forceinline__ BufRsrc MakeRawRsrc(const void* base, uint32_t numBytes) {
  const uint64_t b = reinterpret_cast<uintptr_t>(base);
  BufRsrc r;
  r.x = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(b & 0xFFFFFFFFu));  // word0: base low 32b
  r.y = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(b >> 32));          // word1: base high (stride 0)
  r.z = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(numBytes));         // word2: num_records (bytes)
  r.w = RS_BUF_RSRC_WORD3;                                                      // word3: raw-buffer config
  return r;
}

__device__ __forceinline__ V128 BufferLoad128(BufRsrc r, uint32_t voff) {
  BufRsrc v = llvm_amdgcn_raw_buffer_load_v4i32(r, static_cast<int>(voff), /*soffset=*/0, RS_BUF_AUX);
  return __builtin_bit_cast(V128, v);
}

__device__ __forceinline__ void BufferStore128(BufRsrc r, V128 v, uint32_t voff) {
  llvm_amdgcn_raw_buffer_store_v4i32(__builtin_bit_cast(BufRsrc, v), r, static_cast<int>(voff),
                                     /*soffset=*/0, RS_BUF_AUX);
}

// SDMA queue handle augmented with collective-specific ring writers. It adds no
// data members (same layout as the base), so a base handle can be used through it
// via a reinterpret_cast -- mirroring anvil::SdmaQueueSingleProducerDeviceHandle.
struct SdmaCollectiveHandle : anvil::SdmaQueueDeviceHandle {

  // Build a fused copy+atomic packet ENTIRELY in registers and stream it to an
  // ABSOLUTE ring index as four b128 stores. 
  // Layout matches SDMA_PKT_COPY_WITH_ATOMIC, assembled from the two shared dword
  // writers: copy = dwords 0..6 (WriteCopyPacket), atomic = dwords 7..14
  // (WriteAtomicInc32Packet), trailing single-dword NOP = 15.
  // Caller must have reserved a contiguous, non-wrapping 64B slot at wptrIndex.
  __device__ __forceinline__ void placeCopyAtomicPacketAt(const void* srcBuf, const void* dstBuf,
                                                          size_t copyBytes, HSAuint64* signal,
                                                          uint64_t wptrIndex) {
    uint32_t dw[16];
    anvil::WriteCopyPacket(dw, srcBuf, dstBuf, copyBytes);  // copy: dw[0..6]
    anvil::WriteAtomicInc32Packet(dw + 7, signal);    // atomic: dw[7..14]
    dw[15] = 0;                                       // trailing single-dword NOP
    const uint64_t base = WrapIntoRing(wptrIndex) / sizeof(uint32_t);
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
      const V128 v = {dw[i], dw[i + 1], dw[i + 2], dw[i + 3]};
      StreamStore<EAgentScope, 16>(queueBuf + base + i, v);
    }
  }

  // Fill [wptrIndex, wptrIndex+numBytes) with zero dwords (single-dword SDMA NOPs)
  // so the engine harmlessly skips the wrap-around padding region.
  __device__ __forceinline__ void fillNops(uint64_t wptrIndex, uint64_t numBytes) {
    uint64_t base_index_in_dwords = WrapIntoRing(wptrIndex) / sizeof(uint32_t);
    const uint64_t numDwords = numBytes / sizeof(uint32_t);
    for (uint64_t i = 0; i < numDwords; i++) {
      StreamStore<EAgentScope, 4>(queueBuf + base_index_in_dwords + i, 0);
    }
  }

    // Sole-producer reservation: no CAS. Valid only when this queue has exactly
  // one producing thread for the lifetime of the reservation (e.g. one leader
  // lane per distinct queue). Keeps the wrap-pad + back-pressure (CanWriteUpto)
  // logic, drops the compare-exchange arbitration.
  __device__ __forceinline__ uint64_t ReserveQueueSpaceCASFree(
      const size_t size_in_bytes, uint64_t& offset) {
    const uint64_t queue_size_in_bytes = anvil::SDMA_QUEUE_SIZE;
    uint64_t cur_index =
        __hip_atomic_load(cachedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    offset = 0;
    if (WrapIntoRing(cur_index) + size_in_bytes > queue_size_in_bytes) {
      offset = (queue_size_in_bytes - WrapIntoRing(cur_index));
    }
    const uint64_t new_index = cur_index + size_in_bytes + offset;
    // Back-pressure only (not CAS): spin until the ring has room. CanWriteUpto
    // refreshes cachedHwReadIndex from the hardware rptr.
    int64_t retries = 0;
    while (!CanWriteUpto(new_index)) {
#if BREAK_ON_RETRIES
      if (retries++ == anvil::MAX_RETRIES) __builtin_trap();
#endif
    }
    __hip_atomic_store(cachedWptr, new_index, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    return cur_index;
  }
};

static_assert(sizeof(SdmaCollectiveHandle) == sizeof(anvil::SdmaQueueDeviceHandle));

// One warp broadcasts a single slice to every peer's output[myPe] slice (same
// byte offset `dstOff` on each peer), trailing an ADD32(1) into peer
// signalPtrs[signalSlot]. Multi-producer-safe: other slice-groups' last blocks
// may hit the same per-peer queue concurrently, so this uses the CAS-based
// ReserveQueueSpace + ordered submitPacket (NOT the single-producer CAS-free
// fast path). Lane p serves peer p; the whole warp must be active.
template <class T> 
inline __device__ void SdmaBroadcastSliceWarp(const void* src, size_t dstOff, size_t sliceBytes,
                                              int myPe, int npes, int signalSlot, int qId) {
  auto* heapObj = shmem::GetGlobalGpuStatesPtr()->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);
  const int peer = threadIdx.x % warpSize;
  if (peer < npes && peer != myPe) {
    auto& h = *static_cast<SdmaCollectiveHandle*>(
        *(heapObj->deviceHandles_d + peer * numSdmaQ + qId));
    constexpr size_t pkt = sizeof(SDMA_PKT_COPY_WITH_ATOMIC);
    uint64_t offset = 0;
    uint64_t base = h.ReserveQueueSpace(pkt, offset);  // CAS: multi-producer safe
    if (offset) h.fillNops(base, offset);
    auto* dst = reinterpret_cast<uint8_t*>(heapObj->peerPtrs[peer] + dstOff);
    h.placeCopyAtomicPacketAt(
        src, dst, sliceBytes,
        reinterpret_cast<HSAuint64*>(heapObj->peerSignalPtrs[peer] + signalSlot), base + offset);
    h.submitPacket(base, base + offset + pkt);
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------
// Fused SDMA "push" scatter shared by the push collectives (reduce-scatter and
// all-gather). Every block calls it but it no-ops on all but block 0, so the
// internal __syncthreads() are still reached by all of block 0's threads.
//
// Each shard is split into S = 1<<logS slices. Lane (peer,slice) issues an SDMA
// copy of slice `slice` from srcOf(peer) into peer's heap at byte offset
// dstOffOf(peer), followed by an ADD64 of 1 into peer's per-slice completion
// counter signalPtrs[slice]. The atomic targets the *receiver's* counter, so
// completion is observed on the receive side -- fire-and-forget, no local quiet,
// no cross-PE barrier. Per-slice counters make slice order irrelevant; the
// receiver just waits until the counter reaches the number of expected senders.
//
// The three peer-indexed quantities are supplied by the caller as device
// callables (all warp-uniform at peer granularity):
//   activeOf(peer) -> bool         : whether this peer's copy is issued
//   srcOf(peer)    -> const uint8_t*: local source base for peer's chunk (pre-slice)
//   dstOffOf(peer) -> size_t        : byte offset of the destination slot within
//                                     the (symmetric) peer heap
// Callers build these per collective, e.g.:
//   reduce-scatter: active=peer!=myPe, src=input+peer*chunkElems, dst slot packs
//                   staging densely (slot = myPe<peer?myPe:myPe-1). Self folded in
//                   by the reduce.
//   all-gather:     active=true, src=input (single shard to everyone incl. self),
//                   dst slot = myPe (constant offset).
//   all-to-all:     active=true, src=srcPtrs[peer], dst offset = dstPtrs[myPe] slot
//                   (constant, symmetric-heap layout).
//
// srcOf(peer)/dstOffOf(peer) MUST reference the symmetric static heap (ShmemMalloc)
// so the address-based SDMA put can translate local->peer (offset from
// heapBaseAddr). The SDMA fast path caps npes <= 8 (one warp issues all peers).
// ---------------------------------------------------------------------------
template <class ActiveFn, class SrcFn, class DstOffFn>
__device__ __forceinline__ void StartSdmaScatter(
    int myPe, int npes, int logS, size_t chunkElems, size_t elemBytes,
    ActiveFn activeOf, SrcFn srcOf, DstOffFn dstOffOf, int signalSlotBase = 0) {

  // Element type is irrelevant to the byte-oriented SDMA copy: the caller passes
  // elemBytes (bytes per element) so the slice split stays VecBytes-aligned.
  const size_t vecSize = VecBytes / elemBytes;
  const int S = 1 << logS;
  // vecSize-aligned slice length; the last slice absorbs the remainder.
  const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize;
  const int tid = threadIdx.x;

  auto* heapObj = shmem::GetGlobalGpuStatesPtr()->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);

  const size_t sliceBytes = sliceLen * elemBytes;
  const size_t lastBytes = (chunkElems - (sliceLen << logS) + sliceLen) * elemBytes;

  // due to LDS capacity: (npes << logS) <= kRSPushMaxPeers * kRSPushMaxSlices 
  constexpr size_t packetSize = sizeof(SDMA_PKT_COPY_WITH_ATOMIC);
  // --- Fast path (LDS-staged): build packets into bank-conflict-free LDS ----
  // slots, then flush them to the rings with up to npes*S*4 threads (one
  // coalesced b128 store per thread). The build write uses one lane per packet
  // at a 17-dword slot stride (coprime to the 32 LDS banks) so the 16-dword
  // packet write is conflict-free. The flush spans multiple warps, so the ring
  // writes need a block-wide fence + barrier before any leader rings a doorbell
  // (the submitter's own s_waitcnt(0) only drains its own wave).

  uint64_t start_base = 0;
  SdmaCollectiveHandle *handle_base = nullptr;
  using QueuePtr = decltype(handle_base->queueBuf);
  __shared__ uint32_t pktBuf[kRSPushMaxPeers * kRSPushMaxSlices * kRSPushSlotDwords];
  __shared__ uint64_t sPktStart[kRSPushMaxPeers];
  __shared__ QueuePtr sQueuePtr[kRSPushMaxPeers];

  // -- Build phase: one lane per packet (peer = tid/S, slice = tid%S). --
  const int npesS = npes << logS, peer = tid >> logS, slice = tid & (S - 1);
  const bool bactive = (tid < npesS) && activeOf(peer);
  if (bactive) {
    if (slice == 0) {
      // deviceHandles_d is addressed by GLOBAL pe (deviceHandles_d + pe * numQ),
      // so distinct peers always map to distinct handle slots (no collision).
      auto** handles = heapObj->deviceHandles_d + peer * numSdmaQ;
      handle_base = static_cast<SdmaCollectiveHandle*>(*(handles + 0));
      // Leader reserves the whole S-packet block and publishes base/pad to LDS.
      // Peers map to distinct queues (per global pe) and only the slice-0 lane
      // reserves, so each queue has exactly one producer here -- use the CAS-free
      // single-producer reserve.
      uint64_t offset = 0;
      start_base = handle_base->ReserveQueueSpaceCASFree(packetSize << logS, offset);
      if (offset) handle_base->fillNops(start_base, offset);
      sPktStart[peer] = start_base + offset;
      sQueuePtr[peer] = handle_base->queueBuf;
    }
    const size_t off = dstOffOf(peer);
    auto* s = srcOf(peer) + slice * sliceBytes;
    auto* d =
          reinterpret_cast<uint8_t*>(heapObj->peerPtrs[peer] + off) + slice * sliceBytes;
    size_t sz = (slice == S - 1) ? lastBytes : sliceBytes;
    // Build the fused copy+atomic packet DIRECTLY into the LDS slot (dword
    // stores via the DW_x unions), skipping the register-resident struct and
    // the reg->LDS copy. Layout: copy = dwords 0..6, atomic = dwords 7..14,
    // trailing single-dword NOP = dword 15. Per-field cross-lane stride stays
    // kRSPushSlotDwords (17, coprime to 32 banks) so the build is conflict-free.
    uint32_t* dw = &pktBuf[tid * kRSPushSlotDwords];  // slot == peer*S + slice == tid
    anvil::WriteCopyPacket(dw, s, d, sz);
    anvil::WriteAtomicInc32Packet(dw + 7,
                                heapObj->peerSignalPtrs[peer] + signalSlotBase + slice);
    dw[15] = 0;  // trailing single-dword SDMA NOP (must be 0)
  }
  __syncthreads();

  // -- Flush phase: one thread per b128 (packet = fb/4, b128 = fb%4), block-
  // strided so it is robust to any blockDim (<= npes*S*4 total b128s). --
  const int totalB128 = npes << (logS + 2);
  for (int fb = tid; fb < totalB128; fb += blockDim.x) {
    const int fpkt = fb / 4, bb = fb % 4,
               fpeer = fpkt >> logS, fslice = fpkt & (S - 1);
    if (!activeOf(fpeer)) continue;  // uniform across the warp (peer-granular)
    const uint64_t idx = sPktStart[fpeer] + fslice * packetSize,
                 baseDword = SdmaCollectiveHandle::WrapIntoRing(idx) / sizeof(uint32_t);
    // Read the b128 from LDS as 4 dwords (stride-17 slot is 4B- but not
    // 16B-aligned, so avoid ds_read_b128), then store it coalesced to the ring.
    const int slot = fpkt * kRSPushSlotDwords + bb * 4;
    TVecType<16> v = {pktBuf[slot + 0], pktBuf[slot + 1], pktBuf[slot + 2],
                        pktBuf[slot + 3]};
    StreamStore<EAgentScope, 16>(sQueuePtr[fpeer] + baseDword + bb * 4, v);
  }
  // All flush warps must finish + their ring stores be visible to the on-die
  // SDMA engine before any doorbell. The ring lives in local HBM and is
  // consumed by this GPU's own SDMA agent, so an agent-scope RELEASE fence is
  // the exact match for the "agent"-scoped b128 ring stores (the doorbell
  // store inside submitPacket carries its own system-scope push). __syncthreads
  // (workgroup scope) alone is one scope short of what the SDMA agent needs.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();

  // -- Submit phase: each peer's leader rings one doorbell on its own queue. --
  if (bactive && slice == 0) {
    handle_base->submitPacket(start_base, sPktStart[peer] + (packetSize << logS));
  }
}

#endif  // __HIPCC__ || __HIP__

}  // namespace collective
}  // namespace mori

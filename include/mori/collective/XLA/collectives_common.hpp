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
#include "mori/cco/cco.hpp"
#endif  // __HIPCC__ || __HIP__

#include "mori/core/transport/sdma/sdma_pkt_struct.h"

// ---------------------------------------------------------------------------
// Shared building blocks for collective kernels: cache-bypassing vector
// load/store (StreamLoad/StreamStore), a 128-bit SDMA ring-store helper, and the
// warp-cooperative fused copy+signal SDMA put used by the push collectives.
// ---------------------------------------------------------------------------
namespace mori {
namespace collective {

// Fused 64B packet: [NOP][COPY_LINEAR][ATOMIC SWAP64].
static constexpr size_t kSDMACopySignalPkt = 64;
static constexpr int kSDMAPollInterval = 4;
static_assert(sizeof(uint32_t) + sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC) ==
                      kSDMACopySignalPkt,
              "Packet size mismatch: copy+atomic must be 64B");
#if defined(__HIPCC__) || defined(__HIP__)
static_assert(cco::CCO_SDMA_QUEUE_SIZE % kSDMACopySignalPkt == 0,
              "SDMA ring must be a multiple of the 64B fused packet");
#endif

static constexpr int kCCLMaxPeers = 16;
static constexpr int kPushMaxSlices = 8;

// Completion slots: a symmetric-heap uint64 arrived[kCCLMaxPeers][kArrivedStride]
// (zeroed once, never reset). Row p of my copy is written only by PE p's SDMA: each
// copy is trailed by an atomic swap storing the sender's generation into its slot, and
// the receiver waits until the slot reaches the generation it expects. Slots only
// grow, so nothing is ever cleared. 64 bits because a slot a launch does not use
// (e.g. slice 3 under logS < 2) falls arbitrarily far behind the generation; a
// 32-bit wrap-safe compare would misread it as reached after 2^31 launches.
//   [0, kPushMaxSlices) : scatter slice s (all push collectives, launch gen)
//   kBcastArrived + b     : all-reduce broadcast b (launch gen)
//   kPermuteArrived       : push collective-permute (pairwise permute count)
enum ArrivedIdx {
  kBcastArrived = kPushMaxSlices,
  kPermuteArrived = kBcastArrived + 2,
  kArrivedStride = 16,
};
static_assert(kPermuteArrived < kArrivedStride, "arrived row too small");
static constexpr size_t kArrivedBytes = kCCLMaxPeers * kArrivedStride * sizeof(uint64_t);

// groupCtr layout (local uint32 buffer): [0] grid-wide last-block arrival
// counter (reduce-scatter's gen bump and the pull kernels' exit barrier elect the
// tail block through it), then all-reduce's two broadcast release counters, then
// all-reduce's per-launch b=1 release target (elects the tail owner).
enum GroupCtrIdx {
  kArrivalIdx = 0,
  kBcastFlagIdx = kArrivalIdx + 1,
  kBcastTargetIdx = kBcastFlagIdx + 2,
  kGroupCtrTotal = kBcastTargetIdx + 1,
};

// Per-peer all-to-all endpoints: chunk sent from `source` to peer p / received// into `dest` from peer p. Host-fillable, device-readable (host-pinned buffer).
struct AddressPair {
  const void* source;
  void* dest;
};

#if defined(__HIPCC__) || defined(__HIP__)

#define USE_NONTEMPORAL_LOAD 0

// Streaming (cache-bypassing) 16-byte load/store.
#if (defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__))
#if __has_builtin(__builtin_amdgcn_global_load_b128) && \
    __has_builtin(__builtin_amdgcn_global_store_b128)
#elif defined(__HIP_DEVICE_COMPILE__)
#error "Global b128 load/store not supported on this architecture"
#endif
#endif

constexpr uint32_t VecBytes = 16;

template <int TVecBytes>
using TVecType = std::conditional_t<
    TVecBytes == 1, uint8_t,
        std::conditional_t<TVecBytes == 2, uint16_t,
        std::conditional_t<TVecBytes == 4, uint32_t,
        std::conditional_t<TVecBytes == 8, uint64_t,
        std::conditional_t<TVecBytes == 16, cco::ccoUint4, void>>>>>;

enum StreamScope {
  ESystemScope = 0,
  EAgentScope = 1,
};

template <StreamScope Scope, class T>
__device__ __forceinline__ T AtomicLoad(const T* p) {
  return __hip_atomic_load(cco::impl::global(p), __ATOMIC_RELAXED,
              Scope == ESystemScope ? __HIP_MEMORY_SCOPE_SYSTEM
                                    : __HIP_MEMORY_SCOPE_AGENT);
}

template <StreamScope Scope, class T>
__device__ __forceinline__ void AtomicStore(T* p, T v) {
  __hip_atomic_store(cco::impl::global(p), v, __ATOMIC_RELAXED,
              Scope == ESystemScope ? __HIP_MEMORY_SCOPE_SYSTEM
                                    : __HIP_MEMORY_SCOPE_AGENT);
}

template <StreamScope Scope, int Bytes = VecBytes>
__device__ __forceinline__ TVecType<Bytes> StreamLoad(const void* p) {
  static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8 || Bytes == 16,
                "StreamLoad supports 1/2/4/8/16 byte accesses");
  // Intrinsic takes a non-const pointer
  auto non_const = const_cast<void*>(p);
  auto ptr = cco::impl::global(static_cast<TVecType<Bytes>*>(non_const)); 
#if USE_NONTEMPORAL_LOAD
  return __builtin_nontemporal_load(ptr);
#else
  using V128 = CCO_GLOBAL_SPACE cco::ccoUint4*;
  if constexpr (Bytes == 16) {
    if constexpr (Scope == ESystemScope) {
      return __builtin_amdgcn_global_load_b128((V128)ptr, "");
    } else {
      return __builtin_amdgcn_global_load_b128((V128)ptr, "agent");
    }
  } else {
    return AtomicLoad<Scope>(ptr);
  }
#endif
}

template <StreamScope Scope, int Bytes = VecBytes>
__device__ __forceinline__ void StreamStore(void* p, TVecType<Bytes> v) {
  static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8 || Bytes == 16,
                "StreamStore supports 1/2/4/8/16 byte accesses");
  auto ptr = cco::impl::global(static_cast<TVecType<Bytes>*>(p));
#if USE_NONTEMPORAL_LOAD
  __builtin_nontemporal_store(v, ptr);
#else
  using V128 = CCO_GLOBAL_SPACE cco::ccoUint4*;
  if constexpr (Bytes == 16) {
    if constexpr (Scope == ESystemScope) {
      __builtin_amdgcn_global_store_b128((V128)ptr, v, "");
    } else {
      __builtin_amdgcn_global_store_b128((V128)ptr, v, "agent");
    }
  } else {
    AtomicStore<Scope>(ptr, v);
  }
#endif
}

#define FORCE_SGPR(x) __builtin_amdgcn_readfirstlane((x))

// Peer `peer`'s copy of a local symmetric-heap pointer: the flat-VA rank delta
// (peer - myPe)*stride4G<<32 used by every push and pull kernel.
template <class T>
__device__ __forceinline__ T* PeerPtr(T* p, int peer, int myPe, uint32_t stride4G) {
  const int32_t stride = (peer - myPe) * static_cast<int32_t>(stride4G);
  const uint64_t ofs = static_cast<uint64_t>(stride) << 32;
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(p) + ofs);
}

template <class A, class B>
__device__ __forceinline__ std::pair<A*, B*> PeerPtrs(A* a, B* b, 
                                      int peer, int myPe, uint32_t stride4G) {
  const int32_t stride = (peer - myPe) * static_cast<int32_t>(stride4G);
  const uint64_t ofs = static_cast<uint64_t>(stride) << 32;
  return std::make_pair(reinterpret_cast<A*>(reinterpret_cast<uintptr_t>(a) + ofs),
                        reinterpret_cast<B*>(reinterpret_cast<uintptr_t>(b) + ofs));
}

// SDMA ATOMIC `operation` takes TC atomic opcodes (SDMA_ATOMIC_ADD64 = 47 is
// TC_OP_ATOMIC_ADD_RTN_64); this is TC_OP_ATOMIC_SWAP_RTN_64. A swap stores the
// generation like a FENCE would, but a FENCE costs the engine ~2-4us more.
// TC_OP_WRITE (0x20) is no substitute: it stores only the low dword.
static constexpr uint32_t kSdmaAtomicSwap64 = 0x27;

// Fused 64B packet: [NOP][COPY_LINEAR src->dst][ATOMIC SWAP64 *slot = value].
// The atomic is ordered after the copy's writes, so a receiver observing
// `value` may read the copied bytes (after an acquire). `slot` is 8B-aligned.
// Lane k writes dwords [4k, 4k+4): lane 0 NOP + copy header/count/param, lane 1
// the copy addresses, lane 2 atomic header/address/data lo, lane 3 data hi,
// then compare data and loop (0).
__device__ __forceinline__ void WriteFusedPacket(int lane, 
     const void* srcBuf, const void* dstBuf, size_t packetSize, const uint64_t* slot,
     uint64_t value, uint32_t* outBasePtr) {
  
  uint32_t dw[4];
  // if (lane == 0) {
  //   decltype(SDMA_PKT_COPY_LINEAR::HEADER_UNION) hdr;
  //   hdr.DW_0_DATA = 0;
  //   hdr.op = SDMA_OP_COPY;
  //   hdr.sub_op = SDMA_SUBOP_COPY_LINEAR;
  //   dw[0] = 0;  // leading single-dword NOP (must be 0)
  //   dw[1] = hdr.DW_0_DATA;
  //   dw[2] = static_cast<uint32_t>(packetSize - 1);  // COUNT_UNION.count (reserved bits 0)
  //   dw[3] = 0;                                      // PARAMETER_UNION (unused)
  // } else if (lane == 2) {
  //   decltype(SDMA_PKT_ATOMIC::HEADER_UNION) hdr;
  //   hdr.DW_0_DATA = 0;
  //   hdr.op = SDMA_OP_ATOMIC;
  //   hdr.operation = SDMA_ATOMIC_ADD64;
  //   dw[0] = hdr.DW_0_DATA;
  //   dw[1] = (uint32_t)((uintptr_t)signal);
  //   dw[2] = (uint32_t)((uintptr_t)signal >> 32);
  //   dw[3] = (uint32_t)addValue;
  // } else if (lane == 3) {
  //   dw[0] = (uint32_t)(addValue >> 32);  // dw[1..3] = 0
  if (lane % 2 == 0) {
  // Header depends only on constants; a scalar-replaceable local keeps the
  // bitfield layout authoritative and constant-folds (no address taken).
    decltype(SDMA_PKT_COPY_LINEAR::HEADER_UNION) cp;
    cp.DW_0_DATA = 0;
    cp.op = SDMA_OP_COPY;
    cp.sub_op = SDMA_SUBOP_COPY_LINEAR;
    decltype(SDMA_PKT_ATOMIC::HEADER_UNION) swp;
    swp.DW_0_DATA = 0;
    swp.op = SDMA_OP_ATOMIC;
    swp.operation = kSdmaAtomicSwap64;

    uint32_t flag = lane >> 1;
    dw[0] = swp.DW_0_DATA & -flag;
    dw[1] = flag == 0 ? cp.DW_0_DATA : (uint32_t)((uintptr_t)slot);
    dw[2] = flag == 0 ? static_cast<uint32_t>(packetSize) - 1 : 
                        (uint32_t)((uintptr_t)slot >> 32);
    dw[3] = static_cast<uint32_t>(value) & -flag;
  } else {
    // lane 1 - copy addresses, lane 3 - data hi, zero compare / loop
    uint32_t mask = lane == 1 ? ~0u : 0;
    dw[0] = lane == 1 ? (uint32_t)(uintptr_t)srcBuf : static_cast<uint32_t>(value >> 32);
    dw[1] = (uint32_t)((uintptr_t)srcBuf >> 32) & mask;
    dw[2] = (uint32_t)(uintptr_t)dstBuf & mask;
    dw[3] = (uint32_t)((uintptr_t)dstBuf >> 32) & mask;
  }
  const cco::ccoUint4 v = {dw[0], dw[1], dw[2], dw[3]};
  StreamStore<EAgentScope, 16>(outBasePtr + lane * 4, v);
} 

// SDMA re-reads the flag every `interval` clocks while parked on a POLL_REGMEM.
static_assert(sizeof(SDMA_PKT_POLL_REGMEM) == 6 * sizeof(uint32_t),
              "POLL_REGMEM must be 6 DW so the 64B slot pads with 10 NOPs");

// One POLL_REGMEM (6 DW) padded with single-DW NOPs (zero dwords) to a full
// 64B ring slot, so every ring reservation stays kSDMACopySignalPkt. The
// engine stalls this queue until *flag == value; retry_count 0xfff = forever.
// Four consecutive lanes each write one b128, like WriteFusedPacket.
__device__ __forceinline__ void WritePollPacket(int lane, const uint32_t* flag, uint32_t value,
                                                uint32_t* outBasePtr) {
  uint32_t dw[4] = {0, 0, 0, 0};
  if (lane == 0) {
    decltype(SDMA_PKT_POLL_REGMEM::HEADER_UNION) hdr;
    hdr.DW_0_DATA = 0;
    hdr.op = SDMA_OP_POLL_REGMEM;
    hdr.mem_poll = 1;
    hdr.func = 3;  // equal
    dw[0] = hdr.DW_0_DATA;
    dw[1] = (uint32_t)((uintptr_t)flag);
    dw[2] = (uint32_t)((uintptr_t)flag >> 32);
    dw[3] = value;
  } else if (lane == 1) {
    decltype(SDMA_PKT_POLL_REGMEM::DW5_UNION) d5;
    d5.DW_5_DATA = 0;
    d5.interval = kSDMAPollInterval;
    d5.retry_count = 0xfff;
    dw[0] = 0xffffffffu;  // mask: compare the whole dword
    dw[1] = d5.DW_5_DATA;
  }
  const cco::ccoUint4 v = {dw[0], dw[1], dw[2], dw[3]};
  StreamStore<EAgentScope, 16>(outBasePtr + lane * 4, v);
}

// Broadcast `v` from lane 4k to lanes 4k..4k+3 (DPP quad_perm [0,0,0,0]: a VALU
// modifier, no LDS round trip). Lane 4k must be active.
template <typename T>
__device__ __forceinline__ T BroadcastQuad(T v) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0, "T must be a multiple of uint32_t");
  union {
    T v;
    int dw[sizeof(T) / sizeof(int)];
  } S = {.v = v};
#pragma unroll
  for (int i = 0; i < sizeof(T) / sizeof(int); i++) {
    S.dw[i] = __builtin_amdgcn_update_dpp(0, S.dw[i], 
            /*quad_perm [0,0,0,0]*/ 0x00, 0xF, 0xF, false);
  }
  return S.v;
}

// Broadcast `v` from lane 0 to the whole wave. Lane 0 must be active.
// ds_bpermute, not v_readlane: the scalar result measured ~1us slower on the
// election paths (LastBlockOut, all-reduce tail).
template <typename T>
__device__ __forceinline__ T BroadcastLane0(T v) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0, "T must be a multiple of uint32_t");
  union {
    T v;
    int dw[sizeof(T) / sizeof(int)];
  } S = {.v = v};
#pragma unroll
  for (int i = 0; i < sizeof(T) / sizeof(int); i++) {
    S.dw[i] = __builtin_amdgcn_ds_bpermute(0, S.dw[i]);
  }
  return S.v;
}

// SDMA queue handle augmented with collective-specific ring writers. It adds no
// data members (same layout as the base), so a base handle can be used through it
// via a reinterpret_cast -- mirroring anvil::SdmaQueueSingleProducerDeviceHandle.
struct SdmaCollectiveHandle : cco::ccoSdmaQueueDeviceHandle {

  static __device__ __forceinline__ uint64_t WrapIntoRing(uint64_t index) {
    return index % cco::CCO_SDMA_QUEUE_SIZE;
  }

  __device__ __forceinline__ bool CanWriteUpto(uint64_t uptoIndex) {
    const uint64_t qSize = cco::CCO_SDMA_QUEUE_SIZE;
    if ((uptoIndex - cachedHwReadIndex) < qSize) {
      return true;
    }
    // Only read hardware register if the queue is full based on cached index
    cachedHwReadIndex = AtomicLoad<ESystemScope>(rptr);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return (uptoIndex - cachedHwReadIndex) < qSize;
  }

  // Sole-producer reservation: no CAS. Valid only when this queue has exactly
  // one producing thread for the lifetime of the reservation (e.g. one leader
  // lane per distinct queue). Claim-then-wait like cco ReserveSlot: fetch_add
  // first, then poll rptr if occupancy exceeds the ring.
  //
  // Reservations are multiples of kSDMACopySignalPkt and the ring is a
  // multiple of that, so a multi-packet reservation may span the wrap but no
  // single 64B packet straddles it: wrap each packet's offset separately.
  //
  // cachedHwReadIndex is by-value. The caller snapshots the handle once and
  // reuses it across packets so a refreshed rptr stays in the local copy;
  // write the hint back to `shared` once at the end if it moved. The hint only
  // moves forward to a value rptr actually had, so a later writer is harmless.
  __device__ __forceinline__ uint64_t ReserveSingleProducer(const size_t size_in_bytes) {
    constexpr uint64_t qSize = cco::CCO_SDMA_QUEUE_SIZE;
    uint64_t base =
        __hip_atomic_fetch_add(cco::impl::global(cachedWptr), size_in_bytes, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t end = base + size_in_bytes;
    if (end - cachedHwReadIndex > qSize) {
      int64_t retries = 0;
      do {
        cachedHwReadIndex = AtomicLoad<ESystemScope>(rptr);
        if constexpr (cco::CCO_SDMA_BREAK_ON_RETRIES) {
          if (retries++ == cco::CCO_SDMA_MAX_RETRIES) __builtin_trap();
        }
      } while (end - cachedHwReadIndex > qSize);
    }
    return base;
  }
};

static_assert(sizeof(SdmaCollectiveHandle) == sizeof(cco::ccoSdmaQueueDeviceHandle));

inline __device__ SdmaCollectiveHandle *GetSharedHandle(cco::ccoSdmaQueueDeviceHandle **table) {
  return reinterpret_cast<SdmaCollectiveHandle *>(*cco::impl::global(table));
}

// ---------------------------------------------------------------------------
// Fused SDMA "push" scatter shared by the push collectives (reduce-scatter and
// all-gather). Four consecutive lanes write one fused 64B copy+signal via
// WriteFusedPacket (one b128 store each). nWork = npes*4; a peer never straddles
// a warp (warpSize % 4 == 0). sub==0 snapshots the queue handle once, then for
// each slice s: reserves 64B, the four lanes write, sub==0 rings that packet.
// Same-lane sequential doorbells; distinct peers are distinct queues, so several
// sub==0 lanes in one warp do not share a commit chain. No LDS / syncthreads.
// Occupancy (rptr) stays; the local cachedHwReadIndex is reused across s and
// written back once if it moved.
//
// Each shard is split into S = 1<<logS slices. Slice s is an SDMA copy from
// p.src + s*sliceBytes to p.dst + s*sliceBytes, followed by an atomic swap
// storing `value` into p.sigBase[s]. sigBase is the peer's copy of MY row of the
// symmetric arrived array (PeerPtr(&arrived[myPe*kArrivedStride + slotBase])),
// so the swap lands in a slot only I write on that peer. Completion is observed
// on the receive side (WaitArrived) -- fire-and-forget, no local quiet, no
// cross-PE barrier.
//
// The caller describes each peer's copy with peerOf(peer) -> ScatterPeer
// (warp-uniform at peer granularity). dst and sigBase are peer addresses,
// usually built with PeerPtr. Examples:
//   reduce-scatter: active=peer!=myPe, src=input+peer*chunkElems, dst slot packs
//                   staging densely (slot = myPe<peer?myPe:myPe-1). Self folded in
//                   by the reduce.
//   all-gather:     active=true, src=input (single shard to everyone incl. self),
//                   dst = output[myPe] (constant).
//   all-to-all:     active=true, src=srcPtrs[peer], dst = dstPtrs[myPe] (constant).
//
// src/dst MUST reference the symmetric static heap (ShmemMalloc) so the
// address-based SDMA put can translate local->peer.
// ---------------------------------------------------------------------------
// elems is the DESTINATION shard's element count, so each peer's copy can be a
// different length: the all-reduce push path uses 16B-aligned variable shards,
// while the uniform callers (all-gather / all-to-all / collective-permute /
// reduce-scatter push) pass a constant. The slice geometry is recomputed per peer
// from that length. Callers must pick logS small enough that every active shard
// yields sliceLen >= vecSize (no empty slices), since the receiver still waits on
// all S slices; the reduce facades clamp logS by the smallest shard.
struct ScatterPeer {
  bool active;         // issue this peer's copy
  size_t elems;        // destination shard length, in ElemBytes units
  const uint8_t* src;  // local source base (pre-slice)
  uint8_t* dst;        // peer's destination base (pre-slice)
  uint64_t* sigBase;   // peer's copy of my arrived row; slice s signals sigBase[s]
};

template <uint64_t ElemBytes = 1, class PeerFn>
__device__ __forceinline__ void StartSdmaScatter(cco::ccoSdmaContext sdma, int npes, int logS,
                                                 uint64_t value, PeerFn peerOf) {

  static_assert(ElemBytes > 0 && VecBytes % ElemBytes == 0,
                "ElemBytes must divide VecBytes");
  constexpr size_t vecSize = VecBytes / ElemBytes;

  const int S = 1 << logS, tid = static_cast<int>(threadIdx.x),
            B = static_cast<int>(blockDim.x);

  const int nWork = npes << 2;
  for (int i = tid; i < nWork; i += B) {
    const int peer = i >> 2, sub = i & 3;
    const ScatterPeer p = peerOf(peer);
    const bool active = p.active;

    // Per-peer slice geometry derived from the destination shard's length.
    const size_t chunkElems = active ? p.elems : 0;
    const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize,
            sliceBytes = sliceLen * ElemBytes,
            lastBytes = (chunkElems - (sliceLen << logS) + sliceLen) * ElemBytes;

    SdmaCollectiveHandle handle;
    SdmaCollectiveHandle* shared = nullptr;
    uint32_t* queueBuf = nullptr;
    uint64_t hint0 = 0;
    if (active && sub == 0) {
      shared = GetSharedHandle(sdma.deviceHandles + peer/* *sdma.sdmaNumQueue*/);
      handle = *cco::impl::global(shared);
      hint0 = handle.cachedHwReadIndex;
      queueBuf = handle.queueBuf;
    }
    queueBuf = BroadcastQuad(queueBuf);

    for (int s = 0; s < S; s++) {
      uint64_t pktBase = 0;
      if (active && sub == 0) {
        pktBase = handle.ReserveSingleProducer(kSDMACopySignalPkt);
      }
      pktBase = BroadcastQuad(pktBase);

      if (active) {
        auto* srcp = p.src + s * sliceBytes;
        auto* dstp = p.dst + s * sliceBytes;
        const size_t sz = (s == S - 1) ? lastBytes : sliceBytes;
        const uint64_t* sig = p.sigBase + s;
        const uint64_t ringDw =
            SdmaCollectiveHandle::WrapIntoRing(pktBase) / sizeof(uint32_t);
        WriteFusedPacket(sub, srcp, dstp, sz, sig, value, queueBuf + ringDw);
      }

      cco::ccoSdmaPublishStores();
      __builtin_amdgcn_wave_barrier();

      if (active && sub == 0) {
        handle.submitPacket(pktBase, pktBase + kSDMACopySignalPkt);
      }
    }

    if (active && sub == 0 && handle.cachedHwReadIndex != hint0) {
      AtomicStore<EAgentScope>(&shared->cachedHwReadIndex, handle.cachedHwReadIndex);
    }
  }
}

// Writes one pre-posted broadcast into the two 64B ring slots starting at pktBase:
// [POLL *flag == value][COPY src->dst + SWAP *slot = slotValue]. The copy reads
// src only once its poll passes, so src need not be ready yet. Each slot wraps on its
// own: the reservation may span the ring end, a 64B packet never does.
__device__ __forceinline__ void WriteGatedBroadcast(int sub, uint32_t* queueBuf, uint64_t pktBase,
                                                    const void* src, const void* dst, size_t bytes,
                                                    const uint64_t* slot, uint64_t slotValue,
                                                    const uint32_t* flag, uint32_t value) {
  WritePollPacket(sub, flag, value,
                  queueBuf + SdmaCollectiveHandle::WrapIntoRing(pktBase) / sizeof(uint32_t));
  WriteFusedPacket(sub, src, dst, bytes, slot, slotValue,
                   queueBuf + SdmaCollectiveHandle::WrapIntoRing(pktBase + kSDMACopySignalPkt) /
                                  sizeof(uint32_t));
}

// Per peer queue 0: one reservation of nSlots*64B, filled by
// writeFn(peer, sub, queueBuf, pktBase), then one publish and one doorbell.
// Uses the same lane-to-peer mapping as StartSdmaScatter, so when called right
// after it by the same block, each queue keeps a single producing lane and these
// packets land behind that peer's scatter packets.
template <class WriteFn>
__device__ __forceinline__ void SdmaPostPerPeer(cco::ccoSdmaContext sdma, int npes,
                                                int myPe, int nSlots, WriteFn writeFn) {
  const int tid = static_cast<int>(threadIdx.x), B = static_cast<int>(blockDim.x);
  const int nWork = npes << 2;

  for (int i = tid; i < nWork; i += B) {
    const int peer = i >> 2, sub = i & 3;
    const bool active = peer != myPe;

    SdmaCollectiveHandle handle;
    SdmaCollectiveHandle* shared = nullptr;
    uint32_t* queueBuf = nullptr;
    uint64_t hint0 = 0;
    if (active && sub == 0) {
      shared = GetSharedHandle(sdma.deviceHandles + peer/* *sdma.sdmaNumQueue*/);
      handle = *cco::impl::global(shared);
      hint0 = handle.cachedHwReadIndex;
      queueBuf = handle.queueBuf;
    }
    queueBuf = BroadcastQuad(queueBuf);

    const uint64_t bytes = nSlots * kSDMACopySignalPkt;
    uint64_t pktBase = 0;
    if (active && sub == 0) {
      pktBase = handle.ReserveSingleProducer(bytes);
    }
    pktBase = BroadcastQuad(pktBase);

    if (active) writeFn(peer, sub, queueBuf, pktBase);

    cco::ccoSdmaPublishStores();
    __builtin_amdgcn_wave_barrier();

    if (active && sub == 0) {
      handle.submitPacket(pktBase, pktBase + bytes);
    }

    if (active && sub == 0 && handle.cachedHwReadIndex != hint0) {
      AtomicStore<EAgentScope>(&shared->cachedHwReadIndex, handle.cachedHwReadIndex);
    }
  }
}

// ---------------------------------------------------------------------------
// Generation barrier shared by every push and pull collective.
//
// The array is a symmetric-heap uint64[npes] (barrierCtr; zeroed once at facade
// creation, never reset). Slot p of every PE's copy is written only by PE p, so
// my own slot doubles as the launch generation: between launches it is always
// even, and gen = slot & ~1. Every launch advances it by exactly 2 whatever the
// collective -- push kernels post gen+2 in one entry round, pull kernels post
// gen+1 at entry and gen+2 at exit -- so as long as every PE issues the same
// sequence of collectives, every PE posts the same value in every round.
//
// Arriving posts the value into slot myPe of every PE's copy (plain one-way
// stores, no remote RMW), then waits on local loads only, until every slot of my
// copy has reached it. Lane p posts to and polls for PE p, so one load + wave
// vote checks every slot. `>=` tolerates a fast peer that already posted a
// later round.
//
// A peer that has arrived has entered this launch, so its previous kernel on its
// stream has retired: it no longer reads the staging / recv slots we are about
// to overwrite. (Completion slots need no such ordering: they only grow, see
// kArrivedStride.)
// ---------------------------------------------------------------------------

// One barrier round on a symmetric uint64[npes] `arrived` array: post `value`
// into slot myPe of every PE's copy, then wait (local loads only) until every
// slot of my copy has reached it. Executed by ONE full wave (npes <= warpSize)
// per calling block. With LeaderOnly, only block 0 posts and every calling
// block just waits, so a whole grid can gate on one arrival per PE.
// `value` must be identical on every PE for a given round and strictly increase
// across rounds on the same array; `>=` then tolerates a fast peer that already
// posted a later round.
template <bool LeaderOnly = false>
__device__ __forceinline__ void GenBarrier(uint64_t* arrived, uint64_t value, int myPe,
                                           int npes, int32_t stride4G) {
  const int lane = __lane_id();
  const bool active = lane < npes;
  if (!LeaderOnly || blockIdx.x == 0) {
    // One system release orders the caller's prior stores ahead of every arrival.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    if (active) {
      auto* dst = PeerPtr(&arrived[myPe], lane, myPe, stride4G);
      AtomicStore<ESystemScope>(dst, value);
    }
  }
  while (active) {
    const uint64_t v = AtomicLoad<ESystemScope>(&arrived[lane]);
    if (v >= value) break;
    __builtin_amdgcn_s_sleep(1);
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

// Waits until completion slot `slot` of every sender row in my local copy of
// `arrived` has reached `gen` (rows q >= npes, and row myPe with skipSelf, count
// as done). Executed by ONE full wave (npes <= warpSize): lane q polls row q,
// one load + wave vote per round. No fence: the caller syncs its block and
// issues the acquire before reading the copied bytes.
__device__ __forceinline__ void WaitArrived(const uint64_t* arrived, int slot, uint64_t gen,
                                            int myPe, int npes, bool skipSelf) {
  const int lane = __lane_id();
  const bool active = lane < npes && !(skipSelf && lane == myPe);
  const uint64_t* addr = arrived + lane * kArrivedStride + slot;
  while (active) {
    const uint64_t v = AtomicLoad<ESystemScope>(addr);
    if (v >= gen) break;
    __builtin_amdgcn_s_sleep(1);
  }
}

// Push-collective launch generation, derived from the gen-barrier flags like the
// pull kernels' gen: a push launch enters with gen+1 (odd, posted by block 0)
// and its tail stores gen+2 into my own slot once no block of
// the launch can still read it. My slot therefore holds gen or gen+1 while the
// launch runs, so every block reads the same value; the launch signals gen+2.
// Every PE runs the same collective sequence, so every PE signals the same value.
__device__ __forceinline__ uint64_t LoadPushGen(const uint64_t* barrierCtr, int myPe) {
  return (AtomicLoad<EAgentScope>(&barrierCtr[myPe]) & ~1ULL) + 2;
}

// Elects the last block of the grid to arrive. Called by the whole first wave
// of every block, after the block's __syncthreads; true in every lane of that
// wave of the last block. groupCtr[kArrivalIdx] wraps back to 0 (atomicInc),
// so it needs no reset.
__device__ __forceinline__ bool LastBlockOut(uint32_t* groupCtr) {
  uint32_t last = 0;
  if (threadIdx.x == 0) {
    last = atomicInc(&groupCtr[kArrivalIdx], gridDim.x - 1) == gridDim.x - 1;
  }
  return BroadcastLane0(last) != 0;
}

// Grid-strided copy of nSeg segments of `bytes` each, segment j going from
// srcOf(j) to dstOf(j). Peer-outer / grid-strided-inner: each segment's base is
// hoisted and every address is base + idx*16 (shift-add, no per-element divide),
// so the NumVecs independent b128 loads issue back-to-back for maximum memory-level
// parallelism across the long XGMI read latency (mirrors PullReduceShard). Loads
// are system scope (sources are peers' buffers); stores use StoreScope.
template <int NumVecs, StreamScope StoreScope = ESystemScope, class SrcFn, class DstFn>
__device__ __forceinline__ void PullCopySegments(int nSeg, size_t bytes, SrcFn srcOf,
                                                 DstFn dstOf) {
  using Vec = TVecType<VecBytes>;
  const size_t n = bytes / VecBytes;  // whole 16B units per segment
  const size_t gstride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int seg = 0; seg < nSeg; ++seg) {
    const uint8_t* __restrict__ src = srcOf(seg);
    uint8_t* __restrict__ dst = dstOf(seg);
    size_t u = gtid;
    for (; u + (NumVecs - 1) * gstride < n; u += NumVecs * gstride) {
      Vec v[NumVecs];
#pragma unroll
      for (int m = 0; m < NumVecs; m++) {
        v[m] = StreamLoad<ESystemScope>(src + (u + m * gstride) * VecBytes);
      }
#pragma unroll
      for (int m = 0; m < NumVecs; m++) {
        StreamStore<StoreScope>(dst + (u + m * gstride) * VecBytes, v[m]);
      }
    }
    for (; u < n; u += gstride) {
      StreamStore<StoreScope>(
          dst + u * VecBytes, StreamLoad<ESystemScope>(src + u * VecBytes));
    }
    // Byte tail for the final < 16B (only non-empty when bytes % 16 != 0).
    for (size_t i = n * VecBytes + gtid; i < bytes; i += gstride) {
      StreamStore<StoreScope, 1>(dst + i, StreamLoad<ESystemScope, 1>(src + i));
    }
  }
}

// Peer-interleaved pull copy. Unlike PullCopySegments (peer-outer: the whole grid
// drains one peer before the next, so only one XGMI read link is busy at a time),
// this issues NumVecs b128 loads spread across all NPES peers at once -- V =
// NumVecs/NPES units per peer -- so NPES read links run concurrently and pull
// bandwidth scales past a single link. NPES is a compile-time constant so the
// peer bases live in SGPRs and the AA[] batch is statically indexed; requires
// NPES to divide NumVecs. srcOf(j)/dstOf(j) return peer j's chunk base (src = a
// peer pointer, dst = a local slot), both indexed by the same unit offset.
// Loads are system scope (sources are peers' buffers); stores use StoreScope.
template <int NPES, int NumVecs, StreamScope StoreScope = ESystemScope,
          class SrcFn, class DstFn>
__device__ __forceinline__ void PullCopyInterleaved(size_t bytes, SrcFn srcOf, DstFn dstOf) {
  static_assert(NumVecs % NPES == 0, "NPES must divide NumVecs");
  constexpr int V = NumVecs / NPES;  // units in flight per peer
  using Vec = TVecType<VecBytes>;
  const size_t n = bytes / VecBytes;  // whole 16B units per peer chunk
  const size_t gstride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // Hoist the per-peer bases once (uniform across the block -> SGPRs).
  const uint8_t* __restrict__ src[NPES];
  uint8_t* __restrict__ dst[NPES];
#pragma unroll
  for (int j = 0; j < NPES; j++) {
    src[j] = srcOf(j);
    dst[j] = dstOf(j);
  }

  // Main batch: each thread holds V units for every peer => NumVecs loads in
  // flight across NPES links before any store. Advance by V*gstride per step.
  const size_t step = static_cast<size_t>(V) * gstride;
  size_t u0 = gtid;
  for (; u0 + (V - 1) * gstride < n; u0 += step) {
    Vec aa[NumVecs];
#pragma unroll
    for (int j = 0; j < NPES; j++) {
#pragma unroll
      for (int i = 0; i < V; i++) {
        aa[j * V + i] =
            StreamLoad<ESystemScope>(src[j] + (u0 + i * gstride) * VecBytes);
      }
    }
#pragma unroll
    for (int j = 0; j < NPES; j++) {
#pragma unroll
      for (int i = 0; i < V; i++) {
        StreamStore<StoreScope>(dst[j] + (u0 + i * gstride) * VecBytes,
                                                aa[j * V + i]);
      }
    }
  }
  // Remainder units (still interleaved across peers to keep links busy).
  for (size_t u = u0; u < n; u += gstride) {
#pragma unroll
    for (int j = 0; j < NPES; j++) {
      StreamStore<StoreScope>(
          dst[j] + u * VecBytes,
          StreamLoad<ESystemScope>(src[j] + u * VecBytes));
    }
  }
  // Byte tail for a final < 16B per peer (empty when bytes % 16 == 0).
  for (size_t i = n * VecBytes + gtid; i < bytes; i += gstride) {
#pragma unroll
    for (int j = 0; j < NPES; j++) {
      StreamStore<StoreScope, 1>(dst[j] + i, 
        StreamLoad<ESystemScope, 1>(src[j] + i));
    }
  }
}

#endif  // __HIPCC__ || __HIP__

}  // namespace collective
}  // namespace mori

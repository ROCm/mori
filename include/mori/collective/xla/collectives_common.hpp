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
// warp-cooperative fused copy+atomic SDMA put used by the push collectives.
// ---------------------------------------------------------------------------
namespace mori {
namespace collective {

static constexpr size_t kSDMACopyAtomicPktSize = 64;
static_assert(sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC) 
                                  + sizeof(uint32_t) == kSDMACopyAtomicPktSize,
              "Packet size mismatch: copy+atomic must be 64B");
#if defined(__HIPCC__) || defined(__HIP__)
static_assert(mori::cco::CCO_SDMA_QUEUE_SIZE % kSDMACopyAtomicPktSize == 0,
              "SDMA ring must be a multiple of the 64B fused packet");
#endif

static constexpr int kRSPushMaxPeers = 16;
static constexpr int kRSPushMaxSlices = 8;

// Per-peer all-to-all endpoints: chunk sent from `source` to peer p / received// into `dest` from peer p. Host-fillable, device-readable (host-pinned buffer).
struct AddressPair {
  const void* source;
  void* dest;
};

#if defined(__HIPCC__) || defined(__HIP__)

#define USE_NONTEMPORAL_LOAD 0
#define GLOBAL_SPACE __attribute__((address_space(1)))
#define BREAK_ON_RETRIES 1

// Streaming (cache-bypassing) 16-byte load/store.
#if (defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__))
#if __has_builtin(__builtin_amdgcn_global_load_b128) && \
    __has_builtin(__builtin_amdgcn_global_store_b128)
#elif defined(__HIP_DEVICE_COMPILE__)
#error "Global b128 load/store not supported on this architecture"
#endif
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
__device__ __host__ inline static T* Iglobal(T* ptr) {
  return (T*)(GLOBAL_SPACE T*)reinterpret_cast<uintptr_t>(ptr);
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

__device__ __forceinline__ void WriteFusedPacket(int lane, 
     const void* srcBuf, const void* dstBuf, size_t packetSize, HSAuint64* signal,
     uint32_t* outBasePtr) {
  
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
  //   hdr.operation = SDMA_ATOMIC_ADD32;
  //   dw[0] = hdr.DW_0_DATA;
  //   dw[1] = (uint32_t)((uintptr_t)signal);
  //   dw[2] = (uint32_t)((uintptr_t)signal >> 32);
  //   dw[3] = 1;
  if (lane % 2 == 0) {
  // Header depends only on constants; a scalar-replaceable local keeps the
  // bitfield layout authoritative and constant-folds (no address taken).
    decltype(SDMA_PKT_COPY_LINEAR::HEADER_UNION) cp;
    cp.DW_0_DATA = 0;
    cp.op = SDMA_OP_COPY;
    cp.sub_op = SDMA_SUBOP_COPY_LINEAR;
    decltype(SDMA_PKT_ATOMIC::HEADER_UNION) inc;
    inc.DW_0_DATA = 0;
    inc.op = SDMA_OP_ATOMIC;
    inc.operation = SDMA_ATOMIC_ADD32;

    uint32_t flag = lane >> 1;
    dw[0] = inc.DW_0_DATA & -flag;
    dw[1] = flag == 0 ? cp.DW_0_DATA : (uint32_t)((uintptr_t)signal);
    dw[2] = flag == 0 ? static_cast<uint32_t>(packetSize) - 1 : 
                        (uint32_t)((uintptr_t)signal >> 32);
    dw[3] = flag;
  } else {
    // lane 1 - real addresses, lane 3 - all zeros
    uint32_t mask = lane == 1 ? ~0u : 0;
    dw[0] = (uint32_t)(uintptr_t)srcBuf & mask;
    dw[1] = (uint32_t)((uintptr_t)srcBuf >> 32) & mask;
    dw[2] = (uint32_t)(uintptr_t)dstBuf & mask;
    dw[3] = (uint32_t)((uintptr_t)dstBuf >> 32) & mask;
  }
  const V128 v = {dw[0], dw[1], dw[2], dw[3]};
  StreamStore<EAgentScope, 16>(outBasePtr + lane * 4, v);
} 

// Broadcast `v` from `srcLane` of this wave. ds_bpermute is lane-addressed;
// srcLane must be < warpSize (64 on gfx950, 32 on gfx1250).
template <typename T>
__device__ __forceinline__ T broadcast_warp(T v, int srcLane) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0, 
                        "T must be a multiple of uint32_t");
  union {
    T v;
    uint32_t dw[sizeof(T) / sizeof(uint32_t)];
  } S = {.v = v};
#pragma unroll
  for (int i = 0; i < sizeof(T) / sizeof(uint32_t); i++) {
    S.dw[i] = __builtin_amdgcn_ds_bpermute(srcLane << 2, S.dw[i]);
  }
  return S.v;
}

// SDMA queue handle augmented with collective-specific ring writers. It adds no
// data members (same layout as the base), so a base handle can be used through it
// via a reinterpret_cast -- mirroring anvil::SdmaQueueSingleProducerDeviceHandle.
struct SdmaCollectiveHandle : mori::cco::ccoSdmaQueueDeviceHandle {

  static __device__ __forceinline__ uint64_t WrapIntoRing(uint64_t index) {
    return index % mori::cco::CCO_SDMA_QUEUE_SIZE;
  }

  __device__ __forceinline__ bool CanWriteUpto(uint64_t uptoIndex) {
    const uint64_t queue_size_in_bytes = mori::cco::CCO_SDMA_QUEUE_SIZE;
    if ((uptoIndex - cachedHwReadIndex) < queue_size_in_bytes) {
      return true;
    }
    // Only read hardware register if the queue is full based on cached index
    cachedHwReadIndex = __hip_atomic_load(rptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return (uptoIndex - cachedHwReadIndex) < queue_size_in_bytes;
  }

  // Multi-producer CAS reserve. Packets are always kSDMACopyAtomicPktSize and
  // the ring is a multiple of that, so a reservation never straddles the wrap.
  __device__ __forceinline__ uint64_t ReserveQueueSpace(const size_t size_in_bytes) {
    uint64_t cur_index;
    long long retries = 0;

    while (true) {
      cur_index = __hip_atomic_load(cachedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t new_index = cur_index + size_in_bytes;

      if (CanWriteUpto(new_index)) {
        if (__hip_atomic_compare_exchange_strong(cachedWptr, &cur_index, new_index,
                                                 __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
      // Never return an unreserved index: fail loud rather than corrupt the queue.
      if (++retries > mori::cco::CCO_SDMA_MAX_RETRIES) __builtin_trap();
    }
    return cur_index;
  }

  // Sole-producer reservation: no CAS. Valid only when this queue has exactly
  // one producing thread for the lifetime of the reservation (e.g. one leader
  // lane per distinct queue). Claim-then-wait like cco ReserveSlot: fetch_add
  // first, then poll rptr if occupancy exceeds the ring.
  //
  // Packets are always kSDMACopyAtomicPktSize and the ring is a multiple of that,
  // so a reservation never straddles the wrap.
  //
  // cachedHwReadIndex is by-value. The caller snapshots the handle once and
  // reuses it across packets so a refreshed rptr stays in the local copy;
  // write the hint back to `shared` once at the end if it moved. The hint only
  // moves forward to a value rptr actually had, so a later writer is harmless.
  __device__ __forceinline__ uint64_t ReserveSingleProducer(const size_t size_in_bytes) {
    constexpr uint64_t q_size = mori::cco::CCO_SDMA_QUEUE_SIZE;
    uint64_t base =
        __hip_atomic_fetch_add(cachedWptr, size_in_bytes, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t end = base + size_in_bytes;
    if (end - cachedHwReadIndex > q_size) {
      int64_t retries = 0;
      do {
        cachedHwReadIndex =
            __hip_atomic_load(rptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        if constexpr (mori::cco::CCO_SDMA_BREAK_ON_RETRIES) {
          if (retries++ == mori::cco::CCO_SDMA_MAX_RETRIES) __builtin_trap();
        }
      } while (end - cachedHwReadIndex > q_size);
    }
    return base;
  }
};

static_assert(sizeof(SdmaCollectiveHandle) == sizeof(mori::cco::ccoSdmaQueueDeviceHandle));

// One warp broadcasts a single slice to every peer's destination slice,
// trailing an ADD32(1) into peer signalPtrs[signalSlot]. The per-peer
// destination pointer is supplied by the caller via dstOf(peer) -> uint8_t*
// (warp-uniform at peer granularity), mirroring StartSdmaScatter's dstOf. Four
// consecutive lanes write one fused 64B packet via WriteFusedPacket.
// Multi-producer-safe: other slice-groups' last blocks may hit the same
// per-peer queue concurrently, so this uses the CAS-based ReserveQueueSpace +
// ordered submitPacket (NOT the single-producer CAS-free fast path). The leader
// copies the handle (VGPR pointer fields for submitPacket) and writes
// cachedHwReadIndex back if CanWriteUpto refreshed it. sub==0 of each peer group
// reserves/rings; the whole warp must execute the barriers.
template <class DstFn>
inline __device__ void SdmaBroadcastSliceWarp(
  mori::cco::ccoSdmaContext sdma,
  const void* src, DstFn dstOf, size_t sliceBytes,
                                              int myPe, int npes, int signalSlot, int qId) {
  const uint32_t numSdmaQ = sdma.sdmaNumQueue;
  constexpr int W = warpSize;
  const int lane = static_cast<int>(threadIdx.x) % W;
  const int nWork = npes << 2;

  for (int i = lane; i < nWork; i += W) {
    const int peer = i >> 2, sub = i & 3, leader = lane & ~3;
    const bool active = peer != myPe;

    uint64_t base = 0;
    SdmaCollectiveHandle handle;
    uint32_t* queueBuf = nullptr;
    if (active && sub == 0) {
      auto* shared = static_cast<SdmaCollectiveHandle*>(
          *(sdma.deviceHandles + peer * numSdmaQ + qId));
      handle = *shared;
      const uint64_t hint = handle.cachedHwReadIndex;
      base = handle.ReserveQueueSpace(kSDMACopyAtomicPktSize);  // CAS: multi-producer safe
      if (handle.cachedHwReadIndex != hint) {
        __hip_atomic_store(&shared->cachedHwReadIndex, handle.cachedHwReadIndex,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      queueBuf = handle.queueBuf;
    }
    base = broadcast_warp(base, leader);
    queueBuf = broadcast_warp(queueBuf, leader);

    if (active) {
      // Peer's destination slice, computed by the caller-supplied dstOf(peer).
      uint8_t* dst = dstOf(peer);
      auto* signal = sdma.peerSignalPtrs[peer] + signalSlot;
      const uint64_t Xbase =
          SdmaCollectiveHandle::WrapIntoRing(base) / sizeof(uint32_t);
      WriteFusedPacket(sub, src, dst, sliceBytes, signal, queueBuf + Xbase);
    }

    mori::cco::ccoSdmaPublishStores();
    __builtin_amdgcn_wave_barrier();

    if (active && sub == 0) {
      handle.submitPacket(base, base + kSDMACopyAtomicPktSize);
    }
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------
// Fused SDMA "push" scatter shared by the push collectives (reduce-scatter and
// all-gather). Four consecutive lanes write one fused 64B copy+atomic via
// WriteFusedPacket (one b128 store each). nWork = npes*4; a peer never straddles
// a warp (warpSize % 4 == 0). sub==0 snapshots the queue handle once, then for
// each slice s: reserves 64B, the four lanes write, sub==0 rings that packet.
// Same-lane sequential doorbells; distinct peers are distinct queues, so several
// sub==0 lanes in one warp do not share a commit chain. No LDS / syncthreads.
// Occupancy (rptr) stays; the local cachedHwReadIndex is reused across s and
// written back once if it moved.
//
// Each shard is split into S = 1<<logS slices. Slice s is an SDMA copy from
// srcOf(peer) into dstOf(peer), followed by an ADD32(1) into peer's per-slice
// completion counter signalPtrs[s]. The atomic targets the *receiver's* counter,
// so completion is observed on the receive side -- fire-and-forget, no local
// quiet, no cross-PE barrier.
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
// heapBaseAddr).
// ---------------------------------------------------------------------------
template <uint64_t ElemBytes = 1, class ActiveFn, class SrcFn, class DstFn>
__device__ __forceinline__ void StartSdmaScatter(
    mori::cco::ccoSdmaContext sdma, int npes, int logS, size_t chunkElems,
    ActiveFn activeOf, SrcFn srcOf, DstFn dstOf) {

  static_assert(ElemBytes > 0 && VecBytes % ElemBytes == 0,
                "ElemBytes must divide VecBytes");
  constexpr size_t vecSize = VecBytes / ElemBytes;
  constexpr int W = warpSize;

  const int S = 1 << logS, tid = static_cast<int>(threadIdx.x),
            B = static_cast<int>(blockDim.x);

  const uint32_t numSdmaQ = sdma.sdmaNumQueue;
  const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize,
          sliceBytes = sliceLen * ElemBytes,
          lastBytes = (chunkElems - (sliceLen << logS) + sliceLen) * ElemBytes;

  const int nWork = npes << 2, lane = tid % W;
  for (int i = tid; i < nWork; i += B) {
    const int peer = i >> 2, sub = i & 3, leader = lane & ~3;
    const bool active = activeOf(peer);

    SdmaCollectiveHandle handle;
    SdmaCollectiveHandle* shared = nullptr;
    uint32_t* queueBuf = nullptr;
    uint64_t hint0 = 0;
    if (active && sub == 0) {
      shared = static_cast<SdmaCollectiveHandle*>(
          *(sdma.deviceHandles + peer * numSdmaQ));
      handle = *shared;
      hint0 = handle.cachedHwReadIndex;
      queueBuf = handle.queueBuf;
    }
    queueBuf = broadcast_warp(queueBuf, leader);

    for (int s = 0; s < S; s++) {
      uint64_t pktBase = 0;
      if (active && sub == 0) {
        pktBase = handle.ReserveSingleProducer(kSDMACopyAtomicPktSize);
      }
      pktBase = broadcast_warp(pktBase, leader);

      if (active) {
        auto* srcp = srcOf(peer) + s * sliceBytes;
        auto* dstp = dstOf(peer) + s * sliceBytes;
        const size_t sz = (s == S - 1) ? lastBytes : sliceBytes;
        auto* signal = sdma.peerSignalPtrs[peer] + s;
        const uint64_t ringDw =
            SdmaCollectiveHandle::WrapIntoRing(pktBase) / sizeof(uint32_t);
        WriteFusedPacket(sub, srcp, dstp, sz, signal, queueBuf + ringDw);
      }

      mori::cco::ccoSdmaPublishStores();
      __builtin_amdgcn_wave_barrier();

      if (active && sub == 0) {
        handle.submitPacket(pktBase, pktBase + kSDMACopyAtomicPktSize);
      }
    }

    if (active && sub == 0 && handle.cachedHwReadIndex != hint0) {
      __hip_atomic_store(&shared->cachedHwReadIndex, handle.cachedHwReadIndex,
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

#endif  // __HIPCC__ || __HIP__

}  // namespace collective
}  // namespace mori

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

#include "mori/core/transport/sdma/anvil_device.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"

// ---------------------------------------------------------------------------
// Shared building blocks for collective kernels: cache-bypassing vector
// load/store (StreamLoad/StreamStore), a 128-bit SDMA ring-store helper, and the
// warp-cooperative fused copy+atomic SDMA put used by the push collectives.
// ---------------------------------------------------------------------------

#define USE_FLAT_MEMORY 0
#define USE_NONTEMPORAL_LOAD 0

#if USE_FLAT_MEMORY
#define MEM_SPACE __attribute__((address_space(0)))
#else
#define MEM_SPACE __attribute__((address_space(1)))
#endif
#define GLOBAL_SPACE __attribute__((address_space(1)))

// Streaming (cache-bypassing) 16-byte load/store.
#if (defined(__gfx942__) || defined(__gfx950__)) &&     \
    __has_builtin(__builtin_amdgcn_global_load_b128) && \
    __has_builtin(__builtin_amdgcn_global_store_b128)
#elif defined(__HIP_DEVICE_COMPILE__)
#error "Global b128 load/store not supported on this architecture"
#endif

namespace mori {
namespace collective {

using V128 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using V128_GLOBAL = GLOBAL_SPACE V128*;
template <int VecBytes>
using TVecType = std::conditional_t<
    VecBytes == 1, uint8_t,
        std::conditional_t<VecBytes == 2, uint16_t,
        std::conditional_t<VecBytes == 4, uint32_t,
        std::conditional_t<VecBytes == 8, uint64_t,
        std::conditional_t<VecBytes == 16, V128, void>>>>>;

template <typename T>
__device__ __host__ inline static T* Tglobal(T* ptr) {
  return (T*)(GLOBAL_SPACE T*)reinterpret_cast<uintptr_t>(ptr);
}

template <typename T>
__device__ __host__ inline static MEM_SPACE T* MemSpace(T* ptr) {
  uintptr_t u = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<MEM_SPACE T*>(u);
}

template <int Bytes>
__device__ __forceinline__ TVecType<Bytes> StreamLoad(const void* p, bool system_scope = true) {
  static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8 || Bytes == 16,
                "StreamLoad supports 1/2/4/8/16 byte accesses");
  auto ptr = reinterpret_cast<const TVecType<Bytes>*>(p); 
#if USE_NONTEMPORAL_LOAD
  return __builtin_nontemporal_load(MemSpace(ptr));
#else
  if constexpr (Bytes == 16) {
    if (system_scope) {
      return __builtin_amdgcn_global_load_b128((V128_GLOBAL)p, "");
    } else {
      return __builtin_amdgcn_global_load_b128((V128_GLOBAL)p, "agent");
    }
  } else {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED,
                             system_scope ? __HIP_MEMORY_SCOPE_SYSTEM : __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

template <int Bytes>
__device__ __forceinline__ void StreamStore(void* p, TVecType<Bytes> v, bool system_scope = true) {
  static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8 || Bytes == 16,
                "StreamStore supports 1/2/4/8/16 byte accesses");
  auto ptr = reinterpret_cast<TVecType<Bytes>*>(p);
#if USE_NONTEMPORAL_LOAD
  __builtin_nontemporal_store(v, MemSpace(ptr));
#else
  if constexpr (Bytes == 16) {
    if (system_scope) {
      __builtin_amdgcn_global_store_b128((V128_GLOBAL)p, v, "");
    } else {
      __builtin_amdgcn_global_store_b128((V128_GLOBAL)p, v, "agent");
    }
  } else {
    __hip_atomic_store(ptr, v, __ATOMIC_RELAXED,
                       system_scope ? __HIP_MEMORY_SCOPE_SYSTEM : __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

// SDMA queue handle augmented with collective-specific ring writers. It adds no
// data members (same layout as the base), so a base handle can be used through it
// via a reinterpret_cast -- mirroring anvil::SdmaQueueSingleProducerDeviceHandle.
struct SdmaCollectiveHandle : anvil::SdmaQueueDeviceHandle {

  // Build a fused copy+atomic packet ENTIRELY in registers and stream it to an
  // ABSOLUTE ring index as four b128 stores. 
  // Layout matches SDMA_PKT_COPY_WITH_ATOMIC, assembled from the two shared dword
  // writers: copy = dwords 0..6 (WriteCopyPacket), atomic = dwords 7..14
  // (WriteAtomicAddPacket), trailing single-dword NOP = 15.
  // Caller must have reserved a contiguous, non-wrapping 64B slot at wptrIndex.
  __device__ __forceinline__ void placeCopyAtomicPacketAt(const void* srcBuf, const void* dstBuf,
                                                          size_t copyBytes, HSAuint64* signal,
                                                          uint64_t addVal, uint64_t wptrIndex) {
    uint32_t dw[16];
    anvil::WriteCopyPacket(dw, srcBuf, dstBuf, copyBytes);  // copy: dw[0..6]
    anvil::WriteAtomicAddPacket(dw + 7, signal, addVal);    // atomic: dw[7..14]
    dw[15] = 0;                                             // trailing single-dword NOP
    const uint64_t base = WrapIntoRing(wptrIndex) / sizeof(uint32_t);
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
      const V128 v = {dw[i], dw[i + 1], dw[i + 2], dw[i + 3]};
      StreamStore<16>(queueBuf + base + i, v, false);
    }
  }

  // Fill [wptrIndex, wptrIndex+numBytes) with zero dwords (single-dword SDMA NOPs)
  // so the engine harmlessly skips the wrap-around padding region.
  __device__ __forceinline__ void fillNops(uint64_t wptrIndex, uint64_t numBytes) {
    uint64_t base_index_in_dwords = WrapIntoRing(wptrIndex) / sizeof(uint32_t);
    const uint64_t numDwords = numBytes / sizeof(uint32_t);
    for (uint64_t i = 0; i < numDwords; i++) {
      StreamStore<4>(queueBuf + base_index_in_dwords + i, 0, false);
    }
  }
};

static_assert(sizeof(SdmaCollectiveHandle) == sizeof(anvil::SdmaQueueDeviceHandle));

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

// Warp-cooperative issue of S fused copy+atomic packets into ONE queue. Lane s
// (s in [0,S)) issues slice s: a copy of [srcBase + s*sliceBytes -> dstBase +
// s*sliceBytes] of `sliceBytes` (last lane uses `lastSliceBytes`) followed by an
// ADD64(addVal) into signalsBase[s].
//
// Only lane 0 reserves the whole S-packet block and only lane 0 submits, so there
// is exactly one spinning thread per warp -> no intra-warp producer dependency and
// hence no SIMT deadlock (unlike issuing one packet per thread, where every thread
// would spin in submitPacket). The S packets are written in parallel by the lanes
// into their pre-reserved, contiguous, non-wrapping slots.
//
// MUST be called with the whole warp active (uses __shfl / __syncwarp). S must be
// <= warpSize.
inline __device__ void SdmaPutWarpFusedS(const void* srcBase, void* dstBase, size_t sliceBytes,
                                         size_t lastSliceBytes,
                                         anvil::SdmaQueueDeviceHandle** deviceHandles,
                                         HSAuint64* signalsBase, uint32_t qId, int logS,
                                         uint64_t addVal = 1) {
  const int S = 1 << logS;
  const int lane = threadIdx.x % warpSize;
  constexpr size_t packetSize = sizeof(SDMA_PKT_COPY_WITH_ATOMIC);
  // Use the collective-augmented handle (identical layout) so we can issue the
  // b128 placeCopyAtomicPacketAt / fillNops ring writes.
  auto& handle =
      *static_cast<SdmaCollectiveHandle*>(*(deviceHandles + qId));  // same queue for all lanes

  // Single contiguous reservation for all S packets (done by lane 0). `offset` is
  // wrap-around NOP padding placed before the block so it cannot wrap internally.
  uint64_t offset = 0, startBase = 0;
  if (lane == 0) startBase = handle.ReserveQueueSpace(packetSize << logS, offset);
  // Broadcast both startBase and offset from lane 0 (HIP provides 64-bit __shfl).
  startBase = __shfl(startBase, 0);
  offset = __shfl(offset, 0);

  // Lane 0 fills the wrap padding (rare); each lane writes its own packet slot.
  if (lane == 0 && offset) handle.fillNops(startBase, offset);
  if (lane < S) {
    auto* s = reinterpret_cast<const char*>(srcBase) + lane * sliceBytes;
    auto* d = reinterpret_cast<char*>(dstBase) + lane * sliceBytes;
    size_t sz = (lane == S - 1) ? lastSliceBytes : sliceBytes;
    // Register-resident build + b128 stores (no scratch spill); see placeCopyAtomicPacketAt.
    handle.placeCopyAtomicPacketAt(s, d, sz, signalsBase + lane, addVal,
                                   startBase + offset + lane * packetSize);
  }
  // Reconverge so all lanes have issued their stores before lane 0 submits; the
  // s_waitcnt(0) inside submitPacket then drains the wave-shared vmcnt for them all.
  __syncwarp();
  if (lane == 0) {
    handle.submitPacket(startBase, startBase + offset + (packetSize << logS));
  }
}

// Multi-producer-safe single-thread put: copy packet + completion atomic.
//
// Unlike SdmaPutThread, this reserves space for BOTH packets in ONE
// ReserveQueueSpace call and writes them with ONE placePacket call, so the
// producer's reserved range [startBase, pendingWptr) is a single contiguous
// block.
inline __device__ void SdmaPutThreadFused(void* srcBuf, void* dstBuf, size_t copy_size,
                                          anvil::SdmaQueueDeviceHandle** deviceHandles,
                                          HSAuint64* signals, uint32_t queNum, uint32_t qId,
                                          uint64_t addVal = 1) {
  // A copy-linear packet immediately followed by its completion atomic. Both SDMA
  // packet structs are made entirely of 4-byte fields, so this aggregate has no
  // internal padding and its 15 dwords land in the ring exactly as two adjacent
  // packets would (copy at dwords 0..6, atomic at 7..14).
  struct SDMA_PKT_COPY_WITH_ATOMIC {
    SDMA_PKT_COPY_LINEAR copy;
    SDMA_PKT_ATOMIC atomic;
  };
  static_assert(
      sizeof(SDMA_PKT_COPY_WITH_ATOMIC) == sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC),
      "combined SDMA copy+atomic packet must be tightly packed");
  uint64_t offset = 0;
  char* srcPtr = reinterpret_cast<char*>(srcBuf);
  char* dstPtr = reinterpret_cast<char*>(dstBuf);

  auto& handle = *static_cast<SdmaCollectiveHandle*>(*(deviceHandles + qId));

  // Single contiguous reservation for the copy + atomic packets. `offset` is the
  // wrap-around NOP padding; it precedes the combined packet, which cannot wrap
  // internally because it was reserved as one block.
  const uint64_t startBase = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_WITH_ATOMIC), offset);
  uint64_t pendingWptr = startBase;

  SDMA_PKT_COPY_WITH_ATOMIC packet;
  packet.copy = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size);
  packet.atomic = anvil::CreateAtomicAddPacket(signals + qId, addVal);
  handle.placePacket(packet, pendingWptr, offset);

  handle.submitPacket(startBase, pendingWptr);
}

}  // namespace collective
}  // namespace mori

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

#include "mori/application/transport/sdma/anvil_device.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"  // load<N>/store<N>, VecTypeSelector
#include "mori/core/transport/sdma/device_primitives.hpp"

// ---------------------------------------------------------------------------
// Shared building blocks for collective kernels: cache-bypassing vector
// load/store (StreamLoad/StreamStore), a 128-bit SDMA ring-store helper, and the
// warp-cooperative fused copy+atomic SDMA put used by the push collectives.
// ---------------------------------------------------------------------------

#define GLOBAL_SPACE __attribute__((address_space(1)))

// Streaming (cache-bypassing) 16-byte load/store.
#if (defined(__gfx942__) || defined(__gfx950__)) && \
    __has_builtin(__builtin_amdgcn_global_load_b128) &&  \
    __has_builtin(__builtin_amdgcn_global_store_b128)
#elif defined(__HIP_DEVICE_COMPILE__)
#error "Global b128 load/store not supported on this architecture"
#endif

namespace mori {
namespace collective {

template <int Bytes>
using TVecType = typename mori::core::VecTypeSelector<Bytes>::dataType;

using V128 = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;
using V128_GLOBAL = GLOBAL_SPACE V128*;

template <typename T>
__device__ __host__ inline static T* Tglobal(T* ptr) {
  return (T*)(GLOBAL_SPACE T*)reinterpret_cast<uintptr_t>(ptr);
}

template <int Bytes>
__device__ __forceinline__ TVecType<Bytes> StreamLoad(const void* p) {
  return mori::core::load<Bytes>(p);  // generic fallback (8/4/2/1 byte tails)
}
template <int Bytes>
__device__ __forceinline__ void StreamStore(void* p, TVecType<Bytes> v, bool isAgent = false) {
  mori::core::store<Bytes>(p, v);
}

template <>
__device__ __forceinline__ TVecType<16> StreamLoad<16>(const void* p) {
  V128 raw = __builtin_amdgcn_global_load_b128((V128_GLOBAL)p, "");
  return __builtin_bit_cast(TVecType<16>, raw);
}
template <>
__device__ __forceinline__ void StreamStore<16>(void* p, TVecType<16> v, bool isAgent) {
  V128 raw = __builtin_bit_cast(V128, v);
  if (isAgent) {
    __builtin_amdgcn_global_store_b128((V128_GLOBAL)p, raw, "agent");
  } else {
    __builtin_amdgcn_global_store_b128((V128_GLOBAL)p, raw, "");
  }
}

// SDMA queue handle augmented with collective-specific ring writers. It adds no
// data members (same layout as the base), so a base handle can be used through it
// via a reinterpret_cast -- mirroring anvil::SdmaQueueSingleProducerDeviceHandle.
struct SdmaCollectiveHandle : anvil::SdmaQueueDeviceHandle {
  // Write a packet at an ABSOLUTE ring index, with no wrap-NOP padding and without
  // mutating any write pointer. Used by warp-cooperative issue where a block of S
  // packets is reserved once (contiguous, pre-padded past the ring end so it
  // cannot wrap internally) and each lane writes its own slot in parallel.
  template <typename PacketType>
  __device__ __forceinline__ void placePacketAt(PacketType& packet, uint64_t wptrIndex) {
    static_assert(sizeof(PacketType) / sizeof(uint32_t) <= 64);
    constexpr size_t numDwords = sizeof(PacketType) / sizeof(uint32_t);
    uint32_t* packetPtr = reinterpret_cast<uint32_t*>(&packet);
    uint64_t base_index_in_dwords = WrapIntoRing(wptrIndex) / sizeof(uint32_t);
    // b128 bulk (groups of 4 dwords) + dword tail for any remainder.
    size_t i = 0;
#pragma unroll
    for (; i + 4 <= numDwords; i += 4) {
      StreamStore<16>(queueBuf + base_index_in_dwords + i, 
                   *reinterpret_cast<TVecType<16>*>(packetPtr + i), true);
    }
#pragma unroll
    for (; i < numDwords; i++) {
      __hip_atomic_store(queueBuf + base_index_in_dwords + i, packetPtr[i], __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  // Fill [wptrIndex, wptrIndex+numBytes) with zero dwords (single-dword SDMA NOPs)
  // so the engine harmlessly skips the wrap-around padding region.
  __device__ __forceinline__ void fillNops(uint64_t wptrIndex, uint64_t numBytes) {
    uint64_t base_index_in_dwords = WrapIntoRing(wptrIndex) / sizeof(uint32_t);
    const uint64_t numDwords = numBytes / sizeof(uint32_t);
    for (uint64_t i = 0; i < numDwords; i++) {
      __hip_atomic_store(queueBuf + base_index_in_dwords + i, 0, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
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
inline __device__ void SdmaPutWarpFusedS(void* srcBase, void* dstBase, size_t sliceBytes,
                                         size_t lastSliceBytes,
                                         anvil::SdmaQueueDeviceHandle** deviceHandles,
                                         HSAuint64* signalsBase, uint32_t qId, int S,
                                         uint64_t addVal = 1) {
  const int lane = threadIdx.x % warpSize;
  constexpr size_t pkt = sizeof(SDMA_PKT_COPY_WITH_ATOMIC);
  // Use the collective-augmented handle (identical layout) so we can issue the
  // b128 placePacketAt / fillNops ring writes.
  SdmaCollectiveHandle handle =
      *reinterpret_cast<SdmaCollectiveHandle*>(*(deviceHandles + qId));  // same queue for all lanes

  // Single contiguous reservation for all S packets (done by lane 0). `offset` is
  // wrap-around NOP padding placed before the block so it cannot wrap internally.
  uint64_t offset = 0;
  uint64_t startBase = 0;
  if (lane == 0) startBase = handle.ReserveQueueSpace(pkt * S, offset);
  // Broadcast both startBase and offset from lane 0 (HIP provides 64-bit __shfl).
  startBase = __shfl(startBase, 0);
  offset = __shfl(offset, 0);

  // Lane 0 fills the wrap padding (rare); each lane writes its own packet slot.
  if (lane == 0 && offset) handle.fillNops(startBase, offset);
  if (lane < S) {
    char* s = reinterpret_cast<char*>(srcBase) + lane * sliceBytes;
    char* d = reinterpret_cast<char*>(dstBase) + lane * sliceBytes;
    size_t sz = (lane == S - 1) ? lastSliceBytes : sliceBytes;
    SDMA_PKT_COPY_WITH_ATOMIC packet = {
      .copy = anvil::CreateCopyPacket(s, d, sz),
      .atomic = anvil::CreateAtomicAddPacket(signalsBase + lane, addVal)
    };
    handle.placePacketAt(packet, startBase + offset + lane * pkt);
  }
  // Reconverge so all lanes have issued their stores before lane 0 submits; the
  // s_waitcnt(0) inside submitPacket then drains the wave-shared vmcnt for them all.
  __syncwarp();
  if (lane == 0) handle.submitPacket(startBase, startBase + offset + static_cast<uint64_t>(pkt) * S);
}

}  // namespace collective
}  // namespace mori

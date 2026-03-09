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
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdint.h>  // Required for uint32_t

#include <cassert>

#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"
#include "sdma_pkt_struct.h"

namespace anvil {

// TODO
constexpr uint64_t SDMA_QUEUE_SIZE = 256 * 1024;  // 256KB
constexpr HSA_QUEUE_PRIORITY DEFAULT_PRIORITY = HSA_QUEUE_PRIORITY_NORMAL;
constexpr unsigned int DEFAULT_QUEUE_PERCENTAGE = 100;
constexpr int MAX_RETRIES = 1 << 30;
constexpr bool BREAK_ON_RETRIES = true;

__device__ __forceinline__ SDMA_PKT_COPY_LINEAR CreateCopyPacket(void* srcBuf, void* dstBuf,
                                                                 long long int packetSize) {
  SDMA_PKT_COPY_LINEAR copy_packet = {};

  copy_packet.HEADER_UNION.op = SDMA_OP_COPY;
  copy_packet.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;

  copy_packet.COUNT_UNION.count = (uint32_t)(packetSize - 1);
  copy_packet.SRC_ADDR_LO_UNION.src_addr_31_0 = (uint32_t)(uintptr_t)srcBuf;
  copy_packet.SRC_ADDR_HI_UNION.src_addr_63_32 = (uint32_t)((uintptr_t)srcBuf >> 32);
  copy_packet.DST_ADDR_LO_UNION.dst_addr_31_0 = (uint32_t)(uintptr_t)dstBuf;
  copy_packet.DST_ADDR_HI_UNION.dst_addr_63_32 = (uint32_t)((uintptr_t)dstBuf >> 32);

  return copy_packet;
}

__device__ __forceinline__ SDMA_PKT_ATOMIC CreateAtomicIncPacket(HSAuint64* signal) {
  SDMA_PKT_ATOMIC packet = {};

  packet.HEADER_UNION.op = SDMA_OP_ATOMIC;
  packet.HEADER_UNION.operation = SDMA_ATOMIC_ADD64;

  packet.ADDR_LO_UNION.addr_31_0 = (uint32_t)((uintptr_t)signal);
  packet.ADDR_HI_UNION.addr_63_32 = (uint32_t)((uintptr_t)signal >> 32);

  packet.SRC_DATA_LO_UNION.src_data_31_0 = 0x1;
  packet.SRC_DATA_HI_UNION.src_data_63_32 = 0x0;

  return packet;
}

__device__ __forceinline__ SDMA_PKT_FENCE CreateFencePacket(HSAuint64* address, uint32_t data = 1) {
  SDMA_PKT_FENCE packet = {};

  packet.HEADER_UNION.op = SDMA_OP_FENCE;

  packet.ADDR_LO_UNION.addr_31_0 = (uint32_t)((uintptr_t)address);
  packet.ADDR_HI_UNION.addr_63_32 = (uint32_t)((uintptr_t)address >> 32);

  packet.DATA_UNION.data = data;

  return packet;
}

// Assumes signal is allocated in device memory
__device__ __forceinline__ bool waitForSignal(HSAuint64* addr, uint64_t expected) {
  int retries = 0;
  while (true) {
    uint64_t value = __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    if (value == expected) {
      return true;
    }
    __builtin_amdgcn_s_sleep(1);
    if constexpr (BREAK_ON_RETRIES) {
      if (retries++ == MAX_RETRIES) {
        break;
      }
    }
  }
  return false;
}

struct SdmaQueueDeviceHandle {
  __device__ __forceinline__ uint64_t WrapIntoRing(uint64_t index) {
    const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;
    return index % queue_size_in_bytes;
  }

  __device__ __forceinline__ bool CanWriteUpto(uint64_t uptoIndex) {
    const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;
    if ((uptoIndex - cachedHwReadIndex) < queue_size_in_bytes) {
      return true;
    }
    // Only read hardware register if the queue is full based on cached index
    cachedHwReadIndex = __hip_atomic_load(rptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return (uptoIndex - cachedHwReadIndex) < queue_size_in_bytes;
  }
#if 0
   __device__ __forceinline__ void PadRingToEnd(uint64_t cur_index)
   {
      const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;
      uint64_t new_index = cur_index + (queue_size_in_bytes - WrapIntoRing(cur_index));

      if (!CanWriteUpto(new_index))
      {
         return;
      }

      if (nontemporal_compare_exchange(cachedWptr, cur_index, new_index))
      {
         uint64_t index_in_dwords = WrapIntoRing(cur_index) / sizeof(uint32_t);
         int nDwords = (new_index - cur_index) / sizeof(uint32_t);

         for (int i = 0; i < nDwords; i++)
         {
            queueBuf[index_in_dwords + i] = (uint32_t)0;
         }

         submitPacket(cur_index, new_index);
      }
   }
#endif
  __device__ __forceinline__ uint64_t ReserveQueueSpace(const size_t size_in_bytes,
                                                        uint64_t& offset) {
    const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;

    uint64_t cur_index;
    int retries = 0;

    while (true) {
      cur_index = __hip_atomic_load(cachedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      offset = 0;

      // Wraparound and Pad NOPs on remaining bytes
      if (WrapIntoRing(cur_index) + size_in_bytes > queue_size_in_bytes) {
        // PadRingToEnd(cur_index);
        offset = (queue_size_in_bytes - WrapIntoRing(cur_index));
        // continue;
      }
      uint64_t new_index = cur_index + size_in_bytes + offset;

      if (CanWriteUpto(new_index)) {
        if (__hip_atomic_compare_exchange_strong(cachedWptr, &cur_index, new_index,
                                                 __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT))
        // if (nontemporal_compare_exchange(cachedWptr, expected, new_index))
        {
          break;
        }
      }
      __builtin_amdgcn_s_sleep(1);
      if constexpr (BREAK_ON_RETRIES) {
        if (retries++ == MAX_RETRIES) {
          assert(false && "Retry limit exceed on reserve queue space");
          break;
        }
      }
    }
    return cur_index;
  }

  template <typename PacketType>
  __device__ __forceinline__ void placePacket(PacketType& packet, uint64_t& pendingWptr,
                                              uint64_t offset) {
    // Ensure that one warp can write the whole packet
    static_assert(sizeof(PacketType) / sizeof(uint32_t) <= 64);

    const uint32_t numOffsetDwords = offset / sizeof(uint32_t);
    const uint32_t numDwords = sizeof(PacketType) / sizeof(uint32_t);
    uint32_t* packetPtr = reinterpret_cast<uint32_t*>(&packet);

    uint64_t base_index_in_dwords = WrapIntoRing(pendingWptr) / sizeof(uint32_t);

    for (int i = 0; i < numOffsetDwords; i++) {
      __hip_atomic_store(queueBuf + base_index_in_dwords + i, 0, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    pendingWptr += offset;
    base_index_in_dwords = WrapIntoRing(pendingWptr) / sizeof(uint32_t);

    for (int i = 0; i < numDwords; i++) {
      __hip_atomic_store(queueBuf + base_index_in_dwords + i, packetPtr[i], __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    pendingWptr += sizeof(PacketType);
  }

  __device__ __forceinline__ void submitPacket(uint64_t base, uint64_t pendingWptr) {
    int retries = 0;
    while (true) {
      uint64_t val = __hip_atomic_load(committedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      if (val == base) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);

      if constexpr (BREAK_ON_RETRIES) {
        if (retries++ == MAX_RETRIES) {
          assert(false && "submitPacket: Retry limit exceeded");
          break;
        }
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    // This is nop on gfx942
    __builtin_amdgcn_wave_barrier();
    // Ensure no re-ordering (not part of assembly)
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    // Write to the queue went to fine-grained memory
    // Use release to flush before we update wptr
    __hip_atomic_store(wptr, pendingWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    // Ensure all updates are visible before ringing the doorbell
    // This assumes we write to uncached memory
    // Wait for all stores to be committed
    __builtin_amdgcn_s_waitcnt(0);
    // This is nop on gfx942
    __builtin_amdgcn_wave_barrier();
    // Ensure no re-ordering (not part of assembly)
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    // TODO agent should be sufficient
    __hip_atomic_store(doorbell, pendingWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

    // Ensure no re-ordering (not part of assembly)
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_store(committedWptr, pendingWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    // TODO why is this needed?
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  }

  // Queue resources
  uint32_t* queueBuf;
  HSAuint64* rptr;
  HSAuint64* wptr;
  HSAuint64* doorbell;

  // shared variables
  uint64_t* cachedWptr;
  uint64_t* committedWptr;
  // local variables
  uint64_t cachedHwReadIndex;

 private:
  __device__ __forceinline__ bool nontemporal_compare_exchange(uint64_t* vaddr, uint64_t expected,
                                                               uint64_t value) {
    uint64_t vdst;
    unsigned __int128 vdata = ((unsigned __int128)expected << 64) | value;
    __asm__ __volatile__("flat_atomic_cmpswap_x2 %0, %1, %2 sc0 nt;\n s_waitcnt vmcnt(0); \n\t"
                         : "=v"(vdst)
                         : "v"(vaddr), "v"(vdata));
    return (vdst == expected);
  }
};

struct SdmaQueueSingleProducerDeviceHandle : SdmaQueueDeviceHandle {
#if 0
   __device__ __forceinline__ void PadRingToEnd(uint64_t cur_index)
   {
      const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;
      uint64_t new_index = cur_index + (queue_size_in_bytes - WrapIntoRing(cur_index));

      if (!CanWriteUpto(new_index))
      {
         return;
      }

      // Update cachedWptr
      *cachedWptr = new_index;
      // Place packet
      uint64_t index_in_dwords = WrapIntoRing(cur_index) / sizeof(uint32_t);
      int nDwords = (new_index - cur_index) / sizeof(uint32_t);

      for (int i = 0; i < nDwords; i++)
      {
         queueBuf[index_in_dwords + i] = (uint32_t)0;
      }

      submitPacket(cur_index, new_index);
   }
#endif
  __device__ __forceinline__ uint64_t ReserveQueueSpace(const size_t size_in_bytes,
                                                        uint64_t& offset) {
    const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;

    uint64_t cur_index;
    int retries = 0;

    while (true) {
      cur_index = __hip_atomic_load(cachedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      offset = 0;

      // Wraparound and Pad NOPs on remaining bytes
      if (WrapIntoRing(cur_index) + size_in_bytes > queue_size_in_bytes) {
        // PadRingToEnd(cur_index);
        offset = (queue_size_in_bytes - WrapIntoRing(cur_index));
        // continue;
      }
      uint64_t new_index = cur_index + size_in_bytes + offset;

      if (CanWriteUpto(new_index)) {
        if (__hip_atomic_compare_exchange_strong(cachedWptr, &cur_index, new_index,
                                                 __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT))
        // if (nontemporal_compare_exchange(cachedWptr, expected, new_index))
        {
          break;
        }
      }
      __builtin_amdgcn_s_sleep(1);
      if constexpr (BREAK_ON_RETRIES) {
        if (retries++ == MAX_RETRIES) {
          assert(false && "Retry limit exceed on reserve queue space");
          break;
        }
      }
    }
    return cur_index;
  }

  __device__ __forceinline__ void submitPacket(uint64_t base, uint64_t pendingWptr) {
    int retries = 0;
    while (true) {
      uint64_t val = __hip_atomic_load(committedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      if (val == base) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);

      if constexpr (BREAK_ON_RETRIES) {
        if (retries++ == MAX_RETRIES) {
          assert(false && "submitPacket: Retry limit exceeded");
          break;
        }
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    // This is nop on gfx942
    __builtin_amdgcn_wave_barrier();
    // Ensure no re-ordering (not part of assembly)
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    // Write to the queue went to fine-grained memory
    // Use release to flush before we update wptr
    __hip_atomic_store(wptr, pendingWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    // Ensure all updates are visible before ringing the doorbell
    // This assumes we write to uncached memory
    // Wait for all stores to be committed
    __builtin_amdgcn_s_waitcnt(0);
    // This is nop on gfx942
    __builtin_amdgcn_wave_barrier();
    // Ensure no re-ordering (not part of assembly)
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    // TODO agent should be sufficient
    __hip_atomic_store(doorbell, pendingWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

    // Ensure no re-ordering (not part of assembly)
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_store(committedWptr, pendingWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    // TODO why is this needed?
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  }
};

static_assert(sizeof(SdmaQueueSingleProducerDeviceHandle) == sizeof(SdmaQueueDeviceHandle));

__device__ __forceinline__ void put(SdmaQueueDeviceHandle& handle, void* dst, void* src,
                                    size_t size) {
  uint64_t offset = 0;
  auto base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
  auto packet = CreateCopyPacket(src, dst, size);
  uint64_t pendingWptr = base;
  handle.placePacket(packet, pendingWptr, offset);
  handle.submitPacket(base, pendingWptr);
}

__device__ __forceinline__ void signal(SdmaQueueDeviceHandle& handle, void* signal) {
  uint64_t offset;
  auto base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
  auto packet = CreateAtomicIncPacket(reinterpret_cast<HSAuint64*>(signal));
  uint64_t pendingWptr = base;
  handle.placePacket(packet, pendingWptr, offset);
  handle.submitPacket(base, pendingWptr);
}

__device__ __forceinline__ void putWithSignal(SdmaQueueDeviceHandle& handle, void* dst, void* src,
                                              size_t size, void* signal) {
  uint64_t offset = 0;
  auto base =
      handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC), offset);
  auto copy_packet = CreateCopyPacket(src, dst, size);
  auto signal_packet = CreateAtomicIncPacket(reinterpret_cast<HSAuint64*>(signal));
  uint64_t pendingWptr = base;
  handle.placePacket(copy_packet, pendingWptr, offset);
  handle.placePacket(signal_packet, pendingWptr, 0);
  handle.submitPacket(base, pendingWptr);
}

}  // namespace anvil

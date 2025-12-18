#pragma once

#include <stdint.h> // Required for uint32_t

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"

#include "common/sdma_pkt_struct.h"

namespace anvil
{

// TODO
constexpr uint32_t SDMA_QUEUE_SIZE = 256 * 1024; // 256KB
constexpr HSA_QUEUE_PRIORITY DEFAULT_PRIORITY = HSA_QUEUE_PRIORITY_NORMAL;
constexpr unsigned int DEFAULT_QUEUE_PERCENTAGE = 100;
constexpr int MAX_RETRIES = 1 << 30;
constexpr bool BREAK_ON_RETRIES = false;

__device__ __forceinline__ SDMA_PKT_COPY_LINEAR CreateCopyPacket(void* srcBuf, void* dstBuf, long long int packetSize)
{
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

__device__ __forceinline__ SDMA_PKT_ATOMIC CreateAtomicIncPacket(HSAuint64* signal)
{
   SDMA_PKT_ATOMIC packet = {};

   packet.HEADER_UNION.op = SDMA_OP_ATOMIC;
   packet.HEADER_UNION.operation = SDMA_ATOMIC_ADD64;

   packet.ADDR_LO_UNION.addr_31_0 = (uint32_t)((uintptr_t)signal);
   packet.ADDR_HI_UNION.addr_63_32 = (uint32_t)((uintptr_t)signal >> 32);

   packet.SRC_DATA_LO_UNION.src_data_31_0 = 0x1;
   packet.SRC_DATA_HI_UNION.src_data_63_32 = 0x0;

   return packet;
}

__device__ __forceinline__ SDMA_PKT_FENCE CreateFencePacket(HSAuint64* address, uint32_t data = 1)
{
   SDMA_PKT_FENCE packet = {};

   packet.HEADER_UNION.op = SDMA_OP_FENCE;

   packet.ADDR_LO_UNION.addr_31_0 = (uint32_t)((uintptr_t)address);
   packet.ADDR_HI_UNION.addr_63_32 = (uint32_t)((uintptr_t)address >> 32);

   packet.DATA_UNION.data = data;

   return packet;
}

// Assumes signal is allocated in device memory
__device__ __forceinline__ bool waitForSignal(HSAuint64* addr, uint64_t expected)
{
   int retries = 0;

   while (true)
   {
      uint64_t value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (value == expected)
      {
         return true;
      }
      if constexpr (BREAK_ON_RETRIES)
      {
         if (retries++ == MAX_RETRIES)
         {
            break;
         }
      }
   }
   return false;
}

struct SdmaQueueDeviceHandle
{
   __device__ __forceinline__ uint64_t WrapIntoRing(uint64_t index)
   {
      const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;
      return index % queue_size_in_bytes;
   }

   __device__ __forceinline__ bool CanWriteUpto(uint64_t uptoIndex)
   {
      const uint64_t queue_size_in_bytes = SDMA_QUEUE_SIZE;
      if ((uptoIndex - cachedHwReadIndex) < queue_size_in_bytes)
      {
         return true;
      }
      // Only read hardware register if the queue is full based on cached index
      cachedHwReadIndex = __hip_atomic_load(rptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      return (uptoIndex - cachedHwReadIndex) < queue_size_in_bytes;
   }

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

   __device__ __forceinline__ uint64_t ReserveQueueSpace(const size_t size_in_bytes)
   {
      const uint32_t queue_size_in_bytes = SDMA_QUEUE_SIZE;

      uint64_t cur_index;
      int retries = 0;

      while (true)
      {
         cur_index = __hip_atomic_load(cachedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
         uint64_t new_index = cur_index + size_in_bytes;

         // Wraparound and Pad NOPs on remaining bytes
         if (WrapIntoRing(cur_index) + size_in_bytes > queue_size_in_bytes)
         {
            PadRingToEnd(cur_index);
            continue;
         }

         if (!CanWriteUpto(new_index))
         {
            continue;
         }

         uint64_t expected = cur_index;

         if (nontemporal_compare_exchange(cachedWptr, expected, new_index))
         {
            break;
         }
         if constexpr (BREAK_ON_RETRIES)
         {
            if (retries++ == MAX_RETRIES)
            {
               assert(false && "Retry limit exceed on reserve queue space");
               break;
            }
         }
      }
      return cur_index;
   }

   template <typename PacketType> __device__ __forceinline__ void placePacket(PacketType& packet, uint64_t& pendingWptr)
   {
      // Ensure that one warp can write the whole packet
      static_assert(sizeof(PacketType) / sizeof(uint32_t) <= 64);

      const uint32_t numDwords = sizeof(PacketType) / sizeof(uint32_t);
      uint32_t* packetPtr = reinterpret_cast<uint32_t*>(&packet);

      uint64_t base_index_in_dwords = WrapIntoRing(pendingWptr) / sizeof(uint32_t);

      for (int i = 0; i < numDwords; i++)
      {
         queueBuf[base_index_in_dwords + i] = packetPtr[i];
      }
      pendingWptr += sizeof(PacketType);
   }

   __device__ __forceinline__ void submitPacket(uint64_t base, uint64_t pendingWptr)
   {
      int retries = 0;
      while (true)
      {
         uint64_t val = __hip_atomic_load(committedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
         __atomic_signal_fence(__ATOMIC_SEQ_CST);
         if (val == base)
         {
            break;
         }

         if constexpr (BREAK_ON_RETRIES)
         {
            if (retries++ == MAX_RETRIES)
            {
               assert(false && "submitPacket: Retry limit exceeded");
               break;
            }
         }
      }

      *wptr = (HSAuint64)pendingWptr;

      // Ensure all updates are visisble before ringing the doorbell
      // This assumes we write to uncached memory
      // Wait for all stores to be commited
      __builtin_amdgcn_s_waitcnt(0);
      // This is nop on gfx942
      __builtin_amdgcn_wave_barrier();
      // Ensure no re-ordering (not part of assembly)
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      *doorbell = (HSAuint64)pendingWptr;

      // Ensure no re-ordering (not part of assembly)
      __builtin_amdgcn_s_waitcnt(0);
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      __builtin_nontemporal_store(pendingWptr, committedWptr);
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
   __device__ __forceinline__ bool nontemporal_compare_exchange(uint64_t* vaddr, uint64_t expected, uint64_t value)
   {
      uint64_t vdst;
      unsigned __int128 vdata = ((unsigned __int128)expected << 64) | value;
      __asm__ __volatile__("flat_atomic_cmpswap_x2 %0, %1, %2 sc0 nt;\n s_waitcnt vmcnt(0); \n\t"
                           : "=v"(vdst)
                           : "v"(vaddr), "v"(vdata));
      return (vdst == expected);
   }
};

struct SdmaQueueSingleProducerDeviceHandle : SdmaQueueDeviceHandle
{
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

   __device__ __forceinline__ uint64_t ReserveQueueSpace(const size_t size_in_bytes)
   {
      const uint32_t queue_size_in_bytes = SDMA_QUEUE_SIZE;
      uint64_t cur_index;

      while (true)
      {
         cur_index = *cachedWptr;
         uint64_t new_index = cur_index + size_in_bytes;

         // Wraparound and Pad NOPs on remaining bytes
         if (WrapIntoRing(cur_index) + size_in_bytes > queue_size_in_bytes)
         {
            PadRingToEnd(cur_index);
            continue;
         }

         if (!CanWriteUpto(new_index))
         {
            continue;
         }

         *cachedWptr = new_index;
         break;
      }
      return cur_index;
   }

   __device__ __forceinline__ void submitPacket(uint64_t base, uint64_t pendingWptr)
   {
      *wptr = (HSAuint64)pendingWptr;

      // Ensure all updates are visisble before ringing the doorbell
      // This assumes we write to uncached memory
      // Wait for all stores to be commited
      __builtin_amdgcn_s_waitcnt(0);
      // This is nop on gfx942
      __builtin_amdgcn_wave_barrier();
      // Ensure no re-ordering (not part of assembly)
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      *doorbell = (HSAuint64)pendingWptr;
   }
};

static_assert(sizeof(SdmaQueueSingleProducerDeviceHandle) == sizeof(SdmaQueueDeviceHandle));

__device__ __forceinline__ void put(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size)
{
   auto base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR));
   auto packet = CreateCopyPacket(src, dst, size);
   uint64_t pendingWptr = base;
   handle.placePacket(packet, pendingWptr);
   handle.submitPacket(base, pendingWptr);
}

__device__ __forceinline__ void signal(SdmaQueueDeviceHandle& handle, void* signal)
{
   auto base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC));
   auto packet = CreateAtomicIncPacket(reinterpret_cast<HSAuint64*>(signal));
   uint64_t pendingWptr = base;
   handle.placePacket(packet, pendingWptr);
   handle.submitPacket(base, pendingWptr);
}

__device__ __forceinline__ void putWithSignal(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size,
                                              void* signal)
{
   auto base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC));
   auto copy_packet = CreateCopyPacket(src, dst, size);
   auto signal_packet = CreateAtomicIncPacket(reinterpret_cast<HSAuint64*>(signal));
   uint64_t pendingWptr = base;
   handle.placePacket(copy_packet, pendingWptr);
   handle.placePacket(signal_packet, pendingWptr);
   handle.submitPacket(base, pendingWptr);
}

} // namespace anvil
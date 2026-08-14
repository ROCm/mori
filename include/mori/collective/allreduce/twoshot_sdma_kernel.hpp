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

#include <hip/hip_runtime.h>

#include <cstddef>

#include "mori/collective/allreduce/twoshot_sdma_common.hpp"
#include "mori/collective/intra_node/kernels/vec_type.cuh"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

constexpr int kRSMaxBlocks = 80;

__device__ __forceinline__ void SdmaTransitCacheWriteback() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  asm volatile("buffer_wbl2\n\ts_waitcnt 0" ::: "memory");
#elif defined(__gfx950__)
  asm volatile("buffer_inv\n\ts_waitcnt 0" ::: "memory");
#endif
}

__device__ __forceinline__ void SdmaDirectOutputCacheInvalidate() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
  asm volatile("buffer_inv\n\ts_waitcnt 0" ::: "memory");
#endif
}

template <typename P>
__device__ __forceinline__ void SdmaVisibleStore(P* destination, const P& value) {
  using Uint4 = uint32_t __attribute__((ext_vector_type(4)));
  static_assert(sizeof(P) == sizeof(Uint4));
  const Uint4 bits = *reinterpret_cast<const Uint4*>(&value);
  __builtin_nontemporal_store(bits, reinterpret_cast<Uint4*>(destination));
}

__device__ __forceinline__ void PublishControlFlag(const application::SymmMemObjPtr flagsMemObj,
                                                   size_t flagIndex, uint64_t value, int remotePe) {
  auto* remoteFlags = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[remotePe]);
  __scoped_atomic_store_n(remoteFlags + flagIndex, value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t LoadControlFlag(const uint64_t* flag) {
  return __scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// Legacy per-block barrier for ReduceScatterKernel / standalone Allreduce_sdma.
struct alignas(128) RSBarrierSignal {
  uint32_t sync[kRSMaxBlocks][8];
  alignas(128) uint32_t flag[kRSMaxBlocks];
};

// ============================================================================
// ReduceScatterKernel (LEGACY) — IPC reads with per-block start_sync barrier
//
// Before reading peerPtrs, every block executes a start_sync barrier
// (system-scope atomic store + device-scope atomic load) identical to
// cross_device_reduce_2stage in kernel_impl.cuh.
// This replaces the previous hipStreamSynchronize — no host blocking needed.
// ============================================================================
template <typename T>
__device__ void ReduceScatterKernel_body(int myPe, int npes,
                                         const application::SymmMemObjPtr srcMemObj,
                                         const application::SymmMemObjPtr dstMemObj,
                                         const application::SymmMemObjPtr barrierObj,
                                         size_t elementCount) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t packedPerRank = elementCountPerRank / pack_size;

  if (elementCountPerRank == 0) {
    return;
  }

  // --- start_sync barrier (same as kernel_impl.cuh) --------------------------
  {
    RSBarrierSignal* self_sg = reinterpret_cast<RSBarrierSignal*>(barrierObj->localPtr);
    uint32_t next_flag = self_sg->flag[blockIdx.x] + 1;

    if (threadIdx.x < static_cast<unsigned>(npes)) {
      RSBarrierSignal* remote_sg =
          reinterpret_cast<RSBarrierSignal*>(barrierObj->peerPtrs[threadIdx.x]);

      __scoped_atomic_store_n(&remote_sg->sync[blockIdx.x][myPe], next_flag, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_SYSTEM);

      while (__scoped_atomic_load_n(&self_sg->sync[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_DEVICE) < next_flag);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      self_sg->flag[blockIdx.x] = next_flag;
    }
  }
  // --- barrier done ----------------------------------------------------------

  const size_t totalPacked = static_cast<size_t>(npes) * packedPerRank;
  const size_t start = static_cast<size_t>(myPe) * packedPerRank;
  const size_t end = (myPe == npes - 1) ? totalPacked : start + packedPerRank;

  P* __restrict__ result = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ myDst = result + start;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t idx = start + threadLinearId; idx < end; idx += threadsPerGrid) {
    const P* p0 = reinterpret_cast<const P*>(srcMemObj->peerPtrs[0]);
    A add_reg = upcast_v<typename P::type, pack_size>(p0[idx]);
    for (int pe = 1; pe < npes; ++pe) {
      const P* pp = reinterpret_cast<const P*>(srcMemObj->peerPtrs[pe]);
      packed_assign_add(add_reg, upcast_v<typename P::type, pack_size>(pp[idx]));
    }
    myDst[idx - start] = downcast_v<typename P::type, pack_size>(add_reg);
  }
}

template <typename T>
__global__ void ReduceScatterKernel(int myPe, int npes, const application::SymmMemObjPtr srcMemObj,
                                    const application::SymmMemObjPtr dstMemObj,
                                    const application::SymmMemObjPtr barrierObj,
                                    size_t elementCount) {
  ReduceScatterKernel_body<T>(myPe, npes, srcMemObj, dstMemObj, barrierObj, elementCount);
}

// ============================================================================
// SdmaReduceScatterKernel — SDMA scatter + local reduce in ONE kernel
//
// Replaces ReduceScatterKernel.  Eliminates IPC registration, D2D copy,
// and cross-PE system-scope barriers.
//
//   Phase 1 (block 0):  SDMA scatter — each PE sends partition[destPe] from
//                        its *input* directly to destPe's gather buffer.
//                        No IPC registration of input needed.
//   Phase 2 (block 0):  Wait for all peers' scatter to complete (SDMA flags).
//   Phase 3 (all blocks): Local reduce from gather buffer (HBM only).
//                        No cross-PE reads → no CU-count limit.
//
// Block-0-to-all broadcast uses a device-scope generation counter so the
// kernel works under CUDA graph replay.
// Requirement: gridDim.x <= multiProcessorCount (co-resident blocks).
// ============================================================================
template <typename T>
__device__ void SdmaReduceScatterKernel_body(int myPe, int npes, const T* __restrict__ input,
                                             const application::SymmMemObjPtr dstMemObj,
                                             const application::SymmMemObjPtr flagsMemObj,
                                             CrossPeBarrier* __restrict__ barrier,
                                             size_t elementCount, size_t slotStrideElements,
                                             T* output = nullptr) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t chunkBytes = elementCountPerRank * bytesPerElement;
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t slotStrideBytes = slotStrideElements * bytesPerElement;
  const size_t slotStridePacked = slotStrideElements / pack_size;
  if (elementCountPerRank == 0) return;

  // --- generation counter for device-scope broadcast -------------------------
  __shared__ uint64_t s_next;
  __shared__ uint64_t s_reuse;
  __shared__ uint32_t s_needs_reuse;
  if (threadIdx.x == 0) {
    s_next = barrier->flag + 1ULL;
    s_reuse = barrier->reuseGeneration + 1ULL;
    s_needs_reuse = barrier->needsReuseHandshake && chunkBytes > barrier->reuseSafeChunkBytes;
  }
  __syncthreads();

  if (blockIdx.x == 0) {
    uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
    uint64_t flag_val = s_next;
    uint64_t reuse_val = s_reuse;

    const int warpId = static_cast<int>(threadIdx.x) / warpSize;
    const int laneId = static_cast<int>(threadIdx.x) % warpSize;

    // Protect scratch reuse without a separate stream-barrier launch. Reaching
    // this point proves that the prior operation's local copy-out has finished.
    if (s_needs_reuse && warpId < npes && laneId == 0 && warpId != myPe) {
      PublishControlFlag(flagsMemObj, npes + myPe, reuse_val, warpId);
    }
    if (s_needs_reuse) {
      if (warpId < npes && laneId == 0 && warpId != myPe) {
        while (LoadControlFlag(flags + npes + warpId) < reuse_val);
      }
      __syncthreads();
      if (threadIdx.x == 0) barrier->reuseGeneration = s_reuse;
    }

    // === Phase 1: SDMA scatter ===============================================
    // Each warp handles one destination PE.
    if (warpId < npes && laneId == 0 && warpId != myPe) {
      int destPe = warpId;

      uint8_t* srcPtr = reinterpret_cast<uint8_t*>(const_cast<T*>(input)) +
                        static_cast<size_t>(destPe) * chunkBytes;

      uint8_t* remoteDst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe]) +
                           static_cast<size_t>(myPe) * slotStrideBytes;

      anvil::SdmaQueueDeviceHandle** dh =
          dstMemObj->deviceHandles_d + destPe * dstMemObj->sdmaNumQueue;
      HSAuint64* sig = dstMemObj->signalPtrs + destPe * dstMemObj->sdmaNumQueue;
      HSAuint64* esig = dstMemObj->expectSignalsPtr + destPe * dstMemObj->sdmaNumQueue;
      core::SdmaPutThread(srcPtr, remoteDst, chunkBytes, dh, sig, esig, dstMemObj->sdmaNumQueue, 0);
    }

    // Notify remote PEs that our data has landed
    if (warpId < npes && laneId == 0 && warpId != myPe) {
      int destPe = warpId;
      shmem::ShmemQuietThread(destPe, dstMemObj);
      PublishControlFlag(flagsMemObj, myPe, flag_val, destPe);
    }
    __syncthreads();

    // === Phase 2: Wait for all peers' scatter ================================
    if (warpId < npes && laneId == 0 && warpId != myPe) {
      const int sender = warpId;
      int spin = 0;
      bool warned = false;
      while (LoadControlFlag(flags + sender) < flag_val) {
        if (++spin > 100000000 && !warned) {
          printf("PE %d: SdmaScatter timeout waiting for peer %d\n", myPe, sender);
          warned = true;
        }
      }
    }
    __syncthreads();

    // === Broadcast to all local blocks: scatter done =========================
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(&barrier->flag, s_next, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
  } else {
    // Non-zero blocks: wait for block 0's broadcast (device-scope, L2 only)
    if (threadIdx.x == 0) {
      while (__scoped_atomic_load_n(&barrier->flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
             s_next);
    }
    __syncthreads();
  }

  // The local contribution is read directly from input. This avoids a
  // redundant self-SDMA transfer and never gives a receive slot a CU-written
  // role, which also removes the old local-slot L2 alias hazard.
  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ reduced = buf + static_cast<size_t>(npes) * slotStridePacked;
  P* __restrict__ directOutput =
      output == nullptr || chunkBytes < (64U << 10)
          ? nullptr
          : reinterpret_cast<P*>(output) + static_cast<size_t>(myPe) * packedPerRank;
  const P* __restrict__ inputSlot =
      reinterpret_cast<const P*>(input) + static_cast<size_t>(myPe) * packedPerRank;

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  // === Phase 3: Local reduce (all blocks) ==================================
  for (size_t k = tid; k < packedPerRank; k += stride) {
    A acc = upcast_v<typename P::type, pack_size>(myPe == 0 ? inputSlot[k] : buf[k]);
    for (int pe = 1; pe < npes; ++pe) {
      const P value =
          pe == myPe ? inputSlot[k] : buf[static_cast<size_t>(pe) * slotStridePacked + k];
      packed_assign_add(acc, upcast_v<typename P::type, pack_size>(value));
    }
    const P value = downcast_v<typename P::type, pack_size>(acc);
    SdmaVisibleStore(reduced + k, value);
    if (directOutput != nullptr) SdmaVisibleStore(directOutput + k, value);
  }

  if (output == nullptr) return;

  __syncthreads();
  if (threadIdx.x == 0) {
    __scoped_atomic_store_n(&barrier->blockDone[blockIdx.x], s_next, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_DEVICE);
  }
  if (blockIdx.x != 0) return;
  __syncthreads();

  if (threadIdx.x < gridDim.x) {
    while (__scoped_atomic_load_n(&barrier->blockDone[threadIdx.x], __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE) < s_next);
  }
  __syncthreads();

  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  const int warpId = static_cast<int>(threadIdx.x) / warpSize;
  const int laneId = static_cast<int>(threadIdx.x) % warpSize;
  uint64_t ready_val = static_cast<uint64_t>(s_next) + 1ULL;

  if (warpId < npes && laneId == 0 && warpId != myPe) {
    int remotePe = warpId;
    PublishControlFlag(flagsMemObj, myPe, ready_val, remotePe);
  }
  __syncthreads();
  if (warpId < npes && laneId == 0 && warpId != myPe) {
    while (LoadControlFlag(flags + warpId) < ready_val);
  }
  __syncthreads();

  if (warpId < npes && laneId == 0 && (warpId != myPe || directOutput == nullptr)) {
    int sourcePe = warpId;
    uint8_t* source = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[sourcePe]) +
                      static_cast<size_t>(npes) * slotStrideBytes;
    uint8_t* destination =
        reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(sourcePe) * chunkBytes;
    anvil::SdmaQueueDeviceHandle** handles =
        dstMemObj->deviceHandles_d + sourcePe * dstMemObj->sdmaNumQueue;
    HSAuint64* signals = dstMemObj->signalPtrs + sourcePe * dstMemObj->sdmaNumQueue;
    HSAuint64* expectedSignals = dstMemObj->expectSignalsPtr + sourcePe * dstMemObj->sdmaNumQueue;
    core::SdmaPutThread(source, destination, chunkBytes, handles, signals, expectedSignals,
                        dstMemObj->sdmaNumQueue, 0);
    shmem::ShmemQuietThread(sourcePe, dstMemObj);
  }
  __syncthreads();
  if (threadIdx.x == 0) SdmaDirectOutputCacheInvalidate();

  uint64_t done_val = ready_val + 1ULL;
  if (warpId < npes && laneId == 0 && warpId != myPe) {
    int remotePe = warpId;
    PublishControlFlag(flagsMemObj, myPe, done_val, remotePe);
  }
  __syncthreads();
  if (warpId < npes && laneId == 0 && warpId != myPe) {
    while (LoadControlFlag(flags + warpId) < done_val);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    barrier->flag = done_val;
    barrier->needsReuseHandshake = 0;
  }
}

template <typename T>
__global__ void SdmaReduceScatterKernel(int myPe, int npes, const T* __restrict__ input,
                                        const application::SymmMemObjPtr dstMemObj,
                                        const application::SymmMemObjPtr flagsMemObj,
                                        CrossPeBarrier* __restrict__ barrier, size_t elementCount,
                                        size_t slotStrideElements) {
  SdmaReduceScatterKernel_body<T>(myPe, npes, input, dstMemObj, flagsMemObj, barrier, elementCount,
                                  slotStrideElements);
}

// ============================================================================
// AllGatherSdmaKernel — AllGather via SDMA
//
// Each rank sends its reduced shard (at dstMemObj->localPtr + myPe * stride)
// to every rank via SDMA put, then waits for all peers to finish.
// ============================================================================
template <typename T>
__device__ void AllGatherSdmaKernel_body(int myPe, int npes,
                                         const application::SymmMemObjPtr srcMemObj,
                                         const application::SymmMemObjPtr dstMemObj,
                                         const application::SymmMemObjPtr flagsMemObj,
                                         CrossPeBarrier* __restrict__ barrier, size_t elementCount,
                                         size_t slotStrideElements, size_t outputBaseOffsetBytes,
                                         size_t sourceOffsetElements = SIZE_MAX) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;

  if (elementCountPerRank == 0) {
    return;
  }

  const size_t bytesPerElement = sizeof(T);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  __shared__ uint64_t ag_token;
  if (threadIdx.x == 0) {
    ag_token = static_cast<uint64_t>(barrier->flag) + 1ULL;
  }
  __syncthreads();
  uint64_t ready_val = ag_token;
  uint64_t done_val = ag_token + 1ULL;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // A nonzero output base lies wholly in slot-0 padding and cannot overwrite
  // receive data. Compact output at base zero still needs the cross-rank ready
  // rendezvous before any peer writes into another rank's receive slots.
  if (outputBaseOffsetBytes == 0) {
    if (warpId < npes && laneId == 0 && warpId != myPe) {
      PublishControlFlag(flagsMemObj, myPe, ready_val, warpId);
    }
    __syncthreads();
    if (warpId < npes && laneId == 0 && warpId != myPe) {
      while (LoadControlFlag(flags + warpId) < ready_val);
    }
    __syncthreads();
  }

  // --- SDMA put: send my reduced shard to every rank -------------------------
  if (sourceOffsetElements == SIZE_MAX) {
    sourceOffsetElements = static_cast<size_t>(myPe) * slotStrideElements;
  }
  uint8_t* agSrcPtr =
      reinterpret_cast<uint8_t*>(srcMemObj->localPtr) + sourceOffsetElements * bytesPerElement;
  size_t agSendBytes = elementCountPerRank * bytesPerElement;

  if (warpId < npes && laneId == 0 && (outputBaseOffsetBytes == 0 || warpId != myPe)) {
    int remotePe = warpId;
    application::SymmMemObjPtr dest = dstMemObj;

    const size_t remoteOutputBase =
        outputBaseOffsetBytes == 0
            ? 0
            : static_cast<size_t>(static_cast<ptrdiff_t>(outputBaseOffsetBytes) +
                                  static_cast<ptrdiff_t>(myPe - remotePe) *
                                      static_cast<ptrdiff_t>(agSendBytes));
    uint8_t* agDstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + remoteOutputBase +
                        static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;

    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(agSrcPtr, agDstPtr, agSendBytes, devicehandles, signals, expectedSignals,
                        dest->sdmaNumQueue, 0);
  }

  // --- Notify remote PEs that our data is in place ---------------------------
  if (warpId < npes && laneId == 0 && (outputBaseOffsetBytes == 0 || warpId != myPe)) {
    int remotePe = warpId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    if (remotePe != myPe) {
      PublishControlFlag(flagsMemObj, myPe, done_val, remotePe);
    }
  }
  __syncthreads();

  // --- Wait for all peers to finish AllGather --------------------------------
  if (warpId < npes && laneId == 0 && warpId != myPe) {
    const int sender = warpId;
    int spinCount = 0;
    bool warned = false;
    while (LoadControlFlag(flags + sender) < done_val) {
      ++spinCount;
      if (spinCount > 10000000 && !warned) {
        printf("PE %d: AllGather timeout waiting for peer %d\n", myPe, sender);
        warned = true;
      }
    }
  }
  __syncthreads();

  if (threadLinearId == 0) {
    barrier->flag = done_val;
    barrier->needsReuseHandshake = 1;
    // The next scatter writes the prefix of every fixed receive slot. Use the
    // most restrictive rank's padding so every rank makes the same handshake
    // decision; a rank-local threshold would deadlock when only some ranks
    // enter the collective reuse rendezvous.
    const size_t slotStrideBytes = slotStrideElements * bytesPerElement;
    barrier->reuseSafeChunkBytes =
        outputBaseOffsetBytes == 0 ? 0
                                   : slotStrideBytes - static_cast<size_t>(npes - 1) * agSendBytes;
  }

  // Flags are monotonic generation tokens (AMO_SET), so no reset is needed.
}

template <typename T>
__global__ void AllGatherSdmaKernel(int myPe, int npes, const application::SymmMemObjPtr srcMemObj,
                                    const application::SymmMemObjPtr dstMemObj,
                                    const application::SymmMemObjPtr flagsMemObj,
                                    CrossPeBarrier* __restrict__ barrier, size_t elementCount,
                                    size_t slotStrideElements, size_t outputBaseOffsetBytes) {
  AllGatherSdmaKernel_body<T>(myPe, npes, srcMemObj, dstMemObj, flagsMemObj, barrier, elementCount,
                              slotStrideElements, outputBaseOffsetBytes);
}

}  // namespace collective
}  // namespace mori

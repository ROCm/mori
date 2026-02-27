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

#include "mori/shmem/shmem.hpp"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"
#include "mori/collective/intra_node/kernels/vec_type.cuh"

namespace mori {
namespace collective {

constexpr int kRSMaxBlocks = 80;

// Legacy per-block barrier for ReduceScatterKernel / standalone Allreduce_sdma.
struct alignas(128) RSBarrierSignal {
  uint32_t sync[kRSMaxBlocks][8];
  alignas(128) uint32_t flag[kRSMaxBlocks];
};

// Lightweight barrier for SdmaReduceScatterKernel.
// Block 0 does the SDMA scatter + wait, then device-scope broadcasts to
// all other blocks.  Device-side generation counter → graph-safe.
struct alignas(128) CrossPeBarrier {
  alignas(128) uint32_t flag;
};

inline int getDeviceMaxBlocks() {
  int dev = 0;
  hipGetDevice(&dev);
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, dev);
  return (prop.multiProcessorCount > 0) ? prop.multiProcessorCount : 80;
}

// ============================================================================
// ReduceScatterKernel (LEGACY) — IPC reads with per-block start_sync barrier
//
// Before reading peerPtrs, every block executes a start_sync barrier
// (system-scope atomic store + device-scope atomic load) identical to
// cross_device_reduce_2stage in kernel_impl.cuh.
// This replaces the previous hipStreamSynchronize — no host blocking needed.
// ============================================================================
template <typename T>
__global__ void ReduceScatterKernel(int myPe, int npes,
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
    RSBarrierSignal* self_sg =
        reinterpret_cast<RSBarrierSignal*>(barrierObj->localPtr);
    uint32_t next_flag = self_sg->flag[blockIdx.x] + 1;

    if (threadIdx.x < static_cast<unsigned>(npes)) {
      RSBarrierSignal* remote_sg =
          reinterpret_cast<RSBarrierSignal*>(barrierObj->peerPtrs[threadIdx.x]);

      __scoped_atomic_store_n(
          &remote_sg->sync[blockIdx.x][myPe],
          next_flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);

      while (__scoped_atomic_load_n(
                 &self_sg->sync[blockIdx.x][threadIdx.x],
                 __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < next_flag)
        ;
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
__global__ void SdmaReduceScatterKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr flagsMemObj,
    CrossPeBarrier* __restrict__ barrier,
    size_t elementCount) {

  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t chunkBytes = elementCountPerRank * bytesPerElement;
  const size_t packedPerRank = elementCountPerRank / pack_size;
  if (elementCountPerRank == 0) return;

  // --- generation counter for device-scope broadcast -------------------------
  __shared__ uint32_t s_next;
  if (threadIdx.x == 0) {
    s_next = barrier->flag + 1;
  }
  __syncthreads();

  if (blockIdx.x == 0) {
    // === Phase 1: SDMA scatter ===============================================
    // Each warp handles one destination PE.
    uint64_t* __restrict__ flags =
        reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
    int flag_val = 1;

    const int warpId  = static_cast<int>(threadIdx.x) / warpSize;
    const int laneId  = static_cast<int>(threadIdx.x) % warpSize;

    if (warpId < npes && laneId == 0) {
      int destPe = warpId;

      uint8_t* srcPtr = reinterpret_cast<uint8_t*>(
                             const_cast<T*>(input))
                         + static_cast<size_t>(destPe) * chunkBytes;

      uint8_t* remoteDst = reinterpret_cast<uint8_t*>(
                               dstMemObj->peerPtrs[destPe])
                           + static_cast<size_t>(myPe) * chunkBytes;

      anvil::SdmaQueueDeviceHandle** dh =
          dstMemObj->deviceHandles_d + destPe * dstMemObj->sdmaNumQueue;
      HSAuint64* sig  = dstMemObj->signalPtrs
                        + destPe * dstMemObj->sdmaNumQueue;
      HSAuint64* esig = dstMemObj->expectSignalsPtr
                        + destPe * dstMemObj->sdmaNumQueue;

      core::SdmaPutThread(srcPtr, remoteDst, chunkBytes,
                          dh, sig, esig, dstMemObj->sdmaNumQueue, 0);
    }

    // Notify remote PEs that our data has landed
    if (warpId < npes && laneId == 0) {
      int destPe = warpId;
      shmem::ShmemQuietThread(destPe, dstMemObj);
      shmem::ShmemAtomicSizeNonFetchThread(
          flagsMemObj,
          static_cast<size_t>(myPe) * sizeof(uint64_t),
          &flag_val, 8, core::atomicType::AMO_ADD, destPe);
    }
    __syncthreads();

    // === Phase 2: Wait for all peers' scatter ================================
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) continue;
      if (threadIdx.x == 0) {
        int spin = 0;
        while (core::AtomicLoadRelaxed(flags + sender) == 0) {
          if (++spin > 100000000) {
            printf("PE %d: SdmaScatter timeout waiting for peer %d\n",
                   myPe, sender);
            break;
          }
        }
      }
      __syncthreads();
    }

    // Reset flags for the AllGather phase that follows
    if (threadIdx.x < static_cast<unsigned>(npes)) {
      flags[threadIdx.x] = 0;
    }
    __syncthreads();

    // === Broadcast to all local blocks: scatter done =========================
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(
          &barrier->flag, s_next,
          __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
  } else {
    // Non-zero blocks: wait for block 0's broadcast (device-scope, L2 only)
    if (threadIdx.x == 0) {
      while (__scoped_atomic_load_n(
                 &barrier->flag,
                 __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < s_next)
        ;
    }
    __syncthreads();
  }

  // === Phase 3: Local reduce (all blocks, HBM only) =========================
  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ myDst = buf + static_cast<size_t>(myPe) * packedPerRank;

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x)
      + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t k = tid; k < packedPerRank; k += stride) {
    A acc = upcast_v<typename P::type, pack_size>(buf[k]);
    for (int pe = 1; pe < npes; ++pe) {
      packed_assign_add(
          acc,
          upcast_v<typename P::type, pack_size>(
              buf[static_cast<size_t>(pe) * packedPerRank + k]));
    }
    myDst[k] = downcast_v<typename P::type, pack_size>(acc);
  }
}

// ============================================================================
// AllGatherSdmaKernel — AllGather via SDMA
//
// Each rank sends its reduced shard (at dstMemObj->localPtr + myPe * stride)
// to every rank via SDMA put, then waits for all peers to finish.
// ============================================================================
template <typename T>
__global__ void AllGatherSdmaKernel(int myPe, int npes,
                                    const application::SymmMemObjPtr dstMemObj,
                                    const application::SymmMemObjPtr flagsMemObj,
                                    size_t elementCount) {
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
  int flag_val = 1;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // --- SDMA put: send my reduced shard to every rank -------------------------
  uint8_t* agSrcPtr = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                      + static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;
  size_t agSendBytes = elementCountPerRank * bytesPerElement;

  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    application::SymmMemObjPtr dest = dstMemObj;

    uint8_t* agDstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe])
                        + static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;

    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals         = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(agSrcPtr, agDstPtr, agSendBytes,
                        devicehandles, signals, expectedSignals,
                        dest->sdmaNumQueue, 0);
  }

  // --- Notify remote PEs that our data is in place ---------------------------
  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThread(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t),
        &flag_val, 8, core::atomicType::AMO_ADD, remotePe);
  }
  __syncthreads();

  // --- Wait for all peers to finish AllGather --------------------------------
  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) {
      continue;
    }
    if (threadLinearId == 0) {
      int spinCount = 0;
      while (core::AtomicLoadRelaxed(flags + sender) == 0) {
        ++spinCount;
        if (spinCount > 10000000) {
          printf("PE %d: AllGather timeout waiting for peer %d\n", myPe, sender);
          break;
        }
      }
    }
    __syncthreads();
  }

  // Reset flags for potential subsequent invocations.
  if (threadLinearId < static_cast<size_t>(npes)) {
    flags[threadLinearId] = 0;
  }
  __syncthreads();
}

}  // namespace collective
}  // namespace mori

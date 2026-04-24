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

#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/collective/intra_node/kernels/vec_type.cuh"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// ============================================================
// Phase 1: Reduce-Scatter SDMA PUT kernel
//
// Each PE sends shard_j of its input to PE j's output buffer
// at slot myPe. Uses multi-queue SDMA for bandwidth.
//
// After all PEs complete, PE j's output buffer has:
//   slot 0: PE 0's shard_j
//   slot 1: PE 1's shard_j
//   ...
//   slot npes-1: PE (npes-1)'s shard_j
// ============================================================
template <typename T>
__global__ void ReduceScatterSdmaPutKernel(int myPe, int npes, T* input,
                                           const application::SymmMemObjPtr dstMemObj,
                                           size_t elementCountPerRank) {
  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCountPerRank * bytesPerElement;
  const int numQueues = dstMemObj->sdmaNumQueue;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  if (threadLinearId < npes * numQueues) {
    int qId = threadLinearId % numQueues;
    int remotePe = threadLinearId / numQueues;

    const size_t sendBytesBase = bytesPerPeer / static_cast<size_t>(numQueues);
    size_t sendBytes =
        (qId == numQueues - 1) ? (bytesPerPeer - static_cast<size_t>(numQueues - 1) * sendBytesBase)
                               : sendBytesBase;

    // Source: shard for remotePe in my input, split by queue
    size_t srcByteOffset = remotePe * bytesPerPeer + qId * sendBytesBase;
    // Destination: remotePe's output buffer, slot myPe, split by queue
    size_t destByteOffset = myPe * bytesPerPeer + qId * sendBytesBase;

    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(input) + srcByteOffset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + destByteOffset;

    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * numQueues;
    HSAuint64* signals = dest->signalPtrs + remotePe * numQueues;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * numQueues;
    core::SdmaPutThread(srcPtr, dstPtr, sendBytes, devicehandles, signals, expectedSignals,
                        numQueues, qId);
  }
}

// ============================================================
// AllGather async PUT kernel
//
// Sends reduced shard + ShmemQuiet + AMO_SET notification.
// Uses barrier->flag as generation counter (same as AllGatherSdmaKernel).
// Paired with AllGatherAsyncWaitKernel below.
// ============================================================
template <typename T>
__global__ void AllGatherAsyncPutKernel(int myPe, int npes,
                                        const application::SymmMemObjPtr dstMemObj,
                                        const application::SymmMemObjPtr flagsMemObj,
                                        CrossPeBarrier* __restrict__ barrier, size_t elementCount) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;
  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  if (elementCountPerRank == 0) return;

  const size_t bytesPerElement = sizeof(T);

  // Generation counter — matches AllGatherSdmaKernel's protocol
  __shared__ uint64_t ag_token;
  if (threadIdx.x == 0) {
    ag_token = static_cast<uint64_t>(barrier->flag) + 1ULL;
    barrier->flag = static_cast<uint32_t>(ag_token);
  }
  __syncthreads();
  uint64_t flag_val = ag_token;

  int warpId = static_cast<int>(threadIdx.x) / warpSize;
  const int laneId = static_cast<int>(threadIdx.x) % warpSize;

  // SDMA PUT: send my reduced shard to every rank
  uint8_t* agSrcPtr = reinterpret_cast<uint8_t*>(dstMemObj->localPtr) +
                      static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;
  size_t agSendBytes = elementCountPerRank * bytesPerElement;

  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    application::SymmMemObjPtr dest = dstMemObj;

    uint8_t* agDstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) +
                        static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;

    anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* sig = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* esig = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(agSrcPtr, agDstPtr, agSendBytes, dh, sig, esig, dest->sdmaNumQueue, 0);
  }

  // Notify remote PEs
  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flag_val, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
}

// ============================================================
// AllGather async WAIT kernel
//
// Waits for all peers' AllGather PUT to complete.
// Uses AMO_SET + generation counter (< flag_val comparison).
// Paired with AllGatherAsyncPutKernel above.
// ============================================================
__global__ void AllGatherAsyncWaitKernel(int myPe, int npes,
                                         const application::SymmMemObjPtr flagsMemObj,
                                         CrossPeBarrier* __restrict__ barrier,
                                         size_t elementCount) {
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  // Read the generation token set by AllGatherAsyncPutKernel
  uint64_t flag_val = static_cast<uint64_t>(barrier->flag);

  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) continue;
    if (threadIdx.x == 0) {
      int spin = 0;
      bool warned = false;
      while (core::AtomicLoadRelaxed(flags + sender) < flag_val) {
        if (++spin > 100000000 && !warned) {
          printf("PE %d: AllGather wait timeout for peer %d\n", myPe, sender);
          warned = true;
        }
      }
    }
    __syncthreads();
  }

  // Ensure SDMA-written data is visible to subsequent CU reads.
  // SDMA AllGather writes bypass L2/L1, so flush caches to force re-fetch.
  __threadfence_system();
  if (threadIdx.x == 0) {
    // asm volatile("buffer_wbinvl1_vol" ::: "memory");
  }
  __syncthreads();
}

// ============================================================
// Phase 2: Local reduce kernel for reduce-scatter
//
// After scatter, PE j has npes copies of shard_j in its output
// buffer. Sum them element-wise and write the result to slot myPe.
// Grid size should be proportional to elementCountPerRank.
// ============================================================
template <typename T>
__device__ __forceinline__ T reduce_add(T a, T b) {
  return a + b;
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__CUDA_ARCH__)
template <>
__device__ __forceinline__ __half reduce_add(__half a, __half b) {
  return __hadd(a, b);
}
#endif

template <typename T>
__global__ void ReduceScatterLocalReduceKernel(T* gathered, const T* input,
                                               size_t elementCountPerRank, int myPe, int npes) {
  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  T* myDst = gathered + static_cast<size_t>(myPe) * elementCountPerRank;

  // L2 coherence fix: SDMA scatter bypasses L2, but the previous reduce left
  // stale data in L2 for slot[myPe]. Overwrite slot[myPe] from the original
  // input via CU stores to force L2 to match HBM for subsequent reads.
  const T* inputSlot = input + static_cast<size_t>(myPe) * elementCountPerRank;
  for (size_t i = threadLinearId; i < elementCountPerRank; i += threadsPerGrid) {
    myDst[i] = inputSlot[i];
  }

  // Local reduce: sum all npes copies of my shard
  for (size_t i = threadLinearId; i < elementCountPerRank; i += threadsPerGrid) {
    T sum = gathered[i];
    for (int pe = 1; pe < npes; pe++) {
      sum = reduce_add(sum, gathered[static_cast<size_t>(pe) * elementCountPerRank + i]);
    }
    myDst[i] = sum;
  }

  // Flush L2 dirty lines to HBM so AllGather SDMA reads see fresh data.
  // CU reduce writes land in L2 (write-back); SDMA reads bypass L2.
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

// ============================================================
// Phase 1+2 (P2P variant): ReduceScatter via P2P memory reads
//
// Each PE directly reads all PEs' input data from symmetric memory
// (srcMemObj->peerPtrs[pe]) and reduces its own shard in one step.
// No SDMA scatter or explicit wait needed — the GPU does P2P reads.
//
// Requires: user data has been copied to input_transit_buffer (srcMemObj)
// Result: reduced shard written to dstMemObj->localPtr at slot myPe
// ============================================================
template <typename T>
__global__ void ReduceScatterP2pKernel(int myPe, int npes,
                                       const application::SymmMemObjPtr srcMemObj,
                                       const application::SymmMemObjPtr dstMemObj,
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

  P* __restrict__ result = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ myDst = result + static_cast<size_t>(myPe) * packedPerRank;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  const size_t myStart = static_cast<size_t>(myPe) * packedPerRank;

  for (size_t idx = threadLinearId; idx < packedPerRank; idx += threadsPerGrid) {
    size_t globalIdx = myStart + idx;
    const P* p0 = reinterpret_cast<const P*>(srcMemObj->peerPtrs[0]);
    A add_reg = upcast_v<typename P::type, pack_size>(p0[globalIdx]);
    for (int pe = 1; pe < npes; ++pe) {
      const P* pp = reinterpret_cast<const P*>(srcMemObj->peerPtrs[pe]);
      packed_assign_add(add_reg, upcast_v<typename P::type, pack_size>(pp[globalIdx]));
    }
    myDst[idx] = downcast_v<typename P::type, pack_size>(add_reg);
  }

  // Flush L2 dirty lines to HBM so AllGather SDMA reads see fresh data.
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

// ============================================================
// Phase 3: AllGather SDMA PUT kernel for reduced shards
//
// Each PE sends its reduced shard (at slot myPe in the output
// buffer) to every remote PE's output buffer at the same slot.
// Uses multi-queue SDMA for bandwidth.
// ============================================================
template <typename T>
__global__ void AllGatherReducedSdmaPutKernel(int myPe, int npes,
                                              const application::SymmMemObjPtr dstMemObj,
                                              size_t elementCountPerRank) {
  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCountPerRank * bytesPerElement;
  const int numQueues = dstMemObj->sdmaNumQueue;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  if (threadLinearId < npes * numQueues) {
    int qId = threadLinearId % numQueues;
    int remotePe = threadLinearId / numQueues;

    const size_t sendBytesBase = bytesPerPeer / static_cast<size_t>(numQueues);
    size_t sendBytes =
        (qId == numQueues - 1) ? (bytesPerPeer - static_cast<size_t>(numQueues - 1) * sendBytesBase)
                               : sendBytesBase;

    // Both source and destination are at slot myPe, split by queue
    size_t byteOffset = myPe * bytesPerPeer + qId * sendBytesBase;

    application::SymmMemObjPtr dest = dstMemObj;
    T* gathered = reinterpret_cast<T*>(dstMemObj->localPtr);
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(gathered) + byteOffset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + byteOffset;

    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * numQueues;
    HSAuint64* signals = dest->signalPtrs + remotePe * numQueues;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * numQueues;
    core::SdmaPutThread(srcPtr, dstPtr, sendBytes, devicehandles, signals, expectedSignals,
                        numQueues, qId);
  }
}

// ============================================================
// Fused kernel: ReduceScatter + AllGather PUT in one launch.
//
// Block 0: SDMA scatter → wait → broadcast "scatter done"
// All blocks: L2 fix → local reduce → buffer_wbl2
// Block 0: AllGather SDMA PUT
//
// Reduces 4 kernel launches (scatter + wait + reduce + AG-PUT) to 1.
// Requirement: gridDim.x <= multiProcessorCount (co-resident blocks).
// ============================================================
template <typename T>
__global__ void ReduceScatterAllGatherFusedKernel(int myPe, int npes, const T* __restrict__ input,
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

  // =========================================================================
  // Phase 1 + 2: SDMA scatter + wait (block 0 only)
  // Same logic as SdmaReduceScatterKernel: warp-based scatter, AMO_SET flags
  // with generation counter, device-scope broadcast to all blocks.
  // =========================================================================
  if (blockIdx.x == 0) {
    uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
    uint64_t flag_val = static_cast<uint64_t>(s_next);

    const int warpId = static_cast<int>(threadIdx.x) / warpSize;
    const int laneId = static_cast<int>(threadIdx.x) % warpSize;

    // Phase 1: SDMA scatter — each warp handles one destination PE
    if (warpId < npes && laneId == 0) {
      int destPe = warpId;

      uint8_t* srcPtr = reinterpret_cast<uint8_t*>(const_cast<T*>(input)) +
                        static_cast<size_t>(destPe) * chunkBytes;
      uint8_t* remoteDst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe]) +
                           static_cast<size_t>(myPe) * chunkBytes;

      anvil::SdmaQueueDeviceHandle** dh =
          dstMemObj->deviceHandles_d + destPe * dstMemObj->sdmaNumQueue;
      HSAuint64* sig = dstMemObj->signalPtrs + destPe * dstMemObj->sdmaNumQueue;
      HSAuint64* esig = dstMemObj->expectSignalsPtr + destPe * dstMemObj->sdmaNumQueue;
      core::SdmaPutThread(srcPtr, remoteDst, chunkBytes, dh, sig, esig, dstMemObj->sdmaNumQueue, 0);
    }

    // Notify remote PEs that our data has landed
    if (warpId < npes && laneId == 0) {
      int destPe = warpId;
      shmem::ShmemQuietThread(destPe, dstMemObj);
      shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
          flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flag_val, 8,
          core::atomicType::AMO_SET, destPe, 0);
    }
    __syncthreads();

    // Phase 2: Wait for all peers' scatter
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) continue;
      if (threadIdx.x == 0) {
        int spin = 0;
        bool warned = false;
        while (core::AtomicLoadRelaxed(flags + sender) < flag_val) {
          if (++spin > 100000000 && !warned) {
            printf("PE %d: Fused scatter timeout waiting for peer %d\n", myPe, sender);
            warned = true;
          }
        }
      }
      __syncthreads();
    }

    // Broadcast to all local blocks: scatter done
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

  // =========================================================================
  // Phase 2.5 + 3: L2 coherence fix + local reduce (all blocks)
  // Same as SdmaReduceScatterKernel Phase 2.5 + Phase 3.
  // =========================================================================
  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ myDst = buf + static_cast<size_t>(myPe) * packedPerRank;

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  // L2 fix: overwrite slot[myPe] with fresh input via CU stores
  {
    const P* __restrict__ inputSlot =
        reinterpret_cast<const P*>(input) + static_cast<size_t>(myPe) * packedPerRank;
    for (size_t k = tid; k < packedPerRank; k += stride) {
      myDst[k] = inputSlot[k];
    }
  }

  // Local reduce
  for (size_t k = tid; k < packedPerRank; k += stride) {
    A acc = upcast_v<typename P::type, pack_size>(buf[k]);
    for (int pe = 1; pe < npes; ++pe) {
      packed_assign_add(acc, upcast_v<typename P::type, pack_size>(
                                 buf[static_cast<size_t>(pe) * packedPerRank + k]));
    }
    myDst[k] = downcast_v<typename P::type, pack_size>(acc);
  }

  // Flush dirty L2 lines to HBM so SDMA-based AllGather reads fresh data.
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

}  // namespace collective
}  // namespace mori

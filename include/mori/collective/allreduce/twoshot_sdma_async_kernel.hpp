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
#include "mori/collective/intra_node/kernels/vec_type.cuh"

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
__global__ void ReduceScatterSdmaPutKernel(int myPe, int npes,
                                           T* input,
                                           const application::SymmMemObjPtr dstMemObj,
                                           size_t elementCountPerRank) {
  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCountPerRank * bytesPerElement;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  if (threadLinearId < npes * dstMemObj->sdmaNumQueue) {
    int qId = threadLinearId % dstMemObj->sdmaNumQueue;
    int remotePe = threadLinearId / dstMemObj->sdmaNumQueue;

    const size_t sendBytesBase = bytesPerPeer / 8;
    size_t sendBytes = (qId == 7) ? (bytesPerPeer - 7 * sendBytesBase) : sendBytesBase;

    // Source: shard for remotePe in my input, split by queue
    size_t srcByteOffset = remotePe * bytesPerPeer + qId * sendBytesBase;
    // Destination: remotePe's output buffer, slot myPe, split by queue
    size_t destByteOffset = myPe * bytesPerPeer + qId * sendBytesBase;

    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(input) + srcByteOffset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + destByteOffset;

    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(srcPtr, dstPtr, sendBytes, devicehandles, signals, expectedSignals, dest->sdmaNumQueue, qId);
  }
}

// ============================================================
// Wait kernel: signals remote PEs, waits for all SDMA transfers
// to complete, and resets flags. Reused for both scatter-wait
// and allgather-wait phases.
// ============================================================
__global__ void TwoShotAllReduceSdmaAsyncWaitKernel(int myPe, int npes,
                                        const application::SymmMemObjPtr dstMemObj,
                                        const application::SymmMemObjPtr flagsMemObj) {
  int flag_val = 1;
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  if (threadLinearId < npes) {
    int remotePe = threadLinearId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThread(flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flag_val, 8, core::atomicType::AMO_ADD, remotePe);
  }
  __syncthreads();

  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) {
      continue;
    }

    if (threadLinearId == 0) {
      int spinCount = 0;
      while (core::AtomicLoadRelaxed(flags + sender) == 0) {
        ++spinCount;
        if (spinCount > 10000000) {
          printf("PE %d: Timeout waiting for data from peer %d\n", myPe, sender);
          break;
        }
      }
    }
    __syncthreads();
  }

  if (threadLinearId < npes) {
    flags[threadLinearId] = 0;
  }
}

// ============================================================
// Phase 2: Local reduce kernel for reduce-scatter
//
// After scatter, PE j has npes copies of shard_j in its output
// buffer. Sum them element-wise and write the result to slot myPe.
// Grid size should be proportional to elementCountPerRank.
// ============================================================
template <typename T>
__global__ void ReduceScatterLocalReduceKernel(T* gathered, size_t elementCountPerRank,
                                               int myPe, int npes) {
  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  T* myDst = gathered + static_cast<size_t>(myPe) * elementCountPerRank;

  for (size_t i = threadLinearId; i < elementCountPerRank; i += threadsPerGrid) {
    T sum = gathered[i];
    for (int pe = 1; pe < npes; pe++) {
      sum += gathered[static_cast<size_t>(pe) * elementCountPerRank + i];
    }
    myDst[i] = sum;
  }
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
//
// Minimal register usage to maximize occupancy (wavefront count).
// For P2P high-latency reads, the GPU's wavefront scheduler hides
// latency better than explicit pipelining with high register pressure.
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

  if (packedPerRank == 0) {
    return;
  }

  P* __restrict__ myDst = reinterpret_cast<P*>(dstMemObj->localPtr)
                          + static_cast<size_t>(myPe) * packedPerRank;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
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

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  if (threadLinearId < npes * dstMemObj->sdmaNumQueue) {
    int qId = threadLinearId % dstMemObj->sdmaNumQueue;
    int remotePe = threadLinearId / dstMemObj->sdmaNumQueue;

    const size_t sendBytesBase = bytesPerPeer / 8;
    size_t sendBytes = (qId == 7) ? (bytesPerPeer - 7 * sendBytesBase) : sendBytesBase;

    // Both source and destination are at slot myPe, split by queue
    size_t byteOffset = myPe * bytesPerPeer + qId * sendBytesBase;

    application::SymmMemObjPtr dest = dstMemObj;
    T* gathered = reinterpret_cast<T*>(dstMemObj->localPtr);
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(gathered) + byteOffset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + byteOffset;

    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(srcPtr, dstPtr, sendBytes, devicehandles, signals, expectedSignals, dest->sdmaNumQueue, qId);
  }
}

}  // namespace collective
}  // namespace mori

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
template <typename T>
__global__ void ReduceScatterKernel(int myPe, int npes,
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
// Kernel 2: AllGather via SDMA
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

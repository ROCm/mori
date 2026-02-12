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
__global__ void TwoShotAllReduceSdmaKernel(int myPe, int npes,
		                               T* input,
                                       const application::SymmMemObjPtr srcMemObj,
                                       const application::SymmMemObjPtr dstMemObj,
                                       const application::SymmMemObjPtr flagsMemObj,
                                       size_t elementCount) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  // --- Packed-type declarations ---------------------------------------------
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  // --- Uniform, pack_size-aligned shard size --------------------------------
  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t packedPerRank = elementCountPerRank / pack_size;  // exact

  // Number of valid input elements in the last shard (may be < elementCountPerRank).
  const size_t lastShardActualSize =
      (elementCount > static_cast<size_t>(npes - 1) * elementCountPerRank)
      ? (elementCount - static_cast<size_t>(npes - 1) * elementCountPerRank)
      : 0;

  if (elementCountPerRank == 0) {
    return;
  }

  // --- Common variables -----------------------------------------------------
  T* __restrict__ inputData = input;
  T* __restrict__ gathered  = reinterpret_cast<T*>(dstMemObj->localPtr);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  int flag_val = 1;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  const size_t bytesPerElement = sizeof(T);

  int warpId     = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // =========================================================================
  // Phase 1: Gather via SDMA
  //
  // Each warp sends shard_{remotePe} of this rank's input to remotePe.
  // All shards are elementCountPerRank elements except that the LAST shard
  // has only lastShardActualSize valid input elements — we send only the
  // valid portion to avoid reading past the input buffer.
  // =========================================================================
  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;

    // Only the last shard may have fewer valid elements.
    size_t sendCount = (remotePe == npes - 1) ? lastShardActualSize : elementCountPerRank;

    application::SymmMemObjPtr dest = dstMemObj;

    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(inputData)
                      + static_cast<size_t>(remotePe) * elementCountPerRank * bytesPerElement;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe])
                      + static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;
    size_t sendBytes = sendCount * bytesPerElement;

    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals         = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(srcPtr, dstPtr, sendBytes,
                        devicehandles, signals, expectedSignals,
                        dest->sdmaNumQueue, 0);
  }

  // --- Notify remote PEs that our data is in place --------------------------
  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThread(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t),
        &flag_val, 8, core::atomicType::AMO_ADD, remotePe);
  }
  __syncthreads();

  // --- Wait for all peers to finish writing into our dst buffer -------------
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

  if (threadLinearId < static_cast<size_t>(npes)) {
    flags[threadLinearId] = 0;
  }
  __syncthreads();

  // =========================================================================
  // Phase 2: Local reduce (sum) over the npes gathered copies of our shard.
  //
  // elementCountPerRank is a multiple of pack_size, so every PE's slot
  // starts at a pack_size-aligned boundary and no scalar tail is needed.
  // =========================================================================
  P* __restrict__ packedGathered = reinterpret_cast<P*>(gathered);
  P* __restrict__ myDst = packedGathered + static_cast<size_t>(myPe) * packedPerRank;

  for (size_t idx = threadLinearId; idx < packedPerRank; idx += threadsPerGrid) {
    A add_reg = upcast_v<typename P::type, pack_size>(
        *(packedGathered + 0 * packedPerRank + idx));
    for (int pe = 1; pe < npes; ++pe) {
      P tmp = *(packedGathered + static_cast<size_t>(pe) * packedPerRank + idx);
      packed_assign_add(add_reg, upcast_v<typename P::type, pack_size>(tmp));
    }
    myDst[idx] = downcast_v<typename P::type, pack_size>(add_reg);
  }

  // Ensure all Phase 2 reduce writes are globally visible before SDMA reads.
  __threadfence();
  __syncthreads();

  // =========================================================================
  // Phase 3: AllGather via SDMA  (output → dstMemObj / gathered)
  //
  // Each rank sends its reduced shard (elementCountPerRank elements) to
  // every rank.  Source and destination use the same uniform stride
  // (elementCountPerRank), so they coincide — no overlap concern.
  // =========================================================================
  uint8_t* agSrcPtr = reinterpret_cast<uint8_t*>(gathered)
                      + static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;
  size_t   agSendBytes = elementCountPerRank * bytesPerElement;

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

  // --- Notify remote PEs that our AllGather data is in place ----------------
  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThread(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t),
        &flag_val, 8, core::atomicType::AMO_ADD, remotePe);
  }
  __syncthreads();

  // --- Wait for all peers to finish AllGather into our dstMemObj buffer -----
  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) {
      continue;
    }
    if (threadLinearId == 0) {
      int spinCount = 0;
      while (core::AtomicLoadRelaxed(flags + sender) == 0) {
        ++spinCount;
        if (spinCount > 10000000) {
          printf("PE %d: Phase 3 timeout waiting for peer %d\n", myPe, sender);
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

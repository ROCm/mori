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
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

template <typename T>
__global__ void AllGatherRingKernel(int myPe, int npes, const application::SymmMemObjPtr memObj,
                                    const application::SymmMemObjPtr flagsObj) {
  int nextPeer = (myPe + 1) % npes;
  size_t peChunkSize = memObj->size / npes;  // bytes per chunk
  int maxRounds = npes - 1;

  uint64_t* flagsArray = reinterpret_cast<uint64_t*>(flagsObj->localPtr);

  application::RdmaMemoryRegion source;
  source.addr = reinterpret_cast<uintptr_t>(memObj->localPtr);
  source.lkey = memObj->lkey;

  const int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  const int threadLinearId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const size_t bytesPerThread =
      threadsPerBlock > 0 ? (peChunkSize + threadsPerBlock - 1) / threadsPerBlock : peChunkSize;
  int warpId = threadLinearId / warpSize;

  for (int i = 0; i < maxRounds; i++) {
    int sendDataRank = (myPe - i + npes) % npes;
    int recvDataRank = (myPe - i - 1 + npes) % npes;

    size_t chunkBaseOffset = static_cast<size_t>(sendDataRank) * peChunkSize;
#if 0
    size_t threadOffsetWithinChunk = bytesPerThread * static_cast<size_t>(threadLinearId);

    if (threadOffsetWithinChunk < peChunkSize) {
      size_t sendBytes = bytesPerThread;
      size_t remaining = peChunkSize - threadOffsetWithinChunk;
      if (sendBytes > remaining) {
        sendBytes = remaining;
      }
      size_t sourceOffset = chunkBaseOffset + threadOffsetWithinChunk;

      // Each thread pushes a disjoint slice of the current chunk to the next peer.
      shmem::ShmemPutMemNbiThread(memObj, sourceOffset, source, sourceOffset, sendBytes, nextPeer);
    }
#endif
    if(warpId == 0){
      if(sendDataRank != npes-1){
        shmem::ShmemPutMemNbiWarp(memObj,  chunkBaseOffset, source,  chunkBaseOffset, peChunkSize, nextPeer);
      }
      else{
        size_t sendBytes = memObj->size - peChunkSize*(npes-1);
        shmem::ShmemPutMemNbiWarp(memObj,  chunkBaseOffset, source,  chunkBaseOffset, sendBytes, nextPeer);
      }
    }

//    __threadfence_system();
    if(threadLinearId == 0){
      shmem::ShmemQuietThread(nextPeer,scratchMemObj);
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj, sendDataRank * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, nextPeer);
    }
    __syncthreads();

//    if (threadLinearId == 0) {
//      __threadfence_system();
//      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj, sendDataRank * sizeof(uint64_t), 1,
//                                                     core::atomicType::AMO_ADD, nextPeer);
//    }
//    __syncthreads();

    if (threadLinearId == 0) {
      int spinCount = 0;
      while (core::AtomicLoadRelaxed(flagsArray + recvDataRank) != i + 1) {
        spinCount++;
        if (spinCount > 10000000) {  // Increased timeout threshold
          printf(
              "PE %d: Timeout waiting for data from peer %d (round %d, expected flag %d, actual "
              "flag %lu)\n",
              myPe, recvDataRank, i, i + 1, flagsArray[recvDataRank]);
          break;
        }
      }
    }
    __syncthreads();
  }

  size_t flagCount = flagsObj->size / sizeof(uint64_t);
  for (size_t idx = static_cast<size_t>(threadLinearId); idx < flagCount; idx += threadsPerBlock) {
    flagsArray[idx] = 0;
  }
  __syncthreads();
  if (threadLinearId == 0) {
    __threadfence_system();
  }
}

}  // namespace collective
}  // namespace mori

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

__global__ void ReduceScatterRingKernel(int myPe, int npes, const application::SymmMemObjPtr memObj,
                                        const application::SymmMemObjPtr recvMemObj,
                                        const application::SymmMemObjPtr flagsObj) {
  // memObj is the user provided buffer, recvMemObj is the buffer for receiving data then accumulate
  // to memObj, flagsObj is the buffer for flags
  int nextPeer = (myPe + 1) % npes;
  int prevPeer = (myPe - 1 + npes) % npes;
  int peChunkSize = memObj->size / npes;  // bytes per chunk
  int elemsPerPe = memObj->size / sizeof(uint32_t);
  int elemsPerChunk = peChunkSize / sizeof(uint32_t);  // elements per chunk
  int maxRounds = npes - 1;

  uint64_t* flagsArray = reinterpret_cast<uint64_t*>(flagsObj->localPtr);
  uint32_t* recvBase = reinterpret_cast<uint32_t*>(recvMemObj->localPtr);
  uint32_t* srcBase = reinterpret_cast<uint32_t*>(memObj->localPtr);

  application::RdmaMemoryRegion source;
  source.addr = reinterpret_cast<uintptr_t>(memObj->localPtr);
  source.lkey = memObj->lkey;

  for (int i = 0; i < maxRounds; i++) {
    int sendDataRank = ((myPe - i) + npes) % npes;
    int sourceOffset = sendDataRank * peChunkSize;

    int recvDataRank = (myPe - i - 1 + npes) % npes;
    int recvOffset = recvDataRank * peChunkSize;

    uint32_t* recvPtr = recvBase + recvOffset / sizeof(uint32_t);
    uint32_t* oldPtr = srcBase + recvOffset / sizeof(uint32_t);

    // Send data to next peer
    shmem::ShmemPutMemNbiThread(recvMemObj, sourceOffset, source, sourceOffset, peChunkSize,
                                nextPeer);
    __threadfence_system();  // Ensure data is visible to remote PE
    shmem::ShmemQuietThread();
    shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj, sendDataRank * sizeof(uint64_t), 1,
                                                   core::atomicType::AMO_ADD, nextPeer);

    // __threadfence_system();
    // Wait for data from previous peer
    int spinCount = 0;
    while (core::AtomicLoadRelaxed(flagsArray + recvDataRank) != i + 1) {
      spinCount++;
      if (spinCount > 10000000) {  // Increased timeout threshold
        printf(
            "PE %d: Timeout waiting for data from peer %d (round %d, expected flag %d, actual flag "
            "%lu)\n",
            myPe, recvDataRank, i, i + 1, flagsArray[recvDataRank]);
        break;
      }
    }

    // Accumulate the received data
    for (int j = 0; j < elemsPerChunk; j++) {
      oldPtr[j] += recvPtr[j];
    }
    // __threadfence_system();
  }
  memset(flagsArray, 0, flagsObj->size);
  __threadfence_system();
}

}  // namespace collective
}  // namespace mori

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
  int peChunkSize = memObj->size / npes;  // bytes per chunk
  int maxRounds = npes - 1;

  uint64_t* flagsArray = reinterpret_cast<uint64_t*>(flagsObj->localPtr);

  application::RdmaMemoryRegion source;
  source.addr = reinterpret_cast<uintptr_t>(memObj->localPtr);
  source.lkey = memObj->lkey;

  for (int i = 0; i < maxRounds; i++) {
    int sendDataRank = ((myPe - i + 1) + npes) % npes;
    int sourceOffset = sendDataRank * peChunkSize;

    int recvDataRank = (myPe - i + npes) % npes;

    // Send data to next peer
    shmem::ShmemPutMemNbiThread(memObj, sourceOffset, source, sourceOffset, peChunkSize, nextPeer);
    __threadfence_system();  // Ensure data is visible to remote PE
    shmem::ShmemQuietThread();
    shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj, sendDataRank * sizeof(uint64_t), 1,
                                                   core::atomicType::AMO_ADD, nextPeer);

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
  }

  memset(flagsArray, 0, flagsObj->size);
  __threadfence_system();
}

}  // namespace collective
}  // namespace mori

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
#include <mpi.h>

#include <cassert>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr ProviderType PrvdType = ProviderType::MLX5;

__global__ void RingAllGatherWithPutMemAPIKernel(int myPe, int npes, const SymmMemObjPtr memObj) {
  int nextPeer = (myPe + 1) % npes;
  int peChunkSize = memObj->size / npes;

  RdmaMemoryRegion source;
  source.addr = reinterpret_cast<uintptr_t>(memObj->localPtr);
  source.lkey = memObj->lkey;

  for (int i = 0; i < npes - 1; i++) {
    int sendDataRank = ((myPe - i) + npes) % npes;
    int sendOffset = sendDataRank * peChunkSize;

    ShmemPutMemNbiThread(memObj, sendOffset, source, sendOffset, peChunkSize, nextPeer);
    ShmemQuietThread(nextPeer,memObj);

    int recvDataRank = ((sendDataRank - 1) + npes) % npes;
    int recvOffset = recvDataRank * peChunkSize;
    void* recvAddr = reinterpret_cast<char*>(memObj->localPtr) + recvOffset;

    // Wait until received
    while ((atomicAdd(reinterpret_cast<uint32_t*>(recvAddr), 0)) != (recvDataRank + 1)) {
    }
  }
}

inline void checkHipError(hipError_t err, const char* msg, const char* file, int line)
{
   if (err != hipSuccess)
   {
      std::cerr << "HIP error at " << file << ":" << line << " — " << msg << "\n"
                << "  Code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl;
      std::exit(EXIT_FAILURE);
   }
}

// Allow access to peerDeviceId from deviceId
inline void EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
   int canAccess;
   CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
   if (!canAccess)
   {
      std::cerr << "Unable to enable peer access from GPU devices " << deviceId << " to " << peerDeviceId << "\n";
   }

   CHECK_HIP_ERROR(hipSetDevice(deviceId));
   hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
   if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
   {
      std::cerr << "Unable to enable peer to peer access from " << deviceId << "  to " << peerDeviceId << " ("
                << hipGetErrorString(error) << ")\n";
   }
}

void RingAllGatherWithPutMemAPI() {
  int status;
  MPI_Init(NULL, NULL);
  
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  HIP_RUNTIME_CHECK(hipSetDevice(myPe%8));
  // Allocate buffer
  int buffSize = npes * 1024 * sizeof(uint32_t);
  int peChunkSize = buffSize / npes / sizeof(uint32_t);

  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(
      hipMemsetD32(reinterpret_cast<uint32_t*>(buff) + myPe * peChunkSize, myPe + 1, peChunkSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  for (int i = 0; i < npes; i++) {
    printf("Before rank %d, got %d on %dth chunk\n", myPe,
           reinterpret_cast<uint32_t*>(buff)[i * peChunkSize], i);
  }
  // Run put
  RingAllGatherWithPutMemAPIKernel<<<1, 1>>>(myPe, npes, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < npes; i++) {
    printf("After rank %d, got %d on %dth chunk\n", myPe,
           reinterpret_cast<uint32_t*>(buff)[i * peChunkSize], i);
    for (int j = i * peChunkSize; j < ((i + 1) * peChunkSize); j++) {
      assert(reinterpret_cast<uint32_t*>(buff)[j] == i + 1);
    }
  }

  // Finalize
  ShmemFree(buff);
  ShmemFinalize();
  //MPI_Finalize();
}

int main(int argc, char* argv[]) {
  RingAllGatherWithPutMemAPI();
  return 0;
}

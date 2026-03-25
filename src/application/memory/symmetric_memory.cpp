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
#include "mori/application/memory/symmetric_memory.hpp"

#include "hip/hip_runtime.h"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/core.hpp"
#include "mori/application/transport/sdma/anvil.hpp"

#include <vector>

namespace mori {
namespace application {

SymmMemManager::SymmMemManager(BootstrapNetwork& bootNet, Context& context)
    : bootNet(bootNet), context(context) {}

SymmMemManager::~SymmMemManager() {
  while (!memObjPool.empty()) {
    DeregisterSymmMemObj(memObjPool.begin()->first);
  }
}

SymmMemObjPtr SymmMemManager::HostMalloc(size_t size, size_t alignment) {
  void* ptr = nullptr;
  int status = posix_memalign(&ptr, alignment, size);
  assert(!status);
  memset(ptr, 0, size);
  return RegisterSymmMemObj(ptr, size);
}

void SymmMemManager::HostFree(void* localPtr) {
  free(localPtr);
  DeregisterSymmMemObj(localPtr);
}

SymmMemObjPtr SymmMemManager::Malloc(size_t size) {
  void* ptr = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&ptr, size, hipDeviceMallocUncached));
  //HIP_RUNTIME_CHECK(hipMalloc(&ptr, size));
  HIP_RUNTIME_CHECK(hipMemset(ptr, 0, size));
  return RegisterSymmMemObj(ptr, size);
}

SymmMemObjPtr SymmMemManager::ExtMallocWithFlags(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&ptr, size, flags));
  HIP_RUNTIME_CHECK(hipMemset(ptr, 0, size));
  return RegisterSymmMemObj(ptr, size);
}

void SymmMemManager::Free(void* localPtr) {
  HIP_RUNTIME_CHECK(hipFree(localPtr));
  DeregisterSymmMemObj(localPtr);
}

SymmMemObjPtr SymmMemManager::RegisterSymmMemObj(void* localPtr, size_t size) {
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  SymmMemObj* cpuMemObj = new SymmMemObj();
  cpuMemObj->localPtr = localPtr;
  cpuMemObj->size = size;

  // Exchange pointers
  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  bootNet.Allgather(&localPtr, cpuMemObj->peerPtrs, sizeof(uintptr_t));
  // cpuMemObj->peerPtrs[rank] = reinterpret_cast<uintptr_t>(cpuMemObj->localPtr);

  // P2P context: exchange ipc mem handles
  hipIpcMemHandle_t handle;
  HIP_RUNTIME_CHECK(hipIpcGetMemHandle(&handle, localPtr));
  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));
  bootNet.Allgather(&handle, cpuMemObj->ipcMemHandles, sizeof(hipIpcMemHandle_t));
  for (int i = 0; i < worldSize; i++) {
    if ((context.GetTransportType(i) != TransportType::P2P) && (context.GetTransportType(i) != TransportType::SDMA) ) continue;
    if (i == rank) continue;

    HIP_RUNTIME_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&cpuMemObj->peerPtrs[i]),
                                          cpuMemObj->ipcMemHandles[i],
                                          hipIpcMemLazyEnablePeerAccess));
  }
 
  // Rdma context: set lkey and exchange rkeys
  cpuMemObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  cpuMemObj->peerRkeys[rank] = 0;
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) {
    application::RdmaMemoryRegion mr = rdmaDeviceContext->RegisterRdmaMemoryRegion(localPtr, size);
    cpuMemObj->lkey = mr.lkey;
    cpuMemObj->peerRkeys[rank] = mr.rkey;
  }
  bootNet.Allgather(&cpuMemObj->peerRkeys[rank], cpuMemObj->peerRkeys, sizeof(uint32_t));

  // Copy memory object to GPU memory, we need to access it from GPU directly
  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  bool anySdma = false;
  for (int i = 0; i < worldSize; i++) {
    if (context.GetTransportType(i) == TransportType::SDMA) {
      anySdma = true;
      break;
    }
  }
  if (anySdma) {
    int myHipDevice = 0;
    HIP_RUNTIME_CHECK(hipGetDevice(&myHipDevice));

    std::vector<int> hipDevicePerRank(static_cast<size_t>(worldSize), -1);
    bootNet.Allgather(&myHipDevice, hipDevicePerRank.data(), sizeof(int));

    int numOfQueuesPerDevice = gpuMemObj->sdmaNumQueue;
    const size_t dhRowBytes =
        static_cast<size_t>(numOfQueuesPerDevice) *
        sizeof(anvil::SdmaQueueDeviceHandle*);
    const size_t dhBytes = static_cast<size_t>(worldSize) * dhRowBytes;
    HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->deviceHandles_d, dhBytes));
    HIP_RUNTIME_CHECK(hipMemset(gpuMemObj->deviceHandles_d, 0, dhBytes));

    // Rows indexed by MPI rank (destPe in kernels), not by i%8.
    for (int dstRank = 0; dstRank < worldSize; ++dstRank) {
      if (context.GetTransportType(dstRank) != TransportType::SDMA) continue;
      if (dstRank == rank) continue;
      const int dstGpu = hipDevicePerRank[static_cast<size_t>(dstRank)];
      if (dstGpu < 0) continue;
      for (int q = 0; q < numOfQueuesPerDevice; ++q) {
        gpuMemObj->deviceHandles_d[static_cast<size_t>(dstRank) *
                                       static_cast<size_t>(numOfQueuesPerDevice) +
                                   static_cast<size_t>(q)] =
            anvil::anvil.getSdmaQueue(myHipDevice, dstGpu, q)->deviceHandle();
      }
    }

    // Allocate local signal memory: worldSize * numQueues slots
    // Indexed as [sourcePe * numQueues + qId] — each source PE writes to its own slots
    size_t signalArraySize =
        sizeof(HSAuint64) * static_cast<size_t>(worldSize) *
        static_cast<size_t>(numOfQueuesPerDevice);
    HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->signalPtrs, signalArraySize));
    HIP_RUNTIME_CHECK(hipMemset(gpuMemObj->signalPtrs, 0, signalArraySize));
    HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->expectSignalsPtr, signalArraySize));
    HIP_RUNTIME_CHECK(hipMemset(gpuMemObj->expectSignalsPtr, 0, signalArraySize));

    // Exchange signal memory via IPC so each PE can write to remote PE's signalPtrs
    hipIpcMemHandle_t signalHandle;
    HIP_RUNTIME_CHECK(hipIpcGetMemHandle(&signalHandle, gpuMemObj->signalPtrs));

    auto* signalHandles = static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));
    bootNet.Allgather(&signalHandle, signalHandles, sizeof(hipIpcMemHandle_t));

    // Map remote signal memory into local address space
    auto* peerSignalPtrsHost = static_cast<HSAuint64**>(calloc(worldSize, sizeof(HSAuint64*)));
    peerSignalPtrsHost[rank] = gpuMemObj->signalPtrs;  // self points to own signal
    for (int i = 0; i < worldSize; i++) {
      if (context.GetTransportType(i) != TransportType::SDMA) continue;
      if (i == rank) continue;
      void* mappedPtr = nullptr;
      HIP_RUNTIME_CHECK(hipIpcOpenMemHandle(&mappedPtr, signalHandles[i],
                                            hipIpcMemLazyEnablePeerAccess));
      peerSignalPtrsHost[i] = reinterpret_cast<HSAuint64*>(mappedPtr);
    }

    // Copy peerSignalPtrs array to GPU
    HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerSignalPtrs, sizeof(HSAuint64*) * worldSize));
    HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerSignalPtrs, peerSignalPtrsHost,
                                sizeof(HSAuint64*) * worldSize, hipMemcpyHostToDevice));
    free(signalHandles);
    free(peerSignalPtrsHost);

  }
  memObjPool.insert({localPtr, SymmMemObjPtr{cpuMemObj, gpuMemObj}});
  return memObjPool.at(localPtr);
}

void SymmMemManager::DeregisterSymmMemObj(void* localPtr) {
  if (memObjPool.find(localPtr) == memObjPool.end()) return;

  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) rdmaDeviceContext->DeregisterRdmaMemoryRegion(localPtr);

  SymmMemObjPtr memObjPtr = memObjPool.at(localPtr);
  free(memObjPtr.cpu->peerPtrs);
  free(memObjPtr.cpu->peerRkeys);
  free(memObjPtr.cpu->ipcMemHandles);
  free(memObjPtr.cpu);
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerPtrs));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerRkeys));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu));

  memObjPool.erase(localPtr);
}

SymmMemObjPtr SymmMemManager::Get(void* localPtr) const {
  if (memObjPool.find(localPtr) == memObjPool.end()) return SymmMemObjPtr{};
  return memObjPool.at(localPtr);
}

}  // namespace application
}  // namespace mori

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

#include <cstdlib>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/mori_log.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Initialization */
/* ---------------------------------------------------------------------------------------------- */
__constant__ GpuStates globalGpuStates;

void RdmaStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->rdmaStates = new RdmaStates();
  RdmaStates* rdmaStates = states->rdmaStates;

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;

  rdmaStates->commContext = new application::Context(*states->bootStates->bootNet);
}

void MemoryStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  application::Context* context = states->rdmaStates->commContext;

  states->memoryStates = new MemoryStates();
  states->memoryStates->symmMemMgr =
      new application::SymmMemManager(*states->bootStates->bootNet, *context);
  states->memoryStates->mrMgr =
      new application::RdmaMemoryRegionManager(*context->GetRdmaDeviceContext());

  // Allocate static symmetric heap
  // Size can be configured via environment variable
  const char* heapSizeEnv = std::getenv("MORI_SHMEM_HEAP_SIZE");
  size_t heapSize = DEFAULT_SYMMETRIC_HEAP_SIZE;
  
  if (heapSizeEnv) {
    std::string heapSizeStr(heapSizeEnv);
    size_t multiplier = 1;
    
    // Check for suffix
    if (!heapSizeStr.empty()) {
      char lastChar = heapSizeStr.back();
      if (lastChar == 'G' || lastChar == 'g') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;  // GiB
        heapSizeStr.pop_back();
      } else if (lastChar == 'M' || lastChar == 'm') {
        multiplier = 1024ULL * 1024ULL;  // MiB
        heapSizeStr.pop_back();
      }
    }
    
    heapSize = std::stoull(heapSizeStr) * multiplier;
  }

  MORI_SHMEM_INFO("Allocating static symmetric heap of size {} bytes ({} MB)", heapSize,
                  heapSize / (1024 * 1024));

  // Allocate the symmetric heap using the SymmMemManager
  application::SymmMemObjPtr heapObj =
      states->memoryStates->symmMemMgr->ExtMallocWithFlags(heapSize, hipDeviceMallocUncached);
  if (!heapObj.IsValid()) {
    MORI_SHMEM_ERROR("Failed to allocate static symmetric heap!");
    throw std::runtime_error("Failed to allocate static symmetric heap");
  }

  states->memoryStates->staticHeapBasePtr = heapObj.cpu->localPtr;
  states->memoryStates->staticHeapSize = heapSize;
  // IMPORTANT: Start with a small offset to avoid collision between heap base address
  // and first ShmemMalloc allocation. Without this, when staticHeapUsed == 0,
  // the first ShmemMalloc would return staticHeapBasePtr, which is the same address
  // as the heap itself in memObjPool, causing the heap's SymmMemObj to be overwritten.
  constexpr size_t HEAP_INITIAL_OFFSET = 256;
  states->memoryStates->staticHeapUsed = HEAP_INITIAL_OFFSET;
  states->memoryStates->staticHeapObj = heapObj;

  MORI_SHMEM_INFO("Static symmetric heap allocated at {} (local), size {} bytes, initial offset {} bytes",
                  states->memoryStates->staticHeapBasePtr, heapSize, HEAP_INITIAL_OFFSET);
}

void GpuStateInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  RdmaStates* rdmaStates = states->rdmaStates;

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;

  // Copy to gpu constance memory
  GpuStates gpuStates;
  gpuStates.rank = rank;
  gpuStates.worldSize = worldSize;
  gpuStates.numQpPerPe = rdmaStates->commContext->GetNumQpPerPe();

  // Copy transport types to GPU
  HIP_RUNTIME_CHECK(
      hipMalloc(&gpuStates.transportTypes, sizeof(application::TransportType) * worldSize));
  HIP_RUNTIME_CHECK(
      hipMemcpy(gpuStates.transportTypes, rdmaStates->commContext->GetTransportTypes().data(),
                sizeof(application::TransportType) * worldSize, hipMemcpyHostToDevice));

  // Copy endpoints to GPU
  if (rdmaStates->commContext->RdmaTransportEnabled()) {
    size_t numEndpoints = gpuStates.worldSize * gpuStates.numQpPerPe;
    HIP_RUNTIME_CHECK(
        hipMalloc(&gpuStates.rdmaEndpoints, sizeof(application::RdmaEndpoint) * numEndpoints));
    HIP_RUNTIME_CHECK(
        hipMemcpy(gpuStates.rdmaEndpoints, rdmaStates->commContext->GetRdmaEndpoints().data(),
                  sizeof(application::RdmaEndpoint) * numEndpoints, hipMemcpyHostToDevice));

    size_t lockSize = numEndpoints * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMalloc(&gpuStates.endpointLock, lockSize));
    HIP_RUNTIME_CHECK(hipMemset(gpuStates.endpointLock, 0, lockSize));
  }

  // Copy static symmetric heap info to GPU
  uintptr_t heapBase = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
  gpuStates.heapBaseAddr = heapBase;
  gpuStates.heapEndAddr = heapBase + states->memoryStates->staticHeapSize;

  // Use the GPU-side SymmMemObj pointer that was already allocated and initialized
  // by RegisterSymmMemObj (which properly set up peerPtrs and peerRkeys on GPU)
  gpuStates.heapObj = states->memoryStates->staticHeapObj.gpu;

  MORI_SHMEM_INFO("Heap info copied to GPU: base=0x{:x}, end=0x{:x}, size={} bytes, heapObj=0x{:x}",
                  gpuStates.heapBaseAddr, gpuStates.heapEndAddr,
                  gpuStates.heapEndAddr - gpuStates.heapBaseAddr,
                  reinterpret_cast<uintptr_t>(gpuStates.heapObj));

  // Copy gpu states to constant memory
  HIP_RUNTIME_CHECK(
      hipMemcpyToSymbol(globalGpuStates, &gpuStates, sizeof(GpuStates), 0, hipMemcpyHostToDevice));
}

int ShmemInit(application::BootstrapNetwork* bootNet) {
  int status;

  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  states->bootStates = new BootStates();
  states->bootStates->bootNet = bootNet;
  states->bootStates->bootNet->Initialize();
  states->bootStates->rank = states->bootStates->bootNet->GetLocalRank();
  states->bootStates->worldSize = states->bootStates->bootNet->GetWorldSize();

  RdmaStatesInit();
  MemoryStatesInit();
  GpuStateInit();
  states->status = ShmemStatesStatus::Initialized;
  return 0;
}

int ShmemFinalize() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  HIP_RUNTIME_CHECK(hipFree(globalGpuStates.transportTypes));
  HIP_RUNTIME_CHECK(hipFree(globalGpuStates.rdmaEndpoints));


  // Free the static symmetric heap through SymmMemManager
  if (states->memoryStates->staticHeapObj.IsValid()) {
    states->memoryStates->symmMemMgr->Free(states->memoryStates->staticHeapBasePtr);
  }

  delete states->memoryStates->symmMemMgr;
  delete states->memoryStates->mrMgr;
  delete states->memoryStates;

  delete states->rdmaStates->commContext;
  delete states->rdmaStates;

  states->bootStates->bootNet->Finalize();
  delete states->bootStates->bootNet;

  states->status = ShmemStatesStatus::Finalized;
  return 0;
}

int ShmemMpiInit(MPI_Comm mpiComm) {
  return ShmemInit(new application::MpiBootstrapNetwork(mpiComm));
}

int ShmemInit() {
  return ShmemMpiInit(MPI_COMM_WORLD);
}

int ShmemTorchProcessGroupInit(const std::string& groupName) {
  return ShmemInit(new application::TorchBootstrapNetwork(groupName));
}

int ShmemMyPe() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->bootStates->rank;
}

int ShmemNPes() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->bootStates->worldSize;
}

// int ShmemTeamMyPe(ShmemTeamType);
// int ShmemTeamNPes(ShmemTeamType);

}  // namespace shmem
}  // namespace mori

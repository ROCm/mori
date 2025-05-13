#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Intialization                                         */
/* ---------------------------------------------------------------------------------------------- */
__constant__ GpuStates globalGpuStates;

void RdmaStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->rdmaStates = new RdmaStates();
  RdmaStates* rdmaStates = states->rdmaStates;

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;
  assert(worldSize * worldSize <= MaxRdmaEndpointNum);

  // TODO: select device and port
  rdmaStates->context = new application::RdmaContext();
  const application::RdmaDeviceList& devices = rdmaStates->context->GetRdmaDeviceList();

  // Find the first device with active port
  int portId = 0;
  application::RdmaDevice* selectedDevice = nullptr;
  for (auto* device : devices) {
    std::vector<int> activePorts = device->GetActivePortIds();
    if (activePorts.empty()) continue;
    portId = activePorts[0];
    selectedDevice = device;
  }
  assert(portId > 0);
  rdmaStates->deviceContext = selectedDevice->CreateRdmaDeviceContext();

  application::RdmaEndpointConfig config;
  config.portId = portId;
  config.gidIdx = 1;
  // TODO: expose cqe / wqe numfig con
  config.maxMsgsNum = 1024;
  config.maxCqeNum = 1024;
  config.alignment = 4096;
  config.onGpu = true;

  // Create ep for each other rank and connect
  for (int i = 0; i < worldSize; i++) {
    if (rank == i) {
      rdmaStates->localEps.push_back({});
    } else {
      application::RdmaEndpoint ep = rdmaStates->deviceContext->CreateRdmaEndpoint(config);
      rdmaStates->localEps.push_back(ep);
    }

    rdmaStates->remoteEpHandles.push_back(RdmaEndpointHandleList(worldSize));
    states->bootStates->bootNet->Allgather(&(rdmaStates->localEps.data()[i].handle),
                                           rdmaStates->remoteEpHandles[i].data(),
                                           sizeof(application::RdmaEndpointHandle));
  }

  for (int i = 0; i < worldSize; i++) {
    if (rank == i) continue;
    rdmaStates->deviceContext->ConnectEndpoint(rdmaStates->localEps[i].handle,
                                               rdmaStates->remoteEpHandles[rank][i]);
  }
}

void MemoryStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->memoryStates = new MemoryStates();
  states->memoryStates->symmMemMgr = new application::SymmMemManager(
      *states->bootStates->bootNet, *states->rdmaStates->deviceContext);
  states->memoryStates->mrMgr =
      new application::MemoryRegionManager(*states->rdmaStates->deviceContext);
}

void GpuStateInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  RdmaStates* rdmaStates = states->rdmaStates;

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;

  // Copy endpoints to GPU
  HIP_RUNTIME_CHECK(
      hipMalloc(&rdmaStates->localEpsGpu, sizeof(application::RdmaEndpoint) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(rdmaStates->localEpsGpu, rdmaStates->localEps.data(),
                              sizeof(application::RdmaEndpoint) * worldSize,
                              hipMemcpyHostToDevice));

  // Allocate locks
  HIP_RUNTIME_CHECK(hipMalloc(&rdmaStates->epCqLockMemGpu, MaxRdmaEndpointNum * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(rdmaStates->epCqLockMemGpu, 0, MaxRdmaEndpointNum * sizeof(uint32_t)));

  // Copy to gpu constance memory
  GpuStates gpuStates;
  gpuStates.rank = rank;
  gpuStates.worldSize = worldSize;
  gpuStates.epsStartAddr = rdmaStates->localEpsGpu;
  gpuStates.epCqLockMemGpu = rdmaStates->epCqLockMemGpu;

  HIP_RUNTIME_CHECK(
      hipMemcpyToSymbol(globalGpuStates, &gpuStates, sizeof(GpuStates), 0, hipMemcpyHostToDevice));
}

int ShmemMpiInit(MPI_Comm mpi_comm) {
  int status;

  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  states->bootStates = new BootStates();
  states->bootStates->bootNet = new application::MpiBootstrapNetwork(mpi_comm);
  states->bootStates->bootNet->Initialize();
  states->bootStates->rank = states->bootStates->bootNet->GetLocalRank();
  states->bootStates->worldSize = states->bootStates->bootNet->GetWorldSize();

  // TODO: use in-node rank
  HIP_RUNTIME_CHECK(hipSetDevice(states->bootStates->rank));

  RdmaStatesInit();
  MemoryStatesInit();
  GpuStateInit();
  return 0;
}

int ShmemMpiFinalize() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  delete states->memoryStates->symmMemMgr;
  delete states->memoryStates->mrMgr;
  delete states->memoryStates;

  delete states->rdmaStates->deviceContext;
  delete states->rdmaStates->context;
  delete states->rdmaStates;

  states->bootStates->bootNet->Finalize();
  delete states->bootStates->bootNet;
  return 0;
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

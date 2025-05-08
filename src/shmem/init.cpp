#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Intialization                                         */
/* ---------------------------------------------------------------------------------------------- */
__constant__ application::RdmaEndpoint* epsStartAddr;

__global__ void Test(int rank) { printf("%d %p\n", rank, epsStartAddr); }

void RdmaStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->rdmaStates = new RdmaStates();
  RdmaStates* rdmaStates = states->rdmaStates;

  rdmaStates->context = new application::RdmaContext();

  // TODO: select device and port
  const application::RdmaDeviceList& devices = rdmaStates->context->GetRdmaDeviceList();
  rdmaStates->deviceContext = devices[0]->CreateRdmaDeviceContext();

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;
  assert(worldSize * worldSize <= MaxRdmaEndpointNum);

  application::RdmaEndpointConfig config;
  config.portId = 1;
  config.gidIdx = 1;
  config.maxMsgsNum = 1024;
  config.maxCqeNum = 256;
  config.alignment = 4096;
  config.onGpu = true;

  // Create ep for each other rank and connect
  for (int i = 0; i < worldSize; i++) {
    if (rank == i) {
      rdmaStates->localEps.push_back({});
      rdmaStates->remoteEpHandles.push_back(RdmaEndpointHandleList(worldSize));
      continue;
    };

    application::RdmaEndpoint ep = rdmaStates->deviceContext->CreateRdmaEndpoint(config);
    rdmaStates->localEps.push_back(ep);

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

  // Move ep handle to constant gpu memory
  HIP_RUNTIME_CHECK(
      hipMalloc(&rdmaStates->localEpsGpu, sizeof(application::RdmaEndpoint) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(rdmaStates->localEpsGpu, rdmaStates->localEps.data(),
                              sizeof(application::RdmaEndpoint) * worldSize,
                              hipMemcpyHostToDevice));
  HIP_RUNTIME_CHECK(hipMemcpyToSymbol(epsStartAddr, &rdmaStates->localEpsGpu, sizeof(void*), 0,
                                      hipMemcpyHostToDevice));
}

void MemoryStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->memoryStates = new MemoryStates();
  states->memoryStates->symmMemMgr = new application::SymmMemManager(
      *states->bootStates->bootNet, *states->rdmaStates->deviceContext);
  states->memoryStates->mrMgr =
      new application::MemoryRegionManager(*states->rdmaStates->deviceContext);
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

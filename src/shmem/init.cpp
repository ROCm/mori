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

  rdmaStates->commContext = new application::Context(*states->bootStates->bootNet);
}

void MemoryStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  application::Context* context = states->rdmaStates->commContext;

  states->memoryStates = new MemoryStates();
  states->memoryStates->symmMemMgr =
      new application::SymmMemManager(*states->bootStates->bootNet, *context);
  states->memoryStates->mrMgr =
      new application::MemoryRegionManager(*context->GetRdmaDeviceContext());
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

  // Copy transport types to GPU
  HIP_RUNTIME_CHECK(
      hipMalloc(&gpuStates.transportTypes, sizeof(application::TransportType) * worldSize));
  HIP_RUNTIME_CHECK(
      hipMemcpy(gpuStates.transportTypes, rdmaStates->commContext->GetTransportTypes().data(),
                sizeof(application::TransportType) * worldSize, hipMemcpyHostToDevice));

  // Copy endpoints to GPU
  if (rdmaStates->commContext->RdmaTransportEnabled()) {
    HIP_RUNTIME_CHECK(
        hipMalloc(&gpuStates.rdmaEndpoints, sizeof(application::RdmaEndpoint) * worldSize));
    HIP_RUNTIME_CHECK(
        hipMemcpy(gpuStates.rdmaEndpoints, rdmaStates->commContext->GetRdmaEndpoints().data(),
                  sizeof(application::RdmaEndpoint) * worldSize, hipMemcpyHostToDevice));
  }

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

  // TODO: use in-node rank
  HIP_RUNTIME_CHECK(hipSetDevice(states->bootStates->rank));

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

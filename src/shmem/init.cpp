#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Intialization                                         */
/* ---------------------------------------------------------------------------------------------- */
void RdmaStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->rdmaStates = new RdmaStates();
  states->rdmaStates->context = new application::RdmaContext();
  // TODO: select device
  const application::RdmaDeviceList& devices = states->rdmaStates->context->GetRdmaDeviceList();
  states->rdmaStates->deviceContext = devices[0]->CreateRdmaDeviceContext();
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

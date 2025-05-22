#pragma once

#include <mpi.h>

#include "mori/application/application.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Initialization                                         */
/* ---------------------------------------------------------------------------------------------- */

int ShmemMpiInit(MPI_Comm);
int ShmemMpiFinalize();

int ShmemMyPe();
int ShmemNPes();

enum ShmemTeamType {
  INVALID = -1,
  WORLD = 0,
  SHARED = 1,
  TEAM_NODE = 2,
};

// TODO: finish team pe api
// int ShmemTeamMyPe(ShmemTeamType);
// int ShmemTeamNPes(ShmemTeamType);

/* ---------------------------------------------------------------------------------------------- */
/*                                        Symmetric Memory                                        */
/* ---------------------------------------------------------------------------------------------- */

void* ShmemMalloc(size_t size);
void* ShmemExtMallocWithFlags(size_t size, unsigned int flags);
void ShmemFree(void*);

// Note: temporary API for testing
application::SymmMemObjPtr ShmemQueryMemObjPtr(void*);

int ShmemBufferRegister(void* ptr, size_t size);
int ShmemBufferDeRegister(void* ptr, size_t size);

}  // namespace shmem
}  // namespace mori

#pragma once

#include <mpi.h>

#include "mori/application/application.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Initialization                                         */
/* ---------------------------------------------------------------------------------------------- */

int ShmemMpiInit(MPI_Comm);

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
void ShmemFree(void*);

int ShmemBufferRegister(void* ptr, size_t size);
int ShmemBufferUnRegister(void* ptr, size_t size);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */

__device__ void ShmemPutMemNbiThread(const application::SymmMemObj* dest,
                                     const application::MemoryRegion& source, size_t nelems,
                                     int pe);

__device__ void ShmemPutMemNbiWarp(const application::SymmMemObj* dest,
                                   const application::MemoryRegion& source, size_t nelems, int pe);

}  // namespace shmem
}  // namespace mori

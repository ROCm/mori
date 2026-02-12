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

// Full shmem API -- includes host-only APIs plus HIP/MPI-dependent APIs.
// Must be compiled with hipcc (or a HIP-aware compiler).

#pragma once

#include <mpi.h>

#include <string>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem_host_api.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*               Additional initialization APIs (require MPI / application / Torch)               */
/* ---------------------------------------------------------------------------------------------- */

// TODO: provide unified initialize / finalize APIs
int ShmemInit(application::BootstrapNetwork* bootNet);
int ShmemInit();  // Default initialization using MPI_COMM_WORLD
int ShmemMpiInit(MPI_Comm);
#ifdef MORI_ENABLE_TORCH
int ShmemTorchProcessGroupInit(const std::string& groupName);
#endif

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
/*                     APIs that depend on application types (require hipcc)                       */
/* ---------------------------------------------------------------------------------------------- */

// Note: temporary API for testing
application::SymmMemObjPtr ShmemQueryMemObjPtr(void*);

}  // namespace shmem
}  // namespace mori

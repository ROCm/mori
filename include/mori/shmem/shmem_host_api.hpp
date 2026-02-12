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

// Host-only shmem API -- safe to include from plain clang (no hipcc required).
// Does NOT pull in HIP runtime, MPI, or any device (__device__/__host__) code.
// For use by frameworks like XLA that compile host C++ with a standard compiler.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Initialization                                         */
/* ---------------------------------------------------------------------------------------------- */
#ifndef MORI_SHMEM_UNIQUE_ID_BYTES
#define MORI_SHMEM_UNIQUE_ID_BYTES 128
#endif
using mori_shmem_uniqueid_t = std::array<uint8_t, MORI_SHMEM_UNIQUE_ID_BYTES>;

struct mori_shmem_init_attr_t {
  int32_t rank;
  int32_t nranks;
  mori_shmem_uniqueid_t uid;
  void* mpi_comm;  // Optional MPI_Comm pointer
};

// Initialization flags
constexpr unsigned int MORI_SHMEM_INIT_WITH_MPI_COMM = 0;
constexpr unsigned int MORI_SHMEM_INIT_WITH_UNIQUEID = 1;

// UniqueId-based initialization APIs (nvshmem/rocshmem compatible)
int ShmemGetUniqueId(mori_shmem_uniqueid_t* uid);
int ShmemSetAttrUniqueIdArgs(int rank, int nranks, mori_shmem_uniqueid_t* uid,
                             mori_shmem_init_attr_t* attr);
int ShmemInitAttr(unsigned int flags, mori_shmem_init_attr_t* attr);

int ShmemFinalize();

int ShmemModuleInit(void* hipModule);

int ShmemMyPe();
int ShmemNPes();

void ShmemBarrierAll();

int ShmemNumQpPerPe();

/* ---------------------------------------------------------------------------------------------- */
/*                                        Symmetric Memory                                        */
/* ---------------------------------------------------------------------------------------------- */

void* ShmemMalloc(size_t size);
void* ShmemMallocAlign(size_t alignment, size_t size);
void* ShmemExtMallocWithFlags(size_t size, unsigned int flags);
void ShmemFree(void*);

int ShmemBufferRegister(void* ptr, size_t size);
int ShmemBufferDeregister(void* ptr, size_t size);

uint64_t ShmemPtrP2p(const uint64_t destPtr, const int myPe, int destPe);

}  // namespace shmem
}  // namespace mori

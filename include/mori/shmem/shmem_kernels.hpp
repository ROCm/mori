#pragma once

#include <assert.h>
#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
__device__ GpuStates* GetGlobalGpuStatesPtr() { return &globalGpuStates; }

template <core::ProviderType PrvdType>
__device__ void ShmemPutMemNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                     const application::MemoryRegion& source, size_t sourceOffset,
                                     size_t bytes, int pe) {
  if (bytes == 0) return;
  uintptr_t laddr = source.addr + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->epsStartAddr;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;

  int rank = globalGpuStates->rank;
  uint64_t dbrVal =
      core::PostWrite<PrvdType>(wq.sqAddr, wq.sqWqeNum, &wq.postIdx, ep[pe].handle.qpn, laddr,
                                source.lkey, raddr, rkey, bytes);
  core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, wq.postIdx);
  __threadfence_system();
  core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
  __threadfence_system();
}

template <core::ProviderType PrvdType>
__device__ void ShmemPutMemNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                     const application::SymmMemObjPtr source, size_t sourceOffset,
                                     size_t bytes, int pe) {
  int rank = GetGlobalGpuStatesPtr()->rank;
  ShmemPutMemNbiThread<PrvdType>(dest, destOffset, source->GetMemoryRegion(rank), sourceOffset,
                                 bytes, pe);
}

template <core::ProviderType PrvdType, typename T>
__device__ void ShmemPutTypeNbiThread(const application::SymmMemObjPtr dest, size_t destElmOffset,
                                      const application::MemoryRegion& source, size_t srcElmOffset,
                                      size_t nelems, int pe) {
  constexpr size_t typeSize = sizeof(T);
  ShmemPutMemNbiThread<PrvdType>(dest, destElmOffset * typeSize, source, srcElmOffset * typeSize,
                                 nelems * typeSize, pe);
}

template <core::ProviderType PrvdType, typename T>
__device__ void ShmemPutTypeNbiThread(const application::SymmMemObjPtr dest, size_t destElmOffset,
                                      const application::SymmMemObjPtr source, size_t srcElmOffset,
                                      size_t nelems, int pe) {
  int rank = GetGlobalGpuStatesPtr()->rank;
  ShmemPutTypeNbiThread<PrvdType, T>(dest, destElmOffset, source->GetMemoryRegion(rank),
                                     srcElmOffset, nelems, pe);
}

#define DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(TypeName, T)                                      \
  template <core::ProviderType PrvdType>                                                       \
  __device__ void ShmemPut##TypeName##NbiThread(                                               \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                             \
      const application::MemoryRegion& source, size_t srcElmOffset, size_t nelems, int pe) {   \
    ShmemPutTypeNbiThread<PrvdType, T>(dest, destElmOffset, source, srcElmOffset, nelems, pe); \
  }                                                                                            \
  template <core::ProviderType PrvdType>                                                       \
  __device__ void ShmemPut##TypeName##NbiThread(                                               \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                             \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe) {   \
    ShmemPutTypeNbiThread<PrvdType, T>(dest, destElmOffset, source, srcElmOffset, nelems, pe); \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint8, uint8_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint16, uint16_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint32, uint32_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint64, uint64_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Float, float)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Double, double)

// TODO: deal with bytes count limit
template <core::ProviderType PrvdType, uint32_t SIZE>
__device__ void ShmemPutSizeImmNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                         void* val, int pe) {
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->epsStartAddr;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;

  int rank = globalGpuStates->rank;
  uint64_t dbrVal = core::PostWriteInline<PrvdType>(wq.sqAddr, wq.sqWqeNum, &wq.postIdx,
                                                    ep[pe].handle.qpn, val, raddr, rkey, SIZE);
  core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, wq.postIdx);
  __threadfence_system();
  core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
  __threadfence_system();
}

template <core::ProviderType PrvdType, typename T>
__device__ void ShmemPutTypeImmNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                         T val, int pe) {
  static_assert(sizeof(T) <= core::MaxInlineDataSizePerWqe);
  ShmemPutSizeImmNbiThread<PrvdType, sizeof(T)>(dest, destOffset, &val, pe);
}

#define DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(TypeName, T)                                 \
  template <core::ProviderType PrvdType>                                                      \
  __device__ void ShmemPut##TypeName##ImmNbiThread(const application::SymmMemObjPtr dest,     \
                                                   size_t destOffset, uint32_t val, int pe) { \
    ShmemPutTypeImmNbiThread<PrvdType, uint32_t>(dest, destOffset, val, pe);                  \
  }

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint8, uint8_t)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint16, uint16_t)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint32, uint32_t)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint64, uint64_t)

__device__ void ShmemPutMemNbiWarp(const application::SymmMemObj* dest,
                                   const application::MemoryRegion& source, size_t nelems, int pe);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <core::ProviderType PrvdType>
__device__ void ShmemQuietThread() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->epsStartAddr;

  int rank = globalGpuStates->rank;
  int worldSize = globalGpuStates->worldSize;

  for (int i = 0; i < worldSize; i++) {
    // This assume we do not have endpoint to self
    if (i == rank) continue;

    application::CompletionQueueHandle& cq = ep[i].cqHandle;
    application::WorkQueueHandle& wq = ep[i].wqHandle;

    // Assume every wqe generates a cqe, so we can use work queue postIdx
    // TODO: 1 Should not use postIdx since 1 wqe can inc postIdx by > 1
    // TODO: 2 How to prevent cqe overflow?
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    while (true) {
      uint32_t consIdx = core::AtomicLoadSeqCst(&cq.consIdx);
      uint32_t postIdx = core::AtomicLoadSeqCst(&wq.postIdx);
      if ((consIdx + 1) >= postIdx) break;
      int opcode = core::PollCq<PrvdType>(cq.cqAddr, cq.cqeSize, cq.cqeNum, &cq.consIdx);
      if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
        printf("rank %d dest pe %d consIdx %d opcode %d\n", rank, i, consIdx, opcode);
        core::DumpWqe(wq.sqAddr, consIdx);
        assert(false);
      }
      core::UpdateCqDbrRecord<PrvdType>(cq.dbrRecAddr, consIdx);
    }
  }
}

template <core::ProviderType PrvdType, typename T>
__device__ void ShmemTypeWaitUntilGreaterThan(T* addr, T val) {
  while (core::AtomicLoadRelaxed(addr) <= val) {
  }
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(TypeName, T)            \
  template <core::ProviderType PrvdType>                                  \
  __device__ void Shmem##TypeName##WaitUntilGreaterThan(T* addr, T val) { \
    ShmemTypeWaitUntilGreaterThan<PrvdType, T>(addr, val);                \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint64, uint64_t)

template <core::ProviderType PrvdType, typename T>
__device__ void ShmemTypeWaitUntilEquals(T* addr, T val) {
  while (core::AtomicLoadRelaxed(addr) != val) {
  }
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(TypeName, T)              \
  template <core::ProviderType PrvdType>                             \
  __device__ void Shmem##TypeName##WaitUntilEquals(T* addr, T val) { \
    ShmemTypeWaitUntilEquals<PrvdType, T>(addr, val);                \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint64, uint64_t)

}  // namespace shmem
}  // namespace mori

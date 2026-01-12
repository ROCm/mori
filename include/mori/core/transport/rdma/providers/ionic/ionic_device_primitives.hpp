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
#pragma once

#include <hip/hip_runtime.h>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/rdma/providers/ionic/ionic_defs.hpp"
#include "mori/core/transport/rdma/providers/ionic/ionic_fw.h"
#include "mori/core/transport/rdma/providers/utils.h"
#include "mori/core/utils.hpp"

namespace mori {
namespace core {
// #ifdef ENABLE_IONIC
/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------- */
/*                                        Send / Recv APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t IonicPostSend(WorkQueueHandle& wq, uint32_t curPostIdx, bool cqeSignal,
                                         uint32_t qpn, uintptr_t laddr, uint64_t lkey,
                                         size_t bytes) {
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  int32_t size = (int32_t)bytes;
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  char* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  // to do: need to clear memory
  if ((wqeNum & curPostIdx) == 0) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }

  wqe->base.wqe_idx = curPostIdx;
  wqe->base.op = IONIC_V2_OP_SEND;
  wqe->base.num_sge_key = size ? 1 : 0;
  wqe->base.imm_data_key = HTOBE32(0);

  wqe->common.length = HTOBE32(size);
  wqe->common.pld.sgl[0].va = HTOBE64(reinterpret_cast<uint64_t>(laddr));
  wqe->common.pld.sgl[0].len = HTOBE32(size);
  wqe->common.pld.sgl[0].lkey = HTOBE32(lkey);

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if 0
  MORI_PRINTF("send, curPostIdx:%u, wqeIdx:%u, doorbell:0x%x\n", curPostIdx, curPostIdx, ((curPostIdx + 1) & (wqeNum - 1)));
#endif
  // return doorbell value
  return wq.sq_dbval | ((curPostIdx + 1) & (wqeNum - 1));
}

template <>
inline __device__ uint64_t PostSend<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t postIdx,
                                                       uint32_t curMsntblSlotIdx,
                                                       uint32_t curPsnIdx, bool cqeSignal,
                                                       uint32_t qpn, uintptr_t laddr, uint64_t lkey,
                                                       size_t bytes) {
  return IonicPostSend(wq, postIdx, cqeSignal, qpn, laddr, lkey, bytes);
}

template <>
inline __device__ uint64_t PostSend<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t qpn,
                                                       uintptr_t laddr, uint64_t lkey,
                                                       size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return IonicPostSend(wq, curPostIdx, true, qpn, laddr, lkey, bytes);
}

inline __device__ uint64_t IonicPostRecv(WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t qpn,
                                         uintptr_t laddr, uint64_t lkey, size_t bytes) {
  void* queueBuffAddr = wq.rqAddr;
  uint32_t wqeNum = wq.rqWqeNum;
  int32_t size = (int32_t)bytes;
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  char* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  wqe->base.wqe_idx = curPostIdx;
  wqe->base.num_sge_key = 1;
  wqe->base.imm_data_key = HTOBE32(0);

#if 0
  wqe->common.rdma.remote_va_high = HTOBE32(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->common.rdma.remote_va_low = HTOBE32(reinterpret_cast<uint64_t>(raddr));
  wqe->common.rdma.remote_rkey = HTOBE32(rkey);
#endif
  wqe->common.length = HTOBE32(size);
  wqe->common.pld.sgl[0].va = HTOBE64(reinterpret_cast<uint64_t>(laddr));
  wqe->common.pld.sgl[0].len = HTOBE32(size);
  wqe->common.pld.sgl[0].lkey = HTOBE32(lkey);
#if 0
  MORI_PRINTF("recv, curPostIdx:%u, wqeIdx:%u, doorbell:0x%x\n", curPostIdx, curPostIdx, ((curPostIdx + 1) & (wqeNum - 1)));
#endif
  // return doorbell value
  return wq.rq_dbval | ((curPostIdx + 1) & (wqeNum - 1));
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                       bool cqeSignal, uint32_t qpn,
                                                       uintptr_t laddr, uint64_t lkey,
                                                       size_t bytes) {
  return IonicPostRecv(wq, curPostIdx, qpn, laddr, lkey, bytes);
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t qpn,
                                                       uintptr_t laddr, uint64_t lkey,
                                                       size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return IonicPostRecv(wq, curPostIdx, qpn, laddr, lkey, bytes);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Read / Write APIs                                       */
/* ---------------------------------------------------------------------------------------------- */
// TODO: convert raddr/rkey laddr/lkey to big endien in advance to save cycles
inline __device__ uint64_t IonicPostReadWrite(WorkQueueHandle& wq, uint32_t curPostIdx,
                                              bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                              uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                              size_t bytes, bool isRead) {
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  int32_t size = (int32_t)bytes;
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  char* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  // MORI_PRINTF("IonicPostReadWrite, wqe:%p, curPostIdx:%d, wqeIdx:%d\n", wqe, curPostIdx, wqeIdx);
  // to do: need to clear memory
  if ((wqeNum & curPostIdx) == 0) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }

  wqe->base.wqe_idx = curPostIdx;
  wqe->base.op = isRead ? IONIC_V2_OP_RDMA_READ : IONIC_V2_OP_RDMA_WRITE;
  wqe->base.num_sge_key = size ? 1 : 0;
  wqe->base.imm_data_key = HTOBE32(0);

  wqe->common.rdma.remote_va_high = HTOBE32(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->common.rdma.remote_va_low = HTOBE32(reinterpret_cast<uint64_t>(raddr));
  wqe->common.rdma.remote_rkey = HTOBE32(rkey);

  wqe->common.length = HTOBE32(size);
  wqe->common.pld.sgl[0].va = HTOBE64(reinterpret_cast<uint64_t>(laddr));
  wqe->common.pld.sgl[0].len = HTOBE32(size);
  wqe->common.pld.sgl[0].lkey = HTOBE32(lkey);

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

#if 0
  MORI_PRINTF("dump wqe at addr:%p\n", wqeAddr);
  for (int i = 0; i < 64; i++) {
    MORI_PRINTF("%02x", (unsigned char)wqeAddr[i]);
    if ((i+1)%4 == 0)
      MORI_PRINTF("\n");
  }
#endif
#if 0
  MORI_PRINTF("Write, block:%u, warp:%u, lane:%u, wqe:%p, raddr:%p, rkey:%lu, len:%u, curPostIdx:%u, wqeIdx:%u, doorbell:0x%x\n",
         blockIdx.x, threadIdx.x/warpSize, __lane_id(),
	 wqe, raddr, rkey, size, curPostIdx, curPostIdx, ((curPostIdx + 1) & (wqeNum - 1)));
#endif
  //__threadfence_system();
  // asm volatile("" ::: "memory");
  // return doorbell value
  return wq.sq_dbval | ((curPostIdx + 1) & (wqeNum - 1));
}

template <>
inline __device__ uint64_t PostWrite<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                        uint32_t curMsntblSlotIdx,
                                                        uint32_t curPsnIdx, bool cqeSignal,
                                                        uint32_t qpn, uintptr_t laddr,
                                                        uint64_t lkey, uintptr_t raddr,
                                                        uint64_t rkey, size_t bytes) {
  return IonicPostReadWrite(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey, bytes, false);
}

template <>
inline __device__ uint64_t PostRead<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                       uint32_t curMsntblSlotIdx,
                                                       uint32_t curPsnIdx, bool cqeSignal,
                                                       uint32_t qpn, uintptr_t laddr, uint64_t lkey,
                                                       uintptr_t raddr, uint64_t rkey,
                                                       size_t bytes) {
  return IonicPostReadWrite(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey, bytes, true);
}

template <>
inline __device__ uint64_t PostWrite<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        uintptr_t raddr, uint64_t rkey,
                                                        size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return IonicPostReadWrite(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, bytes, false);
}

template <>
inline __device__ uint64_t PostRead<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t qpn,
                                                       uintptr_t laddr, uint64_t lkey,
                                                       uintptr_t raddr, uint64_t rkey,
                                                       size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return IonicPostReadWrite(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, bytes, true);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        WriteInline APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t IonicPostWriteInline(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                bool cqeSignal, uint32_t qpn, void* val,
                                                uintptr_t raddr, uint64_t rkey, size_t bytes) {
  assert(bytes <= MAX_INLINE_SIZE);
  assert(val);
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  int32_t size = (int32_t)bytes;
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  char* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  // to do: need to clear memory
  wqe_flags |= HTOBE16(IONIC_V1_FLAG_INL);
  if ((wqeNum & curPostIdx) == 0) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }

  wqe->base.wqe_idx = curPostIdx;
  wqe->base.op = IONIC_V2_OP_RDMA_WRITE;
  wqe->base.num_sge_key = 0;
  wqe->base.imm_data_key = HTOBE32(0);

  wqe->common.rdma.remote_va_high = HTOBE32(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->common.rdma.remote_va_low = HTOBE32(reinterpret_cast<uint64_t>(raddr));
  wqe->common.rdma.remote_rkey = HTOBE32(rkey);
  wqe->common.length = HTOBE32(size);
  memcpy(wqe->common.pld.data, val, size);

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#if 0
  MORI_PRINTF("write inline, block:%u, warp:%u, lane:%u, wqe:%p, raddr:%p, rkey:%lu, size:%u, curPostIdx:%u, wqeIdx:%u, doorbell:0x%x\n",
	 blockIdx.x, threadIdx.x/warpSize, __lane_id(),
         wqe, raddr, rkey, size, curPostIdx, curPostIdx, ((curPostIdx + 1) & (wqeNum - 1)));
#endif
  // asm volatile("" ::: "memory");
  // return doorbell value
  return wq.sq_dbval | ((curPostIdx + 1) & (wqeNum - 1));
}

template <>
inline __device__ uint64_t PostWriteInline<ProviderType::PSD>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, void* val, uintptr_t raddr, uint64_t rkey, size_t bytes) {
  return IonicPostWriteInline(wq, curPostIdx, cqeSignal, qpn, val, raddr, rkey, bytes);
}

template <>
inline __device__ uint64_t PostWriteInline<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t qpn,
                                                              void* val, uintptr_t raddr,
                                                              uint64_t rkey, size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  // MORI_PRINTF("PostWriteInline, val:%p\n", val);
  return IonicPostWriteInline(wq, curPostIdx, true, qpn, val, raddr, rkey, bytes);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Atomic APIs                                             */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t IonicPrepareAtomicWqe(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                 bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                                 uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                 void* val_1, void* val_2, uint32_t bytes,
                                                 atomicType amo_op) {
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  int32_t size = (int32_t)bytes;
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  char* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  if ((wqeNum & curPostIdx) == 0) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }

  uint64_t atomic_data = val_1 ? *static_cast<uint64_t*>(val_1) : 0;
  uint64_t atomic_cmp = val_2 ? *static_cast<uint64_t*>(val_2) : 0;
  uint32_t opcode;

  switch (amo_op) {
    case AMO_FETCH_INC:
    case AMO_INC: {
      opcode = IONIC_V2_OP_ATOMIC_FA;
      atomic_data = 1;
      break;
    }
    case AMO_FETCH_ADD:
    case AMO_SIGNAL_ADD:
    case AMO_ADD: {
      opcode = IONIC_V2_OP_ATOMIC_FA;
      break;
    }
    case AMO_FETCH: {
      opcode = IONIC_V2_OP_ATOMIC_FA;
      atomic_data = 0;
      break;
    }
    case AMO_COMPARE_SWAP: {
      opcode = IONIC_V2_OP_ATOMIC_CS;
      break;
    }
    default: {
      MORI_PRINTF("Error: unsupported atomic type (%d)\n", amo_op);
      assert(0);
    }
  }

  wqe->base.wqe_idx = curPostIdx;
  wqe->base.op = opcode;
  wqe->base.num_sge_key = 1;
  wqe->base.imm_data_key = HTOBE32(0);

  wqe->atomic_v2.remote_va_high = HTOBE32(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->atomic_v2.remote_va_low = HTOBE32(reinterpret_cast<uint64_t>(raddr));
  wqe->atomic_v2.remote_rkey = HTOBE32(rkey);
  wqe->atomic_v2.swap_add_high = HTOBE32(atomic_data >> 32);
  wqe->atomic_v2.swap_add_low = HTOBE32(atomic_data);
  wqe->atomic_v2.compare_high = HTOBE32(atomic_cmp >> 32);
  wqe->atomic_v2.compare_low = HTOBE32(atomic_cmp);

  wqe->atomic_v2.local_va = HTOBE64(reinterpret_cast<uint64_t>(laddr));
  wqe->atomic_v2.lkey = HTOBE32(lkey);

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

#if 0
  MORI_PRINTF("atomic,block:%u, warp:%u, lane:%u, wqe:%p, curPostIdx:%u, wqeIdx:%u, doorbell:0x%x\n",
	 blockIdx.x, threadIdx.x/warpSize, __lane_id(),
         wqe, curPostIdx, wqeIdx, ((curPostIdx + 1) & (wqeNum - 1)));
#endif
  // asm volatile("" ::: "memory");
  // return doorbell value
  return wq.sq_dbval | ((curPostIdx + 1) & (wqeNum - 1));
}

template <>
inline __device__ uint64_t PostAtomic<ProviderType::PSD>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr, uint64_t rkey,
    void* val_1, void* val_2, uint32_t typeBytes, atomicType amo_op) {
  return IonicPrepareAtomicWqe(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey, val_1,
                               val_2, typeBytes, amo_op);
}

template <>
inline __device__ uint64_t PostAtomic<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t qpn,
                                                         uintptr_t laddr, uint64_t lkey,
                                                         uintptr_t raddr, uint64_t rkey,
                                                         void* val_1, void* val_2,
                                                         uint32_t typeBytes, atomicType amo_op) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return IonicPrepareAtomicWqe(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, val_1, val_2,
                               typeBytes, amo_op);
}

#define DEFINE_IONIC_POST_ATOMIC_SPEC(TYPE)                                                     \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::PSD, TYPE>(                               \
      WorkQueueHandle & wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, \
      bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,            \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    return IonicPrepareAtomicWqe(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey,      \
                                 (void*)&val_1, (void*)&val_2, sizeof(TYPE), amo_op);           \
  }                                                                                             \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::PSD, TYPE>(                               \
      WorkQueueHandle & wq, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,      \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    uint32_t typeBytes = sizeof(TYPE);                                                          \
    uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);                                            \
    return IonicPrepareAtomicWqe(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey,           \
                                 (void*)&val_1, (void*)&val_2, typeBytes, amo_op);              \
  }

DEFINE_IONIC_POST_ATOMIC_SPEC(uint32_t)
DEFINE_IONIC_POST_ATOMIC_SPEC(uint64_t)
DEFINE_IONIC_POST_ATOMIC_SPEC(int32_t)
DEFINE_IONIC_POST_ATOMIC_SPEC(int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void UpdateSendDbrRecord<ProviderType::PSD>(void* dbrRecAddr, uint32_t wqeIdx) {
  return;
}

template <>
inline __device__ void UpdateRecvDbrRecord<ProviderType::PSD>(void* dbrRecAddr, uint32_t wqeIdx) {
  return;
}

template <>
inline __device__ void RingDoorbell<ProviderType::PSD>(void* dbrAddr, uint64_t dbrVal) {
#if 0
  MORI_PRINTF("really update sq doorbell, block:%u, warp:%u, lane:%u, sq/rq dbrAddr:%p, dbrVal:0x%lx\n",
         blockIdx.x, threadIdx.x/warpSize, __lane_id(), reinterpret_cast<uint64_t*>(dbrAddr), dbrVal);
#endif
  // asm volatile("" ::: "memory");
  core::AtomicStoreSeqCstSystem(reinterpret_cast<uint64_t*>(dbrAddr), dbrVal);
}

template <>
inline __device__ void UpdateDbrAndRingDbSend<ProviderType::PSD>(void* dbrRecAddr, uint32_t wqeIdx,
                                                                 void* dbrAddr, uint64_t dbrVal,
                                                                 uint32_t* lockVar) {
  AcquireLock(lockVar);

  UpdateSendDbrRecord<ProviderType::PSD>(dbrRecAddr, wqeIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::PSD>(dbrAddr, dbrVal);

  ReleaseLock(lockVar);
}

template <>
inline __device__ void UpdateDbrAndRingDbRecv<ProviderType::PSD>(void* dbrRecAddr, uint32_t wqeIdx,
                                                                 void* dbrAddr, uint64_t dbrVal,
                                                                 uint32_t* lockVar) {
  AcquireLock(lockVar);

  UpdateRecvDbrRecord<ProviderType::PSD>(dbrRecAddr, wqeIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::PSD>(dbrAddr, dbrVal);

  ReleaseLock(lockVar);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Completion Queue                                        */
/* ---------------------------------------------------------------------------------------------- */
#ifdef IONIC_CCQE
template <>
inline __device__ int PollCqOnce<ProviderType::PSD>(void* cqeAddr, uint32_t cqeNum,
                                                    uint32_t consIdx, uint32_t* wqeIdx) {
  volatile struct ionic_v1_cqe* cqe = reinterpret_cast<ionic_v1_cqe*>(cqeAddr);
  uint32_t old, msn = HTOBE32(cqe->send.msg_msn);

  MORI_PRINTF("ABH %s:%d here cons %#x msn %#x\n", __func__, __LINE__, consIdx, msn);
  while ((msn - consIdx) & 0x800000) {
    old = msn;
    msn = HTOBE32(cqe->send.msg_msn);
    if (msn != old) {
      MORI_PRINTF("ABH %s:%d here cons %#x msn %#x\n", __func__, __LINE__, consIdx, msn);
    }
  }
  MORI_PRINTF("ABH %s:%d here - msn %#x\n", __func__, __LINE__, msn);

  *wqeIdx = msn;

  return 0;
}
#else
template <>
inline __device__ int PollCqOnce<ProviderType::PSD>(void* cqeAddr, uint32_t cqeNum,
                                                    uint32_t consIdx, uint32_t* wqeIdx) {
  uint32_t cqeIdx = consIdx & (cqeNum - 1);
  char* Addr = reinterpret_cast<char*>(cqeAddr) + (cqeIdx * sizeof(struct ionic_v1_cqe));
  struct ionic_v1_cqe* cqe = reinterpret_cast<ionic_v1_cqe*>(Addr);

  MORI_PRINTF("ABH %s:%d consIdx:%u, cqeIdx:%u, cqeAddr:%p, qtf_be:0x%08x, cqe->status_length:%d\n",
         __func__, __LINE__, consIdx, cqeIdx, Addr, *(volatile uint32_t*)(&cqe->qid_type_flags),
         HTOBE32(cqe->status_length));
#if 1
  MORI_PRINTF("dump cqe at addr:%p\n", Addr);
  for (int i = 0; i < 32; i++) {
    MORI_PRINTF("%02x", (unsigned char)Addr[i]);
    if ((i + 1) % 4 == 0) MORI_PRINTF("\n");
  }
#endif
  /* Determine expected color based on cq wrap count */
  uint32_t qtf_color_bit = HTOBE32(IONIC_V1_CQE_COLOR);
  uint32_t qtf_color_exp = qtf_color_bit;
  if (cqeIdx & cqeNum) {
    qtf_color_exp = 0;
  }

  /* Check if my cqe color == expected color */
  // first round: 1 == 1, second round: 0 == 0
  uint32_t qtf_be = *(volatile uint32_t*)(&cqe->qid_type_flags);
  if ((qtf_be & qtf_color_bit) != qtf_color_exp) {
    MORI_PRINTF("cqe not ready\n");
    return -1;  // CQE just not ready yet, try again
  }

  uint32_t msn = HTOBE32(cqe->send.msg_msn);

  /* Report if the completion indicates an error. */
  if (!!(qtf_be & HTOBE32(IONIC_V1_CQE_ERROR))) {
    uint32_t qtf = HTOBE32(qtf_be);
    uint32_t qid = qtf >> IONIC_V1_CQE_QID_SHIFT;
    uint32_t type = (qtf >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
    uint32_t flag = qtf & 0xf;
    uint32_t status = cqe->status_length;
    uint64_t npg = cqe->send.npg_wqe_idx_timestamp & IONIC_V1_CQE_WQE_IDX_MASK;
    MORI_PRINTF("QUIET ERROR: qid %u type %u flag %#x status %u msn %u npg %lu\n", qid, type, flag,
           status, msn, npg);
    return HTOBE32(cqe->status_length);
  }

  MORI_PRINTF("poll cqe one, success\n");

  return 0;
}
#endif
template <>
inline __device__ int PollCq<ProviderType::PSD>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx) {
  const uint32_t curConsIdx = atomicAdd(consIdx, 1);
  int err = -1;

  // ABH: polls until each thread sees a ready cqe
  //   (what if not all threads see a ready cqe?)
  do {
    err = PollCqOnce<ProviderType::PSD>(cqAddr, cqeNum, curConsIdx, nullptr);
    // TODO: Explain clearly why adding a compiler barrier fix hang issue
    asm volatile("" ::: "memory");
  } while (err < 0);

  // Handle error cases
  if (err) {
    auto error = IonicHandleErrorCqe(err);
    MORI_PRINTF("[IONIC PollCq] CQE error: %s (opcode: %d) at %s:%d\n", IbvWcStatusString(error), err,
           __FILE__, __LINE__);
    return err;
  }

  return 0;
}

template <>
inline __device__ int PollCq<ProviderType::PSD>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx,
                                                uint32_t* wqeCounter) {
  const uint32_t curConsIdx = *consIdx;
  const uint32_t cqeIdx = curConsIdx & (cqeNum - 1);

  // Get CQE pointer
  char* cqeAddr = reinterpret_cast<char*>(cqAddr) + (cqeIdx * sizeof(struct ionic_v1_cqe));
  struct ionic_v1_cqe* cqe = reinterpret_cast<ionic_v1_cqe*>(cqeAddr);

  // Check color bit to determine if CQE is ready
  constexpr uint32_t colorBit = IONIC_V1_CQE_COLOR;
  const uint32_t expectedColor = (curConsIdx & cqeNum) ? 0 : colorBit;
  const uint32_t qtfBe = BE32TOH(*(volatile uint32_t*)(&cqe->qid_type_flags));

  if ((qtfBe & colorBit) != expectedColor) {
    return -1;  // CQE not ready yet, try again
  }

  // Check for errors
  if (qtfBe & IONIC_V1_CQE_ERROR) {
    const uint32_t qid = qtfBe >> IONIC_V1_CQE_QID_SHIFT;
    const uint32_t type = (qtfBe >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
    const uint32_t flags = qtfBe & 0xf;
    const uint32_t status = BE32TOH(cqe->status_length);
    const uint64_t npg = cqe->send.npg_wqe_idx_timestamp & IONIC_V1_CQE_WQE_IDX_MASK;
    const uint32_t msn = BE32TOH(cqe->send.msg_msn) & 0xFFFF;
    const uint8_t error = IonicHandleErrorCqe(status);

    // MORI_PRINTF(
    //     "PollCqOnce2, QUIET ERROR: block:%u, warp:%u, lane:%u, cqeAddr:%p, error:%u "
    //     "qid %u type %u flag %#x status 0x%08x msn %u npg %lu\n",
    //     blockIdx.x, threadIdx.x / warpSize, __lane_id(), cqeAddr, error, qid, type, flags, status,
    //     msn, npg);

#if 0
    // Debug: dump raw CQE contents
    MORI_PRINTF("dump cqe at addr:%p\n", cqeAddr);
    for (int i = 0; i < 32; i++) {
      MORI_PRINTF("%02x", static_cast<unsigned char>(cqeAddr[i]));
      if ((i + 1) % 4 == 0) {
        MORI_PRINTF("\n");
      }
    }
#endif

    return error;
  }

  *wqeCounter = BE32TOH(cqe->send.msg_msn);
  return 0;
}


#ifdef IONIC_CCQE
inline __device__ int PollCqOnce2(WorkQueueHandle& wqHandle, CompletionQueueHandle& cqHandle,
                                  uint64_t activemask, void* cqeAddr, uint32_t cqeNum,
                                  uint32_t consIdx) {
  volatile struct ionic_v1_cqe* cqe = reinterpret_cast<ionic_v1_cqe*>(cqeAddr);
  uint32_t old, msn = HTOBE32(cqe->send.msg_msn);

  consIdx = wqHandle.dbTouchIdx;

  // MORI_PRINTF("ABH %s:%d here cons %#x msn %#x\n", __func__, __LINE__, consIdx, msn);
  while ((msn - consIdx) & 0x800000) {
    old = msn;
    msn = HTOBE32(cqe->send.msg_msn);
    if (msn != old) {
      // MORI_PRINTF("ABH %s:%d here cons %#x msn %#x\n", __func__, __LINE__, consIdx, msn);
    }
  }

  wqHandle.doneIdx = msn;
  return 0;
}
#else
inline __device__ int PollCqOnce2(WorkQueueHandle& wqHandle, CompletionQueueHandle& cqHandle,
                                  uint64_t activemask, void* cqeAddr, uint32_t cqeNum,
                                  uint32_t consIdx) {
  uint32_t my_logical_lane_id = get_active_lane_num(activemask);
  uint32_t my_cq_pos = cqHandle.cq_consumer + my_logical_lane_id;

  uint32_t cqeIdx = my_cq_pos & (cqeNum - 1);
  char* Addr = reinterpret_cast<char*>(cqeAddr) + (cqeIdx * sizeof(struct ionic_v1_cqe));

  struct ionic_v1_cqe* cqe = reinterpret_cast<ionic_v1_cqe*>(Addr);
#if 0
  MORI_PRINTF("PollCqOnce2, block:%u, warp:%u, lane:%u, consIdx:%u, cqeIdx:%u, cqeAddr:%p, qtf_be:0x%08x, cqe->status_length:%d, msn:%u\n",
	 blockIdx.x, threadIdx.x/warpSize, __lane_id(), my_cq_pos, cqeIdx, Addr,
	 *(volatile uint32_t *)(&cqe->qid_type_flags), BE32TOH(cqe->status_length), BE32TOH(cqe->send.msg_msn));
#endif
#if 0
  MORI_PRINTF("dump cqe at addr:%p\n", Addr);
  for (int i = 0; i < 32; i++) {
    MORI_PRINTF("%02x", (unsigned char)Addr[i]);
    if ((i+1)%4 == 0)
      MORI_PRINTF("\n");
  }
#endif
  /* Determine expected color based on cq wrap count */
  uint32_t qtf_color_bit = IONIC_V1_CQE_COLOR;
  uint32_t qtf_color_exp = qtf_color_bit;
  if (my_cq_pos & cqeNum) {
    qtf_color_exp = 0;
  }

  /* Check if my cqe color == expected color */
  // first round: 1 == 1, second round: 0 == 0
  uint32_t qtf_be = BE32TOH(*(volatile uint32_t*)(&cqe->qid_type_flags));
  if ((qtf_be & qtf_color_bit) != qtf_color_exp) {
#if 0
    MORI_PRINTF("PollCqOnce2, not ready, block:%u, warp:%u, lane:%u, consIdx:%u, cqeIdx:%u, cqeAddr:%p, qtf_be:0x%08x, cqe->status_length:0x%08x, msn:%u\n",
           blockIdx.x, threadIdx.x/warpSize, __lane_id(), my_cq_pos, cqeIdx, Addr,
           *(volatile uint32_t *)(&cqe->qid_type_flags), BE32TOH(cqe->status_length), BE32TOH(cqe->send.msg_msn));
#endif
    return 0;  // CQE just not ready yet, try again
  }

  uint32_t msn = BE32TOH(cqe->send.msg_msn);

  /* Report if the completion indicates an error. */
  if (!!(qtf_be & IONIC_V1_CQE_ERROR)) {
    uint32_t qtf = qtf_be;
    uint32_t qid = qtf >> IONIC_V1_CQE_QID_SHIFT;
    uint32_t type = (qtf >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
    uint32_t flag = qtf & 0xf;
    uint32_t status = cqe->status_length;
    uint64_t npg = cqe->send.npg_wqe_idx_timestamp & IONIC_V1_CQE_WQE_IDX_MASK;
    uint8_t error = IonicHandleErrorCqe(BE32TOH(cqe->status_length));
    MORI_PRINTF(
        "PollCqOnce2, QUIET ERROR: block:%u, warp:%u, lane:%u, cqeAddr:%p, error:%u qid %u type %u "
        "flag %#x status 0x%08x msn %u npg %lu\n",
        blockIdx.x, threadIdx.x / warpSize, __lane_id(), Addr, error, qid, type, flag, status, msn,
        npg);
#if 1
    MORI_PRINTF("dump cqe at addr:%p\n", Addr);
    for (int i = 0; i < 32; i++) {
      MORI_PRINTF("%02x", (unsigned char)Addr[i]);
      if ((i + 1) % 4 == 0) MORI_PRINTF("\n");
    }
#endif
    /* No other way to signal an error, so just crash. */
    // abort();
    return error;
  }

#if 0
  MORI_PRINTF("PollCqOnce2, success, block:%u, warp:%u, lane:%u, qp:%u, cqeAddr:%p, my_cq_pos:%u, cqeNum:%u, msn:%u\n",
          blockIdx.x, threadIdx.x/warpSize, __lane_id(),
	  qtf_be >> IONIC_V1_CQE_QID_SHIFT, Addr, my_cq_pos, cqHandle.cqeNum, msn);
#endif
  /* Only proceed with the furthest ahead cqe to update the sq state */
  uint64_t my_lane_mask = 1ull << __lane_id();
  uint64_t lesser_lane_mask = my_lane_mask - 1;
  if (my_lane_mask != (__ballot(true) & activemask & ~lesser_lane_mask)) {
    return 0;
  }

  /* update position in the cq */
  cqHandle.cq_consumer = my_cq_pos + 1;

  /*
   * Ring cq doorbell frequently enough to avoid cq full.
   *
   * NB: IONIC_CQ_GRACE is 100
   */
  if (((cqHandle.cq_consumer - cqHandle.cq_dbpos) & (cqHandle.cqeNum - 1)) >= 100) {
    cqHandle.cq_dbpos = cqHandle.cq_consumer;
    uint64_t dbrVal = cqHandle.cq_dbval | ((cqHandle.cqeNum - 1) & (cqHandle.cq_dbpos));
#if 0
    MORI_PRINTF("update cq doorbell, block:%u, warp:%u, lane:%u, cq dbrAddr:%p, dbrVal:0x%lx, cq_consumer:%u\n",
           blockIdx.x, threadIdx.x/warpSize, __lane_id(), reinterpret_cast<uint64_t*>(cqHandle.dbrRecAddr), dbrVal, cqHandle.cq_consumer);
#endif
    __atomic_store_n(reinterpret_cast<uint64_t*>(cqHandle.dbrRecAddr), dbrVal,
                     __ATOMIC_SEQ_CST);  // TODO:maybe relaxed?
  }

  wqHandle.doneIdx = msn;
  return 0;
}
#endif

#ifdef IONIC_CCQE
template <>
inline __device__ int PollCq<ProviderType::PSD>(WorkQueueHandle& wqHandle,
                                                CompletionQueueHandle& cqHandle, void* cqAddr,
                                                uint32_t cqeNum, uint32_t* consIdx,
                                                uint16_t* wqeCounter) {
  PollCqOnce2(wqHandle, cqHandle, 1, cqAddr, cqeNum, *consIdx);
  *wqeCounter = *consIdx;
  return 0;
}
#else
template <>
inline __device__ int PollCq<ProviderType::PSD>(WorkQueueHandle& wqHandle,
                                                CompletionQueueHandle& cqHandle, void* cqAddr,
                                                uint32_t cqeNum, uint32_t* consIdx,
                                                uint16_t* wqeCounter) {
  uint32_t greed = 10;
  const uint32_t curConsIdx = *consIdx;
  uint64_t activemask = GetActiveLaneMask();
  uint32_t cons = wqHandle.dbTouchIdx;
  int err;
  /* wait for sq_msn to catch up or pass cons. */
  /* 0x800000 - sign bit for 24-bit fields     */
  while ((wqHandle.doneIdx - cons) & 0x800000) {
    if (!spin_lock_try_acquire_shared(&cqHandle.pollCqLock, activemask)) {
      continue;
    }

    /* with lock acquired, this wave polls cqes until caught up */
    while ((wqHandle.doneIdx - cons) & 0x800000) {
      uint32_t old_sq_msn = wqHandle.doneIdx;
      // MORI_PRINTF("PollCq, before PollCqOnce2, curConsIdx:%u\n", curConsIdx);
      // asm volatile("" ::: "memory");
      err = PollCqOnce2(wqHandle, cqHandle, activemask, cqAddr, cqeNum, curConsIdx);
      if (err != 0) {
        MORI_PRINTF("PollCq, PollCqOnce2 failed, err:%u\n", err);
        return err;
      }
      asm volatile("" ::: "memory");
      // MORI_PRINTF("PollCq, after PollCqOnce2, curConsIdx:%u\n", curConsIdx);
      if (!((wqHandle.doneIdx - cons) & 0x800000)) {
        if (wqHandle.doneIdx == old_sq_msn) {
          break;
        }
        if (!greed) {
          break;
        }
        --greed;
      }
    }

    spin_lock_release_shared(&cqHandle.pollCqLock, activemask);
    break;
  }

  return 0;
}
#endif

template <>
inline __device__ void UpdateCqDbrRecord<ProviderType::PSD>(CompletionQueueHandle& cq,
                                                            uint32_t consIdx) {
#if 1
  uint64_t dbrVal = cq.cq_dbval | ((cq.cqeNum - 1) & consIdx);  // don't add 1 to consIdx
  __atomic_store_n(reinterpret_cast<uint64_t*>(cq.dbrRecAddr), dbrVal,
                     __ATOMIC_SEQ_CST); 
  // MORI_PRINTF("UpdateCqDbrRecord, dbrRecAddr:%p, dbrVal:%#lx\n",
  // reinterpret_cast<uint64_t*>(cq.dbrRecAddr), dbrVal);
  return;
#endif
}

template <>
inline __device__ int PollCqAndUpdateDbr<ProviderType::PSD>(CompletionQueueHandle& cq,
                                                            uint32_t* consIdx, uint32_t* lockVar) {
  AcquireLock(lockVar);

  int err = PollCq<ProviderType::PSD>(cq.cqAddr, cq.cqeNum, consIdx);
  if (err >= 0) {
    UpdateCqDbrRecord<ProviderType::PSD>(cq, *consIdx);
  }

  ReleaseLock(lockVar);
  return err;
}
// #endif
}  // namespace core
}  // namespace mori

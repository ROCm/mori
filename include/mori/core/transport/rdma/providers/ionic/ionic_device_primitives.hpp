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

#include "mori/core/transport/rdma/providers/ionic/ionic_fw.h"
#include "mori/application/transport/rdma/rdma.hpp"

#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/rdma/providers/ionic/ionic_defs.hpp"
#include "mori/core/transport/rdma/providers/utils.h"
#include "mori/core/utils.hpp"

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------- */
/*                                        Send / Recv APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t IonicPostSend(WorkQueueHandle& wq, uint32_t curPostIdx, bool cqeSignal,
                                         uint32_t qpn, uintptr_t laddr, uint64_t lkey, size_t bytes) {
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  int32_t size = (int32_t)bytes;
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  char* wqeAddr = reinterpret_cast<char *>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  //to do: need to clear memory
  if (wq.color) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }

  wqe->base.wqe_idx = wqeIdx;
  wqe->base.op = IONIC_V2_OP_SEND;
  wqe->base.num_sge_key = size ? 1 : 0;
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

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  printf("curPostIdx:%u, wqeIdx:%u, doorbell:%lu", curPostIdx, wqeIdx, wq.sq_dbval | wqeIdx);
  
  //return doorbell value
  return wq.sq_dbval | wqeIdx;
}

template <>
inline __device__ uint64_t PostSend<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t postIdx,
                                                        uint32_t curMsntblSlotIdx,
                                                        uint32_t curPsnIdx, bool cqeSignal,
                                                        uint32_t qpn, uintptr_t laddr,
                                                        uint64_t lkey, size_t bytes) {
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
  char* wqeAddr = reinterpret_cast<char *>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  wqe->base.wqe_idx = wqeIdx;
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

  printf("curPostIdx:%u, wqeIdx:%u, doorbell:%lu", curPostIdx, wqeIdx, wq.rq_dbval | wqeIdx);
  
  //return doorbell value
  return wq.rq_dbval | wqeIdx;
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                       bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                                       uint64_t lkey, size_t bytes) {
  return IonicPostRecv(wq, curPostIdx, qpn, laddr, lkey, bytes);
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::PSD>(WorkQueueHandle& wq, uint32_t qpn,
                                                       uintptr_t laddr, uint64_t lkey, size_t bytes) {
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
  char* wqeAddr = reinterpret_cast<char *>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  //to do: need to clear memory
  if (wq.color) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }

  wqe->base.wqe_idx = wqeIdx;
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

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  printf("curPostIdx:%u, wqeIdx:%u, doorbell:%lu", curPostIdx, wqeIdx, wq.sq_dbval | wqeIdx);
  
  //return doorbell value
  return wq.sq_dbval | wqeIdx;
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
                                                       uint32_t qpn, uintptr_t laddr,
                                                       uint64_t lkey, uintptr_t raddr,
                                                       uint64_t rkey, size_t bytes) {
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
  assert(!val);
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  int32_t size = (int32_t)bytes;
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  char* wqeAddr = reinterpret_cast<char *>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  //to do: need to clear memory
  wqe_flags |= HTOBE16(IONIC_V1_FLAG_INL);
  if (wq.color) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }

  wqe->base.wqe_idx = wqeIdx;
  wqe->base.op = IONIC_V2_OP_RDMA_WRITE;
  wqe->base.num_sge_key = 0;
  wqe->base.imm_data_key = HTOBE32(0);
  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  wqe->common.rdma.remote_va_high = HTOBE32(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->common.rdma.remote_va_low = HTOBE32(reinterpret_cast<uint64_t>(raddr));
  wqe->common.rdma.remote_rkey = HTOBE32(rkey);
  
  wqe->common.length = HTOBE32(size);
  memcpy(wqe->common.pld.data, val, size);

  printf("curPostIdx:%u, wqeIdx:%u, doorbell:%lu", curPostIdx, wqeIdx, wq.sq_dbval | wqeIdx);
  
  //return doorbell value
  return wq.sq_dbval | wqeIdx;
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
  char* wqeAddr = reinterpret_cast<char *>(queueBuffAddr) + (wqeIdx * sizeof(struct ionic_v1_wqe));
  struct ionic_v1_wqe* wqe = reinterpret_cast<ionic_v1_wqe*>(wqeAddr);
  uint16_t wqe_flags = 0;

  if (wq.color) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_COLOR);
  }

  if (cqeSignal) {
    wqe_flags |= HTOBE16(IONIC_V1_FLAG_SIG);
  }
  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

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
      printf("Error: unsupported atomic type (%d)\n", amo_op);
      assert(0);
    }
  }

  wqe->base.wqe_idx = wqeIdx;
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
  printf("curPostIdx:%u, wqeIdx:%u, doorbell:%lu", curPostIdx, wqeIdx, wq.sq_dbval | wqeIdx);
	
  //return doorbell value
  return wq.sq_dbval | wqeIdx;
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

#define DEFINE_IONIC_POST_ATOMIC_SPEC(TYPE)                                                      \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::PSD, TYPE>(                              \
      WorkQueueHandle & wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, \
      bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,            \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    return IonicPrepareAtomicWqe(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey,       \
                                 (void*)&val_1, (void*)&val_2, sizeof(TYPE), amo_op);            \
  }                                                                                             \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::PSD, TYPE>(                              \
      WorkQueueHandle & wq, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,      \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    uint32_t typeBytes = sizeof(TYPE);                                                          \
    uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);                                \
    return IonicPrepareAtomicWqe(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey,            \
                                 (void*)&val_1, (void*)&val_2, typeBytes, amo_op);               \
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
template <>
inline __device__ int PollCqOnce<ProviderType::PSD>(void* cqeAddr, uint32_t cqeNum,
                                                    uint32_t consIdx, uint32_t* wqeIdx) {
  uint32_t cqeIdx = consIdx & (cqeNum - 1);
  char* Addr = reinterpret_cast<char *>(cqeAddr) + (cqeIdx * sizeof(struct ionic_v1_cqe));
  struct ionic_v1_cqe* cqe = reinterpret_cast<ionic_v1_cqe*>(Addr);
                                                    
  printf("consIdx:%u, cqeIdx:%u, cqeAddr:%p", consIdx, cqeIdx, cqeAddr);

  /* Determine expected color based on cq wrap count */
  uint32_t qtf_color_bit = HTOBE32(IONIC_V1_CQE_COLOR);
  uint32_t qtf_color_exp = qtf_color_bit;
  if (cqeIdx & cqeNum) {
    qtf_color_exp = 0;
  }
  
  /* Check if my cqe color == expected color */
  // first round: 1 == 1, second round: 0 == 0
  uint32_t qtf_be = *(volatile uint32_t *)(&cqe->qid_type_flags);
  if ((qtf_be & qtf_color_bit) != qtf_color_exp) {
  	printf("cqe not ready\n");
    return -1;// CQE just not ready yet, try again
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
  
    printf("QUIET ERROR: qid %u type %u flag %#x status %u msn %u npg %lu\n",
           qid, type, flag, status, msn, npg);
  }

  int32_t err = HTOBE32(cqe->status_length);
  printf("poll cqe once, err:%d\n", err);
  
  return err;
}
 
template <>
inline __device__ int PollCq<ProviderType::PSD>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx) {
  const uint32_t curConsIdx = atomicAdd(consIdx, 1);
  int err = -1;
  
  do {
	 err = PollCqOnce<ProviderType::PSD>(cqAddr, cqeNum, curConsIdx, nullptr);
	 // TODO: Explain clearly why adding a compiler barrier fix hang issue
	 asm volatile("" ::: "memory");
  } while (err < 0);

  // Handle error cases
  if (err) {
	 auto error = IonicHandleErrorCqe(err);
	 printf("[BNXT PollCq] CQE error: %s (opcode: %d) at %s:%d\n", IbvWcStatusString(error), err,
			__FILE__, __LINE__);
	 return err;
  }

  return 0;
}

template <>
inline __device__ int PollCq<ProviderType::PSD>(void* cqAddr, uint32_t cqeNum, 
                                                uint32_t* consIdx, uint16_t* wqeCounter) {
  const uint32_t curConsIdx = *consIdx;
  int err = -1;
  uint32_t wqeIdx;
  do {
	 err = PollCqOnce<ProviderType::PSD>(cqAddr, cqeNum, curConsIdx, &wqeIdx);
	 asm volatile("" ::: "memory");
  } while (err < 0);

  //to do: check 16bit is enough
  *wqeCounter = (uint16_t)(wqeIdx & 0xFFFF);
  
  if (err) {
	 auto error = IonicHandleErrorCqe(err);
	 printf("[IONIC PollCq] CQE error: %s (opcode: %d), wqeCounter: %u at %s:%d\n",
			IbvWcStatusString(error), err, *wqeCounter, __FILE__, __LINE__);
	 return err;
  }

  return 0;
}

template <>
inline __device__ void UpdateCqDbrRecord<ProviderType::PSD>(CompletionQueueHandle& cq, void* dbrRecAddr, uint32_t cons_idx,
                                                            uint32_t cqeNum) {
  uint64_t dbrVal = cq.cq_dbval | ((cq.cqeNum - 1) & (cons_idx + 1));                                                         
  core::AtomicStoreSeqCstSystem(reinterpret_cast<uint64_t*>(dbrRecAddr), dbrVal);
  return;
}

template <>
inline __device__ int PollCqAndUpdateDbr<ProviderType::PSD>(CompletionQueueHandle& cq, void* cqAddr, uint32_t cqeSize,
                                                            uint32_t cqeNum, uint32_t* consIdx,
                                                            void* dbrRecAddr, uint32_t* lockVar) {
  AcquireLock(lockVar);

  int err = PollCq<ProviderType::PSD>(cqAddr, cqeNum, consIdx);
  if (err >= 0) {
    UpdateCqDbrRecord<ProviderType::PSD>(cq, dbrRecAddr, *consIdx, cqeNum);
  }

  ReleaseLock(lockVar);
  return err;
}

}  // namespace core
}  // namespace mori

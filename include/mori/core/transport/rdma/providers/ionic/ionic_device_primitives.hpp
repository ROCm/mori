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
template <bool IsRead>
inline __device__ uint64_t IonicPostReadWriteImpl(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                  bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                                  uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                  size_t bytes) {
  constexpr uint8_t opcode = IsRead ? IONIC_V2_OP_RDMA_READ : IONIC_V2_OP_RDMA_WRITE;
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

  wqe->base.wqe_idx = curPostIdx;
  wqe->base.op = opcode;
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

  return wq.sq_dbval | ((curPostIdx + 1) & (wqeNum - 1));
}

template <>
inline __device__ uint64_t PostReadWrite<ProviderType::PSD, false>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr, uint64_t rkey,
    size_t bytes) {
  return IonicPostReadWriteImpl<false>(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey,
                                       bytes);
}

template <>
inline __device__ uint64_t PostReadWrite<ProviderType::PSD, true>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr, uint64_t rkey,
    size_t bytes) {
  return IonicPostReadWriteImpl<true>(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey,
                                      bytes);
}

template <>
inline __device__ uint64_t PostReadWrite<ProviderType::PSD, false>(WorkQueueHandle& wq,
                                                                   uint32_t qpn, uintptr_t laddr,
                                                                   uint64_t lkey, uintptr_t raddr,
                                                                   uint64_t rkey, size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return IonicPostReadWriteImpl<false>(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, bytes);
}

template <>
inline __device__ uint64_t PostReadWrite<ProviderType::PSD, true>(WorkQueueHandle& wq, uint32_t qpn,
                                                                  uintptr_t laddr, uint64_t lkey,
                                                                  uintptr_t raddr, uint64_t rkey,
                                                                  size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return IonicPostReadWriteImpl<true>(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, bytes);
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
inline __device__ int PollCq<ProviderType::PSD>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx,
                                                uint32_t* wqeCounter) {
  const uint32_t curConsIdx = *consIdx;

  volatile struct ionic_v1_cqe* cqe = reinterpret_cast<volatile ionic_v1_cqe*>(cqAddr);
  const uint32_t msn = BE32TOH(*(volatile uint32_t*)(&cqe->send.msg_msn));
  if ((msn - (curConsIdx + 1)) & 0x800000) {
    return -1;  // firmware hasn't produced enough completions yet
  }

  *wqeCounter = msn;
  return 0;
}

#else

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

    return error;
  }

  *wqeCounter = BE32TOH(cqe->send.msg_msn);
  return 0;
}
#endif  // end of PollCq<ProviderType::PSD>

template <>
inline __device__ void UpdateCqDbrRecord<ProviderType::PSD>(CompletionQueueHandle& cq,
                                                            uint32_t consIdx) {
#if 1
  uint64_t dbrVal = cq.cq_dbval | ((cq.cqeNum - 1) & consIdx);  // don't add 1 to consIdx
  __atomic_store_n(reinterpret_cast<uint64_t*>(cq.dbrRecAddr), dbrVal, __ATOMIC_SEQ_CST);
  // MORI_PRINTF("UpdateCqDbrRecord, dbrRecAddr:%p, dbrVal:%#lx\n",
  // reinterpret_cast<uint64_t*>(cq.dbrRecAddr), dbrVal);
  return;
#endif
}

// #endif
}  // namespace core
}  // namespace mori

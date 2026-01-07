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

#include "mori/core/transport/p2p/device_primitives.hpp"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/rdma/providers/bnxt/bnxt_defs.hpp"
#include "mori/core/transport/rdma/providers/utils.h"
#include "mori/core/utils.hpp"
#ifdef ENABLE_BNXT
extern "C" {
#include <infiniband/bnxt_re_dv.h>
#include <infiniband/bnxt_re_hsi.h>
}  // ENABLE_BNXT
#else
extern "C" {
#include "mori/core/transport/rdma/providers/bnxt/bnxt_re_dv.h"
#include "mori/core/transport/rdma/providers/bnxt/bnxt_re_hsi.h"
}
#endif

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                         DB Header                                              */
/* ---------------------------------------------------------------------------------------------- */
// struct bnxt_re_db_hdr {
// 	__u64 typ_qid_indx; /* typ: 4, qid:20, indx:24 */
// };
inline __device__ uint64_t bnxt_re_init_db_hdr(int32_t indx, uint32_t toggle, uint32_t qid,
                                               uint32_t typ) {
  uint64_t key_lo = indx | toggle;

  uint64_t key_hi = (static_cast<uint64_t>(qid) & BNXT_RE_DB_QID_MASK);
  key_hi |= (static_cast<uint64_t>(typ) & BNXT_RE_DB_TYP_MASK) << BNXT_RE_DB_TYP_SHIFT;
  key_hi |= 0x1UL << BNXT_RE_DB_VALID_SHIFT;

  return key_lo | (key_hi << 32);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Fill MSN Table                                           */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void atomic_add_packed_msn_and_psn(uint64_t* msnPack, uint32_t incSlot,
                                                     uint32_t incPsn, uint32_t* oldSlot,
                                                     uint32_t* oldPsn) {
  uint64_t expected = __hip_atomic_load(msnPack, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  while (true) {
    uint32_t curSlot = static_cast<uint32_t>(expected & 0xFFFFFFFF);
    uint32_t curPsn = static_cast<uint32_t>((expected >> 32) & 0xFFFFFFFF);

    uint32_t newSlot = curSlot + incSlot;
    uint32_t newPsn = curPsn + incPsn;

    uint64_t desired = (static_cast<uint64_t>(newPsn) << 32) | static_cast<uint64_t>(newSlot);

    if (__hip_atomic_compare_exchange_strong(msnPack, &expected, desired, __ATOMIC_RELAXED,
                                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
      if (oldSlot) *oldSlot = curSlot;
      if (oldPsn) *oldPsn = curPsn;
      break;
    }
  }
}

inline __device__ uint64_t bnxt_re_update_msn_tbl(uint32_t st_idx, uint32_t npsn,
                                                  uint32_t start_psn) {
  return ((((uint64_t)(st_idx) << BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT) &
           BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK) |
          (((uint64_t)(npsn) << BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT) &
           BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK) |
          (((start_psn) << BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT) &
           BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK));
}

inline __device__ void bnxt_re_fill_psns_for_msntbl(void* msnBuffAddr, uint32_t postIdx,
                                                    uint32_t curPsnIdx, uint32_t psnCnt,
                                                    uint32_t msntblIdx) {
  uint32_t nextPsn = curPsnIdx + psnCnt;
  struct bnxt_re_msns msns;
  msns.start_idx_next_psn_start_psn = 0;

  uint64_t* msns_ptr;
  // Get the MSN table address
  msns_ptr = (uint64_t*)(((char*)msnBuffAddr) + ((msntblIdx) << 3));

  msns.start_idx_next_psn_start_psn |= bnxt_re_update_msn_tbl(postIdx, nextPsn, curPsnIdx);

  *msns_ptr = *((uint64_t*)&msns);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------- */
/*                                        Send / Recv APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t BnxtPostSend(WorkQueueHandle& wq, uint32_t curPostIdx,
                                        uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                        bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                        uint64_t lkey, size_t bytes) {
  // In bnxt, wqeNum mean total sq slot num
  uint8_t signalFlag = cqeSignal ? BNXT_RE_WR_FLAGS_SIGNALED : 0x00;
  void* queueBuffAddr = wq.sqAddr;
  void* msntblAddr = wq.msntblAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  uint32_t mtuSize = wq.mtuSize;

  struct bnxt_re_bsqe hdr;
  struct bnxt_re_send send;
  struct bnxt_re_sge sge;
  // constexpr int sendWqeSize =
  //     sizeof(struct bnxt_re_bsqe) + sizeof(struct bnxt_re_send) + sizeof(struct bnxt_re_sge);
  // constexpr int slotsNum = CeilDiv(sendWqeSize, BNXT_RE_SLOT_SIZE);

  int psnCnt = (bytes == 0) ? 1 : (bytes + mtuSize - 1) / mtuSize;

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uint32_t slotIdx = wqeIdx * BNXT_RE_NUM_SLOT_PER_WQE;

  uint32_t wqe_size = BNXT_RE_HDR_WS_MASK & BNXT_RE_NUM_SLOT_PER_WQE;
  uint32_t hdr_flags = BNXT_RE_HDR_FLAGS_MASK & signalFlag;
  uint32_t wqe_type = BNXT_RE_HDR_WT_MASK & BNXT_RE_WR_OPCD_SEND;
  hdr.rsv_ws_fl_wt =
      (wqe_size << BNXT_RE_HDR_WS_SHIFT) | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT) | wqe_type;
  hdr.key_immd = 0;
  hdr.lhdr.qkey_len = bytes;

  // send slot reserved for UD, set to 0x0
  send.dst_qp = 0;
  send.avid = 0;
  send.rsvd = 0;

  sge.pa = (uint64_t)laddr;
  sge.lkey = lkey & 0xffffffff;
  sge.length = bytes;

  char* base = reinterpret_cast<char*>(queueBuffAddr) + slotIdx * BNXT_RE_SLOT_SIZE;
  ThreadCopy<char>(base + 0 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&hdr), sizeof(hdr));
  *reinterpret_cast<uint64_t*>(base + BNXT_RE_SLOT_SIZE) = 0ULL;
  *reinterpret_cast<uint64_t*>(base + BNXT_RE_SLOT_SIZE + 8) = 0ULL;  // memcpy -> set 0
  ThreadCopy<char>(base + 2 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&sge), sizeof(sge));

  // fill psns in msn Table for retransmissions
  uint32_t msntblIdx = curMsntblSlotIdx % wq.msntblNum;
  bnxt_re_fill_psns_for_msntbl(msntblAddr, slotIdx, curPsnIdx, psnCnt, msntblIdx);

  // get doorbell header
  // struct bnxt_re_db_hdr hdr;
  uint8_t flags = ((curPostIdx + 1) >> (__ffs(wqeNum) - 1)) & 0x1;
  uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  return bnxt_re_init_db_hdr(((((wqeIdx + 1) & (wqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE) | epoch),
                             0, qpn, BNXT_RE_QUE_TYPE_SQ);
}

template <>
inline __device__ uint64_t PostSend<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                        uint32_t curMsntblSlotIdx,
                                                        uint32_t curPsnIdx, bool cqeSignal,
                                                        uint32_t qpn, uintptr_t laddr,
                                                        uint64_t lkey, size_t bytes) {
  return BnxtPostSend(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn, laddr, lkey,
                      bytes);
}

template <>
inline __device__ uint64_t PostSend<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        size_t bytes) {
  uint32_t mtuSize = wq.mtuSize;

  int psnCnt = (bytes == 0) ? 1 : (bytes + mtuSize - 1) / mtuSize;
  uint32_t curMsntblSlotIdx, curPsnIdx, curPostIdx;
  // psn index needs to be strictly ordered
  atomic_add_packed_msn_and_psn(&wq.msnPack, 1, psnCnt, &curMsntblSlotIdx, &curPsnIdx);
  curPostIdx = curMsntblSlotIdx;
  return BnxtPostSend(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, true, qpn, laddr, lkey, bytes);
}

inline __device__ uint64_t BnxtPostRecv(WorkQueueHandle& wq, uint32_t curPostIdx, bool cqeSignal,
                                        uint32_t qpn, uintptr_t laddr, uint64_t lkey,
                                        size_t bytes) {
  uint8_t signalFlag = cqeSignal ? BNXT_RE_WR_FLAGS_SIGNALED : 0x00;
  void* queueBuffAddr = wq.rqAddr;
  uint32_t wqeNum = wq.rqWqeNum;
  struct bnxt_re_brqe hdr;
  struct bnxt_re_rqe recv;
  struct bnxt_re_sge sge;

  constexpr int recvWqeSize =
      sizeof(struct bnxt_re_brqe) + sizeof(struct bnxt_re_rqe) + sizeof(struct bnxt_re_sge);
  constexpr int slotsNum = CeilDiv(recvWqeSize, BNXT_RE_SLOT_SIZE);

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uint32_t slotIdx = wqeIdx * BNXT_RE_NUM_SLOT_PER_WQE;

  uint32_t wqe_size = BNXT_RE_HDR_WS_MASK & slotsNum;
  uint32_t hdr_flags = BNXT_RE_HDR_FLAGS_MASK & signalFlag;
  uint32_t wqe_type = BNXT_RE_HDR_WT_MASK & BNXT_RE_WR_OPCD_RECV;
  hdr.rsv_ws_fl_wt =
      (wqe_size << BNXT_RE_HDR_WS_SHIFT) | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT) | wqe_type;
  hdr.wrid = slotIdx / slotsNum;

  sge.pa = (uint64_t)laddr;
  sge.lkey = lkey & 0xffffffff;
  sge.length = bytes;

  char* base = reinterpret_cast<char*>(queueBuffAddr) + slotIdx * BNXT_RE_SLOT_SIZE;
  ThreadCopy<char>(base + 0 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&hdr), sizeof(hdr));
  *reinterpret_cast<uint64_t*>(base + BNXT_RE_SLOT_SIZE) = 0ULL;
  *reinterpret_cast<uint64_t*>(base + BNXT_RE_SLOT_SIZE + 8) = 0ULL;  // memcpy -> set 0
  ThreadCopy<char>(base + 2 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&sge), sizeof(sge));

  // recv wqe needn't to fill msntbl
  uint8_t flags = ((curPostIdx + 1) >> (__ffs(wqeNum) - 1)) & 0x1;
  uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  return bnxt_re_init_db_hdr(((((wqeIdx + 1) & (wqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE) | epoch),
                             0, qpn, BNXT_RE_QUE_TYPE_RQ);
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                        bool cqeSignal, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        size_t bytes) {
  return BnxtPostRecv(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, bytes);
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return BnxtPostRecv(wq, curPostIdx, true, qpn, laddr, lkey, bytes);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Read / Write APIs                                       */
/* ---------------------------------------------------------------------------------------------- */
// TODO: convert raddr/rkey laddr/lkey to big endien in advance to save cycles
inline __device__ uint64_t BnxtPostReadWrite(WorkQueueHandle& wq, uint32_t curPostIdx,
                                             uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                             bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                             uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                             size_t bytes, bool isRead) {
  uint32_t opcode = isRead ? BNXT_RE_WR_OPCD_RDMA_READ : BNXT_RE_WR_OPCD_RDMA_WRITE;
  uint8_t signalFlag = cqeSignal ? BNXT_RE_WR_FLAGS_SIGNALED : 0x00;
  // In bnxt, wqeNum mean total sq slot num
  void* queueBuffAddr = wq.sqAddr;
  void* msntblAddr = wq.msntblAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  uint32_t mtuSize = wq.mtuSize;

  struct bnxt_re_bsqe hdr;
  struct bnxt_re_rdma rdma;
  struct bnxt_re_sge sge;

  // constexpr int sendWqeSize =
  //     sizeof(struct bnxt_re_bsqe) + sizeof(struct bnxt_re_rdma) + sizeof(struct bnxt_re_sge);
  // constexpr int slotsNum = CeilDiv(sendWqeSize, BNXT_RE_SLOT_SIZE);

  int psnCnt = (bytes == 0) ? 1 : (bytes + mtuSize - 1) / mtuSize;
  // psn index needs to be strictly ordered

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uint32_t slotIdx = wqeIdx * BNXT_RE_NUM_SLOT_PER_WQE;

  uint32_t wqe_size = BNXT_RE_HDR_WS_MASK & BNXT_RE_NUM_SLOT_PER_WQE;
  uint32_t hdr_flags = BNXT_RE_HDR_FLAGS_MASK & signalFlag;
  uint32_t wqe_type = BNXT_RE_HDR_WT_MASK & opcode;
  hdr.rsv_ws_fl_wt =
      (wqe_size << BNXT_RE_HDR_WS_SHIFT) | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT) | wqe_type;
  hdr.key_immd = 0;
  hdr.lhdr.qkey_len = bytes;

  rdma.rva = (uint64_t)raddr;
  rdma.rkey = rkey & 0xffffffff;

  sge.pa = (uint64_t)laddr;
  sge.lkey = lkey & 0xffffffff;
  sge.length = bytes;

  char* base = reinterpret_cast<char*>(queueBuffAddr) + slotIdx * BNXT_RE_SLOT_SIZE;
  ThreadCopy<char>(base + 0 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&hdr), sizeof(hdr));
  ThreadCopy<char>(base + 1 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&rdma), sizeof(rdma));
  ThreadCopy<char>(base + 2 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&sge), sizeof(sge));

  // fill psns in msn Table for retransmissions
  uint32_t msntblIdx = curMsntblSlotIdx % wq.msntblNum;
  bnxt_re_fill_psns_for_msntbl(msntblAddr, slotIdx, curPsnIdx, psnCnt, msntblIdx);

  // get doorbell header
  // struct bnxt_re_db_hdr hdr;
  // uint8_t flags = ((curPostIdx + 1) / wqeNum) & 0x1;
  uint8_t flags = ((curPostIdx + 1) >> (__ffs(wqeNum) - 1)) & 0x1;
  uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  return bnxt_re_init_db_hdr(((((wqeIdx + 1) & (wqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE) | epoch),
                             0, qpn, BNXT_RE_QUE_TYPE_SQ);
}

template <>
inline __device__ uint64_t PostWrite<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                         uint32_t curMsntblSlotIdx,
                                                         uint32_t curPsnIdx, bool cqeSignal,
                                                         uint32_t qpn, uintptr_t laddr,
                                                         uint64_t lkey, uintptr_t raddr,
                                                         uint64_t rkey, size_t bytes) {
  return BnxtPostReadWrite(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn, laddr, lkey,
                           raddr, rkey, bytes, false);
}

template <>
inline __device__ uint64_t PostRead<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                        uint32_t curMsntblSlotIdx,
                                                        uint32_t curPsnIdx, bool cqeSignal,
                                                        uint32_t qpn, uintptr_t laddr,
                                                        uint64_t lkey, uintptr_t raddr,
                                                        uint64_t rkey, size_t bytes) {
  return BnxtPostReadWrite(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn, laddr, lkey,
                           raddr, rkey, bytes, true);
}

template <>
inline __device__ uint64_t PostWrite<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t qpn,
                                                         uintptr_t laddr, uint64_t lkey,
                                                         uintptr_t raddr, uint64_t rkey,
                                                         size_t bytes) {
  uint32_t mtuSize = wq.mtuSize;

  int psnCnt = (bytes == 0) ? 1 : (bytes + mtuSize - 1) / mtuSize;
  // psn index needs to be strictly ordered
  uint32_t curMsntblSlotIdx, curPsnIdx, curPostIdx;
  // psn index needs to be strictly ordered
  atomic_add_packed_msn_and_psn(&wq.msnPack, 1, psnCnt, &curMsntblSlotIdx, &curPsnIdx);
  curPostIdx = curMsntblSlotIdx;
  return BnxtPostReadWrite(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, true, qpn, laddr, lkey,
                           raddr, rkey, bytes, false);
}

template <>
inline __device__ uint64_t PostRead<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        uintptr_t raddr, uint64_t rkey,
                                                        size_t bytes) {
  uint32_t mtuSize = wq.mtuSize;

  int psnCnt = (bytes == 0) ? 1 : (bytes + mtuSize - 1) / mtuSize;
  // psn index needs to be strictly ordered
  uint32_t curMsntblSlotIdx, curPsnIdx, curPostIdx;
  // psn index needs to be strictly ordered
  atomic_add_packed_msn_and_psn(&wq.msnPack, 1, psnCnt, &curMsntblSlotIdx, &curPsnIdx);
  curPostIdx = curMsntblSlotIdx;
  return BnxtPostReadWrite(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, true, qpn, laddr, lkey,
                           raddr, rkey, bytes, true);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        WriteInline APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t BnxtPostWriteInline(WorkQueueHandle& wq, uint32_t curPostIdx,
                                               uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                               bool cqeSignal, uint32_t qpn, void* val,
                                               uintptr_t raddr, uint64_t rkey, size_t bytes) {
  // max is 16 * 13slot, use 1 slot now to align write/read
  assert(bytes <= BNXT_RE_SLOT_SIZE);
  uint8_t signalFlag = cqeSignal ? BNXT_RE_WR_FLAGS_SIGNALED : 0x00;
  // In bnxt, wqeNum mean total sq slot num
  void* queueBuffAddr = wq.sqAddr;
  void* msntblAddr = wq.msntblAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  uint32_t mtuSize = wq.mtuSize;

  struct bnxt_re_bsqe hdr;
  struct bnxt_re_rdma rdma;

  // constexpr int sendWqeSize =
  //     sizeof(struct bnxt_re_bsqe) + sizeof(struct bnxt_re_rdma) + sizeof(struct bnxt_re_sge);
  // constexpr int slotsNum = CeilDiv(sendWqeSize, BNXT_RE_SLOT_SIZE);

  // int psnCnt = 1;
  // psn index needs to be strictly ordered

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uint32_t slotIdx = wqeIdx * BNXT_RE_NUM_SLOT_PER_WQE;
  // TODO： wqeNum should be multiple of slotsNum, BRCM say using a specific conf currently.

  uint32_t wqe_size = BNXT_RE_HDR_WS_MASK & BNXT_RE_NUM_SLOT_PER_WQE;
  uint32_t hdr_flags = BNXT_RE_HDR_FLAGS_MASK & (BNXT_RE_WR_FLAGS_INLINE | signalFlag);
  uint32_t wqe_type = BNXT_RE_HDR_WT_MASK & BNXT_RE_WR_OPCD_RDMA_WRITE;
  hdr.rsv_ws_fl_wt =
      (wqe_size << BNXT_RE_HDR_WS_SHIFT) | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT) | wqe_type;
  hdr.key_immd = 0;
  hdr.lhdr.qkey_len = bytes;

  rdma.rva = (uint64_t)raddr;
  rdma.rkey = rkey & 0xffffffff;

  char* base = reinterpret_cast<char*>(queueBuffAddr) + slotIdx * BNXT_RE_SLOT_SIZE;
  ThreadCopy<char>(base + 0 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&hdr), sizeof(hdr));
  ThreadCopy<char>(base + 1 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&rdma), sizeof(rdma));
  uint32_t* wqeDataPtr = reinterpret_cast<uint32_t*>(base + 2 * BNXT_RE_SLOT_SIZE);
  if (bytes == 4) {
    AtomicStoreRelaxed(reinterpret_cast<uint32_t*>(wqeDataPtr),
                       reinterpret_cast<uint32_t*>(val)[0]);
  } else {
    for (int i = 0; i < bytes; i++) {
      AtomicStoreRelaxed(reinterpret_cast<uint8_t*>(wqeDataPtr) + i,
                         reinterpret_cast<uint8_t*>(val)[i]);
    }
  }

  // fill psns in msn Table for retransmissions
  uint32_t msntblIdx = curMsntblSlotIdx % wq.msntblNum;
  bnxt_re_fill_psns_for_msntbl(msntblAddr, slotIdx, curPsnIdx, 1, msntblIdx);

  // get doorbell header
  // struct bnxt_re_db_hdr hdr;
  uint8_t flags = ((curPostIdx + 1) >> (__ffs(wqeNum) - 1)) & 0x1;
  uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  return bnxt_re_init_db_hdr(((((wqeIdx + 1) & (wqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE) | epoch),
                             0, qpn, BNXT_RE_QUE_TYPE_SQ);
}

template <>
inline __device__ uint64_t PostWriteInline<ProviderType::BNXT>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, void* val, uintptr_t raddr, uint64_t rkey, size_t bytes) {
  return BnxtPostWriteInline(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn, val,
                             raddr, rkey, bytes);
}

template <>
inline __device__ uint64_t PostWriteInline<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t qpn,
                                                               void* val, uintptr_t raddr,
                                                               uint64_t rkey, size_t bytes) {
  // psn index needs to be strictly ordered
  uint32_t curMsntblSlotIdx, curPsnIdx, curPostIdx;
  atomic_add_packed_msn_and_psn(&wq.msnPack, 1, 1, &curMsntblSlotIdx, &curPsnIdx);
  curPostIdx = curMsntblSlotIdx;
  return BnxtPostWriteInline(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, true, qpn, val, raddr,
                             rkey, bytes);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Atomic APIs                                             */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t BnxtPrepareAtomicWqe(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                                bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                                uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                void* val_1, void* val_2, uint32_t bytes,
                                                atomicType amo_op) {
  // In bnxt, wqeNum mean total sq slot num
  uint8_t signalFlag = cqeSignal ? BNXT_RE_WR_FLAGS_SIGNALED : 0x00;
  void* queueBuffAddr = wq.sqAddr;
  void* msntblAddr = wq.msntblAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  uint32_t mtuSize = wq.mtuSize;

  struct bnxt_re_bsqe hdr;
  struct bnxt_re_atomic amo;
  struct bnxt_re_sge sge;

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uint32_t slotIdx = wqeIdx * BNXT_RE_NUM_SLOT_PER_WQE;

  uint32_t opcode = BNXT_RE_WR_OPCD_ATOMIC_FA;
  uint64_t data = val_1 ? *static_cast<uint64_t*>(val_1) : 0;
  uint64_t cmp = val_2 ? *static_cast<uint64_t*>(val_2) : 0;
  // printf("BNXT atomic values: data=0x%lx, cmp=0x%lx\n", data, cmp);

  switch (amo_op) {
    case AMO_FETCH_INC:
    case AMO_INC: {
      opcode = BNXT_RE_WR_OPCD_ATOMIC_FA;
      data = 1;
      break;
    }
    case AMO_FETCH_ADD:
    case AMO_SIGNAL_ADD:
    case AMO_ADD: {
      opcode = BNXT_RE_WR_OPCD_ATOMIC_FA;
      break;
    }
    case AMO_FETCH: {
      opcode = BNXT_RE_WR_OPCD_ATOMIC_FA;
      data = 0;
      break;
    }
    case AMO_COMPARE_SWAP: {
      opcode = BNXT_RE_WR_OPCD_ATOMIC_CS;
      break;
    }
    default: {
      printf("Error: unsupported atomic type (%d)\n", amo_op);
      assert(0);
    }
  }

  uint32_t wqe_size = BNXT_RE_HDR_WS_MASK & BNXT_RE_NUM_SLOT_PER_WQE;
  uint32_t hdr_flags = BNXT_RE_HDR_FLAGS_MASK & signalFlag;
  uint32_t wqe_type = BNXT_RE_HDR_WT_MASK & opcode;
  hdr.rsv_ws_fl_wt =
      (wqe_size << BNXT_RE_HDR_WS_SHIFT) | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT) | wqe_type;
  hdr.key_immd = rkey & 0xffffffff;
  hdr.lhdr.rva = (uint64_t)raddr;

  amo.swp_dt = (uint64_t)data;
  amo.cmp_dt = (uint64_t)cmp;

  sge.pa = (uint64_t)laddr;
  sge.lkey = lkey & 0xffffffff;
  sge.length = 8;

  char* base = reinterpret_cast<char*>(queueBuffAddr) + slotIdx * BNXT_RE_SLOT_SIZE;
  ThreadCopy<char>(base + 0 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&hdr), sizeof(hdr));
  ThreadCopy<char>(base + 1 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&amo), sizeof(amo));
  ThreadCopy<char>(base + 2 * BNXT_RE_SLOT_SIZE, reinterpret_cast<char*>(&sge), sizeof(sge));

  // fill psns in msn Table for retransmissions
  uint32_t msntblIdx = curMsntblSlotIdx % wq.msntblNum;
  bnxt_re_fill_psns_for_msntbl(msntblAddr, slotIdx, curPsnIdx, 1, msntblIdx);

  // get doorbell header
  // struct bnxt_re_db_hdr hdr;
  uint8_t flags = ((curPostIdx + 1) >> (__ffs(wqeNum) - 1)) & 0x1;
  uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  return bnxt_re_init_db_hdr(((((wqeIdx + 1) & (wqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE) | epoch),
                             0, qpn, BNXT_RE_QUE_TYPE_SQ);
}

template <>
inline __device__ uint64_t PostAtomic<ProviderType::BNXT>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr, uint64_t rkey,
    void* val_1, void* val_2, uint32_t typeBytes, atomicType amo_op) {
  return BnxtPrepareAtomicWqe(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn, laddr,
                              lkey, raddr, rkey, val_1, val_2, typeBytes, amo_op);
}

template <>
inline __device__ uint64_t PostAtomic<ProviderType::BNXT>(WorkQueueHandle& wq, uint32_t qpn,
                                                          uintptr_t laddr, uint64_t lkey,
                                                          uintptr_t raddr, uint64_t rkey,
                                                          void* val_1, void* val_2,
                                                          uint32_t typeBytes, atomicType amo_op) {
  // psn index needs to be strictly ordered
  uint32_t curMsntblSlotIdx, curPsnIdx, curPostIdx;
  atomic_add_packed_msn_and_psn(&wq.msnPack, 1, 1, &curMsntblSlotIdx, &curPsnIdx);
  curPostIdx = curMsntblSlotIdx;
  return BnxtPrepareAtomicWqe(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, true, qpn, laddr, lkey,
                              raddr, rkey, val_1, val_2, typeBytes, amo_op);
}

#define DEFINE_BNXT_POST_ATOMIC_SPEC(TYPE)                                                      \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::BNXT, TYPE>(                              \
      WorkQueueHandle & wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, \
      bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,            \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    return BnxtPrepareAtomicWqe(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn,    \
                                laddr, lkey, raddr, rkey, (void*)&val_1, (void*)&val_2,         \
                                sizeof(TYPE), amo_op);                                          \
  }                                                                                             \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::BNXT, TYPE>(                              \
      WorkQueueHandle & wq, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,      \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    uint32_t curMsntblSlotIdx, curPsnIdx, curPostIdx;                                           \
    atomic_add_packed_msn_and_psn(&wq.msnPack, 1, 1, &curMsntblSlotIdx, &curPsnIdx);            \
    curPostIdx = curMsntblSlotIdx;                                                              \
    return BnxtPrepareAtomicWqe(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, true, qpn, laddr,  \
                                lkey, raddr, rkey, (void*)&val_1, (void*)&val_2, sizeof(TYPE),  \
                                amo_op);                                                        \
  }

DEFINE_BNXT_POST_ATOMIC_SPEC(uint32_t)
DEFINE_BNXT_POST_ATOMIC_SPEC(uint64_t)
DEFINE_BNXT_POST_ATOMIC_SPEC(int32_t)
DEFINE_BNXT_POST_ATOMIC_SPEC(int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void UpdateSendDbrRecord<ProviderType::BNXT>(void* dbrRecAddr, uint32_t wqeIdx) {
  ;
}

template <>
inline __device__ void UpdateRecvDbrRecord<ProviderType::BNXT>(void* dbrRecAddr, uint32_t wqeIdx) {
  ;
}

template <>
inline __device__ void RingDoorbell<ProviderType::BNXT>(void* dbrAddr, uint64_t dbrVal) {
  core::AtomicStoreSeqCstSystem(reinterpret_cast<uint64_t*>(dbrAddr), dbrVal);
  // __builtin_nontemporal_store(dbrVal , reinterpret_cast<uint64_t*>(dbrAddr));
}

template <>
inline __device__ void UpdateDbrAndRingDbSend<ProviderType::BNXT>(void* dbrRecAddr, uint32_t wqeIdx,
                                                                  void* dbrAddr, uint64_t dbrVal,
                                                                  uint32_t* lockVar) {
  AcquireLock(lockVar);

  UpdateSendDbrRecord<ProviderType::BNXT>(dbrRecAddr, wqeIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::BNXT>(dbrAddr, dbrVal);

  ReleaseLock(lockVar);
}

template <>
inline __device__ void UpdateDbrAndRingDbRecv<ProviderType::BNXT>(void* dbrRecAddr, uint32_t wqeIdx,
                                                                  void* dbrAddr, uint64_t dbrVal,
                                                                  uint32_t* lockVar) {
  AcquireLock(lockVar);

  UpdateRecvDbrRecord<ProviderType::BNXT>(dbrRecAddr, wqeIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::BNXT>(dbrAddr, dbrVal);

  ReleaseLock(lockVar);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Completion Queue                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ int PollSingleCqe(volatile char* cqe, uint32_t consIdx, uint32_t* wqeIdx) {
  // Extract completion index using HIP atomic load
  const uint32_t con_indx = __hip_atomic_load(
      reinterpret_cast<uint32_t*>(const_cast<char*>(cqe) + offsetof(bnxt_re_req_cqe, con_indx)),
      __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);

  if (wqeIdx) {
    *wqeIdx = con_indx & 0xFFFF;
  }

  // Check completion status using HIP atomic load
  volatile char* flgSrc = cqe + sizeof(struct bnxt_re_req_cqe);
  const uint32_t flg_val = __hip_atomic_load(reinterpret_cast<uint32_t*>(const_cast<char*>(flgSrc)),
                                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  const uint8_t status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT) & BNXT_RE_BCQE_STATUS_MASK;

  if (status == BNXT_RE_REQ_ST_OK) {
    return BNXT_RE_REQ_ST_OK;
  }

  return status;
}

template <>
inline __device__ int PollCqOnce<ProviderType::BNXT>(void* cqeAddr, uint32_t cqeNum,
                                                     uint32_t consIdx, uint32_t* wqeIdx) {
  // Fast path for single CQE (most common case) - eliminate all branching
  if (cqeNum == 1) {
    return PollSingleCqe(static_cast<volatile char*>(cqeAddr), consIdx, wqeIdx);
  }

  // Slower path for multiple CQEs
  const uint32_t cqeIdx = consIdx % cqeNum;
  volatile char* cqe = static_cast<volatile char*>(cqeAddr) + 2 * BNXT_RE_SLOT_SIZE * cqeIdx;
  volatile char* flgSrc = cqe + sizeof(struct bnxt_re_req_cqe);
  const uint32_t flg_val = *reinterpret_cast<volatile uint32_t*>(flgSrc);
  const uint32_t expected_phase = BNXT_RE_QUEUE_START_PHASE ^ ((consIdx / cqeNum) & 0x1);

  if ((flg_val & BNXT_RE_BCQE_PH_MASK) != expected_phase) {
    return -1;  // CQE not ready yet
  }

  // Extract completion index and check status
  const uint32_t con_indx =
      *reinterpret_cast<volatile uint32_t*>(cqe + offsetof(bnxt_re_req_cqe, con_indx));

  if (wqeIdx) {
    *wqeIdx = con_indx & 0xFFFF;
  }

  const uint8_t status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT) & BNXT_RE_BCQE_STATUS_MASK;

  if (__builtin_expect(status == BNXT_RE_REQ_ST_OK, 1)) {
    return BNXT_RE_REQ_ST_OK;
  }

  return status;
}

template <>
inline __device__ int PollCq<ProviderType::BNXT>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx) {
  const uint32_t curConsIdx = atomicAdd(consIdx, 1);
  int opcode = -1;
  do {
    opcode = PollCqOnce<ProviderType::BNXT>(cqAddr, cqeNum, curConsIdx, nullptr);
    // TODO: Explain clearly why adding a compiler barrier fix hang issue
    asm volatile("" ::: "memory");
  } while (opcode < 0);

  // Handle error cases
  if (opcode != BNXT_RE_REQ_ST_OK) {
    auto error = BnxtHandleErrorCqe(opcode);
    printf("[BNXT PollCq] CQE error: %s (opcode: %d) at %s:%d\n", IbvWcStatusString(error), opcode,
           __FILE__, __LINE__);
    return opcode;
  }

  return BNXT_RE_REQ_ST_OK;
}

template <>
inline __device__ int PollCq<ProviderType::BNXT>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx,
                                                 uint32_t* wqeCounter) {
  const uint32_t curConsIdx = *consIdx;
  int opcode = -1;
  uint32_t wqeIdx;
  do {
    opcode = PollCqOnce<ProviderType::BNXT>(cqAddr, cqeNum, curConsIdx, &wqeIdx);
    asm volatile("" ::: "memory");
  } while (opcode < 0);
  // Truncate to 16 bits
  *wqeCounter = wqeIdx & 0xFFFF;
  if (opcode != BNXT_RE_REQ_ST_OK) {
    auto error = BnxtHandleErrorCqe(opcode);
    printf("[BNXT PollCq] CQE error: %s (opcode: %d), wqeCounter: %u at %s:%d\n",
           IbvWcStatusString(error), opcode, *wqeCounter, __FILE__, __LINE__);
    return opcode;
  }

  return BNXT_RE_REQ_ST_OK;
}

template <>
inline __device__ int PollCq<ProviderType::BNXT>(WorkQueueHandle& wqHandle, CompletionQueueHandle& cqHandle,
                                                 void* cqAddr, uint32_t cqeNum, uint32_t* consIdx, uint16_t* wqeCounter)
{
  return 0;
}

template <>
inline __device__ void UpdateCqDbrRecord<ProviderType::BNXT>(CompletionQueueHandle& cq, uint32_t consIdx) {
  uint8_t flags = ((consIdx + 1) / cq.cqeNum) & (1UL << BNXT_RE_FLAG_EPOCH_HEAD_SHIFT);
  uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;
  // uint64_t dbrVal = bnxt_re_init_db_hdr(((consIdx + 1) | epoch), 0, flags, BNXT_RE_QUE_TYPE_CQ);
  uint64_t dbrVal =
      bnxt_re_init_db_hdr((((consIdx + 1) % cq.cqeNum) | epoch), 0, flags, BNXT_RE_QUE_TYPE_CQ);
  core::AtomicStoreSeqCstSystem(reinterpret_cast<uint64_t*>(cq.dbrRecAddr), dbrVal);
}

template <>
inline __device__ int PollCqAndUpdateDbr<ProviderType::BNXT>(CompletionQueueHandle& cq, uint32_t* consIdx,
                                                             uint32_t* lockVar) {
  AcquireLock(lockVar);

  int opcode = PollCq<ProviderType::BNXT>(cq.cqAddr, cq.cqeNum, consIdx);
  if (opcode >= 0) {
    UpdateCqDbrRecord<ProviderType::BNXT>(cq, *consIdx);
  }

  ReleaseLock(lockVar);
  return opcode;
}

}  // namespace core
}  // namespace mori

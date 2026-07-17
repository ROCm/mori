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

#include "mori/core/transport/rdma/core_device_types.hpp"

// using namespace mori::application;

namespace mori {
namespace core {
/* ---------------------------------------------------------------------------------------------- */
/*                                          IBGDA Define                                          */
/* ---------------------------------------------------------------------------------------------- */

#define IBGDA_4_BYTE_EXT_AMO_OPMOD 0x08000000
#define IBGDA_8_BYTE_EXT_AMO_OPMOD 0x09000000

typedef struct {
  uint32_t add_data;
  uint32_t field_boundary;
  uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_32_masked_fa_seg_t) == 16,
              "sizeof(ibgda_atomic_32_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
  uint64_t add_data;
  uint64_t field_boundary;
} __attribute__((__packed__)) ibgda_atomic_64_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_64_masked_fa_seg_t) == 16,
              "sizeof(ibgda_atomic_64_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
  uint32_t swap_data;
  uint32_t compare_data;
  uint32_t swap_mask;
  uint32_t compare_mask;
} __attribute__((__packed__)) ibgda_atomic_32_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_32_masked_cs_seg_t) == 16,
              "sizeof(ibgda_atomic_32_masked_cs_seg_t) == 16 failed.");
#endif

typedef struct {
  uint64_t swap;
  uint64_t compare;
} __attribute__((__packed__)) ibgda_atomic_64_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_64_masked_cs_seg_t) == 16,
              "sizeof(ibgda_atomic_64_masked_cs_seg_t) == 16 failed.");
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */

template <ProviderType PrvdType>
inline __device__ uint64_t PostSend(WorkQueueHandle& wq, uint32_t curPostIdx,
                                    uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, bool cqeSignal,
                                    uint32_t qpn, uintptr_t laddr, uint64_t lkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostSend(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                    uint64_t lkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostRecv(WorkQueueHandle& wq, uint32_t curPostIdx, bool cqeSignal,
                                    uint32_t qpn, uintptr_t laddr, uint64_t lkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostRecv(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                    uint64_t lkey, size_t bytes);

template <ProviderType PrvdType, bool IsRead>
inline __device__ uint64_t PostReadWrite(WorkQueueHandle& wq, uint32_t curPostIdx,
                                         uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                         bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                         uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                         size_t bytes);

template <ProviderType PrvdType, bool IsRead>
inline __device__ uint64_t PostReadWrite(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                         uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                         size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWrite(WorkQueueHandle& wq, uint32_t curPostIdx,
                                     uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, bool cqeSignal,
                                     uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                     uint64_t rkey, size_t bytes) {
  return PostReadWrite<PrvdType, false>(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn,
                                        laddr, lkey, raddr, rkey, bytes);
}

template <ProviderType PrvdType>
inline __device__ uint64_t PostRead(WorkQueueHandle& wq, uint32_t curPostIdx,
                                    uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, bool cqeSignal,
                                    uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                    uint64_t rkey, size_t bytes) {
  return PostReadWrite<PrvdType, true>(wq, curPostIdx, curMsntblSlotIdx, curPsnIdx, cqeSignal, qpn,
                                       laddr, lkey, raddr, rkey, bytes);
}

template <ProviderType PrvdType>
inline __device__ uint64_t PostWrite(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                     uint64_t lkey, uintptr_t raddr, uint64_t rkey, size_t bytes) {
  return PostReadWrite<PrvdType, false>(wq, qpn, laddr, lkey, raddr, rkey, bytes);
}

template <ProviderType PrvdType>
inline __device__ uint64_t PostRead(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                    uint64_t lkey, uintptr_t raddr, uint64_t rkey, size_t bytes) {
  return PostReadWrite<PrvdType, true>(wq, qpn, laddr, lkey, raddr, rkey, bytes);
}

// RDMA WRITE_WITH_IMM: like PostWrite but carries a 32-bit immediate that is
// delivered to the receiver's completion queue when (and only when) the payload
// DMA has landed remotely. This is the transport primitive for the inline-flag
// ring protocol (Phase 5): completion becomes a CQ event that cannot be observed
// before its data, unlike a separate GPU-memory flag AMO. Send-side only; the
// receiver consumes the immediate via a recv-CQ poll (see PollRecvCqImm).
template <ProviderType PrvdType>
inline __device__ uint64_t PostWriteImm(WorkQueueHandle& wq, uint32_t curPostIdx,
                                        uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                        bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                        uint64_t lkey, uintptr_t raddr, uint64_t rkey, uint32_t imm,
                                        size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWriteInline(WorkQueueHandle& wq, uint32_t curPostIdx,
                                           uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                           bool cqeSignal, uint32_t qpn, void* val, uintptr_t raddr,
                                           uint64_t rkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWriteInline(WorkQueueHandle& wq, uint32_t qpn, void* val,
                                           uintptr_t raddr, uint64_t rkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostAtomic(WorkQueueHandle& wq, uint32_t curPostIdx,
                                      uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, bool cqeSignal,
                                      uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                      uint64_t rkey, void* val_1, void* val_2, uint32_t typeBytes,
                                      atomicType amo_op);

template <ProviderType PrvdType>
inline __device__ uint64_t PostAtomic(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                      uint64_t lkey, uintptr_t raddr, uint64_t rkey, void* val_1,
                                      void* val_2, uint32_t typeBytes, atomicType amo_op);

template <ProviderType PrvdType, typename T>
inline __device__ uint64_t PostAtomic(WorkQueueHandle& wq, uint32_t curPostIdx,
                                      uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, bool cqeSignal,
                                      uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                      uint64_t rkey, const T val_1, const T val_2,
                                      atomicType amo_op);

template <ProviderType PrvdType, typename T>
inline __device__ uint64_t PostAtomic(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                      uint64_t lkey, uintptr_t raddr, uint64_t rkey, const T val_1,
                                      const T val_2, atomicType amo_op);

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
inline __device__ void UpdateSendDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
inline __device__ void UpdateRecvDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
inline __device__ void RingDoorbell(void* dbr_addr, uint64_t dbr_val);

template <ProviderType PrvdType>
inline __device__ void UpdateDbrAndRingDbSend(void* dbrRecAddr, uint32_t wqeIdx, void* dbrAddr,
                                              uint64_t dbrVal, uint32_t* lockVar);

template <ProviderType PrvdType>
inline __device__ void UpdateDbrAndRingDbRecv(void* dbrRecAddr, uint32_t wqeIdx, void* dbrAddr,
                                              uint64_t dbrVal, uint32_t* lockVar);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queue                                       */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
inline __device__ int PollCqOnce(void* cqAddr, uint32_t cqeNum, uint32_t consIdx, uint32_t* wqeIdx);

template <ProviderType PrvdType>
inline __device__ int PollCq(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx);

template <ProviderType PrvdType>
inline __device__ int PollCq(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx,
                             uint32_t* wqeCounter);

template <ProviderType PrvdType>
inline __device__ int PollCq(WorkQueueHandle& wqHandle, CompletionQueueHandle& cqHandle,
                             void* cqAddr, uint32_t cqeNum, uint32_t* consIdx,
                             uint16_t* wqeCounter);

// Receiver half of the Phase-5 inline-flag ring protocol. Polls a recv CQ for a
// single RDMA-WRITE-with-immediate completion. Returns 0 when a fresh RDMA_IMM
// CQE for slot *consIdx is ready (its payload DMA is proven globally visible),
// -1 if not ready yet, or a positive error code; on success *imm is set to the
// decoded 32-bit immediate and *consIdx is advanced. Because the immediate rides
// the transport CQ (not a separate GPU-memory flag), it can never be observed
// before its payload lands remotely -- the ordering guarantee the memory-flag
// path lacks. Receiver-side scaffold; nothing calls it yet.
template <ProviderType PrvdType>
inline __device__ int PollRecvCqImm(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx,
                                    uint32_t* imm);

template <ProviderType PrvdType>
inline __device__ void UpdateCqDbrRecord(CompletionQueueHandle& cq, uint32_t consIdx);

template <ProviderType PrvdType>
inline __device__ int PollCqAndUpdateDbr(CompletionQueueHandle& cq, uint32_t* consIdx,
                                         uint32_t* lockVar);

}  // namespace core
}  // namespace mori

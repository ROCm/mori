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

#include "primitives.hpp"
#include "mori/application/transport/rdma/rdma.hpp"

using namespace mori::application;

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
                                    uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, uint32_t qpn,
                                    uintptr_t laddr, uint64_t lkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostSend(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                    uint64_t lkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ void PostRecv(WorkQueueHandle& wq, uint32_t curPostIdx, uintptr_t laddr,
                                uint64_t lkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ void PostRecv(WorkQueueHandle& wq, uintptr_t laddr, uint64_t lkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWrite(WorkQueueHandle& wq, uint32_t curPostIdx,
                                     uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, uint32_t qpn,
                                     uintptr_t laddr, uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                     size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostRead(WorkQueueHandle& wq, uint32_t curPostIdx,
                                    uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, uint32_t qpn,
                                    uintptr_t laddr, uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                    size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWrite(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                     uint64_t lkey, uintptr_t raddr, uint64_t rkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostRead(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                    uint64_t lkey, uintptr_t raddr, uint64_t rkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWriteInline(WorkQueueHandle& wq, uint32_t curPostIdx,
                                           uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
                                           uint32_t qpn, void* val, uintptr_t raddr, uint64_t rkey,
                                           size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWriteInline(WorkQueueHandle& wq, uint32_t qpn, void* val,
                                           uintptr_t raddr, uint64_t rkey, size_t bytes);

template <ProviderType PrvdType>
inline __device__ uint64_t PostAtomic(WorkQueueHandle& wq, uint32_t curPostIdx,
                                      uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, uint32_t qpn,
                                      uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                      uint64_t rkey, void* val_1, void* val_2, uint32_t typeBytes,
                                      atomicType amo_op);

template <ProviderType PrvdType>
inline __device__ uint64_t PostAtomic(WorkQueueHandle& wq, uint32_t qpn, uintptr_t laddr,
                                      uint64_t lkey, uintptr_t raddr, uint64_t rkey, void* val_1,
                                      void* val_2, uint32_t typeBytes, atomicType amo_op);

template <ProviderType PrvdType, typename T>
inline __device__ uint64_t PostAtomic(WorkQueueHandle& wq, uint32_t curPostIdx,
                                      uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, uint32_t qpn,
                                      uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
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
                             uint16_t* wqeCounter);

template <ProviderType PrvdType>
inline __device__ void UpdateCqDbrRecord(void* dbrRecAddr, uint32_t consIdx, uint32_t cqeNum);

template <ProviderType PrvdType>
inline __device__ int PollCqAndUpdateDbr(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum,
                                         uint32_t* consIdx, void* dbrRecAddr, uint32_t* lockVar);

}  // namespace core
}  // namespace mori

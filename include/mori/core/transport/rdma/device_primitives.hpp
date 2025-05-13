#pragma once

#include "primitives.hpp"

namespace mori {
namespace core {
/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */

template <ProviderType PrvdType>
static __device__ uint64_t PostSend(void* queue_buff_addr, uint32_t* postIdx, uint32_t wqe_num,
                                    uint32_t qpn, uintptr_t laddr, uint64_t lkey,
                                    size_t bytes_count);

template <ProviderType PrvdType>
static __device__ void PostRecv(void* queue_buff_addr, uint32_t wqe_num, uint32_t* postIdx,
                                uintptr_t laddr, uint64_t lkey, size_t bytes_count);

template <ProviderType PrvdType>
static __device__ uint64_t PostWrite(void* queue_buff_addr, uint32_t wqe_num, uint32_t* postIdx,
                                     uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                     uint64_t rkey, size_t bytes_count);

template <ProviderType PrvdType>
static __device__ uint64_t PostRead(void* queue_buff_addr, uint32_t wqe_num, uint32_t* postIdx,
                                    uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                    uint64_t rkey, size_t bytes_count);

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
static __device__ void UpdateSendDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
static __device__ void UpdateRecvDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
static __device__ void RingDoorbell(void* dbr_addr, uint64_t dbr_val);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queu                                        */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
static __device__ int PollCqOnce(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum, uint32_t consIdx);

template <ProviderType PrvdType>
static __device__ int PollCq(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum, uint32_t* consIdx);

template <ProviderType PrvdType>
static __device__ void UpdateCqDbrRecord(void* dbrRecAddr, uint32_t consIdx);

}  // namespace core
}  // namespace mori
#pragma once

#include "primitives.hpp"

namespace mori {
namespace core {
/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */

template <ProviderType PrvdType>
static __host__ uint64_t PostSend(void* queue_buff_addr, uint32_t& post_idx, uint32_t wqe_num,
                                  uint32_t qpn, uintptr_t laddr, uint64_t lkey, size_t bytes_count);

template <ProviderType PrvdType>
static __host__ void PostRecv(void* queue_buff_addr, uint32_t wqe_num, uint32_t& post_idx,
                              uintptr_t laddr, uint64_t lkey, size_t bytes_count);

template <ProviderType PrvdType>
static __host__ uint64_t PostWrite(void* queue_buff_addr, uint32_t wqe_num, uint32_t& post_idx,
                                   uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                   uint64_t rkey, size_t bytes_count);

template <ProviderType PrvdType>
static __host__ uint64_t PostRead(void* queue_buff_addr, uint32_t wqe_num, uint32_t& post_idx,
                                  uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                  uint64_t rkey, size_t bytes_count);

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
static __host__ void UpdateSendDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
static __host__ void UpdateRecvDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
static __host__ void RingDoorbell(void* dbr_addr, uint64_t dbr_val);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queu                                        */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
static __host__ int PollCqOnce(CompletionQueueHandle cq);

template <ProviderType PrvdType>
static __host__ int PoolCq(CompletionQueueHandle cq);

template <ProviderType PrvdType>
static __host__ void UpdateCqDbrRecord(void* dbrRecAddr, uint32_t cons_idx);

}  // namespace core
}  // namespace mori
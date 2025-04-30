#pragma once

#include "primitives.hpp"

namespace mori {
namespace core {
namespace transport {
namespace ibgda {
/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */

template <ProviderType PrvdType>
static __host__ uint64_t PostWrite(const IbgdaWriteReq& req);

// template <ProviderType PrvdType>
// static __host__ void PostRead(const IbgdaWriteReq& req);

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
static __host__ void UpdateSendDbrRecord(void* dbr_rec_addr, uint32_t wqe_idx);

template <ProviderType PrvdType>
static __host__ void UpdateRecvDbrRecord(void* dbr_rec_addr, uint32_t wqe_idx);

template <ProviderType PrvdType>
static __host__ void RingDoorbell(void* dbr_addr, uint64_t dbr_val);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queu                                        */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
static __host__ int PollCqOnce(CompletionQueueHandle cq);

template <ProviderType PrvdType>
static __host__ int PoolCq(CompletionQueueHandle cq);

}  // namespace ibgda
}  // namespace transport
}  // namespace core
}  // namespace mori
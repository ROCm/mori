#pragma once

#include "primitives.hpp"

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
inline __device__ uint64_t PostSend(void* queue_buff_addr, uint32_t* postIdx, uint32_t wqe_num,
                                    uint32_t qpn, uintptr_t laddr, uint64_t lkey,
                                    size_t bytes_count);

template <ProviderType PrvdType>
inline __device__ void PostRecv(void* queue_buff_addr, uint32_t wqe_num, uint32_t* postIdx,
                                uintptr_t laddr, uint64_t lkey, size_t bytes_count);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWrite(void* queue_buff_addr, uint32_t wqe_num, uint32_t* postIdx,
                                     uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                     uint64_t rkey, size_t bytes_count);

template <ProviderType PrvdType>
inline __device__ uint64_t PostRead(void* queue_buff_addr, uint32_t wqe_num, uint32_t* postIdx,
                                    uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                    uint64_t rkey, size_t bytes_count);

template <ProviderType PrvdType>
inline __device__ uint64_t PostWriteInline(void* queue_buff_addr, uint32_t wqe_num,
                                           uint32_t* postIdx, uint32_t qpn, void* val,
                                           uintptr_t raddr, uint64_t rkey, size_t bytes_count);

template <ProviderType PrvdType, typename T>
static __device__ uint64_t PostAtomic(void* queue_buff_addr, uint32_t wqe_num, uint32_t* postIdx,
                                      uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                      uint64_t rkey, const T val_1, const T val_2,
                                      atomicType amo_op);

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
inline __device__ void UpdateSendDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
inline __device__ void UpdateRecvDbrRecord(void* dbrRecAddr, uint32_t wqe_idx);

template <ProviderType PrvdType>
inline __device__ void RingDoorbell(void* dbr_addr, uint64_t dbr_val);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queu                                        */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
inline __device__ int PollCqOnce(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum, uint32_t consIdx);

template <ProviderType PrvdType>
inline __device__ int PollCq(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum, uint32_t* consIdx);

template <ProviderType PrvdType>
inline __device__ void UpdateCqDbrRecord(void* dbrRecAddr, uint32_t consIdx);

}  // namespace core
}  // namespace mori
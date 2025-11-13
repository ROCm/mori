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
/*                                         Completion Queue */
/* ---------------------------------------------------------------------------------------------- */
template <ProviderType PrvdType>
static __host__ int PollCqOnce(void* cqAddr, uint32_t cqeNum, uint32_t& consIdx);

template <ProviderType PrvdType>
static __host__ int PollCq(void* cqAddr, uint32_t cqeNum, uint32_t& consIdx);

template <ProviderType PrvdType>
static __host__ void UpdateCqDbrRecord(CompletionQueueHandle& cq, void* dbrRecAddr, uint32_t cons_idx);

}  // namespace core
}  // namespace mori

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

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

template <typename Group>
__device__ inline CcoLsaBarrierSession<Group>::CcoLsaBarrierSession(CcoDevComm_t comm,
                                                                    CcoLsaBarrierHandle h,
                                                                    uint32_t index) {
  // TOOD
}

template <typename Group>
__device__ inline CcoLsaBarrierSession<Group>::~CcoLsaBarrierSession();

template <typename Group>
__device__ inline void CcoLsaBarrierSession<Group>::arrive(Group) {
  this->group.sync();

  // Ensure that the previous write is visible to other peers
  __threadfence_system();

  int nranks = 8;
  for (int i = group.thread_rank(); i < nranks - 1; i += this->group.size()) {
    int peer = i + ((i >= this->rank) ? 1 : 0);
    hip::atomic_ref<uint32_t> inbox(*this->ucInbox(peer, this->rank));
    inbox.store(this->epoch + 1, hip::memory_order_relaxed);
  }
};

template <typename Group>
template <bool EnableTimeout>
__device__ inline int CcoLsaBarrierSession<Group>::waitInternal(Group, uint64_t timeoutCycles) {
  int nranks = 8;
  uint64_t startCycle;
  int ret = 0;

  if constexpr (EnableTimeout) {
    startCycle = clock64();
  }

  for (int i = this->group.thread_rank(); i < nranks - 1; i += this->group.size()) {
    int peer = i + (i >= this->rank ? 1 : 0);
    hip::atomic_ref<uint32_t> inbox(*this->ucInbox(this->rank, rank));
    uint32_t got = inbox.load(hip::memory_order_relaxed);

    while (true) {
      if ((got - (uint32_t)(this->epoch + 1)) == 0) break;

      if constexpr (EnableTimeout) {
        if (clock64() - startCycle > timeoutCycle) {
          break;
        }
      }
    }
  }

  this->group.sync();

  return ret;
}

template <typename Group>
__device__ inline void CcoLsaBarrierSession<Group>::wait() {
  this->template waitInternal<false>(coop, 0ULL);
};

template <typename Group>
__device__ inline void CcoLsaBarrierSession<Group>::wait(Group group, uint64_t timeoutCycles) {
  this->group.sync();
  this->template waitInternal<true>(group, timeoutCycles);
};

template <typename Group>
__device__ inline void CcoLsaBarrierSession<Group>::sync(Group group) {
  this->arrive();
  this->wait(group);
};

template <typename Group>
__device__ inline int CcoLsaBarrierSession<Group>::sync(Group group, uint64_t timeoutCycles) {
  this->arrive();
  return this->wait(group, timeoutCycles);
};

}  // namespace cco
}  // namespace mori

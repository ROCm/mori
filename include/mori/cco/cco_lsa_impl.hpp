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

#include "mori/cco/cco_lsa_types.hpp"

namespace mori {
namespace cco {

template <typename Group>
__device__ inline CcoLsaBarrierSession<Group>::CcoLsaBarrierSession(Group grp, CcoDevComm_t comm,
                                                                    CcoLsaBarrierHandle h,
                                                                    uint32_t idx) {
  this->group = grp;
  this->comm = comm;
  this->handle = h;
  this->index = idx;

  // Restore epoch persisted by the previous session's destructor.
  // Inbox slots are never zeroed, so epoch must be monotonically increasing
  // to avoid false-positive matches against stale inbox values.
  uint32_t* state = CcoGetLocalResourceBuffer(comm, h.bufHandle);
  this->epoch = state[idx];
}

template <typename Group>
__device__ inline CcoLsaBarrierSession<Group>::~CcoLsaBarrierSession() {
  // Persist epoch so the next session on this barrier slot resumes correctly.
  uint32_t* state = CcoGetLocalResourceBuffer(this->comm, this->handle.bufHandle);
  if (this->group.thread_rank() == 0) {
    state[this->index] = this->epoch;
  }
  this->group.sync();
}

template <typename Group>
__device__ inline void CcoLsaBarrierSession<Group>::arrive(Group) {
  this->group.sync();

  const int nranks = this->comm->worldSize;
  const int myRank = this->comm->lsa.lsaRank;

  for (int i = this->group.thread_rank(); i < nranks - 1; i += this->group.size()) {
    int peer = i + ((i >= myRank) ? 1 : 0);
    // Write epoch+1 into peer's inbox slot reserved for us.
    __atomic_store_n(this->ucInbox(peer, myRank), this->epoch + 1, __ATOMIC_RELAXED);
  }
}

template <typename Group>
template <bool EnableTimeout>
__device__ inline int CcoLsaBarrierSession<Group>::waitInternal(Group, uint64_t timeoutCycles) {
  const int nranks = this->comm->worldSize;
  const int myRank = this->comm->lsa.lsaRank;
  int ret = 0;

  uint64_t startCycle;
  if constexpr (EnableTimeout) {
    startCycle = (uint64_t)clock64();
  }

  for (int i = this->group.thread_rank(); i < nranks - 1; i += this->group.size()) {
    int peer = i + ((i >= myRank) ? 1 : 0);
    uint32_t* slot = this->ucInbox(myRank, peer);

    while (true) {
      uint32_t got = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
      if ((got - (uint32_t)(this->epoch + 1)) == 0) break;

      if constexpr (EnableTimeout) {
        if ((uint64_t)clock64() - startCycle >= timeoutCycles) {
          ret = 1;  // timeout
          goto done;
        }
      }
    }
  }

  this->epoch += 1;

done:
  this->group.sync();
  return ret;
}

template <typename Group>
__device__ inline void CcoLsaBarrierSession<Group>::wait(Group g) {
  this->template waitInternal</* DisableTimeout */ false>(g, 0ULL);
}

template <typename Group>
__device__ inline int CcoLsaBarrierSession<Group>::wait(Group g, uint64_t timeoutCycles) {
  return this->template waitInternal</* EnableTimeout */ true>(g, timeoutCycles);
}

template <typename Group>
__device__ inline void CcoLsaBarrierSession<Group>::sync(Group g) {
  this->arrive(g);
  this->wait(g);
}

template <typename Group>
__device__ inline int CcoLsaBarrierSession<Group>::sync(Group g, uint64_t timeoutCycles) {
  this->arrive(g);
  return this->wait(g, timeoutCycles);
}

}  // namespace cco
}  // namespace mori

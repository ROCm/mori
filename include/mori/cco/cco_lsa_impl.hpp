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

template <typename Coop>
__device__ inline ccoLsaBarrierSession<Coop>::ccoLsaBarrierSession(Coop grp, ccoDevComm_t comm,
                                                                    ccoLsaBarrierHandle h,
                                                                    uint32_t idx)
    : group(grp), comm(comm), handle(h), index(idx) {

  // Restore epoch persisted by the previous session's destructor.
  // Inbox slots are never zeroed, so epoch must be monotonically increasing
  // to avoid false-positive matches against stale inbox values.
  //
  // State buffer lives at offset `bufOffset` inside the DevComm's resource
  // window. Use the standard LSA peer-addressing formula off the resource
  // window's own slot (winBase already = flatBase + resource window slotOffset).
  const auto& rw = comm->resourceWindow_inlined;
  char* base = rw.winBase + ((uint64_t)comm->lsaRank * rw.stride4G << 32);
  uint32_t* state = reinterpret_cast<uint32_t*>(base + h.bufOffset);
  this->epoch = state[h.nBarriers + idx];  // unicast epoch slot
}

template <typename Coop>
__device__ inline ccoLsaBarrierSession<Coop>::~ccoLsaBarrierSession() {
  // Persist epoch so the next session on this barrier slot resumes correctly.
  const auto& rw = this->comm->resourceWindow_inlined;
  char* base = rw.winBase + ((uint64_t)this->comm->lsaRank * rw.stride4G << 32);
  uint32_t* state = reinterpret_cast<uint32_t*>(base + this->handle.bufOffset);
  if (this->group.thread_rank() == 0) {
    state[this->handle.nBarriers + this->index] = this->epoch;  // unicast epoch slot
  }
  this->group.sync();
}

template <typename Coop>
__device__ inline void ccoLsaBarrierSession<Coop>::arrive(Coop) {
  this->group.sync();

  const int nranks = this->comm->lsaSize;
  const int myRank = this->comm->lsaRank;

  for (int i = this->group.thread_rank(); i < nranks - 1; i += this->group.size()) {
    int peer = i + ((i >= myRank) ? 1 : 0);
    // Write epoch+1 into peer's inbox slot reserved for us， cross-gpu write
    __hip_atomic_store(this->ucInbox(peer, myRank), this->epoch + 1, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

template <typename Coop>
template <bool EnableTimeout>
__device__ inline int ccoLsaBarrierSession<Coop>::waitInternal(Coop, uint64_t timeoutCycles) {
  const int nranks = this->comm->lsaSize;
  const int myRank = this->comm->lsaRank;
  int ret = 0;

  uint64_t startCycle;
  if constexpr (EnableTimeout) {
    startCycle = (uint64_t)clock64();
  }

  for (int i = this->group.thread_rank(); i < nranks - 1; i += this->group.size()) {
    int peer = i + ((i >= myRank) ? 1 : 0);
    uint32_t* slot = this->ucInbox(myRank, peer);

    while (true) {
      uint32_t got = __hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
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

template <typename Coop>
__device__ inline void ccoLsaBarrierSession<Coop>::wait(Coop g) {
  this->template waitInternal</* DisableTimeout */ false>(g, 0ULL);
}

template <typename Coop>
__device__ inline int ccoLsaBarrierSession<Coop>::wait(Coop g, uint64_t timeoutCycles) {
  return this->template waitInternal</* EnableTimeout */ true>(g, timeoutCycles);
}

template <typename Coop>
__device__ inline void ccoLsaBarrierSession<Coop>::sync(Coop g) {
  this->arrive(g);
  this->wait(g);
}

template <typename Coop>
__device__ inline int ccoLsaBarrierSession<Coop>::sync(Coop g, uint64_t timeoutCycles) {
  this->arrive(g);
  return this->wait(g, timeoutCycles);
}

}  // namespace cco
}  // namespace mori

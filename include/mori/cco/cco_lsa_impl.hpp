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

// Flat-VA addressing helpers — intra-node (LSA) only. The flat VA covers the
// LSA team, so peer indexing is by LSA rank. Cross-node access goes through the
// GDA backend with iova=0 + offset and doesn't need these.
__device__ inline void* ccoGetLsaPeerPtr(ccoWindow_t win, int peerLsaRank, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(peerLsaRank) * win->stride4G) << 32) + offset;
}

__device__ inline void* ccoGetLocalPtr(ccoWindow_t win, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(win->lsaRank) * win->stride4G) << 32) + offset;
}

template <typename Coop>
__device__ inline ccoLsaBarrierSession<Coop>::ccoLsaBarrierSession(Coop coop, ccoDevComm_t comm,
                                                                   ccoTeam_t team,
                                                                   ccoLsaBarrierHandle h,
                                                                   uint32_t idx)
    : coop(coop), team(team), comm(comm), handle(h), index(idx) {
  assert(idx < h.nBarriers);

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
  this->epoch = state[idx];  // unicast epoch slot
}

template <typename Coop>
__device__ inline ccoLsaBarrierSession<Coop>::~ccoLsaBarrierSession() {
  // Persist epoch so the next session on this barrier slot resumes correctly.
  const auto& rw = this->comm->resourceWindow_inlined;
  char* base = rw.winBase + ((uint64_t)this->comm->lsaRank * rw.stride4G << 32);
  uint32_t* state = reinterpret_cast<uint32_t*>(base + this->handle.bufOffset);
  if (this->coop.thread_rank() == 0) {
    state[this->index] = this->epoch;  // unicast epoch slot
  }
  this->coop.sync();
}

template <typename Coop>
__device__ inline void ccoLsaBarrierSession<Coop>::arrive(Coop) {
  this->coop.sync();

  const int nranks = this->team.nRanks;
  const int myRank = this->team.rank;

  // System-scope fence so any prior payload writes from this coop are
  // observable to peers before the relaxed inbox stores below land.
  if (nranks > 1) {
    __threadfence_system();
  }

  for (int i = this->coop.thread_rank(); i < nranks - 1; i += this->coop.size()) {
    int peer = i + ((i >= myRank) ? 1 : 0);
    __hip_atomic_store(this->ucInbox(peer, myRank), this->epoch + 1, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

template <typename Coop>
template <bool EnableTimeout>
__device__ inline int ccoLsaBarrierSession<Coop>::waitInternal(Coop, uint64_t timeoutCycles) {
  const int nranks = this->team.nRanks;
  const int myRank = this->team.rank;
  int ret = 0;

  uint64_t startCycle;
  if constexpr (EnableTimeout) {
    startCycle = (uint64_t)clock64();
  }

  for (int i = this->coop.thread_rank(); i < nranks - 1; i += this->coop.size()) {
    int peer = i + ((i >= myRank) ? 1 : 0);
    uint32_t* slot = this->ucInbox(myRank, peer);

    while (true) {
      uint32_t got = __hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);

      if ((got - (uint32_t)(this->epoch + 1)) <= ((uint32_t)-1 >> 1)) break;

      if constexpr (EnableTimeout) {
        if ((uint64_t)clock64() - startCycle >= timeoutCycles) {
          ret = 1;
          goto done;
        }
      }
    }
  }

  this->epoch += 1;

done:
  this->coop.sync();
  return ret;
}

template <typename Coop>
__device__ inline void ccoLsaBarrierSession<Coop>::wait(Coop coop) {
  this->template waitInternal</* DisableTimeout */ false>(coop, 0ULL);
}

template <typename Coop>
__device__ inline int ccoLsaBarrierSession<Coop>::wait(Coop coop, uint64_t timeoutCycles) {
  return this->template waitInternal</* EnableTimeout */ true>(coop, timeoutCycles);
}

template <typename Coop>
__device__ inline void ccoLsaBarrierSession<Coop>::sync(Coop coop) {
  this->arrive(coop);
  this->wait(coop);
}

template <typename Coop>
__device__ inline int ccoLsaBarrierSession<Coop>::sync(Coop coop, uint64_t timeoutCycles) {
  this->arrive(coop);
  return this->wait(coop, timeoutCycles);
}

}  // namespace cco
}  // namespace mori

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

// State buffer layout (unicast only, no multicast):
//   [ 0, nBarries)                                   unicast epoch
//   [nBarriers, nBarriers + nBarriers*lsaSize)   ucInbox[index][peer]

template <typename Coop>
struct ccoLsaBarrierSession {
  Coop coop;
  ccoTeam_t team;
  ccoDevComm_t comm;
  ccoLsaBarrierHandle handle;
  uint32_t epoch;
  uint32_t index;

  // TODO: support multicast on new generation hardware
  // TODO: add flexible memory order parameters in APIs

  __device__ inline ccoLsaBarrierSession(Coop group, ccoDevComm_t comm, ccoTeam_t team,
                                         ccoLsaBarrierHandle h, uint32_t index);
  __device__ inline ~ccoLsaBarrierSession();

  // Write epoch+1 into peer's inbox slot reserved for us, cross-gpu write
  __device__ inline void arrive(Coop);

  // Read each peer's arrival signal from my own buffer at slot[peer]
  __device__ inline void wait(Coop);
  __device__ inline int wait(Coop, uint64_t timeoutCycles);

  __device__ inline void sync(Coop);
  __device__ inline int sync(Coop, uint64_t timeoutCycles);

 private:
  __device__ inline uint32_t* ucInbox(int owner, int peer) {
    // State buffer lives inside the DevComm's resource window at offset
    // `bufOffset`. Resource window's winBase already = flatBase + the
    // resource window's slotOffset, so applying the canonical LSA peer
    // formula here matches ccoGetLsaPeerPtr / cco_types.hpp::ccoLsaBarrierHandle
    // comment (winBase + peer*stride4G<<32 + bufOffset).
    const auto& rw = comm->resourceWindow_inlined;
    char* base = rw.winBase + ((uint64_t)owner * rw.stride4G << 32);
    uint32_t* state = reinterpret_cast<uint32_t*>(base + handle.bufOffset);
    return state + handle.nBarriers + index * comm->lsaSize + peer;
  }

  template <bool EnableTimeout>
  __device__ inline int waitInternal(Coop, uint64_t timeoutCycles);
};

}  // namespace cco
}  // namespace mori

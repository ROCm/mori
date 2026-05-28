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
//   [ 0, nBarries)                                   multimem epoch
//   [ nBarries,    2 * nBarries)                     unicast epoch
//   [2 * nBarries, 3 * nBarries)                     multimem inbox
//   [3*nBarriers, 3*nBarriers + nBarriers*lsaSize)   ucInbox[index][peer]

template <typename Group>
struct CcoLsaBarrierSession {
  Group group;
  CcoDevComm_t comm;
  CcoLsaBarrierHandle handle;
  uint32_t epoch;
  uint32_t index;

  // TODO: support multicast on new generation hardware
  // TODO: add flexible memory order parameters in APIs

  __device__ inline CcoLsaBarrierSession(Group group, CcoDevComm_t comm, CcoLsaBarrierHandle h,
                                         uint32_t index);
  __device__ inline ~CcoLsaBarrierSession();

  // Write epoch+1 into peer's inbox slot reserved for us, cross-gpu write
  __device__ inline void arrive(Group);

  // Read each peer's arrival signal from my own buffer at slot[peer]
  __device__ inline void wait(Group);
  __device__ inline int wait(Group, uint64_t timeoutCycles);

  __device__ inline void sync(Group);
  __device__ inline int sync(Group, uint64_t timeoutCycles);

 private:
  __device__ inline uint32_t* ucInbox(int owner, int peer) {
    char* base = reinterpret_cast<char*>(comm->flatBase) + (uint64_t)owner * comm->perRankSize;
    uint32_t* state = reinterpret_cast<uint32_t*>(base + handle.bufOffset);
    return state + 3 * handle.nBarriers + index * comm->lsaSize + peer;
  }

  template <bool EnableTimeout>
  __device__ inline int waitInternal(Group, uint64_t timeoutCycles);
};

}  // namespace cco
}  // namespace mori

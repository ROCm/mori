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

#define CCO_BUF_HANDLE_GRANULARITY (128)

// bufHandle is a 128-byte unit index within the rank's slot.
__device__ inline uint32_t* CcoGetResourceBuffer(CcoDevComm_t comm, uint32_t bufHandle, int rank) {
  char* base = reinterpret_cast<char*>(comm->flatBase);
  char* rankBase = base + (uint64_t)rank * comm->perRankSize;
  return reinterpret_cast<uint32_t*>(rankBase + (uint64_t)bufHandle * CCO_BUF_HANDLE_GRANULARITY);
}

__device__ inline uint32_t* CcoGetLocalResourceBuffer(CcoDevComm_t comm, uint32_t bufHandle) {
  return CcoGetResourceBuffer(comm, bufHandle, comm->lsa.lsaRank);
}

// State buffer layout (unicast only, no multicast):
//   [0,          nBarriers)                    epoch[index]       (persisted across sessions)
//   [nBarriers,  nBarriers + nBarriers*nRanks) ucInbox[index][peer]

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

  __device__ inline void arrive(Group);
  __device__ inline void wait(Group);
  __device__ inline int wait(Group, uint64_t timeoutCycles);

  __device__ inline void sync(Group);
  __device__ inline int sync(Group, uint64_t timeoutCycles);

 private:
  // inbox where `owner` expects to receive a signal from `peer`
  __device__ inline uint32_t* ucInbox(int owner, int peer) {
    uint32_t* state = CcoGetResourceBuffer(comm, handle.bufHandle, owner);
    return state + handle.nBarriers + index * comm->worldSize + peer;
  }

  template <bool EnableTimeout>
  __device__ inline int waitInternal(Group, uint64_t timeoutCycles);
};

}  // namespace cco
}  // namespace mori

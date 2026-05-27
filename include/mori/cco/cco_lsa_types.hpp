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

__device__ inline void* CcoGetResourceBufferLocalPointer(CcoDevComm_t comm, uint32_t resHandle) {
  void* lsaFlatBase = comm.lsaFlatBase;
  uint32_t stride4G = 1;
  void* local = lsaFlatBase + (stride4G * comm.lsaRank) << 32;
  return (void*)(reinterpret_cast<char (*)[128]>(local) + resHandle);
}

struct CcoLsaBarrierHandle {
  uint32_t nBarriers;
  uint32_t bufHandle;
};

template <typename Group>
struct CcoLsaBarrierSession {
  Group group; /* thread / warp / block */
  CcoDevComm_t comm;
  CcoLsaBarrierHandle handle;
  uint32_t epoch;
  uint32_t index;
  // (1 * nbarries + nRanks * nBarries) * uint32_t

  // TODO: alloc barrier inbox memory
  // TODO: support multicast on new generation hardware
  // TODO: add flexible memory order parameters in APIs

  __device__ inline CcoLsaBarrierSession(CcoDevComm_t comm, CcoLsaBarrierHandle h, uint32_t index);
  __device__ inline ~CcoLsaBarrierSession();

  __device__ inline void arrive(Group);
  __device__ inline void wait(Group);
  __device__ inline int wait(Group, uint64_t timeoutCycles);

  __device__ inline void sync(Group);
  __device__ inline int sync(Group, uint64_t timeoutCycles);

 private:
  __device__ inline uint32_t* ucInbox(int owner, int peer) {
    uint32_t* state = comm->flatBase + owner * comm->stride4G + h.bufHandle * 128;
    return state + nBarriers + index * ranks + peer;
  }

  template <bool EnableTimeout>
  __device__ inline int waitInternal(Group);
};

}  // namespace cco

}  // namespace mori

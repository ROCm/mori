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

// Concrete group types used as the `Group` template arg of
// ccoLsaBarrierSession<Group>. Each must provide:
//   __device__ int  thread_rank() const   // rank within the group
//   __device__ int  size()        const   // number of threads in the group
//   __device__ void sync()                // group-internal sync barrier
//
// They are intentionally NOT derived from a virtual base: device-side
// virtual dispatch is problematic on AMD GPU (vtable placement, devirt
// reliability), and ccoLsaBarrierSession is a template — polymorphism is
// not required.

struct ccoBlockGroup {
  __device__ inline int  thread_rank() const { return threadIdx.x; }
  __device__ inline int  size()        const { return blockDim.x;  }
  __device__ inline void sync()              { __syncthreads();    }
};

struct ccoWarpGroup {
  __device__ inline int  thread_rank() const { return __lane_id(); }
  // `warpSize` is a HIP/CUDA built-in __device__ const int (64 on AMD gfx9+).
  __device__ inline int  size()        const { return warpSize; }
  // AMD GPU: all lanes of a warp execute in lockstep; wave_barrier is the
  // intrinsic synchronization primitive for a single wavefront.
  __device__ inline void sync()              { __builtin_amdgcn_wave_barrier(); }
};

struct ccoThreadGroup {
  __device__ inline int  thread_rank() const { return 0; }
  __device__ inline int  size()        const { return 1; }
  __device__ inline void sync()              { /* empty: a 1-thread group is trivially synced */ }
};

}  // namespace cco
}  // namespace mori

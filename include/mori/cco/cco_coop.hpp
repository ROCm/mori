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

// Concrete group types used as the `Coop` template arg of
// ccoLsaBarrierSession<Coop>. Each must provide:
//   __device__ int  thread_rank() const   // rank within the group
//   __device__ int  size()        const   // number of threads in the group
//   __device__ void sync()                // group-internal sync barrier
//
// They are intentionally NOT derived from a virtual base: device-side
// virtual dispatch is problematic on AMD GPU (vtable placement, devirt
// reliability), and ccoLsaBarrierSession is a template — polymorphism is
// not required.

struct ccoCoopThread {
  __device__ int thread_rank() const { return 0; }
  __device__ int size() const { return 1; }
  __device__ void sync() {}
};

struct ccoCoopWarp {
  __device__ int thread_rank() const { return threadIdx.x % warpSize; }
  __device__ int size() const { return warpSize; }
  __device__ void sync() { __syncwarp(); }
};

struct ccoCoopBlock {
  __device__ int thread_rank() const { return threadIdx.x; }
  __device__ int size() const { return blockDim.x; }
  __device__ void sync() { __syncthreads(); }
};

}  // namespace cco
}  // namespace mori

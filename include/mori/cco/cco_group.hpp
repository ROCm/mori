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

#pragma

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

struct CcoAnyGroup {
  virtual int thread_rank() const = 0;
  virtual int size() const = 0;
  virtual void sync() = 0;
  virtual ~CcoAnyGroup() = default;
};

struct CcoBlockGroup : CcoAnyGroup {
  __device__ inline int thread_rank() const override { return threadIdx.x; }
  __device__ inline int size() const override{return blockDim.x} __device__
      inline void sync() override {
    __syncthreads();
  }
};

struct CcoWarpGroup : CcoAnyGroup {
  __device__ inline int thread_rank() const override { return __lane_id(); }
  __device__ inline int size() const override { return __AMDGCN_WAVEFRONT_SIZE; }
  // AMD GPU一个warp的所有lane执行同一条指令
  __device__ inline void sync() override { __builtin_amdgcn_wave_barrier(); }
};

struct CcoThreadGroup : CcoAnyGroup {
  __device__ inline int thread_rank() const override { return 0; }
  __device__ inline int size() const override { return 1; }
  __device__ inline void sync() override { /* empty */ }
};

}  // namespace cco
}  // namespace mori

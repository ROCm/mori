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

// Device-side helpers shared across the CCO p2p benchmark kernels. Include only
// from HIP translation units.

#pragma once

#include "mori/cco/cco_scale_out.hpp"
#include "util.hpp"

namespace mori::cco::benchmark {

// Intra-block linear thread index.
__device__ inline int linear_tid() {
  return threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
}

// Strided element copy across [lane, nlanes): dst[i] = src[i] for i in [0, n).
template <typename T>
__device__ inline void lsa_copy_strided(T* __restrict__ dst, const T* __restrict__ src, size_t n,
                                        int lane, int nlanes) {
  for (size_t i = lane; i < n; i += static_cast<size_t>(nlanes)) {
    dst[i] = src[i];
  }
}

// GPU-internal cross-block barrier (single GPU, NOT cross-rank). All nblocks of
// this kernel launch rendezvous at the end of round i. This matches shmem's
// bw_cross_block_barrier_round exactly so the LSA bw measurement window includes
// the same per-round all-block sync — an apples-to-apples comparison against the
// shmem benchmark. (CCO's ccoLsaBarrierSession is a *cross-GPU* barrier and is
// not equivalent here: the LSA bw is unidirectional, only PE 0 runs the kernel.)
//
// counter_d[0] = arrival counter, counter_d[1] = phase counter. Call from ALL
// threads of ALL blocks with the same (counter_d, nblocks, i).
__device__ inline void bw_cross_block_barrier_round(volatile unsigned int* counter_d, int nblocks,
                                                    int i) {
  __syncthreads();
  if (linear_tid() == 0) {
    __threadfence();
    unsigned int c = atomicInc((unsigned int*)counter_d, 0xffffffffu);
    if (c == static_cast<unsigned int>(nblocks * (i + 1) - 1)) {
      counter_d[1] += 1u;
    }
    while (counter_d[1] != static_cast<unsigned int>(i + 1)) {
    }
  }
  __syncthreads();
}

}  // namespace mori::cco::benchmark

// CCO_GDA_DISPATCH is provided by mori/cco/cco_scale_out.hpp: GDA provider is
// fixed at build time (per-NIC, from MORI_DEVICE_NIC_*), so the macro just binds
// `constexpr auto P = CCO_GDA_BUILD_PROVIDER` and runs the statement — no
// runtime provider argument.

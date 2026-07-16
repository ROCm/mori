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

// Contiguous copy kernel for FabricBackend transfers over a UALink super-node
// fabric. JIT-compiled via hipcc --genco at runtime and loaded as a .hsaco
// module (same mechanism as the XGMI scatter/gather kernel).
//
// Why a kernel and not hipMemcpy: on an *imported fabric pointer*,
// hipMemcpy(..., hipMemcpyDeviceToDevice) lowers to a pathological blit path
// that livelocks for large payloads (observed >= 4 MB on ROCm 7.15 / gfx1250).
// A vectorized grid-stride copy drives the fabric link directly — remote loads
// for a read, remote stores for a write — and is the path the 07_ualoe
// micro-benchmark uses to reach line rate.

#include <hip/hip_runtime.h>

#include <cstdint>

// dst/src are byte pointers; either may alias remote (imported) HBM. When both
// are 16-byte aligned the bulk is copied as uint4 (vectorized); a scalar tail
// handles the remainder and the fully-unaligned case.
extern "C" __global__ void fabricCopyKernel(char* __restrict__ dst, const char* __restrict__ src,
                                            size_t nbytes) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t nthreads = static_cast<size_t>(gridDim.x) * blockDim.x;

  uintptr_t s = reinterpret_cast<uintptr_t>(src);
  uintptr_t d = reinterpret_cast<uintptr_t>(dst);

  if ((s & 15) == 0 && (d & 15) == 0 && nbytes >= 16) {
    size_t nvec = nbytes / 16;
    size_t tailStart = nvec * 16;

    const uint4* srcV = reinterpret_cast<const uint4*>(src);
    uint4* dstV = reinterpret_cast<uint4*>(dst);
    for (size_t i = tid; i < nvec; i += nthreads) {
      dstV[i] = srcV[i];
    }
    for (size_t i = tailStart + tid; i < nbytes; i += nthreads) {
      dst[i] = src[i];
    }
  } else {
    for (size_t i = tid; i < nbytes; i += nthreads) {
      dst[i] = src[i];
    }
  }
}

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

#include <hip/hip_runtime.h>

namespace mori {
namespace io {

// Batched scatter/gather memory copy kernel. Processes multiple discrete copy
// operations in a single kernel launch, avoiding per-operation hipMemcpy
// overhead that dominates when there are many small non-contiguous segments.
//
// Layout of metadata buffer: [srcOffsets... | dstOffsets... | sizes...]
// Each array has numEntries elements of type size_t, packed contiguously.
__global__ void scatterGatherCopyKernel(const char* __restrict__ srcBase,
                                        char* __restrict__ dstBase,
                                        const size_t* __restrict__ srcOffsets,
                                        const size_t* __restrict__ dstOffsets,
                                        const size_t* __restrict__ entrySizes, int numEntries) {
  for (int e = blockIdx.x; e < numEntries; e += gridDim.x) {
    const char* src = srcBase + srcOffsets[e];
    char* dst = dstBase + dstOffsets[e];
    const size_t len = entrySizes[e];

    uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst);

    if ((srcAddr & 15) == 0 && (dstAddr & 15) == 0 && len >= 16) {
      const size_t numVec = len / 16;
      const size_t tailStart = numVec * 16;

      const int4* srcV = reinterpret_cast<const int4*>(src);
      int4* dstV = reinterpret_cast<int4*>(dst);

      for (size_t i = threadIdx.x; i < numVec; i += blockDim.x) {
        dstV[i] = srcV[i];
      }
      for (size_t i = tailStart + threadIdx.x; i < len; i += blockDim.x) {
        dst[i] = src[i];
      }
    } else {
      for (size_t i = threadIdx.x; i < len; i += blockDim.x) {
        dst[i] = src[i];
      }
    }
  }
}

}  // namespace io
}  // namespace mori

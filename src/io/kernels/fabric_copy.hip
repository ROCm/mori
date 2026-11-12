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

// Vectorize the bulk whenever both pointers can reach the same alignment.
// Prefix and tail stay inside the authorized byte range; never round a read
// down into a neighboring grant or write past the destination range.
template <class Word>
__device__ __forceinline__ void CopyAligned(char* dst, const char* src, size_t bytes, size_t tid,
                                            size_t stride) {
  constexpr size_t width = sizeof(Word);
  const size_t align = (width - (reinterpret_cast<uintptr_t>(src) & (width - 1))) & (width - 1);
  const size_t prefix = bytes < align ? bytes : align;
  for (size_t i = tid; i < prefix; i += stride) dst[i] = src[i];
  const size_t count = (bytes - prefix) / width;
  const auto* input = reinterpret_cast<const Word*>(src + prefix);
  auto* output = reinterpret_cast<Word*>(dst + prefix);
  for (size_t i = tid; i < count; i += stride) output[i] = input[i];
  for (size_t i = prefix + count * width + tid; i < bytes; i += stride) dst[i] = src[i];
}

__device__ __forceinline__ void CopyRange(char* dst, const char* src, size_t bytes, size_t tid,
                                          size_t stride) {
  const uintptr_t different = reinterpret_cast<uintptr_t>(src) ^ reinterpret_cast<uintptr_t>(dst);
  if ((different & 15) == 0)
    CopyAligned<uint4>(dst, src, bytes, tid, stride);
  else if ((different & 7) == 0)
    CopyAligned<uint2>(dst, src, bytes, tid, stride);
  else if ((different & 3) == 0)
    CopyAligned<uint32_t>(dst, src, bytes, tid, stride);
  else if ((different & 1) == 0)
    CopyAligned<uint16_t>(dst, src, bytes, tid, stride);
  else
    for (size_t i = tid; i < bytes; i += stride) dst[i] = src[i];
}

extern "C" __global__ void fabricCopyKernel(char* __restrict__ dst, const char* __restrict__ src,
                                            size_t nbytes) {
  CopyRange(dst, src, nbytes, static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(gridDim.x) * blockDim.x);
}

struct FabricCopySegment {
  const char* src;
  char* dst;
  size_t size;
};

// A CTA owns one independent segment. Multiple CTAs per segment hide remote
// load latency; grid-stride work handles arbitrary lengths and scalar tails.
extern "C" __global__ void fabricBatchCopyKernel(const FabricCopySegment* segments, size_t count,
                                                 unsigned blocksPerSegment) {
  const size_t segment = blockIdx.x / blocksPerSegment;
  if (segment >= count) return;
  const auto item = segments[segment];
  const size_t tid = (blockIdx.x % blocksPerSegment) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blocksPerSegment) * blockDim.x;
  CopyRange(item.dst, item.src, item.size, tid, stride);
}

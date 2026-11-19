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

// Batched copy kernel for FabricBackend transfers over a UALink super-node
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

#include "src/io/kernels/fabric_copy_abi.hpp"

using namespace mori::io::fabric;

namespace {
// Compile-time memory-level parallelism; keep all U loads ahead of their stores.
constexpr int kLoadsInFlight = 4;
// Recursive inlining gives each in-flight vector its own SSA value. A local
// uint4[U] can be lowered to LDS by the compiler even with unrolled loops.
template <int U>
__device__ __forceinline__ void CopyVectors(uint4* dst, const uint4* src, uint64_t index,
                                            uint64_t vectors) {
  uint4 value;
  if (index < vectors) value = src[index];
  if constexpr (U > 1) CopyVectors<U - 1>(dst, src, index + blockDim.x, vectors);
  if (index < vectors) dst[index] = value;
}

template <int U = kLoadsInFlight>
__device__ __forceinline__ void CopyChunk(unsigned char* dst, const unsigned char* src,
                                          uint64_t bytes) {
  const uint64_t tid = threadIdx.x;
  if ((reinterpret_cast<uintptr_t>(src) & 15) != (reinterpret_cast<uintptr_t>(dst) & 15)) {
    for (uint64_t i = tid; i < bytes; i += blockDim.x) dst[i] = src[i];
    return;
  }
  uint64_t head = (16 - (reinterpret_cast<uintptr_t>(src) & 15)) & 15;
  if (head > bytes) head = bytes;
  for (uint64_t i = tid; i < head; i += blockDim.x) dst[i] = src[i];
  src += head;
  dst += head;
  bytes -= head;
  const uint64_t vectors = bytes / 16;
  auto* vd = reinterpret_cast<uint4*>(dst);
  const auto* vs = reinterpret_cast<const uint4*>(src);
  for (uint64_t base = tid; base < vectors; base += uint64_t{U} * blockDim.x) {
    CopyVectors<U>(vd, vs, base, vectors);
  }
  for (uint64_t i = vectors * 16 + tid; i < bytes; i += blockDim.x) dst[i] = src[i];
}

// Whole-segment access pattern matches the original single-copy kernel.
__device__ __forceinline__ void CopyFlat(unsigned char* dst, const unsigned char* src,
                                        uint64_t bytes) {
  const uint64_t tid = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t{gridDim.x} * blockDim.x;
  if ((reinterpret_cast<uintptr_t>(src) & 15) != (reinterpret_cast<uintptr_t>(dst) & 15)) {
    for (uint64_t i = tid; i < bytes; i += stride) dst[i] = src[i];
    return;
  }
  uint64_t head = (16 - (reinterpret_cast<uintptr_t>(src) & 15)) & 15;
  if (head > bytes) head = bytes;
  for (uint64_t i = tid; i < head; i += stride) dst[i] = src[i];
  src += head;
  dst += head;
  bytes -= head;
  const uint64_t vectors = bytes / 16;
  auto* vd = reinterpret_cast<uint4*>(dst);
  const auto* vs = reinterpret_cast<const uint4*>(src);
  for (uint64_t i = tid; i < vectors; i += stride) vd[i] = vs[i];
  for (uint64_t i = vectors * 16 + tid; i < bytes; i += stride) dst[i] = src[i];
}

// Shared template implementation for all three exported kernel entry points.
template <int K>
__device__ __forceinline__ void FabricCopy(const FabricCopyArgs<K>& a) {
  if (a.numSegs == 1) {
    CopyFlat(reinterpret_cast<unsigned char*>(a.dst[0]),
             reinterpret_cast<const unsigned char*>(a.src[0]), a.size[0]);
    return;
  }
  const uint64_t chunkBytes = a.chunkShift == 64 ? UINT64_MAX : uint64_t{1} << a.chunkShift;
  for (uint64_t c = blockIdx.x; c < a.chunkEnd[a.numSegs - 1]; c += gridDim.x) {
    uint32_t seg = 0;
    {
      uint32_t lo = 0, hi = a.numSegs;
      while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (a.chunkEnd[mid] <= c)
          lo = mid + 1;
        else
          hi = mid;
      }
      seg = lo;
    }
    uint64_t first = seg == 0 ? 0 : a.chunkEnd[seg - 1];
    uint64_t offset = a.chunkShift == 64 ? 0 : (c - first) << a.chunkShift;
    uint64_t bytes = a.size[seg] - offset;
    if (bytes > chunkBytes) bytes = chunkBytes;
    CopyChunk(reinterpret_cast<unsigned char*>(a.dst[seg] + offset),
              reinterpret_cast<const unsigned char*>(a.src[seg] + offset), bytes);
  }
}
}  // namespace

extern "C" __global__ void fabricCopyKernelSingle(FabricCopyArgs<kSingleK> a) { FabricCopy(a); }
extern "C" __global__ void fabricCopyKernelSmall(FabricCopyArgs<kSmallK> a) { FabricCopy(a); }
extern "C" __global__ void fabricCopyKernelLarge(FabricCopyArgs<kLargeK> a) { FabricCopy(a); }

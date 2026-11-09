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

// mode bit 0: the last band's first tile skips its add.

#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr uint32_t kGateAbiVersion = 2;
constexpr uint32_t kRolePublish = 1u;

struct GateView {
  uint32_t abiVersion, structSize, recordSize, count, rank, reserved;
  uint64_t epoch, heapBase, records;
};

struct GateRecord {
  uint32_t id, channel, roleFlags, publishers;
  uint64_t readyOffset, arrivalOffset, completionOffset, errorOffset;
};

static_assert(sizeof(GateView) == 48 && sizeof(GateRecord) == 48, "gate ABI sizes");

__device__ inline uint16_t PlusMinusOne(uint64_t seed, uint32_t row, uint32_t col) {
  uint32_t h = static_cast<uint32_t>(seed) * 2654435761u;
  h ^= row * 0x9E3779B1u;
  h ^= col * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return (h & 1u) ? 0x3F80 : 0xBF80;  // bf16 +1 / -1
}

extern "C" __global__ void TriggerProducer(uint64_t out, uint64_t n, uint64_t mt0, uint64_t mt1,
                                           uint64_t gateView, uint64_t gateTileBase, uint64_t seed,
                                           uint64_t mode) {
  const GateView* v = reinterpret_cast<const GateView*>(gateView);
  const uint32_t tile = static_cast<uint32_t>(gateTileBase) + blockIdx.y;
  if (v->abiVersion != kGateAbiVersion || v->reserved != 0 || tile >= v->count) __builtin_trap();
  const GateRecord* r =
      reinterpret_cast<const GateRecord*>(v->records + uint64_t(tile) * v->recordSize);
  if (r->id != tile || !(r->roleFlags & kRolePublish)) __builtin_trap();
  uint64_t* ready = reinterpret_cast<uint64_t*>(v->heapBase + r->readyOffset);

  const uint32_t col0 = blockIdx.x * mt0;
  const uint32_t width = (col0 + mt0 <= n) ? mt0 : n - col0;
  const uint32_t row0 = blockIdx.y * mt1;
  uint16_t* d = reinterpret_cast<uint16_t*>(out);
  for (uint32_t i = threadIdx.x; i < mt1 * width; i += blockDim.x) {
    const uint32_t row = row0 + i / width, col = col0 + i % width;
    d[uint64_t(row) * n + col] = PlusMinusOne(seed, row, col);
  }

  const bool skip = (mode & 1) && blockIdx.x == 0 && blockIdx.y == gridDim.y - 1;
  // Every wave drains its own stores, then one thread releases the band.
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  if (threadIdx.x == 0 && !skip)
    __hip_atomic_fetch_add(ready, uint64_t(1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// ref[row, col] = sum over the seeds of PlusMinusOne, in bf16 (exact: |sum| <= 16).
extern "C" __global__ void TriggerReference(uint64_t ref, uint64_t n, uint64_t rows, uint64_t seeds,
                                            uint64_t numSeeds) {
  const uint64_t* s = reinterpret_cast<const uint64_t*>(seeds);
  uint16_t* d = reinterpret_cast<uint16_t*>(ref);
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < rows * n;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t row = i / n, col = i % n;
    int sum = 0;
    for (uint64_t k = 0; k < numSeeds; k++) sum += PlusMinusOne(s[k], row, col) == 0x3F80 ? 1 : -1;
    const float f = static_cast<float>(sum);
    d[i] = static_cast<uint16_t>(__float_as_uint(f) >> 16);  // exact for small integers
  }
}

// Holds the stream for about `cycles` clocks: a deliberately slow consumer.
extern "C" __global__ void TriggerSpin(uint64_t cycles) {
  if (threadIdx.x != 0) return;
  const uint64_t start = clock64();
  while (clock64() - start < cycles) __builtin_amdgcn_s_sleep(8);
}

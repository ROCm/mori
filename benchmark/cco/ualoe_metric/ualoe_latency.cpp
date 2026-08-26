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
// Two-GPU one-way latency for a single operation, measured inside the kernel.
//
// One thread, one outstanding operation at a time: every op is followed by the wait that matches
// it, so an iteration is issue -> ack rather than an issue rate. That is the whole point -- the
// bandwidth sweeps next door say how fast a saturated link runs, and this says what one crossing
// costs.
//
// The local baseline runs the identical loop against the issuing GPU's own memory, so remote minus
// local isolates the link and cancels the cost of the instruction itself.
//
// Timed twice over the same interval, with both counters:
//   - s_memtime / readcyclecounter counts shader clocks, which move with DVFS, so cycles are only a
//     time once you know the clock for that run;
//   - wall_clock64() is the constant-rate reference clock and needs no such calibration.
// Reporting both is a cross-check: their ratio has to come out as the shader clock, and if it does
// not, one of the two counters is not measuring what it is assumed to.
//
// Usage: ualoe_latency [srcGpu] [dstGpu | -1 for the local baseline] [iters] [reps] [stride]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HC(x)                                                                  \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP FAIL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// gfx12 split s_waitcnt into per-counter waits and renamed the store mnemonics.
//
// The two are not equivalent for this measurement. gfx12's s_wait_storecnt is a real store counter
// and retires when the write is acknowledged, so the bare-store modes below measure a round trip.
// gfx9's vmcnt retires when the store leaves the CU, so on that arch the bare-store modes read the
// same locally and remotely and only the fence / atomic / read-back modes mean anything.
#if defined(__gfx1250__) || defined(__GFX12__)
#define WAIT_ST "s_wait_storecnt 0"
#define FLAT_ST "flat_store_b32"
#else
#define WAIT_ST "s_waitcnt vmcnt(0)"
#define FLAT_ST "flat_store_dword"
#endif

// sched_barrier on both sides: without it the compiler is free to sink the first read or hoist the
// second into the loop, and the interval stops being the thing that was meant.
__device__ __forceinline__ unsigned long long RdCycle() {
  __builtin_amdgcn_sched_barrier(0);
  unsigned long long t = __builtin_readcyclecounter();
  __builtin_amdgcn_sched_barrier(0);
  return t;
}

__device__ __forceinline__ unsigned long long RdWall() {
  __builtin_amdgcn_sched_barrier(0);
  unsigned long long t = wall_clock64();
  __builtin_amdgcn_sched_barrier(0);
  return t;
}

// hipDeviceAttributeWallClockRate reports 100000 kHz on gfx950 but 0 on gfx1250, and returns
// hipSuccess either way -- dividing by it silently yields inf. Measure the rate against the host
// instead, once per run.
__global__ void WallCalKernel(unsigned long long* out, unsigned long long ticks) {
  if (threadIdx.x || blockIdx.x) return;
  unsigned long long w0 = wall_clock64();
  while (wall_clock64() - w0 < ticks) {
  }
  out[0] = wall_clock64() - w0;
}

enum {
  kGlobalB128,
  kGlobalB32,
  kFlatB32,
  kBufferB32,
  kAtomicAdd,
  kAtomicAddNoRet,
  kStoreFence,
  kStoreReadback,
  kNumModes
};

static const char* kModeNames[kNumModes] = {
    "global_store_b128", "global_store_b32",  "flat_store_b32", "buffer_store_b32",
    "atomic_add(ret)",   "atomic_add(noret)", "store+sysfence", "store+readback"};

// Each mode gets its own `#pragma unroll 1` loop so iterations cannot overlap: with two stores in
// flight the measurement becomes a rate, and the number quietly halves.
__global__ void LatKernel(char* __restrict__ dst, unsigned long long* cyc, unsigned long long* tick,
                          int iters, int stride) {
  if (threadIdx.x || blockIdx.x) return;
  unsigned long long c0, c1, w0, w1;
  unsigned sink = 0;

#define TIME_BEGIN() \
  c0 = RdCycle();    \
  w0 = RdWall()
#define TIME_END(mode) \
  c1 = RdCycle();      \
  w1 = RdWall();       \
  cyc[mode] = c1 - c0; \
  tick[mode] = w1 - w0

  {
    uint4 v = make_uint4(1, 2, 3, 4);
    uint4* p = reinterpret_cast<uint4*>(dst);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      p[(i * stride) & 1023] = v;
      asm volatile(WAIT_ST ::: "memory");
    }
    TIME_END(kGlobalB128);
  }

  {
    unsigned* p = reinterpret_cast<unsigned*>(dst);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      p[(i * stride) & 4095] = static_cast<unsigned>(i);
      asm volatile(WAIT_ST ::: "memory");
    }
    TIME_END(kGlobalB32);
  }

  // Generic address space: the same store through flat rather than global addressing, which on some
  // parts takes a different path to memory.
  {
    unsigned* g = reinterpret_cast<unsigned*>(dst);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      unsigned idx = (i * stride) & 4095;
      asm volatile(FLAT_ST " %0, %1\n " WAIT_ST
                   :
                   : "v"(g + idx), "v"(static_cast<unsigned>(i))
                   : "memory");
    }
    TIME_END(kFlatB32);
  }

  {
#if __has_builtin(__builtin_amdgcn_make_buffer_rsrc)
    __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
        static_cast<void*>(dst), /*stride=*/0, /*num=*/1 << 20, /*flags=*/0);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      __builtin_amdgcn_raw_buffer_store_b32(static_cast<unsigned>(i), rsrc,
                                            ((i * stride) & 4095) * 4, 0, 0);
      asm volatile(WAIT_ST ::: "memory");
    }
    TIME_END(kBufferB32);
#else
    cyc[kBufferB32] = 0;
    tick[kBufferB32] = 0;
#endif
  }

  // Returns the old value, so the load half cannot retire until the peer has answered. This is a
  // genuine round trip on every arch and is the mode to trust when the bare stores are not usable.
  {
    unsigned* p = reinterpret_cast<unsigned*>(dst);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++)
      sink += __hip_atomic_fetch_add(p + ((i * stride) & 4095), 1u, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM);
    TIME_END(kAtomicAdd);
  }

  {
    unsigned* p = reinterpret_cast<unsigned*>(dst);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      __hip_atomic_fetch_add(p + ((i * stride) & 4095), 1u, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_SYSTEM);
      asm volatile(WAIT_ST ::: "memory");
    }
    TIME_END(kAtomicAddNoRet);
  }

  // System-scope release: on gfx9 this is what actually pushes the write out, which is why this
  // mode and not the bare store is the CU remote-write figure there.
  {
    unsigned* p = reinterpret_cast<unsigned*>(dst);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      p[(i * stride) & 4095] = static_cast<unsigned>(i);
      __threadfence_system();
    }
    TIME_END(kStoreFence);
  }

  // Two crossings, so this should land at about twice the delta the single-crossing modes report.
  // It is here as a self-consistency check on them rather than as a number in its own right.
  {
    volatile unsigned* p = reinterpret_cast<volatile unsigned*>(dst);
    TIME_BEGIN();
#pragma unroll 1
    for (int i = 0; i < iters; i++) {
      unsigned idx = (i * stride) & 4095;
      p[idx] = static_cast<unsigned>(i);
      sink += p[idx];
    }
    TIME_END(kStoreReadback);
  }

#undef TIME_BEGIN
#undef TIME_END

  cyc[kNumModes] = sink;  // keep the atomic and read-back results live
}

int main(int argc, char** argv) {
  int src = argc > 1 ? atoi(argv[1]) : 0;
  int dst_gpu = argc > 2 ? atoi(argv[2]) : 1;  // negative: local baseline
  int iters = argc > 3 ? atoi(argv[3]) : 20000;
  int reps = argc > 4 ? atoi(argv[4]) : 5;
  // Odd stride so consecutive iterations touch different cache lines; with stride 1 the second
  // store hits a line the first already owns and the number is a cache hit, not a crossing.
  int stride = argc > 5 ? atoi(argv[5]) : 17;

  const size_t kBufBytes = 1 << 22;
  const bool local = (dst_gpu < 0);
  char* buf = nullptr;

  if (local) {
    HC(hipSetDevice(src));
    HC(hipMalloc(&buf, kBufBytes));
  } else {
    int can = 0;
    HC(hipDeviceCanAccessPeer(&can, src, dst_gpu));
    if (!can) {
      printf("no peer access GPU%d -> GPU%d\n", src, dst_gpu);
      return 1;
    }
    HC(hipSetDevice(dst_gpu));
    HC(hipMalloc(&buf, kBufBytes));
    HC(hipSetDevice(src));
    hipError_t pe = hipDeviceEnablePeerAccess(dst_gpu, 0);
    if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) HC(pe);
  }
  HC(hipMemset(buf, 0, kBufBytes));
  HC(hipSetDevice(src));

  unsigned long long* cyc = nullptr;
  unsigned long long* tick = nullptr;
  unsigned long long* cal = nullptr;
  HC(hipMalloc(&cyc, (kNumModes + 1) * sizeof(unsigned long long)));
  HC(hipMalloc(&tick, (kNumModes + 1) * sizeof(unsigned long long)));
  HC(hipMalloc(&cal, sizeof(unsigned long long)));

  // ~1 s at the 100 MHz these parts run the constant clock at: long enough that launch and sync
  // overhead do not move the result.
  hipLaunchKernelGGL(WallCalKernel, dim3(1), dim3(1), 0, 0, cal, 10000000ULL);
  HC(hipDeviceSynchronize());
  auto cal_start = std::chrono::high_resolution_clock::now();
  hipLaunchKernelGGL(WallCalKernel, dim3(1), dim3(1), 0, 0, cal, 100000000ULL);
  HC(hipDeviceSynchronize());
  double cal_sec =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cal_start).count();
  unsigned long long cal_ticks = 0;
  HC(hipMemcpy(&cal_ticks, cal, sizeof(cal_ticks), hipMemcpyDeviceToHost));
  const double wall_mhz = cal_ticks / cal_sec / 1e6;

  hipLaunchKernelGGL(LatKernel, dim3(1), dim3(1), 0, 0, buf, cyc, tick, 2000, stride);
  HC(hipDeviceSynchronize());

  std::vector<std::vector<double>> per_cyc(kNumModes), per_tick(kNumModes);
  for (int r = 0; r < reps; r++) {
    hipLaunchKernelGGL(LatKernel, dim3(1), dim3(1), 0, 0, buf, cyc, tick, iters, stride);
    HC(hipDeviceSynchronize());
    unsigned long long hc[kNumModes + 1], ht[kNumModes + 1];
    HC(hipMemcpy(hc, cyc, sizeof(hc), hipMemcpyDeviceToHost));
    HC(hipMemcpy(ht, tick, sizeof(ht), hipMemcpyDeviceToHost));
    for (int m = 0; m < kNumModes; m++) {
      per_cyc[m].push_back(static_cast<double>(hc[m]) / iters);
      per_tick[m].push_back(static_cast<double>(ht[m]) / iters);
    }
  }

  int wall_attr = -1;
  (void)hipDeviceGetAttribute(&wall_attr, hipDeviceAttributeWallClockRate, src);
  printf("# issuer=GPU%d target=%s iters=%d reps=%d stride=%d\n", src, local ? "LOCAL" : "peer",
         iters, reps, stride);
  printf("# constant clock %.2f MHz measured against the host (attribute reports %d kHz)\n",
         wall_mhz, wall_attr);

  double sum_cyc = 0, sum_tick = 0;
  for (int m = 0; m < kNumModes; m++) {
    sum_cyc += per_cyc[m][0];
    sum_tick += per_tick[m][0];
  }
  printf("# shader clock implied by the two counters: %.3f GHz\n",
         sum_tick > 0 ? sum_cyc / sum_tick * wall_mhz / 1e3 : 0.0);
  printf("%-20s %12s %12s %10s\n", "op", "cycles/op", "ticks/op", "us/op");
  for (int m = 0; m < kNumModes; m++) {
    std::sort(per_cyc[m].begin(), per_cyc[m].end());
    std::sort(per_tick[m].begin(), per_tick[m].end());
    double c = per_cyc[m][per_cyc[m].size() / 2];
    double t = per_tick[m][per_tick[m].size() / 2];
    printf("%-20s %12.1f %12.1f %10.3f\n", kModeNames[m], c, t, t / wall_mhz);
  }
  return 0;
}

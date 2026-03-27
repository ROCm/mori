#include <hip/hip_runtime.h>

#include <cstdio>

#include "mori/core/transport/p2p/device_primitives.hpp"

template <int Unroll>
__global__ void WarpCopyKernel(uint8_t* dst, uint8_t* src, size_t nbytes) {
  mori::core::WarpCopy<uint8_t, Unroll>(dst, src, nbytes);
}

void BenchWarpCopy(size_t nbytes, int iters = 100) {
  uint8_t *d_src, *d_dst;
  hipMalloc(&d_src, nbytes);
  hipMalloc(&d_dst, nbytes);
  hipMemset(d_src, 0xAB, nbytes);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Warmup
  WarpCopyKernel<1><<<1, 64>>>(d_dst, d_src, nbytes);
  hipDeviceSynchronize();

  // Bench Unroll=1
  hipEventRecord(start);
  for (int i = 0; i < iters; i++) {
    WarpCopyKernel<1><<<1, 64>>>(d_dst, d_src, nbytes);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float ms1 = 0;
  hipEventElapsedTime(&ms1, start, stop);

  // Warmup
  WarpCopyKernel<16><<<1, 64>>>(d_dst, d_src, nbytes);
  hipDeviceSynchronize();

  // Bench Unroll=4
  hipEventRecord(start);
  for (int i = 0; i < iters; i++) {
    WarpCopyKernel<8><<<1, 64>>>(d_dst, d_src, nbytes);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float ms4 = 0;
  hipEventElapsedTime(&ms4, start, stop);

  printf("WarpCopy %zu bytes x %d iters:\n", nbytes, iters);
  printf("  Unroll=1: %.3f ms total, %.3f us/iter\n", ms1, ms1 * 1000.0f / iters);
  printf("  Unroll=4: %.3f ms total, %.3f us/iter\n", ms4, ms4 * 1000.0f / iters);
  printf("  Speedup:  %.2fx\n", ms1 / ms4);

  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipFree(d_src);
  hipFree(d_dst);
}

__global__ void BlockCopyKernel(uint8_t* dst, uint8_t* src, size_t nbytes) {
  mori::core::BlockCopy<uint8_t>(dst, src, nbytes);
}

void TestBlockCopy(size_t nbytes) {
  uint8_t *d_src, *d_dst;
  hipMalloc(&d_src, nbytes);
  hipMalloc(&d_dst, nbytes);
  hipMemset(d_src, 0xAB, nbytes);
  hipMemset(d_dst, 0, nbytes);

  BlockCopyKernel<<<1, 256>>>(d_dst, d_src, nbytes);
  hipDeviceSynchronize();

  uint8_t* h_dst = new uint8_t[nbytes];
  hipMemcpy(h_dst, d_dst, nbytes, hipMemcpyDeviceToHost);

  bool pass = true;
  for (size_t i = 0; i < nbytes; i++) {
    if (h_dst[i] != 0xAB) {
      printf("  FAILED at byte %zu: expected 0xAB, got 0x%02X\n", i, h_dst[i]);
      pass = false;
      break;
    }
  }
  printf("BlockCopy(%zu bytes): %s\n", nbytes, pass ? "PASSED" : "FAILED");

  delete[] h_dst;
  hipFree(d_src);
  hipFree(d_dst);
}

int main() {
  // BenchWarpCopy(204800000);
  TestBlockCopy(20000);
  return 0;
}

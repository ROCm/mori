// Compact / expand between variant A's segmented landing zone and a dense run.
//
//   compact: dense[r]                 = recv[s*stride + (r - prefix_s)]
//   expand : combine_in[s*stride + k] = dense[prefix_s + k]
//
// `counts` is the device int[world] that dispatch now exports; prefix sums are
// recomputed per block (world is 4, so this is a handful of scalar adds and is
// cheaper than a separate pass or a host round trip).
#include <hip/hip_runtime.h>
#include <cstdint>

template <bool COMPACT>
__global__ void k_move(uint4* __restrict__ seg, uint4* __restrict__ dense,
                       const int* __restrict__ counts, int world, int stride, int vpr) {
  // Prefix sums once per BLOCK into LDS. Recomputing them per thread cost ~8
  // dependent global loads each, and with one row per block moving 56 B/thread the
  // kernel is pure latency -- that indirection was 2.3x the whole copy.
  extern __shared__ int pre[];   // world + 1 entries
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int s = 0; s < world; ++s) { pre[s] = acc; acc += counts[s]; }
    pre[world] = acc;
  }
  __syncthreads();
  const int total = pre[world];
  for (int r = blockIdx.x; r < total; r += gridDim.x) {
    int s = 0;
    while (s + 1 < world && r >= pre[s + 1]) ++s;
    uint4* a = seg + (size_t)(s * stride + (r - pre[s])) * vpr;
    uint4* b = dense + (size_t)r * vpr;
    uint4* src = COMPACT ? a : b;
    uint4* dst = COMPACT ? b : a;
    for (int i = threadIdx.x; i < vpr; i += blockDim.x) dst[i] = src[i];
  }
}

extern "C" {
// rowBytes must be a multiple of 16; both fp4 (3584) and bf16 (14336) are.
int ep_compact(void* seg, void* dense, const void* counts, int world, int stride,
               int rowBytes, int maxRows, void* stream) {
  if (rowBytes % 16) return -1;
  int grid = maxRows < 4096 ? maxRows : 4096;
  hipLaunchKernelGGL((k_move<true>), dim3(grid), dim3(256), (world + 1) * sizeof(int), (hipStream_t)stream,
                     (uint4*)seg, (uint4*)dense, (const int*)counts, world, stride, rowBytes / 16);
  return 0;
}
int ep_expand(void* seg, void* dense, const void* counts, int world, int stride,
              int rowBytes, int maxRows, void* stream) {
  if (rowBytes % 16) return -1;
  int grid = maxRows < 4096 ? maxRows : 4096;
  hipLaunchKernelGGL((k_move<false>), dim3(grid), dim3(256), (world + 1) * sizeof(int), (hipStream_t)stream,
                     (uint4*)seg, (uint4*)dense, (const int*)counts, world, stride, rowBytes / 16);
  return 0;
}
}

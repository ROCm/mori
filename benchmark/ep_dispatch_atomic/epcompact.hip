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

// The payload is not the only thing dispatch leaves segmented: the per-slot
// reverse map (disp_tok_id_to_src_tok_id_local) is sized at the recv capacity and
// carries the same gaps, so a consumer reading dense rows needs it in dense order
// too. Same permutation, 4 B rows instead of 3584 -- too narrow for the uint4 path
// above, so it gets one thread per row rather than one block.
template <bool COMPACT>
__global__ void k_move_i32(int* __restrict__ seg, int* __restrict__ dense,
                           const int* __restrict__ counts, int world, int stride) {
  extern __shared__ int pre[];
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int s = 0; s < world; ++s) { pre[s] = acc; acc += counts[s]; }
    pre[world] = acc;
  }
  __syncthreads();
  const int total = pre[world];
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < total;
       r += gridDim.x * blockDim.x) {
    int s = 0;
    while (s + 1 < world && r >= pre[s + 1]) ++s;
    const int slot = s * stride + (r - pre[s]);
    if (COMPACT) dense[r] = seg[slot];
    else         seg[slot] = dense[r];
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
// int32-per-row variant for the reverse map. maxRows sizes the grid only; the
// real bound still comes from counts[] on device.
int ep_compact_idx(void* seg, void* dense, const void* counts, int world, int stride,
                   int maxRows, void* stream) {
  int grid = (maxRows + 255) / 256;
  if (grid < 1) grid = 1;
  hipLaunchKernelGGL((k_move_i32<true>), dim3(grid), dim3(256), (world + 1) * sizeof(int), (hipStream_t)stream,
                     (int*)seg, (int*)dense, (const int*)counts, world, stride);
  return 0;
}
int ep_expand_idx(void* seg, void* dense, const void* counts, int world, int stride,
                  int maxRows, void* stream) {
  int grid = (maxRows + 255) / 256;
  if (grid < 1) grid = 1;
  hipLaunchKernelGGL((k_move_i32<false>), dim3(grid), dim3(256), (world + 1) * sizeof(int), (hipStream_t)stream,
                     (int*)seg, (int*)dense, (const int*)counts, world, stride);
  return 0;
}
}

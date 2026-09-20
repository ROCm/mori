// Cost of compacting the variant-A segmented landing zone into a dense run,
// with real kernels instead of torch's index_select / index_put_.
//
// Both row widths divide by 16, so everything moves as uint4. Note the indices
// are SEQUENTIAL inside a segment (idx[r] = r + delta_s), so the gather's reads
// and the scatter's writes are already contiguous runs -- these kernels are the
// segmented block copy, just expressed through an index vector.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

__global__ void k_gather(uint4* __restrict__ dst, const uint4* __restrict__ src,
                         const int* __restrict__ idx, int nrow, int vpr) {
  for (int r = blockIdx.x; r < nrow; r += gridDim.x) {
    const uint4* s = src + (size_t)idx[r] * vpr;
    uint4* d = dst + (size_t)r * vpr;
    for (int i = threadIdx.x; i < vpr; i += blockDim.x) d[i] = s[i];
  }
}
__global__ void k_scatter(uint4* __restrict__ src, const uint4* __restrict__ dst,
                          const int* __restrict__ idx, int nrow, int vpr) {
  for (int r = blockIdx.x; r < nrow; r += gridDim.x) {
    uint4* s = src + (size_t)idx[r] * vpr;
    const uint4* d = dst + (size_t)r * vpr;
    for (int i = threadIdx.x; i < vpr; i += blockDim.x) s[i] = d[i];
  }
}
__global__ void k_copy(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

#define CHK(x) do { hipError_t e = (x); if (e) { printf("ERR %s\n", hipGetErrorString(e)); return 1; } } while (0)

int main() {
  const int CAP = 2048, WORLD = 4, stride = CAP / WORLD;
  const int cnt[4] = {426, 421, 428, 426};
  int VALID = 0;
  std::vector<int> idx;
  for (int s = 0; s < WORLD; ++s)
    for (int k = 0; k < cnt[s]; ++k) { idx.push_back(s * stride + k); ++VALID; }


  // REP copies of the whole problem, laid out back to back, so the working set can
  // be pushed past any last-level cache. If the per-byte cost is flat in REP the
  // small case was not merely cache-resident.
  const int REP = getenv("REP") ? atoi(getenv("REP")) : 1;
  const int rowB[2] = {3584, 14336};
  const char* nm[2] = {"fp4", "bf16"};
  printf("%-6s %7s %9s %11s %11s %9s   %s\n", "dtype", "row B", "MB", "gather us",
         "scatter us", "copy us", "gather GB/s r+w");
  printf("--------------------------------------------------------------------------------\n");
  for (int t = 0; t < 2; ++t) {
    int vpr = rowB[t] / 16;
    int NROW = VALID * REP;
    std::vector<int> bigidx;
    for (int r = 0; r < REP; ++r)
      for (int i = 0; i < VALID; ++i) bigidx.push_back(r * CAP + idx[i]);
    int* d_idx;
    CHK(hipMalloc(&d_idx, NROW * sizeof(int)));
    CHK(hipMemcpy(d_idx, bigidx.data(), NROW * sizeof(int), hipMemcpyHostToDevice));
    uint4 *src, *dst, *ref;
    CHK(hipMalloc(&src, (size_t)CAP * REP * rowB[t]));
    CHK(hipMalloc(&dst, (size_t)NROW * rowB[t]));
    CHK(hipMalloc(&ref, (size_t)NROW * rowB[t]));
    CHK(hipMemset(src, 1, (size_t)CAP * REP * rowB[t]));
    int grid = NROW < 4096 ? NROW : 4096;
    const int VALIDl = NROW;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    float ms; const int IT = 300;
    auto run = [&](int which) {
      for (int i = 0; i < 30; ++i) {
        if (which == 0) hipLaunchKernelGGL(k_gather, dim3(grid), dim3(256), 0, 0, dst, src, d_idx, VALIDl, vpr);
        else if (which == 1) hipLaunchKernelGGL(k_scatter, dim3(grid), dim3(256), 0, 0, src, dst, d_idx, VALIDl, vpr);
        else hipLaunchKernelGGL(k_copy, dim3(grid), dim3(256), 0, 0, ref, dst, (size_t)VALIDl * vpr);
      }
      hipDeviceSynchronize();
      hipEventRecord(a);
      for (int i = 0; i < IT; ++i) {
        if (which == 0) hipLaunchKernelGGL(k_gather, dim3(grid), dim3(256), 0, 0, dst, src, d_idx, VALIDl, vpr);
        else if (which == 1) hipLaunchKernelGGL(k_scatter, dim3(grid), dim3(256), 0, 0, src, dst, d_idx, VALIDl, vpr);
        else hipLaunchKernelGGL(k_copy, dim3(grid), dim3(256), 0, 0, ref, dst, (size_t)VALIDl * vpr);
      }
      hipEventRecord(b); hipEventSynchronize(b);
      hipEventElapsedTime(&ms, a, b);
      return ms * 1000.0 / IT;
    };
    double g = run(0), s = run(1), c = run(2);
    double mb = (double)VALIDl * rowB[t] / 1e6;
    printf("%-6s %7d %9.1f %11.2f %11.2f %9.2f   %.0f\n", nm[t], rowB[t], mb, g, s, c,
           2.0 * VALIDl * rowB[t] / 1e9 / (g * 1e-6));
    hipFree(src); hipFree(dst); hipFree(ref); hipFree(d_idx);
  }
  return 0;
}

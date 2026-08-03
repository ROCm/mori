// The blockwise quantise pass, alone, on one GPU.
//
// WHY IT EXISTS. In the fp8_blockwise combine the quantise is the single largest term -- 408.2us
// inline at 64 blocks, 163.8 at 256 (full minus MORI_COMB_NOQUANT), against a ~50us floor for the
// 322 MB it touches -- but every one of those numbers is a DIFFERENCE between two runs of a kernel
// that also gathers, folds and barriers. This runs the pass and nothing else, so the number is the
// pass, and the knobs below say which part of it costs what.
//
// It calls mori's OWN device helper rather than a copy of it. A hand-rolled quantise would be a
// measurement of my transcription; WarpQuantizeToFp8Blockwise is the code that actually ships.
//
// THE HYPOTHESIS IT IS BUILT TO TEST. combineInp and shmemInpScalesMemObj are both
// hipDeviceMallocUncached (dispatch_combine.cpp:324,378), because a peer has to be able to read
// them over the fabric. So the quantise streams 212 MB out of ordinary HBM and writes 106 MB of
// fp8 plus 3 MB of scales into UNCACHED memory. The bf16 baseline it has to beat writes none of
// that: at ZC=1 the caller's tensor already IS the registered buffer, so bf16 combine never stores
// a payload at all. Q_UNC below is the A/B for exactly that: same kernel, same geometry, only the
// allocator changes.
//
// MODES (Q_MODE)
//   0 quant      the shipping pass: WarpQuantizeToFp8Blockwise per token per warp
//   1 quant-nosc same, scales written to LDS instead of memory -- prices the scale STORES only
//   2 cast       read the token, write fp8, no block max and no scales -- the streaming floor
//                with the same byte counts as mode 0
//   3 read       read the token, write 4 B per warp -- the read side alone
//   4 write      write the fp8 bytes from a register constant -- the uncached write side alone
//
// Build: hipcc --offload-arch=gfx1250 -std=c++17 -O3 -I<mori>/include q_micro.cc -o q_micro
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mori/core/transport/p2p/device_primitives.hpp"

#define CK(x)                                                                                      \
  do {                                                                                             \
    hipError_t _e = (x);                                                                           \
    if (_e != hipSuccess) {                                                                        \
      printf("HIP err %d (%s) at %d\n", (int)_e, hipGetErrorString(_e), __LINE__);                 \
      fflush(stdout);                                                                              \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

using Fp8T = mori::core::CombineInternalFp8;
using Bf16 = hip_bfloat16;

template <int MODE>
__global__ __launch_bounds__(1024) void qmicro(Fp8T* __restrict__ dst, float* __restrict__ scales,
                                               const Bf16* __restrict__ src, int nTok, int hid,
                                               int scaleDim) {
  const int warpNum = blockDim.x / warpSize;
  const int warpId = threadIdx.x / warpSize;
  const int laneId = threadIdx.x % warpSize;
  const int gwid = blockIdx.x * warpNum + warpId;
  const int gwn = gridDim.x * warpNum;
  // Mode 1 needs somewhere to put the scales that is not memory. One block's worth of scale slots
  // per warp; scaleDim is 56 at the shipping shape.
  extern __shared__ float sScratch[];

  for (int t = gwid; t < nTok; t += gwn) {
    Fp8T* d = dst + (size_t)t * hid;
    const Bf16* s = src + (size_t)t * hid;
    if (MODE == 0) {
      mori::core::WarpQuantizeToFp8Blockwise<Fp8T>(d, scales + (size_t)t * scaleDim, s, hid,
                                                   scaleDim);
    } else if (MODE == 1) {
      mori::core::WarpQuantizeToFp8Blockwise<Fp8T>(d, sScratch + (size_t)warpId * scaleDim, s, hid,
                                                   scaleDim);
    } else if (MODE == 2) {
      // Same bytes in and out as mode 0, none of the block reduction: 16 B of bf16 in, 8 B of fp8
      // out, per lane per step.
      for (int e = laneId * 8; e < hid; e += warpSize * 8) {
        float4 v = *reinterpret_cast<const float4*>(s + e);
        const Bf16* p = reinterpret_cast<const Bf16*>(&v);
        union {
          uint64_t w;
          __hip_fp8_storage_t b[8];
        };
#pragma unroll
        for (int k = 0; k < 8; ++k)
          b[k] = __hip_cvt_float_to_fp8((float)p[k], __HIP_SATFINITE, __HIP_E4M3);
        *reinterpret_cast<uint64_t*>(reinterpret_cast<__hip_fp8_storage_t*>(d) + e) = w;
      }
    } else if (MODE == 3) {
      float acc = 0.f;
      for (int e = laneId * 8; e < hid; e += warpSize * 8) {
        float4 v = *reinterpret_cast<const float4*>(s + e);
        const Bf16* p = reinterpret_cast<const Bf16*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc += (float)p[k];
      }
      if (acc == 1234.5678f && laneId == 0) scales[t] = acc;  // never true; keeps the loads alive
    } else if (MODE == 4) {
      const uint64_t w = 0x3838383838383838ull;
      for (int e = laneId * 8; e < hid; e += warpSize * 8)
        *reinterpret_cast<uint64_t*>(reinterpret_cast<__hip_fp8_storage_t*>(d) + e) = w;
    }
  }
}

// Verification runs on the DEVICE so it goes through the same fp8 conversion the fold uses; a host
// decode would be my own reading of the format, which is the thing least worth trusting here.
__global__ void qcheck(const Fp8T* __restrict__ dst, const float* __restrict__ scales,
                       const Bf16* __restrict__ src, int nTok, int hid, int scaleDim,
                       unsigned* __restrict__ worst) {
  const int blockElems = (hid + scaleDim - 1) / scaleDim;
  const int t = blockIdx.x;
  if (t >= nTok) return;
  for (int e = threadIdx.x; e < hid; e += blockDim.x) {
    const int b = e / blockElems;
    float sc = scales[(size_t)t * scaleDim + b];
    if (b == 0 && sc < 0.f) sc = -sc;  // producer's "really was scaled" sentinel
    const float want = (float)src[(size_t)t * hid + e];
    const float got = (float)dst[(size_t)t * hid + e] * sc;
    if (fabsf(want) > 0.05f) {
      const float rel = fabsf(got - want) / fabsf(want);
      atomicMax(worst, __float_as_uint(rel));
    }
  }
}

static int envi(const char* n, int d) {
  const char* v = getenv(n);
  return (v && *v) ? atoi(v) : d;
}

using KFn = void (*)(Fp8T*, float*, const Bf16*, int, int, int);

int main() {
  hipDeviceProp_t props{};
  CK(hipGetDeviceProperties(&props, 0));
  // Not props.warpSize: on ROCm 7.15 `warpSize` is a macro, so the member access does not parse.
  int warpSz = 0;
  CK(hipDeviceGetAttribute(&warpSz, hipDeviceAttributeWarpSize, 0));
  printf("[info] %s arch=%s CU=%d warpSize=%d\n", props.name, props.gcnArchName,
         props.multiProcessorCount, warpSz);

  const int hid = envi("Q_HID", 7168);
  // 14848 x 7168 bf16 = 212.9 MB, the local bf16 the real pass streams at EP4 4096 tokens top-8.
  const int nTok = envi("Q_TOK", 14848);
  const int scaleDim = envi("Q_SCALEDIM", 56);  // hid / 128
  const int iters = envi("Q_ITERS", 20);
  // 0 = everything cached (hipMalloc), 1 = fp8 destination uncached, 2 = destination AND scales
  // uncached, which is what the shipping allocator does.
  const int unc = envi("Q_UNC", 2);
  const int check = envi("Q_CHECK", 1);

  const size_t srcElems = (size_t)nTok * hid;
  const size_t scaleElems = (size_t)nTok * scaleDim;
  auto alloc = [&](void** p, size_t bytes, bool uncached) {
    if (uncached)
      CK(hipExtMallocWithFlags(p, bytes, hipDeviceMallocUncached));
    else
      CK(hipMalloc(p, bytes));
  };

  Bf16* dSrc = nullptr;
  Fp8T* dDst = nullptr;
  float* dSc = nullptr;
  CK(hipMalloc(&dSrc, srcElems * sizeof(Bf16)));  // the caller's tensor: ordinary device memory
  alloc((void**)&dDst, srcElems * sizeof(Fp8T), unc >= 1);
  alloc((void**)&dSc, scaleElems * sizeof(float), unc >= 2);

  Bf16* hSrc = (Bf16*)malloc(srcElems * sizeof(Bf16));
  srand(4321);
  for (size_t i = 0; i < srcElems; ++i) {
    // Spread over a few orders of magnitude so the per-block max is not degenerate.
    float f = ((float)(rand() % 2001) - 1000.f) / 1000.f;
    if ((i % 128) == 7) f *= 40.f;
    hSrc[i] = (Bf16)f;
  }
  CK(hipMemcpy(dSrc, hSrc, srcElems * sizeof(Bf16), hipMemcpyHostToDevice));
  unsigned* dWorst = nullptr;
  CK(hipMalloc(&dWorst, sizeof(unsigned)));

  const double mbSrc = srcElems * 2 / 1e6, mbDst = srcElems / 1e6, mbSc = scaleElems * 4 / 1e6;
  printf("[cfg] hid=%d tok=%d scaleDim=%d unc=%d  read %.1f MB  write %.1f MB fp8 + %.1f MB "
         "scales\n",
         hid, nTok, scaleDim, unc, mbSrc, mbDst, mbSc);

  KFn kern[5] = {qmicro<0>, qmicro<1>, qmicro<2>, qmicro<3>, qmicro<4>};
  const char* mname[5] = {"0 quant", "1 quant-nosc", "2 cast", "3 read", "4 write"};
  const double mbytes[5] = {mbSrc + mbDst + mbSc, mbSrc + mbDst, mbSrc + mbDst, mbSrc, mbDst};

  char mbuf[64];
  const char* ml = getenv("Q_MODES");
  strncpy(mbuf, (ml && *ml) ? ml : "0 1 2 3 4", 63);
  mbuf[63] = 0;
  int modes[8], nModes = 0;
  for (char* t = strtok(mbuf, " ,"); t && nModes < 8; t = strtok(nullptr, " ,"))
    modes[nModes++] = atoi(t);

  char gbuf[128];
  const char* gl = getenv("Q_GRIDS");
  strncpy(gbuf, (gl && *gl) ? gl : "64 128 256 512 1024", 127);
  gbuf[127] = 0;
  int grids[16], nGrids = 0;
  for (char* t = strtok(gbuf, " ,"); t && nGrids < 16; t = strtok(nullptr, " ,"))
    grids[nGrids++] = atoi(t);

  char wbuf[64];
  const char* wl = getenv("Q_WPB");
  strncpy(wbuf, (wl && *wl) ? wl : "8 16", 63);
  wbuf[63] = 0;
  int wpbs[8], nWpb = 0;
  for (char* t = strtok(wbuf, " ,"); t && nWpb < 8; t = strtok(nullptr, " ,"))
    wpbs[nWpb++] = atoi(t);

  hipEvent_t t0, t1;
  CK(hipEventCreate(&t0));
  CK(hipEventCreate(&t1));
  printf("%-14s %6s %5s %10s %9s %s\n", "mode", "grid", "wpb", "us/iter", "GB/s", "check");

  for (int mi = 0; mi < nModes; ++mi) {
    const int mode = modes[mi];
    for (int wi = 0; wi < nWpb; ++wi) {
      const int wpb = wpbs[wi];
      const int block = wpb * warpSz;
      const size_t ldsB = (mode == 1) ? (size_t)wpb * scaleDim * sizeof(float) : 0;
      for (int gi = 0; gi < nGrids; ++gi) {
        const int grid = grids[gi];
        CK(hipMemset(dDst, 0, srcElems * sizeof(Fp8T)));
        for (int w = 0; w < 3; ++w)
          hipLaunchKernelGGL(kern[mode], dim3(grid), dim3(block), ldsB, 0, dDst, dSc, dSrc, nTok,
                             hid, scaleDim);
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(t0));
        for (int it = 0; it < iters; ++it)
          hipLaunchKernelGGL(kern[mode], dim3(grid), dim3(block), ldsB, 0, dDst, dSc, dSrc, nTok,
                             hid, scaleDim);
        CK(hipEventRecord(t1));
        CK(hipEventSynchronize(t1));
        float ms = 0.f;
        CK(hipEventElapsedTime(&ms, t0, t1));
        const double us = ms * 1e3 / iters;
        char verdict[40] = "";
        if (check && mode == 0) {
          // fp8 e4m3 carries 3 mantissa bits, so a correct blockwise round trip lands within about
          // 2^-4 relative. Anything far outside that means a wrong scale, not a rounding loss.
          unsigned zero = 0;
          CK(hipMemcpy(dWorst, &zero, 4, hipMemcpyHostToDevice));
          hipLaunchKernelGGL(qcheck, dim3(64), dim3(256), 0, 0, dDst, dSc, dSrc, 64, hid, scaleDim,
                             dWorst);
          unsigned w = 0;
          CK(hipDeviceSynchronize());
          CK(hipMemcpy(&w, dWorst, 4, hipMemcpyDeviceToHost));
          const float worst = __builtin_bit_cast(float, w);
          snprintf(verdict, sizeof(verdict), worst < 0.15f ? " ok(%.3f)" : " BAD(%.3f)",
                   (double)worst);
        }
        printf("%-14s %6d %5d %10.2f %9.0f%s\n", mname[mode], grid, wpb, us,
               mbytes[mode] / 1e3 / (us / 1e6), verdict);
        fflush(stdout);
      }
    }
  }
  printf("QM_DONE\n");
  return 0;
}

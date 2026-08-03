// The combine PULL gather, alone, on four GPUs.
//
// WHY IT EXISTS. In the real kernel the fp8 gather reads HALF the bytes of the bf16 gather and
// takes longer doing it -- and it does not care what shape the reads are: chunked 628.7us,
// whole-token 609.2, QUAD depth 4 632.0, 4-byte descriptors 630.0. Every one of those numbers is a
// difference between two runs of a kernel that also quantises, folds, barriers and dedups, so none
// of them can say whether the fabric is slow for 1-byte payloads or whether something around the
// transport is. This runs the gather and nothing else.
//
// WHAT IS FAITHFUL, AND WHY EACH PART HAS TO BE
//   peers    the sources are on OTHER cards, reached by peer pointers, which is what PULL means.
//            A single-GPU stand-in would measure HBM, and HBM is not the thing in question.
//   all four every rank gathers at the same time in the real kernel, so all four devices launch
//            concurrently here. Fabric contention is most of what is being measured; one device
//            reading three idle ones would flatter the result by a wide margin.
//   layout   a source row sits at (destPe * maxTok + tok) * slot in the PEER's buffer, so the four
//            contributions to one token are maxTok*slot apart -- separate streams, not one run.
//   dedup    G_P3 percent of tokens have one source missing, as the destPe dedup leaves them.
//   fold     fp32 accumulate out of LDS, bf16 out, 16 B stores -- the same arithmetic the kernel
//            does, including the per-block scale multiply when the payload is fp8.
//
// THE COMPARISON IT IS FOR. G_BYTES=2 is the bf16 baseline; G_BYTES=1 is fp8, which moves half as
// much. If the fabric does not care about element width, fp8 should take about half as long, and
// the real kernel's fp8 gather is being slowed by something outside the transport. If fp8 is no
// faster than bf16 here, the transport itself is the ceiling and no amount of overlap will fix it.
// G_PIPE answers the other half: does leaving a chunk in flight across the fold recover anything?
//
// Build: hipcc --offload-arch=gfx1250 -std=c++17 -O3 g_micro.cc -o g_micro
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_gfx1250_TDM.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mori/core/transport/p2p/device_primitives.hpp"
using CombFp8 = mori::core::CombineInternalFp8;

#define CK(x)                                                                                      \
  do {                                                                                             \
    hipError_t _e = (x);                                                                           \
    if (_e != hipSuccess) {                                                                        \
      printf("HIP err %d (%s) at %d\n", (int)_e, hipGetErrorString(_e), __LINE__);                 \
      fflush(stdout);                                                                              \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float bf2f(unsigned short v) {
  unsigned int u = (unsigned int)v << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}
__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned int u;
  __builtin_memcpy(&u, &f, 4);
  return (unsigned short)(u >> 16);
}
// The fp8 decode has to be the HARDWARE one. A hand-rolled bit-twiddle with a subnormal branch was
// the first thing this file did, and it made the fp8 fold look expensive for a reason that exists
// nowhere in the shipping kernel -- which converts through mori's CombineInternalFp8 and comes out
// as a cvt instruction. Measuring my own decode would have priced my transcription again.
__device__ __forceinline__ float f8f(unsigned char v) {
  return (float)*reinterpret_cast<const CombFp8*>(&v);
}

// dataSize: 0 = 1 byte, 1 = 2 bytes, 2 = 4 bytes. One row of `elems` elements.
__device__ __forceinline__ gfx1250_TDM_GROUP1 ShapeRow(int elems, int dsz) {
  gfx1250_TDM_GROUP1 g;
  g.dataSize(dsz);
  g.tensorDim0(elems);
  g.tensorDim1(1);
  g.tensorDim0Stride(elems);
  g.tensorDim1Stride(1);
  g.tileDim0(elems);
  g.tileDim1(1);
  return g;
}

__device__ __forceinline__ void IssueLoad(void* lds, const void* src,
                                          const gfx1250_TDM_GROUP1& g1) {
  gfx1250_TDM_GROUP0 g0;
  g0.ldsAddr((uintptr_t)lds);
  g0.globalAddr((uintptr_t)src);
  v4i z4{0, 0, 0, 0};
  v8i z8{0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, 0);
}

// ELEMB: payload element size in bytes (1 = fp8 + scales, 2 = bf16).
// PIPE : 0 = wait(0) then fold, 1 = issue chunk k+1 before folding chunk k.
// DEQ  : 0 = no scales at all, 1 = one scale load per source PER VECTOR, straight off the peer,
//        which is what the shipping fold does; 2 = the token's whole scale row pulled into LDS
//        once per source and read from there.
//        The difference between 1 and 2 is the point of the mode: the scale pointer is a peer
//        pointer into an uncached allocation, so mode 1 issues ~28 uncached cross-card 4 B loads
//        per source per token from every lane, and mode 2 issues scaleDim of them once.
//        3 = load but do not fold (transport floor).
//        4 = one lane per scale block loads, the rest take it by shuffle. MEASURED AND REJECTED:
//        357.3us against mode 2's 133.5 at 256x16 chunk 3584, i.e. barely better than mode 1's
//        349.5. It has the same transaction COUNT as mode 2 and that turns out not to be what
//        costs -- the loads are still inside the fold, one round trip per step, serialised behind
//        the arithmetic that needs them. Mode 2 issues all 56 back to back and waits once.
//        5 = the row prefetched into REGISTERS instead of LDS (scaleDim/warpSize = 2 per lane) and
//        read back by shuffle. Same parallel issue as 2, no LDS, so no host-side layout to agree.
// NSRCC: sources issued per chunk, a compile-time constant so the wait immediate can be one.
template <int ELEMB, int PIPE, int DEQ, int NSRCC>
__global__ __launch_bounds__(1024) void gmicro(unsigned short* __restrict__ out,
                                               const void* const* __restrict__ srcBases,
                                               const float* const* __restrict__ scaleBases,
                                               const unsigned char* __restrict__ peMask,
                                               int nTok, int hid, int chunkElems,
                                               uint32_t slotElems, int scaleDim, int myPe,
                                               int maxTok) {
#if defined(__gfx1250__) || defined(__gfx1251__)
  extern __shared__ char ldsRaw[];
  const int wpb = blockDim.x / warpSize;
  const int warpId = threadIdx.x / warpSize;
  const int laneId = threadIdx.x % warpSize;
  const int gwid = blockIdx.x * wpb + warpId;
  const int gwn = gridDim.x * wpb;
  const int bufs = PIPE ? 2 : 1;
  // Per warp: NSRCC rows of chunkElems, times the buffer count.
  const size_t warpTileB = (size_t)NSRCC * chunkElems * ELEMB * bufs;
  char* tile = ldsRaw + (size_t)warpId * warpTileB;
  const int blockElems = (hid + scaleDim - 1) / scaleDim;
  // DEQ 2 parks the scale rows after all the tiles: NSRCC * scaleDim floats per warp.
  float* ldsSc = nullptr;
  if (DEQ == 2) {
    const int wpbAll = blockDim.x / warpSize;
    ldsSc = (float*)(ldsRaw + (size_t)wpbAll * warpTileB) + (size_t)warpId * NSRCC * scaleDim;
  }

  for (int t = gwid; t < nTok; t += gwn) {
    const unsigned m = peMask[t];
    unsigned short* o = out + (size_t)t * hid;
    // Compact the live sources, exactly as the kernel does before the tile loop.
    const char* sp[NSRCC];
    const float* sc[NSRCC];
    int nSrc = 0;
#pragma unroll
    for (int j = 0; j < NSRCC; ++j) {
      if ((m >> j) & 1) {
        sp[nSrc] = (const char*)srcBases[j] +
                   ((size_t)myPe * maxTok + t) * (size_t)slotElems * ELEMB;
        sc[nSrc] = scaleBases[j] + ((size_t)myPe * maxTok + t) * scaleDim;
        ++nSrc;
      }
    }
    if (nSrc == 0) continue;

    if (DEQ == 2) {
      // One pass over the scale row per source, lane-strided, before any folding. Same bytes as
      // mode 1 would read for a single vector step, and then never read again.
      for (int j = 0; j < nSrc; ++j)
        for (int k = laneId; k < scaleDim; k += warpSize) ldsSc[j * scaleDim + k] = sc[j][k];
      __threadfence_block();
    }
    // Register prefetch: lane k holds entries k, k+warpSize, ... of every source's row. Issued
    // before the first chunk is folded, so the whole row costs one round trip rather than one per
    // step. SCREG is sized for scaleDim <= 2*warpSize (56 at hidden 7168, wave32).
    constexpr int SCREG = 2;
    float scr[NSRCC][SCREG];
    if (DEQ == 5) {
#pragma unroll
      for (int j = 0; j < NSRCC; ++j)
#pragma unroll
        for (int r = 0; r < SCREG; ++r) {
          const int k = r * warpSize + laneId;
          scr[j][r] = (j < nSrc && k < scaleDim) ? sc[j][k] : 1.f;
        }
    }

    const int nChunk = (hid + chunkElems - 1) / chunkElems;
    auto issue = [&](int c, char* tb) {
      const int off = c * chunkElems;
      int n = hid - off;
      if (n > chunkElems) n = chunkElems;
      const gfx1250_TDM_GROUP1 g = ShapeRow(n, ELEMB == 1 ? 0 : 1);
#pragma unroll
      for (int j = 0; j < NSRCC; ++j) {
        // A dead source still gets an issue, of one legal 128 B row, so the number of outstanding
        // ops per chunk is the compile-time constant the wait immediate needs.
        if (j < nSrc)
          IssueLoad(tb + (size_t)j * chunkElems * ELEMB, sp[j] + (size_t)off * ELEMB, g);
        else
          IssueLoad(tb + (size_t)j * chunkElems * ELEMB, sp[0] + (size_t)off * ELEMB,
                    ShapeRow(128 / ELEMB, ELEMB == 1 ? 0 : 1));
      }
    };
    auto fold = [&](int c, const char* tb) {
      const int off = c * chunkElems;
      int n = hid - off;
      if (n > chunkElems) n = chunkElems;
      if (DEQ == 3) {
        // Transport floor, WRONG OUTPUT BY CONSTRUCTION: every peer read is still issued and still
        // waited on, so the fabric traffic is byte for byte what a real fold would need, and the
        // arithmetic is gone. Touch one element so the loads cannot be optimised away.
        if (laneId == 0) o[off] = (unsigned short)tb[0];
        return;
      }
      const int V = 8;  // 16 B out per lane, as in the kernel
      const int nv = (n / (warpSize * V)) * (warpSize * V);
      for (int e = laneId * V; e < nv; e += warpSize * V) {
        float a[V];
#pragma unroll
        for (int k = 0; k < V; ++k) a[k] = 0.f;
        const int sb = (off + e) / blockElems;
        // Lanes are laid out so that a run of blockElems/V of them share one scale block, and the
        // run is aligned to the lane index whenever blockElems divides the step. Only its first
        // lane loads.
        const int grpLanes = blockElems / V;
        const int lead = laneId & ~(grpLanes - 1);
        // Unrolled over the compile-time source count, not the runtime one, so that scr[j] below
        // is a fixed register rather than an indexed one -- an indexed local array goes to scratch
        // and the mode would be measuring the spill instead of the idea.
#pragma unroll
        for (int j = 0; j < NSRCC; ++j) {
          if (j >= nSrc) continue;
          float s = 1.f;
          if (ELEMB == 1 && DEQ == 5) {
            // Shuffle BOTH registers and then select. Selecting first and shuffling once is wrong:
            // which register a lane wants depends on that lane's own sb, so it would broadcast
            // whichever register the SOURCE lane wanted. Both halves of the row are live in the
            // same wave here (56 entries, 32 lanes), so it is wrong in this configuration.
            const float r0 = __shfl(scr[j][0], sb % warpSize);
            const float r1 = __shfl(scr[j][1], sb % warpSize);
            s = (sb >= warpSize) ? r1 : r0;
            if (sb == 0 && s < 0.f) s = -s;
          } else if (ELEMB == 1 && DEQ == 4) {
            float t = 0.f;
            if (laneId == lead) t = sc[j][sb];
            s = __shfl(t, lead);
            if (sb == 0 && s < 0.f) s = -s;
          } else if (ELEMB == 1 && DEQ) {
            s = (DEQ == 2) ? ldsSc[j * scaleDim + sb] : sc[j][sb];
            if (sb == 0 && s < 0.f) s = -s;
          }
          const char* row = tb + (size_t)j * chunkElems * ELEMB + (size_t)e * ELEMB;
          if (ELEMB == 1) {
            const uint64_t w = *reinterpret_cast<const uint64_t*>(row);
            const unsigned char* p = reinterpret_cast<const unsigned char*>(&w);
#pragma unroll
            for (int k = 0; k < V; ++k) a[k] += f8f(p[k]) * s;
          } else {
            const uint4 w = *reinterpret_cast<const uint4*>(row);
            const unsigned short* p = reinterpret_cast<const unsigned short*>(&w);
#pragma unroll
            for (int k = 0; k < V; ++k) a[k] += bf2f(p[k]);
          }
        }
        union {
          uint4 ov;
          unsigned short oe[V];
        };
#pragma unroll
        for (int k = 0; k < V; ++k) oe[k] = f2bf(a[k]);
        *reinterpret_cast<uint4*>(o + off + e) = ov;
      }
      for (int e = nv + laneId; e < n; e += warpSize) {
        float acc = 0.f;
        const int sb = (off + e) / blockElems;
        for (int j = 0; j < nSrc; ++j) {
          float s = 1.f;
          if (ELEMB == 1 && DEQ) {
            s = (DEQ == 2) ? ldsSc[j * scaleDim + sb] : sc[j][sb];
            if (sb == 0 && s < 0.f) s = -s;
          }
          const char* row = tb + (size_t)j * chunkElems * ELEMB + (size_t)e * ELEMB;
          acc += (ELEMB == 1) ? f8f(*(const unsigned char*)row) * s
                              : bf2f(*(const unsigned short*)row);
        }
        o[off + e] = f2bf(acc);
      }
    };
    auto buf = [&](int c) { return tile + (size_t)(c % bufs) * NSRCC * chunkElems * ELEMB; };

    if (PIPE) {
      issue(0, buf(0));
      for (int c = 0; c < nChunk; ++c) {
        if (c + 1 < nChunk) issue(c + 1, buf(c + 1));
        if (c + 1 < nChunk)
          __builtin_amdgcn_s_wait_tensorcnt(NSRCC);
        else
          __builtin_amdgcn_s_wait_tensorcnt(0);
        __threadfence_block();
        fold(c, buf(c));
      }
    } else {
      for (int c = 0; c < nChunk; ++c) {
        issue(c, buf(0));
        __builtin_amdgcn_s_wait_tensorcnt(0);
        __threadfence_block();
        fold(c, buf(0));
      }
    }
  }
#endif
}

static int envi(const char* n, int d) {
  const char* v = getenv(n);
  return (v && *v) ? atoi(v) : d;
}

struct Dev {
  void* stage = nullptr;
  float* scales = nullptr;
  unsigned short* out = nullptr;
  unsigned char* mask = nullptr;
  void** dSrcBases = nullptr;
  float** dScaleBases = nullptr;
  hipStream_t stream{};
  hipEvent_t e0{}, e1{};
};

int main() {
  int nDev = 0;
  CK(hipGetDeviceCount(&nDev));
  const int devs = envi("G_DEVS", nDev < 4 ? nDev : 4);
  hipDeviceProp_t props{};
  CK(hipGetDeviceProperties(&props, 0));
  int warpSz = 0;
  CK(hipDeviceGetAttribute(&warpSz, hipDeviceAttributeWarpSize, 0));
  printf("[info] %s arch=%s CU=%d warp=%d devs=%d/%d\n", props.name, props.gcnArchName,
         props.multiProcessorCount, warpSz, devs, nDev);

  const int hid = envi("G_HID", 7168);
  const int nTok = envi("G_TOK", 4096);
  const int nSrcC = 4;  // == worldSize; the wait immediate needs it constant
  const int p3 = envi("G_P3", 40);
  const int iters = envi("G_ITERS", 20);
  const int scaleDim = envi("G_SCALEDIM", 56);
  const int maxTok = envi("G_MAXTOK", nTok);
  const int elemB = envi("G_BYTES", 1);
  const int pipe = envi("G_PIPE", 0);
  const int deq = envi("G_DEQ", 1);
  const int check = envi("G_CHECK", 1);
  const uint32_t slotElems = (uint32_t)hid;

  // Peer access both ways between every pair, or the peer pointers below are not addressable.
  for (int d = 0; d < devs; ++d) {
    CK(hipSetDevice(d));
    for (int p = 0; p < devs; ++p) {
      if (p == d) continue;
      int can = 0;
      CK(hipDeviceCanAccessPeer(&can, d, p));
      if (!can) {
        printf("[fatal] device %d cannot peer %d\n", d, p);
        return 1;
      }
      hipError_t e = hipDeviceEnablePeerAccess(p, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e);
    }
  }

  const size_t stageElems = (size_t)devs * maxTok * slotElems;
  const size_t scaleElems = (size_t)devs * maxTok * scaleDim;
  Dev dv[8];
  unsigned char* hMask = (unsigned char*)malloc(nTok);
  srand(1234);
  long liveTot = 0;
  for (int i = 0; i < nTok; ++i) {
    unsigned m = (1u << nSrcC) - 1u;
    if (rand() % 100 < p3) m &= ~(1u << (rand() % nSrcC));
    hMask[i] = (unsigned char)m;
    liveTot += __builtin_popcount(m);
  }
  const double liveAvg = (double)liveTot / nTok;

  for (int d = 0; d < devs; ++d) {
    CK(hipSetDevice(d));
    // The staging buffer a peer reads is uncached in mori (dispatch_combine.cpp:324), and that is
    // the allocation whose read behaviour is in question, so match it.
    CK(hipExtMallocWithFlags(&dv[d].stage, stageElems * elemB, hipDeviceMallocUncached));
    CK(hipExtMallocWithFlags((void**)&dv[d].scales, scaleElems * sizeof(float),
                             hipDeviceMallocUncached));
    CK(hipMalloc(&dv[d].out, (size_t)nTok * hid * sizeof(unsigned short)));
    CK(hipMalloc(&dv[d].mask, nTok));
    CK(hipMemcpy(dv[d].mask, hMask, nTok, hipMemcpyHostToDevice));
    // 0x38 as a byte is fp8 e4m3 1.0; as a bf16 pair 0x3838 is 4.5e-5. The check below uses the
    // fp8 reading, so bf16 runs are timed only.
    CK(hipMemset(dv[d].stage, 0x38, stageElems * elemB));
    CK(hipMemset(dv[d].out, 0, (size_t)nTok * hid * sizeof(unsigned short)));
    // Entry k of every row is k+1, not a constant. It used to be 2.0 everywhere, and that made the
    // check below blind to WHICH entry a mode read: the register-prefetch mode had an indexing bug
    // that returned a different entry of the right row and still passed. Distinct values also stay
    // exact through the bf16 store -- 4 sources x entry 56 is 224, inside bf16's integer range.
    float* hs = (float*)malloc(scaleElems * sizeof(float));
    for (size_t i = 0; i < scaleElems; ++i) {
      const size_t k = i % scaleDim;
      hs[i] = (k == 0) ? -1.0f : (float)(k + 1);
    }
    CK(hipMemcpy(dv[d].scales, hs, scaleElems * sizeof(float), hipMemcpyHostToDevice));
    free(hs);
    CK(hipStreamCreate(&dv[d].stream));
    CK(hipEventCreate(&dv[d].e0));
    CK(hipEventCreate(&dv[d].e1));
  }
  // Every device gets the table of all four staging bases, which is what makes the reads peer
  // reads: entry j is device j's buffer, read from device d.
  for (int d = 0; d < devs; ++d) {
    CK(hipSetDevice(d));
    void* hb[8];
    float* hsb[8];
    for (int j = 0; j < devs; ++j) {
      hb[j] = dv[j].stage;
      hsb[j] = dv[j].scales;
    }
    CK(hipMalloc((void**)&dv[d].dSrcBases, sizeof(void*) * devs));
    CK(hipMalloc((void**)&dv[d].dScaleBases, sizeof(float*) * devs));
    CK(hipMemcpy(dv[d].dSrcBases, hb, sizeof(void*) * devs, hipMemcpyHostToDevice));
    CK(hipMemcpy(dv[d].dScaleBases, hsb, sizeof(float*) * devs, hipMemcpyHostToDevice));
  }

  const double mbPerDev = (double)nTok * liveAvg * hid * elemB / 1e6;
  printf("[cfg] hid=%d tok=%d src=%d(live %.2f) bytes=%d pipe=%d deq=%d  peer reads %.1f MB per "
         "device, %.1f MB total\n",
         hid, nTok, nSrcC, liveAvg, elemB, pipe, deq, mbPerDev, mbPerDev * devs);

  char gbuf[128];
  const char* gl = getenv("G_GRIDS");
  strncpy(gbuf, (gl && *gl) ? gl : "64 128 256", 127);
  gbuf[127] = 0;
  int grids[16], nGrids = 0;
  for (char* t = strtok(gbuf, " ,"); t && nGrids < 16; t = strtok(nullptr, " ,"))
    grids[nGrids++] = atoi(t);
  char wbuf[64];
  const char* wl = getenv("G_WPB");
  strncpy(wbuf, (wl && *wl) ? wl : "8", 63);
  wbuf[63] = 0;
  int wpbs[8], nWpb = 0;
  for (char* t = strtok(wbuf, " ,"); t && nWpb < 8; t = strtok(nullptr, " ,"))
    wpbs[nWpb++] = atoi(t);
  char cbuf[64];
  const char* cl = getenv("G_CHUNKS");
  strncpy(cbuf, (cl && *cl) ? cl : "0", 63);
  cbuf[63] = 0;
  int chunks[8], nChunks = 0;
  for (char* t = strtok(cbuf, " ,"); t && nChunks < 8; t = strtok(nullptr, " ,"))
    chunks[nChunks++] = atoi(t);

  printf("%-6s %5s %5s %8s %9s %10s %9s %s\n", "bytes", "grid", "wpb", "chunk", "lds KB", "us/iter",
         "GB/s", "check");

  for (int wi = 0; wi < nWpb; ++wi) {
    const int wpb = wpbs[wi];
    const int block = wpb * warpSz;
    for (int ci = 0; ci < nChunks; ++ci) {
      int chunk = chunks[ci] > 0 ? chunks[ci] : (hid / nSrcC / (pipe ? 2 : 1));
      chunk = (chunk / 64) * 64;
      if (chunk <= 0) continue;
      size_t ldsB = (size_t)wpb * nSrcC * chunk * elemB * (pipe ? 2 : 1);
      if (elemB == 1 && deq == 2) ldsB += (size_t)wpb * nSrcC * scaleDim * sizeof(float);
      if (ldsB > 327680) {
        printf("%-6d %5s %5d %8d %9.0f  SKIP over LDS budget\n", elemB, "-", wpb, chunk,
               ldsB / 1024.0);
        continue;
      }
      for (int gi = 0; gi < nGrids; ++gi) {
        const int grid = grids[gi];
        auto launch = [&](int d) {
          CK(hipSetDevice(d));
          const int variant = (elemB == 1 ? 0 : 16) + (pipe ? 8 : 0) + deq;
#define _GLAUNCH(K)                                                                                \
  hipLaunchKernelGGL((K), dim3(grid), dim3(block), ldsB, dv[d].stream, dv[d].out,                  \
                     (const void* const*)dv[d].dSrcBases,                                          \
                     (const float* const*)dv[d].dScaleBases, dv[d].mask, nTok, hid, chunk,         \
                     slotElems, scaleDim, d, maxTok)
          switch (variant) {
            case 0: _GLAUNCH((gmicro<1, 0, 0, 4>)); break;
            case 1: _GLAUNCH((gmicro<1, 0, 1, 4>)); break;
            case 2: _GLAUNCH((gmicro<1, 0, 2, 4>)); break;
            case 3: _GLAUNCH((gmicro<1, 0, 3, 4>)); break;
            case 4: _GLAUNCH((gmicro<1, 0, 4, 4>)); break;
            case 5: _GLAUNCH((gmicro<1, 0, 5, 4>)); break;
            case 8: _GLAUNCH((gmicro<1, 1, 0, 4>)); break;
            case 9: _GLAUNCH((gmicro<1, 1, 1, 4>)); break;
            case 10: _GLAUNCH((gmicro<1, 1, 2, 4>)); break;
            case 11: _GLAUNCH((gmicro<1, 1, 3, 4>)); break;
            case 12: _GLAUNCH((gmicro<1, 1, 4, 4>)); break;
            case 13: _GLAUNCH((gmicro<1, 1, 5, 4>)); break;
            case 16: _GLAUNCH((gmicro<2, 0, 0, 4>)); break;
            case 19: _GLAUNCH((gmicro<2, 0, 3, 4>)); break;
            case 24: _GLAUNCH((gmicro<2, 1, 0, 4>)); break;
            case 27: _GLAUNCH((gmicro<2, 1, 3, 4>)); break;
            default:
              printf("[fatal] bad variant %d (bf16 takes deq 0 or 3 only)\n", variant);
              exit(1);
          }
#undef _GLAUNCH
        };
        for (int w = 0; w < 3; ++w)
          for (int d = 0; d < devs; ++d) launch(d);
        for (int d = 0; d < devs; ++d) {
          CK(hipSetDevice(d));
          CK(hipStreamSynchronize(dv[d].stream));
        }
        for (int d = 0; d < devs; ++d) {
          CK(hipSetDevice(d));
          CK(hipEventRecord(dv[d].e0, dv[d].stream));
        }
        for (int it = 0; it < iters; ++it)
          for (int d = 0; d < devs; ++d) launch(d);
        for (int d = 0; d < devs; ++d) {
          CK(hipSetDevice(d));
          CK(hipEventRecord(dv[d].e1, dv[d].stream));
        }
        double worstUs = 0;
        for (int d = 0; d < devs; ++d) {
          CK(hipSetDevice(d));
          CK(hipEventSynchronize(dv[d].e1));
          float ms = 0.f;
          CK(hipEventElapsedTime(&ms, dv[d].e0, dv[d].e1));
          const double us = ms * 1e3 / iters;
          if (us > worstUs) worstUs = us;
        }
        char verdict[32] = "";
        if (check && elemB == 1 && deq != 0 && deq != 3) {
          // Every byte is fp8 1.0 and entry k of every scale row is k+1 (entry 0 negated), so
          // element e of a token folds to exactly (e/blockElems + 1) * live sources. Varying with e
          // is the point: it catches reading the wrong entry of the right row, which a constant row
          // cannot. A wrong row stride, a skipped source or a dropped scale still move it too.
          CK(hipSetDevice(0));
          const int nchk = nTok < 32 ? nTok : 32;
          unsigned short* h = (unsigned short*)malloc((size_t)nchk * hid * 2);
          CK(hipMemcpy(h, dv[0].out, (size_t)nchk * hid * 2, hipMemcpyDeviceToHost));
          const int blockElems = (hid + scaleDim - 1) / scaleDim;
          long bad = 0;
          for (int t = 0; t < nchk; ++t) {
            const float live = (float)__builtin_popcount((unsigned)hMask[t]);
            for (int e = 0; e < hid; ++e) {
              const float want = live * (float)(e / blockElems + 1);
              unsigned u;
              memcpy(&u, &want, 4);
              if (h[(size_t)t * hid + e] != (unsigned short)(u >> 16)) ++bad;
            }
          }
          snprintf(verdict, sizeof(verdict), bad ? " BAD %ld" : " ok", bad);
          free(h);
        }
        printf("%-6d %5d %5d %8d %9.0f %10.2f %9.0f%s\n", elemB, grid, wpb, chunk, ldsB / 1024.0,
               worstUs, mbPerDev / 1e3 / (worstUs / 1e6), verdict);
        fflush(stdout);
      }
    }
  }
  printf("GM_DONE\n");
  return 0;
}

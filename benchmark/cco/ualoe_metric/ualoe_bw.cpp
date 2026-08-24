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
// ualoe_bw.cpp -- cross-node UALOE fabric bandwidth: read / write / bidirectional.
//
// Symmetric peer-to-peer. If a GPU index is given, one pair is measured; if it is
// OMITTED, ALL local GPUs are used at once (GPU i <-> peer GPU i) and the
// aggregate is reported. When the two nodes have different GPU counts, the
// minimum is used (paired by index).
//
//   read  : local <- remote, pipelined TDM (tensor_load_to_lds) fabric->LDS->local
//   write : local -> remote, CU uint4 copy (TDM does not help the egress side)
//   bidir : both nodes read from each other concurrently (full-duplex; read-based
//           beats write-based since reads sustain higher)
//
// ubench/06 conventions: WARMUP=10, LOOP=50, wall clock + hipDeviceSynchronize,
// GB/s decimal, payload x1.
//
// Build: hipcc -std=c++17 -O3 ualoe_bw.cpp -o ualoe_bw
//   node A> ./ualoe_bw listen  <port> [gpu]        # omit gpu => all GPUs
//   node B> ./ualoe_bw connect <peer_ip> <port> [gpu]
#include <arpa/inet.h>
#include <hip/hip_runtime.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIPCHECK(x)                                                                               \
  do {                                                                                            \
    hipError_t _e = (x);                                                                          \
    if (_e != hipSuccess) {                                                                       \
      fprintf(stderr, "[FATAL] %s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(_e)); \
      exit(2);                                                                                    \
    }                                                                                             \
  } while (0)
#ifndef WARMUP
#define WARMUP 10
#endif
#ifndef LOOP
#define LOOP 50
#endif
#define MAXG 16

#ifdef SWEEP_ONE
// One saturated size, for fast config iteration: 8GB is within 0.1% of 16GB but runs in half the
// time.
static const size_t SIZES[] = {8192UL << 20};
#elif defined(SWEEP_16)
// Single point at 16GB. CU peaks at 8GB and falls back slightly, while the TDM store path keeps
// climbing to 16GB, so this is where the two transports are furthest apart and a policy change
// shows up most clearly.
static const size_t SIZES[] = {16384UL << 20};
#elif defined(SWEEP_HUGE)
// Turns the asymptote into a measurement: if the ~1655 GB/s extrapolated off 1-4GB is real, the
// measured number here should climb to it on its own rather than staying put.
static const size_t SIZES[] = {4096UL << 20, 8192UL << 20, 16384UL << 20};
#elif defined(SWEEP_BIG)
// Bandwidth still climbed 7.2% from 256MB to 1GB, so the default sweep may stop before the curve
// flattens; this variant starts at 1GB to find where it actually plateaus.
static const size_t SIZES[] = {1024UL << 20, 2048UL << 20, 4096UL << 20};
#elif defined(SWEEP_MATRIX)
// Allocation size only. MATRIX=1 picks every size it measures at runtime out of these buffers, so
// the single entry here just has to be the largest cell the sweep will be asked for; it is also the
// size the correctness gate runs at. 8 GB is the size the epcheck baselines were taken at, so
// building at that value lets the sweep carry an anchor column directly comparable to them; it
// costs ~24 GB of VRAM.
#ifndef MATRIX_MAXB
#define MATRIX_MAXB (1024UL << 20)
#endif
static const size_t SIZES[] = {MATRIX_MAXB};
#else
static const size_t SIZES[] = {1UL << 20, 16UL << 20, 256UL << 20, 512UL << 20,
                               1024UL << 20};  // 1MB,16MB,256MB,512MB,1GB
#endif
static const int N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
static size_t MAX_BYTES = SIZES[N_SIZES - 1];
static int g_numCU = 0;
// Iteration counts as runtime state rather than only the WARMUP/LOOP macros, because the MATRIX
// sweep needs a different count in every cell: at one block a 1 GB pass takes ~200 ms, so the
// compiled-in 50 would cost minutes there, while at 1 KB a pass is shorter than its own launch and
// 50 of them still time nothing. Only write_all and tdmwr_all read these -- every variant the sweep
// does not use keeps the macros, so nothing outside MATRIX changes behaviour.
static int g_warmup = WARMUP, g_loop = LOOP;
// "1,2,4" -> array, shared by the size list and the block list. Both processes parse the same
// string: oneway() carries a barrier pair, so a cell one side skips would leave the two ends
// waiting on each other, and dropping malformed entries has to happen identically on both.
static int parse_list(const char* s, size_t* out, int maxn) {
  int n = 0;
  for (const char* p = s; *p && n < maxn;) {
    while (*p == ',' || *p == ' ') p++;
    if (!*p) break;
    char* e = nullptr;
    unsigned long long v = strtoull(p, &e, 10);
    if (e == p) break;
    p = e;
    if (v) out[n++] = (size_t)v;
  }
  return n;
}
static size_t env_sz(const char* k, size_t d) {
  const char* v = getenv(k);
  return v ? strtoull(v, nullptr, 10) : d;
}

static void send_all(int fd, const void* b, size_t n) {
  const char* p = (const char*)b;
  while (n) {
    ssize_t w = send(fd, p, n, 0);
    if (w <= 0) {
      perror("send");
      exit(3);
    }
    p += w;
    n -= w;
  }
}
static void recv_all(int fd, void* b, size_t n) {
  char* p = (char*)b;
  while (n) {
    ssize_t r = recv(fd, p, n, 0);
    if (r <= 0) {
      perror("recv");
      exit(3);
    }
    p += r;
    n -= r;
  }
}
static void barrier(int fd) {
  char b = 1;
  send_all(fd, &b, 1);
  recv_all(fd, &b, 1);
}
static double exch(int fd, double v) {
  double r;
  send_all(fd, &v, 8);
  recv_all(fd, &r, 8);
  return r;
}
static double gbps(size_t bytes, double ms) { return (double)bytes / (ms / 1e3) / 1e9; }
static void b2s(size_t b, char* s) {
  if (b < (1UL << 20))
    sprintf(s, "%zuKB", b >> 10);
  else if (b < (1UL << 30))
    sprintf(s, "%zuMB", b >> 20);
  else
    sprintf(s, "%.2fGB", b / (1024.0 * 1024 * 1024));
}

// ---- write kernel (CU uint4 copy) ----
__global__ void copyk(uint4* __restrict__ d, const uint4* __restrict__ s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) d[i] = s[i];
}
// copyk's store depends on the load it just issued, so a thread holds one request in flight at a
// time and a narrow grid is limited by how many threads exist rather than by the link. This version
// issues U loads before any of their stores, multiplying the in-flight bytes per thread by U
// without adding threads. It exists because a neighbouring team's per-block numbers are ~4.7x these
// at the same block count, and memory-level parallelism per thread is the only term that differs by
// that much. U is a runtime choice (g_unroll) rather than a build flag so one process can compare
// 1/2/4/8 against the same allocation, the same peer and the same launch path.
template <int U>
__global__ void copyk_u(uint4* __restrict__ d, const uint4* __restrict__ s, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x, big = stride * U;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n; i += big) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = s[i + (size_t)u * stride];
#pragma unroll
    for (int u = 0; u < U; u++) d[i + (size_t)u * stride] = v[u];
  }
  // Tail: whatever the unrolled body could not cover, one element at a time. Without it the last
  // partial group is silently not copied and the verify would fail on exactly the sizes that are
  // not a multiple of U*stride.
  for (; i < n; i += stride) d[i] = s[i];
}
static int g_unroll = 1;
static int g_storeonly = 0;
// Size the multi-issuer copy's tile to the payload and the grid rather than to the compiled
// constant. Off by default so every recorded table stays reproducible from its build flags alone.
static int g_dyntile = 0;
// Floor on the tile. One row is 1 KB at RTD0=256, and below that the row has to narrow instead,
// which is a different descriptor shape -- so how far down it pays to go is a measurement, not a
// constant.
static size_t g_dyntile_min = 1024;
static size_t g_dyntile_bytes = 0;  // tile the last call settled on, for the [MX] line to report
// Same copy on non-temporal accesses. The reason to have it is not steady-state bandwidth -- copyk
// already reaches the link's asymptote -- but the fixed per-launch cost measured on top of it,
// which is ~21 us for copyk against ~11 us for the TDM store path on the same link and the same
// bytes, and which does not move when the grid geometry changes by 8x. What differs between the two
// paths is that copyk's stores land in L2 and have to be released at end of kernel. If that release
// is the cost, streaming the stores past L2 should remove it; if the number does not move, the
// hypothesis is wrong and the cost is somewhere neither kernel controls.
using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
__global__ void copyk_nt(u32x4* __restrict__ d, const u32x4* __restrict__ s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    u32x4 v = __builtin_nontemporal_load(&s[i]);
    __builtin_nontemporal_store(v, &d[i]);
  }
}
// Position-dependent source pattern, so a sample check can tell "the right bytes arrived at the
// right offset" from "something arrived". A constant fill cannot distinguish those, and neither can
// it catch a cache policy that leaves the write in a local cache instead of landing it at the peer.
#define SRCPAT(idx) ((uint32_t)(idx) * 2654435761u)
__global__ void fillk(uint32_t* __restrict__ p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) p[i] = SRCPAT(i);
}
// Store-only variant of the copy: the bytes are recomputed in registers instead of being read from
// local HBM, so a thread issues one memory operation per element rather than two and nothing it
// stores depends on a load. It exists to separate "the fabric write path is slow" from "the read
// that feeds it is". Recomputing SRCPAT rather than storing a constant keeps it verifiable: the
// bytes that land are the same bytes the copy would have moved, so the normal check still applies
// and this cannot pass by moving less.
template <int U>
__device__ __forceinline__ uint4 srcpat4(size_t e) {
  uint32_t b = (uint32_t)(e * 4);
  return make_uint4(SRCPAT(b), SRCPAT(b + 1), SRCPAT(b + 2), SRCPAT(b + 3));
}
template <int U>
__global__ void storek_u(uint4* __restrict__ d, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x, big = stride * U;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n; i += big) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      size_t e = i + (size_t)u * stride;
      d[e] = srcpat4<U>(e);
    }
  }
  for (; i < n; i += stride) d[i] = srcpat4<U>(i);
}
// ---- pipelined-TDM read (gfx1250): async fabric->LDS loads + local store ----
using sg0v = int __attribute__((ext_vector_type(4)));
using sg1v = int __attribute__((ext_vector_type(8)));
using sg2v = int __attribute__((ext_vector_type(4)));
using sg3v = int __attribute__((ext_vector_type(4)));
using sgxv = int __attribute__((ext_vector_type(8)));
__device__ void s_wait_tensorcnt(short) __asm("llvm.amdgcn.s.wait.tensorcnt");
// Cache policy exists in two independent places and both were left at zero/default until now.
// The instruction's CPOL immediate: TH in bits 0-2, SCOPE in 3-4, NV in 5.
#ifndef TDM_TH
#define TDM_TH 0  // 0=rt 1=nt 2=ht 3=lu/bypass
#endif
#ifndef TDM_SCOPE
#define TDM_SCOPE 2  // 0=cu 1=se 2=dev 3=sys
#endif
#ifndef TDM_NV
#define TDM_NV 0
#endif
static constexpr int TDM_CPOL = (TDM_TH) | (TDM_SCOPE << 3) | (TDM_NV << 5);
// The descriptor's own copy, per gfx1250_TDM_GROUP0 in <hip/amd_detail/amd_gfx1250_TDM.h>:
//   m_count:2 | m_is_restore:1 | m_is_store:1 | m_nv:1 | m_scope_trait:2 | m_th:3
#ifndef DESC_TH
#define DESC_TH 0
#endif
#ifndef DESC_SCOPE
#define DESC_SCOPE 0
#endif
#ifndef DESC_NV
#define DESC_NV 0
#endif
static constexpr uint32_t DESC_W0 =
    1u | ((DESC_NV & 1u) << 4) | ((DESC_SCOPE & 3u) << 5) | ((DESC_TH & 7u) << 7);
// gfx1250_TDM_GROUP1 sgpr0: m_workgroup_mask:16 | m_data_size:2 | m_atomic_barrier_enable:1 |
//                           m_iterate_enable:1 | m_pad_enable:1 | m_early_timeout:1 |
//                           pad_interval:3 | pad_amount:7
// Only m_data_size was ever set (to 2). It is the log2 of the element size in bytes -- 2 means the
// 4-byte elements these kernels use -- so changing it also changes how many bytes a tile of the
// same dimensions spans. tdm_desc2d converts, keeping the byte extent fixed so the correctness gate
// stays meaningful and only the granularity the hardware sees varies.
#ifndef TDM_DSZ
#define TDM_DSZ 2
#endif
#ifndef DESC_ITER
#define DESC_ITER 0
#endif
#ifndef DESC_ETO
#define DESC_ETO 0
#endif
#ifndef DESC_WGM
#define DESC_WGM 0
#endif
static constexpr uint32_t DESC_G1W0 = ((TDM_DSZ & 0x3u) << 16) | ((DESC_WGM) & 0xFFFFu) |
                                      ((DESC_ITER & 1u) << 19) | ((DESC_ETO & 1u) << 21);
#ifndef RTD0N
#define RTD0N 256
#endif
#ifndef RTD1N
#define RTD1N 8
#endif
// Tiles the single-issuer staged kernel keeps in flight per round. 4 is what every sweep and every
// recorded table was built with; it used to live only in the scripts' GRID string, which meant a
// build that did not go through them ran a different kernel than the one the tables describe.
#ifndef RPIPEN
#define RPIPEN 4
#endif
static const uint32_t RTD0 = RTD0N, RTD1 = RTD1N;
static const int RPIPE = RPIPEN;
// How many tensor stores may stay in flight across the loop boundary. The operand of
// s_wait_tensorcnt is a threshold on outstanding operations, same family as s_wait_dscnt, which the
// compiler itself emits with 0x2/0x4/0x6 in this very file -- so a partial wait is expressible, and
// wait(0) drains the queue on every iteration for no reason other than that it was the only value
// ever tried.
//
// TDM_WAITN is the probe threshold used by the store-only variant, which writes no LDS and so needs
// no extra buffering. TDMCU_WAITN is separate because a partial wait there is only safe if nothing
// overwrites the LDS a still-running store is reading from: at N outstanding stores the staging
// write of round i targets a buffer last stored in round i+1-NBUF, while the most recent wait only
// guarantees rounds up to i-1-N have landed, so NBUF >= N+2. Keeping the two knobs apart matters
// because that buffer growth costs occupancy -- driving them from one macro dropped the hybrid
// variant by 6 GB/s purely from the LDS the probe asked for. 8 is where the gain saturates:
// measured against a wait(0) instance in the same binary it is worth +2.5 to +5 GB/s, and
// 12/16/24/32/63 give no more.
#ifndef TDM_WAITN
#define TDM_WAITN 8
#endif
#ifndef TDMCU_WAITN
#define TDMCU_WAITN 0
#endif
// Waves per block issuing descriptors in the multi-issuer probe. Capped by the block's wave count
// (TWTH/warpSize); anything above that just leaves the extra waves idle.
// 2 is the best measured; 4 is flat and 8 costs ~9 GB/s. Widening issue is not a lever either way.
#ifndef MWISS
#define MWISS 2
#endif
// Issuing waves per block in the CU-staged copy, the variant that carries the hybrid.
#ifndef CUISS
#define CUISS 2
#endif
// LDS is banked into partitions and waves reading the same partition conflict, so each issuing wave
// is given its own LDS span of LDSPART bytes rather than buffers interleaved by round. At 16384
// this is the same total LDS as the interleaved layout, only redistributed, so it costs no
// occupancy; larger values spread the waves further apart at the price of fewer blocks per CU.
#ifndef LDSPART
#define LDSPART 16384
#endif
// The store-only probe holds RPIPEN tiles per wave, more than LDSPART covers, so its per-wave span
// is widened to fit; otherwise a wave would run into the next wave's partition.
#define MW_TILEB (RTD0N * RTD1N * 4)
#define MW_SPAN ((LDSPART) > (RPIPEN * MW_TILEB) ? (LDSPART) : (RPIPEN * MW_TILEB))
// The staged multi-issuer copy carries the same partitioning into the fastest full-copy variant,
// which still issues from one wave. Its pipe depth and per-wave span are separate knobs from the
// probe's: LDS here is MWSISS * MWS_SPAN per block against 32 KB for the single-issuer staged
// kernel at RPIPEN=4, so the spacing that helped the probe is not free and has to be paid for out
// of occupancy.
//
// Measured, all verified full copies, against a single-issuer staged instance in the same process
// that held 1643.4-1643.7 across every build: 2:2:16K 1631/1634, 4:2 (span widens to 32K) 1637,
// 2:2:32K 1616, 1:4:8K 1628, 2:4:16K 1634. Widening issue costs 6-27 GB/s here, the opposite of its
// +3.9 on the CU-staged copy. Inference, not yet tested: the staged loop is a load batch, a drain,
// a store batch and a drain, and several waves interleave their loads and stores into one tensor
// queue, which the CU-staged copy cannot do because its loads go down the vector path and never
// enter the queue.
//
// That measurement is a full grid on a fixed tile, and it also asks the wrong question. What sets
// the wide-grid ceiling is MWSISS*MWSPIPE*tile, the bytes a block pushes per round, not how many
// waves issue them. At 16 GB / 512 blocks: 32 KB per round reads 1642 (both at 1x4 and 2x2), 64 KB
// reads 1645 (both at 4x2 and 8x1), 128 KB reads 1570 at 8x2 and 1568 at 4x2 on a 16 KB tile --
// different issuer counts and tiles, 0.1% apart -- and 256 KB recovers to 1632. The 4x2-on-16 KB
// row issues the same 8 descriptors as the 4x2-on-8 KB row that reads 1645, so descriptor rate is
// not it either.
//
// 64 KB per round is the optimum, and MWSISS=8 with MWSPIPE=1 is the way to reach it that also
// keeps the narrow-grid gain: 17.98 GB/s at one block against the single issuer's 8.3, 1613 at 128
// blocks, 1645.2 at 512, and the best 64 MB column measured (1412/1404/1371 at 128/256/512). LDS is
// 64 KB per block. These three default to the optimum above: 8 issuers, one tile deep, an 8 KB span
// each. Building without any -D gives 64 KB per block per round and 64 KB of LDS per block, which
// is the fast configuration rather than something a caller has to know to ask for. The earlier
// default was 2:2:16K, and the recorded TDMms columns that predate this were taken with it
// -- reproducing those needs -DMWSISS=2 -DMWSPIPE=2 -DMWSSPAN=16384 spelled out.
#ifndef MWSPIPE
#define MWSPIPE 1
#endif
#ifndef MWSISS
#define MWSISS 8
#endif
#ifndef MWSSPAN
#define MWSSPAN 8192
#endif
#define MWS_SPAN ((MWSSPAN) > (MWSPIPE * MW_TILEB) ? (MWSSPAN) : (MWSPIPE * MW_TILEB))
// Prefetch depth and buffer count of the deep-pipelined staged copy. NBUF >= 2*D is required for
// the partial wait to be safe; see tdm_write_deep. D=2/NBUF=4 holds LDS at the 32 KB the
// single-issuer staged kernel uses at RPIPEN=4.
#ifndef DEEPD
#define DEEPD 2
#endif
#ifndef DEEPNBUF
#define DEEPNBUF (2 * DEEPD)
#endif
// Which TDM kernel the hybrid runs on its second stream: 1 = staged copy, 0 = CU-staged copy.
// 0 despite the staged copy being faster on its own (1643.9 vs 1635): concurrently with the CU
// stream it peaks at 1645.8 against 1646.6, so the two contend for something the CU-staged kernel
// does not.
#ifndef HYBTDM
#define HYBTDM 0
#endif
#define TDM_NBUF (TDMCU_WAITN + 2)
__device__ inline void tdm_desc2d(uint32_t sg0[4], uint32_t sg1[8], uint32_t lds, const void* g,
                                  uint32_t td0, uint32_t td1) {
  // Callers pass tile extents in floats; the descriptor wants them in m_data_size elements.
  constexpr uint32_t ESZ = 1u << TDM_DSZ;
  const uint32_t d0 = td0 * 4u / ESZ, d1 = td1;
  uint64_t ga = reinterpret_cast<uint64_t>(g), s0 = d0;
  sg0[0] = DESC_W0;
  sg0[1] = lds;
  sg0[2] = uint32_t(ga);
  sg0[3] = (1u << 31) | uint32_t((ga >> 32) & 0x01FFFFFFu);
  sg1[0] = DESC_G1W0;
  sg1[1] = (d0 << 16);
  sg1[2] = (d0 >> 16) | (d1 << 16);
  sg1[3] = (d1 >> 16) | ((d0 & 0xFFFFu) << 16);
  sg1[4] = (d1 & 0xFFFFu);
  sg1[5] = uint32_t(s0);
  sg1[6] = uint32_t((s0 >> 32) & 0xFFFFu);
  sg1[7] = 0;
}
template <uint32_t TD0, uint32_t TD1, int PIPE>
__global__ void tdm_read(const float* __restrict__ rem, float* __restrict__ loc,
                         uint32_t num_tiles) {
  extern __shared__ char smem[];
  const uint32_t TILE = TD0 * TD1, TB = TILE * 4;
  for (uint32_t base = blockIdx.x * PIPE; base < num_tiles; base += gridDim.x * PIPE) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int k = 0; k < PIPE; k++) {
        uint32_t t = base + k;
        if (t < num_tiles) {
          uint32_t a[4], b[8];
          tdm_desc2d(a, b, k * TB, rem + (size_t)t * TILE, TD0, TD1);
          __builtin_amdgcn_tensor_load_to_lds(
              __builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b), sg2v{0, 0, 0, 0},
              sg3v{0, 0, 0, 0}, sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
        }
      }
    }
    s_wait_tensorcnt(0);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < PIPE; k++) {
      uint32_t t = base + k;
      if (t < num_tiles) {
        const float* s = (const float*)(smem + (size_t)k * TB);
        float* d = loc + (size_t)t * TILE;
        for (uint32_t i = threadIdx.x; i < TILE; i += blockDim.x) d[i] = s[i];
      }
    }
    __syncthreads();
  }
}

// ---- pipelined-TDM write: the egress direction, which the read kernel above cannot measure.
// Mirrors tdm_read so the two are comparable: PIPE tiles are staged in LDS, then pushed to the
// peer. NOLOAD drops the local staging load, leaving whatever is in LDS, which prices the link's
// write ceiling with nothing serialised ahead of it (no correctness check runs here). WAITN is a
// template parameter, not the macro, so one binary can hold both a wait(0) and a wait(N) instance
// and measure them back to back. Across separate builds the two are not comparable: an untouched
// control variant moved by ~3 GB/s between builds here, which is larger than the effect.
template <uint32_t TD0, uint32_t TD1, int PIPE, bool NOLOAD, int WAITN = 0>
__global__ void tdm_write(float* __restrict__ rem, const float* __restrict__ loc,
                          uint32_t num_tiles) {
  extern __shared__ char smem[];
  const uint32_t TILE = TD0 * TD1, TB = TILE * 4;
  for (uint32_t base = blockIdx.x * PIPE; base < num_tiles; base += gridDim.x * PIPE) {
    if (threadIdx.x == 0) {
      if (!NOLOAD) {
#pragma unroll
        for (int k = 0; k < PIPE; k++) {
          uint32_t t = base + k;
          if (t < num_tiles) {
            uint32_t a[4], b[8];
            tdm_desc2d(a, b, k * TB, loc + (size_t)t * TILE, TD0, TD1);
            __builtin_amdgcn_tensor_load_to_lds(
                __builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b), sg2v{0, 0, 0, 0},
                sg3v{0, 0, 0, 0}, sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
          }
        }
        s_wait_tensorcnt(0);
      }
#pragma unroll
      for (int k = 0; k < PIPE; k++) {
        uint32_t t = base + k;
        if (t < num_tiles) {
          uint32_t a[4], b[8];
          tdm_desc2d(a, b, k * TB, rem + (size_t)t * TILE, TD0, TD1);
          __builtin_amdgcn_tensor_store_from_lds(
              __builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b), sg2v{0, 0, 0, 0},
              sg3v{0, 0, 0, 0}, sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
        }
      }
      // NOLOAD writes nothing into LDS, so outstanding stores cannot be corrupted and the drain is
      // pure overhead. The staging path must keep draining: its next batch of loads would overwrite
      // LDS that in-flight stores are still reading.
      if constexpr (NOLOAD)
        s_wait_tensorcnt(WAITN);
      else
        s_wait_tensorcnt(0);
    }
    __syncthreads();
  }
}

// Multi-issuer probe. tdm_write has one wave (threadIdx.x==0) issuing every descriptor while the
// rest of the block sits on the barrier, so it cannot tell a saturated link from a saturated
// descriptor issue rate. tensor_store_from_lds is a scalar instruction -- extra lanes in the same
// wave issue nothing extra -- so the only way to widen issue is more waves.
//
// ISS waves per block each drive their own tile stream. Everything else matches tdm_write<NOLOAD>:
// no source read, same grid, same wait threshold, and every wave points at the same LDS buffers,
// which costs nothing in correctness because nothing reads the source and the bytes are already
// meaningless. Holding LDS constant matters: growing it per wave would trade occupancy for issue
// width and confound the answer. ISS=1 reproduces the single-issuer case in the same binary, as the
// control. td0_elems/td1_rows shrink the tile at launch; 0 keeps the compiled extent. A payload too
// small to fill blocks*ISS*PIPE tiles otherwise leaves most of the grid with nothing to issue, and
// cutting rows only reaches down to one row -- 1 KB at TD0=256 -- so the row itself has to be
// narrowable as well. Both only ever shrink the tile, so PIPE of them still fit the SPAN the launch
// reserved.
template <uint32_t TD0, uint32_t TD1, int PIPE, int ISS, int WAITN, bool NOLOAD = true,
          uint32_t SPAN = MW_SPAN>
__global__ void tdm_write_mw(float* __restrict__ rem, const float* __restrict__ loc,
                             uint32_t num_tiles, uint32_t td0_elems, uint32_t td1_rows) {
  extern __shared__ char smem[];
  const uint32_t D0 = td0_elems ? td0_elems : TD0, TD1R = td1_rows ? td1_rows : TD1;
  const uint32_t TILE = D0 * TD1R, TB = TILE * 4;
  const uint32_t W = warpSize, wv = threadIdx.x / W, lane = threadIdx.x - wv * W;
  if (wv >= ISS || lane != 0) return;
  for (uint32_t base = (blockIdx.x * ISS + wv) * PIPE; base < num_tiles;
       base += gridDim.x * ISS * PIPE) {
    // Each issuing wave works out of its own LDS partition; sharing one span across waves is what a
    // partition conflict looks like, and it cost 9 GB/s at 8 issuers.
    if constexpr (!NOLOAD) {
#pragma unroll
      for (int k = 0; k < PIPE; k++) {
        uint32_t t = base + k;
        if (t < num_tiles) {
          uint32_t a[4], b[8];
          tdm_desc2d(a, b, wv * SPAN + k * TB, loc + (size_t)t * TILE, D0, TD1R);
          __builtin_amdgcn_tensor_load_to_lds(
              __builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b), sg2v{0, 0, 0, 0},
              sg3v{0, 0, 0, 0}, sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
        }
      }
      s_wait_tensorcnt(0);
    }
#pragma unroll
    for (int k = 0; k < PIPE; k++) {
      uint32_t t = base + k;
      if (t < num_tiles) {
        uint32_t a[4], b[8];
        tdm_desc2d(a, b, wv * SPAN + k * TB, rem + (size_t)t * TILE, D0, TD1R);
        __builtin_amdgcn_tensor_store_from_lds(
            __builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b), sg2v{0, 0, 0, 0},
            sg3v{0, 0, 0, 0}, sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
      }
    }
    // With staging, the next round's loads target the same partition, so the stores must have
    // retired first; the store-only probe has no such dependency.
    if constexpr (NOLOAD)
      s_wait_tensorcnt(WAITN);
    else
      s_wait_tensorcnt(0);
  }
}

// Double-buffered variant of the above. tdm_write serialises its two halves (load batch, wait,
// store batch, wait) so the local read and the remote write never overlap. Here store[i] and
// load[i+1] are issued back to back into opposite LDS buffers and are in flight together; the wait
// then covers both. It still has to be wait(0) rather than a partial wait: the thing we need to
// have landed is load[i+1], which was issued last, so no smaller count can express it.
template <uint32_t TD0, uint32_t TD1>
__global__ void tdm_write_dbuf(float* __restrict__ rem, const float* __restrict__ loc,
                               uint32_t num_tiles) {
  extern __shared__ char smem[];
  const uint32_t TILE = TD0 * TD1, TB = TILE * 4;
  if (threadIdx.x != 0) return;
  uint32_t t = blockIdx.x;
  int cur = 0;
  if (t >= num_tiles) return;
  {
    uint32_t a[4], b[8];
    tdm_desc2d(a, b, 0, loc + (size_t)t * TILE, TD0, TD1);
    __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b),
                                        sg2v{0, 0, 0, 0}, sg3v{0, 0, 0, 0},
                                        sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
  }
  for (; t < num_tiles; t += gridDim.x) {
    uint32_t tn = t + gridDim.x;
    s_wait_tensorcnt(0);  // this tile's load, and the previous tile's store
    {
      uint32_t a[4], b[8];
      tdm_desc2d(a, b, cur * TB, rem + (size_t)t * TILE, TD0, TD1);
      __builtin_amdgcn_tensor_store_from_lds(
          __builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b), sg2v{0, 0, 0, 0},
          sg3v{0, 0, 0, 0}, sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
    }
    if (tn < num_tiles) {
      uint32_t a[4], b[8];
      tdm_desc2d(a, b, (cur ^ 1) * TB, loc + (size_t)tn * TILE, TD0, TD1);
      __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b),
                                          sg2v{0, 0, 0, 0}, sg3v{0, 0, 0, 0},
                                          sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
    }
    cur ^= 1;
  }
  s_wait_tensorcnt(0);
}

// Deep-pipelined staged copy, single issuer. tdm_write serialises a load batch, a full drain, a
// store batch and another full drain, so the local read and the remote write of a block never
// overlap; the only reason it still reaches 1643.5 is that other blocks cover the gap.
// tdm_write_dbuf overlaps them but its wait has to stay at 0 because the load it depends on was
// issued last.
//
// Here the loads run D tiles ahead of the stores in a fixed order -- L(0..D-1), then per round i a
// store of tile i followed by a load of tile i+D -- so the operation a round depends on is never
// the most recent one and a partial wait can express it. Two constraints, both assuming the tensor
// queue retires in issue order, which is what would make s_wait_tensorcnt(K) mean "all but the last
// K have landed":
//   store i reads the buffer load i filled, and load i is followed by 2(D-1) operations, so
//   K=2(D-1); load i+D overwrites the buffer store i+D-NBUF read, and that store is followed by
//   2*NBUF-2D-1
//     operations, which must be at least K, giving NBUF >= 2D.
// The tail runs on wait(0): the prefetch stops early there, and the counts above only hold while
// every round issues exactly two operations.
//
// KEPT AS A REFUTATION, NOT AS A VARIANT TO USE. The in-order assumption is false. Verify at 16 GB:
// D=2/NBUF=4 BAD, D=3/NBUF=6 BAD, D=2/NBUF=8 BAD, D=4/NBUF=8 ok, D=2/NBUF=3 (below the bound, built
// with DEEPFORCE as the negative control) BAD. D=2/NBUF=8 is the decisive one: eight buffers at
// depth two leaves the reuse constraint with enormous slack and it still corrupts, so what breaks
// is the load-before-store dependency, i.e. a partial wait does not guarantee load i has landed.
// D=4 passing is timing slack, not semantics -- loads are local reads and stores are remote, and
// K=6 leaves more of it -- so it is not safe either. Independently of correctness there is nothing
// here: the fastest point, D=2/NBUF=4 at 1642.5, is below the 1643.6-1644.0 the plain staged kernel
// held in the same process while still paying both drains. Removing the drains buys nothing, so
// they were never on the critical path.
template <uint32_t TD0, uint32_t TD1, int D, int NBUF>
__global__ void tdm_write_deep(float* __restrict__ rem, const float* __restrict__ loc,
                               uint32_t num_tiles) {
  extern __shared__ char smem[];
  // -DDEEPFORCE lifts the bound so NBUF=2D-1 can be built and shown to corrupt. Passing at 2D while
  // failing at 2D-1 is the evidence for the in-order retire assumption; passing at both would mean
  // the bound is not what makes it work and the derivation above is wrong.
#ifndef DEEPFORCE
  static_assert(NBUF >= 2 * D, "partial wait cannot cover buffer reuse unless NBUF >= 2*D");
#endif
  const uint32_t TILE = TD0 * TD1, TB = TILE * 4;
  if (threadIdx.x != 0) return;
  constexpr int K = 2 * (D - 1);
  // Same tile order tdm_write<PIPE=D> walks, so the only difference between the two is the
  // pipeline.
  auto tile_at = [&](uint32_t j) -> uint32_t {
    return (blockIdx.x + (j / D) * gridDim.x) * D + (j % D);
  };
  auto issue_load = [&](uint32_t t, uint32_t b) {
    uint32_t a[4], c[8];
    tdm_desc2d(a, c, b * TB, loc + (size_t)t * TILE, TD0, TD1);
    __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, c),
                                        sg2v{0, 0, 0, 0}, sg3v{0, 0, 0, 0},
                                        sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
  };
  auto issue_store = [&](uint32_t t, uint32_t b) {
    uint32_t a[4], c[8];
    tdm_desc2d(a, c, b * TB, rem + (size_t)t * TILE, TD0, TD1);
    __builtin_amdgcn_tensor_store_from_lds(__builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, c),
                                           sg2v{0, 0, 0, 0}, sg3v{0, 0, 0, 0},
                                           sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
  };

  uint32_t nl = 0;
  for (; nl < (uint32_t)D; nl++) {
    uint32_t t = tile_at(nl);
    if (t >= num_tiles) break;
    issue_load(t, nl % NBUF);
  }
  uint32_t i = 0;
  if (nl == (uint32_t)D) {
    for (;; i++) {
      uint32_t tn = tile_at(i + D);
      if (tn >= num_tiles) break;
      s_wait_tensorcnt(K);
      issue_store(tile_at(i), i % NBUF);
      issue_load(tn, (i + D) % NBUF);
    }
  }
  for (;; i++) {
    uint32_t t = tile_at(i);
    if (t >= num_tiles) break;
    s_wait_tensorcnt(0);
    issue_store(t, i % NBUF);
  }
  s_wait_tensorcnt(0);
}

// Staging read on the vector path, egress on TDM. The pure-TDM copy stages with tensor_load_to_lds,
// which shares the tensor queue and the single s_wait_tensorcnt with the store, so the two
// serialise no matter how they are buffered -- that is why double buffering did not help. Plain
// vector loads use a different path, so the next tile's staging overlaps the current tile's store
// for real. ISS waves per block each issue their own tile, so a block covers ISS tiles per
// iteration and the grid can be cut by the same factor for the same number of descriptor streams --
// fewer blocks per CU, better occupancy. The store-only probe gained 12 GB/s from exactly this
// (1628 -> 1640 at half the blocks). Staging stays a whole-block cooperative copy over all ISS
// tiles rather than each wave fetching its own: a wave alone would move 8x the bytes per lane and
// the vector read, not the issue rate, is what limits this variant. LDS is TDM_NBUF groups of ISS
// tiles, since the buffer a store is reading must not be restaged until that store has retired.
template <uint32_t TD0, uint32_t TD1, int ISS>
__global__ void tdm_store_cuload(float* __restrict__ rem, const float* __restrict__ loc,
                                 uint32_t num_tiles) {
  extern __shared__ char smem[];
  const uint32_t TILE = TD0 * TD1, TB = TILE * 4, V = TILE / 4;
  const uint32_t W = warpSize, wv = threadIdx.x / W, lane = threadIdx.x - wv * W;
  const uint32_t step = gridDim.x * ISS;
  uint32_t tbase = blockIdx.x * ISS, cur = 0;
  if (tbase >= num_tiles) return;
  // Wave j owns [j*LDSPART, j*LDSPART+TDM_NBUF*TB): one partition per issuer, rounds inside it.
  for (int j = 0; j < ISS; j++) {
    uint32_t tn = tbase + j;
    if (tn < num_tiles) {
      float4* d = (float4*)(smem + (size_t)j * LDSPART);
      const float4* s = (const float4*)(loc + (size_t)tn * TILE);
      for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) d[i] = s[i];
    }
  }
  __syncthreads();
  for (; tbase < num_tiles; tbase += step) {
    uint32_t nxt = cur + 1;
    if (nxt == TDM_NBUF) nxt = 0;
    if (wv < ISS && lane == 0) {
      uint32_t t = tbase + wv;
      if (t < num_tiles) {
        uint32_t a[4], b[8];
        tdm_desc2d(a, b, wv * LDSPART + cur * TB, rem + (size_t)t * TILE, TD0, TD1);
        __builtin_amdgcn_tensor_store_from_lds(
            __builtin_bit_cast(sg0v, a), __builtin_bit_cast(sg1v, b), sg2v{0, 0, 0, 0},
            sg3v{0, 0, 0, 0}, sgxv{0, 0, 0, 0, 0, 0, 0, 0}, TDM_CPOL);
      }
    }
    for (int j = 0; j < ISS; j++) {
      uint32_t tn = tbase + step + j;
      if (tn < num_tiles) {
        float4* d = (float4*)(smem + (size_t)j * LDSPART + (size_t)nxt * TB);
        const float4* s = (const float4*)(loc + (size_t)tn * TILE);
        for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) d[i] = s[i];
      }
    }
    if (wv < ISS && lane == 0) s_wait_tensorcnt(TDMCU_WAITN);
    __syncthreads();
    cur = nxt;
  }
}

static void feature_check(int dev) {
  int vmm = 0, fab = 0;
  HIPCHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev));
  HIPCHECK(hipDeviceGetAttribute(&fab, hipDeviceAttributeHandleTypeFabricSupported, dev));
  hipDeviceProp_t p;
  HIPCHECK(hipGetDeviceProperties(&p, dev));
  g_numCU = p.multiProcessorCount;
  if (!vmm || !fab) {
    fprintf(stderr, "[FATAL] VMM/fabric unsupported on dev %d\n", dev);
    exit(2);
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  bool listen_mode = false, connect_mode = false;
  if (argc >= 2 && !strcmp(argv[1], "listen"))
    listen_mode = true;
  else if (argc >= 2 && !strcmp(argv[1], "connect"))
    connect_mode = true;
  if (!listen_mode && !connect_mode) {
    fprintf(stderr,
            "usage:\n  %s listen  [-port=N] [-gpu=X]\n  %s connect (<peer_ip>|-peer_ip=IP) "
            "[-port=N] [-gpu=X]\n"
            "  -port default 55552; omit -gpu => all GPUs (GPU i <-> peer GPU i)\n",
            argv[0], argv[0]);
    return 1;
  }
  int port = 55552, gpu = -1;
  const char* peer_ip = nullptr;
  for (int i = 2; i < argc; i++) {
    if (!strncmp(argv[i], "-port=", 6))
      port = atoi(argv[i] + 6);
    else if (!strncmp(argv[i], "-gpu=", 5))
      gpu = atoi(argv[i] + 5);
    else if (!strncmp(argv[i], "-peer_ip=", 9))
      peer_ip = argv[i] + 9;
    else
      peer_ip = argv[i];  // positional peer ip (connect)
  }
  if (connect_mode && !peer_ip) {
    fprintf(stderr, "connect needs <peer_ip>\n");
    return 1;
  }
  bool reporter = !listen_mode;

  // local GPU list: explicit single, or all
  int devcount = 0;
  HIPCHECK(hipGetDeviceCount(&devcount));
  int list[MAXG], ndev;
  if (gpu >= 0) {
    list[0] = gpu;
    ndev = 1;
  } else {
    ndev = devcount > MAXG ? MAXG : devcount;
    for (int i = 0; i < ndev; i++) list[i] = i;
  }
  feature_check(list[0]);

  // rendezvous
  int fd;
  if (listen_mode) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons(port);
    if (bind(s, (sockaddr*)&a, sizeof(a)) < 0) {
      perror("bind");
      exit(3);
    }
    listen(s, 1);
    printf("[listen] waiting on :%d\n", port);
    fd = accept(s, nullptr, nullptr);
    close(s);
  } else {
    fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    if (inet_pton(AF_INET, peer_ip, &a.sin_addr) != 1) {
      fprintf(stderr, "bad ip\n");
      exit(3);
    }
    int ok = 0;
    for (int i = 0; i < 60 && !ok; ++i) {
      if (connect(fd, (sockaddr*)&a, sizeof(a)) == 0) {
        ok = 1;
        break;
      }
      usleep(500000);
      close(fd);
      fd = socket(AF_INET, SOCK_STREAM, 0);
    }
    if (!ok) {
      perror("connect");
      exit(3);
    }
  }
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  // agree on GPU count = min(local, peer); pair by index
  uint32_t myc = ndev, peerc = 0;
  send_all(fd, &myc, 4);
  recv_all(fd, &peerc, 4);
  int NG = (int)(myc < peerc ? myc : peerc);
  if (reporter && NG != (int)myc)
    printf("[note] peer has %u GPUs, this node %u -> using %d pairs\n", peerc, myc, NG);

  // per-GPU: export L[i], alloc local + stream
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypeFabric;
  hipMemAccessDesc ad = {};
  ad.flags = hipMemAccessFlagsProtReadWrite;
  void *L[MAXG], *R[MAXG], *l_src[MAXG], *l_dst[MAXG];
  size_t total[MAXG];
  hipMemGenericAllocationHandle_t hL[MAXG];
  hipStream_t st[MAXG], st2[MAXG];
  hipMemFabricHandle_t fhL[MAXG];
  for (int i = 0; i < NG; i++) {
    int d = list[i];
    HIPCHECK(hipSetDevice(d));
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = d;
    size_t g = 0;
    HIPCHECK(hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityRecommended));
    total[i] = ((MAX_BYTES + g - 1) / g) * g;
    HIPCHECK(hipMemCreate(&hL[i], total[i], &prop, 0));
    HIPCHECK(hipMemAddressReserve(&L[i], total[i], 0, 0, 0));
    HIPCHECK(hipMemMap(L[i], total[i], 0, hL[i], 0));
    ad.location.type = hipMemLocationTypeDevice;
    ad.location.id = d;
    HIPCHECK(hipMemSetAccess(L[i], total[i], &ad, 1));
    HIPCHECK(hipMemExportToShareableHandle(&fhL[i], hL[i], hipMemHandleTypeFabric, 0));
    HIPCHECK(hipMalloc(&l_src[i], MAX_BYTES));
    HIPCHECK(hipMalloc(&l_dst[i], MAX_BYTES));
    fillk<<<1024, 256>>>((uint32_t*)l_src[i], MAX_BYTES / 4);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipStreamCreate(&st[i]));
    HIPCHECK(hipStreamCreate(&st2[i]));
  }
  // exchange NG handles (send all mine, recv all peer's), import R[i]
  for (int i = 0; i < NG; i++) {
    uint64_t sz = total[i];
    send_all(fd, &sz, 8);
    send_all(fd, &fhL[i], sizeof(fhL[i]));
  }
  size_t psz[MAXG];
  hipMemFabricHandle_t fhR[MAXG];
  for (int i = 0; i < NG; i++) {
    recv_all(fd, &psz[i], 8);
    recv_all(fd, &fhR[i], sizeof(fhR[i]));
  }
  for (int i = 0; i < NG; i++) {
    HIPCHECK(hipSetDevice(list[i]));
    hipMemGenericAllocationHandle_t hR;
    HIPCHECK(hipMemImportFromShareableHandle(&hR, (void*)&fhR[i], hipMemHandleTypeFabric));
    HIPCHECK(hipMemAddressReserve(&R[i], psz[i], 0, 0, 0));
    HIPCHECK(hipMemMap(R[i], psz[i], 0, hR, 0));
    ad.location.id = list[i];
    HIPCHECK(hipMemSetAccess(R[i], psz[i], &ad, 1));
  }

// Blocks per CU each transport launches when neither sweep is driving the grid width. Both sweeps
// override these -- the matrix from CUMUL/TDMMUL, the block sweep from its own list -- so these
// only decide the plain run and the full-grid reference row the block sweep prints. 64 and 32 are
// what the scripts have always passed; they live here now so a build outside the scripts matches.
#ifndef BLKMUL
#define BLKMUL 64
#endif
#ifndef WTH
#define WTH 512
#endif
#ifndef TWBLK
#define TWBLK 32
#endif
#ifndef TWTH
#define TWTH 256
#endif
#ifndef HSPLIT
// Percent of the payload on the CU path in the hybrid variant. The optimum moved from 35 to 40 when
// the TDM side gained multi-issue (CUISS): a stronger TDM path earns a larger share of the bytes.
// At 40 the hybrid measures 1646.1-1646.4 against 1642 at 38, 1645.5 at 45-50 and 1639.4 for pure
// CU.
#define HSPLIT 40
#endif
  // Runtime so a range of splits can be measured in one process. Sweeping it by recompiling costs
  // more in build time than in measurement, and cross-build drift here is ~3 GB/s, larger than the
  // differences between neighbouring splits.
  static int g_hsplit = HSPLIT;
#ifndef AB_ROUNDS
#define AB_ROUNDS 1  // repeats of the whole table inside one process, to separate drift from config
#endif
  // blocks / wblocks are deliberately mutable and captured by reference: BLKSWEEP below rewrites
  // them between measurements so the grid width of both transports can be walked inside one
  // process. Every kernel here is a grid-stride loop, so any block count is correct, only slower.
  int TH = WTH, blocks = g_numCU * BLKMUL, rblocks = g_numCU * 8, rTH = 256;
  int wblocks = g_numCU * TWBLK, wTH = TWTH;
  const size_t RTILE = (size_t)RTD0 * RTD1, RTB = RTILE * 4;
  size_t rshared = (size_t)RPIPE * RTB;

  // aggregate bench over all NG pairs, concurrently; returns total GB/s
  // Every variant checks for a launch failure right after its warmup. Without it a rejected launch
  // (an LDS request over the limit, say) costs no time and is reported as a bandwidth figure: an
  // oversized buffer count once produced 7.9e7 GB/s here rather than an error.
  auto write_all = [&](size_t S) -> double {
    size_t n = S / 16;
    auto Ck = [&](int i) {
      if (g_storeonly) {
        switch (g_unroll) {
          case 2:
            storek_u<2><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], n);
            break;
          case 4:
            storek_u<4><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], n);
            break;
          case 8:
            storek_u<8><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], n);
            break;
          case 16:
            storek_u<16><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], n);
            break;
          case 32:
            storek_u<32><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], n);
            break;
          default:
            storek_u<1><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], n);
            break;
        }
        return;
      }
      switch (g_unroll) {
        case 2:
          copyk_u<2><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], (uint4*)l_src[i], n);
          break;
        case 4:
          copyk_u<4><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], (uint4*)l_src[i], n);
          break;
        case 8:
          copyk_u<8><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], (uint4*)l_src[i], n);
          break;
        case 16:
          copyk_u<16><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], (uint4*)l_src[i], n);
          break;
        // 32 holds 512 B of loaded data per thread, i.e. 128 VGPRs before anything else. It is
        // past the point where occupancy survives, which is the point: at one block per CU there
        // is no occupancy to lose and in-flight bytes are all that matter.
        case 32:
          copyk_u<32><<<blocks, TH, 0, st[i]>>>((uint4*)R[i], (uint4*)l_src[i], n);
          break;
        default:
          copyk<<<blocks, TH, 0, st[i]>>>((uint4*)R[i], (uint4*)l_src[i], n);
          break;
      }
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < g_warmup; ++w) Ck(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < g_loop; ++l) Ck(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * S,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / g_loop);
  };
  auto write_nt_all = [&](size_t S) -> double {
    size_t n = S / 16;
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < g_warmup; ++w)
        copyk_nt<<<blocks, TH, 0, st[i]>>>((u32x4*)R[i], (const u32x4*)l_src[i], n);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < g_loop; ++l)
        copyk_nt<<<blocks, TH, 0, st[i]>>>((u32x4*)R[i], (const u32x4*)l_src[i], n);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * S,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / g_loop);
  };
  auto read_all = [&](size_t S) -> double {
    uint32_t nt = (uint32_t)(S / RTB);
    if (nt == 0) return write_all(S);  // tiny -> approx
    auto Lk = [&](int i) {
      tdm_read<RTD0, RTD1, RPIPE>
          <<<rblocks, rTH, rshared, st[i]>>>((const float*)R[i], (float*)l_dst[i], nt);
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < WARMUP; ++w) Lk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < LOOP; ++l) Lk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * (size_t)nt * RTB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / LOOP);
  };

  auto tdmwr_all = [&](size_t S, bool noload, bool partial = false) -> double {
    uint32_t nt = (uint32_t)(S / RTB);
    if (nt == 0) return 0;
    auto Wk = [&](int i) {
      if (noload && partial)
        tdm_write<RTD0, RTD1, RPIPE, true, TDM_WAITN>
            <<<wblocks, wTH, rshared, st[i]>>>((float*)R[i], (const float*)l_src[i], nt);
      else if (noload)
        tdm_write<RTD0, RTD1, RPIPE, true>
            <<<wblocks, wTH, rshared, st[i]>>>((float*)R[i], (const float*)l_src[i], nt);
      else
        tdm_write<RTD0, RTD1, RPIPE, false>
            <<<wblocks, wTH, rshared, st[i]>>>((float*)R[i], (const float*)l_src[i], nt);
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < g_warmup; ++w) Wk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < g_loop; ++l) Wk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * (size_t)nt * RTB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / g_loop);
  };

  auto tdmmw_all = [&](size_t S) -> double {
    uint32_t nt = (uint32_t)(S / RTB);
    if (nt == 0) return 0;
    auto Wk = [&](int i) {
      tdm_write_mw<RTD0, RTD1, RPIPE, MWISS, TDM_WAITN>
          <<<wblocks, wTH, (size_t)MWISS * MW_SPAN, st[i]>>>((float*)R[i], (const float*)l_src[i],
                                                             nt, 0, 0);
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < WARMUP; ++w) Wk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < LOOP; ++l) Wk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * (size_t)nt * RTB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / LOOP);
  };
  // Staged copy with MWSISS waves per block issuing, each out of its own LDS span. Same kernel as
  // the probe above with the staging load switched back on, so it is a real copy and the verify
  // covers it.
  auto tdmmws_all = [&](size_t S) -> double {
    // A fixed tile decides how many blocks can participate, and for anything but the largest
    // payloads that number is below the grid: at 1 MB and an 8 KB tile there are 128 tiles, MWSPIPE
    // of them per issuing wave, so 64 waves -- 8 blocks -- and the rest of the grid idles no matter
    // how wide the launch is. Sizing the tile to the payload instead keeps every block fed; the
    // compiled tile stays the cap, since a larger one would need LDS the launch did not reserve.
    uint32_t rows = RTD1, d0 = RTD0;
    size_t tileB = RTB;
    if (g_dyntile) {
      const size_t rowB = (size_t)RTD0 * 4, want = (size_t)wblocks * MWSISS * MWSPIPE;
      size_t tb = S / (want ? want : 1);  // bytes per issuing slot if the payload is split evenly
      if (tb > RTB) tb = RTB;             // the compiled tile is the cap: LDS is reserved for it
      if (tb < g_dyntile_min) tb = g_dyntile_min;
      size_t p = g_dyntile_min;
      while (p * 2 <= tb) p *= 2;  // power of two so the payload divides evenly
      tileB = p;
      if (tileB >= rowB) {
        d0 = RTD0;
        rows = (uint32_t)(tileB / rowB);
      } else {
        rows = 1;
        d0 = (uint32_t)(tileB / 4);
      }  // narrower than one row
    }
    uint32_t nt = (uint32_t)(S / tileB);
    if (nt == 0) return 0;
    const uint32_t rowarg = (rows == RTD1) ? 0u : rows, d0arg = (d0 == RTD0) ? 0u : d0;
    size_t sh = (size_t)MWSISS * MWS_SPAN;
    auto Wk = [&](int i) {
      tdm_write_mw<RTD0, RTD1, MWSPIPE, MWSISS, 0, false, MWS_SPAN>
          <<<wblocks, wTH, sh, st[i]>>>((float*)R[i], (const float*)l_src[i], nt, d0arg, rowarg);
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < WARMUP; ++w) Wk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < LOOP; ++l) Wk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    g_dyntile_bytes = tileB;
    return gbps((size_t)NG * (size_t)nt * tileB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / LOOP);
  };
  auto tdmdp_all = [&](size_t S) -> double {
    uint32_t nt = (uint32_t)(S / RTB);
    if (nt == 0) return 0;
    size_t sh = (size_t)DEEPNBUF * RTB;
    auto Wk = [&](int i) {
      tdm_write_deep<RTD0, RTD1, DEEPD, DEEPNBUF>
          <<<wblocks, wTH, sh, st[i]>>>((float*)R[i], (const float*)l_src[i], nt);
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < WARMUP; ++w) Wk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < LOOP; ++l) Wk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * (size_t)nt * RTB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / LOOP);
  };
  auto tdmdb_all = [&](size_t S) -> double {
    uint32_t nt = (uint32_t)(S / RTB);
    if (nt == 0) return 0;
    size_t dbshared = (size_t)2 * RTB;
    auto Wk = [&](int i) {
      tdm_write_dbuf<RTD0, RTD1>
          <<<wblocks, wTH, dbshared, st[i]>>>((float*)R[i], (const float*)l_src[i], nt);
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < WARMUP; ++w) Wk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < LOOP; ++l) Wk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * (size_t)nt * RTB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / LOOP);
  };
  // multi=false keeps a single-issuer instance alive as the in-binary control for the CUISS change.
  auto tdmcu_all = [&](size_t S, bool multi) -> double {
    uint32_t nt = (uint32_t)(S / RTB);
    if (nt == 0) return 0;
    size_t dbshared = (size_t)(multi ? CUISS : 1) * LDSPART;
    auto Wk = [&](int i) {
      if (multi)
        tdm_store_cuload<RTD0, RTD1, CUISS>
            <<<wblocks, wTH, dbshared, st[i]>>>((float*)R[i], (const float*)l_src[i], nt);
      else
        tdm_store_cuload<RTD0, RTD1, 1>
            <<<wblocks, wTH, dbshared, st[i]>>>((float*)R[i], (const float*)l_src[i], nt);
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < WARMUP; ++w) Wk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < LOOP; ++l) Wk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * (size_t)nt * RTB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / LOOP);
  };
  // Both transports at once, on separate streams, splitting the payload HSPLIT/100 to CU and the
  // rest to TDM. This is here to decide whether ~1646 GB/s is the link saturating or one initiator
  // running out of steam: if two independent initiators together still land at ~1646 it is the
  // link, and no amount of work on either path can pass it.
  auto hybrid_all = [&](size_t S) -> double {
    uint32_t ntot = (uint32_t)(S / RTB);
    if (ntot == 0) return 0;
    uint32_t ncu = (uint32_t)((uint64_t)ntot * g_hsplit / 100), ntdm = ntot - ncu;
    size_t cub = (size_t)ncu * RTB, dbshared = (size_t)CUISS * LDSPART;
    // The TDM side runs the staged copy by default: it measures 1643.9 against 1635 for the
    // CU-staged one, both verified full copies. HYBTDM=0 selects the older CU-staged kernel.
    auto Wk = [&](int i) {
      if (ncu) copyk<<<blocks, TH, 0, st[i]>>>((uint4*)R[i], (const uint4*)l_src[i], cub / 16);
#if HYBTDM
      if (ntdm)
        tdm_write<RTD0, RTD1, RPIPE, false><<<wblocks, wTH, rshared, st2[i]>>>(
            (float*)((char*)R[i] + cub), (const float*)((char*)l_src[i] + cub), ntdm);
#else
      if (ntdm)
        tdm_store_cuload<RTD0, RTD1, CUISS><<<wblocks, wTH, dbshared, st2[i]>>>(
            (float*)((char*)R[i] + cub), (const float*)((char*)l_src[i] + cub), ntdm);
#endif
    };
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int w = 0; w < WARMUP; ++w) Wk(i);
      HIPCHECK(hipGetLastError());
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
      HIPCHECK(hipStreamSynchronize(st2[i]));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      for (int l = 0; l < LOOP; ++l) Wk(i);
    }
    for (int i = 0; i < NG; i++) {
      HIPCHECK(hipSetDevice(list[i]));
      HIPCHECK(hipStreamSynchronize(st[i]));
      HIPCHECK(hipStreamSynchronize(st2[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return gbps((size_t)NG * (size_t)ntot * RTB,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / LOOP);
  };
  // kind: 0=CU write, 1=TDM write (staged), 2=TDM write (no staging load), 3=TDM store + CU
  // staging,
  //       4=TDM write double buffered, 5=CU and TDM concurrently, 6=kind 2 with a partial wait,
  //       7=kind 6 with MWISS waves per block issuing instead of one, 8=kind 3 with CUISS issuers,
  //       9=kind 1 with MWSISS waves per block issuing, one LDS span each,
  //       10=kind 1 with the loads run DEEPD tiles ahead and a partial wait instead of two drains,
  //       11=kind 0 on non-temporal loads and stores
  auto run_kind = [&](size_t S, int kind) -> double {
    switch (kind) {
      case 0:
        return write_all(S);
      case 1:
        return tdmwr_all(S, false);
      case 2:
        return tdmwr_all(S, true);
      case 3:
        return tdmcu_all(S, false);
      case 4:
        return tdmdb_all(S);
      case 5:
        return hybrid_all(S);
      case 6:
        return tdmwr_all(S, true, true);
      case 7:
        return tdmmw_all(S);
      case 8:
        return tdmcu_all(S, true);
      case 9:
        return tdmmws_all(S);
      case 10:
        return tdmdp_all(S);
      case 11:
        return write_nt_all(S);
      default:
        return read_all(S);
    }
  };
  // One way: only the connect side drives, so the number is that single direction.
  auto oneway = [&](size_t S, int kind) -> double {
    double v = 0;
    barrier(fd);
    if (reporter) v = run_kind(S, kind);
    barrier(fd);
    return v;
  };
  // Zero the peer window, run the variant, then read three windows back across the fabric and
  // compare against the source pattern. Cheap, but enough to catch a variant or a cache policy that
  // reports a bandwidth without the bytes actually landing at the peer.
  auto verify_kind = [&](size_t S, int kind) -> int {
    static const size_t W = 1024;
    static uint32_t buf[W];
    if (reporter) {
      HIPCHECK(hipSetDevice(list[0]));
      HIPCHECK(hipMemset(R[0], 0, S));
      HIPCHECK(hipDeviceSynchronize());
    }
    oneway(S, kind);
    if (!reporter) return 0;
    size_t offs[3] = {0, (S / 2) & ~(size_t)4095, S - 4096};
    int bad = 0;
    for (int k = 0; k < 3; k++) {
      HIPCHECK(hipMemcpy(buf, (char*)R[0] + offs[k], W * 4, hipMemcpyDeviceToHost));
      for (size_t j = 0; j < W; j++)
        if (buf[j] != SRCPAT(offs[k] / 4 + j)) {
          bad++;
          break;
        }
    }
    return bad;
  };
#ifndef ONLY_1WAY
  // Both ways: both sides push at once, but the number returned is only this side's own direction,
  // timed on this side. The peer's figure is still exchanged so both ends stay in step.
  auto bothway = [&](size_t S, int kind) -> double {
    barrier(fd);
    double mine = run_kind(S, kind);
    (void)exch(fd, mine);
    barrier(fd);
    return mine;
  };
#endif

  // Under ONLY_1WAY the 2way columns are omitted rather than printed as zeros: nothing measures
  // them, and a table of literal 0.0 reads like a failure. Note this shifts the column numbers,
  // which epcheck.sh parses positionally.
  if (reporter) {
    printf("copy bandwidth of THIS gpu -> peer, timed on this gpu (one direction's bytes only)\n");
#ifdef ONLY_1WAY
    printf("  1way = only this gpu pushes;  2way not measured (built with ONLY_1WAY)\n");
#else
    printf("  1way = only this gpu pushes;  2way = peer pushes back at the same time\n");
    printf(
        "  both columns count only this direction's bytes, so the link total under 2way is twice "
        "the figure\n");
#endif
    printf("  CU    = uint4 copy on every thread\n");
    printf(
        "  TDM   = tensor_load_to_lds then tensor_store_from_lds to the peer, %d tiles per batch\n",
        RPIPE);
    printf("  TDMc2 = CU staging read + TDM store, %d issuing waves, %d bytes of LDS per wave\n",
           CUISS, LDSPART);
#ifdef ONLY_1WAY
    printf(
        "  TDMnl = store only, no source read (not a copy; the gate reports it BAD by design)\n");
    printf(
        "  TDMnw = TDMnl but waiting on s_wait_tensorcnt(%d) instead of draining the queue every "
        "batch\n",
        TDM_WAITN);
    printf(
        "  TDMmw = TDMnw with %d waves per block issuing descriptors instead of one (same LDS, "
        "same grid)\n",
        MWISS);
    printf("  TDMcu = TDMc2 with a single issuing wave\n");
    printf("  TDMms = TDM (staged) with %d waves per block issuing, %d bytes of LDS each\n", MWSISS,
           MWS_SPAN);
#endif
#ifdef DEEPONLY
    printf(
        "  TDMdp = TDM (staged) with loads %d tiles ahead over %d buffers, wait(%d) instead of two "
        "drains\n",
        DEEPD, DEEPNBUF, 2 * (DEEPD - 1));
#endif
    printf("  HYB   = CU and %s concurrently on two streams, %d%% of the bytes on the CU path\n",
           HYBTDM ? "TDM" : "TDMc2", HSPLIT);
    printf("  grid: CU %d x %d, TDM %d x %d, tile %ux%u pipe %d\n", blocks, TH, wblocks, wTH, RTD0,
           RTD1, RPIPE);
    // The grid is far larger than what fits on the device at once -- BLKMUL=64 blocks per CU is a
    // queue depth, not an occupancy. Printing the real limit keeps "how much hardware did this use"
    // from being read off the launch parameters, which is wrong by more than an order of magnitude.
    {
      int oc = 0, ot = 0, oc2 = 0;
      hipDeviceProp_t dp;
      HIPCHECK(hipGetDeviceProperties(&dp, list[0]));
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&oc, (const void*)copyk, TH, 0));
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &ot, (const void*)tdm_write<RTD0, RTD1, RPIPE, false>, wTH, rshared));
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &oc2, (const void*)tdm_store_cuload<RTD0, RTD1, CUISS>, wTH, (size_t)CUISS * LDSPART));
      printf("  device: %d CUs, warpSize %d, %zu KB LDS/CU\n", dp.multiProcessorCount, dp.warpSize,
             (size_t)dp.maxSharedMemoryPerMultiProcessor / 1024);
      printf(
          "  max resident blocks per CU: CU %d, TDM %d, TDMc2 %d"
          "  -> resident waves on the whole device: CU %d, TDM %d, TDMc2 %d\n",
          oc, ot, oc2, oc * dp.multiProcessorCount * (TH / dp.warpSize),
          ot * dp.multiProcessorCount * (wTH / dp.warpSize),
          oc2 * dp.multiProcessorCount * (wTH / dp.warpSize));
      printf(
          "  launched blocks are %.0fx / %.0fx that, so the grid is a queue, not an occupancy\n\n",
          (double)blocks / (oc * dp.multiProcessorCount),
          (double)wblocks / (ot * dp.multiProcessorCount));
    }
#ifdef ONLY_1WAY
    printf("%9s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "size", "CU 1way",
           "TDM 1way", "TDMdb 1w", "TDMnl 1w", "TDMnw 1w", "TDMmw 1w", "TDMcu 1w", "TDMc2 1w",
           "TDMms 1w", "TDMdp 1w", "HYB 1w");
    printf("%9s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "----", "-------",
           "--------", "--------", "--------", "--------", "--------", "--------", "--------",
           "--------", "--------", "------");
#else
    // Only variants the correctness gate passes appear here. The store-only probes are not copies
    // and the rejected ones are slower, so a table meant to be quoted has no business carrying
    // them.
    printf("%9s %10s %10s %10s %10s %10s %10s %10s %10s\n", "size", "CU 1way", "CU 2way",
           "TDM 1way", "TDM 2way", "TDMc2 1way", "TDMc2 2way", "HYB 1way", "HYB 2way");
    printf("%9s %10s %10s %10s %10s %10s %10s %10s %10s\n", "----", "-------", "-------",
           "--------", "--------", "----------", "----------", "--------", "--------");
#endif
  }
#ifndef NOVERIFY
  // Runs on the smallest size only; both sides must walk this loop because oneway() has barriers in
  // it.
  {
    size_t VS = SIZES[0];
    char vsz[16];
    b2s(VS, vsz);
    // TDMdp is only gated under DEEPONLY: it is a refuted variant that always reports BAD, and
    // leaving it in the default gate would make every run look like a failure.
#ifdef DEEPONLY
    const int vk[] = {0, 1, 2, 3, 4, 5, 8, 9, 10};
    const char* vn[] = {"CU", "TDM", "TDMnl", "TDMcu", "TDMdb", "HYB", "TDMc2", "TDMms", "TDMdp"};
    const int NV = 9;
#else
    const int vk[] = {0, 1, 2, 3, 4, 5, 8, 9};
    const char* vn[] = {"CU", "TDM", "TDMnl", "TDMcu", "TDMdb", "HYB", "TDMc2", "TDMms"};
    const int NV = 8;
#endif
    int vb[NV];
    for (int k = 0; k < NV; k++) vb[k] = verify_kind(VS, vk[k]);
    if (reporter) {
      printf("[VERIFY] size=%s", vsz);
      for (int k = 0; k < NV; k++) printf(" %s=%s", vn[k], vb[k] ? "BAD" : "ok");
      printf("   (TDMnl BAD is expected: it stores stale LDS without reading the source)\n\n");
    }
  }
#endif
  // MATRIX=1 replaces the table with a (block count) x (transfer size) sweep of the two transports
  // the gate reports, one cell per line. It exists because every other mode in this file samples
  // one corner of that plane: the table walks sizes at a fixed grid, BLKSWEEP walks the grid at a
  // fixed size, and neither shows where the two curves cross.
  //
  // Both ends must take this branch, and must derive the same cell order from the same environment,
  // or they deadlock: oneway() is a barrier, a measurement and a barrier, so the sequence of
  // barriers has to match on the two sides.
  //
  // Two properties of the plane are structural rather than measured, and reading the output without
  // them is misleading:
  //   - the TDM tile is RTD0 x RTD1 floats, so a transfer smaller than that carries no tile at all
  //   and
  //     tdmwr_all returns 0. Those cells are a 0, not a measurement.
  //   - the TDM kernel advances gridDim.x*PIPE tiles per iteration, so once the block count exceeds
  //     num_tiles/PIPE the extra blocks never enter the loop and the row stops responding to the CU
  //     axis. Same for the CU kernel against S/16 elements.
  if (getenv("MATRIX") && atoi(getenv("MATRIX"))) {
    size_t szl[64], cul[512];
    int nsz = 0, ncu = 0;
    if (const char* s = getenv("MATRIX_SZ"))
      nsz = parse_list(s, szl, 64);
    else
      for (size_t S = 1024; S <= MAX_BYTES && nsz < 64; S <<= 1) szl[nsz++] = S;
    if (const char* c = getenv("MATRIX_CU"))
      ncu = parse_list(c, cul, 512);
    else
      for (size_t b = 1; b <= (size_t)g_numCU && ncu < 512; b <<= 1) cul[ncu++] = b;
    // A fixed iteration count cannot serve a plane that spans six orders of magnitude in size and
    // two in bandwidth. Holding the bytes per cell roughly constant does: MINIT keeps the largest
    // cells from being timed off a single pass, MAXIT keeps the smallest from spending all their
    // time in launch overhead for no extra resolution.
    size_t budget = env_sz("MATRIX_BYTES", 1UL << 30);
    int minit = (int)env_sz("MATRIX_MINIT", 5), maxit = (int)env_sz("MATRIX_MAXIT", 200);
    // Blocks launched per unit of the CU axis. There is no CU masking here -- grid width is the
    // only lever -- so what a row means depends entirely on this multiplier, and the two transports
    // need different values to mean the same thing. At 1 the axis is "one block per CU", the
    // convention mori's dispatch and combine are pinned to. The device reports a residency of 4
    // blocks/CU for the CU kernel and 8 for the TDM one, and epcheck launches 64 and 32
    // respectively, i.e. it oversubscribes residency by 16x and 4x on purpose: its grid is a queue,
    // not an occupancy. Setting these to 64 and 32 therefore makes the right-hand edge of the sweep
    // the exact configuration the published baselines were measured with.
    size_t cumul = env_sz("MATRIX_CUMUL", 1), tdmmul = env_sz("MATRIX_TDMMUL", 1);
    // MATRIX_CUMASK=1 turns the axis into a physical CU count instead of a grid width. The stream
    // is rebuilt per row with hipExtStreamCreateWithCUMask so the hardware may only schedule on the
    // first N CUs; the grid still scales with N, so each of those N CUs gets the same cumul/tdmmul
    // blocks as it would at the full width. Without this the axis cannot answer "what does one CU
    // deliver": a 256-block launch already touches all 256 CUs, so every row from 4 up runs on the
    // whole device and differs only in how deep its queue is.
    bool cumask = env_sz("MATRIX_CUMASK", 0) != 0;
    g_storeonly = (int)env_sz("MATRIX_SO", 0);
    // MATRIX_DYNTILE=1 lets the tdmmws column pick its tile per cell instead of using the compiled
    // one, so the payload divides evenly across the issuing waves instead of leaving most of them
    // without a tile at the smaller sizes. On by default; MXCFG reports which way it ran. Set it to
    // 0 to reproduce a table taken with a fixed tile.
    g_dyntile = (int)env_sz("MATRIX_DYNTILE", 1);
    g_dyntile_min = env_sz("MATRIX_DYNMIN", 1024);
    // MATRIX_HYB=1 adds the two-stream variant to each cell. It is off by default because it
    // doubles the cell cost and because it is not one transport: it splits the payload between the
    // CU kernel and the CU-staged TDM one, so it is the answer to "does a second initiator help",
    // which only becomes the question once a single one has stopped scaling.
    bool hyb = env_sz("MATRIX_HYB", 0) != 0;
    bool c2 = env_sz("MATRIX_C2", 0) != 0;
    bool nt = env_sz("MATRIX_NT", 0) != 0;
    // Which kernel the TDM column reports. 9 is tdmmws_all and the default: MWSISS waves per block,
    // each driving its own tile stream out of its own LDS partition, MWSPIPE tiles per round. 1 is
    // tdm_write, one issuing wave per block, which is what the matrices recorded before this
    // default changed were taken with, so reproducing those needs MATRIX_TDMKIND=1. Both are
    // verified full copies, so the column stays the same kind of measurement; what changes is how
    // many waves inside one block issue descriptors, which the BLOCK axis cannot express -- a block
    // is one issuer at kind 1 no matter how wide it is launched.
    int tdmkind = (int)env_sz("MATRIX_TDMKIND", 9);
    // Block width as an innermost axis rather than a compile-time constant, one list per transport.
    // This is a lever on the per-launch fixed cost rather than on steady-state bandwidth: the same
    // number of threads delivered as fewer, wider blocks is less for the dispatcher to walk.
    // Sweeping it inside one process matters because the effect being chased is a few microseconds,
    // which is smaller than the drift between two builds of this file. Unroll depth of the CU copy
    // as an innermost axis: how many loads a thread keeps in flight. Only the CU path has it; the
    // TDM path's depth is RPIPE and is set at build time.
    size_t unl[8];
    int nun = 0;
    if (const char* u = getenv("MATRIX_UNS")) nun = parse_list(u, unl, 8);
    if (nun == 0) {
      unl[0] = 1;
      nun = 1;
    }
    size_t thl[16], wthl[16];
    int nth = 0, nwth = 0;
    if (const char* t = getenv("MATRIX_THS")) nth = parse_list(t, thl, 16);
    if (const char* w = getenv("MATRIX_WTHS")) nwth = parse_list(w, wthl, 16);
    if (nth == 0) {
      thl[0] = (size_t)TH;
      nth = 1;
    }
    if (nwth == 0) {
      wthl[0] = (size_t)wTH;
      nwth = 1;
    }
    if (reporter) {
      printf(
          "[MXCFG] cells=%dx%d budget=%zu minit=%d maxit=%d CU=%dthr TDM=%dthr tile=%ux%u "
          "pipe=%d\n",
          ncu, nsz, budget, minit, maxit, TH, wTH, RTD0, RTD1, RPIPE);
      printf("[MXCFG] tdm_tile_bytes=%zu (sizes below it cannot carry a tile and report 0)\n", RTB);
      printf("[MXCFG] blocks_per_cu: CU=%zu TDM=%zu (alloc=%zu) th_axis=%d wth_axis=%d\n", cumul,
             tdmmul, MAX_BYTES, nth, nwth);
      printf("[MXCFG] cu_axis=%s (device has %d CUs)\n",
             cumask ? "physical CU mask via hipExtStreamCreateWithCUMask"
                    : "grid width only, no masking",
             g_numCU);
    }
    // The multi-issuer column needs ISS waves per block and ISS partitions of LDS, and neither is
    // implied by the BLOCK axis. Neither fails loudly on its own either: a block with fewer waves
    // than issuers silently leaves the missing issuers' tiles uncopied, and a partition stride that
    // does not cover MWSPIPE tiles addresses past the allocation, which is how a node was lost on
    // 2026-08-11. Both ends evaluate this from the same environment and abort together rather than
    // deadlock.
    if (tdmkind == 9) {
      hipDeviceProp_t dp;
      HIPCHECK(hipGetDeviceProperties(&dp, list[0]));
      size_t lds = (size_t)MWSISS * MWS_SPAN, span = (size_t)MWSPIPE * MW_TILEB;
      int waves = (int)wthl[0] / dp.warpSize;
      if (reporter)
        printf(
            "[MXCFG] tdm column = tdmmws: %d waves x %d tiles x %zuB = %zuB per block per round,"
            " LDS %zuKB/block, block is %d waves of %d threads\n",
            MWSISS, MWSPIPE, (size_t)MW_TILEB, (size_t)MWSISS * span, lds / 1024, waves,
            dp.warpSize);
      if (reporter && g_dyntile)
        printf(
            "[MXCFG] dyntile=1: tile = clamp(bytes/(blocks*%d*%d), %zuB, %zuB) rounded down to a"
            " power of two; below one %dB row the row narrows instead\n",
            MWSISS, MWSPIPE, g_dyntile_min, (size_t)MW_TILEB, RTD0N * 4);
      if (waves < MWSISS) {
        if (reporter)
          printf(
              "[FATAL] block holds %d waves but %d must issue: the missing issuers'"
              " tiles would never be copied. Raise TWTH.\n",
              waves, MWSISS);
        return 1;
      }
      if (span > (size_t)MWS_SPAN) {
        if (reporter)
          printf("[FATAL] MWS_SPAN=%zuB does not cover %zuB of tiles per wave\n", (size_t)MWS_SPAN,
                 span);
        return 1;
      }
      if (lds > (size_t)dp.sharedMemPerBlock) {
        if (reporter)
          printf("[FATAL] LDS %zuB per block > device limit %zuB\n", lds,
                 (size_t)dp.sharedMemPerBlock);
        return 1;
      }
    }
    // A grid-stride kernel is correct at any block count, but that is an argument, not a
    // measurement, and the entire left edge of the plane rests on it. Check it once at the
    // narrowest grid.
    blocks = 1;
    wblocks = 1;
    g_loop = 1;
    g_warmup = 1;
    // Every unroll depth gets its own check at a size that is deliberately not a multiple of the
    // unrolled stride, so a tail the unrolled body fails to copy shows up as a mismatch instead of
    // as a bandwidth figure for a copy that moved fewer bytes than it was credited with.
    for (int ui = 0; ui < nun; ui++) {
      g_unroll = (int)unl[ui];
      int bad = verify_kind(1UL << 20, 0) | verify_kind((1UL << 20) + 16 * 3, 0);
      if (reporter) printf("[MXV] unroll=%d CU=%s\n", g_unroll, bad ? "BAD" : "ok");
      if (bad) {
        if (reporter) printf("[FATAL] unrolled copy is wrong, refusing to report its bandwidth\n");
        return 1;
      }
    }
    g_unroll = 1;
    int vb0 = verify_kind(1UL << 20, 0), vb1 = verify_kind(1UL << 20, tdmkind);
    int vbn = nt ? verify_kind(1UL << 20, 11) : 0;
    if (reporter) {
      printf("[MXV] blocks=1 size=1MB CU=%s TDM=%s", vb0 ? "BAD" : "ok", vb1 ? "BAD" : "ok");
      if (nt) printf(" NT=%s", vbn ? "BAD" : "ok");
      printf("\n");
    }

    for (int ci = 0; ci < ncu; ci++) {
      if (cumask) {
        // Rebuilt rather than set on the fly: a CU mask is a property of the queue the stream owns,
        // so it can only be chosen at creation. Sync before destroying or the pending work of the
        // previous row is cancelled and the next row times a launch that never ran.
        uint32_t m[32] = {0};
        int nw = (g_numCU + 31) / 32;
        if (nw > 32) nw = 32;
        size_t nset = cul[ci] < (size_t)g_numCU ? cul[ci] : (size_t)g_numCU;
        for (size_t b = 0; b < nset; b++) m[b >> 5] |= 1u << (b & 31);
        // st2 is only touched by the hybrid variant, and a masked stream costs a dedicated hardware
        // queue: masking it too doubled the queue demand and the second row failed to create one
        // ("out of memory"). Leave it alone unless the hybrid is actually being measured.
        for (int i = 0; i < NG; i++) {
          HIPCHECK(hipSetDevice(list[i]));
          HIPCHECK(hipStreamSynchronize(st[i]));
          HIPCHECK(hipStreamDestroy(st[i]));
          HIPCHECK(hipExtStreamCreateWithCUMask(&st[i], (uint32_t)nw, m));
          if (hyb) {
            HIPCHECK(hipStreamSynchronize(st2[i]));
            HIPCHECK(hipStreamDestroy(st2[i]));
            HIPCHECK(hipExtStreamCreateWithCUMask(&st2[i], (uint32_t)nw, m));
          }
        }
      }
      for (int si = 0; si < nsz; si++) {
        size_t S = szl[si];
        size_t k = budget / S;
        if (k < (size_t)minit) k = minit;
        if (k > (size_t)maxit) k = maxit;
        blocks = (int)(cul[ci] * cumul);
        wblocks = (int)(cul[ci] * tdmmul);
        g_loop = (int)k;
        g_warmup = (int)(k / 10) > 2 ? (int)(k / 10) : 2;
        for (int ti = 0; ti < nth; ti++) {
          for (int wi = 0; wi < nwth; wi++) {
            for (int ui = 0; ui < nun; ui++) {
              TH = (int)thl[ti];
              wTH = (int)wthl[wi];
              g_unroll = (int)unl[ui];
              double cv = oneway(S, 0);
              double tv = oneway(S, tdmkind);
              // TDMc2 stages through the vector path with CUISS waves issuing, so a block narrower
              // than that leaves the tiles owned by the missing issuers unwritten. Skipped rather
              // than measured: it would report a bandwidth for a copy that did not happen.
              double xv = (c2 && wTH >= CUISS * 32) ? oneway(S, 8) : 0;
              double nv = nt ? oneway(S, 11) : 0;
              double hv = hyb ? oneway(S, 5) : 0;
              // The launched grids and widths are printed, not just the axis value: the row label
              // is a choice made by the multiplier and the two paths do not share it, so the
              // numbers alone would not say what was run.
              if (reporter) {
                printf(
                    "[MX] cu=%zu cublk=%d cuth=%d tdmblk=%d tdmth=%d un=%d bytes=%zu iters=%d "
                    "CU=%.3f TDM=%.3f",
                    cul[ci], blocks, TH, wblocks, wTH, g_unroll, S, g_loop, cv, tv);
                if (g_dyntile) printf(" tileB=%zu", g_dyntile_bytes);
                if (c2) printf(" C2=%.3f", xv);
                if (nt) printf(" NT=%.3f", nv);
                if (hyb) printf(" HYB=%.3f", hv);
                printf("\n");
              }
            }
          }
        }
      }
    }
    barrier(fd);
    close(fd);
    return 0;
  }

  // Repeat the whole table AB_ROUNDS times inside one process. The variants are measured back to
  // back in a fixed order, so a value that decays round over round is drift within the process,
  // while a value that is stable here but differs from an earlier process points at state that
  // survives across processes. Comparing two variants measured in different processes is not valid
  // either way.
  for (int r = 0; r < AB_ROUNDS; r++) {
    for (int i = 0; i < N_SIZES; ++i) {
      size_t S = SIZES[i];
      char sz[16];
      b2s(S, sz);
#ifdef ONLY_1WAY
      // TDMnl and TDMnw differ only in the wait threshold and run adjacent, which is the only way
      // to resolve an effect this small: build-to-build drift is larger than the difference being
      // tested.
#ifdef HYBONLY
      // Tuning the hybrid split does not need the other variants; CU stays as a drift control.
      double c1 = oneway(S, 0), t1 = 0, d1 = 0, n1 = 0, w1 = 0, m1 = 0, u1 = 0, x1 = 0, s1 = 0,
             p1 = 0, h1 = oneway(S, 5);
#elif defined(CUONLY)
      // Tuning the CU-staged copy: its single-issuer control, the new one, the hybrid it feeds, and
      // CU.
      double c1 = oneway(S, 0), t1 = 0, d1 = 0, n1 = 0, w1 = 0, m1 = 0, u1 = oneway(S, 3),
             x1 = oneway(S, 8), s1 = 0, p1 = 0, h1 = oneway(S, 5);
#elif defined(MWSONLY)
      // Tuning the staged multi-issuer copy against the single-issuer staged copy it derives from,
      // measured back to back in one process because the difference is smaller than build-to-build
      // drift.
      double c1 = oneway(S, 0), t1 = oneway(S, 1), d1 = 0, n1 = 0, w1 = 0, m1 = 0, u1 = 0, x1 = 0,
             s1 = oneway(S, 9), p1 = 0, h1 = 0;
#elif defined(BLKONLY)
      // Companion to BLKSWEEP: the same three transports at the full grid, so the sweep has a
      // reference point measured under identical conditions at the end of its own curve.
      double c1 = oneway(S, 0), t1 = oneway(S, 1), d1 = 0, n1 = 0, w1 = 0, m1 = 0, u1 = 0,
             x1 = oneway(S, 8), s1 = 0, p1 = 0, h1 = 0;
#elif defined(DEEPONLY)
      // The deep pipeline against the staged kernel it replaces, plus the double-buffered attempt
      // that could not express a partial wait, all in one process.
      double c1 = oneway(S, 0), t1 = oneway(S, 1), d1 = oneway(S, 4), n1 = 0, w1 = 0, m1 = 0,
             u1 = 0, x1 = 0, s1 = 0, p1 = oneway(S, 10), h1 = 0;
#else
      // kind 10 is deliberately absent: it is refuted and slower, and running it here would cost a
      // pass over 16 GB per round for a number nothing can use.
      double c1 = oneway(S, 0), t1 = oneway(S, 1), d1 = oneway(S, 4), n1 = oneway(S, 2),
             w1 = oneway(S, 6), m1 = oneway(S, 7), u1 = oneway(S, 3), x1 = oneway(S, 8),
             s1 = oneway(S, 9), p1 = 0, h1 = oneway(S, 5);
#endif
      // BLKSWEEP="16,32,...,16384" walks the grid width of both transports inside one process, so
      // the block count each one needs to saturate is measured against the other under identical
      // conditions. Both ends must read it: oneway() has barriers in it, so the two sides have to
      // walk the same number of them. The two are not the same amount of hardware per block -- CU
      // runs WTH threads all moving data, TDM runs TWTH threads of which exactly one issues
      // descriptors and the rest return at once -- so the columns are per block, not per thread.
      // That asymmetry is the point of the table.
      if (const char* bs = getenv("BLKSWEEP")) {
        int sb = blocks, swb = wblocks;
        for (const char* p = bs; *p;) {
          int nb = atoi(p);
          if (nb > 0) {
            blocks = nb;
            wblocks = nb;
            double cv = oneway(S, 0), tv = oneway(S, 1), xv = oneway(S, 8);
            // TDMms is the same staged copy with MWSISS waves per block issuing instead of one.
            // At the full grid that was a loss, but the grid was already saturated there; the
            // thing this column answers is whether the real independent variable is issuing
            // waves rather than blocks, which is what a 64-block grid with 8 issuers implies.
            int sb2 = verify_kind(S, 9);
            double sv = oneway(S, 9);
            if (reporter)
              printf(
                  "[BLK] round=%d size=%s blocks=%d issuers/blk=%d CU=%.1f TDM=%.1f TDMc2=%.1f "
                  "TDMms=%.1f%s\n",
                  r, sz, nb, MWSISS, cv, tv, xv, sv, sb2 ? "(BAD)" : "");
          }
          while (*p && *p != ',') p++;
          if (*p == ',') p++;
        }
        blocks = sb;
        wblocks = swb;
      }
      // THSWEEP="32,64,...,1024" walks the block width of the TDM kernels at a fixed block count.
      // wTH is a launch parameter, not a macro, so every point is measured in one process and the
      // ~3 GB/s of cross-build drift cannot get into the comparison.
      // The two kernels should not respond the same way: the staged copy issues from threadIdx.x==0
      // only, so extra waves buy no issue and cost occupancy plus a wider barrier, while every wave
      // of TDMc2 stages bytes through the vector path. Each point is verified, not just timed --
      // TDMc2 needs at least CUISS waves or the tiles owned by the missing issuers are never
      // stored, which a bandwidth number alone would not show.
      if (const char* ts = getenv("THSWEEP")) {
        int sw = wTH;
        for (const char* p = ts; *p;) {
          int nt2 = atoi(p);
          if (nt2 > 0) {
            wTH = nt2;
            int tb = verify_kind(S, 1);
            double tv = oneway(S, 1);
            int xb = 0;
            double xv = 0;
            if (nt2 >= CUISS * 32) {
              xb = verify_kind(S, 8);
              xv = oneway(S, 8);
            }
            if (reporter)
              printf("[TH] round=%d size=%s wTH=%d waves=%d TDM=%.1f%s TDMc2=%.1f%s\n", r, sz, nt2,
                     nt2 / 32, tv, tb ? "(BAD)" : "", xv,
                     nt2 < CUISS * 32 ? "(skipped, needs CUISS waves)" : (xb ? "(BAD)" : ""));
          }
          while (*p && *p != ',') p++;
          if (*p == ',') p++;
        }
        wTH = sw;
      }
      // HSPLITS="30,35,40" walks the split inside one process. Both ends read it, so both walk the
      // same number of barriers.
      if (const char* hs = getenv("HSPLITS")) {
        int save = g_hsplit;
        for (const char* p = hs; *p;) {
          g_hsplit = atoi(p);
          double hv = oneway(S, 5);
          if (reporter) printf("[HSW] round=%d size=%s split=%d HYB=%.1f\n", r, sz, g_hsplit, hv);
          while (*p && *p != ',') p++;
          if (*p == ',') p++;
        }
        g_hsplit = save;
      }
      if (reporter) {
        printf("%9s %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n",
               sz, c1, t1, d1, n1, w1, m1, u1, x1, s1, p1, h1);
        // Named fields so consumers do not have to track column positions across variant changes.
        printf(
            "[KV] round=%d size=%s CU1=%.1f TDM1=%.1f TDMdb1=%.1f TDMnl1=%.1f TDMnw1=%.1f "
            "TDMmw1=%.1f TDMcu1=%.1f TDMc21=%.1f TDMms1=%.1f TDMdp1=%.1f HYB1=%.1f\n",
            r, sz, c1, t1, d1, n1, w1, m1, u1, x1, s1, p1, h1);
      }
#else
      double c1 = oneway(S, 0), c2 = bothway(S, 0);
      double t1 = oneway(S, 1), t2 = bothway(S, 1);
      double x1 = oneway(S, 8), x2 = bothway(S, 8);
      double h1 = oneway(S, 5), h2 = bothway(S, 5);
      if (reporter) {
        printf("%9s %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n", sz, c1, c2, t1, t2,
               x1, x2, h1, h2);
        printf(
            "[KV] round=%d size=%s CU1=%.1f TDM1=%.1f TDMc21=%.1f HYB1=%.1f"
            " CU2=%.1f TDM2=%.1f TDMc22=%.1f HYB2=%.1f\n",
            r, sz, c1, t1, x1, h1, c2, t2, x2, h2);
      }
#endif
    }
  }

  barrier(fd);
  close(fd);
  return 0;
}

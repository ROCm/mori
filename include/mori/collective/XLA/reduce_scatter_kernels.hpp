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

// ===========================================================================
// reduce_scatter_kernels.hpp
//
// Device/template kernels for the reduce-scatter example. Both modes share the
// same streaming load/store, reduction op functors, and the generic vectorized
// reduce core (ReduceVecGroup):
//
//   * ReduceScatterPushKernel  — fused SDMA "push" scatter + receiver-side
//                                completion signal + grid-strided reduce.
//   * ReduceScatterPullKernel  — direct P2P "pull": read each peer's shard over
//                                XGMI and reduce in one pass (no staging/SDMA).
//
// All host-only test code (fill/verify, threading, main) stays in the .cpp.
// ===========================================================================
#pragma once

#include <array>
#include <cstdint>

#include "mori/collective/XLA/collectives_common.hpp"  
#include "mori/core/transport/p2p/device_primitives.hpp"  // Bf16BitsToF32
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;
using namespace mori::collective;

// StartSdmaScatter and its push constants (kRSPush*, RS_ENABLE_FALLBACK) now live
// in collectives_common.hpp so both reduce-scatter and all-gather share them.

// Push Phase-3 reduce path: 1 = range-checked raw-buffer loads/stores (no scalar
// tail), 0 = original global b128 load/store + partial-group + scalar tail.
#ifndef RS_USE_BUFFER_REDUCE
#define RS_USE_BUFFER_REDUCE 0
#endif

// ---------------------------------------------------------------------------
// Reduction Op functors. The accumulator stays in the element type T (so bf16
// reduces in bf16-width registers -- half the VGPRs of a float accumulator and
// no spilling for the 8x8 register tile). Numerical accuracy of the add is
// handled inside the Op functor: SumOp<hip_bfloat16> promotes each add to float
// and rounds the result back to bf16 (per-add rounding, not a float running sum).
// ---------------------------------------------------------------------------
template < class T >
struct AccumulatorType {
  using type = T;
};

// Generic up/down cast (identity for float; specialize for fp16/bf16 if needed).
template <typename T>
__device__ __forceinline__ typename AccumulatorType<T>::type UpcastF(T v) {
  return static_cast<typename AccumulatorType<T>::type>(v);
}

template <typename T>
__device__ __forceinline__ T DowncastF(typename AccumulatorType<T>::type v) {
  return static_cast<T>(v);
}

template < typename T >
struct SumOp {
  using Type = T;
  __device__ T operator()(T a, T b) { return a + b; }
};
template < class T>
struct MaxOp {
  using Type = T;
  __device__ T operator()(T a, T b) { return std::max(a, b); }
};
template < class T >
struct MinOp {
  using Type = T;
  __device__ T operator()(T a, T b) { return std::min(a, b); }
};
template < class T >
struct ProdOp {
  using Type = T;
  __device__ T operator()(T a, T b) { return a * b; }
};

struct alignas(4) BF16Pack {
  hip_bfloat16 x, y;
};
static_assert(sizeof(BF16Pack) == 4 && alignof(BF16Pack) == 4);

template <>
struct SumOp<BF16Pack> {
  using Type = BF16Pack;
  __device__ BF16Pack operator()(BF16Pack a, BF16Pack b) {
    const uint32_t ua = __builtin_bit_cast(uint32_t, a);
    const uint32_t ub = __builtin_bit_cast(uint32_t, b);
    const float x = Bf16BitsToF32(static_cast<uint16_t>(ua)) +
                    Bf16BitsToF32(static_cast<uint16_t>(ub));
    const float y = Bf16BitsToF32(static_cast<uint16_t>(ua >> 16)) +
                    Bf16BitsToF32(static_cast<uint16_t>(ub >> 16));
    const auto r = static_cast<__hip_bfloat162_raw>(__float22bfloat162_rn(float2{x, y}));
    const auto packed = static_cast<uint32_t>(r.x) | (static_cast<uint32_t>(r.y) << 16);
    return __builtin_bit_cast(BF16Pack, packed);
  }
};

// Maps a storage element type to the type used to INSTANTIATE the reduce kernels.
// float reduces as float; bf16 reduces as a packed pair (BF16Pack) so vecSize stays
// at fp32 parity and the 8x8 accumulator tile does not spill. Data buffers stay
// physically ElemT; host code reinterpret_casts them to ComputeT and passes counts
// in packs (kPack = sizeof(ComputeT)/sizeof(ElemT)).
template <class T> struct ReduceComputeType { using type = T; };
template <> struct ReduceComputeType<hip_bfloat16> { using type = BF16Pack; };

// Reduce one group of NV vectors (each NV-member at vector index g + i*gstride)
// across all npes staging slots into the output. Callers guarantee every member
// index is in-bounds, so there are NO per-lane guards here. Used with NV=NumVecs
// for the full-group fast path and NV=1 for the single trailing partial group.
// Generic core: srcBase(pe) returns the base pointer of peer pe's contribution
// to THIS PE's shard. Peer 0 seeds the accumulators, peers 1..npes-1 reduce in.
template <int NV, class ReduceOp, StreamScope Scope,
          class T = typename ReduceOp::Type>
__device__ __forceinline__ void ReduceVecGroup(const T* __restrict__ input, 
                                               const T* __restrict__ staging, 
                                  T* __restrict__ output, int npes, int myPe,
                                  size_t v, size_t lstride, size_t chunkElems) {
  constexpr int vecSize = VecBytes / sizeof(T);
  using Vec = TVecType<VecBytes>;
  using AccType = typename AccumulatorType<T>::type;
  using Data = std::array<T, vecSize>;

  AccType acc[NV][vecSize];
  Vec vec[NV];
  const T* p[NV];
  // Seed accumulators from peer 0: load its NV vectors, then upcast lanes.
  p[0] = input + myPe * chunkElems;
#pragma unroll
  for (int i = 0; i < NV; i++) {
    const size_t idx = (v + i * lstride) * vecSize;
    vec[i] = StreamLoad<Scope>(p[0] + idx);
  }
#pragma unroll
  for (int i = 0; i < NV; i++) {
    Data lanes = __builtin_bit_cast(Data, vec[i]);
#pragma unroll
    for (int j = 0; j < vecSize; j++) acc[i][j] = UpcastF<T>(lanes[j]);
  }
  
  #pragma unroll
  for (int i = 0; i < NV; i++) {
    p[i] = staging + (v + i*lstride) * vecSize; // do manual hoisting
  }
  // Reduce peers 1..npes-1 in: load their NV vectors, then fold lanes into acc.
  for (int pe = 1; pe < npes; pe++) {
#pragma unroll
    for (int i = 0; i < NV; i++) {
      vec[i] = StreamLoad<Scope>(p[i]);
      p[i] += chunkElems;
    }
#pragma unroll
    for (int i = 0; i < NV; i++) {
      Data lanes = __builtin_bit_cast(Data, vec[i]);
#pragma unroll
      for (int j = 0; j < vecSize; j++) {
        acc[i][j] = ReduceOp()(acc[i][j], UpcastF<T>(lanes[j]));
      }
    }
  }
  // Downcast the accumulators back into NV vectors and store them.
#pragma unroll
  for (int i = 0; i < NV; i++) {
    const size_t idx = v + i * lstride;
    Data lanes;
#pragma unroll
    for (int j = 0; j < vecSize; j++) lanes[j] = DowncastF<T>(acc[i][j]);
    StreamStore<Scope>(output + idx * vecSize, __builtin_bit_cast(Vec, lanes));
  }
}

// Buffered variant of ReduceVecGroup for the push path: same NV-grouped reduce,
// but every access goes through a range-checked RAW buffer resource (V#) instead
// of a raw pointer. srcRsrc(pe) returns peer pe's slice descriptor and outR is the
// output slice descriptor; each descriptor's num_records is the valid slice byte
// extent, so the hardware per-component range check (CDNA4 ISA 9.1.5 note 4)
// handles the final partial vector -- OOB reads return 0 and OOB stores are
// dropped -- with NO per-lane guard and NO scalar tail, for ANY reduction op.
// The byte offset of vector index k is k * VecBytes (uint32; slice < 4 GiB).
// Cache policy / non-temporal hint rides RS_BUF_AUX inside BufferLoad/Store128.
template <int NV, class ReduceOp, class SrcRsrcFn, class T = typename ReduceOp::Type>
__device__ __forceinline__ void ReduceVecGroupBuffered(SrcRsrcFn srcRsrc, BufRsrc outR,
                                                       int npes, uint32_t g, uint32_t gstride) {
  static_assert(VecBytes == 16, "ReduceVecGroupBuffered uses b128 buffer ops");
  constexpr int vecSize = VecBytes / sizeof(T);
  using Vec = TVecType<VecBytes>;
  using AccType = typename AccumulatorType<T>::type;
  using Data = std::array<T, vecSize>;

  AccType acc[NV][vecSize];
  Vec vec[NV];

  // Seed accumulators from peer 0.
  BufRsrc r0 = srcRsrc(0);
#pragma unroll
  for (int i = 0; i < NV; i++) {
    uint32_t voff = static_cast<uint32_t>((g + i * gstride) * VecBytes);
    vec[i] = BufferLoad128(r0, voff);
  }
#pragma unroll
  for (int i = 0; i < NV; i++) {
    Data lanes = __builtin_bit_cast(Data, vec[i]);
#pragma unroll
    for (int j = 0; j < vecSize; j++) acc[i][j] = UpcastF<T>(lanes[j]);
  }
  for (int pe = 1; pe < npes; pe++) {
    BufRsrc rp = srcRsrc(pe);
#pragma unroll
    for (int i = 0; i < NV; i++) {
      uint32_t voff = static_cast<uint32_t>((g + i * gstride) * VecBytes);
      vec[i] = BufferLoad128(rp, voff);
    }
#pragma unroll
    for (int i = 0; i < NV; i++) {
      Data lanes = __builtin_bit_cast(Data, vec[i]);
#pragma unroll
      for (int j = 0; j < vecSize; j++) {
        acc[i][j] = ReduceOp()(acc[i][j], UpcastF<T>(lanes[j]));
      }
    }
  }
#pragma unroll
  for (int i = 0; i < NV; i++) {
    Data lanes;
#pragma unroll
    for (int j = 0; j < vecSize; j++) lanes[j] = DowncastF<T>(acc[i][j]);
    vec[i] = __builtin_bit_cast(Vec, lanes);
  }
#pragma unroll
  for (int i = 0; i < NV; i++) {
    uint32_t voff = static_cast<uint32_t>((g + i * gstride) * VecBytes);
    BufferStore128(outR, vec[i], voff);
  }
}

// ---------------------------------------------------------------------------
// Fused reduce-scatter kernel ("push", sliced into S = 1<<logS slices)
//
//   input         : raw symmetric-heap pointer, N = npes*chunkElems elements
//   staging       : raw symmetric-heap pointer, npes slots of chunkElems elements
//   output        : raw symmetric-heap pointer, chunkElems elements
//   groupCounters : plain device buffer (>= S uint32), local-only block counters
//
// Each shard is split into S slices. A sender issues S separate SDMA copies; copy
// s bumps the receiver's per-slice completion counter signalPtrs[s] via an SDMA
// ADD64 of 1 (one copy + one atomic per (sender,slice)). The receiver's grid is
// partitioned into S groups of G = gridDim.x/S blocks; group g spins on slice g's
// counter until it reaches npes-1 senders and reduces only slice g, so a group
// starts as soon as its slice lands (pipelining). The last block of each group
// zeroes that slice's counter + its local counter (groupCounters[g]) for the
// next launch -- no monotonic generation counter needed.
//
// Phase 1 (npes <= 8, i.e. npes*S <= warpSize) uses ONE warp to issue every
// (peer, slice) copy at once: lane `lane` serves peer = lane/S, slice = lane%S.
// Each peer's slice-0 lane reserves the S contiguous packets on that peer's queue
// and submits once (single doorbell); all lanes write their packet in parallel.
// Peers map to distinct queues (peer%8 is distinct for npes<=8), so the per-warp
// multi-queue submit has exactly one sole producer per queue and cannot deadlock,
// and all npes doorbells fire concurrently (no warp-strided waves). For larger
// npes (npes*S > warpSize) it falls back to the warp-strided SdmaPutWarpFusedS
// path (one warp per peer, lane 0 reserves/submits), which is also deadlock-free.
//
// input/staging/output MUST live in the symmetric static heap (ShmemMalloc) so
// the address-based SDMA put can translate local->peer (offset from heapBaseAddr);
// groupCounters is a plain hipMalloc'd buffer (never peer-written).
template <int NumVecs, class ReduceOp, class T = typename ReduceOp::Type>
__device__ __forceinline__ void ReduceScatterPushBody(
    int myPe, int npes, int logS, const T* __restrict__ input,
    T* __restrict__ staging, T* __restrict__ shardOut, size_t chunkElems) {
  if (blockIdx.x == 0) {
    // reduce-scatter: per-peer source slice (stride=chunkElems), dst=staging,
    // no self-copy (self is folded in by the Phase-3 reduce reading local input).
    // Staging is packed densely (no self hole): slot = myPe<peer?myPe:myPe-1, a
    // bijection over the npes-1 non-self peers.
    const size_t heapBase = GetGlobalGpuStatesPtr()->heapBaseAddr;
    const size_t dstBaseOff = reinterpret_cast<uintptr_t>(staging) - heapBase;
    const size_t chunkBytes = chunkElems * sizeof(T);
    StartSdmaScatter(
        myPe, npes, logS, chunkElems, sizeof(T),
        [=](int peer) { return peer != myPe; },
        [=](int peer) -> const uint8_t* {
          return reinterpret_cast<const uint8_t*>(input + peer * chunkElems);
        },
        [=](int peer) -> size_t {
          const int slot = (myPe < peer ? myPe : myPe - 1);
          return dstBaseOff + slot * chunkBytes;
        });
  }
  const uint32_t BlockDimX = blockDim.x;

  // Grouping: G blocks per slice; group g (= slice g) handles only its slice.
  const uint32_t G = gridDim.x >> logS,    // host guarantees gridDim.x % S == 0
                 g = blockIdx.x / G,       // slice / group index
                 lb = blockIdx.x - g * G;  // local block within the group

  // === Phase 2: wait until slice g's counter reaches all npes-1 senders ========
  // Each slice's counter lives in my local HBM (bumped by remote SDMA ADD64s).
  // One thread per block polls signalPtrs[g]; every block does its own
  // wait+acquire (read-only -> L2 hits). At npes==1 want==0, so no spin.
  if (threadIdx.x == 0) {
    auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
    // Self-copy is skipped, so exactly npes-1 senders bump this slice's counter.
    const uint32_t want = static_cast<uint32_t>(npes - 1);
    // The scatter atomic is a 32-bit ADD into the low dword, so poll 32 bits.
    auto *addr = Tglobal(reinterpret_cast<uint32_t*>(&heapObj->signalPtrs[g]));
    while (__hip_atomic_load(addr, __ATOMIC_RELAXED,
       __HIP_MEMORY_SCOPE_AGENT) < want) {
    }
  }
  __syncthreads();
  // System-scope ACQUIRE (not the full seq_cst __threadfence_system): the staging
  // was written by peers' SDMA over XGMI, so we must invalidate against another
  // device's writes -- scope "" (system) is required -- but only the acquire half
  // is needed here (we publish nothing at this point), so drop the release/waitcnt.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  constexpr int vecSize = VecBytes / sizeof(T);
  // vecSize-aligned slice length; the last slice absorbs the remainder.
  const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize;

  // === Phase 3: grid-strided vectorized reduce over slice g ===================
  // Streaming reduction over slice g only, strided over the group's G blocks. Uses
  // the nontemporal load<16>/store<16> primitives (single-use data, bypass L2).
  const size_t sOfs = FORCE_SGPR(g * sliceLen), 
         sCnt = FORCE_SGPR((g == (1 << logS) - 1) ? (chunkElems - sOfs) : sliceLen);
  uint32_t lstride = FORCE_SGPR(G * BlockDimX);
  input += sOfs;
  staging += sOfs;
  shardOut += sOfs;

#if RS_USE_BUFFER_REDUCE
  // Range-checked raw-buffer path. Each descriptor's num_records is the valid
  // slice byte extent (sCnt elements), so the hardware per-component range check
  // covers the final partial vector: OOB reads return 0 and OOB stores are
  // dropped. No partial-group loop, no scalar tail -- correct for any op. We only
  // iterate to the vector ceiling; groups whose members run past the end are
  // no-ops (bounded by NumVecs-1 extra chunks per thread).
  const uint32_t sliceBytes32 = static_cast<uint32_t>(sCnt * sizeof(T));
  auto srcRsrc = [input, staging, chunkElems, myPe, sliceBytes32](int pe) -> BufRsrc {
    const T *ptr = pe == 0 ? input + myPe * chunkElems : 
                             staging + (pe - 1) * chunkElems;
    return MakeRawRsrc(ptr, sliceBytes32);
  };
  const BufRsrc outR = MakeRawRsrc(shardOut, sliceBytes32);
  // Index in 32-bit: within a slice the vector index is bounded by the same
  // < 4 GiB extent the buffer descriptor (num_records) already assumes, and the
  // grid dims are small -- so 32-bit avoids 64-bit address math and VGPRs.
  const uint32_t totalVecs = static_cast<uint32_t>(sCnt / vecSize);
  const uint32_t totalVecsCeil = static_cast<uint32_t>((sCnt + vecSize - 1) / vecSize);
  // Full NumVecs groups only while every member is in-bounds (no wasted OOB
  // chunks), then NV=1 buffered groups to the ceiling so only the genuine final
  // sub-vector element(s) hit the hardware range check.
  uint32_t v = lb * BlockDimX + threadIdx.x;
  for (; v + (NumVecs - 1) * lstride < totalVecs; v += lstride * NumVecs) {
    ReduceVecGroupBuffered<NumVecs, ReduceOp>(srcRsrc, outR, npes, v, lstride);
  }
  for (; v < totalVecsCeil; v += lstride) {
    ReduceVecGroupBuffered<1, ReduceOp>(srcRsrc, outR, npes, v, lstride);
  }
#else
  // Push path: every source (input + staging) is LOCAL HBM and the Phase-2
  // __threadfence_system() already made the remote-DMA'd staging visible, so the
  // per-access loads/stores use agent scope (scope=EAgentScope).
  const size_t totalVecs = sCnt / vecSize;
  // Grid-strided thread id / stride. The grid is capped by CU count
  size_t v = lb * BlockDimX + threadIdx.x;
  for (; v + (NumVecs - 1) * lstride < totalVecs; v += lstride * NumVecs) {
    ReduceVecGroup<NumVecs, ReduceOp, EAgentScope>(input, staging, shardOut, 
                                        npes, myPe, v, lstride, chunkElems);
  }
  // Trailing partial group: the remaining in-bounds vectors for this thread.
  for (; v < totalVecs; v += lstride) {
    ReduceVecGroup<1, ReduceOp, EAgentScope>(input, staging, shardOut, 
                                        npes, myPe, v, lstride, chunkElems);
  }
  // Scalar tail (only the last slice can be non-vecSize-aligned).
  v = lb * BlockDimX + threadIdx.x + totalVecs * vecSize;
  for (; v < sCnt; v += lstride) {
    using Vec = TVecType<sizeof(T)>;
    using AccType = typename AccumulatorType<T>::type;
    const T *ptr = input + myPe * chunkElems;
    Vec V = StreamLoad<EAgentScope, sizeof(T)>(ptr + v);
    AccType a = UpcastF<T>(__builtin_bit_cast(T, V));
    ptr = staging + v;
    for (int pe = 1; pe < npes; pe++, ptr += chunkElems) {
      V = StreamLoad<EAgentScope, sizeof(T)>(ptr);
      a = ReduceOp()(a, UpcastF<T>(__builtin_bit_cast(T, V)));
    }
    V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    StreamStore<EAgentScope, sizeof(T)>(shardOut + v, V);
  }
#endif // RS_USE_BUFFER_REDUCE
}

template <int NumVecs, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void __launch_bounds__(256, 1)
ReduceScatterPushKernel(int myPe, int npes, int logS, const T* __restrict__ input,
                                    T* __restrict__ staging, T* __restrict__ output,
                                    uint32_t* __restrict__ groupCounters, size_t chunkElems) {
  // Phase 1-3 (scatter + completion wait + acquire + grid-strided reduce) into
  // this PE's shard-sized output buffer.
  ReduceScatterPushBody<NumVecs, ReduceOp, T>(myPe, npes, logS, input, staging, output,
                                              chunkElems);

  // === Reset: the last block of group g zeroes slice g's counter + counter ======
  // A block only reaches here after passing Phase 2 (it has observed the full count
  // for slice g), so when the group counter hits G every block in the group is past
  // Phase 2 and clearing the counter cannot drop an unseen signal. Cross-iteration
  // ordering is guaranteed by the host hipStreamSynchronize + ShmemBarrierAll
  // between launches. Counters are local-only (plain hipMalloc'd buffer).
  if (threadIdx.x == 0) {
    const uint32_t G = gridDim.x >> logS, g = blockIdx.x / G;
    uint32_t z = atomicAdd(&groupCounters[g], 1u);
    if (z + 1 == G) {
      auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
      heapObj->signalPtrs[g] = 0;  // clear slice g's completion counter
      groupCounters[g] = 0;        // clear group g's local counter
      __threadfence_system(); // keep it for next launch's peer SDMA ??
    }
  }
}

// Load M output positions x NPES peers ALL up front into distinct registers, then
// reduce each position. Unlike ReduceVecGroup (which reuses one buffer per peer and
// so serializes the remote loads peer-by-peer via a WAR dependency), every load
// here is independent: with all indices compile-time the regs[][] array stays in
// VGPRs and the M*NPES loads issue back-to-back, so the long remote-load latency is
// paid once and the per-position reductions overlap with still-in-flight loads.
// Callers guarantee every member index (g + m*gstride) is in-bounds.
template <int M, int NPES, class ReduceOp, StreamScope Scope, class SrcBaseFn, 
          class T = typename ReduceOp::Type>
__device__ __forceinline__ void ReduceAllPeersGroup(SrcBaseFn srcBase, T* __restrict__ output,
                                                    size_t g, size_t gstride) {
  constexpr int vecSize = VecBytes / sizeof(T);
  using Vec = TVecType<VecBytes>;
  using AccType = typename AccumulatorType<T>::type;
  using Data = std::array<T, vecSize>;
  Vec regs[M][NPES];
#pragma unroll
  for (int m = 0; m < M; m++) {
    size_t idx = g + static_cast<size_t>(m) * gstride;
#pragma unroll
    for (int pe = 0; pe < NPES; pe++)
      regs[m][pe] = StreamLoad<Scope>(srcBase(pe) + idx * vecSize);
  }
#pragma unroll
  for (int m = 0; m < M; m++) {
    AccType acc[vecSize];
    Data l0 = __builtin_bit_cast(Data, regs[m][0]);
#pragma unroll
    for (int j = 0; j < vecSize; j++) acc[j] = UpcastF<T>(l0[j]);
#pragma unroll
    for (int pe = 1; pe < NPES; pe++) {
      Data l = __builtin_bit_cast(Data, regs[m][pe]);
#pragma unroll
      for (int j = 0; j < vecSize; j++) {
        acc[j] = ReduceOp()(acc[j], UpcastF<T>(l[j]));
      }
    }
    Data o;
#pragma unroll
    for (int j = 0; j < vecSize; j++) o[j] = DowncastF<T>(acc[j]);
    StreamStore<Scope>(output + (g + static_cast<size_t>(m) * gstride) * vecSize,
                       __builtin_bit_cast(Vec, o));
  }
}

// ---------------------------------------------------------------------------
// Direct "pull" reduce-scatter kernel (no staging, no SDMA scatter).
//
// Each PE reads its shard directly from every peer's input buffer over the P2P
// fabric (XGMI) and reduces in one pass:
//
//   output[j] = REDUCE_p( input_p[ myPe*chunkElems + j ] )
//
// input_p's base is srcObj->peerPtrs[p] (the symmetric peer pointer); this PE's
// shard within each peer starts at peerPtrs[p] + myPe*chunkElems. There is no
// staging buffer and no cross-block flag handoff, so every block is independent
// (no co-residency cap) and the kernel is a single fused grid-strided reduce.
//
// The fast path uses ReduceAllPeersGroup: each group reduces M = NumVecs/NPES
// output positions and issues all M*NPES remote loads up front, for maximum
// memory-level parallelism across peers (the long XGMI read latency is paid once
// per group instead of once per peer). NPES is a compile-time template arg so the
// per-position/per-peer register tile stays in VGPRs; the host dispatches on the
// real npes.
//
// Correctness requires all PEs to have produced their input before launch; the
// host issues a ShmemBarrierAll() before timing.
// ---------------------------------------------------------------------------
template <int NumVecs, int NPES, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void ReduceScatterPullKernel(int myPe,
                                        mori::application::SymmMemObjPtr srcObj,
                                        T* __restrict__ output, size_t chunkElems) {
  constexpr int vecSize = VecBytes / sizeof(T);
  constexpr int M = NumVecs / NPES;  // output positions per group (M >= 1 for NPES <= NumVecs)
  const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t gstride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t totalVecs = chunkElems / vecSize;
  using AccType = typename AccumulatorType<T>::type;
  
  // Peer pe's contribution to my shard: base of peer pe's input + myPe*chunkElems.
  const size_t myOfs = static_cast<size_t>(myPe) * chunkElems;
  auto srcBase = [srcObj, myOfs](int pe) -> const T* {
    return reinterpret_cast<const T*>(srcObj->peerPtrs[pe]) + myOfs;
  };

  // Fast path: M positions per group, all M*NPES loads issued up front.
  size_t g = gtid;
  for (; g + static_cast<size_t>(M - 1) * gstride < totalVecs; g += gstride * M) {
    ReduceAllPeersGroup<M, NPES, ReduceOp, ESystemScope>(srcBase, output, g, gstride);
  }
  // Trailing in-bounds vectors for this thread (fewer than M left). Reuse the
  // all-peers primitive with M=1 (one position, NPES loads up front) so the pull
  // kernel stays on a single reduction path and needs no runtime-npes helper.
  for (size_t idx = g; idx < totalVecs; idx += gstride) {
    ReduceAllPeersGroup<1, NPES, ReduceOp, ESystemScope>(srcBase, output, idx, gstride);
  }

  // Scalar tail for elements not covered by the vectorized loop.
  for (size_t i = totalVecs * vecSize + gtid; i < chunkElems; i += gstride) {
    using Vec = TVecType<sizeof(T)>;
    AccType a = UpcastF<T>(__builtin_bit_cast(T, 
                  StreamLoad<ESystemScope, sizeof(T)>(srcBase(0) + i)));
    for (int pe = 1; pe < NPES; pe++) {
      auto V = StreamLoad<ESystemScope, sizeof(T)>(srcBase(pe) + i);
      a = ReduceOp()(a, UpcastF<T>(__builtin_bit_cast(T, V)));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    StreamStore<ESystemScope, sizeof(T)>(output + i, V);
  }
}

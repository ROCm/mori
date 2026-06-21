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

#include "mori/core/transport/p2p/device_primitives.hpp"  // load<N>/store<N>
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

#define GLOBAL_SPACE __attribute__((address_space(1)))

// ---------------------------------------------------------------------------
// Streaming (cache-bypassing) 16-byte load/store, RCCL-style (see rccl op128.h).
#if (defined(__gfx942__) || defined(__gfx950__)) && \
    __has_builtin(__builtin_amdgcn_global_load_b128) &&  \
    __has_builtin(__builtin_amdgcn_global_store_b128)
#define RS_HAVE_GLOBAL_B128 1
#else
#define RS_HAVE_GLOBAL_B128 0
#endif

template <int Bytes>
using TVecType = typename mori::core::VecTypeSelector<Bytes>::dataType;

/*
using index_t = int32_t;
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
__device__ void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
*/

using rs_v4u = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;
using rs_v4u_gptr = GLOBAL_SPACE rs_v4u*;


template <int Bytes>
__device__ __forceinline__ TVecType<Bytes> StreamLoad(
    const void* p) {
  return mori::core::load<Bytes>(p);  // generic fallback (8/4/2/1 byte tails)
}
template <int Bytes>
__device__ __forceinline__ void StreamStore(
    void* p, TVecType<Bytes> v) {
  mori::core::store<Bytes>(p, v);
}

template <>
__device__ __forceinline__ TVecType<16> StreamLoad<16>(
    const void* p) {
#if RS_HAVE_GLOBAL_B128
  rs_v4u raw = __builtin_amdgcn_global_load_b128((rs_v4u_gptr)p, "");
  return __builtin_bit_cast(TVecType<16>, raw);
#else
  return mori::core::load<16>(p);
#endif
}
template <>
__device__ __forceinline__ void StreamStore<16>(void* p, TVecType<16> v) {
#if RS_HAVE_GLOBAL_B128
  rs_v4u raw = __builtin_bit_cast(rs_v4u, v);
  __builtin_amdgcn_global_store_b128((rs_v4u_gptr)p, raw, "");
#else
  mori::core::store<16>(p, v);
#endif
}

// ---------------------------------------------------------------------------
// Reduction Op functors. Accumulation is done in float (lossless for the
// small integer-valued test patterns, and avoids precision loss for fp16/bf16).
// ---------------------------------------------------------------------------
template < class T >
struct AccumulatorType {
  using type = T;
};

template <>
struct AccumulatorType<hip_bfloat16> {
  using type = float;
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
  __device__ T operator()(T a, T b) { return a + b; }
};
template < class T>
struct MaxOp {
  __device__ T operator()(T a, T b) { return std::max(a, b); }
};
template < class T >
struct MinOp {
  __device__ T operator()(T a, T b) { return std::min(a, b); }
};
template < class T >
struct ProdOp {
  __device__ T operator()(T a, T b) { return a * b; }
};

template < int VecBytes, int NumVecs, class T >
struct PackedVec {
  static constexpr int VecSize = VecBytes / sizeof(T);
  static constexpr int TotalElems = NumVecs * VecSize;
  using Vec = TVecType<VecBytes>;
  using AccType = typename AccumulatorType<T>::type;
  using Data = std::array<T, VecSize>;
  std::array<Vec, NumVecs> vec;

  __device__ void upcastTo(AccType (&acc)[TotalElems]) {
#pragma unroll
      for (int i = 0; i < NumVecs; i++) {
        Data lanes = __builtin_bit_cast(Data, vec[i]);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
          acc[i * VecSize + j] = UpcastF<T>(lanes[j]);
        }
      }
  }
  __device__ void downcastFrom(AccType (&acc)[TotalElems]) {
#pragma unroll
      for (int i = 0; i < NumVecs; i++) {
        Data lanes;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
          lanes[j] = DowncastF<T>(acc[i * VecSize + j]);
        }
        vec[i] = __builtin_bit_cast(Vec, lanes);
      }
  }
  template < template < class > class OpT >
  __device__ void reduce(AccType (&acc)[TotalElems], OpT<T> reduceOp) {
#pragma unroll
    for (int i = 0; i < NumVecs; i++) {
      Data lanes = __builtin_bit_cast(Data, vec[i]);
#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        acc[i * VecSize + j] = reduceOp(acc[i * VecSize + j], UpcastF<T>(lanes[j]));
      }
    }
  } 
};


// Reduce one group of NV vectors (each NV-member at vector index g + i*gstride)
// across all npes staging slots into the output. Callers guarantee every member
// index is in-bounds, so there are NO per-lane guards here. Used with NV=NumVecs
// for the full-group fast path and NV=1 for the single trailing partial group.
// Generic core: srcBase(pe) returns the base pointer of peer pe's contribution
// to THIS PE's shard. Peer 0 seeds the accumulators, peers 1..npes-1 reduce in.
// This decouples the data layout from the reduction so the same code serves the
// staging "push" path, the "pull" path, and the per-hop 2-source ring reduction.
template <int VecBytes, int NV, class T, template <class> class OpT, class SrcBaseFn>
__device__ __forceinline__ void ReduceVecGroup(SrcBaseFn srcBase, T* __restrict__ output,
                                                  int npes, size_t g, size_t gstride) {
  constexpr int vecSize = VecBytes / sizeof(T);
  using AccType = typename AccumulatorType<T>::type;
  AccType acc[NV * vecSize];
  PackedVec<VecBytes, NV, T> packed;
  const T* b0 = srcBase(0);
#pragma unroll
  for (int i = 0; i < NV; i++) {
    size_t idx = g + static_cast<size_t>(i) * gstride;
    packed.vec[i] = StreamLoad<VecBytes>(b0 + idx * vecSize);
  }
  packed.upcastTo(acc);
  for (int pe = 1; pe < npes; pe++) {
    const T* bp = srcBase(pe);
#pragma unroll
    for (int i = 0; i < NV; i++) {
      size_t idx = g + static_cast<size_t>(i) * gstride;
      packed.vec[i] = StreamLoad<VecBytes>(bp + idx * vecSize);
    }
    packed.reduce(acc, OpT<T>());
  }
  packed.downcastFrom(acc);
#pragma unroll
  for (int i = 0; i < NV; i++) {
    size_t idx = g + static_cast<size_t>(i) * gstride;
    StreamStore<VecBytes>(output + idx * vecSize, packed.vec[i]);
  }
}

// Load M output positions x NPES peers ALL up front into distinct registers, then
// reduce each position. Unlike ReduceVecGroup (which reuses one buffer per peer and
// so serializes the remote loads peer-by-peer via a WAR dependency), every load
// here is independent: with all indices compile-time the regs[][] array stays in
// VGPRs and the M*NPES loads issue back-to-back, so the long remote-load latency is
// paid once and the per-position reductions overlap with still-in-flight loads.
// Callers guarantee every member index (g + m*gstride) is in-bounds.
template <int VecBytes, int M, int NPES, class T,
          template <class> class OpT, class SrcBaseFn>
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
      regs[m][pe] = StreamLoad<VecBytes>(srcBase(pe) + idx * vecSize);
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
      for (int j = 0; j < vecSize; j++) acc[j] = OpT<T>()(acc[j], UpcastF<T>(l[j]));
    }
    Data o;
#pragma unroll
    for (int j = 0; j < vecSize; j++) o[j] = DowncastF<T>(acc[j]);
    StreamStore<VecBytes>(output + (g + static_cast<size_t>(m) * gstride) * vecSize,
                          __builtin_bit_cast(Vec, o));
  }
}

// ---------------------------------------------------------------------------
// Fused reduce-scatter kernel
//
//   input    : raw symmetric-heap pointer, N = npes*chunkElems elements
//   staging  : raw symmetric-heap pointer, npes slots of chunkElems elements
//   output   : raw symmetric-heap pointer, chunkElems elements
//   gen      : monotonic launch generation for the receive-side signal wait
//
// All three buffers MUST live in the symmetric static heap (ShmemMalloc) so the
// address-based SDMA put can translate local->peer (offset from heapBaseAddr).
// ---------------------------------------------------------------------------
template <int VecBytes, int NumVecs, class T, template <class> class OpT>
__global__ void ReduceScatterPushKernel(int myPe, int npes, int S, const T* __restrict__ input,
                                    T* __restrict__ staging, T* __restrict__ output,
                                    size_t chunkElems, uint64_t gen) {
  auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);

  // The shard is split into S slices (vecSize-aligned; the last slice absorbs the
  // remainder). Each slice is sent as one SDMA sub-chunk on a SINGLE queue, in
  // order, so the per-sender completion counter encodes pipeline progress and
  // group g in Phase 3 reduces exactly the slice-g region.
  constexpr int vecSize = VecBytes / sizeof(T);
  const size_t sliceLen = (chunkElems / static_cast<size_t>(S) / vecSize) * vecSize;

  // === Phase 1: SDMA scatter (block 0 only), ONE thread per destination peer ===
  // Each destPe thread issues all S sub-chunk copies for that peer sequentially on
  // queue 0, each followed by an atomic increment of the SAME per-sender signal
  // slot on the receiver. Because a SINGLE thread issues them in loop order and
  // submitPacket commits in order, the receiver's counter increments 1,2,...,S as
  // slices 0,1,...,S-1 land -- so a counter value of v means slices [0, v) are
  // done. Splitting a peer's S transfers across threads would interleave their
  // commits and break this mapping. Fire-and-forget: the copy + completion atomic
  // ride the same queue and the atomic targets the *receiver's* signalPtrs, so
  // completion is observed on the receive side (Phase 2) -- no local quiet, no
  // cross-PE barrier, no block-0 -> all-blocks handoff.
  int tid = threadIdx.x;
  if (blockIdx.x == 0 && tid < npes && tid != myPe) {
    int destPe = tid;
    // Local symmetric address of peer destPe's staging slot for me (slot myPe).
    T* dstLocal = staging + myPe * chunkElems;
    const T* src = input + destPe * chunkElems;

    auto** handles = heapObj->deviceHandles_d + (destPe % 8) * numSdmaQ;
    // SdmaPutThreadFused still does expectedSignals[0]++ on this local array; that
    // is now dead bookkeeping (nobody quiets locally) but harmless.
    HSAuint64* expectedSignals = heapObj->expectSignalsPtr + (destPe % 8) * numSdmaQ;
    // Completion atomic -> destPe's single per-sender slot (queue 0), keyed by ME:
    //   destPe.signalPtrs[(myPe % 8) * numSdmaQ]
    HSAuint64* remoteSig = heapObj->peerSignalPtrs[destPe] + (myPe % 8) * numSdmaQ;

    for (int s = 0; s < S; s++) {
      size_t sOfs = static_cast<size_t>(s) * sliceLen;
      size_t sElems = (s == S - 1) ? (chunkElems - sOfs) : sliceLen;

      uintptr_t destAddr = reinterpret_cast<uintptr_t>(dstLocal + sOfs);
      size_t offset = destAddr - GetGlobalGpuStatesPtr()->heapBaseAddr;
      uint8_t* srcPtr =
          const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src + sOfs));
      uint8_t* dstPtr = reinterpret_cast<uint8_t*>(heapObj->peerPtrs[destPe] + offset);
      // qId = 0: single queue. signals base = remoteSig so signals[qId] == remoteSig,
      // i.e. every sub-chunk's atomic increments the same per-sender counter.
      mori::core::SdmaPutThreadFused(srcPtr, dstPtr, sElems * sizeof(T), handles, remoteSig,
                                expectedSignals, numSdmaQ, /*qId=*/0);
    }
  }

  // Grouping: partition the N blocks into S groups of G = N/S blocks; group g owns
  // slice g (the region [g*sliceLen, ...)) and works INDEPENDENTLY of the other
  // groups -- it waits only until slice g has landed from every sender and reduces
  // only slice g. S==1 => one group of all blocks reducing the whole chunk.
  const int N = static_cast<int>(gridDim.x);
  const int G = N / S;              // blocks per group (host guarantees N % S == 0)
  const int bidx = static_cast<int>(blockIdx.x);
  const int g = bidx / G;  // this block's group/slice
  const int lb = bidx - g * G; // local block index within group

  // === Phase 2: wait until THIS slice has landed from all npes senders ========
  // The per-sender counters live in my local HBM (written by remote SDMA), are
  // monotonic and never reset, and advance by exactly S per launch. Slice g
  // (0-indexed) is ready once a sender's counter reaches (gen-1)*S + g + 1. We use
  // a `>=` wait because a later slice's atomic may have already advanced the
  // counter past this group's exact threshold. One thread per sender polls its
  // per-sender slot signalPtrs[(p%8)*numSdmaQ]; every block in the group does its
  // own wait+acquire (read-only -> L2 hits).
  if (tid < npes && tid != myPe) {
    uint64_t want = (gen - 1) * static_cast<uint64_t>(S) + static_cast<uint64_t>(g) + 1;
    anvil::waitForSignalAtLeast(&heapObj->signalPtrs[(tid % 8) * numSdmaQ], want);
  }
  __syncthreads();
  __threadfence_system();  // acquire: staging visible before Phase 3 reads it

  // === Phase 3: grid-strided vectorized reduce over slice g (this group) ======
  // Streaming reduction: every staging slot is read exactly once and the output
  // written once, so use the nontemporal load<16>/store<16> primitives from
  // device_primitives.hpp. They move 16-byte vectors with a streaming (LRU
  // bypass) hint, avoiding L2 pollution from single-use data.
  const size_t sOfs = static_cast<size_t>(g) * sliceLen;            // slice base (elems)
  const size_t sCnt = (g == S - 1) ? (chunkElems - sOfs) : sliceLen;  // slice length

  // Stride over the GROUP only (G blocks), so the G blocks cooperatively reduce
  // just this slice.
  const size_t ltid = static_cast<size_t>(lb) * blockDim.x + threadIdx.x;
  const size_t lstride = static_cast<size_t>(G) * blockDim.x;
  const size_t totalVecs = sCnt / vecSize;  // full VecBytes-wide vectors in this slice
  using AccType = typename AccumulatorType<T>::type;

  // Same guard-free fast path + single trailing partial group as before, but
  // scoped to this slice (offset sOfs) and strided over the group.
  auto srcBase = [staging, input, chunkElems, sOfs, myPe](int pe) -> const T* {
    return (pe == myPe ? input : staging) + pe * chunkElems + sOfs;
  };
  T* out = output + sOfs;

  size_t v = ltid;
  for (; v + static_cast<size_t>(NumVecs - 1) * lstride < totalVecs; v += lstride * NumVecs) {
    ReduceVecGroup<VecBytes, NumVecs, T, OpT>(srcBase, out, npes, v, lstride);
  }
  // Trailing partial group: the remaining in-bounds vectors for this thread.
  for (size_t idx = v; idx < totalVecs; idx += lstride) {
    ReduceVecGroup<VecBytes, 1, T, OpT>(srcBase, out, npes, idx, lstride);
  }

  // Scalar tail (only the last slice can be non-vecSize-aligned).
  for (size_t i = totalVecs * vecSize + ltid; i < sCnt; i += lstride) {
    using Vec = TVecType<sizeof(T)>;
    const T* base = srcBase(0) + i;
    AccType a = UpcastF<T>(load<sizeof(T)>(base));
    base += chunkElems;
    for (int pe = 1; pe < npes; pe++, base += chunkElems) {
      auto V = load<sizeof(T)>(base);
      a = OpT<T>()(a, UpcastF<T>(V));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    store<sizeof(T)>(out + i, V);
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
template <int VecBytes, int NumVecs, int NPES, class T, template <class> class OpT>
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
    ReduceAllPeersGroup<VecBytes, M, NPES, T, OpT>(srcBase, output, g, gstride);
  }
  // Trailing in-bounds vectors for this thread (fewer than M left).
  for (size_t idx = g; idx < totalVecs; idx += gstride) {
    ReduceVecGroup<VecBytes, 1, T, OpT>(srcBase, output, NPES, idx, gstride);
  }

  // Scalar tail for elements not covered by the vectorized loop.
  for (size_t i = totalVecs * vecSize + gtid; i < chunkElems; i += gstride) {
    using Vec = TVecType<sizeof(T)>;
    AccType a = UpcastF<T>(load<sizeof(T)>(srcBase(0) + i));
    for (int pe = 1; pe < NPES; pe++) {
      auto V = load<sizeof(T)>(srcBase(pe) + i);
      a = OpT<T>()(a, UpcastF<T>(V));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    store<sizeof(T)>(output + i, V);
  }
}

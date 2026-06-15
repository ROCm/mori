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
// Device/template kernels for the reduce-scatter example. Three modes share the
// same streaming load/store, reduction op functors, and the generic vectorized
// reduce core (ReduceVecGroup):
//
//   * ReduceScatterKernel      — fused SDMA "push" scatter + receiver-side
//                                completion signal + grid-strided reduce.
//   * ReduceScatterPullKernel  — direct P2P "pull": read each peer's shard over
//                                XGMI and reduce in one pass (no staging/SDMA).
//   * ReduceScatterRingKernel  — proof-of-concept SDMA *ring* reduce-scatter:
//                                block 0 walks the ring prev->me->next once per
//                                chunk, slice-pipelined, with per-hop SDMA copy +
//                                receiver-side data-ready signal and a fused
//                                ReduceVecGroup reduction.
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

template<typename T>
__device__ __host__ inline static  T* Xglobal(T* ptr) { 
  return (T*)(T GLOBAL_SPACE *)reinterpret_cast<uintptr_t>(ptr); 
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
__global__ void ReduceScatterKernel(int myPe, int npes, int numQ, const T* __restrict__ input,
                                    T* __restrict__ staging, T* __restrict__ output,
                                    size_t chunkElems, uint64_t gen) {
  auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);

  // === Phase 1: SDMA scatter (block 0 only), one thread per (destPe, queue) ===
  // Fire-and-forget: the copy packet and its completion atomic ride the SAME
  // queue, and the atomic targets the *receiver's* signalPtrs, so it fires only
  // after the data has landed in the peer's staging. There is NO local quiet here
  // -- completion is observed on the receive side (Phase 2), which also lets us
  // drop the cross-PE barrier and the block-0 -> all-blocks flag handoff.
  int tid = threadIdx.x;
  if (blockIdx.x == 0 && tid < npes * numQ) {
    int destPe = tid / numQ, q = tid - destPe * numQ;
    // Local symmetric address of peer destPe's staging slot for me (slot myPe).
    T* dstLocal = staging + static_cast<size_t>(myPe) * chunkElems;
    const T* src = input + static_cast<size_t>(destPe) * chunkElems;

    // This thread's slice of the chunk (queue q of numQ).
    size_t qChunk = chunkElems / numQ;
    size_t qOfs = static_cast<size_t>(q) * qChunk;
    size_t qElems = (q == numQ - 1) ? (chunkElems - qOfs) : qChunk;

    uintptr_t destAddr = reinterpret_cast<uintptr_t>(dstLocal + qOfs);
    size_t offset = destAddr - GetGlobalGpuStatesPtr()->heapBaseAddr;
    uint8_t* srcPtr = Xglobal(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src + qOfs)));
    uint8_t* dstPtr = Xglobal(reinterpret_cast<uint8_t*>(heapObj->peerPtrs[destPe] + offset));

    auto** handles = Xglobal(heapObj->deviceHandles_d + (destPe % 8) * numSdmaQ);
    // SdmaPutThread still does expectedSignals[q]++ on this local array; that is
    // now dead bookkeeping (nobody quiets locally) but harmless.
    HSAuint64* expectedSignals = Xglobal(heapObj->expectSignalsPtr + (destPe % 8) * numSdmaQ);
    // Completion atomic -> destPe's signalPtrs at the slot keyed by ME (sender):
    //   destPe.signalPtrs[(myPe % 8) * numSdmaQ + q]
    HSAuint64* remoteSig = Xglobal(heapObj->peerSignalPtrs[destPe] + (myPe % 8) * numSdmaQ);
    mori::core::SdmaPutThread(srcPtr, dstPtr, qElems * sizeof(T), handles, remoteSig,
                              expectedSignals, numSdmaQ, q);
  }
  
  // === Phase 2: every block waits until ALL peers wrote my staging ============
  // The receive signals live in my local HBM (written by remote SDMA), are
  // monotonic and never reset, so we wait for the exact generation `gen`. One
  // thread per (sender, queue) slot polls; all blocks share these few cache lines
  // (read-only -> L2 hits; a slot is invalidated only once per iteration when its
  // remote atomic lands).
  if (tid < npes * numQ) {
    int p = tid / numQ, q = tid - p * numQ;
    anvil::waitForSignal(&heapObj->signalPtrs[(p % 8) * numSdmaQ + q], gen);
  }
  __syncthreads();
  __threadfence_system();  // acquire: staging visible before Phase 3 reads it

  // === Phase 3: grid-strided vectorized reduce (all blocks) ================
  // Streaming reduction: every staging slot is read exactly once and the output
  // written once, so use the nontemporal load<16>/store<16> primitives from
  // device_primitives.hpp. They move 16-byte vectors with a streaming (LRU
  // bypass) hint, avoiding L2 pollution from single-use data.
  constexpr int vecSize = VecBytes / sizeof(T);  // elements per VecBytes-wide vector

  const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t gstride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t totalVecs = chunkElems / vecSize;  // total full VecBytes-wide vectors
  using AccType = typename AccumulatorType<T>::type;

  // Coalescing-friendly unroll: each thread owns NumVecs vectors spaced gstride
  // apart, so for a fixed lane index i, neighbouring threads read contiguous
  // vectors (the load stays coalesced across the warp), while the NumVecs loads
  // per peer give independent memory ops in flight (ILP/MLP).
  //
  // Because the group stride is gstride*NumVecs, each thread has at most ONE
  // partial group, and it is the last one: once a group's first member is out of
  // range, the next group (gstride*NumVecs further) is entirely out of range too.
  // So we run a GUARD-FREE fast path over complete groups (all NumVecs members
  // in-bounds), then mop up the single trailing partial group one vector at a
  // time. No per-lane "if (idx < totalVecs)" in the hot loop.
  // Staging "push" layout: peer pe's contribution is the contiguous chunkElems
  // slot at staging + pe*chunkElems. Same generic core as the pull kernel.
  auto srcBase = [staging, chunkElems](int pe) -> const T* {
    return staging + static_cast<size_t>(pe) * chunkElems;
  };

  size_t g = gtid;
  for (; g + static_cast<size_t>(NumVecs - 1) * gstride < totalVecs; g += gstride * NumVecs) {
    ReduceVecGroup<VecBytes, NumVecs, T, OpT>(srcBase, output, npes, g, gstride);
  }
  // Trailing partial group: the remaining in-bounds vectors for this thread.
  for (size_t idx = g; idx < totalVecs; idx += gstride) {
    ReduceVecGroup<VecBytes, 1, T, OpT>(srcBase, output, npes, idx, gstride);
  }

  // Scalar tail for elements not covered by the vectorized loop.
  for (size_t i = totalVecs * vecSize + gtid; i < chunkElems; i += gstride) {
    using Vec = TVecType<sizeof(T)>;
    const T* base = staging + i;
    AccType a = UpcastF<T>(load<sizeof(T)>(base));
    base += chunkElems;
    for (int pe = 1; pe < npes; pe++, base += chunkElems) {
      auto V = load<sizeof(T)>(base);
      a = OpT<T>()(a, UpcastF<T>(V));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    store<sizeof(T)>(output + i, V);
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
// Correctness requires all PEs to have produced their input before launch; the
// host issues a ShmemBarrierAll() before timing. Best for small/medium chunks
// where the staging round-trip (extra 2N local HBM traffic) and the serial
// block-0 scatter dominate.
// ---------------------------------------------------------------------------
template <int VecBytes, int NumVecs, class T, template <class> class OpT>
__global__ void ReduceScatterPullKernel(int myPe, int npes,
                                        mori::application::SymmMemObjPtr srcObj,
                                        T* __restrict__ output, size_t chunkElems) {
  constexpr int vecSize = VecBytes / sizeof(T);
  const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t gstride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t totalVecs = chunkElems / vecSize;
  using AccType = typename AccumulatorType<T>::type;

  // Peer pe's contribution to my shard: base of peer pe's input + myPe*chunkElems.
  const size_t myOfs = static_cast<size_t>(myPe) * chunkElems;
  auto srcBase = [srcObj, myOfs](int pe) -> const T* {
    return reinterpret_cast<const T*>(srcObj->peerPtrs[pe]) + myOfs;
  };

  // Same guard-free fast path + single trailing partial group as the push kernel.
  size_t g = gtid;
  for (; g + static_cast<size_t>(NumVecs - 1) * gstride < totalVecs; g += gstride * NumVecs) {
    ReduceVecGroup<VecBytes, NumVecs, T, OpT>(srcBase, output, npes, g, gstride);
  }
  for (size_t idx = g; idx < totalVecs; idx += gstride) {
    ReduceVecGroup<VecBytes, 1, T, OpT>(srcBase, output, npes, idx, gstride);
  }

  // Scalar tail for elements not covered by the vectorized loop.
  for (size_t i = totalVecs * vecSize + gtid; i < chunkElems; i += gstride) {
    using Vec = TVecType<sizeof(T)>;
    AccType a = UpcastF<T>(load<sizeof(T)>(srcBase(0) + i));
    for (int pe = 1; pe < npes; pe++) {
      auto V = load<sizeof(T)>(srcBase(pe) + i);
      a = OpT<T>()(a, UpcastF<T>(V));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    store<sizeof(T)>(output + i, V);
  }
}

// ===========================================================================
// SDMA ring reduce-scatter (proof of concept)
// ===========================================================================
// Topology: next = (myPe+1)%npes, prev = (myPe-1+npes)%npes. The shard that
// must end at rank R is seeded by rank R+1 and travels R+1 -> R+2 -> ... -> R,
// each visited rank adding its own input contribution. Equivalently, at "step"
// k = 0..npes-1 rank R handles chunk
//
//   chunk(k) = (R - 1 - k + npes) % npes
//
// where step 0 is the seed (send R's own input shard chunk(0) to next), steps
// 1..npes-2 receive a partial from prev / reduce in our local input / forward to
// next, and step npes-1 receives the final partial and writes output (chunk ==
// R). Each step's payload is split into S slices for SDMA granularity.
//
// Buffers (ALL in the symmetric heap so the address-based SDMA put translates
// local->peer via heapBaseAddr offsets):
//   input    : N = npes*chunkElems elements (read-only).
//   recvBuf  : (npes-1) slots of chunkElems. Slot s holds the partial that the
//              producer wrote at its step s; this rank consumes slot (k-1) at its
//              step k. Each slot is written EXACTLY ONCE per launch.
//   sendBuf  : (npes-1) slots of chunkElems. Forward step k reduces into slot k
//              and SDMAs that slot to next (distinct slot per step => no in-flight
//              source reuse hazard within a launch).
//   output   : chunkElems (final shard).
//   readySig : (npes-1)*S uint64. readySig[s*S+j] is bumped by the producer's
//              SDMA completion atomic when slot s, slice j lands here; the
//              consumer waits for it to reach `gen`. Monotonic, never reset, one
//              increment per slot/slice per launch (so expected == gen).
//
// PoC scope: block 0 only (single pipeline). Cross-step overlap is natural (a
// forward SDMA send of step k overlaps the recv-wait/reduce of step k+1); the
// channel-parallel version (each block owns a sub-range + its own queue) is the
// documented follow-up for bandwidth.
//
// L2 coherence (same concern as the push/two-shot kernels): SDMA writes bypass
// L2 and land in HBM, and SDMA reads bypass L2 too. So (a) after the CU reduces
// into sendBuf we write it back to HBM before the SDMA reads it, and (b) after
// the data-ready wait we invalidate L2 before the CU reads recvBuf. On gfx950 a
// system-scope fence emits buffer_wbl2/buffer_inv; on gfx94x we add the explicit
// buffer_wbl2 as the two-shot kernel does.
// ---------------------------------------------------------------------------

// Issue one SDMA slice copy local `src` -> next's `dstSlotLocal` (a local heap
// address whose heap offset is identical on `next`), with a completion atomic
// that bumps next's `sigSlotLocal`. `dstSlotLocal`/`sigSlotLocal` are this rank's
// local-equivalent addresses inside the symmetric heap.
template <class T>
__device__ __forceinline__ void RingSdmaSend(const T* src, T* dstSlotLocal,
                                             HSAuint64* sigSlotLocal, size_t bytes, int next,
                                             int q) {
  auto* states = GetGlobalGpuStatesPtr();
  auto* heapObj = states->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);
  const uintptr_t heapBase = states->heapBaseAddr;

  size_t dOff = reinterpret_cast<uintptr_t>(dstSlotLocal) - heapBase;
  size_t sOff = reinterpret_cast<uintptr_t>(sigSlotLocal) - heapBase;

  uint8_t* srcPtr = Xglobal(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src)));
  uint8_t* dstPtr = Xglobal(reinterpret_cast<uint8_t*>(heapObj->peerPtrs[next] + dOff));

  auto** handles = Xglobal(heapObj->deviceHandles_d + (next % 8) * numSdmaQ);
  HSAuint64* expectedSignals = Xglobal(heapObj->expectSignalsPtr + (next % 8) * numSdmaQ);
  // SdmaPutThread fires its atomic at signals[q]; we want next's readySig slot,
  // so bias the base by -q (pointer arithmetic) => signals[q] == that slot.
  HSAuint64* nextSig = reinterpret_cast<HSAuint64*>(heapObj->peerPtrs[next] + sOff);
  HSAuint64* signals = Xglobal(nextSig - q);
  mori::core::SdmaPutThread(srcPtr, dstPtr, bytes, handles, signals, expectedSignals, numSdmaQ, q);
}

// Block-strided fused 2-source reduce: out[i] = Op(a[i], b[i]) over `cnt` elems.
// Reuses ReduceVecGroup with a 2-entry srcBase (npes==2). Single block.
template <int VecBytes, int NumVecs, class T, template <class> class OpT>
__device__ __forceinline__ void ReduceTwoBlock(const T* a, const T* b, T* out, size_t cnt) {
  constexpr int vecSize = VecBytes / sizeof(T);
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  const size_t totalVecs = cnt / vecSize;
  using AccType = typename AccumulatorType<T>::type;
  auto srcBase = [a, b](int s) -> const T* { return s == 0 ? a : b; };

  size_t g = tid;
  for (; g + static_cast<size_t>(NumVecs - 1) * stride < totalVecs; g += stride * NumVecs) {
    ReduceVecGroup<VecBytes, NumVecs, T, OpT>(srcBase, out, 2, g, stride);
  }
  for (size_t idx = g; idx < totalVecs; idx += stride) {
    ReduceVecGroup<VecBytes, 1, T, OpT>(srcBase, out, 2, idx, stride);
  }
  for (size_t i = totalVecs * vecSize + tid; i < cnt; i += stride) {
    using Vec = TVecType<sizeof(T)>;
    AccType acc = UpcastF<T>(load<sizeof(T)>(a + i));
    acc = OpT<T>()(acc, UpcastF<T>(load<sizeof(T)>(b + i)));
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(acc));
    store<sizeof(T)>(out + i, V);
  }
}

// Writeback dirty L2 to HBM so the following SDMA read sees the CU's stores.
__device__ __forceinline__ void RingL2Writeback() {
  __syncthreads();
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
  __threadfence_system();  // gfx950: emits buffer_wbl2 (release) for system scope
  __syncthreads();
}

template <int VecBytes, int NumVecs, class T, template <class> class OpT>
__global__ void ReduceScatterRingKernel(int myPe, int npes, const T* __restrict__ input,
                                        T* __restrict__ recvBuf, T* __restrict__ sendBuf,
                                        T* __restrict__ output, HSAuint64* __restrict__ readySig,
                                        size_t chunkElems, int S, uint64_t gen) {
  if (blockIdx.x != 0) return;
  const int tid = static_cast<int>(threadIdx.x);

  const int next = (myPe + 1) % npes;
  const auto chunkOf = [npes, myPe](int k) -> int {
    int c = (myPe - 1 - k) % npes;
    return (c < 0) ? c + npes : c;
  };
  const auto sliceOfs = [chunkElems, S](int j) -> size_t {
    return static_cast<size_t>(j) * (chunkElems / static_cast<size_t>(S));
  };
  const auto sliceCnt = [chunkElems, S](int j) -> size_t {
    size_t base = chunkElems / static_cast<size_t>(S);
    return (j == S - 1) ? (chunkElems - base * j) : base;
  };

  auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);

  // --- Step 0 (seed): send our own input shard chunk(0) to next.recvBuf[slot 0].
  // One thread issues all S slice copies sequentially (round-robin over the SDMA
  // queues). Sequential enqueue avoids a same-queue write-pointer race when
  // S > numSdmaQ, while the copies still run asynchronously on the SDMA engine
  // and overlap the CU reductions of later steps.
  {
    const int sc = chunkOf(0);
    const T* src = input + sc * chunkElems;
    if (tid == 0) {
      for (int j = 0; j < S; ++j) {
        size_t base = chunkElems / S, ofs = j * base,
               cnt = (j == S - 1) ? (chunkElems - ofs) : base;
        RingSdmaSend<T>(src + ofs, recvBuf + ofs, readySig + j,
                        cnt * sizeof(T), next, j % numSdmaQ);
      }
    }
    __syncthreads();
  }

  // --- Steps 1..npes-1: per-slice software pipeline. For each slice we wait the
  // prev's data-ready signal, reduce that slice with our local input, and (unless
  // final) forward it to next. The SDMA send of slice j is fire-and-forget, so it
  // overlaps the wait+reduce of slice j+1 (copy(j) || compute(j+1)); and because
  // signalling is per-slice, `next` can begin reducing slice j as soon as it lands.
  // With S==1 this collapses to the original wait-whole / reduce-whole / send-whole.
  for (int k = 1; k < npes; ++k) {
    const int recvSlot = k - 1;  // slot prev wrote at its step k-1
    const int rc = chunkOf(k);
    const bool isFinal = (k == npes - 1);

    const T* a = recvBuf + recvSlot * chunkElems;  // SDMA-written partial from prev
    const T* b = input + rc * chunkElems;          // our local contribution
    T* dest = isFinal ? output : (sendBuf + k * chunkElems);

    for (int j = 0; j < S; ++j) {
      size_t base = chunkElems / S, ofs = j * base,
             cnt = (j == S - 1) ? (chunkElems - ofs) : base;

      // Wait for slice j of recvSlot to land, then acquire (invalidate L2) so the
      // CU reads the SDMA-written staging fresh from HBM.
      if (tid == 0) {
        anvil::waitForSignal(&readySig[static_cast<size_t>(recvSlot) * S + j], gen);
      }
      __syncthreads();
      __threadfence_system();

      // Reduce just this slice: dest[slice j] = recvBuf[slice j] (+) input[slice j].
      ReduceTwoBlock<VecBytes, NumVecs, T, OpT>(a + ofs, b + ofs, dest + ofs, cnt);

      if (!isFinal) {
        RingL2Writeback();  // make this slice of dest visible in HBM before SDMA reads it
        if (tid == 0) {
          T* dstSlot = recvBuf + k * chunkElems;  // next.recvBuf slot k
          RingSdmaSend<T>(dest + ofs, dstSlot + ofs, readySig + static_cast<size_t>(k) * S + j,
                          cnt * sizeof(T), next, j % numSdmaQ);
        }
        // No trailing __syncthreads: slice j's dest region is not touched again this
        // launch, and the next slice's wait path re-synchronises the block.
      }
    }
  }
}

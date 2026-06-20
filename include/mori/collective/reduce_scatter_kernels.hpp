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
//   * ReduceScatterPushKernel  — fused SDMA "push" scatter + receiver-side
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
    T* dstLocal = staging + static_cast<size_t>(myPe) * chunkElems;
    const T* src = input + static_cast<size_t>(destPe) * chunkElems;

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
  const int g = static_cast<int>(blockIdx.x) / G;  // this block's group/slice
  const int lb = static_cast<int>(blockIdx.x) % G; // local block index within group

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
    return (pe == myPe ? input : staging) + static_cast<size_t>(pe) * chunkElems + sOfs;
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
//   readySig : (npes-1)*S uint64 (S = number of slices). readySig[s*S+g] is
//              bumped by the producer group's single SDMA completion atomic when
//              group g's slice of slot s lands here; every consumer block in
//              group g waits for it to reach `gen`. Monotonic, never reset, one
//              increment per slot/group per launch (so expected == gen).
//   groupCnt : (npes-1)*S uint32 arrival counters (local, not symmetric). At each
//              forward step every block in group g increments groupCnt[(k-1)*S+g]
//              after flushing its sub-portion; the last arriver issues the slice
//              SDMA. Monotonic (G increments/launch), so (prev+1)%G==0 elects it.
//
// Grouped channel-parallel: N thread blocks, group g = b/G (G = N/S) cooperatively
// reduces slice g (vecSize-aligned, chunkElems/S elems), block b handling its
// localIdx = b%G sub-portion. The whole slice moves as ONE large SDMA per (step,
// group), issued by the group's last arriver; this decouples transfer granularity
// (S) from compute parallelism (N). G==1 (S==N) degenerates to one send per block.
// The partition is identical on every rank, so group g on rank R produces the
// partial that group g on rank R+1 consumes. Cross-step overlap is natural (a
// forward SDMA of step k overlaps the recv-wait/reduce of step k+1).
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

  uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src));
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(heapObj->peerPtrs[next] + dOff);

  auto** handles = heapObj->deviceHandles_d + (next % 8) * numSdmaQ;
  HSAuint64* expectedSignals = heapObj->expectSignalsPtr + (next % 8) * numSdmaQ;
  // SdmaPutThreadFused fires its atomic at signals[q]; we want next's readySig slot,
  // so bias the base by -q (pointer arithmetic) => signals[q] == that slot.
  HSAuint64* nextSig = reinterpret_cast<HSAuint64*>(heapObj->peerPtrs[next] + sOff);
  HSAuint64* signals = nextSig - q;
  // Fused (single-reservation) put: required because with N blocks > numSdmaQ
  // multiple blocks issue concurrently on the same queue.
  mori::core::SdmaPutThreadFused(srcPtr, dstPtr, bytes, handles, signals, 
            expectedSignals, numSdmaQ, q);
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
                                        uint32_t* __restrict__ groupCnt, size_t chunkElems, int S,
                                        uint64_t gen) {
  const int tid = static_cast<int>(threadIdx.x);
  const int N = static_cast<int>(gridDim.x);  // number of channel blocks
  const int b = static_cast<int>(blockIdx.x);

  const int next = (myPe + 1) % npes;
  const auto chunkOf = [npes, myPe](int k) -> int {
    int c = (myPe - 1 - k) % npes;
    return (c < 0) ? c + npes : c;
  };

  auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);

  // Grouping: the shard is split into S slices; group g = b / G owns slice g and
  // is reduced cooperatively by G = N/S blocks (localIdx = b % G handles its
  // sub-portion). The whole slice travels as ONE large SDMA transfer issued by
  // the last block in the group to finish (elected via groupCnt), which keeps the
  // transfer granularity (chunkElems/S) decoupled from compute parallelism (N).
  constexpr int vecSize = VecBytes / sizeof(T);
  const int G = N / S;          // blocks per slice (>= 1; host guarantees N % S == 0)
  const int g = b / G;          // this block's slice/group index
  const int localIdx = b % G;   // position within the group
  const int q = g % numSdmaQ;   // SDMA queue routing per group/slice

  // Slice geometry (vecSize-aligned; last slice absorbs the shard remainder).
  const size_t sliceBase = (chunkElems / static_cast<size_t>(S) / vecSize) * vecSize;
  const size_t sOfs = static_cast<size_t>(g) * sliceBase;
  const size_t sCnt = (g == S - 1) ? (chunkElems - sOfs) : sliceBase;  // whole-slice length

  // This block's reduce sub-portion within the slice (last localIdx absorbs the
  // slice remainder so the G sub-portions tile the slice exactly).
  const size_t subBase = (sliceBase / static_cast<size_t>(G) / vecSize) * vecSize;
  const size_t bOfs = sOfs + static_cast<size_t>(localIdx) * subBase;
  const size_t bCnt = (localIdx == G - 1) ? (sOfs + sCnt - bOfs) : subBase;

  // --- Step 0 (seed): send our own input slice chunk(0) to next.recvBuf[slot 0].
  // No reduction and no cross-block sync needed (input is read-only / already in
  // HBM), so one block per group (localIdx == 0) issues the whole-slice copy.
  // Signal at the per-(slot, group) slot readySig[0*S + g].
  {
    const int sc = chunkOf(0);
    const T* src = input + sc * chunkElems;
    if (tid == 0 && localIdx == 0) {
      RingSdmaSend<T>(src + sOfs, recvBuf + sOfs, readySig + g, sCnt * sizeof(T), next, q);
    }
    __syncthreads();
  }

  // --- Steps 1..npes-1: grouped ring. Every block waits its group's data-ready
  // signal, reduces its sub-portion of the slice with our local input, and (unless
  // final) flushes it to HBM and joins the group's arrival counter. The last
  // arriver issues ONE large SDMA for the whole slice; the others fall straight
  // through to the next step's wait. Signals are per-(slot, group): readySig[slot*S+g].
  for (int k = 1; k < npes; ++k) {
    const int recvSlot = k - 1;  // slot prev wrote at its step k-1
    const int rc = chunkOf(k);
    const bool isFinal = (k == npes - 1);

    const T* a = recvBuf + recvSlot * chunkElems;  // SDMA-written partial from prev
    const T* bin = input + rc * chunkElems;        // our local contribution
    T* dest = isFinal ? output : (sendBuf + k * chunkElems);

    // Wait for our group's slice of recvSlot to land, then acquire (invalidate L2)
    // so the CU reads the SDMA-written staging fresh from HBM.
    if (tid == 0) {
      anvil::waitForSignal(&readySig[static_cast<size_t>(recvSlot) * S + g], gen);
    }
    __syncthreads();
    __threadfence_system();

    // Reduce just our sub-portion: dest[range] = recvBuf[range] (+) input[range].
    ReduceTwoBlock<VecBytes, NumVecs, T, OpT>(a + bOfs, bin + bOfs, dest + bOfs, bCnt);

    if (!isFinal) {
      RingL2Writeback();  // make our sub-portion of dest visible in HBM before SDMA reads it
      if (tid == 0) {
        // Join the group's arrival barrier. The counter is monotonic and gets
        // exactly G increments per launch, so (prev+1) % G == 0 uniquely elects
        // the last arriver every launch with no reset. After it observes all G
        // arrivals, every sub-portion of the slice is flushed to HBM (each block
        // did RingL2Writeback before incrementing), so the single SDMA is safe.
        uint32_t prev = atomicAdd(&groupCnt[static_cast<size_t>(recvSlot) * S + g], 1u);
        if ((prev + 1) % static_cast<uint32_t>(G) == 0) {
          T* dstSlot = recvBuf + k * chunkElems;  // next.recvBuf slot k
          RingSdmaSend<T>(dest + sOfs, dstSlot + sOfs,
                          readySig + static_cast<size_t>(k) * S + g, sCnt * sizeof(T), next, q);
        }
      }
      // No trailing __syncthreads: this step's dest region is not touched again
      // this launch, and the next step's wait path re-synchronises the block.
    }
  }
}

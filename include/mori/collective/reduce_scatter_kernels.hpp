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

#include "mori/collective/collectives_common.hpp"  // StreamLoad/Store, SdmaPutWarpFusedS
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;
using namespace mori::collective;

// Phase-1 packet emission strategy for the push kernel's single-warp fast path.
//   RS_PUSH_LDS_STAGE == 0 : each lane writes its packet directly to the ring.
//   RS_PUSH_LDS_STAGE == 1 : lanes stage packets in LDS (bank-conflict-free,
//        17-dword padded slots), then up to npes*S*4 threads flush them to the
//        ring as coalesced b128 stores (one b128 per thread).
#ifndef RS_PUSH_LDS_STAGE
#define RS_PUSH_LDS_STAGE 0
#endif

// Fast-path capacity: npes <= 8 (one warp), S = 1<<logS <= 8. A fused
// copy+atomic packet is 64B = 16 dwords; pad each LDS slot to 17 dwords so the
// per-lane build write (stride 17, coprime to the 32 LDS banks) is
// conflict-free.
static constexpr int kRSPushMaxPeers = 8;
static constexpr int kRSPushMaxSlices = 8;
static constexpr int kRSPushPktDwords = 16;   // sizeof(SDMA_PKT_COPY_WITH_ATOMIC)/4
static constexpr int kRSPushSlotDwords = 17;  // padded stride (16 + 1)

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

// Spin until every bit in `mask` is set in *addr. Used by the push kernel's
// receiver-side completion flag, where each sender ORs in its own bit.
__device__ __forceinline__ void waitForSignalMask(HSAuint64* addr, uint64_t mask) {
  while ((__hip_atomic_load(Tglobal(addr), __ATOMIC_RELAXED, 
                  __HIP_MEMORY_SCOPE_AGENT) & mask) != mask) {
  }
}

// ---------------------------------------------------------------------------
// Fused reduce-scatter kernel ("push", sliced into S = 1<<logS slices)
//
//   input         : raw symmetric-heap pointer, N = npes*chunkElems elements
//   staging       : raw symmetric-heap pointer, npes slots of chunkElems elements
//   output        : raw symmetric-heap pointer, chunkElems elements
//   groupCounters : plain device buffer (>= S uint64), local-only block counters
//
// Each shard is split into S slices. A sender issues S separate SDMA copies; copy
// s ORs bit (1<<senderPe) via an SDMA ADD64 into the receiver's per-slice bitmask
// flag signalPtrs[s] (one copy + one atomic per (sender,slice), so adds never
// collide). The receiver's grid is partitioned into S groups of G = gridDim.x/S
// blocks; group g spins only on slice g's flag and reduces only slice g, so a
// group starts as soon as its slice lands (pipelining). The last block of each
// group zeroes that slice's flag + its local counter (groupCounters[g]) for the
// next launch -- no monotonic generation counter needed. Limited to npes <= 64.
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
// ---------------------------------------------------------------------------
template <int VecBytes, int NumVecs, class T, template <class> class OpT>
__global__ void ReduceScatterPushKernel(int myPe, int npes, int logS, const T* __restrict__ input,
                                    T* __restrict__ staging, T* __restrict__ output,
                                    uint64_t* __restrict__ groupCounters, size_t chunkElems) {
  auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
  const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);
  constexpr int vecSize = VecBytes / sizeof(T);

  const int S = 1 << logS;
  // vecSize-aligned slice length; the last slice absorbs the remainder.
  const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize;

  // Completion bitmask: bit p means "sender p's copy has landed". Self-copy is
  // skipped, so we wait for every peer bit except our own.
  const uint64_t allMask = (npes >= 64) ? ~0ull : ((1ull << npes) - 1);
  const uint64_t wantMask = allMask & ~(1ull << myPe);

  int tid = threadIdx.x;

  // === Phase 1: SDMA scatter (block 0 only) ====================================
  // Lane (peer,slice) issues an SDMA copy of slice `slice` into destPe's staging
  // slot followed by an ADD64 of (1<<myPe) into destPe's per-slice flag
  // signalPtrs[slice]. The atomic targets the *receiver's* flag, so completion is
  // observed on the receive side (Phase 2) -- fire-and-forget, no local quiet, no
  // cross-PE barrier. Per-slice flags make the slice order irrelevant on the
  // receive side.
  //
  // Fast path (npes*S <= warpSize, i.e. npes<=8): a single warp issues ALL peers'
  // copies concurrently; each peer's slice-0 leader reserves the S-packet block and
  // submits once. Peers use distinct queues (peer%8), so the per-warp multi-queue
  // reserve/submit has one sole producer per queue and cannot deadlock, and the S
  // reservations + S doorbells per peer collapse into one of each. Fallback path
  // (larger npes): warp-strided one-warp-per-peer, lane 0 reserves/submits.
  if (blockIdx.x == 0) {
    T* dstLocal = staging + myPe * chunkElems;
    const size_t heapBase = GetGlobalGpuStatesPtr()->heapBaseAddr;
    const size_t sliceBytes = sliceLen * sizeof(T);
    const size_t lastBytes = (chunkElems - static_cast<size_t>(S - 1) * sliceLen) * sizeof(T);
    // off is uniform across peers: my data lands in every peer's staging[myPe] slot.
    const size_t off = reinterpret_cast<uintptr_t>(dstLocal) - heapBase;

    if (npes * S <= warpSize) {
      constexpr size_t pkt = sizeof(SDMA_PKT_COPY_WITH_ATOMIC);
#if RS_PUSH_LDS_STAGE
      // --- Fast path (LDS-staged): build packets into bank-conflict-free LDS ----
      // slots, then flush them to the rings with up to npes*S*4 threads (one
      // coalesced b128 store per thread). The build write uses one lane per packet
      // at a 17-dword slot stride (coprime to the 32 LDS banks) so the 16-dword
      // packet write is conflict-free. The flush spans multiple warps, so the ring
      // writes need a block-wide fence + barrier before any leader rings a doorbell
      // (the submitter's own s_waitcnt(0) only drains its own wave).
      __shared__ uint32_t pktBuf[kRSPushMaxPeers * kRSPushMaxSlices * kRSPushSlotDwords];
      __shared__ uint64_t sStart[kRSPushMaxPeers];
      __shared__ uint64_t sOff[kRSPushMaxPeers];

      // -- Build phase: one lane per packet (peer = lane/S, slice = lane%S). --
      const int lane = tid;
      const int peer = (lane < npes * S) ? lane / S : npes;
      const int slice = (lane < npes * S) ? lane % S : 0;
      const bool bactive = (lane < npes * S) && (peer != myPe);
      if (bactive) {
        auto** handles = heapObj->deviceHandles_d + (peer % 8) * numSdmaQ;
        auto* handle = static_cast<SdmaCollectiveHandle*>(*(handles + 0));
        if (slice == 0) {
          // Leader reserves the whole S-packet block and publishes base/pad to LDS.
          uint64_t offset = 0;
          uint64_t sb = handle->ReserveQueueSpace(pkt * S, offset);
          if (offset) handle->fillNops(sb, offset);
          sStart[peer] = sb;
          sOff[peer] = offset;
        }
        const uint8_t* s =
            reinterpret_cast<const uint8_t*>(input + peer * chunkElems) + slice * sliceBytes;
        uint8_t* d =
            reinterpret_cast<uint8_t*>(heapObj->peerPtrs[peer] + off) + slice * sliceBytes;
        size_t sz = (slice == S - 1) ? lastBytes : sliceBytes;
        // Build the fused copy+atomic packet DIRECTLY into the LDS slot (dword
        // stores via the DW_x unions), skipping the register-resident struct and
        // the reg->LDS copy. Layout: copy = dwords 0..6, atomic = dwords 7..14,
        // trailing single-dword NOP = dword 15. Per-field cross-lane stride stays
        // kRSPushSlotDwords (17, coprime to 32 banks) so the build is conflict-free.
        uint32_t* dw = &pktBuf[lane * kRSPushSlotDwords];  // slot == peer*S + slice == lane
        anvil::WriteCopyPacket(dw, const_cast<uint8_t*>(s), d, sz);
        anvil::WriteAtomicAddPacket(dw + 7, heapObj->peerSignalPtrs[peer] + slice,
                                    (1ull << myPe));
        dw[15] = 0;  // trailing single-dword SDMA NOP (must be 0)
      }
      __syncthreads();

      // -- Flush phase: one thread per b128 (packet = fb/4, b128 = fb%4), block-
      // strided so it is robust to any blockDim (<= npes*S*4 total b128s). --
      const int totalB128 = npes * S * 4;
      for (int fb = tid; fb < totalB128; fb += blockDim.x) {
        const int fpkt = fb / 4;
        const int bb = fb % 4;
        const int fpeer = fpkt / S;
        const int fslice = fpkt % S;
        if (fpeer == myPe) continue;  // uniform across the warp (peer-granular)
        auto** handles = heapObj->deviceHandles_d + (fpeer % 8) * numSdmaQ;
        auto* handle = static_cast<SdmaCollectiveHandle*>(*(handles + 0));
        const uint64_t wptrIndex = sStart[fpeer] + sOff[fpeer] + static_cast<uint64_t>(fslice) * pkt;
        const uint64_t baseDword = handle->WrapIntoRing(wptrIndex) / sizeof(uint32_t);
        // Read the b128 from LDS as 4 dwords (stride-17 slot is 4B- but not
        // 16B-aligned, so avoid ds_read_b128), then store it coalesced to the ring.
        const int slotBase = fpkt * kRSPushSlotDwords + bb * 4;
        TVecType<16> v = {pktBuf[slotBase + 0], pktBuf[slotBase + 1], pktBuf[slotBase + 2],
                          pktBuf[slotBase + 3]};
        StreamStore<16>(handle->queueBuf + baseDword + bb * 4, v, /*system_scope=*/false);
      }
      // All flush warps must finish + be system-visible before any doorbell.
      __threadfence_system();
      __syncthreads();

      // -- Submit phase: each peer's leader rings one doorbell on its own queue. --
      if (bactive && slice == 0) {
        auto** handles = heapObj->deviceHandles_d + (peer % 8) * numSdmaQ;
        auto* handle = static_cast<SdmaCollectiveHandle*>(*(handles + 0));
        handle->submitPacket(sStart[peer], sStart[peer] + sOff[peer] + static_cast<uint64_t>(pkt) * S);
      }
#else
      // --- Fast path (direct): one warp issues EVERY (peer, slice) copy at once. -
      // Lane `lane` serves peer = lane/S, slice = lane%S (npes*S <= warpSize). Each
      // peer's slice-0 leader reserves the S-packet block and submits once; all
      // lanes write their own packet in parallel. Distinct queues per peer make the
      // per-warp multi-queue reserve/submit deadlock-free.
      const int lane = tid;
      const int peer = lane / S;
      const int slice = lane % S;
      const bool active = (tid < warpSize) && (lane < npes * S) && (peer != myPe);

      SdmaCollectiveHandle* handle = nullptr;
      if (active) {
        auto** handles = heapObj->deviceHandles_d + (peer % 8) * numSdmaQ;
        handle = static_cast<SdmaCollectiveHandle*>(*(handles + 0));
      }

      // Slice-0 leader of each peer reserves the whole S-packet block on its queue.
      uint64_t startBase = 0, offset = 0;
      if (active && slice == 0) startBase = handle->ReserveQueueSpace(pkt * S, offset);
      // Broadcast (startBase, offset) from each peer's leader lane (= (lane/S)*S).
      // Executed by the full warp so __shfl stays convergent.
      if (tid < warpSize) {
        const int leaderLane = (lane / S) * S;
        startBase = __shfl(startBase, leaderLane);
        offset = __shfl(offset, leaderLane);
      }
      if (active && slice == 0 && offset) handle->fillNops(startBase, offset);

      if (active) {
        const uint8_t* s =
            reinterpret_cast<const uint8_t*>(input + peer * chunkElems) + slice * sliceBytes;
        uint8_t* d =
            reinterpret_cast<uint8_t*>(heapObj->peerPtrs[peer] + off) + slice * sliceBytes;
        size_t sz = (slice == S - 1) ? lastBytes : sliceBytes;
        SDMA_PKT_COPY_WITH_ATOMIC packet = {
            .copy = anvil::CreateCopyPacket(const_cast<uint8_t*>(s), d, sz),
            .atomic = anvil::CreateAtomicAddPacket(heapObj->peerSignalPtrs[peer] + slice,
                                                   (1ull << myPe))};
        handle->placePacketAt(packet, startBase + offset + static_cast<uint64_t>(slice) * pkt);
      }
      // Reconverge so all packet stores are visible before any leader submits; the
      // submitter's s_waitcnt(0) inside submitPacket then drains the wave's vmcnt.
      if (tid < warpSize) __syncwarp();
      if (active && slice == 0)
        handle->submitPacket(startBase, startBase + offset + static_cast<uint64_t>(pkt) * S);
#endif
    } else {
      // --- Fallback (npes*S > warpSize): warp-strided, one warp per peer. ---
      const int w = tid / warpSize;
      const int numW = blockDim.x / warpSize;
      for (int destPe = w; destPe < npes; destPe += numW) {
        if (destPe == myPe) continue;  // uniform across the warp
        auto** handles = heapObj->deviceHandles_d + (destPe % 8) * numSdmaQ;
        const T* src = input + destPe * chunkElems;
        uint8_t* dst = reinterpret_cast<uint8_t*>(heapObj->peerPtrs[destPe] + off);
        mori::collective::SdmaPutWarpFusedS(
            const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src)), dst, sliceBytes, lastBytes,
            handles, heapObj->peerSignalPtrs[destPe], /*qId=*/0, S, /*addVal=*/(1ull << myPe));
      }
    }
  }

  // Grouping: G blocks per slice; group g (= slice g) handles only its slice.
  const int G = gridDim.x / S;            // host guarantees gridDim.x % S == 0
  const int g = blockIdx.x / G;           // slice / group index
  const int lb = blockIdx.x % G;          // local block within the group

  // === Phase 2: wait until every sender's bit is set in slice g's flag =========
  // Each slice's bitmask lives in my local HBM (written by remote SDMA atomics).
  // One thread per block polls signalPtrs[g]; every block does its own
  // wait+acquire (read-only -> L2 hits). npes==1 has no peers, nothing to wait for.
  if (npes > 1 && tid == 0) {
    waitForSignalMask(&heapObj->signalPtrs[g], wantMask);
  }
  __syncthreads();
  __threadfence_system();  // acquire: staging visible before Phase 3 reads it

  // === Phase 3: grid-strided vectorized reduce over slice g ===================
  // Streaming reduction over slice g only, strided over the group's G blocks. Uses
  // the nontemporal load<16>/store<16> primitives (single-use data, bypass L2).
  const size_t sOfs = static_cast<size_t>(g) * sliceLen;
  const size_t sCnt = (g == S - 1) ? (chunkElems - sOfs) : sliceLen;
  const size_t ltid = static_cast<size_t>(lb) * blockDim.x + threadIdx.x;
  const size_t lstride = static_cast<size_t>(G) * blockDim.x;
  const size_t totalVecs = sCnt / vecSize;
  using AccType = typename AccumulatorType<T>::type;

  // Pass the RUNTIME npes to ReduceVecGroup so its per-peer reduce loop stays
  // rolled (constant-folding it raises VGPR pressure and lowers occupancy).
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
    AccType a = UpcastF<T>(StreamLoad<sizeof(T)>(base));
    base += chunkElems;
    for (int pe = 1; pe < npes; pe++, base += chunkElems) {
      auto V = StreamLoad<sizeof(T)>(base);
      a = OpT<T>()(a, UpcastF<T>(V));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    StreamStore<sizeof(T)>(out + i, V);
  }

  // === Reset: the last block of group g zeroes slice g's flag + counter =========
  // A block only reaches here after passing Phase 2 (it has observed the full mask
  // for slice g), so when the group counter hits G every block in the group is past
  // Phase 2 and clearing the flag cannot drop an unseen bit. Cross-iteration
  // ordering is guaranteed by the host hipStreamSynchronize + ShmemBarrierAll
  // between launches. Counters are local-only (plain hipMalloc'd buffer).
  if (npes > 1 && tid == 0) {
    uint64_t z = atomicAdd(reinterpret_cast<unsigned long long*>(&groupCounters[g]), 1ull);
    if (z + 1 == static_cast<uint64_t>(G)) {
      heapObj->signalPtrs[g] = 0;  // clear slice g's bitmask flag
      groupCounters[g] = 0;        // clear group g's local counter
      __threadfence_system();
    }
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
  
  // Add poison to disable promotion to global memory space
#if USE_FLAT_MEMORY
  if (blockIdx.x + myPe == 1777) {
    output = reinterpret_cast<T*>(srcObj->peerPtrs[777]);
  }
#endif

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
    AccType a = UpcastF<T>(StreamLoad<sizeof(T)>(srcBase(0) + i));
    for (int pe = 1; pe < NPES; pe++) {
      auto V = StreamLoad<sizeof(T)>(srcBase(pe) + i);
      a = OpT<T>()(a, UpcastF<T>(V));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    StreamStore<sizeof(T)>(output + i, V);
  }
}

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

// Reduce-scatter test using a SINGLE fused SDMA kernel (single-process,
// multi-threaded: one thread per GPU/PE, no MPI, no file-based bootstrap).
//
// Reduce-scatter semantics: every PE owns the full vector of N = npes * chunk
// elements. After the collective, PE q holds the reduction (over all PEs) of
// shard q:
//
//   output_q[j] = REDUCE_p( input_p[ q*chunkElems + j ] )   for j in [0, chunkElems)
//
// Algorithm (all three steps fused into ONE kernel launch):
//   Phase 1 (block 0 only): SDMA "push" scatter. For every destination peer p,
//     PE myPe sends input[p*chunkElems ..] into peer p's staging slot myPe,
//     split across SDMA queues for bandwidth. Fire-and-forget: the completion
//     atomic rides the same queue as the copy and targets the *receiver's*
//     signalPtrs, so it lands only after the data does. No local quiet.
//   Phase 2 (all blocks): receiver-side wait. Each PE spins on its own signalPtrs
//     until every (sender, queue) slot reaches the launch generation `gen`,
//     i.e. all peers finished writing our staging. This replaces both the local
//     SDMA quiet and the cross-PE barrier, and -- since the signals are global --
//     needs no block-0 -> all-blocks flag handoff, so every block is independent.
//   Phase 3 (all blocks): grid-strided vectorized reduction of the npes staging
//     slots into the output shard. Templated by element type T and reduction Op.
//
// Because there is no co-resident flag handoff, the reduce grid is NOT capped by
// multiProcessorCount -- push uses full SM occupancy, like the pull kernel.
//
// This file also contains a small, self-contained COOPERATIVE-LAUNCH demo
// kernel (CoopGridSyncDemoKernel) that uses cooperative_groups::grid_group and
// grid.sync() for study/comparison. It is unrelated to shmem and runs locally.
//
// Usage: ./reduce_scatter_test [num_gpus] [num_elems]
//   num_gpus  : number of GPUs/PEs to run in this process.
//   num_elems : TOTAL input element count (matches XLA's num_elems). The per-rank
//               output shard is chunkElems = num_elems / num_gpus; its byte size
//               (chunkElems * sizeof(ElemT)) must be a multiple of 16.

// Sofar the best:
// reduce_scatter_test: 4 PEs, 10485760 bytes/shard (2621440 elems), 41943040 bytes input/PE
// reduce_scatter_test: numQ=2 numChannels=80 (SMs=256)
// Rank 2: PASS (0 errors) | 5 warmup + 20 runs | avg 0.232 ms (180.836 GB/s) min 0.229 ms (182.929 GB/s) max 0.238 ms

// RCCL:
// I0000 00:00:1781190820.849456  195820 gpu_collectives_test.cc:798] bytes: 4194304 ms: 0.034509 alg_bw: 121.542 GB/s  bus_bw: 91.1567 GB/s
// I0000 00:00:1781190820.850156  195820 gpu_collectives_test.cc:798] bytes: 8388608 ms: 0.0528935 alg_bw: 158.594 GB/s  bus_bw: 118.946 GB/s
// I0000 00:00:1781190820.851236  195820 gpu_collectives_test.cc:798] bytes: 16777216 ms: 0.0879905 alg_bw: 190.671 GB/s  bus_bw: 143.003 GB/s
// I0000 00:00:1781190820.853203  195820 gpu_collectives_test.cc:798] bytes: 33554432 ms: 0.160136 alg_bw: 209.536 GB/s  bus_bw: 157.152 GB/s
// I0000 00:00:1781190820.856929  195820 gpu_collectives_test.cc:798] bytes: 67108864 ms: 0.307485 alg_bw: 218.251 GB/s  bus_bw: 163.688 GB/s
// I0000 00:00:1781190820.864209  195820 gpu_collectives_test.cc:798] bytes: 134217728 ms: 0.605257 alg_bw: 221.753 GB/s  bus_bw: 166.315 GB/s

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"  // load<N>/store<N>
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/internal.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

using ElemT = float;  // element type used by the test instantiation
#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// Streaming (cache-bypassing) 16-byte load/store, RCCL-style (see rccl op128.h).
#if (defined(__gfx942__) || defined(__gfx950__)) && \
    __has_builtin(__builtin_amdgcn_global_load_b128) &&  \
    __has_builtin(__builtin_amdgcn_global_store_b128)
#define RS_HAVE_GLOBAL_B128 1
#else
#define RS_HAVE_GLOBAL_B128 0
#endif

using rs_v4u = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;
using rs_v4u_gptr = __attribute__((address_space(1))) rs_v4u*;

template <int Bytes>
__device__ __forceinline__ typename mori::core::VecTypeSelector<Bytes>::dataType StreamLoad(
    const void* p) {
  return mori::core::load<Bytes>(p);  // generic fallback (8/4/2/1 byte tails)
}
template <int Bytes>
__device__ __forceinline__ void StreamStore(
    void* p, typename mori::core::VecTypeSelector<Bytes>::dataType v) {
  mori::core::store<Bytes>(p, v);
}

template <>
__device__ __forceinline__ typename mori::core::VecTypeSelector<16>::dataType StreamLoad<16>(
    const void* p) {
#if RS_HAVE_GLOBAL_B128
  rs_v4u raw = __builtin_amdgcn_global_load_b128((rs_v4u_gptr)p, "");
  return __builtin_bit_cast(mori::core::VecTypeSelector<16>::dataType, raw);
#else
  return mori::core::load<16>(p);
#endif
}
template <>
__device__ __forceinline__ void StreamStore<16>(
    void* p, typename mori::core::VecTypeSelector<16>::dataType v) {
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
  using Vec = typename VecTypeSelector<VecBytes>::dataType;
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
// This decouples the data layout from the reduction so the same code serves both
// the staging "push" path (peers are contiguous slots in one local buffer) and
// the "pull" path (peers are separate symmetric peer pointers).
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
    uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src + qOfs));
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(heapObj->peerPtrs[destPe] + offset);

    auto** handles = heapObj->deviceHandles_d + (destPe % 8) * numSdmaQ;
    // SdmaPutThread still does expectedSignals[q]++ on this local array; that is
    // now dead bookkeeping (nobody quiets locally) but harmless.
    HSAuint64* expectedSignals = heapObj->expectSignalsPtr + (destPe % 8) * numSdmaQ;
    // Completion atomic -> destPe's signalPtrs at the slot keyed by ME (sender):
    //   destPe.signalPtrs[(myPe % 8) * numSdmaQ + q]
    HSAuint64* remoteSig = heapObj->peerSignalPtrs[destPe] + (myPe % 8) * numSdmaQ;
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
    using Vec = typename VecTypeSelector<sizeof(T)>::dataType;
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
    using Vec = typename VecTypeSelector<sizeof(T)>::dataType;
    AccType a = UpcastF<T>(load<sizeof(T)>(srcBase(0) + i));
    for (int pe = 1; pe < npes; pe++) {
      auto V = load<sizeof(T)>(srcBase(pe) + i);
      a = OpT<T>()(a, UpcastF<T>(V));
    }
    Vec V = __builtin_bit_cast(Vec, DowncastF<T>(a));
    store<sizeof(T)>(output + i, V);
  }
}

// ---------------------------------------------------------------------------
// Fill / verify kernels
// ---------------------------------------------------------------------------
// input[k] = (myPe + 1) + (k % 8). All small integers (exact in float). With
// this pattern the reduce-scatter SUM result on shard owned by PE q is:
//   output_q[j] = sum_p[(p+1) + ((q*chunkElems + j) % 8)]
//              = npes*(npes+1)/2 + npes * ((q*chunkElems + j) % 8)
__global__ void FillPatternKernel(ElemT* buf, size_t numElements, int myPe) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = static_cast<ElemT>((myPe + 1) + static_cast<int>(i % 8));
  }
}

__global__ void VerifyKernel(const ElemT* output, size_t chunkElems, int myPe, int npes,
                             uint32_t* errorCount) {
  const float base = static_cast<float>(npes) * (npes + 1) / 2.0f;
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < chunkElems;
       j += (size_t)gridDim.x * blockDim.x) {
    size_t globalIdx = static_cast<size_t>(myPe) * chunkElems + j;
    float expected = base + static_cast<float>(npes) * static_cast<float>(globalIdx % 8);
    if (fabsf(static_cast<float>(output[j]) - expected) > 1e-3f) {
      atomicAdd(errorCount, 1u);
    }
  }
}
struct ThreadInfo {
  int rank{-1};
  int worldSize{-1};
  int deviceId{-1};
  int ret_code{-1};
};

// ---------------------------------------------------------------------------
// Test body (runs after ShmemInit)
// ---------------------------------------------------------------------------
static void RunReduceScatterThreadedTest(size_t numElems, const UniqueId& uid, ThreadInfo& info) {
  HIP_RUNTIME_CHECK(hipSetDevice(info.deviceId));

  auto* bootstrap = new SocketBootstrapNetwork(uid, info.rank, info.worldSize);
  int status = ShmemInit(bootstrap);
  if (status != 0) {
    XPUT("ERROR: ShmemInit failed (ret=%d)", status);
    info.ret_code = status;
    return;
  }

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // numElems is the TOTAL input element count (matches XLA's num_elems); the
  // per-rank output shard is chunkElems = numElems / npes.
  const size_t chunkElems = numElems / npes;
  const size_t chunkBytes = chunkElems * sizeof(ElemT);
  const size_t N = static_cast<size_t>(npes) * chunkElems;  // input/staging element count
  const size_t inBytes = N * sizeof(ElemT);
  const size_t stagingBytes = N * sizeof(ElemT);
  const size_t outBytes = chunkBytes;
  const size_t totalBytes = inBytes + stagingBytes + outBytes;
  ShmemBarrierAll();

  if (info.deviceId == 0) {
    XPUT("reduce_scatter_test: %d PEs, %zu bytes/shard (%zu elems), %zu bytes input/PE", npes,
         chunkBytes, chunkElems, inBytes);
  }

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  // Single symmetric-heap allocation: [ input(N) | staging(N) | output(chunk) ].
  void* baseBuf = ShmemMalloc(totalBytes);
  if (baseBuf == nullptr) {
    XPUT("ERROR: ShmemMalloc(%zu) failed", totalBytes);
    info.ret_code = -1;
    return;
  }
  SymmMemObjPtr baseObj = ShmemQueryMemObjPtr(baseBuf);
  assert(baseObj.IsValid());

  ElemT* input = reinterpret_cast<ElemT*>(baseBuf);
  ElemT* staging = input + N;
  ElemT* output = staging + N;

  HIP_RUNTIME_CHECK(hipMemsetAsync(baseBuf, 0, totalBytes, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

  // Fill input with the per-PE pattern.
  constexpr int kThreads = 256;
  int fillBlocks = static_cast<int>(std::min<size_t>(1024, (N + kThreads - 1) / kThreads));
  FillPatternKernel<<<fillBlocks, kThreads, 0, stream>>>(input, N, myPe);
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  ShmemBarrierAll();

  // Channel sizing. With the receiver-side completion signal (Phase 2) the push
  // kernel no longer has a cross-block flag handoff or co-residency requirement,
  // so BOTH push and pull use full SM occupancy.
  const int numQ = static_cast<int>(std::max(1u, baseObj->sdmaNumQueue));
  hipDeviceProp_t prop;
  HIP_RUNTIME_CHECK(hipGetDeviceProperties(&prop, info.deviceId));
  constexpr int VecBytes = 16,  NumVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ElemT);
  size_t totalVecs = chunkElems / (VecSize * NumVecs);
  int wantBlocks = static_cast<int>(std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  int blocks = std::min(wantBlocks, std::max(1, prop.multiProcessorCount));

  const bool usePull = [] {
    const char* e = std::getenv("RS_PULL");
    return e != nullptr && std::atoi(e) != 0;
  }();

  if (info.deviceId == 0) {
    XPUT("reduce_scatter_test: mode=%s numQ=%d blocks=%d (SMs=%d)",
         usePull ? "PULL" : "PUSH", numQ, blocks, prop.multiProcessorCount);
  }

  // --- Benchmark ---
  constexpr int nWarmup = 5;
  constexpr int nRuns = 20;
  hipEvent_t tStart, tStop;
  HIP_RUNTIME_CHECK(hipEventCreate(&tStart));
  HIP_RUNTIME_CHECK(hipEventCreate(&tStop));

  float totalMs = 0, minMs = 1e9f, maxMs = 0;
  for (int iter = 0; iter < nWarmup + nRuns; iter++) {
    ShmemBarrierAll();
    HIP_RUNTIME_CHECK(hipEventRecord(tStart, stream));
    if (usePull) {
      // input lives at offset 0 of baseBuf, so baseObj->peerPtrs[pe] is peer pe's
      // input base. output is the local shard buffer.
      ReduceScatterPullKernel<VecBytes, NumVecs, ElemT, SumOp><<<blocks, kThreads, 0, stream>>>(
          myPe, npes, baseObj, output, chunkElems);
    } else {
      // gen = monotonic launch generation (signals are zeroed once and never
      // reset, so the receive-side wait matches the exact per-launch value).
      ReduceScatterKernel<VecBytes, NumVecs, ElemT, SumOp><<<blocks, kThreads, 0, stream>>>(
          myPe, npes, numQ, input, staging, output, chunkElems,
          static_cast<uint64_t>(iter) + 1);
    }
    HIP_RUNTIME_CHECK(hipEventRecord(tStop, stream));
    HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));

    float iterMs = 0;
    HIP_RUNTIME_CHECK(hipEventElapsedTime(&iterMs, tStart, tStop));
    if (iter >= nWarmup) {
      totalMs += iterMs;
      minMs = std::min(minMs, iterMs);
      maxMs = std::max(maxMs, iterMs);
    }
  }

  // After every PE's last kernel has been stream-synced above, each PE has
  // observed all incoming completion signals -> every outgoing DMA (incoming to
  // some peer) has landed. This host barrier makes that global before teardown,
  // so no in-flight peer DMA can target our staging after we free it.
  ShmemBarrierAll();

  float avgMs = totalMs / nRuns;
  // Bytes scattered over the network per PE ~ input bytes (N elements).
  double avgBw = (inBytes / 1e9) / (avgMs / 1e3);
  double maxBw = (inBytes / 1e9) / (minMs / 1e3);

  // --- Verify (last iteration result) ---
  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemsetAsync(dErrors, 0, sizeof(uint32_t), stream));
  int vBlocks =
      static_cast<int>(std::min<size_t>(1024, (chunkElems + kThreads - 1) / kThreads));
  VerifyKernel<<<vBlocks, kThreads, 0, stream>>>(output, chunkElems, myPe, npes, dErrors);
  uint32_t hErrors = 0;
  HIP_RUNTIME_CHECK(hipMemcpyAsync(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  HIP_RUNTIME_CHECK(hipFree(dErrors));

  if (hErrors != 0 || myPe == 0) {
    XPUT("Rank %d: %s | %d warmup + %d runs | avg %.3f ms (%.3f GB/s) "
       "min %.3f ms (%.3f GB/s) max %.3f ms\n--------------------",
       myPe, hErrors == 0 ? "PASS" : "FAIL", nWarmup, nRuns, avgMs, avgBw, minMs, maxBw,
       maxMs);
  }

  HIP_RUNTIME_CHECK(hipEventDestroy(tStart));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStop));
  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  ShmemFree(baseBuf);
  ShmemFinalize();
  info.ret_code = 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  int deviceCount = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));
  if (argc < 3) {
    XPUT("Usage: %s [num_gpus] [num_elems]\n", argv[0]);
    return 1;
  }
  int numGpus = std::atoi(argv[1]);
  if (numGpus < 1 || numGpus > deviceCount) {
    XPUT("Usage: %s [num_gpus] [num_elems]   (num_gpus in 1..%d)\n", argv[0],
            deviceCount);
    return 1;
  }

  // num_elems is the TOTAL input element count (matches XLA's num_elems). The
  // per-rank output shard is chunkElems = num_elems / num_gpus.
  size_t numElems = std::atol(argv[2]);
  assert(numElems % numGpus == 0 && "num_elems must be divisible by num_gpus");
  size_t chunkElems = numElems / numGpus;
  size_t chunkBytes = chunkElems * sizeof(ElemT);
  assert(chunkBytes >= 16 && (chunkBytes % 16) == 0 &&
         "per-shard bytes (num_elems/num_gpus * sizeof) must be a multiple of 16");

  // Single in-process UniqueId shared by all threads (no file/MPI needed).
  mori_shmem_uniqueid_t uid_bytes{};
  int ret = ShmemGetUniqueId(&uid_bytes);
  if (ret != 0) {
    XPUT("ERROR: ShmemGetUniqueId failed (ret=%d)", ret);
    return 1;
  }
  UniqueId uid;
  static_assert(sizeof(uid) == sizeof(uid_bytes), "UniqueId size mismatch");
  std::memcpy(&uid, uid_bytes.data(), sizeof(uid));

  std::vector<std::thread> threads;
  std::vector<ThreadInfo> infos(numGpus);
  threads.reserve(numGpus);
  for (int i = 0; i < numGpus; i++) {
    infos[i].rank = i;
    infos[i].worldSize = numGpus;
    infos[i].deviceId = i;
    threads.emplace_back(RunReduceScatterThreadedTest, numElems, std::cref(uid),
                         std::ref(infos[i]));
  }
  for (auto& t : threads) t.join();

  for (const auto& inf : infos) {
    if (inf.ret_code != 0) {
      XPUT("ERROR: Rank %d returned non-zero ret_code %d", inf.rank, inf.ret_code);
      return 1;
    }
  }
  return 0;
}

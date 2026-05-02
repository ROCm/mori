// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce — SDMA scatter + reduce + SDMA AllGather.
//
// SCATTER_MODE=0: Single-kernel AllReduce (SDMA scatter + reduce + SDMA AG)
//   1-chunk mode (shard < 2×kMinChunkShardBytes):
//     Block 0: burst scatter → cc wait → cross-PE reduce_complete → SDMA AG → AG wait.
//     Compute blocks: scatter-poll → reduce → wbl2+fence → chunks_complete.
//   Multi-chunk mode (shard ≥ 2×kMinChunkShardBytes, default 2 chunks):
//     Block 0: burst scatter → per-chunk (cc wait → cross-PE barrier → AG) → AG wait.
//     Compute blocks: scatter-poll → reduce → wbl2+fence → chunks_complete.
//     Overlaps AG(c) SDMA transfer with scatter(c+1)+reduce(c+1) on CU.
//     wbl2+CC for intermediate chunks runs on wavefront 1 (thread 64),
//     parallel with scatter-poll on wavefront 0.
//
//   Cross-PE reduce_complete barrier is REQUIRED before AG to prevent data
//   corruption: my AG writes to peer's transit[myPe-slot] via SDMA; peer's
//   reduce reads transit[all slots] including that slot as part of its
//   compute. If my AG fires before peer's reduce reads, peer sees my AG
//   value instead of the original scatter value — visible as inplace 2nd
//   call verification failures with stale transit.
//
//   Barrier uses flagsMemObj (symmetric memory) + reduceCompleteBase
//   counter — separate from qId=0 SDMA scatter signal — to avoid
//   cross-iteration races.
//
// SCATTER_MODE=1: P2P read + CU AG (legacy path).
//
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <type_traits>

#include "mori/shmem/shmem.hpp"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"
#include "mori/collective/intra_node/kernels/vec_type.cuh"
#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"

namespace mori {
namespace collective {

static_assert(kMaxPipelineBlocks <= 385,
              "compute block count must fit in grid launch");

// ---- Phase-level timestamp instrumentation (optional) ----------------------
// When phase_ts != nullptr, block 0 thread 0 writes __builtin_amdgcn_s_memtime()
// at each phase boundary. Used to diagnose per-phase cost of AR[0] cold path.
// Slot layout (see file-level comment for PipelinedAllReduceSdmaKernel):
//   0: kernel entry
//   1: scatter submit done
//   2 + 3*c + {0,1,2}: chunk c {compute-wait, cross-PE-barrier, AG-submit} done
//   2 + 3*numChunks:   AG wait done (all peers)
//   3 + 3*numChunks:   block 0 exit
//   --- Stage 2b-0 per-chunk AG completion timestamps (E' prototype) ---
//   kArPhaseAgDoneBase + c: block 0 observed chunk c's AG completion
//   --- post-AG wait instrumentation (E' prototype) ---
//   30: block 0 post_ag_flag set (after AG wait done, before block 0 exit)
//   31: compute block 1 post_ag_flag observed (spin-wait finished)
// MUST stay in sync with kPhaseTsCapacity in twoshot_allreduce_sdma_class.hpp.
// 256 leaves headroom for block0 + AG-done + CB + A-group slots. Was 32 which OOB'd in Plan A
// (CB1 max slot 11 + 3*numChunks = 35 at numChunks=8) and corrupted adjacent
// device memory, breaking CrossPeBarrier sync state → deadlock. See Entry 19.
static constexpr int kArPhaseTsCapacity = 256;
static constexpr int kArPhaseAgDoneBase = 64;
static constexpr int kArPhaseCbBase = 88;
static constexpr int kArPhaseABase = 144;

__device__ inline void ar_write_phase_ts(uint64_t* ts, int idx) {
  if (ts != nullptr && blockIdx.x == 0 && threadIdx.x == 0 &&
      static_cast<unsigned>(idx) < static_cast<unsigned>(kArPhaseTsCapacity)) {
    ts[idx] = __builtin_amdgcn_s_memtime();
  }
}

// Same as ar_write_phase_ts but only the first compute block (block 1) writes.
// Used to measure compute-block-internal phases (dispatch latency, scatter-poll
// wait, reduce execution, fetch_add). Call sites pass the historical logical
// layout (10, 11+3c+...), but the helper remaps it to kArPhaseCbBase+...
// so it never collides with block-0 slots when numChunks >= 4.
__device__ inline void ar_write_phase_ts_cb1(uint64_t* ts, int idx) {
  const int mapped = kArPhaseCbBase + (idx - 10);
  if (ts != nullptr && blockIdx.x == 1 && threadIdx.x == 0 &&
      idx >= 10 &&
      static_cast<unsigned>(mapped) < static_cast<unsigned>(kArPhaseTsCapacity)) {
    ts[mapped] = __builtin_amdgcn_s_memtime();
  }
}

// Same timestamp helper for the first AG-pull block in Plan A v2.
// Call sites pass logical slots 49 / 50+3c+..., but the helper remaps them
// to kArPhaseABase+... to avoid colliding with CB slots.
__device__ inline void ar_write_phase_ts_cbA(uint64_t* ts, int idx, int nR) {
  const int mapped = kArPhaseABase + (idx - 49);
  if (ts != nullptr && blockIdx.x == static_cast<unsigned>(nR + 1) &&
      threadIdx.x == 0 &&
      idx >= 49 &&
      static_cast<unsigned>(mapped) < static_cast<unsigned>(kArPhaseTsCapacity)) {
    ts[mapped] = __builtin_amdgcn_s_memtime();
  }
}

// ============================================================================
// OneShotDirectOutputKernel — correctness-first direct-output cadence probe.
//
// Each rank copies its input into a symmetric input buffer before launch. The
// kernel waits for all peers to signal input readiness, reads every peer input
// directly, reduces locally, and writes the full result to user_output.
// ============================================================================
template <typename T>
__global__ void OneShotDirectOutputKernel(
    int myPe, int npes,
    const application::SymmMemObjPtr inputObj,
    const application::SymmMemObjPtr flagsMemObj,
    T* __restrict__ user_output,
    size_t elementCount,
    uint64_t readyBase,
    uint64_t* phase_ts) {
  ar_write_phase_ts(phase_ts, 0);
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;
  const size_t packedCount =
      (elementCount + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);

  if (blockIdx.x == 0) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
      HSAuint64* myFlag = reinterpret_cast<HSAuint64*>(flagsMemObj->localPtr);
      __hip_atomic_fetch_add(myFlag, 1ULL,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    if (tid < npes && tid != myPe) {
      const uint64_t expected = readyBase + 1ULL;
      HSAuint64* peerFlag = reinterpret_cast<HSAuint64*>(flagsMemObj->peerPtrs[tid]);
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(peerFlag) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d ONESHOT_DIRECT wait peer=%d expected=%llu got=%llu\n",
                 myPe, tid, (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(peerFlag));
          stuck = 0;
        }
      }
    }
    __syncthreads();
    ar_write_phase_ts(phase_ts, 1);
  }
  __syncthreads();

  const size_t linear =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  P* __restrict__ outP = reinterpret_cast<P*>(user_output);

  if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                std::is_same<typename P::type, int32_t>::value) {
    for (size_t k = linear; k < packedCount; k += stride) {
      const P* p0 = reinterpret_cast<const P*>(inputObj->peerPtrs[0]);
      P acc = p0[k];
      for (int pe = 1; pe < npes; ++pe) {
        const P* pp = reinterpret_cast<const P*>(inputObj->peerPtrs[pe]);
        packed_assign_add(acc, pp[k]);
      }
      outP[k] = acc;
    }
  } else {
    for (size_t k = linear; k < packedCount; k += stride) {
      const P* p0 = reinterpret_cast<const P*>(inputObj->peerPtrs[0]);
      A acc = upcast_v<typename P::type, pack_size>(p0[k]);
      for (int pe = 1; pe < npes; ++pe) {
        const P* pp = reinterpret_cast<const P*>(inputObj->peerPtrs[pe]);
        packed_assign_add(acc, upcast_v<typename P::type, pack_size>(pp[k]));
      }
      outP[k] = downcast_v<typename P::type, pack_size>(acc);
    }
  }
  __syncthreads();
  ar_write_phase_ts(phase_ts, 2);
}

// Chunked variant of OneShotDirectOutputKernel. It keeps the direct-output
// semantics but services only one chunk per launch, so the experiment can test
// whether smaller AR service units reduce continuous backlog.
template <typename T>
__global__ void ChunkedDirectOutputKernel(
    int myPe, int npes,
    const application::SymmMemObjPtr inputObj,
    const application::SymmMemObjPtr flagsMemObj,
    T* __restrict__ user_output,
    size_t elementCount,
    size_t chunkElementCount,
    int chunkIdx,
    uint64_t readyBase,
    uint64_t* phase_ts) {
  ar_write_phase_ts(phase_ts, 0);
  if (elementCount == 0 || chunkElementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;
  const size_t packedCount =
      (elementCount + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const size_t packedChunk =
      (chunkElementCount + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const size_t off = static_cast<size_t>(chunkIdx) * packedChunk;
  if (off >= packedCount) return;
  size_t cnt = packedChunk;
  if (off + cnt > packedCount) cnt = packedCount - off;

  if (blockIdx.x == 0) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
      HSAuint64* myFlag = reinterpret_cast<HSAuint64*>(flagsMemObj->localPtr);
      __hip_atomic_fetch_add(myFlag, 1ULL,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    if (tid < npes && tid != myPe) {
      const uint64_t expected = readyBase + static_cast<uint64_t>(chunkIdx + 1);
      HSAuint64* peerFlag = reinterpret_cast<HSAuint64*>(flagsMemObj->peerPtrs[tid]);
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(peerFlag) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d CHUNKED_DIRECT c=%d wait peer=%d expected=%llu got=%llu\n",
                 myPe, chunkIdx, tid, (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(peerFlag));
          stuck = 0;
        }
      }
    }
    __syncthreads();
    ar_write_phase_ts(phase_ts, 1);
  }
  __syncthreads();

  const size_t linear =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  P* __restrict__ outP = reinterpret_cast<P*>(user_output);

  if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                std::is_same<typename P::type, int32_t>::value) {
    for (size_t k = linear; k < cnt; k += stride) {
      const size_t idx = off + k;
      const P* p0 = reinterpret_cast<const P*>(inputObj->peerPtrs[0]);
      P acc = p0[idx];
      for (int pe = 1; pe < npes; ++pe) {
        const P* pp = reinterpret_cast<const P*>(inputObj->peerPtrs[pe]);
        packed_assign_add(acc, pp[idx]);
      }
      outP[idx] = acc;
    }
  } else {
    for (size_t k = linear; k < cnt; k += stride) {
      const size_t idx = off + k;
      const P* p0 = reinterpret_cast<const P*>(inputObj->peerPtrs[0]);
      A acc = upcast_v<typename P::type, pack_size>(p0[idx]);
      for (int pe = 1; pe < npes; ++pe) {
        const P* pp = reinterpret_cast<const P*>(inputObj->peerPtrs[pe]);
        packed_assign_add(acc, upcast_v<typename P::type, pack_size>(pp[idx]));
      }
      outP[idx] = downcast_v<typename P::type, pack_size>(acc);
    }
  }
  __syncthreads();
  ar_write_phase_ts(phase_ts, 2);
}

// Multi-lane direct-output probe. Lanes partition the output index space and
// traverse peers in forward/reverse order to exercise bidirectional XGMI while
// preserving correctness (each output element is produced by exactly one lane).
template <typename T>
__global__ void MultiLaneDirectOutputKernel(
    int myPe, int npes,
    const application::SymmMemObjPtr inputObj,
    const application::SymmMemObjPtr flagsMemObj,
    T* __restrict__ user_output,
    size_t elementCount,
    int forwardLanes,
    int reverseLanes,
    int blocksPerLane,
    uint64_t readyBase,
    uint64_t* phase_ts) {
  ar_write_phase_ts(phase_ts, 0);
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;
  const size_t packedCount =
      (elementCount + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const int laneCount = forwardLanes + reverseLanes;
  if (laneCount <= 0 || blocksPerLane <= 0) return;

  if (blockIdx.x == 0) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
      HSAuint64* myFlag = reinterpret_cast<HSAuint64*>(flagsMemObj->localPtr);
      __hip_atomic_fetch_add(myFlag, 1ULL,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    if (tid < npes && tid != myPe) {
      const uint64_t expected = readyBase + 1ULL;
      HSAuint64* peerFlag = reinterpret_cast<HSAuint64*>(flagsMemObj->peerPtrs[tid]);
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(peerFlag) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d MULTILANE_DIRECT wait peer=%d expected=%llu got=%llu\n",
                 myPe, tid, (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(peerFlag));
          stuck = 0;
        }
      }
    }
    __syncthreads();
    ar_write_phase_ts(phase_ts, 1);
  }
  __syncthreads();

  const int blockId = static_cast<int>(blockIdx.x);
  const int lane = blockId / blocksPerLane;
  const int posInLane = blockId - lane * blocksPerLane;
  if (lane >= laneCount) return;
  const bool reverse = lane >= forwardLanes;
  const size_t laneLinear =
      static_cast<size_t>(posInLane) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t laneStride =
      static_cast<size_t>(blocksPerLane) * static_cast<size_t>(blockDim.x);

  P* __restrict__ outP = reinterpret_cast<P*>(user_output);

  if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                std::is_same<typename P::type, int32_t>::value) {
    for (size_t logical = laneLinear;; logical += laneStride) {
      const size_t idx = static_cast<size_t>(lane) + logical * static_cast<size_t>(laneCount);
      if (idx >= packedCount) break;
      const P* self = reinterpret_cast<const P*>(inputObj->peerPtrs[myPe]);
      P acc = self[idx];
      for (int hop = 1; hop < npes; ++hop) {
        const int pe = reverse ? (myPe - hop + npes) % npes : (myPe + hop) % npes;
        const P* pp = reinterpret_cast<const P*>(inputObj->peerPtrs[pe]);
        packed_assign_add(acc, pp[idx]);
      }
      outP[idx] = acc;
    }
  } else {
    for (size_t logical = laneLinear;; logical += laneStride) {
      const size_t idx = static_cast<size_t>(lane) + logical * static_cast<size_t>(laneCount);
      if (idx >= packedCount) break;
      const P* self = reinterpret_cast<const P*>(inputObj->peerPtrs[myPe]);
      A acc = upcast_v<typename P::type, pack_size>(self[idx]);
      for (int hop = 1; hop < npes; ++hop) {
        const int pe = reverse ? (myPe - hop + npes) % npes : (myPe + hop) % npes;
        const P* pp = reinterpret_cast<const P*>(inputObj->peerPtrs[pe]);
        packed_assign_add(acc, upcast_v<typename P::type, pack_size>(pp[idx]));
      }
      outP[idx] = downcast_v<typename P::type, pack_size>(acc);
    }
  }
  __syncthreads();
  ar_write_phase_ts(phase_ts, 2);
}

template <typename T>
__global__ void RingShardReduceScatterRoundKernel(
    int myPe, int npes,
    const application::SymmMemObjPtr accumObj,
    const application::SymmMemObjPtr recvObj,
    CrossPeBarrier* __restrict__ barrier,
    size_t elementCount,
    int round,
    uint64_t signalBase) {
  if (elementCount == 0 || npes <= 1) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;
  const size_t elemsPerShard =
      (elementCount + static_cast<size_t>(npes) - 1) / static_cast<size_t>(npes);
  const size_t packedPerShard =
      (elemsPerShard + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const size_t packedTotal =
      (elementCount + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const int next = (myPe + 1) % npes;
  const int prev = (myPe - 1 + npes) % npes;
  const int sendShard = (myPe - round + npes) % npes;
  const int recvShard = (myPe - round - 1 + npes) % npes;
  const size_t sendOff = static_cast<size_t>(sendShard) * packedPerShard;
  const size_t recvOff = static_cast<size_t>(recvShard) * packedPerShard;
  size_t recvCnt = packedPerShard;
  if (recvOff + recvCnt > packedTotal) recvCnt = packedTotal - recvOff;
  size_t sendCnt = packedPerShard;
  if (sendOff + sendCnt > packedTotal) sendCnt = packedTotal - sendOff;
  const size_t sendBytes = sendCnt * sizeof(P);
  const uint32_t numQ = recvObj->sdmaNumQueue;
  constexpr uint32_t qId = 0;

  if (blockIdx.x == 0) {
    if (threadIdx.x == 0 && sendBytes > 0) {
      anvil::SdmaQueueDeviceHandle** dh =
          recvObj->deviceHandles_d + static_cast<size_t>(next) * numQ;
      HSAuint64* rSig = recvObj->peerSignalPtrs[next]
          + static_cast<size_t>(myPe) * numQ;
      uint8_t* src = reinterpret_cast<uint8_t*>(accumObj->localPtr)
          + sendOff * sizeof(P);
      uint8_t* dst = reinterpret_cast<uint8_t*>(recvObj->peerPtrs[next])
          + sendOff * sizeof(P);
      core::SdmaPutThread(src, dst, sendBytes, dh, rSig, numQ, qId);
    }
    if (threadIdx.x == 0) {
      const uint64_t expected = signalBase + static_cast<uint64_t>(round + 1);
      HSAuint64* sig = recvObj->signalPtrs + static_cast<size_t>(prev) * numQ + qId;
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(sig) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d RING_RS round=%d prev=%d expected=%llu got=%llu\n",
                 myPe, round, prev, (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(sig));
          stuck = 0;
        }
      }
      __threadfence();
      __hip_atomic_store(&barrier->flag, 1u, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  while (__hip_atomic_load(&barrier->flag, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT) < 1u) {
    __builtin_amdgcn_s_sleep(1);
  }

  const size_t linear =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  P* __restrict__ accum = reinterpret_cast<P*>(accumObj->localPtr) + recvOff;
  const P* __restrict__ recv = reinterpret_cast<const P*>(recvObj->localPtr) + recvOff;

  if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                std::is_same<typename P::type, int32_t>::value) {
    for (size_t k = linear; k < recvCnt; k += stride) {
      P v = accum[k];
      packed_assign_add(v, recv[k]);
      accum[k] = v;
    }
  } else {
    for (size_t k = linear; k < recvCnt; k += stride) {
      A v = upcast_v<typename P::type, pack_size>(accum[k]);
      packed_assign_add(v, upcast_v<typename P::type, pack_size>(recv[k]));
      accum[k] = downcast_v<typename P::type, pack_size>(v);
    }
  }
}

template <typename T>
__global__ void RingShardAllGatherRoundKernel(
    int myPe, int npes,
    const application::SymmMemObjPtr accumObj,
    const application::SymmMemObjPtr recvObj,
    CrossPeBarrier* __restrict__ barrier,
    T* __restrict__ user_output,
    size_t elementCount,
    int round,
    uint64_t signalBase) {
  if (elementCount == 0 || npes <= 1) return;

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;
  const size_t elemsPerShard =
      (elementCount + static_cast<size_t>(npes) - 1) / static_cast<size_t>(npes);
  const size_t packedPerShard =
      (elemsPerShard + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const size_t packedTotal =
      (elementCount + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const int next = (myPe + 1) % npes;
  const int prev = (myPe - 1 + npes) % npes;
  const int sendShard = (myPe - round + 1 + npes) % npes;
  const int recvShard = (myPe - round + npes) % npes;
  const size_t sendOff = static_cast<size_t>(sendShard) * packedPerShard;
  const size_t recvOff = static_cast<size_t>(recvShard) * packedPerShard;
  size_t recvCnt = packedPerShard;
  if (recvOff + recvCnt > packedTotal) recvCnt = packedTotal - recvOff;
  size_t sendCnt = packedPerShard;
  if (sendOff + sendCnt > packedTotal) sendCnt = packedTotal - sendOff;
  const size_t sendBytes = sendCnt * sizeof(P);
  const uint32_t numQ = recvObj->sdmaNumQueue;
  constexpr uint32_t qId = 1;

  if (blockIdx.x == 0) {
    if (threadIdx.x == 0 && sendBytes > 0) {
      anvil::SdmaQueueDeviceHandle** dh =
          recvObj->deviceHandles_d + static_cast<size_t>(next) * numQ;
      HSAuint64* rSig = recvObj->peerSignalPtrs[next]
          + static_cast<size_t>(myPe) * numQ;
      uint8_t* src = reinterpret_cast<uint8_t*>(accumObj->localPtr)
          + sendOff * sizeof(P);
      uint8_t* dst = reinterpret_cast<uint8_t*>(recvObj->peerPtrs[next])
          + sendOff * sizeof(P);
      core::SdmaPutThread(src, dst, sendBytes, dh, rSig, numQ, qId);
    }
    if (threadIdx.x == 0) {
      const uint64_t expected = signalBase + static_cast<uint64_t>(round + 1);
      HSAuint64* sig = recvObj->signalPtrs + static_cast<size_t>(prev) * numQ + qId;
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(sig) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d RING_AG round=%d prev=%d expected=%llu got=%llu\n",
                 myPe, round, prev, (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(sig));
          stuck = 0;
        }
      }
      __threadfence();
      __hip_atomic_store(&barrier->flag, 1u, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  while (__hip_atomic_load(&barrier->flag, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT) < 1u) {
    __builtin_amdgcn_s_sleep(1);
  }

  const size_t linear =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  P* __restrict__ accum = reinterpret_cast<P*>(accumObj->localPtr) + recvOff;
  P* __restrict__ outP = reinterpret_cast<P*>(user_output) + recvOff;
  const P* __restrict__ recv = reinterpret_cast<const P*>(recvObj->localPtr) + recvOff;
  for (size_t k = linear; k < recvCnt; k += stride) {
    const P v = recv[k];
    accum[k] = v;
    outP[k] = v;
  }
  P* __restrict__ outSend = reinterpret_cast<P*>(user_output) + sendOff;
  const P* __restrict__ accumSend = reinterpret_cast<const P*>(accumObj->localPtr) + sendOff;
  for (size_t k = linear; k < sendCnt; k += stride) {
    outSend[k] = accumSend[k];
  }
}

template <typename T>
__global__ void RingShardDirectKernel(
    int myPe, int npes,
    const application::SymmMemObjPtr accumObj,
    const application::SymmMemObjPtr recvObj,
    T* __restrict__ user_output,
    size_t elementCount,
    uint64_t scatterBase,
    uint64_t agBase,
    uint64_t* phase_ts) {
  ar_write_phase_ts(phase_ts, 0);
  if (elementCount == 0 || npes <= 1) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;
  const size_t elemsPerShard =
      (elementCount + static_cast<size_t>(npes) - 1) / static_cast<size_t>(npes);
  const size_t packedPerShard =
      (elemsPerShard + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const size_t packedTotal =
      (elementCount + static_cast<size_t>(pack_size) - 1) /
      static_cast<size_t>(pack_size);
  const int next = (myPe + 1) % npes;
  const int prev = (myPe - 1 + npes) % npes;
  const uint32_t numQ = recvObj->sdmaNumQueue;

  const size_t linear =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);

  for (int round = 0; round < npes - 1; ++round) {
    const int sendShard = (myPe - round + npes) % npes;
    const int recvShard = (myPe - round - 1 + npes) % npes;
    const size_t sendOff = static_cast<size_t>(sendShard) * packedPerShard;
    const size_t recvOff = static_cast<size_t>(recvShard) * packedPerShard;
    size_t recvCnt = packedPerShard;
    if (recvOff + recvCnt > packedTotal) recvCnt = packedTotal - recvOff;
    size_t sendCnt = packedPerShard;
    if (sendOff + sendCnt > packedTotal) sendCnt = packedTotal - sendOff;
    const size_t sendBytes = sendCnt * sizeof(P);

    if (blockIdx.x == 0 && threadIdx.x == 0 && sendBytes > 0) {
      anvil::SdmaQueueDeviceHandle** dh =
          recvObj->deviceHandles_d + static_cast<size_t>(next) * numQ;
      HSAuint64* rSig = recvObj->peerSignalPtrs[next]
          + static_cast<size_t>(myPe) * numQ;
      uint8_t* src = reinterpret_cast<uint8_t*>(accumObj->localPtr)
          + sendOff * sizeof(P);
      uint8_t* dst = reinterpret_cast<uint8_t*>(recvObj->peerPtrs[next])
          + sendOff * sizeof(P);
      core::SdmaPutThread(src, dst, sendBytes, dh, rSig, numQ, 0);
    }
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      const uint64_t expected = scatterBase + static_cast<uint64_t>(round + 1);
      HSAuint64* sig = recvObj->signalPtrs + static_cast<size_t>(prev) * numQ;
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(sig) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d RING_FUSED_RS round=%d prev=%d expected=%llu got=%llu\n",
                 myPe, round, prev, (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(sig));
          stuck = 0;
        }
      }
    }
    __syncthreads();

    P* __restrict__ accum = reinterpret_cast<P*>(accumObj->localPtr) + recvOff;
    const P* __restrict__ recv = reinterpret_cast<const P*>(recvObj->localPtr) + recvOff;
    if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                  std::is_same<typename P::type, int32_t>::value) {
      for (size_t k = linear; k < recvCnt; k += stride) {
        P v = accum[k];
        packed_assign_add(v, recv[k]);
        accum[k] = v;
      }
    } else {
      for (size_t k = linear; k < recvCnt; k += stride) {
        A v = upcast_v<typename P::type, pack_size>(accum[k]);
        packed_assign_add(v, upcast_v<typename P::type, pack_size>(recv[k]));
        accum[k] = downcast_v<typename P::type, pack_size>(v);
      }
    }
    __syncthreads();
  }

  for (int round = 0; round < npes - 1; ++round) {
    const int sendShard = (myPe - round + 1 + npes) % npes;
    const int recvShard = (myPe - round + npes) % npes;
    const size_t sendOff = static_cast<size_t>(sendShard) * packedPerShard;
    const size_t recvOff = static_cast<size_t>(recvShard) * packedPerShard;
    size_t recvCnt = packedPerShard;
    if (recvOff + recvCnt > packedTotal) recvCnt = packedTotal - recvOff;
    size_t sendCnt = packedPerShard;
    if (sendOff + sendCnt > packedTotal) sendCnt = packedTotal - sendOff;
    const size_t sendBytes = sendCnt * sizeof(P);

    if (blockIdx.x == 0 && threadIdx.x == 0 && sendBytes > 0) {
      anvil::SdmaQueueDeviceHandle** dh =
          recvObj->deviceHandles_d + static_cast<size_t>(next) * numQ;
      HSAuint64* rSig = recvObj->peerSignalPtrs[next]
          + static_cast<size_t>(myPe) * numQ + 1;
      uint8_t* src = reinterpret_cast<uint8_t*>(accumObj->localPtr)
          + sendOff * sizeof(P);
      uint8_t* dst = reinterpret_cast<uint8_t*>(recvObj->peerPtrs[next])
          + sendOff * sizeof(P);
      core::SdmaPutThread(src, dst, sendBytes, dh, rSig, numQ, 1);
    }
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      const uint64_t expected = agBase + static_cast<uint64_t>(round + 1);
      HSAuint64* sig = recvObj->signalPtrs + static_cast<size_t>(prev) * numQ + 1;
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(sig) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d RING_FUSED_AG round=%d prev=%d expected=%llu got=%llu\n",
                 myPe, round, prev, (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(sig));
          stuck = 0;
        }
      }
    }
    __syncthreads();

    P* __restrict__ accum = reinterpret_cast<P*>(accumObj->localPtr) + recvOff;
    P* __restrict__ outP = reinterpret_cast<P*>(user_output) + recvOff;
    const P* __restrict__ recv = reinterpret_cast<const P*>(recvObj->localPtr) + recvOff;
    for (size_t k = linear; k < recvCnt; k += stride) {
      const P v = recv[k];
      accum[k] = v;
      outP[k] = v;
    }
    const size_t selfShard = static_cast<size_t>(myPe) * packedPerShard;
    if (selfShard < packedTotal) {
      size_t selfCnt = packedPerShard;
      if (selfShard + selfCnt > packedTotal) selfCnt = packedTotal - selfShard;
      P* __restrict__ outSelf = reinterpret_cast<P*>(user_output) + selfShard;
      const P* __restrict__ accumSelf = reinterpret_cast<const P*>(accumObj->localPtr) + selfShard;
      for (size_t k = linear; k < selfCnt; k += stride) {
        outSelf[k] = accumSelf[k];
      }
    }
    __syncthreads();
  }
  ar_write_phase_ts(phase_ts, 2);
}
// ---------------------------------------------------------------------------

// ============================================================================
// ScatterSdmaOnlyKernel — 1-block kernel that submits SDMA scatter to all
// peers and waits for all incoming scatter data.  Paired with
// PipelinedAllReduceSdmaKernel<..., EXTERNAL_SCATTER=true> so compute blocks
// start reducing immediately without spin-waiting on scatter signals.
// ============================================================================
template <typename T>
__global__ void ScatterSdmaOnlyKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t scatterBase) {

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t chunkPerRank =
      ((chunkElementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t packedChunkPerRank = chunkPerRank / pack_size;

  if (elementCountPerRank == 0 || chunkPerRank == 0) return;

  const int numChunks =
      static_cast<int>((packedPerRank + packedChunkPerRank - 1) / packedChunkPerRank);
  const size_t chunkBytes = chunkPerRank * bytesPerElement;
  const size_t totalShardBytes = elementCountPerRank * bytesPerElement;
  const uint32_t numQ = dstMemObj->sdmaNumQueue;
  const int thr = static_cast<int>(threadIdx.x);

  // Phase 1: submit SDMA scatter to every peer for every chunk
  if (thr < npes && thr != myPe) {
    const int destPe = thr;
    anvil::SdmaQueueDeviceHandle** dh =
        dstMemObj->deviceHandles_d + destPe * numQ;
    HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
        + static_cast<size_t>(myPe) * numQ;
    for (int c = 0; c < numChunks; c++) {
      const size_t cOff = static_cast<size_t>(c) * chunkBytes;
      size_t actualBytes = chunkBytes;
      if (cOff + actualBytes > totalShardBytes)
        actualBytes = totalShardBytes - cOff;
      if (actualBytes > 0) {
        uint8_t* src = reinterpret_cast<uint8_t*>(const_cast<T*>(input))
            + static_cast<size_t>(destPe) * totalShardBytes + cOff;
        uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
            + static_cast<size_t>(myPe) * totalShardBytes + cOff;
        core::SdmaPutThread(src, dst, actualBytes, dh, rSig, numQ, 0);
      }
    }
  }
  __syncthreads();

  // Phase 2: wait for all incoming scatter data from every peer
  if (thr < npes && thr != myPe) {
    const int sender = thr;
    const uint64_t expected =
        scatterBase + static_cast<uint64_t>(numChunks);
    HSAuint64* sig = dstMemObj->signalPtrs
        + static_cast<size_t>(sender) * numQ;
    uint64_t stuck = 0;
    while (core::AtomicLoadRelaxed(sig) < expected) {
      __builtin_amdgcn_s_sleep(1);
      if (++stuck >= 100000000ULL) {
        printf("[STUCK] PE %d K1 thr %d scatter-wait sender=%d expected=%llu got=%llu\n",
               myPe, thr, sender,
               (unsigned long long)expected,
               (unsigned long long)core::AtomicLoadRelaxed(sig));
        stuck = 0;
      }
    }
  }
}

// Same data movement as ScatterSdmaOnlyKernel, but waits after each chunk.
// This is used by experimental copy-pipe paths where burst-submitting all
// chunks on qId=0 has been observed to leave the final signal short.
template <typename T>
__global__ void ScatterSdmaOnlyWaitEachChunkKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t scatterBase) {

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t chunkPerRank =
      ((chunkElementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t packedChunkPerRank = chunkPerRank / pack_size;

  if (elementCountPerRank == 0 || chunkPerRank == 0) return;

  const int numChunks =
      static_cast<int>((packedPerRank + packedChunkPerRank - 1) / packedChunkPerRank);
  const size_t chunkBytes = chunkPerRank * bytesPerElement;
  const size_t totalShardBytes = elementCountPerRank * bytesPerElement;
  const uint32_t numQ = dstMemObj->sdmaNumQueue;
  const int thr = static_cast<int>(threadIdx.x);

  for (int c = 0; c < numChunks; c++) {
    const size_t cOff = static_cast<size_t>(c) * chunkBytes;
    size_t actualBytes = chunkBytes;
    if (cOff + actualBytes > totalShardBytes)
      actualBytes = totalShardBytes - cOff;

    if (thr < npes && thr != myPe) {
      const int destPe = thr;
      anvil::SdmaQueueDeviceHandle** dh =
          dstMemObj->deviceHandles_d + destPe * numQ;
      HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
          + static_cast<size_t>(myPe) * numQ;
      uint8_t* src = reinterpret_cast<uint8_t*>(const_cast<T*>(input))
          + static_cast<size_t>(destPe) * totalShardBytes + cOff;
      uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
          + static_cast<size_t>(myPe) * totalShardBytes + cOff;
      core::SdmaPutThread(src, dst, actualBytes, dh, rSig, numQ, 0);
    }
    __syncthreads();

    if (thr < npes && thr != myPe) {
      const int sender = thr;
      const uint64_t expected = scatterBase + static_cast<uint64_t>(c + 1);
      HSAuint64* sig = dstMemObj->signalPtrs
          + static_cast<size_t>(sender) * numQ;
      uint64_t stuck = 0;
      while (core::AtomicLoadRelaxed(sig) < expected) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          printf("[STUCK] PE %d K1-chunked thr %d chunk=%d scatter-wait sender=%d expected=%llu got=%llu\n",
                 myPe, thr, c, sender,
                 (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(sig));
          stuck = 0;
        }
      }
    }
    __syncthreads();
  }
}

template <typename T, int SCATTER_MODE = 0, bool MULTI_CHUNK = false,
          bool EXTERNAL_SCATTER = false>
__global__ void PipelinedAllReduceSdmaKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr agDstMemObj,
    const application::SymmMemObjPtr flagsMemObj,
    CrossPeBarrier* __restrict__ barrier,
    const application::SymmMemObjPtr inputSymmObj,
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t scatterBase,
    uint64_t agBase,
    const uint64_t* __restrict__ agBaseByQ,
    uint64_t reduceCompleteBase,
    uint64_t* phase_ts,
    bool multi_q_ag,
    uint32_t* post_ag_flag /* nullptr = old behavior; non-null = Stage 1 E'
                              prototype: compute blocks stay alive after
                              reduce until block 0 sets this flag (after AG
                              wait done). Used to measure how much CU
                              occupancy during AG wait costs the overlap
                              benchmark (GEMM variability, wall time). */ ) {

  ar_write_phase_ts(phase_ts, 0);  // phase 0: kernel entry

  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t chunkPerRank =
      ((chunkElementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t packedChunkPerRank = chunkPerRank / pack_size;

  if (elementCountPerRank == 0 || chunkPerRank == 0) return;

  const int numChunks =
      static_cast<int>((packedPerRank + packedChunkPerRank - 1) / packedChunkPerRank);
  const size_t chunkBytes = chunkPerRank * bytesPerElement;
  const size_t totalShardBytes = elementCountPerRank * bytesPerElement;

  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  const bool separateAgBuffer = (agDstMemObj.gpu != nullptr);
  const uint32_t numQ = dstMemObj->sdmaNumQueue;
  const uint32_t agNumQ = separateAgBuffer ? agDstMemObj->sdmaNumQueue : numQ;
  const int compBlocks = static_cast<int>(gridDim.x) - 1;

  __shared__ uint64_t s_scatter_by_sender[64];
  __shared__ uint64_t s_ag_by_sender[64];
  __shared__ uint32_t s_cc_base;
  __shared__ const P* s_pe_ptrs[8];

  // chunks_complete is zeroed by the host on stream BEFORE each kernel launch
  // (hipMemsetAsync in pipelined()), so this launch's atomic increments are
  // absolute counts starting from 0. The old design read chunks_complete at
  // kernel entry as a baseline, which had a race: compute blocks could
  // already have incremented the counter before block 0 read it, making the
  // baseline too large and the ccTarget unreachable — concrete cause of the
  // probabilistic overlap-mode deadlock.
  if (threadIdx.x == 0) {
    s_cc_base = 0u;
  }
  {
    const int s = static_cast<int>(threadIdx.x);
    if (s < npes && s != myPe) {
      s_scatter_by_sender[s] = scatterBase;
      if (blockIdx.x == 0) {
        s_ag_by_sender[s] = agBase;
      }
    }
  }
  __syncthreads();

  // =========================================================================
  // SCATTER_MODE = 0 — SDMA scatter(qId=0) + AG(qId=1)
  // =========================================================================
  if constexpr (SCATTER_MODE == 0) {

    if (numQ < 2) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("PE %d: pipelined SDMA needs sdmaNumQueue>=2 (got %u)\n",
               myPe, numQ);
      }
      return;
    }

    if (blockIdx.x != 0) {
      // =================================================================
      // COMPUTE BLOCKS (1..N): scatter-poll → reduce → chunks_complete
      // =================================================================
      ar_write_phase_ts_cb1(phase_ts, 10);  // compute block entry (block 1 thr 0)

      const size_t compTid =
          static_cast<size_t>(blockIdx.x - 1) * static_cast<size_t>(blockDim.x)
          + threadIdx.x;
      const size_t compStride =
          static_cast<size_t>(compBlocks) * static_cast<size_t>(blockDim.x);

      for (int c = 0; c < numChunks; c++) {
        ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 0);  // chunk c: loop start
        const size_t off = static_cast<size_t>(c) * packedChunkPerRank;

        if (c > 0 && threadIdx.x == 64) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
          asm volatile("buffer_wbl2" ::: "memory");
#endif
          __threadfence();
          __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                                 __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        if constexpr (!EXTERNAL_SCATTER) {
          if (threadIdx.x < static_cast<unsigned>(npes - 1)) {
            const int idx = static_cast<int>(threadIdx.x);
            const int sender = idx < myPe ? idx : idx + 1;
            const uint64_t expected =
                s_scatter_by_sender[sender] + static_cast<uint64_t>(c + 1);
            HSAuint64* sig = dstMemObj->signalPtrs
                + static_cast<size_t>(sender) * numQ;
            while (core::AtomicLoadRelaxed(sig) < expected) {
              __builtin_amdgcn_s_sleep(2);
              __builtin_amdgcn_s_sleep(2);
            }
          }
        }
        ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 1);  // chunk c: scatter-poll done

        if (threadIdx.x < static_cast<unsigned>(npes)) {
          const int pe = static_cast<int>(threadIdx.x);
          s_pe_ptrs[pe] = (pe == myPe)
              ? reinterpret_cast<const P*>(input)
                  + static_cast<size_t>(myPe) * packedPerRank + off
              : buf + static_cast<size_t>(pe) * packedPerRank + off;
        }
        __syncthreads();

        {
          size_t cnt = packedChunkPerRank;
          if (off + cnt > packedPerRank) cnt = packedPerRank - off;

          P* __restrict__ myDst =
              buf + static_cast<size_t>(myPe) * packedPerRank + off;

          size_t k = compTid;
          if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                        std::is_same<typename P::type, int32_t>::value) {
            for (; k + compStride < cnt; k += compStride * 2) {
              P acc0 = s_pe_ptrs[0][k];
              P acc1 = s_pe_ptrs[0][k + compStride];
              for (int pe = 1; pe < npes; ++pe) {
                packed_assign_add(acc0, s_pe_ptrs[pe][k]);
                packed_assign_add(acc1, s_pe_ptrs[pe][k + compStride]);
              }
              myDst[k] = acc0;
              myDst[k + compStride] = acc1;
            }
            if (k < cnt) {
              P acc = s_pe_ptrs[0][k];
              for (int pe = 1; pe < npes; ++pe) {
                packed_assign_add(acc, s_pe_ptrs[pe][k]);
              }
              myDst[k] = acc;
            }
          } else {
            for (; k + compStride < cnt; k += compStride * 2) {
              A acc0 = upcast_v<typename P::type, pack_size>(s_pe_ptrs[0][k]);
              A acc1 = upcast_v<typename P::type, pack_size>(
                  s_pe_ptrs[0][k + compStride]);
              for (int pe = 1; pe < npes; ++pe) {
                packed_assign_add(
                    acc0,
                    upcast_v<typename P::type, pack_size>(s_pe_ptrs[pe][k]));
                packed_assign_add(
                    acc1,
                    upcast_v<typename P::type, pack_size>(
                        s_pe_ptrs[pe][k + compStride]));
              }
              myDst[k] = downcast_v<typename P::type, pack_size>(acc0);
              myDst[k + compStride] =
                  downcast_v<typename P::type, pack_size>(acc1);
            }
            if (k < cnt) {
              A acc = upcast_v<typename P::type, pack_size>(s_pe_ptrs[0][k]);
              for (int pe = 1; pe < npes; ++pe) {
                packed_assign_add(
                    acc,
                    upcast_v<typename P::type, pack_size>(s_pe_ptrs[pe][k]));
              }
              myDst[k] = downcast_v<typename P::type, pack_size>(acc);
            }
          }
        }

        __syncthreads();
        ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 2);  // chunk c: reduce done
      }

      if (threadIdx.x == 0) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        asm volatile("buffer_wbl2" ::: "memory");
#endif
        __threadfence();
        __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }

      if (post_ag_flag != nullptr) {
        if (threadIdx.x == 0) {
          while (__hip_atomic_load(
                     post_ag_flag,
                     __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0u) {
            __builtin_amdgcn_s_sleep(2);
          }
        }
        __syncthreads();
        ar_write_phase_ts_cb1(phase_ts, 31);  // post_ag_flag observed
      }

      ar_write_phase_ts_cb1(phase_ts, 11 + 3 * numChunks);  // compute block exit

    } else {
      // =================================================================
      // BLOCK 0 — Burst-Submit Scatter + AG
      // =================================================================
      const int thr = static_cast<int>(threadIdx.x);
      const uint32_t ccBase = s_cc_base;

      if constexpr (!EXTERNAL_SCATTER) {
        if (thr < npes && thr != myPe) {
          const int destPe = thr;
          anvil::SdmaQueueDeviceHandle** dh =
              dstMemObj->deviceHandles_d + destPe * numQ;
          HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
              + static_cast<size_t>(myPe) * numQ;
          for (int c = 0; c < numChunks; c++) {
            const size_t cOff = static_cast<size_t>(c) * chunkBytes;
            size_t actualBytes = chunkBytes;
            if (cOff + actualBytes > totalShardBytes)
              actualBytes = totalShardBytes - cOff;
            if (actualBytes > 0) {
              uint8_t* src = reinterpret_cast<uint8_t*>(const_cast<T*>(input))
                  + static_cast<size_t>(destPe) * totalShardBytes + cOff;
              uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
                  + static_cast<size_t>(myPe) * totalShardBytes + cOff;
              core::SdmaPutThread(src, dst, actualBytes, dh, rSig, numQ, 0);
            }
          }
        }
      }
      ar_write_phase_ts(phase_ts, 1);  // phase 1: scatter submit done (block 0 thr 0)

      // ==============================================================
      // Per-chunk flow: for each chunk c
      //   1. wait local reduce(c) complete  (cc_wait)
      //   2. cross-PE barrier: wait ALL peers reduce(c) complete (flags poll)
      //   3. SDMA AG(c)
      //
      // Why cross-PE barrier is REQUIRED:
      //   My AG writes to peer's transit[myPe-slot] via SDMA. Peer's reduce
      //   reads transit[all slots including myPe-slot-from-peer's-perspective]
      //   as its input. If I issue AG before peer finishes reduce, peer may
      //   read my AG-overwritten transit slot instead of the original scatter
      //   value — corrupting peer's reduce sum.
      //
      //   This manifests as inplace-2nd-call verification failures when PE
      //   timing differs enough that fast PE's AG arrives at slow PE's
      //   transit before slow PE's reduce reads it.
      //
      //   The barrier uses flagsMemObj (symmetric memory) with a dedicated
      //   counter reduceCompleteBase, separate from qId=0 SDMA scatter signal
      //   to avoid cross-iteration races.
      // ==============================================================

      if constexpr (MULTI_CHUNK) {
        for (int c = 0; c < numChunks; c++) {
          // 1. Wait local compute blocks to finish chunk c reduce.
          const uint32_t ccTarget =
              ccBase + static_cast<uint32_t>((c + 1) * compBlocks);
          if (thr == 0) {
            while (__scoped_atomic_load_n(
                       &barrier->chunks_complete,
                       __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget)
              __builtin_amdgcn_s_sleep(1);
          }
          __syncthreads();
          ar_write_phase_ts(phase_ts, 2 + 3 * c + 0);  // chunk c: compute-wait done

          // 2. Cross-PE reduce_complete barrier: signal + wait all peers.
          //    Required so my AG doesn't overwrite peer's transit slot
          //    before peer's reduce reads it.
          if (thr == 0) {
            HSAuint64* myFlag =
                reinterpret_cast<HSAuint64*>(flagsMemObj->localPtr);
            __hip_atomic_fetch_add(myFlag, 1ULL,
                                   __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
          }
          if (!separateAgBuffer && thr < npes && thr != myPe) {
            const int pe = thr;
            const uint64_t expected =
                reduceCompleteBase + static_cast<uint64_t>(c + 1);
            HSAuint64* remoteFlag =
                reinterpret_cast<HSAuint64*>(flagsMemObj->peerPtrs[pe]);
            while (core::AtomicLoadRelaxed(remoteFlag) < expected)
              __builtin_amdgcn_s_sleep(1);
          }
          ar_write_phase_ts(phase_ts, 2 + 3 * c + 1);  // chunk c: cross-PE barrier done

          // 3. SDMA AG for chunk c.
          if (thr < npes && thr != myPe) {
            const int destPe = thr;
            const application::SymmMemObjPtr agObj =
                separateAgBuffer ? agDstMemObj : dstMemObj;
            anvil::SdmaQueueDeviceHandle** dh =
                agObj->deviceHandles_d + destPe * agNumQ;
            const uint32_t qSpan = (agNumQ > 1)
                ? ((agNumQ - 1u) < 15u ? (agNumQ - 1u) : 15u) : 1u;
            const uint32_t agQ = (multi_q_ag && qSpan > 1)
                ? static_cast<uint32_t>(1 + (c % static_cast<int>(qSpan)))
                : 1u;
            HSAuint64* rSig = agObj->peerSignalPtrs[destPe]
                + static_cast<size_t>(myPe) * agNumQ;

            const size_t cOff = static_cast<size_t>(c) * chunkBytes;
            size_t agBytes = chunkBytes;
            if (cOff + agBytes > totalShardBytes)
              agBytes = totalShardBytes - cOff;

            uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            uint8_t* dst = reinterpret_cast<uint8_t*>(agObj->peerPtrs[destPe])
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            core::SdmaPutThread(src, dst, agBytes, dh, rSig, agNumQ, agQ);
          }
          ar_write_phase_ts(phase_ts, 2 + 3 * c + 2);  // chunk c: AG submit done
        }

        for (int c = 0; c < numChunks; c++) {
          if (thr < npes && thr != myPe) {
            const int sender = thr;
              const uint32_t qSpan = (agNumQ > 1)
                  ? ((agNumQ - 1u) < 15u ? (agNumQ - 1u) : 15u) : 1u;
              const uint32_t agQ = (multi_q_ag && qSpan > 1)
                  ? static_cast<uint32_t>(1 + (c % static_cast<int>(qSpan)))
                  : 1u;
              const uint64_t qBase = (multi_q_ag && agBaseByQ != nullptr)
                  ? agBaseByQ[agQ] : s_ag_by_sender[sender];
              const uint64_t qCount = (multi_q_ag && qSpan > 1)
                  ? static_cast<uint64_t>(c / static_cast<int>(qSpan) + 1)
                  : static_cast<uint64_t>(c + 1);
              const uint64_t expected_c = qBase + qCount;
              const application::SymmMemObjPtr agObj =
                  separateAgBuffer ? agDstMemObj : dstMemObj;
              HSAuint64* sig = agObj->signalPtrs
                  + static_cast<size_t>(sender) * agNumQ + agQ;
            while (core::AtomicLoadRelaxed(sig) < expected_c)
              __builtin_amdgcn_s_sleep(1);
          }
          __syncthreads();
          if (phase_ts != nullptr && threadIdx.x == 0 &&
              (kArPhaseAgDoneBase + c) < kArPhaseTsCapacity) {
            phase_ts[kArPhaseAgDoneBase + c] =
                __builtin_amdgcn_s_memtime();  // chunk c AG done
          }
        }
        ar_write_phase_ts(phase_ts, 2 + 3 * numChunks);  // AG wait done

        if (post_ag_flag != nullptr && threadIdx.x == 0) {
          __threadfence();
          __hip_atomic_store(post_ag_flag, 1u,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
          ar_write_phase_ts(phase_ts, 30);  // post_ag_flag set
        }
      } else {
        // Single-chunk AG. Same barrier requirement as MULTI_CHUNK.
        const uint32_t ccTarget =
            ccBase + static_cast<uint32_t>(compBlocks);
        if (thr == 0) {
          while (__scoped_atomic_load_n(
                     &barrier->chunks_complete,
                     __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget)
            __builtin_amdgcn_s_sleep(1);
        }
        __syncthreads();
        ar_write_phase_ts(phase_ts, 2);  // chunk 0: compute-wait done

        // Cross-PE reduce_complete barrier.
        if (thr == 0) {
          HSAuint64* myFlag =
              reinterpret_cast<HSAuint64*>(flagsMemObj->localPtr);
          __hip_atomic_fetch_add(myFlag, 1ULL,
                                 __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        if (thr < npes && thr != myPe) {
          const int pe = thr;
          const uint64_t expected = reduceCompleteBase + 1ULL;
          HSAuint64* remoteFlag =
              reinterpret_cast<HSAuint64*>(flagsMemObj->peerPtrs[pe]);
          while (core::AtomicLoadRelaxed(remoteFlag) < expected)
            __builtin_amdgcn_s_sleep(1);
        }
        ar_write_phase_ts(phase_ts, 3);  // chunk 0: cross-PE barrier done

        if (thr < npes && thr != myPe) {
          const int destPe = thr;
          anvil::SdmaQueueDeviceHandle** dh =
              dstMemObj->deviceHandles_d + destPe * numQ;
          HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
              + static_cast<size_t>(myPe) * numQ;

          uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
              + static_cast<size_t>(myPe) * totalShardBytes;
          uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
              + static_cast<size_t>(myPe) * totalShardBytes;
          core::SdmaPutThread(src, dst, totalShardBytes, dh, rSig, numQ, 1);
        }
        ar_write_phase_ts(phase_ts, 4);  // chunk 0: AG submit done

        if (thr < npes && thr != myPe) {
          const int sender = thr;
          const uint64_t expected = s_ag_by_sender[sender] + 1ULL;
          HSAuint64* sig = dstMemObj->signalPtrs
              + static_cast<size_t>(sender) * numQ + 1;
          while (core::AtomicLoadRelaxed(sig) < expected)
            __builtin_amdgcn_s_sleep(1);
        }
        ar_write_phase_ts(phase_ts, 5);  // AG wait done (single-chunk, numChunks=1)

        // Stage 1 E' prototype (single-chunk path).
        if (post_ag_flag != nullptr && threadIdx.x == 0) {
          __threadfence();
          __hip_atomic_store(post_ag_flag, 1u,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
          ar_write_phase_ts(phase_ts, 30);  // post_ag_flag set
        }
      }

      ar_write_phase_ts(phase_ts, 3 + 3 * numChunks);  // block 0 exit

    }  // end block 0 vs compute

  } else {
    // =========================================================================
    // SCATTER_MODE = 1 — P2P fused reduce+CU AG (legacy)
    // =========================================================================
    const size_t tid =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

    __shared__ uint32_t s_gen;
    __shared__ uint32_t s_gather_count;
    if (threadIdx.x == 0) {
      s_gen = barrier->flag;
      s_gather_count = 0;
    }
    __syncthreads();

    auto gatherBarrier = [&]() {
      __syncthreads();
      if (threadIdx.x == 0) {
        s_gather_count++;
        if (blockIdx.x != 0) {
          __scoped_atomic_fetch_add(&barrier->ag_sync, 1u,
                                    __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
        } else {
          uint32_t target =
              s_gather_count * static_cast<uint32_t>(gridDim.x - 1);
          while (__scoped_atomic_load_n(&barrier->ag_sync,
                                        __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < target)
            __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
    };

    for (int c = 0; c < numChunks; c++) {
      const size_t off = static_cast<size_t>(c) * packedChunkPerRank;
      size_t cnt = packedChunkPerRank;
      if (off + cnt > packedPerRank) cnt = packedPerRank - off;
      P* __restrict__ myDst =
          buf + static_cast<size_t>(myPe) * packedPerRank + off;
      const size_t myStart =
          static_cast<size_t>(myPe) * packedPerRank + off;
      for (size_t k = tid; k < cnt; k += stride) {
        const size_t gK = myStart + k;
        const P* p0 = reinterpret_cast<const P*>(inputSymmObj->peerPtrs[0]);
        A acc = upcast_v<typename P::type, pack_size>(p0[gK]);
        for (int pe = 1; pe < npes; ++pe) {
          const P* pp = reinterpret_cast<const P*>(inputSymmObj->peerPtrs[pe]);
          packed_assign_add(acc, upcast_v<typename P::type, pack_size>(pp[gK]));
        }
        P val = downcast_v<typename P::type, pack_size>(acc);
        myDst[k] = val;
        for (int pe = 0; pe < npes; ++pe) {
          if (pe == myPe) continue;
          reinterpret_cast<P*>(dstMemObj->peerPtrs[pe])
              [static_cast<size_t>(myPe) * packedPerRank + off + k] = val;
        }
      }
      __threadfence_system();
      gatherBarrier();
      if (blockIdx.x == 0) {
        const int pe = static_cast<int>(threadIdx.x);
        if (pe < npes && pe != myPe) {
          HSAuint64* sig = dstMemObj->peerSignalPtrs[pe]
              + static_cast<size_t>(myPe) * numQ + 1;
          __hip_atomic_fetch_add(sig, 1ULL,
                                 __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }

    if (blockIdx.x == 0) {
      const int thr = static_cast<int>(threadIdx.x);
      if (thr < npes - 1) {
        const int sender = thr < myPe ? thr : thr + 1;
        const uint64_t expected =
            s_ag_by_sender[sender] + static_cast<uint64_t>(numChunks);
        HSAuint64* sig = dstMemObj->signalPtrs
            + static_cast<size_t>(sender) * numQ + 1;
        while (core::AtomicLoadRelaxed(sig) < expected)
          __builtin_amdgcn_s_sleep(1);
      }
    }
  }

}

// ============================================================================
// PipelinedXGMIPullKernel — Plan A (2-kernel, strict spec per user design).
//
// Paired with ScatterSdmaOnlyKernel which must be launched FIRST on the same
// stream (SDMA scatter runs on the SDMA engine; this kernel runs on CU only).
//
// Plan A v2 per-chunk flow (two compute groups, pipelined):
//
//   Stage 1 (R-group reduce):
//     R blocks wait scatter signals for chunk c, read local
//     transit[all peer slots, chunk c region], sum in registers, write
//     ONLY to local transit[myPe slot, chunk c region]. (方式 2: no double
//     write to user_output; self-slot copy happens uniformly in Stage 3.)
//
//   Stage 2 (barrier):
//     R blocks: buffer_wbl2 + threadfence_system +
//       fetch_add(chunks_complete)  — makes reduce result visible to
//       peers via XGMI and to block 0 for counting.
//     Block 0: wait chunks_complete ≥ (c+1)*compBlocks; do cross-PE
//       reduce_complete barrier via flagsMemObj; then fetch_add(ag_sync)
//       so compute blocks know chunk c's peer transits are now safe to
//       XGMI-read.
//
//   Stage 3 (AG via CU XGMI pull):
//     A blocks wait ag_sync ≥ c+1. Each block takes slot p =
//     a_id % npes, pos = a_id / npes. Cross-XGMI load peer[p].transit[p slot, chunk c
//     region] → local user_output[p slot, chunk c region]. For p == myPe,
//     peerPtrs[myPe] == localPtr so the same code path is a local-to-local
//     copy (reduce just wrote that region so it's L2-hot).
//
// R-group can reduce chunk c+1 while A-group pulls chunk c, restoring the
// pipeline that the first Plan A implementation accidentally serialized.
//
// Compared to PipelinedAllReduceSdmaKernel<T,0,true,true> (baseline
// EXTERNAL_SCATTER + MULTI_CHUNK SDMA path): stages 1 + 2 are identical
// except no per-chunk SDMA AG submit/wait; stage 3 replaces SDMA AG with
// CU XGMI pull + direct write to user_output, eliminating the external
// hipMemcpyAsync (~0.35 ms/AR per perf_history Entry 16).
// ============================================================================
template <typename T>
__global__ void PipelinedXGMIPullKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,     // transit (symm)
    const application::SymmMemObjPtr flagsMemObj,   // cross-PE reduce_complete flag
    CrossPeBarrier* __restrict__ barrier,           // chunks_complete + ag_sync
    T* __restrict__ user_output,                    // user-provided output tensor
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t scatterBase,
    uint64_t agBase,               // unused in Plan A (kept for API symmetry)
    uint64_t reduceCompleteBase,
    int nR_requested,
    uint64_t* phase_ts) {

  (void)agBase;

  ar_write_phase_ts(phase_ts, 0);

  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t chunkPerRank =
      ((chunkElementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t packedChunkPerRank = chunkPerRank / pack_size;
  if (elementCountPerRank == 0 || chunkPerRank == 0) return;

  const int numChunks =
      static_cast<int>((packedPerRank + packedChunkPerRank - 1) / packedChunkPerRank);
  const size_t chunkBytes = chunkPerRank * bytesPerElement;
  const size_t totalShardBytes = elementCountPerRank * bytesPerElement;

  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  const uint32_t numQ = dstMemObj->sdmaNumQueue;
  const int compBlocks = static_cast<int>(gridDim.x) - 1;
  int nR = nR_requested;
  if (nR < 1) nR = 1;
  if (nR > compBlocks - 1) nR = compBlocks - 1;
  const int nA = compBlocks - nR;

  __shared__ uint64_t s_scatter_by_sender[64];
  __shared__ const P* s_pe_ptrs[8];              // peer transit localPtr view for reduce
  __shared__ uintptr_t s_peer_transit_base[8];   // peer transit peerPtrs[p] for AG pull

  if (threadIdx.x < 64) {
    const int s = static_cast<int>(threadIdx.x);
    if (s < npes && s != myPe) {
      s_scatter_by_sender[s] = scatterBase;
    }
    if (s < npes) {
      s_peer_transit_base[s] = dstMemObj->peerPtrs[s];
    }
  }
  __syncthreads();

  // =========================================================================
  // Block 0 — orchestrator: per-chunk cc-wait + cross-PE barrier + ag_sync signal
  // =========================================================================
  if (blockIdx.x == 0) {
    const int thr = static_cast<int>(threadIdx.x);

    for (int c = 0; c < numChunks; c++) {
      // 1. Wait all R-group blocks to finish reduce for chunk c.
      const uint32_t ccTarget =
          static_cast<uint32_t>((c + 1) * nR);
      if (thr == 0) {
        uint64_t stuck = 0;
        while (__scoped_atomic_load_n(
                   &barrier->chunks_complete,
                   __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget) {
          __builtin_amdgcn_s_sleep(1);
          if (++stuck >= 100000000ULL) {
            printf("[STUCK] PE %d K2 b0 chunk %d wait chunks_complete expected=%u got=%u\n",
                   myPe, c, ccTarget,
                   __scoped_atomic_load_n(
                       &barrier->chunks_complete,
                       __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE));
            stuck = 0;
          }
        }
      }
      __syncthreads();
      ar_write_phase_ts(phase_ts, 2 + 3 * c + 0);  // chunk c compute-wait done

      // 2. Cross-PE reduce_complete barrier. Same mechanism as baseline
      //    (PipelinedAllReduceSdmaKernel). Required so my XGMI pull later
      //    reads peer's post-reduce transit[peer slot], not pre-reduce
      //    scatter value.
      if (thr == 0) {
        HSAuint64* myFlag =
            reinterpret_cast<HSAuint64*>(flagsMemObj->localPtr);
        __hip_atomic_fetch_add(myFlag, 1ULL,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      if (thr < npes && thr != myPe) {
        const int pe = thr;
        const uint64_t expected =
            reduceCompleteBase + static_cast<uint64_t>(c + 1);
        HSAuint64* remoteFlag =
            reinterpret_cast<HSAuint64*>(flagsMemObj->peerPtrs[pe]);
        uint64_t stuck = 0;
        while (core::AtomicLoadRelaxed(remoteFlag) < expected) {
          __builtin_amdgcn_s_sleep(1);
          if (++stuck >= 100000000ULL) {
            printf("[STUCK] PE %d K2 b0 thr %d chunk %d cross-PE-barrier peer=%d expected=%llu got=%llu\n",
                   myPe, thr, c, pe,
                   (unsigned long long)expected,
                   (unsigned long long)core::AtomicLoadRelaxed(remoteFlag));
            stuck = 0;
          }
        }
      }
      __syncthreads();
      ar_write_phase_ts(phase_ts, 2 + 3 * c + 1);  // chunk c cross-PE barrier done

      // 3. Signal compute blocks: chunk c's peer transits are now safe
      //    to XGMI-read. ag_sync is zeroed by host before launch.
      if (thr == 0) {
        __threadfence();
        __hip_atomic_fetch_add(&barrier->ag_sync, 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
      ar_write_phase_ts(phase_ts, 2 + 3 * c + 2);  // chunk c ag_sync signaled
    }

    ar_write_phase_ts(phase_ts, 3 + 3 * numChunks);  // block 0 exit
    return;
  }

  // =========================================================================
  // Compute blocks split into two independent groups:
  //   R-group (blockIdx 1..nR) reduces chunk c+1 while
  //   A-group (blockIdx nR+1..gridDim-1) pulls chunk c to user_output.
  // =========================================================================
  const int cb_id = static_cast<int>(blockIdx.x) - 1;
  const bool is_R = cb_id < nR;

  if (is_R) {
    const int r_id = cb_id;
    const size_t rTid =
        static_cast<size_t>(r_id) * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t rStride =
        static_cast<size_t>(nR) * static_cast<size_t>(blockDim.x);

    ar_write_phase_ts_cb1(phase_ts, 10);  // first R-block entry

    for (int c = 0; c < numChunks; c++) {
      ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 0);  // chunk c R loop start
      const size_t off = static_cast<size_t>(c) * packedChunkPerRank;

      // 1. Wait scatter signals for chunk c from all peers.
      if (threadIdx.x < static_cast<unsigned>(npes - 1)) {
        const int idx = static_cast<int>(threadIdx.x);
        const int sender = idx < myPe ? idx : idx + 1;
        const uint64_t expected =
            s_scatter_by_sender[sender] + static_cast<uint64_t>(c + 1);
        HSAuint64* sig = dstMemObj->signalPtrs
            + static_cast<size_t>(sender) * numQ;
        uint64_t stuck = 0;
        while (core::AtomicLoadRelaxed(sig) < expected) {
          __builtin_amdgcn_s_sleep(2);
          if (++stuck >= 50000000ULL) {
            if (blockIdx.x == 1) {
              printf("[STUCK] PE %d K2 R b%d thr %d chunk %d scatter-wait sender=%d expected=%llu got=%llu\n",
                     myPe, (int)blockIdx.x, (int)threadIdx.x, c, sender,
                     (unsigned long long)expected,
                     (unsigned long long)core::AtomicLoadRelaxed(sig));
            }
            stuck = 0;
          }
        }
      }
      ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 1);  // scatter-poll done

      if (threadIdx.x < static_cast<unsigned>(npes)) {
        const int pe = static_cast<int>(threadIdx.x);
        s_pe_ptrs[pe] = (pe == myPe)
            ? reinterpret_cast<const P*>(input)
                + static_cast<size_t>(myPe) * packedPerRank + off
            : buf + static_cast<size_t>(pe) * packedPerRank + off;
      }
      __syncthreads();

      {
        size_t cnt = packedChunkPerRank;
        if (off + cnt > packedPerRank) cnt = packedPerRank - off;
        P* __restrict__ myDst =
            buf + static_cast<size_t>(myPe) * packedPerRank + off;

        size_t k = rTid;
        for (; k + rStride < cnt; k += rStride * 2) {
          A acc0 = upcast_v<typename P::type, pack_size>(s_pe_ptrs[0][k]);
          A acc1 = upcast_v<typename P::type, pack_size>(
              s_pe_ptrs[0][k + rStride]);
          for (int pe = 1; pe < npes; ++pe) {
            packed_assign_add(
                acc0,
                upcast_v<typename P::type, pack_size>(s_pe_ptrs[pe][k]));
            packed_assign_add(
                acc1,
                upcast_v<typename P::type, pack_size>(
                    s_pe_ptrs[pe][k + rStride]));
          }
          myDst[k] = downcast_v<typename P::type, pack_size>(acc0);
          myDst[k + rStride] =
              downcast_v<typename P::type, pack_size>(acc1);
        }
        if (k < cnt) {
          A acc = upcast_v<typename P::type, pack_size>(s_pe_ptrs[0][k]);
          for (int pe = 1; pe < npes; ++pe) {
            packed_assign_add(
                acc,
                upcast_v<typename P::type, pack_size>(s_pe_ptrs[pe][k]));
          }
          myDst[k] = downcast_v<typename P::type, pack_size>(acc);
        }
      }
      __syncthreads();
      ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 2);  // reduce done

      if (threadIdx.x == 0) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        asm volatile("buffer_wbl2" ::: "memory");
#endif
        __threadfence_system();
        __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
      __syncthreads();
    }

    ar_write_phase_ts_cb1(phase_ts, 11 + 3 * numChunks);  // R-block exit
    return;
  }

  // A-group: wait per-chunk ag_ready and pull that chunk to user_output.
  const int a_id = cb_id - nR;
  ar_write_phase_ts_cbA(phase_ts, 49, nR);  // first A-block entry

  for (int c = 0; c < numChunks; c++) {
    ar_write_phase_ts_cbA(phase_ts, 50 + 3 * c + 0, nR);  // A wait start
    if (threadIdx.x == 0) {
      uint64_t stuck = 0;
      while (__scoped_atomic_load_n(
                 &barrier->ag_sync,
                 __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) <
             static_cast<uint32_t>(c + 1)) {
        __builtin_amdgcn_s_sleep(1);
        if (++stuck >= 100000000ULL) {
          if (blockIdx.x == static_cast<unsigned>(nR + 1)) {
            printf("[STUCK] PE %d K2 A b%d chunk %d wait ag_sync expected=%u got=%u\n",
                   myPe, (int)blockIdx.x, c, (uint32_t)(c + 1),
                   __scoped_atomic_load_n(
                       &barrier->ag_sync,
                       __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE));
          }
          stuck = 0;
        }
      }
    }
    __syncthreads();
    ar_write_phase_ts_cbA(phase_ts, 50 + 3 * c + 1, nR);  // A ready

    using vec_t = ulonglong2;  // 16 B vectorized
    const size_t cOff = static_cast<size_t>(c) * chunkBytes;
    size_t agBytes = chunkBytes;
    if (cOff + agBytes > totalShardBytes) agBytes = totalShardBytes - cOff;
    const size_t totalVecs = agBytes / sizeof(vec_t);

    const int slot = a_id % npes;
    const int blocks_per_slot_floor = nA / npes;
    const int extra = nA - blocks_per_slot_floor * npes;
    const int pos_in_slot = a_id / npes;
    const int blocks_on_this_slot = (slot < extra)
        ? blocks_per_slot_floor + 1 : blocks_per_slot_floor;

    if (blocks_on_this_slot > 0 && pos_in_slot < blocks_on_this_slot) {
      const size_t vecsPerBlock =
          (totalVecs + blocks_on_this_slot - 1) / blocks_on_this_slot;
      const size_t vStart =
          static_cast<size_t>(pos_in_slot) * vecsPerBlock;
      const size_t vEnd =
          (vStart + vecsPerBlock > totalVecs) ? totalVecs
                                              : (vStart + vecsPerBlock);

      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(s_peer_transit_base[slot])
          + static_cast<size_t>(slot) * totalShardBytes + cOff;
      uint8_t* dst =
          reinterpret_cast<uint8_t*>(user_output)
          + static_cast<size_t>(slot) * totalShardBytes + cOff;

      const vec_t* sVec = reinterpret_cast<const vec_t*>(src);
      vec_t* dVec = reinterpret_cast<vec_t*>(dst);
      for (size_t v = vStart + threadIdx.x; v < vEnd; v += blockDim.x) {
        dVec[v] = sVec[v];
      }
    }
    __syncthreads();
    ar_write_phase_ts_cbA(phase_ts, 50 + 3 * c + 2, nR);  // A pull done
  }

  ar_write_phase_ts_cbA(phase_ts, 50 + 3 * numChunks, nR);  // A-block exit
}

// ============================================================================
// FullMeshChannelizedAllReduceKernel — experimental drop-in fullmesh pipeline.
//
// One kernel loops over chunks and performs:
//   block0: fullmesh SDMA scatter chunk c -> transit
//   compute blocks: reduce chunk c -> transit[myPe]
//   block0: fullmesh SDMA AG chunk c -> separate internal AG buffer
//   compute blocks: local copy AG buffer chunk c -> user_output
//
// Remote writes never target user_output and never overwrite the scatter/reduce
// transit input while peers may still read it. This is correctness-first and
// intended to test cadence/output semantics before deeper optimization.
// ============================================================================
template <typename T>
__global__ void FullMeshChannelizedAllReduceKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr transitObj,
    const application::SymmMemObjPtr agObj,
    CrossPeBarrier* __restrict__ barrier,
    T* __restrict__ user_output,
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t scatterBase,
    uint64_t agBase) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t chunkPerRank =
      ((chunkElementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t packedChunkPerRank = chunkPerRank / pack_size;
  if (elementCountPerRank == 0 || chunkPerRank == 0) return;

  const int numChunks =
      static_cast<int>((packedPerRank + packedChunkPerRank - 1) / packedChunkPerRank);
  const size_t chunkBytes = chunkPerRank * bytesPerElement;
  const size_t totalShardBytes = elementCountPerRank * bytesPerElement;
  const uint32_t numQ = transitObj->sdmaNumQueue;
  const uint32_t agNumQ = agObj->sdmaNumQueue;
  const int compBlocks = static_cast<int>(gridDim.x) - 1;
  P* __restrict__ transit = reinterpret_cast<P*>(transitObj->localPtr);
  P* __restrict__ agBuf = reinterpret_cast<P*>(agObj->localPtr);
  P* __restrict__ outP = reinterpret_cast<P*>(user_output);

  __shared__ const P* s_pe_ptrs[8];

  if (blockIdx.x == 0) {
    const int thr = static_cast<int>(threadIdx.x);
    for (int c = 0; c < numChunks; ++c) {
      const size_t cOffBytes = static_cast<size_t>(c) * chunkBytes;
      size_t actualBytes = chunkBytes;
      if (cOffBytes + actualBytes > totalShardBytes)
        actualBytes = totalShardBytes - cOffBytes;

      // Fullmesh scatter chunk c to every destination PE.
      if (thr < npes && thr != myPe && actualBytes > 0) {
        const int destPe = thr;
        anvil::SdmaQueueDeviceHandle** dh =
            transitObj->deviceHandles_d + destPe * numQ;
        HSAuint64* rSig = transitObj->peerSignalPtrs[destPe]
            + static_cast<size_t>(myPe) * numQ;
        uint8_t* src = reinterpret_cast<uint8_t*>(const_cast<T*>(input))
            + static_cast<size_t>(destPe) * totalShardBytes + cOffBytes;
        uint8_t* dst = reinterpret_cast<uint8_t*>(transitObj->peerPtrs[destPe])
            + static_cast<size_t>(myPe) * totalShardBytes + cOffBytes;
        core::SdmaPutThread(src, dst, actualBytes, dh, rSig, numQ, 0);
      }
      __syncthreads();

      // Wait incoming scatter chunk c.
      if (thr < npes && thr != myPe) {
        const int sender = thr;
        const uint64_t expected = scatterBase + static_cast<uint64_t>(c + 1);
        HSAuint64* sig = transitObj->signalPtrs + static_cast<size_t>(sender) * numQ;
        while (core::AtomicLoadRelaxed(sig) < expected)
          __builtin_amdgcn_s_sleep(1);
      }
      __syncthreads();

      if (thr == 0) {
        __threadfence();
        __hip_atomic_store(&barrier->flag, static_cast<uint32_t>(c + 1),
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }

      // Wait local reduce completion.
      const uint32_t reduceTarget = static_cast<uint32_t>((c + 1) * compBlocks);
      if (thr == 0) {
        while (__scoped_atomic_load_n(&barrier->chunks_complete,
                                      __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < reduceTarget)
          __builtin_amdgcn_s_sleep(1);
      }
      __syncthreads();

      // Fullmesh AG chunk c to separate AG buffer.
      if (thr < npes && thr != myPe && actualBytes > 0) {
        const int destPe = thr;
        anvil::SdmaQueueDeviceHandle** dh =
            agObj->deviceHandles_d + destPe * agNumQ;
        HSAuint64* rSig = agObj->peerSignalPtrs[destPe]
            + static_cast<size_t>(myPe) * agNumQ + 1;
        uint8_t* src = reinterpret_cast<uint8_t*>(transitObj->localPtr)
            + static_cast<size_t>(myPe) * totalShardBytes + cOffBytes;
        uint8_t* dst = reinterpret_cast<uint8_t*>(agObj->peerPtrs[destPe])
            + static_cast<size_t>(myPe) * totalShardBytes + cOffBytes;
        core::SdmaPutThread(src, dst, actualBytes, dh, rSig, agNumQ, 1);
      }
      __syncthreads();

      // Wait incoming AG chunk c.
      if (thr < npes && thr != myPe) {
        const int sender = thr;
        const uint64_t expected = agBase + static_cast<uint64_t>(c + 1);
        HSAuint64* sig = agObj->signalPtrs + static_cast<size_t>(sender) * agNumQ + 1;
        while (core::AtomicLoadRelaxed(sig) < expected)
          __builtin_amdgcn_s_sleep(1);
      }
      __syncthreads();

      if (thr == 0) {
        __threadfence();
        __hip_atomic_store(&barrier->ag_sync, static_cast<uint32_t>(c + 1),
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }

      // Wait local copy completion before reusing this chunk in later logic.
      const uint32_t copyTarget = static_cast<uint32_t>((c + 1) * compBlocks);
      if (thr == 0) {
        while (__scoped_atomic_load_n(&barrier->ag_gate,
                                      __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < copyTarget)
          __builtin_amdgcn_s_sleep(1);
      }
      __syncthreads();
    }
    return;
  }

  const size_t compTid =
      static_cast<size_t>(blockIdx.x - 1) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t compStride =
      static_cast<size_t>(compBlocks) * static_cast<size_t>(blockDim.x);

  for (int c = 0; c < numChunks; ++c) {
    while (__hip_atomic_load(&barrier->flag, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) < static_cast<uint32_t>(c + 1))
      __builtin_amdgcn_s_sleep(1);

    const size_t off = static_cast<size_t>(c) * packedChunkPerRank;
    size_t cnt = packedChunkPerRank;
    if (off + cnt > packedPerRank) cnt = packedPerRank - off;

    if (threadIdx.x < static_cast<unsigned>(npes)) {
      const int pe = static_cast<int>(threadIdx.x);
      s_pe_ptrs[pe] = (pe == myPe)
          ? reinterpret_cast<const P*>(input) + static_cast<size_t>(myPe) * packedPerRank + off
          : transit + static_cast<size_t>(pe) * packedPerRank + off;
    }
    __syncthreads();

    P* __restrict__ myDst = transit + static_cast<size_t>(myPe) * packedPerRank + off;
    size_t k = compTid;
    if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                  std::is_same<typename P::type, int32_t>::value) {
      for (; k < cnt; k += compStride) {
        P acc = s_pe_ptrs[0][k];
        for (int pe = 1; pe < npes; ++pe) packed_assign_add(acc, s_pe_ptrs[pe][k]);
        myDst[k] = acc;
      }
    } else {
      for (; k < cnt; k += compStride) {
        A acc = upcast_v<typename P::type, pack_size>(s_pe_ptrs[0][k]);
        for (int pe = 1; pe < npes; ++pe)
          packed_assign_add(acc, upcast_v<typename P::type, pack_size>(s_pe_ptrs[pe][k]));
        myDst[k] = downcast_v<typename P::type, pack_size>(acc);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
      asm volatile("buffer_wbl2" ::: "memory");
#endif
      __threadfence_system();
      __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();

    while (__hip_atomic_load(&barrier->ag_sync, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) < static_cast<uint32_t>(c + 1))
      __builtin_amdgcn_s_sleep(1);

    const size_t totalCnt = static_cast<size_t>(npes) * cnt;
    for (size_t linear = compTid; linear < totalCnt; linear += compStride) {
      const size_t pe = linear / cnt;
      const size_t elem = linear - pe * cnt;
      outP[pe * packedPerRank + off + elem] =
          agBuf[pe * packedPerRank + off + elem];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(&barrier->ag_gate, 1u,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
  }
}

// ============================================================================
// PipelinedSdmaAgCopyKernel — copy-mode pipeline using SDMA AG + local copy.
//
// Paired with ScatterSdmaOnlyKernel (K1). K2 splits compute blocks into:
//   block0: wait R-group reduce(c), SDMA AG(c) to separate agObj, wait incoming
//           AG(c), then signal ag_sync=c+1
//   R-group: reduce chunk c from transit into transit[myPe]
//   C-group: wait ag_sync(c), copy agObj[all slots, c] -> user_output[all slots, c]
//
// This keeps remote writes off user_output and avoids a final monolithic
// transit->user_output copy by pipelining local output writes with later reduce.
// ============================================================================
template <typename T>
__global__ void PipelinedSdmaAgCopyKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr transitObj,
    const application::SymmMemObjPtr agObj,
    CrossPeBarrier* __restrict__ barrier,
    T* __restrict__ user_output,
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t agBase,
    int nR_requested) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t chunkPerRank =
      ((chunkElementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t packedChunkPerRank = chunkPerRank / pack_size;
  if (elementCountPerRank == 0 || chunkPerRank == 0) return;

  const int numChunks =
      static_cast<int>((packedPerRank + packedChunkPerRank - 1) / packedChunkPerRank);
  const size_t chunkBytes = chunkPerRank * bytesPerElement;
  const size_t totalShardBytes = elementCountPerRank * bytesPerElement;
  const int compBlocks = static_cast<int>(gridDim.x) - 1;
  int nR = nR_requested;
  if (nR < 1) nR = 1;
  if (nR > compBlocks - 1) nR = compBlocks - 1;
  const int nC = compBlocks - nR;

  P* __restrict__ transit = reinterpret_cast<P*>(transitObj->localPtr);
  P* __restrict__ agBuf = reinterpret_cast<P*>(agObj->localPtr);
  P* __restrict__ outP = reinterpret_cast<P*>(user_output);
  const uint32_t agNumQ = agObj->sdmaNumQueue;

  __shared__ const P* s_pe_ptrs[8];

  if (blockIdx.x == 0) {
    const int thr = static_cast<int>(threadIdx.x);
    for (int c = 0; c < numChunks; ++c) {
      const uint32_t reduceTarget = static_cast<uint32_t>((c + 1) * nR);
      if (thr == 0) {
        while (__scoped_atomic_load_n(&barrier->chunks_complete,
                                      __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < reduceTarget)
          __builtin_amdgcn_s_sleep(1);
      }
      __syncthreads();

      const size_t cOffBytes = static_cast<size_t>(c) * chunkBytes;
      size_t agBytes = chunkBytes;
      if (cOffBytes + agBytes > totalShardBytes) agBytes = totalShardBytes - cOffBytes;

      if (thr < npes && thr != myPe && agBytes > 0) {
        const int destPe = thr;
        anvil::SdmaQueueDeviceHandle** dh =
            agObj->deviceHandles_d + destPe * agNumQ;
        HSAuint64* rSig = agObj->peerSignalPtrs[destPe]
            + static_cast<size_t>(myPe) * agNumQ + 1;
        uint8_t* src = reinterpret_cast<uint8_t*>(transitObj->localPtr)
            + static_cast<size_t>(myPe) * totalShardBytes + cOffBytes;
        uint8_t* dst = reinterpret_cast<uint8_t*>(agObj->peerPtrs[destPe])
            + static_cast<size_t>(myPe) * totalShardBytes + cOffBytes;
        core::SdmaPutThread(src, dst, agBytes, dh, rSig, agNumQ, 1);
      }
      __syncthreads();

      if (thr < npes && thr != myPe) {
        const int sender = thr;
        const uint64_t expected = agBase + static_cast<uint64_t>(c + 1);
        HSAuint64* sig = agObj->signalPtrs + static_cast<size_t>(sender) * agNumQ + 1;
        while (core::AtomicLoadRelaxed(sig) < expected)
          __builtin_amdgcn_s_sleep(1);
      }
      __syncthreads();
      if (thr == 0) {
        __threadfence();
        __hip_atomic_store(&barrier->ag_sync, static_cast<uint32_t>(c + 1),
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    return;
  }

  const int cb_id = static_cast<int>(blockIdx.x) - 1;
  const bool is_R = cb_id < nR;
  if (is_R) {
    const size_t rTid =
        static_cast<size_t>(cb_id) * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t rStride = static_cast<size_t>(nR) * static_cast<size_t>(blockDim.x);
    for (int c = 0; c < numChunks; ++c) {
      const size_t off = static_cast<size_t>(c) * packedChunkPerRank;
      size_t cnt = packedChunkPerRank;
      if (off + cnt > packedPerRank) cnt = packedPerRank - off;
      if (threadIdx.x < static_cast<unsigned>(npes)) {
        const int pe = static_cast<int>(threadIdx.x);
        s_pe_ptrs[pe] = (pe == myPe)
            ? reinterpret_cast<const P*>(input) + static_cast<size_t>(myPe) * packedPerRank + off
            : transit + static_cast<size_t>(pe) * packedPerRank + off;
      }
      __syncthreads();
      P* __restrict__ myDst = transit + static_cast<size_t>(myPe) * packedPerRank + off;
      for (size_t k = rTid; k < cnt; k += rStride) {
        if constexpr (std::is_same<typename P::type, uint32_t>::value ||
                      std::is_same<typename P::type, int32_t>::value) {
          P acc = s_pe_ptrs[0][k];
          for (int pe = 1; pe < npes; ++pe) packed_assign_add(acc, s_pe_ptrs[pe][k]);
          myDst[k] = acc;
        } else {
          A acc = upcast_v<typename P::type, pack_size>(s_pe_ptrs[0][k]);
          for (int pe = 1; pe < npes; ++pe)
            packed_assign_add(acc, upcast_v<typename P::type, pack_size>(s_pe_ptrs[pe][k]));
          myDst[k] = downcast_v<typename P::type, pack_size>(acc);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        asm volatile("buffer_wbl2" ::: "memory");
#endif
        __threadfence_system();
        __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
      __syncthreads();
    }
    return;
  }

  const int c_id = cb_id - nR;
  for (int c = 0; c < numChunks; ++c) {
    if (threadIdx.x == 0) {
      while (__hip_atomic_load(&barrier->ag_sync, __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) < static_cast<uint32_t>(c + 1))
        __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();

    const size_t off = static_cast<size_t>(c) * packedChunkPerRank;
    size_t cnt = packedChunkPerRank;
    if (off + cnt > packedPerRank) cnt = packedPerRank - off;
    const size_t totalCnt = static_cast<size_t>(npes) * cnt;
    const size_t cTid =
        static_cast<size_t>(c_id) * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t cStride = static_cast<size_t>(nC) * static_cast<size_t>(blockDim.x);
    for (size_t linear = cTid; linear < totalCnt; linear += cStride) {
      const size_t pe = linear / cnt;
      const size_t elem = linear - pe * cnt;
      outP[pe * packedPerRank + off + elem] =
          agBuf[pe * packedPerRank + off + elem];
    }
    __syncthreads();
  }
}

}  // namespace collective
}  // namespace mori

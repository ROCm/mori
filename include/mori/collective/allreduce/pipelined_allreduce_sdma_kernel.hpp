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
//   20 + c: block 0 observed chunk c's AG completion (MULTI_CHUNK path)
//   --- post-AG wait instrumentation (E' prototype) ---
//   30: block 0 post_ag_flag set (after AG wait done, before block 0 exit)
//   31: compute block 1 post_ag_flag observed (spin-wait finished)
static constexpr int kArPhaseTsCapacity = 32;

__device__ inline void ar_write_phase_ts(uint64_t* ts, int idx) {
  if (ts != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
    ts[idx] = __builtin_amdgcn_s_memtime();
  }
}

// Same as ar_write_phase_ts but only the first compute block (block 1) writes.
// Used to measure compute-block-internal phases (dispatch latency, scatter-poll
// wait, reduce execution, fetch_add). Block 1 slots use idx 10..17:
//   10: compute block entry
//   11 + 3c + {0,1,2}: chunk c {loop-start, scatter-poll done, reduce done}
//   11 + 3*numChunks: compute block exit (after final fetch_add)
__device__ inline void ar_write_phase_ts_cb1(uint64_t* ts, int idx) {
  if (ts != nullptr && blockIdx.x == 1 && threadIdx.x == 0) {
    ts[idx] = __builtin_amdgcn_s_memtime();
  }
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
    while (core::AtomicLoadRelaxed(sig) < expected)
      __builtin_amdgcn_s_sleep(1);
  }
}

template <typename T, int SCATTER_MODE = 0, bool MULTI_CHUNK = false,
          bool EXTERNAL_SCATTER = false>
__global__ void PipelinedAllReduceSdmaKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr flagsMemObj,
    CrossPeBarrier* __restrict__ barrier,
    const application::SymmMemObjPtr inputSymmObj,
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t scatterBase,
    uint64_t agBase,
    uint64_t reduceCompleteBase,
    uint64_t* phase_ts,
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
  const uint32_t numQ = dstMemObj->sdmaNumQueue;
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
          if (thr < npes && thr != myPe) {
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
            anvil::SdmaQueueDeviceHandle** dh =
                dstMemObj->deviceHandles_d + destPe * numQ;
            HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
                + static_cast<size_t>(myPe) * numQ;

            const size_t cOff = static_cast<size_t>(c) * chunkBytes;
            size_t agBytes = chunkBytes;
            if (cOff + agBytes > totalShardBytes)
              agBytes = totalShardBytes - cOff;

            uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            core::SdmaPutThread(src, dst, agBytes, dh, rSig, numQ, 1);
          }
          ar_write_phase_ts(phase_ts, 2 + 3 * c + 2);  // chunk c: AG submit done
        }

        for (int c = 0; c < numChunks; c++) {
          if (thr < npes && thr != myPe) {
            const int sender = thr;
            const uint64_t expected_c =
                s_ag_by_sender[sender] + static_cast<uint64_t>(c + 1);
            HSAuint64* sig = dstMemObj->signalPtrs
                + static_cast<size_t>(sender) * numQ + 1;
            while (core::AtomicLoadRelaxed(sig) < expected_c)
              __builtin_amdgcn_s_sleep(1);
          }
          __syncthreads();
          if (phase_ts != nullptr && threadIdx.x == 0 && (20 + c) < kArPhaseTsCapacity) {
            phase_ts[20 + c] = __builtin_amdgcn_s_memtime();  // chunk c AG done
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
// PipelinedCuAgKernel — "Plan A / direct-output" slim kernel (perf_history
// Entry 18). NO SDMA AG code at all — relies on a preceding
// ScatterSdmaOnlyKernel to have already started the SDMA scatter phase.
//
// Compiler-wise, this is a SEPARATE function from PipelinedAllReduceSdmaKernel
// so the scalar / vector register budget is dedicated to reduce + CU XGMI AG
// code. The single-kernel approach (plan A v1) added the CU XGMI AG inline
// and the compiler's register allocator serialized previously parallel reduce
// ops, making reduce 4× slower. This separate kernel avoids that.
//
// Phases (all in this one kernel, but CU-only; SDMA scatter runs in parallel
// on the SDMA engine from the preceding kernel):
//   1. Compute blocks: poll peer scatter signals; reduce chunk c → transit
//      [myPe slot]; fetch_add chunks_complete.
//   2. Block 0: wait chunks_complete ≥ target; cross-PE reduce barrier
//      (fetch_add myFlag + wait peer flags). No SDMA AG.
//   3. Block 0: set post_ag_flag = 1 (signals compute blocks that all peers
//      have finished their reduce → safe to XGMI load peer transits).
//   4. Compute blocks: after post_ag_flag observed, do CU XGMI AG — each
//      block assigned to one peer (kBppA=16 per peer; validated 370 GB/s in
//      examples/sdma/cu_xgmi_bench), read peer_transit[peer_slot] via P2P
//      pointer and write directly into user_output[peer_slot]. For
//      peer==myPe, read local transit[myPe_slot] and copy locally.
// ============================================================================
template <typename T>
__global__ void PipelinedCuAgKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr flagsMemObj,
    CrossPeBarrier* __restrict__ barrier,
    T* __restrict__ user_output,
    size_t elementCount,
    size_t chunkElementCount,
    uint64_t scatterBase,
    uint64_t reduceCompleteBase,
    uint64_t* phase_ts,
    uint32_t* post_ag_flag) {

  ar_write_phase_ts(phase_ts, 0);  // entry

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
  const size_t totalShardBytes = elementCountPerRank * bytesPerElement;

  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  const uint32_t numQ = dstMemObj->sdmaNumQueue;
  const int compBlocks = static_cast<int>(gridDim.x) - 1;

  __shared__ uint64_t s_scatter_by_sender[64];
  __shared__ uint32_t s_cc_base;
  __shared__ const P* s_pe_ptrs[8];

  if (threadIdx.x == 0) s_cc_base = 0u;
  {
    const int s = static_cast<int>(threadIdx.x);
    if (s < npes && s != myPe) {
      s_scatter_by_sender[s] = scatterBase;
    }
  }
  __syncthreads();

  if (blockIdx.x != 0) {
    // ---- Compute blocks: scatter-poll → reduce → chunks_complete ----
    ar_write_phase_ts_cb1(phase_ts, 10);

    const size_t compTid =
        static_cast<size_t>(blockIdx.x - 1) * static_cast<size_t>(blockDim.x)
        + threadIdx.x;
    const size_t compStride =
        static_cast<size_t>(compBlocks) * static_cast<size_t>(blockDim.x);

    for (int c = 0; c < numChunks; c++) {
      ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 0);
      const size_t off = static_cast<size_t>(c) * packedChunkPerRank;

      if (c > 0 && threadIdx.x == 64) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        asm volatile("buffer_wbl2" ::: "memory");
#endif
        __threadfence();
        __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
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
      ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 1);

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

      __syncthreads();
      ar_write_phase_ts_cb1(phase_ts, 11 + 3 * c + 2);
    }

    if (threadIdx.x == 0) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
      asm volatile("buffer_wbl2" ::: "memory");
#endif
      __threadfence();
      __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }

    // Wait for post_ag_flag (cross-PE reduce barrier done signal from block 0).
    if (post_ag_flag != nullptr) {
      if (threadIdx.x == 0) {
        while (__hip_atomic_load(
                   post_ag_flag,
                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0u) {
          __builtin_amdgcn_s_sleep(2);
        }
      }
      __syncthreads();
      ar_write_phase_ts_cb1(phase_ts, 31);
    }

    // ---- Phase 4: CU XGMI AG + direct write user_output ----
    // Each peer gets bpp blocks (bounded above by 16, the bench-optimal
    // saturation point per cu_xgmi_bench @ 32 MB shard = 370 GB/s).
    // Adaptive for small compBlocks so every peer gets at least 1 worker.
    if (compBlocks > 0 && user_output != nullptr) {
      int bpp = compBlocks / npes;
      if (bpp > 16) bpp = 16;
      if (bpp < 1)  bpp = 1;
      const int totalWorkers = npes * bpp;
      const int cbId = static_cast<int>(blockIdx.x) - 1;
      const int peer = (cbId < totalWorkers) ? (cbId / bpp) : npes;  // peer==npes → idle
      const int posInPeer = (cbId < totalWorkers) ? (cbId % bpp) : 0;

      if (peer < npes) {
        using vec_t = ulonglong2;  // 16 B
        static_assert(sizeof(vec_t) == 16, "vec_t must be 16B");

        const size_t peerOffset =
            static_cast<size_t>(peer) * totalShardBytes;
        const uintptr_t srcBaseAddr = (peer == myPe)
            ? reinterpret_cast<uintptr_t>(dstMemObj->localPtr)
            : dstMemObj->peerPtrs[peer];
        const uint8_t* srcBase =
            reinterpret_cast<const uint8_t*>(srcBaseAddr) + peerOffset;
        uint8_t* dstBase =
            reinterpret_cast<uint8_t*>(user_output) + peerOffset;

        const size_t totalVecs = totalShardBytes / sizeof(vec_t);
        const size_t vecsPerBlock =
            (totalVecs + bpp - 1) / bpp;
        const size_t vStart =
            static_cast<size_t>(posInPeer) * vecsPerBlock;
        const size_t vEnd =
            (vStart + vecsPerBlock > totalVecs) ? totalVecs
                                                 : (vStart + vecsPerBlock);

        const vec_t* src = reinterpret_cast<const vec_t*>(srcBase);
        vec_t* dst = reinterpret_cast<vec_t*>(dstBase);

        for (size_t v = vStart + threadIdx.x; v < vEnd; v += blockDim.x) {
          dst[v] = src[v];
        }
        const size_t tailStart = totalVecs * sizeof(vec_t);
        if (tailStart < totalShardBytes
            && peer == npes - 1 && posInPeer == bpp - 1) {
          for (size_t t = tailStart + threadIdx.x;
               t < totalShardBytes; t += blockDim.x) {
            dstBase[t] = srcBase[t];
          }
        }
        __syncthreads();
      }
    }

    ar_write_phase_ts_cb1(phase_ts, 11 + 3 * numChunks);  // compute block exit

  } else {
    // ---- Block 0: wait compute blocks + cross-PE reduce barrier ----
    const int thr = static_cast<int>(threadIdx.x);
    const uint32_t ccBase = s_cc_base;

    ar_write_phase_ts(phase_ts, 1);  // placeholder, scatter already done externally

    for (int c = 0; c < numChunks; c++) {
      const uint32_t ccTarget =
          ccBase + static_cast<uint32_t>((c + 1) * compBlocks);
      if (thr == 0) {
        while (__scoped_atomic_load_n(
                   &barrier->chunks_complete,
                   __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget)
          __builtin_amdgcn_s_sleep(1);
      }
      __syncthreads();
      ar_write_phase_ts(phase_ts, 2 + 3 * c + 0);

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
        while (core::AtomicLoadRelaxed(remoteFlag) < expected)
          __builtin_amdgcn_s_sleep(1);
      }
      ar_write_phase_ts(phase_ts, 2 + 3 * c + 1);
      // No SDMA AG submit; leave slot 2+3*c+2 unused
    }
    ar_write_phase_ts(phase_ts, 2 + 3 * numChunks);  // barriers all done

    // Signal compute blocks: safe to XGMI load peer transits.
    if (post_ag_flag != nullptr && threadIdx.x == 0) {
      __threadfence_system();
      __hip_atomic_store(post_ag_flag, 1u,
                         __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      ar_write_phase_ts(phase_ts, 30);
    }

    ar_write_phase_ts(phase_ts, 3 + 3 * numChunks);  // block 0 exit
  }
}

}  // namespace collective
}  // namespace mori

// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce — burst-submit SDMA pipeline with cross-PE barrier.
//
// SCATTER_MODE=0: Burst-Submit Pipeline with Remote-Read AllGather
//   Block 0 (threads 0..npes-1, one per peer):
//     Phase 1: burst-submit ALL scatter packets (all chunks) to SDMA queues.
//              Scatter uses SDMA write → remote (outbound XGMI).
//     Phase 2: per-thread AG loop (zero __syncthreads in hot path):
//              poll chunks_complete → signal_q2 → rd_wait → AG submit.
//              AG uses SDMA read ← remote (inbound XGMI), enabling true
//              bidirectional pipeline overlap with scatter.
//     Phase 3: final AG wait.
//
//   Compute blocks (1..N):
//     scatter-poll → reduce → wbl2+fence → atomic_add(chunks_complete).
//
//   qId=0 scatter (outbound), qId=1 AG (inbound), qId=2 reduce-done.
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

// Burst-submit pipeline: Block 0 only uses threads 0..npes-1 (one per peer).
// kMaxPipelineBlocks still caps compute blocks for register pressure.
static_assert(kMaxPipelineBlocks <= 385,
              "compute block count must fit in grid launch");

template <typename T, int SCATTER_MODE = 0>
__global__ void PipelinedAllReduceSdmaKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr flagsMemObj,
    CrossPeBarrier* __restrict__ barrier,
    const application::SymmMemObjPtr inputSymmObj,
    size_t elementCount,
    size_t chunkElementCount) {

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

  // ---- Signal baselines (per-sender, requires sdmaNumQueue >= 3) ----
  __shared__ uint64_t s_scatter_by_sender[64];
  __shared__ uint64_t s_ag_by_sender[64];
  __shared__ uint64_t s_rd_by_sender[64];
  __shared__ uint32_t s_cc_base;

  if (threadIdx.x == 0) {
    s_cc_base = (blockIdx.x == 0)
        ? __scoped_atomic_load_n(
              &barrier->chunks_complete, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
        : 0u;
    for (int s = 0; s < npes && s < 64; ++s) {
      if (s == myPe) continue;
      const size_t row = static_cast<size_t>(s) * numQ;
      s_scatter_by_sender[s] =
          core::AtomicLoadRelaxed(dstMemObj->signalPtrs + row + 0);
      if (blockIdx.x == 0) {
        s_ag_by_sender[s] =
            core::AtomicLoadRelaxed(dstMemObj->signalPtrs + row + 1);
        s_rd_by_sender[s] =
            core::AtomicLoadRelaxed(dstMemObj->signalPtrs + row + 2);
      }
    }
  }
  __syncthreads();

  // =========================================================================
  // SCATTER_MODE = 0 — Single-buffer SDMA + qId=2 cross-PE reduce barrier
  // =========================================================================
  if constexpr (SCATTER_MODE == 0) {

    if (numQ < 3) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("PE %d: pipelined SDMA needs sdmaNumQueue>=3 (got %u)\n",
               myPe, numQ);
      }
      return;
    }

    if (blockIdx.x != 0) {
      // =================================================================
      // COMPUTE BLOCKS (1..N): scatter-poll → reduce → chunks_complete
      // =================================================================
      const size_t compTid =
          static_cast<size_t>(blockIdx.x - 1) * static_cast<size_t>(blockDim.x)
          + threadIdx.x;
      const size_t compStride =
          static_cast<size_t>(compBlocks) * static_cast<size_t>(blockDim.x);

      for (int c = 0; c < numChunks; c++) {
        if (threadIdx.x < static_cast<unsigned>(npes - 1)) {
          const int idx = static_cast<int>(threadIdx.x);
          const int sender = idx < myPe ? idx : idx + 1;
          const uint64_t expected =
              s_scatter_by_sender[sender] + static_cast<uint64_t>(c + 1);
          HSAuint64* sig = dstMemObj->signalPtrs
              + static_cast<size_t>(sender) * numQ;
          while (core::AtomicLoadRelaxed(sig) < expected)
            ;
        }
        __syncthreads();

        {
          const size_t off = static_cast<size_t>(c) * packedChunkPerRank;
          size_t cnt = packedChunkPerRank;
          if (off + cnt > packedPerRank) cnt = packedPerRank - off;

          P* __restrict__ myDst =
              buf + static_cast<size_t>(myPe) * packedPerRank + off;
          const P* __restrict__ myInput =
              reinterpret_cast<const P*>(input)
              + static_cast<size_t>(myPe) * packedPerRank + off;

          for (size_t k = compTid; k < cnt; k += compStride) {
            A acc = upcast_v<typename P::type, pack_size>(myInput[k]);
            for (int pe = 0; pe < npes; ++pe) {
              if (pe == myPe) continue;
              packed_assign_add(
                  acc,
                  upcast_v<typename P::type, pack_size>(
                      buf[static_cast<size_t>(pe) * packedPerRank + off + k]));
            }
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
      }

    } else {
      // =================================================================
      // BLOCK 0 — Burst-Submit Pipeline (zero __syncthreads in hot path)
      //
      // Phase 1: burst-submit ALL scatter packets to SDMA queues at once.
      //          Eliminates SDMA idle gaps between per-chunk submissions.
      // Phase 2: per-thread AG loop — each thread handles one peer:
      //          poll chunks_complete → signal_q2 → rd_wait → AG submit.
      //          Threads in the same wavefront naturally synchronize on
      //          the shared chunks_complete counter; no __syncthreads.
      // Phase 3: final AG wait.
      // =================================================================
      const int thr = static_cast<int>(threadIdx.x);
      const uint32_t ccBase = s_cc_base;

      // ---- Phase 1: burst scatter (all chunks, all peers) ----
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

      // ---- Phase 2: per-thread AG loop (no __syncthreads) ----
      // Remote-read AG: pull each peer's reduced shard to local buffer.
      // Scatter → outbound XGMI, AG ← inbound XGMI: no bandwidth contention.
      if (thr < npes && thr != myPe) {
        const int peer = thr;
        anvil::SdmaQueueDeviceHandle** dh =
            dstMemObj->deviceHandles_d + peer * numQ;

        for (int c = 0; c < numChunks; c++) {
          // Wait for ALL compute blocks to finish reduce(c)
          const uint32_t ccTarget =
              ccBase + static_cast<uint32_t>((c + 1) * compBlocks);
          while (__scoped_atomic_load_n(
                     &barrier->chunks_complete,
                     __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget)
            ;

          // Signal remote PE: our reduce(c) is done (they can pull our data)
          HSAuint64* rs = dstMemObj->peerSignalPtrs[peer]
              + static_cast<size_t>(myPe) * numQ + 2;
          __hip_atomic_fetch_add(rs, 1ULL,
              __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

          // Wait for remote PE's reduce(c) done before pulling their data
          const uint64_t rdNeed =
              s_rd_by_sender[peer] + static_cast<uint64_t>(c + 1);
          HSAuint64* rdSig = dstMemObj->signalPtrs
              + static_cast<size_t>(peer) * numQ + 2;
          while (core::AtomicLoadRelaxed(rdSig) < rdNeed)
            ;

          // Pull AG(c): SDMA read from remote peer's reduced shard to local.
          // src = peer's transit buffer (peer's shard), dst = our local buffer.
          // Signal goes to OUR signalPtrs (local atomic, faster detection).
          const size_t cOff = static_cast<size_t>(c) * chunkBytes;
          size_t actualBytes = chunkBytes;
          if (cOff + actualBytes > totalShardBytes)
            actualBytes = totalShardBytes - cOff;
          if (actualBytes > 0) {
            uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[peer])
                + static_cast<size_t>(peer) * totalShardBytes + cOff;
            uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                + static_cast<size_t>(peer) * totalShardBytes + cOff;
            HSAuint64* localSig = dstMemObj->signalPtrs
                + static_cast<size_t>(peer) * numQ;
            core::SdmaPutThread(src, dst, actualBytes, dh, localSig, numQ, 1);
          }
        }
      }

      // ---- Phase 3: final AG wait ----
      if (thr < npes && thr != myPe) {
        const int sender = thr;
        const uint64_t expected =
            s_ag_by_sender[sender] + static_cast<uint64_t>(numChunks);
        HSAuint64* sig = dstMemObj->signalPtrs
            + static_cast<size_t>(sender) * numQ + 1;
        int spin = 0;
        while (core::AtomicLoadRelaxed(sig) < expected) {
          if (++spin > 100000000) {
            printf("PE %d: final AG timeout peer=%d exp=%llu act=%llu\n",
                   myPe, sender,
                   (unsigned long long)expected,
                   (unsigned long long)core::AtomicLoadRelaxed(sig));
            break;
          }
        }
      }

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
            ;
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
        int spin = 0;
        while (core::AtomicLoadRelaxed(sig) < expected) {
          if (++spin > 100000000) {
            printf("PE %d: final AG timeout peer=%d exp=%llu act=%llu\n",
                   myPe, sender,
                   (unsigned long long)expected,
                   (unsigned long long)core::AtomicLoadRelaxed(sig));
            break;
          }
        }
      }
    }
  }

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

}  // namespace collective
}  // namespace mori

// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce — SDMA scatter + reduce + SDMA AllGather.
//
// sdma-total signal architecture: SdmaPutThread (8 args) always increments
// both the SDMA atomic target (signals[qId]) and local expectedSignals[qId].
// The serial kernel uses q=0 for outgoing-completion tracking.  To avoid
// corrupting the serial kernel's signal counters, the pipeline uses q=1
// exclusively for both scatter and AG.  Because every PE sends the same
// number of scatter+AG operations (symmetric), the incoming SDMA atomics
// on q=1 always balance the outgoing kernel-store increments, keeping
// SdmaQueitThread's exact-equality invariant satisfied.
//
// SCATTER_MODE=0: scatter + AG both on queue 1 (queue 0 reserved for serial).
// SCATTER_MODE=1: P2P read + CU AG (legacy).
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

template <typename T, int SCATTER_MODE = 0, bool MULTI_CHUNK = false>
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

  // ---- Per-sender signal baselines (all blocks snapshot signalPtrs) ----
  __shared__ uint64_t s_scatter_by_sender[64];
  __shared__ uint64_t s_ag_by_sender[64];
  __shared__ uint32_t s_cc_base;
  __shared__ const P* s_pe_ptrs[8];

  if (threadIdx.x == 0) {
    s_cc_base = (blockIdx.x == 0)
        ? __scoped_atomic_load_n(
              &barrier->chunks_complete, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
        : 0u;
  }
  {
    const int s = static_cast<int>(threadIdx.x);
    if (s < npes && s != myPe) {
      const size_t row = static_cast<size_t>(s) * numQ;
      // Both scatter and AG use q=1; read the single shared baseline.
      s_scatter_by_sender[s] =
          core::AtomicLoadRelaxed(dstMemObj->signalPtrs + row + 1);
      if (blockIdx.x == 0) {
        s_ag_by_sender[s] = s_scatter_by_sender[s];
      }
    }
  }
  __syncthreads();

  // =========================================================================
  // SCATTER_MODE = 0 — SDMA scatter(q1) + reduce + SDMA AG(q1)
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
      // COMPUTE BLOCKS — scatter-poll → reduce → chunks_complete
      // =================================================================
      const size_t compTid =
          static_cast<size_t>(blockIdx.x - 1) * static_cast<size_t>(blockDim.x)
          + threadIdx.x;
      const size_t compStride =
          static_cast<size_t>(compBlocks) * static_cast<size_t>(blockDim.x);

      for (int c = 0; c < numChunks; c++) {
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
              + static_cast<size_t>(sender) * numQ + 1;
          while (core::AtomicLoadRelaxed(sig) < expected)
            ;
        }

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
      }

      if (threadIdx.x == 0) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        asm volatile("buffer_wbl2" ::: "memory");
#endif
        __threadfence();
        __hip_atomic_fetch_add(&barrier->chunks_complete, 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }

    } else {
      // =================================================================
      // BLOCK 0 — Scatter burst + cc wait + AG
      // =================================================================
      const int thr = static_cast<int>(threadIdx.x);
      const uint32_t ccBase = s_cc_base;

      // ---- Phase 1: burst scatter (all chunks, all peers) on queue 1 ----
      if (thr < npes && thr != myPe) {
        const int destPe = thr;
        anvil::SdmaQueueDeviceHandle** dh =
            dstMemObj->deviceHandles_d + destPe * numQ;
        HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
            + static_cast<size_t>(myPe) * numQ;
        HSAuint64* eSig = dstMemObj->expectSignalsPtr
            + static_cast<size_t>(destPe) * numQ;
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
            core::SdmaPutThread(src, dst, actualBytes, dh, rSig, eSig, numQ, 1);
          }
        }
      }

      // ---- Phase 2: per-chunk cc wait → AG on queue 1 ----
      if constexpr (MULTI_CHUNK) {
        if (thr < npes && thr != myPe) {
          const int destPe = thr;
          anvil::SdmaQueueDeviceHandle** dh =
              dstMemObj->deviceHandles_d + destPe * numQ;
          HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
              + static_cast<size_t>(myPe) * numQ;
          HSAuint64* eSig = dstMemObj->expectSignalsPtr
              + static_cast<size_t>(destPe) * numQ;

          for (int c = 0; c < numChunks; c++) {
            const uint32_t ccTarget =
                ccBase + static_cast<uint32_t>((c + 1) * compBlocks);
            while (__scoped_atomic_load_n(
                       &barrier->chunks_complete,
                       __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget)
              ;

            const size_t cOff = static_cast<size_t>(c) * chunkBytes;
            size_t agBytes = chunkBytes;
            if (cOff + agBytes > totalShardBytes)
              agBytes = totalShardBytes - cOff;

            uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            core::SdmaPutThread(src, dst, agBytes, dh, rSig, eSig, numQ, 1);
          }
        }
      } else {
        if (thr < npes && thr != myPe) {
          const int destPe = thr;
          anvil::SdmaQueueDeviceHandle** dh =
              dstMemObj->deviceHandles_d + destPe * numQ;
          HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
              + static_cast<size_t>(myPe) * numQ;
          HSAuint64* eSig = dstMemObj->expectSignalsPtr
              + static_cast<size_t>(destPe) * numQ;

          const uint32_t ccTarget =
              ccBase + static_cast<uint32_t>(compBlocks);
          while (__scoped_atomic_load_n(
                     &barrier->chunks_complete,
                     __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget)
            ;

          uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
              + static_cast<size_t>(myPe) * totalShardBytes;
          uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
              + static_cast<size_t>(myPe) * totalShardBytes;
          core::SdmaPutThread(src, dst, totalShardBytes, dh, rSig, eSig, numQ, 1);
        }
      }

      // ---- Phase 3: wait for AG from all senders ----
      // Scatter+AG share q=1: each pipeline call sends numChunks scatter +
      // numChunks AG per peer, so total expected increments = 2*numChunks.
      if (thr < npes && thr != myPe) {
        const int sender = thr;
        const uint64_t expected =
            s_ag_by_sender[sender] + static_cast<uint64_t>(
                MULTI_CHUNK ? 2 * numChunks : 2);
        HSAuint64* sig = dstMemObj->signalPtrs
            + static_cast<size_t>(sender) * numQ + 1;
        while (core::AtomicLoadRelaxed(sig) < expected)
          ;
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

}

}  // namespace collective
}  // namespace mori

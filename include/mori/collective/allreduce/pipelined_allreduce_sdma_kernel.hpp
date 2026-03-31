// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce — SDMA scatter + reduce + SDMA AllGather.
//
// SCATTER_MODE=0: Single-kernel AllReduce (SDMA scatter + reduce + SDMA AG)
//   1-chunk mode (shard < 2×kMinChunkShardBytes):
//     Block 0: read baselines → broadcast → burst scatter → cc wait → AG → AG wait.
//     Compute blocks: wait broadcast → scatter-poll → reduce → wbl2+fence → cc.
//   Multi-chunk mode (shard ≥ 2×kMinChunkShardBytes, default 2 chunks):
//     Block 0: read baselines → broadcast → burst scatter → per-chunk (cc→AG) → AG wait.
//     Compute blocks: wait broadcast → scatter-poll → reduce → wbl2+fence → cc.
//     Overlaps AG(c) SDMA transfer with scatter(c+1)+reduce(c+1) on CU.
//     wbl2+CC for intermediate chunks runs on wavefront 1 (thread 64),
//     parallel with scatter-poll on wavefront 0.
//
// SCATTER_MODE=1: P2P read + CU AG (legacy path).
//
// Baseline protocol: Block 0 reads expectSignalsPtr once (before SdmaPutThread
// modifies it), stores to barrier->scatter_base/ag_base with device-scope
// release, then compute blocks acquire-poll barrier->ag_sync.  One-time
// broadcast at kernel start, ~100ns overhead.
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

template <typename T, int SCATTER_MODE = 0, bool MULTI_CHUNK = false>
__global__ void PipelinedAllReduceSdmaKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr flagsMemObj,
    CrossPeBarrier* __restrict__ barrier,
    const application::SymmMemObjPtr inputSymmObj,
    size_t elementCount,
    size_t chunkElementCount,
    uint32_t pipelineGen) {

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

  __shared__ uint32_t s_cc_base;
  __shared__ const P* s_pe_ptrs[8];

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    s_cc_base = __scoped_atomic_load_n(
        &barrier->chunks_complete, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
  __syncthreads();

  // =========================================================================
  // SCATTER_MODE = 0 — Single-buffer SDMA + kernel-side baseline broadcast
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
      // COMPUTE BLOCKS (1..N): wait baseline → scatter-poll → reduce → cc
      // =================================================================
      __shared__ uint64_t s_scatter_base;

      if (threadIdx.x == 0) {
        while (__scoped_atomic_load_n(
                   &barrier->pipeline_gen, __ATOMIC_ACQUIRE,
                   __MEMORY_SCOPE_DEVICE) < pipelineGen)
          ;
        s_scatter_base = barrier->scatter_base;
      }
      __syncthreads();

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

        {
          const int s = static_cast<int>(threadIdx.x);
          if (s < npes && s != myPe) {
            HSAuint64* sig = dstMemObj->signalPtrs
                + static_cast<size_t>(s) * numQ;
            const uint64_t expected =
                s_scatter_base + static_cast<uint64_t>(c + 1);
            while (core::AtomicLoadRelaxed(sig) < expected)
              ;
          }
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
      // BLOCK 0 — Read baselines → broadcast → scatter → cc wait → AG
      // =================================================================
      const int thr = static_cast<int>(threadIdx.x);
      const uint32_t ccBase = s_cc_base;

      __shared__ uint64_t s_ag_base;

      // ---- Read baselines and broadcast to compute blocks ----
      if (thr == 0) {
        const int refPe = (myPe == 0) ? 1 : 0;
        const size_t refOff = static_cast<size_t>(refPe) * numQ;

        uint64_t sb = dstMemObj->expectSignalsPtr[refOff + 0];
        uint64_t ab = dstMemObj->expectSignalsPtr[refOff + 1];
        s_ag_base = ab;

        barrier->scatter_base = sb;
        barrier->ag_base = ab;
        __threadfence();
        __scoped_atomic_store_n(&barrier->pipeline_gen, pipelineGen,
                                __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
      }
      __syncthreads();

      // ---- Phase 1: burst scatter (all chunks, all peers) ----
      if (thr < npes && thr != myPe) {
        const int destPe = thr;
        anvil::SdmaQueueDeviceHandle** dh =
            dstMemObj->deviceHandles_d + destPe * numQ;
        HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
            + static_cast<size_t>(myPe) * numQ;
        HSAuint64* eSig = dstMemObj->expectSignalsPtr
            + destPe * numQ;
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
            core::SdmaPutThread(src, dst, actualBytes, dh, rSig, eSig, numQ, 0);
          }
        }
      }

      // ---- Phase 2: per-chunk cc wait → AG ----
      if constexpr (MULTI_CHUNK) {
        for (int c = 0; c < numChunks; c++) {
          if (thr < npes && thr != myPe) {
            const int destPe = thr;
            anvil::SdmaQueueDeviceHandle** dh =
                dstMemObj->deviceHandles_d + destPe * numQ;
            HSAuint64* rSigAG = dstMemObj->peerSignalPtrs[destPe]
                + static_cast<size_t>(myPe) * numQ + 1;
            HSAuint64* eSigAG = dstMemObj->expectSignalsPtr
                + destPe * numQ + 1;

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
            core::SdmaPutThread(src, dst, agBytes, dh, rSigAG, eSigAG, numQ, 0);
          }
          __syncthreads();
        }
      } else {
        // Single-chunk: one cc wait → one AG
        if (thr < npes && thr != myPe) {
          const int destPe = thr;
          anvil::SdmaQueueDeviceHandle** dh =
              dstMemObj->deviceHandles_d + destPe * numQ;
          HSAuint64* rSigAG = dstMemObj->peerSignalPtrs[destPe]
              + static_cast<size_t>(myPe) * numQ + 1;
          HSAuint64* eSigAG = dstMemObj->expectSignalsPtr
              + destPe * numQ + 1;

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
          core::SdmaPutThread(src, dst, totalShardBytes, dh, rSigAG, eSigAG, numQ, 0);
        }
      }

      // ---- Phase 3: wait for AG from all senders ----
      if (thr < npes && thr != myPe) {
        const int sender = thr;
        const uint64_t expected =
            s_ag_base + static_cast<uint64_t>(numChunks);
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

    __shared__ uint64_t s_ag_by_sender[64];
    if (blockIdx.x == 0) {
      const int s = static_cast<int>(threadIdx.x);
      if (s < npes && s != myPe) {
        s_ag_by_sender[s] = dstMemObj->expectSignalsPtr[
            static_cast<size_t>(s) * numQ + 1];
      }
    }
    __syncthreads();

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

// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce — SDMA scatter + reduce + SDMA AllGather.
//
// SCATTER_MODE=0: Single-kernel AllReduce (SDMA scatter + reduce + SDMA AG)
//   1-chunk mode (shard < 2×kMinChunkShardBytes):
//     Block 0: burst scatter → cc wait → SDMA AG push → AG wait.
//     Compute blocks: scatter-poll → reduce → wbl2+fence → chunks_complete.
//     No signal/barrier zeroing: monotonic ATOMIC_INC + baseline protocol.
//   Multi-chunk mode (shard ≥ 2×kMinChunkShardBytes, default 2 chunks):
//     Block 0: burst scatter → per-chunk (cc wait → SDMA AG push) → AG wait.
//     Compute blocks: scatter-poll → reduce → wbl2+fence → chunks_complete.
//     Overlaps AG(c) SDMA transfer with scatter(c+1)+reduce(c+1) on CU.
//     wbl2+CC for intermediate chunks runs on wavefront 1 (thread 64),
//     parallel with scatter-poll on wavefront 0.
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
      s_scatter_by_sender[s] =
          core::AtomicLoadRelaxed(dstMemObj->signalPtrs + row + 0);
      if (blockIdx.x == 0) {
        s_ag_by_sender[s] =
            core::AtomicLoadRelaxed(dstMemObj->signalPtrs + row + 1);
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

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("PE%d: ENTER numQ=%u compBlk=%d chunks=%d shardB=%zu ccBase=%u\n",
             myPe, numQ, compBlocks, numChunks,
             totalShardBytes, s_cc_base);
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
          int sSpin = 0;
          while (core::AtomicLoadRelaxed(sig) < expected) {
            if (++sSpin > 200000000 && blockIdx.x == 1 && threadIdx.x == 0) {
              printf("PE%d blk%d: SCATTER SIG TIMEOUT sender=%d c=%d exp=%llu act=%llu base=%llu\n",
                     myPe, (int)blockIdx.x, sender, c,
                     (unsigned long long)expected,
                     (unsigned long long)core::AtomicLoadRelaxed(sig),
                     (unsigned long long)s_scatter_by_sender[sender]);
              break;
            }
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
      // BLOCK 0 — Burst-Submit Scatter + AG
      // =================================================================
      const int thr = static_cast<int>(threadIdx.x);
      const uint32_t ccBase = s_cc_base;

      if (thr < npes && thr != myPe) {
        const int destPe = thr;
        anvil::SdmaQueueDeviceHandle** dh =
            dstMemObj->deviceHandles_d + destPe * numQ;
        HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
            + static_cast<size_t>(myPe) * numQ;
        if (thr == (myPe == 0 ? 1 : 0)) {
          printf("PE%d: SCATTER dh=%p rSig=%p bytes=%zu\n",
                 myPe, (void*)dh, (void*)rSig, totalShardBytes);
        }
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
      if (thr == 0) {
        printf("PE%d: SCATTER DONE\n", myPe);
      }

      if constexpr (MULTI_CHUNK) {
        if (thr < npes && thr != myPe) {
          const int destPe = thr;
          anvil::SdmaQueueDeviceHandle** dh =
              dstMemObj->deviceHandles_d + destPe * numQ;
          HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
              + static_cast<size_t>(myPe) * numQ;

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
            core::SdmaPutThread(src, dst, agBytes, dh, rSig, numQ, 1);
          }
        }

        if (thr < npes && thr != myPe) {
          const int sender = thr;
          const uint64_t expected =
              s_ag_by_sender[sender] + static_cast<uint64_t>(numChunks);
          HSAuint64* sig = dstMemObj->signalPtrs
              + static_cast<size_t>(sender) * numQ + 1;
          while (core::AtomicLoadRelaxed(sig) < expected)
            ;
        }
      } else {
        if (thr == 0) {
          printf("PE%d: CC_WAIT target=%u\n", myPe,
                 ccBase + static_cast<uint32_t>(compBlocks));
        }
        if (thr < npes && thr != myPe) {
          const int destPe = thr;
          anvil::SdmaQueueDeviceHandle** dh =
              dstMemObj->deviceHandles_d + destPe * numQ;
          HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
              + static_cast<size_t>(myPe) * numQ;

          const uint32_t ccTarget =
              ccBase + static_cast<uint32_t>(compBlocks);
          int ccSpin = 0;
          while (__scoped_atomic_load_n(
                     &barrier->chunks_complete,
                     __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < ccTarget) {
            if (++ccSpin > 200000000 && thr == (myPe == 0 ? 1 : 0)) {
              printf("PE%d: CC TIMEOUT target=%u act=%u\n", myPe, ccTarget,
                     __scoped_atomic_load_n(&barrier->chunks_complete,
                                            __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE));
              break;
            }
          }

          uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
              + static_cast<size_t>(myPe) * totalShardBytes;
          uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
              + static_cast<size_t>(myPe) * totalShardBytes;
          core::SdmaPutThread(src, dst, totalShardBytes, dh, rSig, numQ, 1);
        }
        __syncthreads();
        if (thr == 0) {
          printf("PE%d: AG DONE, waiting sigs\n", myPe);
        }

        if (thr < npes && thr != myPe) {
          const int sender = thr;
          const uint64_t expected = s_ag_by_sender[sender] + 1ULL;
          HSAuint64* sig = dstMemObj->signalPtrs
              + static_cast<size_t>(sender) * numQ + 1;
          int agSpin = 0;
          while (core::AtomicLoadRelaxed(sig) < expected) {
            if (++agSpin > 200000000 && thr == (myPe == 0 ? 1 : 0)) {
              printf("PE%d: AG SIG TIMEOUT sender=%d exp=%llu act=%llu\n",
                     myPe, sender,
                     (unsigned long long)expected,
                     (unsigned long long)core::AtomicLoadRelaxed(sig));
              break;
            }
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
        while (core::AtomicLoadRelaxed(sig) < expected)
          ;
      }
    }
  }

}

}  // namespace collective
}  // namespace mori

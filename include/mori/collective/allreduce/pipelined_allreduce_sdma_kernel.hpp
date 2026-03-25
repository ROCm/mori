// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce — single-buffer SDMA pipeline with cross-PE barrier.
//
// SCATTER_MODE=0: 3-stage pipeline (loop i = 0 .. numChunks+1)
//   scatter(i) | reduce(i-1) local | AG(i-2)
//
//   qId=0 scatter, qId=1 AG, qId=2 reduce-done (CU atomic on peerSignalPtrs).
//   Before AG(k), all PEs must finish reduce(k); wait on signalPtrs[*][2].
//
//   block_done[blk] = bdBase + (c+1) after reduce(chunk c) (numChunks==1 safe).
//
//   Block 0: parallel wavefronts — scatter / wait_rd(q=2) / block_done poll;
//   then wbl2 + signal q=2 + AG (wavefront lockstep for ordering).
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

  // ---- Signal baselines (dstMemObj; requires sdmaNumQueue >= 3 for qId=2) ----
  __shared__ uint64_t s_scatter_base;
  __shared__ uint64_t s_ag_base;
  __shared__ uint64_t s_rd_base;
  __shared__ uint32_t s_bd_base;

  if (threadIdx.x == 0) {
    s_ag_base = 0;
    s_rd_base = 0;
    if (blockIdx.x == 0 && compBlocks > 0) {
      s_bd_base = __scoped_atomic_load_n(
          &barrier->block_done[1], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    } else if (blockIdx.x != 0) {
      s_bd_base = __scoped_atomic_load_n(
          &barrier->block_done[blockIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    } else {
      s_bd_base = 0;
    }
    for (int i = 0; i < npes; ++i) {
      if (i != myPe) {
        s_scatter_base = core::AtomicLoadRelaxed(
            dstMemObj->signalPtrs + static_cast<size_t>(i) * numQ + 0);
        if (blockIdx.x == 0) {
          s_ag_base = core::AtomicLoadRelaxed(
              dstMemObj->signalPtrs + static_cast<size_t>(i) * numQ + 1);
          s_rd_base = core::AtomicLoadRelaxed(
              dstMemObj->signalPtrs + static_cast<size_t>(i) * numQ + 2);
        }
        break;
      }
    }
  }
  __syncthreads();

  const uint64_t scatterBase = s_scatter_base;
  const uint64_t agBase = s_ag_base;
  const uint64_t rdBase = s_rd_base;
  const uint32_t bdBase = s_bd_base;

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
      // COMPUTE BLOCKS (1..N): parallel scatter-poll -> reduce -> flag
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
              scatterBase + static_cast<uint64_t>(c + 1);
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

        // After reduce(chunk c): release block 0 for AG(c). Required for
        // numChunks==1 (otherwise bdBase+1 is never written inside the loop).
        __syncthreads();
        if (threadIdx.x == 0) {
          __scoped_atomic_store_n(
              &barrier->block_done[blockIdx.x],
              bdBase + static_cast<uint32_t>(c + 1),
              __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
        }
      }

    } else {
      // =================================================================
      // BLOCK 0: i = 0..numChunks+1
      //   Ph1: scatter(i) | wait_rd(i) | bd_poll(i)  [parallel]
      //   Ph2: wbl2 + signal q=2 for reduce(i-1) | AG(i-2)
      //   wait_rd: need rdBase + (i-1) from each peer before AG(i-2) (i>=2)
      //   bd_poll: i in [1,numChunks] target bdBase + i
      // =================================================================
      for (int i = 0; i <= numChunks + 1; i++) {
        const int thr = static_cast<int>(threadIdx.x);

        if (i < numChunks && thr < npes && thr != myPe) {
          const int destPe = thr;
          const size_t cOff = static_cast<size_t>(i) * chunkBytes;
          size_t actualBytes = chunkBytes;
          if (cOff + actualBytes > totalShardBytes)
            actualBytes = totalShardBytes - cOff;
          if (actualBytes > 0) {
            uint8_t* src = reinterpret_cast<uint8_t*>(const_cast<T*>(input))
                + static_cast<size_t>(destPe) * totalShardBytes + cOff;
            uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            anvil::SdmaQueueDeviceHandle** dh =
                dstMemObj->deviceHandles_d + destPe * numQ;
            HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
                + static_cast<size_t>(myPe) * numQ;
            core::SdmaPutThread(src, dst, actualBytes, dh, rSig, numQ, 0);
          }
        }

        if (i >= 2 && i <= numChunks + 1 && thr >= 64 &&
            thr < 64 + npes - 1) {
          const int idx = thr - 64;
          const int sender = idx < myPe ? idx : idx + 1;
          const uint64_t need = rdBase + static_cast<uint64_t>(i - 1);
          HSAuint64* sig = dstMemObj->signalPtrs
              + static_cast<size_t>(sender) * numQ + 2;
          while (core::AtomicLoadRelaxed(sig) < need)
            ;
        }

        if (i >= 1 && i <= numChunks && thr >= 128 &&
            thr < 128 + compBlocks) {
          const int blk = thr - 128 + 1;
          const uint32_t target = bdBase + static_cast<uint32_t>(i);
          while (__scoped_atomic_load_n(
                     &barrier->block_done[blk],
                     __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < target)
            ;
        }

        __syncthreads();

        if (i >= 1 && i <= numChunks) {
          if (thr == 0) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
            asm volatile("buffer_wbl2" ::: "memory");
#endif
            __threadfence_system();
          }
        }
        __syncthreads();

        if (i >= 1 && i <= numChunks) {
          if (thr < npes && thr != myPe) {
            HSAuint64* rs = dstMemObj->peerSignalPtrs[thr]
                + static_cast<size_t>(myPe) * numQ + 2;
            __hip_atomic_fetch_add(rs, 1ULL,
                __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
          }
        }
        __syncthreads();

        if (i >= 2 && i <= numChunks + 1 && thr < npes && thr != myPe) {
          const int destPe = thr;
          const int agChunk = i - 2;
          const size_t cOff = static_cast<size_t>(agChunk) * chunkBytes;
          size_t actualBytes = chunkBytes;
          if (cOff + actualBytes > totalShardBytes)
            actualBytes = totalShardBytes - cOff;
          if (actualBytes > 0) {
            uint8_t* src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            uint8_t* dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
                + static_cast<size_t>(myPe) * totalShardBytes + cOff;
            anvil::SdmaQueueDeviceHandle** dh =
                dstMemObj->deviceHandles_d + destPe * numQ;
            HSAuint64* rSig = dstMemObj->peerSignalPtrs[destPe]
                + static_cast<size_t>(myPe) * numQ;
            core::SdmaPutThread(src, dst, actualBytes, dh, rSig, numQ, 1);
          }
        }
      }

      // Final AG wait
      {
        const int thr = static_cast<int>(threadIdx.x);
        if (thr < npes - 1) {
          const int sender = thr < myPe ? thr : thr + 1;
          const uint64_t expected =
              agBase + static_cast<uint64_t>(numChunks);
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
            agBase + static_cast<uint64_t>(numChunks);
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

// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce SDMA kernel.
//
// SCATTER_MODE=0: SDMA scatter → reduce → cross-PE barrier → CU AllGather
//   The fused reduce+CU-AG has a cross-PE HBM race: PE X's CU AG write to
//   PE Y's HBM can arrive before PE Y reads the scatter data at the same
//   address (xGMI bypasses L2; L2 cold-miss reads from HBM which may
//   already be overwritten). Fix: split reduce / CU-AG with a lightweight
//   cross-PE barrier on signal-slot qId=2.
//
// SCATTER_MODE=1: P2P read (inputSymmObj) + CU AG write (dstMemObj)
//   Input and output are separate buffers → no cross-PE race → fused.
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

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  const uint32_t numQ = dstMemObj->sdmaNumQueue;

  // ---- Shared state ----
  __shared__ uint32_t s_gen;
  __shared__ uint64_t s_scatter_base;
  __shared__ uint64_t s_ag_base;
  __shared__ uint64_t s_rd_base;  // reduce-done baseline (qId=2)
  __shared__ uint32_t s_gather_count;

  if (threadIdx.x == 0) {
    s_gen = barrier->flag;
    s_scatter_base = static_cast<uint64_t>(barrier->flag);
    s_ag_base = 0;
    s_rd_base = 0;
    s_gather_count = 0;
  }
  __syncthreads();

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < npes; ++i) {
      if (i != myPe) {
        s_ag_base = core::AtomicLoadRelaxed(
            dstMemObj->signalPtrs + static_cast<size_t>(i) * numQ + 1);
        s_rd_base = core::AtomicLoadRelaxed(
            dstMemObj->signalPtrs + static_cast<size_t>(i) * numQ + 2);
        break;
      }
    }
  }
  if (blockIdx.x == 0) __syncthreads();

  const uint64_t scatterBase = s_scatter_base;
  const uint64_t agBase = s_ag_base;
  const uint64_t rdBase = s_rd_base;

  // =========================================================================
  // Intra-PE barriers (block 0 ↔ all blocks)
  // =========================================================================
  auto broadcastBarrier = [&]() {
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        s_gen++;
        __scoped_atomic_store_n(&barrier->flag, s_gen,
                                __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      }
    } else {
      if (threadIdx.x == 0) {
        uint32_t expected = s_gen + 1;
        while (__scoped_atomic_load_n(&barrier->flag,
                                      __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < expected)
          ;
        s_gen = expected;
      }
    }
    __syncthreads();
  };

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

  // =========================================================================
  // Reduce only — reads scatter data from buf, writes to local myDst.
  // =========================================================================
  auto reduceOnly = [&](int chunkIdx) {
    const size_t off = static_cast<size_t>(chunkIdx) * packedChunkPerRank;
    size_t cnt = packedChunkPerRank;
    if (off + cnt > packedPerRank) cnt = packedPerRank - off;

    P* __restrict__ myDst =
        buf + static_cast<size_t>(myPe) * packedPerRank + off;
    const P* __restrict__ myInput = reinterpret_cast<const P*>(input)
        + static_cast<size_t>(myPe) * packedPerRank + off;

    for (size_t k = tid; k < cnt; k += stride) {
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
  };

  // =========================================================================
  // CU AllGather only — reads local reduce result, writes to remote peers.
  // =========================================================================
  auto cuAgOnly = [&](int chunkIdx) {
    const size_t off = static_cast<size_t>(chunkIdx) * packedChunkPerRank;
    size_t cnt = packedChunkPerRank;
    if (off + cnt > packedPerRank) cnt = packedPerRank - off;

    const P* __restrict__ myResult =
        buf + static_cast<size_t>(myPe) * packedPerRank + off;

    for (size_t k = tid; k < cnt; k += stride) {
      P val = myResult[k];
      for (int pe = 0; pe < npes; ++pe) {
        if (pe == myPe) continue;
        reinterpret_cast<P*>(dstMemObj->peerPtrs[pe])
            [static_cast<size_t>(myPe) * packedPerRank + off + k] = val;
      }
    }
  };

  // =========================================================================
  // Fused P2P read-reduce + CU AllGather (SCATTER_MODE=1 only)
  // Input/output are separate symmetric buffers → no cross-PE race.
  // =========================================================================
  auto p2pReduceAndCuAg = [&](int chunkIdx) {
    const size_t off = static_cast<size_t>(chunkIdx) * packedChunkPerRank;
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
  };

  // =========================================================================
  // Cross-PE signaling helpers
  // =========================================================================

  // Signal all remote PEs that our reduce is done (qId=2, system-scope).
  // RELEASE ordering ensures prior stores (reduce result in L2/HBM) are
  // visible before the signal becomes visible at the remote PE.
  auto signalReduceDone = [&]() {
    if (blockIdx.x == 0) {
      const int pe = static_cast<int>(threadIdx.x);
      if (pe < npes && pe != myPe) {
        HSAuint64* sig = dstMemObj->peerSignalPtrs[pe]
            + static_cast<size_t>(myPe) * numQ + 2;
        __hip_atomic_fetch_add(sig, 1ULL,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  };

  // Block 0 waits until all remote PEs have finished their reduce for the
  // current chunk.  After this returns, it is safe for any PE to issue
  // CU AG writes that touch remote PEs' transit buffers.
  auto waitReduceDone = [&](int chunkOrdinal) {
    if (blockIdx.x != 0) return;
    const int thr = static_cast<int>(threadIdx.x);
    if (thr < npes - 1) {
      const int sender = thr < myPe ? thr : thr + 1;
      const uint64_t expected =
          rdBase + static_cast<uint64_t>(chunkOrdinal);
      HSAuint64* sig = dstMemObj->signalPtrs
          + static_cast<size_t>(sender) * numQ + 2;
      int spin = 0;
      while (core::AtomicLoadRelaxed(sig) < expected) {
        if (++spin > 100000000) {
          printf("PE %d: reduce-done timeout chunk=%d peer=%d exp=%llu act=%llu\n",
                 myPe, chunkOrdinal - 1, sender,
                 (unsigned long long)expected,
                 (unsigned long long)core::AtomicLoadRelaxed(sig));
          break;
        }
      }
    }
    __syncthreads();
  };

  // Signal all remote PEs that our AG write for one chunk is done (qId=1).
  auto signalAg = [&]() {
    if (blockIdx.x == 0) {
      const int pe = static_cast<int>(threadIdx.x);
      if (pe < npes && pe != myPe) {
        HSAuint64* sig = dstMemObj->peerSignalPtrs[pe]
            + static_cast<size_t>(myPe) * numQ + 1;
        __hip_atomic_fetch_add(sig, 1ULL,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  };

  // Block 0 waits for cumulative AG signals from all peers (all chunks).
  auto finalAgWait = [&]() {
    if (blockIdx.x != 0) return;
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
    __syncthreads();
  };

  // =========================================================================
  // Main pipeline — SCATTER_MODE = 0 (SDMA scatter + CU AllGather)
  // =========================================================================
  //
  // Per-chunk timeline:
  //
  //   Block 0: scatter submit(c) | scatter wait(c-1)   [parallel wavefronts]
  //   __syncthreads
  //   broadcastBarrier            → all blocks know scatter(c-1) data ready
  //   reduceOnly(c-1)            → all blocks read scatter data, write local
  //   __threadfence_system        → flush local reduce result to HBM
  //   gatherBarrier               → all local blocks done with reduce
  //   signalReduceDone            → block 0 tells remote PEs "reduce done"
  //   waitReduceDone              → block 0 waits for all remote PEs
  //   broadcastBarrier            → all blocks know it's safe to CU AG
  //   cuAgOnly(c-1)              → all blocks write to remote peers via xGMI
  //   __threadfence               → drain xGMI stores
  //   gatherBarrier               → all local blocks done with CU AG
  //   signalAg                    → block 0 tells remote PEs "AG done"
  //
  if constexpr (SCATTER_MODE == 0) {

    int rdChunkOrd = 0;

    for (int c = 0; c <= numChunks; c++) {

      // ---- Management phase (block 0 only) ----
      if (blockIdx.x == 0) {
        const int thr = static_cast<int>(threadIdx.x);

        // Warp 0: SDMA scatter submit for chunk c
        if (c < numChunks && thr < npes && thr != myPe) {
          const int destPe = thr;
          const size_t cOff = static_cast<size_t>(c) * chunkBytes;
          const size_t totalShardBytes = elementCountPerRank * bytesPerElement;
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

        // Warp 1: parallel scatter wait for chunk c-1
        if (c >= 1 && thr >= 64 && thr < 64 + npes - 1) {
          const int idx = thr - 64;
          const int sender = idx < myPe ? idx : idx + 1;
          const uint64_t expected =
              scatterBase + static_cast<uint64_t>(c);
          HSAuint64* sig = dstMemObj->signalPtrs
              + static_cast<size_t>(sender) * numQ;
          int spin = 0;
          while (core::AtomicLoadRelaxed(sig) < expected) {
            if (++spin > 100000000) {
              printf("PE %d: scatter timeout c=%d peer=%d exp=%llu act=%llu\n",
                     myPe, c - 1, sender,
                     (unsigned long long)expected,
                     (unsigned long long)core::AtomicLoadRelaxed(sig));
              break;
            }
          }
        }

        __syncthreads();
      }

      // ---- Compute phase (all blocks, for chunk c-1) ----
      if (c >= 1) {
        // Phase 1: Reduce — all PEs read scatter data, write local result.
        broadcastBarrier();
        reduceOnly(c - 1);
        __threadfence_system();
        gatherBarrier();

        // Cross-PE barrier: all PEs must finish reduce before any PE
        // starts CU AG (which writes to remote PEs' transit buffers).
        rdChunkOrd++;
        signalReduceDone();
        waitReduceDone(rdChunkOrd);

        // Phase 2: CU AllGather — safe now that all PEs finished reduce.
        broadcastBarrier();
        cuAgOnly(c - 1);
        __threadfence();
        gatherBarrier();
        signalAg();
      }
    }

    finalAgWait();

  } else {
    // =========================================================================
    // Main pipeline — SCATTER_MODE = 1 (P2P read + CU AllGather)
    // =========================================================================
    // Input/output are separate symmetric buffers → fused reduce+AG is safe.

    for (int c = 0; c < numChunks; c++) {
      p2pReduceAndCuAg(c);
      __threadfence_system();
      gatherBarrier();
      signalAg();
    }

    finalAgWait();
  }

  // Flush local reduce results from L2 → HBM for subsequent host/DMA reads.
  // CU AG stores bypass L2 (fine-grained xGMI) — only local writes need flush.
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

}  // namespace collective
}  // namespace mori

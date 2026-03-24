// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see twoshot_sdma_kernel.hpp for full text)
//
// Pipelined AllReduce SDMA kernel.
//
// Overlaps SDMA scatter/allgather with CU reduce by processing data in chunks.
// Inspired by NCCL NVLS warp-role division, adapted for SDMA-based transport.
//
// Data flow per chunk iteration:
//   Block 0: AG-wait(i-2) → Scatter(i) → Wait(i-1) → [barrier] → Reduce(i-1) → wbl2 → AG-put(i-1)
//   Other blocks:                                      [barrier] → Reduce(i-1)
//
// scatter_mode: 0 = SDMA scatter (block 0 submits SDMA PUT), 1 = P2P read (all blocks read remotely)
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

// ============================================================================
// PipelinedAllReduceSdmaKernel
//
// Template parameters:
//   T            - data type (uint32_t, float, half, __hip_bfloat16, ...)
//   SCATTER_MODE - 0: SDMA scatter, 1: P2P read
// ============================================================================
template <typename T, int SCATTER_MODE = 0>
__global__ void PipelinedAllReduceSdmaKernel(
    int myPe, int npes,
    const T* __restrict__ input,
    const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr flagsMemObj,
    CrossPeBarrier* __restrict__ barrier,
    // For P2P read mode (SCATTER_MODE=1): input registered in symmetric memory
    const application::SymmMemObjPtr inputSymmObj,
    size_t elementCount,
    size_t chunkElementCount) {

  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  // Align chunk to pack_size
  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t chunkPerRank =
      ((chunkElementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t packedPerRank = elementCountPerRank / pack_size;
  const size_t packedChunkPerRank = chunkPerRank / pack_size;

  if (elementCountPerRank == 0 || chunkPerRank == 0) return;

  const int numChunks = (packedPerRank + packedChunkPerRank - 1) / packedChunkPerRank;
  const size_t chunkBytes = chunkPerRank * bytesPerElement;

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);

  // Generation counter for block-0-to-all broadcast
  __shared__ uint32_t s_gen;
  if (threadIdx.x == 0) {
    s_gen = barrier->flag;
  }
  __syncthreads();

  // ========================================================================
  // Helper lambdas (device code, inlined)
  // ========================================================================

  // --- SDMA scatter: send chunk c's shard to each remote PE ---
  auto sdmaScatterChunk = [&](int chunkIdx) {
    if (blockIdx.x != 0) return;
    int warpId = static_cast<int>(threadIdx.x) / warpSize;
    int laneId = static_cast<int>(threadIdx.x) % warpSize;

    if (warpId < npes && laneId == 0) {
      int destPe = warpId;
      size_t chunkOffset = static_cast<size_t>(chunkIdx) * chunkBytes;
      size_t actualChunkBytes = chunkBytes;
      // Last chunk may be smaller
      size_t totalShardBytes = elementCountPerRank * bytesPerElement;
      if (chunkOffset + actualChunkBytes > totalShardBytes)
        actualChunkBytes = totalShardBytes - chunkOffset;
      if (actualChunkBytes == 0) return;

      uint8_t* srcPtr = reinterpret_cast<uint8_t*>(const_cast<T*>(input))
                        + static_cast<size_t>(destPe) * elementCountPerRank * bytesPerElement
                        + chunkOffset;
      uint8_t* remoteDst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe])
                           + static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement
                           + chunkOffset;

      anvil::SdmaQueueDeviceHandle** dh =
          dstMemObj->deviceHandles_d + destPe * dstMemObj->sdmaNumQueue;
      HSAuint64* remoteSignal = dstMemObj->peerSignalPtrs[destPe]
                                + static_cast<size_t>(myPe) * dstMemObj->sdmaNumQueue;

      core::SdmaPutThread(srcPtr, remoteDst, actualChunkBytes,
                          dh, remoteSignal, dstMemObj->sdmaNumQueue, 0);
    }
  };

  // --- Wait for scatter of chunk c to complete (all PEs' data arrived) ---
  auto waitScatterChunk = [&](int chunkIdx, uint64_t flagVal) {
    if (blockIdx.x != 0) return;
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) continue;
      if (threadIdx.x == 0) {
        HSAuint64* mySignal = dstMemObj->signalPtrs
                              + static_cast<size_t>(sender) * dstMemObj->sdmaNumQueue;
        int spin = 0;
        while (core::AtomicLoadRelaxed(mySignal) < flagVal) {
          if (++spin > 100000000) {
            printf("PE %d: scatter timeout chunk %d peer %d (expect %llu actual %llu base %llu)\n",
                   myPe, chunkIdx, sender,
                   (unsigned long long)flagVal,
                   (unsigned long long)core::AtomicLoadRelaxed(mySignal),
                   (unsigned long long)s_gen);
            break;
          }
        }
      }
      __syncthreads();
    }
  };

  // --- Broadcast from block 0 to all blocks ---
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

  // --- L2 coherence fix for chunk c: CU store input's own shard ---
  auto l2FixChunk = [&](int chunkIdx) {
    size_t chunkPackedOffset = static_cast<size_t>(chunkIdx) * packedChunkPerRank;
    size_t thisChunkPacked = packedChunkPerRank;
    if (chunkPackedOffset + thisChunkPacked > packedPerRank)
      thisChunkPacked = packedPerRank - chunkPackedOffset;

    P* __restrict__ myDst = buf + static_cast<size_t>(myPe) * packedPerRank + chunkPackedOffset;
    const P* __restrict__ inputSlot =
        reinterpret_cast<const P*>(input)
        + static_cast<size_t>(myPe) * packedPerRank + chunkPackedOffset;

    for (size_t k = tid; k < thisChunkPacked; k += stride) {
      myDst[k] = inputSlot[k];
    }
  };

  // --- Reduce chunk c (all blocks) ---
  auto reduceChunk = [&](int chunkIdx) {
    size_t chunkPackedOffset = static_cast<size_t>(chunkIdx) * packedChunkPerRank;
    size_t thisChunkPacked = packedChunkPerRank;
    if (chunkPackedOffset + thisChunkPacked > packedPerRank)
      thisChunkPacked = packedPerRank - chunkPackedOffset;

    P* __restrict__ myDst = buf + static_cast<size_t>(myPe) * packedPerRank + chunkPackedOffset;

    for (size_t k = tid; k < thisChunkPacked; k += stride) {
      size_t globalK = chunkPackedOffset + k;
      A acc = upcast_v<typename P::type, pack_size>(buf[globalK]);
      for (int pe = 1; pe < npes; ++pe) {
        packed_assign_add(
            acc,
            upcast_v<typename P::type, pack_size>(
                buf[static_cast<size_t>(pe) * packedPerRank + globalK]));
      }
      myDst[k] = downcast_v<typename P::type, pack_size>(acc);
    }
  };

  // --- Flush L2 for chunk c ---
  auto flushL2 = [&]() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("buffer_wbl2" ::: "memory");
    }
#endif
    __syncthreads();
  };

  // --- AllGather PUT for chunk c ---
  auto agPutChunk = [&](int chunkIdx) {
    if (blockIdx.x != 0) return;
    int warpId = static_cast<int>(threadIdx.x) / warpSize;
    int laneId = static_cast<int>(threadIdx.x) % warpSize;

    if (warpId < npes && laneId == 0) {
      int remotePe = warpId;
      size_t chunkOffset = static_cast<size_t>(chunkIdx) * chunkBytes;
      size_t actualChunkBytes = chunkBytes;
      size_t totalShardBytes = elementCountPerRank * bytesPerElement;
      if (chunkOffset + actualChunkBytes > totalShardBytes)
        actualChunkBytes = totalShardBytes - chunkOffset;
      if (actualChunkBytes == 0) return;

      uint8_t* agSrcPtr = reinterpret_cast<uint8_t*>(dstMemObj->localPtr)
                          + static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement
                          + chunkOffset;
      uint8_t* agDstPtr = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[remotePe])
                          + static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement
                          + chunkOffset;

      anvil::SdmaQueueDeviceHandle** dh =
          dstMemObj->deviceHandles_d + remotePe * dstMemObj->sdmaNumQueue;
      HSAuint64* remoteSignal = dstMemObj->peerSignalPtrs[remotePe]
                                + static_cast<size_t>(myPe) * dstMemObj->sdmaNumQueue;

      core::SdmaPutThread(agSrcPtr, agDstPtr, actualChunkBytes,
                          dh, remoteSignal, dstMemObj->sdmaNumQueue, 0);
    }
  };

  // --- Wait for AllGather of chunk c ---
  auto agWaitChunk = [&](int chunkIdx, uint64_t flagVal) {
    if (blockIdx.x != 0) return;
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) continue;
      if (threadIdx.x == 0) {
        HSAuint64* mySignal = dstMemObj->signalPtrs
                              + static_cast<size_t>(sender) * dstMemObj->sdmaNumQueue;
        int spin = 0;
        while (core::AtomicLoadRelaxed(mySignal) < flagVal) {
          if (++spin > 100000000) {
            printf("PE %d: AG timeout chunk %d peer %d (expect %llu actual %llu base %llu nChunks %d)\n",
                   myPe, chunkIdx, sender,
                   (unsigned long long)flagVal,
                   (unsigned long long)core::AtomicLoadRelaxed(mySignal),
                   (unsigned long long)s_gen,
                   numChunks);
            break;
          }
        }
      }
      __syncthreads();
    }
  };

  // ========================================================================
  // P2P read + reduce (SCATTER_MODE=1): all blocks read and reduce directly
  // ========================================================================
  auto p2pReadReduceChunk = [&](int chunkIdx) {
    size_t chunkPackedOffset = static_cast<size_t>(chunkIdx) * packedChunkPerRank;
    size_t thisChunkPacked = packedChunkPerRank;
    if (chunkPackedOffset + thisChunkPacked > packedPerRank)
      thisChunkPacked = packedPerRank - chunkPackedOffset;

    P* __restrict__ myDst = buf + static_cast<size_t>(myPe) * packedPerRank + chunkPackedOffset;
    size_t myStart = static_cast<size_t>(myPe) * packedPerRank + chunkPackedOffset;

    for (size_t k = tid; k < thisChunkPacked; k += stride) {
      size_t globalK = myStart + k;
      const P* p0 = reinterpret_cast<const P*>(inputSymmObj->peerPtrs[0]);
      A acc = upcast_v<typename P::type, pack_size>(p0[globalK]);
      for (int pe = 1; pe < npes; ++pe) {
        const P* pp = reinterpret_cast<const P*>(inputSymmObj->peerPtrs[pe]);
        packed_assign_add(acc, upcast_v<typename P::type, pack_size>(pp[globalK]));
      }
      myDst[k] = downcast_v<typename P::type, pack_size>(acc);
    }
  };

  // ========================================================================
  // Main pipeline loop
  // ========================================================================

  // SDMA signal accounting: scatter and AG share the same signal slot (qId=0)
  // per (sender→receiver) pair.  Each SdmaPutThread atomically increments the
  // signal by 1.  The submission order from any PE X targeting PE Y is:
  //
  //   scatter(0), scatter(1), AG(0), scatter(2), AG(1), ... scatter(N-1), AG(N-2), AG(N-1)
  //
  // So the signal value after each operation (relative to signalBase) is:
  //   scatter(c): max(1, 2c)        — positions 1, 2, 4, 6, 8, ...
  //   AG(k):      min(2k+3, 2N)     — positions 3, 5, 7, ..., 2N
  //
  const uint64_t signalBase = static_cast<uint64_t>(s_gen);

  auto scatterFlagVal = [&](int c) -> uint64_t {
      int pos = (2 * c > 1) ? 2 * c : 1;
      return signalBase + static_cast<uint64_t>(pos);
  };
  auto agFlagVal = [&](int k) -> uint64_t {
      int pos = 2 * k + 3;
      int cap = 2 * numChunks;
      return signalBase + static_cast<uint64_t>(pos < cap ? pos : cap);
  };

  // One-time diagnostic: verify signalBase matches actual SDMA signal values
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("PE %d: pipeline start signalBase=%llu barrier=%u numChunks=%d\n",
           myPe, (unsigned long long)signalBase, barrier->flag, numChunks);
    for (int i = 0; i < npes; ++i) {
      if (i == myPe) continue;
      HSAuint64* sig = dstMemObj->signalPtrs
                       + static_cast<size_t>(i) * dstMemObj->sdmaNumQueue;
      printf("  signal[peer %d] = %llu\n", i, (unsigned long long)core::AtomicLoadRelaxed(sig));
    }
  }
  __syncthreads();

  if constexpr (SCATTER_MODE == 0) {
    // === SDMA Scatter mode ===
    // Pipeline: overlap scatter(i) with reduce(i-1) and AG-put(i-2)

    for (int c = 0; c < numChunks + 2; c++) {
      // Step 1: AG-wait for chunk c-2
      if (c >= 2 && c - 2 < numChunks) {
        agWaitChunk(c - 2, agFlagVal(c - 2));
      }

      // Step 2: Scatter chunk c (block 0 only, non-blocking SDMA submit)
      if (c < numChunks) {
        sdmaScatterChunk(c);
      }

      // Step 3: Wait for scatter of chunk c-1
      if (c >= 1 && c - 1 < numChunks) {
        waitScatterChunk(c - 1, scatterFlagVal(c - 1));
      }

      // Step 4: Broadcast barrier + L2 fix + Reduce chunk c-1 (all blocks)
      if (c >= 1 && c - 1 < numChunks) {
        broadcastBarrier();
        l2FixChunk(c - 1);
        reduceChunk(c - 1);
        flushL2();
      }

      // Step 5: AG-put for chunk c-1 (block 0, after reduce completes)
      if (c >= 1 && c - 1 < numChunks) {
        agPutChunk(c - 1);
      }

      if (blockIdx.x == 0) __syncthreads();
    }

    // Final AG-wait for last chunk
    if (numChunks >= 1) agWaitChunk(numChunks - 1, agFlagVal(numChunks - 1));

    // broadcastBarrier() incremented barrier->flag by numChunks, but SDMA
    // signals incremented by 2*numChunks.  Correct barrier->flag so the next
    // kernel invocation computes the right signalBase.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      __scoped_atomic_store_n(&barrier->flag,
          static_cast<uint32_t>(signalBase + 2 * numChunks),
          __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }

  } else {
    // === P2P Read mode ===
    // No SDMA scatter needed; all blocks read remotely and reduce directly.
    // Pipeline: overlap P2P-read-reduce(i) with AG-put(i-1)

    // P2P mode: only AG puts use SDMA (no scatter).  Each AG(k) is the
    // (k+1)-th SDMA op per peer, so expected signal = signalBase + k + 1.
    for (int c = 0; c < numChunks + 1; c++) {
      // Step 1: AG-wait for chunk c-2
      if (c >= 2 && c - 2 < numChunks) {
        agWaitChunk(c - 2, signalBase + static_cast<uint64_t>(c - 2) + 1);
      }

      // Step 2: P2P read + reduce chunk c (all blocks)
      if (c < numChunks) {
        p2pReadReduceChunk(c);
        flushL2();
      }

      // Step 3: AG-put for chunk c (block 0, after reduce completes)
      if (c < numChunks) {
        if (blockIdx.x == 0) __syncthreads();
        agPutChunk(c);
      }
    }

    // Final AG-waits
    for (int c = numChunks >= 2 ? numChunks - 2 : 0; c < numChunks; c++) {
      agWaitChunk(c, signalBase + static_cast<uint64_t>(c) + 1);
    }

    // No broadcastBarrier in P2P mode, so barrier->flag is still signalBase.
    // SDMA signals advanced by numChunks (one AG per chunk).  Correct it.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      __scoped_atomic_store_n(&barrier->flag,
          static_cast<uint32_t>(signalBase + numChunks),
          __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
  }
}

}  // namespace collective
}  // namespace mori

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
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include "src/ops/dispatch_combine/intranode_rdna4.hpp"

namespace mori {
namespace moe {

template <typename T>
inline __device__ void Ep2FullBarrier(EpDispatchCombineArgs<T> args,
                                      uint64_t crossDeviceBarrierFlag) {
  const int thdId = threadIdx.x;
  // One block exchanges peer flags, then releases the local grid. Having every
  // block poll and invalidate peer-visible memory is expensive on PCIe.
  __syncthreads();
  if (thdId == 0) {
    __threadfence_system();
    atomicAdd(args.combineGridBarrier, 1);
  }
  if (blockIdx.x == 0) {
    if (thdId == 0) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, gridDim.x);
    }
    __syncthreads();
    if (thdId < args.config.worldSize) {
      core::AtomicStoreReleaseSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(thdId) + args.config.rank,
          crossDeviceBarrierFlag);
      uint64_t* localFlags = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
      // A fast peer can publish the next generation before all readers finish.
      while (core::AtomicLoadSeqCstSystem(localFlags + thdId) < crossDeviceBarrierFlag) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    if (thdId == 0) {
      __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(args.crossDeviceBarrierFlag, crossDeviceBarrierFlag + 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  if (thdId == 0) {
    while (__hip_atomic_load(args.crossDeviceBarrierFlag, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) < crossDeviceBarrierFlag + 1) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
}

// Keep the general entry's two generations so ranks may select small/large
// entries independently. LocalCounts is only selected for symmetric small
// capacities, where every rank uses this count protocol. Join writes at agent scope,
// acquire the counter's release sequence once, then publish with system release.
// Poll relaxed and acquire only after success, not on every unsuccessful poll.
template <typename T, bool PublishCounts = false, bool LocalCounts = false>
inline __device__ void Ep2SmallBarrier(EpDispatchCombineArgs<T> args, uint64_t generation) {
  __syncthreads();
  if (threadIdx.x == 0)
    __hip_atomic_fetch_add(args.combineGridBarrier, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      while (__hip_atomic_load(args.combineGridBarrier, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) != gridDim.x)
        __builtin_amdgcn_s_sleep(1);
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      if constexpr (PublishCounts) {
        if (!args.replayMode) {
          for (int p = 0; p < 2; ++p) {
            if constexpr (LocalCounts) {
              args.sendTokenNumMemObj->template GetAs<index_t*>()[p] = args.destPeTokenCounter[p];
            } else {
              index_t* signal =
                  args.recvTokenNumMemObj->template GetAs<index_t*>(p) + args.config.rank;
              shmem::ShmemInt32WaitUntilEquals(signal, 0);
              core::AtomicStoreRelaxedSystem(signal, args.destPeTokenCounter[p] + 1);
            }
          }
        }
      }
      const int peer = args.config.rank ^ 1;
      core::AtomicStoreReleaseSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(peer) + args.config.rank,
          generation);
      uint64_t* flag = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>() + peer;
      while (core::AtomicLoadRelaxedSystem(flag) < generation) __builtin_amdgcn_s_sleep(1);
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(args.crossDeviceBarrierFlag, generation + 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  } else if (threadIdx.x == 0) {
    while (__hip_atomic_load(args.crossDeviceBarrierFlag, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT) < generation + 1)
      __builtin_amdgcn_s_sleep(1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
}

// Push combine reads only local inputs/inbox. The last finishing block can
// acknowledge completion without waiting for its peer: the next dispatch's
// first barrier, or the next push combine's entry wait, protects inbox reuse.
// Preserve the same final peer flag and local generation as the full barrier.
template <typename T>
inline __device__ void Ep2SmallComplete(EpDispatchCombineArgs<T> args, uint64_t generation) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned arrived = __hip_atomic_fetch_add(args.combineGridBarrier, 1u, __ATOMIC_RELEASE,
                                                    __HIP_MEMORY_SCOPE_AGENT);
    if (arrived == gridDim.x - 1) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      core::AtomicStoreReleaseSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(args.config.rank ^ 1) +
              args.config.rank,
          generation);
      __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(args.crossDeviceBarrierFlag, generation + 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// Counts/slots remain compact and rank ordered, matching the public routing ABI.
// Small entries group routing allocations across 32 tokens and broadcasts payload vectors
// to both destinations. Large entries retain whole-token copies.
// TopK=0 uses the runtime top-k; retain a separate top-8 instantiation for the
// measured fast path. Keep its original runtime top-k arithmetic and single-pass
// metadata operations: constant folding also changes register allocation.
template <typename T, bool Small = false, bool LocalCounts = false, int TopK = 8>
__device__ void EpDispatchIntraNodeEp2PushKernel_body(EpDispatchCombineArgs<T> args) {
  static_assert(TopK == 0 || TopK == 8);
  static_assert(!LocalCounts || Small, "local counts require the paired V2 protocol");
  const auto& config = args.config;
  const int rank = config.rank;
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int warps = gridDim.x * blockDim.x / warpSize;
  const int topk = config.numExpertPerToken;
  const size_t hidden = config.HiddenDimSz();
  const index_t stride = config.MaxNumTokensToSend();
  const uint64_t generation = *args.crossDeviceBarrierFlag;
  if (!args.replayMode) {
    if constexpr (Small || TopK == 0) {
      // One lane classifies one token. Allocate a whole wave's compact rows
      // with two atomics, instead of two competing atomics for every token.
      for (int base = warp * warpSize; base < args.curRankNumToken; base += warps * warpSize) {
        const int token = base + lane;
        int first[2] = {-1, -1};
        if (token < args.curRankNumToken) {
          for (int k = 0; k < topk; ++k) {
            const index_t expert = args.tokenIndices[size_t(token) * topk + k];
            const int p =
                expert >= 0 && expert < config.numExpertPerRank                             ? 0
                : expert >= config.numExpertPerRank && expert < 2 * config.numExpertPerRank ? 1
                                                                                            : 2;
            if (p < 2 && first[p] < 0) first[p] = k;
            args.dispDestTokIdMap[size_t(token) * topk + k] = FlatTokenIndex(config, 2, 0);
          }
        }
        for (int p = 0; p < 2; ++p) {
          const uint32_t mask = uint32_t(__ballot(first[p] >= 0));
          if (!mask) continue;
          index_t slot = 0;
          if (lane == 0) slot = atomicAdd(args.destPeTokenCounter + p, __popc(mask));
          slot = __shfl(slot, 0) + __popc(mask & ((uint32_t(1) << lane) - 1));
          if (first[p] >= 0)
            args.dispDestTokIdMap[size_t(token) * topk + first[p]] =
                FlatTokenIndex(config, p, slot);
        }
      }
    } else {
      for (int token = warp; token < args.curRankNumToken; token += warps) {
        const index_t expert = lane < topk ? args.tokenIndices[size_t(token) * topk + lane] : -1;
        const int dest = expert >= 0 ? expert / config.numExpertPerRank : 2;
        index_t output = FlatTokenIndex(config, 2, 0);
        for (int p = 0; p < 2; ++p) {
          const uint32_t mask = uint32_t(__ballot(dest == p));
          if (!mask) continue;
          index_t slot = 0;
          if (lane == 0) slot = atomicAdd(args.destPeTokenCounter + p, 1);
          slot = __shfl(slot, 0);
          if (lane == __ffs(mask) - 1) output = FlatTokenIndex(config, p, slot);
        }
        if (lane < topk) args.dispDestTokIdMap[size_t(token) * topk + lane] = output;
      }
    }
    if constexpr (!Small) {
      __syncthreads();
      if (threadIdx.x == 0) atomicAdd(args.dispatchGridBarrier, 1);
      if (warp == 0) {
        if (lane == 0) {
          shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
          *args.dispatchGridBarrier = 0;
        }
        __syncwarp();
        if (lane < 2) {
          index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(lane) + rank;
          shmem::ShmemInt32WaitUntilEquals(signal, 0);
          core::AtomicStoreRelaxedSystem(signal, args.destPeTokenCounter[lane] + 1);
        }
      }
    }
  }
  if constexpr (Small)
    Ep2SmallBarrier<T, true, LocalCounts>(args, generation);
  else
    Ep2FullBarrier(args, generation);
  index_t bases[2] = {0, 0};
  if (!args.replayMode) {
    if constexpr (LocalCounts) {
      if (rank == 1) {
        const index_t* rank0 = args.sendTokenNumMemObj->template GetAs<index_t*>(0);
        bases[0] = rank0[0];
        bases[1] = rank0[1];
      }
      if (warp == 0 && lane == 0)
        *args.totalRecvTokenNum = args.sendTokenNumMemObj->template GetAs<index_t*>(0)[rank] +
                                  args.sendTokenNumMemObj->template GetAs<index_t*>(1)[rank];
    } else {
      if (rank == 1) {
        bases[0] = args.recvTokenNumMemObj->template GetAs<index_t*>(0)[0] - 1;
        bases[1] = args.recvTokenNumMemObj->template GetAs<index_t*>(1)[0] - 1;
      }
      if (warp == 0 && lane == 0) {
        const index_t* counts = args.recvTokenNumMemObj->template GetAs<index_t*>();
        *args.totalRecvTokenNum = counts[0] + counts[1] - 2;
      }
    }
  }
  for (int token = warp; token < args.curRankNumToken; token += warps) {
    if constexpr (TopK == 0) {
      // Update all expert slots, including the second batch at top-k > 32.
      if (!args.replayMode)
        for (int k = lane; k < topk; k += warpSize) {
          index_t& slot = args.dispDestTokIdMap[size_t(token) * topk + k];
          const int p = slot / stride;
          if (p < 2) slot += bases[p];
        }
    }
    index_t flat = lane < topk ? args.dispDestTokIdMap[size_t(token) * topk + lane]
                               : FlatTokenIndex(config, 2, 0);
    const int dest = Small ? (flat < stride       ? 0
                              : flat < 2 * stride ? 1
                                                  : 2)
                           : PeFromFlatTokenIndex(config, flat);
    if (TopK != 0 && !args.replayMode && lane < topk && dest < 2) {
      if constexpr (Small)
        flat += dest == 0 ? bases[0] : bases[1];
      else
        flat =
            FlatTokenIndex(config, dest, bases[dest] + LocalTokIdFromFlatTokenIndex(config, flat));
      args.dispDestTokIdMap[size_t(token) * topk + lane] = flat;
    }
    T* destinations[2] = {nullptr, nullptr};
    if constexpr (TopK == 8) {
      for (int p = 0; p < 2; ++p) {
        const uint32_t mask = uint32_t(__ballot(dest == p));
        if (!mask) continue;
        const index_t target = __shfl(flat, __ffs(mask) - 1);
        const index_t row =
            Small ? target - p * stride : LocalTokIdFromFlatTokenIndex(config, target);
        assert(row < config.MaxNumTokensToRecv());
        T* targetPayload =
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(p) + size_t(row) * hidden;
        if constexpr (!Small)
          core::WarpCopy<T, 4>(targetPayload, args.inpTokenBuf + size_t(token) * hidden, hidden);
        if constexpr (Small) destinations[p] = targetPayload;
        if (lane < topk) {
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(p)[size_t(row) * topk + lane] =
              args.tokenIndices[size_t(token) * topk + lane];
          if (args.weightsBuf)
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                p)[size_t(row) * topk + lane] = args.weightsBuf[size_t(token) * topk + lane];
        }
        if (!args.replayMode && lane == 0) {
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(p)[row] =
              FlatTokenIndex(config, rank, token);
        }
      }
    } else {
      for (int p = 0; p < 2; ++p) {
        index_t target;
        target =
            Rdna4FindRankToken<2>(args.dispDestTokIdMap + size_t(token) * topk, topk, stride, p);
        if (target == 2 * stride) continue;
        const index_t row =
            Small ? target - p * stride : LocalTokIdFromFlatTokenIndex(config, target);
        assert(row < config.MaxNumTokensToRecv());
        T* targetPayload =
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(p) + size_t(row) * hidden;
        if constexpr (!Small)
          core::WarpCopy<T, 4>(targetPayload, args.inpTokenBuf + size_t(token) * hidden, hidden);
        if constexpr (Small) destinations[p] = targetPayload;
        for (int k = lane; k < topk; k += warpSize) {
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(p)[size_t(row) * topk + k] =
              args.tokenIndices[size_t(token) * topk + k];
          if (args.weightsBuf)
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(p)[size_t(row) * topk + k] =
                args.weightsBuf[size_t(token) * topk + k];
        }
        if (!args.replayMode && lane == 0) {
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(p)[row] =
              FlatTokenIndex(config, rank, token);
        }
      }
    }
    if constexpr (Small) {
      // Broadcast each loaded vector to both destinations. Metadata has already
      // been issued; the final system barrier covers metadata and payload alike.
      constexpr int vec = 16 / sizeof(T);
      constexpr int step = 4 * warpSize * vec;
      const T* source = args.inpTokenBuf + size_t(token) * hidden;
      size_t offset = 0;
      for (; offset + step <= hidden; offset += step) {
        typename core::VecTypeSelector<16>::dataType data[4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
          data[u] = core::load<16>(source + offset + (lane + u * warpSize) * vec);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          if (!destinations[p]) continue;
#pragma unroll
          for (int u = 0; u < 4; ++u)
            core::store<16>(destinations[p] + offset + (lane + u * warpSize) * vec, data[u]);
        }
      }
      for (size_t j = offset + lane * vec; j < hidden; j += warpSize * vec) {
        const auto data = core::load<16>(source + j);
        for (int p = 0; p < 2; ++p)
          if (destinations[p]) core::store<16>(destinations[p] + j, data);
      }
    }
  }
  if constexpr (Small)
    Ep2SmallBarrier(args, generation + 1);
  else
    Ep2FullBarrier(args, generation + 1);
  if (warp == 0 && lane < 2) {
    args.destPeTokenCounter[lane] = 0;
    args.localPeTokenCounter[lane] = 0;
    args.recvTokenNumMemObj->template GetAs<index_t*>()[lane] = 0;
  }
}

// EP2 has at most one local and one remote contribution after rank deduplication.
// Large entries retain their whole-token copy/reduction and full barriers.
// Small entries use hidden tiles and a completion acknowledgement. V2 matches
// unrolling to tile width and replaces general flat-map division with EP2 bounds.
template <typename T, bool Small = false, int SmallTile = 512, int TopK = 8>
__device__ void EpCombineIntraNodeEp2Kernel_body(EpDispatchCombineArgs<T> args) {
  static_assert(TopK == 0 || TopK == 8);
  constexpr int unroll = Small ? SmallTile / 256 : 4;
  const auto& config = args.config;
  const int rank = config.rank;
  const int peer = rank ^ 1;
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int warps = gridDim.x * blockDim.x / warpSize;
  const size_t hidden = config.HiddenDimSz();
  const index_t stride = config.MaxNumTokensToSend();
  const int tiles = Small ? (hidden + SmallTile - 1) / SmallTile : 1;
  const int topk = config.numExpertPerToken;
  const uint64_t generation = *args.crossDeviceBarrierFlag;
  const index_t received = *args.totalRecvTokenNum;
  const index_t* reverse = args.dispTokIdToSrcTokIdLocal != nullptr
                               ? args.dispTokIdToSrcTokIdLocal
                               : args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>();

  if constexpr (Small) {
    if (threadIdx.x == 0) {
      uint64_t* done = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>() + peer;
      while (core::AtomicLoadRelaxedSystem(done) < generation - 1) __builtin_amdgcn_s_sleep(1);
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    }
    __syncthreads();
  }

  for (int item = warp; item < received * tiles; item += warps) {
    const int row = item / tiles;
    const int tile = item % tiles;
    const size_t start = Small ? size_t(tile) * SmallTile : 0;
    const size_t length = Small ? min(size_t(SmallTile), hidden - start) : hidden;
    const index_t flat = reverse[row];
    const int sourceRank = Small ? int(flat >= stride) : PeFromFlatTokenIndex(config, flat);
    if (sourceRank == rank) continue;
    const index_t token =
        Small ? flat - sourceRank * stride : LocalTokIdFromFlatTokenIndex(config, flat);
    const int targetRank = peer;
    const index_t targetRow = token;
    core::WarpCopy<T, unroll>(args.intraNodeTokBufs.combineInp->template GetAs<T*>(targetRank) +
                                  size_t(targetRow) * hidden + start,
                              args.inpTokenBuf + size_t(row) * hidden + start, length);
    if constexpr (TopK == 8) {
      if (tile == 0 && args.weightsBuf && lane < topk) {
        args.shmemInpWeightsMemObj->template GetAs<float*>(
            targetRank)[size_t(targetRow) * topk + lane] =
            args.weightsBuf[size_t(row) * topk + lane];
      }
    } else {
      if (tile == 0 && args.weightsBuf)
        for (int k = lane; k < topk; k += warpSize)
          args.shmemInpWeightsMemObj->template GetAs<float*>(
              targetRank)[size_t(targetRow) * topk + k] = args.weightsBuf[size_t(row) * topk + k];
    }
  }
  if constexpr (Small)
    Ep2SmallBarrier(args, generation);
  else
    Ep2FullBarrier(args, generation);
  if (warp == 0 && lane == 0 && args.dispTokIdToSrcTokIdLocal == nullptr) {
    *args.totalRecvTokenNum = 0;
  }
  T* remoteBase = args.intraNodeTokBufs.combineInp->template GetAs<T*>(rank);

  for (int item = warp; item < args.curRankNumToken * tiles; item += warps) {
    const int token = item / tiles;
    const int tile = item % tiles;
    const size_t start = Small ? size_t(tile) * SmallTile : 0;
    const size_t length = Small ? min(size_t(SmallTile), hidden - start) : hidden;
    index_t flat = lane < topk ? args.dispDestTokIdMap[size_t(token) * topk + lane]
                               : FlatTokenIndex(config, 2, 0);
    const int dest = Small ? (flat < stride       ? 0
                              : flat < 2 * stride ? 1
                                                  : 2)
                           : PeFromFlatTokenIndex(config, flat);
    uint32_t localMask, remoteMask;
    index_t localFlat;
    if constexpr (TopK == 0) {
      const index_t* slots = args.dispDestTokIdMap + size_t(token) * topk;
      localFlat = Rdna4FindRankToken<2>(slots, topk, stride, rank);
      localMask = localFlat != 2 * stride;
      remoteMask = Rdna4FindRankToken<2>(slots, topk, stride, peer) != 2 * stride;
    } else {
      localMask = uint32_t(__ballot(dest == rank));
      remoteMask = uint32_t(__ballot(dest == peer));
      localFlat = __shfl(flat, localMask ? __ffs(localMask) - 1 : 0);
    }
    const index_t localRow =
        Small ? localFlat - rank * stride : LocalTokIdFromFlatTokenIndex(config, localFlat);
    const index_t remoteRow = token;
    T* sources[2] = {localMask ? args.inpTokenBuf + size_t(localRow) * hidden + start : nullptr,
                     remoteMask ? remoteBase + size_t(remoteRow) * hidden + start : nullptr};
    T* output =
        args.intraNodeTokBufs.combineOut->template GetAs<T*>() + size_t(token) * hidden + start;
    size_t offset = 0;
    core::WarpAccumLFImpl<T, 16, 2, unroll>(output, sources, nullptr, offset, length);
    core::WarpAccumLFImpl<T, 16, 2, 1>(output, sources, nullptr, offset, length);
    for (size_t j = offset + lane; j < length; j += warpSize) {
      float value = 0;
      if (sources[0]) value += float(sources[0][j]);
      if (sources[1]) value += float(sources[1][j]);
      output[j] = T(value);
    }
    if constexpr (TopK == 8) {
      if (tile == 0 && args.weightsBuf && lane < topk) {
        float value = 0;
        if (localMask) value += args.weightsBuf[size_t(localRow) * topk + lane];
        if (remoteMask)
          value += args.shmemInpWeightsMemObj->template GetAs<float*>(
              rank)[size_t(remoteRow) * topk + lane];
        args.shmemCombineOutWeightsMemObj->template GetAs<float*>()[size_t(token) * topk + lane] =
            value;
      }
    } else {
      if (tile == 0 && args.weightsBuf)
        for (int k = lane; k < topk; k += warpSize) {
          float value = 0;
          if (localMask) value += args.weightsBuf[size_t(localRow) * topk + k];
          if (remoteMask)
            value += args.shmemInpWeightsMemObj->template GetAs<float*>(
                rank)[size_t(remoteRow) * topk + k];
          args.shmemCombineOutWeightsMemObj->template GetAs<float*>()[size_t(token) * topk + k] =
              value;
        }
    }
  }
  if constexpr (Small)
    Ep2SmallComplete(args, generation + 1);
  else
    Ep2FullBarrier(args, generation + 1);
}

}  // namespace moe
}  // namespace mori

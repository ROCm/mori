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

struct alignas(64) Ep4TokenMetadata {
  index_t indices[8];
  float weights[8];
};
static_assert(sizeof(Ep4TokenMetadata) == 64);

template <typename T>
inline __device__ Ep4TokenMetadata* Ep4Metadata(EpDispatchCombineArgs<T> args, int peer) {
  // The first suffix slice is reserved for live Combine weights. Dispatch
  // uses the next 80 bytes per source token (metadata + four reverse slots).
  // Together these consume 208*capacity bytes of the 272*capacity-byte suffix.
  char* suffix =
      reinterpret_cast<char*>(args.intraNodeTokBufs.combineInp->template GetAs<T*>(peer) +
                              size_t(args.config.MaxNumTokensToRecv()) * args.config.HiddenDimSz());
  return reinterpret_cast<Ep4TokenMetadata*>(suffix + size_t(args.config.MaxNumTokensToRecv()) * 8 *
                                                          sizeof(float));
}

template <typename T>
inline __device__ index_t* Ep4SourceMap(EpDispatchCombineArgs<T> args, int peer) {
  return reinterpret_cast<index_t*>(Ep4Metadata(args, peer) +
                                    args.config.MaxNumTokensToSendPerRank());
}

// Runtime-top-k metadata uses variable-width records.
template <typename T, int TopK>
inline __device__ index_t* Ep4Metadata(EpDispatchCombineArgs<T> args, int peer, int token = 0) {
  const int topk = TopK ? TopK : args.config.numExpertPerToken;
  // The first suffix slice is reserved for live Combine weights. Dispatch
  // follows with records [topk indices, topk weights], then four reverse slots
  // per source token. Total suffix use: capacity*(24*topk + 16) bytes, within
  // the existing capacity*(32*topk + 16) allocation even for topk=1.
  static_assert(sizeof(index_t) == sizeof(float));
  char* suffix =
      reinterpret_cast<char*>(args.intraNodeTokBufs.combineInp->template GetAs<T*>(peer) +
                              size_t(args.config.MaxNumTokensToRecv()) * args.config.HiddenDimSz());
  return reinterpret_cast<index_t*>(suffix + size_t(args.config.MaxNumTokensToRecv()) * topk *
                                                 sizeof(float)) +
         size_t(token) * 2 * topk;
}

template <typename T, int TopK>
inline __device__ index_t* Ep4SourceMap(EpDispatchCombineArgs<T> args, int peer) {
  if constexpr (TopK == 8) {
    return Ep4SourceMap(args, peer);
  } else {
    const int topk = TopK ? TopK : args.config.numExpertPerToken;
    return Ep4Metadata<T, TopK>(args, peer) +
           size_t(args.config.MaxNumTokensToSendPerRank()) * 2 * topk;
  }
}

// Paired opt-in EP4 protocol. Only block zero polls peers; local grid joins
// use agent release/acquire, followed by system publication of payload/counts.
template <typename T, bool PublishCounts = false>
inline __device__ void Ep4Barrier(EpDispatchCombineArgs<T> args, uint64_t generation) {
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
        if (!args.replayMode)
          for (int p = 0; p < 4; ++p)
            args.sendTokenNumMemObj->template GetAs<index_t*>()[p] = args.destPeTokenCounter[p];
      }
    }
    __syncthreads();
    if (threadIdx.x < 4 && threadIdx.x != args.config.rank) {
      const int peer = threadIdx.x;
      core::AtomicStoreReleaseSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(peer) + args.config.rank,
          generation);
      const uint64_t* flag = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>() + peer;
      while (core::AtomicLoadRelaxedSystem(flag) < generation) __builtin_amdgcn_s_sleep(1);
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
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

// Hidden inboxes and the weights suffix are not reused by the next dispatch.
// The next combine's entry wait protects both, including peer weights readers.
template <typename T>
inline __device__ void Ep4Complete(EpDispatchCombineArgs<T> args, uint64_t generation) {
  __shared__ unsigned ep4LastBlock;
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned arrived = __hip_atomic_fetch_add(args.combineGridBarrier, 1u, __ATOMIC_RELEASE,
                                                    __HIP_MEMORY_SCOPE_AGENT);
    ep4LastBlock = arrived == gridDim.x - 1;
    if (ep4LastBlock) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
  if (ep4LastBlock) {
    if (threadIdx.x < 4 && threadIdx.x != args.config.rank)
      core::AtomicStoreReleaseSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(threadIdx.x) + args.config.rank,
          generation);
    __syncthreads();
    if (threadIdx.x == 0) {
      __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(args.crossDeviceBarrierFlag, generation + 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// TopK=0 is the general wave32 implementation. Preserve top-8 control flow,
// aligned metadata accesses and variable scopes to retain its measured codegen.
template <typename T, int TopK = 8>
__device__ void EpDispatchIntraNodeEp4HybridKernel_body(EpDispatchCombineArgs<T> args) {
  static_assert(TopK == 0 || TopK == 8);
  const auto& config = args.config;
  const int rank = config.rank;
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int warps = gridDim.x * blockDim.x / warpSize;
  const size_t hidden = config.HiddenDimSz();
  const int topk = TopK ? TopK : config.numExpertPerToken;
  const index_t stride = config.MaxNumTokensToSend();
  const uint64_t generation = *args.crossDeviceBarrierFlag;
  if (!args.replayMode) {
    for (int base = warp * warpSize; base < args.curRankNumToken; base += warps * warpSize) {
      const int token = base + lane;
      int first[4] = {-1, -1, -1, -1};
      if (token < args.curRankNumToken) {
        for (int k = 0; k < topk; ++k) {
          const index_t expert = args.tokenIndices[size_t(token) * topk + k];
          const int p = expert >= 0 ? expert / config.numExpertPerRank : 4;
          if (p < 4 && first[p] < 0) first[p] = k;
          args.dispDestTokIdMap[size_t(token) * topk + k] = 4 * stride;
        }
      }
      for (int p = 0; p < 4; ++p) {
        const uint32_t mask = uint32_t(__ballot(first[p] >= 0));
        if (!mask) continue;
        index_t slot = 0;
        if (lane == 0) slot = atomicAdd(args.destPeTokenCounter + p, __popc(mask));
        slot = __shfl(slot, 0) + __popc(mask & ((uint32_t(1) << lane) - 1));
        if (first[p] >= 0) {
          args.dispDestTokIdMap[size_t(token) * topk + first[p]] = p * stride + slot;
          Ep4SourceMap<T, TopK>(args, rank)[p * config.MaxNumTokensToSendPerRank() + slot] = token;
        }
      }
    }
  }
  {
    for (int token = warp; token < args.curRankNumToken; token += warps) {
      if constexpr (TopK == 8) {
        if (lane < 8) {
          Ep4Metadata(args, rank)[token].indices[lane] =
              args.tokenIndices[size_t(token) * 8 + lane];
          if (args.weightsBuf)
            Ep4Metadata(args, rank)[token].weights[lane] =
                args.weightsBuf[size_t(token) * 8 + lane];
        }
      } else {
        index_t* metadata = Ep4Metadata<T, TopK>(args, rank, token);
        for (int k = lane; k < topk; k += warpSize) {
          metadata[k] = args.tokenIndices[size_t(token) * topk + k];
          if (args.weightsBuf)
            reinterpret_cast<float*>(metadata + topk)[k] =
                args.weightsBuf[size_t(token) * topk + k];
        }
      }
    }
  }
  Ep4Barrier<T, true>(args, generation);
  // Load all peer count vectors concurrently once per block. Re-reading them
  // in source-rank order in every wave serializes several PCIe read latencies.
  __shared__ index_t peerCounts[4][4];
  if (threadIdx.x < 16) {
    const int source = threadIdx.x / 4;
    const int dest = threadIdx.x % 4;
    peerCounts[source][dest] = args.sendTokenNumMemObj->template GetAs<index_t*>(source)[dest];
  }
  __syncthreads();
  index_t bases[4] = {0, 0, 0, 0};
  if (!args.replayMode) {
    for (int src = 0; src < rank; ++src) {
      for (int dest = 0; dest < 4; ++dest) bases[dest] += peerCounts[src][dest];
    }
    if (warp == 0 && lane == 0) {
      index_t received = 0;
      for (int src = 0; src < 4; ++src) received += peerCounts[src][rank];
      *args.totalRecvTokenNum = received;
    }
    if (warp == 0 && lane < 4)
      args.recvTokenNumMemObj->template GetAs<index_t*>()[lane] = peerCounts[lane][rank] + 1;
  }
  for (int token = warp; token < args.curRankNumToken; token += warps) {
    if constexpr (TopK == 0) {
      if (!args.replayMode)
        for (int k = lane; k < topk; k += warpSize) {
          index_t& slot = args.dispDestTokIdMap[size_t(token) * topk + k];
          const int p = slot / stride;
          if (p < 4) slot += bases[p];
        }
    }
    index_t flat = lane < topk ? args.dispDestTokIdMap[size_t(token) * topk + lane] : 4 * stride;
    const int dest = flat / stride;
    if (TopK != 0 && !args.replayMode && lane < topk && dest < 4) {
      flat += bases[dest];
      args.dispDestTokIdMap[size_t(token) * topk + lane] = flat;
    }
    {
      T* destinations[4] = {nullptr, nullptr, nullptr, nullptr};
      if constexpr (TopK == 8) {
        for (int p = 0; p < 4; ++p) {
          const uint32_t mask = uint32_t(__ballot(dest == p));
          if (!mask) continue;
          const index_t row = __shfl(flat, __ffs(mask) - 1) - p * stride;
          assert(row < config.MaxNumTokensToRecv());
          destinations[p] =
              args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(p) + size_t(row) * hidden;
        }
      } else {
        for (int p = 0; p < 4; ++p) {
          index_t target;
          target =
              Rdna4FindRankToken<4>(args.dispDestTokIdMap + size_t(token) * topk, topk, stride, p);
          if (target == 4 * stride) continue;
          const index_t row = target - p * stride;
          assert(row < config.MaxNumTokensToRecv());
          destinations[p] =
              args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(p) + size_t(row) * hidden;
        }
      }
      constexpr int vec = 16 / sizeof(T);
      constexpr int step = 4 * 32 * vec;
      const T* source = args.inpTokenBuf + size_t(token) * hidden;
      size_t offset = 0;
      for (; offset + step <= hidden; offset += step) {
        typename core::VecTypeSelector<16>::dataType data[4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
          data[u] = core::load<16>(source + offset + (lane + u * 32) * vec);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          if (!destinations[p]) continue;
#pragma unroll
          for (int u = 0; u < 4; ++u)
            core::store<16>(destinations[p] + offset + (lane + u * 32) * vec, data[u]);
        }
      }
      for (size_t j = offset + lane * vec; j < hidden; j += 32 * vec) {
        const auto data = core::load<16>(source + j);
        for (int p = 0; p < 4; ++p)
          if (destinations[p]) core::store<16>(destinations[p] + j, data);
      }
    }
  }
  {
    int counts[4], prefixes[4];
    int sum = 0, maximum = 0;
    for (int p = 0; p < 4; ++p) {
      prefixes[p] = sum;
      counts[p] = peerCounts[p][rank];
      sum += counts[p];
      maximum = max(maximum, counts[p]);
    }
    const int items = args.replayMode ? *args.totalRecvTokenNum : maximum * 4;
    MultiWarpIter iter(warps, max(items, 1), size_t(256), 256);
    for (int w = warp; w < items * iter.warpsPerItem; w += warps) {
      int item, part;
      size_t offset, length;
      iter.Decode(w, item, part, offset, length);
      if (!length) continue;
      int src, token, row;
      if (args.replayMode) {
        row = item;
        const index_t* reverse = args.dispTokIdToSrcTokIdLocal != nullptr
                                     ? args.dispTokIdToSrcTokIdLocal
                                     : args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>();
        const index_t flat = reverse[row];
        src = flat / stride;
        token = flat - src * stride;
      } else {
        src = item % 4;
        const int pos = item / 4;
        if (pos >= counts[src]) continue;
        row = prefixes[src] + pos;
        token = Ep4SourceMap<T, TopK>(args, src)[rank * config.MaxNumTokensToSendPerRank() + pos];
        if (part == 0 && lane == 0)
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[row] = src * stride + token;
      }
      if constexpr (TopK == 8) {
        if (part == 0 && lane < 8) {
          args.shmemOutIndicesMemObj->template GetAs<index_t*>()[size_t(row) * 8 + lane] =
              Ep4Metadata(args, src)[token].indices[lane];
          if (args.weightsBuf)
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>()[size_t(row) * 8 + lane] =
                Ep4Metadata(args, src)[token].weights[lane];
        }
      } else {
        if (part == 0) {
          const index_t* metadata = Ep4Metadata<T, TopK>(args, src, token);
          for (int k = lane; k < topk; k += warpSize) {
            args.shmemOutIndicesMemObj->template GetAs<index_t*>()[size_t(row) * topk + k] =
                metadata[k];
            if (args.weightsBuf)
              args.shmemDispatchOutWeightsMemObj->template GetAs<float*>()[size_t(row) * topk + k] =
                  reinterpret_cast<const float*>(metadata + topk)[k];
          }
        }
      }
    }
  }
  Ep4Barrier(args, generation + 1);
  if (warp == 0 && lane < 4) {
    args.destPeTokenCounter[lane] = 0;
    args.localPeTokenCounter[lane] = 0;
    // Keep incoming counts for the paired combine's peer-interleaved traversal.
  }
}

template <typename T, int Tile = 0, int TopK = 8>
__device__ void EpCombineIntraNodeEp4PushKernel_body(EpDispatchCombineArgs<T> args) {
  static_assert(TopK == 0 || TopK == 8);
  const auto& config = args.config;
  const int rank = config.rank;
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int warps = gridDim.x * blockDim.x / warpSize;
  const size_t hidden = config.HiddenDimSz();
  const int topk = TopK ? TopK : config.numExpertPerToken;
  const index_t stride = config.MaxNumTokensToSend();
  const index_t capacity = config.MaxNumTokensToSendPerRank();
  // combineInp reserves MaxNumTokensToRecv()*MaxXferBytesPerToken(), including
  // weights/indices space in addition to hidden. Keep staged weights AFTER the
  // four hidden inbox planes. Dispatch stages only the hidden prefix and its
  // separate metadata buffers, so deferred completion cannot expose weights to
  // a fast peer's next dispatch. User-visible dispatch weights stay untouched.
  const size_t weightOffset = size_t(config.MaxNumTokensToRecv()) * hidden;
  float* weightStage = reinterpret_cast<float*>(
      args.intraNodeTokBufs.combineInp->template GetAs<T*>() + weightOffset);
  const int tiles = Tile ? (hidden + Tile - 1) / Tile : 1;
  constexpr int unroll = Tile ? Tile / 256 : 4;
  const uint64_t generation = *args.crossDeviceBarrierFlag;
  const index_t received = *args.totalRecvTokenNum;
  const index_t* reverse = args.dispTokIdToSrcTokIdLocal != nullptr
                               ? args.dispTokIdToSrcTokIdLocal
                               : args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>();
  if (threadIdx.x < 4 && threadIdx.x != rank) {
    const uint64_t* done = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>() + threadIdx.x;
    while (core::AtomicLoadRelaxedSystem(done) < generation - 1) __builtin_amdgcn_s_sleep(1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  __shared__ index_t incomingCounts[4];
  if (threadIdx.x < 4 && args.dispTokIdToSrcTokIdLocal == nullptr)
    incomingCounts[threadIdx.x] =
        args.recvTokenNumMemObj->template GetAs<index_t*>()[threadIdx.x] - 1;
  __syncthreads();
  int counts[4], prefixes[4];
  int sum = 0, maximum = 0;
  const bool interleaved = args.dispTokIdToSrcTokIdLocal == nullptr;
  if (interleaved) {
    for (int p = 0; p < 4; ++p) {
      prefixes[p] = sum;
      counts[p] = incomingCounts[p];
      sum += counts[p];
      if (p != rank) maximum = max(maximum, counts[p]);
    }
    assert(sum == received);
  }
  // One protocol for every capacity/actual-token combination, including ragged
  // small/large ranks. Hidden is pushed into rank planes; weights are staged
  // contiguously and pulled by their original receive slot after the barrier.
  if (args.weightsBuf)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < received * topk;
         i += gridDim.x * blockDim.x)
      weightStage[i] = args.weightsBuf[i];
  const int pushItems = interleaved ? maximum * 3 * tiles : received * tiles;
  for (int item = warp; item < pushItems; item += warps) {
    int row = item / tiles;
    int tile = item % tiles;
    if (interleaved) {
      const int peer = (rank + 1 + item % 3) % 4;
      const int pos = item / (3 * tiles);
      tile = (item / 3) % tiles;
      if (pos >= counts[peer]) continue;
      row = prefixes[peer] + pos;
    }
    const size_t start = Tile ? size_t(tile) * Tile : 0;
    const size_t length = Tile ? min(size_t(Tile), hidden - start) : hidden;
    const index_t flat = reverse[row];
    const int sourceRank = flat / stride;
    if (sourceRank == rank) continue;
    const index_t token = flat - sourceRank * stride;
    const size_t inboxRow = size_t(rank) * capacity + token;
    core::WarpCopy<T, unroll>(args.intraNodeTokBufs.combineInp->template GetAs<T*>(sourceRank) +
                                  inboxRow * hidden + start,
                              args.inpTokenBuf + size_t(row) * hidden + start, length);
  }
  Ep4Barrier(args, generation);
  if (warp == 0 && lane == 0 && args.dispTokIdToSrcTokIdLocal == nullptr)
    *args.totalRecvTokenNum = 0;
  for (int item = warp; item < args.curRankNumToken * tiles; item += warps) {
    const int token = item / tiles;
    const size_t start = Tile ? size_t(item % tiles) * Tile : 0;
    const size_t length = Tile ? min(size_t(Tile), hidden - start) : hidden;
    const index_t flat =
        lane < topk ? args.dispDestTokIdMap[size_t(token) * topk + lane] : 4 * stride;
    const int dest = flat / stride;
    T* sources[4] = {nullptr, nullptr, nullptr, nullptr};
    float* weights[4] = {nullptr, nullptr, nullptr, nullptr};
    if constexpr (TopK == 8) {
      for (int p = 0; p < 4; ++p) {
        const uint32_t mask = uint32_t(__ballot(dest == p));
        if (!mask) continue;
        const index_t localRow = __shfl(flat, __ffs(mask) - 1) - p * stride;
        const size_t inboxRow = size_t(p) * capacity + token;
        sources[p] = p == rank ? args.inpTokenBuf + size_t(localRow) * hidden + start
                               : args.intraNodeTokBufs.combineInp->template GetAs<T*>() +
                                     inboxRow * hidden + start;
        if (args.weightsBuf)
          weights[p] =
              p == rank
                  ? args.weightsBuf + size_t(localRow) * 8
                  : reinterpret_cast<float*>(
                        args.intraNodeTokBufs.combineInp->template GetAs<T*>(p) + weightOffset) +
                        size_t(localRow) * 8;
      }
    } else {
      for (int p = 0; p < 4; ++p) {
        index_t target;
        target =
            Rdna4FindRankToken<4>(args.dispDestTokIdMap + size_t(token) * topk, topk, stride, p);
        if (target == 4 * stride) continue;
        const index_t localRow = target - p * stride;
        const size_t inboxRow = size_t(p) * capacity + token;
        sources[p] = p == rank ? args.inpTokenBuf + size_t(localRow) * hidden + start
                               : args.intraNodeTokBufs.combineInp->template GetAs<T*>() +
                                     inboxRow * hidden + start;
        if (args.weightsBuf)
          weights[p] =
              p == rank
                  ? args.weightsBuf + size_t(localRow) * topk
                  : reinterpret_cast<float*>(
                        args.intraNodeTokBufs.combineInp->template GetAs<T*>(p) + weightOffset) +
                        size_t(localRow) * topk;
      }
    }
    T* output =
        args.intraNodeTokBufs.combineOut->template GetAs<T*>() + size_t(token) * hidden + start;
    size_t offset = 0;
    core::WarpAccumLFImpl<T, 16, 4, unroll>(output, sources, nullptr, offset, length);
    core::WarpAccumLFImpl<T, 16, 4, 1>(output, sources, nullptr, offset, length);
    for (size_t j = offset + lane; j < length; j += 32) {
      float value = 0;
      for (int p = 0; p < 4; ++p)
        if (sources[p]) value += float(sources[p][j]);
      output[j] = T(value);
    }
    if constexpr (TopK == 8) {
      if (start == 0 && args.weightsBuf && lane < 8) {
        float value = 0;
        for (int p = 0; p < 4; ++p)
          if (weights[p]) value += weights[p][lane];
        args.shmemCombineOutWeightsMemObj->template GetAs<float*>()[size_t(token) * 8 + lane] =
            value;
      }
    } else {
      if (start == 0 && args.weightsBuf)
        for (int k = lane; k < topk; k += warpSize) {
          float value = 0;
          for (int p = 0; p < 4; ++p)
            if (weights[p]) value += weights[p][k];
          args.shmemCombineOutWeightsMemObj->template GetAs<float*>()[size_t(token) * topk + k] =
              value;
        }
    }
  }
  Ep4Complete(args, generation + 1);
}

}  // namespace moe
}  // namespace mori

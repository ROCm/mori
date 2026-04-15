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
#include "src/ops/dispatch_combine/low_latency_async.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <type_traits>

#include "mori/core/core.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/profiler/profiler.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

/* ---------------------------------------------------------------------------------------------- */
/*                               EpDispatchLowLatencyAsyncSendCopy                                */
/* ---------------------------------------------------------------------------------------------- */

template <typename T>
__device__ void EpDispatchLowLatencyAsyncSendCopy_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::DispatchAsyncSendCopy);
  for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
       i += globalWarpNum) {
    index_t srcTokId = i / config.numExpertPerToken;
    index_t destExpert = args.tokenIndices[i];
    index_t destPe = destExpert / config.numExpertPerRank;
    index_t destTokId = 0;

    // Deduplicate
    assert(config.numExpertPerToken < warpSize);
    int condition = 0;
    if (laneId < (i % config.numExpertPerToken)) {
      condition = destPe == (args.tokenIndices[srcTokId * config.numExpertPerToken + laneId] /
                             config.numExpertPerRank);
    }
    if (__any(condition)) {
      // Indicate that this token is already sent to the destination PE by setting an overflow
      // token index
      if (laneId == 0) args.dispDestTokIdMap[i] = NullSendBufSlotOffset(config);
      continue;
    }

    if (laneId == 0) {
      // decide token id in dest pe
      destTokId = atomicAdd(args.destPeTokenCounter + destPe, 1);
      args.dispDestTokIdMap[i] = SendBufSlotOffset(config, destPe, destTokId);
    }
    destTokId = __shfl(destTokId, 0);

    index_t destTokOffset = SendBufSlotOffset(config, destPe, destTokId);

    uint8_t* stagingPtr = args.interNodeTokBufs.staging->template GetAs<uint8_t*>();
    size_t stagingTokOffset = destTokOffset * xferBytes;
    core::WarpCopy<uint8_t, 4>(
        stagingPtr + stagingTokOffset,
        reinterpret_cast<uint8_t*>(args.inpTokenBuf) + srcTokId * hiddenBytes, hiddenBytes);
    core::WarpCopy<uint8_t, 4>(
        stagingPtr + stagingTokOffset + hiddenBytes,
        reinterpret_cast<uint8_t*>(args.tokenIndices) + srcTokId * indexBytes, indexBytes);
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes,
                               reinterpret_cast<uint8_t*>(args.weightsBuf) + srcTokId * weightBytes,
                               weightBytes);
    if (args.scalesBuf && (scaleBytes > 0))
      core::WarpCopy<uint8_t, 4>(
          stagingPtr + stagingTokOffset + hiddenBytes + indexBytes + weightBytes,
          reinterpret_cast<uint8_t*>(args.scalesBuf) + srcTokId * scaleBytes, scaleBytes);
    if (laneId == 0) {
      reinterpret_cast<index_t*>(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes +
                                 weightBytes + scaleBytes)[0] =
          srcTokId + config.rank * config.maxNumInpTokenPerRank;
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                     EpDispatchLowLatencyAsyncSendCopy — Two-Phase (Multi-Block)                */
/* ---------------------------------------------------------------------------------------------- */

// Phase 1: Lightweight slot assignment using warp shuffle for dedup.
// Groups of numExpertPerToken threads within a warp cooperate on one token.
// Each thread loads its own expert's destPe, then uses __shfl to read previous
// experts' destPe for dedup — no extra memory loads needed.
template <typename T>
__device__ void EpDispatchLowLatencyAsyncSendCopySlotAssign_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::DispatchAsyncSlotAssign);
  int numEpt = config.numExpertPerToken;
  int tokensPerWarp = warpSize / numEpt;
  int expertIdx = laneId % numEpt;
  int inWarpTokIdx = laneId / numEpt;
  int baseLane = inWarpTokIdx * numEpt;

  for (int warpTokBase = globalWarpId * tokensPerWarp; warpTokBase < args.curRankNumToken;
       warpTokBase += globalWarpNum * tokensPerWarp) {
    int tokenId = warpTokBase + inWarpTokIdx;
    if (tokenId >= args.curRankNumToken) continue;

    int i = tokenId * numEpt + expertIdx;
    index_t destExpert = args.tokenIndices[i];
    index_t destPe = destExpert / config.numExpertPerRank;

    // Deduplicate via shuffle: read previous experts' destPe from registers.
    // All threads loop the same number of iterations to keep __shfl uniform.
    bool isDuplicate = false;
    for (int j = 0; j < numEpt - 1; ++j) {
      index_t prevPe = __shfl(destPe, baseLane + j);
      if (j < expertIdx) {
        isDuplicate = isDuplicate || (prevPe == destPe);
      }
    }

    if (isDuplicate) {
      args.dispDestTokIdMap[i] = NullSendBufSlotOffset(config);
    } else {
      index_t destTokId = atomicAdd(args.destPeTokenCounter + destPe, 1);
      args.dispDestTokIdMap[i] = SendBufSlotOffset(config, destPe, destTokId);
    }
  }
  if (globalThdId == 0) args.totalRecvTokenNum[0] = 0;
}

// Phase 2: Data copy using pre-computed slot assignments from dispDestTokIdMap.
// Multiple warps cooperate to copy a single token for higher bandwidth.
template <typename T>
__device__ void EpDispatchLowLatencyAsyncSendCopyMultiBlock_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::DispatchAsyncSendCopyMultiBlock);
  index_t totalEntries = args.curRankNumToken * config.numExpertPerToken;
  index_t warpsPerToken = (globalWarpNum + totalEntries - 1) / totalEntries;
  index_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~(size_t)15;

  for (int i = globalWarpId; i < totalEntries * warpsPerToken; i += globalWarpNum) {
    index_t entryId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;

    index_t destTokOffset = args.dispDestTokIdMap[entryId];

    // Skip deduplicated (overflow) entries
    if (destTokOffset >= NullSendBufSlotOffset(config)) continue;

    index_t srcTokId = entryId / config.numExpertPerToken;

    uint8_t* stagingPtr = args.interNodeTokBufs.staging->template GetAs<uint8_t*>();
    uint8_t* dst = stagingPtr + destTokOffset * xferBytes;

    // Each sub-warp copies its portion of hidden bytes
    size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenOffset,
          reinterpret_cast<uint8_t*>(args.inpTokenBuf) + srcTokId * hiddenBytes + hiddenOffset,
          len);
    }

    // First sub-warp handles metadata copies
    if (inTokenPartId == 0) {
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenBytes, reinterpret_cast<uint8_t*>(args.tokenIndices) + srcTokId * indexBytes,
          indexBytes);
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenBytes + indexBytes,
          reinterpret_cast<uint8_t*>(args.weightsBuf) + srcTokId * weightBytes, weightBytes);
      if (args.scalesBuf && (scaleBytes > 0))
        core::WarpCopy<uint8_t, 1>(
            dst + hiddenBytes + indexBytes + weightBytes,
            reinterpret_cast<uint8_t*>(args.scalesBuf) + srcTokId * scaleBytes, scaleBytes);
      if (laneId == 0) {
        reinterpret_cast<index_t*>(dst + hiddenBytes + indexBytes + weightBytes + scaleBytes)[0] =
            FlatTokenIndex(config, myPe, srcTokId);
        // srcTokId + config.rank * config.maxNumInpTokenPerRank;
      }
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                             EpDispatchLowLatencyAsyncSendTransfer                              */
/* ---------------------------------------------------------------------------------------------- */

template <typename T>
__device__ void EpDispatchLowLatencyAsyncSendTransfer_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::DispatchAsyncSendTransfer);
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
      int tokenChunkNum = core::CeilDiv(tokenNum, config.numQpPerPe);
      int thisChunkTokenNum = std::min(tokenChunkNum, tokenNum - qpId * tokenChunkNum);
      size_t remoteOffset = SendBufSlotOffset(config, myPe, tokenChunkNum * qpId) * xferBytes;
      size_t localOffset = SendBufSlotOffset(config, destPe, tokenChunkNum * qpId) * xferBytes;
      if ((destPe != myPe) && (laneId == 0) && (thisChunkTokenNum > 0)) {
        shmem::ShmemPutMemNbiThread(args.interNodeTokBufs.dispatchInp, remoteOffset,
                                    args.interNodeTokBufs.staging, localOffset,
                                    thisChunkTokenNum * xferBytes, destPe, qpId);
      }
      // TODO(ditian12): index value is wrong if signal completion here, investigate the reason
      // shmem::ShmemAtomicTypeNonFetchWarp<uint64_t>(
      //     args.recvTokenNumMemObj, (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
      //     static_cast<uint64_t>(tokenNum + 1), core::AMO_ADD, destPe, qpId);
    }
  }

  if (globalThdId == 0) args.totalRecvTokenNum[0] = 0;
}

/* ---------------------------------------------------------------------------------------------- */
/*                             EpDispatchLowLatencyAsyncRecvTransfer                              */
/* ---------------------------------------------------------------------------------------------- */

template <typename T>
__device__ void EpDispatchLowLatencyAsyncRecvTransfer_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::DispatchAsyncRecvTransfer);
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      if (laneId == 0) {
        if (destPe / config.gpuPerNode == myNode && config.enableSdma) {
          shmem::ShmemQuietThreadKernel<application::TransportType::SDMA>(
              destPe, args.interNodeTokBufs.dispatchInp);
        } else {
          shmem::ShmemQuietThread(destPe, qpId);
        }
        int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
        shmem::ShmemPutUint64ImmNbiThread(args.recvTokenNumMemObj,
                                          (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
                                          static_cast<uint64_t>(tokenNum + 1), destPe, qpId);
      }
    }
  }

  // Polling recv token number signal
  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (laneId < config.numQpPerPe) {
      shmem::ShmemUint64WaitUntilGreaterThan(recvTokenNums + destPe * config.numQpPerPe + laneId,
                                             0);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpDispatchLowLatencyAsyncRecvCopyMultiBlock                            */
/* ---------------------------------------------------------------------------------------------- */

// Prefix-sum variant: eliminates atomicAdd on totalRecvTokenNum by computing
// per-PE offsets via warp-shuffle prefix sum and per-warp offsets arithmetically.
template <typename T>
__device__ void EpDispatchLowLatencyAsyncRecvCopyMultiBlock_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::DispatchAsyncRecvCopy);

  int blocksPerPe = blockNum / npes;
  int destPe = blockId / blocksPerPe;

  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();

  // Step 1: Parallel load + warp-shuffle prefix sum for per-PE offsets
  uint64_t myPeTokens = 0;
  if (laneId < npes) {
    myPeTokens = recvTokenNums[laneId * config.numQpPerPe] - 1;
  }
  uint64_t inclusive = myPeTokens;
  for (int offset = 1; offset < npes; offset <<= 1) {
    uint64_t n = __shfl_up(inclusive, offset);
    if (laneId >= offset) inclusive += n;
  }
  uint64_t peExclusive = __shfl_up(inclusive, 1);
  if (laneId == 0) peExclusive = 0;

  uint64_t peOffset = __shfl(peExclusive, destPe);
  uint64_t recvTokenNum = __shfl(myPeTokens, destPe);
  uint64_t totalTokens = __shfl(inclusive, npes - 1);
  if (laneId == 0) {
    // NOTE: totalTokens is a count number here, so use <=
    assert(totalTokens <= config.MaxNumTokensToRecv() &&
           "Total recv token overflow: increase maxTotalRecvTokens");
  }

  // Copy data — multiple warps cooperate per token
  uint8_t* stagingPtr = (destPe != myPe)
                            ? args.interNodeTokBufs.dispatchInp->template GetAs<uint8_t*>()
                            : args.interNodeTokBufs.staging->template GetAs<uint8_t*>();
  stagingPtr += SendBufSlotOffset(config, destPe, 0) * xferBytes;

  int peWarps = blocksPerPe * warpNum;
  int localWarpId = (blockId % blocksPerPe) * warpNum + warpId;
  index_t warpsPerToken = (recvTokenNum > 0) ? (peWarps + recvTokenNum - 1) / recvTokenNum : 1;
  size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~(size_t)15;

  for (int i = localWarpId; i < recvTokenNum * warpsPerToken; i += peWarps) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t destTokId = peOffset + tokenId;

    // Each sub-warp copies its portion of hidden bytes
    size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      core::WarpCopy<uint8_t, 1>(args.interNodeTokBufs.dispatchOut->template GetAs<uint8_t*>() +
                                     destTokId * hiddenBytes + hiddenOffset,
                                 stagingPtr + tokenId * xferBytes + hiddenOffset, len);
    }

    // First sub-warp handles metadata and validation
    if (inTokenPartId == 0) {
      if (laneId < config.numExpertPerToken) {
        index_t id =
            reinterpret_cast<index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes)[laneId];
        index_t pe = id / config.numExpertPerRank;
        if (!((pe >= 0) && (pe < config.worldSize))) {
          assert((pe >= 0) && (pe < config.worldSize));
        }
      }
      core::WarpCopy<uint8_t, 1>(
          args.shmemOutIndicesMemObj->template GetAs<uint8_t*>() + destTokId * indexBytes,
          stagingPtr + tokenId * xferBytes + hiddenBytes, indexBytes);
      core::WarpCopy<uint8_t, 1>(
          args.shmemDispatchOutWeightsMemObj->template GetAs<uint8_t*>() + destTokId * weightBytes,
          stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes, weightBytes);
      if (scaleBytes > 0) {
        core::WarpCopy<uint8_t, 1>(
            args.shmemOutScalesMemObj->template GetAs<uint8_t*>() + destTokId * scaleBytes,
            stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes + weightBytes, scaleBytes);
      }
      if (laneId == 0) {
        // A map used to recover token ordering at combine send phase
        args.dispReceiverIdxMap[destTokId] = SendBufSlotOffset(config, destPe, tokenId);
        // A map used for unit test correctness check
        args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[destTokId] =
            reinterpret_cast<index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes +
                                       weightBytes + scaleBytes)[0];
      }
    }
  }

  if (globalWarpId == 0) {
    if (laneId == 0) {
      args.totalRecvTokenNum[0] = totalTokens;
    }
    if (laneId < npes) {
      args.destPeTokenCounter[laneId] = 0;
    }
    if (laneId == 0) {
      args.dispatchGridBarrier[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                EpCombineLowLatencyAsyncSendCopy                                */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool UseFp8DirectCast>
__device__ void EpCombineLowLatencyAsyncSendCopy_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::CombineAsyncSendCopy);
  using TokT = std::conditional_t<UseFp8DirectCast, core::CombineInternalFp8, T>;
  static_assert(!UseFp8DirectCast || std::is_same_v<T, hip_bfloat16>,
                "Fp8 direct cast combine currently only supports bf16 input");
  const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);

  // Copy token onto staging buffer for later IBGDA transfer
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  uint8_t* stagingPtr = args.interNodeTokBufs.staging->template GetAs<uint8_t*>();
  for (int tokenId = globalWarpId; tokenId < totalRecvTokenNum; tokenId += globalWarpNum) {
    index_t stagingTokId = 0;
    if (laneId == 0) stagingTokId = args.dispReceiverIdxMap[tokenId];
    stagingTokId = __shfl(stagingTokId, 0);
    if constexpr (UseFp8DirectCast) {
      core::WarpCastBf16ToCombineInternalFp8<T>(
          reinterpret_cast<TokT*>(stagingPtr + stagingTokId * tokHiddenBytes),
          args.inpTokenBuf + tokenId * hiddenDim, hiddenDim, laneId);
    } else {
      core::WarpCopy<uint8_t, 4>(
          stagingPtr + stagingTokId * tokHiddenBytes,
          reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * tokHiddenBytes, tokHiddenBytes);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                              EpCombineLowLatencyAsyncSendTransfer                              */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool UseFp8DirectCast>
__device__ void EpCombineLowLatencyAsyncSendTransfer_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::CombineAsyncSendTransfer);
  using TokT = std::conditional_t<UseFp8DirectCast, core::CombineInternalFp8, T>;
  static_assert(!UseFp8DirectCast || std::is_same_v<T, hip_bfloat16>,
                "Fp8 direct cast combine currently only supports bf16 input");
  const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);

  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      int tokenNum = 0;
      if (laneId == 0) {
        tokenNum = recvTokenNums[destPe * config.numQpPerPe + qpId] - 1;
        core::AtomicStoreRelaxedSystem(&recvTokenNums[destPe * config.numQpPerPe + qpId],
                                       uint64_t{0});
      }
      tokenNum = __shfl(tokenNum, 0);
      int tokenChunkNum = core::CeilDiv(tokenNum, config.numQpPerPe);
      int thisChunkTokenNum = std::min(tokenChunkNum, tokenNum - qpId * tokenChunkNum);
      size_t remoteOffset = SendBufSlotOffset(config, myPe, tokenChunkNum * qpId) * tokHiddenBytes;
      size_t localOffset = SendBufSlotOffset(config, destPe, tokenChunkNum * qpId) * tokHiddenBytes;
      if ((destPe != myPe) && (laneId == 0) && (thisChunkTokenNum > 0))
        shmem::ShmemPutMemNbiThread(args.interNodeTokBufs.combineInp, remoteOffset,
                                    args.interNodeTokBufs.staging, localOffset,
                                    thisChunkTokenNum * tokHiddenBytes, destPe, qpId);
      // if (laneId == 0)
      // shmem::ShmemQuietThread(destPe, qpId);
      // shmem::ShmemAtomicTypeNonFetchWarp<uint64_t>(
      //     args.crossDeviceBarrierMemObj, myPe * sizeof(uint64_t), 1, core::AMO_ADD, destPe,
      //     qpId);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                              EpCombineLowLatencyAsyncRecvTransfer                              */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool UseFp8DirectCast>
__device__ void EpCombineLowLatencyAsyncRecvTransfer_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::CombineAsyncRecvTransfer);
  using TokT = std::conditional_t<UseFp8DirectCast, core::CombineInternalFp8, T>;
  static_assert(!UseFp8DirectCast || std::is_same_v<T, hip_bfloat16>,
                "Fp8 direct cast combine currently only supports bf16 input");
  (void)sizeof(TokT);

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      if (laneId == 0) {
        if (destPe / config.gpuPerNode == myNode && config.enableSdma) {
          shmem::ShmemQuietThreadKernel<application::TransportType::SDMA>(
              destPe, args.interNodeTokBufs.combineInp);
        } else {
          shmem::ShmemQuietThread(destPe, qpId);
        }
        uint64_t flag = args.crossDeviceBarrierFlag[0];
        shmem::ShmemPutUint64ImmNbiThread(args.crossDeviceBarrierMemObj,
                                          (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
                                          flag, destPe, qpId);
      }
    }
  }

  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint64_t barrierFlag = args.crossDeviceBarrierFlag[0];
    for (int i = 0; i < config.numQpPerPe; i++) {
      shmem::ShmemUint64WaitUntilEquals(args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>() +
                                            destPe * config.numQpPerPe + i,
                                        barrierFlag);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                EpCombineLowLatencyAsyncRecvCopy                                */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool UseFp8DirectCast>
__device__ void EpCombineLowLatencyAsyncRecvCopy_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  LOW_LATENCY_ASYNC_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId);
  MORI_TRACE_SPAN(profiler, Slot::CombineAsyncRecvCopy);
  using TokT = std::conditional_t<UseFp8DirectCast, core::CombineInternalFp8, T>;
  static_assert(!UseFp8DirectCast || std::is_same_v<T, hip_bfloat16>,
                "Fp8 direct cast combine currently only supports bf16 input");

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;

  if (args.curRankNumToken != 0) {
    MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

    for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem);
         i += globalWarpNum) {
      int tokenId, inTokenPartId;
      size_t hiddenDimOffset, hiddenDimSize;
      mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

      for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
        index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
        index_t destPe = PeFromSendBufSlotOffset(config, destTokId);

        TokT* stagingPtr = (destPe != myPe)
                               ? args.interNodeTokBufs.combineInp->template GetAs<TokT*>()
                               : args.interNodeTokBufs.staging->template GetAs<TokT*>();
        if (destPe < npes) {
          srcPtrs[j] = stagingPtr + destTokId * hiddenDim + hiddenDimOffset;
        } else {
          srcPtrs[j] = nullptr;
        }
      }

      T* outPtr = args.interNodeTokBufs.combineOut->template GetAs<T*>() + tokenId * hiddenDim +
                  hiddenDimOffset;
      if constexpr (UseFp8DirectCast) {
        core::WarpAccumCombineInternalFp8ToBf16(outPtr,
                                                reinterpret_cast<const TokT* const*>(srcPtrs),
                                                config.numExpertPerToken, laneId, hiddenDimSize);
      } else {
        core::WarpAccum<T, 4>(outPtr, srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);
      }
    }
  }
}

}  // namespace moe
}  // namespace mori

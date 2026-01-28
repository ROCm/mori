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

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

/* ---------------------------------------------------------------------------------------------- */
/*                                  EpDispatchLowLatencyAsyncSend                                 */
/* ---------------------------------------------------------------------------------------------- */

template <typename T>
__global__ void EpDispatchLowLatencyAsyncSend(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
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
      if (laneId == 0)
        args.dispDestTokIdMap[i] = config.worldSize * config.MaxNumTokensToSendPerRank();
      continue;
    }

    if (laneId == 0) {
      // decide token id in dest pe
      destTokId = atomicAdd(args.destPeTokenCounter + destPe, 1);
      args.dispDestTokIdMap[i] = destTokId + config.MaxNumTokensToSendPerRank() * destPe;
    }
    destTokId = __shfl(destTokId, 0);

    index_t destTokOffset = destTokId + config.MaxNumTokensToSendPerRank() * destPe;

    uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
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
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      if (laneId == 0) shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
      int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
      int tokenChunkNum = core::CeilDiv(tokenNum, config.numQpPerPe);
      int thisChunkTokenNum = std::min(tokenChunkNum, tokenNum - qpId * tokenChunkNum);
      size_t remoteOffset =
          (config.MaxNumTokensToSendPerRank() * myPe + tokenChunkNum * qpId) * xferBytes;
      size_t localOffset =
          (config.MaxNumTokensToSendPerRank() * destPe + tokenChunkNum * qpId) * xferBytes;
      if (destPe != myPe)
        shmem::ShmemPutMemNbiWarp(args.shmemDispatchInpTokMemObj, remoteOffset,
                                  args.shmemStagingTokMemObj, localOffset,
                                  thisChunkTokenNum * xferBytes, destPe, qpId);
      // TODO(ditian12): index value is wrong if signal completion here, investigate the reason
      // shmem::ShmemAtomicTypeNonFetchWarp<uint64_t>(
      //     args.recvTokenNumMemObj, (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
      //     static_cast<uint64_t>(tokenNum + 1), core::AMO_ADD, destPe, qpId);
    }
  }
  if (globalThdId == 0) args.totalRecvTokenNum[0] = 0;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                  EpDispatchLowLatencyAsyncRecv                                 */
/* ---------------------------------------------------------------------------------------------- */

template <typename T>
__global__ void EpDispatchLowLatencyAsyncRecv(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  int blocksPerPe = blockNum / npes;
  int destPe = blockId / blocksPerPe;

  // TODO(ditian12): index value is wrong when signal completion at send phase, hence we signal at
  // recv phase as a workaround at the cost of extra latency, we should still investigate the reason
  if ((blockId % blocksPerPe) == 0) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
      if (laneId == 0) shmem::ShmemQuietThread(destPe, qpId);
      shmem::ShmemAtomicTypeNonFetchWarp<uint64_t>(
          args.recvTokenNumMemObj, (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
          static_cast<uint64_t>(tokenNum + 1), core::AMO_ADD, destPe, qpId);
    }
  }

  // Polling recv token number signal
  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  uint64_t recvTokenNum = 0;
  if (laneId < config.numQpPerPe) {
    recvTokenNum = shmem::ShmemUint64WaitUntilGreaterThan(
                       recvTokenNums + destPe * config.numQpPerPe + laneId, 0) -
                   1;
  }
  recvTokenNum = __shfl(recvTokenNum, 0);

  // Copy data
  uint8_t* stagingPtr = (destPe != myPe)
                            ? args.shmemDispatchInpTokMemObj->template GetAs<uint8_t*>()
                            : args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
  stagingPtr += (config.MaxNumTokensToSendPerRank() * destPe) * xferBytes;

  for (int tokenId = (blockId % blocksPerPe) * warpNum + warpId; tokenId < recvTokenNum;
       tokenId += blocksPerPe * warpNum) {
    index_t destTokId = 0;
    if (laneId == 0) destTokId = atomicAdd(args.totalRecvTokenNum, 1);
    destTokId = __shfl(destTokId, 0);
    core::WarpCopy<uint8_t, 4>(
        args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>() + destTokId * hiddenBytes,
        stagingPtr + tokenId * xferBytes, hiddenBytes);
    // if (laneId < config.numExpertPerToken) {
    //   index_t id =
    //       reinterpret_cast<index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes)[laneId];
    //   index_t pe = id / config.numExpertPerRank;
    //   if (!((pe >= 0) && (pe < config.worldSize))) {
    //     printf("myPe %d destPe %d get topk id %d %d %lu %d\n", myPe, destPe, id, pe,
    //     recvTokenNum,
    //            destTokId);
    //     assert((pe >= 0) && (pe < config.worldSize));
    //   }
    // }
    core::WarpCopy<uint8_t, 4>(
        args.shmemOutIndicesMemObj->template GetAs<uint8_t*>() + destTokId * indexBytes,
        stagingPtr + tokenId * xferBytes + hiddenBytes, indexBytes);
    core::WarpCopy<uint8_t, 4>(
        args.shmemDispatchOutWeightsMemObj->template GetAs<uint8_t*>() + destTokId * weightBytes,
        stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes, weightBytes);
    if (scaleBytes > 0) {
      core::WarpCopy<uint8_t, 4>(
          args.shmemOutScalesMemObj->template GetAs<uint8_t*>() + destTokId * scaleBytes,
          stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes + weightBytes, scaleBytes);
    }
    if (laneId == 0) {
      // A map used to recover token ordering at combine send phase
      args.dispReceiverIdxMap[destTokId] = config.MaxNumTokensToSendPerRank() * destPe + tokenId;
      // A map used for unit test correctness check
      args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[destTokId] =
          reinterpret_cast<index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes +
                                     weightBytes + scaleBytes)[0];
    }
  }

  uint32_t finishedWarpNum = 0;
  if (laneId == 0) {
    finishedWarpNum = atomicAdd(args.dispatchGridBarrier, 1);
  }
  finishedWarpNum = __shfl(finishedWarpNum, 0);

  if ((finishedWarpNum == (2 * globalWarpNum - 1)) && (laneId < npes)) {
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
/*                                  EpCombineLowLatencyAsyncRecv                                  */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineLowLatencyAsyncSend(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  // Copy token onto staing buffer for later IBGDA transfer
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
  for (int tokenId = globalWarpId; tokenId < totalRecvTokenNum; tokenId += globalWarpNum) {
    index_t stagingTokId = 0;
    if (laneId == 0) stagingTokId = args.dispReceiverIdxMap[tokenId];
    stagingTokId = __shfl(stagingTokId, 0);
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokId * hiddenBytes,
                               reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes,
                               hiddenBytes);
  }
  if (laneId == 0) {
    atomicAdd(args.combineGridBarrier, 1);
  }

  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      int tokenNum = 0;
      if (laneId == 0) {
        shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
        tokenNum = recvTokenNums[destPe * config.numQpPerPe + qpId];
        core::AtomicStoreRelaxedSystem(&recvTokenNums[destPe * config.numQpPerPe + qpId],
                                       uint64_t{0});
      }
      tokenNum = __shfl(tokenNum, 0);
      int tokenChunkNum = core::CeilDiv(tokenNum, config.numQpPerPe);
      int thisChunkTokenNum = std::min(tokenChunkNum, tokenNum - qpId * tokenChunkNum);
      size_t remoteOffset =
          (config.MaxNumTokensToSendPerRank() * myPe + tokenChunkNum * qpId) * hiddenBytes;
      size_t localOffset =
          (config.MaxNumTokensToSendPerRank() * destPe + tokenChunkNum * qpId) * hiddenBytes;
      if (destPe != myPe)
        shmem::ShmemPutMemNbiWarp(args.shmemCombineInpTokMemObj, remoteOffset,
                                  args.shmemStagingTokMemObj, localOffset,
                                  thisChunkTokenNum * hiddenBytes, destPe, qpId);
      // if (laneId == 0)
      // shmem::ShmemQuietThread(destPe, qpId);
      // shmem::ShmemAtomicTypeNonFetchWarp<uint64_t>(
      //     args.crossDeviceBarrierMemObj, myPe * sizeof(uint64_t), 1, core::AMO_ADD, destPe,
      //     qpId);
    }
  }
}

template <typename T>
__global__ void EpCombineLowLatencyAsyncRecv(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
      if (laneId == 0) shmem::ShmemQuietThread(destPe, qpId);
      shmem::ShmemAtomicTypeNonFetchWarp<uint64_t>(
          args.crossDeviceBarrierMemObj, myPe * sizeof(uint64_t), 1, core::AMO_ADD, destPe, qpId);
    }
  }

  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint32_t barrierFlag = args.crossDeviceBarrierFlag[0];
    shmem::ShmemUint64WaitUntilEquals(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>() + destPe,
        barrierFlag * config.numQpPerPe);
  }

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;

  if (args.curRankNumToken != 0) {
    index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
    index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

    for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
      index_t tokenId = i / warpsPerToken;
      index_t inTokenPartId = i % warpsPerToken;
      index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
      index_t hiddenDimSize =
          std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

      for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
        index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
        index_t destPe = destTokId / config.MaxNumTokensToSendPerRank();

        T* stagingPtr = (destPe != myPe) ? args.shmemCombineInpTokMemObj->template GetAs<T*>()
                                         : args.shmemStagingTokMemObj->template GetAs<T*>();
        if (destPe < npes) {
          srcPtrs[j] = stagingPtr + destTokId * config.hiddenDim + hiddenDimOffset;
        } else {
          srcPtrs[j] = nullptr;
        }
      }

      core::WarpAccum<T, 4>(args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                                tokenId * config.hiddenDim + hiddenDimOffset,
                            srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);
    }
  }

  uint32_t finishedWarpNum = 0;
  if (laneId == 0) finishedWarpNum = atomicAdd(args.combineGridBarrier, 1);
  finishedWarpNum = __shfl(finishedWarpNum, 0);

  if (finishedWarpNum == (2 * globalWarpNum - 1)) {
    if (laneId == 0) {
      args.combineGridBarrier[0] = 0;
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Template Specialization                                    */
/* ---------------------------------------------------------------------------------------------- */
template __global__ void EpDispatchLowLatencyAsyncSend<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpDispatchLowLatencyAsyncSend<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpDispatchLowLatencyAsyncSend<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpDispatchLowLatencyAsyncSend<float>(EpDispatchCombineArgs<float> args);

template __global__ void EpDispatchLowLatencyAsyncRecv<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpDispatchLowLatencyAsyncRecv<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpDispatchLowLatencyAsyncRecv<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpDispatchLowLatencyAsyncRecv<float>(EpDispatchCombineArgs<float> args);

template __global__ void EpCombineLowLatencyAsyncSend<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpCombineLowLatencyAsyncSend<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpCombineLowLatencyAsyncSend<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpCombineLowLatencyAsyncSend<float>(EpDispatchCombineArgs<float> args);

template __global__ void EpCombineLowLatencyAsyncRecv<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpCombineLowLatencyAsyncRecv<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpCombineLowLatencyAsyncRecv<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpCombineLowLatencyAsyncRecv<float>(EpDispatchCombineArgs<float> args);

}  // namespace moe
}  // namespace mori

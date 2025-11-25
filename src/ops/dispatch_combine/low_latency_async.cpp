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
      if (laneId == 0) args.dispDestTokIdMap[i] = config.worldSize * config.MaxNumTokensToSend();
      continue;
    }

    if (laneId == 0) {
      // decide token id in dest pe
      destTokId = atomicAdd(args.destPeTokenCounter + destPe, 1);
      args.dispDestTokIdMap[i] = destPe * config.MaxNumTokensToSend() + destTokId;
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

  for (int destPe = blockId; (destPe < npes) && (warpId == 0); destPe += blockNum) {
    if (laneId == 0) shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
    int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
    size_t remoteOffset = (config.MaxNumTokensToSendPerRank() * myPe) * xferBytes;
    size_t localOffset = (config.MaxNumTokensToSendPerRank() * destPe) * xferBytes;
    if (destPe != myPe)
      shmem::ShmemPutMemNbiWarp(args.shmemDispatchInpTokMemObj, remoteOffset,
                                args.shmemStagingTokMemObj, localOffset, tokenNum * xferBytes,
                                destPe);
    shmem::ShmemPutInt32ImmNbiWarp(args.recvTokenNumMemObj, myPe * sizeof(index_t), tokenNum + 1,
                                   destPe);
    // shmem::ShmemPutMemNbiSignalWarp(args.shmemDispatchInpTokMemObj, remoteOffset,
    //                                 args.shmemStagingTokMemObj, localOffset, tokenNum *
    //                                 xferBytes, args.recvTokenNumMemObj, myPe * sizeof(index_t),
    //                                 tokenNum + 1, core::atomicType::AMO_SET, destPe);
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

  // Polling recv token number signal
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  index_t recvTokenNum = 0;
  if (laneId == 0) {
    recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(recvTokenNums + destPe, 0) - 1;
  }
  recvTokenNum = __shfl(recvTokenNum, 0);

  // Copy data
  uint8_t* stagingPtr = (destPe != myPe)
                            ? args.shmemDispatchInpTokMemObj->template GetAs<uint8_t*>()
                            : args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
  // uint8_t* stagingPtr = args.shmemDispatchInpTokMemObj->template GetAs<uint8_t*>();
  stagingPtr += (config.MaxNumTokensToSendPerRank() * destPe) * xferBytes;

  for (int tokenId = (blockId % blocksPerPe) * warpNum + warpId; tokenId < recvTokenNum;
       tokenId += blocksPerPe * warpNum) {
    index_t destTokId = 0;
    if (laneId == 0) destTokId = atomicAdd(args.totalRecvTokenNum, 1);
    destTokId = __shfl(destTokId, 0);
    core::WarpCopy<uint8_t, 4>(
        args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>() + destTokId * hiddenBytes,
        stagingPtr + tokenId * xferBytes, hiddenBytes);
    core::WarpCopy<uint8_t, 4>(
        args.shmemOutIndicesMemObj->template GetAs<uint8_t*>() + destTokId * indexBytes,
        stagingPtr + tokenId * xferBytes + hiddenBytes, indexBytes);
    core::WarpCopy<uint8_t, 4>(
        args.shmemDispatchOutWeightsMemObj->template GetAs<uint8_t*>() + destTokId * weightBytes,
        stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes, weightBytes);
    if (args.scalesBuf && (scaleBytes > 0)) {
      core::WarpCopy<uint8_t, 4>(
          args.shmemOutScalesMemObj->template GetAs<uint8_t*>() + destTokId * scaleBytes,
          stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes + weightBytes, scaleBytes);
    }
    if (laneId == 0) {
      // printf("myPe %d destPe %d token id %d src token id %d\n", myPe, destPe, destTokId,
      // reinterpret_cast<index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes +
      //                              weightBytes + scaleBytes)[0]);
      args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[destTokId] =
          reinterpret_cast<index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes +
                                     weightBytes + scaleBytes)[0];
    }
  }

  // Last warp to reset states
  uint32_t finishedWarpNum = 0;
  if (laneId == 0) {
    finishedWarpNum = atomicAdd(args.dispatchGridBarrier, 1);
  }
  finishedWarpNum = __shfl(finishedWarpNum, 0);

  if ((finishedWarpNum == (2 * globalWarpNum - 1)) && (laneId < npes)) {
    if (laneId < npes) {
      recvTokenNums[laneId] = 0;
      args.destPeTokenCounter[laneId] = 0;
    }
    if (laneId == 0) {
      args.dispatchGridBarrier[0] = 0;
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

}  // namespace moe
}  // namespace mori

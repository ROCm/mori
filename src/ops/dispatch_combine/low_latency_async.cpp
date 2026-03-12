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
__global__ void EpDispatchLowLatencyAsyncCopy(EpDispatchCombineArgs<T> args) {
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
  if (globalThdId == 0) args.totalRecvTokenNum[0] = 0;
}

template <typename T>
__global__ void EpDispatchLowLatencyAsyncDataTransfer(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
      int tokenChunkNum = core::CeilDiv(tokenNum, config.numQpPerPe);
      int thisChunkTokenNum = std::min(tokenChunkNum, tokenNum - qpId * tokenChunkNum);
      size_t remoteOffset =
          (config.MaxNumTokensToSendPerRank() * myPe + tokenChunkNum * qpId) * xferBytes;
      size_t localOffset =
          (config.MaxNumTokensToSendPerRank() * destPe + tokenChunkNum * qpId) * xferBytes;

      if ((destPe != myPe) && (laneId == 0)) {
        shmem::ShmemPutMemNbiThread(args.shmemDispatchInpTokMemObj, remoteOffset,
                                    args.shmemStagingTokMemObj, localOffset,
                                    thisChunkTokenNum * xferBytes, destPe, qpId);
      }
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                  EpDispatchLowLatencyAsyncRecv                                 */
/* ---------------------------------------------------------------------------------------------- */

template <typename T>
__global__ void EpDispatchLowLatencyAsyncRecvDataTransfer(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      if (laneId == 0) {
        if (args.enableSDMA && (destPe / config.gpuPerNode == myNode)) {
          shmem::ShmemQuietThreadKernel<application::TransportType::SDMA>(
              destPe, args.shmemDispatchInpTokMemObj);
        } else {
          shmem::ShmemQuietThread(destPe, qpId);
        }
        int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
        // TODO(ditian12): send atomic op right after quiet lead to hang issue, need to investigate
        // shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
        //     args.recvTokenNumMemObj, (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
        //     static_cast<uint64_t>(tokenNum + 1), core::AMO_ADD, destPe, qpId);
        shmem::ShmemPutUint64ImmNbiThread(args.recvTokenNumMemObj,
                                          (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
                                          static_cast<uint64_t>(tokenNum + 1), destPe, qpId);
      }
    }
  }
  // Polling recv token number signal
  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    if (laneId < config.numQpPerPe) {
      shmem::ShmemUint64WaitUntilGreaterThan(recvTokenNums + destPe * config.numQpPerPe + laneId,
                                             0);
    }
  }
}

template <typename T>
__global__ void EpDispatchLowLatencyAsyncRecvCopy(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  int blocksPerPe = blockNum / npes;
  int destPe = blockId / blocksPerPe;

  uint64_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint64_t*>();
  uint64_t recvTokenNum = recvTokenNums[destPe * config.numQpPerPe] - 1;

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
    if (laneId < config.numExpertPerToken) {
      index_t id =
          reinterpret_cast<index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes)[laneId];
      index_t pe = id / config.numExpertPerRank;
      if (!((pe >= 0) && (pe < config.worldSize))) {
        assert((pe >= 0) && (pe < config.worldSize));
      }
    }
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

  if (globalWarpId == 0) {
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
template <typename T, bool UseFp8DirectCast>
__global__ void EpCombineLowLatencyAsyncCopy(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  using TokT = std::conditional_t<UseFp8DirectCast, core::CombineInternalFp8, T>;
  static_assert(!UseFp8DirectCast || std::is_same_v<T, hip_bfloat16>,
                "Fp8 direct cast combine currently only supports bf16 input");
  const size_t tokHiddenBytes = config.hiddenDim * sizeof(TokT);

  // Copy token onto staging buffer for later IBGDA transfer
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
  for (int tokenId = globalWarpId; tokenId < totalRecvTokenNum; tokenId += globalWarpNum) {
    index_t stagingTokId = 0;
    if (laneId == 0) stagingTokId = args.dispReceiverIdxMap[tokenId];
    stagingTokId = __shfl(stagingTokId, 0);
    if constexpr (UseFp8DirectCast) {
      core::WarpCastBf16ToCombineInternalFp8<T>(
          reinterpret_cast<TokT*>(stagingPtr + stagingTokId * tokHiddenBytes),
          args.inpTokenBuf + tokenId * config.hiddenDim, config.hiddenDim, laneId);
    } else {
      core::WarpCopy<uint8_t, 4>(
          stagingPtr + stagingTokId * tokHiddenBytes,
          reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * tokHiddenBytes, tokHiddenBytes);
    }
  }
}

template <typename T, bool UseFp8DirectCast>
__global__ void EpCombineLowLatencyAsyncDataTransfer(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  using TokT = std::conditional_t<UseFp8DirectCast, core::CombineInternalFp8, T>;
  static_assert(!UseFp8DirectCast || std::is_same_v<T, hip_bfloat16>,
                "Fp8 direct cast combine currently only supports bf16 input");
  const size_t tokHiddenBytes = config.hiddenDim * sizeof(TokT);

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
      size_t remoteOffset =
          (config.MaxNumTokensToSendPerRank() * myPe + tokenChunkNum * qpId) * tokHiddenBytes;
      size_t localOffset =
          (config.MaxNumTokensToSendPerRank() * destPe + tokenChunkNum * qpId) * tokHiddenBytes;
      if ((destPe != myPe) && (laneId == 0) && (thisChunkTokenNum > 0))
        shmem::ShmemPutMemNbiThread(args.shmemCombineInpTokMemObj, remoteOffset,
                                    args.shmemStagingTokMemObj, localOffset,
                                    thisChunkTokenNum * tokHiddenBytes, destPe, qpId);
    }
  }
}

template <typename T, bool UseFp8DirectCast>
__global__ void EpCombineLowLatencyAsyncRecvDataTransfer(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    for (int qpId = warpId; qpId < config.numQpPerPe; qpId += warpNum) {
      if (laneId == 0) {
        if (args.enableSDMA && (destPe / config.gpuPerNode == myNode)) {
          shmem::ShmemQuietThreadKernel<application::TransportType::SDMA>(
              destPe, args.shmemCombineInpTokMemObj);
        } else {
          shmem::ShmemQuietThread(destPe, qpId);
        }
        // TODO(ditian12): send atomic op right after quiet lead to hang issue, need to investigate
        // shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
        // args.crossDeviceBarrierMemObj, myPe * sizeof(uint64_t), 1, core::AMO_ADD, destPe, qpId);
        uint64_t flag = args.crossDeviceBarrierFlag[0];
        shmem::ShmemPutUint64ImmNbiThread(args.crossDeviceBarrierMemObj,
                                          (myPe * config.numQpPerPe + qpId) * sizeof(uint64_t),
                                          flag, destPe, qpId);
      }
    }
  }
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    uint64_t barrierFlag = args.crossDeviceBarrierFlag[0];
    if (laneId < config.numQpPerPe)
      shmem::ShmemUint64WaitUntilEquals(args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>() +
                                            destPe * config.numQpPerPe + laneId,
                                        barrierFlag);
  }
}

template <typename T, bool UseFp8DirectCast>
__global__ void EpCombineLowLatencyAsyncRecvCopy(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  using TokT = std::conditional_t<UseFp8DirectCast, core::CombineInternalFp8, T>;
  static_assert(!UseFp8DirectCast || std::is_same_v<T, hip_bfloat16>,
                "Fp8 direct cast combine currently only supports bf16 input");

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
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

        TokT* stagingPtr = (destPe != myPe) ? args.shmemCombineInpTokMemObj->template GetAs<TokT*>()
                                            : args.shmemStagingTokMemObj->template GetAs<TokT*>();
        if (destPe < npes) {
          srcPtrs[j] = stagingPtr + destTokId * config.hiddenDim + hiddenDimOffset;
        } else {
          srcPtrs[j] = nullptr;
        }
      }

      T* outPtr = args.shmemCombineOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim +
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

/* ---------------------------------------------------------------------------------------------- */
/*                                     Template Specialization                                    */
/* ---------------------------------------------------------------------------------------------- */
// Helper macros for conditional FP8 compilation
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
#define MORI_FP8_FNUZ(...) __VA_ARGS__
#else
#define MORI_FP8_FNUZ(...)
#endif

#ifdef MORI_FP8_TYPE_OCP_ENABLED
#define MORI_FP8_OCP(...) __VA_ARGS__
#else
#define MORI_FP8_OCP(...)
#endif

#if defined(MORI_FP8_TYPE_OCP_ENABLED) || defined(MORI_FP8_TYPE_FNUZ_ENABLED)
#define MORI_FP8_ANY(...) __VA_ARGS__
#else
#define MORI_FP8_ANY(...)
#endif

// Macro to instantiate async kernels for all data types
#define INSTANTIATE_ASYNC_KERNEL(KernelName)                                                   \
  template __global__ void KernelName<hip_bfloat16>(EpDispatchCombineArgs<hip_bfloat16> args); \
  MORI_FP8_FNUZ(template __global__ void KernelName<__hip_fp8_e4m3_fnuz>(                      \
                    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);)                         \
  MORI_FP8_OCP(template __global__ void KernelName<__hip_fp8_e4m3>(                            \
                   EpDispatchCombineArgs<__hip_fp8_e4m3> args);)                               \
  template __global__ void KernelName<mori_fp4x2_e2m1>(                                        \
      EpDispatchCombineArgs<mori_fp4x2_e2m1> args);                                            \
  template __global__ void KernelName<float>(EpDispatchCombineArgs<float> args);

// Macro to instantiate async combine kernels (includes optional bf16->fp8 direct-cast path)
#define INSTANTIATE_ASYNC_COMBINE_KERNEL(KernelName)                                           \
  template __global__ void KernelName<hip_bfloat16>(EpDispatchCombineArgs<hip_bfloat16> args); \
  MORI_FP8_ANY(template __global__ void KernelName<hip_bfloat16, true>(                        \
                   EpDispatchCombineArgs<hip_bfloat16> args);)                                 \
  MORI_FP8_FNUZ(template __global__ void KernelName<__hip_fp8_e4m3_fnuz>(                      \
                    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);)                         \
  MORI_FP8_OCP(template __global__ void KernelName<__hip_fp8_e4m3>(                            \
                   EpDispatchCombineArgs<__hip_fp8_e4m3> args);)                               \
  template __global__ void KernelName<mori_fp4x2_e2m1>(                                        \
      EpDispatchCombineArgs<mori_fp4x2_e2m1> args);                                            \
  template __global__ void KernelName<float>(EpDispatchCombineArgs<float> args);

INSTANTIATE_ASYNC_KERNEL(EpDispatchLowLatencyAsyncCopy)
INSTANTIATE_ASYNC_KERNEL(EpDispatchLowLatencyAsyncDataTransfer)
INSTANTIATE_ASYNC_KERNEL(EpDispatchLowLatencyAsyncRecvDataTransfer)
INSTANTIATE_ASYNC_KERNEL(EpDispatchLowLatencyAsyncRecvCopy)
INSTANTIATE_ASYNC_COMBINE_KERNEL(EpCombineLowLatencyAsyncCopy)
INSTANTIATE_ASYNC_COMBINE_KERNEL(EpCombineLowLatencyAsyncDataTransfer)
INSTANTIATE_ASYNC_COMBINE_KERNEL(EpCombineLowLatencyAsyncRecvDataTransfer)
INSTANTIATE_ASYNC_COMBINE_KERNEL(EpCombineLowLatencyAsyncRecvCopy)

#undef MORI_FP8_FNUZ
#undef MORI_FP8_OCP
#undef MORI_FP8_ANY
#undef INSTANTIATE_ASYNC_KERNEL
#undef INSTANTIATE_ASYNC_COMBINE_KERNEL

}  // namespace moe
}  // namespace mori

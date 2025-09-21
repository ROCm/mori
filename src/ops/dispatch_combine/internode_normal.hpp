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
#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {
/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchInterNodeNormalKernel                             */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchInterNodeNormalKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  // TODO check if need sync
  if (!(args.tokenIndices && args.inpTokenBuf)) {
    return;
  }
  const int thdId = threadIdx.x;
  const int thdNum = blockDim.x;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int blockId = blockIdx.x;
  const int blockNum = gridDim.x;

  const int myPe = config.rank;
  const int npes = config.worldSize;
  constexpr int nlocalPes = MAX_GPUS_PER_NODE;
  const int myLocalPe = config.rank % nlocalPes;
  const int myNode = myPe / nlocalPes;
  const int nNodes = npes / nlocalPes;

  const index_t tokenNum = args.curRankNumToken;
  const int numExpertPerToken = config.numExpertPerToken;
  const size_t tokenBytes = config.hiddenDim * sizeof(T);
  const size_t weightBytes = args.weightsBuf ? sizeof(float) * numExpertPerToken : 0;
  const size_t indiceBytes = sizeof(index_t) * numExpertPerToken;
  const size_t scaleBytes = args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)
                                ? config.scaleTypeSize * config.scaleDim
                                : 0;
  const size_t tokenPackBytes = tokenBytes + weightBytes + indiceBytes + scaleBytes;

  const int nChannels = blockNum / 2;
  const int isSender = blockId < nChannels;
  const int channelId = blockId % nChannels;

  const size_t maxNumInpTokenPerRank = config.maxNumInpTokenPerRank;
  index_t baseTokensPerChannel = tokenNum / nChannels;
  index_t remTokens = tokenNum % nChannels;
  const index_t channelStartOffset = channelId * baseTokensPerChannel + min(channelId, remTokens);
  const index_t tokensPerChannel = baseTokensPerChannel + (channelId < remTokens ? 1 : 0);
  const index_t channelEndOffset = channelStartOffset + tokensPerChannel;
  // TODO modify maxTokensPerChannel 目前是channel所需的最大size，可以根据实际改小
  const size_t maxTokensPerChannel = baseTokensPerChannel + 1;
  const size_t maxNumToken = maxTokensPerChannel * nChannels;
  // constexpr int maxNumRDMASendTokens = 30;
  // TODO maxRDMAStagingTokens是RDMA从某一个node收发数据的最大值；maxP2PStagingTokens是转发数据上限
  const size_t maxRDMAStagingTokens = maxTokensPerChannel;
  const size_t maxP2PStagingTokens = maxTokensPerChannel * nNodes;
  const int maxNumRDMASendTokens = maxNumInpTokenPerRank;

  // for (int tokenIdx = channelStartOffset + warpId; tokenIdx < channelEndOffset;
  //      tokenIdx += warpNum) {
  // }
  // TODO uodate localPeBuf size
  // For sender
  index_t* tokenIdxToSlotMap = reinterpret_cast<index_t*>(args.localPeBuf);
  index_t* slotToTokenIdxMap =
      reinterpret_cast<index_t*>(reinterpret_cast<char*>(tokenIdxToSlotMap) +
                                 maxNumInpTokenPerRank * nNodes * sizeof(index_t));
  // For Reveiver
  index_t* rdmaRecvTokensNum = reinterpret_cast<index_t*>(
      reinterpret_cast<char*>(slotToTokenIdxMap) + nNodes * maxNumToken * sizeof(index_t));

  if (isSender) {
    constexpr int kSendWarpCount = 15;
    __shared__ volatile index_t tokenProgress[kSendWarpCount + 1];
    // __shared__ volatile size_t slotProgress[MAX_NODES][kSendWarpCount+1];
    

    if (warpId == kSendWarpCount) {
      if (laneId < kSendWarpCount + 1) tokenProgress[laneId] = -1;
      // TODO maybe change to warp barrier
      __syncthreads();
      __shared__ index_t nodeTokenCount[MAX_NODES];
      if (laneId < nNodes) nodeTokenCount[laneId] = 0;
      for (int tokenIdx = channelStartOffset; tokenIdx < channelEndOffset; ++tokenIdx) {
        index_t destNode = -1;
        // int tmp = myPe;
        if (laneId < numExpertPerToken) {
          index_t destExpert = args.tokenIndices[tokenIdx * numExpertPerToken + laneId];
          index_t destPe = destExpert / config.numExpertPerRank;
          destNode = destPe / nlocalPes;

          unsigned long long dupMask = __match_any(destNode);
          bool isFirst = (dupMask & ((1ULL << laneId) - 1)) == 0;
          if (isFirst) {
            index_t slot = nodeTokenCount[destNode];
            tokenIdxToSlotMap[tokenIdx * nNodes + destNode] = slot + 1;
            slotToTokenIdxMap[(channelId * nNodes + destNode) * maxTokensPerChannel + slot] =
                tokenIdx;
            // slotProgress[kSendWarpCount] = destNode * maxTokensPerChannel + slot;
            ++nodeTokenCount[destNode];
          }

          // if (tokenIdx == 0) {
          //   while (tmp < npes) {
          //     // 空循环，什么都不做
          //   }
          // }
        }
        tokenProgress[kSendWarpCount] = tokenIdx;
      }

      // if (laneId == 0) {
      //   for (int i = 0; i < nNodes; ++i) {
      //     printf("rank=%d warpId=%d nodeTokenCount[%d]=%d tokenProgress[15]=%d\n", myPe, warpId, i,
      //            nodeTokenCount[i], tokenProgress[kSendWarpCount]);
      //   }
      // }

      for (int i = 0; i < nNodes; ++i) {
        int destNode = (myNode + i) % nNodes;
        int destPe = destNode * nlocalPes + myLocalPe;

        if (laneId == 0) {
          shmem::ShmemPutInt32ImmNbiThread(args.recvTokenNumMemObj,
                                           (myNode + channelId * nNodes) * sizeof(index_t),
                                           nodeTokenCount[destNode] + 1, destPe);
        }
      }

      index_t numTokensToSend = laneId < nNodes ? nodeTokenCount[laneId] : 0;
      index_t sendSlotStart = 0;
      uint64_t tailCache =
          laneId < nNodes ? args.localTail[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;
      while (__any(numTokensToSend > 0)) {
        for (int i = 0; i < nNodes; ++i) {
          int destNode = (myNode + i) % nNodes;
          int destPe = destNode * nlocalPes + myLocalPe;
          index_t destNodeNumTokensToSend = __shfl(numTokensToSend, destNode);
          if (destNodeNumTokensToSend == 0) continue;

          index_t destNodeSendSlotStart = __shfl(sendSlotStart, destNode);
          // TODO modify maxNumRDMASendTokens
          index_t sendTokenNum =
              min(maxNumRDMASendTokens, destNodeNumTokensToSend - destNodeSendSlotStart);
          index_t lastTokenIdx = slotToTokenIdxMap[destNodeSendSlotStart + sendTokenNum - 1];
          while (true) {
            bool dataReady = laneId < kSendWarpCount ? tokenProgress[laneId] >= lastTokenIdx : true;
            if (__all(dataReady)) break;
          }

          size_t srcStagingOffset = 0, dstStagingOffset = 0;
          if (laneId == destNode) {
            srcStagingOffset = ((channelId * nNodes + destNode) * maxRDMAStagingTokens +
                                tailCache % maxRDMAStagingTokens) *
                               tokenPackBytes;
            dstStagingOffset = ((channelId * nNodes + myNode) * maxRDMAStagingTokens +
                                tailCache % maxRDMAStagingTokens) *
                               tokenPackBytes;
          }
          srcStagingOffset = __shfl(srcStagingOffset, destNode);
          dstStagingOffset = __shfl(dstStagingOffset, destNode);
          if (destNode == myNode) {
            core::WarpCopy(args.shmemInpTokMemObj->template GetAs<char*>(destPe) + dstStagingOffset,
                           args.shmemStagingTokMemObj->template GetAs<char*>() + srcStagingOffset,
                           sendTokenNum * tokenPackBytes);
          } else {
            shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, dstStagingOffset,
                                                args.shmemStagingTokMemObj, srcStagingOffset,
                                                sendTokenNum * tokenPackBytes, destPe);
            // shmem::ShmemPutTypeNbiThread<uint8_t>(args.shmemInpTokMemObj, dstStagingOffset,
            //                                       args.shmemStagingTokMemObj, srcStagingOffset,
            //                                       sendTokenNum * tokenPackBytes, destPe);
          }
          if (laneId == destNode) {
            numTokensToSend -= sendTokenNum;
            sendSlotStart += sendTokenNum;
            tailCache += sendTokenNum;
            assert(numTokensToSend >= 0);

            // TODO use amo (amo hang)
            if (destNode == myNode) {
              // *(args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes) =
              //     tailCache;
              __hip_atomic_store(
                  args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes,
                  tailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
              if (laneId == 0)
                printf("rank=%d warpId=%d destPe=%d rdma tail=%lu\n", myPe, warpId, destPe,
                       *(args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe +
                         channelId * npes));
            } else {
              shmem::ShmemPutUint64ImmNbiThread(
                  args.tailMemObj, (myPe + channelId * npes) * sizeof(uint64_t), tailCache, destPe);
            }
          }
        }
      }
      if (laneId < nNodes) {
        args.localTail[(laneId * nlocalPes + myLocalPe) + channelId * npes] = tailCache;
        printf("rank=%d warpId=%d peerRank=%d args.localTail=%lu\n", myPe, warpId,
               laneId * nlocalPes + myLocalPe, tailCache);
      }
    } else if (warpId < kSendWarpCount) {
      __syncthreads();
      uint64_t tailCache[MAX_NODES] = {};
      for (int i = 0; i < nNodes; ++i) {
        tailCache[i] = args.localTail[(i * nlocalPes + myLocalPe) + channelId * npes];
      }
      uint64_t headCache = 0;
      for (int tokenIdx = channelStartOffset + warpId; tokenIdx < channelEndOffset;
           tokenIdx += (kSendWarpCount - 1)) {
        while (tokenIdx > tokenProgress[kSendWarpCount]);

        for (int node = 0; node < nNodes; ++node) {
          index_t slot = tokenIdxToSlotMap[tokenIdx * nNodes + node] - 1;
          // TODO clear tokenIdxToSlotMap after every call
          if (slot == -1) continue;

          // TODO check buffer ready
          size_t tail = tailCache[node] + slot;
          if (laneId == node) {
            uint64_t* headPtr = args.headMemObj->template GetAs<uint64_t*>() + channelId * npes +
                                (node * nlocalPes + myLocalPe);
            while (tail - headCache >= maxRDMAStagingTokens) {
              headCache = __hip_atomic_load(headPtr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
          }

          // TODO modify shmemStagingTokMemObj size
          size_t stagingOffset =
              ((channelId * nNodes + node) * maxRDMAStagingTokens + tail % maxRDMAStagingTokens) *
              tokenPackBytes;
          core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + stagingOffset,
                         reinterpret_cast<char*>(args.inpTokenBuf) + tokenIdx * tokenBytes,
                         tokenBytes);
          stagingOffset += tokenBytes;
          if (weightBytes) {
            core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + stagingOffset,
                           reinterpret_cast<char*>(args.weightsBuf) + tokenIdx * weightBytes,
                           weightBytes);
            stagingOffset += weightBytes;
          }
          core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + stagingOffset,
                         reinterpret_cast<char*>(args.tokenIndices) + tokenIdx * indiceBytes,
                         indiceBytes);
          stagingOffset += indiceBytes;
          if (scaleBytes) {
            core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + stagingOffset,
                           reinterpret_cast<char*>(args.scalesBuf) + tokenIdx * scaleBytes,
                           scaleBytes);
          }
        }
        tokenProgress[warpId] = tokenIdx;
      }
      tokenProgress[warpId] = channelEndOffset - 1;
    }
  } else {  // Receiver
    constexpr int kfwdWarpCount = 8;
    __shared__ volatile index_t fwdHead[MAX_GPUS_PER_NODE][MAX_NODES];
    __shared__ volatile int fwdFinish;

    if (warpId < kfwdWarpCount) {
      if (warpId == 0) fwdHead[laneId / MAX_NODES][laneId % MAX_NODES] = 0;
      if (laneId == 0) fwdFinish = 0;
      const int fwdLocalPe = warpId;
      const int fwdPe = myNode * MAX_GPUS_PER_NODE + fwdLocalPe;
      index_t numTokensToRecv = 0;
      if (laneId < nNodes) {
        index_t* signal =
            args.recvTokenNumMemObj->template GetAs<index_t*>() + laneId + channelId * nNodes;
        numTokensToRecv = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
        rdmaRecvTokensNum[channelId * nNodes + laneId] = numTokensToRecv;
      }
      __syncthreads();

      uint64_t headBase =
          laneId < nNodes ? args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;
      uint64_t headCache =
          laneId < nNodes ? args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;
      uint64_t tailCache = 0;

      uint64_t p2pHeadCache = 0;
      uint64_t p2pTailCache = __hip_atomic_load(
          args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe,
          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      int srcNode = myNode;
      while (__any(numTokensToRecv > 0)) {
        srcNode = (srcNode - 1 + nNodes) % nNodes;
        int srcPe = srcNode * nlocalPes + myLocalPe;
        index_t destNodeNumTokensToRecv = __shfl(numTokensToRecv, srcNode);
        if (destNodeNumTokensToRecv == 0) continue;

        // check rdma data ready
        if (laneId == srcNode) {
          while (true) {
            tailCache = __hip_atomic_load(
                args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            if (tailCache >= headCache + 1) break;
          }
        }
        headCache = __shfl(headCache, srcNode);
        tailCache = __shfl(tailCache, srcNode);

        // check buffer ready
        if (laneId == 0 && srcNode != myNode) {
          while (true) {
            p2pHeadCache = __hip_atomic_load(
                args.headMemObj->template GetAs<uint64_t*>() + channelId * npes + fwdPe,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            if (p2pTailCache - p2pHeadCache < maxP2PStagingTokens) break;
          }
        }

        for (uint64_t i = headCache; i < tailCache; ++i) {
          if (srcNode == laneId) --numTokensToRecv;
          index_t slot = i % maxRDMAStagingTokens;
          char* srcPtr = args.shmemInpTokMemObj->template GetAs<char*>() +
                         (channelId * nNodes + srcNode) * maxRDMAStagingTokens +
                         slot * tokenPackBytes;
          index_t* srcIndicesPtr = reinterpret_cast<index_t*>(srcPtr + tokenBytes + weightBytes);
          int destPe = -1;
          if (laneId < numExpertPerToken) destPe = srcIndicesPtr[laneId] / config.numExpertPerRank;
          if (!__any(destPe == fwdPe)) {
            continue;
          }

          if (fwdPe == myPe) {
            index_t localTokenIdx = 0;
            if (laneId == 0) {
              localTokenIdx = atomicAdd(args.localPeTokenCounter, 1);
            }
            localTokenIdx = __shfl(localTokenIdx, 0);
            // TODO abstrct into a function
            char* dstPtr = reinterpret_cast<char*>(args.outTokenBuf) + localTokenIdx * tokenBytes;
            core::WarpCopy(dstPtr, srcPtr, tokenBytes);
            srcPtr += tokenBytes;
            if (weightBytes) {
              core::WarpCopy(
                  args.shmemOutWeightsMemObj->template GetAs<char*>() + localTokenIdx * weightBytes,
                  srcPtr, weightBytes);
              srcPtr += weightBytes;
            }
            core::WarpCopy(
                args.shmemOutIndicesMemObj->template GetAs<char*>() + localTokenIdx * indiceBytes,
                srcPtr, indiceBytes);
            srcPtr += indiceBytes;
            if (scaleBytes) {
              core::WarpCopy(
                  args.shmemOutScalesMemObj->template GetAs<char*>() + localTokenIdx * scaleBytes,
                  srcPtr, scaleBytes);
            }
          } else {
            char* dstPtr = args.shmemOutTokMemObj->template GetAs<char*>() +
                           ((channelId * nlocalPes + fwdLocalPe) * maxP2PStagingTokens +
                            p2pTailCache % maxP2PStagingTokens) *
                               tokenPackBytes;
            core::WarpCopy(dstPtr, srcPtr, tokenPackBytes);
          }
          ++p2pTailCache;
        }
        if (laneId == 0 && fwdPe != myPe) {
          __hip_atomic_store(
              args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe,
              p2pTailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          printf("rank=%d warpId=%d fwdPe=%d p2p tail=%lu\n", myPe, warpId, fwdPe,
                 *(args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe));
        }
        if (laneId == srcNode) {
          headCache = tailCache;
          fwdHead[fwdLocalPe][srcNode] = headCache - headBase;
        }
      }
      if (fwdLocalPe != myLocalPe) {
        constexpr int fwdDone = 1;
        int* statusPtr = args.intraNodeBarrierMemObj->template GetAs<int*>(fwdPe) + myPe;
        shmem::ShmemInt32WaitUntilEquals(statusPtr, 0);
        core::AtomicStoreRelaxedSystem(statusPtr, fwdDone);
        // if(laneId == 0) printf("rank=%d warpId=%d fwdPe=%d\n", myPe, warpId, fwdPe);
      }
    } else if (warpId == kfwdWarpCount) {
      __syncthreads();
      index_t numTokensToRecv =
          laneId < nNodes ? rdmaRecvTokensNum[channelId * nNodes + laneId] : 0;
      uint64_t headBase =
          laneId < nNodes ? args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;

      index_t curMinHead = 0;
      while (__any(numTokensToRecv > 0)) {
        if (laneId < nNodes) {
          index_t minHead = fwdHead[0][laneId];
          for (int i = 1; i < MAX_GPUS_PER_NODE; ++i) {
            minHead = min(minHead, fwdHead[i][laneId]);
          }

          // TODO use amo (amo hang)
          if (minHead > curMinHead) {
            shmem::ShmemPutUint64ImmNbiThread(args.headMemObj,
                                              (myPe + channelId * npes) * sizeof(uint64_t),
                                              headBase + minHead, laneId * nlocalPes + myLocalPe);
            numTokensToRecv -= (minHead - curMinHead);
            curMinHead = minHead;
            assert(numTokensToRecv >= 0);
          }
        }
      }

      if (laneId < nNodes) {
        args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] = headBase + curMinHead;
      }

    } else {  // copy from shmemOutTokMemObj to output
      __syncthreads();
      int srcLocalPe = warpId - (kfwdWarpCount + 1);
      // skip self PE
      if (srcLocalPe >= myLocalPe) ++srcLocalPe;
      int srcPe = myNode * MAX_GPUS_PER_NODE + srcLocalPe;

      uint64_t p2pHeadCache = __hip_atomic_load(
          args.headMemObj->template GetAs<uint64_t*>(srcPe) + channelId * npes + myPe,
          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      uint64_t p2pTailCache = 0;

      int done = 0;
      int* statusPtr = args.intraNodeBarrierMemObj->template GetAs<int*>() + srcPe;
      while (true) {
        done = __shfl(done, 0);
        if (done) {
          core::AtomicStoreRelaxedSystem(statusPtr, 0);
          break;
        }

        // check data ready
        if (laneId == 0) {
          while (true) {
            p2pTailCache = __hip_atomic_load(
                args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            done = core::AtomicLoadRelaxedSystem(statusPtr);
            if (p2pTailCache >= p2pHeadCache + 1 || done) break;
          }
        }
        p2pTailCache = __shfl(p2pTailCache, 0);

        for (uint64_t i = p2pHeadCache; i < p2pTailCache; ++i) {
          index_t slot = i % maxP2PStagingTokens;
          char* srcPtr =
              args.shmemOutTokMemObj->template GetAs<char*>(srcPe) +
              ((channelId * nlocalPes + myLocalPe) * maxP2PStagingTokens + slot) * tokenPackBytes;

          index_t localTokenIdx = 0;
          if (laneId == 0) {
            localTokenIdx = atomicAdd(args.localPeTokenCounter, 1);
          }
          localTokenIdx = __shfl(localTokenIdx, 0);

          // TODO use args.outTokenBuf
          core::WarpCopy(reinterpret_cast<char*>(args.outTokenBuf) + localTokenIdx * tokenBytes,
                         srcPtr, tokenBytes);
          srcPtr += tokenBytes;
          if (weightBytes) {
            core::WarpCopy(
                args.shmemOutWeightsMemObj->template GetAs<char*>() + localTokenIdx * weightBytes,
                srcPtr, weightBytes);
            srcPtr += weightBytes;
          }
          core::WarpCopy(
              args.shmemOutIndicesMemObj->template GetAs<char*>() + localTokenIdx * indiceBytes,
              srcPtr, indiceBytes);
          srcPtr += indiceBytes;
          if (scaleBytes) {
            core::WarpCopy(
                args.shmemOutScalesMemObj->template GetAs<char*>() + localTokenIdx * scaleBytes,
                srcPtr, scaleBytes);
          }
        }
        if (laneId == 0) {
          p2pHeadCache = p2pTailCache;
          *(args.headMemObj->template GetAs<uint64_t*>(srcPe) + channelId* npes + myPe) = p2pHeadCache;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeNormalKernel                              */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineIntraNodeNormalKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
}

}  // namespace moe
}  // namespace mori
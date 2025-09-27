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
#define DEBUG 0
#define DEBUG_DATA 0
#define SPINS_CNT 100000

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
  const int nlocalPes = config.numGPUsPerNode;
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
  const size_t metaBytes = sizeof(size_t);
  const size_t tokenPackBytes = tokenBytes + weightBytes + indiceBytes + scaleBytes + metaBytes;

  const int nChannels = blockNum / 2;
  const int isSender = blockId < nChannels;
  const int channelId = blockId % nChannels;

  const size_t maxNumInpTokenPerRank = config.maxNumInpTokenPerRank;
  index_t baseTokensPerChannel = maxNumInpTokenPerRank / nChannels;
  index_t remTokens = maxNumInpTokenPerRank % nChannels;
  const index_t channelStartOffset = channelId * baseTokensPerChannel + min(channelId, remTokens);
  const index_t tokensPerChannel = baseTokensPerChannel + (channelId < remTokens ? 1 : 0);
  const index_t channelEndOffset = min(channelStartOffset + tokensPerChannel, tokenNum);

  // TODO maxRDMAStagingTokens 是RDMA从某一个node收发数据的最大值；maxP2PStagingTokens是转发数据上限
  // TODO staging buffer每个channel不能小于maxNumRDMASendTokens？
  const int stepRDMATokens = config.maxRDMAStepTokens;
  // if (blockId==0 && thdId == 0) {
  //   printf("nlocalPes=%d stepRDMATokens=%d\n", nlocalPes, stepRDMATokens);
  // }

  // TODO modify maxTokensPerChannel 目前是channel所需的最大size，可以根据实际改小
  const size_t maxTokensPerChannel = max(baseTokensPerChannel + (remTokens ? 1 : 0), stepRDMATokens);
  const size_t maxNumToken = maxTokensPerChannel * nChannels;

  const size_t maxRDMAStagingTokens =
      maxTokensPerChannel / stepRDMATokens * stepRDMATokens;
  const size_t maxRDMASteps = maxRDMAStagingTokens / stepRDMATokens;

  const int maxNumP2pSendTokens = 32;
  const size_t maxP2PStagingTokens = maxTokensPerChannel * nNodes;

  // TODO uodate localPeBuf size
  // For sender
  index_t* tokenIdxToSlotMap = reinterpret_cast<index_t*>(args.localPeBuf);
  index_t* slotToTokenIdxMap =
      reinterpret_cast<index_t*>(reinterpret_cast<char*>(tokenIdxToSlotMap) +
                                 maxNumInpTokenPerRank * nNodes * sizeof(index_t));
  // For Reveiver
  index_t* rdmaRecvTokensNum = reinterpret_cast<index_t*>(
      reinterpret_cast<char*>(slotToTokenIdxMap) + nNodes * maxNumToken * sizeof(index_t));
  slotToTokenIdxMap += channelId * nNodes * maxTokensPerChannel;
  rdmaRecvTokensNum += channelId * nNodes;

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
        if (laneId < numExpertPerToken) {
          index_t destExpert = args.tokenIndices[tokenIdx * numExpertPerToken + laneId];
          index_t destPe = destExpert / config.numExpertPerRank;
          destNode = destPe / nlocalPes;

          unsigned long long dupMask = __match_any(destNode);
          bool isFirst = (dupMask & ((1ULL << laneId) - 1)) == 0;
          if (isFirst) {
            index_t slot = nodeTokenCount[destNode];
            tokenIdxToSlotMap[tokenIdx * nNodes + destNode] = slot + 1;
            slotToTokenIdxMap[destNode * maxTokensPerChannel + slot] = tokenIdx;
            ++nodeTokenCount[destNode];
          }
        }
        tokenProgress[kSendWarpCount] = tokenIdx;
      }
#if DEBUG == 1
      // if (laneId == 0) {
      //   for (int i = 0; i < nNodes; ++i) {
      //     printf("rank=%d warpId=%d nodeTokenCount[%d]=%d tokenProgress[15]=%d\n", myPe, warpId,
      //     i,
      //            nodeTokenCount[i], tokenProgress[kSendWarpCount]);
      //   }
      // }
#endif

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
#if DEBUG == 1
      // if (laneId < nNodes) {
      //   printf("send RDMA before rank=%d call=%d warpId=%d node=%d localTail=%lu\n", myPe,
      //          args.crossDeviceBarrierFlag, warpId, laneId, tailCache);
      // }
#endif
      while (__any(numTokensToSend > 0)) {
        for (int i = 0; i < nNodes; ++i) {
          int destNode = (myNode + i) % nNodes;
          int destPe = destNode * nlocalPes + myLocalPe;
          index_t syncNumTokensToSend = __shfl(numTokensToSend, destNode);
          if (syncNumTokensToSend == 0) continue;

          index_t syncNodeSendSlotStart = __shfl(sendSlotStart, destNode);
          // TODO modify stepRDMATokens
          index_t sendTokenNum = min(stepRDMATokens, syncNumTokensToSend);
          index_t lastTokenIdx = slotToTokenIdxMap[destNode * maxTokensPerChannel +
                                                   syncNodeSendSlotStart + sendTokenNum - 1];
          while (true) {
            bool dataReady = laneId < kSendWarpCount ? tokenProgress[laneId] >= lastTokenIdx : true;
            if (__all(dataReady)) break;
          }

          size_t srcStagingOffset = 0, dstStagingOffset = 0;
          if (laneId == destNode) {
            // for RDMA, tailCache save send step
            srcStagingOffset = ((channelId * nNodes + destNode) * maxRDMAStagingTokens +
                                (tailCache % maxRDMASteps) * stepRDMATokens) *
                               tokenPackBytes;
            dstStagingOffset = ((channelId * nNodes + myNode) * maxRDMAStagingTokens +
                                (tailCache % maxRDMASteps) * stepRDMATokens) *
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
            shmem::ShmemQuietThread();
          }
#if DEBUG_DATA == 1
          if (laneId == 0) {
            // assert(float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
            //                     srcStagingOffset) +
            //                1)) == float(myPe + 1));
            printf("DATA CHECK SEND RDMA PUT destNode=%d step=%lu srcData=%f\n", destNode,
                   tailCache,
                   float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
                                srcStagingOffset))));
          }
#endif
          if (laneId == destNode) {
            numTokensToSend -= sendTokenNum;
            sendSlotStart += sendTokenNum;
            // tailCache += sendTokenNum;
            tailCache += 1;
#if DEBUG == 1
            assert(numTokensToSend >= 0 && sendSlotStart <= nodeTokenCount[destNode]);
#endif

            // Update rdma tail
            // TODO use amo (amo hang)
            if (destNode == myNode) {
              // *(args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes) =
              //     tailCache;
              __hip_atomic_store(
                  args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes,
                  tailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if DEBUG == 1
              // printf(
              //     "send RDMA put rank=%d call=%d warpId=%d destPe=%d numTokensToSend=%d "
              //     "sendSlotStart=%d offset=%d "
              //     "tail=%lu\n",
              //     myPe, args.crossDeviceBarrierFlag, warpId, destPe, numTokensToSend, sendSlotStart,
              //     myPe + channelId * npes,
              //     *(args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes));
#endif
            } else {
              shmem::ShmemPutUint64ImmNbiThread(
                  args.tailMemObj, (myPe + channelId * npes) * sizeof(uint64_t), tailCache, destPe);
              shmem::ShmemQuietThread();
#if DEBUG == 1
              printf(
                  "send RDMA put rank=%d call=%d warpId=%d destPe=%d numTokensToSend=%d "
                  "sendSlotStart=%d offset=%d "
                  "tailCache=%lu\n",
                  myPe, args.crossDeviceBarrierFlag, warpId, destPe, numTokensToSend, sendSlotStart,
                  myPe + channelId * npes, tailCache);
#endif
            }
          }

          // if (destNode != myNode) {
          //   uint64_t syncTailCache = __shfl(tailCache, destNode);
          //   shmem::ShmemPutUint64ImmNbiWarp(args.tailMemObj,
          //                                   (myPe + channelId * npes) * sizeof(uint64_t),
          //                                   syncTailCache, destPe);
          //   if (laneId == warpSize - 1) {
          //     printf("send RDMA put rank=%d call=%d warpId=%d destPe=%d tail=%lu offset=%d\n", myPe,
          //            args.crossDeviceBarrierFlag, warpId, destPe, syncTailCache,
          //            myPe + channelId * npes);
          //   }
          // }

        }
      }
      if (laneId < nNodes) {
        args.localTail[(laneId * nlocalPes + myLocalPe) + channelId * npes] = tailCache;
        __threadfence();
#if DEBUG == 1
        printf("send RDMA after rank=%d warpId=%d peerRank=%d args.localTail=%lu\n", myPe, warpId,
               laneId * nlocalPes + myLocalPe, tailCache);
#endif
      }
      // clear data
      for (int i = laneId; i < nNodes * maxTokensPerChannel; i += warpSize) {
        slotToTokenIdxMap[i] = 0;
      }
    } else if (warpId < kSendWarpCount) {
      __syncthreads();
      uint64_t nodetailCache[MAX_NODES] = {};
      for (int i = 0; i < nNodes; ++i) {
        nodetailCache[i] = args.localTail[(i * nlocalPes + myLocalPe) + channelId * npes];
      }
      uint64_t headCache = 0;
      for (int tokenIdx = channelStartOffset + warpId; tokenIdx < channelEndOffset;
           tokenIdx += (kSendWarpCount - 1)) {
        while (tokenIdx > tokenProgress[kSendWarpCount]);

        for (int node = 0; node < nNodes; ++node) {
          index_t slot = tokenIdxToSlotMap[tokenIdx * nNodes + node] - 1;
          // clear tokenIdxToSlotMap
          if (laneId) tokenIdxToSlotMap[tokenIdx * nNodes + node] = 0;
          if (slot == -1) continue;

          // TODO check RDMA buffer ready (shmemInpTokMemObj)
          uint64_t tailCache = nodetailCache[node] + slot / stepRDMATokens;
          // if (laneId == node) {
          //   while (tailCache - headCache >= maxRDMASteps) {
          //     headCache = __hip_atomic_load(args.headMemObj->template GetAs<uint64_t*>() +
          //                                       channelId * npes + (node * nlocalPes + myLocalPe),
          //                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          //   }
          // }
          int spins = 0;
          while (tailCache - headCache >= maxRDMASteps) {
            if (laneId == node) {
              headCache = __hip_atomic_load(args.headMemObj->template GetAs<uint64_t*>() +
                                                channelId * npes + (node * nlocalPes + myLocalPe),
                                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
              ++spins;
              // if (spins == SPINS_CNT) {
              //   printf(
              //       "send to RDMA staging TIMEOUT rank=%d wait peer=%d call=%d warpId=%d "
              //       "node=%d "
              //       "channelId=%d tailCache=%lu "
              //       "headCache=%lu\n",
              //       myPe, node * nlocalPes + myLocalPe, args.crossDeviceBarrierFlag, warpId, node,
              //       channelId, tailCache, headCache);
              // }
            }
            headCache = __shfl(headCache, node);
          }

          // TODO modify shmemStagingTokMemObj size
          size_t stagingOffset =
              ((channelId * nNodes + node) * maxRDMAStagingTokens +
               (tailCache % maxRDMASteps) * stepRDMATokens + slot % stepRDMATokens) *
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
            stagingOffset += scaleBytes;
          }
          if (laneId == 0) {
            // write meta - record source token info
            *(reinterpret_cast<size_t*>(args.shmemStagingTokMemObj->template GetAs<char*>() +
                                        stagingOffset)) = myPe * maxNumInpTokenPerRank + tokenIdx;
#if DEBUG_DATA == 1
            printf(
                "DATA CHECK SEND copy to RDMA staging node=%d slot=%d tailCache=%lu srcData=%f\n",
                node, slot, tailCache,
                float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
                             ((channelId * nNodes + node) * maxRDMAStagingTokens +
                              tailCache % maxRDMAStagingTokens) *
                                 tokenPackBytes) +
                        0)));
            // assert(float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
            //                     ((channelId * nNodes + node) * maxRDMAStagingTokens +
            //                      tailCache % maxRDMAStagingTokens) *
            //                         tokenPackBytes) +
            //                1)) == (float)(myPe + 1));
#endif
          }
        }
        __threadfence();
        tokenProgress[warpId] = tokenIdx;
      }
      tokenProgress[warpId] = channelEndOffset - 1;
    }
  } else {  // Receiver
    constexpr int kfwdWarpCount = 8;
    __shared__ volatile index_t fwdHead[MAX_GPUS_PER_NODE][MAX_NODES];
    __shared__ volatile int fwdFinish;

    if (warpId < kfwdWarpCount) {  // copy from shmemInpTokMemObj to shmemOutTokMemObj
      if (warpId == 0) fwdHead[laneId / MAX_NODES][laneId % MAX_NODES] = 0;
      if (laneId == 0) fwdFinish = 0;
      const int fwdLocalPe = warpId;
      const int fwdPe = myNode * nlocalPes + fwdLocalPe;
      index_t numTokensToRecv = 0;
      if (laneId < nNodes) {
        index_t* signal =
            args.recvTokenNumMemObj->template GetAs<index_t*>() + laneId + channelId * nNodes;
        numTokensToRecv = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
        rdmaRecvTokensNum[laneId] = numTokensToRecv;
      }
      __syncthreads();
      if (fwdLocalPe >= nlocalPes) {
        __syncthreads();
        return;
      }
      if(warpId == 0) {
        // clear recvTokenNumMemObj
        // TODO maybe hang
        if (laneId < nNodes) {
          index_t* signal =
              args.recvTokenNumMemObj->template GetAs<index_t*>() + laneId + channelId * nNodes;
          core::AtomicStoreRelaxedSystem(signal, 0);
        }
      }

      uint64_t headBase =
          laneId < nNodes ? args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;
      uint64_t headCache = headBase;
      uint64_t tailCache = headCache;

#if DEBUG == 1
      if (laneId < nNodes) {
        printf("recv fwd before rank=%d call=%d warpId=%d node=%d localHead=%lu\n", myPe,
               args.crossDeviceBarrierFlag, warpId, laneId, headCache);
      }
#endif

      uint64_t p2pHeadCache =
          __hip_atomic_load(args.headMemObj->template GetAs<uint64_t*>() + channelId * npes + fwdPe,
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      uint64_t p2pTailCache = __hip_atomic_load(
          args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe,
          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      uint64_t lastP2pTail = p2pTailCache;

      index_t fwdCounter = 0;
      int srcNode = myNode;
      while (__any(numTokensToRecv > 0)) {
        srcNode = (srcNode - 1 + nNodes) % nNodes;
        int srcPe = srcNode * nlocalPes + myLocalPe;
        index_t syncNumTokensToRecv = __shfl(numTokensToRecv, srcNode);
        if (syncNumTokensToRecv == 0) continue;

        // check RDMA data ready (shmemInpTokMemObj), wait data from same-id GPU
#if 0
        if (laneId == srcNode) {
          int spins = 0;
          while (true) {
            if (tailCache >= headCache + 1) break;
            tailCache = __hip_atomic_load(
                args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            // if (spins == SPINS_CNT) {
            //   printf(
            //       "recv fwd wait data TIMEOUT rank=%d srcPe=%d call=%d warpId=%d srcNode=%d "
            //       "syncNumTokensToRecv=%d tailCache=%lu "
            //       "headCache=%lu\n",
            //       myPe, srcPe, args.crossDeviceBarrierFlag, warpId, srcNode, syncNumTokensToRecv,
            //       tailCache, headCache);
            // }
            ++spins;
          }
        }
#else
        uint64_t syncHeadCache = __shfl(headCache, srcNode);
        uint64_t syncTailCache = __shfl(tailCache, srcNode);
        while (syncTailCache < syncHeadCache + 1) {
          if (laneId == srcNode) {
            tailCache = __hip_atomic_load(
                args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
          syncTailCache = __shfl(tailCache, srcNode);
        }
#endif

        // check buffer ready (shmemOutTokMemObj)
#if 0
        if (laneId == 0 && srcNode != myNode) {
          int spins = 0;
          while (true) {
            if (p2pTailCache - p2pHeadCache < maxP2PStagingTokens) break;
            p2pHeadCache = __hip_atomic_load(
                args.headMemObj->template GetAs<uint64_t*>() + channelId * npes + fwdPe,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            // if (spins == SPINS_CNT) {
            //   printf(
            //       "recv fwd wait buffer TIMEOUT rank=%d fwdPe=%d call=%d warpId=%d srcNode=%d "
            //       "channelId=%d p2pTailCache=%lu "
            //       "p2pHeadCache=%lu\n",
            //       myPe, fwdPe, args.crossDeviceBarrierFlag, warpId, srcNode, channelId,
            //       p2pTailCache, p2pHeadCache);
            // }
          }
        }
#else
        if (srcNode != myNode) {
          while (p2pTailCache - p2pHeadCache >= maxP2PStagingTokens) {
            if(laneId == 0) {
              p2pHeadCache = __hip_atomic_load(
                  args.headMemObj->template GetAs<uint64_t*>() + channelId * npes + fwdPe,
                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
            p2pHeadCache = __shfl(p2pHeadCache, 0);
          }
        }
#endif

#if 0
        // uint64_t syncHeadCache = __shfl(headCache, srcNode);
        // uint64_t syncTailCache = __shfl(tailCache, srcNode);
#endif
        for (uint64_t step = syncHeadCache; step < syncTailCache; ++step) {
        uint64_t tokenStart = (syncHeadCache % maxRDMASteps) * stepRDMATokens;
        syncNumTokensToRecv = __shfl(numTokensToRecv, srcNode);
        uint64_t tokenEnd = tokenStart + min(syncNumTokensToRecv, stepRDMATokens);
        for (uint64_t i = tokenStart; i < tokenEnd; ++i) {
          if (srcNode == laneId) {
            --numTokensToRecv;
#if DEBUG == 1
            assert(numTokensToRecv >= 0);
#endif
          }
          index_t slot = i;
          char* srcPtr =
              args.shmemInpTokMemObj->template GetAs<char*>() +
              ((channelId * nNodes + srcNode) * maxRDMAStagingTokens + slot) * tokenPackBytes;
          index_t* srcIndicesPtr = reinterpret_cast<index_t*>(srcPtr + tokenBytes + weightBytes);
          int destPe = -1;
          if (laneId < numExpertPerToken) destPe = srcIndicesPtr[laneId] / config.numExpertPerRank;
          if (!__any(destPe == fwdPe)) {
            continue;
          }

#if DEBUG == 1
          if (laneId == 0) {
            assert(float(*(T*)srcPtr) != 0);
          }
#endif
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
              srcPtr += scaleBytes;
            }
            if (laneId == 0) {
              // get meta
              args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[localTokenIdx] =
                  *(reinterpret_cast<index_t*>(srcPtr));
#if DEBUG_DATA == 1
              // assert(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
              //                     localTokenIdx * tokenBytes))) ==
              //        (float)(*(reinterpret_cast<size_t*>(srcPtr)) / maxNumInpTokenPerRank + 1));
              printf(
                  "DATA CHECK RECV fwd to output srcStep=%lu srcSlot=%d srcData=%f dst(output) "
                  "localTokenIdx=%d\n",
                  step, slot,
                  float(*((T*)(args.shmemInpTokMemObj->template GetAs<char*>() +
                               ((channelId * nNodes + srcNode) * maxRDMAStagingTokens + slot) *
                                   tokenPackBytes))),
                  localTokenIdx);
#endif
            }
          } else {
            char* dstPtr = args.shmemOutTokMemObj->template GetAs<char*>() +
                           ((channelId * nlocalPes + fwdLocalPe) * maxP2PStagingTokens +
                            p2pTailCache % maxP2PStagingTokens) *
                               tokenPackBytes;
            core::WarpCopy(dstPtr, srcPtr, tokenPackBytes);
#if DEBUG_DATA == 1
            printf(
                "DATA CHECK RECV fwd to shmemOutTokMemObj srcStep=%lu srcSlot=%d srcData=%f "
                "dst p2pTailCache=%lu dstSlot=%lu\n",
                step, slot,
                float(*((T*)(args.shmemInpTokMemObj->template GetAs<char*>() +
                             ((channelId * nNodes + srcNode) * maxRDMAStagingTokens + slot) *
                                 tokenPackBytes))),
                p2pTailCache, p2pTailCache % maxP2PStagingTokens);
#endif
          }
          ++p2pTailCache, ++fwdCounter;
        }
        }
        __threadfence_system();
        // TODO add content in comment
        if (laneId == 0 && fwdPe != myPe /*&&
            (numTokensToRecv == 0 || p2pTailCache - lastP2pTail >= maxNumP2pSendTokens)*/) {
          // Update p2p tail
          __hip_atomic_store(
              args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe,
              p2pTailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          lastP2pTail = p2pTailCache;
#if DEBUG == 1
          printf("recv fwd after rank=%d warpId=%d fwdPe=%d p2p tail=%lu\n", myPe, warpId, fwdPe,
                 *(args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe));
#endif
        }
        if (laneId == srcNode) {
          headCache = tailCache;
          fwdHead[fwdLocalPe][srcNode] = headCache - headBase;
        }
      }
      if (fwdLocalPe != myLocalPe) {
        __threadfence_system();
        int* statusPtr =
            args.intraNodeBarrierMemObj->template GetAs<int*>(fwdPe) + myPe + channelId * npes;
        shmem::ShmemInt32WaitUntilEquals(statusPtr, 0);
        core::AtomicStoreRelaxedSystem(statusPtr, fwdCounter + 1);
#if DEBUG==1
        if (laneId == 0) {
          printf("recv fwd rank=%d warpId=%d fwdPe=%d fwdCounter=%d\n", myPe, warpId, fwdPe,
                 fwdCounter);
        }
#endif
      }
    } else if (warpId == kfwdWarpCount) {
      __syncthreads();
      index_t numTokensToRecv =
          laneId < nNodes ? rdmaRecvTokensNum[laneId] : 0;
      uint64_t headBase =
          laneId < nNodes ? args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;
#if DEBUG == 1
      if (laneId < nNodes) {
        printf(
            "recv Ctl before rank=%d call=%d warpId=%d node=%d numTokensToRecv=%d localHead=%lu\n",
            myPe, args.crossDeviceBarrierFlag, warpId, laneId, numTokensToRecv, headBase);
      }
#endif
      index_t curMinHead = 0;
      index_t minHead = 0;
      while (__any(numTokensToRecv > 0)) {
        for (int node = 0; node < nNodes; ++node) {
          if (laneId == node) {
            minHead = fwdHead[0][laneId];
            for (int i = 1; i < nlocalPes; ++i) {
              minHead = min(minHead, fwdHead[i][laneId]);
            }
          }

          index_t syncMinHead = __shfl(minHead, node);
          index_t syncCurMinHead = __shfl(curMinHead, node);
          // TODO use amo (amo hang)
          if (syncMinHead > syncCurMinHead) {
            uint64_t syncHeadBase = __shfl(headBase, node);
            // Update rdma head
            // ShmemPutUint64ImmNbiThread may lead to hang
            shmem::ShmemPutUint64ImmNbiWarp(
                args.headMemObj, (myPe + channelId * npes) * sizeof(uint64_t),
                syncHeadBase + syncMinHead, node * nlocalPes + myLocalPe);
            if (laneId == node) {
              numTokensToRecv -= (minHead - curMinHead) * stepRDMATokens;
              curMinHead = minHead;
#if DEBUG == 1
              assert(numTokensToRecv > -stepRDMATokens);
#endif
            }
          }
        }
      }

      if (laneId < nNodes) {
        args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] = headBase + curMinHead;
#if DEBUG == 1
        printf("recv Ctl rank=%d call=%d warpId=%d node=%d numRecvTokens=%d localHead=%lu\n", myPe,
               args.crossDeviceBarrierFlag, warpId, laneId, rdmaRecvTokensNum[laneId],
               args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes]);
#endif
        // clear rdmaRecvTokensNum
        rdmaRecvTokensNum[laneId] = 0;
      }

    } else {  // copy from shmemOutTokMemObj to output
      __syncthreads();
      int srcLocalPe = warpId - (kfwdWarpCount + 1);
      // skip self PE
      if (srcLocalPe >= myLocalPe) ++srcLocalPe;
      if (srcLocalPe >= nlocalPes) {
        __syncthreads();
        return;
      }
      int srcPe = myNode * nlocalPes + srcLocalPe;

      uint64_t p2pHeadCache = __hip_atomic_load(
          args.headMemObj->template GetAs<uint64_t*>(srcPe) + channelId * npes + myPe,
          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      uint64_t p2pTailCache =
          __hip_atomic_load(args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

      int done = 0;
      int* statusPtr =
          args.intraNodeBarrierMemObj->template GetAs<int*>() + srcPe + channelId * npes;
      while (true) {
        done = __shfl(done, 0);
        if (done) {
          if(laneId == 0) core::AtomicStoreRelaxedSystem(statusPtr, 0);
          break;
        }

        // check data ready (shmemOutTokMemObj)
#if 0
        if (laneId == 0) {
          int spins = 0;
          while (true) {
            if (p2pTailCache >= p2pHeadCache + 1 || done) break;
            p2pTailCache = __hip_atomic_load(
                args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            done = core::AtomicLoadRelaxedSystem(statusPtr);
            // ++spins;
            // if (spins == SPINS_CNT) {
            //   printf(
            //       "recv copy wait data TIMEOUT rank=%d srcPe=%d call=%d warpId=%d "
            //       "channelId=%d p2pTailCache=%lu "
            //       "p2pHeadCache=%lu\n",
            //       myPe, srcPe, args.crossDeviceBarrierFlag, warpId, channelId, p2pTailCache,
            //       p2pHeadCache);
            // }
          }
        }
        p2pTailCache = __shfl(p2pTailCache, 0);
#else
        while (p2pTailCache < p2pHeadCache + 1 && !done) {
          if(laneId == 0){
            p2pTailCache = __hip_atomic_load(
                args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            done = core::AtomicLoadRelaxedSystem(statusPtr);
          }
          p2pTailCache = __shfl(p2pTailCache, 0);
          done = __shfl(done, 0);
        }
#endif
        for (uint64_t i = p2pHeadCache; i < p2pTailCache; ++i) {
          index_t slot = i % maxP2PStagingTokens;
          char* srcPtr =
              args.shmemOutTokMemObj->template GetAs<char*>(srcPe) +
              ((channelId * nlocalPes + myLocalPe) * maxP2PStagingTokens + slot) * tokenPackBytes;

#if DEBUG == 1
          if (laneId == 0) {
            assert(float(*(T*)srcPtr) != 0);
          }
#endif

          index_t localTokenIdx = 0;
          if (laneId == 0) {
            localTokenIdx = atomicAdd(args.localPeTokenCounter, 1);
          }
          localTokenIdx = __shfl(localTokenIdx, 0);

          // TODO use args.outTokenBuf
          core::WarpCopy(reinterpret_cast<char*>(args.outTokenBuf) + localTokenIdx * tokenBytes,
                         srcPtr, tokenBytes);
#if DEBUG_DATA == 1
          if (laneId == 0) {
            printf(
                "DATA CHECK COPY to output srcSlot=%d srcData=%f "
                "dst output localTokenIdx=%d\n",
                slot, (float)(*((T*)srcPtr)), localTokenIdx);
          }
#endif
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
            srcPtr += scaleBytes;
          }
          if (laneId == 0) {
            // get meta
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[localTokenIdx] =
                *(reinterpret_cast<index_t*>(srcPtr));
#if DEBUG_DATA == 1
            assert(float(*((T*)(args.shmemOutTokMemObj->template GetAs<char*>(srcPe) +
                                ((channelId * nlocalPes + myLocalPe) * maxP2PStagingTokens + slot) *
                                    tokenPackBytes) )) ==
                   (float)(*(reinterpret_cast<size_t*>(srcPtr)) / maxNumInpTokenPerRank + 1));
#endif
          }
        }
        p2pHeadCache = p2pTailCache;
        // Update p2p head
        if (laneId == 0) {
          // *(args.headMemObj->template GetAs<uint64_t*>(srcPe) + channelId * npes + myPe) =
          //     p2pHeadCache;
          __hip_atomic_store(
              args.headMemObj->template GetAs<uint64_t*>(srcPe) + channelId * npes + myPe,
              p2pHeadCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
      __threadfence_system();
    }
    __syncthreads();
    shmem::ShmemQuietThread();

    // all recv block barrier
    if (thdId == 0) {
      // TODO check hang
      atomicAdd(args.dispatchGridBarrier, 1);
      while (atomicCAS(args.dispatchGridBarrier, nChannels, 0) != 0);

      if (channelId == 0) {
        // totalRecvTokenNum will be used in GetDispatchSrcTokenId
        *(args.totalRecvTokenNum) = *(args.localPeTokenCounter);
#if DEBUG == 1
        printf("rank=%d totalRecvTokenNum=%d\n", myPe, *args.totalRecvTokenNum);
#endif
        // clear localPeTokenCounter
        *args.localPeTokenCounter = 0;
      }
    }
    __syncthreads();
  }  // Receiver end
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

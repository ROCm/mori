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
#define ASSERT_ON 1
#define SPINS_CNT 100000

__device__ inline float tokenCheckValue(int srcPe, int srcTokenId) {
  return srcPe * 0.1f + 1.0f + srcTokenId;
}

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

  const size_t weightsOffset = tokenBytes;
  const size_t indiceOffset = weightsOffset + weightBytes;
  const size_t scalesOffset = indiceOffset + indiceBytes;
  const size_t metaOffset = scalesOffset + scaleBytes;

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
  const size_t maxTokensPerChannel = baseTokensPerChannel + (remTokens ? 1 : 0);
  const size_t maxNumToken = maxTokensPerChannel * nChannels;

  const size_t maxRDMAStagingTokens =
      ((maxTokensPerChannel + stepRDMATokens - 1) / stepRDMATokens) * stepRDMATokens;
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
        __threadfence_block();
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
      //   printf("send RDMA before rank=%d ch=%d call=%d warpId=%d node=%d localTail=%lu\n", myPe,
      //          channelId, args.crossDeviceBarrierFlag, warpId, laneId, tailCache);
      // }
#endif
      while (__any(numTokensToSend > 0)) {
        for (int i = 0; i < nNodes; ++i) {
          int destNode = (myNode + i) % nNodes;
          int destPe = destNode * nlocalPes + myLocalPe;
          index_t syncNumTokensToSend = __shfl(numTokensToSend, destNode);
          if (syncNumTokensToSend == 0) continue;

          index_t syncSendSlotStart = __shfl(sendSlotStart, destNode);
          // TODO modify stepRDMATokens
          index_t sendTokenNum = min(stepRDMATokens, syncNumTokensToSend);
          index_t lastTokenIdx = slotToTokenIdxMap[destNode * maxTokensPerChannel +
                                                   syncSendSlotStart + sendTokenNum - 1];
          while (true) {
            bool dataReady = laneId < kSendWarpCount ? tokenProgress[laneId] >= lastTokenIdx : true;
            if (__all(dataReady)) break;
          }

          uint64_t syncTailCache = __shfl(tailCache, destNode);
          // for RDMA, tailCache save send step
          size_t srcStagingOffset = ((channelId * nNodes + destNode) * maxRDMAStagingTokens +
                                     (syncTailCache % maxRDMASteps) * stepRDMATokens) *
                                    tokenPackBytes;
          size_t dstStagingOffset = ((channelId * nNodes + myNode) * maxRDMAStagingTokens +
                                     (syncTailCache % maxRDMASteps) * stepRDMATokens) *
                                    tokenPackBytes;
#if ASSERT_ON == 1
          assert(sendTokenNum > 0);
#endif
          if (destNode == myNode) {
            core::WarpCopy(args.shmemInpTokMemObj->template GetAs<char*>(destPe) + dstStagingOffset,
                           args.shmemStagingTokMemObj->template GetAs<char*>() + srcStagingOffset,
                           sendTokenNum * tokenPackBytes);
#if DEBUG == 1
            if (laneId == warpSize - 1)
              printf(
                  "send RDMA putDataLocal rank=%d ch=%d laneId=%d call=%d warpId=%d destPe=%d "
                  "sendTokenNum=%d "
                  "syncSendSlotStart=%d syncTailCache=%lu srcStagingOffset=%zu "
                  "dstStagingOffset=%zu\n",
                  myPe, channelId, laneId, args.crossDeviceBarrierFlag, warpId, destPe,
                  sendTokenNum, syncSendSlotStart, syncTailCache, srcStagingOffset,
                  dstStagingOffset);
#endif
          } else {
            // if (laneId == destNode) {
            shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, dstStagingOffset,
                                                args.shmemStagingTokMemObj, srcStagingOffset,
                                                sendTokenNum * tokenPackBytes, destPe);
#if DEBUG == 1
            if (laneId == warpSize - 1)
              printf(
                  "send RDMA putData rank=%d ch=%d laneId=%d call=%d warpId=%d destPe=%d "
                  "sendTokenNum=%d "
                  "syncSendSlotStart=%d syncTailCache=%lu maxRDMASteps=%zu stepRDMATokens=%d\n",
                  myPe, channelId, laneId, args.crossDeviceBarrierFlag, warpId, destPe,
                  sendTokenNum, syncSendSlotStart, syncTailCache, maxRDMASteps, stepRDMATokens);
#endif
            // }
            shmem::ShmemQuietThread();
          }
#if DEBUG_DATA == 1
          if (laneId == 0) {
#if ASSERT_ON == 1
            // assert(float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
            //                     srcStagingOffset) +
            //                1)) == tokenCheckValue(myPe, 0));
            assert(float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
                                srcStagingOffset) +
                           1)) != float(0));
#endif
            // printf("DATA CHECK SEND RDMA PUT destNode=%d step=%lu srcData=%f\n", destNode,
            //        tailCache,
            //        float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
            //                     srcStagingOffset))));
          }
#endif
          if (laneId == destNode) {
            numTokensToSend -= sendTokenNum;
            sendSlotStart += sendTokenNum;
            // tailCache += sendTokenNum;
            tailCache += 1;
#if ASSERT_ON == 1
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
              printf(
                  "send RDMA putSignalLocal rank=%d call=%d warpId=%d destPe=%d numTokensToSend=%d "
                  "sendSlotStart=%d offset=%d "
                  "tail=%lu\n",
                  myPe, args.crossDeviceBarrierFlag, warpId, destPe, numTokensToSend, sendSlotStart,
                  myPe + channelId * npes,
                  *(args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes));
#endif
            } else {
              shmem::ShmemPutUint64ImmNbiThread(
                  args.tailMemObj, (myPe + channelId * npes) * sizeof(uint64_t), tailCache, destPe);
              // will report cqe error
              // shmem::ShmemQuietThread();
#if DEBUG == 1
              printf(
                  "send RDMA putSignal rank=%d ch=%d laneId=%d call=%d warpId=%d destPe=%d "
                  "numTokensToSend=%d "
                  "sendSlotStart=%d offset=%d "
                  "tailCache=%lu\n",
                  myPe, channelId, laneId, args.crossDeviceBarrierFlag, warpId, destPe,
                  numTokensToSend, sendSlotStart, myPe + channelId * npes, tailCache);
#endif
            }
          }
          // __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
          // __builtin_amdgcn_wave_barrier();
          // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
          // __threadfence_block();
        }
      }
      shmem::ShmemQuietThread();
      if (laneId < nNodes) {
        args.localTail[(laneId * nlocalPes + myLocalPe) + channelId * npes] = tailCache;
#if DEBUG == 1
        printf("send RDMA after rank=%d ch=%d warpId=%d peerRank=%d args.localTail=%lu\n", myPe,
               channelId, warpId, laneId * nlocalPes + myLocalPe, tailCache);
#endif
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

        char* sendStagingPtr[MAX_NODES];
        int numNodesToSend = 0;
        for (int node = 0; node < nNodes; ++node) {
          index_t slot = tokenIdxToSlotMap[tokenIdx * nNodes + node] - 1;
          if (slot == -1) continue;

          // TODO check RDMA buffer ready (shmemInpTokMemObj)
          uint64_t tailCache = nodetailCache[node] + slot / stepRDMATokens;
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
          sendStagingPtr[numNodesToSend++] =
              args.shmemStagingTokMemObj->template GetAs<char*>() +
              ((channelId * nNodes + node) * maxRDMAStagingTokens +
               (tailCache % maxRDMASteps) * stepRDMATokens + slot % stepRDMATokens) *
                  tokenPackBytes;
        }

        core::WarpBroadcast<T, 8>(
            reinterpret_cast<T**>(sendStagingPtr),
            reinterpret_cast<T*>(args.inpTokenBuf + tokenIdx * config.hiddenDim), numNodesToSend,
            config.hiddenDim);

#if DEBUG_DATA == 1
        for (int node = 0; node < numNodesToSend; ++node) {
          // printf(
          //     "DATA CHECK SEND copy to RDMA staging node=%d slot=%d tailCache=%lu srcData=%f\n",
          //     node, slot, tailCache, float(*((T*)sendStagingPtr[node] + 0)));
#if ASSERT_ON == 1
          assert((float)(*((T*)sendStagingPtr[node] + 1)) != (float)(0));
          assert((float)(*((T*)sendStagingPtr[node] + 1)) == tokenCheckValue(myPe, tokenIdx));
          assert((float)(*((T*)sendStagingPtr[node] + config.hiddenDim - 1)) ==
                 tokenCheckValue(myPe, tokenIdx));
#endif
        }
#endif

        if (weightBytes) {
          for (int i = laneId; i < numExpertPerToken * numNodesToSend; i += warpSize) {
            int node = i / numExpertPerToken;
            int weightIdx = i % numExpertPerToken;
            auto weightVal = __builtin_nontemporal_load(args.weightsBuf + tokenIdx * numExpertPerToken +
                                                  weightIdx);
            *(reinterpret_cast<float*>(sendStagingPtr[node] + weightsOffset) + weightIdx) =
                weightVal;
          }
        }

        for (int i = laneId; i < numExpertPerToken * numNodesToSend; i += warpSize) {
          int node = i / numExpertPerToken;
          int indexIdx = i % numExpertPerToken;
          auto indexVal = __builtin_nontemporal_load(args.tokenIndices +
                                                     tokenIdx * numExpertPerToken + indexIdx);
          *(reinterpret_cast<index_t*>(sendStagingPtr[node] + indiceOffset) + indexIdx) = indexVal;
        }

        if (scaleBytes) {
          for (int i = laneId; i < config.scaleDim * config.scaleTypeSize; i += warpSize) {
            auto scaleVal = __builtin_nontemporal_load(reinterpret_cast<char*>(args.scalesBuf) +
                                                       tokenIdx * scaleBytes + i);
            for (int node = 0; node < numNodesToSend; ++node) {
              *(sendStagingPtr[node] + scalesOffset + i) = scaleVal;
            }
          }
        }
        
        if (laneId < numNodesToSend) {
          // write meta - record source token info
          *(reinterpret_cast<size_t*>(sendStagingPtr[laneId] + metaOffset)) =
              myPe * maxNumInpTokenPerRank + tokenIdx;
        }

        tokenProgress[warpId] = tokenIdx;
        __threadfence_block();
      }
      tokenProgress[warpId] = channelEndOffset - 1;
    }
    
    __syncthreads();
    // clear slotToTokenIdxMap
    for (int i = thdId; i < nNodes * maxTokensPerChannel; i += thdNum) {
      slotToTokenIdxMap[i] = 0;
    }
    // clear tokenIdxToSlotMap
    for (int i = thdId + channelStartOffset * nNodes; i < channelEndOffset * nNodes; i += thdNum) {
      tokenIdxToSlotMap[i] = 0;
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
        printf("recv fwd before rank=%d ch=%d call=%d warpId=%d node=%d localHead=%lu\n", myPe,
               channelId, args.crossDeviceBarrierFlag, warpId, laneId, headCache);
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
        // syncNumTokensToRecv = __shfl(numTokensToRecv, srcNode);
        uint64_t tokenEnd = tokenStart + min(syncNumTokensToRecv, stepRDMATokens);
        for (uint64_t i = tokenStart; i < tokenEnd; ++i) {
          if (srcNode == laneId) {
            --numTokensToRecv;
#if ASSERT_ON == 1
            assert(numTokensToRecv >= 0);
#endif
          }
          --syncNumTokensToRecv;
          index_t slot = i;
          char* srcPtr =
              args.shmemInpTokMemObj->template GetAs<char*>() +
              ((channelId * nNodes + srcNode) * maxRDMAStagingTokens + slot) * tokenPackBytes;
          index_t* srcIndicesPtr = reinterpret_cast<index_t*>(srcPtr + indiceOffset);
          int destPe = -1;
          if (laneId < numExpertPerToken) destPe = srcIndicesPtr[laneId] / config.numExpertPerRank;
          if (!__any(destPe == fwdPe)) {
            continue;
          }

#if DEBUG_DATA == 1 && ASSERT_ON == 1 
          if (laneId == 0) {
            assert(float(*(T*)srcPtr) != float(0));
          }
#endif
          if (fwdPe == myPe) {
            index_t localTokenIdx = 0;
            if (laneId == 0) {
              localTokenIdx = atomicAdd(args.localPeTokenCounter, 1);
            }
            localTokenIdx = __shfl(localTokenIdx, 0);
            // TODO abstrct into a function
            core::WarpCopy(reinterpret_cast<char*>(args.outTokenBuf) + localTokenIdx * tokenBytes,
                           srcPtr, tokenBytes);
            if (weightBytes) {
              core::WarpCopy(
                  args.shmemOutWeightsMemObj->template GetAs<char*>() + localTokenIdx * weightBytes,
                  srcPtr + weightsOffset, weightBytes);
            }
            core::WarpCopy(
                args.shmemOutIndicesMemObj->template GetAs<char*>() + localTokenIdx * indiceBytes,
                srcPtr + indiceOffset, indiceBytes);
            if (scaleBytes) {
              core::WarpCopy(
                  args.shmemOutScalesMemObj->template GetAs<char*>() + localTokenIdx * scaleBytes,
                  srcPtr + scalesOffset, scaleBytes);
            }
            if (laneId == 0) {
              // get meta
              args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[localTokenIdx] =
                  *(reinterpret_cast<index_t*>(srcPtr + metaOffset));
#if DEBUG_DATA == 1
              index_t meta = *(reinterpret_cast<index_t*>(srcPtr + metaOffset));
              int srcPe = meta / maxNumInpTokenPerRank;
              int srcTokenId = meta % maxNumInpTokenPerRank;
#if ASSERT_ON == 1
              assert(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                  localTokenIdx * tokenBytes))) != (float)(0));
              assert(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                  localTokenIdx * tokenBytes) +
                             config.hiddenDim - 1)) != (float)(0));

              assert(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                  localTokenIdx * tokenBytes))) ==
                     tokenCheckValue(srcPe, srcTokenId));
              assert(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                  localTokenIdx * tokenBytes) +
                             config.hiddenDim - 1)) == tokenCheckValue(srcPe, srcTokenId));
#endif
              // printf(
              //     "DATA CHECK RECV fwd to output srcStep=%lu srcSlot=%d srcData=%f dst(output) "
              //     "localTokenIdx=%d\n",
              //     step, slot,
              //     float(*((T*)(args.shmemInpTokMemObj->template GetAs<char*>() +
              //                  ((channelId * nNodes + srcNode) * maxRDMAStagingTokens + slot) *
              //                      tokenPackBytes))),
              //     localTokenIdx);
#endif
            }
          } else {
            char* dstPtr = args.shmemOutTokMemObj->template GetAs<char*>() +
                           ((channelId * nlocalPes + fwdLocalPe) * maxP2PStagingTokens +
                            p2pTailCache % maxP2PStagingTokens) *
                               tokenPackBytes;
            core::WarpCopy(dstPtr, srcPtr, tokenPackBytes);
#if DEBUG_DATA == 1
#if ASSERT_ON == 1
            assert((float)(*(T*)dstPtr) != float(0));
#endif
            // printf(
            //     "DATA CHECK RECV fwd to shmemOutTokMemObj srcStep=%lu srcSlot=%d srcData=%f "
            //     "dst p2pTailCache=%lu dstSlot=%lu\n",
            //     step, slot, float(*((T*)(srcPtr))), p2pTailCache,
            //     p2pTailCache % maxP2PStagingTokens);
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
          printf("recv fwd after rank=%d ch=%d warpId=%d fwdPe=%d p2p tail=%lu\n", myPe, channelId,
                 warpId, fwdPe,
                 *(args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe));
#endif
        }
        if (laneId == srcNode) {
          headCache = tailCache;
          fwdHead[fwdLocalPe][srcNode] = headCache - headBase;
        }
      }
      if (fwdLocalPe != myLocalPe) {
        int* statusPtr =
            args.intraNodeBarrierMemObj->template GetAs<int*>(fwdPe) + myPe + channelId * npes;
        shmem::ShmemInt32WaitUntilEquals(statusPtr, 0);
        core::AtomicStoreRelaxedSystem(statusPtr, fwdCounter + 1);
#if DEBUG==1
        if (laneId == 0) {
          printf("recv fwd rank=%d ch=%d warpId=%d fwdPe=%d fwdCounter=%d\n", myPe, channelId,
                 warpId, fwdPe, fwdCounter);
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
            "recv Ctl before rank=%d ch=%d call=%d warpId=%d node=%d numTokensToRecv=%d "
            "localHead=%lu\n",
            myPe, channelId, args.crossDeviceBarrierFlag, warpId, laneId, numTokensToRecv,
            headBase);
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
#if ASSERT_ON == 1
              assert(numTokensToRecv > -stepRDMATokens);
#endif
            }
          }
        }
        shmem::ShmemQuietThread();
      }

      if (laneId < nNodes) {
        args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] = headBase + curMinHead;
#if DEBUG == 1
        printf("recv Ctl rank=%d ch=%d call=%d warpId=%d node=%d numRecvTokens=%d localHead=%lu\n",
               myPe, channelId, args.crossDeviceBarrierFlag, warpId, laneId,
               rdmaRecvTokensNum[laneId],
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
            //       "recv copy wait data TIMEOUT rank=%d ch=%d srcPe=%d call=%d warpId=%d "
            //       "channelId=%d p2pTailCache=%lu "
            //       "p2pHeadCache=%lu\n",
            //       myPe, channelId, srcPe, args.crossDeviceBarrierFlag, warpId, channelId,
            //       p2pTailCache, p2pHeadCache);
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

#if DEBUG_DATA == 1 && ASSERT_ON == 1
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
          // if (laneId == 0) {
          //   printf(
          //       "DATA CHECK COPY to output srcSlot=%d srcData=%f "
          //       "dst output localTokenIdx=%d\n",
          //       slot, (float)(*((T*)srcPtr)), localTokenIdx);
          // }
#if ASSERT_ON == 1
          assert((float)(*(T*)(srcPtr)) != 0);
#endif
#endif
          if (weightBytes) {
            core::WarpCopy(
                args.shmemOutWeightsMemObj->template GetAs<char*>() + localTokenIdx * weightBytes,
                srcPtr + weightsOffset, weightBytes);
          }
          core::WarpCopy(
              args.shmemOutIndicesMemObj->template GetAs<char*>() + localTokenIdx * indiceBytes,
              srcPtr + indiceOffset, indiceBytes);
          if (scaleBytes) {
            core::WarpCopy(
                args.shmemOutScalesMemObj->template GetAs<char*>() + localTokenIdx * scaleBytes,
                srcPtr + scalesOffset, scaleBytes);
          }
          if (laneId == 0) {
            // get meta
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[localTokenIdx] =
                *(reinterpret_cast<index_t*>(srcPtr + metaOffset));
#if DEBUG_DATA == 1
            index_t meta = *(reinterpret_cast<index_t*>(srcPtr + metaOffset));
            int srcPe = meta / maxNumInpTokenPerRank;
            int srcTokenId = meta % maxNumInpTokenPerRank;
#if ASSERT_ON == 1
            assert(float(*((T*)(srcPtr))) == tokenCheckValue(srcPe, srcTokenId));
            assert(float(*((T*)(srcPtr) + config.hiddenDim - 1)) ==
                   tokenCheckValue(srcPe, srcTokenId));
#endif
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

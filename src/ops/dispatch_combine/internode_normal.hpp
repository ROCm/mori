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
#define RECV_DATA 1
#define SEND_DATA 1
#define COPY_DATA 1

#define INTRA_NODE_WRITE 1

#if ASSERT_ON == 1
#define KERNEL_ASSERT(cond)                                                                  \
  do {                                                                                       \
    if (!(cond)) {                                                                           \
      printf("[ASSERT] Block %d Thread %d - %s:%d: %s\n", blockIdx.x, threadIdx.x, __FILE__, \
             __LINE__, #cond);                                                               \
      abort();                                                                               \
    }                                                                                        \
  } while (0)
#else
#define KERNEL_ASSERT(cond) ((void)0)
#endif

__device__ inline float tokenCheckValue(int srcPe, int srcTokenId) {
  return srcPe * 0.1f + 1.0f + srcTokenId;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchInterNodeNormalKernel                             */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchInterNodeNormalKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  const index_t tokenNum = args.curRankNumToken;
#if DEBUG == 1
  if (tokenNum != 0) {
    KERNEL_ASSERT(args.tokenIndices && args.inpTokenBuf);
  }
#endif

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

  // calculate channel size and offset
  const index_t baseTokensPerChannel = tokenNum / nChannels;
  const index_t remTokens = tokenNum % nChannels;
  const index_t channelStartOffset = channelId * baseTokensPerChannel + min(channelId, remTokens);
  const index_t tokensPerChannel = baseTokensPerChannel + (channelId < remTokens ? 1 : 0);
  const index_t channelEndOffset = min(channelStartOffset + tokensPerChannel, tokenNum);

  const size_t maxTokensPerChannel = baseTokensPerChannel + (remTokens ? 1 : 0);
  const size_t maxNumToken = maxTokensPerChannel * nChannels;
  const size_t maxChannelRecvTokensPerRank = maxTokensPerChannel * nNodes;

  // define staging buffer size
  const int stepRDMATokens = config.maxRDMAStepTokens;
  const int stepP2pTokens = config.maxP2PStepTokens;
#if DEBUG == 1
  if (blockId==0 && thdId == 0) {
    printf("nlocalPes=%d stepRDMATokens=%d\n", nlocalPes, stepRDMATokens);
  }
#endif
  const size_t maxChannelStagingTokens = config.maxChannelStagingTokens;

  const size_t maxRDMAStagingTokens = maxChannelStagingTokens;
  const size_t maxRDMASteps = maxChannelStagingTokens / stepRDMATokens;

  const size_t maxP2PStagingTokens = maxChannelStagingTokens * nNodes;

  // For sender
  index_t* tokenIdxToSlotMap = reinterpret_cast<index_t*>(args.localPeBuf);
  index_t* slotToTokenIdxMap =
      reinterpret_cast<index_t*>(reinterpret_cast<char*>(tokenIdxToSlotMap) +
                                 maxNumInpTokenPerRank * nNodes * sizeof(index_t));
  // // For Reveiver
  // index_t* rdmaRecvTokensNum = reinterpret_cast<index_t*>(
  //     reinterpret_cast<char*>(slotToTokenIdxMap) + nNodes * maxNumToken * sizeof(index_t));

  slotToTokenIdxMap += channelId * nNodes * maxTokensPerChannel;

  // For Reveiver
  index_t* rdmaRecvTokensNum = args.rdmaRecvTokensNum + channelId * nNodes;

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
        // __threadfence_block();
      }
#if DEBUG == 1
      if (myPe == 0 && laneId == 0 && channelId == 0) {
        for (int i = 0; i < nNodes; ++i) {
          printf("send Ctl rank=%d warpId=%d nodeTokenCount[%d]=%d tokenProgress[15]=%d\n", myPe,
                 warpId, i, nodeTokenCount[i], tokenProgress[kSendWarpCount]);
        }
      }
#endif

      for (int i = 0; i < nNodes; ++i) {
        int destNode = (myNode + i) % nNodes;
        int destPe = destNode * nlocalPes + myLocalPe;

        if (laneId == 0) {
          shmem::ShmemPutInt32ImmNbiThread(args.recvTokenNumMemObj,
                                           (myNode + channelId * nNodes) * sizeof(index_t),
                                           nodeTokenCount[destNode] + 1, destPe, channelId);
        }
      }

      index_t numTokensToSend = laneId < nNodes ? nodeTokenCount[laneId] : 0;
      index_t sendSlotStart = 0;
      uint64_t tailCache =
          laneId < nNodes ? args.localTail[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;
#if DEBUG == 1
      // if (myPe == 0 && laneId < nNodes && channelId == 0) {
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
          KERNEL_ASSERT(sendTokenNum > 0);

#if SEND_DATA == 1
          if (destNode != myNode) {
            // for RDMA, tailCache save send step
            size_t srcStagingOffset = ((channelId * nNodes + destNode) * maxRDMAStagingTokens +
                                       (syncTailCache % maxRDMASteps) * stepRDMATokens) *
                                      tokenPackBytes;
            size_t dstStagingOffset = ((channelId * nNodes + myNode) * maxRDMAStagingTokens +
                                       (syncTailCache % maxRDMASteps) * stepRDMATokens) *
                                      tokenPackBytes;
            // if (laneId == destNode) {
            shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, dstStagingOffset,
                                                args.shmemStagingTokMemObj, srcStagingOffset,
                                                sendTokenNum * tokenPackBytes, destPe, channelId);
#if DEBUG == 1
            if (myPe == 0 && laneId == warpSize - 1 && channelId == 0) {
              printf(
                  "send RDMA putData rank=%d ch=%d laneId=%d call=%d warpId=%d destPe=%d "
                  "sendTokenNum=%d "
                  "syncSendSlotStart=%d syncTailCache=%lu maxRDMASteps=%zu stepRDMATokens=%d\n",
                  myPe, channelId, laneId, args.crossDeviceBarrierFlag, warpId, destPe,
                  sendTokenNum, syncSendSlotStart, syncTailCache, maxRDMASteps, stepRDMATokens);
            }
            shmem::ShmemQuietThread(destPe, channelId);
#endif
            // }
#if DEBUG_DATA == 1
            if (laneId == 0) {
              // KERNEL_ASSERT(float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
              //                     srcStagingOffset) +
              //                1)) == tokenCheckValue(myPe, 0));
              KERNEL_ASSERT(float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
                                         srcStagingOffset) +
                                    1)) != float(0));
              // printf("DATA CHECK SEND RDMA PUT destNode=%d step=%lu srcData=%f\n", destNode,
              //        tailCache,
              //        float(*((T*)(args.shmemStagingTokMemObj->template GetAs<char*>() +
              //                     srcStagingOffset))));
            }
#endif
          }
#endif

          if (laneId == destNode) {
            numTokensToSend -= sendTokenNum;
            sendSlotStart += sendTokenNum;
            // tailCache += sendTokenNum;
            tailCache += 1;
            KERNEL_ASSERT(numTokensToSend >= 0 && sendSlotStart <= nodeTokenCount[destNode]);

            // Update rdma tail
            // TODO use amo (amo hang)
            if (destNode == myNode) {
              __hip_atomic_store(
                  args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes,
                  tailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if DEBUG == 1
              if (myPe == 0 && channelId == 0) {
                printf(
                    "send RDMA putSignal(local) rank=%d call=%d warpId=%d destPe=%d "
                    "numTokensToSend=%d "
                    "sendSlotStart=%d offset=%d "
                    "tail=%lu\n",
                    myPe, args.crossDeviceBarrierFlag, warpId, destPe, numTokensToSend,
                    sendSlotStart, myPe + channelId * npes,
                    *(args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe +
                      channelId * npes));
              }
#endif
            } else {
              shmem::ShmemPutUint64ImmNbiThread(args.tailMemObj,
                                                (myPe + channelId * npes) * sizeof(uint64_t),
                                                tailCache, destPe, channelId);
#if DEBUG == 1
              if (myPe == 0 && channelId == 0) {
                printf(
                    "send RDMA putSignal rank=%d ch=%d laneId=%d call=%d warpId=%d destPe=%d "
                    "numTokensToSend=%d "
                    "sendSlotStart=%d offset=%d "
                    "tailCache=%lu\n",
                    myPe, channelId, laneId, args.crossDeviceBarrierFlag, warpId, destPe,
                    numTokensToSend, sendSlotStart, myPe + channelId * npes, tailCache);
              }
              // will report cqe error
              shmem::ShmemQuietThread(destPe, channelId);
#endif
            }
          }
          // __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
          // __builtin_amdgcn_wave_barrier();
          // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
          // __threadfence_block();
        }
      }

      if (laneId < nNodes) {
        args.localTail[(laneId * nlocalPes + myLocalPe) + channelId * npes] = tailCache;
#if DEBUG == 1
        if (myPe == 0 && channelId == 0) {
          printf("send RDMA after rank=%d ch=%d warpId=%d peerRank=%d args.localTail=%lu\n", myPe,
                 channelId, warpId, laneId * nlocalPes + myLocalPe, tailCache);
        }
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
          do {
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
              //       myPe, node * nlocalPes + myLocalPe, args.crossDeviceBarrierFlag, warpId,
              //       node, channelId, tailCache, headCache);
              // }
            }
            headCache = __shfl(headCache, node);
          } while (tailCache - headCache >= maxRDMASteps);

          // TODO modify shmemStagingTokMemObj size
          sendStagingPtr[numNodesToSend++] =
              (node == myNode ? args.shmemInpTokMemObj->template GetAs<char*>()
                              : args.shmemStagingTokMemObj->template GetAs<char*>()) +
              ((channelId * nNodes + node) * maxRDMAStagingTokens +
               (tailCache % maxRDMASteps) * stepRDMATokens + slot % stepRDMATokens) *
                  tokenPackBytes;
        }

#if COPY_DATA == 1
        core::WarpBroadcast<T, 8>(reinterpret_cast<T**>(sendStagingPtr),
                                  args.inpTokenBuf + tokenIdx * config.hiddenDim, numNodesToSend,
                                  config.hiddenDim);

#if DEBUG_DATA == 1
        for (int node = 0; node < numNodesToSend; ++node) {
          // printf(
          //     "DATA CHECK SEND copy to RDMA staging node=%d slot=%d tailCache=%lu srcData=%f\n",
          //     node, slot, tailCache, float(*((T*)sendStagingPtr[node] + 0)));
          KERNEL_ASSERT((float)(*((T*)sendStagingPtr[node] + 1)) != (float)(0));
          KERNEL_ASSERT((float)(*((T*)sendStagingPtr[node] + 1)) ==
                        tokenCheckValue(myPe, tokenIdx));
          KERNEL_ASSERT((float)(*((T*)sendStagingPtr[node] + config.hiddenDim - 1)) ==
                        tokenCheckValue(myPe, tokenIdx));
        }
#endif

        if (weightBytes) {
          for (int i = laneId; i < numExpertPerToken * numNodesToSend; i += warpSize) {
            int node = i / numExpertPerToken;
            int weightIdx = i % numExpertPerToken;
            auto weightVal = __builtin_nontemporal_load(args.weightsBuf +
                                                        tokenIdx * numExpertPerToken + weightIdx);
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
#endif

        tokenProgress[warpId] = tokenIdx;
        // __threadfence_block();
      }
      tokenProgress[warpId] = channelEndOffset - 1;
    } else {
      __syncthreads();
    }

    for (int node = 0; node < nNodes; ++node) {
      int destPe = node * nlocalPes + myLocalPe;
      shmem::ShmemQuietThread(destPe, channelId);
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
      if (warpId == 0) {
        fwdHead[laneId / MAX_NODES][laneId % MAX_NODES] = 0;
        if (laneId == 0) fwdFinish = 0;
      }
      const int fwdLocalPe = warpId;
      const int fwdPe = myNode * nlocalPes + fwdLocalPe;
      index_t numTokensToRecv = 0;
      if (laneId < nNodes) {
        index_t* signal =
            args.recvTokenNumMemObj->template GetAs<index_t*>() + laneId + channelId * nNodes;
        numTokensToRecv = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
        if (warpId == 0) rdmaRecvTokensNum[laneId] = numTokensToRecv;
      }
      __syncthreads();
      if (fwdLocalPe >= nlocalPes) {
        __syncthreads();
        return;
      }
      if (warpId == 0) {
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
      if (laneId < nNodes && myPe == 0 && channelId == 0) {
        printf("recv fwd before rank=%d ch=%d fwdPe=%d call=%d warpId=%d node=%d localHead=%lu\n",
               myPe, channelId, fwdPe, args.crossDeviceBarrierFlag, warpId, laneId, headCache);
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
      int srcNode = (myNode + 1) % nNodes;
      index_t syncNumTokensToRecv = 0;
      // update map for combine
      index_t srcNodeTokensCounter = 0;
      index_t* recvTokenMap = args.srcPeTokenIdxMap +
                              (channelId * nlocalPes + fwdLocalPe) * maxChannelRecvTokensPerRank;
      index_t* fwdTokenMap =
          laneId < nNodes
              ? args.fwdTokenMap + (channelId * nNodes + laneId) * maxTokensPerChannel * nlocalPes +
                    fwdLocalPe
              : nullptr;
      while (__any(numTokensToRecv > 0)) {
        // check buffer ready (shmemOutTokMemObj)
        if (srcNode != myNode) {
          while (p2pTailCache - p2pHeadCache >= maxP2PStagingTokens) {
            if (laneId == 0) {
              p2pHeadCache = __hip_atomic_load(
                  args.headMemObj->template GetAs<uint64_t*>() + channelId * npes + fwdPe,
                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
            p2pHeadCache = __shfl(p2pHeadCache, 0);
          }
        }

        // check RDMA data ready (shmemInpTokMemObj), wait data from same-id GPU
        while (true) {
          srcNode = (srcNode - 1 + nNodes) % nNodes;
          int srcPe = srcNode * nlocalPes + myLocalPe;
          syncNumTokensToRecv = __shfl(numTokensToRecv, srcNode);
          if (syncNumTokensToRecv > 0) {
            if (laneId == srcNode && tailCache < headCache + 1) {
              tailCache = __hip_atomic_load(
                  args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
            if (__shfl(tailCache >= headCache, srcNode)) break;
          }
        }
        uint64_t syncHeadCache = __shfl(headCache, srcNode);
        uint64_t syncTailCache = __shfl(tailCache, srcNode);

        for (uint64_t step = syncHeadCache; step < syncTailCache; ++step) {
          uint64_t tokenStart = (syncHeadCache % maxRDMASteps) * stepRDMATokens;
          // syncNumTokensToRecv = __shfl(numTokensToRecv, srcNode);
          uint64_t tokenEnd = tokenStart + min(syncNumTokensToRecv, stepRDMATokens);
          for (uint64_t i = tokenStart; i < tokenEnd; ++i) {
            if (srcNode == laneId) {
              --numTokensToRecv;
              KERNEL_ASSERT(numTokensToRecv >= 0);
              ++srcNodeTokensCounter;
            }
            --syncNumTokensToRecv;
            index_t slot = i;
            char* srcPtr =
                args.shmemInpTokMemObj->template GetAs<char*>() +
                ((channelId * nNodes + srcNode) * maxRDMAStagingTokens + slot) * tokenPackBytes;
            index_t* srcIndicesPtr = reinterpret_cast<index_t*>(srcPtr + indiceOffset);
            int destPe = -1;
            if (laneId < numExpertPerToken)
              destPe = srcIndicesPtr[laneId] / config.numExpertPerRank;
            if (!__any(destPe == fwdPe)) {
              continue;
            }

#if DEBUG_DATA == 1
            if (laneId == 0) {
              KERNEL_ASSERT(float(*(T*)srcPtr) != float(0));
            }
#endif

#if RECV_DATA == 1
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
                core::WarpCopy(args.shmemOutWeightsMemObj->template GetAs<char*>() +
                                   localTokenIdx * weightBytes,
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
                // update map for combine
                // args.dispReceiverIdxMap[localTokenIdx] = srcNode;
                recvTokenMap[fwdCounter] = localTokenIdx;
#if DEBUG_DATA == 1
                index_t meta = *(reinterpret_cast<index_t*>(srcPtr + metaOffset));
                int srcPe = meta / maxNumInpTokenPerRank;
                int srcTokenId = meta % maxNumInpTokenPerRank;
                KERNEL_ASSERT(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                           localTokenIdx * tokenBytes))) != (float)(0));
                KERNEL_ASSERT(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                           localTokenIdx * tokenBytes) +
                                      config.hiddenDim - 1)) != (float)(0));

                KERNEL_ASSERT(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                           localTokenIdx * tokenBytes))) ==
                              tokenCheckValue(srcPe, srcTokenId));
                KERNEL_ASSERT(float(*((T*)(reinterpret_cast<char*>(args.outTokenBuf) +
                                           localTokenIdx * tokenBytes) +
                                      config.hiddenDim - 1)) == tokenCheckValue(srcPe, srcTokenId));
                // printf(
                //     "DATA CHECK RECV fwd to output srcStep=%lu srcSlot=%d srcData=%f dst(output)
                //     " "localTokenIdx=%d\n", step, slot,
                //     float(*((T*)(args.shmemInpTokMemObj->template GetAs<char*>() +
                //                  ((channelId * nNodes + srcNode) * maxRDMAStagingTokens + slot) *
                //                      tokenPackBytes))),
                //     localTokenIdx);
#endif
              }
            } else {
#if INTRA_NODE_WRITE == 1
              char* dstPtr = args.shmemOutTokMemObj->template GetAs<char*>(fwdPe) +
                             ((channelId * nlocalPes + myLocalPe) * maxP2PStagingTokens +
                              p2pTailCache % maxP2PStagingTokens) *
                                 tokenPackBytes;
#else
              char* dstPtr = args.shmemOutTokMemObj->template GetAs<char*>() +
                             ((channelId * nlocalPes + fwdLocalPe) * maxP2PStagingTokens +
                              p2pTailCache % maxP2PStagingTokens) *
                                 tokenPackBytes;
#endif
              core::WarpCopy(dstPtr, srcPtr, tokenPackBytes);
#if DEBUG_DATA == 1
              KERNEL_ASSERT((float)(*(T*)dstPtr) != float(0));
              // printf(
              //     "DATA CHECK RECV fwd to shmemOutTokMemObj srcStep=%lu srcSlot=%d srcData=%f "
              //     "dst p2pTailCache=%lu dstSlot=%lu\n",
              //     step, slot, float(*((T*)(srcPtr))), p2pTailCache,
              //     p2pTailCache % maxP2PStagingTokens);
#endif
              ++p2pTailCache;
              if (syncNumTokensToRecv == 0 || p2pTailCache - lastP2pTail >= stepP2pTokens) {
                __threadfence_system();
                // Update p2p tail
                if (laneId == 0) {
                  __hip_atomic_store(
                      args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes + myPe,
                      p2pTailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                }
                lastP2pTail = p2pTailCache;
#if DEBUG == 1
                if (myPe == 0 && channelId == 0 && laneId == warpSize - 1) {
                  printf(
                      "recv fwd after rank=%d ch=%d warpId=%d laneId=%d srcNode=%d fwdPe=%d p2p "
                      "tail=%lu\n",
                      myPe, channelId, warpId, laneId, srcNode, fwdPe,
                      *(args.tailMemObj->template GetAs<uint64_t*>(fwdPe) + channelId * npes +
                        myPe));
                }
#endif
              }
            }
#endif

            if (laneId == srcNode) {
              fwdTokenMap[((srcNodeTokensCounter - 1)) * nlocalPes] = fwdCounter + 1;
            }
            ++fwdCounter;
          }

          if (laneId == srcNode) {
            headCache = tailCache;
            fwdHead[fwdLocalPe][srcNode] = headCache - headBase;
          }
        }
      }

      // Set intra-node forward finished flag
      if (fwdLocalPe != myLocalPe) {
        int* statusPtr =
            args.intraNodeBarrierMemObj->template GetAs<int*>(fwdPe) + myPe + channelId * npes;
        shmem::ShmemInt32WaitUntilEquals(statusPtr, 0);
        if (laneId == 0) {
          core::AtomicStoreRelaxedSystem(statusPtr, fwdCounter + 1);
        }
#if DEBUG == 1
        if (laneId == 0 && myPe == 0 && channelId == 0) {
          printf("recv fwd done rank=%d ch=%d warpId=%d fwdPe=%d fwdCounter=%d\n", myPe, channelId,
                 warpId, fwdPe, fwdCounter);
        }
#endif
      } else {
        if (laneId == 0) {
          args.p2pRecvTokenNum[channelId * nlocalPes + fwdLocalPe] = fwdCounter;
        }
      }
    } else if (warpId == kfwdWarpCount) {
      __syncthreads();
      index_t numTokensToRecv = laneId < nNodes ? rdmaRecvTokensNum[laneId] : 0;
      uint64_t headBase =
          laneId < nNodes ? args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] : 0;
#if DEBUG == 1
      if (laneId < nNodes && myPe == 0 && channelId == 0) {
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
                syncHeadBase + syncMinHead, node * nlocalPes + myLocalPe, channelId);
#if DEBUG == 1
            shmem::ShmemQuietThread(node * nlocalPes + myLocalPe, channelId);
#endif
            if (laneId == node) {
              numTokensToRecv -= (minHead - curMinHead) * stepRDMATokens;
              curMinHead = minHead;
              KERNEL_ASSERT(numTokensToRecv > -stepRDMATokens);
            }
          }
        }
      }

      if (laneId < nNodes) {
        args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes] = headBase + curMinHead;
#if DEBUG == 1
        if (myPe == 0 && channelId == 0) {
          printf(
              "recv Ctl rank=%d ch=%d call=%d warpId=%d node=%d numRecvTokens=%d localHead=%lu\n",
              myPe, channelId, args.crossDeviceBarrierFlag, warpId, laneId,
              rdmaRecvTokensNum[laneId],
              args.localHead[(laneId * nlocalPes + myLocalPe) + channelId * npes]);
        }
#endif
        // // clear rdmaRecvTokensNum
        // rdmaRecvTokensNum[laneId] = 0;
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

      // update map for combine
      index_t recvTokensCounter = 0;
      index_t* recvTokenMap = args.srcPeTokenIdxMap +
                              (channelId * nlocalPes + srcLocalPe) * maxChannelRecvTokensPerRank;
      while (true) {
        done = __shfl(done, 0);
        if (done) {
          if (laneId == 0) core::AtomicStoreRelaxedSystem(statusPtr, 0);
          break;
        }

        // check data ready (shmemOutTokMemObj)
        while (p2pTailCache < p2pHeadCache + 1 && !done) {
          if (laneId == 0) {
            p2pTailCache = __hip_atomic_load(
                args.tailMemObj->template GetAs<uint64_t*>() + srcPe + channelId * npes,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            done = core::AtomicLoadRelaxedSystem(statusPtr);
          }
          p2pTailCache = __shfl(p2pTailCache, 0);
          done = __shfl(done, 0);
        }

        for (uint64_t i = p2pHeadCache; i < p2pTailCache; ++i) {
          index_t slot = i % maxP2PStagingTokens;
#if INTRA_NODE_WRITE == 1
          char* srcPtr =
              args.shmemOutTokMemObj->template GetAs<char*>() +
              ((channelId * nlocalPes + srcLocalPe) * maxP2PStagingTokens + slot) * tokenPackBytes;
#else
          char* srcPtr =
              args.shmemOutTokMemObj->template GetAs<char*>(srcPe) +
              ((channelId * nlocalPes + myLocalPe) * maxP2PStagingTokens + slot) * tokenPackBytes;
#endif

#if DEBUG_DATA == 1
          if (laneId == 0) {
            KERNEL_ASSERT(float(*(T*)srcPtr) != 0);
          }
#endif

#if RECV_DATA == 1
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
          KERNEL_ASSERT((float)(*(T*)(srcPtr)) != 0);
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
            index_t meta = *(reinterpret_cast<index_t*>(srcPtr + metaOffset));
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[localTokenIdx] = meta;
#if DEBUG_DATA == 1
            index_t meta = *(reinterpret_cast<index_t*>(srcPtr + metaOffset));
            int srcPe = meta / maxNumInpTokenPerRank;
            int srcTokenId = meta % maxNumInpTokenPerRank;
            KERNEL_ASSERT(float(*((T*)(srcPtr))) == tokenCheckValue(srcPe, srcTokenId));
            KERNEL_ASSERT(float(*((T*)(srcPtr) + config.hiddenDim - 1)) ==
                          tokenCheckValue(srcPe, srcTokenId));
#endif
            // record source node for combine
            // args.dispReceiverIdxMap[localTokenIdx] = (meta / maxNumInpTokenPerRank) / nNodes;
            // record output slot index for combine
            recvTokenMap[recvTokensCounter++] = localTokenIdx;
          }
#endif
        }

        p2pHeadCache = p2pTailCache;
        // Update p2p head
        if (laneId == 0) {
          __hip_atomic_store(
              args.headMemObj->template GetAs<uint64_t*>(srcPe) + channelId * npes + myPe,
              p2pHeadCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }

      // record recvTokensCounter for combine
      if (laneId == 0){
        args.p2pRecvTokenNum[channelId * nlocalPes + srcLocalPe] = recvTokensCounter;
      }
    }

    for (int node = 0; node < nNodes; ++node) {
      int destPe = node * nlocalPes + myLocalPe;
      shmem::ShmemQuietThread(destPe, channelId);
    }
    __syncthreads();

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
        // args.p2pRecvTokenNum[nChannels * nlocalPes] = maxChannelRecvTokensPerRank;
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

  const int numExpertPerToken = config.numExpertPerToken;
  const size_t tokenBytes = config.hiddenDim * sizeof(T);
  const size_t weightBytes = args.weightsBuf ? sizeof(float) * numExpertPerToken : 0;
  const size_t indiceBytes = 0;
  const size_t scaleBytes = 0;
  const size_t metaBytes = 0;
  const size_t tokenPackBytes = tokenBytes + weightBytes + indiceBytes + scaleBytes + metaBytes;

  const size_t weightsOffset = tokenBytes;
  // const size_t indiceOffset = weightsOffset + weightBytes;
  // const size_t scalesOffset = indiceOffset + indiceBytes;
  // const size_t metaOffset = scalesOffset + scaleBytes;

  const int nChannels = blockNum / 2;
  const int isSender = blockId < nChannels;
  const int channelId = blockId % nChannels;

  // calculate channel size and offset
  const index_t tokenNum = args.curRankNumToken;
  index_t baseTokensPerChannel = tokenNum / nChannels;
  const size_t maxTokensPerChannel = baseTokensPerChannel + (remTokens ? 1 : 0);
  const size_t maxChannelRecvTokensPerRank = maxTokensPerChannel * nNodes;
#if DEBUG == 1
  if (blockId == 0 && thdId == 0) {
    printf("maxTokensPerChannel=%zu\n", maxTokensPerChannel);
  }
#endif

  // define staging buffer size
  const int stepRDMATokens = config.maxRDMAStepTokens;
  const int stepP2pTokens = config.maxP2PStepTokens;
  const size_t maxChannelStagingTokens = config.maxChannelStagingTokens;

  const size_t maxRDMAStagingTokens = maxChannelStagingTokens;
  const size_t maxRDMASteps = maxChannelStagingTokens / stepRDMATokens;

  // differs from dispatch
  const size_t maxP2PStagingTokens = maxChannelStagingTokens * nNodes;

  // clear totalRecvTokenNum
  if (blockId == 0 && thdId == 0) {
    args.totalRecvTokenNum[0] = 0;
  }

  if (isSender) {
    constexpr int kSendWarpCount = 8;
    if (warpId < kSendWarpCount) {
      const int dstLocalPe = warpId;
      const int destPe = myNode * nlocalPes + dstLocalPe;
      const index_t numChannelTokens = args.p2pRecvTokenNum[channelId * nlocalPes + dstLocalPe];

      uint64_t p2pHeadCache = __hip_atomic_load(
          args.headMemObj->template GetAs<uint64_t*>(destPe) + channelId * npes + myPe,
          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      uint64_t p2pTailCache =
          __hip_atomic_load(args.tailMemObj->template GetAs<uint64_t*>() + destPe + channelId * npes,
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

      index_t* recvTokenMap = args.srcPeTokenIdxMap +
                              (channelId * nlocalPes + dstLocalPe) * maxChannelRecvTokensPerRank;

      for (int idx = 0; idx < numChannelTokens; ++idx) {
        index_t localTokenIdx = recvTokenMap[idx];
        // int destNode = args.dispReceiverIdxMap[localTokenIdx];

        // TODO check p2p buffer ready
        while (p2pTailCache - p2pHeadCache >= maxP2PStagingTokens) {
          if (laneId == 0) {
            p2pHeadCache = __hip_atomic_load(
                args.headMemObj->template GetAs<uint64_t*>() + channelId * npes + destPe,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
          p2pHeadCache = __shfl(p2pHeadCache, 0);
        }

        index_t slot = p2pTailCache % maxP2PStagingTokens;
        char* dstPtr =
            args.shmemOutTokMemObj->template GetAs<char*>(destPe) +
            ((channelId * nlocalPes + myLocalPe) * maxP2PStagingTokens + slot) * tokenPackBytes;
        core::WarpCopy(dstPtr,
                       reinterpret_cast<char*>(args.inpTokenBuf) + localTokenIdx * tokenBytes,
                       tokenBytes);
        if (args.weightsBuf) {
          core::WarpCopy(dstPtr + weightsOffset,
                         reinterpret_cast<char*>(args.weightsBuf) + localTokenIdx * weightBytes,
                         weightBytes);
        }

        ++p2pTailCache;
        if ((idx + 1) % maxRDMAStagingTokens == 0 || idx == numChannelTokens - 1) {
          if (laneId == 0) {
            __hip_atomic_store(
                args.tailMemObj->template GetAs<uint64_t*>(destPe) + channelId * npes + myPe,
                p2pTailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
        }
      }
    } else {
      const int destNode = warpId % nNodes;
      const int destPe = destNode * nlocalPes + myLocalPe;
      const int subWarpId = warpId / nNodes;
      const int nSubWarps = (warpNum - kSendWarpCount) / nNodes;

      index_t* rdmaRecvTokensNum = args.rdmaRecvTokensNum + channelId * nNodes + destNode;
      const index_t numTokensToCombine = *rdmaRecvTokensNum;
      if (subWarpId == 0 && laneId == 0) {
        // clear rdmaRecvTokensNum
        *rdmaRecvTokensNum = 0;
      }

      index_t* fwdTokenMap =
          args.fwdTokenMap + (channelId * nNodes + destNode) * maxTokensPerChannel * nlocalPes;

      uint64_t* p2pHeadPtr =
          laneId < nlocalPes
              ? args.headMemObj->template GetAs<uint64_t*>(destNode * nlocalPes + laneId) +
                    channelId * npes + myPe
              : nullptr;
      uint64_t* p2pTailPtr = laneId < nlocalPes
                                 ? args.tailMemObj->template GetAs<uint64_t*>() + channelId * npes +
                                       (destNode * nlocalPes + laneId)
                                 : nullptr;

      uint64_t p2pHeadCache = laneId < nlocalPes ? __hip_atomic_load(p2pHeadPtr, __ATOMIC_RELAXED,
                                                                     __HIP_MEMORY_SCOPE_SYSTEM)
                                                 : 0;
      uint64_t p2pTailCache = laneId < nlocalPes ? __hip_atomic_load(p2pTailPtr, __ATOMIC_RELAXED,
                                                                     __HIP_MEMORY_SCOPE_SYSTEM)
                                                 : 0;
      // TODO 这里会有问题，tail由其他rank更新
      uint64_t p2pTailBase = p2pTailCache;

      uint64_t headCache = __hip_atomic_load(args.headMemObj->template GetAs<uint64_t*>() +
                                                 channelId * npes + (node * nlocalPes + myLocalPe),
                                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      uint64_t tailCache = args.localTail[channelId * npes + destPe];

      for (int idx = 0; idx < numTokensToCombine; idx += stepRDMATokens) {
        // TODO check rdma buffer ready
        while (tailCache - headCache >= maxRDMASteps) {
          if (laneId == node) {
            headCache = __hip_atomic_load(args.headMemObj->template GetAs<uint64_t*>() +
                                              channelId * npes + (node * nlocalPes + myLocalPe),
                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
          headCache = __shfl(headCache, node);
        }

        for (int i = subWarpId; i < stepRDMATokens && i + idx < numTokensToCombine;
             i += nSubWarps) {
          index_t expectTail =
              laneId < nlocalPes ? fwdTokenMap[(idx + i) * nlocalPes + laneId] - 1 : -1;

          // wait data
          while(true) {
            bool dataReady = p2pTailCache > expectTail + p2pTailBase;
            if (__all(dataReady)) break;
            if (!dataReady) {
              p2pTailCache =
                  __hip_atomic_load(p2pTailPtr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
          }

          char* dstPtr =
              (destNode == myNode ? args.shmemInpTokMemObj->template GetAs<char*>()
                                  : args.shmemStagingTokMemObj->template GetAs<char*>()) +
              ((channelId * nNodes + destNode) * maxRDMAStagingTokens +
               (tailCache % maxRDMASteps) * stepRDMATokens + i) *
                  tokenPackBytes;

          T* srcPtrs[MAX_GPUS_PER_NODE];
          int accumNum = 0;
          for (int pe = 0; pe < nlocalPes; ++pe) {
            uint64_t slot = __shfl(expectTail + p2pTailBase, pe);
            if (slot != -1) {
              srcPtrs[accumNum++] = args.shmemOutTokMemObj->template GetAs<char*>() +
                                    ((channelId * nlocalPes + pe) * maxP2PStagingTokens +
                                     slot % maxP2PStagingTokens) *
                                        tokenPackBytes;
            }
          }
          // TODO 记录当前slot，维护一个nNodes*nlocalPes的shmem，用来更新p2p buffer的head

          core::WarpAccum<T, 8>(reinterpret_cast<T*>(dstPtr), srcPtrs, nullptr, accumNum,
                                config.hiddenDim);

          if (args.weightsBuf) {
            float* srcWeightPtrs[MAX_GPUS_PER_NODE];
            for (int n = 0 ; n < accumNum ; ++n) {
              srcWeightPtrs[n] = reinterpret_cast<char*>(srcPtrs[n]) + tokenBytes;
            }
            core::WarpAccum<float, 4>(
                reinterpret_cast<float*>(reinterpret_cast<char*>(dstPtr) + tokenBytes),
                srcWeightPtrs, nullptr, accumNum, config.numExpertPerToken);
          }
        }

        // TODO sub warp barrier

        if (subWarpId == nSubWarps - 1 && destNode != myNode) {
          size_t dstStagingOffset = ((channelId * nNodes + myNode) * maxRDMAStagingTokens +
                                     (tailCache % maxRDMASteps) * stepRDMATokens) *
                                    tokenPackBytes;
          size_t srcStagingOffset = ((channelId * nNodes + destNode) * maxRDMAStagingTokens +
                                     (tailCache % maxRDMASteps) * stepRDMATokens) *
                                    tokenPackBytes;
          shmem::ShmemPutTypeNbiWarp<uint8_t>(
              args.shmemInpTokMemObj, dstStagingOffset, args.shmemStagingTokMemObj,
              srcStagingOffset, min(stepRDMATokens, numTokensToCombine - idx) * tokenPackBytes,
              destPe, channelId);
        }

        // Update rmda tail
        ++tailCache;
        if (subWarpId == nSubWarps - 1 && laneId == destNode) {
          // if (destNode == myNode) {
          //   __hip_atomic_store(
          //       args.tailMemObj->template GetAs<uint64_t*>(destPe) + myPe + channelId * npes,
          //       tailCache, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          // } else {
            shmem::ShmemPutUint64ImmNbiThread(args.tailMemObj,
                                              (myPe + channelId * npes) * sizeof(uint64_t),
                                              tailCache, destPe, channelId);
#if COMB_DEBUG == 1
            // will report cqe error
            shmem::ShmemQuietThread(destPe, channelId);
#endif
          // }
        }
      }
    }
  } else {  // Receiver
    constexpr int kRecvWarpCount = 8;
    if (warpId < kRecvWarpCount){
      const index_t remTokens = tokenNum % nChannels;
      const index_t channelStartOffset =
          channelId * baseTokensPerChannel + min(channelId, remTokens);
      const index_t tokensPerChannel = baseTokensPerChannel + (channelId < remTokens ? 1 : 0);
      const index_t channelEndOffset = min(channelStartOffset + tokensPerChannel, tokenNum);
      index_t* tokenIdxToSlotMap = reinterpret_cast<index_t*>(args.localPeBuf);

      uint64_t* tailPtr = laneId < nNodes ? args.tailMemObj->template GetAs<uint64_t*>() +
                                                channelId * npes + (laneId * nlocalPes + myLocalPe)
                                          : nullptr;
      uint64_t headBase =
          laneId < nNodes ? args.localHead[channelId * npes + (laneId * nlocalPes + myLocalPe)] : 0;
      uint64_t tailCache =
          laneId < nNodes ? __hip_atomic_load(tailPtr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                          : 0;
      uint64_t tailBase = tailCache;

      index_t* tokenIdxToSlotMap = reinterpret_cast<index_t*>(args.localPeBuf);
      for (int tokenIdx = channelStartOffset + warpId; tokenIdx < channelEndOffset;
           tokenIdx += (kRecvWarpCount - 1)) {
        index_t expectTail =
            laneId < nNodes ? tokenIdxToSlotMap[tokenIdx * nNodes + laneId] - 1 : -1;
        // wait data
        while (true) {
          bool dataReady = tailCache * stepRDMATokens > expectTail + tailBase * stepRDMATokens;
          if (__all(dataReady)) break;
          if (!dataReady) {
            tailCache = __hip_atomic_load(tailPtr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
        }

        T* srcPtrs[MAX_NODES];
        int accumNum = 0;
        for (int node = 0; node < nNodes; ++node) {
          uint64_t slot = __shfl(expectTail + tailBase * stepRDMATokens, node);
          if (slot != -1) {
            srcPtrs[accumNum++] =
                args.shmemInpTokMemObj->template GetAs<char*>() +
                ((channelId * nNodes + node) * maxRDMAStagingTokens + slot % maxRDMAStagingTokens) *
                    tokenPackBytes;
          }
        }
        core::WarpAccum<T, 8>(args.outTokenBuf + tokenIdx * config.hiddenDim, srcPtrs, nullptr,
                              accumNum, config.hiddenDim);

        if (args.weightsBuf) {
          float* srcWeightPtrs[MAX_GPUS_PER_MAX_NODESNODE];
          for (int n = 0; n < accumNum; ++n) {
            srcWeightPtrs[n] = reinterpret_cast<char*>(srcPtrs[n]) + tokenBytes;
          }
          core::WarpAccum<float, 4>(args.shmemOutWeightsMemObj->template GetAs<float*>() +
                                        tokenIdx * config.numExpertPerToken,
                                    srcWeightPtrs, nullptr, accumNum, config.numExpertPerToken);
        }
      }
      // TODO update rdma head
    }
  }
}

}  // namespace moe
}  // namespace mori

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

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

namespace mori {
namespace moe {

// Encode/decode a (pe, localTokId) pair into a flat global token index for bookkeeping.
// Stride: MaxNumTokensToRecv(), which guarantees uniqueness across all PEs.
inline __device__ int FlatTokenIndex(const EpDispatchCombineConfig& config, int pe,
                                     int localTokId) {
  return pe * config.MaxNumTokensToRecv() + localTokId;
}
inline __device__ int PeFromFlatTokenIndex(const EpDispatchCombineConfig& config, int flatIdx) {
  return flatIdx / config.MaxNumTokensToRecv();
}
inline __device__ int LocalTokIdFromFlatTokenIndex(const EpDispatchCombineConfig& config,
                                                   int flatIdx) {
  return flatIdx % config.MaxNumTokensToRecv();
}
inline __device__ int NullFlatTokenIndex(const EpDispatchCombineConfig& config) {
  return config.worldSize * config.MaxNumTokensToRecv();
}

// Encode/decode a flat offset into a per-PE staging buffer.
// Stride: MaxNumTokensToRecvPerRank, which determines how much buffer space is allocated per PE.
inline __device__ int BufSlotOffset(const EpDispatchCombineConfig& config, int pe, int slotId) {
  return pe * config.MaxNumTokensToRecvPerRank() + slotId;
}
inline __device__ int PeFromBufSlotOffset(const EpDispatchCombineConfig& config, int flatIdx) {
  return flatIdx / config.MaxNumTokensToRecvPerRank();
}
inline __device__ int SlotIdFromBufSlotOffset(const EpDispatchCombineConfig& config, int flatIdx) {
  return flatIdx % config.MaxNumTokensToRecvPerRank();
}

// Encode/decode a flat offset into a per-PE send staging buffer.
// Stride: MaxNumTokensToSendPerRank, which determines how much buffer space is allocated per PE.
// NullSendBufSlotOffset() returns an out-of-range sentinel indicating "already sent".
inline __device__ int SendBufSlotOffset(const EpDispatchCombineConfig& config, int pe, int slotId) {
  return pe * config.MaxNumTokensToSendPerRank() + slotId;
}
inline __device__ int PeFromSendBufSlotOffset(const EpDispatchCombineConfig& config, int flatIdx) {
  return flatIdx / config.MaxNumTokensToSendPerRank();
}
inline __device__ int SlotIdFromSendBufSlotOffset(const EpDispatchCombineConfig& config,
                                                  int flatIdx) {
  return flatIdx % config.MaxNumTokensToSendPerRank();
}
inline __device__ int NullSendBufSlotOffset(const EpDispatchCombineConfig& config) {
  return config.worldSize * config.MaxNumTokensToSendPerRank();
}

#define DEF_COMMON_VARS                                                                         \
  const EpDispatchCombineConfig& config = args.config;                                          \
  int thdId = threadIdx.x;                                                                      \
  int thdNum = blockDim.x;                                                                      \
  int laneId = threadIdx.x & (warpSize - 1);                                                    \
  int warpId = thdId / warpSize;                                                                \
  int warpNum = blockDim.x / warpSize;                                                          \
  int blockNum = gridDim.x;                                                                     \
  int blockId = blockIdx.x;                                                                     \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;                                      \
  int globalThdNum = gridDim.x * blockDim.x;                                                    \
  int globalWarpId = blockIdx.x * warpNum + warpId;                                             \
  int globalWarpNum = gridDim.x * warpNum;                                                      \
  int nullTokenId = NullFlatTokenIndex(config);                                                 \
  int myPe = config.rank;                                                                       \
  int npes = config.worldSize;                                                                  \
  int myNode = myPe / config.gpuPerNode;                                                        \
  int nNodes = npes / config.gpuPerNode;                                                        \
  int numExpertPerToken = config.numExpertPerToken;                                             \
  assert(numExpertPerToken < warpSize);                                                         \
  size_t hiddenBytes = config.hiddenDim * sizeof(T);                                            \
  size_t indexBytes = config.numExpertPerToken * sizeof(index_t);                               \
  size_t weightBytes = config.numExpertPerToken * sizeof(float);                                \
  size_t srcTokenIdBytes = sizeof(index_t);                                                     \
  size_t scaleBytes = (args.config.scaleDim == 0) ? 0 : config.scaleDim * config.scaleTypeSize; \
  size_t xferBytes = hiddenBytes + indexBytes + weightBytes + srcTokenIdBytes + scaleBytes;     \
  size_t combXferBytes = (args.weightsBuf == nullptr) ? hiddenBytes : hiddenBytes + weightBytes;

}  // namespace moe
}  // namespace mori

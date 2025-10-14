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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/library_types.h>

#include <sstream>
#include <variant>

#include "mori/application/application.hpp"

namespace mori {
namespace moe {

enum KernelType {
  IntraNode = 0,
  InterNode = 1,
};

inline const char* HipDataTypeToString(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_16F:
      return "HIP_R_16F";
    case HIP_R_32F:
      return "HIP_R_32F";
    case HIP_R_16BF:
      return "HIP_R_16BF";
    case HIP_R_8F_E4M3_FNUZ:
      return "HIP_R_8F_E4M3_FNUZ";
    default:
      return "Unknown";
  }
}

inline size_t GetHipDataTypeSize(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_32F:
      return sizeof(float);
    case HIP_R_16BF:
      return sizeof(hip_bfloat16);
    case HIP_R_8F_E4M3_FNUZ:
      return sizeof(__hip_fp8_e4m3_fnuz);
    default:
      throw std::runtime_error("Unknown hipDataType");
  }
}

using index_t = int32_t;

#define MAX_EXPERTS_PER_TOKEN (8)
struct EpDispatchCombineConfig {
  int rank{0};
  int worldSize{0};
  int hiddenDim{4096};
  int scaleDim{32};
  int scaleTypeSize{1};
  int maxTokenTypeSize{4};
  int maxNumInpTokenPerRank{128};
  int numExpertPerRank{1};
  int numExpertPerToken{2};
  int warpNumPerBlock{1};
  int blockNum{1};
  // If true, use external buffer which incurs extra copy overhead; otherwise, the kernel assumes
  // the provided buffer is shmemInpTokMemObj
  bool useExternalInpBuffer{true};

  inline __host__ __device__ int MaxNumTokensToSendPerRank() const { return maxNumInpTokenPerRank; }

  inline __host__ __device__ int MaxNumTokensToSend() const {
    return worldSize * MaxNumTokensToSendPerRank();
  }

  inline __host__ __device__ int MaxNumTokensToRecvPerRank() const { return maxNumInpTokenPerRank; }

  inline __host__ __device__ int MaxNumTokensToRecv() const {
    return worldSize * MaxNumTokensToRecvPerRank();
  }
};

class EpDispatchCombineHandle {
 public:
  EpDispatchCombineHandle(EpDispatchCombineConfig config);
  ~EpDispatchCombineHandle();

  void PrepareInference(hipDataType inputType, void* input, void* output, float* weights,
                        index_t* tokenIndices, index_t numToken) {
    this->inputType = inputType;
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->weightsBuf = weights;
    this->tokenIndices = tokenIndices;
    this->curRankNumToken = numToken;
    // printf("handle inputType %s\n", HipDataTypeToString(inputType));
  }

  void PrepareInference(hipDataType inputType, void* input, void* output, float* weights,
                        uint8_t* scales, index_t* tokenIndices, index_t numToken) {
    this->inputType = inputType;
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->weightsBuf = weights;
    this->scalesBuf = scales;
    this->tokenIndices = tokenIndices;
    this->curRankNumToken = numToken;
    // printf("handle inputType %s\n", HipDataTypeToString(inputType));
  }

  // When blockNum and warpPerBlock <= 0, kernel will use default values in config
  void LaunchIntraNodeDispatch(int blockNum = -1, int warpPerBlock = -1, hipStream_t = 0);
  void LaunchInterNodeDispatch(int blockNum = -1, int warpPerBlock = -1, hipStream_t = 0);
  void LaunchIntraNodeCombine(int blockNum = -1, int warpPerBlock = -1, hipStream_t = 0);
  void LaunchInterNodeCombine(int blockNum = -1, int warpPerBlock = -1, hipStream_t = 0);

  void LaunchDispatch(KernelType, int blockNum = -1, int warpPerBlock = -1, hipStream_t = 0);
  void LaunchCombine(KernelType, int blockNum = -1, int warpPerBlock = -1, hipStream_t = 0);
  void LaunchReset(hipStream_t = 0);

  index_t GetCurRankNumToken() const { return curRankNumToken; }

 private:
  void InitializeShmemBuf();
  void FinalizeShmemBuf();

  void InitializeTokenNumSignalBuf();
  void FinalizeTokenNumSignalBuf();

  void InitializeOrderMapBuf();
  void FinalizeOrderMapBuf();

  void InitializeBarrier();
  void FinalizeBarrier();

 public:
  // Number of tokens on this rank and size of scale data type, updated at each round of inference
  index_t curRankNumToken{0};

 public:
  // Config
  EpDispatchCombineConfig config;
  // Routed expert indices for tokens
  index_t* tokenIndices{nullptr};

  // Kernel input/output buffer
  void* inpTokenBuf{nullptr};
  void* outTokenBuf{nullptr};
  hipDataType inputType;
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};

  // Registered buffers for tokens, shmemOutTokMemObj will be returned to user as output
  mori::application::SymmMemObjPtr shmemDispatchInpTokMemObj;
  mori::application::SymmMemObjPtr shmemCombineInpTokMemObj;
  mori::application::SymmMemObjPtr shmemOutTokMemObj;
  mori::application::SymmMemObjPtr shmemStagingTokMemObj;

  // Registered buffer used for weights, indices and scales
  mori::application::SymmMemObjPtr shmemInpWeightsMemObj;
  mori::application::SymmMemObjPtr shmemOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemInpScalesMemObj;
  mori::application::SymmMemObjPtr shmemOutScalesMemObj;
  mori::application::SymmMemObjPtr shmemInpIndicesMemObj;
  mori::application::SymmMemObjPtr shmemOutIndicesMemObj;

  // Record number of tokens that will be received from other PE
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  mori::application::SymmMemObjPtr sendAtomicSignalMemObj;

  // Barrier for intra-grid synchronization
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};

  // Map dispatch input token index to staging buffer index, saved at dispatch send phase and used
  // at combine recv phase
  index_t* dispSenderIdxMap{nullptr};
  // Map dispatch staging buffer index to output buffer index, saved at dispatch recv phase and used
  // at combine send phase
  index_t* dispReceiverIdxMap{nullptr};

  // Map staging buffer index to dispatch input token index, saved at dispatch init phase and used
  // at dispatch send phase
  index_t* destPeTokenIdxMap{nullptr};
  // Map output buffer index to combine input token index, saved at dispatch recv phase and used at
  // combine send phase
  index_t* srcPeTokenIdxMap{nullptr};

  // Count the number of tokens sent to destination pe
  index_t* destPeTokenCounter{nullptr};
  // Count the number of tokens sent to local pe
  index_t* localPeTokenCounter{nullptr};

  // Intra-node kernel parameters
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint32_t crossDeviceBarrierFlag{1};
};

template <typename T>
struct EpDispatchCombineArgs {
  using data_type = T;
  EpDispatchCombineConfig config;
  index_t curRankNumToken{0};
  index_t* tokenIndices{nullptr};
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};
  mori::application::SymmMemObjPtr shmemDispatchInpTokMemObj;
  mori::application::SymmMemObjPtr shmemCombineInpTokMemObj;
  mori::application::SymmMemObjPtr shmemOutTokMemObj;
  mori::application::SymmMemObjPtr shmemStagingTokMemObj;
  mori::application::SymmMemObjPtr shmemInpWeightsMemObj;
  mori::application::SymmMemObjPtr shmemOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemInpScalesMemObj;
  mori::application::SymmMemObjPtr shmemOutScalesMemObj;
  mori::application::SymmMemObjPtr shmemInpIndicesMemObj;
  mori::application::SymmMemObjPtr shmemOutIndicesMemObj;
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  mori::application::SymmMemObjPtr sendAtomicSignalMemObj;
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};
  index_t* destPeTokenCounter{nullptr};
  index_t* localPeTokenCounter{nullptr};
  index_t* dispReceiverIdxMap{nullptr};
  index_t* dispSenderIdxMap{nullptr};
  index_t* destPeTokenIdxMap{nullptr};
  index_t* srcPeTokenIdxMap{nullptr};
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint32_t crossDeviceBarrierFlag{1};
};

using EpDispatchCombineArgsVariant =
    std::variant<EpDispatchCombineArgs<float>, EpDispatchCombineArgs<hip_bfloat16>,
                 EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> >;

template <typename T>
EpDispatchCombineArgs<T> GetEpDispatchCombineArgs(const EpDispatchCombineHandle& handle) {
  EpDispatchCombineArgs<T> args;
  args.config = handle.config;
  args.curRankNumToken = handle.curRankNumToken;
  args.tokenIndices = handle.tokenIndices;
  args.inpTokenBuf = reinterpret_cast<T*>(handle.inpTokenBuf);
  args.outTokenBuf = reinterpret_cast<T*>(handle.outTokenBuf);
  args.weightsBuf = handle.weightsBuf;
  args.scalesBuf = handle.scalesBuf;
  args.destPeTokenCounter = handle.destPeTokenCounter;
  args.localPeTokenCounter = handle.localPeTokenCounter;
  args.shmemDispatchInpTokMemObj = handle.shmemDispatchInpTokMemObj;
  args.shmemCombineInpTokMemObj = handle.shmemCombineInpTokMemObj;
  args.shmemOutTokMemObj = handle.shmemOutTokMemObj;
  args.shmemStagingTokMemObj = handle.shmemStagingTokMemObj;
  args.shmemInpWeightsMemObj = handle.shmemInpWeightsMemObj;
  args.shmemOutWeightsMemObj = handle.shmemOutWeightsMemObj;
  args.shmemInpScalesMemObj = handle.shmemInpScalesMemObj;
  args.shmemOutScalesMemObj = handle.shmemOutScalesMemObj;
  args.shmemInpIndicesMemObj = handle.shmemInpIndicesMemObj;
  args.shmemOutIndicesMemObj = handle.shmemOutIndicesMemObj;
  args.recvTokenNumMemObj = handle.recvTokenNumMemObj;
  args.sendTokenNumMemObj = handle.sendTokenNumMemObj;
  args.sendAtomicSignalMemObj = handle.sendAtomicSignalMemObj;
  args.dispatchGridBarrier = handle.dispatchGridBarrier;
  args.combineGridBarrier = handle.combineGridBarrier;
  args.dispReceiverIdxMap = handle.dispReceiverIdxMap;
  args.dispSenderIdxMap = handle.dispSenderIdxMap;
  args.destPeTokenIdxMap = handle.destPeTokenIdxMap;
  args.srcPeTokenIdxMap = handle.srcPeTokenIdxMap;
  args.dispTokOffsetMemObj = handle.dispTokOffsetMemObj;
  args.dispTokIdToSrcTokIdMemObj = handle.dispTokIdToSrcTokIdMemObj;
  args.dispDestTokIdMap = handle.dispDestTokIdMap;
  args.totalRecvTokenNum = handle.totalRecvTokenNum;
  args.crossDeviceBarrierMemObj = handle.crossDeviceBarrierMemObj;
  args.crossDeviceBarrierFlag = handle.crossDeviceBarrierFlag;
  return args;
}

inline EpDispatchCombineArgsVariant GetEpDispatchCombineArgsByInputType(
    const EpDispatchCombineHandle& handle) {
  switch (handle.inputType) {
    case HIP_R_32F:
      return GetEpDispatchCombineArgs<float>(handle);
    case HIP_R_16BF:
      return GetEpDispatchCombineArgs<hip_bfloat16>(handle);
    case HIP_R_8F_E4M3_FNUZ:
      return GetEpDispatchCombineArgs<__hip_fp8_e4m3_fnuz>(handle);
    default:
      std::ostringstream oss;
      oss << "Unsupported inputType " << HipDataTypeToString(handle.inputType)
          << " in GetEpDispatchCombineArgsByInputType";
      throw std::runtime_error(oss.str());
  }
}

}  // namespace moe
}  // namespace mori

namespace std {

static std::ostream& operator<<(std::ostream& s, mori::moe::EpDispatchCombineConfig config) {
  std::stringstream ss;
  ss << "EpDispatchCombineConfig: " << std::endl
     << "  WorldSize: " << config.worldSize << std::endl
     << "  hiddenDim: " << config.hiddenDim << std::endl
     << "  scaleDim: " << config.scaleDim << std::endl
     << "  scaleTypeSize: " << config.scaleTypeSize << std::endl
     << "  maxTokenTypeSize: " << config.maxTokenTypeSize << std::endl
     << "  maxNumInpTokenPerRank: " << config.maxNumInpTokenPerRank << std::endl
     << "  numExpertPerRank: " << config.numExpertPerRank << std::endl
     << "  numExpertPerToken: " << config.numExpertPerToken << std::endl
     << "  warpNumPerBlock: " << config.warpNumPerBlock << std::endl
     << "  blockNum: " << config.blockNum;
  s << ss.str();
  return s;
}

}  // namespace std

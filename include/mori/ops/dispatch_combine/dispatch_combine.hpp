#pragma once

#include <sstream>

#include "mori/application/application.hpp"

namespace mori {
namespace moe {

enum KernelType {
  IntraNode = 0,
  InterNode = 1,
};

using index_t = int32_t;

struct EpDispatchCombineConfig {
  int rank{0};
  int worldSize{0};
  int hiddenDim{4096};
  int scaleDim{32};
  int scaleTypeSize{1};
  int maxNumInpTokenPerRank{128};
  int numExpertPerRank{1};
  int numExpertPerToken{2};
  int warpNumPerBlock{1};
  int blockNum{1};

  inline __host__ __device__ int MaxNumTokensToSendPerRank() const {
    return maxNumInpTokenPerRank * numExpertPerToken;
  }

  inline __host__ __device__ int MaxNumTokensToSend() const {
    return worldSize * maxNumInpTokenPerRank * numExpertPerToken;
  }

  inline __host__ __device__ int MaxNumTokensToRecvPerRank() const {
    return maxNumInpTokenPerRank * std::min(numExpertPerRank, numExpertPerToken);
  }

  inline __host__ __device__ int MaxNumTokensToRecv() const {
    return worldSize * MaxNumTokensToRecvPerRank();
  }
};

template <typename T>
class EpDispatchCombineHandle {
 public:
  EpDispatchCombineHandle(EpDispatchCombineConfig config);
  ~EpDispatchCombineHandle();

  void PrepareInference(T* input, T* output, float* weights, index_t* tokenIndicies,
                        index_t numToken) {
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->weightsBuf = weights;
    this->tokenIndicies = tokenIndicies;
    this->curRankNumToken = numToken;
  }

  void PrepareInference(T* input, T* output, float* weights, uint8_t* scales,
                        index_t* tokenIndicies, index_t numToken) {
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->weightsBuf = weights;
    this->scalesBuf = scales;
    this->tokenIndicies = tokenIndicies;
    this->curRankNumToken = numToken;
  }

  void LaunchIntraNodeDispatch(hipStream_t = 0);
  void LaunchInterNodeDispatch(hipStream_t = 0);
  void LaunchIntraNodeCombine(hipStream_t = 0);
  void LaunchInterNodeCombine(hipStream_t = 0);

  void LaunchDispatch(KernelType, hipStream_t = 0);
  void LaunchCombine(KernelType, hipStream_t = 0);
  void LaunchReset(hipStream_t = 0);

  index_t GetCurRankNumToken() const { return curRankNumToken; }

 private:
  void IntializeShmemBuf();
  void FinalizeShmemBuf();

  void IntializeTokenNumSignalBuf();
  void FinalizeTokenNumSignalBuf();

  void IntializeOrderMapBuf();
  void FinalizeOrderMapBuf();

  void IntializeBarrier();
  void FinalizeBarrier();

 public:
  // Number of tokens on this rank and size of scale data type, updated at each round of inference
  index_t curRankNumToken{0};

 public:
  // Config
  EpDispatchCombineConfig config;
  // Routed expert indices for tokens
  index_t* tokenIndicies{nullptr};

  // Kernel input/output buffer
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};

  // Registered buffers for tokens, shmemOutTokMemObj will be returned to user as output
  mori::application::SymmMemObjPtr shmemInpTokMemObj;
  mori::application::SymmMemObjPtr shmemOutTokMemObj;
  mori::application::SymmMemObjPtr shmemStagingTokMemObj;

  // Registered buffer used for weights, indicies and scales
  mori::application::SymmMemObjPtr shmemInpWeightsMemObj;
  mori::application::SymmMemObjPtr shmemOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemInpScalesMemObj;
  mori::application::SymmMemObjPtr shmemOutScalesMemObj;
  mori::application::SymmMemObjPtr shmemInpIndiciesMemObj;
  mori::application::SymmMemObjPtr shmemOutIndiciesMemObj;

  // Record number of tokens that will be received from other PE
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;

  // Barrier for intra-grid synchronization
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};

  // Map dispatch input token index to staging buffer index, saved at dispatch send phase and used
  // at combine recv phase
  index_t* dispSenderIdxMap{nullptr};
  // Map dispatch staging buffer index to output buffer index, saved at dispatch recv phase and used
  // at combine send phase
  index_t* dispReceiverIdxMap{nullptr};

  // Count the number of tokens sent to destination pe
  index_t* destPeTokenCounter{nullptr};
  // Count the number of tokens sent to local pe
  index_t* localPeTokenCounter{nullptr};

  // Lock for guarding shmem ops
  uint32_t* lock{nullptr};

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
  EpDispatchCombineConfig config;
  index_t curRankNumToken{0};
  index_t* tokenIndicies{nullptr};
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};
  mori::application::SymmMemObjPtr shmemInpTokMemObj;
  mori::application::SymmMemObjPtr shmemOutTokMemObj;
  mori::application::SymmMemObjPtr shmemStagingTokMemObj;
  mori::application::SymmMemObjPtr shmemInpWeightsMemObj;
  mori::application::SymmMemObjPtr shmemOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemInpScalesMemObj;
  mori::application::SymmMemObjPtr shmemOutScalesMemObj;
  mori::application::SymmMemObjPtr shmemInpIndiciesMemObj;
  mori::application::SymmMemObjPtr shmemOutIndiciesMemObj;
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};
  index_t* destPeTokenCounter{nullptr};
  index_t* localPeTokenCounter{nullptr};
  uint32_t* lock{nullptr};
  index_t* dispReceiverIdxMap{nullptr};
  index_t* dispSenderIdxMap{nullptr};
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint32_t crossDeviceBarrierFlag{1};
};

template <typename T>
EpDispatchCombineArgs<T> GetEpDispatchCombineArgs(const EpDispatchCombineHandle<T>& handle) {
  EpDispatchCombineArgs<T> args;
  args.config = handle.config;
  args.curRankNumToken = handle.curRankNumToken;
  args.tokenIndicies = handle.tokenIndicies;
  args.inpTokenBuf = handle.inpTokenBuf;
  args.outTokenBuf = handle.outTokenBuf;
  args.weightsBuf = handle.weightsBuf;
  args.scalesBuf = handle.scalesBuf;
  args.destPeTokenCounter = handle.destPeTokenCounter;
  args.localPeTokenCounter = handle.localPeTokenCounter;
  args.lock = handle.lock;
  args.shmemInpTokMemObj = handle.shmemInpTokMemObj;
  args.shmemOutTokMemObj = handle.shmemOutTokMemObj;
  args.shmemStagingTokMemObj = handle.shmemStagingTokMemObj;
  args.shmemInpWeightsMemObj = handle.shmemInpWeightsMemObj;
  args.shmemOutWeightsMemObj = handle.shmemOutWeightsMemObj;
  args.shmemInpScalesMemObj = handle.shmemInpScalesMemObj;
  args.shmemOutScalesMemObj = handle.shmemOutScalesMemObj;
  args.shmemInpIndiciesMemObj = handle.shmemInpIndiciesMemObj;
  args.shmemOutIndiciesMemObj = handle.shmemOutIndiciesMemObj;
  args.recvTokenNumMemObj = handle.recvTokenNumMemObj;
  args.sendTokenNumMemObj = handle.sendTokenNumMemObj;
  args.dispatchGridBarrier = handle.dispatchGridBarrier;
  args.combineGridBarrier = handle.combineGridBarrier;
  args.dispReceiverIdxMap = handle.dispReceiverIdxMap;
  args.dispSenderIdxMap = handle.dispSenderIdxMap;
  args.dispTokOffsetMemObj = handle.dispTokOffsetMemObj;
  args.dispTokIdToSrcTokIdMemObj = handle.dispTokIdToSrcTokIdMemObj;
  args.dispDestTokIdMap = handle.dispDestTokIdMap;
  args.totalRecvTokenNum = handle.totalRecvTokenNum;
  args.crossDeviceBarrierMemObj = handle.crossDeviceBarrierMemObj;
  args.crossDeviceBarrierFlag = handle.crossDeviceBarrierFlag;
  return args;
}

}  // namespace moe
}  // namespace mori

namespace std {

static std::ostream& operator<<(std::ostream& s, mori::moe::EpDispatchCombineConfig config) {
  std::stringstream ss;
  ss << "EpDispatchCombineConfig: " << std::endl
     << "  WorlSize: " << config.worldSize << std::endl
     << "  hiddenDim: " << config.hiddenDim << std::endl
     << "  scaleDim: " << config.scaleDim << std::endl
     << "  scaleTypeSize: " << config.scaleTypeSize << std::endl
     << "  maxNumInpTokenPerRank: " << config.maxNumInpTokenPerRank << std::endl
     << "  numExpertPerRank: " << config.numExpertPerRank << std::endl
     << "  numExpertPerToken: " << config.numExpertPerToken << std::endl
     << "  warpNumPerBlock: " << config.warpNumPerBlock << std::endl
     << "  blockNum: " << config.blockNum;
  s << ss.str();
  return s;
}

}  // namespace std
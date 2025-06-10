#pragma once

#include <sstream>

#include "mori/application/application.hpp"

namespace mori {
namespace moe {

enum KernelType {
  IntraNode = 0,
  InterNode = 1,
};

struct EpDispatchCombineConfig {
  int rank{0};
  int worldSize{0};
  int hiddenDim{4096};
  int maxNumInpTokenPerRank{128};
  int numExpertPerRank{1};
  int numExpertPerToken{2};
  int warpNumPerBlock{1};
  int blockNum{1};

  uint32_t MaxNumOutputTokens() { return worldSize * maxNumInpTokenPerRank * numExpertPerRank; }
};

template <typename T>
class EpDispatchCombineHandle {
 public:
  EpDispatchCombineHandle(EpDispatchCombineConfig config);
  ~EpDispatchCombineHandle();

  void PrepareInference(T* input, T* output, float* weights, uint32_t* tokenIndicies,
                        int numToken) {
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->weightsBuf = weights;
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

 private:
  void IntializeShmemInpOutTokBuf();
  void FinalizeShmemInpOutTokBuf();

  void IntializeTokenNumSignalBuf();
  void FinalizeTokenNumSignalBuf();

  void IntializeTokToExptBuf();
  void FinalizeTokToExptBuf();

  void IntializeOrderMapBuf();
  void FinalizeOrderMapBuf();

  void IntializeBarrier();
  void FinalizeBarrier();

 public:
  // Number of tokens on this rank, updated at each round of inference
  int curRankNumToken{-1};

 public:
  // Config
  EpDispatchCombineConfig config;
  // Routed expert indices for tokens
  uint32_t* tokenIndicies{nullptr};

  // Kernel input/output buffer
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};

  // Temporary buffers of input/output tokens used for shmem ops
  mori::application::SymmMemObjPtr shmemInpTokMemObj;
  mori::application::SymmMemObjPtr shmemOutTokMemObj;

  // Record number of tokens that will be received from other PE
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;

  // Barrier for intra-grid synchronization
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};

  // Buffers for token to expert mapping, only used for shmem ops at dispatch phase
  mori::application::SymmMemObjPtr inpTokToExptMapMemObj;
  mori::application::SymmMemObjPtr outTokToExptMapMemObj;

  // Recover from expert sorted order to pe sorted order, filled at dispatch recv phase and used at
  // combine send phase
  uint32_t* exptSortedToPeSortedBuf{nullptr};
  // Recover from pe sorted order to original order, filled at dispatch send phase and used at
  // combine recv phase
  uint32_t* tokenIndicesToPeSortedBuf{nullptr};

  // Counter used for sorting by PE order
  uint32_t* peTokenOffset{nullptr};
  // Counter used for sorting by expert order
  uint32_t* exptTokenOffset{nullptr};

  // Intra-node kernel parameters
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  uint32_t* dispDestTokIdMap{nullptr};
  uint32_t* totalRecvTokenNum{nullptr};
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint32_t crossDeviceBarrierFlag{1};
};

template <typename T>
struct EpDispatchCombineArgs {
  EpDispatchCombineConfig config;
  int curRankNumToken{-1};
  uint32_t* tokenIndicies{nullptr};
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  mori::application::SymmMemObjPtr shmemInpTokMemObj;
  mori::application::SymmMemObjPtr shmemOutTokMemObj;
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};
  mori::application::SymmMemObjPtr inpTokToExptMapMemObj;
  mori::application::SymmMemObjPtr outTokToExptMapMemObj;
  uint32_t* peTokenOffset{nullptr};
  uint32_t* exptTokenOffset{nullptr};
  uint32_t* exptSortedToPeSortedBuf{nullptr};
  uint32_t* tokenIndicesToPeSortedBuf{nullptr};
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  uint32_t* dispDestTokIdMap{nullptr};
  uint32_t* totalRecvTokenNum{nullptr};
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
  args.peTokenOffset = handle.peTokenOffset;
  args.exptTokenOffset = handle.exptTokenOffset;
  args.shmemInpTokMemObj = handle.shmemInpTokMemObj;
  args.shmemOutTokMemObj = handle.shmemOutTokMemObj;
  args.recvTokenNumMemObj = handle.recvTokenNumMemObj;
  args.sendTokenNumMemObj = handle.sendTokenNumMemObj;
  args.dispatchGridBarrier = handle.dispatchGridBarrier;
  args.combineGridBarrier = handle.combineGridBarrier;
  args.inpTokToExptMapMemObj = handle.inpTokToExptMapMemObj;
  args.outTokToExptMapMemObj = handle.outTokToExptMapMemObj;
  args.exptSortedToPeSortedBuf = handle.exptSortedToPeSortedBuf;
  args.tokenIndicesToPeSortedBuf = handle.tokenIndicesToPeSortedBuf;
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
     << "  maxNumInpTokenPerRank: " << config.maxNumInpTokenPerRank << std::endl
     << "  numExpertPerRank: " << config.numExpertPerRank << std::endl
     << "  numExpertPerToken: " << config.numExpertPerToken << std::endl
     << "  warpNumPerBlock: " << config.warpNumPerBlock << std::endl
     << "  blockNum: " << config.blockNum;
  s << ss.str();
  return s;
}

}  // namespace std
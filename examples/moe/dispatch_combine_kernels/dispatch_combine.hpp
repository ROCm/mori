#pragma once

#include "mori/application/application.hpp"

namespace mori {
namespace moe {

struct EpDispatchCombineConfig {
  int randomSeed{0};
  int rank{0};
  int worldSize{0};
  int numExpertPerRank{1};
  int hiddenDim{4096};
  int numExpertPerToken{2};
  int maxNumInpTokenPerRank{128};
  int warpNumPerBlock{1};
  int blockNum{1};
};

template <typename T>
class EpDispatchCombineHandle {
 public:
  EpDispatchCombineHandle(EpDispatchCombineConfig config);
  ~EpDispatchCombineHandle();

  void PrepareInference(T* input, T* output, uint32_t* tokenIndicies, int numToken) {
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->tokenIndicies = tokenIndicies;
    this->curRankNumToken = numToken;
  }

  void LaunchDispatch();
  void LaunchCombine();
  void LaunchReset();

 private:
  void IntializeShmemInpOutTokBuf();
  void FinalizeShmemInpOutTokBuf();

  void IntializeTokenNumSignalBuf();
  void FinalizeTokenNumSignalBuf();

  void IntializeTokToExptBuf();
  void FinalizeTokToExptBuf();

  void IntializeOrderMapBuf();
  void FinalizeOrderMapBuf();

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
  // Temporary buffers of input/output tokens used for shmem ops
  mori::application::SymmMemObjPtr shmemInpTokMemObj;
  mori::application::SymmMemObjPtr shmemOutTokMemObj;
  // Record number of tokens that will be received from other PE
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  uint32_t* gridCopyTokenBarrier{nullptr};
  // Buffers for token to expert mapping, only used for shmem ops at dispatch phase
  mori::application::SymmMemObjPtr inpTokToExptMapMemObj;
  mori::application::SymmMemObjPtr outTokToExptMapMemObj;

  // Recover from expert sorted order to pe sorted order, filled at dispatch recv phase and used at
  // combine send phase
  uint32_t* exptSortedToPeSortedBuf{nullptr};
  // Recover from pe sorted order to original order, filled at dispatch send phase and used at
  // combine recv phase
  uint32_t* tokenIndicesToPeSortedBuf{nullptr};
};

}  // namespace moe
}  // namespace mori
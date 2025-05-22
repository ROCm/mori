#include <hip/hip_cooperative_groups.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <random>
#include <sstream>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;
using namespace std;

constexpr ProviderType PrvdType = ProviderType::MLX5;

#define DEBUG 0

__device__ void SyncIfDebugEnabled(const char* msg) {
#if DEBUG == 1
  __syncthreads();
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    ShmemQuietThread();
    printf("%s\n", msg);
  }
  __syncthreads();
#endif
}

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
  EpDispatchCombineHandle(EpDispatchCombineConfig config) : config(config) {}

  void Intialize() {
    IntializeTokenIndiciesBuf();
    IntializeShmemInpOutTokBuf();
    IntializeTokenNumSignalBuf();
    IntializeTokToExptBuf();
    IntializeOrderMapBuf();
  }

  void SetInputOutput(void* input, void* output) {
    inpTokenBuf = reinterpret_cast<T*>(input);
    outTokenBuf = reinterpret_cast<T*>(output);
  }

  void Reset() {
    int tokenNumSignalSize = config.worldSize * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMemset(recvTokenNumMemObj->localPtr, 0, tokenNumSignalSize));
    HIP_RUNTIME_CHECK(hipMemset(sendTokenNumMemObj->localPtr, 0, tokenNumSignalSize));
    HIP_RUNTIME_CHECK(hipMemset(gridCopyTokenBarrier, 0, tokenNumSignalSize));
  }

  void ResetBarrier() {
    int tokenNumSignalSize = config.worldSize * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMemset(gridCopyTokenBarrier, 0, tokenNumSignalSize));
  }

 private:
  void IntializeTokenIndiciesBuf() {
    int maxNumTokenIndices = config.maxNumInpTokenPerRank * config.numExpertPerToken;
    int tokenIndiciesSize = maxNumTokenIndices * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMalloc(&tokenIndicies, tokenIndiciesSize));
    HIP_RUNTIME_CHECK(hipMemset(tokenIndicies, 0, tokenIndiciesSize));
  }

  void IntializeShmemInpOutTokBuf() {
    int maxTokenSize = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken *
                       config.hiddenDim * sizeof(T);

    void* shmemInpTokBuf = ShmemMalloc(maxTokenSize);
    HIP_RUNTIME_CHECK(hipMemset(shmemInpTokBuf, 0, maxTokenSize));
    shmemInpTokMemObj = ShmemQueryMemObjPtr(shmemInpTokBuf);
    assert(shmemInpTokMemObj.IsValid());

    void* shmemOutTokBuf = ShmemMalloc(maxTokenSize);
    HIP_RUNTIME_CHECK(hipMemset(shmemOutTokBuf, 0, maxTokenSize));
    shmemOutTokMemObj = ShmemQueryMemObjPtr(shmemOutTokBuf);
    assert(shmemOutTokMemObj.IsValid());
  }

  void IntializeTokenNumSignalBuf() {
    int tokenNumSignalSize = config.worldSize * sizeof(uint32_t);

    void* recvTokenNumBuf = ShmemMalloc(tokenNumSignalSize);
    HIP_RUNTIME_CHECK(hipMemset(recvTokenNumBuf, 0, tokenNumSignalSize));
    recvTokenNumMemObj = ShmemQueryMemObjPtr(recvTokenNumBuf);
    assert(recvTokenNumMemObj.IsValid());

    void* sendTokenNumBuf = ShmemMalloc(tokenNumSignalSize);
    HIP_RUNTIME_CHECK(hipMemset(sendTokenNumBuf, 0, tokenNumSignalSize));
    sendTokenNumMemObj = ShmemQueryMemObjPtr(sendTokenNumBuf);
    assert(sendTokenNumMemObj.IsValid());

    HIP_RUNTIME_CHECK(hipMalloc(&gridCopyTokenBarrier, tokenNumSignalSize));
    HIP_RUNTIME_CHECK(hipMemset(gridCopyTokenBarrier, 0, tokenNumSignalSize));
  }

  void IntializeTokToExptBuf() {
    int tokToExptMapSize = config.worldSize * config.maxNumInpTokenPerRank *
                           config.numExpertPerToken * sizeof(uint32_t);
    void* inpTokToExptMapBuf = ShmemMalloc(tokToExptMapSize);
    HIP_RUNTIME_CHECK(hipMemset(inpTokToExptMapBuf, 0, tokToExptMapSize));
    inpTokToExptMapMemObj = ShmemQueryMemObjPtr(inpTokToExptMapBuf);
    assert(inpTokToExptMapMemObj.IsValid());

    void* outTokToExptMapBuf = ShmemMalloc(tokToExptMapSize);
    HIP_RUNTIME_CHECK(hipMemset(outTokToExptMapBuf, 0, tokToExptMapSize));
    outTokToExptMapMemObj = ShmemQueryMemObjPtr(outTokToExptMapBuf);
    assert(outTokToExptMapMemObj.IsValid());
  }

  void IntializeOrderMapBuf() {
    int maxNumOutToken = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken;
    HIP_RUNTIME_CHECK(hipMalloc(&exptSortedToPeSortedBuf, maxNumOutToken * sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipMemset(exptSortedToPeSortedBuf, 0, maxNumOutToken * sizeof(uint32_t)));

    HIP_RUNTIME_CHECK(hipMalloc(&tokenIndicesToPeSortedBuf, maxNumOutToken * sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipMemset(tokenIndicesToPeSortedBuf, 0, maxNumOutToken * sizeof(uint32_t)));
  }

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
  SymmMemObjPtr shmemInpTokMemObj;
  SymmMemObjPtr shmemOutTokMemObj;
  // Record number of tokens that will be received from other PE
  SymmMemObjPtr recvTokenNumMemObj;
  SymmMemObjPtr sendTokenNumMemObj;
  uint32_t* gridCopyTokenBarrier{nullptr};
  // Buffers for token to expert mapping, only used for shmem ops at dispatch phase
  SymmMemObjPtr inpTokToExptMapMemObj;
  SymmMemObjPtr outTokToExptMapMemObj;

  // Recover from expert sorted order to pe sorted order, filled at dispatch recv phase and used at
  // combine send phase
  uint32_t* exptSortedToPeSortedBuf{nullptr};
  // Recover from pe sorted order to original order, filled at dispatch send phase and used at
  // combine recv phase
  uint32_t* tokenIndicesToPeSortedBuf{nullptr};
};

template <typename T>
__global__ void EpDispatchKernel(EpDispatchCombineConfig config,
                                 EpDispatchCombineHandle<T> handle) {
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  T* inpTokenBuf = handle.shmemInpTokMemObj->template GetAs<T*>();
  uint32_t* inpTokToExptBuf = handle.inpTokToExptMapMemObj->template GetAs<uint32_t*>();
  uint32_t* outTokToExptBuf = handle.outTokToExptMapMemObj->template GetAs<uint32_t*>();

  size_t maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;

  // Send out tokens
  extern __shared__ char sharedMem[];

  // Phase1: send token
  // Each warp compute token offset on destinition PE
  uint32_t* peTokenOffset = reinterpret_cast<uint32_t*>(sharedMem) + warpId * npes;
  for (int i = laneId; i < npes; i += warpSize) {
    peTokenOffset[i] = 0;
  }

  for (int i = 0; i < handle.curRankNumToken * config.numExpertPerToken; i++) {
    // located on the same pe
    uint32_t destExpert = handle.tokenIndicies[i];
    uint32_t destPe = destExpert / config.numExpertPerRank;
    uint32_t peTokenIdx = peTokenOffset[destPe];
    if (laneId == 0) {
      atomicAdd(peTokenOffset + destPe, 1);
      inpTokToExptBuf[destPe * maxNumOutTokenPerRank + peTokenIdx] = destExpert;
    }

    if ((i % globalWarpNum) != globalWarpId) continue;  // skip token not assigned for this warp

    uint32_t tokenId = i / config.numExpertPerToken;
    uint32_t tokenOffset = tokenId * config.hiddenDim;

    uint32_t peSortedId = myPe * maxNumOutTokenPerRank + peTokenIdx;
    uint32_t peSortedOffset = peSortedId * config.hiddenDim;

    WarpCopy(handle.shmemInpTokMemObj->template GetAs<T*>() + tokenOffset,
             handle.inpTokenBuf + tokenOffset, config.hiddenDim);
    ShmemPutTypeNbiWarp<T>(handle.shmemOutTokMemObj, peSortedOffset, handle.shmemInpTokMemObj,
                           tokenOffset, config.hiddenDim, destPe);
    __threadfence_system();
    if (laneId == 0) {
      handle.tokenIndicesToPeSortedBuf[i] = destPe * maxNumOutTokenPerRank + peTokenIdx;
      uint32_t destPeCopyCnt = atomicAdd(handle.gridCopyTokenBarrier + destPe, 1);
    }
  }
  // SyncIfDebugEnabled("Dispatch kernel: finished send token");
  // Make sure WarCopy is visible to other blocks
  __threadfence_system();

  // Send token num & token to expert mapping to other ranks
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    // Add 1 so that when token number == 0, receiver side still know the signal is sent
    uint32_t recvTokenNum = peTokenOffset[destPe];

    // Wait until all tokens are sent
    ShmemUint32WaitUntilEquals(handle.gridCopyTokenBarrier + destPe, recvTokenNum);

    ShmemPutUint32NbiWarp(handle.outTokToExptMapMemObj, myPe * maxNumOutTokenPerRank,
                          handle.inpTokToExptMapMemObj, destPe * maxNumOutTokenPerRank,
                          recvTokenNum, destPe);

    uint32_t numTokenSignal = recvTokenNum + 1;
    ShmemPutUint32ImmNbiWarp(handle.recvTokenNumMemObj, myPe * sizeof(uint32_t), numTokenSignal,
                             destPe);
  }
  SyncIfDebugEnabled("Dispatch kernel: finish sending tok2expt mapping & num token signal");
  __threadfence_system();

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  uint32_t* recvTokenNum = reinterpret_cast<uint32_t*>(sharedMem) + warpId * npes;
  uint32_t* accumExpertTokOffsets = reinterpret_cast<uint32_t*>(sharedMem) + warpNum * npes;
  uint32_t* expertTokOff = reinterpret_cast<uint32_t*>(sharedMem) + warpNum * npes +
                           config.numExpertPerRank + warpId * config.numExpertPerRank;
  for (int i = thdId; i < config.numExpertPerRank; i += thdNum) {
    accumExpertTokOffsets[i] = 0;
  }
  __syncthreads();

  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint32_t* signal = handle.recvTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
    ShmemUint32WaitUntilGreaterThan(signal, 0);
    recvTokenNum[destPe] = *signal - 1;
  }
  // SyncIfDebugEnabled("Dispatch kernel: finish waiting num token signal");

  for (int srcPe = warpId; srcPe < npes; srcPe += warpNum) {
    for (int tokId = laneId; tokId < recvTokenNum[srcPe]; tokId += warpSize) {
      uint32_t expertId = outTokToExptBuf[srcPe * maxNumOutTokenPerRank + tokId];
      uint32_t localExpertId = expertId % config.numExpertPerRank;
      if ((expertId / config.numExpertPerRank) != myPe) {
        printf("mype %d srcpe %d expertId %d\n", myPe, srcPe, expertId);
        assert(false);
      }
      atomicAdd(accumExpertTokOffsets + localExpertId, 1);
    }
  }
  __syncthreads();

  // Calculate prefix sum of expert token offset
  assert(config.numExpertPerRank < warpSize);
  if (thdId < config.numExpertPerRank) {
    uint32_t expertTokNum = accumExpertTokOffsets[thdId];
    uint32_t accumOffset = WarpPrefixSum(expertTokNum, config.numExpertPerRank);
    accumExpertTokOffsets[thdId] = accumOffset;
  }
  __syncthreads();

  for (int i = laneId; i < config.numExpertPerRank; i += warpSize) {
    expertTokOff[i] = 0;
  }

  T* outTokenBuf = handle.outTokenBuf;
  T* shmemOutTokenBuf = handle.shmemOutTokMemObj->template GetAs<T*>();

  for (int i = 0, srcTokId = 0, srcPe = 0; srcPe < npes; srcTokId++, i++) {
    if (srcTokId == recvTokenNum[srcPe]) {
      srcPe++;
      srcTokId = -1;
      continue;
    }

    uint32_t localExpertId =
        outTokToExptBuf[srcPe * maxNumOutTokenPerRank + srcTokId] % config.numExpertPerRank;
    if (laneId == 0) {
      expertTokOff[localExpertId] += 1;
    }

    if ((i % globalWarpNum) != globalWarpId) continue;  // skip token not assigned for this warp

    // Copy token
    uint32_t peSortedId = srcPe * maxNumOutTokenPerRank + srcTokId;
    uint32_t srcTokenOff = peSortedId * config.hiddenDim;

    uint32_t exptSortedId = accumExpertTokOffsets[localExpertId] + expertTokOff[localExpertId] - 1;
    uint32_t destTokenOff = exptSortedId * config.hiddenDim;
    WarpCopy(outTokenBuf + destTokenOff, shmemOutTokenBuf + srcTokenOff, config.hiddenDim);

    if (laneId == 0) {
      handle.exptSortedToPeSortedBuf[exptSortedId] = peSortedId;
    }
  }
}

template <typename T>
__global__ void EpCombineKernel(EpDispatchCombineConfig config, EpDispatchCombineHandle<T> handle) {
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  T* inpTokenBuf = handle.inpTokenBuf;
  T* shmemInpTokenBuf = handle.shmemInpTokMemObj->template GetAs<T*>();
  T* shmemOutTokenBuf = handle.shmemOutTokMemObj->template GetAs<T*>();

  size_t maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;

  // Phase 1: recover tokens from expert sorted order to pe sorted order and send token back
  // Each warp compute total number of recveid tokens
  uint32_t* recvTokenNumBuf = handle.recvTokenNumMemObj->template GetAs<uint32_t*>();
  uint32_t totalNumRecvToken = 0;
  for (int i = laneId; i < npes; i += warpSize) {
    totalNumRecvToken += recvTokenNumBuf[i] - 1;
  }
  totalNumRecvToken = WarpReduceSum(totalNumRecvToken);
  totalNumRecvToken = __shfl(totalNumRecvToken, 0);

  // Recover pe sorted order and send back
  for (int exptSortedId = 0; exptSortedId < totalNumRecvToken; exptSortedId++) {
    if ((exptSortedId % globalWarpNum) != globalWarpId) continue;

    uint32_t peSortedId = handle.exptSortedToPeSortedBuf[exptSortedId];
    uint32_t peSortedOffset = peSortedId * config.hiddenDim;

    uint32_t destPe = peSortedId / maxNumOutTokenPerRank;
    uint32_t peerPeSortedOffset =
        (peSortedId - destPe * maxNumOutTokenPerRank + myPe * maxNumOutTokenPerRank) *
        config.hiddenDim;

    uint32_t exptSortedOffset = exptSortedId * config.hiddenDim;

    WarpCopy(shmemInpTokenBuf + peSortedOffset, inpTokenBuf + exptSortedOffset, config.hiddenDim);
    ShmemPutTypeNbiWarp<T>(handle.shmemOutTokMemObj, peerPeSortedOffset, handle.shmemInpTokMemObj,
                           peSortedOffset, config.hiddenDim, destPe);

    __threadfence();
    if (laneId == 0) {
      uint32_t destPeCopyCnt = atomicAdd(handle.gridCopyTokenBarrier + destPe, 1);
    }
  }
  SyncIfDebugEnabled("Combine kernel: finish recovering from expert sorted to pe sorted");

  // TODO: since we don't have atomic yet, we have to wait untill all tokens are sent, then set
  // the remote flag; once we have atomic operation, we can send an atomic rdma op after each
  // token and the remote peer polling the flag to know if the token is finished sent
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    uint32_t numTokenSignal = recvTokenNumBuf[destPe];
    uint32_t recvTokenNum = numTokenSignal - 1;

    ShmemUint32WaitUntilEquals(handle.gridCopyTokenBarrier + destPe, recvTokenNum);
    ShmemPutUint32ImmNbiWarp(handle.sendTokenNumMemObj, myPe * sizeof(uint32_t), numTokenSignal,
                             destPe);
  }
  SyncIfDebugEnabled("Combine kernel: finish sending tokens");

  // Phase 2: recv pe sorted token, reduce accross expert and recover original order
  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint32_t* signal = handle.sendTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
    ShmemUint32WaitUntilGreaterThan(signal, 0);
  }
  SyncIfDebugEnabled("Combine kernel: finish waiting num token signal");

  T* outTokenBuf = handle.outTokenBuf;
  for (int i = 0; i < handle.curRankNumToken; i++) {
    if ((i % globalWarpNum) != globalWarpId) continue;

    uint32_t tokenOffset = i * config.hiddenDim;
    for (int j = 0; j < config.numExpertPerToken; j++) {
      uint32_t peSortedId = handle.tokenIndicesToPeSortedBuf[i * config.numExpertPerToken + j];
      uint32_t peSortedOffset = peSortedId * config.hiddenDim;
      WarpAccum(handle.outTokenBuf + tokenOffset, shmemOutTokenBuf + peSortedOffset,
                config.hiddenDim);
    }
  }
}

template <typename T>
class EpDispatchCombineTestCase {
 public:
  EpDispatchCombineTestCase(EpDispatchCombineHandle<T>& handle) : handle(handle) {
    gen = mt19937(rd());
    gen.seed(ShmemMyPe());

    EpDispatchCombineConfig& config = handle.config;

    // Set kernel input/output token buffer
    int maxTokenSize = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken *
                       config.hiddenDim * sizeof(T);
    HIP_RUNTIME_CHECK(hipMalloc(&inpTokBuf, maxTokenSize));
    HIP_RUNTIME_CHECK(hipMemset(inpTokBuf, 0, maxTokenSize));
    HIP_RUNTIME_CHECK(hipMalloc(&outTokBuf, maxTokenSize));
    HIP_RUNTIME_CHECK(hipMemset(outTokBuf, 0, maxTokenSize));
  }

  ~EpDispatchCombineTestCase() {
    HIP_RUNTIME_CHECK(hipFree(inpTokBuf));
    HIP_RUNTIME_CHECK(hipFree(outTokBuf));
    free(inpTokBufCpu);
  }

  void RandomInitializeHandle() {
    handle.SetInputOutput(inpTokBuf, outTokBuf);
    handle.Reset();
    RandomIntializeNumToken();
    RandomIntializeDispatch();
    RandomInitializeToken();
    PrintDispatch();
  }

  void CompareToken(T* expected, T* got, uint32_t hiddenDim, std::string msg) {
    for (int k = 0; k < hiddenDim; k++) {
      T expectedVal = expected[k];
      T gotVal = got[k];
      bool equal = (expectedVal == gotVal);
      if (!equal) {
        std::cout << "Wrong result at pos " << k << ": " << msg << " expected " << expectedVal
                  << " got " << gotVal << std::endl;
        assert(false);
      }
    }
  };

  void RunDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    dim3 grid(config.blockNum);
    dim3 block(warpSize * config.warpNumPerBlock);
    size_t sharedMemSize = 2 * config.worldSize * config.warpNumPerBlock * sizeof(uint32_t);
    EpDispatchKernel<<<grid, block, sharedMemSize>>>(config, handle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    handle.ResetBarrier();
  }

  void CheckDispatchResult() {
    EpDispatchCombineConfig& config = handle.config;

    // Copy token indices to CPU
    int maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;
    int tokenIndiciesSize = maxNumOutTokenPerRank * sizeof(uint32_t);

    // Collect token indices from all ranks
    void* tokenIndicesCpu = malloc(tokenIndiciesSize);
    HIP_RUNTIME_CHECK(
        hipMemcpy(tokenIndicesCpu, handle.tokenIndicies, tokenIndiciesSize, hipMemcpyDeviceToHost));

    void* globalTokIndiciesCpu = malloc(config.worldSize * tokenIndiciesSize);
    MPI_Allgather(tokenIndicesCpu, tokenIndiciesSize, MPI_CHAR, globalTokIndiciesCpu,
                  tokenIndiciesSize, MPI_CHAR, MPI_COMM_WORLD);

    // Collect token num from all ranks
    uint32_t globalTokenNum[config.worldSize];
    MPI_Allgather(&handle.curRankNumToken, 1, MPI_UINT32_T, globalTokenNum, 1, MPI_UINT32_T,
                  MPI_COMM_WORLD);

    // Collect tokens from all ranks
    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    int inpTokSize = inpTokEleNum * sizeof(T);
    inpTokBufCpu = malloc(inpTokSize);
    HIP_RUNTIME_CHECK(
        hipMemcpy(inpTokBufCpu, handle.inpTokenBuf, inpTokSize, hipMemcpyDeviceToHost));

    void* globalInpTokBufCpu = malloc(config.worldSize * inpTokSize);
    MPI_Allgather(inpTokBufCpu, inpTokSize, MPI_CHAR, globalInpTokBufCpu, inpTokSize, MPI_CHAR,
                  MPI_COMM_WORLD);

    // Check token dispatched to current rank
    struct SrcTokInfo {
      int pe;
      int tokenId;
    };
    std::vector<SrcTokInfo> srcTokInfoList;
    // Tokens are supposed to be sorted 1) in the order of expert then 2) in the order of srcPe
    for (int exptId = 0; exptId < config.numExpertPerRank; exptId++) {
      for (int srcPe = 0; srcPe < config.worldSize; srcPe++) {
        uint32_t* tokenIndiciesAddr =
            reinterpret_cast<uint32_t*>(globalTokIndiciesCpu) + srcPe * maxNumOutTokenPerRank;

        for (int dispatchId = 0; dispatchId < config.numExpertPerToken * globalTokenNum[srcPe];
             dispatchId++) {
          int expertId = tokenIndiciesAddr[dispatchId];
          int rankId = expertId / config.numExpertPerRank;
          if (rankId != config.rank) continue;

          int localExptId = expertId % config.numExpertPerRank;
          if (localExptId != exptId) continue;

          int tokenId = dispatchId / config.numExpertPerToken;
          srcTokInfoList.push_back({srcPe, tokenId});
        }
      }
    }

    for (int localTokId = 0; localTokId < srcTokInfoList.size(); localTokId++) {
      T* localTokBuf = handle.outTokenBuf + localTokId * config.hiddenDim;

      int srcPe = srcTokInfoList[localTokId].pe;
      int srcTokId = srcTokInfoList[localTokId].tokenId;
      int srcTokenOffset = srcPe * inpTokEleNum + srcTokId * config.hiddenDim;
      T* srcTokBuf = reinterpret_cast<T*>(globalInpTokBufCpu) + srcTokenOffset;

      std::stringstream msg;
      msg << "mype " << config.rank << " localTokId " << localTokId << " srcpe " << srcPe
          << " srcTokId " << srcTokId;
      CompareToken(srcTokBuf, localTokBuf, config.hiddenDim, msg.str());
    }

    free(globalInpTokBufCpu);
    free(globalTokIndiciesCpu);
    free(tokenIndicesCpu);
  }

  void RunCombine() {
    EpDispatchCombineConfig& config = handle.config;

    int maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;

    // Use the output of dispatch as the input of combine
    HIP_RUNTIME_CHECK(hipMemcpy(handle.inpTokenBuf, handle.outTokenBuf,
                                maxNumOutTokenPerRank * config.hiddenDim * sizeof(T),
                                hipMemcpyDeviceToDevice));
    HIP_RUNTIME_CHECK(
        hipMemset(handle.outTokenBuf, 0, maxNumOutTokenPerRank * config.hiddenDim * sizeof(T)));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    dim3 grid(config.blockNum);
    dim3 block(warpSize * config.warpNumPerBlock);
    size_t sharedMemSize = 2 * config.worldSize * config.warpNumPerBlock * sizeof(uint32_t);
    EpCombineKernel<<<grid, block, sharedMemSize>>>(config, handle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  }

  void CheckCombineResult() {
    EpDispatchCombineConfig& config = handle.config;

    for (int i = 0; i < handle.curRankNumToken; i++) {
      uint32_t tokenOffset = i * config.hiddenDim;
      for (int j = 0; j < config.hiddenDim; j++) {
        T expected = reinterpret_cast<T*>(inpTokBufCpu)[tokenOffset + j] * config.numExpertPerToken;
        T got = handle.outTokenBuf[tokenOffset + j];
        if (got != expected) {
          printf("Wrong result at pos %d: mype %d tokenId %d expected %d got %d\n", j, config.rank,
                 i, expected, got);
          assert(false);
        }
      }
    }
  }

 private:
  void RandomIntializeNumToken() {
    EpDispatchCombineConfig& config = handle.config;
    uniform_int_distribution<> dist(0, config.maxNumInpTokenPerRank);
    handle.curRankNumToken = dist(gen);
  }

  void RandomIntializeDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    std::vector<int> epRange;
    for (int i = 0; i < config.worldSize * config.numExpertPerRank; i++) epRange.push_back(i);

    for (int i = 0; i < handle.curRankNumToken; i++) {
      std::shuffle(epRange.begin(), epRange.end(), gen);
      for (int j = 0; j < config.numExpertPerToken; j++) {
        handle.tokenIndicies[i * config.numExpertPerRank + j] = epRange[j];
      }
    }
  }

  void PrintDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    stringstream ss;
    for (int i = 0; i < handle.curRankNumToken; i++) {
      ss << "  Token " << i << " dispatch to ";
      for (int j = 0; j < config.numExpertPerToken; j++) {
        ss << handle.tokenIndicies[i * config.numExpertPerRank + j] << " ";
      }
      ss << std::endl;
    }
    std::cout << "Rank " << config.rank << ":" << std::endl;
    std::cout << ss.str() << std::endl;
  }

  void RandomInitializeToken() {
    EpDispatchCombineConfig& config = handle.config;
    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    uniform_int_distribution<> tokValDist(1, config.hiddenDim);
    for (int i = 0; i < inpTokEleNum; i++) {
      reinterpret_cast<T*>(handle.inpTokenBuf)[i] = tokValDist(gen);
    }
  }

 private:
  random_device rd;
  mt19937 gen;

  void* inpTokBuf{nullptr};
  void* inpTokBufCpu{nullptr};
  void* outTokBuf{nullptr};
  EpDispatchCombineHandle<T>& handle;
};

void CheckCoopLaunchSupport() {
  int device = 0;
  int supportCoopLaunch = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&device));
  HIP_RUNTIME_CHECK(
      hipDeviceGetAttribute(&supportCoopLaunch, hipDeviceAttributeCooperativeLaunch, device));
  assert(supportCoopLaunch);
}

// A simple MoE-EP dispatch kernel example, assume dp rank is equal to ep rank
void EpDispatchWithPutMemAPI() {
  int status;

  // Initialize shmem
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  CheckCoopLaunchSupport();

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Setup config
  EpDispatchCombineConfig config;
  config.randomSeed = myPe;
  config.rank = myPe;
  config.worldSize = npes;
  config.numExpertPerRank = 2;
  config.hiddenDim = 4096;
  config.numExpertPerToken = 2;
  config.maxNumInpTokenPerRank = 32;
  config.warpNumPerBlock = 2;
  config.blockNum = 4;

  // Intialize EpDispatchCombineHandle
  using DataType = uint32_t;
  EpDispatchCombineHandle<DataType> handle(config);
  handle.Intialize();

  // Run tests
  EpDispatchCombineTestCase<DataType> testCase(handle);
  for (int i = 0; i < 5; i++) {
    testCase.RandomInitializeHandle();
    MPI_Barrier(MPI_COMM_WORLD);
    testCase.RunDispatch();
    MPI_Barrier(MPI_COMM_WORLD);
    testCase.CheckDispatchResult();
    MPI_Barrier(MPI_COMM_WORLD);
    // testCase.RunCombine();
    // MPI_Barrier(MPI_COMM_WORLD);
    // testCase.CheckCombineResult();
  }

  ShmemMpiFinalize();
}

int main() {
  EpDispatchWithPutMemAPI();
  return 0;
}
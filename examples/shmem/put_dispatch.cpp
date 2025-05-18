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
    ShmemQuietThread<PrvdType>();
    printf("%s\n", msg);
  }
  __syncthreads();
#endif
}

struct EpDispatchConfig {
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
class EpDispatchHandle {
 public:
  EpDispatchHandle(EpDispatchConfig config) : config(config) {}

  void Intialize() {
    IntializeTokenIndiciesBuf();
    IntializeShmemInpTokBuf();
    IntializeShmemOutTokBuf();
    IntializeRecvTokenNumBuf();
    IntializeTokToExptBuf();
  }

 private:
  void IntializeTokenIndiciesBuf() {
    int maxNumTokenIndices = config.maxNumInpTokenPerRank * config.numExpertPerToken;
    int tokenIndiciesSize = maxNumTokenIndices * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMalloc(&tokenIndicies, tokenIndiciesSize));
    HIP_RUNTIME_CHECK(hipMemset(tokenIndicies, 0, tokenIndiciesSize));
  }

  void IntializeShmemInpTokBuf() {
    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    int inpTokSize = inpTokEleNum * sizeof(T);
    void* inpTokBuf = ShmemMalloc(inpTokSize);
    HIP_RUNTIME_CHECK(hipMemset(inpTokBuf, 0, inpTokSize));
    shmemInpTokMemObj = ShmemQueryMemObjPtr(inpTokBuf);
    assert(shmemInpTokMemObj.IsValid());
  }

  void IntializeShmemOutTokBuf() {
    int outTokSize = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken *
                     config.hiddenDim * sizeof(T);
    void* shmemOutTokBuf = ShmemMalloc(outTokSize);
    HIP_RUNTIME_CHECK(hipMemset(shmemOutTokBuf, 0, outTokSize));
    shmemOutTokMemObj = ShmemQueryMemObjPtr(shmemOutTokBuf);
    assert(shmemOutTokMemObj.IsValid());
  }

  void IntializeRecvTokenNumBuf() {
    int recvTokenNumSize = config.worldSize * sizeof(uint32_t);
    void* recvTokenNumBuf = ShmemMalloc(recvTokenNumSize);
    HIP_RUNTIME_CHECK(hipMemset(recvTokenNumBuf, 0, recvTokenNumSize));
    recvTokenNumMemObj = ShmemQueryMemObjPtr(recvTokenNumBuf);
    assert(recvTokenNumMemObj.IsValid());
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

 public:
  // Number of tokens on this rank, updated at each round of inference
  int curRankNumToken{-1};

 public:
  // Config
  EpDispatchConfig config;
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
  // Buffers for token to expert mapping, only used for shmem ops
  SymmMemObjPtr inpTokToExptMapMemObj;
  SymmMemObjPtr outTokToExptMapMemObj;
};

template <typename T>
__device__ T WarpPrefixSum(T val, size_t laneNum) {
  int laneId = threadIdx.x & (warpSize - 1);
  uint32_t prefixSum = 0;
  if (laneId < laneNum) {
    for (int i = 0; i <= laneId; i++) {
      uint32_t targetLaneVal = __shfl(val, i);
      if (laneId > i) prefixSum += targetLaneVal;
    }
  }
  return prefixSum;
}

template <typename T>
__device__ void WarpCopy(T* dst, T* src, size_t nelems) {
  constexpr int vecSize = 16 / sizeof(T);
  int laneId = threadIdx.x & (warpSize - 1);
  int offset = laneId * vecSize;

  while ((offset + vecSize) < nelems) {
    reinterpret_cast<uint4*>(dst + offset)[0] = reinterpret_cast<uint4*>(src + offset)[0];
    offset += warpSize * vecSize;
  }

  while (offset < nelems) {
    dst[offset] = src[offset];
    offset += 1;
  }
}

template <typename T>
__global__ void EpDispatchWithPutMemAPIKernel(EpDispatchConfig config, EpDispatchHandle<T> handle) {
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

    int tokenId = i / config.numExpertPerToken;
    int tokenOffset = tokenId * config.hiddenDim;

    int destEpOffset = myPe * maxNumOutTokenPerRank * config.hiddenDim;
    int destEpTokOffset = peTokenIdx * config.hiddenDim;

    // TODO: call copy since we don't build queue pair to self for now, we should consider
    // calling shmem api directly and let it decide how to transfer data based on transport type
    if (destPe == myPe) {
      WarpCopy(handle.shmemOutTokMemObj->template GetAs<T*>() + destEpOffset + destEpTokOffset,
               handle.inpTokenBuf + tokenOffset, config.hiddenDim);
      continue;
    }
    // First copy into shmem inp token buffer, then put with shmem
    WarpCopy(handle.shmemInpTokMemObj->template GetAs<T*>() + tokenOffset,
             handle.inpTokenBuf + tokenOffset, config.hiddenDim);
    if (laneId == 0) {
      ShmemPutTypeNbiThread<PrvdType, T>(handle.shmemOutTokMemObj, destEpOffset + destEpTokOffset,
                                         handle.shmemInpTokMemObj, tokenOffset, config.hiddenDim,
                                         destPe);
    }
  }
  SyncIfDebugEnabled("finished send token");
  // Make sure WarCopy is visible to other blocks
  __threadfence();

  // Send token num & token to expert mapping to other ranks
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    // Add 1 so that when token number == 0, receiver side still know the signal is sent
    uint32_t numTokenSignal = peTokenOffset[destPe] + 1;
    if (destPe == myPe) {
      WarpCopy(outTokToExptBuf + myPe * maxNumOutTokenPerRank,
               inpTokToExptBuf + destPe * maxNumOutTokenPerRank, peTokenOffset[destPe]);
      __threadfence();  // numTokenSignal should be visible after token to expert mapping is visible
      AtomicStoreRelaxed(handle.recvTokenNumMemObj->template GetAs<uint32_t*>() + myPe,
                         numTokenSignal);
      continue;
    }
    if (laneId == 0) {
      // According to RDMA speces, ops in the same queue pair is properly ordered, hence when
      // numTokenSignal is visible to a pe, token to expert mapping is also visible
      ShmemPutUint32NbiThread<PrvdType>(
          handle.outTokToExptMapMemObj, myPe * maxNumOutTokenPerRank, handle.inpTokToExptMapMemObj,
          destPe * maxNumOutTokenPerRank, peTokenOffset[destPe], destPe);
      ShmemPutUint32ImmNbiThread<PrvdType>(handle.recvTokenNumMemObj, myPe * sizeof(uint32_t),
                                           numTokenSignal, destPe);
    }
  }
  SyncIfDebugEnabled("finish sending tok2expt mapping & num token signal");

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

  uint32_t totalNumRecvToken = 0;
  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint32_t* signal = handle.recvTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
    ShmemUint32WaitUntilGreaterThan<PrvdType>(signal, 0);
    recvTokenNum[destPe] = *signal - 1;
    totalNumRecvToken += *signal - 1;
  }
  SyncIfDebugEnabled("finish waiting num token signal");

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

  for (int i = 0, tokId = 0, srcPe = 0; srcPe < npes; tokId++, i++) {
    if (tokId == recvTokenNum[srcPe]) {
      srcPe++;
      tokId = -1;
      continue;
    }

    uint32_t localExpertId =
        outTokToExptBuf[srcPe * maxNumOutTokenPerRank + tokId] % config.numExpertPerRank;
    if (laneId == 0) {
      expertTokOff[localExpertId] += 1;
    }

    if ((i % globalWarpNum) != globalWarpId) continue;  // skip token not assigned for this warp

    // Copy token
    uint32_t srcTokenOff =
        srcPe * maxNumOutTokenPerRank * config.hiddenDim + tokId * config.hiddenDim;
    uint32_t destTokenOff =
        (accumExpertTokOffsets[localExpertId] + expertTokOff[localExpertId] - 1) * config.hiddenDim;

    WarpCopy(outTokenBuf + destTokenOff, shmemOutTokenBuf + srcTokenOff, config.hiddenDim);
  }
}

template <typename T>
class EpDispatchCombineTestCase {
 public:
  EpDispatchCombineTestCase() {
    gen = mt19937(rd());
    gen.seed(0);
  }
  ~EpDispatchCombineTestCase() = default;

  void RandomInitializeHandle(EpDispatchHandle<T>& handle) {
    EpDispatchConfig& config = handle.config;

    // Set kernel input/output token buffer
    int inpTokSize = config.maxNumInpTokenPerRank * config.hiddenDim * sizeof(T);
    HIP_RUNTIME_CHECK(hipMalloc(&handle.inpTokenBuf, inpTokSize));
    HIP_RUNTIME_CHECK(hipMemset(handle.inpTokenBuf, 0, inpTokSize));

    int outTokSize = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken *
                     config.hiddenDim * sizeof(T);
    HIP_RUNTIME_CHECK(hipMalloc(&handle.outTokenBuf, outTokSize));
    HIP_RUNTIME_CHECK(hipMemset(handle.outTokenBuf, 0, outTokSize));

    RandomIntializeNumToken(handle);
    RandomIntializeDispatch(handle);
    RandomInitializeToken(handle);
  }

  void Run(EpDispatchHandle<T>& handle) {
    EpDispatchConfig& config = handle.config;
    dim3 grid(config.blockNum);
    dim3 block(warpSize * config.warpNumPerBlock);
    size_t sharedMemSize = 2 * config.worldSize * config.warpNumPerBlock * sizeof(uint32_t);
    EpDispatchWithPutMemAPIKernel<<<grid, block, sharedMemSize>>>(config, handle);
  }

  void CehckResult(EpDispatchHandle<T>& handle) {
    EpDispatchConfig& config = handle.config;

    // Copy token indices to CPU
    int maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;
    int tokenIndiciesSize = maxNumOutTokenPerRank * sizeof(uint32_t);

    uint32_t* tokenIndicesCpu = reinterpret_cast<uint32_t*>(malloc(tokenIndiciesSize));
    HIP_RUNTIME_CHECK(
        hipMemcpy(tokenIndicesCpu, handle.tokenIndicies, tokenIndiciesSize, hipMemcpyDeviceToHost));
    // Collect token indices from all ranks
    uint32_t* globalTokIndiciesCpu =
        reinterpret_cast<uint32_t*>(malloc(config.worldSize * tokenIndiciesSize));
    MPI_Allgather(tokenIndicesCpu, tokenIndiciesSize, MPI_CHAR, globalTokIndiciesCpu,
                  tokenIndiciesSize, MPI_CHAR, MPI_COMM_WORLD);
    memcpy(globalTokIndiciesCpu + config.rank * maxNumOutTokenPerRank, tokenIndicesCpu,
           tokenIndiciesSize);

    uint32_t globalTokenNum[config.worldSize];
    MPI_Allgather(&handle.curRankNumToken, 1, MPI_UINT32_T, globalTokenNum, 1, MPI_UINT32_T,
                  MPI_COMM_WORLD);
    globalTokenNum[config.rank] = handle.curRankNumToken;

    // Collect tokens from all ranks
    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    int inpTokSize = inpTokEleNum * sizeof(T);
    void* inpTokBufCpu = malloc(inpTokSize);
    HIP_RUNTIME_CHECK(
        hipMemcpy(inpTokBufCpu, handle.inpTokenBuf, inpTokSize, hipMemcpyDeviceToHost));

    void* globalInpTokBufCpu = malloc(config.worldSize * inpTokSize);
    MPI_Allgather(inpTokBufCpu, inpTokSize, MPI_CHAR, globalInpTokBufCpu, inpTokSize, MPI_CHAR,
                  MPI_COMM_WORLD);
    memcpy(reinterpret_cast<char*>(globalInpTokBufCpu) + config.rank * inpTokSize, inpTokBufCpu,
           inpTokSize);

    std::vector<uint32_t> expertCount(config.numExpertPerRank, 0);

    auto CompareTokenFunc = [](T* expected, T* got, uint32_t hiddenDim, std::string msg) {
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

    // Check result
    for (int srcPe = 0; srcPe < config.worldSize; srcPe++) {
      uint32_t* tokenIndiciesAddr = globalTokIndiciesCpu + srcPe * maxNumOutTokenPerRank;
      uint32_t peTokenOffset = 0;

      for (int tokIdx = 0; tokIdx < config.numExpertPerToken * globalTokenNum[srcPe]; tokIdx++) {
        int expertId = tokenIndiciesAddr[tokIdx];
        int rankId = expertId / config.numExpertPerRank;
        if (rankId != config.rank) continue;

        expertCount[expertId % config.numExpertPerRank] += 1;

        int tokenId = tokIdx / config.numExpertPerToken;
        int srcTokenOffset = srcPe * inpTokEleNum + tokenId * config.hiddenDim;

        int outTokEleNum = maxNumOutTokenPerRank * config.hiddenDim;
        int destTokenOffset = srcPe * outTokEleNum + peTokenOffset * config.hiddenDim;

        // Check shmem out token buffer
        std::stringstream ss;
        ss << "source pe " << srcPe << " dest pe " << config.rank << " expertId " << expertId
           << " srcTokenId " << tokenId << " destTokenId " << peTokenOffset;
        CompareTokenFunc(reinterpret_cast<T*>(globalInpTokBufCpu) + srcTokenOffset,
                         handle.shmemOutTokMemObj->template GetAs<T*>() + destTokenOffset,
                         config.hiddenDim, ss.str());

        // Check token to expert mapping buffer
        uint32_t* outTokToExptMapBuf = handle.outTokToExptMapMemObj->template GetAs<uint32_t*>();
        uint32_t gotExptId = outTokToExptMapBuf[srcPe * maxNumOutTokenPerRank + peTokenOffset];
        if ((gotExptId != expertId)) {
          printf("Wrong: srcpe %d mype %d token offset %d expected %d got %d\n", srcPe, config.rank,
                 peTokenOffset, expertId, gotExptId);
          assert(false);
        }
        peTokenOffset += 1;
      }

      // Check recv token num signal
      uint32_t recvTokenNumSignal = handle.recvTokenNumMemObj->template GetAs<uint32_t*>()[srcPe];
      assert((recvTokenNumSignal - 1) == peTokenOffset);
    }

    // Check out token buffer
    struct SrcTokInfo {
      int pe;
      int tokenId;
    };
    std::vector<SrcTokInfo> srcTokInfoList;
    for (int exptId = 0; exptId < config.numExpertPerRank; exptId++) {
      for (int srcPe = 0; srcPe < config.worldSize; srcPe++) {
        uint32_t* tokenIndiciesAddr = globalTokIndiciesCpu + srcPe * maxNumOutTokenPerRank;

        for (int tokIdx = 0; tokIdx < config.numExpertPerToken * globalTokenNum[srcPe]; tokIdx++) {
          int expertId = tokenIndiciesAddr[tokIdx];
          int rankId = expertId / config.numExpertPerRank;
          if (rankId != config.rank) continue;

          int localExptId = expertId % config.numExpertPerRank;
          if (localExptId != exptId) continue;

          int tokenId = tokIdx / config.numExpertPerToken;

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

      CompareTokenFunc(srcTokBuf, localTokBuf, config.hiddenDim, msg.str());
    }

    // for (int i = 0; i < expertCount.size(); i++) {
    //   std::cout << "Rank " << config.rank << " expert " << i << " token " << expertCount[i]
    //             << std::endl;
    // }
  }

 private:
  void RandomIntializeNumToken(EpDispatchHandle<T>& handle) {
    EpDispatchConfig& config = handle.config;
    uniform_int_distribution<> dist(0, config.maxNumInpTokenPerRank);
    handle.curRankNumToken = dist(gen);
  }

  void RandomIntializeDispatch(EpDispatchHandle<T>& handle) {
    EpDispatchConfig& config = handle.config;

    int totalNumExperts = config.worldSize * config.numExpertPerRank;
    uniform_int_distribution<> distTokIndex(0, totalNumExperts - 1);

    std::vector<int> epRange;
    for (int i = 0; i < config.worldSize * config.numExpertPerRank; i++) epRange.push_back(i);

    stringstream ss;
    for (int i = 0; i < handle.curRankNumToken; i++) {
      std::shuffle(epRange.begin(), epRange.end(), gen);
      ss << "  Token " << i << " dispatch to ";
      for (int j = 0; j < config.numExpertPerToken; j++) {
        handle.tokenIndicies[i * config.numExpertPerRank + j] = epRange[j];
        ss << handle.tokenIndicies[i * config.numExpertPerRank + j] << " ";
      }
      ss << std::endl;
    }
    std::cout << "Rank " << config.rank << ":" << std::endl;
    std::cout << ss.str() << std::endl;
  }

  void RandomInitializeToken(EpDispatchHandle<T>& handle) {
    EpDispatchConfig& config = handle.config;

    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    uniform_int_distribution<> tokValDist(0, config.hiddenDim);
    for (int i = 0; i < inpTokEleNum; i++) {
      reinterpret_cast<T*>(handle.inpTokenBuf)[i] = tokValDist(gen);
    }
  }

 private:
  random_device rd;
  mt19937 gen;
};

// A simple MoE-EP dispatch kernel example, assume dp rank is equal to ep rank
void EpDispatchWithPutMemAPI() {
  int status;

  // Initialize shmem
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Setup config
  EpDispatchConfig config;
  config.randomSeed = myPe;
  config.rank = myPe;
  config.worldSize = npes;
  config.numExpertPerRank = 8;
  config.hiddenDim = 4096;
  config.numExpertPerToken = 8;
  config.maxNumInpTokenPerRank = 128;
  config.warpNumPerBlock = 4;
  config.blockNum = 8;

  // Intialize EpDispatchHandle
  using DataType = uint32_t;
  EpDispatchHandle<DataType> handle(config);
  handle.Intialize();

  EpDispatchCombineTestCase<DataType> testCase;
  testCase.RandomInitializeHandle(handle);
  testCase.Run(handle);
  testCase.CehckResult(handle);

  ShmemMpiFinalize();
  MPI_Finalize();
}

int main() {
  EpDispatchWithPutMemAPI();
  return 0;
}
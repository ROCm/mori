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

struct EpDispatchConfig {
  int randomSeed{0};
  int rank{0};
  int worldSize{0};
  int numExpertPerRank{1};
  int hiddenDim{4096};
  int numExpertPerToken{2};
  int maxNumInpTokenPerRank{128};
  int warpNum{1};
};

struct EpDispatchHandle {
  int curRankNumToken;
  uint32_t* tokenIndicies;
  SymmMemObjPtr inpTokMemObj;
  SymmMemObjPtr outTokMemObj;
  SymmMemObjPtr recvTokenNumMemObj;
};

EpDispatchHandle IntializeTestHandle(EpDispatchConfig config) {
  // Intialize random generator
  random_device rd;
  mt19937 gen(rd());
  gen.seed(config.rank);

  // EpDispatchHandle
  EpDispatchHandle handle;

  // Generate random token number on this rank
  uniform_int_distribution<> dist(0, config.maxNumInpTokenPerRank);
  handle.curRankNumToken = dist(gen);

  // Generate random dispatch indices
  int totalNumExperts = config.worldSize * config.numExpertPerRank;
  uniform_int_distribution<> distTokIndex(0, totalNumExperts - 1);
  int maxNumTokenIndices = config.maxNumInpTokenPerRank * config.numExpertPerToken;
  int tokenIndiciesSize = maxNumTokenIndices * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&handle.tokenIndicies, tokenIndiciesSize));
  HIP_RUNTIME_CHECK(hipMemset(handle.tokenIndicies, 0, tokenIndiciesSize));

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

  // Allocate token input buffer and random initialize
  int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
  int inpTokSize = inpTokEleNum * sizeof(uint32_t);
  void* inpTokBuf = ShmemMalloc(inpTokSize);
  HIP_RUNTIME_CHECK(hipMemset(inpTokBuf, 0, inpTokSize));
  handle.inpTokMemObj = ShmemQueryMemObjPtr(inpTokBuf);
  assert(handle.inpTokMemObj.IsValid());
  uniform_int_distribution<> tokValDist(0, config.hiddenDim);
  for (int i = 0; i < inpTokEleNum; i++) {
    reinterpret_cast<uint32_t*>(inpTokBuf)[i] = tokValDist(gen);
  }

  // Allocate token output buffer
  int outTokSize =
      config.worldSize * config.maxNumInpTokenPerRank * config.hiddenDim * sizeof(uint32_t);
  void* outTokBuf = ShmemMalloc(outTokSize);
  HIP_RUNTIME_CHECK(hipMemset(outTokBuf, 0, outTokSize));
  handle.outTokMemObj = ShmemQueryMemObjPtr(outTokBuf);
  assert(handle.outTokMemObj.IsValid());

  // Allocate recv token num buffer
  int recvTokenNumSize = config.worldSize * sizeof(uint32_t);
  void* recvTokenNumBuf = ShmemMalloc(recvTokenNumSize);
  HIP_RUNTIME_CHECK(hipMemset(recvTokenNumBuf, 0, recvTokenNumSize));
  handle.recvTokenNumMemObj = ShmemQueryMemObjPtr(recvTokenNumBuf);
  assert(handle.recvTokenNumMemObj.IsValid());

  return handle;
}

void CheckTestResult(EpDispatchConfig config, EpDispatchHandle handle) {
  // Copy token indices to CPU
  int maxNumTokenIndices = config.maxNumInpTokenPerRank * config.numExpertPerToken;
  int tokenIndiciesSize = maxNumTokenIndices * sizeof(uint32_t);

  uint32_t* tokenIndicesCpu = reinterpret_cast<uint32_t*>(malloc(tokenIndiciesSize));
  HIP_RUNTIME_CHECK(
      hipMemcpy(tokenIndicesCpu, handle.tokenIndicies, tokenIndiciesSize, hipMemcpyDeviceToHost));
  // Collect token indices from all ranks
  uint32_t* globalInpTokBuf =
      reinterpret_cast<uint32_t*>(malloc(config.worldSize * tokenIndiciesSize));
  MPI_Allgather(tokenIndicesCpu, tokenIndiciesSize, MPI_CHAR, globalInpTokBuf, tokenIndiciesSize,
                MPI_CHAR, MPI_COMM_WORLD);

  uint32_t globalTokenNum[config.worldSize];
  MPI_Allgather(&handle.curRankNumToken, 1, MPI_UINT32_T, globalTokenNum, 1, MPI_UINT32_T,
                MPI_COMM_WORLD);

  // Collect tokens from all ranks
  int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
  int inpTokSize = inpTokEleNum * sizeof(uint32_t);
  void* inpTokBufCpu = malloc(inpTokSize);
  HIP_RUNTIME_CHECK(hipMemcpy(inpTokBufCpu, handle.inpTokMemObj.cpu->localPtr, inpTokSize,
                              hipMemcpyDeviceToHost));

  void* globalInpTokBufCpu = malloc(config.worldSize * inpTokSize);
  MPI_Allgather(inpTokBufCpu, inpTokSize, MPI_CHAR, globalInpTokBufCpu, inpTokSize, MPI_CHAR,
                MPI_COMM_WORLD);

  // Check result
  for (int i = 0; i < config.worldSize; i++) {
    if (i == config.rank) continue;
    // printf("on rank %d got rank %d token num %d\n", config.rank, i, globalTokenNum[i]);
    uint32_t* tokenIndiciesAddr = globalInpTokBuf + i * maxNumTokenIndices;
    uint32_t peTokenOffset = 0;
    for (int j = 0; j < config.numExpertPerToken * globalTokenNum[i]; j++) {
      int expertId = tokenIndiciesAddr[j];
      int rankId = expertId / config.numExpertPerRank;
      if (rankId != config.rank) continue;

      int tokenId = j / config.numExpertPerToken;
      int srcTokenOffset = i * inpTokEleNum + tokenId * config.hiddenDim;
      int destTokenOffset = i * inpTokEleNum + peTokenOffset * config.hiddenDim;
      peTokenOffset += 1;
      // printf("source pe %d mype %d expert %d tokenId %d offset %d\n", i, config.rank, expertId,
      //        tokenId, peTokenOffset - 1);
      for (int k = 0; k < config.hiddenDim; k++) {
        uint32_t expected = reinterpret_cast<uint32_t*>(globalInpTokBufCpu)[srcTokenOffset + k];
        uint32_t got =
            reinterpret_cast<uint32_t*>(handle.outTokMemObj.cpu->localPtr)[destTokenOffset + k];
        bool equal = (expected == got);
        if (!equal) {
          printf(
              "Wrong: source pe %d dest pe %d expertId %d srcTokenId %d destTokenId %d pos %d "
              "expected %u got %u\n",
              i, config.rank, expertId, tokenId, peTokenOffset - 1, k, expected, got);
          assert(false);
        }
      }
    }
    uint32_t recvTokenNum = reinterpret_cast<uint32_t*>(handle.recvTokenNumMemObj.cpu->localPtr)[i];
    printf("source pe %d my pe %d expected %d got %d\n", i, config.rank, peTokenOffset,
           recvTokenNum);
    assert(recvTokenNum == peTokenOffset);
  }
}

__global__ void EpDispatchWithPutMemAPIKernel(EpDispatchConfig config, EpDispatchHandle handle) {
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

  uint32_t* inpTokenBuf = reinterpret_cast<uint32_t*>(handle.inpTokMemObj.gpu->localPtr);
  uint32_t* outTokenBuf = reinterpret_cast<uint32_t*>(handle.outTokMemObj.gpu->localPtr);

  // Send out tokens
  // TODO: 2 token id -> expert id
  // TODO: 3 rank token num
  extern __shared__ char sharedMem[];

  // Phase1: send token
  // Per warp offset array
  uint32_t* peTokenOffset = reinterpret_cast<uint32_t*>(sharedMem) + warpId * npes;
  for (int i = laneId; i < npes; i += warpSize) {
    peTokenOffset[i] = 0;
  }

  for (int i = 0; i < handle.curRankNumToken * config.numExpertPerToken; i++) {
    // TODO: eliminate redundant token transimission when a token is routed to multiple experts
    // located on the same pe
    uint32_t destExpert = handle.tokenIndicies[i];
    uint32_t destPe = destExpert / config.numExpertPerRank;
    uint32_t peTokenIdx = peTokenOffset[destPe];
    if (laneId == 0) {
      peTokenOffset[destPe] += 1;
    }
    if (destPe == myPe) continue;                       // skip sending token to self
    if ((i % globalWarpNum) != globalWarpId) continue;  // skip token not assigned for this warp

    int tokenId = i / config.numExpertPerToken;

    int tokenOffset = tokenId * config.hiddenDim * sizeof(uint32_t);
    int destEpOffset = myPe * config.maxNumInpTokenPerRank * config.hiddenDim * sizeof(uint32_t);
    int destEpTokOffset = peTokenIdx * config.hiddenDim * sizeof(uint32_t);

    if (laneId == 0) {
      MemoryRegion srcMr = handle.inpTokMemObj.gpu->GetMemoryRegion(myPe);
      ShmemPutMemNbiThread<PrvdType>(handle.outTokMemObj.gpu, destEpOffset + destEpTokOffset, srcMr,
                                     tokenOffset, config.hiddenDim * sizeof(uint32_t), destPe);
      // printf("mype %d warp %d exprt %d dest pe %d tokenId %d offset %u\n", myPe, globalWarpId,
      //        destExpert, destPe, tokenId, peTokenIdx);
    }
  }

  // Send token num to other ranks
  if ((globalThdId < npes) && (globalThdId != myPe)) {
    uint32_t destPe = globalThdId;
    atomicStoreSeqCstSystem(
        reinterpret_cast<uint32_t*>(handle.recvTokenNumMemObj.gpu->localPtr) + destPe,
        peTokenOffset[destPe]);
    MemoryRegion srcMr = handle.recvTokenNumMemObj.gpu->GetMemoryRegion(myPe);
    ShmemPutMemNbiThread<PrvdType>(handle.recvTokenNumMemObj.gpu, myPe * sizeof(uint32_t), srcMr,
                                   destPe * sizeof(uint32_t), sizeof(uint32_t), destPe);
    printf("mype %d destpe %d token num %d\n", myPe, destPe, peTokenOffset[destPe]);
  }

  // Phase 2: recv token
}

void LaunchEpDispatchWithPutMemAPIKernel(EpDispatchConfig config, EpDispatchHandle handle) {
  dim3 grid(config.warpNum);
  dim3 block(warpSize);
  size_t sharedMemSize = config.worldSize * 1 * sizeof(uint32_t);
  EpDispatchWithPutMemAPIKernel<<<grid, block, sharedMemSize>>>(config, handle);
}

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
  config.numExpertPerRank = 4;
  config.hiddenDim = 4096;
  config.numExpertPerToken = 4;
  config.maxNumInpTokenPerRank = 32;
  config.warpNum = 8;

  // Intialize data
  EpDispatchHandle handle = IntializeTestHandle(config);

  // Invoke kernel
  LaunchEpDispatchWithPutMemAPIKernel(config, handle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Check results
  CheckTestResult(config, handle);
  ShmemMpiFinalize();
  MPI_Finalize();
}

int main() {
  EpDispatchWithPutMemAPI();
  return 0;
}
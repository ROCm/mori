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
  int hiddenDim{4096};
  int numExpertPerToken{2};
  int maxNumTokenPerRank{128};
  int warpNum{1};
};

struct EpDispatchHandle {
  int curRankNumToken;
  uint32_t* tokenIndicies;
  SymmMemObjPtr inpTokMemObj;
  SymmMemObjPtr outTokMemObj;
};

EpDispatchHandle IntializeTestHandle(EpDispatchConfig config) {
  // Intialize random generator
  random_device rd;
  mt19937 gen(rd());
  gen.seed(config.rank);

  // EpDispatchHandle
  EpDispatchHandle handle;

  // Generate random token number on this rank
  uniform_int_distribution<> dist(0, config.maxNumTokenPerRank);
  handle.curRankNumToken = dist(gen);

  // Generate random dispatch indices
  uniform_int_distribution<> distTokIndex(0, config.worldSize - 1);
  int maxNumTokenIndices = config.maxNumTokenPerRank * config.numExpertPerToken;
  int tokenIndiciesSize = maxNumTokenIndices * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&handle.tokenIndicies, tokenIndiciesSize));
  HIP_RUNTIME_CHECK(hipMemset(handle.tokenIndicies, 0, tokenIndiciesSize));

  std::vector<int> epRange;
  for (int i = 0; i < config.worldSize; i++) epRange.push_back(i);

  stringstream ss;
  for (int i = 0; i < handle.curRankNumToken; i++) {
    std::shuffle(epRange.begin(), epRange.end(), gen);
    ss << "  Token " << i << " dispatch to ";
    for (int j = 0; j < config.numExpertPerToken; j++) {
      handle.tokenIndicies[i] = epRange[j];
      ss << epRange[j] << " ";
    }
    ss << std::endl;
  }
  std::cout << "Rank " << config.rank << ":" << std::endl;
  std::cout << ss.str() << std::endl;

  // Allocate token input buffer and random initialize
  int inpTokEleNum = config.maxNumTokenPerRank * config.hiddenDim;
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
      config.worldSize * config.maxNumTokenPerRank * config.hiddenDim * sizeof(uint32_t);
  void* outTokBuf = ShmemMalloc(outTokSize);
  HIP_RUNTIME_CHECK(hipMemset(outTokBuf, 0, outTokSize));
  handle.outTokMemObj = ShmemQueryMemObjPtr(outTokBuf);
  assert(handle.outTokMemObj.IsValid());

  return handle;
}

void CheckTestResult(EpDispatchConfig config, EpDispatchHandle handle) {
  // Copy token indices to CPU
  int maxNumTokenIndices = config.maxNumTokenPerRank * config.numExpertPerToken;
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
  int inpTokEleNum = config.maxNumTokenPerRank * config.hiddenDim;
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
    printf("on rank %d got rank %d token num %d\n", config.rank, i, globalTokenNum[i]);
    uint32_t* tokenIndiciesAddr = globalInpTokBuf + i * maxNumTokenIndices;
    for (int j = 0; j < config.numExpertPerToken * globalTokenNum[i]; j++) {
      if (tokenIndiciesAddr[j] != config.rank) continue;
      int tokenId = j / config.numExpertPerToken;
      int tokenOffset = i * inpTokEleNum + tokenId * config.hiddenDim;
      printf("rank %d ep %d token id %d offset %d ep offset %d\n", config.rank, i, tokenId,
             tokenOffset, i * maxNumTokenIndices);
      for (int k = 0; k < config.hiddenDim; k++) {
        // printf("%d %d %d\n", tokenOffset + k,
        //        reinterpret_cast<uint32_t*>(globalInpTokBufCpu)[tokenOffset + k],
        //        reinterpret_cast<uint32_t*>(handle.outTokMemObj.cpu->localPtr)[tokenOffset + k]);
        assert(reinterpret_cast<uint32_t*>(globalInpTokBufCpu)[tokenOffset + k] ==
               reinterpret_cast<uint32_t*>(handle.outTokMemObj.cpu->localPtr)[tokenOffset + k]);
      }
    }
  }
}

__global__ void EpDispatchWithPutMemAPIKernel(EpDispatchConfig config, EpDispatchHandle handle) {
  int warpId = blockIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int myPe = config.rank;
  int npes = config.worldSize;

  uint32_t* inpTokenBuf = reinterpret_cast<uint32_t*>(handle.inpTokMemObj.gpu->localPtr);
  uint32_t* outTokenBuf = reinterpret_cast<uint32_t*>(handle.outTokMemObj.gpu->localPtr);

  // Dispatch tokens
  for (int i = warpId; i < config.numExpertPerToken * handle.curRankNumToken; i += gridDim.x) {
    // assume destExpert == destPe
    int destExpert = handle.tokenIndicies[i];
    if (destExpert == myPe) continue;

    int tokenId = i / config.numExpertPerToken;

    int tokenOffset = tokenId * config.hiddenDim * sizeof(uint32_t);
    int destEpOffset = myPe * config.maxNumTokenPerRank * config.hiddenDim * sizeof(uint32_t);

    if (laneId == 0) {
      MemoryRegion srcMr = handle.inpTokMemObj.gpu->GetMemoryRegion(myPe);
      ShmemPutMemNbiThread<PrvdType>(handle.outTokMemObj.gpu, destEpOffset + tokenOffset, srcMr,
                                     tokenOffset, config.hiddenDim * sizeof(uint32_t), destExpert);
      ShmemQuietThread<PrvdType>();
    }
    __syncthreads();
  }
}

void LaunchEpDispatchWithPutMemAPIKernel(EpDispatchConfig config, EpDispatchHandle handle) {
  dim3 grid(config.warpNum);
  dim3 block(1);
  EpDispatchWithPutMemAPIKernel<<<grid, block>>>(config, handle);
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
  config.hiddenDim = 4096;
  config.numExpertPerToken = 2;
  config.maxNumTokenPerRank = 4;
  config.warpNum = 1;

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
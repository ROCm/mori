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
  int warpNumPerBlock{1};
  int blockNum{1};
};

struct EpDispatchHandle {
  int curRankNumToken{-1};
  // Routed expert indices for tokens
  uint32_t* tokenIndicies{nullptr};
  // Compact output token buffer, this is the final output of dispatch
  uint32_t* outTokenBuf{nullptr};
  // Store input token for shmem ops
  SymmMemObjPtr shmemInpTokMemObj;
  // Temporary output token buffer, only used for shmem ops
  SymmMemObjPtr shmemOutTokMemObj;
  // Record number of tokens that will be received from other PE
  SymmMemObjPtr recvTokenNumMemObj;
  // Buffers for token to expert mapping, only used for shmem ops
  SymmMemObjPtr inpTokToExptMapMemObj;
  SymmMemObjPtr outTokToExptMapMemObj;
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
  handle.shmemInpTokMemObj = ShmemQueryMemObjPtr(inpTokBuf);
  assert(handle.shmemInpTokMemObj.IsValid());
  uniform_int_distribution<> tokValDist(0, config.hiddenDim);
  for (int i = 0; i < inpTokEleNum; i++) {
    reinterpret_cast<uint32_t*>(inpTokBuf)[i] = tokValDist(gen);
  }

  // Allocate token output buffer
  int outTokSize = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken *
                   config.hiddenDim * sizeof(uint32_t);
  void* shmemOutTokBuf = ShmemMalloc(outTokSize);
  HIP_RUNTIME_CHECK(hipMemset(shmemOutTokBuf, 0, outTokSize));
  handle.shmemOutTokMemObj = ShmemQueryMemObjPtr(shmemOutTokBuf);
  assert(handle.shmemOutTokMemObj.IsValid());

  HIP_RUNTIME_CHECK(hipMalloc(&handle.outTokenBuf, outTokSize));
  HIP_RUNTIME_CHECK(hipMemset(handle.outTokenBuf, 0, outTokSize));

  // Allocate recv token num buffer
  int recvTokenNumSize = config.worldSize * sizeof(uint32_t);
  void* recvTokenNumBuf = ShmemMalloc(recvTokenNumSize);
  HIP_RUNTIME_CHECK(hipMemset(recvTokenNumBuf, 0, recvTokenNumSize));
  handle.recvTokenNumMemObj = ShmemQueryMemObjPtr(recvTokenNumBuf);
  assert(handle.recvTokenNumMemObj.IsValid());

  // Allocate token id to expert id mapping
  int tokToExptMapSize =
      config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken * sizeof(uint32_t);
  void* inpTokToExptMapBuf = ShmemMalloc(tokToExptMapSize);
  HIP_RUNTIME_CHECK(hipMemset(inpTokToExptMapBuf, 0, tokToExptMapSize));
  handle.inpTokToExptMapMemObj = ShmemQueryMemObjPtr(inpTokToExptMapBuf);
  assert(handle.inpTokToExptMapMemObj.IsValid());

  void* outTokToExptMapBuf = ShmemMalloc(tokToExptMapSize);
  HIP_RUNTIME_CHECK(hipMemset(outTokToExptMapBuf, 0, tokToExptMapSize));
  handle.outTokToExptMapMemObj = ShmemQueryMemObjPtr(outTokToExptMapBuf);
  assert(handle.outTokToExptMapMemObj.IsValid());

  return handle;
}

void CheckTestResult(EpDispatchConfig config, EpDispatchHandle handle) {
  // Copy token indices to CPU
  int maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;
  int tokenIndiciesSize = maxNumOutTokenPerRank * sizeof(uint32_t);

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
  HIP_RUNTIME_CHECK(
      hipMemcpy(inpTokBufCpu, handle.shmemInpTokMemObj->Get(), inpTokSize, hipMemcpyDeviceToHost));

  void* globalInpTokBufCpu = malloc(config.worldSize * inpTokSize);
  MPI_Allgather(inpTokBufCpu, inpTokSize, MPI_CHAR, globalInpTokBufCpu, inpTokSize, MPI_CHAR,
                MPI_COMM_WORLD);

  std::vector<uint32_t> expertCount(config.numExpertPerRank, 0);

  // Check result
  for (int i = 0; i < config.worldSize; i++) {
    if (i == config.rank) continue;
    // printf("on rank %d got rank %d token num %d\n", config.rank, i, globalTokenNum[i]);
    uint32_t* tokenIndiciesAddr = globalInpTokBuf + i * maxNumOutTokenPerRank;
    uint32_t peTokenOffset = 0;
    for (int j = 0; j < config.numExpertPerToken * globalTokenNum[i]; j++) {
      int expertId = tokenIndiciesAddr[j];
      int rankId = expertId / config.numExpertPerRank;
      if (rankId != config.rank) continue;

      expertCount[expertId % config.numExpertPerRank] += 1;

      int tokenId = j / config.numExpertPerToken;
      int srcTokenOffset = i * inpTokEleNum + tokenId * config.hiddenDim;

      int outTokEleNum = maxNumOutTokenPerRank * config.hiddenDim;
      int destTokenOffset = i * outTokEleNum + peTokenOffset * config.hiddenDim;
      // printf("source pe %d mype %d expert %d tokenId %d offset %d\n", i, config.rank, expertId,
      //        tokenId, peTokenOffset );
      for (int k = 0; k < config.hiddenDim; k++) {
        uint32_t expected = reinterpret_cast<uint32_t*>(globalInpTokBufCpu)[srcTokenOffset + k];
        uint32_t got = handle.shmemOutTokMemObj->GetAs<uint32_t*>()[destTokenOffset + k];
        bool equal = (expected == got);
        if (!equal) {
          printf(
              "Wrong: source pe %d dest pe %d expertId %d srcTokenId %d destTokenId %d pos %d "
              "expected %u got %u\n",
              i, config.rank, expertId, tokenId, peTokenOffset, k, expected, got);
          assert(false);
        }
      }

      uint32_t* outTokToExptMapBuf = handle.outTokToExptMapMemObj->GetAs<uint32_t*>();
      uint32_t gotExptId = outTokToExptMapBuf[i * maxNumOutTokenPerRank + peTokenOffset];
      if ((gotExptId != expertId)) {
        printf("Wrong: srcpe %d mype %d token offset %d expected %d got %d\n", i, config.rank,
               peTokenOffset, expertId, gotExptId);
        assert(false);
      }
      peTokenOffset += 1;
    }
    uint32_t recvTokenNumSignal = handle.recvTokenNumMemObj->GetAs<uint32_t*>()[i];
    assert((recvTokenNumSignal - 1) == peTokenOffset);
  }

  for (int i = 0; i < expertCount.size(); i++) {
    std::cout << "Rank " << config.rank << " expert " << i << " token " << expertCount[i]
              << std::endl;
  }
}

template <typename T>
__device__ T WarpReduceSumToRight(T sum) {
  for (int i = warpSize / 2; i > 0; i /= 2) {
    sum += __shfl_up(sum, i);
  }
  return sum;
}

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

  uint32_t* inpTokenBuf = handle.shmemInpTokMemObj->GetAs<uint32_t*>();
  uint32_t* inpTokToExptBuf = handle.inpTokToExptMapMemObj->GetAs<uint32_t*>();
  uint32_t* outTokToExptBuf = handle.outTokToExptMapMemObj->GetAs<uint32_t*>();

  size_t maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;

  // Send out tokens
  // TODO: token id -> expert id
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
      peTokenOffset[destPe] += 1;

      // We must update token to expert mapping using the same warp as the one that sends it,
      // otherwise the data might not be visible to the warp that send it
      if (destPe == globalWarpId) {
        inpTokToExptBuf[destPe * maxNumOutTokenPerRank + peTokenIdx] = destExpert;
      }
    }

    if (destPe == myPe) continue;                       // skip sending token to self
    if ((i % globalWarpNum) != globalWarpId) continue;  // skip token not assigned for this warp

    int tokenId = i / config.numExpertPerToken;
    int tokenOffset = tokenId * config.hiddenDim;

    int destEpOffset = myPe * maxNumOutTokenPerRank * config.hiddenDim;
    int destEpTokOffset = peTokenIdx * config.hiddenDim;

    if (laneId == 0) {
      ShmemPutUint32NbiThread<PrvdType>(handle.shmemOutTokMemObj, destEpOffset + destEpTokOffset,
                                        handle.shmemInpTokMemObj, tokenOffset, config.hiddenDim,
                                        destPe);
    }
  }

  // Send token num & token to expert mapping to other ranks
  for (int destPe = globalWarpId; destPe < npes; destPe++) {
    if (destPe == myPe) continue;
    if (laneId == 0) {
      ShmemPutUint32NbiThread<PrvdType>(
          handle.outTokToExptMapMemObj, myPe * maxNumOutTokenPerRank, handle.inpTokToExptMapMemObj,
          destPe * maxNumOutTokenPerRank, peTokenOffset[destPe], destPe);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      uint32_t numTokenSignal = peTokenOffset[destPe] + 1;
      ShmemPutUint32ImmNbiThread<PrvdType>(handle.recvTokenNumMemObj, myPe * sizeof(uint32_t),
                                           numTokenSignal, destPe);
    }
  }

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
    if (destPe != myPe) {
      uint32_t* signal = handle.recvTokenNumMemObj->GetAs<uint32_t*>() + destPe;
      ShmemUint32WaitUntilGreaterThan<PrvdType>(signal, 0);
      recvTokenNum[destPe] = *signal - 1;
      totalNumRecvToken += *signal - 1;
    }
  }

  // TODO: calculate expert offset
  for (int srcPe = warpId; srcPe < npes; srcPe += warpNum) {
    if (srcPe == myPe) continue;
    for (int tokId = laneId; tokId < recvTokenNum[srcPe]; tokId += warpSize) {
      uint32_t expertId = outTokToExptBuf[srcPe * maxNumOutTokenPerRank + tokId];
      uint32_t localExpertId = expertId % config.numExpertPerRank;
      // TODO: this assert fails occasionally, probably caused by incorrect RDMA memory order
      if ((expertId / config.numExpertPerRank) != myPe) {
        printf("mype %d expertId %d\n", myPe, expertId);
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
    if ((myPe == 1) && (blockIdx.x == 0))
      printf("mype %d expert id %d expert token %d accum offset %d\n", myPe, thdId, expertTokNum,
             accumOffset);
    accumExpertTokOffsets[thdId] = accumOffset;
  }
  __syncthreads();

  for (int i = laneId; i < config.numExpertPerRank; i += warpSize) {
    expertTokOff[i] = 0;
  }

  uint32_t* outTokenBuf = handle.outTokenBuf;
  uint32_t* shmemOutTokenBuf = handle.shmemOutTokMemObj->GetAs<uint32_t*>();

  for (int i = 0, tokId = 0, srcPe = 0; srcPe < npes; tokId++, i++) {
    // Tokens from srcPe are all copied
    // TODO: also copy tokens from my pe
    if ((tokId == recvTokenNum[srcPe]) || (srcPe == myPe)) {
      srcPe++;
      tokId = 0;
      continue;
    }

    uint32_t localExpertId =
        outTokToExptBuf[srcPe * maxNumOutTokenPerRank + tokId] % config.numExpertPerRank;
    if (laneId == 0) {
      expertTokOff[localExpertId] += 1;
    }

    if ((i % globalWarpNum) != globalWarpId) continue;  // skip token not assigned for this warp

    // Copy token
    uint32_t srcTokenOff = i * config.hiddenDim;
    uint32_t destTokenOff =
        (accumExpertTokOffsets[localExpertId] + expertTokOff[localExpertId] - 1) * config.hiddenDim;

    constexpr int vecSize = 16 / sizeof(uint32_t);
    for (int offset = laneId * vecSize; offset < config.hiddenDim; offset += warpSize * vecSize) {
      reinterpret_cast<uint4*>(outTokenBuf + destTokenOff + offset)[0] =
          reinterpret_cast<uint4*>(shmemOutTokenBuf + srcTokenOff + offset)[0];
    }
  }
}

void LaunchEpDispatchWithPutMemAPIKernel(EpDispatchConfig config, EpDispatchHandle handle) {
  dim3 grid(config.blockNum);
  dim3 block(warpSize * config.warpNumPerBlock);
  size_t sharedMemSize = 2 * config.worldSize * config.warpNumPerBlock * sizeof(uint32_t);
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
  config.maxNumInpTokenPerRank = 64;
  config.warpNumPerBlock = 3;
  config.blockNum = 4;

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
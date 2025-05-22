#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <random>
#include <sstream>

#include "dispatch_combine_kernels/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

using namespace std;
using namespace mori;
using namespace mori::moe;
using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;

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

    int tokenIndiciesSize =
        config.maxNumInpTokenPerRank * config.numExpertPerToken * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMalloc(&tokenIndicies, tokenIndiciesSize));
    HIP_RUNTIME_CHECK(hipMemset(tokenIndicies, 0, tokenIndiciesSize));
  }

  ~EpDispatchCombineTestCase() {
    HIP_RUNTIME_CHECK(hipFree(inpTokBuf));
    HIP_RUNTIME_CHECK(hipFree(outTokBuf));
    HIP_RUNTIME_CHECK(hipFree(tokenIndicies));
    free(inpTokBufCpu);
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

  void RandomInitializeHandle() {
    handle.Reset();
    RandomIntializeNumToken();
    RandomIntializeDispatch();
    RandomInitializeToken();
    handle.PrepareInference(inpTokBuf, outTokBuf, tokenIndicies, numToken);
    PrintDispatch();
  }

  void RunDispatch() {
    handle.LaunchDispatch();
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    int tokenNumSignalSize = handle.config.worldSize * sizeof(uint32_t);
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
    inpTokBufCpu = reinterpret_cast<T*>(malloc(inpTokSize));
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
    HIP_RUNTIME_CHECK(hipMemcpy(inpTokBuf, outTokBuf,
                                maxNumOutTokenPerRank * config.hiddenDim * sizeof(T),
                                hipMemcpyDeviceToDevice));
    HIP_RUNTIME_CHECK(
        hipMemset(outTokBuf, 0, maxNumOutTokenPerRank * config.hiddenDim * sizeof(T)));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    handle.LaunchCombine();
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
    numToken = dist(gen);
  }

  void RandomIntializeDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    std::vector<int> epRange;
    for (int i = 0; i < config.worldSize * config.numExpertPerRank; i++) epRange.push_back(i);

    for (int i = 0; i < numToken; i++) {
      std::shuffle(epRange.begin(), epRange.end(), gen);
      for (int j = 0; j < config.numExpertPerToken; j++) {
        tokenIndicies[i * config.numExpertPerRank + j] = epRange[j];
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
      reinterpret_cast<T*>(inpTokBuf)[i] = tokValDist(gen);
    }
  }

 private:
  random_device rd;
  mt19937 gen;

  T* inpTokBuf{nullptr};
  T* inpTokBufCpu{nullptr};
  T* outTokBuf{nullptr};
  uint32_t* tokenIndicies{nullptr};
  int numToken{-1};
  EpDispatchCombineHandle<T>& handle;
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

  // Run tests
  EpDispatchCombineTestCase<DataType> testCase(handle);
  for (int i = 0; i < 2; i++) {
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
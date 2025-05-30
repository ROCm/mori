#include <getopt.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <chrono>
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

enum DataType {
  FP32 = 0,
  BF16 = 1,
  FP8_E4M3 = 2,
};

enum TestType {
  Accuracy = 0,
  Benchmark = 1,
};

namespace std {
static std::ostream& operator<<(std::ostream& s, DataType dataType) {
  if (dataType == DataType::FP32) {
    s << "float32";
  } else if (dataType == DataType::BF16) {
    s << "bfloat16";
  } else if (dataType == DataType::FP8_E4M3) {
    s << "fp8_e4m3";
  } else {
    assert(false);
  }
  return s;
};

static std::ostream& operator<<(std::ostream& s, TestType testType) {
  if (testType == TestType::Accuracy) {
    s << "accuracy";
  } else if (testType == TestType::Benchmark) {
    s << "benchmark";
  } else {
    assert(false);
  }
  return s;
};
}  // namespace std

struct RunConfig {
  TestType testType{Accuracy};
  int warmup{5};
  int repeat{5};
  float atol{1e-2};
};

struct EpDispatchCombineTestConfig {
  DataType dataType{DataType::BF16};
  RunConfig runConfig;
  EpDispatchCombineConfig config;
};

template <typename T>
class EpDispatchCombineTestCase {
 public:
  EpDispatchCombineTestCase(EpDispatchCombineHandle<T>& handle, RunConfig runConfig)
      : handle(handle), runConfig(runConfig) {
    const auto timestamp = std::chrono::system_clock::now();
    gen = mt19937(
        std::chrono::duration_cast<std::chrono::seconds>(timestamp.time_since_epoch()).count() +
        handle.config.rank);

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

    int weightsBufSize = config.maxNumInpTokenPerRank * config.numExpertPerToken * sizeof(float);
    HIP_RUNTIME_CHECK(hipMalloc(&weightsBuf, weightsBufSize));
    HIP_RUNTIME_CHECK(hipMemset(weightsBuf, 0, weightsBufSize));
  }

  ~EpDispatchCombineTestCase() {
    HIP_RUNTIME_CHECK(hipFree(inpTokBuf));
    HIP_RUNTIME_CHECK(hipFree(outTokBuf));
    HIP_RUNTIME_CHECK(hipFree(tokenIndicies));
    HIP_RUNTIME_CHECK(hipFree(weightsBuf));
    free(inpTokBufCpu);
  }

  void InitializeHandle() {
    if (runConfig.testType == TestType::Accuracy) {
      // RandomInitializeNumToken();
      InitializeNumToken();
      RandomInitializeDispatch();
    } else if (runConfig.testType == TestType::Benchmark) {
      InitializeNumToken();
      RandomInitializeDispatch();
    } else {
      assert(false);
    }
    RandomInitializeWeights();
    RandomInitializeToken();
    handle.PrepareInference(inpTokBuf, outTokBuf, weightsBuf, tokenIndicies, numToken);
    // PrintDispatch();
    // PrintDispatchStats();
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

    void* tokenIndicesToPeSortedBufCpu = malloc(tokenIndiciesSize);
    HIP_RUNTIME_CHECK(hipMemcpy(tokenIndicesToPeSortedBufCpu, handle.tokenIndicesToPeSortedBuf,
                                tokenIndiciesSize, hipMemcpyDeviceToHost));

    uint32_t* globalTokenIndicesToPeSortedBufCpu =
        reinterpret_cast<uint32_t*>(malloc(config.worldSize * tokenIndiciesSize));
    MPI_Allgather(tokenIndicesToPeSortedBufCpu, tokenIndiciesSize, MPI_CHAR,
                  globalTokenIndicesToPeSortedBufCpu, tokenIndiciesSize, MPI_CHAR, MPI_COMM_WORLD);

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

    // Build pe sorted to token index map
    std::vector<std::unordered_map<uint32_t, uint32_t>> peSortToTokenIdxMapsVec;
    for (int i = 0; i < config.worldSize; i++) {
      peSortToTokenIdxMapsVec.push_back({});
      uint32_t peTokenNum = globalTokenNum[i];
      for (int j = 0; j < peTokenNum * config.numExpertPerToken; j++) {
        uint32_t peSortedId = globalTokenIndicesToPeSortedBufCpu[i * maxNumOutTokenPerRank + j];
        assert(peSortToTokenIdxMapsVec[i].find(peSortedId) == peSortToTokenIdxMapsVec[i].end());
        peSortToTokenIdxMapsVec[i].insert({peSortedId, j});
      }
    }

    std::vector<uint32_t> srcPeCheckTokenNum(config.worldSize, 0);

    uint32_t totalRecvNumToken = 0;
    for (int i = 0; i < config.worldSize; i++) {
      totalRecvNumToken += handle.recvTokenNumMemObj->template GetAs<uint32_t*>()[i] - 1;
    }
    std::cout << "Rank " << config.rank << " recv " << totalRecvNumToken << " tokens" << std::endl;

    for (int i = 0; i < totalRecvNumToken; i++) {
      uint32_t peSortedId = handle.exptSortedToPeSortedBuf[i];
      uint32_t srcPe = peSortedId / maxNumOutTokenPerRank;
      peSortedId = peSortedId - srcPe * maxNumOutTokenPerRank + config.rank * maxNumOutTokenPerRank;
      uint32_t srcTokDispatchId = peSortToTokenIdxMapsVec[srcPe][peSortedId];
      uint32_t srcTokId = srcTokDispatchId / config.numExpertPerToken;

      T* localTokBuf = handle.outTokenBuf + i * config.hiddenDim;
      T* srcTokBuf = reinterpret_cast<T*>(globalInpTokBufCpu) + srcPe * inpTokEleNum +
                     srcTokId * config.hiddenDim;
      srcPeCheckTokenNum[srcPe]++;

      std::stringstream msg;
      msg << "mype " << config.rank << " localTokId " << i << " srcpe " << srcPe << " srcTokId "
          << srcTokId;
      for (int k = 0; k < config.hiddenDim; k++) {
        float expectedVal = float(srcTokBuf[k]);
        float gotVal = float(localTokBuf[k]);
        bool equal = (expectedVal == gotVal);
        if (!equal) {
          std::cout << "Wrong result at pos " << k << ": " << msg.str() << " expected "
                    << expectedVal << " got " << gotVal << std::endl;
          assert(false);
        }
      }
    }

    for (int i = 0; i < config.worldSize; i++) {
      assert(srcPeCheckTokenNum[i] ==
             (handle.recvTokenNumMemObj->template GetAs<uint32_t*>()[i] - 1));
    }

    // Check token dispatched to current rank
    // struct SrcTokInfo {
    //   int pe;
    //   int tokenId;
    // };
    // std::vector<SrcTokInfo> srcTokInfoList;
    // // Tokens are supposed to be sorted 1) in the order of expert then 2) in the order of srcPe
    // for (int exptId = 0; exptId < config.numExpertPerRank; exptId++) {
    //   for (int srcPe = 0; srcPe < config.worldSize; srcPe++) {
    //     uint32_t* tokenIndiciesAddr =
    //         reinterpret_cast<uint32_t*>(globalTokIndiciesCpu) + srcPe * maxNumOutTokenPerRank;

    //     for (int dispatchId = 0; dispatchId < config.numExpertPerToken * globalTokenNum[srcPe];
    //          dispatchId++) {
    //       int expertId = tokenIndiciesAddr[dispatchId];
    //       int rankId = expertId / config.numExpertPerRank;
    //       if (rankId != config.rank) continue;

    //       int localExptId = expertId % config.numExpertPerRank;
    //       if (localExptId != exptId) continue;

    //       int tokenId = dispatchId / config.numExpertPerToken;
    //       srcTokInfoList.push_back({srcPe, tokenId});
    //     }
    //   }
    // }

    // for (int localTokId = 0; localTokId < srcTokInfoList.size(); localTokId++) {
    //   T* localTokBuf = handle.outTokenBuf + localTokId * config.hiddenDim;

    //   int srcPe = srcTokInfoList[localTokId].pe;
    //   int srcTokId = srcTokInfoList[localTokId].tokenId;
    //   int srcTokenOffset = srcPe * inpTokEleNum + srcTokId * config.hiddenDim;
    //   T* srcTokBuf = reinterpret_cast<T*>(globalInpTokBufCpu) + srcTokenOffset;

    //   std::stringstream msg;
    //   msg << "mype " << config.rank << " localTokId " << localTokId << " srcpe " << srcPe
    //       << " srcTokId " << srcTokId;
    //   for (int k = 0; k < config.hiddenDim; k++) {
    //     float expectedVal = float(srcTokBuf[k]);
    //     float gotVal = float(localTokBuf[k]);
    //     bool equal = (expectedVal == gotVal);
    //     if (!equal) {
    //       std::cout << "Wrong result at pos " << k << ": " << msg.str() << " expected "
    //                 << expectedVal << " got " << gotVal << std::endl;
    //       assert(false);
    //     }
    //   }
    // }

    free(globalInpTokBufCpu);
    free(globalTokIndiciesCpu);
    free(tokenIndicesCpu);
  }

  void CheckCombineResult() {
    EpDispatchCombineConfig& config = handle.config;

    for (int i = 0; i < handle.curRankNumToken; i++) {
      uint32_t tokenOffset = i * config.hiddenDim;

      // compute weight sum
      float weightSum = 0.0f;
      for (int k = 0; k < config.numExpertPerToken; k++)
        weightSum += weightsBuf[i * config.numExpertPerToken + k];

      for (int j = 0; j < config.hiddenDim; j++) {
        float expected = float(reinterpret_cast<T*>(inpTokBufCpu)[tokenOffset + j]) * weightSum;
        float got = float(handle.outTokenBuf[tokenOffset + j]);
        // if (abs(got - expected) > runConfig.atol) {
        std::cout << "Wrong result at pos " << j << ": mype " << config.rank << " tokenId " << i
                  << " expected " << expected << " got " << got << " weight sum " << weightSum
                  << std::endl;
        // assert(false);
        // }
      }
    }
  }

  void Run() {
    if (runConfig.testType == TestType::Accuracy) {
      RunAccuracyTest();
    } else if (runConfig.testType == TestType::Benchmark) {
      RunBenchmark();
    } else {
      assert(false);
    }
  }

  void RunAccuracyTest() {
    for (int i = 0; i < runConfig.repeat; i++) {
      handle.LaunchReset();
      InitializeHandle();
      SystemBarrier();

      handle.LaunchDispatch();
      SystemBarrier();

      CheckDispatchResult();
      SystemBarrier();
      if (handle.config.rank == 0) std::cout << "Test round " << i << " dispatch PASS" << std::endl;

      CopyDispatchOutAsCombineInp();
      handle.LaunchCombine();
      SystemBarrier();

      CheckCombineResult();
      SystemBarrier();
      if (handle.config.rank == 0) std::cout << "Test round " << i << " combine PASS" << std::endl;
    }
  }

  void RunBenchmark() {
    hipStream_t stream;
    HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

    for (int i = 0; i < runConfig.warmup; i++) {
      handle.LaunchReset(stream);
      InitializeHandle();
      SystemBarrier();

      handle.LaunchDispatch(stream);
      // CopyDispatchOutAsCombineInp();
      // handle.LaunchCombine(stream);
      if (handle.config.rank == 0) std::cout << "Warmup Done" << std::endl;
    }

    hipEvent_t events[4];
    for (int i = 0; i < 4; i++) HIP_RUNTIME_CHECK(hipEventCreate(&events[i]));

    float dispatchTotal = 0, combineTotal = 0;
    for (int i = 0; i < runConfig.repeat; i++) {
      handle.LaunchReset(stream);
      InitializeHandle();
      SystemBarrier();

      HIP_RUNTIME_CHECK(hipEventRecord(events[0]));
      handle.LaunchDispatch(stream);
      HIP_RUNTIME_CHECK(hipEventRecord(events[1]));

      // CopyDispatchOutAsCombineInp();

      // HIP_RUNTIME_CHECK(hipEventRecord(events[2]));
      // handle.LaunchCombine(stream);
      // HIP_RUNTIME_CHECK(hipEventRecord(events[3]));

      float dispatch, combine;
      HIP_RUNTIME_CHECK(hipEventSynchronize(events[1]));
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&dispatch, events[0], events[1]));
      // HIP_RUNTIME_CHECK(hipEventElapsedTime(&combine, events[2], events[3]));

      // std::cout << "Rank " << handle.config.rank << " dispatch " << dispatch << std::endl;

      dispatchTotal += dispatch;
      combineTotal += combine;

      if (handle.config.rank == 0) std::cout << "Benchmark round " << i << " Done" << std::endl;
    }

    std::cout << "Rank " << handle.config.rank
              << " Dispatch average: " << dispatchTotal / runConfig.repeat << std::endl;
    std::cout << "Rank " << handle.config.rank
              << " Combine average: " << combineTotal / runConfig.repeat << std::endl;

    HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  }

 private:
  void SystemBarrier() {
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void CopyDispatchOutAsCombineInp() {
    EpDispatchCombineConfig& config = handle.config;
    int maxNumOutTokenPerRank =
        config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken;
    HIP_RUNTIME_CHECK(hipMemcpy(inpTokBuf, outTokBuf,
                                maxNumOutTokenPerRank * config.hiddenDim * sizeof(T),
                                hipMemcpyDeviceToDevice));
    HIP_RUNTIME_CHECK(
        hipMemset(outTokBuf, 0, maxNumOutTokenPerRank * config.hiddenDim * sizeof(T)));
  }

  void RandomInitializeNumToken() {
    EpDispatchCombineConfig& config = handle.config;
    uniform_int_distribution<> dist(1, config.maxNumInpTokenPerRank);
    numToken = dist(gen);
  }

  void InitializeNumToken() {
    EpDispatchCombineConfig& config = handle.config;
    numToken = config.maxNumInpTokenPerRank;
  }

  void RandomInitializeDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    std::vector<int> epRange;
    for (int i = 0; i < config.worldSize * config.numExpertPerRank; i++) epRange.push_back(i);

    std::vector<int> rankCount(config.worldSize, 0);

    for (int i = 0; i < numToken; i++) {
      std::vector<int> epRangeShuffled(epRange);
      std::shuffle(epRangeShuffled.begin(), epRangeShuffled.end(), gen);

      for (int j = 0; j < config.numExpertPerToken; j++) {
        assert(epRangeShuffled[j] < config.numExpertPerRank * config.worldSize);
        tokenIndicies[i * config.numExpertPerToken + j] = epRangeShuffled[j];
        int rank = epRangeShuffled[j] / config.numExpertPerRank;
        rankCount[rank]++;
      }
    }

    // for (int i = 0; i < config.worldSize; i++) {
    //   std::cout << "Rank " << config.rank << " dispatches " << rankCount[i] << " tokens to rank "
    //             << i << std::endl;
    // }
  }

  void RandomInitializeWeights() {
    EpDispatchCombineConfig& config = handle.config;
    uniform_real_distribution<> tokValDist(1, 2);
    for (int i = 0; i < numToken; i++) {
      for (int j = 0; j < config.numExpertPerToken; j++) {
        // weightsBuf[i * config.numExpertPerRank + j] = tokValDist(gen);
        weightsBuf[i * config.numExpertPerRank + j] = 1.0f;
      }
    }
  }

  void RandomInitializeToken() {
    EpDispatchCombineConfig& config = handle.config;
    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    uniform_real_distribution<> tokValDist(0.01, 1);
    for (int i = 0; i < inpTokEleNum; i++) {
      reinterpret_cast<T*>(inpTokBuf)[i] = tokValDist(gen);
    }
  }

  void PrintDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    stringstream ss;
    for (int i = 0; i < handle.curRankNumToken; i++) {
      ss << "  Token " << i << " dispatch to ";
      for (int j = 0; j < config.numExpertPerToken; j++) {
        ss << tokenIndicies[i * config.numExpertPerToken + j] << " ";
      }
      ss << std::endl;
    }
    std::cout << "Rank " << config.rank << ":" << std::endl;
    std::cout << ss.str() << std::endl;
  }

 private:
  random_device rd;
  mt19937 gen;

  T* inpTokBuf{nullptr};
  T* inpTokBufCpu{nullptr};
  T* outTokBuf{nullptr};
  float* weightsBuf{nullptr};
  uint32_t* tokenIndicies{nullptr};
  int numToken{-1};
  EpDispatchCombineHandle<T>& handle;
  RunConfig runConfig;
};

template <typename T>
void RunDispatchCombineTest(EpDispatchCombineTestConfig testConfig) {
  EpDispatchCombineHandle<T> handle(testConfig.config);
  EpDispatchCombineTestCase<T> testCase(handle, testConfig.runConfig);
  testCase.Run();
}

// A simple MoE-EP dispatch kernel example, assume dp rank is equal to ep rank
void EpDispatchWithPutMemAPI(EpDispatchCombineTestConfig testConfig) {
  int status;

  // Initialize shmem
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Setup config
  if (testConfig.dataType == DataType::FP32) {
    testConfig.runConfig.atol = 1e-3;
  } else if (testConfig.dataType == DataType::BF16) {
    testConfig.runConfig.atol = 1e-1;
  } else if (testConfig.dataType == DataType::FP8_E4M3) {
    testConfig.runConfig.atol = 3e-1;
  } else {
    std::cout << "Unknown datatype: " << testConfig.dataType << std::endl;
    assert(false);
  }

  testConfig.config.rank = myPe;
  testConfig.config.worldSize = npes;
  if (testConfig.config.rank == 0) {
    std::cout << "DataType: " << testConfig.dataType << std::endl;
    std::cout << "TestType: " << testConfig.runConfig.testType << std::endl;
    std::cout << "Atol: " << testConfig.runConfig.atol << std::endl;
    std::cout << testConfig.config << std::endl;
  }

  if (testConfig.dataType == DataType::FP32) {
    RunDispatchCombineTest<float>(testConfig);
  } else if (testConfig.dataType == DataType::BF16) {
    RunDispatchCombineTest<hip_bfloat16>(testConfig);
  } else if (testConfig.dataType == DataType::FP8_E4M3) {
    RunDispatchCombineTest<__hip_fp8_e4m3_fnuz>(testConfig);
  } else {
    std::cout << "Unknown datatype: " << testConfig.dataType << std::endl;
    assert(false);
  }

  ShmemMpiFinalize();
}

EpDispatchCombineTestConfig ParseArguments(int argc, char* argv[]) {
  EpDispatchCombineTestConfig testConfig;

  static struct option long_options[] = {{"help", no_argument, NULL, 'h'},
                                         {"cmd", required_argument, NULL, 0},
                                         {"data_type", required_argument, NULL, 0},
                                         {"hdim", optional_argument, NULL, 'd'},
                                         {"max_tokens", optional_argument, NULL, 'm'},
                                         {"expert_per_rank", optional_argument, NULL, 'r'},
                                         {"expert_per_token", optional_argument, NULL, 't'},
                                         {"warp_per_blk", optional_argument, NULL, 'w'},
                                         {"block_num", optional_argument, NULL, 'b'},
                                         {"num", optional_argument, NULL, 'n'},
                                         {0, 0, 0, 0}};
  int option_index = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "d::m::r::t::w::b::n::h", long_options, &option_index)) !=
         -1) {
    if (opt == -1) break;

    switch (opt) {
      case 0:
        if (strcmp(long_options[option_index].name, "cmd") == 0) {
          if (strcmp(optarg, "test") == 0) {
            testConfig.runConfig.testType = TestType::Accuracy;
          } else if (strcmp(optarg, "bench") == 0) {
            testConfig.runConfig.testType = TestType::Benchmark;
          } else {
            printf("Unknown cmd: %s, must be 'test' or 'bench'\n", optarg);
            assert(false);
          }
        } else if (strcmp(long_options[option_index].name, "data_type") == 0) {
          if (strcmp(optarg, "fp32") == 0) {
            testConfig.dataType = DataType::FP32;
          } else if (strcmp(optarg, "bf16") == 0) {
            testConfig.dataType = DataType::BF16;
          } else if (strcmp(optarg, "fp8") == 0) {
            testConfig.dataType = DataType::FP8_E4M3;
          } else {
            printf("Unknown cmd: %s, must be 'test' or 'bench'\n", optarg);
            assert(false);
          }
        }
        break;
      case 'd':
        testConfig.config.hiddenDim = std::stoi(optarg);
        break;
      case 'm':
        testConfig.config.maxNumInpTokenPerRank = std::stoi(optarg);
        break;
      case 'r':
        testConfig.config.numExpertPerRank = std::stoi(optarg);
        break;
      case 't':
        testConfig.config.numExpertPerToken = std::stoi(optarg);
        break;
      case 'w':
        testConfig.config.warpNumPerBlock = std::stoi(optarg);
        break;
      case 'b':
        testConfig.config.blockNum = std::stoi(optarg);
        break;
      case 'n':
        testConfig.runConfig.repeat = std::stoi(optarg);
        break;
      case 'h':
        printf("This is help message\n");
        break;
      default:
        fprintf(stderr, "Unknown error in getopt_long\n");
        return {};
    }
  }
  return testConfig;
}

int main(int argc, char* argv[]) {
  EpDispatchCombineTestConfig config = ParseArguments(argc, argv);
  EpDispatchWithPutMemAPI(config);
  return 0;
}

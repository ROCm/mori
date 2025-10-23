// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <getopt.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <random>
#include <sstream>
#include <unordered_set>

#include "mori/ops/ops.hpp"
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
  KernelType kernelType{KernelType::IntraNode};
  int warmup{5};
  int repeat{5};
  float atol{1e-2};
  bool isAiterMoE{true};
};

struct EpDispatchCombineTestConfig {
  DataType dataType{DataType::BF16};
  RunConfig runConfig;
  EpDispatchCombineConfig config;
};

template <typename T>
hipDataType GetHipDataType();

template <>
hipDataType GetHipDataType<float>() {
  return HIP_R_32F;
}

template <>
hipDataType GetHipDataType<hip_bfloat16>() {
  return HIP_R_16BF;
}

template <>
hipDataType GetHipDataType<__hip_fp8_e4m3_fnuz>() {
  return HIP_R_8F_E4M3_FNUZ;
}

// inline bool IsNoDataRank(const EpDispatchCombineConfig& config) { return config.rank < 4; }
inline bool IsNoDataRank(const EpDispatchCombineConfig& config) { return false; }

template <typename T>
class EpDispatchCombineTestCase {
 public:
  EpDispatchCombineTestCase(EpDispatchCombineHandle& handle, RunConfig runConfig)
      : handle(handle), runConfig(runConfig) {
    const auto timestamp = std::chrono::system_clock::now();
    gen = mt19937(
        std::chrono::duration_cast<std::chrono::seconds>(timestamp.time_since_epoch()).count() +
        handle.config.rank);

    EpDispatchCombineConfig& config = handle.config;

    // Set kernel input/output token buffer
    int maxTokenSize = config.MaxNumTokensToRecvPerRank() * config.hiddenDim * sizeof(T);
    HIP_RUNTIME_CHECK(hipMalloc(&inpTokBuf, maxTokenSize));
    HIP_RUNTIME_CHECK(hipMemset(inpTokBuf, 0, maxTokenSize));
    HIP_RUNTIME_CHECK(hipMalloc(&outTokBuf, maxTokenSize));
    HIP_RUNTIME_CHECK(hipMemset(outTokBuf, 0, maxTokenSize));
    HIP_RUNTIME_CHECK(hipHostMalloc(&inpTokBufCpu, maxTokenSize));
    HIP_RUNTIME_CHECK(hipMemset(inpTokBufCpu, 0, maxTokenSize));

    int tokenIndicesSize = config.MaxNumTokensToSendPerRank() * sizeof(index_t);
    HIP_RUNTIME_CHECK(hipMalloc(&tokenIndices, tokenIndicesSize));
    HIP_RUNTIME_CHECK(hipMemset(tokenIndices, 0, tokenIndicesSize));

    int weightsBufSize = config.MaxNumTokensToSendPerRank() * sizeof(float);
    HIP_RUNTIME_CHECK(hipMalloc(&weightsBuf, weightsBufSize));
    HIP_RUNTIME_CHECK(hipMemset(weightsBuf, 0, weightsBufSize));

    if (config.scaleDim > 0) {
      int scalesBufSize =
          config.MaxNumTokensToSendPerRank() * config.scaleDim * config.scaleTypeSize;
      HIP_RUNTIME_CHECK(hipMalloc(&scalesBuf, scalesBufSize));
      HIP_RUNTIME_CHECK(hipMemset(scalesBuf, 0, scalesBufSize));
    }
  }

  ~EpDispatchCombineTestCase() {
    HIP_RUNTIME_CHECK(hipFree(inpTokBuf));
    HIP_RUNTIME_CHECK(hipFree(outTokBuf));
    HIP_RUNTIME_CHECK(hipFree(tokenIndices));
    HIP_RUNTIME_CHECK(hipFree(weightsBuf));
    HIP_RUNTIME_CHECK(hipFree(scalesBuf));
    HIP_RUNTIME_CHECK(hipFree(inpTokBufCpu));
  }

  void InitializeHandle() {
    if (runConfig.testType == TestType::Accuracy) {
      InitializeNumToken();
      // RandomInitializeNumToken();

      RandomInitializeDispatch();
      // RoundRobinInitializeDispatch();
    } else if (runConfig.testType == TestType::Benchmark) {
      InitializeNumToken();
      // RandomInitializeDispatch();
      RoundRobinInitializeDispatch();
    } else {
      assert(false);
    }
    RandomInitializeWeights();
    RandomInitializeScales();
    RandomInitializeToken();
    if (IsNoDataRank(handle.config)) {
      handle.PrepareInference(GetHipDataType<T>(), nullptr, outTokBuf, weightsBuf, scalesBuf,
                              nullptr, 0);
    } else {
      handle.PrepareInference(GetHipDataType<T>(), inpTokBuf, outTokBuf, weightsBuf, scalesBuf,
                              tokenIndices, numToken);
    }
    // PrintDispatch();
    // PrintDispatchStats();
  }

  void CheckDispatchResult() {
    EpDispatchCombineConfig& config = handle.config;

    // Copy token indices to CPU
    int MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
    int tokenIndicesSize = config.MaxNumTokensToSendPerRank() * sizeof(index_t);

    // Collect token indices from all ranks
    void* tokenIndicesCpu = malloc(tokenIndicesSize);
    memset(tokenIndicesCpu, 0, tokenIndicesSize);
    if (handle.tokenIndices) {
      HIP_RUNTIME_CHECK(
          hipMemcpy(tokenIndicesCpu, handle.tokenIndices, tokenIndicesSize, hipMemcpyDeviceToHost));
    }

    void* globalTokIndicesCpu = malloc(config.worldSize * tokenIndicesSize);
    MPI_Allgather(tokenIndicesCpu, tokenIndicesSize, MPI_CHAR, globalTokIndicesCpu,
                  tokenIndicesSize, MPI_CHAR, MPI_COMM_WORLD);

    void* dispSenderIdxMapCpu = malloc(tokenIndicesSize);
    HIP_RUNTIME_CHECK(hipMemcpy(dispSenderIdxMapCpu, handle.dispSenderIdxMap, tokenIndicesSize,
                                hipMemcpyDeviceToHost));

    index_t* globaldispSenderIdxMapCpu =
        reinterpret_cast<index_t*>(malloc(config.worldSize * tokenIndicesSize));
    MPI_Allgather(dispSenderIdxMapCpu, tokenIndicesSize, MPI_CHAR, globaldispSenderIdxMapCpu,
                  tokenIndicesSize, MPI_CHAR, MPI_COMM_WORLD);

    // Collect token num from all ranks
    std::vector<index_t> globalTokenNum(config.worldSize);
    MPI_Allgather(&handle.curRankNumToken, 1, MPI_INT32_T, globalTokenNum.data(), 1, MPI_INT32_T,
                  MPI_COMM_WORLD);

    // Collect tokens from all ranks
    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    int inpTokSize = inpTokEleNum * sizeof(T);
    HIP_RUNTIME_CHECK(hipMemset(inpTokBufCpu, 0, inpTokSize));
    if (handle.inpTokenBuf) {
      HIP_RUNTIME_CHECK(
          hipMemcpy(inpTokBufCpu, handle.inpTokenBuf, inpTokSize, hipMemcpyDeviceToHost));
    }

    void* globalInpTokBufCpu = malloc(config.worldSize * inpTokSize);
    MPI_Allgather(inpTokBufCpu, inpTokSize, MPI_CHAR, globalInpTokBufCpu, inpTokSize, MPI_CHAR,
                  MPI_COMM_WORLD);

    // TODO: Collect scales from all ranks

    index_t totalRecvNumToken = 0;
    for (int i = 0; i < config.worldSize; i++) {
      totalRecvNumToken += handle.recvTokenNumMemObj->template GetAs<index_t*>()[i] - 1;
    }
    std::cout << "Rank " << config.rank << " recv " << totalRecvNumToken << " tokens" << std::endl;

    if (runConfig.kernelType == IntraNode) {
      for (int i = 0; i < totalRecvNumToken; i++) {
        index_t srcTokId = handle.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[i];

        index_t srcPe = srcTokId / config.maxNumInpTokenPerRank;
        index_t localSrcTokId = srcTokId - srcPe * config.maxNumInpTokenPerRank;

        T* localTokBuf =
            handle.shmemDispatchOutTokMemObj->template GetAs<T*>() + i * config.hiddenDim;
        T* srcTokBuf = reinterpret_cast<T*>(globalInpTokBufCpu) + srcPe * inpTokEleNum +
                       localSrcTokId * config.hiddenDim;

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
          assert(expectedVal != 0);
        }
      }
    } else if (runConfig.kernelType == InterNode) {
      // Build pe sorted to token index map
      std::vector<std::unordered_map<index_t, index_t>> peSortToTokenIdxMapsVec;
      for (int i = 0; i < config.worldSize; i++) {
        peSortToTokenIdxMapsVec.push_back({});
        index_t peTokenNum = globalTokenNum[i];
        for (int j = 0; j < peTokenNum * config.numExpertPerToken; j++) {
          index_t peSortedId =
              globaldispSenderIdxMapCpu[i * config.MaxNumTokensToSendPerRank() + j];
          assert(peSortToTokenIdxMapsVec[i].find(peSortedId) == peSortToTokenIdxMapsVec[i].end());
          peSortToTokenIdxMapsVec[i].insert({peSortedId, j});
        }
      }

      std::vector<index_t> srcPeCheckTokenNum(config.worldSize, 0);
      for (int i = 0; i < totalRecvNumToken; i++) {
        index_t peSortedId = handle.dispReceiverIdxMap[i];
        index_t srcPe = peSortedId / config.MaxNumTokensToSendPerRank();
        peSortedId = peSortedId - srcPe * config.MaxNumTokensToSendPerRank() +
                     config.rank * config.MaxNumTokensToSendPerRank();
        index_t srcTokDispatchId = peSortToTokenIdxMapsVec[srcPe][peSortedId];
        index_t srcTokId = srcTokDispatchId / config.numExpertPerToken;

        T* localTokBuf =
            handle.shmemDispatchOutTokMemObj->template GetAs<T*>() + i * config.hiddenDim;

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
               (handle.recvTokenNumMemObj->template GetAs<index_t*>()[i] - 1));
      }
    }

    free(globalInpTokBufCpu);
    free(globalTokIndicesCpu);
    free(tokenIndicesCpu);
  }

  void CheckCombineResult() {
    EpDispatchCombineConfig& config = handle.config;
    for (int i = 0; i < handle.curRankNumToken; i++) {
      index_t tokenOffset = i * config.hiddenDim;

      // compute weight sum
      float weightSum = 0.0f;
      if (runConfig.isAiterMoE) {
        std::unordered_set<index_t> pes;
        for (int j = 0; j < config.numExpertPerToken; j++) {
          index_t exptId = handle.tokenIndices[i * config.numExpertPerToken + j];
          index_t destPe = exptId / config.numExpertPerRank;
          pes.insert(destPe);
        }
        weightSum = 1.0f * pes.size();
      } else {
        for (int j = 0; j < config.numExpertPerToken; j++) {
          weightSum += weightsBuf[i * config.numExpertPerToken + j];
        }
      }

      for (int j = 0; j < config.hiddenDim; j++) {
        float expected =
            float(T(float(reinterpret_cast<T*>(inpTokBufCpu)[tokenOffset + j]) * weightSum));
        // float got = float(handle.outTokenBuf[tokenOffset + j]);
        float got = float(handle.shmemCombineOutTokMemObj->template GetAs<T*>()[tokenOffset + j]);
        assert(weightSum != 0);
        if (abs(got - expected) > runConfig.atol) {
          std::cout << "Wrong result at pos " << j << ": mype " << config.rank << " tokenId " << i
                    << " expected " << expected << " got " << got << " weight sum " << weightSum
                    << " src " << float(reinterpret_cast<T*>(inpTokBufCpu)[tokenOffset + j])
                    << std::endl;
          assert(false);
        }
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
      InitializeHandle();

      handle.LaunchDispatch(runConfig.kernelType);

      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
      CheckDispatchResult();
      if (handle.config.rank == 0) std::cout << "Test round " << i << " dispatch PASS" << std::endl;

      CopyDispatchOutAsCombineInp();

      handle.LaunchCombine(runConfig.kernelType);

      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
      CheckCombineResult();
      if (handle.config.rank == 0) std::cout << "Test round " << i << " combine PASS" << std::endl;
    }
    SystemBarrier();
    std::cout << "rank " << handle.config.rank << " done" << std::endl;
  }

  void RunBenchmark() {
    hipStream_t stream;
    HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

    for (int i = 0; i < runConfig.warmup; i++) {
      InitializeHandle();
      SystemBarrier();

      handle.LaunchDispatch(runConfig.kernelType, -1, -1, stream);
      CopyDispatchOutAsCombineInp();
      SystemBarrier();
      handle.LaunchCombine(runConfig.kernelType, -1, -1, stream);
      if (handle.config.rank == 0) std::cout << "Warmup Done" << std::endl;
    }

    hipEvent_t events[4];
    for (int i = 0; i < 4; i++) HIP_RUNTIME_CHECK(hipEventCreate(&events[i]));

    float dispatchTotal = 0, combineTotal = 0;
    int dispatchTotalRecvTokenNum = 0;
    for (int i = 0; i < runConfig.repeat; i++) {
      InitializeHandle();
      SystemBarrier();

      HIP_RUNTIME_CHECK(hipEventRecord(events[0]));
      handle.LaunchDispatch(runConfig.kernelType, -1, -1, stream);
      HIP_RUNTIME_CHECK(hipEventRecord(events[1]));

      HIP_RUNTIME_CHECK(hipMemcpy(&dispatchTotalRecvTokenNum, handle.totalRecvTokenNum, sizeof(int),
                                  hipMemcpyDeviceToHost));
      CopyDispatchOutAsCombineInp();
      SystemBarrier();

      HIP_RUNTIME_CHECK(hipEventRecord(events[2]));
      handle.LaunchCombine(runConfig.kernelType, -1, -1, stream);
      HIP_RUNTIME_CHECK(hipEventRecord(events[3]));

      float dispatch, combine;
      HIP_RUNTIME_CHECK(hipEventSynchronize(events[3]));
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&dispatch, events[0], events[1]));
      HIP_RUNTIME_CHECK(hipEventElapsedTime(&combine, events[2], events[3]));

      dispatchTotal += dispatch;
      combineTotal += combine;

      if (handle.config.rank == 0) std::cout << "Benchmark round " << i << " Done" << std::endl;
    }

    size_t total_bytes = dispatchTotalRecvTokenNum * handle.config.hiddenDim * sizeof(T);
    float dispatchBw = (total_bytes / 1e9f) / (dispatchTotal / runConfig.repeat / 1000.0f);
    float combineBw = (total_bytes / 1e9f) / (combineTotal / runConfig.repeat / 1000.0f);
    std::cout << "Rank " << handle.config.rank << " recvTokenNum " << dispatchTotalRecvTokenNum
              << " Dispatch average time: " << dispatchTotal / runConfig.repeat
              << " bw: " << dispatchBw << std::endl;
    std::cout << "Rank " << handle.config.rank << " recvTokenNum " << dispatchTotalRecvTokenNum
              << " Combine average time: " << combineTotal / runConfig.repeat
              << " bw: " << combineBw << std::endl;
    HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  }

 private:
  void SystemBarrier() {
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void CopyDispatchOutAsCombineInp() {
    EpDispatchCombineConfig& config = handle.config;
    if (IsNoDataRank(handle.config)) {
      handle.PrepareInference(GetHipDataType<T>(), inpTokBuf, outTokBuf, weightsBuf, scalesBuf,
                              nullptr, 0);
    }
    // HIP_RUNTIME_CHECK(hipMemcpy(inpTokBuf, outTokBuf,
    //                             config.MaxNumTokensToRecvPerRank() * config.hiddenDim *
    //                             sizeof(T), hipMemcpyDeviceToDevice));
    HIP_RUNTIME_CHECK(hipMemcpy(inpTokBuf, handle.shmemDispatchOutTokMemObj->template GetAs<T*>(),
                                config.MaxNumTokensToRecvPerRank() * config.hiddenDim * sizeof(T),
                                hipMemcpyDeviceToDevice));
    HIP_RUNTIME_CHECK(
        hipMemset(outTokBuf, 0, config.MaxNumTokensToRecvPerRank() * config.hiddenDim * sizeof(T)));
    // HIP_RUNTIME_CHECK(hipDeviceSynchronize());
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
        tokenIndices[i * config.numExpertPerToken + j] = epRangeShuffled[j];
        int rank = epRangeShuffled[j] / config.numExpertPerRank;
        rankCount[rank]++;
      }
    }

    for (int i = 0; i < config.worldSize; i++) {
      std::cout << "Rank " << config.rank << " dispatches " << rankCount[i] << " tokens to rank "
                << i << std::endl;
    }
  }

  void RoundRobinInitializeDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    std::vector<int> epRange;
    for (int i = 0; i < config.worldSize * config.numExpertPerRank; i++) epRange.push_back(i);

    std::vector<int> rankCount(config.worldSize, 0);

    for (int i = 0; i < numToken; i++) {
      for (int j = 0; j < config.numExpertPerToken; j++) {
        index_t dispIdx = i * config.numExpertPerToken + j;
        index_t destPe = dispIdx % config.worldSize;

        index_t localExpertId = dispIdx / config.worldSize % config.numExpertPerRank;

        tokenIndices[dispIdx] = destPe * config.numExpertPerRank + localExpertId;

        rankCount[destPe]++;
      }
    }

    for (int i = 0; i < config.worldSize; i++) {
      std::cout << "Rank " << config.rank << " dispatches " << rankCount[i] << " tokens to rank "
                << i << std::endl;
    }
  }

  void RandomInitializeWeights() {
    EpDispatchCombineConfig& config = handle.config;
    uniform_real_distribution<> tokValDist(1, 2);
    for (int i = 0; i < numToken; i++) {
      for (int j = 0; j < config.numExpertPerToken; j++) {
        weightsBuf[i * config.numExpertPerToken + j] = tokValDist(gen);
      }
    }
  }

  void RandomInitializeScales() {
    if (!scalesBuf) return;
    EpDispatchCombineConfig& config = handle.config;
    uniform_real_distribution<> tokValDist(0, 1);
    for (int i = 0; i < numToken; i++) {
      for (int j = 0; j < config.scaleDim; j++) {
        if (config.scaleTypeSize == sizeof(float)) {
          reinterpret_cast<float*>(scalesBuf)[i * config.scaleDim + j] = tokValDist(gen);
        } else {
          reinterpret_cast<__hip_fp8_e4m3_fnuz*>(scalesBuf)[i * config.scaleDim + j] =
              tokValDist(gen);
        }
      }
    }
  }

  void RandomInitializeToken() {
    EpDispatchCombineConfig& config = handle.config;
    int maxTokenSize = config.MaxNumTokensToRecvPerRank() * config.hiddenDim * sizeof(T);
    HIP_RUNTIME_CHECK(hipMemset(inpTokBuf, 0, maxTokenSize));

    if (IsNoDataRank(config)) {
      return;
    }

    HIP_RUNTIME_CHECK(hipMemset(inpTokBufCpu, 0, maxTokenSize));

    int inpTokEleNum = config.maxNumInpTokenPerRank * config.hiddenDim;
    size_t inpTokSize = inpTokEleNum * sizeof(T);
    uniform_real_distribution<> tokValDist(0.01, 1);
    for (int i = 0; i < inpTokEleNum; i++) {
      reinterpret_cast<T*>(inpTokBufCpu)[i] = tokValDist(gen);
      // reinterpret_cast<T*>(inpTokBufCpu)[i] = config.rank + 1;
    }
    HIP_RUNTIME_CHECK(hipMemcpy(inpTokBuf, inpTokBufCpu, inpTokSize, hipMemcpyHostToDevice));
    HIP_RUNTIME_CHECK(hipMemset(inpTokBufCpu, 0, maxTokenSize));
  }

  void PrintDispatch() {
    EpDispatchCombineConfig& config = handle.config;
    stringstream ss;
    for (int i = 0; i < handle.curRankNumToken; i++) {
      ss << "  Token " << i << " dispatch to ";
      for (int j = 0; j < config.numExpertPerToken; j++) {
        ss << tokenIndices[i * config.numExpertPerToken + j] << " ";
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
  uint8_t* scalesBuf{nullptr};
  index_t* tokenIndices{nullptr};
  int numToken{-1};
  EpDispatchCombineHandle& handle;
  RunConfig runConfig;
};

template <typename T>
void RunDispatchCombineTest(EpDispatchCombineTestConfig testConfig) {
  EpDispatchCombineHandle handle(testConfig.config);
  EpDispatchCombineTestCase<T> testCase(handle, testConfig.runConfig);
  testCase.Run();
}

// A simple MoE-EP dispatch kernel example, assume dp rank is equal to ep rank
void EpDispatchWithPutMemAPI(EpDispatchCombineTestConfig testConfig) {
  int status;

  // Initialize shmem
  MPI_Init(NULL, NULL);
  int localRank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &localRank);
  HIP_RUNTIME_CHECK(hipSetDevice(localRank % 8));

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

  ShmemFinalize();
}

EpDispatchCombineTestConfig ParseArguments(int argc, char* argv[]) {
  EpDispatchCombineTestConfig testConfig;

  static struct option long_options[] = {{"help", no_argument, NULL, 'h'},
                                         {"cmd", required_argument, NULL, 0},
                                         {"data_type", required_argument, NULL, 0},
                                         {"hdim", optional_argument, NULL, 'd'},
                                         {"scale_dim", optional_argument, NULL, 's'},
                                         {"scale_type", optional_argument, NULL, 0},
                                         {"max_tokens", optional_argument, NULL, 'm'},
                                         {"max_token_type_size", optional_argument, NULL, 0},
                                         {"expert_per_rank", optional_argument, NULL, 'r'},
                                         {"expert_per_token", optional_argument, NULL, 't'},
                                         {"warp_per_blk", optional_argument, NULL, 'w'},
                                         {"block_num", optional_argument, NULL, 'b'},
                                         {"num", optional_argument, NULL, 'n'},
                                         {"kernel_type", optional_argument, NULL, 'k'},
                                         {0, 0, 0, 0}};
  int option_index = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "d::m::r::t::w::b::n::s::k::h", long_options,
                            &option_index)) != -1) {
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
            printf("Unknown data type: %s, must be 'fp32', 'bf16' or 'fp8'\n", optarg);
            assert(false);
          }
        } else if (strcmp(long_options[option_index].name, "scale_type") == 0) {
          if (strcmp(optarg, "fp32") == 0) {
            testConfig.config.scaleTypeSize = 4;
          } else if (strcmp(optarg, "fp8") == 0) {
            testConfig.config.scaleTypeSize = 1;
          } else {
            printf("Unknown scale type: %s, must be 'fp8' or 'fp32'\n", optarg);
            assert(false);
          }
        } else if (strcmp(long_options[option_index].name, "max_token_type_size") == 0) {
          testConfig.config.maxTokenTypeSize = std::stoi(optarg);
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
      case 's':
        testConfig.config.scaleDim = std::stoi(optarg);
        break;
      case 'n':
        testConfig.runConfig.repeat = std::stoi(optarg);
        break;
      case 'k':
        if (strcmp(optarg, "intra") == 0) {
          testConfig.runConfig.kernelType = KernelType::IntraNode;
        } else if (strcmp(optarg, "inter") == 0) {
          testConfig.runConfig.kernelType = KernelType::InterNode;
        } else {
          printf("Unknown kernel type: %s, must be 'inter' or 'intra'\n", optarg);
          assert(false);
        }
        break;
      case 'h':
        printf("This is help message\n");
        break;
      case '?':
        fprintf(stderr, "Unknown option or missing argument\n");
        return {};
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

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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/ops/dispatch_combine/launch.hpp"
#include "mori/shmem/shmem_api.hpp"

using mori::moe::EpDispatchCombineConfig;
using mori::moe::index_t;
using mori::moe::KernelRegistry;
using mori::moe::LaunchLocalExpertCount;
using mori::shmem::mori_shmem_init_attr_t;
using mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID;
using mori::shmem::mori_shmem_uniqueid_t;
using mori::shmem::ShmemFinalize;
using mori::shmem::ShmemGetUniqueId;
using mori::shmem::ShmemInitAttr;
using mori::shmem::ShmemSetAttrUniqueIdArgs;

namespace {

void CheckHip(hipError_t err, const char* expr) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(expr) + " failed: " + hipGetErrorString(err));
  }
}

void CheckShmem(int err, const char* expr) {
  if (err != 0) {
    throw std::runtime_error(std::string(expr) + " failed");
  }
}

void InitShmemSingleton() {
  mori_shmem_uniqueid_t uid{};
  mori_shmem_init_attr_t attr{};
  CheckShmem(ShmemGetUniqueId(&uid), "ShmemGetUniqueId");
  CheckShmem(ShmemSetAttrUniqueIdArgs(/*rank=*/0, /*nranks=*/1, &uid, &attr),
             "ShmemSetAttrUniqueIdArgs");
  CheckShmem(ShmemInitAttr(MORI_SHMEM_INIT_WITH_UNIQUEID, &attr), "ShmemInitAttr");
}

std::vector<int> RunCase(const EpDispatchCombineConfig& config, const std::vector<index_t>& indices,
                         index_t totalRecvTokenNum, hipStream_t stream) {
  index_t* d_indices = nullptr;
  index_t* d_total = nullptr;
  int* d_counts = nullptr;

  const size_t indicesBytes = indices.size() * sizeof(index_t);
  const size_t countsBytes = static_cast<size_t>(config.numExpertPerRank) * sizeof(int);

  CheckHip(hipMalloc(&d_indices, indicesBytes), "hipMalloc(d_indices)");
  CheckHip(hipMalloc(&d_total, sizeof(index_t)), "hipMalloc(d_total)");
  CheckHip(hipMalloc(&d_counts, countsBytes), "hipMalloc(d_counts)");

  CheckHip(hipMemcpyAsync(d_indices, indices.data(), indicesBytes, hipMemcpyHostToDevice, stream),
           "hipMemcpyAsync(indices)");
  CheckHip(
      hipMemcpyAsync(d_total, &totalRecvTokenNum, sizeof(index_t), hipMemcpyHostToDevice, stream),
      "hipMemcpyAsync(totalRecvTokenNum)");

  LaunchLocalExpertCount(config, d_indices, d_total, d_counts, /*block_num=*/4,
                         /*warp_per_block=*/2, stream);
  CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize");

  std::vector<int> counts(config.numExpertPerRank, -1);
  CheckHip(hipMemcpy(counts.data(), d_counts, countsBytes, hipMemcpyDeviceToHost),
           "hipMemcpy(counts)");

  CheckHip(hipFree(d_counts), "hipFree(d_counts)");
  CheckHip(hipFree(d_total), "hipFree(d_total)");
  CheckHip(hipFree(d_indices), "hipFree(d_indices)");
  return counts;
}

void ExpectEqual(const std::vector<int>& actual, const std::vector<int>& expected,
                 const char* label) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(label) + " size mismatch");
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      throw std::runtime_error(std::string(label) + " mismatch at expert " + std::to_string(i) +
                               ": got " + std::to_string(actual[i]) + ", expected " +
                               std::to_string(expected[i]));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  CheckHip(hipSetDevice(0), "hipSetDevice");
  InitShmemSingleton();

  const std::string baseDir = (argc > 1) ? argv[1] : "lib";
  KernelRegistry::Instance().AutoLoad(baseDir);

  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "hipStreamCreate");

  EpDispatchCombineConfig config{};
  config.rank = 1;
  config.worldSize = 4;
  config.numExpertPerRank = 3;
  config.numExpertPerToken = 2;
  config.warpNumPerBlock = 2;
  config.blockNum = 4;

  const std::vector<index_t> indices = {
      3, 5, 4, 1, 5, 5, 0, 3, 4, 4,
  };
  ExpectEqual(RunCase(config, indices, /*totalRecvTokenNum=*/4, stream), {2, 1, 3},
              "non_zero_case");
  ExpectEqual(RunCase(config, indices, /*totalRecvTokenNum=*/0, stream), {0, 0, 0}, "zero_case");

  CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
  CheckShmem(ShmemFinalize(), "ShmemFinalize");
  std::printf("\n=== local_expert_count test PASSED ===\n");
  return 0;
}

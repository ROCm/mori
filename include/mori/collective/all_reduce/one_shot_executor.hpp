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
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstring>

#include "mori/application/utils/check.hpp"
#include "mori/collective/all_reduce/allreduce_config.hpp"
#include "mori/collective/all_reduce/allreduce_executor.hpp"
#include "mori/collective/device/one_shot_allreduce.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

template <typename T>
class OneShotAllReduceExecutor : public AllReduceExecutor<T> {
 public:
  OneShotAllReduceExecutor(int num_ranks, int rank,
                           const AllReduceConfig& config = AllReduceConfig());
  ~OneShotAllReduceExecutor() override = default;

  int Execute(T* input, T* output, size_t count, hipStream_t stream) override;

 private:
  int numRanks;
  int rank;
  AllReduceConfig config;
};

template <typename T>
OneShotAllReduceExecutor<T>::OneShotAllReduceExecutor(int num_ranks, int rank,
                                                      const AllReduceConfig& config)
    : numRanks(num_ranks), rank(rank), config(config) {}

template <typename T>
int OneShotAllReduceExecutor<T>::Execute(T* input, T* output, size_t count, hipStream_t stream) {
  if (count == 0) {
    return 0;
  }

  const size_t totalBytes = count * sizeof(T);
  const size_t scratchBytes = totalBytes * static_cast<size_t>(numRanks);

  application::SymmMemObjPtr srcMemObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(input), totalBytes);
  if (!srcMemObj.IsValid()) {
    return -1;
  }

  application::SymmMemObjPtr dstMemObj;
  if (input == output) {
    dstMemObj = srcMemObj;
  } else {
    dstMemObj = shmem::ShmemSymmetricRegister(static_cast<void*>(output), totalBytes);
    if (!dstMemObj.IsValid()) {
      return -1;
    }
  }

  void* scratchBuffer = shmem::ShmemMalloc(scratchBytes);
  if (scratchBuffer == nullptr) {
    return -1;
  }
  application::SymmMemObjPtr scratchMemObj = shmem::ShmemQueryMemObjPtr(scratchBuffer);
  if (!scratchMemObj.IsValid()) {
    shmem::ShmemFree(scratchBuffer);
    return -1;
  }

  const size_t flagsBytes = static_cast<size_t>(numRanks) * sizeof(uint64_t);
  void* flagsBuffer = shmem::ShmemMalloc(flagsBytes);
  if (flagsBuffer == nullptr) {
    shmem::ShmemFree(scratchBuffer);
    return -1;
  }
  std::memset(flagsBuffer, 0, flagsBytes);
  application::SymmMemObjPtr flagsMemObj = shmem::ShmemQueryMemObjPtr(flagsBuffer);
  if (!flagsMemObj.IsValid()) {
    shmem::ShmemFree(flagsBuffer);
    shmem::ShmemFree(scratchBuffer);
    return -1;
  }

  const int threadsPerBlock = config.threadsPerBlock > 0 ? config.threadsPerBlock : 1;
  int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
  if (blocks <= 0) {
    blocks = 1;
  }
  if (config.maxBlocks > 0 && blocks > config.maxBlocks) {
    blocks = config.maxBlocks;
  }

  OneShotAllReduceKernel<T><<<blocks, threadsPerBlock, 0, stream>>>(
      rank, numRanks, srcMemObj, dstMemObj, scratchMemObj, flagsMemObj, count);
  hipError_t kernelStatus = hipGetLastError();
  shmem::ShmemFree(scratchBuffer);
  shmem::ShmemFree(flagsBuffer);
  HIP_RUNTIME_CHECK(kernelStatus);

  return 0;
}

}  // namespace collective
}  // namespace mori

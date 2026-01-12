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
#include "mori/collective/core/allreduce_config.hpp"
#include "mori/collective/core/allreduce_executor.hpp"
#include "mori/collective/inter_node/kernels/one_shot_kernel.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

template <typename T>
class OneShotAllReduceExecutor : public AllReduceExecutor<T> {
 public:
  OneShotAllReduceExecutor(int num_ranks, int rank,
                           const AllReduceConfig& config = AllReduceConfig());
  ~OneShotAllReduceExecutor() override;

  int Execute(T* input, T* output, size_t count, hipStream_t stream) override;

  int RegisterBuffers(T* input, T* output, size_t maxCount);
  void DeregisterBuffers();

 private:
  int InitializeScratchBuffers(size_t maxBytes, hipStream_t stream);
  void FinalizeScratchBuffers();

  int numRanks;
  int rank;
  AllReduceConfig config;

  void* scratchBuffer{nullptr};
  void* flagsBuffer{nullptr};
  application::SymmMemObjPtr scratchMemObj;
  application::SymmMemObjPtr flagsMemObj;
  size_t currentMaxBytes{0};

  application::SymmMemObjPtr registeredInputMemObj;
  application::SymmMemObjPtr registeredOutputMemObj;
  T* registeredInput{nullptr};
  T* registeredOutput{nullptr};
  size_t registeredMaxCount{0};

  uint64_t epoch{0};
};

template <typename T>
OneShotAllReduceExecutor<T>::OneShotAllReduceExecutor(int num_ranks, int rank,
                                                      const AllReduceConfig& config)
    : numRanks(num_ranks), rank(rank), config(config) {}

template <typename T>
OneShotAllReduceExecutor<T>::~OneShotAllReduceExecutor() {
  DeregisterBuffers();
  FinalizeScratchBuffers();
}

template <typename T>
int OneShotAllReduceExecutor<T>::InitializeScratchBuffers(size_t maxBytes, hipStream_t stream) {
  if (maxBytes <= currentMaxBytes && scratchBuffer != nullptr && flagsBuffer != nullptr) {
    return 0;
  }

  FinalizeScratchBuffers();

  const size_t scratchBytes = maxBytes * static_cast<size_t>(numRanks);
  scratchBuffer = shmem::ShmemMalloc(scratchBytes);
  if (scratchBuffer == nullptr) {
    return -1;
  }
  scratchMemObj = shmem::ShmemQueryMemObjPtr(scratchBuffer);
  if (!scratchMemObj.IsValid()) {
    shmem::ShmemFree(scratchBuffer);
    scratchBuffer = nullptr;
    return -1;
  }

  const size_t flagsBytes = static_cast<size_t>(numRanks) * sizeof(uint64_t);
  flagsBuffer = shmem::ShmemMalloc(flagsBytes);
  if (flagsBuffer == nullptr) {
    shmem::ShmemFree(scratchBuffer);
    scratchBuffer = nullptr;
    return -1;
  }
  flagsMemObj = shmem::ShmemQueryMemObjPtr(flagsBuffer);
  if (!flagsMemObj.IsValid()) {
    shmem::ShmemFree(flagsBuffer);
    shmem::ShmemFree(scratchBuffer);
    flagsBuffer = nullptr;
    scratchBuffer = nullptr;
    return -1;
  }

  hipError_t err = hipMemsetAsync(flagsBuffer, 0, flagsBytes, stream);
  if (err != hipSuccess) {
    shmem::ShmemFree(flagsBuffer);
    shmem::ShmemFree(scratchBuffer);
    flagsBuffer = nullptr;
    scratchBuffer = nullptr;
    return -1;
  }
  hipStreamSynchronize(stream);

  currentMaxBytes = maxBytes;
  return 0;
}

template <typename T>
void OneShotAllReduceExecutor<T>::FinalizeScratchBuffers() {
  hipDeviceSynchronize();

  if (flagsBuffer) {
    shmem::ShmemFree(flagsBuffer);
    flagsBuffer = nullptr;
  }
  if (scratchBuffer) {
    shmem::ShmemFree(scratchBuffer);
    scratchBuffer = nullptr;
  }
  currentMaxBytes = 0;
  flagsMemObj = {};
  scratchMemObj = {};
}

template <typename T>
int OneShotAllReduceExecutor<T>::RegisterBuffers(T* input, T* output, size_t maxCount) {
  DeregisterBuffers();

  const size_t totalBytes = maxCount * sizeof(T);
  registeredInputMemObj = shmem::ShmemSymmetricRegister(static_cast<void*>(input), totalBytes);
  if (!registeredInputMemObj.IsValid()) {
    return -1;
  }

  if (input == output) {
    registeredOutputMemObj = registeredInputMemObj;
  } else {
    registeredOutputMemObj = shmem::ShmemSymmetricRegister(static_cast<void*>(output), totalBytes);
    if (!registeredOutputMemObj.IsValid()) {
      shmem::ShmemSymmetricDeregister(input, totalBytes);
      registeredInputMemObj = {};
      return -1;
    }
  }

  registeredInput = input;
  registeredOutput = output;
  registeredMaxCount = maxCount;
  return 0;
}

template <typename T>
void OneShotAllReduceExecutor<T>::DeregisterBuffers() {
  if (registeredInput && registeredInputMemObj.IsValid()) {
    shmem::ShmemSymmetricDeregister(registeredInput, registeredMaxCount * sizeof(T));
  }
  if (registeredOutput && registeredOutput != registeredInput && registeredOutputMemObj.IsValid()) {
    shmem::ShmemSymmetricDeregister(registeredOutput, registeredMaxCount * sizeof(T));
  }
  registeredInputMemObj = {};
  registeredOutputMemObj = {};
  registeredInput = nullptr;
  registeredOutput = nullptr;
  registeredMaxCount = 0;
}

template <typename T>
int OneShotAllReduceExecutor<T>::Execute(T* input, T* output, size_t count, hipStream_t stream) {
  if (count == 0) {
    return 0;
  }

  const size_t totalBytes = count * sizeof(T);

  if (InitializeScratchBuffers(totalBytes, stream) != 0) {
    return -1;
  }

  application::SymmMemObjPtr srcMemObj;
  application::SymmMemObjPtr dstMemObj;
  bool needTempInputReg = false;
  bool needTempOutputReg = false;

  if (registeredInput == input && registeredInputMemObj.IsValid() && count <= registeredMaxCount) {
    srcMemObj = registeredInputMemObj;
  } else {
    srcMemObj = shmem::ShmemSymmetricRegister(static_cast<void*>(input), totalBytes);
    if (!srcMemObj.IsValid()) {
      return -1;
    }
    needTempInputReg = true;
  }

  if (input == output) {
    dstMemObj = srcMemObj;
  } else if (registeredOutput == output && registeredOutputMemObj.IsValid() &&
             count <= registeredMaxCount) {
    dstMemObj = registeredOutputMemObj;
  } else {
    dstMemObj = shmem::ShmemSymmetricRegister(static_cast<void*>(output), totalBytes);
    if (!dstMemObj.IsValid()) {
      if (needTempInputReg) {
        shmem::ShmemSymmetricDeregister(input, totalBytes);
      }
      return -1;
    }
    needTempOutputReg = true;
  }

  epoch++;

  const int threadsPerBlock = config.threadsPerBlock > 0 ? config.threadsPerBlock : 256;

  OneShotAllReduceKernelSingleBlock<T><<<1, threadsPerBlock, 0, stream>>>(
      rank, numRanks, srcMemObj, dstMemObj, scratchMemObj, flagsMemObj, count, epoch);

  hipError_t kernelStatus = hipGetLastError();
  if (kernelStatus != hipSuccess) {
    if (needTempInputReg || needTempOutputReg) {
      hipStreamSynchronize(stream);
      if (needTempInputReg) {
        shmem::ShmemSymmetricDeregister(input, totalBytes);
      }
      if (needTempOutputReg && input != output) {
        shmem::ShmemSymmetricDeregister(output, totalBytes);
      }
    }
    return -1;
  }

  if (needTempInputReg || needTempOutputReg) {
    hipStreamSynchronize(stream);
    if (needTempInputReg) {
      shmem::ShmemSymmetricDeregister(input, totalBytes);
    }
    if (needTempOutputReg && input != output) {
      shmem::ShmemSymmetricDeregister(output, totalBytes);
    }
  }

  return 0;
}

}  // namespace collective
}  // namespace mori

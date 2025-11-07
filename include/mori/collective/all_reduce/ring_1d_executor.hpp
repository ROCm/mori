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

#include <cstring>

#include "mori/application/utils/check.hpp"
#include "mori/collective/all_reduce/allreduce_config.hpp"
#include "mori/collective/all_reduce/allreduce_executor.hpp"
#include "mori/collective/device/all_gather.hpp"
#include "mori/collective/device/reduce_scatter.hpp"
#include "mori/collective/topology_detector.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

/**
 * Ring1DAllReduceExecutor: Simple 1D Ring All-Reduce for inter-node
 *
 * Implements classic Ring AllReduce:
 * Phase 1: Reduce-Scatter (N-1 rounds)
 * Phase 2: AllGather (N-1 rounds)
 *
 * Optimal for small to medium data sizes
 */
template <typename T>
class Ring1DAllReduceExecutor : public AllReduceExecutor<T> {
 public:
  /**
   * Initialize 1D Ring executor
   *
   * @param num_ranks Total number of ranks
   * @param rank Current rank
   * @param config Configuration parameters
   */
  Ring1DAllReduceExecutor(int num_ranks, int rank,
                          const AllReduceConfig& config = AllReduceConfig());

  ~Ring1DAllReduceExecutor() override = default;

  int Execute(T* input, T* output, size_t count, hipStream_t stream) override;

 private:
  int numRanks;
  int rank;
  AllReduceConfig config;

  // Phase 1: Reduce-Scatter
  int ReduceScatter(T* input, T* output, size_t total_count, hipStream_t stream);

  // Phase 2: AllGather
  int AllGather(T* input, T* output, size_t total_count, hipStream_t stream);
};

template <typename T>
Ring1DAllReduceExecutor<T>::Ring1DAllReduceExecutor(int num_ranks, int rank,
                                                    const AllReduceConfig& config)
    : numRanks(num_ranks), rank(rank), config(config) {}

template <typename T>
int Ring1DAllReduceExecutor<T>::Execute(T* input, T* output, size_t count, hipStream_t stream) {
  int status = ReduceScatter(input, output, count, stream);
  if (status != 0) {
    return status;
  }

  status = AllGather(input, output, count, stream);
  if (status != 0) {
    return status;
  }

  return status;
}

template <typename T>
int Ring1DAllReduceExecutor<T>::ReduceScatter(T* input, T* output, size_t total_count,
                                              hipStream_t stream) {
  int myPe = TopologyDetector::GetMyPe();
  int npes = TopologyDetector::GetNPes();
  size_t dtype_size = sizeof(T);
  void* tempOutput = nullptr;
  application::SymmMemObjPtr recvMemObj;

  application::SymmMemObjPtr memObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(input), total_count * dtype_size);
  if (input != output) {
    recvMemObj =
        shmem::ShmemSymmetricRegister(static_cast<void*>(output), total_count * dtype_size);
  } else {
    tempOutput = shmem::ShmemMalloc(total_count * dtype_size);
    recvMemObj = shmem::ShmemQueryMemObjPtr(tempOutput);
  }

  assert(recvMemObj.IsValid());

  int flagsSize = npes * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (flags == nullptr) {
    return -1;
  }
  std::memset(flags, 0, flagsSize);
  application::SymmMemObjPtr flagsObj = shmem::ShmemQueryMemObjPtr(flags);
  ReduceScatterRingKernel<T><<<1, 1, 0, stream>>>(myPe, npes, memObj, recvMemObj, flagsObj);

  shmem::ShmemFree(flags);

  if (input != output) {
    HIP_RUNTIME_CHECK(
        hipMemcpyAsync(output, input, total_count * dtype_size, hipMemcpyDeviceToDevice, stream));
  } else
    shmem::ShmemFree(tempOutput);

  return 0;
}

template <typename T>
int Ring1DAllReduceExecutor<T>::AllGather(T* input, T* output, size_t total_count,
                                          hipStream_t stream) {
  int myPe = TopologyDetector::GetMyPe();
  int npes = TopologyDetector::GetNPes();
  size_t dtype_size = sizeof(T);

  application::SymmMemObjPtr memObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(input), total_count * dtype_size);

  int flagsSize = npes * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (flags == nullptr) {
    return -1;
  }
  std::memset(flags, 0, flagsSize);
  application::SymmMemObjPtr flagsObj = shmem::ShmemQueryMemObjPtr(flags);

  AllGatherRingKernel<T><<<1, config.threadsPerBlock, 0, stream>>>(myPe, npes, memObj, flagsObj);

  if (input != output) {
    HIP_RUNTIME_CHECK(
        hipMemcpyAsync(output, input, total_count * dtype_size, hipMemcpyDeviceToDevice, stream));
  }

  shmem::ShmemFree(flags);
  return 0;
}

}  // namespace collective
}  // namespace mori

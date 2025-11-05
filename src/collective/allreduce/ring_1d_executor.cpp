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
#include "mori/collective/all_reduce/ring_1d_executor.hpp"

#include "mori/collective/device/all_gather.hpp"
#include "mori/collective/device/reduce_scatter.hpp"

namespace mori {
namespace collective {

Ring1DAllReduceExecutor::Ring1DAllReduceExecutor(int num_ranks, int rank,
                                                 const AllReduceConfig& config)
    : numRanks(num_ranks), rank(rank), config(config) {}

Ring1DAllReduceExecutor::~Ring1DAllReduceExecutor() {}

int Ring1DAllReduceExecutor::Execute(void* input, void* output, size_t count, size_t dtype_size,
                                     hipStream_t stream) {
  // fake input and output
  int status = ReduceScatter(input, output, count, dtype_size, stream);
  if (status != 0) {
    return status;
  }
  memset(output, 0, count * dtype_size);
  status = AllGather(input, output, count, dtype_size, stream);
  if (status != 0) {
    return status;
  }
  return status;
}

int Ring1DAllReduceExecutor::ReduceScatter(void* input, void* output_chunk, size_t total_count,
                                           size_t dtype_size, hipStream_t stream) {
  int myPe = TopologyDetector::GetMyPe();
  int npes = TopologyDetector::GetNPes();
  application::SymmMemObjPtr memObj =
      shmem::ShmemSymmetricRegister(input, total_count * dtype_size);
  application::SymmMemObjPtr recvMemObj =
      shmem::ShmemSymmetricRegister(output_chunk, total_count * dtype_size);

  int flagsSize = npes * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (flags == nullptr) {
    return -1;
  }
  memset(flags, 0, flagsSize);
  application::SymmMemObjPtr flagsObj = shmem::ShmemQueryMemObjPtr(flags);
  ReduceScatterRingKernel<<<1, 1, 0, stream>>>(myPe, npes, memObj, recvMemObj, flagsObj);

  shmem::ShmemFree(flags);

  return 0;
}

int Ring1DAllReduceExecutor::AllGather(void* input_chunk, void* output, size_t total_count,
                                       size_t dtype_size, hipStream_t stream) {
  int myPe = TopologyDetector::GetMyPe();
  int npes = TopologyDetector::GetNPes();
  application::SymmMemObjPtr memObj =
      shmem::ShmemSymmetricRegister(input_chunk, total_count * dtype_size);
  application::SymmMemObjPtr recvMemObj =
      shmem::ShmemSymmetricRegister(output, total_count * dtype_size);

  int flagsSize = npes * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (flags == nullptr) {
    return -1;
  }
  memset(flags, 0, flagsSize);
  application::SymmMemObjPtr flagsObj = shmem::ShmemQueryMemObjPtr(flags);
  AllGatherRingKernel<<<1, 1, 0, stream>>>(myPe, npes, memObj, recvMemObj, flagsObj);

  shmem::ShmemFree(flags);
  return 0;
}
}  // namespace collective
}  // namespace mori

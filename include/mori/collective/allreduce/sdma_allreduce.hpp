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
#include "twoshot_sdma_kernel.hpp"
#include <mpi.h>

namespace mori {
namespace collective {

template <typename T>
double Allreduce_sdma(T* input, T* output, size_t total_count,
                      hipStream_t stream) {
  int myPe = shmem::ShmemMyPe();
  int npes = shmem::ShmemNPes();
  size_t dtype_size = sizeof(T);

  // Register input buffer (elementCount elements per rank)
  application::SymmMemObjPtr inPutBuffObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(input), total_count * dtype_size);

  // Compute the padded shard size to determine transit buffer size.
  // This mirrors the kernel's elementCountPerRank calculation.
  // We don't know pack_size here at compile time for all T, so over-allocate
  // conservatively: npes * (total_count / npes + 16) * dtype_size.
  size_t ecpr_approx = (total_count / npes + 16);  // generous upper bound
  size_t transit_size = npes * ecpr_approx * dtype_size;

  // Allocate transit buffer (dstMemObj) for intermediate gather + reduce + allgather
  void* transit = shmem::ShmemMalloc(transit_size);
  if (transit == nullptr) {
    return -1;
  }
  application::SymmMemObjPtr transitObj =
      shmem::ShmemSymmetricRegister(transit, transit_size);

  // Allocate flags
  int flagsSize = npes * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (flags == nullptr) {
    return -1;
  }
  memset(flags, 0, flagsSize);
  application::SymmMemObjPtr flagsObj = shmem::ShmemQueryMemObjPtr(flags);

  assert(inPutBuffObj.IsValid());
  assert(transitObj.IsValid());
  assert(flagsObj.IsValid());

  double start = MPI_Wtime();
  TwoShotAllReduceSdmaKernel<T><<<1, 512, 0, stream>>>(
      myPe, npes, input, inPutBuffObj, transitObj, flagsObj, total_count);

  // Synchronize GPU to ensure kernel completion
  hipError_t sync_err;
  if (stream != nullptr) {
    sync_err = hipStreamSynchronize(stream);
  } else {
    sync_err = hipDeviceSynchronize();
  }

  if (sync_err != hipSuccess) {
    fprintf(stderr, "PE %d: Failed to synchronize: %s\n", myPe, hipGetErrorString(sync_err));
    return -1.0;
  }

  double end = MPI_Wtime();

  // Copy result from transit buffer to user output (first total_count elements)
  hipError_t copy_err = hipMemcpy(output, transit, total_count * dtype_size,
                                  hipMemcpyDeviceToDevice);
  if (copy_err != hipSuccess) {
    fprintf(stderr, "PE %d: Failed to copy result: %s\n", myPe, hipGetErrorString(copy_err));
    return -1.0;
  }

  return end - start;
}

}  // namespace collective
}  // namespace mori

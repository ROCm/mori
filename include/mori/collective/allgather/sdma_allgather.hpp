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
#include "oneshot_sdma_kernel.hpp"

namespace mori {
namespace collective {

template <typename T>
int AllGather_sdma(T* input, T* output, size_t total_count,
                                          hipStream_t stream) {
  int myPe =  shmem::ShmemMyPe();
  int npes =  shmem::ShmemNPes();
  size_t dtype_size = sizeof(T);

  application::SymmMemObjPtr inPutBuffObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(input), total_count * dtype_size);

  application::SymmMemObjPtr outPutBuffObj =
      shmem::ShmemSymmetricRegister(static_cast<void*>(input), total_count * dtype_size * npes);

  int flagsSize = npes * sizeof(uint64_t);
  void* flags = shmem::ShmemMalloc(flagsSize);
  if (flags == nullptr) {
    return -1;
  }
  memset(flags, 0, flagsSize);
  application::SymmMemObjPtr flagsObj = shmem::ShmemQueryMemObjPtr(flags);

  OneShotAllGatherSdmaKernel<T><<<1, 256, 0, stream>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsObj, total_count);

  shmem::ShmemFree(flags);
  return 0;
}
}
}

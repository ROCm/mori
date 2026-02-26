// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime.h"
#include "hip/hip_version.h"

/**
 * @brief Compatibility wrapper for hipMemImportFromShareableHandle
 *
 * ROCm 7.0.x expects (void*)&fd, 7.1.0+ changed to expect (void*)(uintptr_t)fd
 */
inline hipError_t hipMemImportFromShareableHandleCompat(hipMemGenericAllocationHandle_t* handle,
                                                        int fd,
                                                        hipMemAllocationHandleType handleType) {
#if HIP_VERSION >= 70100000
  // ROCm 7.1.0+: FD value as pointer
  return hipMemImportFromShareableHandle(
      handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)), handleType);
#else
  // ROCm 7.0.x or older: Pointer to FD
  return hipMemImportFromShareableHandle(handle, (void*)&fd, handleType);
#endif
}

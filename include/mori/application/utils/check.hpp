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

#include <execinfo.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ROCm version detection (may already be defined by gpu.hpp)
#ifndef MORI_USE_AMDSMI
#if __has_include(<rocm-core/rocm_version.h>)
#include <rocm-core/rocm_version.h>
#elif __has_include(<rocm_version.h>)
#include <rocm_version.h>
#endif
#if defined(ROCM_VERSION_MAJOR) && \
    ((ROCM_VERSION_MAJOR > 10) || (ROCM_VERSION_MAJOR == 10 && ROCM_VERSION_MINOR >= 1))
#define MORI_USE_AMDSMI 1
#else
#define MORI_USE_AMDSMI 0
#endif
#endif

#if MORI_USE_AMDSMI
#include "amd_smi/amdsmi.h"
#else
#include "rocm_smi/rocm_smi.h"
#endif

namespace mori {
namespace application {

#define HIP_RUNTIME_CHECK(stmt)                                            \
  do {                                                                     \
    hipError_t result = (stmt);                                            \
    if (hipSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__, \
              hipGetErrorString(result));                                  \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define HIP_RUNTIME_CHECK_WITH_BACKTRACE(stmt)                             \
  do {                                                                     \
    hipError_t result = (stmt);                                            \
    if (hipSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__, \
              hipGetErrorString(result));                                  \
      void* array[20];                                                     \
      int size = backtrace(array, 20);                                     \
      backtrace_symbols_fd(array, size, STDERR_FILENO);                    \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define SYSCALL_RETURN_ZERO(stmt)                                                               \
  do {                                                                                          \
    auto _ret = (stmt);                                                                         \
    if (_ret != 0) {                                                                            \
      fprintf(stderr, "[%s:%d] syscall failed with %s\n", __FILE__, __LINE__, strerror(errno)); \
      exit(-1);                                                                                 \
    }                                                                                           \
  } while (0)

#define SYSCALL_RETURN_ZERO_IGNORE_ERROR(stmt, ignored)                                           \
  do {                                                                                            \
    auto _ret = (stmt);                                                                           \
    if (_ret != 0) {                                                                              \
      int err = errno;                                                                            \
      if (err != ignored) {                                                                       \
        fprintf(stderr, "[%s:%d] syscall failed with %s\n", __FILE__, __LINE__, strerror(errno)); \
        exit(-1);                                                                                 \
      }                                                                                           \
    }                                                                                             \
  } while (0)

#if MORI_USE_AMDSMI
#define ROCM_SMI_CHECK(stmt)                                                              \
  do {                                                                                    \
    amdsmi_status_t result = (stmt);                                                      \
    if (AMDSMI_STATUS_SUCCESS != result) {                                                \
      const char* msg;                                                                    \
      amdsmi_status_code_to_string(result, &msg);                                         \
      fprintf(stderr, "[%s:%d] amd smi failed with %s \n", __FILE__, __LINE__, msg);      \
      exit(-1);                                                                           \
    }                                                                                     \
  } while (0)
#else
#define ROCM_SMI_CHECK(stmt)                                                          \
  do {                                                                                \
    rsmi_status_t result = (stmt);                                                    \
    if (RSMI_STATUS_SUCCESS != result) {                                              \
      const char* msg;                                                                \
      rsmi_status_string(result, &msg);                                               \
      fprintf(stderr, "[%s:%d] rocm smi failed with %s \n", __FILE__, __LINE__, msg); \
      exit(-1);                                                                       \
    }                                                                                 \
  } while (0)
#endif

}  // namespace application
}  // namespace mori

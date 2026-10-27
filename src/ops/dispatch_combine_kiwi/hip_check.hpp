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

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace mori::kiwi::ep {
namespace internal {

inline void hip_check_abort(hipError_t error, const char* file, int line) {
  if (error == hipSuccess) {
    return;
  }
  std::fprintf(stderr, "HIP error at %s:%d: %s\n", file, line, hipGetErrorString(error));
  std::abort();
}

inline void hip_check_throw(hipError_t error, const char* file, int line) {
  if (error == hipSuccess) {
    return;
  }
  throw std::runtime_error(std::string("HIP error at ") + file + ":" + std::to_string(line) + ": " +
                           hipGetErrorString(error));
}

}  // namespace internal
}  // namespace mori::kiwi::ep

#ifdef HIP_CHECK
#undef HIP_CHECK
#endif
#define HIP_CHECK(call) ::mori::kiwi::ep::internal::hip_check_abort((call), __FILE__, __LINE__)

#define HIP_CHECK_THROW(call) \
  ::mori::kiwi::ep::internal::hip_check_throw((call), __FILE__, __LINE__)

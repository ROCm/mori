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

namespace mori::kiwi::ep {

[[noreturn]] __host__ __device__ inline void kiwi_ep_assert_fail(const char* message) {
#if defined(__HIP_DEVICE_COMPILE__)
  printf("kiwi EP device assert failed: %s\n", message);
  __builtin_trap();
#else
  std::fprintf(stderr, "kiwi EP assert failed: %s\n", message);
  std::abort();
#endif
}

}  // namespace mori::kiwi::ep

#define KIWI_EP_ASSERT(condition, message) \
  do {                                     \
    if (!(condition)) {                    \
      kiwi_ep_assert_fail(message);        \
    }                                      \
  } while (0)

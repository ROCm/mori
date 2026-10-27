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
/**
 * @file error.hpp
 * @brief Error handling for queue
 *
 * Push/pop operations return bool (true=success, false=full/empty).
 * Fatal errors throw std::runtime_error.
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace mori::kiwi::queue {

/**
 * @brief Exception thrown for fatal queue errors
 *
 * Used for allocation failures, invalid parameters, and other
 * unrecoverable errors during queue construction/initialization.
 */
class queue_error : public std::runtime_error {
 public:
  explicit queue_error(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace mori::kiwi::queue

// Debug-only assert, usable in both host and device code.
// Compiles to nothing in release (NDEBUG defined).
#ifndef NDEBUG
#define KIWI_QUEUE_ASSERT(cond)    \
  do {                             \
    if (!(cond)) __builtin_trap(); \
  } while (0)
#else
#define KIWI_QUEUE_ASSERT(cond) ((void)0)
#endif

// Check HIP call and throw on failure
#define HIP_CHECK(call)                                                                           \
  do {                                                                                            \
    hipError_t err_ = (call);                                                                     \
    if (err_ != hipSuccess) {                                                                     \
      throw kiwi::queue::queue_error(std::string(#call) + " failed: " + hipGetErrorString(err_)); \
    }                                                                                             \
  } while (0)

// Check HIP call in noexcept context (destructors, cleanup paths).
// Asserts in debug, logs to stderr in release.
#define HIP_CHECK_NOEXCEPT(call)                                                                  \
  do {                                                                                            \
    hipError_t err_ = (call);                                                                     \
    if (err_ != hipSuccess) {                                                                     \
      fprintf(stderr, "queue: %s failed: %s (%s:%d)\n", #call, hipGetErrorString(err_), __FILE__, \
              __LINE__);                                                                          \
      assert(err_ == hipSuccess && #call " failed");                                              \
    }                                                                                             \
  } while (0)

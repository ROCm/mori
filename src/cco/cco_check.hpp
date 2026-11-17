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
//
// Error handling for the cco host control plane (libmori_cco only).
//
// The host API reports failures by return code and never ends the process, so it
// does not use HIP_RUNTIME_CHECK, which calls exit(). Errors are logged and also
// kept per thread, so the C ABI (cco_c_api.cpp) can return them as text.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>

#include "hip/hip_runtime_api.h"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace cco {
namespace detail {

// Per-thread error text with no destructor. exit() destroys C++ thread_locals
// before it runs atexit hooks and static destructors, and the C ABI may be
// called from those, so this storage must outlive them. Truncates.
template <size_t N>
struct TlsText {
  char text[N];
  size_t len;

  void Clear() {
    len = 0;
    text[0] = '\0';
  }
  void Append(const char* s, size_t n) {
    n = std::min(n, N - 1 - len);
    std::memcpy(text + len, s, n);
    len += n;
    text[len] = '\0';
  }
  void Append(const std::string& s) { Append(s.data(), s.size()); }
};
static_assert(std::is_trivially_destructible_v<TlsText<2>>, "TlsText must have no destructor");

// Appends to this thread's error text (the first error is usually the cause).
void RecordError(const std::string& msg);
// Returns this thread's error text and clears it.
std::string TakeErrors();

inline bool HipOk(hipError_t err, const char* expr, const char* file, int line) {
  if (err == hipSuccess) return true;
  const char* base = std::strrchr(file, '/');
  base = base ? base + 1 : file;
  MORI_SHMEM_ERROR("{}:{}: {} failed: {}", base, line, expr, hipGetErrorString(err));
  RecordError(fmt::format("{} failed: {}", expr, hipGetErrorString(err)));
  return false;
}

}  // namespace detail
}  // namespace cco
}  // namespace mori

// HIP check that logs and evaluates to false instead of exiting.
#define CCO_HIP_OK(stmt) ::mori::cco::detail::HipOk((stmt), #stmt, __FILE__, __LINE__)

// Returns -1 from the enclosing function when the HIP call fails.
#define CCO_HIP_TRY(stmt)             \
  do {                                \
    if (!CCO_HIP_OK(stmt)) return -1; \
  } while (0)

// Logs an error and keeps it for the C ABI's last-error text. Spelled with
// MORI_ERROR so a TU may redefine MORI_SHMEM_ERROR as this macro.
#define CCO_ERROR(...)                                            \
  do {                                                            \
    MORI_ERROR(::mori::modules::SHMEM, __VA_ARGS__);              \
    ::mori::cco::detail::RecordError(::fmt::format(__VA_ARGS__)); \
  } while (0)

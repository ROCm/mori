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

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

#include "mori/io/logging.hpp"

namespace mori {
namespace io {
namespace env {

inline const char* Get(const char* name) { return std::getenv(name); }

inline std::optional<std::string> GetString(const char* name) {
  const char* value = Get(name);
  if (value == nullptr || value[0] == '\0') return std::nullopt;
  return std::string(value);
}

namespace detail {

inline bool EqualsIgnoreCase(const char* lhs, const char* rhs) {
  while (*lhs != '\0' && *rhs != '\0') {
    if (std::tolower(static_cast<unsigned char>(*lhs)) !=
        std::tolower(static_cast<unsigned char>(*rhs))) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == '\0' && *rhs == '\0';
}

inline std::optional<bool> ParseBool(const char* raw) {
  if (std::strcmp(raw, "1") == 0 || EqualsIgnoreCase(raw, "true") || EqualsIgnoreCase(raw, "on") ||
      EqualsIgnoreCase(raw, "yes")) {
    return true;
  }
  if (std::strcmp(raw, "0") == 0 || EqualsIgnoreCase(raw, "false") ||
      EqualsIgnoreCase(raw, "off") || EqualsIgnoreCase(raw, "no")) {
    return false;
  }
  return std::nullopt;
}

inline std::optional<uint32_t> ParsePositiveU32(const char* raw) {
  errno = 0;
  char* end = nullptr;
  unsigned long parsed = std::strtoul(raw, &end, 10);
  if (end == raw || *end != '\0' || errno != 0 || parsed == 0 ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(parsed);
}

inline std::optional<int> ParsePositiveInt(const char* raw) {
  errno = 0;
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || errno != 0 || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return std::nullopt;
  }
  return static_cast<int>(parsed);
}

}  // namespace detail

template <typename Target, typename V>
inline void Override(const char* name, Target& target, std::optional<V> (*parse)(const char*)) {
  const char* raw = Get(name);
  if (raw == nullptr || raw[0] == '\0') return;
  if (auto value = parse(raw); value.has_value()) {
    target = *value;
  } else {
    MORI_IO_WARN("Ignore invalid env {}={}", name, raw);
  }
}

}  // namespace env
}  // namespace io
}  // namespace mori

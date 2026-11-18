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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>

// Strict parsing of configuration variables: an unset or blank variable leaves
// *out as it is, and a malformed one is an error for the caller to refuse to
// start on, rather than a default it silently falls back to (the lenient kind,
// for tuning knobs, is env_time.h).
namespace mori::umbp::env_parse {

inline std::optional<std::string> EnvString(const char* name) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') return std::nullopt;
  return std::string(value);
}

inline bool ParseSizeEnv(const char* name, size_t* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  try {
    *out = static_cast<size_t>(std::stoull(*raw));
    return true;
  } catch (const std::exception& exc) {
    *error = std::string("invalid ") + name + ": " + exc.what();
    return false;
  }
}

inline bool ParseUint16Env(const char* name, uint16_t* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  try {
    unsigned long value = std::stoul(*raw);
    if (value > 65535) {
      *error = std::string(name) + " must be <= 65535";
      return false;
    }
    *out = static_cast<uint16_t>(value);
    return true;
  } catch (const std::exception& exc) {
    *error = std::string("invalid ") + name + ": " + exc.what();
    return false;
  }
}

inline bool ParseIntEnv(const char* name, int* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  try {
    *out = std::stoi(*raw);
    return true;
  } catch (const std::exception& exc) {
    *error = std::string("invalid ") + name + ": " + exc.what();
    return false;
  }
}

inline bool ParseBoolEnv(const char* name, bool* out, std::string* error) {
  auto raw = EnvString(name);
  if (!raw.has_value()) return true;
  std::string value = *raw;
  for (char& ch : value) ch = static_cast<char>(std::tolower(ch));
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    *out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    *out = false;
    return true;
  }
  *error = std::string("invalid boolean ") + name + "=" + *raw;
  return false;
}

}  // namespace mori::umbp::env_parse

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

// Unified parsers for UMBP_* timing/count env vars.
//
// Semantics:
//   - Unset or empty env -> return `def` silently (no log).
//   - Invalid value (non-numeric, negative, or < min_allowed) -> return `def`
//     and emit one WARN per env name per process (keyed by the env name
//     string). Subsequent invalid reads of the same name are suppressed.
//   - Valid value -> return the parsed value converted to the requested unit.
//
// Caching policy:
//   These helpers do NOT cache the result across calls; they re-read the env
//   on every invocation. Callers that sit on hot paths MUST wrap the call in
//   a function-local `static const auto v = GetEnv...(...)` to freeze the
//   value at first use (C++11 magic statics, thread-safe).
//
// Thread / signal safety:
//   std::getenv and the underlying logger are NOT async-signal-safe. First
//   initialization of any `static const` wrapper must happen on a normal
//   thread, not inside a signal handler.

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

#include "umbp/common/log.h"

namespace mori::umbp {

namespace env_time_detail {

inline std::mutex& WarnMutex() {
  static std::mutex m;
  return m;
}

inline std::unordered_set<std::string>& WarnedNames() {
  static std::unordered_set<std::string> s;
  return s;
}

inline void WarnOnce(const char* name, const char* reason, const char* raw) {
  std::lock_guard<std::mutex> lock(WarnMutex());
  if (!WarnedNames().insert(name).second) return;
  UMBP_LOG_WARN("env %s: %s (value='%s'); using default", name, reason, raw ? raw : "");
}

// Parse an env value as signed integer. On any failure / out-of-range,
// returns false and fills `reason` with a short description; otherwise
// writes the parsed value to `*out` and returns true.
inline bool ParseSignedEnv(const char* raw, int64_t* out, const char** reason) {
  if (raw == nullptr || *raw == '\0') {
    *reason = "empty";
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const long long v = std::strtoll(raw, &end, 10);
  if (errno != 0) {
    *reason = "out of range";
    return false;
  }
  if (end == raw || (end && *end != '\0')) {
    *reason = "not a number";
    return false;
  }
  *out = static_cast<int64_t>(v);
  return true;
}

// Core resolver shared by the typed wrappers.
// Returns parsed value if valid and >= min_allowed, otherwise `def`.
inline int64_t ResolveEnvInt(const char* name, int64_t def, int64_t min_allowed) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return def;
  int64_t v = 0;
  const char* reason = nullptr;
  if (!ParseSignedEnv(raw, &v, &reason)) {
    WarnOnce(name, reason, raw);
    return def;
  }
  if (v < min_allowed) {
    WarnOnce(name, "below min_allowed", raw);
    return def;
  }
  return v;
}

}  // namespace env_time_detail

inline std::chrono::seconds GetEnvSeconds(const char* name, std::chrono::seconds def,
                                          int64_t min_allowed = 0) {
  const int64_t v = env_time_detail::ResolveEnvInt(name, def.count(), min_allowed);
  return std::chrono::seconds(v);
}

inline std::chrono::milliseconds GetEnvMilliseconds(const char* name, std::chrono::milliseconds def,
                                                    int64_t min_allowed = 0) {
  const int64_t v = env_time_detail::ResolveEnvInt(name, def.count(), min_allowed);
  return std::chrono::milliseconds(v);
}

inline std::chrono::microseconds GetEnvMicroseconds(const char* name, std::chrono::microseconds def,
                                                    int64_t min_allowed = 0) {
  const int64_t v = env_time_detail::ResolveEnvInt(name, def.count(), min_allowed);
  return std::chrono::microseconds(v);
}

inline uint32_t GetEnvUint32(const char* name, uint32_t def, uint32_t min_allowed = 0) {
  const int64_t v = env_time_detail::ResolveEnvInt(name, static_cast<int64_t>(def),
                                                   static_cast<int64_t>(min_allowed));
  // ResolveEnvInt already rejects negatives via min_allowed, but guard once
  // more so the narrowing conversion is well-defined for any future path.
  if (v < 0 || v > static_cast<int64_t>(UINT32_MAX)) {
    env_time_detail::WarnOnce(name, "outside uint32 range", std::getenv(name));
    return def;
  }
  return static_cast<uint32_t>(v);
}

// Test-only: clear the WARN-once registry. Not thread-safe vs readers.
inline void ResetEnvWarnStateForTesting() {
  std::lock_guard<std::mutex> lock(env_time_detail::WarnMutex());
  env_time_detail::WarnedNames().clear();
}

}  // namespace mori::umbp

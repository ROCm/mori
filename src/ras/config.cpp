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
#include "src/ras/config.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <set>
#include <sstream>

#include "mori/utils/env_utils.hpp"

namespace mori {
namespace ras {
namespace {

constexpr int kMaxCpu = 65535;

bool ParseUnsigned(const std::string& s, uint64_t max, uint64_t* out) {
  if (s.empty() || s[0] < '0' || s[0] > '9') return false;
  errno = 0;
  char* end = nullptr;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || *end != '\0' || v > max) return false;
  *out = v;
  return true;
}

std::string Trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return "";
  return s.substr(b, s.find_last_not_of(" \t") - b + 1);
}

// "0-3,8,10-11" in ascending order without duplicates. False on any malformed item.
bool ParseCpuList(const std::string& spec, std::vector<int>* out) {
  std::set<int> cpus;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = Trim(item);
    size_t dash = item.find('-');
    uint64_t lo, hi;
    if (dash == std::string::npos) {
      if (!ParseUnsigned(item, kMaxCpu, &lo)) return false;
      hi = lo;
    } else if (!ParseUnsigned(item.substr(0, dash), kMaxCpu, &lo) ||
               !ParseUnsigned(item.substr(dash + 1), kMaxCpu, &hi) || lo > hi) {
      return false;
    }
    for (uint64_t c = lo; c <= hi; ++c) cpus.insert(static_cast<int>(c));
  }
  if (cpus.empty()) return false;
  out->assign(cpus.begin(), cpus.end());
  return true;
}

}  // namespace

RasConfig ConfigFromEnv(const GetEnvFn& getenv) {
  RasConfig c;
  auto get = [&](const char* name) -> std::string {
    const char* v = getenv(name);
    return v == nullptr ? std::string() : std::string(v);
  };
  auto invalid = [&](const char* name, const std::string& value, const std::string& fallback) {
    c.warnings.push_back(std::string(name) + "=" + value + " is invalid; using " + fallback);
  };

  std::string v = get("MORI_RAS_ENABLE");
  if (!v.empty()) {
    std::optional<bool> b = env::detail::ParseBool(v.c_str());
    if (b.has_value()) {
      c.enable = *b;
    } else {
      invalid("MORI_RAS_ENABLE", v, "0");
    }
  }

  c.ifname = get("MORI_RAS_IFNAME");

  v = get("MORI_RAS_PORT");
  if (!v.empty()) {
    uint64_t port;
    if (!ParseUnsigned(v, 65535, &port)) {
      invalid("MORI_RAS_PORT", v, "an ephemeral port");
    } else if (port == kRcclRasPort) {
      c.warnings.push_back(
          "MORI_RAS_PORT=28028 is RCCL RAS's client port; using an ephemeral port");
    } else {
      c.port = static_cast<uint16_t>(port);
    }
  }

  v = get("MORI_RAS_TIMEOUT_FACTOR");
  if (!v.empty()) {
    errno = 0;
    char* end = nullptr;
    double f = std::strtod(v.c_str(), &end);
    if (errno != 0 || *end != '\0' || !(f > 0.0) || f > 100.0) {
      invalid("MORI_RAS_TIMEOUT_FACTOR", v, "1.0");
    } else {
      c.timeoutFactor = f;
    }
  }

  v = get("MORI_RAS_HEARTBEAT_MS");
  if (!v.empty()) {
    uint64_t ms;
    if (!ParseUnsigned(v, 600000, &ms) || ms < 10) {
      invalid("MORI_RAS_HEARTBEAT_MS", v, "1000");
    } else {
      c.heartbeatMs = static_cast<uint32_t>(ms);
    }
  }

  v = get("MORI_RAS_MAX_LINKS");
  if (!v.empty()) {
    uint64_t n;
    if (!ParseUnsigned(v, 65536, &n) || n == 0) {
      invalid("MORI_RAS_MAX_LINKS", v, "256");
    } else {
      c.maxLinks = static_cast<uint32_t>(n);
    }
  }

  v = get("MORI_RAS_CPU_AFFINITY");
  if (!v.empty() && !ParseCpuList(v, &c.cpuAffinity)) {
    c.cpuAffinity.clear();
    invalid("MORI_RAS_CPU_AFFINITY", v, "no pinning");
  }
  return c;
}

RasConfig ConfigFromEnv() {
  return ConfigFromEnv([](const char* name) { return std::getenv(name); });
}

}  // namespace ras
}  // namespace mori

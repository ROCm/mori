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

#include <cerrno>
#include <cstdlib>
#include <optional>

#include "mori/utils/cpu_list.hpp"
#include "mori/utils/env_utils.hpp"

namespace mori {
namespace ras {

RasConfig ConfigFromEnv() {
  RasConfig c;
  auto invalid = [&](const char* name, const std::string& value, const std::string& fallback) {
    c.warnings.push_back(std::string(name) + "=" + value + " is invalid; using " + fallback);
  };

  if (std::optional<std::string> v = env::GetString("MORI_RAS_ENABLE")) {
    std::optional<bool> enable = env::detail::ParseBool(v->c_str());
    c.requested = enable.value_or(true);
    if (enable.has_value()) {
      c.enable = *enable;
    } else {
      invalid("MORI_RAS_ENABLE", *v, "0");
    }
  }

  c.ifname = env::GetString("MORI_RAS_IFNAME").value_or("");

  if (std::optional<std::string> v = env::GetString("MORI_RAS_PORT")) {
    std::optional<int> port = env::GetInt("MORI_RAS_PORT");
    if (!port.has_value() || *port < 0 || *port > 65535) {
      invalid("MORI_RAS_PORT", *v, "an ephemeral port");
    } else if (*port == kRcclRasPort) {
      c.warnings.push_back(
          "MORI_RAS_PORT=28028 is RCCL RAS's client port; using an ephemeral port");
    } else {
      c.port = static_cast<uint16_t>(*port);
    }
  }

  if (std::optional<std::string> v = env::GetString("MORI_RAS_TIMEOUT_FACTOR")) {
    errno = 0;
    char* end = nullptr;
    double f = std::strtod(v->c_str(), &end);
    if (errno != 0 || *end != '\0' || !(f > 0.0) || f > 100.0) {
      invalid("MORI_RAS_TIMEOUT_FACTOR", *v, "1.0");
    } else {
      c.timeoutFactor = f;
    }
  }

  if (std::optional<std::string> v = env::GetString("MORI_RAS_HEARTBEAT_MS")) {
    std::optional<int> ms = env::GetInt("MORI_RAS_HEARTBEAT_MS");
    if (!ms.has_value() || *ms < 10 || *ms > 600000) {
      invalid("MORI_RAS_HEARTBEAT_MS", *v, "1000");
    } else {
      c.heartbeatMs = static_cast<uint32_t>(*ms);
    }
  }

  if (std::optional<std::string> v = env::GetString("MORI_RAS_CPU_AFFINITY")) {
    std::optional<std::vector<int>> cpus = ParseCpuList(*v);
    if (cpus.has_value()) {
      c.cpuAffinity = *cpus;
    } else {
      invalid("MORI_RAS_CPU_AFFINITY", *v, "no pinning");
    }
  }
  return c;
}

}  // namespace ras
}  // namespace mori

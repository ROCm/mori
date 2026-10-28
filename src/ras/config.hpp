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

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mori {
namespace ras {

// RCCL's RAS listens for clients here with SO_REUSEPORT, so a MORI listener on it could be
// handed RCCL's connections. MORI_RAS_PORT never uses it.
constexpr uint16_t kRcclRasPort = 28028;

// MORI_RAS_* settings. An invalid value falls back to its default and leaves a warning, so a
// typo never turns RAS into a failure.
struct RasConfig {
  bool enable{false};            // MORI_RAS_ENABLE: off by default through P3
  std::string ifname;            // MORI_RAS_IFNAME: the interface for RAS links
  uint16_t port{0};              // MORI_RAS_PORT: the peer listener port; 0 picks one
  double timeoutFactor{1.0};     // MORI_RAS_TIMEOUT_FACTOR: scales every timeout and backoff
  uint32_t heartbeatMs{1000};    // MORI_RAS_HEARTBEAT_MS: the heartbeat period
  uint32_t maxLinks{256};        // MORI_RAS_MAX_LINKS: the link budget, enforced from part 2
  std::vector<int> cpuAffinity;  // MORI_RAS_CPU_AFFINITY: CPUs for the RAS thread, e.g. 0-3,8
  std::vector<std::string> warnings;

  uint64_t HeartbeatNs() const { return uint64_t{heartbeatMs} * 1000000; }
  // Connect plus HELLO, in either direction: RCCL's 5 s connect warning.
  uint64_t HandshakeTimeoutNs() const { return ScaledMs(5000); }
  uint64_t ReconnectBackoffInitialNs() const { return ScaledMs(1000); }
  uint64_t ReconnectBackoffMaxNs() const { return ScaledMs(30000); }
  uint64_t ScaledMs(uint64_t ms) const {
    return static_cast<uint64_t>(static_cast<double>(ms) * timeoutFactor * 1e6);
  }
};

using GetEnvFn = std::function<const char*(const char*)>;
RasConfig ConfigFromEnv(const GetEnvFn& getenv);
RasConfig ConfigFromEnv();

}  // namespace ras
}  // namespace mori

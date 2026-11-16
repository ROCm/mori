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

// Identity and state types shared by the RAS runtime and the subsystems that feed it. Plain
// data only, so this header stays usable when RAS is built out.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mori {
namespace ras {

// 128-bit runtime UUID, random (version 4), one per RAS runtime. A process that forks or
// restarts gets a new one, so a UUID names one process lifetime.
struct RuntimeUuid {
  uint64_t hi{0};
  uint64_t lo{0};

  bool IsZero() const { return hi == 0 && lo == 0; }
  bool operator==(const RuntimeUuid& o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(const RuntimeUuid& o) const { return !(*this == o); }
  bool operator<(const RuntimeUuid& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
  std::string ToString() const {
    char buf[37];
    std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx", static_cast<unsigned>(hi >> 32),
                  static_cast<unsigned>((hi >> 16) & 0xffff), static_cast<unsigned>(hi & 0xffff),
                  static_cast<unsigned>(lo >> 48),
                  static_cast<unsigned long long>(lo & 0xffffffffffffull));
    return buf;
  }
};

// Who a RAS runtime is. nodeId, hostname, host, and pid come from the EngineDesc of the engine
// that started RAS; the PID namespace and start time let a peer on the same node confirm an
// exit (P3) without trusting a reused PID.
struct RasProcessId {
  RuntimeUuid uuid;
  std::string nodeId;
  std::string hostname;
  std::string host;
  int32_t pid{0};
  uint64_t pidNamespace{0};    // inode of /proc/self/ns/pid; 0 if unknown
  uint64_t startTimeTicks{0};  // field 22 of /proc/self/stat, in clock ticks since boot
  std::string bootId;          // /proc/sys/kernel/random/boot_id; scopes startTimeTicks

  bool operator==(const RasProcessId& o) const {
    return uuid == o.uuid && nodeId == o.nodeId && hostname == o.hostname && host == o.host &&
           pid == o.pid && pidNamespace == o.pidNamespace && startTimeTicks == o.startTimeTicks &&
           bootId == o.bootId;
  }
};

// Why a runtime is RAS_DEGRADED: it runs, but its view may be incomplete.
enum DegradedReason : uint32_t {
  kDegradedInitFailed = 1u << 0,  // the runtime couldn't start; engines advertise no endpoint
  kDegradedLoopError = 1u << 1,   // the RAS thread hit an unexpected error and carried on
  kDegradedForTest = 1u << 31,    // set by tests
};

struct EventStats {
  size_t capacity{0};
  uint64_t enqueued{0};
  uint64_t dropped{0};
  std::vector<uint64_t> droppedByKind;  // indexed like RasEvent's alternatives
  size_t depthHighWater{0};
};

struct RuntimeSnapshot {
  RasProcessId self;
  uint16_t listenPort{0};
  uint32_t degraded{0};  // DegradedReason bits
  EventStats events;
  uint64_t inboundAccepted{0};  // TCP connections accepted
};

}  // namespace ras
}  // namespace mori

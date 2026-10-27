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
// data only: the wire mapping lives in src/ras/protocol.cpp, so this header stays usable when
// RAS is built out.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mori {
namespace ras {

// 128-bit runtime UUID, random (version 4), one per RAS runtime. A process that forks or
// restarts gets a new one, so a UUID names one process lifetime. Wire headers carry `lo`.
struct RuntimeUuid {
  uint64_t hi{0};
  uint64_t lo{0};

  bool IsZero() const { return hi == 0 && lo == 0; }
  bool operator==(const RuntimeUuid& o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(const RuntimeUuid& o) const { return !(*this == o); }
  bool operator<(const RuntimeUuid& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
  std::string ToString() const;
};

// One RDMA port this host offers, with where it attaches in the fabric when that is known.
struct NicAttachment {
  std::string device;      // RDMA device, e.g. "ionic_3"
  uint32_t port{0};        // RDMA port number
  std::string netdev;      // netdev behind the RoCE GID, e.g. "enp9s0"
  std::string address;     // RoCE GID address: IPv4 for IPv4-mapped GIDs, else IPv6
  std::string rail;        // the address's subnet, e.g. "10.3.0.0/24", or a topology-file name
  std::string switchName;  // attached switch, from LLDP data
  std::string switchPort;  // its port, from LLDP data

  bool operator==(const NicAttachment& o) const {
    return device == o.device && port == o.port && netdev == o.netdev && address == o.address &&
           rail == o.rail && switchName == o.switchName && switchPort == o.switchPort;
  }
};

struct Topology {
  std::string rack;    // from the scheduler or a topology file
  std::string leaf;    // leaf (top-of-rack) switch, same sources
  std::string source;  // comma-separated sources the values came from, e.g. "sysfs,file"
  std::vector<NicAttachment> nics;

  bool operator==(const Topology& o) const {
    return rack == o.rack && leaf == o.leaf && source == o.source && nics == o.nics;
  }
};

using Labels = std::map<std::string, std::string>;

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
  Topology topology;
  Labels labels;  // process labels set by the framework

  bool operator==(const RasProcessId& o) const {
    return uuid == o.uuid && nodeId == o.nodeId && hostname == o.hostname && host == o.host &&
           pid == o.pid && pidNamespace == o.pidNamespace && startTimeTicks == o.startTimeTicks &&
           bootId == o.bootId && topology == o.topology && labels == o.labels;
  }
};

}  // namespace ras
}  // namespace mori

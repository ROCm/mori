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

// Builds a runtime's RasProcessId. Every source is unprivileged and read once at startup.
//
// - Process facts from /proc: the PID namespace (inode of /proc/self/ns/pid), the start time
//   (field 22 of /proc/self/stat), and the boot ID that scopes it.
// - NICs from /sys/class/infiniband: every RDMA port on the host, filtered by
//   MORI_RDMA_DEVICES the way MORI's RDMA context filters it, with its RoCEv2 GID address and
//   netdev. The rail is the address's subnet, from the netdev's prefix length. A port's
//   relationship to the engines that use it arrives with the QP table in P4.
// - Rack, leaf switch, rail names, and LLDP neighbors from the file MORI_RAS_TOPOLOGY_FILE
//   names, written by site tooling. It holds key=value lines:
//       rack=<name>
//       leaf=<name>
//       rail.<netdev or device>=<name>
//       lldp.<netdev>.chassis.name=<switch>
//       lldp.<netdev>.port.{ifname,local,descr,mac}=<port>
//   The lldp keys are exactly what `lldpcli -f keyvalue show neighbors` prints, so a prolog
//   can append its output. RAS doesn't query lldpd itself: its socket is usually limited to
//   the _lldpd group, and spawning lldpcli from a library thread isn't safe.
// - The leaf switch from Slurm's SLURM_TOPOLOGY_ADDR and SLURM_TOPOLOGY_ADDR_PATTERN, when the
//   topology file doesn't name one.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "mori/ras/types.hpp"

namespace mori {
namespace ras {

// Where collection reads from. LiveIdentitySources() is the running system; tests point the
// roots at fixture trees.
struct IdentitySources {
  std::string procRoot{"/proc"};
  std::string sysRoot{"/sys"};
  std::string topologyFile;          // MORI_RAS_TOPOLOGY_FILE
  std::string rdmaDevices;           // MORI_RDMA_DEVICES
  std::string slurmTopologyAddr;     // SLURM_TOPOLOGY_ADDR
  std::string slurmTopologyPattern;  // SLURM_TOPOLOGY_ADDR_PATTERN
  // Prefix length of `address` on `netdev`, or -1 if it isn't configured there.
  std::function<int(const std::string& netdev, const std::string& address)> prefixLength;
};

IdentitySources LiveIdentitySources();

struct ProcessFacts {
  uint64_t pidNamespace{0};
  uint64_t startTimeTicks{0};
  std::string bootId;
};
// `pid` is "self" or a decimal PID. Missing facts stay 0 or empty.
ProcessFacts ReadProcessFacts(const std::string& procRoot, const std::string& pid);

std::vector<NicAttachment> ReadNicInventory(const IdentitySources& src);

// The topology file's keys, applied to `topo`, whose NICs must already be filled in.
void ApplyTopologyFile(const std::string& content, Topology* topo);
// The leaf switch from Slurm's topology address, unless `topo` already names one.
void ApplySlurmTopology(const std::string& addr, const std::string& pattern, Topology* topo);

RasProcessId CollectIdentity(const std::string& nodeId, const std::string& hostname,
                             const std::string& host, int32_t pid, RuntimeUuid uuid,
                             const IdentitySources& src);

// A random version-4 UUID. Its variant bits keep `lo` nonzero, so it can be a header sender.
RuntimeUuid NewRuntimeUuid();
// 128 random bits as 32 hex digits.
std::string NewToken();

// True if peers would accept these labels in a HELLO.
bool ValidLabels(const Labels& labels);

}  // namespace ras
}  // namespace mori

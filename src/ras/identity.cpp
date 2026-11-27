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
#include "src/ras/identity.hpp"

#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/random.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "mori/utils/env_utils.hpp"
#include "src/ras/protocol.hpp"

namespace mori {
namespace ras {
namespace {

constexpr size_t kMaxTopologyFileBytes = 64 * 1024;
constexpr int kMaxGidsScanned = 256;

std::string FirstLine(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  return line;
}

std::string Trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// A sysfs attribute: one line.
std::string Attr(const std::string& path) { return Trim(FirstLine(path)); }

std::string Clip(std::string s) {
  if (s.size() > proto::kMaxIdentityString) s.resize(proto::kMaxIdentityString);
  return s;
}

std::vector<std::string> ListDir(const std::string& path) {
  std::vector<std::string> names;
  DIR* d = opendir(path.c_str());
  if (d == nullptr) return names;
  while (dirent* e = readdir(d)) {
    if (std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0) {
      names.emplace_back(e->d_name);
    }
  }
  closedir(d);
  std::sort(names.begin(), names.end());
  return names;
}

bool ParseU64(const std::string& s, uint64_t* out) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
  errno = 0;
  unsigned long long v = std::strtoull(s.c_str(), nullptr, 10);
  if (errno != 0) return false;
  *out = v;
  return true;
}

void AddSource(Topology* topo, const char* source) {
  std::stringstream ss(topo->source);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item == source) return;
  }
  if (!topo->source.empty()) topo->source += ",";
  topo->source += source;
}

// MORI_RDMA_DEVICES: a comma list of device names, or "^" and a list to exclude.
struct DeviceFilter {
  bool exclude{false};
  std::set<std::string> names;

  explicit DeviceFilter(std::string spec) {
    if (!spec.empty() && spec[0] == '^') {
      exclude = true;
      spec = spec.substr(1);
    }
    std::stringstream ss(spec);
    std::string name;
    while (std::getline(ss, name, ',')) {
      name = Trim(name);
      if (!name.empty()) names.insert(name);
    }
  }
  bool Allows(const std::string& dev) const {
    if (names.empty()) return true;
    return exclude != (names.count(dev) != 0);
  }
};

std::string FormatAddress(int family, const void* bytes) {
  char buf[INET6_ADDRSTRLEN] = {0};
  if (inet_ntop(family, bytes, buf, sizeof(buf)) == nullptr) return "";
  return buf;
}

// `bytes` masked to `prefixLen` bits, as "address/len".
std::string Subnet(int family, const uint8_t* bytes, int prefixLen) {
  int len = family == AF_INET ? 4 : 16;
  if (prefixLen < 0 || prefixLen > len * 8) return "";
  uint8_t masked[16] = {0};
  for (int i = 0; i < len; ++i) {
    int bits = std::min(8, std::max(0, prefixLen - 8 * i));
    masked[i] = static_cast<uint8_t>(bytes[i] & (bits == 0 ? 0 : 0xff << (8 - bits)));
  }
  return FormatAddress(family, masked) + "/" + std::to_string(prefixLen);
}

bool IsZero(const uint8_t* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (b[i] != 0) return false;
  }
  return true;
}

// Fills address, netdev, and rail from the port's RoCEv2 GIDs: the first IPv4-mapped one,
// else the first global IPv6 one. Link-local GIDs are skipped.
void SelectRoceGid(const std::string& portDir, const TopologySources& src, NicAttachment* nic) {
  std::vector<int> indices;
  for (const std::string& name : ListDir(portDir + "/gids")) {
    uint64_t i;
    if (ParseU64(name, &i) && i < kMaxGidsScanned) indices.push_back(static_cast<int>(i));
  }
  std::sort(indices.begin(), indices.end());
  int v6Index = -1;
  uint8_t v6Bytes[16];
  for (int i : indices) {
    std::string idx = std::to_string(i);
    if (Attr(portDir + "/gid_attrs/types/" + idx) != "RoCE v2") continue;
    uint8_t g[16];
    if (inet_pton(AF_INET6, Attr(portDir + "/gids/" + idx).c_str(), g) != 1) continue;
    if (IsZero(g, 16)) continue;
    if (g[0] == 0xfe && (g[1] & 0xc0) == 0x80) continue;
    bool v4Mapped = IsZero(g, 10) && g[10] == 0xff && g[11] == 0xff;
    if (v4Mapped) {
      nic->netdev = Attr(portDir + "/gid_attrs/ndevs/" + idx);
      nic->address = FormatAddress(AF_INET, g + 12);
      if (src.prefixLength) {
        nic->rail = Subnet(AF_INET, g + 12, src.prefixLength(nic->netdev, nic->address));
      }
      return;
    }
    if (v6Index < 0) {
      v6Index = i;
      std::memcpy(v6Bytes, g, 16);
    }
  }
  if (v6Index >= 0) {
    nic->netdev = Attr(portDir + "/gid_attrs/ndevs/" + std::to_string(v6Index));
    nic->address = FormatAddress(AF_INET6, v6Bytes);
    if (src.prefixLength) {
      nic->rail = Subnet(AF_INET6, v6Bytes, src.prefixLength(nic->netdev, nic->address));
    }
  }
}

int LivePrefixLength(const std::string& netdev, const std::string& address) {
  uint8_t want[16];
  int family = address.find(':') != std::string::npos ? AF_INET6 : AF_INET;
  if (inet_pton(family, address.c_str(), want) != 1) return -1;
  ifaddrs* list = nullptr;
  if (getifaddrs(&list) != 0) return -1;
  int result = -1;
  for (ifaddrs* ifa = list; ifa != nullptr && result < 0; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_netmask == nullptr || netdev != ifa->ifa_name) {
      continue;
    }
    if (ifa->ifa_addr->sa_family != family) continue;
    const uint8_t* addr;
    const uint8_t* mask;
    int len;
    if (family == AF_INET) {
      addr = reinterpret_cast<const uint8_t*>(
          &reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr)->sin_addr);
      mask = reinterpret_cast<const uint8_t*>(
          &reinterpret_cast<const sockaddr_in*>(ifa->ifa_netmask)->sin_addr);
      len = 4;
    } else {
      addr = reinterpret_cast<const sockaddr_in6*>(ifa->ifa_addr)->sin6_addr.s6_addr;
      mask = reinterpret_cast<const sockaddr_in6*>(ifa->ifa_netmask)->sin6_addr.s6_addr;
      len = 16;
    }
    if (std::memcmp(addr, want, len) != 0) continue;
    int bits = 0;
    for (int i = 0; i < len; ++i) bits += __builtin_popcount(mask[i]);
    result = bits;
  }
  freeifaddrs(list);
  return result;
}

// Requests are at most 16 bytes, which getrandom returns in full once the kernel's pool is
// seeded. It fails only where the syscall is missing or filtered out.
bool FillRandom(void* out, size_t n) { return getrandom(out, n, 0) == static_cast<ssize_t>(n); }

}  // namespace

uint64_t StartTimeFromStat(const std::string& stat) {
  // Field 2, the command name, is parenthesized and may hold spaces and parentheses.
  size_t close = stat.rfind(')');
  if (close == std::string::npos) return 0;
  std::istringstream in(stat.substr(close + 1));
  std::string skipped;
  for (int field = 3; field < 22; ++field) in >> skipped;
  uint64_t ticks = 0;
  in >> ticks;
  return ticks;
}

uint64_t InodeFromNsLink(const std::string& target) {
  unsigned long long inode = 0;
  return std::sscanf(target.c_str(), "pid:[%llu]", &inode) == 1 ? inode : 0;
}

ProcessFacts ReadProcessFacts() {
  ProcessFacts f;
  f.startTimeTicks = StartTimeFromStat(FirstLine("/proc/self/stat"));
  char link[64];
  ssize_t n = readlink("/proc/self/ns/pid", link, sizeof(link) - 1);
  if (n > 0) f.pidNamespace = InodeFromNsLink(std::string(link, static_cast<size_t>(n)));
  f.bootId = Clip(FirstLine("/proc/sys/kernel/random/boot_id"));
  return f;
}

RasProcessId CollectIdentity(const std::string& nodeId, const std::string& hostname,
                             const std::string& host, int32_t pid, RuntimeUuid uuid) {
  RasProcessId id;
  id.uuid = uuid;
  id.nodeId = Clip(nodeId);
  id.hostname = Clip(hostname);
  id.host = Clip(host);
  id.pid = pid;
  ProcessFacts facts = ReadProcessFacts();
  id.pidNamespace = facts.pidNamespace;
  id.startTimeTicks = facts.startTimeTicks;
  id.bootId = facts.bootId;
  return id;
}

TopologySources LiveTopologySources() {
  TopologySources s;
  s.topologyFile = env::GetString("MORI_RAS_TOPOLOGY_FILE").value_or("");
  s.rdmaDevices = env::GetString("MORI_RDMA_DEVICES").value_or("");
  s.slurmTopologyAddr = env::GetString("SLURM_TOPOLOGY_ADDR").value_or("");
  s.slurmTopologyPattern = env::GetString("SLURM_TOPOLOGY_ADDR_PATTERN").value_or("");
  s.prefixLength = LivePrefixLength;
  return s;
}

std::vector<NicAttachment> ReadNicInventory(const TopologySources& src) {
  std::vector<NicAttachment> nics;
  DeviceFilter filter(src.rdmaDevices);
  std::string root = src.sysRoot + "/class/infiniband";
  for (const std::string& dev : ListDir(root)) {
    if (!filter.Allows(dev)) continue;
    for (const std::string& portName : ListDir(root + "/" + dev + "/ports")) {
      uint64_t port;
      if (!ParseU64(portName, &port)) continue;
      std::string portDir = root + "/" + dev + "/ports/" + portName;
      NicAttachment nic;
      nic.device = Clip(dev);
      nic.port = static_cast<uint32_t>(port);
      if (Attr(portDir + "/link_layer") == "InfiniBand") {
        uint8_t g[16];
        if (inet_pton(AF_INET6, Attr(portDir + "/gids/0").c_str(), g) == 1) {
          nic.address = FormatAddress(AF_INET6, g);
        }
      } else {
        SelectRoceGid(portDir, src, &nic);
      }
      nic.netdev = Clip(nic.netdev);
      nics.push_back(std::move(nic));
      if (nics.size() == proto::kMaxNics) return nics;
    }
  }
  return nics;
}

void ApplyTopologyFile(const std::string& content, Topology* topo) {
  static const std::pair<const char*, int> kPortKeys[] = {
      {".port.ifname", 4}, {".port.local", 3}, {".port.descr", 2}, {".port.mac", 1}};
  static const char kChassisName[] = ".chassis.name";
  auto endsWith = [](const std::string& s, const std::string& suffix) {
    return s.size() > suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  bool usedFile = false;
  bool usedLldp = false;
  std::map<std::string, std::pair<int, std::string>> bestPort;  // netdev -> (priority, port)

  std::stringstream lines(content);
  std::string line;
  while (std::getline(lines, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = Trim(line.substr(0, eq));
    std::string value = Trim(line.substr(eq + 1));
    if (value.empty() || value.size() > proto::kMaxIdentityString) continue;
    if (key == "rack") {
      topo->rack = value;
      usedFile = true;
    } else if (key == "leaf") {
      topo->leaf = value;
      usedFile = true;
    } else if (key.rfind("rail.", 0) == 0) {
      std::string name = key.substr(5);
      for (NicAttachment& nic : topo->nics) {
        if (nic.netdev == name || nic.device == name) {
          nic.rail = value;
          usedFile = true;
        }
      }
    } else if (key.rfind("lldp.", 0) == 0) {
      // An interface name may itself contain dots (VLANs), so match the suffix.
      std::string rest = key.substr(5);
      if (endsWith(rest, kChassisName)) {
        std::string ifname = rest.substr(0, rest.size() - (sizeof(kChassisName) - 1));
        for (NicAttachment& nic : topo->nics) {
          if (!nic.netdev.empty() && nic.netdev == ifname) {
            nic.switchName = value;
            usedLldp = true;
          }
        }
        continue;
      }
      for (const auto& [suffix, priority] : kPortKeys) {
        if (!endsWith(rest, suffix)) continue;
        std::string ifname = rest.substr(0, rest.size() - std::strlen(suffix));
        auto& best = bestPort[ifname];
        if (priority > best.first) best = {priority, value};
        break;
      }
    }
  }
  for (NicAttachment& nic : topo->nics) {
    auto it = nic.netdev.empty() ? bestPort.end() : bestPort.find(nic.netdev);
    if (it != bestPort.end()) {
      nic.switchPort = it->second.second;
      usedLldp = true;
    }
  }
  if (usedFile) AddSource(topo, "file");
  if (usedLldp) AddSource(topo, "lldp");
}

void ApplySlurmTopology(const std::string& addr, const std::string& pattern, Topology* topo) {
  if (!topo->leaf.empty() || addr.empty() || pattern.empty()) return;
  auto split = [](const std::string& s) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, '.')) parts.push_back(part);
    return parts;
  };
  std::vector<std::string> a = split(addr);
  std::vector<std::string> p = split(pattern);
  if (a.size() != p.size() || a.size() < 2 || p.back() != "node") return;
  if (p[p.size() - 2] != "switch" || a[a.size() - 2].empty()) return;
  topo->leaf = Clip(a[a.size() - 2]);
  AddSource(topo, "slurm");
}

Topology CollectTopology(const TopologySources& src, std::string* problem) {
  Topology topo;
  topo.nics = ReadNicInventory(src);
  if (!topo.nics.empty()) AddSource(&topo, "sysfs");
  if (!src.topologyFile.empty()) {
    std::ifstream in(src.topologyFile);
    std::string content(kMaxTopologyFileBytes + 1, '\0');
    in.read(&content[0], static_cast<std::streamsize>(content.size()));
    content.resize(static_cast<size_t>(in.gcount()));
    if (!in && !in.eof()) {
      *problem = "MORI_RAS_TOPOLOGY_FILE=" + src.topologyFile + " can't be read";
    } else if (content.size() > kMaxTopologyFileBytes) {
      *problem = "MORI_RAS_TOPOLOGY_FILE=" + src.topologyFile + " is over 64 KiB";
    } else {
      ApplyTopologyFile(content, &topo);
    }
  }
  ApplySlurmTopology(src.slurmTopologyAddr, src.slurmTopologyPattern, &topo);
  return topo;
}

RuntimeUuid NewRuntimeUuid() {
  RuntimeUuid u;
  if (!FillRandom(&u.hi, sizeof(u.hi)) || !FillRandom(&u.lo, sizeof(u.lo))) return RuntimeUuid{};
  u.hi = (u.hi & ~0xf000ull) | 0x4000ull;        // version 4
  u.lo = (u.lo & ~(3ull << 62)) | (2ull << 62);  // RFC 4122 variant
  return u;
}

std::string NewToken() {
  uint8_t bytes[16];
  if (!FillRandom(bytes, sizeof(bytes))) return "";
  static const char kHex[] = "0123456789abcdef";
  std::string token;
  for (uint8_t b : bytes) {
    token += kHex[b >> 4];
    token += kHex[b & 0xf];
  }
  return token;
}

bool ValidLabels(const Labels& labels) {
  if (labels.size() > proto::kMaxLabels) return false;
  for (const auto& [key, value] : labels) {
    if (key.empty() || key.size() > proto::kMaxLabelKey || value.size() > proto::kMaxLabelValue) {
      return false;
    }
  }
  return true;
}

}  // namespace ras
}  // namespace mori

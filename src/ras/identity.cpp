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

#include <sys/random.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "src/ras/protocol.hpp"

namespace mori {
namespace ras {
namespace {

std::string FirstLine(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  return line;
}

std::string Clip(std::string s) {
  if (s.size() > proto::kMaxIdentityString) s.resize(proto::kMaxIdentityString);
  return s;
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

}  // namespace ras
}  // namespace mori

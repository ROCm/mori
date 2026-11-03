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

#include <fcntl.h>
#include <sys/random.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <sstream>

#include "src/ras/protocol.hpp"

namespace mori {
namespace ras {
namespace {

bool ReadFile(const std::string& path, std::string* out, size_t maxBytes) {
  int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  std::string buf(maxBytes + 1, '\0');
  size_t total = 0;
  while (total < buf.size()) {
    ssize_t n = read(fd, &buf[total], buf.size() - total);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    total += static_cast<size_t>(n);
  }
  close(fd);
  if (total > maxBytes) return false;
  out->assign(buf.data(), total);
  return true;
}

std::string Trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string ReadTrimmed(const std::string& path) {
  std::string s;
  return ReadFile(path, &s, 4096) ? Trim(s) : "";
}

bool ParseU64(const std::string& s, uint64_t* out) {
  if (s.empty() || s[0] < '0' || s[0] > '9') return false;
  errno = 0;
  char* end = nullptr;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  *out = v;
  return true;
}

std::string Clip(std::string s) {
  if (s.size() > proto::kMaxIdentityString) s.resize(proto::kMaxIdentityString);
  return s;
}

void FillRandom(void* out, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(out);
  size_t got = 0;
  while (got < n) {
    ssize_t r = getrandom(p + got, n - got, GRND_NONBLOCK);
    if (r > 0) {
      got += static_cast<size_t>(r);
    } else if (r < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
  if (got == n) return;
  int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  if (fd >= 0) {
    while (got < n) {
      ssize_t r = read(fd, p + got, n - got);
      if (r <= 0) break;
      got += static_cast<size_t>(r);
    }
    close(fd);
  }
  if (got == n) return;
  // No kernel randomness at all: still unique per process, though not unpredictable.
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t seed = static_cast<uint64_t>(ts.tv_nsec) ^ (static_cast<uint64_t>(ts.tv_sec) << 30) ^
                  (static_cast<uint64_t>(getpid()) << 48) ^ reinterpret_cast<uintptr_t>(out);
  for (; got < n; ++got) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    p[got] = static_cast<uint8_t>(seed >> 56);
  }
}

}  // namespace

IdentitySources LiveIdentitySources() { return IdentitySources{}; }

ProcessFacts ReadProcessFacts(const std::string& procRoot, const std::string& pid) {
  ProcessFacts f;
  std::string base = procRoot + "/" + pid;
  std::string stat;
  if (ReadFile(base + "/stat", &stat, 4096)) {
    // Field 2, the command name, is parenthesized and may hold spaces and parentheses.
    size_t close = stat.rfind(')');
    if (close != std::string::npos) {
      std::istringstream in(stat.substr(close + 1));
      std::string field;
      for (int index = 3; in >> field; ++index) {
        if (index == 22) {
          ParseU64(field, &f.startTimeTicks);
          break;
        }
      }
    }
  }
  char link[64];
  ssize_t n = readlink((base + "/ns/pid").c_str(), link, sizeof(link) - 1);
  if (n > 0) {
    std::string target(link, static_cast<size_t>(n));
    size_t open = target.find('[');
    size_t close = target.find(']');
    if (open != std::string::npos && close != std::string::npos && close > open) {
      ParseU64(target.substr(open + 1, close - open - 1), &f.pidNamespace);
    }
  }
  f.bootId = Clip(ReadTrimmed(procRoot + "/sys/kernel/random/boot_id"));
  return f;
}

RasProcessId CollectIdentity(const std::string& nodeId, const std::string& hostname,
                             const std::string& host, int32_t pid, RuntimeUuid uuid,
                             const IdentitySources& src) {
  RasProcessId id;
  id.uuid = uuid;
  id.nodeId = Clip(nodeId);
  id.hostname = Clip(hostname);
  id.host = Clip(host);
  id.pid = pid;
  ProcessFacts facts = ReadProcessFacts(src.procRoot, "self");
  id.pidNamespace = facts.pidNamespace;
  id.startTimeTicks = facts.startTimeTicks;
  id.bootId = facts.bootId;
  return id;
}

RuntimeUuid NewRuntimeUuid() {
  RuntimeUuid u;
  FillRandom(&u.hi, sizeof(u.hi));
  FillRandom(&u.lo, sizeof(u.lo));
  u.hi = (u.hi & ~0xf000ull) | 0x4000ull;        // version 4
  u.lo = (u.lo & ~(3ull << 62)) | (2ull << 62);  // RFC 4122 variant
  return u;
}

std::string NewToken() {
  uint8_t bytes[16];
  FillRandom(bytes, sizeof(bytes));
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

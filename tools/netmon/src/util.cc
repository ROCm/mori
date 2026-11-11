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
#include "util.h"

#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace netmon {

Status Status::Errno(const std::string& what, int err) {
  return Status(what + ": " + std::strerror(err));
}

static uint64_t ClockNs(clockid_t c) {
  timespec ts{};
  clock_gettime(c, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t MonoNs() { return ClockNs(CLOCK_MONOTONIC); }
uint64_t RealNs() { return ClockNs(CLOCK_REALTIME); }

std::string Trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) b++;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
  return s.substr(b, e - b);
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ReadTextFile(const std::string& path, std::string* out) {
  std::ifstream f(path);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

std::string Hostname() {
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf) - 1) != 0) return "unknown";
  return buf;
}

bool ParseDurationNs(const std::string& in, uint64_t* ns) {
  std::string s = Trim(in);
  if (s == "0") {
    *ns = 0;
    return true;
  }
  double mult;
  std::string num;
  auto ends = [&](const char* suf) {
    size_t n = std::strlen(suf);
    return s.size() > n && s.compare(s.size() - n, n, suf) == 0;
  };
  if (ends("us")) {
    mult = 1e3;
    num = s.substr(0, s.size() - 2);
  } else if (ends("ms")) {
    mult = 1e6;
    num = s.substr(0, s.size() - 2);
  } else if (ends("s")) {
    mult = 1e9;
    num = s.substr(0, s.size() - 1);
  } else if (ends("m")) {
    mult = 60e9;
    num = s.substr(0, s.size() - 1);
  } else {
    return false;
  }
  if (num.empty()) return false;
  char* end = nullptr;
  errno = 0;
  double v = std::strtod(num.c_str(), &end);
  if (errno != 0 || end == num.c_str() || *end != '\0' || !std::isfinite(v) || v < 0) return false;
  double total = v * mult;
  if (total > 86400e9) return false;  // 1 day cap keeps all arithmetic far from overflow
  *ns = static_cast<uint64_t>(std::llround(total));
  return true;
}

std::string FormatDurationNs(uint64_t ns) {
  char buf[64];
  if (ns == 0) return "0";
  if (ns % 1000000000ull == 0) {
    snprintf(buf, sizeof(buf), "%llus", static_cast<unsigned long long>(ns / 1000000000ull));
  } else if (ns % 1000000ull == 0) {
    snprintf(buf, sizeof(buf), "%llums", static_cast<unsigned long long>(ns / 1000000ull));
  } else {
    snprintf(buf, sizeof(buf), "%lluus", static_cast<unsigned long long>(ns / 1000ull));
  }
  return buf;
}

}  // namespace netmon

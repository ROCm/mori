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
#include "discover.h"

#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>

#include "ethtool_stats.h"

namespace netmon {

namespace {

std::vector<std::string> ListDir(const std::string& path) {
  std::vector<std::string> out;
  DIR* d = opendir(path.c_str());
  if (!d) return out;
  while (dirent* e = readdir(d)) {
    std::string n = e->d_name;
    if (n != "." && n != "..") out.push_back(n);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

std::string RealPath(const std::string& p) {
  char buf[PATH_MAX];
  if (!realpath(p.c_str(), buf)) return "";
  return buf;
}

std::string Basename(const std::string& p) {
  size_t s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

std::string Dirname(const std::string& p) {
  size_t s = p.find_last_of('/');
  return s == std::string::npos ? "" : p.substr(0, s);
}

std::string ReadLinkBase(const std::string& p) {
  char buf[PATH_MAX];
  ssize_t n = readlink(p.c_str(), buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = 0;
  return Basename(buf);
}

std::string GrandparentBdf(const std::string& real) { return Basename(Dirname(Dirname(real))); }

}  // namespace

Status DiscoverNics(std::vector<Nic>* out, const std::string& root) {
  out->clear();
  // card BDF -> /dev node of its tawk_ipc function.
  std::vector<std::pair<std::string, std::string>> ipc;
  const std::string tawk = root + "/sys/bus/pci/drivers/tawk_ipc";
  for (const auto& fn : ListDir(tawk)) {
    if (fn.find(':') == std::string::npos) continue;
    std::string real = RealPath(tawk + "/" + fn);
    if (real.empty()) continue;
    auto misc = ListDir(real + "/misc");
    if (misc.empty()) continue;
    ipc.emplace_back(GrandparentBdf(real), "/dev/" + misc.front());
  }

  const std::string net = root + "/sys/class/net";
  for (const auto& ifname : ListDir(net)) {
    const std::string dev = net + "/" + ifname + "/device";
    if (ReadLinkBase(dev + "/driver") != "ionic") continue;
    std::string real = RealPath(dev);
    if (real.empty()) continue;
    Nic n;
    n.netdev = ifname;
    n.pci = Basename(real);
    n.card = GrandparentBdf(real);
    auto ib = ListDir(dev + "/infiniband");
    if (!ib.empty()) n.ibdev = ib.front();
    for (const auto& [card, node] : ipc) {
      if (card == n.card) n.ipc_dev = node;
    }
    std::string s;
    if (ReadTextFile(net + "/" + ifname + "/speed", &s)) {
      const std::string text = Trim(s);  // must outlive `end`
      char* end = nullptr;
      long long v = strtoll(text.c_str(), &end, 10);
      if (!text.empty() && end && *end == '\0') n.speed_mbps = v;
    }
    if (ReadTextFile(root + "/sys/module/ionic/version", &s)) n.driver_version = Trim(s);
    if (root.empty()) {
      DriverInfo di;
      if (QueryDriverInfo(ifname, &di).ok()) {
        n.fw_version = di.fw_version;
        if (n.driver_version.empty()) n.driver_version = di.version;
      }
    }
    out->push_back(n);
  }
  return Status::Ok();
}

}  // namespace netmon

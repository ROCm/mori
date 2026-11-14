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
#include "topology.h"

#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

namespace xgmi {
namespace {

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

bool IsCardName(const std::string& name) {
  if (name.size() <= 4 || name.compare(0, 4, "card") != 0) return false;
  return std::all_of(name.begin() + 4, name.end(), [](char c) { return c >= '0' && c <= '9'; });
}

}  // namespace

bool ParsePortNum(const std::string& text, std::vector<PortLine>* out, std::string* err) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("->") == std::string::npos) continue;
    PortLine p;
    int dst_port;
    if (std::sscanf(line.c_str(), " %d:%d -> %d:%d", &p.src_node, &p.src_port, &p.dst_node,
                    &dst_port) != 4) {
      *err = "xgmi_port_num: cannot parse \"" + line + "\"";
      return false;
    }
    out->push_back(p);
  }
  return true;
}

bool Discover(const std::string& drm_root, std::vector<Gpu>* gpus, std::string* err) {
  std::vector<std::pair<int, std::string>> cards;
  DIR* d = opendir(drm_root.c_str());
  if (!d) {
    *err = "cannot open " + drm_root;
    return false;
  }
  while (dirent* e = readdir(d)) {
    const std::string name = e->d_name;
    if (IsCardName(name) &&
        access((drm_root + "/" + name + "/device/xgmi_device_id").c_str(), F_OK) == 0)
      cards.emplace_back(std::stoi(name.substr(4)), name);
  }
  closedir(d);
  std::sort(cards.begin(), cards.end());

  std::map<int, int> node_to_idx;
  std::vector<std::vector<PortLine>> ports(cards.size());
  for (size_t i = 0; i < cards.size(); ++i) {
    const std::string dev = drm_root + "/" + cards[i].second + "/device";
    Gpu g;
    g.idx = static_cast<int>(i);
    g.card = cards[i].second;
    g.metrics_path = dev + "/gpu_metrics";
    std::fill(std::begin(g.peer), std::end(g.peer), -1);
    char real[PATH_MAX];
    if (realpath(dev.c_str(), real)) {
      const std::string r = real;
      g.pci = r.substr(r.find_last_of('/') + 1);
    }
    std::string text, raw;
    if (!ReadFile(dev + "/xgmi_port_num", &text)) {
      *err = "cannot read " + dev + "/xgmi_port_num";
      return false;
    }
    if (!ParsePortNum(text, &ports[i], err)) return false;
    if (!ports[i].empty()) node_to_idx[ports[i][0].src_node] = g.idx;
    if (!ReadFile(g.metrics_path, &raw)) {
      *err = "cannot read " + g.metrics_path;
      return false;
    }
    std::string e;
    if (!ParseLayout(reinterpret_cast<const uint8_t*>(raw.data()), raw.size(), &g.layout, &e)) {
      *err = g.card + ": " + e;
      return false;
    }
    g.link_max = LinkMaxBytesPerSecond(reinterpret_cast<const uint8_t*>(raw.data()), g.layout);
    gpus->push_back(g);
  }
  for (size_t i = 0; i < gpus->size(); ++i) {
    for (const PortLine& p : ports[i]) {
      auto it = node_to_idx.find(p.dst_node);
      if (it != node_to_idx.end() && p.src_port >= 0 && p.src_port < kNumLinks)
        (*gpus)[i].peer[p.src_port] = it->second;
    }
  }
  return true;
}

}  // namespace xgmi

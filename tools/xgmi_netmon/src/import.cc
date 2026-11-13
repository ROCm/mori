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
#include "import.h"

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <set>

#include "rates.h"
#include "recording.h"
#include "vm.h"

namespace xgmi {
namespace {

struct GpuFile {
  std::string path;
  FileHeader header;
  std::vector<Record> records;
  size_t partial_bytes = 0;
};

struct HostRecording {
  std::string dir, host;
  std::vector<GpuFile> gpus;
};

bool LoadDir(const std::string& dir, HostRecording* out, std::string* err) {
  DIR* d = opendir(dir.c_str());
  if (!d) {
    *err = "cannot open " + dir;
    return false;
  }
  std::vector<std::string> files;
  while (dirent* e = readdir(d)) {
    const std::string n = e->d_name;
    if (n.size() > 7 && n.compare(0, 3, "gpu") == 0 && n.compare(n.size() - 4, 4, ".bin") == 0)
      files.push_back(dir + "/" + n);
  }
  closedir(d);
  if (files.empty()) {
    *err = dir + ": no gpu<N>.bin files";
    return false;
  }
  std::sort(files.begin(), files.end());
  out->dir = dir;
  for (const std::string& f : files) {
    GpuFile g;
    g.path = f;
    if (!ReadRecording(f, &g.header, &g.records, &g.partial_bytes, err)) return false;
    const std::string host = g.header.host;
    if (!out->host.empty() && host != out->host) {
      *err = dir + ": files of different hosts (" + out->host + ", " + host + ")";
      return false;
    }
    out->host = host;
    out->gpus.push_back(std::move(g));
  }
  return true;
}

std::string UtcTime(uint64_t real_ns) {
  const time_t t = static_cast<time_t>(real_ns / 1000000000ull);
  tm u;
  gmtime_r(&t, &u);
  char b[32];
  std::strftime(b, sizeof(b), "%Y-%m-%d %H:%M:%S", &u);
  return b;
}

}  // namespace

int RunImport(const ImportOptions& opt) {
  std::string err;
  std::vector<HostRecording> hosts(opt.dirs.size());
  std::string experiment = opt.experiment;
  std::set<std::string> seen;
  for (size_t i = 0; i < opt.dirs.size(); ++i) {
    if (!LoadDir(opt.dirs[i], &hosts[i], &err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    if (!seen.insert(hosts[i].host).second) {
      std::fprintf(stderr, "error: host \"%s\" appears in more than one recording\n",
                   hosts[i].host.c_str());
      return 1;
    }
    for (const GpuFile& g : hosts[i].gpus) {
      const std::string e = g.header.experiment;
      if (opt.experiment.empty() && !experiment.empty() && e != experiment) {
        std::fprintf(
            stderr,
            "error: recordings of different experiments (\"%s\", \"%s\"); use --experiment\n",
            experiment.c_str(), e.c_str());
        return 1;
      }
      if (opt.experiment.empty()) experiment = e;
    }
  }

  Url url;
  if (!ParseUrl(opt.url, &url, &err) || !VmHealthy(url, &err)) {
    std::fprintf(stderr, "error: VictoriaMetrics at %s: %s\n", opt.url.c_str(), err.c_str());
    return 1;
  }
  for (const HostRecording& h : hosts) {
    const int has = VmHasSeries(url, experiment, h.host, &err);
    if (has < 0) {
      std::fprintf(stderr, "error: VictoriaMetrics: %s\n", err.c_str());
      return 1;
    }
    if (has && !opt.replace) {
      std::fprintf(
          stderr,
          "error: VictoriaMetrics already has xgmi data for experiment \"%s\", host \"%s\" "
          "(e.g. from the live push); use --replace to replace it\n",
          experiment.c_str(), h.host.c_str());
      return 1;
    }
    if (has && !VmDeleteSeries(url, experiment, h.host, &err)) {
      std::fprintf(stderr, "error: VictoriaMetrics: %s\n", err.c_str());
      return 1;
    }
  }

  constexpr size_t kBodyLimit = 16 << 20;
  RateStats total;
  uint64_t records = 0, first_ns = UINT64_MAX, last_ns = 0;
  std::string body;
  auto flush = [&]() {
    if (body.empty()) return true;
    if (!VmImport(url, body, 120000, &err)) return false;
    body.clear();
    return true;
  };
  for (const HostRecording& h : hosts) {
    for (const GpuFile& g : h.gpus) {
      RateCalc calc(experiment, h.host, g.header.gpu_idx, g.header.peer, g.header.link_max);
      for (const Record& r : g.records) {
        calc.Add(r, &body);
        if (body.size() > kBodyLimit && !flush()) {
          std::fprintf(stderr, "error: import incomplete, rerun with --replace: %s\n", err.c_str());
          return 1;
        }
      }
      records += g.records.size();
      if (!g.records.empty()) {
        first_ns = std::min(first_ns, g.records.front().real_ns);
        last_ns = std::max(last_ns, g.records.back().real_ns);
      }
      if (g.partial_bytes)
        std::fprintf(stderr, "note: %s ends with a partial record (%zu bytes, dropped)\n",
                     g.path.c_str(), g.partial_bytes);
      const RateStats& s = calc.stats();
      total.intervals += s.intervals;
      total.points += s.points;
      total.stale += s.stale;
      total.decreases += s.decreases;
      total.implausible += s.implausible;
    }
  }
  if (!flush()) {
    std::fprintf(stderr, "error: import incomplete, rerun with --replace: %s\n", err.c_str());
    return 1;
  }
  VmResetCache(url);
  std::fprintf(stderr, "imported experiment \"%s\": %zu host(s), %llu records, %llu points\n",
               experiment.c_str(), hosts.size(), static_cast<unsigned long long>(records),
               static_cast<unsigned long long>(total.points));
  if (records)
    std::fprintf(stderr, "time range (UTC): %s .. %s\n", UtcTime(first_ns).c_str(),
                 UtcTime(last_ns).c_str());
  std::fprintf(
      stderr,
      "dropped intervals: %llu counter decreases, %llu implausible, %llu stale timestamps\n",
      static_cast<unsigned long long>(total.decreases),
      static_cast<unsigned long long>(total.implausible),
      static_cast<unsigned long long>(total.stale));
  return 0;
}

}  // namespace xgmi

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
#include "cli.h"

#include <getopt.h>

#include <cstdlib>
#include <set>

#include "http_client.h"
#include "util.h"

namespace netmon {

const char* RecordUsage() {
  return R"(usage: ainic-netmon record -o DIR [options]

  -o, --out DIR          output directory; must not exist or be empty. "{host}" expands to
                         the host id (e.g. -o /shared/run1/{host})
      --host-id ID       host id for {host} and meta.json (default: hostname)
      --nics N1,N2       netdevs to record (default: all ionic NICs)
      --sources LIST     auto (default) | any of rdma,ipc,ethtool. auto = rdma + ethtool, plus
                         ipc when running as root on a verified firmware. Explicitly listed
                         sources that are unavailable are an error.
      --ipc-unverified   enable ipc on a firmware outside the verified list
      --interval SPEC    e.g. rdma=2ms,ipc=2ms,ethtool=1s (default); 0 = back-to-back
  -d, --duration DUR     e.g. 60s, 10m (default: until Ctrl-C / SIGTERM)
      --tag K=V          free-form metadata stored in meta.json, repeatable
      --cpus LIST        pin sampler threads round-robin, e.g. 0-7,64
      --owner UID:GID    when running as root: owner of the output (see README)
      --sudo             not root: re-run this binary with sudo (asks for the password)
  -q, --quiet            no per-second console summary

  live push to VictoriaMetrics (view in Grafana while recording; see examples/monitoring/netmon):
      --push URL         e.g. http://127.0.0.1:8428. The recording is unaffected if VictoriaMetrics
                         is slow or goes away (points are dropped and counted). Re-import the
                         recording afterwards (ainic-netmon-import --replace) for exact numbers.
      --experiment NAME  experiment label (default: name of DIR's parent directory)
      --push-interval DUR  how often to push (default 1s)
      --push-counters key|all  key counters + PFC (default), or every counter that changes
      --grafana URL      only for the dashboard links printed at start (default
                         http://127.0.0.1:3000)
)";
}

bool ParseCpuList(const std::string& spec, std::vector<int>* out) {
  out->clear();
  for (const auto& part : Split(spec, ',')) {
    if (part.empty()) return false;
    auto range = Split(part, '-');
    char* e = nullptr;
    long a = strtol(range[0].c_str(), &e, 10);
    if (*e || range[0].empty() || a < 0 || a > 65535) return false;
    long b = a;
    if (range.size() == 2) {
      b = strtol(range[1].c_str(), &e, 10);
      if (*e || range[1].empty() || b < a || b > 65535) return false;
    } else if (range.size() > 2) {
      return false;
    }
    for (long c = a; c <= b; c++) out->push_back(static_cast<int>(c));
  }
  return !out->empty();
}

bool ParseIntervals(const std::string& spec, RecordArgs* out, std::string* error) {
  for (const auto& item : Split(spec, ',')) {
    auto kv = Split(item, '=');
    uint64_t ns;
    if (kv.size() != 2 || !ParseDurationNs(kv[1], &ns)) {
      *error = "bad --interval item '" + item + "' (expected SOURCE=DURATION)";
      return false;
    }
    if (ns > kMaxIntervalNs) {
      *error = "--interval " + kv[0] + " exceeds 60s";
      return false;
    }
    if (kv[0] == "rdma") {
      out->interval_rdma_ns = ns;
    } else if (kv[0] == "ipc") {
      out->interval_ipc_ns = ns;
    } else if (kv[0] == "ethtool") {
      out->interval_ethtool_ns = ns;
    } else {
      *error = "--interval: unknown source '" + kv[0] + "'";
      return false;
    }
  }
  return true;
}

std::string ExpandHost(const std::string& path, const std::string& host_id) {
  std::string out;
  const std::string key = "{host}";
  size_t i = 0;
  while (i < path.size()) {
    if (path.compare(i, key.size(), key) == 0) {
      out += host_id;
      i += key.size();
    } else {
      out += path[i++];
    }
  }
  return out;
}

bool ParseRecordArgs(int argc, char** argv, RecordArgs* a, std::string* error) {
  enum {
    kHostId = 1000,
    kNics,
    kSources,
    kIpcUnverified,
    kInterval,
    kTag,
    kCpus,
    kOwner,
    kSudo,
    kPush,
    kExperiment,
    kPushInterval,
    kPushCounters,
    kGrafana
  };
  static const option opts[] = {
      {"out", required_argument, nullptr, 'o'},
      {"host-id", required_argument, nullptr, kHostId},
      {"nics", required_argument, nullptr, kNics},
      {"sources", required_argument, nullptr, kSources},
      {"ipc-unverified", no_argument, nullptr, kIpcUnverified},
      {"interval", required_argument, nullptr, kInterval},
      {"duration", required_argument, nullptr, 'd'},
      {"tag", required_argument, nullptr, kTag},
      {"cpus", required_argument, nullptr, kCpus},
      {"owner", required_argument, nullptr, kOwner},
      {"sudo", no_argument, nullptr, kSudo},
      {"quiet", no_argument, nullptr, 'q'},
      {"help", no_argument, nullptr, 'h'},
      {"push", required_argument, nullptr, kPush},
      {"experiment", required_argument, nullptr, kExperiment},
      {"push-interval", required_argument, nullptr, kPushInterval},
      {"push-counters", required_argument, nullptr, kPushCounters},
      {"grafana", required_argument, nullptr, kGrafana},
      {nullptr, 0, nullptr, 0},
  };
  std::set<int> seen;
  optind = 1;
  opterr = 0;
  int c;
  while ((c = getopt_long(argc, argv, "+o:d:qh", opts, nullptr)) != -1) {
    if (c != kTag && !seen.insert(c).second) {
      *error = std::string("option given more than once: ") + argv[optind - 1];
      return false;
    }
    std::string v = optarg ? optarg : "";
    switch (c) {
      case 'o':
        a->out = v;
        break;
      case kHostId:
        if (v.empty() || v.find('/') != std::string::npos) {
          *error = "--host-id must be non-empty and contain no '/'";
          return false;
        }
        a->host_id = v;
        break;
      case kNics:
        a->nics = Split(v, ',');
        for (const auto& n : a->nics) {
          if (n.empty()) {
            *error = "--nics: empty name";
            return false;
          }
        }
        break;
      case kSources:
        if (v == "auto") {
          a->sources_auto = true;
          a->want_rdma = a->want_ipc = a->want_ethtool = true;
          break;
        }
        a->sources_auto = false;
        a->want_rdma = a->want_ipc = a->want_ethtool = false;
        for (const auto& s : Split(v, ',')) {
          if (s == "rdma")
            a->want_rdma = true;
          else if (s == "ipc")
            a->want_ipc = true;
          else if (s == "ethtool")
            a->want_ethtool = true;
          else {
            *error = "--sources: unknown source '" + s + "'";
            return false;
          }
        }
        break;
      case kIpcUnverified:
        a->ipc_unverified = true;
        break;
      case kInterval:
        if (!ParseIntervals(v, a, error)) return false;
        break;
      case 'd':
        if (!ParseDurationNs(v, &a->duration_ns) || a->duration_ns == 0) {
          *error = "bad --duration '" + v + "'";
          return false;
        }
        break;
      case kTag: {
        size_t eq = v.find('=');
        if (eq == std::string::npos || eq == 0) {
          *error = "--tag must be KEY=VALUE";
          return false;
        }
        a->tags[v.substr(0, eq)] = v.substr(eq + 1);
        break;
      }
      case kCpus:
        if (!ParseCpuList(v, &a->cpus)) {
          *error = "bad --cpus '" + v + "'";
          return false;
        }
        break;
      case kOwner:
        a->owner = v;
        break;
      case kSudo:
        a->sudo = true;
        break;
      case 'q':
        a->quiet = true;
        break;
      case kPush: {
        HttpUrl u;
        if (!ParseHttpUrl(v, &u, error)) return false;
        a->push_url = v;
        while (!a->push_url.empty() && a->push_url.back() == '/') a->push_url.pop_back();
        break;
      }
      case kExperiment:
        if (v.empty()) {
          *error = "--experiment must not be empty";
          return false;
        }
        a->experiment = v;
        break;
      case kPushInterval:
        if (!ParseDurationNs(v, &a->push_interval_ns) || a->push_interval_ns < 100000000ull ||
            a->push_interval_ns > kMaxIntervalNs) {
          *error = "bad --push-interval '" + v + "' (100ms .. 60s)";
          return false;
        }
        break;
      case kPushCounters:
        if (v != "key" && v != "all") {
          *error = "--push-counters must be key or all";
          return false;
        }
        a->push_all = v == "all";
        break;
      case kGrafana:
        a->grafana = v;
        break;
      case 'h':
        *error = "help";
        return false;
      default:
        *error = std::string("unknown or incomplete option: ") + argv[optind - 1];
        return false;
    }
  }
  if (optind < argc) {
    *error = std::string("unexpected argument: ") + argv[optind];
    return false;
  }
  if (a->out.empty()) {
    *error = "-o/--out is required";
    return false;
  }
  if (a->push_url.empty() && (seen.count(kExperiment) || seen.count(kPushInterval) ||
                              seen.count(kPushCounters) || seen.count(kGrafana))) {
    *error = "--experiment / --push-interval / --push-counters / --grafana need --push";
    return false;
  }
  return true;
}

}  // namespace netmon

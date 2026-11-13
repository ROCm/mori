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
// xgmi-netmon: per-link XGMI bandwidth of AMD Instinct GPUs at 2 ms resolution.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "import.h"
#include "recorder.h"
#include "topology.h"

namespace {

const char kUsage[] =
    "usage:\n"
    "  xgmi-netmon list\n"
    "  xgmi-netmon record -o DIR [-d DURATION] [--interval-ms MS] [--push URL]\n"
    "                     [--experiment NAME] [--host-id NAME]\n"
    "  xgmi-netmon import DIR... [--url URL] [--experiment NAME] [--replace]\n"
    "  xgmi-netmon version\n"
    "\n"
    "record:\n"
    "  -o DIR            output directory; {host} expands to the host id. Must not exist or be "
    "empty.\n"
    "  -d DURATION       e.g. 60, 60s, 5m, 1h (default: until Ctrl-C / SIGTERM)\n"
    "  --interval-ms MS  read period per GPU (default 2, as ainic-netmon)\n"
    "  --push URL        also push live to VictoriaMetrics, e.g. http://127.0.0.1:28428\n"
    "  --experiment NAME experiment label (default: name of DIR's parent directory)\n"
    "  --host-id NAME    host label (default: first DNS label of the hostname)\n"
    "import:\n"
    "  --url URL         VictoriaMetrics (default http://127.0.0.1:28428)\n"
    "  --experiment NAME override the experiment stored in the recordings\n"
    "  --replace         replace xgmi data already in VictoriaMetrics for these hosts\n";

bool ParseDuration(const std::string& s, double* out) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  const std::string unit = end;
  const double mult = unit.empty() || unit == "s" ? 1 : unit == "m" ? 60 : unit == "h" ? 3600 : -1;
  if (end == s.c_str() || mult < 0 || v <= 0) return false;
  *out = v * mult;
  return true;
}

int Usage(const char* msg) {
  if (msg) std::fprintf(stderr, "error: %s\n\n", msg);
  std::fputs(kUsage, stderr);
  return 2;
}

int List() {
  std::vector<xgmi::Gpu> gpus;
  std::string err;
  if (!xgmi::Discover("/sys/class/drm", &gpus, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  if (gpus.empty()) {
    std::fprintf(stderr, "no GPU with XGMI found\n");
    return 1;
  }
  std::printf("%3s  %-7s %-13s %12s  %s\n", "gpu", "card", "pci", "link max/dir", "port->peer");
  for (const xgmi::Gpu& g : gpus) {
    std::string links;
    for (int p = 0; p < xgmi::kNumLinks; ++p)
      if (g.peer[p] >= 0) links += std::to_string(p) + "->GPU" + std::to_string(g.peer[p]) + " ";
    std::printf("%3d  %-7s %-13s %7.1f GB/s  %s\n", g.idx, g.card.c_str(), g.pci.c_str(),
                g.link_max / 1e9, links.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return Usage(nullptr);
  const std::string cmd = argv[1];
  std::vector<std::string> args(argv + 2, argv + argc);
  auto value = [&](size_t* i, std::string* out) {  // the option's argument
    if (*i + 1 >= args.size()) return false;
    *out = args[++*i];
    return true;
  };

  if (cmd == "list") return List();
  if (cmd == "version" || cmd == "--version") {
    std::printf("xgmi-netmon %s\n", XGMI_NETMON_VERSION);
    return 0;
  }
  if (cmd == "-h" || cmd == "--help" || cmd == "help") {
    std::fputs(kUsage, stdout);
    return 0;
  }
  if (cmd == "record") {
    xgmi::RecordOptions o;
    for (size_t i = 0; i < args.size(); ++i) {
      const std::string& a = args[i];
      std::string v;
      if ((a == "-o" || a == "--output") && value(&i, &v)) {
        o.out = v;
      } else if ((a == "-d" || a == "--duration") && value(&i, &v)) {
        if (!ParseDuration(v, &o.duration_s)) return Usage("bad duration");
      } else if (a == "--interval-ms" && value(&i, &v)) {
        const double ms = std::atof(v.c_str());
        if (ms < 0.1 || ms > 1000) return Usage("--interval-ms must be 0.1 .. 1000");
        o.period_ns = static_cast<uint64_t>(ms * 1e6);
      } else if (a == "--push" && value(&i, &v)) {
        o.push_url = v;
      } else if (a == "--experiment" && value(&i, &v)) {
        o.experiment = v;
      } else if (a == "--host-id" && value(&i, &v)) {
        o.host_id = v;
      } else {
        return Usage(("unknown or incomplete option " + a).c_str());
      }
    }
    if (o.out.empty()) return Usage("record needs -o DIR");
    return xgmi::RunRecord(o);
  }
  if (cmd == "import") {
    xgmi::ImportOptions o;
    for (size_t i = 0; i < args.size(); ++i) {
      const std::string& a = args[i];
      std::string v;
      if (a == "--url" && value(&i, &v)) {
        o.url = v;
      } else if (a == "--experiment" && value(&i, &v)) {
        o.experiment = v;
      } else if (a == "--replace") {
        o.replace = true;
      } else if (!a.empty() && a[0] == '-') {
        return Usage(("unknown or incomplete option " + a).c_str());
      } else {
        o.dirs.push_back(a);
      }
    }
    if (o.dirs.empty()) return Usage("import needs at least one recording directory");
    return xgmi::RunImport(o);
  }
  return Usage(("unknown command " + cmd).c_str());
}

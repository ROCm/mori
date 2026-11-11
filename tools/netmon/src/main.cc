// ainic-netmon: millisecond-level counter collector for AMD AINIC (ionic).
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cli.h"
#include "discover.h"
#include "ethtool_stats.h"
#include "ionic_ipc.h"
#include "meta_json.h"
#include "privilege.h"
#include "rdma_netlink.h"
#include "recorder.h"

using namespace netmon;

namespace {

constexpr uint64_t kNetlinkTimeoutNs = 2000000000ull;
const std::array<uint8_t, 2> kDefaultIpcPort = {0x01, 0x11};

const char* kUsage = R"(ainic-netmon: millisecond-level counter collector for AMD AINIC (ionic)

usage:
  ainic-netmon list [--json]           NICs, mapping and available sources
  ainic-netmon verify [--nics N1,..]   check the firmware IPC layout (needs root)
  ainic-netmon record -o DIR [...]     record counters (see `ainic-netmon record --help`)
  ainic-netmon path                    print this binary's absolute path
  ainic-netmon version

Sources: rdma (RDMA netlink hw counters, unprivileged), ipc (firmware port stats, root),
ethtool (ethtool -S; its port part refreshes only every ~5 s on ionic).
Analysis and CSV export: ainic-netmon-report (Python, tools/netmon/report/).
)";

bool IsRoot() { return geteuid() == 0; }

void NoNicsHint() {
  fprintf(stderr,
          "no ionic NICs found in /sys/class/net. In a container, use --network host "
          "(NIC discovery and ethtool need the host network namespace).\n");
}

std::vector<Nic> FilterNics(const std::vector<Nic>& all, const std::vector<std::string>& want,
                            bool* ok) {
  *ok = true;
  if (want.empty()) return all;
  std::vector<Nic> out;
  for (const auto& w : want) {
    bool found = false;
    for (const auto& n : all) {
      if (n.netdev == w) {
        out.push_back(n);
        found = true;
      }
    }
    if (!found) {
      fprintf(stderr, "error: %s is not an ionic NIC on this host\n", w.c_str());
      *ok = false;
    }
  }
  return out;
}

int CmdList(int argc, char** argv) {
  bool json = argc > 1 && std::strcmp(argv[1], "--json") == 0;
  std::vector<Nic> nics;
  DiscoverNics(&nics);
  if (nics.empty()) {
    NoNicsHint();
    return 1;
  }
  struct Row {
    Nic n;
    int rdma = -1, eth = -1;
    bool ipc_verified = false;
  };
  std::vector<Row> rows;
  for (const auto& n : nics) {
    Row r{n};
    if (!n.ibdev.empty()) {
      RdmaNetlinkSource s;
      if (s.Open(n.ibdev, 1, kNetlinkTimeoutNs).ok()) r.rdma = static_cast<int>(s.names().size());
    }
    EthtoolSource e;
    if (e.Open(n.netdev).ok()) r.eth = static_cast<int>(e.names().size());
    r.ipc_verified = ipc::FindVerified(n.fw_version) != nullptr;
    rows.push_back(r);
  }
  if (json) {
    JsonWriter j;
    j.BeginArray();
    for (const auto& r : rows) {
      j.BeginObject().Key("netdev").String(r.n.netdev).Key("pci").String(r.n.pci);
      j.Key("card").String(r.n.card).Key("ibdev").String(r.n.ibdev);
      j.Key("ipc_dev").String(r.n.ipc_dev).Key("fw_version").String(r.n.fw_version);
      j.Key("ipc_fw_verified").Bool(r.ipc_verified).Key("speed_mbps").Int(r.n.speed_mbps);
      j.Key("rdma_counters").Int(r.rdma).Key("ethtool_stats").Int(r.eth).EndObject();
    }
    j.EndArray();
    printf("%s\n", j.str().c_str());
    return 0;
  }
  printf("%-10s %-13s %-13s %-8s %-20s %-14s %6s %8s %7s\n", "netdev", "pci", "card", "ibdev",
         "ipc", "fw", "rdma#", "ethtool#", "speed");
  std::string devices;
  for (const auto& r : rows) {
    std::string fw = r.n.fw_version.empty() ? "-" : r.n.fw_version + (r.ipc_verified ? "" : "*");
    printf("%-10s %-13s %-13s %-8s %-20s %-14s %6s %8s %7s\n", r.n.netdev.c_str(), r.n.pci.c_str(),
           r.n.card.c_str(), r.n.ibdev.empty() ? "-" : r.n.ibdev.c_str(),
           r.n.ipc_dev.empty() ? "-" : r.n.ipc_dev.c_str(), fw.c_str(),
           r.rdma < 0 ? "-" : std::to_string(r.rdma).c_str(),
           r.eth < 0 ? "-" : std::to_string(r.eth).c_str(),
           r.n.speed_mbps < 0 ? "-" : (std::to_string(r.n.speed_mbps / 1000) + "G").c_str());
    if (!r.n.ipc_dev.empty()) devices += " --device " + r.n.ipc_dev;
  }
  printf("\n* firmware not in the verified IPC list (ipc needs --ipc-unverified)\n");
  if (!IsRoot()) printf("ipc needs root: use a privileged container, sudo, or --sudo.\n");
  if (!devices.empty()) {
    printf("container without --privileged: docker run --network host%s ...\n", devices.c_str());
  }
  return 0;
}

int CmdVerify(int argc, char** argv) {
  std::vector<std::string> want;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--nics") == 0 && i + 1 < argc) {
      want = Split(argv[++i], ',');
    } else {
      fprintf(stderr, "usage: ainic-netmon verify [--nics N1,N2]\n");
      return 1;
    }
  }
  if (!IsRoot()) {
    fprintf(stderr, "verify needs root (firmware IPC)\n");
    return 1;
  }
  std::vector<Nic> all;
  DiscoverNics(&all);
  bool ok = true;
  auto nics = FilterNics(all, want, &ok);
  if (!ok) return 1;
  int failures = 0, checked = 0;
  for (const auto& n : nics) {
    if (n.ipc_dev.empty()) {
      printf("%s: no IPC node, skipped\n", n.netdev.c_str());
      continue;
    }
    const auto* v = ipc::FindVerified(n.fw_version);
    IonicIpcSource s;
    Status st = s.Open(n.ipc_dev, v ? v->port : kDefaultIpcPort, n.netdev);
    std::string detail;
    if (st.ok()) st = s.Probe(&detail);
    checked++;
    printf("%s (%s, fw %s%s): %s\n", n.netdev.c_str(), n.ipc_dev.c_str(), n.fw_version.c_str(),
           v ? ", verified" : ", NOT in verified list",
           st.ok() ? detail.c_str() : st.message().c_str());
    if (!st.ok()) failures++;
  }
  if (checked == 0) {
    printf("no NIC with an IPC node\n");
    return 1;
  }
  return failures ? 1 : 0;
}

int CmdRecord(int argc, char** argv) {
  RecordArgs a;
  std::string err;
  if (!ParseRecordArgs(argc, argv, &a, &err)) {
    if (err == "help") {
      printf("%s", RecordUsage());
      return 0;
    }
    fprintf(stderr, "error: %s\n\n%s", err.c_str(), RecordUsage());
    return kExitSetup;
  }

  // --sudo: explicit re-exec of this binary by absolute path; never implicit.
  if (a.sudo && !IsRoot()) {
    std::vector<std::string> args(argv, argv + argc);  // argv[0] is "record"
    auto sv = BuildSudoArgv(SelfExePath(), args, getuid(), getgid());
    std::vector<char*> cargv;
    for (auto& s : sv) cargv.push_back(s.data());
    cargv.push_back(nullptr);
    fflush(stdout);
    execvp("sudo", cargv.data());
    fprintf(stderr,
            "error: could not run sudo (%s). Run without --sudo to record rdma+ethtool only.\n",
            std::strerror(errno));
    return kExitSetup;
  }

  if (a.host_id.empty()) a.host_id = Hostname();
  const std::string out = ExpandHost(a.out, a.host_id);

  std::vector<Nic> all;
  DiscoverNics(&all);
  if (all.empty()) {
    NoNicsHint();
    return kExitSetup;
  }
  bool ok = true;
  auto nics = FilterNics(all, a.nics, &ok);
  if (!ok) return kExitSetup;

  RecordOptions opt;
  opt.out_dir = out;
  opt.duration_ns = a.duration_ns;
  opt.quiet = a.quiet;
  opt.host_id = a.host_id;
  opt.tags = a.tags;
  opt.command.assign(argv, argv + argc);
  opt.command.insert(opt.command.begin(), "ainic-netmon");
  opt.nics = nics;

  std::vector<std::unique_ptr<Stream>> streams;
  bool fatal = false;
  auto unavailable = [&](const Nic& n, const char* src, const std::string& why) {
    if (!a.sources_auto) {
      fprintf(stderr, "error: %s.%s unavailable: %s\n", n.netdev.c_str(), src, why.c_str());
      fatal = true;
    } else {
      opt.degraded.push_back({n.netdev, src, why});
    }
  };
  auto add = [&](const Nic& n, std::unique_ptr<Source> src, uint64_t iv, bool unverified) {
    auto s = std::make_unique<Stream>();
    s->netdev = n.netdev;
    s->ibdev = n.ibdev;
    s->pci = n.pci;
    s->src = std::move(src);
    s->interval_ns = iv;
    s->required = !a.sources_auto;
    s->ipc_unverified = unverified;
    streams.push_back(std::move(s));
  };
  for (const auto& n : nics) {
    if (a.want_rdma) {
      if (n.ibdev.empty()) {
        unavailable(n, "rdma", "no RDMA device");
      } else {
        auto s = std::make_unique<RdmaNetlinkSource>();
        Status st = s->Open(n.ibdev, 1, kNetlinkTimeoutNs);
        if (st.ok())
          add(n, std::move(s), a.interval_rdma_ns, false);
        else
          unavailable(n, "rdma", st.message());
      }
    }
    if (a.want_ipc) {
      const auto* v = ipc::FindVerified(n.fw_version);
      if (!IsRoot()) {
        unavailable(n, "ipc", "not root (use a privileged container, sudo, or --sudo)");
      } else if (n.ipc_dev.empty()) {
        unavailable(n, "ipc", "no tawk_ipc node (in a container: --privileged or --device)");
      } else if (!v && !a.ipc_unverified) {
        unavailable(
            n, "ipc",
            "firmware " + n.fw_version + " is not in the verified list (use --ipc-unverified)");
      } else {
        auto s = std::make_unique<IonicIpcSource>();
        Status st = s->Open(n.ipc_dev, v ? v->port : kDefaultIpcPort, n.netdev);
        if (st.ok()) {
          add(n, std::move(s), a.interval_ipc_ns, v == nullptr);
        } else {
          // Visible in sysfs but no /dev node: typically a container without the device.
          unavailable(n, "ipc",
                      st.message() + (access(n.ipc_dev.c_str(), F_OK) != 0
                                          ? " (in a container: --privileged or --device)"
                                          : ""));
        }
      }
    }
    if (a.want_ethtool) {
      auto s = std::make_unique<EthtoolSource>();
      Status st = s->Open(n.netdev);
      if (st.ok())
        add(n, std::move(s), a.interval_ethtool_ns, false);
      else
        unavailable(n, "ethtool", st.message());
    }
  }
  if (fatal) return kExitSetup;
  for (const auto& d : opt.degraded) {
    fprintf(stderr, "skip %s.%s: %s\n", d.netdev.c_str(), d.source.c_str(), d.reason.c_str());
  }
  if (streams.empty()) {
    fprintf(stderr, "error: nothing to record\n");
    return kExitSetup;
  }
  for (size_t i = 0; i < streams.size() && !a.cpus.empty(); i++) {
    streams[i]->cpu = a.cpus[i % a.cpus.size()];
  }

  // Drop root now: every privileged fd is open, and no thread exists yet.
  if (IsRoot()) {
    Owner owner;
    Status st;
    if (DetermineOwner(a.owner, out, &owner, &st)) {
      st = DropPrivileges(owner);
      if (!st.ok()) {
        fprintf(stderr, "error: %s\n", st.message().c_str());
        return kExitSetup;
      }
      fprintf(stderr, "running as uid %u gid %u (%s)\n", owner.uid, owner.gid,
              owner.source.c_str());
    } else if (!st.ok()) {
      fprintf(stderr, "error: %s\n", st.message().c_str());
      return kExitSetup;
    } else {
      fprintf(stderr,
              "warning: no non-root owner found (--owner, SUDO_UID, or a non-root parent "
              "directory); output files will belong to root\n");
    }
  }
  return RunRecording(&streams, &opt);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "%s", kUsage);
    return 1;
  }
  std::string cmd = argv[1];
  if (cmd == "list") return CmdList(argc - 1, argv + 1);
  if (cmd == "verify") return CmdVerify(argc - 1, argv + 1);
  if (cmd == "record") return CmdRecord(argc - 1, argv + 1);
  if (cmd == "path") {
    printf("%s\n", SelfExePath().c_str());
    return 0;
  }
  if (cmd == "version" || cmd == "--version") {
    printf("ainic-netmon %s\n", NETMON_VERSION);
    return 0;
  }
  if (cmd == "-h" || cmd == "--help" || cmd == "help") {
    printf("%s", kUsage);
    return 0;
  }
  fprintf(stderr, "unknown command '%s'\n\n%s", cmd.c_str(), kUsage);
  return 1;
}

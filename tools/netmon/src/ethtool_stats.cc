#include "ethtool_stats.h"

#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace netmon {

namespace {

int OpenCtlSocket() { return socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0); }

Status DoIoctl(int fd, const std::string& netdev, void* data, const char* what) {
  ifreq ifr{};
  if (netdev.size() >= IFNAMSIZ) return Status::Error("interface name too long: " + netdev);
  std::memcpy(ifr.ifr_name, netdev.c_str(), netdev.size());
  ifr.ifr_data = static_cast<char*>(data);
  if (ioctl(fd, SIOCETHTOOL, &ifr) != 0) {
    return Status::Errno(std::string("SIOCETHTOOL ") + what + " " + netdev, errno);
  }
  return Status::Ok();
}

std::string FixedStr(const char* p, size_t n) {
  size_t e = 0;
  while (e < n && p[e] != 0) e++;
  return std::string(p, e);
}

}  // namespace

Status QueryDriverInfo(const std::string& netdev, DriverInfo* out) {
  int fd = OpenCtlSocket();
  if (fd < 0) return Status::Errno("socket", errno);
  ethtool_drvinfo di{};
  di.cmd = ETHTOOL_GDRVINFO;
  Status st = DoIoctl(fd, netdev, &di, "GDRVINFO");
  close(fd);
  if (!st.ok()) return st;
  out->driver = FixedStr(di.driver, sizeof(di.driver));
  out->version = FixedStr(di.version, sizeof(di.version));
  out->fw_version = FixedStr(di.fw_version, sizeof(di.fw_version));
  out->bus_info = FixedStr(di.bus_info, sizeof(di.bus_info));
  return Status::Ok();
}

EthtoolSource::~EthtoolSource() {
  if (fd_ >= 0) close(fd_);
}

Status EthtoolSource::Ioctl(void* data, const char* what) {
  return DoIoctl(fd_, netdev_, data, what);
}

Status EthtoolSource::Open(const std::string& netdev) {
  netdev_ = netdev;
  fd_ = OpenCtlSocket();
  if (fd_ < 0) return Status::Errno("socket", errno);

  ethtool_drvinfo di{};
  di.cmd = ETHTOOL_GDRVINFO;
  Status st = Ioctl(&di, "GDRVINFO");
  if (!st.ok()) return st;
  const size_t n = di.n_stats;
  if (n == 0 || n > kMaxStats) return Status::Error(netdev + ": unexpected stat count");

  // GSTRINGS: the kernel writes as many strings as the driver currently has, whatever len we
  // pass, so the buffer is guarded rather than sized from n.
  GuardedBuffer strings;
  st = strings.Allocate(sizeof(ethtool_gstrings) + kMaxStats * ETH_GSTRING_LEN);
  if (!st.ok()) return st;
  auto* gs = reinterpret_cast<ethtool_gstrings*>(strings.data());
  gs->cmd = ETHTOOL_GSTRINGS;
  gs->string_set = ETH_SS_STATS;
  gs->len = static_cast<uint32_t>(n);
  st = Ioctl(gs, "GSTRINGS");
  if (!st.ok()) return st;
  if (gs->len != n) {
    return Status::Error(netdev + ": stat count changed during discovery (" + std::to_string(n) +
                         " -> " + std::to_string(gs->len) + ")");
  }
  names_.reserve(n);
  for (size_t i = 0; i < n; i++) {
    names_.push_back(
        FixedStr(reinterpret_cast<const char*>(gs->data) + i * ETH_GSTRING_LEN, ETH_GSTRING_LEN));
  }

  return stats_buf_.Allocate(sizeof(ethtool_stats) + kMaxStats * sizeof(uint64_t));
}

Status EthtoolSource::Query(uint64_t* out) {
  auto* es = reinterpret_cast<ethtool_stats*>(stats_buf_.data());
  es->cmd = ETHTOOL_GSTATS;
  es->n_stats = static_cast<uint32_t>(names_.size());
  // On failure (including EFAULT at the guard page) the whole result is discarded.
  Status st = Ioctl(es, "GSTATS");
  if (!st.ok()) return st;
  const size_t got = es->n_stats;
  if (got > kMaxStats || got != names_.size()) {
    return Status::Error("ethtool stat count changed: " + std::to_string(names_.size()) + " -> " +
                         std::to_string(got) + " (queue reconfiguration?)");
  }
  std::memcpy(out, es->data, got * sizeof(uint64_t));
  return Status::Ok();
}

}  // namespace netmon

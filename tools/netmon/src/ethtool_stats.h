// `ethtool -S` statistics via SIOCETHTOOL (ETHTOOL_GSTRINGS / ETHTOOL_GSTATS).
//
// On ionic the ioctl is cheap (~1.2 us for ~600 stats) but the port (MAC) part is a firmware
// DMA region refreshed only every ~5 s; per-queue software stats are live.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "guarded_buffer.h"
#include "source.h"

namespace netmon {

struct DriverInfo {
  std::string driver, version, fw_version, bus_info;
};

// One-shot GDRVINFO; works for any netdev the caller can see.
Status QueryDriverInfo(const std::string& netdev, DriverInfo* out);

class EthtoolSource : public Source {
 public:
  // Capacity of the guarded buffers, in entries.
  static constexpr size_t kMaxStats = 131072;

  ~EthtoolSource() override;
  Status Open(const std::string& netdev);

  const char* kind() const override { return "ethtool"; }
  const std::vector<std::string>& names() const override { return names_; }
  double snapshot_fraction() const override { return 0.5; }
  Status Query(uint64_t* out) override;

 private:
  Status Ioctl(void* data, const char* what);

  int fd_ = -1;
  std::string netdev_;
  GuardedBuffer stats_buf_;
  std::vector<std::string> names_;
};

}  // namespace netmon

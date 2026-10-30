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
#include "umbp/common/nic_region_limit.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "mori/application/transport/rdma/providers/ionic/ionic_fw_version.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori::umbp {
namespace {

constexpr uint64_t kNicRegionPages = uint64_t{1} << 18;
constexpr application::IonicFwVersion kFixedFirmware{1, 117, 5, 190};

std::string Name(const std::optional<application::IonicFwVersion>& v) {
  if (!v) return "unknown";
  const auto& [major, minor, patch, build] = *v;
  return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch) + "-a-" +
         std::to_string(build);
}

// The firmware of this host's AINICs, the RDMA devices the ionic driver binds;
// nullopt for one whose version cannot be read.  Read once.
const std::vector<std::optional<application::IonicFwVersion>>& AinicFirmware() {
  static const auto versions = [] {
    std::vector<std::optional<application::IonicFwVersion>> found;
    std::error_code ec;
    for (const auto& dev : std::filesystem::directory_iterator("/sys/class/infiniband", ec)) {
      std::error_code link_ec;
      const auto driver = std::filesystem::read_symlink(dev.path() / "device" / "driver", link_ec);
      if (link_ec || driver.filename() != "ionic") continue;
      found.push_back(application::ReadIonicFwVersion(dev.path().filename().c_str()));
    }
    return found;
  }();
  return versions;
}

}  // namespace

void WarnIfNicRegionAliases(std::string_view what, uint64_t bytes, uint64_t page_size,
                            std::string_view remedy) {
  if (page_size == 0 || bytes / page_size <= kNicRegionPages) return;
  // Quiet with no AINIC here, or with every one at or past the fix.
  const auto& firmware = AinicFirmware();
  if (std::all_of(firmware.begin(), firmware.end(),
                  [](const auto& v) { return v && *v >= kFixedFirmware; })) {
    return;
  }
  std::set<std::string> names;
  for (const auto& v : firmware) names.insert(Name(v));
  std::string versions;
  for (const auto& name : names) versions += (versions.empty() ? "" : ", ") + name;
  MORI_UMBP_WARN(
      "[UMBP] {} of {} MiB on {} KiB pages is {} NIC page entries; AINIC firmware before {} "
      "translates only the first {} of a memory region and silently aliases the rest "
      "(this host's AINICs: {}). {}",
      what, bytes >> 20, page_size >> 10, bytes / page_size, Name(kFixedFirmware), kNicRegionPages,
      versions, remedy);
}

}  // namespace mori::umbp

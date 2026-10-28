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
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {
namespace {

constexpr uint64_t kNicRegionPages = uint64_t{1} << 18;
constexpr const char* kFixedFirmware = "1.117.5-a-190";

// The firmware of this host's AINICs, the RDMA devices the ionic driver binds;
// "unknown" for one whose version cannot be read.  Read once.
const std::set<std::string>& AinicFirmware() {
  static const std::set<std::string> versions = [] {
    std::set<std::string> found;
    std::error_code ec;
    for (const auto& dev : std::filesystem::directory_iterator("/sys/class/infiniband", ec)) {
      std::error_code link_ec;
      const auto driver = std::filesystem::read_symlink(dev.path() / "device" / "driver", link_ec);
      if (link_ec || driver.filename() != "ionic") continue;
      std::ifstream in(dev.path() / "fw_ver");
      std::string version;
      found.insert(std::getline(in, version) && !version.empty() ? version : "unknown");
    }
    return found;
  }();
  return versions;
}

// A version's numbers in order: "1.117.5-a-77" is 1 117 5 77.
std::vector<uint64_t> Numbers(const std::string& version) {
  std::vector<uint64_t> numbers;
  for (size_t i = 0; i < version.size();) {
    if (!std::isdigit(static_cast<unsigned char>(version[i]))) {
      ++i;
      continue;
    }
    uint64_t n = 0;
    for (; i < version.size() && std::isdigit(static_cast<unsigned char>(version[i])); ++i) {
      n = n * 10 + static_cast<uint64_t>(version[i] - '0');
    }
    numbers.push_back(n);
  }
  return numbers;
}

}  // namespace

bool AinicFirmwareTranslatesEveryPage(const std::string& version) {
  static const std::vector<uint64_t> fixed = Numbers(kFixedFirmware);
  return !(Numbers(version) < fixed);
}

void WarnIfNicRegionAliases(std::string_view what, uint64_t bytes, uint64_t page_size,
                            std::string_view remedy) {
  if (page_size == 0 || bytes / page_size <= kNicRegionPages) return;
  const auto& firmware = AinicFirmware();
  if (firmware.empty() ||
      std::all_of(firmware.begin(), firmware.end(), AinicFirmwareTranslatesEveryPage))
    return;
  std::string versions;
  for (const auto& version : firmware) versions += (versions.empty() ? "" : ", ") + version;
  MORI_UMBP_WARN(
      "[UMBP] {} of {} MiB on {} KiB pages is {} NIC page entries; AINIC firmware before {} "
      "translates only the first {} of a memory region and silently aliases the rest "
      "(this host's AINICs: {}). {}",
      what, bytes >> 20, page_size >> 10, bytes / page_size, kFixedFirmware, kNicRegionPages,
      versions, remedy);
}

}  // namespace mori::umbp

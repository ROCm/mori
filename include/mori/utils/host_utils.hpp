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
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

#include "mori/utils/env_utils.hpp"

namespace mori {

// First line of a sysfs/procfs file with trailing whitespace stripped, or empty
// if the file is absent or unreadable.
inline std::string ReadSysfsLine(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  if (f && std::getline(f, line)) {
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
    return line;
  }
  return {};
}

// Identical for all processes on one physical node, distinct across nodes;
// empty if unavailable.
inline std::string ReadKernelBootId() { return ReadSysfsLine("/proc/sys/kernel/random/boot_id"); }

// Firmware version of an RDMA device as reported by the kernel. Every vendor
// (bnxt_re, mlx5, ionic) exposes it at the same path, so this needs no
// per-provider special casing.
inline std::string ReadNicFirmware(const std::string& ibDev) {
  return ReadSysfsLine("/sys/class/infiniband/" + ibDev + "/fw_ver");
}

// PCI vendor id of an RDMA device, e.g. "0x14e4". Same key `mori check` uses to
// pick a vendor, and more reliable than the device name (ionic cards appear as
// ionic_*, roceensp*, rocep* depending on the host).
inline std::string ReadNicVendorId(const std::string& ibDev) {
  return ReadSysfsLine("/sys/class/infiniband/" + ibDev + "/device/vendor");
}

// Dotted-numeric version compare. Handles the AINIC "1.117.5-a-45" shape by
// treating any non-digit run as a separator, so it orders the trailing build
// number too. Returns <0, 0, >0 like strcmp.
inline int CompareFwVersion(const std::string& a, const std::string& b) {
  auto parts = [](const std::string& s) {
    std::vector<long> v;
    for (size_t i = 0; i < s.size();) {
      if (std::isdigit(static_cast<unsigned char>(s[i])) == 0) {
        ++i;
        continue;
      }
      size_t j = i;
      while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j])) != 0) ++j;
      v.push_back(std::stol(s.substr(i, j - i)));
      i = j;
    }
    return v;
  };
  const std::vector<long> va = parts(a), vb = parts(b);
  for (size_t i = 0; i < std::max(va.size(), vb.size()); ++i) {
    const long x = i < va.size() ? va[i] : 0;
    const long y = i < vb.size() ? vb[i] : 0;
    if (x != y) return x < y ? -1 : 1;
  }
  return 0;
}

// Verdict on a NIC firmware version, mirroring the table `mori check`
// (tools/env_check.sh) applies in Step 1. Kept in sync by hand: the shell tool
// and the library both need it and share no code, so the constants below must
// match AINIC_MIN_VER / BNXT_MIN_VER_235 / BNXT_MIN_VER_237 there.
//
// Ok      - meets a known-good minimum, or the vendor has no known minimum
// Unknown - empty or unparseable: a gap in detection, not a verdict
// Bad     - below a known-good minimum, on a known-bad branch, or on a branch
//           nobody has validated
enum class FwVerdict { Ok, Unknown, Bad };

struct FwCheck {
  FwVerdict verdict{FwVerdict::Unknown};
  std::string detail;
};

inline FwCheck CheckNicFirmware(const std::string& vendorId, const std::string& fw) {
  static constexpr const char* kAinicMin = "1.117.5-a-45";
  static constexpr const char* kBnxtMin235 = "235.2.86.0";
  static constexpr const char* kBnxtMin237 = "237.1.137.0";

  if (fw.empty()) return {FwVerdict::Unknown, "firmware version unavailable"};
  if (std::isdigit(static_cast<unsigned char>(fw[0])) == 0) {
    return {FwVerdict::Unknown, "cannot parse firmware version '" + fw + "'"};
  }

  if (vendorId == "0x1dd8") {  // AMD Pensando / ionic (AINIC)
    if (fw.rfind("1.117.1", 0) == 0 && (fw.size() == 7 || fw[7] == '.' || fw[7] == '-')) {
      return {FwVerdict::Bad, "AINIC firmware " + fw +
                                  " is on the 1.117.1 branch, which does NOT support IBGDA "
                                  "-- upgrade to >= " +
                                  kAinicMin};
    }
    if (CompareFwVersion(fw, kAinicMin) >= 0) return {FwVerdict::Ok, ""};
    return {FwVerdict::Bad, "AINIC firmware " + fw + " is below the required minimum (>= " +
                                kAinicMin + ") for cross-node IBGDA"};
  }

  if (vendorId == "0x14e4") {  // Broadcom bnxt_re
    const std::string major = fw.substr(0, fw.find('.'));
    if (major == "231") {
      return {FwVerdict::Bad, "Broadcom firmware " + fw +
                                  " is on the 231.x branch, which is too old for IBGDA "
                                  "-- upgrade to >= " +
                                  kBnxtMin235 + " or >= " + kBnxtMin237};
    }
    if (major == "232") {
      return {FwVerdict::Bad, "Broadcom firmware " + fw +
                                  " is on the 232.x branch, which is known not to work on Thor2 "
                                  "-- upgrade to >= " +
                                  kBnxtMin235 + " or >= " + kBnxtMin237};
    }
    const char* min = major == "235" ? kBnxtMin235 : (major == "237" ? kBnxtMin237 : nullptr);
    if (min == nullptr) {
      return {FwVerdict::Bad, "Broadcom firmware " + fw + " is on an unverified branch (" + major +
                                  ".x) -- known-good: >= " + kBnxtMin235 + " on 235.x, >= " +
                                  kBnxtMin237 + " on 237.x; known-bad: 231.x, 232.x"};
    }
    if (CompareFwVersion(fw, min) >= 0) return {FwVerdict::Ok, ""};
    return {FwVerdict::Bad, "Broadcom firmware " + fw + " is below the required minimum on the " +
                                major + ".x branch (>= " + min + ")"};
  }

  // mlx5 (0x15b3) and anything else: no minimum MORI knows of.
  return {FwVerdict::Ok, ""};
}

// MEC (MicroEngine Compute) firmware of the GPU at a PCI BDF such as
// "0000:e8:00.0" -- the form hipDeviceGetPCIBusId returns, which already
// matches the sysfs spelling. Keyed on BDF rather than a drm card index,
// because XCP-partitioned GPUs do not map onto /sys/class/drm/card<N> in any
// straightforward order. Returned as the raw hex sysfs value (e.g. 0x00000022),
// which rocm-smi renders in decimal as "MEC firmware version: 34".
inline std::string ReadGpuMecFirmware(const std::string& pciBdf) {
  return ReadSysfsLine("/sys/bus/pci/devices/" + pciBdf + "/fw_version/mec_fw_version");
}

// Per-physical-node identity, by priority: MORI_NODE_ID override, boot_id, then
// hostname. boot_id keeps it correct when machines share one hostname.
inline std::string ResolveNodeId(const std::string& hostname) {
  if (auto nodeId = env::GetString("MORI_NODE_ID"); nodeId.has_value()) {
    return *nodeId;
  }
  std::string bootId = ReadKernelBootId();
  if (!bootId.empty()) return bootId;
  return hostname;
}

}  // namespace mori

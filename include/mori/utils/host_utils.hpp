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

#include <cctype>
#include <fstream>
#include <string>

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
inline std::string ReadKernelBootId() {
  return ReadSysfsLine("/proc/sys/kernel/random/boot_id");
}

// Firmware version of an RDMA device as reported by the kernel. Every vendor
// (bnxt_re, mlx5, ionic) exposes it at the same path, so this needs no
// per-provider special casing.
inline std::string ReadNicFirmware(const std::string& ibDev) {
  return ReadSysfsLine("/sys/class/infiniband/" + ibDev + "/fw_ver");
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

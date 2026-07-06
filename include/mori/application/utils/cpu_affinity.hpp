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

#include <hip/hip_runtime_api.h>
#include <sched.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

namespace detail {

// Parse a Linux cpulist string like "0-3,8,12-15" into individual CPU ids.
inline std::vector<int> ParseCpuList(const std::string& s) {
  std::vector<int> cpus;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    size_t dash = tok.find('-');
    try {
      if (dash == std::string::npos) {
        cpus.push_back(std::stoi(tok));
      } else {
        int lo = std::stoi(tok.substr(0, dash));
        int hi = std::stoi(tok.substr(dash + 1));
        for (int c = lo; c <= hi; ++c) cpus.push_back(c);
      }
    } catch (...) {
    }
  }
  return cpus;
}

// CPUs local to a GPU, from its PCI BDF's sysfs local_cpulist -- the same source
// NCCL uses. numaNode is filled (for logging) when available.
inline std::optional<std::string> GpuLocalCpuList(int deviceId, int& numaNode) {
  char bdf[32] = {0};
  if (hipDeviceGetPCIBusId(bdf, sizeof(bdf), deviceId) != hipSuccess) return std::nullopt;
  std::string id(bdf);
  std::transform(id.begin(), id.end(), id.begin(), ::tolower);  // sysfs uses lowercase
  const std::string dir = "/sys/bus/pci/devices/" + id;
  {
    std::ifstream nf(dir + "/numa_node");
    if (nf) nf >> numaNode;
  }
  std::ifstream f(dir + "/local_cpulist");
  std::string line;
  if (!f || !std::getline(f, line) || line.empty()) return std::nullopt;
  return line;
}

}  // namespace detail

// Bind the CALLING thread to the CPUs local to its current HIP device's NUMA node,
// at most once per thread. Because the target NUMA node is derived from the
// thread's *current* device, this is correct under SPMT too (each op thread runs a
// single GPU): every per-GPU thread pins to its own GPU's CPUs without affecting
// the others. The GPU-local CPU set is intersected with the existing cpuset, so
// cgroup limits and any outer numactl/torchrun binding are respected, and binding
// is skipped if the intersection is empty (mirrors NCCL). Threads spawned by a
// bound thread inherit the affinity. Disable with MORI_IGNORE_CPU_AFFINITY=1.
inline void BindCallingThreadToGpuNumaOnce() {
  static thread_local bool attempted = false;
  if (attempted) return;
  attempted = true;
  if (env::IsEnvVarEnabled("MORI_IGNORE_CPU_AFFINITY")) return;

  int deviceId = -1;
  if (hipGetDevice(&deviceId) != hipSuccess) return;
  int numaNode = -1;
  std::optional<std::string> cpulist = detail::GpuLocalCpuList(deviceId, numaNode);
  if (!cpulist.has_value()) {
    MORI_APP_WARN("CPU affinity: cannot read GPU {} local_cpulist; skipping", deviceId);
    return;
  }
  std::vector<int> localCpus = detail::ParseCpuList(cpulist.value());

  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

  cpu_set_t target;
  CPU_ZERO(&target);
  int count = 0;
  for (int c : localCpus) {
    if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(c, &allowed)) {
      CPU_SET(c, &target);
      ++count;
    }
  }
  if (count == 0) {
    MORI_APP_WARN(
        "CPU affinity: GPU {} local CPU set empty after cpuset intersect; keeping current",
        deviceId);
    return;
  }
  if (sched_setaffinity(0, sizeof(target), &target) != 0) {
    MORI_APP_WARN("CPU affinity: sched_setaffinity for GPU {} failed: {}", deviceId,
                  strerror(errno));
    return;
  }
  MORI_APP_INFO("thread bound to {} CPUs local to GPU {} (numa node {})", count, deviceId,
                numaNode);
}

}  // namespace application
}  // namespace mori

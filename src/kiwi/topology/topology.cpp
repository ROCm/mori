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
#include "mori/kiwi/topology/topology.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifdef __linux__
#include <errno.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace mori::kiwi::p2p {
namespace topology {

namespace {

#ifdef __linux__

bool proxy_affinity_debug_enabled() {
  const char* value = std::getenv("KIWI_PROXY_AFFINITY_DEBUG");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::vector<int> cpus_in_set(const cpu_set_t& set) {
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

std::string summarize_cpus(const std::vector<int>& cpus) {
  std::string out = "[";
  for (size_t i = 0; i < cpus.size();) {
    const int begin = cpus[i];
    int end = begin;
    while (++i < cpus.size() && cpus[i] == end + 1) {
      end = cpus[i];
    }
    if (out.size() > 1) {
      out += ",";
    }
    out += std::to_string(begin);
    if (end != begin) {
      out += "-" + std::to_string(end);
    }
  }
  return out + "]";
}

// Linux CPU ids local to the current HIP device, parsed from its PCI sysfs
// local_cpulist (for example "56-111,168-223"). KFD's cpu_core_id_base is a
// hardware (APIC) id, not a Linux CPU id, so it is not used. Empty if the
// device or the file cannot be read.
std::vector<int> current_gpu_local_cpus(int* hip_device_out) {
  int hip_device = -1;
  char bus_id[64] = {};
  if (hipGetDevice(&hip_device) != hipSuccess ||
      hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), hip_device) != hipSuccess) {
    return {};
  }
  *hip_device_out = hip_device;
  std::string path = std::string("/sys/bus/pci/devices/") + bus_id;
  std::transform(path.begin(), path.end(), path.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::ifstream in(path + "/local_cpulist");
  std::vector<int> cpus;
  std::string range;
  while (std::getline(in, range, ',')) {
    int first = -1;
    int last = -1;
    const int fields = std::sscanf(range.c_str(), "%d-%d", &first, &last);
    if (fields == 1) {
      last = first;
    }
    if (fields < 1 || first < 0 || last < first) {
      return {};
    }
    for (int cpu = first; cpu <= last; ++cpu) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

#endif

}  // namespace

std::vector<int> proxy_thread_cpu_set() {
#ifdef __linux__
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    return {};
  }
  int hip_device = -1;
  const std::vector<int> local = current_gpu_local_cpus(&hip_device);
  std::vector<int> cpus;
  for (int cpu : local) {
    if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
      cpus.push_back(cpu);
    }
  }
  std::sort(cpus.begin(), cpus.end());
  cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
  if (proxy_affinity_debug_enabled()) {
    std::fprintf(stderr,
                 "KIWI_PROXY_AFFINITY_DEBUG pid=%ld hip_device=%d gpu_local=%s "
                 "allowed=%s proxy_set=%s\n",
                 static_cast<long>(getpid()), hip_device, summarize_cpus(local).c_str(),
                 summarize_cpus(cpus_in_set(allowed)).c_str(),
                 cpus.empty() ? "inherited" : summarize_cpus(cpus).c_str());
  }
  return cpus;
#else
  return {};
#endif
}

void bind_current_thread_to_cpus(const std::vector<int>& cpus) {
#ifdef __linux__
  if (cpus.empty()) {
    return;
  }
  cpu_set_t target;
  CPU_ZERO(&target);
  for (int cpu : cpus) {
    if (cpu >= 0 && cpu < CPU_SETSIZE) {
      CPU_SET(cpu, &target);
    }
  }
  if (sched_setaffinity(0, sizeof(target), &target) != 0) {
    std::fprintf(stderr, "ERROR: failed to bind proxy thread to CPUs %s: %s\n",
                 summarize_cpus(cpus).c_str(), std::strerror(errno));
    std::abort();
  }
  if (proxy_affinity_debug_enabled()) {
    cpu_set_t actual;
    CPU_ZERO(&actual);
    sched_getaffinity(0, sizeof(actual), &actual);
    std::fprintf(stderr, "KIWI_PROXY_AFFINITY_DEBUG pid=%ld tid=%ld bound=%s\n",
                 static_cast<long>(getpid()), static_cast<long>(gettid()),
                 summarize_cpus(cpus_in_set(actual)).c_str());
  }
#else
  (void)cpus;
#endif
}

}  // namespace topology
}  // namespace mori::kiwi::p2p

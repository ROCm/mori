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
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef __linux__
#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <unistd.h>
#endif

#undef HIP_CHECK
#define HIP_CHECK(call)                                                                        \
  do {                                                                                         \
    hipError_t err = (call);                                                                   \
    if (err != hipSuccess) {                                                                   \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      abort();                                                                                 \
    }                                                                                          \
  } while (0)

namespace mori::kiwi::p2p {
namespace topology {

namespace {

#ifdef __linux__

struct pci_bdf_t {
  int domain = 0;
  int bus = -1;
  int device = -1;
  int function = -1;
};

struct kfd_node_t {
  int node_id = -1;
  int cpu_core_id_base = -1;
  int cpu_cores_count = 0;
  long long location_id = -1;
  long long domain = -1;
};

std::optional<long long> parse_integer(const std::string& value) {
  char* end = nullptr;
  errno = 0;
  long long parsed = std::strtoll(value.c_str(), &end, 0);
  if (errno != 0 || end == value.c_str()) {
    return std::nullopt;
  }
  return parsed;
}

std::unordered_map<std::string, long long> read_properties(const std::string& path) {
  std::unordered_map<std::string, long long> props;
  std::ifstream in(path);
  std::string key;
  std::string value;
  while (in >> key >> value) {
    auto parsed = parse_integer(value);
    if (parsed) {
      props.emplace(std::move(key), *parsed);
    }
  }
  return props;
}

template <typename MapT>
long long get_prop_or(const MapT& props, const std::string& key, long long fallback) {
  auto it = props.find(key);
  return it == props.end() ? fallback : it->second;
}

std::vector<int> list_numeric_dirs(const std::string& path) {
  std::vector<int> ids;
  DIR* dir = opendir(path.c_str());
  if (!dir) {
    return ids;
  }
  while (dirent* ent = readdir(dir)) {
    if (ent->d_name[0] == '.') {
      continue;
    }
    char* end = nullptr;
    long parsed = std::strtol(ent->d_name, &end, 10);
    if (end != ent->d_name && *end == '\0' && parsed >= 0) {
      ids.push_back(static_cast<int>(parsed));
    }
  }
  closedir(dir);
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::optional<int> get_current_hip_device() {
  int hip_device = -1;
  hipError_t err = hipGetDevice(&hip_device);
  if (err != hipSuccess || hip_device < 0) {
    return std::nullopt;
  }
  return hip_device;
}

std::optional<pci_bdf_t> get_gpu_bdf(int hip_device) {
  char bus_id[64] = {};
  hipError_t err = hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), hip_device);
  if (err != hipSuccess) {
    return std::nullopt;
  }

  pci_bdf_t bdf;
  int parsed =
      std::sscanf(bus_id, "%x:%x:%x.%x", &bdf.domain, &bdf.bus, &bdf.device, &bdf.function);
  if (parsed != 4) {
    bdf.domain = 0;
    parsed = std::sscanf(bus_id, "%x:%x.%x", &bdf.bus, &bdf.device, &bdf.function);
    if (parsed != 3) {
      return std::nullopt;
    }
  }
  return bdf;
}

// Linux CPU ids local to a PCI device, parsed from its sysfs local_cpulist
// (for example "56-111,168-223"). Empty if the file is missing or malformed.
std::vector<int> read_pci_local_cpus(const pci_bdf_t& bdf) {
  char path[96];
  std::snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%x/local_cpulist",
                bdf.domain, bdf.bus, bdf.device, bdf.function);
  std::ifstream in(path);
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

bool location_matches_bdf(const kfd_node_t& node, const pci_bdf_t& bdf) {
  if (node.location_id < 0 || bdf.bus < 0 || bdf.device < 0 || bdf.function < 0) {
    return false;
  }

  // KFD location_id is the PCI BDF without the domain on current ROCm
  // systems: bus:device.function encoded as bus << 8 | device << 3 | fn.
  // Also accept a domain-extended form for portability across kernels.
  const long long bdf_no_domain = (static_cast<long long>(bdf.bus) << 8) |
                                  (static_cast<long long>(bdf.device) << 3) |
                                  static_cast<long long>(bdf.function);
  const long long bdf_with_domain = (static_cast<long long>(bdf.domain) << 16) | bdf_no_domain;

  if (node.location_id == bdf_no_domain || node.location_id == bdf_with_domain) {
    return node.domain < 0 || node.domain == bdf.domain;
  }
  return false;
}

std::map<int, kfd_node_t> read_kfd_nodes() {
  static constexpr const char* kTopologyRoot = "/sys/class/kfd/kfd/topology/nodes";
  std::map<int, kfd_node_t> nodes;
  for (int node_id : list_numeric_dirs(kTopologyRoot)) {
    auto props =
        read_properties(std::string(kTopologyRoot) + "/" + std::to_string(node_id) + "/properties");
    if (props.empty()) {
      continue;
    }
    kfd_node_t node;
    node.node_id = node_id;
    node.cpu_core_id_base = static_cast<int>(get_prop_or(props, "cpu_core_id_base", -1));
    node.cpu_cores_count = static_cast<int>(get_prop_or(props, "cpu_cores_count", 0));
    node.location_id = get_prop_or(props, "location_id", -1);
    node.domain = get_prop_or(props, "domain", -1);
    nodes.emplace(node_id, node);
  }
  return nodes;
}

std::optional<int> find_linked_cpu_node(int gpu_node_id, const std::map<int, kfd_node_t>& nodes) {
  static constexpr const char* kTopologyRoot = "/sys/class/kfd/kfd/topology/nodes";

  auto choose_cpu_endpoint =
      [&](const std::unordered_map<std::string, long long>& props) -> std::optional<int> {
    if (get_prop_or(props, "type", -1) != 2) {
      return std::nullopt;
    }
    const int from = static_cast<int>(get_prop_or(props, "node_from", -1));
    const int to = static_cast<int>(get_prop_or(props, "node_to", -1));
    int candidate = -1;
    if (from == gpu_node_id) {
      candidate = to;
    } else if (to == gpu_node_id) {
      candidate = from;
    }
    auto it = nodes.find(candidate);
    if (it != nodes.end() && it->second.cpu_cores_count > 0 && it->second.cpu_core_id_base >= 0) {
      return candidate;
    }
    return std::nullopt;
  };

  // Prefer links listed under the GPU node itself.
  const std::string gpu_links_dir =
      std::string(kTopologyRoot) + "/" + std::to_string(gpu_node_id) + "/io_links";
  for (int link_id : list_numeric_dirs(gpu_links_dir)) {
    auto props = read_properties(gpu_links_dir + "/" + std::to_string(link_id) + "/properties");
    if (auto cpu_node = choose_cpu_endpoint(props)) {
      return cpu_node;
    }
  }

  // Some topologies expose the CPU<->GPU edge from the CPU node side only.
  for (const auto& [node_id, node] : nodes) {
    (void)node;
    const std::string links_dir =
        std::string(kTopologyRoot) + "/" + std::to_string(node_id) + "/io_links";
    for (int link_id : list_numeric_dirs(links_dir)) {
      auto props = read_properties(links_dir + "/" + std::to_string(link_id) + "/properties");
      if (auto cpu_node = choose_cpu_endpoint(props)) {
        return cpu_node;
      }
    }
  }

  return std::nullopt;
}

std::vector<int> current_allowed_cpus() {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    return {};
  }

  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

bool proxy_affinity_debug_enabled() {
  const char* value = std::getenv("KIWI_PROXY_AFFINITY_DEBUG");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::string summarize_cpus(const std::vector<int>& cpus) {
  if (cpus.empty()) {
    return "[]";
  }

  std::vector<int> sorted = cpus;
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

  std::string out = "[";
  for (size_t i = 0; i < sorted.size();) {
    const int begin = sorted[i];
    int end = begin;
    ++i;
    while (i < sorted.size() && sorted[i] == end + 1) {
      end = sorted[i];
      ++i;
    }
    if (out.size() > 1) {
      out += ",";
    }
    out += std::to_string(begin);
    if (end != begin) {
      out += "-" + std::to_string(end);
    }
  }
  out += "]";
  return out;
}

std::string bdf_to_string(const pci_bdf_t& bdf) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x", bdf.domain, bdf.bus, bdf.device,
                bdf.function);
  return std::string(buf);
}

std::string current_rank_debug_summary() {
  int device = -1;
  (void)hipGetDevice(&device);
  return std::string("pid=") + std::to_string(getpid()) + " hip_device=" + std::to_string(device);
}

std::vector<int> intersect_cpus_preserve_order(const std::vector<int>& cpus,
                                               const std::vector<int>& allowed) {
  std::vector<int> out;
  out.reserve(cpus.size());
  for (int cpu : cpus) {
    if (std::find(allowed.begin(), allowed.end(), cpu) != allowed.end()) {
      out.push_back(cpu);
    }
  }
  return out;
}

std::vector<int> affinity_summary_for_current_thread() { return current_allowed_cpus(); }

struct gpu_locality_t {
  int hip_device = -1;
  pci_bdf_t bdf;
  bool has_bdf = false;
  int gpu_node_id = -1;
  int cpu_node_id = -1;
  int gpu_slot_on_cpu_node = -1;
  int gpus_on_cpu_node = 0;
  std::vector<int> local_cpus;
};

struct gpu_slot_candidate_t {
  int node_id = -1;
  long long domain = -1;
  long long location_id = -1;
};

gpu_locality_t query_current_gpu_locality() {
  gpu_locality_t locality;
  auto hip_device = get_current_hip_device();
  if (!hip_device) {
    return locality;
  }
  locality.hip_device = *hip_device;

  auto bdf = get_gpu_bdf(*hip_device);
  if (!bdf) {
    return locality;
  }
  locality.bdf = *bdf;
  locality.has_bdf = true;
  // KFD's cpu_core_id_base is a hardware (APIC) id, not a Linux CPU id: on
  // two-socket SMT hosts the second socket reports 128 while Linux numbers
  // its CPUs 56-111,168-223. The PCI device's local_cpulist uses Linux ids.
  locality.local_cpus = read_pci_local_cpus(*bdf);

  auto nodes = read_kfd_nodes();
  for (const auto& [node_id, node] : nodes) {
    if (node.cpu_cores_count == 0 && location_matches_bdf(node, *bdf)) {
      locality.gpu_node_id = node_id;
      break;
    }
  }
  if (locality.gpu_node_id < 0) {
    return locality;
  }

  auto cpu_node_id = find_linked_cpu_node(locality.gpu_node_id, nodes);
  if (!cpu_node_id) {
    return locality;
  }
  locality.cpu_node_id = *cpu_node_id;

  auto cpu_node = nodes.find(*cpu_node_id);
  if (cpu_node == nodes.end() || cpu_node->second.cpu_core_id_base < 0 ||
      cpu_node->second.cpu_cores_count <= 0) {
    return locality;
  }

  std::vector<gpu_slot_candidate_t> peers;
  for (const auto& [node_id, node] : nodes) {
    if (node.cpu_cores_count != 0) {
      continue;
    }
    auto peer_cpu_node_id = find_linked_cpu_node(node_id, nodes);
    if (!peer_cpu_node_id || *peer_cpu_node_id != *cpu_node_id) {
      continue;
    }
    peers.push_back({node_id, node.domain, node.location_id});
  }

  std::sort(peers.begin(), peers.end(),
            [](const gpu_slot_candidate_t& a, const gpu_slot_candidate_t& b) {
              if (a.domain != b.domain) {
                return a.domain < b.domain;
              }
              if (a.location_id != b.location_id) {
                return a.location_id < b.location_id;
              }
              return a.node_id < b.node_id;
            });

  locality.gpus_on_cpu_node = static_cast<int>(peers.size());
  for (size_t i = 0; i < peers.size(); ++i) {
    if (peers[i].node_id == locality.gpu_node_id) {
      locality.gpu_slot_on_cpu_node = static_cast<int>(i);
      break;
    }
  }

  return locality;
}

int select_partitioned_cpu(const std::vector<int>& cpus, int gpu_slot, int hip_device,
                           int num_proxy_threads, int thread_id) {
  if (cpus.empty()) {
    return -1;
  }
  const int slot = gpu_slot >= 0 ? gpu_slot : std::max(hip_device, 0);
  const size_t offset =
      (static_cast<size_t>(slot) * static_cast<size_t>(std::max(num_proxy_threads, 1)) +
       static_cast<size_t>(std::max(thread_id, 0))) %
      cpus.size();
  return cpus[offset];
}

void debug_log_proxy_choice(const gpu_locality_t& locality, const std::vector<int>& allowed,
                            const std::vector<int>& local_allowed, int thread_id,
                            int num_proxy_threads, int selected_cpu, const char* selected_source) {
  if (!proxy_affinity_debug_enabled()) {
    return;
  }
  std::fprintf(
      stderr,
      "KIWI_PROXY_AFFINITY_DEBUG choose pid=%ld %s hip_device=%d bdf=%s "
      "kfd_gpu_node=%d kfd_cpu_node=%d gpu_slot=%d gpus_on_cpu_node=%d "
      "thread=%d/%d selected_source=%s selected_cpu=%d "
      "local_cpus=%s local_allowed=%s allowed=%s\n",
      static_cast<long>(getpid()), current_rank_debug_summary().c_str(), locality.hip_device,
      locality.has_bdf ? bdf_to_string(locality.bdf).c_str() : "unknown", locality.gpu_node_id,
      locality.cpu_node_id, locality.gpu_slot_on_cpu_node, locality.gpus_on_cpu_node, thread_id,
      num_proxy_threads, selected_source, selected_cpu, summarize_cpus(locality.local_cpus).c_str(),
      summarize_cpus(local_allowed).c_str(), summarize_cpus(allowed).c_str());
}

std::vector<int> build_proxy_thread_cpus(int num_proxy_threads) {
  if (num_proxy_threads <= 0) {
    return {};
  }
  std::vector<int> assignments(static_cast<size_t>(num_proxy_threads), -1);

  const std::vector<int> allowed = current_allowed_cpus();
  const gpu_locality_t locality = query_current_gpu_locality();
  const std::vector<int> local_allowed =
      intersect_cpus_preserve_order(locality.local_cpus, allowed);
  // Partition the *pin-able* local CPU mask when available. Some cluster
  // launch paths expose holes in the NUMA-local CPU range, for example CPU1-2
  // unavailable while CPU0,3-55 are allowed. Selecting only from CPUs that are
  // already in the process affinity mask avoids per-thread fallback pinning.
  const bool use_local_allowed = !local_allowed.empty();
  const std::vector<int>& selected_pool = use_local_allowed ? local_allowed : allowed;
  const char* selected_source =
      use_local_allowed ? "local_pci_allowed_cpu_slice" : "allowed_cpu_slice_by_hip_device";
  const int selected_slot = use_local_allowed ? locality.gpu_slot_on_cpu_node : locality.hip_device;

  for (int thread_id = 0; thread_id < num_proxy_threads; ++thread_id) {
    const int selected_cpu = select_partitioned_cpu(
        selected_pool, selected_slot, locality.hip_device, num_proxy_threads, thread_id);
    assignments[static_cast<size_t>(thread_id)] = selected_cpu;
    debug_log_proxy_choice(locality, allowed, local_allowed, thread_id, num_proxy_threads,
                           selected_cpu, selected_source);
  }
  return assignments;
}

int try_pin_to_cpu(int selected_cpu) {
  if (selected_cpu < 0 || selected_cpu >= CPU_SETSIZE) {
    errno = EINVAL;
    return -1;
  }

  cpu_set_t target;
  CPU_ZERO(&target);
  CPU_SET(selected_cpu, &target);
  return sched_setaffinity(0, sizeof(target), &target);
}

void debug_log_proxy_pin(int thread_id, int num_proxy_threads, const char* phase, int selected_cpu,
                         int rc, int saved_errno) {
  if (!proxy_affinity_debug_enabled()) {
    return;
  }

  const auto locality = query_current_gpu_locality();
  const std::vector<int> actual = affinity_summary_for_current_thread();
  std::fprintf(stderr,
               "KIWI_PROXY_AFFINITY_DEBUG pin pid=%ld %s phase=%s "
               "thread=%d/%d selected_cpu=%d sched_setaffinity_rc=%d errno=%d (%s) "
               "actual_affinity=%s hip_device=%d bdf=%s kfd_gpu_node=%d "
               "kfd_cpu_node=%d gpu_slot=%d\n",
               static_cast<long>(getpid()), current_rank_debug_summary().c_str(), phase, thread_id,
               num_proxy_threads, selected_cpu, rc, saved_errno,
               rc == 0 ? "success" : std::strerror(saved_errno), summarize_cpus(actual).c_str(),
               locality.hip_device,
               locality.has_bdf ? bdf_to_string(locality.bdf).c_str() : "unknown",
               locality.gpu_node_id, locality.cpu_node_id, locality.gpu_slot_on_cpu_node);
}

void pin_selected_proxy_thread_to_cpu(int thread_id, int num_proxy_threads, int selected_cpu) {
  if (selected_cpu < 0) {
    debug_log_proxy_pin(thread_id, num_proxy_threads, "none", -1, -1, EINVAL);
    return;
  }

  errno = 0;
  const int rc = try_pin_to_cpu(selected_cpu);
  const int saved_errno = rc == 0 ? 0 : errno;
  debug_log_proxy_pin(thread_id, num_proxy_threads, "selected", selected_cpu, rc, saved_errno);
  if (rc != 0) {
    std::fprintf(stderr, "ERROR: failed to pin proxy thread %d/%d to CPU core %d: %s\n", thread_id,
                 num_proxy_threads, selected_cpu, std::strerror(saved_errno));
    std::abort();
  }
}

#else

std::vector<int> build_proxy_thread_cpus(int num_proxy_threads) {
  if (num_proxy_threads <= 0) {
    return {};
  }
  return std::vector<int>(static_cast<size_t>(num_proxy_threads), -1);
}

void pin_selected_proxy_thread_to_cpu(int thread_id, int num_proxy_threads, int selected_cpu) {
  (void)thread_id;
  (void)num_proxy_threads;
  (void)selected_cpu;
}

#endif

}  // namespace

std::vector<int> choose_proxy_thread_cpus(int num_proxy_threads) {
  return build_proxy_thread_cpus(num_proxy_threads);
}

void pin_proxy_thread_to_cpu(int thread_id, int num_proxy_threads, int selected_cpu) {
  pin_selected_proxy_thread_to_cpu(thread_id, num_proxy_threads, selected_cpu);
}

namespace nic_selection_detail {

struct pci_location_t {
  std::string bus_id;
  std::filesystem::path canonical_path;
  int numa_node = -1;

  bool is_valid() const { return !bus_id.empty() && !canonical_path.empty(); }
};

struct hca_info_t {
  std::string name;
  pci_location_t pci;
  double total_gbps = 0.0;
};

struct hca_rank_key_t {
  double total_gbps = 0.0;
  bool same_numa = false;
  int hops = std::numeric_limits<int>::max();
};

inline std::string trim_copy(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    begin++;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    end--;
  }
  return value.substr(begin, end - begin);
}

inline std::string lowercase_ascii(std::string value) {
  for (char& c : value) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return value;
}

inline std::string read_first_line(const std::filesystem::path& path) {
  std::ifstream f(path);
  std::string line;
  if (!f.is_open() || !std::getline(f, line)) return {};
  return trim_copy(line);
}

inline int read_int_file(const std::filesystem::path& path, int fallback = -1) {
  std::ifstream f(path);
  int value = fallback;
  if (!(f >> value)) return fallback;
  return value;
}

inline double read_rate_gbps(const std::filesystem::path& path) {
  std::ifstream f(path);
  double value = 0.0;
  if (!(f >> value)) return 0.0;
  return value;
}

inline std::filesystem::path canonical_or_empty(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path resolved = std::filesystem::canonical(path, ec);
  if (ec) return {};
  return resolved;
}

inline bool is_bdf_like(const std::string& name) {
  int colon_count = 0;
  int dot_count = 0;
  for (char c : name) {
    if (c == ':')
      colon_count++;
    else if (c == '.')
      dot_count++;
  }
  return (colon_count == 2 || colon_count == 1) && dot_count == 1;
}

inline std::vector<std::string> expand_bus_id_candidates(const std::string& raw_bus_id) {
  std::string bus_id = lowercase_ascii(trim_copy(raw_bus_id));
  std::vector<std::string> candidates;
  if (!bus_id.empty()) candidates.push_back(bus_id);
  if (!bus_id.empty() && is_bdf_like(bus_id) && bus_id.find(':') == bus_id.rfind(':')) {
    candidates.push_back("0000:" + bus_id);
  }
  return candidates;
}

inline pci_location_t resolve_pci_location(const std::string& raw_bus_id) {
  for (const std::string& candidate : expand_bus_id_candidates(raw_bus_id)) {
    std::filesystem::path dev_path =
        canonical_or_empty(std::filesystem::path("/sys/bus/pci/devices") / candidate);
    if (dev_path.empty()) continue;
    return {
        candidate, dev_path,
        read_int_file(std::filesystem::path("/sys/bus/pci/devices") / candidate / "numa_node", -1)};
  }
  return {};
}

inline bool is_active_port(const std::filesystem::path& port_path) {
  std::string state = read_first_line(port_path / "state");
  return state.find("ACTIVE") != std::string::npos;
}

inline std::vector<std::string> split_path_components(const std::filesystem::path& path) {
  std::vector<std::string> parts;
  for (const auto& part : path) parts.push_back(part.string());
  return parts;
}

inline int pcie_hops(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
  if (lhs.empty() || rhs.empty()) return std::numeric_limits<int>::max();

  std::vector<std::string> lhs_parts = split_path_components(lhs);
  std::vector<std::string> rhs_parts = split_path_components(rhs);
  size_t common = 0;
  while (common < lhs_parts.size() && common < rhs_parts.size() &&
         lhs_parts[common] == rhs_parts[common]) {
    common++;
  }
  size_t hops = (lhs_parts.size() - common) + (rhs_parts.size() - common);
  if (hops > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(hops);
}

inline pci_location_t get_gpu_pci_location(int gpu_id) {
  char pci_id[64] = {};
  if (hipDeviceGetPCIBusId(pci_id, sizeof(pci_id), gpu_id) != hipSuccess) {
    return {};
  }
  return resolve_pci_location(pci_id);
}

inline std::vector<hca_info_t> enumerate_active_hcas() {
  std::vector<hca_info_t> hcas;
  std::filesystem::path hca_root("/sys/class/infiniband");
  std::error_code root_ec;
  if (!std::filesystem::exists(hca_root, root_ec)) return hcas;

  std::error_code list_ec;
  for (const auto& entry : std::filesystem::directory_iterator(hca_root, list_ec)) {
    if (list_ec) break;

    hca_info_t hca;
    hca.name = entry.path().filename().string();

    std::filesystem::path dev_path = canonical_or_empty(entry.path() / "device");
    if (dev_path.empty()) continue;

    hca.pci = resolve_pci_location(dev_path.filename().string());
    if (hca.pci.bus_id.empty()) {
      hca.pci.bus_id = lowercase_ascii(dev_path.filename().string());
      hca.pci.canonical_path = dev_path;
      hca.pci.numa_node = read_int_file(
          std::filesystem::path("/sys/bus/pci/devices") / hca.pci.bus_id / "numa_node", -1);
    }

    std::filesystem::path ports_dir = entry.path() / "ports";
    bool has_active_port = false;
    std::error_code ports_ec;
    for (const auto& port_entry : std::filesystem::directory_iterator(ports_dir, ports_ec)) {
      if (ports_ec) break;
      if (!is_active_port(port_entry.path())) continue;
      has_active_port = true;
      hca.total_gbps += read_rate_gbps(port_entry.path() / "rate");
    }

    if (!has_active_port) continue;
    hcas.push_back(std::move(hca));
  }

  std::sort(hcas.begin(), hcas.end(),
            [](const hca_info_t& a, const hca_info_t& b) { return a.name < b.name; });
  return hcas;
}

inline hca_rank_key_t rank_hca_for_gpu(const hca_info_t& hca, const pci_location_t& gpu) {
  hca_rank_key_t key;
  key.total_gbps = hca.total_gbps;
  key.same_numa = gpu.numa_node >= 0 && hca.pci.numa_node == gpu.numa_node;
  key.hops = pcie_hops(gpu.canonical_path, hca.pci.canonical_path);
  return key;
}

inline bool same_rank_key(const hca_rank_key_t& a, const hca_rank_key_t& b) {
  return a.total_gbps == b.total_gbps && a.same_numa == b.same_numa && a.hops == b.hops;
}

inline bool better_rank_key(const hca_rank_key_t& a, const hca_rank_key_t& b) {
  if (a.total_gbps != b.total_gbps) return a.total_gbps > b.total_gbps;
  if (a.same_numa != b.same_numa) return a.same_numa && !b.same_numa;
  if (a.hops != b.hops) return a.hops < b.hops;
  return false;
}

inline std::vector<std::string> build_gpu_hca_map() {
  int num_gpus = 0;
  if (hipGetDeviceCount(&num_gpus) != hipSuccess || num_gpus <= 0) return {};

  std::vector<hca_info_t> hcas = enumerate_active_hcas();
  std::vector<std::string> mapping(num_gpus);
  if (hcas.empty()) return mapping;

  std::vector<int> assigned_count(hcas.size(), 0);
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    pci_location_t gpu_pci = get_gpu_pci_location(gpu);
    if (!gpu_pci.is_valid()) {
      mapping[gpu] = hcas[gpu % hcas.size()].name;
      assigned_count[gpu % hcas.size()]++;
      continue;
    }

    std::vector<int> candidate_indices(hcas.size());
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
    std::sort(candidate_indices.begin(), candidate_indices.end(), [&](int lhs, int rhs) {
      hca_rank_key_t lhs_key = rank_hca_for_gpu(hcas[lhs], gpu_pci);
      hca_rank_key_t rhs_key = rank_hca_for_gpu(hcas[rhs], gpu_pci);
      if (better_rank_key(lhs_key, rhs_key)) return true;
      if (better_rank_key(rhs_key, lhs_key)) return false;
      return hcas[lhs].name < hcas[rhs].name;
    });

    int selected = candidate_indices.front();
    hca_rank_key_t best_key = rank_hca_for_gpu(hcas[selected], gpu_pci);
    for (int idx : candidate_indices) {
      hca_rank_key_t key = rank_hca_for_gpu(hcas[idx], gpu_pci);
      if (!same_rank_key(key, best_key)) break;
      if (assigned_count[idx] < assigned_count[selected] ||
          (assigned_count[idx] == assigned_count[selected] &&
           hcas[idx].name < hcas[selected].name)) {
        selected = idx;
      }
    }

    assigned_count[selected]++;
    mapping[gpu] = hcas[selected].name;
  }
  return mapping;
}

}  // namespace nic_selection_detail

// Return the nearest active HCA/NIC name for a HIP GPU id.  The mapping is
// computed across all visible GPUs and balances exact ties so ranks do not all
// bind to the same NIC when several equally-near HCAs are available.
std::string get_hca_name_for_gpu(int gpu_id) {
  static const std::vector<std::string> gpu_to_hca = nic_selection_detail::build_gpu_hca_map();
  if (gpu_id < 0 || gpu_id >= static_cast<int>(gpu_to_hca.size())) return {};
  return gpu_to_hca[gpu_id];
}

std::string get_hca_name_for_current_gpu() {
  int gpu_id = -1;
  if (hipGetDevice(&gpu_id) != hipSuccess) return {};
  return get_hca_name_for_gpu(gpu_id);
}

std::string get_requested_or_auto_hca_name_for_current_gpu() {
  const char* requested = std::getenv("MORI_KIWI_DEVICE_NAME");
  if (requested != nullptr && requested[0] != '\0') return requested;
  return get_hca_name_for_current_gpu();
}

}  // namespace topology
}  // namespace mori::kiwi::p2p

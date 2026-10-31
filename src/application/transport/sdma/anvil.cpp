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
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#include "mori/application/transport/sdma/anvil.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "mori/utils/mori_log.hpp"
namespace anvil {

namespace {

#define CHECK_HSA_ERROR(cmd)                                                                 \
  do {                                                                                       \
    if (auto s = (cmd); s != HSA_STATUS_SUCCESS) {                                           \
      const char* hsa_err_msg;                                                               \
      hsa_status_string(s, &hsa_err_msg);                                                    \
      throw std::runtime_error{std::string("HSA error at " __FILE__ ":") +                   \
                               std::to_string(__LINE__) + std::string(" - ") + hsa_err_msg}; \
    }                                                                                        \
  } while (0)

// `call` is evaluated ONCE, into _st. It used to appear twice -- in the test and
// again in the error message -- so every failure ran the KFD call a second time:
// hsaKmtAllocMemory leaked the first block, hsaKmtFreeMemory double-freed,
// hsaKmtCreateQueueExt asked for a second queue, and the code printed was the
// RETRY's, which need not equal the one that actually failed.
#define CHECK_HSAKMT_SUCCESS(call, msg)                                                   \
  do {                                                                                    \
    const HSAKMT_STATUS _st = (call);                                                     \
    if (_st != HSAKMT_STATUS_SUCCESS) {                                                   \
      std::cout << "ERROR code: " << std::dec << _st << " " << msg << " (File: " __FILE__ \
                << ", Line: " << __LINE__ << ")" << std::endl;                            \
      exit(EXIT_FAILURE);                                                                 \
    }                                                                                     \
  } while (0)

// HSA agents
std::vector<hsa_agent_t> cpuAgents_;
std::vector<hsa_agent_t> gpuAgents_;

hsa_status_t rocm_hsa_agent_callback(hsa_agent_t agent, hsa_device_type_t target_device_type,
                                     [[maybe_unused]] void* vector) {
  std::vector<hsa_agent_t>* agents = static_cast<std::vector<hsa_agent_t>*>(vector);
  hsa_device_type_t device_type{};
  hsa_status_t status{hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)};
  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get device type: 0x%x", status);
    return status;
  }
  if (device_type == target_device_type) {
    agents->push_back(agent);
  }
  return status;
}

hsa_status_t rocm_hsa_gpu_agent_callback(hsa_agent_t agent, [[maybe_unused]] void* context) {
  return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_GPU, context);
}
hsa_status_t rocm_hsa_cpu_agent_callback(hsa_agent_t agent, [[maybe_unused]] void* context) {
  return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_CPU, context);
}

void SetUpKFD() {
  CHECK_HSAKMT_SUCCESS(hsaKmtOpenKFD(), "hsaKmtOpenKFD() failed!");
  HsaSystemProperties m_SystemProperties;
  memset(&m_SystemProperties, 0, sizeof(m_SystemProperties));
  CHECK_HSAKMT_SUCCESS(hsaKmtAcquireSystemProperties(&m_SystemProperties), "Failed!");
}

// void SetUpKFD(uint32_t targetDevice) {
//     HsaNodeProperties m_node_props;
//     CHECK_HSAKMT_SUCCESS(hsaKmtGetNodeProperties(targetDevice, &m_node_props), "Failed!");
//     std::cout << "Num of PCIe SDMA Queues: " << m_node_props.NumSdmaEngines << std::endl;
//     std::cout << "Num of XGMI SDMA Queues: " << m_node_props.NumSdmaXgmiEngines << std::endl;
//     std::cout << "Device Id: " << m_node_props.DeviceId << std::endl;
// }

void CloseKFD() { (void)hsaKmtCloseKFD(); }

// PCI bus id ("domain:bus:dev.func") of a HIP device ordinal.
// Optionally returns the domain.
uint32_t getBusId(int deviceId, uint32_t* pdomain = nullptr) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busId[] = "00000000:00:00.0";
  CHECK_HIP_ERROR(hipDeviceGetPCIBusId(busId, sizeof(busId), deviceId));
  uint32_t domain = 0, bus = 0, dev = 0, func = 0;
  if (std::sscanf(busId, "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4) {
    MORI_APP_ERROR("Failed to parse PCI bus ID for device {}", deviceId);
    return ~0u;
  }
  if (pdomain) *pdomain = domain;
  return ((bus & 0xFF) << 8) | ((dev & 0x1F) << 3) | (func & 0x7);
}

std::tuple<uint32_t, uint32_t, bool> locIdAndDomainForNode(int node) {
  char path[128];
  snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/properties", node);
  std::ifstream f(path);
  if (!f.is_open()) return std::tuple{0, 0, false};
  uint32_t locId = ~0u, domain = ~0u;
  std::string key, valStr;
  // Read tokens as strings: some KFD properties (e.g. hive_id) are 64-bit values
  // that would fail a numeric extraction and abort the scan early.
  while (f >> key >> valStr) {
    if (key == "location_id")
      locId = std::strtol(valStr.c_str(), nullptr, 0);
    else if (key == "domain")
      domain = std::strtol(valStr.c_str(), nullptr, 0);
  }
  return std::tuple{locId, domain, true};
}

// "name value" pairs of a KFD sysfs properties file. Empty if the file is missing,
// or if KFD refuses the read (EPERM) because it belongs to a GPU the device cgroup
// keeps from this process.
std::unordered_map<std::string, uint64_t> readKfdProperties(std::istream& in) {
  std::unordered_map<std::string, uint64_t> props;
  std::string key, val;
  while (in >> key >> val) props[key] = std::strtoull(val.c_str(), nullptr, 0);
  return props;
}

std::unordered_map<std::string, uint64_t> readKfdProperties(const std::string& path) {
  std::ifstream f(path);
  return readKfdProperties(f);
}

std::string kfdNodePath(int node) {
  return "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(node);
}

uint64_t kfdNodeProperty(int node, const char* name) {
  auto props = readKfdProperties(kfdNodePath(node) + "/properties");
  auto it = props.find(name);
  return it == props.end() ? 0 : it->second;
}

// gpu_id has its own file, not a properties line; 0 for a CPU node.
uint64_t kfdGpuId(int node) {
  std::ifstream f(kfdNodePath(node) + "/gpu_id");
  uint64_t id = 0;
  f >> id;
  return id;
}

struct KfdLink {
  int nodeTo;
  uint32_t recSdmaEngIdMask;
};

// io_links then p2p_links, as libhsakmt lists them, in sysfs node ids. A node's
// own io_links stay readable even to GPUs this process cannot open, which
// libhsakmt drops. A link KFD gates on such a GPU (a p2p link carries the peer's
// device) reads as EPERM and is skipped, as libhsakmt skips it.
std::vector<KfdLink> kfdLinks(int node) {
  std::vector<KfdLink> links;
  for (const char* dir : {"io_links", "p2p_links"}) {
    for (int i = 0;; ++i) {
      std::ifstream f(kfdNodePath(node) + "/" + dir + "/" + std::to_string(i) + "/properties");
      if (!f.is_open()) break;
      auto props = readKfdProperties(f);
      if (props.empty()) continue;
      links.push_back({static_cast<int>(props["node_to"]),
                       static_cast<uint32_t>(props["recommended_sdma_engine_id_mask"])});
    }
  }
  return links;
}

// libhsakmt numbers only the nodes this process can open, 0..NumNodes-1, so in a
// container given some of the host's GPUs its ids are not the sysfs ones. Match
// the GPU by gpu_id.
uint32_t hsakmtNodeForKfdNode(int node) {
  const uint64_t gpuId = kfdGpuId(node);
  HsaSystemProperties sys{};
  if (gpuId != 0 && hsaKmtAcquireSystemProperties(&sys) == HSAKMT_STATUS_SUCCESS) {
    for (uint32_t n = 0; n < sys.NumNodes; ++n) {
      HsaNodeProperties props{};
      if (hsaKmtGetNodeProperties(n, &props) == HSAKMT_STATUS_SUCCESS && props.KFDGpuID == gpuId)
        return n;
    }
  }
  // Exit like a failed hsaKmt call in SdmaQueue: shmem would catch an exception
  // and blame the network, and cco cannot propagate one into Python.
  MORI_APP_ERROR("SDMA: no libhsakmt node has the gpu_id of KFD node {}", node);
  exit(EXIT_FAILURE);
}

}  // namespace

SdmaQueue::SdmaQueue(uint32_t hsakmtNodeId, uint32_t engineId) {
  // Allocate SDMA queue buffer on device side, requires ExecuteAccess
  HsaMemFlags memFlags = {};
  memFlags.ui32.NonPaged = 1;
  memFlags.ui32.HostAccess = 1;
  memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  memFlags.ui32.NoNUMABind = 1;
  memFlags.ui32.ExecuteAccess = 1;
  memFlags.ui32.Uncached = 1;

  // std::cout << "Allocating SDMA Queue Buffer for device: " << hsakmtNodeId << std::endl <<
  // std::flush;

  CHECK_HSAKMT_SUCCESS(hsaKmtAllocMemory(hsakmtNodeId, SDMA_QUEUE_SIZE, memFlags, &queueBuffer_),
                       "Failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE, NULL), "Failed");

  // Create SDMA Queue
  // TODO needed here?
  memset(&queue_, 0, sizeof(HsaQueueResource));

  CHECK_HSAKMT_SUCCESS(
      hsaKmtCreateQueueExt(hsakmtNodeId, HSA_QUEUE_SDMA_BY_ENG_ID, 100, HSA_QUEUE_PRIORITY_MAXIMUM,
                           engineId, queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_),
      "Failed");

  // Populate Device Handle
  // TODO uncached
  CHECK_HIP_ERROR(hipMalloc(&deviceHandle_, sizeof(SdmaQueueDeviceHandle)));
  CHECK_HIP_ERROR(
      hipExtMallocWithFlags((void**)&cachedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));
  CHECK_HIP_ERROR(
      hipExtMallocWithFlags((void**)&committedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));
  CHECK_HIP_ERROR(
      hipExtMallocWithFlags((void**)&retiredWptr_, 2 * sizeof(uint64_t), hipDeviceMallocUncached));

  uint64_t cachedWptr = (uint64_t)*(queue_.Queue_write_ptr_aql);
  uint64_t committedWptr = (uint64_t)*(queue_.Queue_write_ptr_aql);
  uint64_t retiredWptr[2] = {committedWptr, committedWptr};
  SdmaQueueDeviceHandle handle = {
      .queueBuf = static_cast<uint32_t*>(queueBuffer_),
      .rptr = queue_.Queue_read_ptr_aql,
      .wptr = queue_.Queue_write_ptr_aql,
      .doorbell = queue_.Queue_DoorBell_aql,
      .cachedWptr = cachedWptr_,
      .committedWptr = committedWptr_,
      .cachedHwReadIndex = (uint64_t)*(queue_.Queue_read_ptr_aql),
      .retiredWptr = retiredWptr_,
  };

  CHECK_HIP_ERROR(
      hipMemcpy(deviceHandle_, &handle, sizeof(SdmaQueueDeviceHandle), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(cachedWptr_, &cachedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(
      hipMemcpy(committedWptr_, &committedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(retiredWptr_, retiredWptr, sizeof(retiredWptr), hipMemcpyHostToDevice));
}

SdmaQueue::~SdmaQueue() {
  CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId), "Failed to destroy queue.");
  CHECK_HIP_ERROR(hipFree(deviceHandle_));
  CHECK_HIP_ERROR(hipFree(cachedWptr_));
  CHECK_HIP_ERROR(hipFree(committedWptr_));
  CHECK_HIP_ERROR(hipFree(retiredWptr_));
  CHECK_HSAKMT_SUCCESS(hsaKmtUnmapMemoryToGPU(queueBuffer_), "Failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE), "Failed");
}

SdmaQueueDeviceHandle* SdmaQueue::deviceHandle() const { return deviceHandle_; }

AnvilLib::~AnvilLib() {
  for (auto& p : sdma_channels_) {
    p.second.clear();
  }
  CloseKFD();
  hsa_shut_down();
}

void AnvilLib::init() {
  std::call_once(init_flag, []() {
    //   std::atexit(CloseKFD); // Register cleanup

    // HSA
    hsa_status_t status{hsa_init()};
    if (status != HSA_STATUS_SUCCESS) {
      printf("Failure to open HSA connection: 0x%x", status);
      // return 1;
    }
    status = hsa_iterate_agents(&rocm_hsa_gpu_agent_callback, &gpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      printf("Failure to iterate HSA agents: 0x%x", status);
      // return 1;
    }
    status = hsa_iterate_agents(&rocm_hsa_cpu_agent_callback, &cpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      printf("Failure to iterate HSA agents: 0x%x", status);
      // return 1;
    }

    SetUpKFD();
  });
}

// Map a HIP device ordinal to its KFD node id, the id connect() takes.
/*static*/ uint32_t AnvilLib::nodeForHipDevice(int hipDev) {
  int node = kfdNodeIdForHipDevice(hipDev);
  if (node < 0)
    throw std::runtime_error("SDMA: no KFD node for HIP device " + std::to_string(hipDev));
  return static_cast<uint32_t>(node);
}

// Resolve the KFD topology node id of the given HIP device WITHOUT initializing HSA.
/*static*/ int AnvilLib::kfdNodeIdForHipDevice(int hipDev) {
  uint32_t wantDomain = 0, wantLocId = getBusId(hipDev, &wantDomain);
  // N in /sys/class/kfd/kfd/topology/nodes/N is the counter the KFD driver
  // assigns while walking topology_device_list (kfd_build_sysfs_node_tree in
  // kfd_topology.c: kobject "%d", i++). It's the enumeration index, not a
  // sparse gpu_id/domain, so node ids are always contiguous 0..N-1 (CPU NUMA
  // nodes occupy the low indices). Stop at the first missing directory.
  for (int node = 0;; node++) {
    auto [locId, domain, valid] = locIdAndDomainForNode(node);
    if (!valid) break;
    if (locId == wantLocId && domain == wantDomain) {
      return node;
    }
  }
  return -1;
}

bool AnvilLib::connect(int srcNode, int dstNode, int numChannels) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  // Spread the channels across the engines recommended for this peer link. On
  // MI350 the mask typically reports 2 engines per peer; on platforms with a
  // single recommended engine all channels share it.
  std::vector<uint32_t> engines;
  engines.reserve(2);
  if (srcNode == dstNode) {
    // Loopback has no self io_link, so KFD recommends no engine. On gfx1250 each
    // engine holds only 6 queues and ROCr's blit queues already sit on the low
    // ones, so pinning every loopback channel to engine 0 hits NO_MEMORY at the
    // sixth channel; spread over the CPU-link engines (all 16 there) instead.
    // Other archs are not engine-0-bound for loopback (gfx950: 8 queues/engine,
    // only engines 0-1 general) and regress if a channel lands on a busy engine,
    // so keep them pinned to engine 0. Some gfx1250 GPUs report no CPU link at all
    // (two of four on HELIOS), so they take the engines of their peer links instead.
    if (isGfx1250(srcNode)) {
      uint32_t mask = getHostLinkEngineMask(srcNode);
      if (!mask) mask = getAnyLinkEngineMask(srcNode);
      for (uint32_t b = 0; b < 32; ++b) {
        if (mask & (1u << b)) engines.push_back(b);
      }
    }
    if (engines.empty()) engines.push_back(0);
  } else {
    uint32_t mask = getRecommendedEngineMask(srcNode, dstNode);
    for (uint32_t b = 0; b < 32; ++b) {
      if (mask & (1u << b)) engines.push_back(b);
    }
    // Fall back to the static OAM table if KFD did not report a mask.
    if (engines.empty()) {
      int e = getSdmaEngineId(srcNode, dstNode);
      engines.push_back(e);
    }
  }

  // MI308X is a 4-XCD harvest of the 8-XCD MI300X: only 4 physical SDMA engines
  // exist, exposed at the even KFD logical engine ids (0,2,4,6); the odd ids
  // (1,3,5,7) name fused-off engines. IP-discovery firmware still advertises the
  // full MI300X layout (2 general + 6 xGMI = 8) and recommended_sdma_engine_id_mask
  // points at odd ids for many links (this box lacks the upstream amdkfd
  // "rec SDMA engines with limited XGMI" fix). Creating a queue on an odd id and
  // ringing its doorbell hangs, so fold every selected id down to its backing
  // even engine (id & ~1) and dedup, preserving order, so the per-engine queue
  // budget is not double-charged.
  if (isMi308x(srcNode)) {
    uint32_t seen = 0;
    std::vector<uint32_t> evenEngines;
    for (uint32_t e : engines) {
      uint32_t backing = e & ~1u;
      if (!(seen & (1u << backing))) {
        seen |= (1u << backing);
        evenEngines.push_back(backing);
      }
    }
    engines.swap(evenEngines);
  }

  int numEngines = static_cast<int>(engines.size());

  // Queues live in this process-global singleton and are shared across every
  // Context/comm for this node pair (getSdmaQueue keys on KFD node ids, not on
  // the comm), and are only reclaimed when the process exits. So create just the
  // shortfall: appending on every connect() would pile up unused duplicate
  // hardware queues (getSdmaQueue only ever indexes the first numChannels) and
  // eventually exhaust the per-engine queue slots.
  auto& channels = sdma_channels_[std::make_pair(srcNode, dstNode)];
  if (static_cast<int>(channels.size()) < numChannels) {
    const uint32_t hsakmtNode = hsakmtNodeForKfdNode(srcNode);
    for (int c = static_cast<int>(channels.size()); c < numChannels; ++c) {
      uint32_t engineId = engines[c % numEngines];
      channels.emplace_back(std::make_unique<SdmaQueue>(hsakmtNode, engineId));
    }
  }
  return true;
}

uint32_t AnvilLib::getRecommendedEngineMask(int srcNode, int dstNode) {
  for (const auto& link : kfdLinks(srcNode)) {
    if (link.nodeTo == dstNode) return link.recSdmaEngIdMask;
  }
  return 0;
}

// Engines KFD recommends for this GPU's link to a CPU node, i.e. the general
// (non-xGMI) ones. Zero if the node reports no such link.
uint32_t AnvilLib::getHostLinkEngineMask(int srcNode) {
  for (const auto& link : kfdLinks(srcNode)) {
    // A GPU, or a node KFD will not show us (only a GPU is ever hidden): not the host.
    auto to = readKfdProperties(kfdNodePath(link.nodeTo) + "/properties");
    if (to.empty() || to["simd_count"] != 0) continue;
    if (link.recSdmaEngIdMask) return link.recSdmaEngIdMask;
  }
  return 0;
}

uint32_t AnvilLib::getAnyLinkEngineMask(int srcNode) {
  uint32_t mask = 0;
  for (const auto& link : kfdLinks(srcNode)) mask |= link.recSdmaEngIdMask;
  return mask;
}

// gfx12.5+ (gfx1250): the only arch whose loopback channels are spread over
// engines. See connect() for why.
bool AnvilLib::isGfx1250(int node) {
  // libhsakmt's EngineId: major = v / 10000 % 100, minor = v / 100 % 100.
  const uint64_t v = kfdNodeProperty(node, "gfx_target_version");
  return v / 10000 % 100 == 12 && v / 100 % 100 == 5;
}

// MI308X (PCI device_id 0x74A2) is a 4-XCD harvest of the 8-XCD MI300X. Both are
// gfx942, so EngineId cannot tell them apart; match the PCI device id instead.
// See connect() for why its odd SDMA engine ids must be avoided.
bool AnvilLib::isMi308x(int node) { return kfdNodeProperty(node, "device_id") == 0x74A2; }

SdmaQueue* AnvilLib::getSdmaQueue(int srcNode, int dstNode, int idx) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  auto key = std::make_pair(srcNode, dstNode);
  auto it = sdma_channels_.find(key);
  if (it == sdma_channels_.end() || idx >= static_cast<int>(it->second.size())) {
    return nullptr;
  }
  return it->second[idx].get();
}

AnvilLib& AnvilLib::getInstance() {
  // Keep pre-SDMA-collective behavior: do not run ~AnvilLib during process teardown.
  // Worker exits can otherwise stall in ROCm/HSA shutdown ordering.
  static AnvilLib* instance;
  if (instance == nullptr) {
    instance = new AnvilLib();
  }
  return *instance;
}

int AnvilLib::getOamId(int node) {
  auto [locId, domain, valid] = locIdAndDomainForNode(node);
  if (valid && locId == ~0u) {
    throw std::runtime_error("SDMA: cannot read KFD node " + std::to_string(node) +
                             " (a GPU this process cannot open) to find its xGMI id");
  }
  uint32_t bus = (locId >> 8) & 0xFF, dev = (locId >> 3) & 0x1F, func = locId & 0x7;

  char fpath[128];
  std::snprintf(fpath, sizeof(fpath), "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/xgmi_physical_id",
                domain, bus, dev, func);
  std::ifstream file(fpath);
  if (!(valid && file.is_open())) {
    throw std::runtime_error("Unable to open xGMI physical id file: " + std::string(fpath));
  }
  int xgmi_physical_id;
  if (!(file >> xgmi_physical_id)) {
    throw std::runtime_error("Failed to read xGMI physical id from file: " + std::string(fpath));
  }
  return xgmi_physical_id;
}

int AnvilLib::getSdmaEngineId(int srcNode, int dstNode) {
  int srcOamId = getOamId(srcNode);
  int dstOamId = getOamId(dstNode);

  // Use even engines only
  return mi300xOamMap[srcOamId][dstOamId] * 2;
}

AnvilLib& anvil = anvil.getInstance();

}  // namespace anvil

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

#define CHECK_HSA_ERROR(cmd)                                                               \
  if (auto s = (cmd); s != HSA_STATUS_SUCCESS) {                                           \
    const char* hsa_err_msg;                                                               \
    hsa_status_string(s, &hsa_err_msg);                                                    \
    throw std::runtime_error{std::string("HSA error at " __FILE__ ":") +                   \
                             std::to_string(__LINE__) + std::string(" - ") + hsa_err_msg}; \
  }

#define CHECK_HSAKMT_SUCCESS(call, msg)                                                       \
  do {                                                                                        \
    if ((call) != HSAKMT_STATUS_SUCCESS) {                                                    \
      std::cout << "ERROR code: " << std::dec << call << " " << msg << " (File: " << __FILE__ \
                << ", Line: " << __LINE__ << ")" << std::endl;                                \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
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

std::pair<uint32_t, uint32_t> locIdAndDomainForNode(int node) {
  uint32_t locId = ~0u, domain = ~0u;
  std::string path = "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(node) + "/properties";
  std::ifstream f(path);
  if (!f.is_open()) return std::pair{locId, domain};
  std::string key, valStr;
  // Read tokens as strings: some KFD properties (e.g. hive_id) are 64-bit values
  // that would fail a numeric extraction and abort the scan early.
  while (f >> key >> valStr) {
    if (key == "location_id")
      locId = std::strtol(valStr.c_str(), nullptr, 0);
    else if (key == "domain")
      domain = std::strtol(valStr.c_str(), nullptr, 0);
  }
  return std::pair{locId, domain};
}

// hsa_iterate_agents (SetUp) enumerates ALL physical GPU agents in HSA order,
// which is NOT filtered by HIP_VISIBLE_DEVICES. The srcDeviceId/deviceId that the
// collective passes in is a HIP device ORDINAL (indexes only visible devices).
// When the two diverge (e.g. HIP_VISIBLE_DEVICES=4,5,6,7) indexing gpuAgents_ by
// the raw HIP ordinal selects the WRONG physical GPU -> the SDMA queue is created
// on the wrong KFD node while the compute kernel runs on the intended GPU ->
// "Memory access fault by GPU node-N". Match HIP ordinal -> HSA agent by PCI BDF
// so the selection is correct in all cases. In the common HIP_VISIBLE_DEVICES=
// 0..N-1 case this resolves to the identity map (BDF matches at the same index)
// so the default path is behavior-identical.
hsa_agent_t gpuAgentForHipDevice(int hipDeviceId) {
  static std::mutex mapMutex;
  static std::unordered_map<int, int> hipToAgent;
  std::lock_guard<std::mutex> lock(mapMutex);
  auto it = hipToAgent.find(hipDeviceId);
  if (it != hipToAgent.end()) return gpuAgents_[it->second];

  // BDF of the HIP device, parsed from its "domain:bus:device.function" string.
  uint32_t hipBdf = getBusId(hipDeviceId);
  // HSA_AMD_AGENT_INFO_BDFID exposes only the 16-bit bus/device/function, not the
  // PCI domain, so on a multi-segment machine two GPUs can share the same 16-bit
  // BDF. Only trust the match when it is UNIQUE; otherwise keep the identity
  // fallback rather than risk selecting the wrong agent. domain is parsed but
  // cannot be matched against the HSA side.
  int match = hipDeviceId;  // identity fallback (also correct when HIP and HSA order align)
  int nMatch = 0, firstMatch = -1;
  for (size_t a = 0; a < gpuAgents_.size(); ++a) {
    uint32_t bdfid = 0;
    if (hsa_agent_get_info(gpuAgents_[a], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdfid) !=
        HSA_STATUS_SUCCESS)
      continue;
    if ((bdfid & 0xFFFF) == (hipBdf & 0xFFFF)) {
      ++nMatch;
      if (firstMatch < 0) firstMatch = static_cast<int>(a);
    }
  }
  if (nMatch == 1) match = firstMatch;
  hipToAgent[hipDeviceId] = match;
  return gpuAgents_[match];
}

}  // namespace

SdmaQueue::SdmaQueue(uint32_t localNodeId, uint32_t engineId) {
  // Allocate SDMA queue buffer on device side, requires ExecuteAccess
  HsaMemFlags memFlags = {};
  memFlags.ui32.NonPaged = 1;
  memFlags.ui32.HostAccess = 1;
  memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  memFlags.ui32.NoNUMABind = 1;
  memFlags.ui32.ExecuteAccess = 1;
  memFlags.ui32.Uncached = 1;

  // std::cout << "Allocating SDMA Queue Buffer for device: " << localNodeId << std::endl <<
  // std::flush;

  CHECK_HSAKMT_SUCCESS(hsaKmtAllocMemory(localNodeId, SDMA_QUEUE_SIZE, memFlags, &queueBuffer_),
                       "Failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE, NULL), "Failed");

  // Create SDMA Queue
  // TODO needed here?
  memset(&queue_, 0, sizeof(HsaQueueResource));

  CHECK_HSAKMT_SUCCESS(
      hsaKmtCreateQueueExt(localNodeId, HSA_QUEUE_SDMA_BY_ENG_ID, 100, HSA_QUEUE_PRIORITY_MAXIMUM,
                           engineId, queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_),
      "Failed");

  // Populate Device Handle
  // TODO uncached
  CHECK_HIP_ERROR(hipMalloc(&deviceHandle_, sizeof(SdmaQueueDeviceHandle)));
  CHECK_HIP_ERROR(
      hipExtMallocWithFlags((void**)&cachedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));
  CHECK_HIP_ERROR(
      hipExtMallocWithFlags((void**)&committedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));

  uint64_t cachedWptr = (uint64_t)*(queue_.Queue_write_ptr_aql);
  uint64_t committedWptr = (uint64_t)*(queue_.Queue_write_ptr_aql);
  SdmaQueueDeviceHandle handle = {
      .queueBuf = static_cast<uint32_t*>(queueBuffer_),
      .rptr = queue_.Queue_read_ptr_aql,
      .wptr = queue_.Queue_write_ptr_aql,
      .doorbell = queue_.Queue_DoorBell_aql,
      .cachedWptr = cachedWptr_,
      .committedWptr = committedWptr_,
      .cachedHwReadIndex = (uint64_t)*(queue_.Queue_read_ptr_aql),
  };

  CHECK_HIP_ERROR(
      hipMemcpy(deviceHandle_, &handle, sizeof(SdmaQueueDeviceHandle), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(cachedWptr_, &cachedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(
      hipMemcpy(committedWptr_, &committedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
}

SdmaQueue::~SdmaQueue() {
  CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId), "Failed to destroy queue.");
  CHECK_HIP_ERROR(hipFree(deviceHandle_));
  CHECK_HIP_ERROR(hipFree(cachedWptr_));
  CHECK_HIP_ERROR(hipFree(committedWptr_));
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

// Map a HIP device ordinal to its KFD node id.
/*static*/ uint32_t AnvilLib::nodeForHipDevice(int hipDev) {
  uint32_t nodeId = 0;
  CHECK_HSA_ERROR(hsa_agent_get_info(gpuAgentForHipDevice(hipDev), HSA_AGENT_INFO_NODE, &nodeId));
  return nodeId;
}

// Resolve the KFD topology node id of the given HIP device WITHOUT initializing HSA.
/*static*/ int AnvilLib::kfdNodeIdForHipDevice(int hipDev) {
  uint32_t wantDomain = 0, wantLocId = getBusId(hipDev, &wantDomain);
  // KFD node ids are contiguous from 0; stop at the first gap.
  for (int node = 0;; node++) {
    auto [locId, domain] = locIdAndDomainForNode(node);
    if (locId == ~0u && domain == ~0u) break;
    if (locId == wantLocId && domain == wantDomain) {
      return node;
    }
  }
  MORI_APP_ERROR("Failed to find KFD node for device {}", hipDev);
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
    // so keep them pinned to engine 0.
    if (isGfx1250(srcNode)) {
      uint32_t mask = getHostLinkEngineMask(srcNode);
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
  int numEngines = static_cast<int>(engines.size());

  // Queues live in this process-global singleton and are shared across every
  // Context/comm for this node pair (getSdmaQueue keys on KFD node ids, not on
  // the comm), and are only reclaimed when the process exits. So create just the
  // shortfall: appending on every connect() would pile up unused duplicate
  // hardware queues (getSdmaQueue only ever indexes the first numChannels) and
  // eventually exhaust the per-engine queue slots.
  auto& channels = sdma_channels_[std::make_pair(srcNode, dstNode)];
  for (int c = static_cast<int>(channels.size()); c < numChannels; ++c) {
    uint32_t engineId = engines[c % numEngines];
    channels.emplace_back(std::make_unique<SdmaQueue>(srcNode, engineId));
  }
  return true;
}

uint32_t AnvilLib::getRecommendedEngineMask(int srcNode, int dstNode) {
  HsaNodeProperties props{};
  if (hsaKmtGetNodeProperties(srcNode, &props) != HSAKMT_STATUS_SUCCESS || props.NumIOLinks == 0) {
    return 0;
  }

  std::vector<HsaIoLinkProperties> links(props.NumIOLinks);
  if (hsaKmtGetNodeIoLinkProperties(srcNode, props.NumIOLinks, links.data()) !=
      HSAKMT_STATUS_SUCCESS) {
    return 0;
  }
  for (const auto& link : links) {
    if (link.NodeTo == dstNode) {
      return link.RecSdmaEngIdMask;
    }
  }
  return 0;
}

// Engines KFD recommends for this GPU's link to a CPU node, i.e. the general
// (non-xGMI) ones. Zero if the node reports no such link.
uint32_t AnvilLib::getHostLinkEngineMask(int srcNode) {
  HsaNodeProperties props{};
  if (hsaKmtGetNodeProperties(srcNode, &props) != HSAKMT_STATUS_SUCCESS || props.NumIOLinks == 0) {
    return 0;
  }
  std::vector<HsaIoLinkProperties> links(props.NumIOLinks);
  if (hsaKmtGetNodeIoLinkProperties(srcNode, props.NumIOLinks, links.data()) !=
      HSAKMT_STATUS_SUCCESS) {
    return 0;
  }
  for (const auto& link : links) {
    HsaNodeProperties to{};
    if (hsaKmtGetNodeProperties(link.NodeTo, &to) != HSAKMT_STATUS_SUCCESS) continue;
    if (to.NumFComputeCores != 0) continue;  // GPU node, not the host
    if (link.RecSdmaEngIdMask) return link.RecSdmaEngIdMask;
  }
  return 0;
}

// gfx12.5+ (gfx1250): the only arch whose loopback channels are spread over
// engines. See connect() for why.
bool AnvilLib::isGfx1250(int node) {
  HsaNodeProperties props{};
  if (hsaKmtGetNodeProperties(node, &props) != HSAKMT_STATUS_SUCCESS) return false;
  return props.EngineId.ui32.Major == 12 && props.EngineId.ui32.Minor == 5;
}

SdmaQueue* AnvilLib::getSdmaQueue(int srcNode, int dstNode, int channel_idx) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  auto key = std::make_pair(srcNode, dstNode);
  auto it = sdma_channels_.find(key);
  if (it == sdma_channels_.end()) {
    return nullptr;
  }
  if (!(channel_idx < static_cast<int>(it->second.size()))) {
    return nullptr;
  }
  return it->second[channel_idx].get();
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
  auto [locId, domain] = locIdAndDomainForNode(node);
  uint32_t bus = (locId >> 8) & 0xFF, dev = (locId >> 3) & 0x1F, func = locId & 0x7;

  char fpath[128];
  std::snprintf(fpath, sizeof(fpath), "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/xgmi_physical_id",
                domain, bus, dev, func);
  std::ifstream file(fpath);
  int xgmi_physical_id;
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + std::string(fpath));
  }
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

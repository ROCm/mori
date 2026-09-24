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

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
namespace anvil {

// How many SDMA queues this process has built, for the failure message below:
// it separates "mori asked for an unreasonable number" from "the box was
// already full when we asked for our first".
std::atomic<int> queuesCreated_{0};

auto checkHsaError = [](hsa_status_t s, const char* msg, const char* file, int line) {
  if (s != HSA_STATUS_SUCCESS) {
    const char* hsa_err_msg;
    hsa_status_string(s, &hsa_err_msg);
    throw(std::runtime_error{std::string("HSA error at ") + file + std::string(":") +
                             std::to_string(line) + std::string(" - ") + hsa_err_msg});
  }
};

#define CHECK_HSA_ERROR(cmd) checkHsaError((cmd), #cmd, __FILE__, __LINE__)

// `call` is evaluated once. It used to appear a second time in the message, so
// every failing hsaKmt call was issued twice on the way out: a second
// CreateQueueExt against a node that had just refused one, a second AllocMemory
// that leaks, a double DestroyQueue.
#define CHECK_HSAKMT_SUCCESS(call, msg)                                                  \
  do {                                                                                   \
    HSAKMT_STATUS _hsakmt_status = (call);                                               \
    if (_hsakmt_status != HSAKMT_STATUS_SUCCESS) {                                       \
      std::cout << "ERROR code: " << std::dec << _hsakmt_status << " " << msg            \
                << " (File: " << __FILE__ << ", Line: " << __LINE__ << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                                \
    }                                                                                    \
  } while (0)

#if 0
inline void checkHipError(hipError_t err, const char* msg, const char* file, int line)
{
   if (err != hipSuccess)
   {
      std::cerr << "HIP error at " << file << ":" << line << " — " << msg << "\n"
                << "  Code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl;
      std::exit(EXIT_FAILURE);
   }
}

#define CHECK_HIP_ERROR(cmd) checkHipError((cmd), #cmd, __FILE__, __LINE__)

// Allow access to peerDeviceId from deviceId
inline void EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
   int canAccess;
   CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
   if (!canAccess)
   {
      std::cerr << "Unable to enable peer access from GPU devices " << deviceId << " to " << peerDeviceId << "\n";
   }

   CHECK_HIP_ERROR(hipSetDevice(deviceId));
   hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
   if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
   {
      std::cerr << "Unable to enable peer to peer access from " << deviceId << "  to " << peerDeviceId << " ("
                << hipGetErrorString(error) << ")\n";
   }
}
#endif

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

// Convert a logical deviceId index to the NVML device minor number
static const std::string getBusId(int deviceId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdChar[] = "00000000:00:00.0";
  CHECK_HIP_ERROR(hipDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  // we need the hex in lower case format
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
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
static int gpuAgentIndexForHipDevice(int hipDeviceId) {
  static std::mutex mapMutex;
  static std::unordered_map<int, int> hipToAgent;
  std::lock_guard<std::mutex> lock(mapMutex);
  auto it = hipToAgent.find(hipDeviceId);
  if (it != hipToAgent.end()) return it->second;

  // BDF of the HIP device, parsed from its "domain:bus:device.function" string.
  std::string busId = getBusId(hipDeviceId);
  unsigned domain = 0, bus = 0, dev = 0, func = 0;
  std::sscanf(busId.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &func);
  uint32_t hipBdf = ((bus & 0xFF) << 8) | ((dev & 0x1F) << 3) | (func & 0x7);

  // HSA_AMD_AGENT_INFO_BDFID exposes only the 16-bit bus/device/function, not the
  // PCI domain, so on a multi-segment machine two GPUs can share the same 16-bit
  // BDF. Only trust the match when it is UNIQUE; otherwise keep the identity
  // fallback rather than risk selecting the wrong agent. domain is parsed but
  // cannot be matched against the HSA side.
  (void)domain;
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
  return match;
}

SdmaQueue::SdmaQueue(int localDeviceId, int remoteDeviceId, hsa_agent_t& localAgent,
                     uint32_t engineId)
    : remoteDeviceId_(remoteDeviceId) {
  // cachedWptr_(detail::gpuCallocUncachedShared<uint64_t>()),
  // committedWptr_(detail::gpuCallocUncachedShared<uint64_t>()) {
  int originalDeviceId;

  CHECK_HIP_ERROR(hipGetDevice(&originalDeviceId));  // Save the current device

  uint32_t localNodeId;
  hsa_status_t status = hsa_agent_get_info(localAgent, HSA_AGENT_INFO_NODE, &localNodeId);
  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get device info: 0x%x", status);
    // return status;
  }

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

  // Two very different failures arrive here with nothing to tell them apart, and
  // the difference decides who has to act. Measured on MI355X (gfx950), which has
  // 2 PCIe + 14 XGMI SDMA engines and 8 queues per engine:
  //
  //   HSAKMT_STATUS_ERROR (1)      the kernel refused. Every form of "no queue
  //                                for you" lands here: the per-engine cap
  //                                (verified at 8/engine, and at the full 128 on
  //                                a node), and an engine id the node does not
  //                                have. A dead process's slots come back before
  //                                a new process can reach this call, so a retry
  //                                buys nothing.
  //   HSAKMT_STATUS_NO_MEMORY (6)  libhsakmt could not allocate for the queue in
  //                                THIS process's address space. Reproduced by
  //                                constraining RLIMIT_AS: the refusal moves to
  //                                the 11th, 5th, 3rd queue as the limit drops,
  //                                while an unconstrained process reaches 128.
  //                                Nothing about SDMA is exhausted; the process
  //                                or the box is out of memory.
  //
  // Say which one it is, with the numbers needed to size it, so this does not
  // have to be re-derived from a bare error code. See ROCm/mori#685.
  HSAKMT_STATUS queueStatus =
      hsaKmtCreateQueueExt(localNodeId, HSA_QUEUE_SDMA_BY_ENG_ID, 100, HSA_QUEUE_PRIORITY_MAXIMUM,
                           engineId, queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_);
  if (queueStatus != HSAKMT_STATUS_SUCCESS) {
    HsaNodeProperties props{};
    bool haveProps = hsaKmtGetNodeProperties(localNodeId, &props) == HSAKMT_STATUS_SUCCESS;
    std::cout << "SDMA queue creation failed: status=" << std::dec << queueStatus << " ("
              << (queueStatus == HSAKMT_STATUS_NO_MEMORY
                      ? "NO_MEMORY: out of memory in this process, not out of SDMA queues"
                      : "ERROR: the kernel refused this engine")
              << ")\n  node=" << localNodeId << " engine=" << engineId
              << " queues_created_by_this_process=" << queuesCreated_.load();
    if (haveProps) {
      std::cout << "\n  node has " << props.NumSdmaEngines << " PCIe + " << props.NumSdmaXgmiEngines
                << " XGMI SDMA engines, " << props.NumSdmaQueuesPerEngine << " queues per engine";
    }
    std::cout << std::endl;
  }
  CHECK_HSAKMT_SUCCESS(queueStatus, "Failed");
  queuesCreated_.fetch_add(1);

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

bool AnvilLib::connect(int srcDeviceId, int dstDeviceId, int numChannels) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  // Spread the channels across the engines recommended for this peer link. On
  // MI350 the mask typically reports 2 engines per peer; on platforms with a
  // single recommended engine all channels share it.
  std::vector<uint32_t> engines;
  if (srcDeviceId == dstDeviceId) {
    // Loopback has no self io_link, so KFD recommends no engine. On gfx1250 each
    // engine holds only 6 queues and ROCr's blit queues already sit on the low
    // ones, so pinning every loopback channel to engine 0 hits NO_MEMORY at the
    // sixth channel; spread over the CPU-link engines (all 16 there) instead.
    // Other archs are not engine-0-bound for loopback (gfx950: 8 queues/engine,
    // only engines 0-1 general) and regress if a channel lands on a busy engine,
    // so keep them pinned to engine 0.
    if (isGfx1250(srcDeviceId)) {
      uint32_t mask = getHostLinkEngineMask(srcDeviceId);
      for (uint32_t b = 0; b < 32; ++b) {
        if (mask & (1u << b)) engines.push_back(b);
      }
    }
    if (engines.empty()) engines.push_back(0);
  } else {
    uint32_t mask = getRecommendedEngineMask(srcDeviceId, dstDeviceId);
    for (uint32_t b = 0; b < 32; ++b) {
      if (mask & (1u << b)) engines.push_back(b);
    }
    // Fall back to the static OAM table if KFD did not report a mask.
    if (engines.empty()) {
      int e = getSdmaEngineId(srcDeviceId, dstDeviceId);
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
  if (isMi308x(srcDeviceId)) {
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
  // Context/comm for this device pair (getSdmaQueue keys on device ids, not on
  // the comm), and are only reclaimed when the process exits. So create just the
  // shortfall: appending on every connect() would pile up unused duplicate
  // hardware queues (getSdmaQueue only ever indexes the first numChannels) and
  // eventually exhaust the per-engine queue slots.
  auto& channels = sdma_channels_[std::make_pair(srcDeviceId, dstDeviceId)];
  for (int c = static_cast<int>(channels.size()); c < numChannels; ++c) {
    uint32_t engineId = engines[c % numEngines];
    channels.emplace_back(std::make_unique<SdmaQueue>(
        srcDeviceId, dstDeviceId, gpuAgents_[gpuAgentIndexForHipDevice(srcDeviceId)], engineId));
  }
  return true;
}

uint32_t AnvilLib::getNodeId(int deviceId) {
  uint32_t nodeId = 0;
  CHECK_HSA_ERROR(hsa_agent_get_info(gpuAgents_[gpuAgentIndexForHipDevice(deviceId)],
                                     HSA_AGENT_INFO_NODE, &nodeId));
  return nodeId;
}

uint32_t AnvilLib::getRecommendedEngineMask(int srcDeviceId, int dstDeviceId) {
  uint32_t srcNode = getNodeId(srcDeviceId), dstNode = getNodeId(dstDeviceId);

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
uint32_t AnvilLib::getHostLinkEngineMask(int srcDeviceId) {
  const uint32_t srcNode = getNodeId(srcDeviceId);
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
bool AnvilLib::isGfx1250(int deviceId) {
  HsaNodeProperties props{};
  if (hsaKmtGetNodeProperties(getNodeId(deviceId), &props) != HSAKMT_STATUS_SUCCESS) return false;
  return props.EngineId.ui32.Major == 12 && props.EngineId.ui32.Minor == 5;
}

// MI308X (PCI device_id 0x74A2) is a 4-XCD harvest of the 8-XCD MI300X. Both are
// gfx942, so EngineId cannot tell them apart; match the PCI device id instead.
// See connect() for why its odd SDMA engine ids must be avoided.
bool AnvilLib::isMi308x(int deviceId) {
  HsaNodeProperties props{};
  if (hsaKmtGetNodeProperties(getNodeId(deviceId), &props) != HSAKMT_STATUS_SUCCESS) return false;
  return props.DeviceId == 0x74A2;
}

SdmaQueue* AnvilLib::getSdmaQueue(int srcDeviceId, int dstDeviceId, int channel_idx) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  auto key = std::make_pair(srcDeviceId, dstDeviceId);
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

int AnvilLib::getOamId(int deviceId) {
  std::string busId = getBusId(deviceId);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/xgmi_physical_id";
  std::ifstream file(file_str);
  int xgmi_physical_id;
  if (file.is_open()) {
    if (!(file >> xgmi_physical_id)) {
      throw std::runtime_error("Failed to read xGMI physical id from file: " + file_str);
    }
  } else {
    throw std::runtime_error("Failed to open file: " + file_str);
  }
  return xgmi_physical_id;
}

int AnvilLib::getSdmaEngineId(int srcDeviceId, int dstDeviceId) {
  int srcOamId = getOamId(srcDeviceId);
  int dstOamId = getOamId(dstDeviceId);

  // Use even engines only
  return mi300xOamMap[srcOamId][dstOamId] * 2;
}

AnvilLib& anvil = anvil.getInstance();

}  // namespace anvil

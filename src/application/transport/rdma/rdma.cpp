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
#include "mori/application/transport/rdma/rdma.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "infiniband/verbs.h"
#include "mori/application/transport/rdma/providers/bnxt/bnxt.hpp"
#include "mori/application/transport/rdma/providers/ibverbs/ibverbs.hpp"
#include "mori/application/transport/rdma/providers/ionic/ionic.hpp"
#include "mori/application/transport/rdma/providers/mlx5/mlx5.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

namespace {
std::string TrimWhitespace(const std::string& input) {
  auto begin = std::find_if_not(input.begin(), input.end(),
                                [](unsigned char ch) { return std::isspace(ch); });
  auto end = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) {
               return std::isspace(ch);
             }).base();
  if (begin >= end) return {};
  return std::string(begin, end);
}

bool IsZeroGid(const union ibv_gid& gid) {
  for (const auto byte : gid.raw) {
    if (byte != 0) return false;
  }
  return true;
}

bool IsIpv4MappedGid(const union ibv_gid& gid) {
  for (int i = 0; i < 10; ++i) {
    if (gid.raw[i] != 0) return false;
  }
  return gid.raw[10] == 0xff && gid.raw[11] == 0xff;
}

bool IsLinkLocal(const union ibv_gid& gid) {
  return gid.raw[0] == 0xfe && (gid.raw[1] & 0xc0) == 0x80;
}

bool ReadGidTypeFromSysfs(ibv_context* context, uint32_t portId, int index,
                          ibv_gid_type* out_type) {
  if (!context || !context->device || !out_type || index < 0) return false;

  char path[PATH_MAX];
  int written = snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%u/gid_attrs/types/%d",
                         context->device->name, portId, index);
  if (written <= 0 || written >= static_cast<int>(sizeof(path))) return false;

  std::ifstream typeFile(path);
  if (!typeFile.is_open()) return false;

  std::string line;
  std::getline(typeFile, line);
  typeFile.close();
  line = TrimWhitespace(line);
  if (line.empty()) return false;

  std::transform(line.begin(), line.end(), line.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });

  if (line.find("v2") != std::string::npos) {
    *out_type = IBV_GID_TYPE_ROCE_V2;
    return true;
  }
  if (line.find("v1") != std::string::npos) {
    *out_type = IBV_GID_TYPE_ROCE_V1;
    return true;
  }
  if (line.find("ib") != std::string::npos) {
    *out_type = IBV_GID_TYPE_IB;
    return true;
  }

  return false;
}

bool QueryGidAtIndex(ibv_context* context, uint32_t portId, int index,
                     const ibv_port_attr* portAttr, union ibv_gid* out_gid,
                     ibv_gid_type* out_type) {
  if (!context || index < 0) return false;

  ibv_gid_entry entry{};
  if (ibv_query_gid_ex(context, portId, index, &entry, 0) == 0) {
    if (out_gid) *out_gid = entry.gid;
    if (out_type) *out_type = static_cast<ibv_gid_type>(entry.gid_type);
    return true;
  }

  union ibv_gid legacy_gid{};
  if (ibv_query_gid(context, portId, index, &legacy_gid) == 0) {
    if (out_gid) *out_gid = legacy_gid;
    if (out_type) {
      ibv_gid_type legacy_type = IBV_GID_TYPE_IB;
      if (portAttr && portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        legacy_type = IBV_GID_TYPE_IB;
      } else {
        if (!ReadGidTypeFromSysfs(context, portId, index, &legacy_type)) {
          legacy_type = IBV_GID_TYPE_ROCE_V1;
        }
      }
      *out_type = legacy_type;
    }
    return true;
  }

  return false;
}

int ScoreGidCandidate(const union ibv_gid& gid, ibv_gid_type gidType, int index) {
  if (IsZeroGid(gid)) return INT_MIN;

  bool isIpv4 = IsIpv4MappedGid(gid);
  bool isLinkLocalGid = IsLinkLocal(gid);
  bool isGlobalIpv6 = !isIpv4 && !isLinkLocalGid;

  int score = 0;

  if (gidType == IBV_GID_TYPE_ROCE_V2) {
    score += 1000;
  } else if (gidType == IBV_GID_TYPE_ROCE_V1) {
    score += 500;
  } else {
    score += 100;
  }

  if (isIpv4) {
    score += 200;
  } else if (isGlobalIpv6) {
    score += 100;
  }

  score -= index;  // Prefer smaller indices for deterministic behavior
  return score;
}

}  // namespace

GidSelectionResult AutoSelectGidIndex(ibv_context* context, uint32_t portId,
                                      const ibv_port_attr* portAttr, int32_t configuredGidIdx) {
  GidSelectionResult result{};
  result.gidIdx = configuredGidIdx;
  if (!context) return result;

  if (configuredGidIdx >= 0) {
    result.fromUser = true;
    if (QueryGidAtIndex(context, portId, configuredGidIdx, portAttr, &result.gid,
                        &result.gidType)) {
      result.valid = true;
    } else {
      MORI_APP_WARN("Failed to query user-specified gid index {} on port {}", configuredGidIdx,
                    portId);
    }
    return result;
  }

  int gidTableLen = portAttr ? static_cast<int>(portAttr->gid_tbl_len) : 0;
  if (gidTableLen <= 0) gidTableLen = 128;  // Conservative fallback

  int bestScore = INT_MIN;
  int bestIdx = -1;
  union ibv_gid bestGid{};
  ibv_gid_type bestType = IBV_GID_TYPE_IB;
  bool found = false;

  for (int idx = 0; idx < gidTableLen; ++idx) {
    union ibv_gid gid{};
    ibv_gid_type gidType = IBV_GID_TYPE_IB;
    if (!QueryGidAtIndex(context, portId, idx, portAttr, &gid, &gidType)) continue;

    int score = ScoreGidCandidate(gid, gidType, idx);
    if (score > bestScore) {
      bestScore = score;
      bestIdx = idx;
      bestGid = gid;
      bestType = gidType;
      found = true;

      if (score > 1000 + 200) break;  // Optimal candidate found
    }
  }

  if (!found) {
    if (QueryGidAtIndex(context, portId, 0, portAttr, &bestGid, &bestType)) {
      bestIdx = 0;
      found = true;
    }
  }

  if (found) {
    result.gidIdx = bestIdx;
    result.gid = bestGid;
    result.gidType = bestType;
    result.valid = true;
    MORI_APP_TRACE("Auto-selected GID index {} (type={}) on port {}", bestIdx,
                   static_cast<int>(bestType), portId);
  } else {
    MORI_APP_ERROR("Failed to auto-detect a valid GID on port {}", portId);
  }

  return result;
}

/* -------------------------------------------------------------------------- */
/*                             Rdma Configurations                            */
/* -------------------------------------------------------------------------- */

std::optional<uint8_t> ReadUint8FromEnvVar(const std::string& name) {
  const char* val = std::getenv(name.c_str());
  if (!val) {
    return std::nullopt;  // env not set
  }

  // Check conversion errors
  errno = 0;
  char* end = nullptr;
  unsigned long parsed = std::strtoul(val, &end, 10);
  if (errno != 0 || end == val || *end != '\0') {
    return std::nullopt;
  }

  // Range check for uint8_t
  if (parsed > std::numeric_limits<uint8_t>::max()) {
    return std::nullopt;
  }

  return static_cast<uint8_t>(parsed);
}

std::optional<uint8_t> ReadRdmaServiceLevelEnv() { return ReadUint8FromEnvVar("MORI_RDMA_SL"); }

std::optional<uint8_t> ReadRdmaTrafficClassEnv() { return ReadUint8FromEnvVar("MORI_RDMA_TC"); }

std::optional<uint8_t> ReadIoServiceLevelEnv() { return ReadUint8FromEnvVar("MORI_IO_SL"); }

std::optional<uint8_t> ReadIoTrafficClassEnv() { return ReadUint8FromEnvVar("MORI_IO_TC"); }

bool ReadIoTrafficClassDisableEnv() {
  std::optional<uint8_t> disable = ReadUint8FromEnvVar("MORI_IO_TC_DISABLE");
  return disable.has_value() && disable.value() == 1;
}

bool ReadIbEnableRelaxedOrderingEnv() {
  std::optional<uint8_t> enable = ReadUint8FromEnvVar("MORI_IB_ENABLE_RELAXED_ORDERING");
  return enable.has_value() && enable.value() == 1;
}

int MaybeAddRelaxedOrderingFlag(int accessFlag) {
#ifdef IBV_ACCESS_RELAXED_ORDERING
  if (ReadIbEnableRelaxedOrderingEnv()) {
    return accessFlag | IBV_ACCESS_RELAXED_ORDERING;
  }
#endif
  return accessFlag;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        RdmaDeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
RdmaDeviceContext::RdmaDeviceContext(RdmaDevice* device, ibv_pd* inPd) : device(device), pd(inPd) {
  InitializeUdpSportConfiguration();
}

RdmaDeviceContext::~RdmaDeviceContext() {
  for (auto& it : mrPool) {
    ibv_dereg_mr(it.second);
  }
  mrPool.clear();

  if (srq != nullptr) ibv_destroy_srq(srq);
  ibv_dealloc_pd(pd);
}

RdmaDevice* RdmaDeviceContext::GetRdmaDevice() { return device; }

ibv_context* RdmaDeviceContext::GetIbvContext() { return GetRdmaDevice()->defaultContext; }

application::RdmaMemoryRegion RdmaDeviceContext::RegisterRdmaMemoryRegion(void* ptr, size_t size,
                                                                          int accessFlag) {
  int effectiveAccessFlag = MaybeAddRelaxedOrderingFlag(accessFlag);
  ibv_mr* mr = ibv_reg_mr(pd, ptr, size, effectiveAccessFlag);
  if (!mr) {
    MORI_APP_ERROR(
        "RegisterRdmaMemoryRegion failed! addr:{}, size:{}, accessFlag:{}, errno:{} ({})", ptr,
        size, effectiveAccessFlag, errno, strerror(errno));
    std::abort();
  }
  MORI_APP_TRACE("RegisterRdmaMemoryRegion, addr:{}, size:{}, lkey:{}, rkey:{}\n", ptr, size,
                 mr->lkey, mr->rkey);
  mrPool.insert({ptr, mr});
  application::RdmaMemoryRegion handle;
  handle.addr = reinterpret_cast<uintptr_t>(ptr);
  handle.lkey = mr->lkey;
  handle.rkey = mr->rkey;
  handle.length = mr->length;
  return handle;
}

application::RdmaMemoryRegion RdmaDeviceContext::RegisterRdmaMemoryRegionDmabuf(void* ptr,
                                                                                size_t size,
                                                                                int dmabuf_fd,
                                                                                int accessFlag) {
  int effectiveAccessFlag = MaybeAddRelaxedOrderingFlag(accessFlag);
  ibv_mr* mr =
      ibv_reg_dmabuf_mr(pd, 0, size, reinterpret_cast<uint64_t>(ptr), dmabuf_fd,
                        effectiveAccessFlag);
  if (!mr) {
    MORI_APP_ERROR(
        "RegisterRdmaMemoryRegionDmabuf failed! addr:{}, size:{}, dmabuf_fd:{}, accessFlag:{}, "
        "errno:{} ({})",
        ptr, size, dmabuf_fd, effectiveAccessFlag, errno, strerror(errno));
    std::abort();
  }
  MORI_APP_TRACE(
      "RegisterRdmaMemoryRegionDmabuf, addr:{}, size:{}, dmabuf_fd:{}, lkey:{}, rkey:{}\n", ptr,
      size, dmabuf_fd, mr->lkey, mr->rkey);
  mrPool.insert({ptr, mr});
  application::RdmaMemoryRegion handle;
  handle.addr = reinterpret_cast<uintptr_t>(ptr);
  handle.lkey = mr->lkey;
  handle.rkey = mr->rkey;
  handle.length = mr->length;
  return handle;
}

void RdmaDeviceContext::DeregisterRdmaMemoryRegion(void* ptr) {
  if (mrPool.find(ptr) == mrPool.end()) return;
  ibv_mr* mr = mrPool[ptr];
  ibv_dereg_mr(mr);
  mrPool.erase(ptr);
}

ibv_srq* RdmaDeviceContext::CreateRdmaSrqIfNx(const RdmaEndpointConfig& config) {
  assert(config.maxMsgSge <= GetRdmaDevice()->GetDeviceAttr()->orig_attr.max_sge);
  if (srq == nullptr) {
    ibv_srq_init_attr srqAttr = {
        .attr = {.max_wr = config.maxMsgsNum, .max_sge = config.maxMsgSge, .srq_limit = 0}};
    srq = ibv_create_srq(pd, &srqAttr);
  }
  return srq;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaDevice                                           */
/* ---------------------------------------------------------------------------------------------- */
RdmaDevice::RdmaDevice(ibv_device* device) : device(device) {
  assert(device);

  defaultContext = ibv_open_device(device);
  assert(defaultContext);

  deviceAttr.reset(new ibv_device_attr_ex{});
  int status = ibv_query_device_ex(defaultContext, NULL, deviceAttr.get());
  assert(!status);

  for (uint32_t port = 1; port <= deviceAttr->orig_attr.phys_port_cnt; ++port) {
    std::unique_ptr<ibv_port_attr> portAttr(new ibv_port_attr{});
    int ret = ibv_query_port(defaultContext, port, portAttr.get());
    assert(!ret);
    portAttrMap.emplace(port, std::move(portAttr));
  }
}

RdmaDevice::~RdmaDevice() { ibv_close_device(defaultContext); }

int RdmaDevice::GetDevicePortNum() const { return deviceAttr->orig_attr.phys_port_cnt; }

std::vector<uint32_t> RdmaDevice::GetActivePortIds() const {
  std::vector<uint32_t> activePorts;
  for (uint32_t port = 1; port <= deviceAttr->orig_attr.phys_port_cnt; ++port) {
    auto it = portAttrMap.find(port);
    if (it != portAttrMap.end() && it->second) {
      if (it->second->state == IBV_PORT_ACTIVE) {
        activePorts.push_back(port);
      }
    }
  }
  return activePorts;
}

std::string RdmaDevice::Name() const { return device->name; }

const ibv_device_attr_ex* RdmaDevice::GetDeviceAttr() const { return deviceAttr.get(); }

const std::unordered_map<uint32_t, std::unique_ptr<ibv_port_attr>>* RdmaDevice::GetPortAttrMap()
    const {
  return &portAttrMap;
}

const ibv_port_attr* RdmaDevice::GetPortAttr(uint32_t portId) const {
  auto mapPtr = GetPortAttrMap();
  auto it = mapPtr->find(portId);
  if (it != mapPtr->end() && it->second) return it->second.get();
  return nullptr;
}

double RdmaDevice::ActiveGbps(uint32_t portId) const {
  static constexpr std::array<double, 10> SpeedTable = {
      0,         // 0 unused
      2.5,       // 1 SDR
      5.0,       // 2 DDR
      10.0,      // 4 QDR
      10.3125,   // 8 FDR10
      14.0625,   // 16 FDR
      25.78125,  // 32 EDR
      50.0,      // 64 HDR
      100.0,     // 128 NDR
      250.0      // 256 XDR
  };

  static constexpr std::array<double, 6> WidthTable = {
      0,  // 0 unused
      1,  // 1 IBV_WIDTH_1X
      4,  // 2 IBV_WIDTH_4X
      0,  // 3 unused
      8,  // 4 IBV_WIDTH_8X
      12  // 5 IBV_WIDTH_12X
  };

  const ibv_port_attr* attr = GetPortAttr(portId);
  if (!attr) return 0;

  int speedIdx = 1;
  for (; speedIdx < 9; speedIdx++) {
    if ((attr->active_speed >> (speedIdx - 1)) & 0x1) break;
  }

  double laneSpeed = SpeedTable[speedIdx];
  double lanes = WidthTable[attr->active_width];
  return laneSpeed * lanes;
}

double RdmaDevice::TotalActiveGbps() const {
  uint32_t gbps = 0;
  for (auto port : GetActivePortIds()) {
    gbps += ActiveGbps(port);
  }
  return gbps;
}

RdmaDeviceContext* RdmaDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  return new RdmaDeviceContext(this, pd);
}

ActiveDevicePortList GetActiveDevicePortList(const RdmaDeviceList& devices) {
  ActiveDevicePortList activeDevPortList;
  for (RdmaDevice* device : devices) {
    std::vector<uint32_t> activePorts = device->GetActivePortIds();
    for (uint32_t port : activePorts) {
      activeDevPortList.push_back({device, port});
    }
  }
  return activeDevPortList;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaContext                                          */
/* ---------------------------------------------------------------------------------------------- */
RdmaContext::RdmaContext(RdmaBackendType backendType) : backendType(backendType) {
  deviceList = ibv_get_device_list(&nums_device);
  MORI_APP_TRACE("ibv_get_device_list nums_device: {}", nums_device);
  Initialize();
}

RdmaContext::~RdmaContext() {
  if (deviceList) ibv_free_device_list(deviceList);
  for (RdmaDevice* device : rdmaDeviceList) delete device;
}

const RdmaDeviceList& RdmaContext::GetRdmaDeviceList() const { return rdmaDeviceList; }

RdmaDevice* RdmaContext::RdmaDeviceFactory(ibv_device* inDevice) {
  ibv_context* context = ibv_open_device(inDevice);
  assert(context);

  ibv_device_attr_ex device_attr_ex;
  int status = ibv_query_device_ex(context, NULL, &device_attr_ex);
  assert(!status);
  ibv_close_device(context);

  // device_attr_ex.orig_attr.vendor_id = 0x14E4;
  if (backendType == RdmaBackendType::IBVerbs) {
    return new IBVerbsDevice(inDevice);
  } else if (backendType == RdmaBackendType::DirectVerbs) {
    switch (device_attr_ex.orig_attr.vendor_id) {
      case (static_cast<uint32_t>(RdmaDeviceVendorId::Mellanox)):
        return new Mlx5Device(inDevice);
        break;
#ifdef ENABLE_BNXT
      case (static_cast<uint32_t>(RdmaDeviceVendorId::Broadcom)):
        return new BnxtDevice(inDevice);
        break;
#endif
#ifdef ENABLE_IONIC
      case (static_cast<uint32_t>(RdmaDeviceVendorId::Pensando)):
        return new IonicDevice(inDevice);
        break;
#endif
      default:
        return nullptr;
    }
  } else {
    assert(false && "unsupported backend type");
  }
}

void RdmaContext::Initialize() {
  rdmaDeviceList.clear();

  const char* envDevices = std::getenv("MORI_RDMA_DEVICES");
  std::unordered_set<std::string> envDeviceSet;
  std::vector<std::string> envDeviceInOrder;
  bool isExcludeMode = false;
  if (envDevices) {
    std::string envDevicesStr(envDevices);
    if (!envDevicesStr.empty() && envDevicesStr[0] == '^') {
      isExcludeMode = true;
      envDevicesStr = envDevicesStr.substr(1);
    }

    std::stringstream ss(envDevicesStr);
    std::string device;
    while (std::getline(ss, device, ',')) {
      device.erase(0, device.find_first_not_of(" \t"));
      device.erase(device.find_last_not_of(" \t") + 1);
      if (!device.empty()) {
        envDeviceSet.insert(device);
        envDeviceInOrder.push_back(device);
      }
    }
  }

  if (isExcludeMode || envDeviceSet.empty()) {
    for (int i = 0; deviceList[i] != nullptr; i++) {
      if (!envDeviceSet.empty() &&
          (isExcludeMode ^ (envDeviceSet.find(deviceList[i]->name) == envDeviceSet.end()))) {
        continue;
      }
      RdmaDevice* device = RdmaDeviceFactory(deviceList[i]);
      if (device != nullptr) {
        rdmaDeviceList.push_back(device);
      }
    }
  } else {
    for (const std::string& envDevice : envDeviceInOrder) {
      for (int i = 0; deviceList[i] != nullptr; i++) {
        if (deviceList[i]->name == envDevice) {
          RdmaDevice* device = RdmaDeviceFactory(deviceList[i]);
          if (device != nullptr) {
            rdmaDeviceList.push_back(device);
          }
          break;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                 UDP Sport Configuration                                        */
/* ---------------------------------------------------------------------------------------------- */
void RdmaDeviceContext::InitializeUdpSportConfiguration() {
  // Default UDP sport configuration
  static constexpr uint16_t DEFAULT_UDP_SPORTS[RDMA_UDP_SPORT_ARRAY_SIZE] = {49153, 49154, 49155,
                                                                             49156};

  // Initialize with defaults
  for (uint32_t i = 0; i < RDMA_UDP_SPORT_ARRAY_SIZE; i++) {
    udp_sport_setting[i] = DEFAULT_UDP_SPORTS[i];
  }

  // Check for environment variable configurations
  const char* gor_port_batch = std::getenv("MORI_GOR_PORT");
  if (gor_port_batch != nullptr) {
    // Parse comma-separated values
    std::string batch_str(gor_port_batch);
    std::stringstream ss(batch_str);
    std::string item;
    uint32_t index = 0;

    while (std::getline(ss, item, ',') && index < RDMA_UDP_SPORT_ARRAY_SIZE) {
      try {
        // Support both decimal and hexadecimal (0x prefix) formats
        uint16_t port_val = static_cast<uint16_t>(std::stoul(item, nullptr, 0));
        udp_sport_setting[index] = port_val;
        index++;
      } catch (const std::exception& e) {
        MORI_APP_WARN("Invalid UDP sport value in MORI_GOR_PORT: {}, using default value", item);
      }
    }
  } else {
    // Check individual environment variables
    const char* env_vars[RDMA_UDP_SPORT_ARRAY_SIZE] = {"MORI_GOR_PORT1", "MORI_GOR_PORT2",
                                                       "MORI_GOR_PORT3", "MORI_GOR_PORT4"};

    for (uint32_t i = 0; i < RDMA_UDP_SPORT_ARRAY_SIZE; i++) {
      const char* env_val = std::getenv(env_vars[i]);
      if (env_val != nullptr) {
        try {
          uint16_t port_val = static_cast<uint16_t>(std::stoul(env_val, nullptr, 0));
          udp_sport_setting[i] = port_val;
        } catch (const std::exception& e) {
          MORI_APP_WARN("Invalid UDP sport value in {}: {}, using default value", env_vars[i],
                        env_val);
        }
      }
    }
  }

  // Log final configuration
  for (uint32_t i = 0; i < RDMA_UDP_SPORT_ARRAY_SIZE; i++) {
    MORI_APP_INFO("UDP sport[{}] = {}", i, udp_sport_setting[i]);
  }
}

uint16_t RdmaDeviceContext::GetUdpSport(uint32_t qpId) const {
  return udp_sport_setting[qpId % RDMA_UDP_SPORT_ARRAY_SIZE];
}

}  // namespace application
}  // namespace mori

#include "mori/application/transport/rdma/rdma.hpp"

#include <cassert>
#include <iostream>

#include "infiniband/verbs.h"
#include "mori/application/transport/rdma/providers/ibverbs/ibverbs.hpp"
#include "mori/application/transport/rdma/providers/mlx5/mlx5.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                        RdmaDeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
RdmaDeviceContext::RdmaDeviceContext(RdmaDevice* device, ibv_pd* inPd) : device(device), pd(inPd) {}

RdmaDeviceContext::~RdmaDeviceContext() {
  ibv_dealloc_pd(pd);
  if (srq != nullptr) ibv_destroy_srq(srq);
}

RdmaDevice* RdmaDeviceContext::GetRdmaDevice() { return device; }

ibv_context* RdmaDeviceContext::GetIbvContext() { return GetRdmaDevice()->defaultContext; }

application::RdmaMemoryRegion RdmaDeviceContext::RegisterRdmaMemoryRegion(void* ptr, size_t size,
                                                                          int accessFlag) {
  ibv_mr* mr = ibv_reg_mr(pd, ptr, size, accessFlag);
  assert(mr);
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
  deviceList = ibv_get_device_list(nullptr);
  Intialize();
}

RdmaContext::~RdmaContext() {
  if (deviceList) ibv_free_device_list(deviceList);
  for (RdmaDevice* device : rdmaDeviceList) free(device);
}

const RdmaDeviceList& RdmaContext::GetRdmaDeviceList() const { return rdmaDeviceList; }

RdmaDevice* RdmaContext::RdmaDeviceFactory(ibv_device* inDevice) {
  ibv_context* context = ibv_open_device(inDevice);
  assert(context);

  ibv_device_attr_ex device_attr_ex;
  int status = ibv_query_device_ex(context, NULL, &device_attr_ex);
  assert(!status);

  if (backendType == RdmaBackendType::IBVerbs) {
    return new IBVerbsDevice(inDevice);
  } else if (backendType == RdmaBackendType::DirectVerbs) {
    switch (device_attr_ex.orig_attr.vendor_id) {
      case (static_cast<uint32_t>(RdmaDeviceVendorId::Mellanox)):
        return new Mlx5Device(inDevice);
        break;
      default:
        return nullptr;
    }
  } else {
    assert(false && "unsupported backend type");
  }
}

void RdmaContext::Intialize() {
  rdmaDeviceList.clear();
  for (int i = 0; deviceList[i] != nullptr; i++) {
    RdmaDevice* device = RdmaDeviceFactory(deviceList[i]);
    if (device == nullptr) continue;
    rdmaDeviceList.push_back(device);
  }
}

}  // namespace application
}  // namespace mori
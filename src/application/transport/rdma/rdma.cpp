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

void RdmaDeviceContext::DeRegisterRdmaMemoryRegion(void* ptr) {
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
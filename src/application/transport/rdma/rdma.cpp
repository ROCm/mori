#include "mori/application/transport/rdma/rdma.hpp"

#include <cassert>
#include <iostream>

#include "infiniband/verbs.h"
#include "mori/application/transport/rdma/providers/mlx5/mlx5.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                        RdmaDeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
RdmaDeviceContext::RdmaDeviceContext(RdmaDevice* device, ibv_pd* inPd) : device(device), pd(inPd) {}

RdmaDeviceContext::~RdmaDeviceContext() { ibv_dealloc_pd(pd); }

RdmaDevice* RdmaDeviceContext::GetRdmaDevice() { return device; }

ibv_context* RdmaDeviceContext::GetIbvContext() { return GetRdmaDevice()->defaultContext; }

application::MemoryRegion RdmaDeviceContext::RegisterMemoryRegion(void* ptr, size_t size,
                                                                  int accessFlag) {
  ibv_mr* mr = ibv_reg_mr(pd, ptr, size, accessFlag);
  mrPool.insert({ptr, mr});
  application::MemoryRegion handle;
  handle.addr = reinterpret_cast<uintptr_t>(ptr);
  handle.lkey = mr->lkey;
  handle.rkey = mr->rkey;
  handle.length = mr->length;
  return handle;
}

void RdmaDeviceContext::DeRegisterMemoryRegion(void* ptr) {
  if (mrPool.find(ptr) == mrPool.end()) return;
  ibv_mr* mr = mrPool[ptr];
  ibv_dereg_mr(mr);
  mrPool.erase(ptr);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaDevice                                           */
/* ---------------------------------------------------------------------------------------------- */
RdmaDevice::RdmaDevice(ibv_device* device) : device(device) {
  defaultContext = ibv_open_device(device);
  assert(defaultContext);

  deviceAttr.reset(new ibv_device_attr_ex{});
  int status = ibv_query_device_ex(defaultContext, NULL, deviceAttr.get());
  assert(!status);
}

RdmaDevice::~RdmaDevice() { ibv_close_device(defaultContext); }

int RdmaDevice::GetDevicePortNum() const { return deviceAttr->orig_attr.phys_port_cnt; }

RdmaDeviceContext* RdmaDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  return new RdmaDeviceContext(this, pd);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaContext                                          */
/* ---------------------------------------------------------------------------------------------- */
RdmaContext::RdmaContext() {
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

  switch (device_attr_ex.orig_attr.vendor_id) {
    case (RdmaDeviceVendorId::Mellanox):
      return new Mlx5Device(inDevice);
      break;
    default:
      assert(false);
  }

  ibv_close_device(context);
}

void RdmaContext::Intialize() {
  rdmaDeviceList.clear();
  for (int i = 0; deviceList[i] != nullptr; i++) {
    rdmaDeviceList.push_back(RdmaDeviceFactory(deviceList[i]));
  }
}

}  // namespace application
}  // namespace mori
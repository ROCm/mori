#include "mori/application/transport/rdma/rdma.hpp"

#include <cassert>
#include <iostream>

#include "infiniband/verbs.h"

namespace mori {
namespace application {
namespace transport {
namespace rdma {

/* ---------------------------------------------------------------------------------------------- */
/*                                        RdmaDeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
RdmaDeviceContext::RdmaDeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd)
    : rdma_device(rdma_device), pd(in_pd) {}

RdmaDeviceContext::~RdmaDeviceContext() { ibv_dealloc_pd(pd); }

RdmaDevice* RdmaDeviceContext::GetRdmaDevice() { return rdma_device; }

ibv_context* RdmaDeviceContext::GetIbvContext() { return GetRdmaDevice()->default_context; }

core::transport::ibgda::MemoryRegion RdmaDeviceContext::RegisterMemoryRegion(void* ptr, size_t size,
                                                                             int access_flag) {
  ibv_mr* mr = ibv_reg_mr(pd, ptr, size, access_flag);
  core::transport::ibgda::MemoryRegion handle;
  handle.addr = reinterpret_cast<uintptr_t>(handle.addr);
  handle.lkey = mr->lkey;
  handle.rkey = mr->rkey;
  handle.length = mr->length;
  return handle;
}

void RdmaDeviceContext::DeRegisterMemoryRegion(void* ptr) {
  if (mr_pool.find(ptr) == mr_pool.end()) return;
  ibv_mr* mr = mr_pool[ptr];
  ibv_dereg_mr(mr);
  mr_pool.erase(ptr);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaDevice                                           */
/* ---------------------------------------------------------------------------------------------- */
RdmaDevice::RdmaDevice(ibv_device* in_device) : device(in_device) {
  default_context = ibv_open_device(device);
  assert(default_context);

  device_attr.reset(new ibv_device_attr_ex{});
  int status = ibv_query_device_ex(default_context, NULL, device_attr.get());
  assert(!status);
}

RdmaDevice::~RdmaDevice() { ibv_close_device(default_context); }

int RdmaDevice::GetDevicePortNum() const { return device_attr->orig_attr.phys_port_cnt; }

RdmaDeviceContext* RdmaDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(default_context);
  return new RdmaDeviceContext(this, pd);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaContext                                          */
/* ---------------------------------------------------------------------------------------------- */
RdmaContext::RdmaContext() {
  device_list = ibv_get_device_list(nullptr);
  Intialize();
}

RdmaContext::~RdmaContext() {
  if (device_list) ibv_free_device_list(device_list);
  for (RdmaDevice* rdma_device : rdma_device_list) free(rdma_device);
}

const RdmaDeviceList& RdmaContext::GetRdmaDeviceList() const { return rdma_device_list; }

RdmaDevice* RdmaContext::RdmaDeviceFactory(ibv_device* in_device) {
  ibv_context* context = ibv_open_device(in_device);
  assert(context);

  ibv_device_attr_ex device_attr_ex;
  int status = ibv_query_device_ex(context, NULL, &device_attr_ex);
  assert(!status);

  switch (device_attr_ex.orig_attr.vendor_id) {
    case (RdmaDeviceVendorId::Mellanox):
      return new Mlx5Device(in_device);
      break;
    default:
      assert(false);
  }

  ibv_close_device(context);
}

void RdmaContext::Intialize() {
  rdma_device_list.clear();
  for (int i = 0; device_list[i] != nullptr; i++) {
    rdma_device_list.push_back(RdmaDeviceFactory(device_list[i]));
  }
}

}  // namespace rdma
}  // namespace transport
}  // namespace application
}  // namespace mori
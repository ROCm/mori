#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "infiniband/verbs.h"

namespace mori {
namespace application {
namespace transport {
namespace rdma {

enum RdmaDeviceVendorId {
  Mellanox = 0x02c9,
  // Broadcom =
};

// TODO: set defalut values
struct RdmaEndpointConfig {
  size_t sq_max_wqe_num;
  size_t rq_max_wqe_num;
  size_t max_cqe_num;
  size_t alignment;
};

struct InfiniBandEndpointHandle {
  uint32_t qpn;
  uint32_t lid;
};

struct EthernetEndpointHandle {
  char gid[16];
  char mac[ETHERNET_LL_SIZE];
};

struct RdmaEndpointHandle {
  uint32_t psn;
  struct InfiniBandEndpointHandle;
  struct EthernetEndpointHandle;
};

struct RdmaEndpoint {
  RdmaEndpointHandle handle;
};

class RdmaDevice;

class RdmaDeviceContext {
 public:
  RdmaDeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd);
  ~RdmaDeviceContext();

  virtual void RegisterMemoryRegion(void* ptr, size_t size, int access_flag);
  virtual void DeRegisterMemoryRegion(void* ptr);

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) {
    assert(false && "not implementd");
  }

  RdmaDevice* GetRdmaDevice();

 protected:
  ibv_context* GetIbvContext();

 protected:
  ibv_pd* pd;

 private:
  RdmaDevice* rdma_device;
  std::unordered_map<void*, ibv_mr*> mr_pool;
};

class RdmaDevice {
 public:
  RdmaDevice(ibv_device* device);
  virtual ~RdmaDevice();

  int GetDevicePortNum() const;

  virtual RdmaDeviceContext* CreateRdmaDeviceContext();

 protected:
  friend class RdmaDeviceContext;

  // managed by RdmaContext
  ibv_device* device;
  ibv_context* default_context;

  // Managed by this object
  std::unique_ptr<ibv_device_attr_ex> device_attr;
};

using RdmaDeviceList = std::vector<RdmaDevice*>;

class RdmaContext {
 public:
  RdmaContext();
  ~RdmaContext();

  const RdmaDeviceList& GetRdmaDeviceList() const;

 private:
  RdmaDevice* RdmaDeviceFactory(ibv_device* in_device);
  void Intialize();

 private:
  ibv_device** device_list{nullptr};
  RdmaDeviceList rdma_device_list;
};

}  // namespace rdma
}  // namespace transport
}  // namespace application
}  // namespace mori
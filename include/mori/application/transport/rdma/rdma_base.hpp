#pragma once

#include <unistd.h>

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "infiniband/verbs.h"
#include "mori/core/transport/ibgda/ibgda.hpp"

namespace mori {
namespace application {
namespace transport {
namespace rdma {

enum RdmaDeviceVendorId {
  Mellanox = 0x02c9,
  // Broadcom =
};

#define PAGESIZE uint32_t(sysconf(_SC_PAGE_SIZE))

// TODO: set defalut values
struct RdmaEndpointConfig {
  uint32_t port_id{1};
  uint32_t gid_index{1};  // TODO: auto detect?
  uint32_t max_msgs_num{128};
  uint32_t max_cqe_num{128};
  uint32_t alignment{PAGESIZE};
  bool on_gpu{false};
};

struct InfiniBandEndpointHandle {
  uint32_t lid{0};
};

struct EthernetEndpointHandle {
  uint8_t gid[16];
  uint8_t mac[ETHERNET_LL_SIZE];
};

// TODO: add gid type
struct RdmaEndpointHandle {
  uint32_t psn{0};
  uint32_t qpn{0};
  struct InfiniBandEndpointHandle ib;
  struct EthernetEndpointHandle eth;
};

struct WorkQueueHandle {
  void* sq_addr{nullptr};
  void* rq_addr{nullptr};
  void* dbr_rec_addr{nullptr};
  void* dbr_addr{nullptr};
  uint32_t sq_wqe_num{0};
  uint32_t rq_wqe_num{0};
};

struct RdmaEndpoint {
  RdmaEndpointHandle handle;
  WorkQueueHandle wq_handle;
  core::transport::ibgda::CompletionQueueHandle cq_handle;
};

class RdmaDevice;

class RdmaDeviceContext {
 public:
  RdmaDeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd);
  ~RdmaDeviceContext();

  virtual core::transport::ibgda::MemoryRegion RegisterMemoryRegion(void* ptr, size_t size,
                                                                    int access_flag);
  virtual void DeRegisterMemoryRegion(void* ptr);

  // TODO: query gid entry by ibv_query_gid_table
  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) {
    assert(false && "not implementd");
  }
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local, const RdmaEndpointHandle& remote) {
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

namespace std {

static std::ostream& operator<<(
    std::ostream& s, const mori::application::transport::rdma::EthernetEndpointHandle handle) {
  std::stringstream ss;
  ss << "gid: " << std::hex;
  for (int i = 0; i < sizeof(handle.gid); i++) {
    ss << int(handle.gid[i]);
  }
  ss << ", mac: " << std::hex;
  for (int i = 0; i < sizeof(handle.mac); i++) {
    ss << int(handle.mac[i]);
  }
  s << ss.str();
  return s;
}

static std::ostream& operator<<(
    std::ostream& s, const mori::application::transport::rdma::RdmaEndpointHandle handle) {
  std::stringstream ss;
  ss << "psn: " << handle.psn << " qpn: " << handle.qpn;
  s << ss.str();
  return s;
}

}  // namespace std

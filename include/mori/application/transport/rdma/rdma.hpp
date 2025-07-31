#pragma once

#include <unistd.h>

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "infiniband/verbs.h"
#include "mori/core/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

#define MR_DEFAULT_ACCESS_FLAG                                                \
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | \
      IBV_ACCESS_REMOTE_ATOMIC

enum RdmaDeviceVendorId {
  Unknown = 0,
  Mellanox = 0x02c9,
  // Broadcom =
};

#define PAGESIZE uint32_t(sysconf(_SC_PAGE_SIZE))
#define OUTSTANDING_TABLE_SIZE (65536)

/* -------------------------------------------------------------------------- */
/*                             Rdma Data Structure                            */
/* -------------------------------------------------------------------------- */
struct RdmaEndpointConfig {
  uint32_t portId{1};
  uint32_t gidIdx{3};  // TODO: auto detect?
  uint32_t maxMsgsNum{128};
  uint32_t maxCqeNum{128};
  uint32_t alignment{PAGESIZE};
  bool onGpu{false};
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
  uint32_t portId{0};
  InfiniBandEndpointHandle ib;
  EthernetEndpointHandle eth;
};

struct WorkQueueHandle {
  uint32_t postIdx{0}; // numbers of wqe that post to work queue
  uint32_t dbTouchIdx{0}; // numbers of wqe that touched doorbell
  uint32_t doneIdx{0}; // numbers of wqe that have been consumed by nic
  uint32_t readyIdx{0};
  void* sqAddr{nullptr};
  void* rqAddr{nullptr};
  void* dbrRecAddr{nullptr};
  void* dbrAddr{nullptr};
  uint32_t sqWqeNum{0};
  uint32_t rqWqeNum{0};
  uint32_t postSendLock{0};
  uint64_t outstandingWqe[OUTSTANDING_TABLE_SIZE]{0};
};

struct CompletionQueueHandle {
  void* cqAddr{nullptr};
  void* dbrRecAddr{nullptr};
  uint32_t consIdx{0}; // numbers of cqe that have been completed
  uint32_t needConsIdx{0}; // numbers of cqe that should be consumed
  uint32_t activeIdx{0}; // numbers of cqe that under processing but not completed
  uint32_t cq_consumer{0};
  uint32_t cqeNum{0};
  uint32_t cqeSize{0};
  uint32_t pollCqLock{0};
};

struct RdmaEndpoint {
  RdmaDeviceVendorId vendorId{RdmaDeviceVendorId::Unknown};
  RdmaEndpointHandle handle;
  WorkQueueHandle wqHandle;
  CompletionQueueHandle cqHandle;

  __device__ __host__ core::ProviderType GetProviderType() {
    if (vendorId == Mellanox) {
      return core::ProviderType::MLX5;
    } else {
      printf("unknown vendorId %d", vendorId);
      assert(false);
    }
    return core::ProviderType::Unknown;
  }
};

class RdmaDevice;

struct MemoryRegion {
  uintptr_t addr;
  uint32_t lkey;
  uint32_t rkey;
  size_t length;
};

/* -------------------------------------------------------------------------- */
/*                              RdmaDeviceContext                             */
/* -------------------------------------------------------------------------- */
class RdmaDeviceContext {
 public:
  RdmaDeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd);
  ~RdmaDeviceContext();

  virtual MemoryRegion RegisterMemoryRegion(void* ptr, size_t size,
                                            int accessFlag = MR_DEFAULT_ACCESS_FLAG);
  virtual void DeRegisterMemoryRegion(void* ptr);

  // TODO: query gid entry by ibv_query_gid_table
  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) {
    assert(false && "not implementd");
  }
  void ConnectEndpoint(const RdmaEndpoint& local, const RdmaEndpoint& remote) {
    ConnectEndpoint(local.handle, remote.handle);
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
  RdmaDevice* device;
  std::unordered_map<void*, ibv_mr*> mrPool;
};

/* -------------------------------------------------------------------------- */
/*                                 RdmaDevice                                 */
/* -------------------------------------------------------------------------- */
class RdmaDevice {
 public:
  RdmaDevice(ibv_device* device);
  virtual ~RdmaDevice();

  int GetDevicePortNum() const;
  std::vector<uint32_t> GetActivePortIds() const;
  const ibv_device_attr_ex* GetDeviceAttr() const;
  const std::unordered_map<uint32_t, std::unique_ptr<ibv_port_attr>>* GetPortAttrMap() const;

  std::string Name() const;

  virtual RdmaDeviceContext* CreateRdmaDeviceContext();

 protected:
  friend class RdmaDeviceContext;

  ibv_device* device;
  ibv_context* defaultContext;

  std::unique_ptr<ibv_device_attr_ex> deviceAttr;
  std::unordered_map<uint32_t, std::unique_ptr<ibv_port_attr>> portAttrMap;
};

using RdmaDeviceList = std::vector<RdmaDevice*>;
using ActiveDevicePort = std::pair<RdmaDevice*, uint32_t>;
using ActiveDevicePortList = std::vector<ActiveDevicePort>;

ActiveDevicePortList GetActiveDevicePortList(const RdmaDeviceList&);

/* -------------------------------------------------------------------------- */
/*                                 RdmaContext                                */
/* -------------------------------------------------------------------------- */
class RdmaContext {
 public:
  RdmaContext();
  ~RdmaContext();

  const RdmaDeviceList& GetRdmaDeviceList() const;

 private:
  RdmaDevice* RdmaDeviceFactory(ibv_device* inDevice);
  void Intialize();

 private:
  ibv_device** deviceList{nullptr};
  RdmaDeviceList rdmaDeviceList;
};

}  // namespace application
}  // namespace mori

namespace std {

static std::ostream& operator<<(std::ostream& s,
                                const mori::application::EthernetEndpointHandle handle) {
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

static std::ostream& operator<<(std::ostream& s,
                                const mori::application::RdmaEndpointHandle handle) {
  std::stringstream ss;
  ss << "psn: " << handle.psn << " qpn: " << handle.qpn << " ib [" << handle.ib.lid << "] "
     << " eth [" << handle.eth << "]";
  s << ss.str();
  return s;
}

}  // namespace std

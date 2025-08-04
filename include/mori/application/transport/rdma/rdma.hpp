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

enum class RdmaBackendType : uint32_t {
  Unknown = 0,
  DirectVerbs = 1,
  IBVerbs = 2,
};

enum class RdmaDeviceVendorId : uint32_t {
  Unknown = 0,
  Mellanox = 0x02c9,
  Broadcom = 0x14E4,
};

template <typename T>
RdmaDeviceVendorId ToRdmaDeviceVendorId(T v) {
  switch (v) {
    case static_cast<T>(RdmaDeviceVendorId::Mellanox):
      return RdmaDeviceVendorId::Mellanox;
    default:
      return RdmaDeviceVendorId::Unknown;
  }
  return RdmaDeviceVendorId::Unknown;
}

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
  uint32_t maxMsgSge{1};
  uint32_t alignment{PAGESIZE};
  bool onGpu{false};
  bool withCompChannel{false};
  bool enableSrq{false};
};

struct InfiniBandEndpointHandle {
  uint32_t lid{0};
  constexpr bool operator==(const InfiniBandEndpointHandle& rhs) const noexcept {
    return lid == rhs.lid;
  }
};

struct EthernetEndpointHandle {
  uint8_t gid[16];
  uint8_t mac[ETHERNET_LL_SIZE];

  constexpr bool operator==(const EthernetEndpointHandle& rhs) const noexcept {
    return std::equal(std::begin(gid), std::end(gid), std::begin(rhs.gid)) &&
           std::equal(std::begin(mac), std::end(mac), std::begin(rhs.mac));
  }
};

// TODO: add gid type
struct RdmaEndpointHandle {
  uint32_t psn{0};
  uint32_t qpn{0};
  uint32_t portId{0};
  InfiniBandEndpointHandle ib;
  EthernetEndpointHandle eth;

  constexpr bool operator==(const RdmaEndpointHandle& rhs) const noexcept {
    return psn == rhs.psn && qpn == rhs.qpn && portId == rhs.portId && ib == rhs.ib &&
           eth == rhs.eth;
  }
};

using RdmaEndpointHandleVec = std::vector<RdmaEndpointHandle>;

struct WorkQueueAttrs {
  uint32_t wqeNum{0};
  uint32_t wqeSize{0};
  uint64_t wqSize{0};
  uint32_t head{0};
  uint32_t postIdx{0};
  uint32_t wqeShift{0};
  uint32_t offset{0};
};

struct WorkQueueHandle {
  uint32_t postIdx{0};     // numbers of wqe that post to work queue
  uint32_t dbTouchIdx{0};  // numbers of wqe that touched doorbell
  uint32_t doneIdx{0};     // numbers of wqe that have been consumed by nic
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
  void* dbrAddr{nullptr};
  uint32_t consIdx{0}; // numbers of cqe that have been completed
  uint32_t needConsIdx{0}; // numbers of cqe that should be consumed
  uint32_t activeIdx{0}; // numbers of cqe that under processing but not completed
  uint32_t cq_consumer{0};
  uint32_t cqeNum{0};
  uint32_t cqeSize{0};
  uint32_t pollCqLock{0};
};

struct IBVerbsHandle {
  ibv_qp* qp{nullptr};
  ibv_cq* cq{nullptr};
  ibv_srq* srq{nullptr};
  ibv_comp_channel* compCh{nullptr};
};

struct RdmaEndpoint {
  RdmaDeviceVendorId vendorId{RdmaDeviceVendorId::Unknown};
  RdmaEndpointHandle handle;
  // TODO(@ditian12): we should use an opaque handle to reference the actual transport context
  // handles, for direct verbs it should be WorkQueueHandle/CompletionQueueHandle, for ib verbs, it
  // should be ibv structures
  WorkQueueHandle wqHandle;
  CompletionQueueHandle cqHandle;
  IBVerbsHandle ibvHandle;

  __device__ __host__ core::ProviderType GetProviderType() {
    if (vendorId == RdmaDeviceVendorId::Mellanox) {
      return core::ProviderType::MLX5;
    } if (vendorId == Broadcom) {
      return core::ProviderType::BNXT;
    } else {
      printf("unknown vendorId %d", vendorId);
      assert(false);
    }
    return core::ProviderType::Unknown;
  }
};

class RdmaDevice;

struct RdmaMemoryRegion {
  uintptr_t addr{0};
  uint32_t lkey{0};
  uint32_t rkey{0};
  size_t length{0};
};

/* -------------------------------------------------------------------------- */
/*                              RdmaDeviceContext                             */
/* -------------------------------------------------------------------------- */
class RdmaDeviceContext {
 public:
  RdmaDeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd);
  virtual ~RdmaDeviceContext();

  virtual RdmaMemoryRegion RegisterRdmaMemoryRegion(void* ptr, size_t size,
                                                    int accessFlag = MR_DEFAULT_ACCESS_FLAG);
  virtual void DeregisterRdmaMemoryRegion(void* ptr);

  // TODO: query gid entry by ibv_query_gid_table
  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) {
    assert(false && "not implemented");
  }
  void ConnectEndpoint(const RdmaEndpoint& local, const RdmaEndpoint& remote) {
    ConnectEndpoint(local.handle, remote.handle);
  }
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local, const RdmaEndpointHandle& remote) {
    assert(false && "not implemented");
  }

  ibv_srq* CreateRdmaSrqIfNx(const RdmaEndpointConfig&);

  RdmaDevice* GetRdmaDevice();
  ibv_context* GetIbvContext();
  ibv_srq* GetIbvSrq() { return srq; }

 protected:
  ibv_pd* pd{nullptr};
  ibv_srq* srq{nullptr};

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
  const ibv_port_attr* GetPortAttr(uint32_t portId) const;

  double ActiveGbps(uint32_t portId) const;
  double TotalActiveGbps() const;

  std::string Name() const;

  virtual RdmaDeviceContext* CreateRdmaDeviceContext();
  ibv_context* GetIbvContext() const { return defaultContext; }
  ibv_device* GetIbvDevice() const { return device; }

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
  RdmaContext(RdmaBackendType);
  ~RdmaContext();

  const RdmaDeviceList& GetRdmaDeviceList() const;
  int nums_device;
 private:
  RdmaDevice* RdmaDeviceFactory(ibv_device* inDevice);
  void Initialize();

 public:
  RdmaBackendType backendType;

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

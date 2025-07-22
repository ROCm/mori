#pragma once

#include "infiniband/verbs.h"
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

class IBVerbsDeviceContext : public RdmaDeviceContext {
 public:
  IBVerbsDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd);
  ~IBVerbsDeviceContext() override;

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local,
                               const RdmaEndpointHandle& remote) override;

 private:
  std::unordered_map<void*, std::unique_ptr<ibv_cq>> cqPool;
  std::unordered_map<uint32_t, std::unique_ptr<ibv_qp>> qpPool;
};

class IBVerbsDevice : public RdmaDevice {
 public:
  IBVerbsDevice(ibv_device* device);
  ~IBVerbsDevice() override;

  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};

}  // namespace application
}  // namespace mori

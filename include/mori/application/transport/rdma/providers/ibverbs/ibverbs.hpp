#pragma once

#include "infiniband/verbs.h"
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

class IBVerbsDeviceContext : public RdmaDeviceContext {
 public:
  IBVerbsDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd);
  ~IBVerbsDeviceContext();

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local,
                               const RdmaEndpointHandle& remote) override;
};

class IBVerbsDevice : public RdmaDevice {
 public:
  IBVerbsDevice(ibv_device* device);
  ~IBVerbsDevice();

  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};

}  // namespace application
}  // namespace mori

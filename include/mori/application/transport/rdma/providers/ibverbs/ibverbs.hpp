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
                               const RdmaEndpointHandle& remote, uint32_t qpId = 0) override;

 private:
  std::unordered_map<void*, ibv_cq*> cqPool;
  std::unordered_map<uint32_t, ibv_qp*> qpPool;
};

class IBVerbsDevice : public RdmaDevice {
 public:
  IBVerbsDevice(ibv_device* device);
  ~IBVerbsDevice() override;

  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};

}  // namespace application
}  // namespace mori

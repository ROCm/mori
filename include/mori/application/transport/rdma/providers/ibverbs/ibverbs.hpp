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

#include <mutex>

#include "infiniband/verbs.h"
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

class IBVerbsDeviceContext : public RdmaDeviceContext {
 public:
  IBVerbsDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd, bool ownsPd = true);
  ~IBVerbsDeviceContext() override;

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local, const RdmaEndpointHandle& remote,
                               uint32_t qpId = 0) override;
  bool DestroyRdmaEndpointNoThrow(const RdmaEndpoint&) noexcept override;

  struct ProxyRecvInfo {
    void* buf;
    uint32_t lkey;
    uint32_t count;
  };
  ProxyRecvInfo GetProxyRecvInfo(uint32_t qpn) const;
  // ConnectEndpoint gives each QP a registered receive buffer for the shmem
  // proxy's SEND_WITH_IMM emulation (GetProxyRecvInfo) unless this is turned
  // off, e.g. for QPs that never receive. Applies to later ConnectEndpoint
  // calls. The buffer is freed with its endpoint or with this context.
  void SetProxyRecvBuffersEnabled(bool enabled) { proxyRecvBuffersEnabled = enabled; }

 private:
  struct ProxyRecvBuffer {
    ProxyRecvInfo info;
    ibv_mr* mr;
  };
  void FreeProxyRecvBufferLocked(uint32_t qpn);  // caller holds poolMu

  mutable std::mutex poolMu;
  std::unordered_map<void*, ibv_cq*> cqPool;
  std::unordered_map<uint32_t, ibv_qp*> qpPool;
  std::vector<ibv_comp_channel*> compChPool;
  std::unordered_map<uint32_t, ProxyRecvBuffer> proxyRecvBuffers;
  bool proxyRecvBuffersEnabled{true};
};

class IBVerbsDevice : public RdmaDevice {
 public:
  IBVerbsDevice(ibv_device* device);
  ~IBVerbsDevice() override;

  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};

}  // namespace application
}  // namespace mori

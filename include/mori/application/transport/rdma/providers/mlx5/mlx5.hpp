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
#include <set>

#include "mori/application/transport/rdma/providers/dv_loader.hpp"
#include "mori/application/transport/rdma/providers/mlx5/mlx5_dv.h"
#include "mori/application/transport/rdma/providers/mlx5/mlx5_ifc.hpp"
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                        Device Attributes                                       */
/* ---------------------------------------------------------------------------------------------- */
static size_t GetMlx5CqeSize() { return sizeof(mlx5_cqe64); }

// TODO: figure out how does 192 computed?
static size_t GetMlx5SqWqeSize() {
  return (192 + sizeof(mlx5_wqe_data_seg) + MLX5_SEND_WQE_BB - 1) / MLX5_SEND_WQE_BB *
         MLX5_SEND_WQE_BB;
}

static size_t GetMlx5RqWqeSize() { return sizeof(mlx5_wqe_data_seg); }

struct HcaCapability {
  uint32_t portType{0};
  uint32_t dbrRegSize{0};

  bool IsEthernet() const { return portType == MLX5_CAP_PORT_TYPE_ETH; }
  bool IsInfiniBand() const { return portType == MLX5_CAP_PORT_TYPE_IB; }
};

HcaCapability QueryHcaCap(ibv_context* context);

/* ---------------------------------------------------------------------------------------------- */
/*                                 Device Data Structure Container                                */
/* ---------------------------------------------------------------------------------------------- */
// TODO: refactor Mlx5CqContainer so its structure is similar to Mlx5QpContainer
class Mlx5CqContainer {
 public:
  Mlx5CqContainer(ibv_context* context, const RdmaEndpointConfig& config);
  ~Mlx5CqContainer();

 public:
  RdmaEndpointConfig config;
  uint32_t cqeNum;

 public:
  uint32_t cqn{0};
  void* cqUmemAddr{nullptr};
  void* cqDbrUmemAddr{nullptr};
  mlx5dv_devx_umem* cqUmem{nullptr};
  mlx5dv_devx_umem* cqDbrUmem{nullptr};
  mlx5dv_devx_uar* uar{nullptr};
  mlx5dv_devx_obj* cq{nullptr};
};

class Mlx5DeviceContext;  // Forward declaration

class Mlx5QpContainer {
 public:
  Mlx5QpContainer(ibv_context* context, const RdmaEndpointConfig& config, uint32_t cqn,
                  uint32_t pdn, Mlx5DeviceContext* device_context);
  ~Mlx5QpContainer();

  void ModifyRst2Init();
  void ModifyInit2Rtr(const RdmaEndpointHandle& local_handle,
                      const RdmaEndpointHandle& remote_handle, const ibv_port_attr& portAttr,
                      uint32_t qpId = 0);
  void ModifyRtr2Rts(const RdmaEndpointHandle& local_handle);

  void* GetSqAddress();
  void* GetRqAddress();

  Mlx5DeviceContext* GetDeviceContext() { return device_context; }

 private:
  void ComputeQueueAttrs(const RdmaEndpointConfig& config);
  void CreateQueuePair(uint32_t cqn, uint32_t pdn);
  void DestroyQueuePair();

 public:
  ibv_context* context;
  Mlx5DeviceContext* device_context;

 public:
  RdmaEndpointConfig config;
  WorkQueueAttrs rqAttrs;
  WorkQueueAttrs sqAttrs;
  size_t qpTotalSize{0};

 public:
  size_t qpn{0};
  void* qpUmemAddr{nullptr};
  void* qpDbrUmemAddr{nullptr};
  mlx5dv_devx_umem* qpUmem{nullptr};
  mlx5dv_devx_umem* qpDbrUmem{nullptr};
  mlx5dv_devx_uar* qpUar{nullptr};
  void* qpUarPtr{nullptr};
  mlx5dv_devx_obj* qp{nullptr};

  // Atomic internal buffer fields
  void* atomicIbufAddr{nullptr};
  size_t atomicIbufSize{0};
  ibv_mr* atomicIbufMr{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                        Mlx5DeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
class Mlx5DeviceContext : public RdmaDeviceContext {
 public:
  Mlx5DeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd);
  ~Mlx5DeviceContext() override;

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local, const RdmaEndpointHandle& remote,
                               uint32_t qpId = 0) override;

  // rdma-core hands out ONE non-cached UAR per ibv_context: _mlx5dv_devx_alloc_uar() short
  // circuits MLX5DV_UAR_ALLOC_TYPE_NC to mlx5_get_singleton_nc_uar(), so every QP on this device
  // gets the identical reg_addr. Register it with HIP exactly once (bnxt already does this).
  bool TryRegisterUar(void* uar_addr);
  bool TryUnregisterUar(void* uar_addr);

 private:
  uint32_t pdn;

  // Declared BEFORE the pools: members are destroyed in reverse declaration order, and
  // ~Mlx5QpContainer calls TryUnregisterUar. If these came after qpPool they would already be
  // gone by the time the QPs are destroyed, and the unregister would lock a destroyed mutex and
  // mutate a destroyed set -- silent heap corruption at teardown.
  std::set<void*> registeredUars;
  std::mutex uarMutex;

  std::unordered_map<uint32_t, std::unique_ptr<Mlx5CqContainer>> cqPool;
  std::unordered_map<uint32_t, std::unique_ptr<Mlx5QpContainer>> qpPool;
};

class Mlx5Device : public RdmaDevice {
 public:
  Mlx5Device(ibv_device* device);
  ~Mlx5Device() override;

  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};

}  // namespace application
}  // namespace mori

namespace std {

static std::ostream& operator<<(std::ostream& s, const mori::application::WorkQueueAttrs wq_attrs) {
  std::stringstream ss;
  ss << "wqeNum: " << wq_attrs.wqeNum << " wqeSize: " << wq_attrs.wqeSize
     << " wqSize: " << wq_attrs.wqSize << " postIdx: " << wq_attrs.postIdx
     << " wqeShift: " << wq_attrs.wqeShift << " offset: " << wq_attrs.offset;
  s << ss.str();
  return s;
}

}  // namespace std

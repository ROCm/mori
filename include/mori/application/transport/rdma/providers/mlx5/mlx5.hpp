#pragma once

#include "infiniband/mlx5dv.h"
#include "mori/application/transport/rdma/rdma.hpp"
#include "src/application/transport/rdma/providers/mlx5/mlx5_ifc.hpp"

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

struct WorkQueueAttrs {
  uint32_t wqeNum{0};
  uint32_t wqeSize{0};
  uint64_t wqSize{0};
  uint32_t head{0};
  uint32_t postIdx{0};
  uint32_t wqeShift{0};
  uint32_t offset{0};
};

class Mlx5QpContainer {
 public:
  Mlx5QpContainer(ibv_context* context, const RdmaEndpointConfig& config, uint32_t cqn,
                  uint32_t pdn);
  ~Mlx5QpContainer();

  void ModifyRst2Init();
  void ModifyInit2Rtr(const RdmaEndpointHandle& remote_handle);
  void ModifyRtr2Rts(const RdmaEndpointHandle& local_handle);

  void* GetSqAddress();
  void* GetRqAddress();

 private:
  void ComputeQueueAttrs(const RdmaEndpointConfig& config);
  void CreateQueuePair(uint32_t cqn, uint32_t pdn);
  void DestroyQueuePair();

 public:
  ibv_context* context;

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
};

/* ---------------------------------------------------------------------------------------------- */
/*                                        Mlx5DeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
class Mlx5DeviceContext : public RdmaDeviceContext {
 public:
  Mlx5DeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd);
  ~Mlx5DeviceContext();

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local,
                               const RdmaEndpointHandle& remote) override;

 private:
  uint32_t pdn;

  // std::unordered_map<uint32_t, std::unique_ptr<Mlx5CqContainer>> cq_pool;
  // std::unordered_map<uint32_t, std::unique_ptr<Mlx5QpContainer>> qp_pool;

  std::unordered_map<uint32_t, Mlx5CqContainer*> cqPool;
  std::unordered_map<uint32_t, Mlx5QpContainer*> qpPool;
};

class Mlx5Device : public RdmaDevice {
 public:
  Mlx5Device(ibv_device* device);
  ~Mlx5Device();

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
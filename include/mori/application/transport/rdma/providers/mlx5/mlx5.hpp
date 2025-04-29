#pragma once

#include "infiniband/mlx5dv.h"
#include "mori/application/transport/rdma/rdma_base.hpp"
#include "src/application/transport/rdma/providers/mlx5/mlx5_ifc.hpp"

namespace mori {
namespace application {
namespace transport {
namespace rdma {

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
  uint32_t port_type;
  uint32_t dbr_reg_size;

  bool IsEthernet() const { return port_type == MLX5_CAP_PORT_TYPE_ETH; }
  bool IsInfiniBand() const { return port_type == MLX5_CAP_PORT_TYPE_IB; }
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
  uint32_t cqn{0};
  void* cq_umem_addr{nullptr};
  void* cq_dbr_umem_addr{nullptr};
  mlx5dv_devx_umem* cq_umem{nullptr};
  mlx5dv_devx_umem* cq_dbr_umem{nullptr};
  mlx5dv_devx_uar* uar{nullptr};
  mlx5dv_devx_obj* cq{nullptr};
};

struct WorkQueueAttrs {
  uint32_t wqe_num{0};
  uint32_t wqe_size{0};
  uint64_t wq_size{0};
  uint32_t head{0};
  uint32_t post_idx{0};
  uint32_t wqe_shift{0};
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

 private:
  void ComputeQueueAttrs(const RdmaEndpointConfig& config);
  void CreateQueuePair(uint32_t cqn, uint32_t pdn);
  void DestroyQueuePair();

 public:
  ibv_context* context;

 public:
  RdmaEndpointConfig config;
  WorkQueueAttrs rq_attrs;
  WorkQueueAttrs sq_attrs;
  size_t qp_total_size{0};

 public:
  size_t qpn{0};
  void* qp_umem_addr{nullptr};
  void* qp_dbr_umem_addr{nullptr};
  mlx5dv_devx_umem* qp_umem{nullptr};
  mlx5dv_devx_umem* qp_dbr_umem{nullptr};
  mlx5dv_devx_uar* qp_uar{nullptr};
  void* qp_uar_ptr{nullptr};
  mlx5dv_devx_obj* qp{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                        Mlx5DeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
class Mlx5DeviceContext : public RdmaDeviceContext {
 public:
  Mlx5DeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd);
  ~Mlx5DeviceContext();

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local,
                               const RdmaEndpointHandle& remote) override;

 private:
  uint32_t pdn;

  std::unordered_map<uint32_t, std::unique_ptr<Mlx5CqContainer>> cq_pool;
  std::unordered_map<uint32_t, std::unique_ptr<Mlx5QpContainer>> qp_pool;
};

class Mlx5Device : public RdmaDevice {
 public:
  Mlx5Device(ibv_device* device);
  ~Mlx5Device();

  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};

}  // namespace rdma
}  // namespace transport
}  // namespace application
}  // namespace mori
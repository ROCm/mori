#pragma once

#include "infiniband/mlx5dv.h"
#include "mori/application/transport/rdma/rdma_base.hpp"

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
  return (192 + sizeof(mlx5_wqe_data_seg) + MLX5_SEND_WQE_BB) / MLX5_SEND_WQE_BB * MLX5_SEND_WQE_BB;
}

static size_t GetMlx5RqWqeSize() { return sizeof(mlx5_wqe_data_seg); }

struct HcaCapability {
  uint32_t port_type;
  uint32_t dbr_reg_size;
};

HcaCapability QueryHcaCap(ibv_context* context);

/* ---------------------------------------------------------------------------------------------- */
/*                                 Device Data Structure Container                                */
/* ---------------------------------------------------------------------------------------------- */
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

class Mlx5QpContainer {
 public:
  Mlx5QpContainer(ibv_context* context, const RdmaEndpointConfig& config, uint32_t cqn,
                  uint32_t pdn);
  ~Mlx5QpContainer();

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

  RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;

 private:
  uint32_t pdn;

  std::unordered_map<uint32_t, Mlx5CqContainer*> cq_pool;
  std::unordered_map<uint32_t, Mlx5QpContainer*> qp_pool;
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
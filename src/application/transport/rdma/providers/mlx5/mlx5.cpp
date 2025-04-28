#include "mori/application/transport/rdma/providers/mlx5/mlx5.hpp"

#include <hip/hip_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include "mori/application/utils/hip_check.hpp"
#include "mori/application/utils/math.hpp"
#include "src/application/transport/rdma/providers/mlx5/mlx5_ifc.hpp"
#include "src/application/transport/rdma/providers/mlx5/mlx5_prm.hpp"

namespace mori {
namespace application {
namespace transport {
namespace rdma {

/* ---------------------------------------------------------------------------------------------- */
/*                                        Device Attributes                                       */
/* ---------------------------------------------------------------------------------------------- */
HcaCapability QueryHcaCap(ibv_context* context) {
  int status;
  uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
      0,
  };
  uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
      0,
  };

  DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
  DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod, HCA_CAP_OPMOD_GET_CUR);

  status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                   sizeof(cmd_cap_out));
  assert(!status);

  HcaCapability hca_cap;

  hca_cap.port_type = DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.port_type);

  uint32_t log_bf_reg_size =
      DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.log_bf_reg_size);
  hca_cap.dbr_reg_size = 1LLU << log_bf_reg_size;

  return hca_cap;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Mlx5CqContainer */
/* ---------------------------------------------------------------------------------------------- */
Mlx5CqContainer::Mlx5CqContainer(ibv_context* context, const RdmaEndpointConfig& config) {
  int status;
  uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_cq_in)] = {
      0,
  };
  uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_cq_out)] = {
      0,
  };

  // Allocate user memory for CQ
  // TODO: accept memory allocated by user?
  int cq_size = RoundUpPowOfTwo(GetMlx5CqeSize() * config.max_cqe_num);
  cq_size = (cq_size + config.alignment - 1) / config.alignment * config.alignment;
  HIP_RUNTIME_CHECK(hipMalloc(&cq_umem_addr, cq_size));
  HIP_RUNTIME_CHECK(hipMemset(cq_umem_addr, 0, cq_size));

  cq_umem = mlx5dv_devx_umem_reg(context, cq_umem_addr, cq_size, IBV_ACCESS_LOCAL_WRITE);
  assert(cq_umem);

  // Allocate user memory for CQ DBR (doorbell?)
  HIP_RUNTIME_CHECK(hipMalloc(&cq_dbr_umem_addr, 8));
  HIP_RUNTIME_CHECK(hipMemset(cq_dbr_umem_addr, 0, 8));

  cq_dbr_umem = mlx5dv_devx_umem_reg(context, cq_dbr_umem_addr, 8, IBV_ACCESS_LOCAL_WRITE);
  assert(cq_dbr_umem);

  // Allocate user access region
  uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
  assert(uar->page_id != 0);

  // Initialize CQ
  DEVX_SET(create_cq_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_CQ);
  DEVX_SET(create_cq_in, cmd_in, cq_umem_valid, 0x1);
  DEVX_SET(create_cq_in, cmd_in, cq_umem_id, cq_umem->umem_id);

  void* cq_context = DEVX_ADDR_OF(create_cq_in, cmd_in, cq_context);
  DEVX_SET(cqc, cq_context, dbr_umem_valid, 0x1);
  DEVX_SET(cqc, cq_context, dbr_umem_id, cq_dbr_umem->umem_id);
  DEVX_SET(cqc, cq_context, log_cq_size, LogCeil2(config.max_cqe_num));
  DEVX_SET(cqc, cq_context, uar_page, uar->page_id);

  uint32_t eqn;
  status = mlx5dv_devx_query_eqn(context, 0, &eqn);
  assert(!status);
  DEVX_SET(cqc, cq_context, c_eqn, eqn);

  cq = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
  assert(cq);

  cqn = DEVX_GET(create_cq_out, cmd_out, cqn);
}

Mlx5CqContainer::~Mlx5CqContainer() {
  mlx5dv_devx_umem_dereg(cq_umem);
  mlx5dv_devx_umem_dereg(cq_dbr_umem);
  mlx5dv_devx_free_uar(uar);
  mlx5dv_devx_obj_destroy(cq);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Mlx5QpContainer                                        */
/* ---------------------------------------------------------------------------------------------- */
Mlx5QpContainer::Mlx5QpContainer(ibv_context* context, const RdmaEndpointConfig& config,
                                 uint32_t cqn, uint32_t pdn) {
  int status = 0;
  uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {
      0,
  };
  uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {
      0,
  };

  // Allocate user memory for QP
  int qp_size =
      config.rq_max_wqe_num * GetMlx5RqWqeSize() + config.sq_max_wqe_num * GetMlx5SqWqeSize();
  qp_size = RoundUpPowOfTwo(qp_size);
  qp_size = (qp_size + config.alignment - 1) / config.alignment * config.alignment;

  HIP_RUNTIME_CHECK(hipMalloc(&qp_umem_addr, qp_size));
  HIP_RUNTIME_CHECK(hipMemset(qp_umem_addr, 0, qp_size));

  qp_umem = mlx5dv_devx_umem_reg(context, qp_umem_addr, qp_size, IBV_ACCESS_LOCAL_WRITE);
  assert(qp_umem);

  // Allocate user memory for DBR (doorbell?)
  HIP_RUNTIME_CHECK(hipMalloc(&qp_dbr_umem_addr, 8));
  HIP_RUNTIME_CHECK(hipMemset(qp_dbr_umem_addr, 0, 8));

  qp_dbr_umem = mlx5dv_devx_umem_reg(context, qp_dbr_umem_addr, 8, IBV_ACCESS_LOCAL_WRITE);
  assert(qp_dbr_umem);

  // Allocate user access region
  qp_uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
  assert(qp_uar);
  assert(qp_uar->page_id != 0);

  uint32_t flag = hipHostRegisterPortable | hipHostRegisterMapped | hipHostRegisterIoMemory;
  HIP_RUNTIME_CHECK(hipHostRegister(qp_uar->reg_addr, QueryHcaCap(context).dbr_reg_size, flag));
  HIP_RUNTIME_CHECK(hipHostGetDevicePointer(&qp_uar_ptr, qp_uar->reg_addr, 0));

  // TODO: check for correctness
  int log_rq_size = int(log2(config.rq_max_wqe_num - 1)) + 1;
  int rq_wqe_shift = int(log2(RoundUpPowOfTwo(GetMlx5RqWqeSize() * config.rq_max_wqe_num) - 1)) + 1;
  int log_rq_stride = rq_wqe_shift - 4;
  int log_sq_size = int(log2(config.sq_max_wqe_num - 1)) + 1;

  // Initialize QP
  DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
  DEVX_SET(create_qp_in, cmd_in, wq_umem_id, qp_umem->umem_id);
  DEVX_SET64(create_qp_in, cmd_in, wq_umem_offset, 0);
  DEVX_SET(create_qp_in, cmd_in, wq_umem_valid, 0x1);

  void* qp_context = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
  DEVX_SET(qpc, qp_context, st, MLX5_QPC_ST_RC);
  DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
  DEVX_SET(qpc, qp_context, pd, pdn);
  DEVX_SET(qpc, qp_context, uar_page, qp_uar->page_id);  // BF register
  DEVX_SET(qpc, qp_context, cqn_snd, cqn);
  DEVX_SET(qpc, qp_context, cqn_rcv, cqn);
  DEVX_SET(qpc, qp_context, log_sq_size, log_sq_size);
  DEVX_SET(qpc, qp_context, log_rq_size, log_rq_size);
  DEVX_SET(qpc, qp_context, log_rq_stride, log_rq_stride);
  DEVX_SET(qpc, qp_context, ts_format, 0x1);
  DEVX_SET(qpc, qp_context, cs_req, 0);
  DEVX_SET(qpc, qp_context, cs_res, 0);
  DEVX_SET(qpc, qp_context, dbr_umem_valid, 0x1);  // Enable dbr_umem_id
  DEVX_SET64(qpc, qp_context, dbr_addr,
             0);  // Offset of dbr_umem_id (behavior changed because of dbr_umem_valid)
  DEVX_SET(qpc, qp_context, dbr_umem_id, qp_dbr_umem->umem_id);  // DBR buffer
  DEVX_SET(qpc, qp_context, page_offset, 0);

  qp = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
  assert(qp);

  qpn = DEVX_GET(create_qp_out, cmd_out, qpn);
}

Mlx5QpContainer::~Mlx5QpContainer() {}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Mlx5DeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
Mlx5DeviceContext::Mlx5DeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd)
    : RdmaDeviceContext(rdma_device, in_pd) {
  mlx5dv_obj dv_obj{};
  mlx5dv_pd dvpd{};
  // TODO: should we zero init?
  // memset(dvpd, 0, sizeof(mlx5dv_pd));
  // memset(dv_obj, 0, sizeof(mlx5dv_obj));
  dv_obj.pd.in = pd;
  dv_obj.pd.out = &dvpd;
  int status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
  assert(!status);
  pdn = dvpd.pdn;
}

Mlx5DeviceContext::~Mlx5DeviceContext() {}

RdmaEndpoint Mlx5DeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  ibv_context* context = GetIbvContext();

  Mlx5CqContainer* cq = new Mlx5CqContainer(context, config);
  cq_pool.insert({cq->cqn, cq});

  Mlx5QpContainer* qp = new Mlx5QpContainer(context, config, cq->cqn, pdn);
  qp_pool.insert({qp->qpn, qp});

  return {};
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           Mlx5Device                                           */
/* ---------------------------------------------------------------------------------------------- */
Mlx5Device::Mlx5Device(ibv_device* in_device) : RdmaDevice(in_device) {}
Mlx5Device::~Mlx5Device() {}

RdmaDeviceContext* Mlx5Device::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(default_context);
  return new Mlx5DeviceContext(this, pd);
}

}  // namespace rdma
}  // namespace transport
}  // namespace application
}  // namespace mori
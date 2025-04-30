#include "mori/application/transport/rdma/providers/mlx5/mlx5.hpp"

#include <hip/hip_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include <iostream>

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
Mlx5CqContainer::Mlx5CqContainer(ibv_context* context, const RdmaEndpointConfig& config)
    : config(config) {
  int status;
  uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_cq_in)] = {
      0,
  };
  uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_cq_out)] = {
      0,
  };

  // Allocate user memory for CQ
  // TODO: accept memory allocated by user?
  cqe_num = config.max_cqe_num;
  int cq_size = RoundUpPowOfTwo(GetMlx5CqeSize() * cqe_num);
  // TODO: adjust cqe_num after aligning?
  cq_size = (cq_size + config.alignment - 1) / config.alignment * config.alignment;

  if (config.on_gpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&cq_umem_addr, cq_size));
    HIP_RUNTIME_CHECK(hipMemset(cq_umem_addr, 0, cq_size));
  } else {
    int status = posix_memalign(&cq_umem_addr, config.alignment, cq_size);
    memset(cq_umem_addr, 0, cq_size);
    assert(!status);
  }

  cq_umem = mlx5dv_devx_umem_reg(context, cq_umem_addr, cq_size, IBV_ACCESS_LOCAL_WRITE);
  assert(cq_umem);

  // Allocate user memory for CQ DBR (doorbell?)
  if (config.on_gpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&cq_dbr_umem_addr, 8));
    HIP_RUNTIME_CHECK(hipMemset(cq_dbr_umem_addr, 0, 8));
  } else {
    int status = posix_memalign(&cq_dbr_umem_addr, 8, 8);
    memset(cq_dbr_umem_addr, 0, 8);
    assert(!status);
  }

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
  DEVX_SET(cqc, cq_context, log_cq_size, LogCeil2(cqe_num));
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
                                 uint32_t cqn, uint32_t pdn)
    : context(context), config(config) {
  ComputeQueueAttrs(config);
  CreateQueuePair(cqn, pdn);
}

Mlx5QpContainer::~Mlx5QpContainer() { DestroyQueuePair(); }

void Mlx5QpContainer::ComputeQueueAttrs(const RdmaEndpointConfig& config) {
  // Receive queue attributes
  rq_attrs.wqe_size = RoundUpPowOfTwo(GetMlx5RqWqeSize() * config.max_recv_sge);
  uint32_t max_msgs_num = RoundUpPowOfTwo(config.max_msgs_num);
  rq_attrs.wq_size = std::max(rq_attrs.wqe_size * max_msgs_num, uint32_t(MLX5_SEND_WQE_BB));
  rq_attrs.wqe_num = ceil(rq_attrs.wq_size / rq_attrs.wqe_size);
  rq_attrs.wqe_shift = log2(rq_attrs.wqe_size - 1) + 1;
  rq_attrs.offset = 0;

  // Send queue attributes
  sq_attrs.offset = rq_attrs.wq_size;
  sq_attrs.wqe_size = GetMlx5SqWqeSize();
  sq_attrs.wq_size = RoundUpPowOfTwo(sq_attrs.wqe_size * config.max_msgs_num);
  sq_attrs.wqe_num = ceil(sq_attrs.wq_size / MLX5_SEND_WQE_BB);
  sq_attrs.wqe_shift = MLX5_SEND_WQE_SHIFT;

  // Queue pair attributes
  qp_total_size = RoundUpPowOfTwo(rq_attrs.wq_size + sq_attrs.wq_size);
  qp_total_size = (qp_total_size + config.alignment - 1) / config.alignment * config.alignment;

  std::cout << "rq[ " << rq_attrs << "] sq[ " << sq_attrs << "]" << std::endl;
}

void Mlx5QpContainer::CreateQueuePair(uint32_t cqn, uint32_t pdn) {
  int status = 0;
  uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {
      0,
  };
  uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {
      0,
  };

  // Allocate user memory for QP

  if (config.on_gpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&qp_umem_addr, qp_total_size));
    HIP_RUNTIME_CHECK(hipMemset(qp_umem_addr, 0, qp_total_size));
  } else {
    status = posix_memalign(&qp_umem_addr, config.alignment, qp_total_size);
    memset(qp_umem_addr, 0, qp_total_size);
    assert(!status);
  }

  qp_umem = mlx5dv_devx_umem_reg(context, qp_umem_addr, qp_total_size, IBV_ACCESS_LOCAL_WRITE);
  assert(qp_umem);

  // Allocate user memory for DBR (doorbell?)
  if (config.on_gpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&qp_dbr_umem_addr, 8));
    HIP_RUNTIME_CHECK(hipMemset(qp_dbr_umem_addr, 0, 8));
  } else {
    status = posix_memalign(&qp_dbr_umem_addr, 8, 8);
    memset(qp_dbr_umem_addr, 0, 8);
    assert(!status);
  }

  qp_dbr_umem = mlx5dv_devx_umem_reg(context, qp_dbr_umem_addr, 8, IBV_ACCESS_LOCAL_WRITE);
  assert(qp_dbr_umem);

  // Allocate user access region
  qp_uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
  assert(qp_uar);
  assert(qp_uar->page_id != 0);

  if (config.on_gpu) {
    uint32_t flag = hipHostRegisterPortable | hipHostRegisterMapped | hipHostRegisterIoMemory;
    HIP_RUNTIME_CHECK(hipHostRegister(qp_uar->reg_addr, QueryHcaCap(context).dbr_reg_size, flag));
    HIP_RUNTIME_CHECK(hipHostGetDevicePointer(&qp_uar_ptr, qp_uar->reg_addr, 0));
  } else {
    qp_uar_ptr = qp_uar->reg_addr;
  }

  // TODO: check for correctness
  uint32_t log_rq_size = int(log2(rq_attrs.wqe_num - 1)) + 1;
  uint32_t log_rq_stride = rq_attrs.wqe_shift - 4;
  uint32_t log_sq_size = int(log2(sq_attrs.wqe_num - 1)) + 1;

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

void Mlx5QpContainer::DestroyQueuePair() {
  if (qp_umem_addr) HIP_RUNTIME_CHECK(hipFree(qp_umem_addr));
  if (qp_dbr_umem_addr) HIP_RUNTIME_CHECK(hipFree(qp_dbr_umem_addr));
  if (qp_dbr_umem) mlx5dv_devx_umem_dereg(qp_dbr_umem);
  if (qp_uar) {
    mlx5dv_devx_free_uar(qp_uar);
    HIP_RUNTIME_CHECK(hipHostUnregister(qp_uar->reg_addr));
  }
  if (qp) mlx5dv_devx_obj_destroy(qp);
}

void* Mlx5QpContainer::GetSqAddress() { return static_cast<char*>(qp_umem_addr) + sq_attrs.offset; }

void* Mlx5QpContainer::GetRqAddress() { return static_cast<char*>(qp_umem_addr) + rq_attrs.offset; }

void Mlx5QpContainer::ModifyRst2Init() {
  uint8_t rst2init_cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {
      0,
  };
  uint8_t rst2init_cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {
      0,
  };

  DEVX_SET(rst2init_qp_in, rst2init_cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
  DEVX_SET(rst2init_qp_in, rst2init_cmd_in, qpn, qpn);

  void* qpc = DEVX_ADDR_OF(rst2init_qp_in, rst2init_cmd_in, qpc);
  DEVX_SET(qpc, qpc, rwe, 1); /* remote write access */
  DEVX_SET(qpc, qpc, rre, 1); /* remote read access */
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.port_id);
  int status = mlx5dv_devx_obj_modify(qp, rst2init_cmd_in, sizeof(rst2init_cmd_in),
                                      rst2init_cmd_out, sizeof(rst2init_cmd_out));
  assert(!status);
}

void Mlx5QpContainer::ModifyInit2Rtr(const RdmaEndpointHandle& remote_handle) {
  uint8_t init2rtr_cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
      0,
  };
  uint8_t init2rtr_cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
      0,
  };

  DEVX_SET(init2rtr_qp_in, init2rtr_cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
  DEVX_SET(init2rtr_qp_in, init2rtr_cmd_in, qpn, qpn);

  void* qpc = DEVX_ADDR_OF(init2rtr_qp_in, init2rtr_cmd_in, qpc);
  DEVX_SET(qpc, qpc, mtu, IBV_MTU_1024);
  DEVX_SET(qpc, qpc, log_msg_max, 20);
  DEVX_SET(qpc, qpc, remote_qpn, remote_handle.qpn);
  DEVX_SET(qpc, qpc, next_rcv_psn, remote_handle.psn);
  DEVX_SET(qpc, qpc, min_rnr_nak, 12);

  qpc = DEVX_ADDR_OF(init2rtr_qp_in, init2rtr_cmd_in, qpc);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.port_id);

  HcaCapability hca_cap = QueryHcaCap(context);
  if (hca_cap.IsEthernet()) {
    memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip), remote_handle.eth.gid,
           sizeof(remote_handle.eth.gid));

    memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), remote_handle.eth.mac,
           sizeof(remote_handle.eth.mac));
    DEVX_SET(qpc, qpc, primary_address_path.hop_limit, 64);
    DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, config.gid_index);
    DEVX_SET(qpc, qpc, primary_address_path.udp_sport, 0xC000);
  } else if (hca_cap.IsInfiniBand()) {
    DEVX_SET(qpc, qpc, primary_address_path.rlid, remote_handle.ib.lid);
  } else {
    assert(false);
  }

  int status = mlx5dv_devx_obj_modify(qp, init2rtr_cmd_in, sizeof(init2rtr_cmd_in),
                                      init2rtr_cmd_out, sizeof(init2rtr_cmd_out));
  assert(!status);
}

void Mlx5QpContainer::ModifyRtr2Rts(const RdmaEndpointHandle& local_handle) {
  uint8_t rtr2rts_cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {
      0,
  };
  uint8_t rtr2rts_cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {
      0,
  };

  DEVX_SET(rtr2rts_qp_in, rtr2rts_cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
  DEVX_SET(rtr2rts_qp_in, rtr2rts_cmd_in, qpn, qpn);

  void* qpc = DEVX_ADDR_OF(rtr2rts_qp_in, rtr2rts_cmd_in, qpc);
  DEVX_SET(qpc, qpc, log_sra_max, 20);
  DEVX_SET(qpc, qpc, next_send_psn, local_handle.psn);
  DEVX_SET(qpc, qpc, retry_count, 7);
  DEVX_SET(qpc, qpc, rnr_retry, 7);
  DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 14);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.port_id);

  int status = mlx5dv_devx_obj_modify(qp, rtr2rts_cmd_in, sizeof(rtr2rts_cmd_in), rtr2rts_cmd_out,
                                      sizeof(rtr2rts_cmd_out));
  assert(!status);
}

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
  Mlx5QpContainer* qp = new Mlx5QpContainer(context, config, cq->cqn, pdn);

  RdmaEndpoint endpoint;
  endpoint.handle.psn = 0;

  HcaCapability hca_cap = QueryHcaCap(context);

  endpoint.handle.qpn = qp->qpn;
  if (hca_cap.IsEthernet()) {
    uint32_t out[DEVX_ST_SZ_DW(query_roce_address_out)] = {};
    uint32_t in[DEVX_ST_SZ_DW(query_roce_address_in)] = {};

    DEVX_SET(query_roce_address_in, in, opcode, MLX5_CMD_OP_QUERY_ROCE_ADDRESS);
    DEVX_SET(query_roce_address_in, in, roce_address_index, config.gid_index);
    DEVX_SET(query_roce_address_in, in, vhca_port_num, config.port_id);

    int status = mlx5dv_devx_general_cmd(context, in, sizeof(in), out, sizeof(out));
    assert(!status);

    memcpy(endpoint.handle.eth.gid,
           DEVX_ADDR_OF(query_roce_address_out, out, roce_address.source_l3_address),
           sizeof(endpoint.handle.eth.gid));

    memcpy(endpoint.handle.eth.mac,
           DEVX_ADDR_OF(query_roce_address_out, out, roce_address.source_mac_47_32),
           sizeof(endpoint.handle.eth.mac));
  } else if (hca_cap.IsInfiniBand()) {
    ibv_port_attr port_attr;
    int status = ibv_query_port(context, config.port_id, &port_attr);
    assert(!status);
    endpoint.handle.ib.lid = port_attr.lid;
  } else {
    assert(false);
  }

  endpoint.wq_handle.sq_addr = qp->GetSqAddress();
  endpoint.wq_handle.rq_addr = qp->GetRqAddress();
  endpoint.wq_handle.dbr_rec_addr = qp->qp_dbr_umem_addr;
  endpoint.wq_handle.dbr_addr = qp->qp_uar_ptr;

  endpoint.cq_handle.cq_addr = cq->cq_umem_addr;
  endpoint.cq_handle.consumer_idx = 0;
  endpoint.cq_handle.cqe_num = cq->cqe_num;
  endpoint.cq_handle.cqe_size = GetMlx5CqeSize();

  // cq_pool.insert({cq->cqn, std::move(std::unique_ptr<Mlx5CqContainer>(cq))});
  // qp_pool.insert({qp->qpn, std::move(std::unique_ptr<Mlx5QpContainer>(qp))});

  cq_pool.insert({cq->cqn, cq});
  qp_pool.insert({qp->qpn, qp});

  return endpoint;
}

void Mlx5DeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                        const RdmaEndpointHandle& remote) {
  uint32_t local_qpn = local.qpn;
  assert(qp_pool.find(local_qpn) != qp_pool.end());
  Mlx5QpContainer* qp = qp_pool.at(local_qpn);  //.get();
  qp->ModifyRst2Init();
  qp->ModifyInit2Rtr(remote);
  qp->ModifyRtr2Rts(local);
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
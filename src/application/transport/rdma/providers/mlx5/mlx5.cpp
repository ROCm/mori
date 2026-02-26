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
#include "mori/application/transport/rdma/providers/mlx5/mlx5.hpp"

#include <hip/hip_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include <iostream>

#include "mori/application/utils/check.hpp"
#include "mori/application/utils/math.hpp"
#include "mori/utils/mori_log.hpp"
#include "mori/application/transport/rdma/providers/mlx5/mlx5_ifc.hpp"
#include "mori/application/transport/rdma/providers/mlx5/mlx5_prm.hpp"

namespace mori {
namespace application {

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

  hca_cap.portType = DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.port_type);

  uint32_t logBfRegSize =
      DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.log_bf_reg_size);
  hca_cap.dbrRegSize = 1LLU << logBfRegSize;

  MORI_APP_TRACE("MLX5 HCA capabilities: portType={}, dbrRegSize={}", hca_cap.portType,
                 hca_cap.dbrRegSize);

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
  cqeNum = config.maxCqeNum;
  int cqSize = RoundUpPowOfTwo(GetMlx5CqeSize() * cqeNum);
  // TODO: adjust cqe_num after aligning?
  cqSize = (cqSize + config.alignment - 1) / config.alignment * config.alignment;

  if (config.onGpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&cqUmemAddr, cqSize));
    HIP_RUNTIME_CHECK(hipMemset(cqUmemAddr, 0, cqSize));
  } else {
    int status = posix_memalign(&cqUmemAddr, config.alignment, cqSize);
    memset(cqUmemAddr, 0, cqSize);
    assert(!status);
  }

  cqUmem = mlx5dv_devx_umem_reg(context, cqUmemAddr, cqSize, IBV_ACCESS_LOCAL_WRITE);
  assert(cqUmem);

  // Allocate user memory for CQ DBR (doorbell?)
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&cqDbrUmemAddr, 8));
    HIP_RUNTIME_CHECK(hipMemset(cqDbrUmemAddr, 0, 8));
  } else {
    int status = posix_memalign(&cqDbrUmemAddr, 8, 8);
    memset(cqDbrUmemAddr, 0, 8);
    assert(!status);
  }

  cqDbrUmem = mlx5dv_devx_umem_reg(context, cqDbrUmemAddr, 8, IBV_ACCESS_LOCAL_WRITE);
  assert(cqDbrUmem);

  // Allocate user access region
  uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
  assert(uar->page_id != 0);

  // Initialize CQ
  DEVX_SET(create_cq_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_CQ);
  DEVX_SET(create_cq_in, cmd_in, cq_umem_valid, 0x1);
  DEVX_SET(create_cq_in, cmd_in, cq_umem_id, cqUmem->umem_id);

  void* cq_context = DEVX_ADDR_OF(create_cq_in, cmd_in, cq_context);
  DEVX_SET(cqc, cq_context, dbr_umem_valid, 0x1);
  DEVX_SET(cqc, cq_context, dbr_umem_id, cqDbrUmem->umem_id);
  DEVX_SET(cqc, cq_context, log_cq_size, LogCeil2(cqeNum));
  DEVX_SET(cqc, cq_context, uar_page, uar->page_id);

  uint32_t eqn;
  status = mlx5dv_devx_query_eqn(context, 0, &eqn);
  assert(!status);
  DEVX_SET(cqc, cq_context, c_eqn, eqn);

  cq = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
  assert(cq);

  cqn = DEVX_GET(create_cq_out, cmd_out, cqn);

  MORI_APP_TRACE("MLX5 CQ created: cqn={}, cqeNum={}, cqSize={}, cqUmemAddr=0x{:x}, uar_page_id={}",
                 cqn, cqeNum, cqSize, reinterpret_cast<uintptr_t>(cqUmemAddr), uar->page_id);
}

Mlx5CqContainer::~Mlx5CqContainer() {
  mlx5dv_devx_umem_dereg(cqUmem);
  mlx5dv_devx_umem_dereg(cqDbrUmem);
  mlx5dv_devx_free_uar(uar);
  mlx5dv_devx_obj_destroy(cq);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Mlx5QpContainer                                        */
/* ---------------------------------------------------------------------------------------------- */
Mlx5QpContainer::Mlx5QpContainer(ibv_context* context, const RdmaEndpointConfig& config,
                                 uint32_t cqn, uint32_t pdn, Mlx5DeviceContext* device_context)
    : context(context), config(config), device_context(device_context) {
  ComputeQueueAttrs(config);
  CreateQueuePair(cqn, pdn);
}

Mlx5QpContainer::~Mlx5QpContainer() { DestroyQueuePair(); }

void Mlx5QpContainer::ComputeQueueAttrs(const RdmaEndpointConfig& config) {
  // Receive queue attributes//
  rqAttrs.wqeSize = GetMlx5RqWqeSize();
  uint32_t maxMsgsNum = RoundUpPowOfTwo(config.maxMsgsNum);
  rqAttrs.wqSize = std::max(rqAttrs.wqeSize * maxMsgsNum, uint32_t(MLX5_SEND_WQE_BB));
  rqAttrs.wqeNum = ceil(rqAttrs.wqSize / rqAttrs.wqeSize);
  rqAttrs.wqeShift = log2(rqAttrs.wqeSize - 1) + 1;
  rqAttrs.offset = 0;

  // Send queue attributes
  sqAttrs.offset = rqAttrs.wqSize;
  sqAttrs.wqeSize = GetMlx5SqWqeSize();
  sqAttrs.wqSize = RoundUpPowOfTwo(sqAttrs.wqeSize * maxMsgsNum);
  sqAttrs.wqeNum = ceil(sqAttrs.wqSize / MLX5_SEND_WQE_BB);
  sqAttrs.wqeShift = MLX5_SEND_WQE_SHIFT;

  // Queue pair attributes
  qpTotalSize = RoundUpPowOfTwo(rqAttrs.wqSize + sqAttrs.wqSize);
  qpTotalSize = (qpTotalSize + config.alignment - 1) / config.alignment * config.alignment;

  MORI_APP_TRACE(
      "MLX5 Queue attributes computed - RQ: wqeSize={}, wqSize={}, wqeNum={}, offset={} | SQ: "
      "wqeSize={}, wqSize={}, wqeNum={}, offset={} | Total: {}",
      rqAttrs.wqeSize, rqAttrs.wqSize, rqAttrs.wqeNum, rqAttrs.offset, sqAttrs.wqeSize,
      sqAttrs.wqSize, sqAttrs.wqeNum, sqAttrs.offset, qpTotalSize);
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

  if (config.onGpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&qpUmemAddr, qpTotalSize));
    HIP_RUNTIME_CHECK(hipMemset(qpUmemAddr, 0, qpTotalSize));
  } else {
    status = posix_memalign(&qpUmemAddr, config.alignment, qpTotalSize);
    memset(qpUmemAddr, 0, qpTotalSize);
    assert(!status);
  }

  qpUmem = mlx5dv_devx_umem_reg(context, qpUmemAddr, qpTotalSize, IBV_ACCESS_LOCAL_WRITE);
  assert(qpUmem);

  // Allocate user memory for DBR (doorbell?)
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&qpDbrUmemAddr, 8));
    HIP_RUNTIME_CHECK(hipMemset(qpDbrUmemAddr, 0, 8));
  } else {
    status = posix_memalign(&qpDbrUmemAddr, 8, 8);
    memset(qpDbrUmemAddr, 0, 8);
    assert(!status);
  }

  qpDbrUmem = mlx5dv_devx_umem_reg(context, qpDbrUmemAddr, 8, IBV_ACCESS_LOCAL_WRITE);
  assert(qpDbrUmem);

  // Allocate and register atomic internal buffer (ibuf)
  atomicIbufSize = (RoundUpPowOfTwo(config.atomicIbufSlots) + 1) * ATOMIC_IBUF_SLOT_SIZE;
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&atomicIbufAddr, atomicIbufSize));
    HIP_RUNTIME_CHECK(hipMemset(atomicIbufAddr, 0, atomicIbufSize));
  } else {
    status = posix_memalign(&atomicIbufAddr, config.alignment, atomicIbufSize);
    memset(atomicIbufAddr, 0, atomicIbufSize);
    assert(!status);
  }

  // Register atomic ibuf as independent memory region
  int atomicIbufAccessFlag =
      MaybeAddRelaxedOrderingFlag(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  atomicIbufMr = ibv_reg_mr(device_context->GetIbvPd(), atomicIbufAddr, atomicIbufSize,
                            atomicIbufAccessFlag);
  assert(atomicIbufMr);

  MORI_APP_TRACE(
      "MLX5 Atomic ibuf allocated: addr=0x{:x}, slots={}, size={}, lkey=0x{:x}, rkey=0x{:x}",
      reinterpret_cast<uintptr_t>(atomicIbufAddr), RoundUpPowOfTwo(config.atomicIbufSlots),
      atomicIbufSize, atomicIbufMr->lkey, atomicIbufMr->rkey);

  // Allocate user access region
  qpUar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
  assert(qpUar);
  assert(qpUar->page_id != 0);

  if (config.onGpu) {
    uint32_t flag = hipHostRegisterPortable | hipHostRegisterMapped;
    HIP_RUNTIME_CHECK(hipHostRegister(qpUar->reg_addr, QueryHcaCap(context).dbrRegSize, flag));
    HIP_RUNTIME_CHECK(hipHostGetDevicePointer(&qpUarPtr, qpUar->reg_addr, 0));
  } else {
    qpUarPtr = qpUar->reg_addr;
  }

  // TODO: check for correctness
  uint32_t logRqSize = int(log2(rqAttrs.wqeNum - 1)) + 1;
  uint32_t logRqStride = rqAttrs.wqeShift - 4;
  uint32_t logSqSize = int(log2(sqAttrs.wqeNum - 1)) + 1;

  // Initialize QP
  DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
  DEVX_SET(create_qp_in, cmd_in, wq_umem_id, qpUmem->umem_id);
  DEVX_SET64(create_qp_in, cmd_in, wq_umem_offset, 0);
  DEVX_SET(create_qp_in, cmd_in, wq_umem_valid, 0x1);

  void* qp_context = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
  DEVX_SET(qpc, qp_context, st, MLX5_QPC_ST_RC);
  DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
  DEVX_SET(qpc, qp_context, pd, pdn);
  DEVX_SET(qpc, qp_context, uar_page, qpUar->page_id);  // BF register
  DEVX_SET(qpc, qp_context, cqn_snd, cqn);
  DEVX_SET(qpc, qp_context, cqn_rcv, cqn);
  DEVX_SET(qpc, qp_context, log_sq_size, logSqSize);
  DEVX_SET(qpc, qp_context, log_rq_size, logRqSize);
  DEVX_SET(qpc, qp_context, log_rq_stride, logRqStride);
  DEVX_SET(qpc, qp_context, ts_format, 0x1);
  DEVX_SET(qpc, qp_context, cs_req, 0);
  DEVX_SET(qpc, qp_context, cs_res, 0);
  DEVX_SET(qpc, qp_context, dbr_umem_valid, 0x1);  // Enable dbr_umem_id
  DEVX_SET64(qpc, qp_context, dbr_addr,
             0);  // Offset of dbr_umem_id (behavior changed because of dbr_umem_valid)
  DEVX_SET(qpc, qp_context, dbr_umem_id, qpDbrUmem->umem_id);  // DBR buffer
  DEVX_SET(qpc, qp_context, page_offset, 0);

  qp = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
  assert(qp);

  qpn = DEVX_GET(create_qp_out, cmd_out, qpn);

  MORI_APP_TRACE(
      "MLX5 QP created: qpn={}, qpTotalSize={}, sqWqeNum={}, rqWqeNum={}, sqAddr=0x{:x}, "
      "rqAddr=0x{:x}",
      qpn, qpTotalSize, sqAttrs.wqeNum, rqAttrs.wqeNum, reinterpret_cast<uintptr_t>(GetSqAddress()),
      reinterpret_cast<uintptr_t>(GetRqAddress()));
}

void Mlx5QpContainer::DestroyQueuePair() {
  if (atomicIbufMr) {
    ibv_dereg_mr(atomicIbufMr);
    atomicIbufMr = nullptr;
  }
  if (atomicIbufAddr) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipFree(atomicIbufAddr));
    } else {
      free(atomicIbufAddr);
    }
    atomicIbufAddr = nullptr;
  }

  if (qpUmem) mlx5dv_devx_umem_dereg(qpUmem);
  if (qpUmemAddr) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipFree(qpUmemAddr));
    } else {
      free(qpUmemAddr);
    }
  }
  if (qpDbrUmem) mlx5dv_devx_umem_dereg(qpDbrUmem);
  if (qpDbrUmemAddr) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipFree(qpDbrUmemAddr));
    } else {
      free(qpDbrUmemAddr);
    }
  }
  if (qpUar) {
    if (config.onGpu) {
      hipPointerAttribute_t attr;
      HIP_RUNTIME_CHECK(hipPointerGetAttributes(&attr, qpUar->reg_addr));
      // Multiple qp may share the same uar address, only unregister once
      if ((attr.type == hipMemoryTypeHost) && (attr.hostPointer != nullptr)) {
        HIP_RUNTIME_CHECK(hipHostUnregister(qpUar->reg_addr));
      }
    }
    mlx5dv_devx_free_uar(qpUar);
  }
  if (qp) mlx5dv_devx_obj_destroy(qp);
}

void* Mlx5QpContainer::GetSqAddress() { return static_cast<char*>(qpUmemAddr) + sqAttrs.offset; }

void* Mlx5QpContainer::GetRqAddress() { return static_cast<char*>(qpUmemAddr) + rqAttrs.offset; }

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
  DEVX_SET(qpc, qpc, rae, 1);
  DEVX_SET(qpc, qpc, atomic_mode, 0x3);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.portId);

  DEVX_SET(qpc, qpc, pm_state, 0x3);
  DEVX_SET(qpc, qpc, counter_set_id, 0x0);

  int status = mlx5dv_devx_obj_modify(qp, rst2init_cmd_in, sizeof(rst2init_cmd_in),
                                      rst2init_cmd_out, sizeof(rst2init_cmd_out));
  assert(!status);
}

void Mlx5QpContainer::ModifyInit2Rtr(const RdmaEndpointHandle& local_handle,
                                     const RdmaEndpointHandle& remote_handle,
                                     const ibv_port_attr& portAttr, uint32_t qpId) {
  uint8_t init2rtr_cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
      0,
  };
  uint8_t init2rtr_cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
      0,
  };

  DEVX_SET(init2rtr_qp_in, init2rtr_cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
  DEVX_SET(init2rtr_qp_in, init2rtr_cmd_in, qpn, qpn);

  void* qpc = DEVX_ADDR_OF(init2rtr_qp_in, init2rtr_cmd_in, qpc);
  DEVX_SET(qpc, qpc, mtu, portAttr.active_mtu);
  DEVX_SET(qpc, qpc, log_msg_max, 30);
  DEVX_SET(qpc, qpc, remote_qpn, remote_handle.qpn);
  DEVX_SET(qpc, qpc, next_rcv_psn, remote_handle.psn);
  DEVX_SET(qpc, qpc, min_rnr_nak, 12);
  DEVX_SET(qpc, qpc, log_rra_max, 20);

  qpc = DEVX_ADDR_OF(init2rtr_qp_in, init2rtr_cmd_in, qpc);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.portId);

  // HcaCapability hca_cap = QueryHcaCap(context);
  if (portAttr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip), remote_handle.eth.gid,
           sizeof(remote_handle.eth.gid));

    memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), remote_handle.eth.mac,
           sizeof(remote_handle.eth.mac));
    DEVX_SET(qpc, qpc, primary_address_path.hop_limit, 64);
    DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, local_handle.eth.gidIdx);
    // Use shared UDP sport configuration with qpId-based selection
    uint16_t selected_udp_sport = device_context->GetUdpSport(qpId);
    DEVX_SET(qpc, qpc, primary_address_path.udp_sport, selected_udp_sport | 0xC000);
    MORI_APP_TRACE("MLX5 QP {} using UDP sport {} (qpId={}, index={})", qpn, selected_udp_sport,
                   qpId, qpId % RDMA_UDP_SPORT_ARRAY_SIZE);
  } else if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
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
  DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 20);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.portId);

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
  dv_obj.pd.in = pd;
  dv_obj.pd.out = &dvpd;
  int status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
  assert(!status);
  pdn = dvpd.pdn;
}

Mlx5DeviceContext::~Mlx5DeviceContext() {}

RdmaEndpoint Mlx5DeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  assert(!config.withCompChannel && !config.enableSrq && "not implemented");
  ibv_context* context = GetIbvContext();

  Mlx5CqContainer* cq = new Mlx5CqContainer(context, config);
  Mlx5QpContainer* qp = new Mlx5QpContainer(context, config, cq->cqn, pdn, this);
  const ibv_device_attr_ex* deviceAttr = GetRdmaDevice()->GetDeviceAttr();

  RdmaEndpoint endpoint;
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;
  endpoint.handle.maxSge = config.maxMsgSge;

  const ibv_port_attr* portAttr = GetRdmaDevice()->GetPortAttr(config.portId);
  assert(portAttr);
  HcaCapability hca_cap = QueryHcaCap(context);

  endpoint.handle.qpn = qp->qpn;
  if (hca_cap.IsEthernet()) {
    GidSelectionResult gidSelection =
        AutoSelectGidIndex(context, config.portId, portAttr, config.gidIdx);
    assert(gidSelection.gidIdx >= 0 && gidSelection.valid);
    int gidIdx = gidSelection.gidIdx;

    uint32_t out[DEVX_ST_SZ_DW(query_roce_address_out)] = {};
    uint32_t in[DEVX_ST_SZ_DW(query_roce_address_in)] = {};

    DEVX_SET(query_roce_address_in, in, opcode, MLX5_CMD_OP_QUERY_ROCE_ADDRESS);
    DEVX_SET(query_roce_address_in, in, roce_address_index, gidIdx);
    DEVX_SET(query_roce_address_in, in, vhca_port_num, config.portId);

    int status = mlx5dv_devx_general_cmd(context, in, sizeof(in), out, sizeof(out));
    assert(!status);

    memcpy(endpoint.handle.eth.gid,
           DEVX_ADDR_OF(query_roce_address_out, out, roce_address.source_l3_address),
           sizeof(endpoint.handle.eth.gid));

    memcpy(endpoint.handle.eth.mac,
           DEVX_ADDR_OF(query_roce_address_out, out, roce_address.source_mac_47_32),
           sizeof(endpoint.handle.eth.mac));
    endpoint.handle.eth.gidIdx = gidIdx;
  } else if (hca_cap.IsInfiniBand()) {
    auto mapPtr = GetRdmaDevice()->GetPortAttrMap();
    auto it = mapPtr->find(config.portId);
    if (it != mapPtr->end() && it->second) {
      ibv_port_attr* port_attr = it->second.get();
      endpoint.handle.ib.lid = port_attr->lid;
    } else {
      assert(false && "Port attribute not found for given port ID");
    }
  } else {
    assert(false);
  }

  endpoint.vendorId = RdmaDeviceVendorId::Mellanox;

  endpoint.wqHandle.sqAddr = qp->GetSqAddress();
  endpoint.wqHandle.rqAddr = qp->GetRqAddress();
  endpoint.wqHandle.dbrRecAddr = qp->qpDbrUmemAddr;
  endpoint.wqHandle.dbrAddr = qp->qpUarPtr;
  endpoint.wqHandle.sqWqeNum = qp->sqAttrs.wqeNum;
  endpoint.wqHandle.rqWqeNum = qp->rqAttrs.wqeNum;

  endpoint.cqHandle.cqAddr = cq->cqUmemAddr;
  endpoint.cqHandle.consIdx = 0;
  endpoint.cqHandle.cqeNum = cq->cqeNum;
  endpoint.cqHandle.cqeSize = GetMlx5CqeSize();
  endpoint.cqHandle.dbrRecAddr = cq->cqDbrUmemAddr;

  // Set atomic internal buffer information
  endpoint.atomicIbuf.addr = reinterpret_cast<uintptr_t>(qp->atomicIbufAddr);
  endpoint.atomicIbuf.lkey = qp->atomicIbufMr->lkey;
  endpoint.atomicIbuf.rkey = qp->atomicIbufMr->rkey;
  endpoint.atomicIbuf.nslots = RoundUpPowOfTwo(config.atomicIbufSlots);

  cqPool.insert({cq->cqn, std::move(std::unique_ptr<Mlx5CqContainer>(cq))});
  qpPool.insert({qp->qpn, std::move(std::unique_ptr<Mlx5QpContainer>(qp))});

  MORI_APP_TRACE(
      "MLX5 endpoint created: qpn={}, cqn={}, portId={}, gidIdx={}, atomicIbuf addr=0x{:x}, "
      "nslots={}",
      qp->qpn, cq->cqn, config.portId, endpoint.handle.eth.gidIdx, endpoint.atomicIbuf.addr,
      endpoint.atomicIbuf.nslots);

  return endpoint;
}

void Mlx5DeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                        const RdmaEndpointHandle& remote, uint32_t qpId) {
  uint32_t local_qpn = local.qpn;
  assert(qpPool.find(local_qpn) != qpPool.end());
  Mlx5QpContainer* qp = qpPool.at(local_qpn).get();

  MORI_APP_TRACE("MLX5 connecting endpoint: local_qpn={}, remote_qpn={}, qpId={}", local_qpn,
                 remote.qpn, qpId);

  RdmaDevice* rdmaDevice = GetRdmaDevice();
  const ibv_device_attr_ex* deviceAttr = rdmaDevice->GetDeviceAttr();
  const ibv_port_attr& portAttr = *(rdmaDevice->GetPortAttrMap()->find(local.portId)->second);
  qp->ModifyRst2Init();
  qp->ModifyInit2Rtr(local, remote, portAttr, qpId);
  qp->ModifyRtr2Rts(local);

  MORI_APP_TRACE("MLX5 endpoint connected successfully: local_qpn={}, remote_qpn={}", local_qpn,
                 remote.qpn);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           Mlx5Device                                           */
/* ---------------------------------------------------------------------------------------------- */
Mlx5Device::Mlx5Device(ibv_device* in_device) : RdmaDevice(in_device) {}
Mlx5Device::~Mlx5Device() {}

RdmaDeviceContext* Mlx5Device::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  return new Mlx5DeviceContext(this, pd);
}

}  // namespace application
}  // namespace mori

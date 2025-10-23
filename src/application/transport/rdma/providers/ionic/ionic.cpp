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
#include "mori/application/transport/rdma/providers/ionic/ionic.hpp"

#include <hip/hip_runtime.h>
#include <infiniband/ionic_dv.h>
#include <infiniband/ionic_fw.h>
#include <infiniband/verbs.h>
#include <iostream>

#include "mori/application/utils/check.hpp"
#include "mori/application/utils/math.hpp"
#include "mori/utils/mori_log.hpp"
//#include "src/application/transport/rdma/providers/mlx5/mlx5_ifc.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                        Device Attributes                                       */
/* ---------------------------------------------------------------------------------------------- */


/* ---------------------------------------------------------------------------------------------- */
/*                                          IonicCqContainer 				                	  */
/* ---------------------------------------------------------------------------------------------- */
IonicCqContainer::IonicCqContainer(ibv_context* context, const RdmaEndpointConfig& config)
    : config(config) {
  int status;
  struct ibv_cq_init_attr_ex cq_attr;
  struct ibv_cq_ex *cq_ex;

  cqeNum = config.maxCqeNum;
  size_t cqSize = 0;// not device memory

  memset(&cq_attr, 0, sizeof(struct ibv_cq_init_attr_ex));
  cq_attr.cqe           = cqeNum * 2; //from rocshmem, send&recv?
  cq_attr.cq_context    = nullptr;
  cq_attr.channel       = nullptr;
  cq_attr.comp_vector   = 0;
  cq_attr.flags         = 0;
  cq_attr.comp_mask     = IBV_CQ_INIT_ATTR_MASK_PD;
  cq_attr.parent_domain = pd_uxdma[0];

  cq_ex = ibv_create_cq_ex(context, &cq_attr);
  CHECK_NNULL(cq_ex, "ibv_create_cq_ex");
  assert(cq_ex);
  
  cq = ibv_cq_ex_to_cq(cq_ex);
  assert(cq);

  MORI_APP_TRACE("IONIC CQ created: cqn={}, cqeNum={}, cqSize={}",
                 cqn, cqeNum, cqSize);
}

IonicCqContainer::~IonicCqContainer() {
  int err;

  err = ibv_destroy_cq(cq);
  assert(err == 0);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         IonicQpContainer                                        */
/* ---------------------------------------------------------------------------------------------- */
IonicQpContainer::IonicQpContainer(ibv_context* context, const RdmaEndpointConfig& config,
                                   ibv_cq* cq, struct ibv_pd *pd_uxdma, 
                                   IonicDeviceContext* device_context)
    : context(context), config(config), device_context(device_context) {
  struct ibv_qp_init_attr_ex attr;
  int hip_dev_id{-1};

  wqeNum = config.maxMsgsNum;
  memset(&attr, 0, sizeof(struct ibv_qp_init_attr_ex));
  attr.cap.max_send_wr     = wqeNum;
  //attr.cap.max_recv_wr     = wqeNum;
  attr.cap.max_send_sge    = 1;
  attr.cap.max_inline_data = MAX_INLINE_SIZE;
  attr.sq_sig_all          = 0;
  attr.qp_type             = IBV_QPT_RC;
  attr.comp_mask           = IBV_QP_INIT_ATTR_PD;
  attr.cap.max_send_sge    = 1; 
  //attr.cap.max_recv_sge    = 1; 
  attr.pd                  = pd_uxdma;
  attr.send_cq             = cq;
  attr.recv_cq             = cq;
  qp = ibv_create_qp_ex(context, &attr);
  assert(qp);

  HIP_RUNTIME_CHECK(hipGetDevice(&hip_dev_id));
  ionic_dv_get_ctx(&dvctx, context);

  rocm_memory_lock_to_fine_grain(dvctx.db_page, 0x1000, &gpu_db_page, hip_dev_id);

  db_page_u64 = reinterpret_cast<uint64_t*>(dvctx.db_page);
  gpu_db_page_u64 = reinterpret_cast<uint64_t*>(gpu_db_page);

  gpu_db_ptr = &gpu_db_page_u64[dvctx.db_ptr - db_page_u64];

  gpu_db_page = gpu_db_page;
  gpu_db_cq = &gpu_db_ptr[dvctx.cq_qtype];
  gpu_db_sq = &gpu_db_ptr[dvctx.sq_qtype];
  gpu_db_rq = &gpu_db_ptr[dvctx.rq_qtype];

  uint8_t udma_idx = ionic_dv_qp_get_udma_idx(qp);
  ionic_dv_get_cq(&dvcq, cq, udma_idx);

  cq_dbreg = gpu_db_cq;
  cq_dbval = dvcq.q.db_val;
  cq_mask = dvcq.q.mask;
  ionic_cq_buf = reinterpret_cast<ionic_v1_cqe*>(dvcq.q.ptr);

  ionic_dv_qp dvqp;
  ionic_dv_get_qp(&dvqp, qp);

  sq_dbreg = gpu_db_sq;
  sq_dbval = dvqp.sq.db_val;
  sq_mask = dvqp.sq.mask;
  ionic_sq_buf = reinterpret_cast<ionic_v1_wqe *>(dvqp.sq.ptr);

  rq_dbreg = gpu_db_rq;
  rq_dbval = dvqp.rq.db_val;
  rq_mask = dvqp.rq.mask;
  ionic_rq_buf = reinterpret_cast<ionic_v1_wqe *>(dvqp.rq.ptr);

  strncpy(dev_name, qp->context->device->name, sizeof(dev_name));
  dev_name[sizeof(dev_name) - 1] = 0;

  qpn = qp->qp_num;

  MORI_APP_TRACE(
		"IONIC QP created: qpn={}, sqWqeNum={}, cqAddr=0x{:x}, sqAddr=0x{:x}",
		qpn, config.maxMsgsNum, 
		reinterpret_cast<uintptr_t>(ionic_cq_buf),
		reinterpret_cast<uintptr_t>(ionic_sq_buf));

  // Allocate and register atomic internal buffer (ibuf)
  atomicIbufSize = (RoundUpPowOfTwo(config.atomicIbufSlots) + 1) * ATOMIC_IBUF_SLOT_SIZE;
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(
        hipExtMallocWithFlags(&atomicIbufAddr, atomicIbufSize, hipDeviceMallocUncached));
    HIP_RUNTIME_CHECK(hipMemset(atomicIbufAddr, 0, atomicIbufSize));
  } else {
    err = posix_memalign(&atomicIbufAddr, config.alignment, atomicIbufSize);
    memset(atomicIbufAddr, 0, atomicIbufSize);
    assert(!err);
  }

  // Register atomic ibuf as independent memory region
  atomicIbufMr = ibv_reg_mr(pd_uxdma, atomicIbufAddr, atomicIbufSize,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  assert(atomicIbufMr);

  MORI_APP_TRACE(
      "BNXT Atomic ibuf allocated: addr=0x{:x}, slots={}, size={}, lkey=0x{:x}, rkey=0x{:x}",
      reinterpret_cast<uintptr_t>(atomicIbufAddr), RoundUpPowOfTwo(config.atomicIbufSlots),
      atomicIbufSize, atomicIbufMr->lkey, atomicIbufMr->rkey);  
}

IonicQpContainer::~IonicQpContainer(struct ibv_qp *qp) { 
  int err;

  // Clean up atomic internal buffer
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
  
  err = ibv_destroy_qp(qp);
  assert(err == 0);
}

void* IonicQpContainer::GetSqAddress() { return static_cast<char*>(ionic_sq_buf)}

void* IonicQpContainer::GetRqAddress() { return static_cast<char*>(ionic_rq_buf) }

void IonicQpContainer::ModifyRst2Init() {
  int err;
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));

  attr.qp_state        = IBV_QPS_INIT;
  attr.pkey_index      = 0;
  attr.port_num        = config.portId;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_READ  | IBV_ACCESS_REMOTE_ATOMIC;

  attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  err = ibv_modify_qp(qp, &attr, attr_mask);
  assert(err == 0);
}

void IonicQpContainer::ModifyInit2Rtr(const RdmaEndpointHandle& remote_handle,
                                     const ibv_port_attr& portAttr, uint32_t qpn) {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = portAttr.active_mtu;
  attr.min_rnr_timer = 12;
  attr.max_dest_rd_atomic = 15;
  attr.rq_psn = remote_handle.psn;
  attr.dest_qp_num = remote_handle.qpn;

  //ah_atter
  memcpy(&attr.ah_attr.grh.dgid, &dest_info[i].gid, 16);
  attr.ah_attr.grh.sgid_index = config.gidIdx;
  attr.ah_attr.port_num = config.portId;
  attr.ah_attr.is_global = 1;
  attr.ah_attr.grh.hop_limit = 1;
  attr.ah_attr.sl = 1;

  attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN | IBV_QP_DEST_QPN
              IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  err = ibv_modify_qp(qp, &attr, attr_mask);
  assert(err == 0);
}

void IonicQpContainer::ModifyRtr2Rts(const RdmaEndpointHandle& local_handle,
                                     const RdmaEndpointHandle& remote_handle) {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state      = IBV_QPS_RTS;
  attr.timeout       = 14;
  attr.retry_cnt     = 7;
  attr.rnr_retry     = 7;
  attr.max_rd_atomic = 15;
  attr.sq_psn = remote_handle.psn;

  attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC |
              IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY;

  err = ibv_modify_qp(qp, &attr, attr_mask);
  assert(!err);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        IonicDeviceContext                                      */
/* ---------------------------------------------------------------------------------------------- */
IonicDeviceContext::IonicDeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd)
    : RdmaDeviceContext(rdma_device, in_pd) {
  pd_uxdma = in_pd;
}

IonicDeviceContext::~IonicDeviceContext() {}

RdmaEndpoint IonicDeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  ibv_context* context = GetIbvContext();
  int ret;

  assert(!config.withCompChannel && !config.enableSrq && "not implemented");

  IonicCqContainer* cq = new IonicCqContainer(context, config);
  IonicQpContainer* qp = new IonicQpContainer(context, config, cq->cq, pd_uxdma, this);

  RdmaEndpoint endpoint;
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;
  endpoint.handle.qpn = qp->qpn;

  // Get gid
  union ibv_gid ibvGid;
  ret = ibv_query_gid(context, config.portId, config.gidIdx, &ibvGid);
  assert(!ret);
  memcpy(endpoint.handle.eth.gid, ibvGid.raw, sizeof(endpoint.handle.eth.gid));
  endpoint.vendorId = RdmaDeviceVendorId::Pensando;
  endpoint.wqHandle.sqAddr = qp->GetSqAddress();
  endpoint.wqHandle.rqAddr = qp->GetRqAddress();
  endpoint.wqHandle.dbrAddr = qp->gpu_db_sq;
  endpoint.wqHandle.sqWqeNum = qp->wqeNum;
  endpoint.wqHandle.rqWqeNum = qp->wqeNum;
  endpoint.wqHandle.color = true;
  endpoint.wqHandle.sq_dbval = qp->sq_dbval;
  endpoint.wqHandle.rq_dbval = qp->rq_dbval;

  endpoint.cqHandle.cqAddr = qp->ionic_cq_buf;
  endpoint.cqHandle.consIdx = 0;
  endpoint.cqHandle.cqeNum = cq->cqeNum;
  endpoint.cqHandle.cqeSize = GetIonicCqeSize();
  endpoint.cqHandle.dbrAddr = qp->gpu_db_cq;
  endpoint.cqHandle.cq_dbval = qp->cq_dbval;

  // Set atomic internal buffer information
  endpoint.atomicIbuf.addr = reinterpret_cast<uintptr_t>(qp->atomicIbufAddr);
  endpoint.atomicIbuf.lkey = qp->atomicIbufMr->lkey;
  endpoint.atomicIbuf.rkey = qp->atomicIbufMr->rkey;
  endpoint.atomicIbuf.nslots = RoundUpPowOfTwo(config.atomicIbufSlots);

  cqPool.insert({cq->cqn, std::move(std::unique_ptr<Mlx5CqContainer>(cq))});
  qpPool.insert({qp->qpn, std::move(std::unique_ptr<Mlx5QpContainer>(qp))});

  MORI_APP_TRACE(
      "Ionic endpoint created: qpn={}, cqn={}, portId={}, gidIdx={}, atomicIbuf addr=0x{:x}, "
      "nslots={}",
      qp->qpn, cq->cqn, config.portId, config.gidIdx, endpoint.atomicIbuf.addr,
      endpoint.atomicIbuf.nslots);

  return endpoint;
}

void IonicDeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                         const RdmaEndpointHandle& remote, uint32_t qpn) {
  uint32_t local_qpn = local.qpn;
  assert(qpPool.find(local_qpn) != qpPool.end());
  Mlx5QpContainer* qp = qpPool.at(local_qpn).get();

  MORI_APP_TRACE("Ionic connecting endpoint: local_qpn={}, remote_qpn={}, qpId={}", local_qpn,
                 remote.qpn, qpId);

  RdmaDevice* rdmaDevice = GetRdmaDevice();
  const ibv_device_attr_ex* deviceAttr = rdmaDevice->GetDeviceAttr();
  const ibv_port_attr& portAttr = *(rdmaDevice->GetPortAttrMap()->find(local.portId)->second);
  qp->ModifyRst2Init();
  //qpn unused for now, other vendor for udp multi-sport
  qp->ModifyInit2Rtr(remote, portAttr, qpn);
  qp->ModifyRtr2Rts(local);

  MORI_APP_TRACE("Ionic endpoint connected successfully: local_qpn={}, remote_qpn={}", local_qpn,
                 remote.qpn);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           IonicDevice                                          */
/* ---------------------------------------------------------------------------------------------- */
IonicDevice::IonicDevice(ibv_device* in_device) : RdmaDevice(in_device) {}
IonicDevice::~IonicDevice() {}

void IonicDevice::pd_release(struct ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type) {
  CHECK_HIP(hipFree(ptr));
}

void* IonicDevice::pd_alloc_device_uncached(struct ibv_pd* pd, void* pd_context, size_t size, 
                                            size_t alignment, uint64_t resource_type) {
  void* dev_ptr{nullptr};
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocUncached));
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

void IonicDevice::create_parent_domain(struct ibv_pd *pd_orig) {
  struct ibv_parent_domain_init_attr pattr;

  memset(&pattr, 0, sizeof(struct ibv_parent_domain_init_attr));
  pattr.pd         = pd_orig;
  pattr.td         = nullptr,
  pattr.comp_mask  = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  pattr.free       = IonicDevice::pd_release;
  pattr.pd_context = nullptr;
  pattr.alloc      = IonicDevice::pd_alloc_device_uncached;

  pd_parent = ibv_alloc_parent_domain(defaultContext, &pattr);
  assert(pd_parent);

  ionic_dv_pd_set_sqcmb(pd_parent, false, false, false);
  ionic_dv_pd_set_rqcmb(pd_parent, false, false, false);

  pd_uxdma = ibv_alloc_parent_domain(defaultContext, &pattr);
  assert(pd_uxdma);

  ionic_dv_pd_set_sqcmb(pd_uxdma, false, false, false);
  ionic_dv_pd_set_rqcmb(pd_uxdma, false, false, false);
  ionic_dv_pd_set_udma_mask(pd_uxdma, 1);
}

RdmaDeviceContext* IonicDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  assert(pd);
  	
  create_parent_domain(pd);
  return new IonicDeviceContext(this, pd_uxdma);
}

}  // namespace application
}  // namespace mori

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
#include "mori/application/transport/rdma/providers/bnxt/bnxt.hpp"

#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "mori/application/utils/check.hpp"
#include "mori/application/utils/math.hpp"
#include "mori/utils/mori_log.hpp"

// #define USE_BNXT_DEFAULT_DBR

#ifdef ENABLE_BNXT
namespace std {
static std::ostream& operator<<(std::ostream& s, const bnxt_re_dv_qp_mem_info& m) {
  std::stringstream ss;
  ss << "qp_handle: 0x" << std::hex << m.qp_handle << std::dec << "  sq_va: 0x" << std::hex
     << m.sq_va << std::dec << "  sq_len: " << m.sq_len << "  sq_slots: " << m.sq_slots
     << "  sq_wqe_sz: " << m.sq_wqe_sz << "  sq_psn_sz: " << m.sq_psn_sz
     << "  sq_npsn: " << m.sq_npsn << "  rq_va: 0x" << std::hex << m.rq_va << std::dec
     << "  rq_len: " << m.rq_len << "  rq_slots: " << m.rq_slots << "  rq_wqe_sz: " << m.rq_wqe_sz
     << "  comp_mask: 0x" << std::hex << m.comp_mask << std::dec;
  s << ss.str();
  return s;
}
}  // namespace std

template <>
struct fmt::formatter<bnxt_re_dv_qp_mem_info> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.end(); }

  template <typename FormatContext>
  auto format(const bnxt_re_dv_qp_mem_info& m, FormatContext& ctx) -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(),
                          "qp_handle: 0x{:x}  sq_va: 0x{:x}  sq_len: {}  sq_slots: {}  "
                          "sq_wqe_sz: {}  sq_psn_sz: {}  sq_npsn: {}  rq_va: 0x{:x}  "
                          "rq_len: {}  rq_slots: {}  rq_wqe_sz: {}  comp_mask: 0x{:x}",
                          m.qp_handle, m.sq_va, m.sq_len, m.sq_slots, m.sq_wqe_sz, m.sq_psn_sz,
                          m.sq_npsn, m.rq_va, m.rq_len, m.rq_slots, m.rq_wqe_sz, m.comp_mask);
  }
};
#endif  // ENABLE_BNXT

namespace mori {
namespace application {
#ifdef ENABLE_BNXT

/* ---------------------------------------------------------------------------------------------- */
/*                                          BnxtCqContainer */
/* ---------------------------------------------------------------------------------------------- */
BnxtCqContainer::BnxtCqContainer(ibv_context* context, const RdmaEndpointConfig& config)
    : config(config) {
  struct bnxt_re_dv_cq_init_attr cq_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;

  cqeNum = config.maxCqeNum;
  size_t cqSize = RoundUpPowOfTwo(GetBnxtCqeSize() * cqeNum);

  if (config.onGpu) {
    HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&cqUmemAddr, cqSize, hipDeviceMallocUncached));
    HIP_RUNTIME_CHECK(hipMemset(cqUmemAddr, 0, cqSize));
  } else {
    int status = posix_memalign(&cqUmemAddr, config.alignment, cqSize);
    memset(cqUmemAddr, 0, cqSize);
    assert(!status);
  }

  memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
  umem_attr.addr = cqUmemAddr;
  umem_attr.size = cqSize;
  umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

  cqUmem = bnxt_re_dv_umem_reg(context, &umem_attr);
  assert(cqUmem);

  memset(&cq_attr, 0, sizeof(struct bnxt_re_dv_cq_init_attr));
  cq_attr.umem_handle = cqUmem;
  cq_attr.ncqe = cqeNum;

  cq = bnxt_re_dv_create_cq(context, &cq_attr);
  assert(cq);

  struct bnxt_re_dv_obj dv_obj{};
  struct bnxt_re_dv_cq dvcq{};
  memset(&dv_obj, 0, sizeof(struct bnxt_re_dv_obj));
  dv_obj.cq.in = cq;
  dv_obj.cq.out = &dvcq;
  int status = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_CQ);
  assert(!status);
  cqn = dvcq.cqn;

  MORI_APP_TRACE("BNXT CQ created: cqn={}, cqeNum={}, cqSize={}, cqUmemAddr=0x{:x}", cqn, cqeNum,
                 cqSize, reinterpret_cast<uintptr_t>(cqUmemAddr));
}

BnxtCqContainer::~BnxtCqContainer() {
  if (cqUmemAddr) HIP_RUNTIME_CHECK(hipFree(cqUmemAddr));
  if (cqDbrUmemAddr) HIP_RUNTIME_CHECK(hipFree(cqDbrUmemAddr));
  if (cqUmem) bnxt_re_dv_umem_dereg(cqUmem);
  if (cqUar) {
    HIP_RUNTIME_CHECK(hipHostUnregister(cqUar));
  }
  if (cq) bnxt_re_dv_destroy_cq(cq);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         BnxtQpContainer                                        */
/* ---------------------------------------------------------------------------------------------- */
int bnxt_re_calc_dv_qp_mem_info(struct ibv_pd* ibvpd, struct ibv_qp_init_attr* attr,
                                struct bnxt_re_dv_qp_mem_info* dv_qp_mem) {
  struct ibv_qp_init_attr_ex attr_ex;
  constexpr int fixed_num_slot_per_wqe = BNXT_RE_NUM_SLOT_PER_WQE;

  uint32_t nwqe;
  uint32_t max_wqesz;
  uint32_t wqe_size;
  uint32_t slots;
  uint32_t psn_sz;
  uint32_t npsn;
  size_t sq_bytes = 0;
  size_t rq_bytes = 0;

  wqe_size = fixed_num_slot_per_wqe * BNXT_RE_SLOT_SIZE;
  nwqe = attr->cap.max_send_wr;
  slots = fixed_num_slot_per_wqe * nwqe;

  // msn mem calc
  npsn = RoundUpPowOfTwo(slots) / 2;
  psn_sz = 8;

  /*sq mem calc*/
  sq_bytes = slots * BNXT_RE_SLOT_SIZE;
  sq_bytes += npsn * psn_sz;
  dv_qp_mem->sq_len = AlignUp(sq_bytes, 4096);
  dv_qp_mem->sq_slots = slots;
  dv_qp_mem->sq_wqe_sz = wqe_size;
  dv_qp_mem->sq_npsn = npsn;
  dv_qp_mem->sq_psn_sz = 8;

  /*rq mem calc*/
  rq_bytes = slots * BNXT_RE_SLOT_SIZE;
  dv_qp_mem->rq_len = AlignUp(rq_bytes, 4096);
  dv_qp_mem->rq_slots = slots;
  dv_qp_mem->rq_wqe_sz = wqe_size;

  return 0;
}

BnxtQpContainer::BnxtQpContainer(ibv_context* context, const RdmaEndpointConfig& config, ibv_cq* cq,
                                 ibv_pd* pd, BnxtDeviceContext* device_context)
    : context(context), config(config), device_context(device_context) {
  struct ibv_qp_init_attr ib_qp_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;
  struct bnxt_re_dv_qp_init_attr dv_qp_attr;
  int err;

  uint32_t maxMsgsNum = RoundUpPowOfTwoAlignUpTo256(config.maxMsgsNum);
  memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_init_attr));
  ib_qp_attr.send_cq = cq;
  ib_qp_attr.recv_cq = cq;
  ib_qp_attr.cap.max_send_wr = maxMsgsNum;
  ib_qp_attr.cap.max_recv_wr = maxMsgsNum;
  ib_qp_attr.cap.max_send_sge = 1;
  ib_qp_attr.cap.max_recv_sge = 1;
  ib_qp_attr.cap.max_inline_data = 16;
  ib_qp_attr.qp_type = IBV_QPT_RC;
  ib_qp_attr.sq_sig_all = 0;

  memset(&qpMemInfo, 0, sizeof(struct bnxt_re_dv_qp_mem_info));
  err = bnxt_re_calc_dv_qp_mem_info(pd, &ib_qp_attr, &qpMemInfo);
  assert(!err);

  // sqUmemAddr
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(
        hipExtMallocWithFlags(&sqUmemAddr, qpMemInfo.sq_len, hipDeviceMallocUncached));
    HIP_RUNTIME_CHECK(hipMemset(sqUmemAddr, 0, qpMemInfo.sq_len));
  } else {
    err = posix_memalign(&sqUmemAddr, config.alignment, qpMemInfo.sq_len);
    memset(sqUmemAddr, 0, qpMemInfo.sq_len);
    assert(!err);
  }
  qpMemInfo.sq_va = reinterpret_cast<uint64_t>(sqUmemAddr);

  // msntblUmemAddr
  uint64_t msntbl_len = (qpMemInfo.sq_psn_sz * qpMemInfo.sq_npsn);
  uint64_t msntbl_offset = qpMemInfo.sq_len - msntbl_len;
  msntblUmemAddr = reinterpret_cast<void*>(reinterpret_cast<char*>(sqUmemAddr) + msntbl_offset);

  // rqUmemAddr
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(
        hipExtMallocWithFlags(&rqUmemAddr, qpMemInfo.rq_len, hipDeviceMallocUncached));
    HIP_RUNTIME_CHECK(hipMemset(rqUmemAddr, 0, qpMemInfo.rq_len));
  } else {
    err = posix_memalign(&rqUmemAddr, config.alignment, qpMemInfo.sq_len);
    memset(rqUmemAddr, 0, qpMemInfo.sq_len);
    assert(!err);
  }
  qpMemInfo.rq_va = reinterpret_cast<uint64_t>(rqUmemAddr);

#ifndef USE_BNXT_DEFAULT_DBR
  // Allocate dedicated db region for this QP
  dbrAttr = bnxt_re_dv_alloc_db_region(context);
  assert(dbrAttr != nullptr);
#endif

  memset(&dv_qp_attr, 0, sizeof(struct bnxt_re_dv_qp_init_attr));
  dv_qp_attr.send_cq = ib_qp_attr.send_cq;
  dv_qp_attr.recv_cq = ib_qp_attr.recv_cq;
  dv_qp_attr.max_send_wr = ib_qp_attr.cap.max_send_wr;
  dv_qp_attr.max_recv_wr = ib_qp_attr.cap.max_recv_wr;
  dv_qp_attr.max_send_sge = ib_qp_attr.cap.max_send_sge;
  dv_qp_attr.max_recv_sge = ib_qp_attr.cap.max_recv_sge;
  dv_qp_attr.max_inline_data = ib_qp_attr.cap.max_inline_data;
  dv_qp_attr.qp_type = ib_qp_attr.qp_type;

  // dv_qp_attr.qp_handle = qpMemInfo.qp_handle;
#ifndef USE_BNXT_DEFAULT_DBR
  dv_qp_attr.dbr_handle = dbrAttr;
#endif
  dv_qp_attr.sq_len = qpMemInfo.sq_len;
  dv_qp_attr.sq_slots = qpMemInfo.sq_slots;
  dv_qp_attr.sq_wqe_sz = qpMemInfo.sq_wqe_sz;
  dv_qp_attr.sq_psn_sz = qpMemInfo.sq_psn_sz;
  dv_qp_attr.sq_npsn = qpMemInfo.sq_npsn;
  dv_qp_attr.rq_len = qpMemInfo.rq_len;
  dv_qp_attr.rq_slots = qpMemInfo.rq_slots;
  dv_qp_attr.rq_wqe_sz = qpMemInfo.rq_wqe_sz;
  dv_qp_attr.comp_mask = qpMemInfo.comp_mask;

  memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
  umem_attr.addr = sqUmemAddr;
  umem_attr.size = qpMemInfo.sq_len;
  umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
  sqUmem = dv_qp_attr.sq_umem_handle = bnxt_re_dv_umem_reg(context, &umem_attr);
  assert(sqUmem);

  memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
  umem_attr.addr = rqUmemAddr;
  umem_attr.size = qpMemInfo.rq_len;
  umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
  rqUmem = dv_qp_attr.rq_umem_handle = bnxt_re_dv_umem_reg(context, &umem_attr);
  assert(rqUmem);

  qp = bnxt_re_dv_create_qp(pd, &dv_qp_attr);
  assert(qp);
  qpn = qp->qp_num;

  // Allocate and register atomic internal buffer (ibuf)
  atomicIbufSize = (RoundUpPowOfTwo(config.atomicIbufSlots) + 1) * ATOMIC_IBUF_SLOT_SIZE;
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(hipMalloc(&atomicIbufAddr, atomicIbufSize));
    HIP_RUNTIME_CHECK(hipMemset(atomicIbufAddr, 0, atomicIbufSize));
  } else {
    int status = posix_memalign(&atomicIbufAddr, config.alignment, atomicIbufSize);
    memset(atomicIbufAddr, 0, atomicIbufSize);
    assert(!status);
  }

  // Register atomic ibuf as independent memory region
  atomicIbufMr = ibv_reg_mr(pd, atomicIbufAddr, atomicIbufSize,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  assert(atomicIbufMr);

  MORI_APP_TRACE(
      "BNXT Atomic ibuf allocated: addr=0x{:x}, slots={}, size={}, lkey=0x{:x}, rkey=0x{:x}",
      reinterpret_cast<uintptr_t>(atomicIbufAddr), RoundUpPowOfTwo(config.atomicIbufSlots),
      atomicIbufSize, atomicIbufMr->lkey, atomicIbufMr->rkey);
  MORI_APP_TRACE(qpMemInfo);
}

BnxtQpContainer::~BnxtQpContainer() { DestroyQueuePair(); }

void BnxtQpContainer::DestroyQueuePair() {
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

  if (sqUmem) bnxt_re_dv_umem_dereg(sqUmem);
  if (rqUmem) bnxt_re_dv_umem_dereg(rqUmem);
  if (sqUmemAddr) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipFree(sqUmemAddr));
    } else {
      free(sqUmemAddr);
    }
  }
  if (rqUmemAddr) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipFree(rqUmemAddr));
    } else {
      free(rqUmemAddr);
    }
  }
  if (qpDbrUmemAddr) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipFree(qpDbrUmemAddr));
    } else {
      free(qpDbrUmemAddr);
    }
  }
  if (qpUar) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipHostUnregister(qpUar));
    }
  }
#ifndef USE_BNXT_DEFAULT_DBR
  if (dbrAttr) {
    int ret = bnxt_re_dv_free_db_region(context, dbrAttr);
    assert(!ret);
    dbrAttr = nullptr;
  }
#endif
  if (qp) bnxt_re_dv_destroy_qp(qp);
}

void* BnxtQpContainer::GetSqAddress() { return static_cast<char*>(sqUmemAddr); }

void* BnxtQpContainer::GetMsntblAddress() { return static_cast<char*>(msntblUmemAddr); }

void* BnxtQpContainer::GetRqAddress() { return static_cast<char*>(rqUmemAddr); }

void BnxtQpContainer::ModifyRst2Init() {
  struct ibv_qp_attr attr;
  int attr_mask;
  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = config.portId;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC;

  attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  int status = bnxt_re_dv_modify_qp(qp, &attr, attr_mask, 0, 0);
  assert(!status);
}

void BnxtQpContainer::ModifyInit2Rtr(const RdmaEndpointHandle& remote_handle,
                                     const ibv_port_attr& portAttr,
                                     const ibv_device_attr_ex& deviceAttr) {
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = portAttr.active_mtu;
  attr.rq_psn = remote_handle.psn;
  attr.dest_qp_num = remote_handle.qpn;

  memcpy(&attr.ah_attr.grh.dgid, remote_handle.eth.gid, 16);
  attr.ah_attr.grh.sgid_index = config.gidIdx;
  attr.ah_attr.grh.hop_limit = 1;
  attr.ah_attr.sl = 1;
  attr.ah_attr.is_global = 1;
  attr.ah_attr.port_num = config.portId;

  // TODO: max_dest_rd_atomic whether affect nums of amo/rd
  attr.max_dest_rd_atomic = deviceAttr.orig_attr.max_qp_rd_atom;
  attr.min_rnr_timer = 12;

  attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN | IBV_QP_DEST_QPN | IBV_QP_AV |
              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  int status = bnxt_re_dv_modify_qp(qp, &attr, attr_mask, 0, 0);
  assert(!status);
}

void BnxtQpContainer::ModifyRtr2Rts(const RdmaEndpointHandle& local_handle,
                                    const RdmaEndpointHandle& remote_handle, uint32_t qpId) {
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = remote_handle.psn;
  attr.max_rd_atomic = 7;
  attr.timeout = 20;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;

  attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_TIMEOUT |
              IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY;

  int status = bnxt_re_dv_modify_qp(qp, &attr, attr_mask, 0, 0);
  assert(!status);
  // Use qpId to select UDP sport value from the shared configuration (round-robin)
  uint16_t selected_udp_sport = GetDeviceContext()->GetUdpSport(qpId) + qpId;
  MORI_APP_TRACE("QP {} using UDP sport {} (qpId={}, index={})", qpn, selected_udp_sport, qpId,
                 qpId % RDMA_UDP_SPORT_ARRAY_SIZE);
  status = bnxt_re_dv_modify_qp_udp_sport(qp, selected_udp_sport);
  if (status) {
    MORI_APP_ERROR("Failed to set UDP sport {} for QP {}: error code {}", selected_udp_sport, qpn,
                   status);
  }
  assert(!status);
  MORI_APP_TRACE("bnxt_re_dv_modify_qp_udp_sport is done, return {}", status);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        BnxtDeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
BnxtDeviceContext::BnxtDeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd)
    : RdmaDeviceContext(rdma_device, in_pd) {
  struct bnxt_re_dv_obj dv_obj{};
  struct bnxt_re_dv_pd dvpd{};

  dv_obj.pd.in = in_pd;
  dv_obj.pd.out = &dvpd;
  int status = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_PD);
  assert(!status);
  pdn = dvpd.pdn;
}

BnxtDeviceContext::~BnxtDeviceContext() {}

RdmaEndpoint BnxtDeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  ibv_context* context = GetIbvContext();

  BnxtCqContainer* cq = new BnxtCqContainer(context, config);

  BnxtQpContainer* qp = new BnxtQpContainer(context, config, cq->cq, pd, this);

  const ibv_device_attr_ex* deviceAttr = GetRdmaDevice()->GetDeviceAttr();
  int ret;

  RdmaEndpoint endpoint;
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;
  endpoint.handle.maxSge = deviceAttr->tm_caps.max_sge;

  endpoint.handle.qpn = qp->qpn;

  // Get gid
  union ibv_gid ibvGid;
  ret = ibv_query_gid(context, config.portId, config.gidIdx, &ibvGid);
  assert(!ret);
  memcpy(endpoint.handle.eth.gid, ibvGid.raw, sizeof(endpoint.handle.eth.gid));

#ifdef USE_BNXT_DEFAULT_DBR
  // Get default shared db region
  struct bnxt_re_dv_db_region_attr dbrAttr{};
  ret = bnxt_re_dv_get_default_db_region(context, &dbrAttr);
  assert(!ret);
  void* uar_host = (void*)dbrAttr.dbr;
#else
  // Use the db region allocated during QP creation
  void* uar_host = (void*)qp->dbrAttr->dbr;
#endif
  void* uar_dev = uar_host;
  if (config.onGpu) {
    constexpr uint32_t flag = hipHostRegisterPortable | hipHostRegisterMapped;

    HIP_RUNTIME_CHECK(hipHostRegister(uar_host, getpagesize(), flag));
    HIP_RUNTIME_CHECK(hipHostGetDevicePointer(&uar_dev, uar_host, 0));
  }
  qp->qpUar = cq->cqUar = uar_host;
  qp->qpUarPtr = cq->cqUarPtr = uar_dev;

  endpoint.vendorId = RdmaDeviceVendorId::Broadcom;

  RdmaDevice* rdmaDevice = GetRdmaDevice();
  const ibv_port_attr& portAttr = *(rdmaDevice->GetPortAttrMap()->find(config.portId)->second);
  endpoint.wqHandle.mtuSize = 256U << (portAttr.active_mtu - 1);
  endpoint.wqHandle.sqAddr = qp->GetSqAddress();
  endpoint.wqHandle.msntblAddr = qp->GetMsntblAddress();
  endpoint.wqHandle.rqAddr = qp->GetRqAddress();
  endpoint.wqHandle.dbrAddr = qp->qpUarPtr;
  assert(qp->qpMemInfo.sq_slots % BNXT_RE_NUM_SLOT_PER_WQE == 0);
  assert(qp->qpMemInfo.rq_slots % BNXT_RE_NUM_SLOT_PER_WQE == 0);
  endpoint.wqHandle.sqWqeNum = qp->qpMemInfo.sq_slots / BNXT_RE_NUM_SLOT_PER_WQE;
  endpoint.wqHandle.rqWqeNum = qp->qpMemInfo.rq_slots / BNXT_RE_NUM_SLOT_PER_WQE;
  endpoint.wqHandle.msntblNum = qp->qpMemInfo.sq_npsn;

  endpoint.cqHandle.cqAddr = cq->cqUmemAddr;
  endpoint.cqHandle.dbrAddr = cq->cqUarPtr;
  endpoint.cqHandle.dbrRecAddr = cq->cqUarPtr;
  endpoint.cqHandle.cqeNum = cq->cqeNum;
  endpoint.cqHandle.cqeSize = GetBnxtCqeSize();

  // Set atomic internal buffer information
  endpoint.atomicIbuf.addr = reinterpret_cast<uintptr_t>(qp->atomicIbufAddr);
  endpoint.atomicIbuf.lkey = qp->atomicIbufMr->lkey;
  endpoint.atomicIbuf.rkey = qp->atomicIbufMr->rkey;
  endpoint.atomicIbuf.nslots = RoundUpPowOfTwo(config.atomicIbufSlots);

  cqPool.insert({cq->cqn, cq});
  qpPool.insert({qp->qpn, qp});

  MORI_APP_TRACE(
      "BNXT endpoint created: qpn={}, cqn={}, portId={}, gidIdx={}, atomicIbuf addr=0x{:x}, "
      "nslots={}, uar_host=0x{:x},uar_dev=0x{:x}",
      qp->qpn, cq->cqn, config.portId, config.gidIdx, endpoint.atomicIbuf.addr,
      endpoint.atomicIbuf.nslots, reinterpret_cast<uintptr_t>(uar_host),
      reinterpret_cast<uintptr_t>(uar_dev));

  return endpoint;
}

void BnxtDeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                        const RdmaEndpointHandle& remote, uint32_t qpId) {
  uint32_t local_qpn = local.qpn;
  assert(qpPool.find(local_qpn) != qpPool.end());
  BnxtQpContainer* qp = qpPool.at(local_qpn);
  RdmaDevice* rdmaDevice = GetRdmaDevice();
  const ibv_device_attr_ex& deviceAttr = *(rdmaDevice->GetDeviceAttr());
  const ibv_port_attr& portAttr = *(rdmaDevice->GetPortAttrMap()->find(local.portId)->second);
  qp->ModifyRst2Init();
  qp->ModifyInit2Rtr(remote, portAttr, deviceAttr);
  qp->ModifyRtr2Rts(local, remote, qpId);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           BnxtDevice                                           */
/* ---------------------------------------------------------------------------------------------- */
BnxtDevice::BnxtDevice(ibv_device* in_device) : RdmaDevice(in_device) {}
BnxtDevice::~BnxtDevice() {}

RdmaDeviceContext* BnxtDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  return new BnxtDeviceContext(this, pd);
}
#endif

}  // namespace application
}  // namespace mori

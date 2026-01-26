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
#include "mori/application/transport/rdma/providers/ibverbs/ibverbs.hpp"

#include "mori/application/utils/check.hpp"
#include "mori/utils/mori_log.hpp"
namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                      IBVerbsDeviceContext                                      */
/* ---------------------------------------------------------------------------------------------- */
IBVerbsDeviceContext::IBVerbsDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd)
    : RdmaDeviceContext(rdma_device, inPd) {}

IBVerbsDeviceContext::~IBVerbsDeviceContext() {
  for (auto& it : qpPool) ibv_destroy_qp(it.second);
  for (auto& it : cqPool) ibv_destroy_cq(it.second);
  for (auto* compCh : compChPool) {
    if (compCh) ibv_destroy_comp_channel(compCh);
  }
}

RdmaEndpoint IBVerbsDeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  ibv_context* context = GetIbvContext();
  const ibv_device_attr_ex* deviceAttr = GetRdmaDevice()->GetDeviceAttr();

  RdmaEndpoint endpoint;
  endpoint.vendorId = ToRdmaDeviceVendorId(deviceAttr->orig_attr.vendor_id);
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;
  endpoint.handle.maxSge = config.maxMsgSge;

  const ibv_port_attr* portAttr = GetRdmaDevice()->GetPortAttr(config.portId);
  assert(portAttr);
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    endpoint.handle.ib.lid = portAttr->lid;
  } else if (portAttr->link_layer == IBV_LINK_LAYER_ETHERNET) {
    GidSelectionResult gidSelection =
        AutoSelectGidIndex(context, config.portId, portAttr, config.gidIdx);
    assert(gidSelection.gidIdx >= 0 && gidSelection.valid);

    memcpy(endpoint.handle.eth.gid, gidSelection.gid.raw, 16);
    endpoint.handle.eth.gidIdx = gidSelection.gidIdx;
  } else {
    assert(false && "unsupported link layer");
  }

  // TODO: we need to add more options in config, include min cqe num for ib_create_cq
  endpoint.ibvHandle.compCh = config.withCompChannel ? ibv_create_comp_channel(context) : nullptr;
  endpoint.ibvHandle.cq =
      ibv_create_cq(context, config.maxCqeNum, NULL, endpoint.ibvHandle.compCh, 0);
  assert(endpoint.ibvHandle.cq);

  // TODO: should also manage the lifecycle of completion channel && srq
  if (config.withCompChannel)
    assert(endpoint.ibvHandle.compCh &&
           (endpoint.ibvHandle.cq->channel == endpoint.ibvHandle.compCh));

  assert(config.maxMsgSge <= GetRdmaDevice()->GetDeviceAttr()->orig_attr.max_sge);
  endpoint.ibvHandle.srq = config.enableSrq ? CreateRdmaSrqIfNx(config) : nullptr;

  ibv_qp_init_attr qpAttr = {.send_cq = endpoint.ibvHandle.cq,
                             .recv_cq = endpoint.ibvHandle.cq,
                             .srq = endpoint.ibvHandle.srq,
                             .cap =
                                 {
                                     .max_send_wr = config.maxMsgsNum,
                                     .max_recv_wr = config.maxMsgsNum,
                                     .max_send_sge = config.maxMsgSge,
                                     .max_recv_sge = config.maxMsgSge,
                                 },
                             .qp_type = IBV_QPT_RC};
  endpoint.ibvHandle.qp = ibv_create_qp(pd, &qpAttr);
  assert(endpoint.ibvHandle.qp);
  endpoint.handle.qpn = endpoint.ibvHandle.qp->qp_num;

  if (config.enableSrq)
    assert(endpoint.ibvHandle.srq && (endpoint.ibvHandle.qp->srq == endpoint.ibvHandle.srq));

  cqPool.insert({endpoint.ibvHandle.cq, endpoint.ibvHandle.cq});
  qpPool.insert({endpoint.ibvHandle.qp->qp_num, endpoint.ibvHandle.qp});
  if (endpoint.ibvHandle.compCh) {
    compChPool.push_back(endpoint.ibvHandle.compCh);
  }
  return endpoint;
}

void IBVerbsDeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                           const RdmaEndpointHandle& remote, uint32_t qpId) {
  ibv_qp_attr attr;
  int flags;

  const ibv_device_attr_ex* devAttr = GetRdmaDevice()->GetDeviceAttr();
  ibv_qp* qp = qpPool.find(local.qpn)->second;

  // INIT
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = local.portId;
  attr.pkey_index = 0;
  attr.qp_access_flags = MR_DEFAULT_ACCESS_FLAG;
  flags = IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS;
  SYSCALL_RETURN_ZERO(ibv_modify_qp(qp, &attr, flags));

  const ibv_port_attr* portAttr = GetRdmaDevice()->GetPortAttr(local.portId);
  assert(portAttr);
  // RTR
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = portAttr->active_mtu;
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = devAttr->orig_attr.max_qp_rd_atom;
  attr.min_rnr_timer = 12;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = local.portId;
  attr.ah_attr.sl = ReadRdmaServiceLevelEnv().value_or(0);
  std::optional<uint8_t> tc = ReadRdmaTrafficClassEnv();
  if (tc.has_value()) {
    attr.ah_attr.grh.traffic_class = tc.value();
  }
  MORI_APP_INFO("ibverbs attr.ah_attr.sl:{} attr.ah_attr.grh.traffic_class:{}", attr.ah_attr.sl,
                attr.ah_attr.grh.traffic_class);

  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    attr.ah_attr.dlid = remote.ib.lid;
  } else if (portAttr->link_layer == IBV_LINK_LAYER_ETHERNET) {
    attr.ah_attr.is_global = 1;
    union ibv_gid dgid;
    memcpy(dgid.raw, remote.eth.gid, 16);
    attr.ah_attr.grh.dgid = dgid;
    attr.ah_attr.grh.sgid_index = local.eth.gidIdx;
    attr.ah_attr.grh.hop_limit = 16;
  }
  flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER | IBV_QP_AV;
  SYSCALL_RETURN_ZERO(ibv_modify_qp(qp, &attr, flags));

  // RTS
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = devAttr->orig_attr.max_qp_init_rd_atom;
  flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_MAX_QP_RD_ATOMIC;
  SYSCALL_RETURN_ZERO(ibv_modify_qp(qp, &attr, flags));
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          IBVerbsDevice                                         */
/* ---------------------------------------------------------------------------------------------- */
IBVerbsDevice::IBVerbsDevice(ibv_device* device) : RdmaDevice(device) {}
IBVerbsDevice::~IBVerbsDevice() {}

RdmaDeviceContext* IBVerbsDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  return new IBVerbsDeviceContext(this, pd);
}

}  // namespace application
}  // namespace mori

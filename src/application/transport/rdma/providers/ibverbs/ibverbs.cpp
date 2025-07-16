#include "mori/application/transport/rdma/providers/ibverbs/ibverbs.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                      IBVerbsDeviceContext                                      */
/* ---------------------------------------------------------------------------------------------- */
IBVerbsDeviceContext::IBVerbsDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd)
    : RdmaDeviceContext(rdma_device, inPd) {}

IBVerbsDeviceContext::~IBVerbsDeviceContext() {}

RdmaEndpoint IBVerbsDeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  ibv_context* context = GetIbvContext();
  const ibv_device_attr_ex* deviceAttr = GetRdmaDevice()->GetDeviceAttr();

  RdmaEndpoint endpoint;
  endpoint.vendorId = ToRdmaDeviceVendorId(deviceAttr->orig_attr.vendor_id);
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;

  const ibv_port_attr* portAttr = GetRdmaDevice()->GetPortAttr(config.portId);
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    endpoint.handle.ib.lid = portAttr->lid;
  } else if (portAttr->link_layer == IBV_LINK_LAYER_ETHERNET) {
    // TODO: implement ethernet
    assert(false && "not implemented");
  } else {
    assert(false && "unsupported link layer");
  }

  // TODO: we need to add more options in config, include min cqe num for ib_create_cq and max sge
  // for qp
  endpoint.ibvHandle.cq = ibv_create_cq(context, config.maxCqeNum, NULL, NULL, 0);
  ibv_qp_init_attr qpAttr = {.send_cq = endpoint.ibvHandle.cq,
                             .recv_cq = endpoint.ibvHandle.cq,
                             .cap = {.max_send_wr = config.maxMsgsNum,
                                     .max_recv_wr = config.maxMsgsNum,
                                     .max_send_sge = 1,
                                     .max_recv_sge = 1},
                             .qp_type = IBV_QPT_RC};
  endpoint.ibvHandle.qp = ibv_create_qp(pd, &qpAttr);
  endpoint.handle.qpn = endpoint.ibvHandle.qp->qp_num;

  cqPool.insert({endpoint.ibvHandle.cq, std::move(std::unique_ptr<ibv_cq>(endpoint.ibvHandle.cq))});
  qpPool.insert(
      {endpoint.ibvHandle.qp->qp_num, std::move(std::unique_ptr<ibv_qp>(endpoint.ibvHandle.qp))});

  return endpoint;
}

void IBVerbsDeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                           const RdmaEndpointHandle& remote) {
  ibv_qp_attr attr;
  int flags;

  ibv_qp* qp = qpPool.find(local.qpn)->second.get();

  // INIT
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = local.portId;
  attr.pkey_index = 0;
  attr.qp_access_flags = MR_DEFAULT_ACCESS_FLAG;
  flags = IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS;
  ibv_modify_qp(qp, &attr, flags);

  // RTR
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = remote.ib.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = local.portId;
  // TODO: complete ethernet logics
  //   if (dgid) {
  //     attr.ah_attr.is_global = 1;
  //     attr.ah_attr.grh.dgid = *dgid;
  //     attr.ah_attr.grh.sgid_index = 0;
  //     attr.ah_attr.grh.hop_limit = 1;
  //   }
  flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER | IBV_QP_AV;
  ibv_modify_qp(qp, &attr, flags);

  // RTS
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_MAX_QP_RD_ATOMIC;
  ibv_modify_qp(qp, &attr, flags);
}

ibv_qp* IBVerbsDeviceContext::GetQp() const {}

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

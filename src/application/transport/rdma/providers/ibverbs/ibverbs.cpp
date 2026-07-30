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

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

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

namespace {

// Releases the endpoint resources created so far if CreateRdmaEndpoint unwinds.
//
// c858f398 registered cq/compCh with the destructor pools at creation so they
// could not be lost, which fixed the leak-past-teardown but NOT the thing the
// error message asks the caller to do. `ibv_create_qp failed ... reduce
// numQpPerPe` invites an in-place retry, and a tracked-but-not-released CQ
// still occupies the HCA's budget until ~IBVerbsDeviceContext -- so each retry
// consumed one more CQ off the very budget the message said to reduce, and the
// advice made the exhaustion monotonically worse. Tracking is the right
// property for teardown; for a retry the resources have to actually go back.
//
// Commit() on the success path; anything else destroys and de-registers. Order
// matters: the QP references the CQ and the CQ references the comp channel, so
// they are torn down in reverse creation order, exactly as
// ~IBVerbsDeviceContext does.
class EndpointCreationGuard {
 public:
  EndpointCreationGuard(std::unordered_map<void*, ibv_cq*>& cqPool,
                        std::vector<ibv_comp_channel*>& compChPool)
      : cqPool_(cqPool), compChPool_(compChPool) {}

  ~EndpointCreationGuard() {
    if (committed_) return;
    if (srq_) ibv_destroy_srq(srq_);
    if (cq_) {
      cqPool_.erase(cq_);
      ibv_destroy_cq(cq_);
    }
    if (compCh_) {
      compChPool_.erase(std::remove(compChPool_.begin(), compChPool_.end(), compCh_),
                        compChPool_.end());
      ibv_destroy_comp_channel(compCh_);
    }
  }

  EndpointCreationGuard(const EndpointCreationGuard&) = delete;
  EndpointCreationGuard& operator=(const EndpointCreationGuard&) = delete;

  void TrackCompCh(ibv_comp_channel* compCh) { compCh_ = compCh; }
  void TrackCq(ibv_cq* cq) { cq_ = cq; }
  void TrackSrq(ibv_srq* srq) { srq_ = srq; }
  void Commit() { committed_ = true; }

 private:
  std::unordered_map<void*, ibv_cq*>& cqPool_;
  std::vector<ibv_comp_channel*>& compChPool_;
  ibv_comp_channel* compCh_{nullptr};
  ibv_cq* cq_{nullptr};
  ibv_srq* srq_{nullptr};
  bool committed_{false};
};

}  // namespace

RdmaEndpoint IBVerbsDeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  EndpointCreationGuard guard(cqPool, compChPool);
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
  //
  // Each resource is registered with the pools THE MOMENT it is created, before
  // anything that can throw. ~IBVerbsDeviceContext destroys exactly what the
  // pools hold, so registering at the end (as this did) means every failure
  // between creation and the end of this function permanently leaks the
  // resources already taken. That was harmless while the failure was an
  // assert() -- the process aborted -- but the ibv_create_qp failure below now
  // unwinds, and its own message tells the caller to RETRY with fewer QPs. A
  // retry that burns a CQ per attempt makes the exhaustion it reports worse
  // each time round.
  endpoint.ibvHandle.compCh = config.withCompChannel ? ibv_create_comp_channel(context) : nullptr;
  if (endpoint.ibvHandle.compCh) {
    compChPool.push_back(endpoint.ibvHandle.compCh);
    guard.TrackCompCh(endpoint.ibvHandle.compCh);
  }
  endpoint.ibvHandle.cq =
      ibv_create_cq(context, config.maxCqeNum, NULL, endpoint.ibvHandle.compCh, 0);
  if (endpoint.ibvHandle.cq == nullptr) {
    // Same defect the ibv_create_qp assert below had, one line up: CQ
    // exhaustion is as likely as QP exhaustion on a busy HCA (mori asks for one
    // CQ per endpoint), and this assert is compiled out under NDEBUG, where
    // ibv_create_qp is then handed a null send_cq.
    const int err = errno;
    throw std::runtime_error(
        "mori: ibv_create_cq failed: " + std::string(std::strerror(err)) +
        " (errno=" + std::to_string(err) + "). Requested cqe=" + std::to_string(config.maxCqeNum) +
        ", with_comp_channel=" + (config.withCompChannel ? "true" : "false") +
        ". ENOMEM usually means the device's CQ resources are exhausted (reduce "
        "numQpPerPe or the EP world size); EINVAL usually means the requested "
        "cqe count exceeds the device attributes.");
  }
  cqPool.insert({endpoint.ibvHandle.cq, endpoint.ibvHandle.cq});
  guard.TrackCq(endpoint.ibvHandle.cq);

  // TODO: should also manage the lifecycle of completion channel && srq
  if (config.withCompChannel)
    assert(endpoint.ibvHandle.compCh &&
           (endpoint.ibvHandle.cq->channel == endpoint.ibvHandle.compCh));

  assert(config.maxMsgSge <= GetRdmaDevice()->GetDeviceAttr()->orig_attr.max_sge);
  endpoint.ibvHandle.srq = config.enableSrq ? CreateRdmaSrqIfNx(config) : nullptr;
  guard.TrackSrq(endpoint.ibvHandle.srq);

  uint32_t maxRecvWr = config.maxRecvWr != 0 ? config.maxRecvWr : config.maxMsgsNum;
  ibv_qp_init_attr qpAttr = {.send_cq = endpoint.ibvHandle.cq,
                             .recv_cq = endpoint.ibvHandle.cq,
                             .srq = endpoint.ibvHandle.srq,
                             .cap =
                                 {
                                     .max_send_wr = config.maxMsgsNum,
                                     .max_recv_wr = maxRecvWr,
                                     .max_send_sge = config.maxMsgSge,
                                     .max_recv_sge = config.maxMsgSge,
                                 },
                             .qp_type = IBV_QPT_RC};
  endpoint.ibvHandle.qp = ibv_create_qp(pd, &qpAttr);
  if (endpoint.ibvHandle.qp == nullptr) {
    // Not an assert(). This is the most common RDMA bring-up failure and an
    // assert discards the one thing that distinguishes its causes -- errno --
    // AND is compiled out entirely under NDEBUG, where the very next line then
    // dereferences nullptr and the process dies with no diagnostic at all.
    // ENOMEM means the HCA's QP/CQ budget is exhausted (too many ranks x
    // numQpPerPe for this device); EINVAL means the requested caps exceed the
    // device attributes, i.e. a config error. Those need opposite fixes and a
    // bare abort cannot tell a caller which one it hit.
    const int err = errno;
    throw std::runtime_error(
        "mori: ibv_create_qp failed: " + std::string(std::strerror(err)) +
        " (errno=" + std::to_string(err) +
        "). Requested max_send_wr=" + std::to_string(config.maxMsgsNum) +
        ", max_recv_wr=" + std::to_string(maxRecvWr) +
        ", max_send_sge=" + std::to_string(config.maxMsgSge) +
        ", cq_size=" + std::to_string(config.maxCqeNum) +
        ". ENOMEM usually means the device's QP/CQ resources are exhausted "
        "(reduce numQpPerPe or the EP world size); EINVAL usually means these "
        "caps exceed the device attributes.");
  }
  endpoint.handle.qpn = endpoint.ibvHandle.qp->qp_num;

  if (config.enableSrq)
    assert(endpoint.ibvHandle.srq && (endpoint.ibvHandle.qp->srq == endpoint.ibvHandle.srq));

  // cq and compCh were registered at creation, above. Only the QP is left, and
  // nothing between its creation and here can throw.
  qpPool.insert({endpoint.ibvHandle.qp->qp_num, endpoint.ibvHandle.qp});
  // Past every throw: the endpoint is the caller's now, and its resources are
  // owned by the pools until ~IBVerbsDeviceContext. Nothing for the guard to
  // release.
  guard.Commit();
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
  {
    ibv_mtu path_mtu = portAttr->active_mtu;
    const char* envMtu = std::getenv("MORI_IB_PATH_MTU");
    if (envMtu != nullptr) {
      int mtuBytes = std::atoi(envMtu);
      if (mtuBytes == 256)
        path_mtu = IBV_MTU_256;
      else if (mtuBytes == 512)
        path_mtu = IBV_MTU_512;
      else if (mtuBytes == 1024)
        path_mtu = IBV_MTU_1024;
      else if (mtuBytes == 2048)
        path_mtu = IBV_MTU_2048;
      else if (mtuBytes == 4096)
        path_mtu = IBV_MTU_4096;
      else
        MORI_APP_WARN("Ignore invalid MORI_IB_PATH_MTU={} (allowed: 256/512/1024/2048/4096)",
                      envMtu);
      MORI_APP_INFO("MORI_IB_PATH_MTU override: {} bytes (ibv_mtu={})", mtuBytes, (int)path_mtu);
    }
    attr.path_mtu = path_mtu;
  }
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = devAttr->orig_attr.max_qp_rd_atom;
  attr.min_rnr_timer = 12;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = local.portId;
  std::optional<uint8_t> sl = ReadIoServiceLevelEnv();
  if (!sl.has_value()) {
    sl = ReadRdmaServiceLevelEnv();
  }
  attr.ah_attr.sl = sl.value_or(0);

  bool disableIoTc = ReadIoTrafficClassDisableEnv();
  if (!disableIoTc) {
    std::optional<uint8_t> tc = ReadIoTrafficClassEnv();
    if (!tc.has_value()) {
      tc = ReadRdmaTrafficClassEnv();
    }
    if (tc.has_value()) {
      attr.ah_attr.grh.traffic_class = tc.value();
    }
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

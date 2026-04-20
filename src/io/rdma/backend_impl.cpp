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
#include "src/io/rdma/backend_impl.hpp"

#include <sys/epoll.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <string>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"
#include "src/io/rdma/protocol.hpp"
namespace mori {
namespace io {

static void ValidateRdmaNotificationConfig(const RdmaBackendConfig& config) {
  if (config.enableNotification && config.notifPerQp == 0) {
    MORI_IO_ERROR(
        "Invalid RDMA config: notifPerQp must be >= 1 when notification is enabled; got {}",
        config.notifPerQp);
    throw std::runtime_error(
        "Invalid RDMA config: notifPerQp must be >= 1 when notification is "
        "enabled");
  }
}

enum class CqeFailureOrigin : uint8_t {
  BatchTransfer = 0,
  NotificationSend,
  NotificationRecv,
  Unknown,
};

struct CqeFailureAdvice {
  const char* statusText{nullptr};
  std::string hint;

  bool HasHint() const { return !hint.empty(); }

  std::string ComposeStatusMessage() const {
    std::string message = statusText != nullptr ? statusText : "unknown";
    if (hint.empty()) return message;
    message += " Hint: ";
    message += hint;
    return message;
  }
};

static void LogAsyncTransferFailureIfNeeded(internal::IoCallDiagnostics* diagnostics, uint32_t code,
                                            const std::string& message) {
  if (diagnostics == nullptr || diagnostics->Label() == nullptr) return;

  internal::IoFailureKind failureKind = diagnostics->CurrentFailureKind();
  if (failureKind != internal::IoFailureKind::FlushCascade ||
      !diagnostics->TryMarkLogged(failureKind)) {
    return;
  }

  MORI_IO_DEBUG("{} error {} message {}", diagnostics->Label(), code, message);
}

static CqeFailureOrigin ClassifyCqeFailureOrigin(uint64_t wrId, uint32_t notifPerQp) {
  if (IsNotifSendWrId(wrId)) return CqeFailureOrigin::NotificationSend;
  if (wrId < notifPerQp) return CqeFailureOrigin::NotificationRecv;
  return CqeFailureOrigin::BatchTransfer;
}

static CqeFailureAdvice DescribeCqeFailure(ibv_wc_status status, CqeFailureOrigin origin,
                                           const RdmaBackendConfig& config) {
  CqeFailureAdvice advice{ibv_wc_status_str(status), {}};
  switch (status) {
    case IBV_WC_RETRY_EXC_ERR:
      advice.hint =
          "transport retry limit exceeded; check peer liveness/connectivity, verify GID "
          "selection (unset or correct MORI_IB_GID_INDEX), and if running RoCE verify QoS "
          "settings such as MORI_IO_SL/MORI_IO_TC or MORI_RDMA_SL/MORI_RDMA_TC.";
      break;
    case IBV_WC_RNR_RETRY_EXC_ERR:
      if (origin == CqeFailureOrigin::NotificationSend) {
        advice.hint =
            "receiver not ready for SEND completions; if notifications are enabled, ensure the "
            "peer pre-posts enough RECV WRs. Try increasing notifPerQp / MORI_IO_QP_MAX_RECV_WR "
            "(current notifPerQp=" +
            std::to_string(config.notifPerQp) +
            "), or set MORI_IO_ENABLE_NOTIFICATION=0 if inbound notification is not required.";
      } else {
        advice.hint =
            "receiver not ready; check the peer receive path. If this is related to MORI "
            "notifications, increase notifPerQp / MORI_IO_QP_MAX_RECV_WR or disable "
            "MORI_IO_ENABLE_NOTIFICATION when inbound notification is not required.";
      }
      break;
    case IBV_WC_LOC_PROT_ERR:
      advice.hint =
          "local protection error; verify the local buffer is still registered with MORI, lkey "
          "matches the posted WR, and transfer offsets/lengths stay within the registered range.";
      break;
    case IBV_WC_LOC_LEN_ERR:
      advice.hint =
          "local length error; verify SGE lengths and transfer offsets stay within the registered "
          "local MR bounds.";
      break;
    case IBV_WC_REM_ACCESS_ERR:
      advice.hint =
          "remote access error; verify the remote buffer is still registered, rkey/permissions "
          "allow this operation, and remote offsets/lengths stay within the registered range.";
      break;
    case IBV_WC_REM_OP_ERR:
      advice.hint =
          "remote operation error; verify both peers use compatible verbs/QP state and the remote "
          "endpoint supports the requested RDMA operation.";
      break;
    default:
      break;
  }
  return advice;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaManager                                          */
/* ---------------------------------------------------------------------------------------------- */

RdmaManager::RdmaManager(const RdmaBackendConfig cfg, application::RdmaContext* ctx)
    : config(cfg), ctx(ctx) {
  application::RdmaDeviceList devices = ctx->GetRdmaDeviceList();
  availDevices = GetActiveDevicePortList(devices);
  assert(availDevices.size() > 0);

  deviceCtxs.resize(availDevices.size(), nullptr);
  topo.reset(new application::TopoSystem());
}

RdmaManager::~RdmaManager() {
  for (auto* devCtx : deviceCtxs) {
    if (devCtx != nullptr) {
      delete devCtx;
    }
  }
  deviceCtxs.clear();

  if (ctx != nullptr) {
    delete ctx;
    ctx = nullptr;
  }
}

std::vector<std::pair<int, int>> RdmaManager::Search(TopoKey key) {
  if (key.loc == MemoryLocationType::GPU) {
    std::string nicName = topo->MatchGpuAndNic(key.deviceId);
    assert(!nicName.empty());
    for (int i = 0; i < availDevices.size(); i++) {
      if (availDevices[i].first->Name() == nicName) {
        return {{i, 1}};
      }
    }
    MORI_IO_WARN("No matching NIC found for GPU {}, nicName: {}", key.deviceId, nicName);
  } else if (key.loc == MemoryLocationType::CPU) {
    if (availDevices.empty()) return {};
    const char* envNic = std::getenv("MORI_IO_RDMA_NIC_IDX");
    if (envNic) {
      int idx = std::atoi(envNic);
      if (idx >= 0 && idx < static_cast<int>(availDevices.size())) {
        return {{idx, 1}};
      }
      MORI_IO_WARN("MORI_IO_RDMA_NIC_IDX={} out of range [0, {}), falling back to round-robin", idx,
                   availDevices.size());
    }
    int idx = (roundRobinCounter.fetch_add(1, std::memory_order_relaxed) % availDevices.size());
    return {{idx, 1}};
  }
  MORI_IO_ERROR(
      "topo searching for device other than CPU/GPU is not implemented yet, returning default "
      "device 0");
  return {{0, 1}};
}

/* ----------------------------------- Local Memory Management ---------------------------------- */
std::optional<application::RdmaMemoryRegion> RdmaManager::GetLocalMemory(int devId,
                                                                         MemoryUniqueId id) {
  std::shared_lock<std::shared_mutex> lock(mu);
  MemoryKey key{devId, id};
  if (mTable.find(key) == mTable.end()) return std::nullopt;
  return mTable[key];
}

application::RdmaMemoryRegion RdmaManager::RegisterLocalMemory(int devId, const MemoryDesc& desc) {
  std::unique_lock<std::shared_mutex> lock(mu);
  MemoryKey key{devId, desc.id};
  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(devId);
  mTable[key] = devCtx->RegisterRdmaMemoryRegion(reinterpret_cast<void*>(desc.data), desc.size);
  return mTable[key];
}

void RdmaManager::DeregisterLocalMemory(int devId, const MemoryDesc& desc) {
  std::unique_lock<std::shared_mutex> lock(mu);
  MemoryKey key{devId, desc.id};
  if (mTable.find(key) != mTable.end()) {
    deviceCtxs[devId]->DeregisterRdmaMemoryRegion(reinterpret_cast<void*>(desc.data));
    mTable.erase(key);
  }
}

/* ---------------------------------- Remote Memory Management ---------------------------------- */
std::optional<application::RdmaMemoryRegion> RdmaManager::GetRemoteMemory(EngineKey ekey,
                                                                          int remRdmaDevId,
                                                                          MemoryUniqueId id) {
  std::shared_lock<std::shared_mutex> lock(mu);
  MemoryKey key{remRdmaDevId, id};
  RemoteEngineMeta& remote = remotes[ekey];
  if (remote.mTable.find(key) == remote.mTable.end()) {
    return std::nullopt;
  }
  return remote.mTable[key];
}

void RdmaManager::RegisterRemoteMemory(EngineKey ekey, int remRdmaDevId, MemoryUniqueId id,
                                       application::RdmaMemoryRegion mr) {
  std::unique_lock<std::shared_mutex> lock(mu);
  MemoryKey key{remRdmaDevId, id};
  RemoteEngineMeta& remote = remotes[ekey];
  remote.mTable[key] = mr;
}

void RdmaManager::DeregisterRemoteMemory(EngineKey ekey, int remRdmaDevId, MemoryUniqueId id) {
  std::unique_lock<std::shared_mutex> lock(mu);
  RemoteEngineMeta& remote = remotes[ekey];
  MemoryKey key{remRdmaDevId, id};
  if (remote.mTable.find(key) != remote.mTable.end()) {
    remote.mTable.erase(key);
  }
}

/* ------------------------------------- Endpoint Management ------------------------------------ */
int RdmaManager::CountEndpoint(EngineKey engine, TopoKeyPair key) {
  std::shared_lock<std::shared_mutex> lock(mu);
  return remotes[engine].rTable[key].size();
}

EpPairVec RdmaManager::GetAllEndpoint(EngineKey engine, TopoKeyPair key) {
  std::shared_lock<std::shared_mutex> lock(mu);
  return remotes[engine].rTable[key];
}

application::RdmaEndpointConfig RdmaManager::GetRdmaEndpointConfig(int devId) {
  const auto& [device, portId] = availDevices[devId];
  const auto* deviceAttr = device->GetDeviceAttr();

  application::RdmaEndpointConfig epConfig{};
  epConfig.portId = portId;
  epConfig.gidIdx = -1;
  const char* envGidIdx = std::getenv("MORI_IB_GID_INDEX");
  if (envGidIdx != nullptr) {
    epConfig.gidIdx = std::atoi(envGidIdx);
  }

  epConfig.enableSrq = false;
  epConfig.alignment = PAGESIZE;
  epConfig.withCompChannel = (config.pollCqMode == PollCqMode::EVENT);

  uint32_t maxQpWr = static_cast<uint32_t>(deviceAttr->orig_attr.max_qp_wr);
  uint32_t maxCqe = static_cast<uint32_t>(deviceAttr->orig_attr.max_cqe);
  uint32_t maxSge = static_cast<uint32_t>(deviceAttr->orig_attr.max_sge);

  if (config.enableNotification && maxQpWr < config.notifPerQp) {
    MORI_IO_ERROR(
        "Device max_qp_wr={} is less than notifPerQp={}; notification requires at least "
        "notifPerQp RQ slots. Either reduce notifPerQp or disable notification.",
        maxQpWr, config.notifPerQp);
    throw std::runtime_error("Device RQ capacity insufficient for configured notifPerQp");
  }

  uint32_t desiredSendWr = config.maxSendWr > 0 ? static_cast<uint32_t>(config.maxSendWr) : 8192u;
  uint32_t desiredRecvWr = config.enableNotification ? config.notifPerQp : 0u;
  uint32_t desiredCqe = config.maxCqeNum > 0 ? static_cast<uint32_t>(config.maxCqeNum) : 16384u;
  std::optional<uint32_t> desiredMsgSge =
      config.maxMsgSge > 0 ? std::optional<uint32_t>(static_cast<uint32_t>(config.maxMsgSge))
                           : std::nullopt;

  env::Override("MORI_IO_QP_MAX_SEND_WR", desiredSendWr, mori::env::detail::ParsePositiveU32);
  env::Override("MORI_IO_QP_MAX_RECV_WR", desiredRecvWr, mori::env::detail::ParsePositiveU32);
  env::Override("MORI_IO_QP_MAX_CQE", desiredCqe, mori::env::detail::ParsePositiveU32);
  env::Override("MORI_IO_QP_MAX_MSG_SGE", desiredMsgSge, mori::env::detail::ParsePositiveU32);
  // Alias for convenience: keep both MORI_IO_QP_MAX_MSG_SGE and MORI_IO_QP_MAX_SGE.
  env::Override("MORI_IO_QP_MAX_SGE", desiredMsgSge, mori::env::detail::ParsePositiveU32);

  if (config.enableNotification && desiredRecvWr < config.notifPerQp) {
    MORI_IO_WARN("MORI_IO_QP_MAX_RECV_WR={} is less than notifPerQp={}; clamping to notifPerQp",
                 desiredRecvWr, config.notifPerQp);
    desiredRecvWr = config.notifPerQp;
  }

  epConfig.maxMsgsNum = std::min(desiredSendWr, maxQpWr);
  // RQ must fit NotifManager's pre-posted recv WQEs (config.notifPerQp) when notification is
  // enabled. MORI_IO_QP_MAX_RECV_WR can raise this baseline, but not lower it.
  epConfig.maxRecvWr = desiredRecvWr > 0 ? std::min(desiredRecvWr, maxQpWr) : 0;
  epConfig.maxCqeNum = std::min(desiredCqe, maxCqe);
  uint32_t minRequiredCqe = epConfig.maxMsgsNum + epConfig.maxRecvWr;
  if (epConfig.maxCqeNum < minRequiredCqe) {
    uint32_t newCqeNum = std::min(minRequiredCqe, maxCqe);
    MORI_IO_WARN(
        "maxCqeNum ({}) is smaller than SQ+RQ depth ({}+{}={}); increasing maxCqeNum to {}",
        epConfig.maxCqeNum, epConfig.maxMsgsNum, epConfig.maxRecvWr, minRequiredCqe, newCqeNum);
    epConfig.maxCqeNum = newCqeNum;
  }
  if (desiredMsgSge.has_value()) {
    epConfig.maxMsgSge = std::min(*desiredMsgSge, maxSge);
  } else {
    bool is_ionic = (deviceAttr->orig_attr.vendor_id ==
                     static_cast<uint32_t>(application::RdmaDeviceVendorId::Pensando));
    epConfig.maxMsgSge = std::min(maxSge, is_ionic ? 2u : 4u);
  }
  return epConfig;
}

application::RdmaEndpoint RdmaManager::CreateEndpoint(int devId) {
  std::unique_lock<std::shared_mutex> lock(mu);

  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(devId);

  application::RdmaEndpoint rdmaEp = devCtx->CreateRdmaEndpoint(GetRdmaEndpointConfig(devId));
  if (config.pollCqMode == PollCqMode::EVENT)
    SYSCALL_RETURN_ZERO(ibv_req_notify_cq(rdmaEp.ibvHandle.cq, 0));
  return rdmaEp;
}

EndpointId RdmaManager::ConnectEndpoint(EngineKey remoteKey, int devId,
                                        application::RdmaEndpoint local, int rdevId,
                                        application::RdmaEndpointHandle remote, TopoKeyPair topoKey,
                                        int weight) {
  std::unique_lock<std::shared_mutex> lock(mu);
  deviceCtxs[devId]->ConnectEndpoint(local.handle, remote);
  RemoteEngineMeta& meta = remotes[remoteKey];
  auto epConfig = GetRdmaEndpointConfig(devId);
  EpPair ep{weight,
            devId,
            rdevId,
            remoteKey,
            local,
            remote,
            std::make_shared<std::atomic<int>>(0),
            static_cast<int>(epConfig.maxMsgsNum),
            std::make_shared<std::atomic<bool>>(false),
            std::make_shared<SubmissionLedger>(config.notifPerQp)};
  meta.rTable[topoKey].push_back(ep);

  EndpointId id = nextEndpointId_.fetch_add(1);
  auto rt = std::make_shared<EndpointRuntime>(id, ep);
  endpointsById_[id] = rt;
  return id;
}

std::shared_ptr<EndpointRuntime> RdmaManager::GetEndpointRuntime(EndpointId id) {
  std::shared_lock<std::shared_mutex> lock(mu);
  auto it = endpointsById_.find(id);
  if (it == endpointsById_.end()) return nullptr;
  return it->second;
}

application::RdmaDeviceContext* RdmaManager::GetRdmaDeviceContext(int devId) {
  std::shared_lock<std::shared_mutex> lock(mu);
  return deviceCtxs[devId];
}

std::vector<std::shared_ptr<EndpointRuntime>> RdmaManager::SnapshotEndpointRuntimes() {
  std::shared_lock<std::shared_mutex> lock(mu);
  std::vector<std::shared_ptr<EndpointRuntime>> result;
  result.reserve(endpointsById_.size());
  for (auto& [_, rt] : endpointsById_) {
    result.push_back(rt);
  }
  return result;
}

application::RdmaDeviceContext* RdmaManager::GetOrCreateDeviceContext(int devId) {
  assert(devId < deviceCtxs.size());
  application::RdmaDeviceContext* devCtx = deviceCtxs[devId];
  if (devCtx == nullptr) {
    devCtx = availDevices[devId].first->CreateRdmaDeviceContext();
    deviceCtxs[devId] = devCtx;
  }
  return devCtx;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Notification Manager                                      */
/* ---------------------------------------------------------------------------------------------- */
NotifManager::NotifManager(RdmaManager* rdmaMgr, const RdmaBackendConfig& cfg)
    : rdma(rdmaMgr), config(cfg) {}

NotifManager::~NotifManager() { Shutdown(); }

void NotifManager::RegisterEndpoint(const std::shared_ptr<EndpointRuntime>& rt) {
  if (config.pollCqMode == PollCqMode::EVENT) {
    epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.u64 = rt->id;
    assert(rt->ep.local.ibvHandle.compCh);
    SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, rt->ep.local.ibvHandle.compCh->fd, &ev));
  }

  // Skip notification setup if disabled
  if (!config.enableNotification) {
    std::lock_guard<std::mutex> lock(mu);
    registeredRuntimes_[rt->id] = rt;
    return;
  }

  std::lock_guard<std::mutex> lock(mu);
  if (notifCtxById_.find(rt->id) != notifCtxById_.end()) return;

  registeredRuntimes_[rt->id] = rt;

  application::RdmaDeviceContext* devCtx = rdma->GetRdmaDeviceContext(rt->ep.ldevId);
  assert(devCtx);

  void* buf;
  SYSCALL_RETURN_ZERO(
      posix_memalign(reinterpret_cast<void**>(&buf), PAGESIZE,
                     static_cast<size_t>(config.notifPerQp) * sizeof(NotifMessage)));
  application::RdmaMemoryRegion mr =
      devCtx->RegisterRdmaMemoryRegion(buf, config.notifPerQp * sizeof(NotifMessage));

  notifCtxById_.insert({rt->id, {mr, buf}});

  struct ibv_qp* qp = rt->ep.local.ibvHandle.qp;
  assert(qp);

  for (uint64_t i = 0; i < config.notifPerQp; i++) {
    struct ibv_sge sge{};
    sge.addr = mr.addr + i * sizeof(NotifMessage);
    sge.length = sizeof(NotifMessage);
    sge.lkey = mr.lkey;

    struct ibv_recv_wr wr{};
    wr.wr_id = i;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr* bad = nullptr;
    SYSCALL_RETURN_ZERO(ibv_post_recv(qp, &wr, &bad));
  }
}

NotifManager::FlushDrainStats NotifManager::ProcessOneCqe(
    const std::shared_ptr<EndpointRuntime>& rt) {
  const EpPair& ep = rt->ep;
  ibv_cq* cq = ep.local.ibvHandle.cq;
  FlushDrainStats flushDrain;

  // Resolve notif context once before the CQ drain loop.
  QpNotifContext* notifCtxPtr = nullptr;
  if (config.enableNotification) {
    std::lock_guard<std::mutex> lock(mu);
    auto nit = notifCtxById_.find(rt->id);
    if (nit != notifCtxById_.end()) notifCtxPtr = &nit->second;
  }

  const int batchSize = 32;
  struct ibv_wc wc[batchSize];
  int n = 0;

  while ((n = ibv_poll_cq(cq, batchSize, wc)) > 0) {
    for (int i = 0; i < n; ++i) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        const bool isFlush = (wc[i].status == IBV_WC_WR_FLUSH_ERR);
        const CqeFailureOrigin failureOrigin =
            ClassifyCqeFailureOrigin(wc[i].wr_id, config.notifPerQp);
        const CqeFailureAdvice failureAdvice =
            isFlush ? CqeFailureAdvice{ibv_wc_status_str(wc[i].status), {}}
                    : DescribeCqeFailure(wc[i].status, failureOrigin, config);

        if (isFlush) {
          flushDrain.Record(wc[i].qp_num);
          MORI_IO_DEBUG("ProcessOneCqe: flush error #{}: wr_id={} qp_num={}", flushDrain.count,
                        wc[i].wr_id, wc[i].qp_num);
        } else {
          // Non-flush error: this is the root cause — always log at ERROR.
          if (failureAdvice.HasHint()) {
            MORI_IO_ERROR(
                "ProcessOneCqe: [ROOT CAUSE] CQE error: wr_id={} status={}({}) qp_num={} "
                "vendor_err={} hint={}",
                wc[i].wr_id, static_cast<uint32_t>(wc[i].status), failureAdvice.statusText,
                wc[i].qp_num, wc[i].vendor_err, failureAdvice.hint);
          } else {
            MORI_IO_ERROR(
                "ProcessOneCqe: [ROOT CAUSE] CQE error: wr_id={} status={}({}) qp_num={} "
                "vendor_err={}",
                wc[i].wr_id, static_cast<uint32_t>(wc[i].status), failureAdvice.statusText,
                wc[i].qp_num, wc[i].vendor_err);
          }
        }

        int mergedBatchSize = 0;
        auto meta = ep.ledger
                        ? ep.ledger->ReleaseByCqe(wc[i].wr_id, ep.sqDepth.get(), &mergedBatchSize)
                        : nullptr;
        if (meta) {
          (void)meta->finishedBatchSize.fetch_add(mergedBatchSize);
          if (isFlush) {
            meta->diagnostics.MarkFlushCascade();
          } else {
            meta->diagnostics.MarkRootCause();
          }
          LogAsyncTransferFailureIfNeeded(&meta->diagnostics,
                                          static_cast<uint32_t>(StatusCode::ERR_RDMA_OP),
                                          failureAdvice.ComposeStatusMessage());
          TransferStatus* statusPtr = meta->status;
          if (statusPtr != nullptr) {
            statusPtr->Update(StatusCode::ERR_RDMA_OP, failureAdvice.ComposeStatusMessage());
            meta->status = nullptr;
          }
          if (ep.degraded && ep.degraded->load(std::memory_order_relaxed) && ep.ledger) {
            const int orphanedReleased = ep.ledger->ReleaseOrphanedByRecovery(ep.sqDepth.get());
            ep.degraded->store(false, std::memory_order_relaxed);
            MORI_IO_WARN(
                "ProcessOneCqe: recovered degraded EP eid={} qpn={} by releasing {} orphaned WRs",
                rt->id, ep.local.handle.qpn, orphanedReleased);
          }
        } else if (IsNotifSendWrId(wc[i].wr_id)) {
          if (ep.sqDepth) ep.sqDepth->fetch_sub(1, std::memory_order_relaxed);
          if (!isFlush) {
            MORI_IO_WARN(
                "ProcessOneCqe: failed notification SEND CQE, transfer_id={}, released 1 sqDepth",
                ExtractTransferIdFromWrId(wc[i].wr_id));
          }
        } else if (wc[i].wr_id < config.notifPerQp) {
          if (!isFlush) {
            MORI_IO_WARN("ProcessOneCqe: failed notification RECV CQE, wr_id={} (recv_idx)",
                         wc[i].wr_id);
          }
        } else {
          if (!isFlush) {
            MORI_IO_WARN(
                "ProcessOneCqe: failed CQE wr_id={} in ledger range but no record found, "
                "sqDepth may be stale",
                wc[i].wr_id);
          }
        }
        continue;
      }

      if (wc[i].opcode == IBV_WC_RECV) {
        // Skip RECV processing if notification is disabled
        if (!config.enableNotification) {
          MORI_IO_WARN("Received unexpected RECV completion when notification is disabled");
          continue;
        }

        std::lock_guard<std::mutex> lock(mu);

        assert(notifCtxPtr != nullptr);
        QpNotifContext& ctx = *notifCtxPtr;

        // FIXME: this notif mechenism has bug when notif index is wrapped around
        uint64_t idx = wc[i].wr_id;
        NotifMessage msg = reinterpret_cast<NotifMessage*>(ctx.mr.addr)[idx];
        assert(msg.totalNum > 0);

        EngineKey ekey = ep.remoteEngineKey;
        if (notifPool[ekey].find(msg.id) == notifPool[ekey].end()) {
          notifPool[ekey][msg.id] = msg.totalNum;
        }
        notifPool[ekey][msg.id] -= 1;
        MORI_IO_TRACE(
            "NotifManager receive notif message from engine {} id {} qp {} total num {} cur num {}",
            ekey.c_str(), msg.id, msg.qpIndex, msg.totalNum, notifPool[ekey][msg.id]);
        // replenish recv wr
        struct ibv_sge sge{};
        sge.addr = ctx.mr.addr + idx * sizeof(NotifMessage);
        sge.length = sizeof(NotifMessage);
        sge.lkey = ctx.mr.lkey;

        struct ibv_recv_wr wr{};
        wr.wr_id = idx;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        struct ibv_recv_wr* bad = nullptr;
        SYSCALL_RETURN_ZERO(ibv_post_recv(ep.local.ibvHandle.qp, &wr, &bad));
      } else if (wc[i].opcode == IBV_WC_SEND) {
        if (!IsNotifSendWrId(wc[i].wr_id)) {
          MORI_IO_WARN(
              "ProcessOneCqe: unexpected SEND completion with non-notification wr_id {}; "
              "releasing 1 sqDepth under current SEND invariant",
              wc[i].wr_id);
        }
        if (ep.sqDepth) ep.sqDepth->fetch_sub(1, std::memory_order_relaxed);
      } else {
        // Batch path: wr_id carries a recordId from the SubmissionLedger.
        uint64_t recordId = wc[i].wr_id;
        int mergedBatchSize = 0;
        auto meta = ep.ledger
                        ? ep.ledger->ReleaseByCqe(recordId, ep.sqDepth.get(), &mergedBatchSize)
                        : nullptr;
        if (meta) {
          uint32_t finishedBefore = meta->finishedBatchSize.fetch_add(mergedBatchSize);
          TransferStatus* statusPtr = meta->status;
          if (statusPtr != nullptr && (finishedBefore + mergedBatchSize) == meta->totalBatchSize) {
            statusPtr->Update(StatusCode::SUCCESS, ibv_wc_status_str(wc[i].status));
          }
          MORI_IO_TRACE("ProcessOneCqe: batch CQE for task {} total={} finished={} cur={}",
                        meta->id, meta->totalBatchSize, finishedBefore, mergedBatchSize);
        } else {
          MORI_IO_WARN(
              "ProcessOneCqe: no ledger record for wr_id {} (recordId {}); sqDepth may be stale",
              wc[i].wr_id, recordId);
        }
      }
    }
  }

  if (!flushDrain.Empty()) {
    MORI_IO_DEBUG("ProcessOneCqe: drain — {} flush errors on eid={} qp_num={}", flushDrain.count,
                  rt->id, flushDrain.firstQpNum);
  }
  return flushDrain;
}

void NotifManager::EmitFlushSummaryIfNeeded(const FlushRoundStats& roundStats) {
  if (roundStats.Empty()) {
    flushSummaryStreak_ = 0;
    return;
  }

  flushSummaryStreak_++;
  const bool shouldLog =
      (flushSummaryStreak_ == 1) ||
      (flushSummaryStreak_ < 64 && (flushSummaryStreak_ & (flushSummaryStreak_ - 1)) == 0) ||
      (flushSummaryStreak_ % 1000 == 0);

  if (shouldLog) {
    if (flushSummaryStreak_ == 1) {
      MORI_IO_ERROR(
          "CQ poll round summary: {} flush errors across {} endpoint(s); "
          "representative eid={} qp_num={}. "
          "Flush errors are cascaded from QP(s) entering Error State. "
          "Check: (1) peer process alive, (2) PFC / network congestion, "
          "(3) ibv_devinfo / dmesg for HW errors",
          roundStats.total, roundStats.endpointCount, roundStats.sampleEndpointId,
          roundStats.sampleQpNum);
    } else {
      MORI_IO_WARN(
          "CQ poll round summary: {} flush errors across {} endpoint(s); "
          "representative eid={} qp_num={}; in "
          "consecutive flush round #{} (rate-limited). "
          "Flush errors are cascaded from QP(s) entering Error State. "
          "Check: (1) peer process alive, (2) PFC / network congestion, "
          "(3) ibv_devinfo / dmesg for HW errors",
          roundStats.total, roundStats.endpointCount, roundStats.sampleEndpointId,
          roundStats.sampleQpNum, flushSummaryStreak_);
    }
  }
}

void NotifManager::MainLoop() {
  if (config.pollCqMode == PollCqMode::EVENT) {
    constexpr int maxEvents = 128;
    epoll_event events[maxEvents];
    while (running.load()) {
      FlushRoundStats roundStats;
      bool handledCqEvent = false;
      int nfds = epoll_wait(epfd, events, maxEvents, 0 /*ms*/);
      for (int i = 0; i < nfds; ++i) {
        EndpointId eid = events[i].data.u64;

        std::shared_ptr<EndpointRuntime> rt;
        {
          std::lock_guard<std::mutex> lock(mu);
          auto it = registeredRuntimes_.find(eid);
          if (it == registeredRuntimes_.end()) continue;
          rt = it->second;
        }

        struct ibv_comp_channel* ch = rt->ep.local.ibvHandle.compCh;

        struct ibv_cq* cq = nullptr;
        void* evCtx = nullptr;
        if (ibv_get_cq_event(ch, &cq, &evCtx)) continue;
        ibv_ack_cq_events(cq, 1);
        ibv_req_notify_cq(cq, 0);

        handledCqEvent = true;
        roundStats.Merge(rt->id, ProcessOneCqe(rt));
      }
      if (handledCqEvent) {
        EmitFlushSummaryIfNeeded(roundStats);
      }
    }
  } else {
    while (running.load()) {
      auto snapshot = rdma->SnapshotEndpointRuntimes();
      if (snapshot.empty()) {
        EmitFlushSummaryIfNeeded(FlushRoundStats{});
        std::this_thread::yield();
        continue;
      }
      FlushRoundStats roundStats;
      for (auto& rt : snapshot) {
        roundStats.Merge(rt->id, ProcessOneCqe(rt));
      }
      EmitFlushSummaryIfNeeded(roundStats);
    }
  }
}

bool NotifManager::PopInboundTransferStatus(const EngineKey& remote, TransferUniqueId id,
                                            TransferStatus* status) {
  std::lock_guard<std::mutex> lock(mu);
  if (notifPool[remote].find(id) != notifPool[remote].end()) {
    if (notifPool[remote][id] == 0) {
      status->SetCode(StatusCode::SUCCESS);
      return true;
    }
  }
  return false;
}

void NotifManager::Start() {
  if (running.load()) return;
  if (config.pollCqMode == PollCqMode::EVENT) {
    epfd = epoll_create1(EPOLL_CLOEXEC);
    assert(epfd >= 0);
  }
  running.store(true);
  thd = std::thread([this] { MainLoop(); });
}

void NotifManager::Shutdown() {
  running.store(false);
  if (config.pollCqMode == PollCqMode::EVENT) {
    epfd = close(epfd);
  }
  if (thd.joinable()) thd.join();
}

/* ----------------------------------------------------------------------------------------------
 */
/*                                      Control Plane Server */
/* ----------------------------------------------------------------------------------------------
 */
ControlPlaneServer::ControlPlaneServer(const std::string& k, const std::string& host, int port,
                                       RdmaManager* rdmaMgr, NotifManager* notifMgr)
    : myEngKey(k) {
  ctx.reset(new application::TCPContext(host, port));
  rdma = rdmaMgr;
  notif = notifMgr;
}

ControlPlaneServer::~ControlPlaneServer() { Shutdown(); }

void ControlPlaneServer::RegisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(mu);
  engines[rdesc.key] = rdesc;
}

void ControlPlaneServer::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  std::lock_guard<std::mutex> lock(mu);
  engines.erase(rdesc.key);
}

void ControlPlaneServer::BuildRdmaConn(EngineKey ekey, TopoKeyPair topo) {
  application::TCPEndpointHandle tcph;
  {
    std::lock_guard<std::mutex> lock(mu);
    assert((engines.find(ekey) != engines.end()) && "register engine first");
    EngineDesc& rdesc = engines[ekey];
    tcph = ctx->Connect(rdesc.host, rdesc.port);
  }

  auto candidates = rdma->Search(topo.local);
  assert(!candidates.empty());
  auto [devId, weight] = candidates[0];

  application::RdmaEndpoint lep = rdma->CreateEndpoint(devId);

  Protocol p(tcph);
  p.WriteMessageRegEndpoint({myEngKey, topo, devId, lep.handle});
  MessageHeader hdr = p.ReadMessageHeader();
  assert(hdr.type == MessageType::RegEndpoint);
  MessageRegEndpoint msg = p.ReadMessageRegEndpoint(hdr.len);

  EndpointId eid = rdma->ConnectEndpoint(ekey, devId, lep, msg.devId, msg.eph, topo, weight);
  auto ert = rdma->GetEndpointRuntime(eid);
  notif->RegisterEndpoint(ert);
  MORI_IO_INFO("Built RdmaConn for engine {} with topo local({},{}) remote({},{})", ekey,
               topo.local.deviceId, topo.local.loc, topo.remote.deviceId, topo.remote.loc);
  ctx->CloseEndpoint(tcph);
}

void ControlPlaneServer::RegisterMemory(MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  mems[desc.id] = desc;
}

void ControlPlaneServer::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  mems.erase(desc.id);
}

application::RdmaMemoryRegion ControlPlaneServer::AskRemoteMemoryRegion(EngineKey ekey, int rdevId,
                                                                        MemoryUniqueId id) {
  application::TCPEndpointHandle tcph;
  {
    std::lock_guard<std::mutex> lock(mu);
    assert((engines.find(ekey) != engines.end()) && "register engine first");
    EngineDesc& rdesc = engines[ekey];
    tcph = ctx->Connect(rdesc.host, rdesc.port);
  }

  Protocol p(tcph);
  p.WriteMessageAskMemoryRegion({ekey, rdevId, id, {}});
  MessageHeader hdr = p.ReadMessageHeader();
  assert(hdr.type == MessageType::AskMemoryRegion);
  MessageAskMemoryRegion msg = p.ReadMessageAskMemoryRegion(hdr.len);

  return msg.mr;
}

void ControlPlaneServer::AcceptRemoteEngineConn() {
  application::TCPEndpointHandleVec newEps = ctx->Accept();
  for (auto& ep : newEps) {
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = ep.fd;
    SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, ep.fd, &ev));
    eps.insert({ep.fd, ep});
  }
}

void ControlPlaneServer::HandleControlPlaneProtocol(int fd) {
  assert(eps.find(fd) != eps.end());
  application::TCPEndpointHandle tcph = eps[fd];

  Protocol p(tcph);
  MessageHeader hdr = p.ReadMessageHeader();

  switch (hdr.type) {
    case MessageType::RegEndpoint: {
      MessageRegEndpoint msg = p.ReadMessageRegEndpoint(hdr.len);
      auto candidates = rdma->Search(msg.topo.remote);
      assert(!candidates.empty());
      int rdevId = msg.devId;
      auto [devId, weight] = candidates[0];
      application::RdmaEndpoint lep = rdma->CreateEndpoint(devId);
      EndpointId eid =
          rdma->ConnectEndpoint(msg.ekey, devId, lep, rdevId, msg.eph, msg.topo, weight);
      auto ert = rdma->GetEndpointRuntime(eid);
      notif->RegisterEndpoint(ert);
      p.WriteMessageRegEndpoint(MessageRegEndpoint{myEngKey, msg.topo, devId, lep.handle});
      SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL));
      break;
    }
    case MessageType::AskMemoryRegion: {
      std::lock_guard<std::mutex> lock(mu);
      MessageAskMemoryRegion msg = p.ReadMessageAskMemoryRegion(hdr.len);
      if (mems.find(msg.id) != mems.end()) {
        MemoryDesc& desc = mems[msg.id];
        auto localMr = rdma->GetLocalMemory(msg.devId, msg.id);
        if (!localMr.has_value()) {
          localMr = rdma->RegisterLocalMemory(msg.devId, desc);
        }
        p.WriteMessageAskMemoryRegion({msg.ekey, msg.devId, msg.id, *localMr});
      } else {
        // TODO: we should add status code for NOT_FOUND
        p.WriteMessageAskMemoryRegion({msg.ekey, msg.devId, msg.id, {}});
      }
      break;
    }
    default:
      assert(false && "not implemented");
  }

  ctx->CloseEndpoint(tcph);
  eps.erase(fd);
}

void ControlPlaneServer::MainLoop() {
  constexpr int maxEvents = 128;
  epoll_event events[maxEvents];

  while (running.load()) {
    int nfds = epoll_wait(epfd, events, maxEvents, 5 /*ms*/);

    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;

      // Add new endpoints into epoll list
      if (fd == ctx->GetListenFd()) {
        AcceptRemoteEngineConn();
        continue;
      }

      HandleControlPlaneProtocol(fd);
    }
  }
}

void ControlPlaneServer::Start() {
  if (running.load()) return;

  // Create epoll fd
  epfd = epoll_create1(EPOLL_CLOEXEC);
  assert(epfd >= 0);

  // Add TCP listen fd
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ctx->Listen();
  ev.data.fd = ctx->GetListenFd();
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, ctx->GetListenFd(), &ev));

  running.store(true);
  thd = std::thread([this] { MainLoop(); });
}

void ControlPlaneServer::Shutdown() {
  running.store(false);
  if (thd.joinable()) thd.join();
  if (epfd >= 0) {
    close(epfd);
    epfd = -1;
  }
}

/* ----------------------------------------------------------------------------------------------
 */
/*                                       RdmaBackendSession */
/* ----------------------------------------------------------------------------------------------
 */
RdmaBackendSession::RdmaBackendSession(const RdmaBackendConfig& config,
                                       const application::RdmaMemoryRegion& l,
                                       const application::RdmaMemoryRegion& r, const EpPairVec& e,
                                       Executor* exec)
    : config(config), local(l), remote(r), eps(e), executor(exec) {}

void RdmaBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                   TransferStatus* status, TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  status->SetCode(StatusCode::IN_PROGRESS);
  auto callbackMeta = std::make_shared<CqCallbackMeta>(status, id, 1);
  internal::PublishCurrentIoCallDiagnostics(callbackMeta);

  RdmaOpRet ret =
      RdmaReadWrite(eps, local, localOffset, remote, remoteOffset, size, callbackMeta, id, isRead);

  assert(!ret.Init());
  if (ret.Failed() || ret.Succeeded()) {
    status->Update(ret.code, ret.message);
  }
  if (!ret.Failed() && config.enableNotification) {
    RdmaOpRet notifRet = RdmaNotifyTransfer(eps, status, id);
    if (notifRet.Failed()) {
      status->Update(notifRet.code, notifRet.message);
    }
  }
}

void RdmaBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                        const SizeVec& sizes, TransferStatus* status,
                                        TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  status->SetCode(StatusCode::IN_PROGRESS);
  auto callbackMeta = std::make_shared<CqCallbackMeta>(status, id, sizes.size());
  internal::PublishCurrentIoCallDiagnostics(callbackMeta);
  RdmaOpRet ret;
  if (executor) {
    ExecutorReq req{eps,          local, localOffsets,         remote, remoteOffsets, sizes,
                    callbackMeta, id,    config.postBatchSize, isRead};
    ret = executor->RdmaBatchReadWrite(req);
  } else {
    ret = RdmaBatchReadWrite(eps, local, localOffsets, remote, remoteOffsets, sizes, callbackMeta,
                             id, isRead, config.postBatchSize);
  }
  assert(!ret.Init());
  if (ret.Failed() || ret.Succeeded()) {
    status->Update(ret.code, ret.message);
  }
  if (!ret.Failed() && config.enableNotification) {
    RdmaOpRet notifRet = RdmaNotifyTransfer(eps, status, id);
    if (notifRet.Failed()) {
      status->Update(notifRet.code, notifRet.message);
    }
  }
}

bool RdmaBackendSession::Alive() const { return true; }

/* ----------------------------------------------------------------------------------------------
 */
/*                                           RdmaBackend */
/* ----------------------------------------------------------------------------------------------
 */

RdmaBackend::RdmaBackend(EngineKey k, const IOEngineConfig& engConfig,
                         const RdmaBackendConfig& beConfig)
    : myEngKey(k), config(beConfig) {
  env::Override("MORI_IO_ENABLE_NOTIFICATION", config.enableNotification,
                mori::env::detail::ParseBool);
  ValidateRdmaNotificationConfig(config);

  application::RdmaContext* ctx =
      new application::RdmaContext(application::RdmaBackendType::IBVerbs);
  rdma.reset(new mori::io::RdmaManager(config, ctx));

  notif.reset(new NotifManager(rdma.get(), config));
  notif->Start();

  server.reset(
      new ControlPlaneServer(myEngKey, engConfig.host, engConfig.port, rdma.get(), notif.get()));
  server->Start();

  if (config.numWorkerThreads > 1) {
    executor.reset(
        new MultithreadExecutor(std::min(config.qpPerTransfer, config.numWorkerThreads)));
    executor->Start();
  }

  std::stringstream ss;
  ss << config;
  MORI_IO_INFO("RdmaBackend created with config: {}", ss.str().c_str());
}

RdmaBackend::~RdmaBackend() {
  notif->Shutdown();
  server->Shutdown();
  if (executor.get() != nullptr) {
    executor->Shutdown();
  }
}

void RdmaBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  server->RegisterRemoteEngine(rdesc);
}

void RdmaBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  server->DeregisterRemoteEngine(rdesc);
}

void RdmaBackend::RegisterMemory(MemoryDesc& desc) { server->RegisterMemory(desc); }

void RdmaBackend::DeregisterMemory(const MemoryDesc& desc) {
  server->DeregisterMemory(desc);
  InvalidateSessionsForMemory(desc.id);
}

void RdmaBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset,
                            const MemoryDesc& remoteSrc, size_t remoteOffset, size_t size,
                            TransferStatus* status, TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  RdmaBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
  sess->ReadWrite(localOffset, remoteOffset, size, status, id, isRead);
}

void RdmaBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                                 const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                                 const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                 bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  assert(localOffsets.size() == remoteOffsets.size());
  assert(sizes.size() == remoteOffsets.size());
  size_t batchSize = sizes.size();
  if (batchSize == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  RdmaBackendSession* sess = GetOrCreateSessionCached(localDest, remoteSrc);
  sess->BatchReadWrite(localOffsets, remoteOffsets, sizes, status, id, isRead);
}

BackendSession* RdmaBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  RdmaBackendSession* sess = new RdmaBackendSession();
  CreateSession(local, remote, *sess);
  return sess;
}

void RdmaBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote,
                                RdmaBackendSession& sess) {
  TopoKey localKey{local.deviceId, local.loc};
  TopoKey remoteKey{remote.deviceId, remote.loc};
  TopoKeyPair kp{localKey, remoteKey};

  EngineKey ekey = remote.engineKey;

  // Create a pair of endpoint if none
  int epNum = rdma->CountEndpoint(ekey, kp);
  for (int i = 0; i < (config.qpPerTransfer - epNum); i++) {
    server->BuildRdmaConn(ekey, kp);
  }
  EpPairVec eps = rdma->GetAllEndpoint(ekey, kp);
  assert(!eps.empty());

  EpPairVec epSet = {eps.begin(), eps.begin() + config.qpPerTransfer};

  // TODO: we assume all eps is on same device and has same ldevId/rdevId
  EpPair ep = epSet[0];
  auto localMr = rdma->GetLocalMemory(ep.ldevId, local.id);
  if (!localMr.has_value()) {
    localMr = rdma->RegisterLocalMemory(ep.ldevId, local);
  }

  auto remoteMr = rdma->GetRemoteMemory(ekey, ep.rdevId, remote.id);
  if (!remoteMr.has_value()) {
    remoteMr = server->AskRemoteMemoryRegion(ekey, ep.rdevId, remote.id);
    // TODO: protocol should return status code
    // Currently we check member equality to ensure correct memory region
    assert(remoteMr->length == remote.size);
    rdma->RegisterRemoteMemory(ekey, ep.rdevId, remote.id, remoteMr.value());
  }

  sess = RdmaBackendSession(config, localMr.value(), remoteMr.value(), epSet, executor.get());
}

bool RdmaBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                           TransferStatus* status) {
  return notif->PopInboundTransferStatus(remote, id, status);
}

RdmaBackendSession* RdmaBackend::GetOrCreateSessionCached(const MemoryDesc& local,
                                                          const MemoryDesc& remote) {
  SessionCacheKey key{remote.engineKey, local.id, remote.id};
  {
    std::lock_guard<std::mutex> lock(sessionCacheMu);
    auto it = sessionCache.find(key);
    if (it != sessionCache.end()) {
      return it->second.get();
    }
  }
  // create outside lock (CreateSession may allocate / block); then insert
  auto newSess = std::make_unique<RdmaBackendSession>();
  CreateSession(local, remote, *newSess);
  std::lock_guard<std::mutex> lock(sessionCacheMu);
  auto it = sessionCache.find(key);
  if (it != sessionCache.end()) {
    return it->second.get();
  }
  auto [emplacedIt, inserted] = sessionCache.emplace(key, std::move(newSess));
  return emplacedIt->second.get();
}

void RdmaBackend::InvalidateSessionsForMemory(MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(sessionCacheMu);
  for (auto it = sessionCache.begin(); it != sessionCache.end();) {
    if (it->first.localMemId == id || it->first.remoteMemId == id) {
      it = sessionCache.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace io
}  // namespace mori

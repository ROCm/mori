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
#include <chrono>
#include <shared_mutex>

#include "mori/io/logging.hpp"
#include "src/io/rdma/protocol.hpp"
namespace mori {
namespace io {

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

std::vector<std::pair<int, int>> RdmaManager::Search(TopoKey key) {
  if (key.loc == MemoryLocationType::GPU) {
    std::string nicName = topo->MatchGpuAndNic(key.deviceId);
    assert(!nicName.empty());
    for (int i = 0; i < availDevices.size(); i++) {
      if (availDevices[i].first->Name() == nicName) {
        return {{i, 1}};
      }
    }
  } else {
    assert("topo searching for device other than GPU is not implemented yet");
    return {};
  }
  return {};
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
  EpPairVec out;
  auto engIt = remotes.find(engine);
  if (engIt == remotes.end()) return out;
  auto rtIt = engIt->second.rTable.find(key);
  if (rtIt == engIt->second.rTable.end()) return out;
  out.reserve(rtIt->second.size());
  for (auto& handle : rtIt->second) {
    out.push_back(handle->shared());
  }
  return out;
}

EpHandleVec RdmaManager::GetAllEndpointHandles(EngineKey engine, TopoKeyPair key) {
  std::shared_lock<std::shared_mutex> lock(mu);
  EpHandleVec out;
  auto engIt = remotes.find(engine);
  if (engIt == remotes.end()) return out;
  auto rtIt = engIt->second.rTable.find(key);
  if (rtIt == engIt->second.rTable.end()) return out;
  out = rtIt->second;  // shallow copy of shared_ptr handles
  return out;
}

application::RdmaEndpointConfig RdmaManager::GetRdmaEndpointConfig(int portId) {
  application::RdmaEndpointConfig config;
  config.portId = portId;
  config.gidIdx = 3;
  config.maxMsgsNum = 8192;
  config.maxMsgSge = 1;
  config.maxCqeNum = 8192;
  config.alignment = PAGESIZE;
  config.withCompChannel = true;
  config.enableSrq = true;
  return config;
}

application::RdmaEndpoint RdmaManager::CreateEndpoint(int devId) {
  std::unique_lock<std::shared_mutex> lock(mu);

  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(devId);

  application::RdmaEndpoint rdmaEp =
      devCtx->CreateRdmaEndpoint(GetRdmaEndpointConfig(availDevices[devId].second));
  if (config.pollCqMode == PollCqMode::EVENT)
    SYSCALL_RETURN_ZERO(ibv_req_notify_cq(rdmaEp.ibvHandle.cq, 0));
  return rdmaEp;
}

void RdmaManager::ConnectEndpoint(EngineKey remoteKey, int devId, application::RdmaEndpoint local,
                                  int rdevId, application::RdmaEndpointHandle remote,
                                  TopoKeyPair topoKey, int weight) {
  std::unique_lock<std::shared_mutex> lock(mu);
  deviceCtxs[devId]->ConnectEndpoint(local.handle, remote);
  RemoteEngineMeta& meta = remotes[remoteKey];
  EpPairPtr epPair = std::make_shared<EpPair>();
  epPair->weight = weight;
  epPair->ldevId = devId;
  epPair->rdevId = rdevId;
  epPair->remoteEngineKey = remoteKey;
  epPair->local = local;
  epPair->remote = remote;
  epPair->topoKey = topoKey;
  EpHandlePtr handle = std::make_shared<EpHandle>(epPair);
  meta.rTable[topoKey].push_back(handle);
  epsMap.insert({epPair->local.handle.qpn, epPair});
}

std::optional<EpPairPtr> RdmaManager::GetEpPairPtrByQpn(uint32_t qpn) {
  std::shared_lock<std::shared_mutex> lock(mu);
  auto it = epsMap.find(qpn);
  if (it == epsMap.end()) return std::nullopt;
  return it->second;
}

std::optional<EpPairPtr> RdmaManager::RebuildEndpoint(uint32_t qpn) {
  std::unique_lock<std::shared_mutex> lock(mu);
  auto it = epsMap.find(qpn);
  if (it == epsMap.end()) return std::nullopt;
  EpPairPtr oldEp = it->second;
  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(oldEp->ldevId);
  // 1. Create brand new endpoint
  application::RdmaEndpoint newLocal =
      devCtx->CreateRdmaEndpoint(GetRdmaEndpointConfig(availDevices[oldEp->ldevId].second));
  deviceCtxs[oldEp->ldevId]->ConnectEndpoint(newLocal.handle, oldEp->remote);
  // 2. Prepare new EpPair
  EpPairPtr newEp = std::make_shared<EpPair>();
  newEp->weight = oldEp->weight;
  newEp->ldevId = oldEp->ldevId;
  newEp->rdevId = oldEp->rdevId;
  newEp->remoteEngineKey = oldEp->remoteEngineKey;
  newEp->remote = oldEp->remote;
  newEp->topoKey = oldEp->topoKey;
  newEp->rebuild_generation.store(oldEp->rebuild_generation.load(std::memory_order_relaxed) + 1,
                                  std::memory_order_relaxed);
  newEp->scheduled_rebuild.store(false, std::memory_order_release);
  newEp->local = newLocal;
  // 3. Find its handle in route table and atomically swap pointer inside handle
  RemoteEngineMeta& meta = remotes[newEp->remoteEngineKey];
  auto& vec = meta.rTable[newEp->topoKey];
  for (auto& h : vec) {
    if (h->qpn() == qpn) {  // match old
      h->update(newEp);     // atomic store of new raw pointer
      break;
    }
  }
  // 4. Update epsMap: remove old qpn entry, insert new qpn->newEp
  epsMap.erase(it);
  epsMap.emplace(newEp->local.handle.qpn, newEp);
  // 5. Mark old as retired (optional flags)
  oldEp->scheduled_rebuild.store(false, std::memory_order_release);
  return newEp;
}

application::RdmaDeviceContext* RdmaManager::GetRdmaDeviceContext(int devId) {
  std::shared_lock<std::shared_mutex> lock(mu);
  return deviceCtxs[devId];
}

void RdmaManager::EnumerateEndpoints(const EnumerateEpCallbackFunc& func) {
  std::shared_lock<std::shared_mutex> lock(mu);
  for (auto& it : epsMap) {
    func(it.first, *it.second);
  }
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
NotifManager::NotifManager(RdmaManager* rdmaMgr, const RdmaBackendConfig& cfg,
                           Backend* ownerBackend)
    : rdma(rdmaMgr), config(cfg), owner(ownerBackend) {}

NotifManager::~NotifManager() { Shutdown(); }

void NotifManager::RegisterEndpointByQpn(uint32_t qpn) {
  if (config.pollCqMode == PollCqMode::EVENT) {
    epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.u32 = qpn;
    std::optional<EpPairPtr> ep = rdma->GetEpPairPtrByQpn(qpn);
    assert(ep.has_value() && (*ep)->local.ibvHandle.compCh);
    SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, (*ep)->local.ibvHandle.compCh->fd, &ev));
  }
}

void NotifManager::UnregisterEndpointByQpn(uint32_t qpn) {
  if (config.pollCqMode == PollCqMode::EVENT) {
    std::optional<EpPairPtr> ep = rdma->GetEpPairPtrByQpn(qpn);
    if (ep.has_value() && (*ep)->local.ibvHandle.compCh) {
      epoll_ctl(epfd, EPOLL_CTL_DEL, (*ep)->local.ibvHandle.compCh->fd, nullptr);
    }
  }
}

void NotifManager::RegisterDevice(int devId) {
  std::lock_guard<std::mutex> lock(mu);
  if (notifCtx.find(devId) != notifCtx.end()) return;

  application::RdmaDeviceContext* devCtx = rdma->GetRdmaDeviceContext(devId);
  assert(devCtx);

  void* buf;
  SYSCALL_RETURN_ZERO(
      posix_memalign(reinterpret_cast<void**>(&buf), PAGESIZE, maxNotifNum * sizeof(NotifMessage)));
  application::RdmaMemoryRegion mr =
      devCtx->RegisterRdmaMemoryRegion(buf, maxNotifNum * sizeof(NotifMessage));
  struct ibv_srq* srq = devCtx->GetIbvSrq();
  assert(srq);
  notifCtx.insert({devId, {srq, mr}});

  // Pre post notification receive wr
  // TODO: should use min(maxNotifNum, maxSrqWrNum)
  for (uint64_t i = 0; i < maxNotifNum; i++) {
    struct ibv_sge sge{};
    sge.addr = mr.addr + i * sizeof(NotifMessage);
    sge.length = sizeof(NotifMessage);
    sge.lkey = mr.lkey;

    struct ibv_recv_wr wr{};
    wr.wr_id = i;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr* bad = nullptr;
    SYSCALL_RETURN_ZERO(ibv_post_srq_recv(srq, &wr, &bad));
  };
}

void NotifManager::ProcessOneCqe(int qpn, const EpPair& ep) {
  ibv_cq* cq = ep.local.ibvHandle.cq;

  struct ibv_wc wc{};
  int rc = 0;
  while ((rc = ibv_poll_cq(cq, 1, &wc)) > 0) {
    if (wc.opcode == IBV_WC_RECV) {
      std::lock_guard<std::mutex> lock(mu);
      int devId = ep.ldevId;

      assert(notifCtx.find(devId) != notifCtx.end());
      DeviceNotifContext& ctx = notifCtx[devId];

      // FIXME: this notif mechenism has bug when notif index is wrapped around
      uint64_t idx = wc.wr_id;
      NotifMessage msg = reinterpret_cast<NotifMessage*>(ctx.mr.addr)[idx];
      assert(msg.totalNum > 0);
      // printf("recv notif for transfer %d\n", tid);

      EngineKey ekey = ep.remoteEngineKey;
      if (notifPool[ekey].find(msg.id) == notifPool[ekey].end()) {
        notifPool[ekey][msg.id] = msg.totalNum;
      }
      notifPool[ekey][msg.id] -= 1;
      MORI_IO_TRACE(
          "NotifManager receive notif message from engine {} id {} qp {} total num {} cur num {}",
          ekey.c_str(), msg.id, msg.qpIndex, msg.totalNum, notifPool[ekey][msg.id]);
      // replenish recv wr
      // TODO(ditian12): we should replenish recv wr faster, insufficient recv wr is met
      // frequently when transfer is very fast. Two way to solve this, 1. use srq_limit to
      // replenish in advance
      // 2. independent srq entry config (now reuse maxMsgNum)
      struct ibv_sge sge{};
      sge.addr = ctx.mr.addr + idx * sizeof(NotifMessage);
      sge.length = sizeof(NotifMessage);
      sge.lkey = ctx.mr.lkey;

      struct ibv_recv_wr wr{};
      wr.wr_id = idx;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      struct ibv_recv_wr* bad = nullptr;
      SYSCALL_RETURN_ZERO(ibv_post_srq_recv(ctx.srq, &wr, &bad));
    } else if (wc.opcode == IBV_WC_SEND) {
      uint64_t id = wc.wr_id;
    } else {
      CqCallbackMessage* msg = reinterpret_cast<CqCallbackMessage*>(wc.wr_id);
      uint32_t lastBatchSize = msg->meta->finishedBatchSize.fetch_add(msg->batchSize);
      if (msg->meta->status != nullptr) {
        if (wc.status == IBV_WC_SUCCESS) {
          if ((lastBatchSize + msg->batchSize) == msg->meta->totalBatchSize) {
            // TODO: should use atomic cas to avoid overwriting failed status
            msg->meta->status->SetCode(StatusCode::SUCCESS);
            msg->meta->status->SetMessage(ibv_wc_status_str(wc.status));
          }
        } else {
          msg->meta->status->SetCode(StatusCode::ERR_RDMA_OP);
          msg->meta->status->SetMessage(ibv_wc_status_str(wc.status));
          // set status to nullptr indicate that transfer failed
          msg->meta->status = nullptr;
          // Error reporting hook: map wc.status to Backend::ErrorRecord severity.
          Backend::ErrorRecord rec{};
          rec.vendor_err = wc.vendor_err;
          rec.msg = ibv_wc_status_str(wc.status);
          switch (wc.status) {
            case IBV_WC_SUCCESS:
              rec.code = Backend::ErrorCode::Ok;
              rec.severity = Backend::Severity::Info;
              break;
            case IBV_WC_RNR_RETRY_EXC_ERR:
              rec.code = Backend::ErrorCode::Transient;
              rec.severity = Backend::Severity::Recoverable;
              break;
            case IBV_WC_RETRY_EXC_ERR:
              rec.code = Backend::ErrorCode::Transient;
              rec.severity = Backend::Severity::Recoverable;
              break;
            case IBV_WC_RESP_TIMEOUT_ERR:
              rec.code = Backend::ErrorCode::Timeout;
              rec.severity = Backend::Severity::Recoverable;
              break;
            case IBV_WC_LOC_PROT_ERR:
              rec.code = Backend::ErrorCode::Protection;
              rec.severity = Backend::Severity::Fatal;
              break;
            case IBV_WC_REM_OP_ERR:
              rec.code = Backend::ErrorCode::RemoteDisconnect;
              rec.severity = Backend::Severity::Recoverable;  // may escalate if frequent
              break;
            case IBV_WC_WR_FLUSH_ERR:
              rec.code = Backend::ErrorCode::Transient;
              rec.severity = Backend::Severity::Recoverable;
              break;
            default:
              rec.code = Backend::ErrorCode::Internal;
              rec.severity = Backend::Severity::Fatal;
              break;
          }
          if (owner) owner->report_error(rec);
          // QP rebuild scheduling logic (lightweight heuristic for now)
          if (rec.severity != Backend::Severity::Info) {
            auto epPtrOpt = rdma->GetEpPairPtrByQpn(ep.local.handle.qpn);
            if (epPtrOpt.has_value()) {
              EpPairPtr epPtr = *epPtrOpt;
              bool needRebuild = false;
              switch (wc.status) {
                case IBV_WC_WR_FLUSH_ERR:
                  epPtr->flush_err_count.fetch_add(1, std::memory_order_relaxed);
                  needRebuild = true;  // flush indicates QP moved to error
                  break;
                case IBV_WC_RETRY_EXC_ERR:
                  if (epPtr->retry_exhausted_count.fetch_add(1, std::memory_order_relaxed) > 5)
                    needRebuild = true;
                  break;
                case IBV_WC_RESP_TIMEOUT_ERR:
                  if (epPtr->timeout_count.fetch_add(1, std::memory_order_relaxed) > 5)
                    needRebuild = true;
                  break;
                default:
                  break;
              }
              if (needRebuild && !epPtr->scheduled_rebuild.exchange(true)) {
                // Rebuild inline (could be offloaded to separate thread later)
                auto rebuilt = rdma->RebuildEndpoint(ep.local.handle.qpn);
                if (rebuilt.has_value()) {
                  MORI_IO_INFO("QP rebuild success old_qpn {} new_qpn {} gen {}",
                               ep.local.handle.qpn, (*rebuilt)->local.handle.qpn,
                               (*rebuilt)->rebuild_generation.load(std::memory_order_relaxed));
                  // Re-register new QP for event mode
                  if (config.pollCqMode == PollCqMode::EVENT) {
                    RegisterEndpointByQpn((*rebuilt)->local.handle.qpn);
                    UnregisterEndpointByQpn(ep.local.handle.qpn);
                  }
                } else {
                  MORI_IO_INFO("QP rebuild failed qpn {}", ep.local.handle.qpn);
                }
              }
            }
          }
        }
      }
      MORI_IO_TRACE(
          "NotifManager receive cqe for task {} code {} total batch size {} last batch size {} cur "
          "batch size {}",
          msg->meta->id, msg->meta->status->CodeUint32(), msg->meta->totalBatchSize, lastBatchSize,
          msg->batchSize);
      if ((lastBatchSize + msg->batchSize) == msg->meta->totalBatchSize) {
        free(msg->meta);
      }
      free(msg);
    }
  }
  if (rc < 0) {
    // ibv_poll_cq error path: record as fatal CQPollFailed
    Backend::ErrorRecord rec{};
    rec.code = Backend::ErrorCode::CQPollFailed;
    rec.severity = Backend::Severity::Fatal;
    rec.msg = "ibv_poll_cq failed";
    if (owner) owner->report_error(rec);
  }
}

void NotifManager::MainLoop() {
  if (config.pollCqMode == PollCqMode::EVENT) {
    constexpr int maxEvents = 128;
    epoll_event events[maxEvents];
    while (running.load()) {
      int nfds = epoll_wait(epfd, events, maxEvents, 0 /*ms*/);
      for (int i = 0; i < nfds; ++i) {
        uint32_t qpn = events[i].data.u32;

        std::optional<EpPairPtr> ep = rdma->GetEpPairPtrByQpn(qpn);
        if (!ep.has_value()) continue;

        struct ibv_comp_channel* ch = (*ep)->local.ibvHandle.compCh;

        struct ibv_cq* cq = nullptr;
        void* evCtx = nullptr;
        if (ibv_get_cq_event(ch, &cq, &evCtx)) continue;
        ibv_ack_cq_events(cq, 1);
        ibv_req_notify_cq(cq, 0);

        ProcessOneCqe(qpn, *(*ep));
      }
    }
  } else {
    while (running.load()) {
      rdma->EnumerateEndpoints([this](int qpn, const EpPair& ep) { this->ProcessOneCqe(qpn, ep); });
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

  rdma->ConnectEndpoint(ekey, devId, lep, msg.devId, msg.eph, topo, weight);
  notif->RegisterEndpointByQpn(lep.handle.qpn);
  notif->RegisterDevice(devId);
  ctx->CloseEndpoint(tcph);
}

void ControlPlaneServer::RegisterMemory(const MemoryDesc& desc) {
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
      p.WriteMessageRegEndpoint(MessageRegEndpoint{myEngKey, msg.topo, devId, lep.handle});
      rdma->ConnectEndpoint(msg.ekey, devId, lep, rdevId, msg.eph, msg.topo, weight);
      notif->RegisterEndpointByQpn(lep.handle.qpn);
      notif->RegisterDevice(devId);
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
}

/* ----------------------------------------------------------------------------------------------
 */
/*                                       RdmaBackendSession */
/* ----------------------------------------------------------------------------------------------
 */
RdmaBackendSession::RdmaBackendSession(const RdmaBackendConfig& config,
                                       const application::RdmaMemoryRegion& l,
                                       const application::RdmaMemoryRegion& r, const EpHandleVec& h,
                                       Executor* exec)
    : config(config), local(l), remote(r), handles(h), executor(exec) {}

void RdmaBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size,
                                   TransferStatus* status, TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  status->SetCode(StatusCode::IN_PROGRESS);
  CqCallbackMeta* callbackMeta = new CqCallbackMeta(status, id, 1);

  // snapshot current EpPairs for this op
  EpPairVec snapshot;
  snapshot.reserve(handles.size());
  for (auto& h : handles) snapshot.push_back(h->shared());
  RdmaOpRet ret = RdmaReadWrite(snapshot, local, localOffset, remote, remoteOffset, size,
                                callbackMeta, id, isRead);

  assert(!ret.Init());
  if (ret.Failed() || ret.Succeeded()) {
    status->SetCode(ret.code);
    status->SetMessage(ret.message);
  }
  if (!ret.Failed()) {
    RdmaNotifyTransfer(snapshot, status, id);
  }
}

void RdmaBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                        const SizeVec& sizes, TransferStatus* status,
                                        TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  status->SetCode(StatusCode::IN_PROGRESS);
  CqCallbackMeta* callbackMeta = new CqCallbackMeta(status, id, sizes.size());
  EpPairVec snapshot;
  snapshot.reserve(handles.size());
  for (auto& h : handles) snapshot.push_back(h->shared());
  RdmaOpRet ret;
  if (executor) {
    ExecutorReq req{snapshot,     local, localOffsets,         remote, remoteOffsets, sizes,
                    callbackMeta, id,    config.postBatchSize, isRead};
    ret = executor->RdmaBatchReadWrite(req);
  } else {
    ret = RdmaBatchReadWrite(snapshot, local, localOffsets, remote, remoteOffsets, sizes,
                             callbackMeta, id, isRead, config.postBatchSize);
  }
  assert(!ret.Init());
  if (ret.Failed() || ret.Succeeded()) {
    status->SetCode(ret.code);
    status->SetMessage(ret.message);
  }
  if (!ret.Failed()) {
    RdmaNotifyTransfer(snapshot, status, id);
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
  application::RdmaContext* ctx =
      new application::RdmaContext(application::RdmaBackendType::IBVerbs);
  rdma.reset(new mori::io::RdmaManager(beConfig, ctx));

  notif.reset(new NotifManager(rdma.get(), beConfig, this));
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

void RdmaBackend::RegisterMemory(const MemoryDesc& desc) { server->RegisterMemory(desc); }

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
  EpHandleVec handles = rdma->GetAllEndpointHandles(ekey, kp);
  assert(!handles.empty());
  if (handles.size() > static_cast<size_t>(config.qpPerTransfer)) {
    handles.resize(config.qpPerTransfer);
  }
  // TODO: we assume all selected endpoints are on same device
  EpPairPtr ep = handles[0]->shared();
  auto localMr = rdma->GetLocalMemory(ep->ldevId, local.id);
  if (!localMr.has_value()) {
    localMr = rdma->RegisterLocalMemory(ep->ldevId, local);
  }

  auto remoteMr = rdma->GetRemoteMemory(ekey, ep->rdevId, remote.id);
  if (!remoteMr.has_value()) {
    remoteMr = server->AskRemoteMemoryRegion(ekey, ep->rdevId, remote.id);
    // TODO: protocol should return status code
    // Currently we check member equality to ensure correct memory region
    assert(remoteMr->length == remote.size);
    rdma->RegisterRemoteMemory(ekey, ep->rdevId, remote.id, remoteMr.value());
  }

  sess = RdmaBackendSession(config, localMr.value(), remoteMr.value(), handles, executor.get());
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

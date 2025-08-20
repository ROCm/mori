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
#include "src/io/rdma/backend_impl_v1.hpp"

#include <sys/epoll.h>

#include <chrono>

#include "src/io/rdma/protocol.hpp"
namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaManager                                          */
/* ---------------------------------------------------------------------------------------------- */

RdmaManager::RdmaManager(application::RdmaContext* ctx) : ctx(ctx) {
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
        printf("gpu %d match nic %s\n", key.deviceId, nicName.c_str());
        return {{i, 1}};
      }
    }
  } else {
    assert("topo searching for device other than GPU is not implemented yet");
  }
}

/* ----------------------------------- Local Memory Management ---------------------------------- */
std::optional<application::RdmaMemoryRegion> RdmaManager::GetLocalMemory(int devId,
                                                                         MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{devId, id};
  if (mTable.find(key) == mTable.end()) return std::nullopt;
  return mTable[key];
}

application::RdmaMemoryRegion RdmaManager::RegisterLocalMemory(int devId, const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{devId, desc.id};
  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(devId);
  mTable[key] = devCtx->RegisterRdmaMemoryRegion(reinterpret_cast<void*>(desc.data), desc.size);
  return mTable[key];
}

void RdmaManager::DeregisterLocalMemory(int devId, const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
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
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{remRdmaDevId, id};
  RemoteEngineMeta& remote = remotes[ekey];
  if (remote.mTable.find(key) == remote.mTable.end()) {
    return std::nullopt;
  }
  return remote.mTable[key];
}

void RdmaManager::RegisterRemoteMemory(EngineKey ekey, int remRdmaDevId, MemoryUniqueId id,
                                       application::RdmaMemoryRegion mr) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{remRdmaDevId, id};
  RemoteEngineMeta& remote = remotes[ekey];
  remote.mTable[key] = mr;
}

void RdmaManager::DeregisterRemoteMemory(EngineKey ekey, int remRdmaDevId, MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(mu);
  RemoteEngineMeta& remote = remotes[ekey];
  MemoryKey key{remRdmaDevId, id};
  if (remote.mTable.find(key) != remote.mTable.end()) {
    remote.mTable.erase(key);
  }
}

/* ------------------------------------- Endpoint Management ------------------------------------ */
int RdmaManager::CountEndpoint(EngineKey engine, TopoKeyPair key) {
  std::lock_guard<std::mutex> lock(mu);
  return remotes[engine].rTable[key].size();
}

EpPairVec RdmaManager::GetAllEndpoint(EngineKey engine, TopoKeyPair key) {
  std::lock_guard<std::mutex> lock(mu);
  return remotes[engine].rTable[key];
}

application::RdmaEndpointConfig RdmaManager::GetRdmaEndpointConfig(int portId) {
  application::RdmaEndpointConfig config;
  config.portId = portId;
  config.gidIdx = 3;
  config.maxMsgsNum = 8192;
  config.maxMsgSge = 1;
  config.maxCqeNum = 8192;
  config.alignment = 4096;
  config.withCompChannel = true;
  config.enableSrq = true;
  return config;
}

application::RdmaEndpoint RdmaManager::CreateEndpoint(int devId) {
  std::lock_guard<std::mutex> lock(mu);

  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(devId);

  application::RdmaEndpoint rdmaEp =
      devCtx->CreateRdmaEndpoint(GetRdmaEndpointConfig(availDevices[devId].second));
  SYSCALL_RETURN_ZERO(ibv_req_notify_cq(rdmaEp.ibvHandle.cq, 0));
  return rdmaEp;
}

void RdmaManager::ConnectEndpoint(EngineKey remoteKey, int devId, application::RdmaEndpoint local,
                                  int rdevId, application::RdmaEndpointHandle remote,
                                  TopoKeyPair topoKey, int weight) {
  std::lock_guard<std::mutex> lock(mu);
  deviceCtxs[devId]->ConnectEndpoint(local.handle, remote);
  RemoteEngineMeta& meta = remotes[remoteKey];
  EpPair ep{weight, devId, rdevId, remoteKey, local, remote};
  meta.rTable[topoKey].push_back(ep);
  epsMap.insert({ep.local.handle.qpn, ep});
}

std::optional<EpPair> RdmaManager::GetEpPairByQpn(uint32_t qpn) {
  std::lock_guard<std::mutex> lock(mu);
  if (epsMap.find(qpn) == epsMap.end()) return std::nullopt;
  return epsMap[qpn];
}

application::RdmaDeviceContext* RdmaManager::GetRdmaDeviceContext(int devId) {
  std::lock_guard<std::mutex> lock(mu);
  return deviceCtxs[devId];
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
NotifManager::NotifManager(RdmaManager* rdmaMgr) : rdma(rdmaMgr) {}

NotifManager::~NotifManager() { Shutdown(); }

void NotifManager::RegisterEndpointByQpn(uint32_t qpn) {
  epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.u32 = qpn;
  std::optional<EpPair> ep = rdma->GetEpPairByQpn(qpn);
  assert(ep.has_value() && ep->local.ibvHandle.compCh);
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, ep->local.ibvHandle.compCh->fd, &ev));
}

void NotifManager::RegisterDevice(int devId) {
  std::lock_guard<std::mutex> lock(mu);
  if (notifCtx.find(devId) != notifCtx.end()) return;

  application::RdmaDeviceContext* devCtx = rdma->GetRdmaDeviceContext(devId);
  assert(devCtx);

  void* buf;
  SYSCALL_RETURN_ZERO(posix_memalign(reinterpret_cast<void**>(&buf), PAGESIZE,
                                     maxNotifNum * sizeof(TransferUniqueId)));
  application::RdmaMemoryRegion mr =
      devCtx->RegisterRdmaMemoryRegion(buf, maxNotifNum * sizeof(TransferUniqueId));
  struct ibv_srq* srq = devCtx->GetIbvSrq();
  assert(srq);
  notifCtx.insert({devId, {srq, mr}});

  // Pre post notification receive wr
  // TODO: should use min(maxNotifNum, maxSrqWrNum)
  for (uint64_t i = 0; i < maxNotifNum; i++) {
    struct ibv_sge sge{};
    sge.addr = mr.addr + i * sizeof(TransferUniqueId);
    sge.length = sizeof(TransferUniqueId);
    sge.lkey = mr.lkey;

    struct ibv_recv_wr wr{};
    wr.wr_id = i;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr* bad = nullptr;
    SYSCALL_RETURN_ZERO(ibv_post_srq_recv(srq, &wr, &bad));
  };
}

void NotifManager::MainLoop() {
  int maxEvents = 128;
  epoll_event events[maxEvents];
  while (running.load()) {
    int nfds = epoll_wait(epfd, events, maxEvents, 0 /*ms*/);
    for (int i = 0; i < nfds; ++i) {
      uint32_t qpn = events[i].data.u32;

      std::optional<EpPair> ep = rdma->GetEpPairByQpn(qpn);
      struct ibv_comp_channel* ch = ep->local.ibvHandle.compCh;

      struct ibv_cq* cq = nullptr;
      void* evCtx = nullptr;
      if (ibv_get_cq_event(ch, &cq, &evCtx)) continue;
      ibv_ack_cq_events(cq, 1);
      ibv_req_notify_cq(cq, 0);

      // TODO: maybe take multiple cqes?
      struct ibv_wc wc{};
      while (ibv_poll_cq(cq, 1, &wc) > 0) {
        if (wc.opcode == IBV_WC_RECV) {
          std::lock_guard<std::mutex> lock(mu);
          int devId = ep->ldevId;

          assert(notifCtx.find(devId) != notifCtx.end());
          DeviceNotifContext& ctx = notifCtx[devId];

          // FIXME: this notif mechenism has bug when notif index is wrapped around
          uint64_t idx = wc.wr_id;
          TransferUniqueId tid = reinterpret_cast<TransferUniqueId*>(ctx.mr.addr)[idx];
          // printf("recv notif for transfer %d\n", tid);

          EngineKey ekey = ep->remoteEngineKey;
          notifPool[ekey].insert(tid);

          // replenish recv wr
          // TODO(ditian12): we should replenish recv wr faster, insufficient recv wr is met
          // frequently when transfer is very fast. Two way to solve this, 1. use srq_limit to
          // replenish in advance
          // 2. independent srq entry config (now reuse maxMsgNum)
          struct ibv_sge sge{};
          sge.addr = ctx.mr.addr + idx * sizeof(TransferUniqueId);
          sge.length = sizeof(TransferUniqueId);
          sge.lkey = ctx.mr.lkey;

          struct ibv_recv_wr wr{};
          wr.wr_id = idx;
          wr.sg_list = &sge;
          wr.num_sge = 1;
          struct ibv_recv_wr* bad = nullptr;
          SYSCALL_RETURN_ZERO(ibv_post_srq_recv(ctx.srq, &wr, &bad));
        } else if (wc.opcode == IBV_WC_SEND) {
          uint64_t id = wc.wr_id;
          // printf("send notif for transfer %d\n", id);
        } else {
          // printf("data mov for transfer %lu %d\n", wc.wr_id, wc.opcode);
          TransferStatus* status = reinterpret_cast<TransferStatus*>(wc.wr_id);
          if (wc.status == IBV_WC_SUCCESS) {
            status->SetCode(StatusCode::SUCCESS);
          } else {
            status->SetCode(StatusCode::ERROR);
          }
          status->SetMessage(ibv_wc_status_str(wc.status));
          // printf("set transfer status\n");
        }
      }
    }
  }
}

void NotifManager::Start() {
  if (running.load()) return;
  epfd = epoll_create1(EPOLL_CLOEXEC);
  assert(epfd >= 0);
  running.store(true);
  thd = std::thread([this] { MainLoop(); });
}

void NotifManager::Shutdown() {
  running.store(false);
  if (thd.joinable()) thd.join();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Control Plane Server                                      */
/* ---------------------------------------------------------------------------------------------- */
ControlPlaneServer::ControlPlaneServer(std::string host, int port, RdmaManager* rdmaMgr,
                                       NotifManager* notifMgr) {
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
  p.WriteMessageRegEndpoint({ekey, topo, devId, lep.handle});
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
      p.WriteMessageRegEndpoint(MessageRegEndpoint{msg.ekey, msg.topo, devId, lep.handle});
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
  int maxEvents = 128;
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

/* ---------------------------------------------------------------------------------------------- */
/*                                         Rdma Utilities                                         */
/* ---------------------------------------------------------------------------------------------- */
namespace {

void RdmaNotifyTransfer(const application::RdmaEndpoint& ep, TransferStatus* status,
                        TransferUniqueId id) {
  struct ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(&id);
  sge.length = sizeof(TransferUniqueId);
  sge.lkey = 0;

  struct ibv_send_wr wr{};
  wr.wr_id = id;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_send_wr* bad_wr = nullptr;
  int ret = ibv_post_send(ep.ibvHandle.qp, &wr, &bad_wr);
  if (ret != 0) {
    status->SetCode(StatusCode::ERROR);
    status->SetMessage(strerror(errno));
  }
}

void RdmaBatchReadWrite(const application::RdmaEndpoint& ep,
                        const application::RdmaMemoryRegion& local, const SizeVec& localOffsets,
                        const application::RdmaMemoryRegion& remote, const SizeVec& remoteOffsets,
                        const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                        bool isRead) {
  assert(localOffsets.size() == remoteOffsets.size());
  assert(sizes.size() == remoteOffsets.size());
  size_t batchSize = sizes.size();
  if (batchSize == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  std::vector<struct ibv_sge> sges(batchSize, ibv_sge{});
  std::vector<struct ibv_send_wr> wrs(batchSize, ibv_send_wr{});
  for (int i = 0; i < batchSize; i++) {
    struct ibv_sge& sge = sges[i];
    sge.addr = reinterpret_cast<uint64_t>(local.addr) + localOffsets[i];
    sge.length = sizes[i];
    sge.lkey = local.lkey;

    struct ibv_send_wr& wr = wrs[i];
    wr.wr_id = (i < (batchSize - 1)) ? 0 : reinterpret_cast<uint64_t>(status);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = isRead ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
    wr.send_flags = (i < (batchSize - 1)) ? 0 : IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remote.addr) + remoteOffsets[i];
    wr.wr.rdma.rkey = remote.rkey;
    wr.next = (i < (batchSize - 1)) ? wrs.data() + i + 1 : nullptr;
  }

  int ret = ibv_post_send(ep.ibvHandle.qp, wrs.data(), nullptr);
  if (ret != 0) {
    status->SetCode(StatusCode::ERROR);
    status->SetMessage(strerror(errno));
  }
}

void RdmaBatchRead(const application::RdmaEndpoint& ep, const application::RdmaMemoryRegion& local,
                   const SizeVec& localOffsets, const application::RdmaMemoryRegion& remote,
                   const SizeVec& remoteOffsets, const SizeVec& sizes, TransferStatus* status,
                   TransferUniqueId id) {
  RdmaBatchReadWrite(ep, local, localOffsets, remote, remoteOffsets, sizes, status, id,
                     true /*isRead */);
}

void RdmaBatchWrite(const application::RdmaEndpoint& ep, const application::RdmaMemoryRegion& local,
                    const SizeVec& localOffsets, const application::RdmaMemoryRegion& remote,
                    const SizeVec& remoteOffsets, const SizeVec& sizes, TransferStatus* status,
                    TransferUniqueId id) {
  RdmaBatchReadWrite(ep, local, localOffsets, remote, remoteOffsets, sizes, status, id,
                     false /*isRead */);
}

void RdmaRead(const application::RdmaEndpoint& ep, const application::RdmaMemoryRegion& local,
              size_t localOffset, const application::RdmaMemoryRegion& remote, size_t remoteOffset,
              size_t size, TransferStatus* status, TransferUniqueId id) {
  RdmaBatchRead(ep, local, {localOffset}, remote, {remoteOffset}, {size}, status, id);
}

void RdmaWrite(const application::RdmaEndpoint& ep, const application::RdmaMemoryRegion& local,
               size_t localOffset, const application::RdmaMemoryRegion& remote, size_t remoteOffset,
               size_t size, TransferStatus* status, TransferUniqueId id) {
  RdmaBatchWrite(ep, local, {localOffset}, remote, {remoteOffset}, {size}, status, id);
}

};  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                       RdmaBackendSession                                       */
/* ---------------------------------------------------------------------------------------------- */
RdmaBackendSession::RdmaBackendSession(const application::RdmaMemoryRegion& l,
                                       const application::RdmaMemoryRegion& r, const EpPair& e)
    : local(l), remote(r), eps(e) {}

void RdmaBackendSession::Read(size_t localOffset, size_t remoteOffset, size_t size,
                              TransferStatus* status, TransferUniqueId id) {
  RdmaRead(eps.local, local, localOffset, remote, remoteOffset, size, status, id);
  RdmaNotifyTransfer(eps.local, status, id);
}

void RdmaBackendSession::Write(size_t localOffset, size_t remoteOffset, size_t size,
                               TransferStatus* status, TransferUniqueId id) {
  RdmaWrite(eps.local, local, localOffset, remote, remoteOffset, size, status, id);
  RdmaNotifyTransfer(eps.local, status, id);
}

void RdmaBackendSession::BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                   const SizeVec& sizes, TransferStatus* status,
                                   TransferUniqueId id) {
  RdmaBatchRead(eps.local, local, localOffsets, remote, remoteOffsets, sizes, status, id);
  RdmaNotifyTransfer(eps.local, status, id);
}

bool RdmaBackendSession::Alive() const { return true; }
/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

RdmaBackend::RdmaBackend(EngineKey key, IOEngineConfig config) {
  application::RdmaContext* ctx =
      new application::RdmaContext(application::RdmaBackendType::IBVerbs);
  rdma.reset(new mori::io::RdmaManager(ctx));

  notif.reset(new NotifManager(rdma.get()));
  notif->Start();

  server.reset(new ControlPlaneServer(config.host, config.port, rdma.get(), notif.get()));
  server->Start();
}

RdmaBackend::~RdmaBackend() {
  notif->Shutdown();
  server->Shutdown();
}

void RdmaBackend::RegisterRemoteEngine(const EngineDesc& rdesc) {
  server->RegisterRemoteEngine(rdesc);
}

void RdmaBackend::DeregisterRemoteEngine(const EngineDesc& rdesc) {
  server->DeregisterRemoteEngine(rdesc);
}

void RdmaBackend::RegisterMemory(const MemoryDesc& desc) { server->RegisterMemory(desc); }

void RdmaBackend::DeregisterMemory(const MemoryDesc& desc) { server->DeregisterMemory(desc); }

void RdmaBackend::Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                       size_t remoteOffset, size_t size, TransferStatus* status,
                       TransferUniqueId id) {
  RdmaBackendSession sess;
  CreateSession(localDest, remoteSrc, sess);
  return sess.Read(localOffset, remoteOffset, size, status, id);
}

void RdmaBackend::Write(const MemoryDesc& localSrc, size_t localOffset,
                        const MemoryDesc& remoteDest, size_t remoteOffset, size_t size,
                        TransferStatus* status, TransferUniqueId id) {
  RdmaBackendSession sess;
  CreateSession(localSrc, remoteDest, sess);
  return sess.Write(localOffset, remoteOffset, size, status, id);
}

void RdmaBackend::BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                            const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                            const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) {
  assert(localOffsets.size() == remoteOffsets.size());
  assert(sizes.size() == remoteOffsets.size());
  size_t batchSize = sizes.size();
  if (batchSize == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  RdmaBackendSession sess;
  CreateSession(localDest, remoteSrc, sess);
  return sess.BatchRead(localOffsets, remoteOffsets, sizes, status, id);
}

BackendSession* RdmaBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  RdmaBackendSession* sess = new RdmaBackendSession();
  CreateSession(local, remote, *sess);
  sessions.emplace_back(sess);
  return sess;
}

void RdmaBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote,
                                RdmaBackendSession& sess) {
  TopoKey localKey{local.deviceId, local.loc};
  TopoKey remoteKey{remote.deviceId, remote.loc};
  TopoKeyPair kp{localKey, remoteKey};

  EngineKey ekey = remote.engineKey;

  // Create a pair of endpoint if none
  if (rdma->CountEndpoint(ekey, kp) == 0) {
    server->BuildRdmaConn(ekey, kp);
  }
  EpPairVec eps = rdma->GetAllEndpoint(ekey, kp);
  assert(!eps.empty());

  EpPair ep = eps[0];
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

  sess = RdmaBackendSession(localMr.value(), remoteMr.value(), ep);
}

bool RdmaBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                           TransferStatus* status) {
  status->SetCode(StatusCode::SUCCESS);
  return true;
}

}  // namespace io
}  // namespace mori

#include "src/io/rdma/backend_impl_v1.hpp"

#include <sys/epoll.h>

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
}

std::vector<std::pair<int, int>> RdmaManager::Search(TopoKey key) { return {{0, 999}}; }

/* ----------------------------------- Local Memory Management ---------------------------------- */
std::optional<application::RdmaMemoryRegion> RdmaManager::GetLocalMemory(int devId,
                                                                         MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{devId, id};
  if (mTable.find(key) == mTable.end()) return std::nullopt;
  return mTable[key];
}

application::RdmaMemoryRegion RdmaManager::RegisterLocalMemory(int devId, MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{devId, desc.id};
  application::RdmaDeviceContext* devCtx = GetOrCreateDeviceContext(devId);
  mTable[key] = devCtx->RegisterRdmaMemoryRegion(desc.data, desc.size);
  return mTable[key];
}

void RdmaManager::DeregisterLocalMemory(int devId, MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{devId, desc.id};
  if (mTable.find(key) != mTable.end()) {
    deviceCtxs[devId]->DeregisterRdmaMemoryRegion(desc.data);
    mTable.erase(key);
  }
}

/* ---------------------------------- Remote Memory Management ---------------------------------- */
std::optional<application::RdmaMemoryRegion> RdmaManager::GetRemoteMemory(EngineKey ekey, int devId,
                                                                          MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{devId, id};
  RemoteEngineMeta remote = remotes[ekey];
  if (remote.mTable.find(key) == remote.mTable.end()) return std::nullopt;
  return remote.mTable[key];
}

void RdmaManager::RegisterRemoteMemory(EngineKey ekey, int devId, MemoryUniqueId id,
                                       application::RdmaMemoryRegion mr) {
  std::lock_guard<std::mutex> lock(mu);
  MemoryKey key{devId, id};
  RemoteEngineMeta remote = remotes[ekey];
  remote.mTable[key] = mr;
}

void RdmaManager::DeregisterRemoteMemory(EngineKey ekey, int devId, MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(mu);
  RemoteEngineMeta remote = remotes[ekey];
  MemoryKey key{devId, id};
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
  config.maxMsgsNum = 1024;
  config.maxMsgSge = 1;
  config.maxCqeNum = 1024;
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
    // devCtx->CreateRdmaSrqIfNx(GetRdmaEndpointConfig(availDevices[devId].second));
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
  ibv_srq* srq = devCtx->GetIbvSrq();
  assert(srq);
  notifCtx.insert({devId, {srq, mr}});

  // Pre post notification receive wr
  for (uint64_t i = 0; i < maxNotifNum; i++) {
    ibv_sge sge{};
    sge.addr = mr.addr + i * sizeof(TransferUniqueId);
    sge.length = sizeof(TransferUniqueId);
    sge.lkey = mr.lkey;

    ibv_recv_wr wr{};
    wr.wr_id = i;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    ibv_recv_wr* bad = nullptr;
    SYSCALL_RETURN_ZERO(ibv_post_srq_recv(srq, &wr, &bad));
  };
}

void NotifManager::MainLoop() {
  int maxEvents = 128;
  epoll_event events[maxEvents];
  while (running.load()) {
    int nfds = epoll_wait(epfd, events, maxEvents, 5 /*ms*/);
    for (int i = 0; i < nfds; ++i) {
      uint32_t qpn = events[i].data.u32;

      std::optional<EpPair> ep = rdma->GetEpPairByQpn(qpn);
      ibv_comp_channel* ch = ep->local.ibvHandle.compCh;

      ibv_cq* cq;
      void* evCtx;
      if (ibv_get_cq_event(ch, &cq, &evCtx)) continue;
      ibv_ack_cq_events(cq, 1);
      ibv_req_notify_cq(cq, 0);

      // TODO: maybe take multiple cqes?
      ibv_wc wc;
      while (ibv_poll_cq(cq, 1, &wc) > 0) {
        if (wc.opcode == IBV_WC_RECV) {
          std::lock_guard<std::mutex> lock(mu);
          int devId = ep->ldevId;

          assert(notifCtx.find(devId) != notifCtx.end());
          DeviceNotifContext& ctx = notifCtx[devId];

          uint64_t idx = wc.wr_id;
          TransferUniqueId tid = reinterpret_cast<TransferUniqueId*>(ctx.mr.addr)[idx];
          printf("recv notif for transfer %d\n", tid);

          EngineKey ekey = ep->remoteEngineKey;
          notifPool[ekey].insert(tid);

          // replenish recv wr
          ibv_sge sge;
          sge.addr = ctx.mr.addr + idx * sizeof(TransferUniqueId);
          sge.length = sizeof(TransferUniqueId);
          sge.lkey = ctx.mr.lkey;

          ibv_recv_wr wr;
          wr.wr_id = idx;
          wr.sg_list = &sge;
          wr.num_sge = 1;
          ibv_recv_wr* bad;
          SYSCALL_RETURN_ZERO(ibv_post_srq_recv(ctx.srq, &wr, &bad));
        } else if (wc.opcode == IBV_WC_SEND) {
          uint64_t id = wc.wr_id;
          printf("send notif for transfer %d\n", id);
        } else {
          printf("data mov for transfer %lu %d\n", wc.wr_id, wc.opcode);
          TransferStatus* status = reinterpret_cast<TransferStatus*>(wc.wr_id);
          if (wc.status == IBV_WC_SUCCESS) {
            status->SetCode(StatusCode::SUCCESS);
          } else {
            status->SetCode(StatusCode::ERROR);
          }
          status->SetMessage(ibv_wc_status_str(wc.status));
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

void RdmaBackend::RegisterMemory(MemoryDesc& desc) { server->RegisterMemory(desc); }

void RdmaBackend::DeregisterMemory(MemoryDesc& desc) { server->DeregisterMemory(desc); }

void RdmaBackend::Read(MemoryDesc localDest, size_t localOffset, MemoryDesc remoteSrc,
                       size_t remoteOffset, size_t size, TransferStatus* status,
                       TransferUniqueId id) {
  TopoKey local{localDest.deviceId, localDest.loc};
  TopoKey remote{remoteSrc.deviceId, remoteSrc.loc};
  TopoKeyPair kp{local, remote};

  EngineKey ekey = remoteSrc.engineKey;

  // Create a pair of endpoint if none
  if (rdma->CountEndpoint(ekey, kp) == 0) server->BuildRdmaConn(ekey, kp);
  EpPairVec eps = rdma->GetAllEndpoint(ekey, kp);
  assert(!eps.empty());

  //
  EpPair ep = eps[0];
  auto localMr = rdma->GetLocalMemory(ep.ldevId, localDest.id);
  if (!localMr.has_value()) {
    localMr = rdma->RegisterLocalMemory(ep.ldevId, localDest);
  }

  auto remoteMr = rdma->GetRemoteMemory(ekey, ep.rdevId, remoteSrc.id);
  if (!remoteMr.has_value()) {
    remoteMr = server->AskRemoteMemoryRegion(ekey, ep.rdevId, remoteSrc.id);
    // TODO: protocol should return status code
    // Currently we check member equality to ensure correct memory region
    assert(remoteMr->length == remoteSrc.size);
  }

  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(localDest.data) + localOffset;
  sge.length = size;
  sge.lkey = localMr->lkey;

  ibv_send_wr wr{};
  ibv_send_wr* bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(status);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remoteSrc.data) + remoteOffset;
  wr.wr.rdma.rkey = remoteMr->rkey;

  int ret = ibv_post_send(ep.local.ibvHandle.qp, &wr, &bad_wr);
  if (ret != 0) {
    status->SetCode(StatusCode::ERROR);
    status->SetMessage(strerror(errno));
  }

  RdmaNotifyTransfer(ep.local, status, id);
}

void RdmaBackend::Write(MemoryDesc localSrc, size_t localOffset, MemoryDesc remoteDest,
                        size_t remoteOffset, size_t size, TransferStatus* status,
                        TransferUniqueId id) {
  status->SetCode(StatusCode::SUCCESS);
}

bool RdmaBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                           TransferStatus* status) {
  status->SetCode(StatusCode::SUCCESS);
  return true;
}

void RdmaBackend::RdmaNotifyTransfer(const application::RdmaEndpoint& ep, TransferStatus* status,
                                     TransferUniqueId id) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(&id);
  sge.length = sizeof(TransferUniqueId);
  sge.lkey = 0;

  ibv_send_wr wr{};
  wr.wr_id = id;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  ibv_send_wr* bad_wr = nullptr;
  int ret = ibv_post_send(ep.ibvHandle.qp, &wr, &bad_wr);
  if (ret != 0) {
    status->SetCode(StatusCode::ERROR);
    status->SetMessage(strerror(errno));
  }
}

}  // namespace io
}  // namespace mori
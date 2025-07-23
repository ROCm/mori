#include "mori/io/engine.hpp"

#include <fcntl.h>
#include <infiniband/verbs.h>
#include <sys/epoll.h>

#include "mori/application/utils/check.hpp"
#include "mori/application/utils/serdes.hpp"
#include "mori/io/meta_data.hpp"
#include "mori/io/protocol.hpp"

namespace mori {
namespace io {

IOEngine::IOEngine(EngineKey key, IOEngineConfig config) : config(config) {
  // Initialize descriptor
  desc.key = key;
  desc.gpuId = config.gpuId;
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  desc.hostname = std::string(hostname);
  desc.backends = BackendBitmap(config.backends);

  // Initialize control plane
  tcpContext.reset(new application::TCPContext(config.host, config.port));
  StartControlPlane();
  desc.tcpHandle = tcpContext->handle;

  // Initialize data plane
  StartDataPlane();
}

IOEngine::~IOEngine() {
  ShutdownControlPlane();
  ShutdownDataPlane();
}

EngineDesc IOEngine::GetEngineDesc() { return desc; }

application::RdmaEndpointConfig IOEngine::GetRdmaEndpointConfig() {
  application::RdmaEndpointConfig config;
  config.portId = devicePort.second;
  config.gidIdx = 1;
  config.maxMsgsNum = 1024;
  config.maxMsgSge = 1;
  config.maxCqeNum = 1024;
  config.alignment = 4096;
  config.withCompChannel = true;
  config.enableSrq = true;
  return config;
}

application::RdmaEndpoint IOEngine::CreateRdmaEndpoint() {
  application::RdmaEndpoint rdmaEp = rdmaDeviceContext->CreateRdmaEndpoint(GetRdmaEndpointConfig());
  // Register notification
  SYSCALL_RETURN_ZERO(ibv_req_notify_cq(rdmaEp.ibvHandle.cq, 0));
  // Add to epoll list
  epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.u32 = rdmaEp.handle.qpn;
  assert(rdmaEp.ibvHandle.compCh);
  SYSCALL_RETURN_ZERO(
      epoll_ctl(rdmaCompChEpollFd, EPOLL_CTL_ADD, rdmaEp.ibvHandle.compCh->fd, &ev));
  return rdmaEp;
}

void IOEngine::RegisterRemoteEngine(EngineDesc remote) {
  if (engineKV.find(remote.key) != engineKV.end()) return;
  application::TCPEndpointHandle tcpEph =
      tcpContext->Connect(remote.tcpHandle.host, remote.tcpHandle.port);

  BackendBitmap commonBes = desc.backends.FindCommonBackends(remote.backends);
  if (commonBes.IsAvailableBackend(BackendType::RDMA)) {
    Protocol protocol(tcpEph);
    application::RdmaEndpoint localRdmaEp = CreateRdmaEndpoint();
    protocol.WriteMessageRegEngine(MessageRegEngine{GetEngineDesc(), localRdmaEp.handle});
    MessageHeader hdr = protocol.ReadMessageHeader();
    assert(hdr.type == MessageType::RegEngine);
    MessageRegEngine msg = protocol.ReadMessageRegEngine(hdr.len);
    rdmaDeviceContext->ConnectEndpoint(localRdmaEp.handle, msg.rdmaEph);

    engineKV.insert({remote.key, remote});
    rdmaEpKV.insert({remote.key, {}});
    rdmaEpKV[remote.key].push_back({localRdmaEp, msg.rdmaEph});
    qpn2EngineKV.insert({localRdmaEp.handle.qpn, {remote.key, {localRdmaEp, msg.rdmaEph}}});
    trsfUidNotifMaps.insert({remote.key, std::make_unique<TransferUidNotifMap>()});
  }

  tcpContext->CloseEndpoint(tcpEph);
}

void IOEngine::DeRegisterRemoteEngine(EngineDesc remote) {
  engineKV.erase(remote.key);
  rdmaEpKV.erase(remote.key);
  trsfUidNotifMaps.erase(remote.key);
}

MemoryDesc IOEngine::RegisterMemory(void* data, size_t length, int deviceId,
                                    MemoryLocationType loc) {
  MemoryDesc memDesc;
  memDesc.engineKey = desc.key;
  memDesc.id = nextMemUid.fetch_add(1, std::memory_order_relaxed);
  memDesc.deviceId = deviceId;
  memDesc.data = data;
  memDesc.length = length;
  memDesc.loc = loc;
  memPool.insert({memDesc.id, memDesc});

  if (desc.backends.IsAvailableBackend(BackendType::RDMA)) {
    application::RdmaMemoryRegion rdmaMr =
        rdmaDeviceContext->RegisterRdmaMemoryRegion(data, length);
    memDesc.backendDesc.rdmaMr = rdmaMr;
  }
  return memDesc;
}

void IOEngine::DeRegisterMemory(const MemoryDesc& desc) {
  memPool.erase(desc.id);
  if (GetEngineDesc().backends.IsAvailableBackend(BackendType::RDMA)) {
    rdmaDeviceContext->DeRegisterRdmaMemoryRegion(desc.data);
  }
}

TransferUniqueId IOEngine::AllocateTransferUniqueId() {
  return nextTransferUid.fetch_add(1, std::memory_order_relaxed);
}

void IOEngine::Read(MemoryDesc localDest, size_t localOffset, MemoryDesc remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id) {
  assert(GetEngineDesc().backends.IsAvailableBackend(BackendType::RDMA) && "not implemented yet");

  assert((engineKV.find(remoteSrc.engineKey) != engineKV.end()) && "register remote engine first");
  assert((memPool.find(localDest.id) != memPool.end()) && "register local memory first");

  if (rdmaEpKV.find(remoteSrc.engineKey) == rdmaEpKV.end()) {
    // TODO make connection
    assert(false && "lazy connection built up has not yet implemented");
  }

  // TODO: add selection logics when qp pool feature is ready
  application::RdmaEndpoint ep = rdmaEpKV[remoteSrc.engineKey][0].first;

  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(localDest.data) + localOffset;
  sge.length = size;
  sge.lkey = localDest.backendDesc.rdmaMr.lkey;

  ibv_send_wr wr{};
  ibv_send_wr* bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(status);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remoteSrc.data) + remoteOffset;
  wr.wr.rdma.rkey = remoteSrc.backendDesc.rdmaMr.rkey;

  int ret = ibv_post_send(ep.ibvHandle.qp, &wr, &bad_wr);
  if (ret != 0) {
    status->SetCode(StatusCode::ERROR);
    status->SetMessage(strerror(errno));
  }

  RdmaNotifyTransfer(ep, status, id);
}

void IOEngine::Write(MemoryDesc localSrc, size_t localOffset, MemoryDesc remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) {
  assert(GetEngineDesc().backends.IsAvailableBackend(BackendType::RDMA) && "not implemented yet");
}

void IOEngine::QueryAndAckInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                                TransferStatus* status) {
  assert(GetEngineDesc().backends.IsAvailableBackend(BackendType::RDMA) && "not implemented yet");

  assert(trsfUidNotifMaps.find(remote) != trsfUidNotifMaps.end());
  TransferUidNotifMap* notifMap = trsfUidNotifMaps[remote].get();
  {
    std::lock_guard<std::mutex> lock(notifMap->mu);
    if (notifMap->map.find(id) == notifMap->map.end()) return;
    notifMap->map.erase(id);
    status->SetCode(StatusCode::SUCCESS);
  }
}

void IOEngine::RdmaNotifyTransfer(const application::RdmaEndpoint& ep, TransferStatus* status,
                                  TransferUniqueId id) {
  ibv_sge sg = {
      .addr = reinterpret_cast<uintptr_t>(&id), .length = sizeof(TransferUniqueId), .lkey = 0};
  struct ibv_send_wr wr = {.wr_id = id,
                           .opcode = IBV_WR_SEND,
                           .send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED,
                           .sg_list = &sg,
                           .num_sge = 1};
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(ep.ibvHandle.qp, &wr, &bad_wr);
  if (ret != 0) {
    status->SetCode(StatusCode::ERROR);
    status->SetMessage(strerror(errno));
  }
  return;
}

void IOEngine::StartControlPlane() {
  if (running.load()) return;

  tcpContext->Listen();
  // Create epoll fd
  epollFd = epoll_create1(EPOLL_CLOEXEC);
  assert(epollFd >= 0);

  // Add TCP listen fd
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ev.data.fd = tcpContext->GetListenFd();
  SYSCALL_RETURN_ZERO(epoll_ctl(epollFd, EPOLL_CTL_ADD, tcpContext->GetListenFd(), &ev));

  running.store(true);
  ctrlPlaneThd = std::thread([this] { ControlPlaneLoop(); });
}

void IOEngine::ShutdownControlPlane() {
  running.store(false);
  if (ctrlPlaneThd.joinable()) ctrlPlaneThd.join();
}

void IOEngine::AcceptRemoteEngineConn() {
  application::TCPEndpointHandleVec eps = tcpContext->Accept();
  for (auto& ep : eps) {
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = ep.fd;
    SYSCALL_RETURN_ZERO(epoll_ctl(epollFd, EPOLL_CTL_ADD, ep.fd, &ev));
    tcpEpKV.insert({ep.fd, ep});
  }
}

void IOEngine::HandleControlPlaneProtocol(int fd) {
  assert(tcpEpKV.find(fd) != tcpEpKV.end());
  application::TCPEndpointHandle eph = tcpEpKV[fd];

  Protocol protocol(eph);
  MessageHeader hdr = protocol.ReadMessageHeader();

  switch (hdr.type) {
    case MessageType::RegEngine: {
      MessageRegEngine msg = protocol.ReadMessageRegEngine(hdr.len);

      BackendBitmap commonBes = desc.backends.FindCommonBackends(msg.engineDesc.backends);
      if (commonBes.IsAvailableBackend(BackendType::RDMA)) {
        application::RdmaEndpoint localRdmaEp = CreateRdmaEndpoint();
        protocol.WriteMessageRegEngine(MessageRegEngine{GetEngineDesc(), localRdmaEp.handle});
        rdmaDeviceContext->ConnectEndpoint(localRdmaEp.handle, msg.rdmaEph);

        engineKV.insert({msg.engineDesc.key, msg.engineDesc});
        if (rdmaEpKV.find(msg.engineDesc.key) == rdmaEpKV.end())
          rdmaEpKV.insert({msg.engineDesc.key, {}});
        rdmaEpKV[msg.engineDesc.key].push_back({localRdmaEp, msg.rdmaEph});
        qpn2EngineKV.insert(
            {localRdmaEp.handle.qpn, {msg.engineDesc.key, {localRdmaEp, msg.rdmaEph}}});
        trsfUidNotifMaps.insert({msg.engineDesc.key, std::make_unique<TransferUidNotifMap>()});

        SYSCALL_RETURN_ZERO(epoll_ctl(epollFd, EPOLL_CTL_DEL, fd, NULL));
      }
      tcpContext->CloseEndpoint(eph);
      tcpEpKV.erase(fd);
      break;
    }
    default:
      assert(false && "not implemented");
  }
}

void IOEngine::ControlPlaneLoop() {
  int maxEvents = 128;
  epoll_event events[maxEvents];
  while (running.load()) {
    int nfds = epoll_wait(epollFd, events, maxEvents, 5 /*ms*/);

    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;

      // Add new endpoints into epoll list
      if (fd == tcpContext->GetListenFd()) {
        AcceptRemoteEngineConn();
        continue;
      }

      HandleControlPlaneProtocol(fd);
    }
  }
}

void IOEngine::RdmaPollLoop() {
  int maxEvents = 128;
  epoll_event events[maxEvents];
  while (running.load()) {
    int nfds = epoll_wait(rdmaCompChEpollFd, events, maxEvents, 5 /*ms*/);
    for (int i = 0; i < nfds; ++i) {
      uint32_t qpn = events[i].data.u32;

      assert(qpn2EngineKV.find(qpn) != qpn2EngineKV.end());
      std::pair<EngineKey, RdmaEpPair> engineEpPair = qpn2EngineKV[qpn];
      EngineKey engineKey = engineEpPair.first;
      ibv_comp_channel* ch = engineEpPair.second.first.ibvHandle.compCh;

      ibv_cq* cq;
      void* evCtx;
      if (ibv_get_cq_event(ch, &cq, &evCtx)) continue;
      ibv_ack_cq_events(cq, 1);
      ibv_req_notify_cq(cq, 0);

      // TODO: maybe take multiple cqes?
      ibv_wc wc;
      while (ibv_poll_cq(cq, 1, &wc) > 0) {
        // Recv is only used for notification of single-sided op (READ/WRITE)
        if (wc.opcode == IBV_WC_RECV) {
          uint64_t uidBufIdx = wc.wr_id;
          TransferUniqueId remoteTrsfId = rdmaTrsfUidBuf[uidBufIdx];
          printf("recv notif for transfer %d\n", remoteTrsfId);

          assert(trsfUidNotifMaps.find(engineKey) != trsfUidNotifMaps.end());
          TransferUidNotifMap* notifMap = trsfUidNotifMaps[engineKey].get();
          {
            std::lock_guard<std::mutex> lock(notifMap->mu);
            notifMap->map.insert(remoteTrsfId);
          }

          // replenish recv wr
          ibv_sge sge = {.addr = reinterpret_cast<uintptr_t>(rdmaTrsfUidBuf + uidBufIdx),
                         .length = sizeof(TransferUniqueId),
                         .lkey = rdmaTrsfUidMr.lkey};
          ibv_recv_wr wr = {.wr_id = uidBufIdx, .sg_list = &sge, .num_sge = 1};
          ibv_recv_wr* bad;
          SYSCALL_RETURN_ZERO(ibv_post_srq_recv(rdmaDeviceContext->GetIbvSrq(), &wr, &bad));
        } else if (wc.opcode == IBV_WC_SEND) {
          uint64_t id = wc.wr_id;
          printf("send notif for transfer %d\n", id);
        } else {
          printf("data mov for transfer %d %d\n", wc.wr_id, wc.opcode);
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

void IOEngine::StartDataPlane() {
  if (desc.backends.IsAvailableBackend(BackendType::RDMA)) {
    // TODO: add topology detection
    rdmaContext.reset(new application::RdmaContext(application::RdmaBackendType::IBVerbs));
    application::RdmaDeviceList devices = rdmaContext->GetRdmaDeviceList();
    application::ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(devices);
    assert(activeDevicePortList.size() > 0);
    devicePort = activeDevicePortList[desc.gpuId % activeDevicePortList.size()];
    application::RdmaDevice* device = devicePort.first;
    rdmaDeviceContext = device->CreateRdmaDeviceContext();
    ibv_srq* srq = rdmaDeviceContext->CreateRdmaSrqIfNx(GetRdmaEndpointConfig());

    // Start RDMA poll thread
    // Create epoll fd
    rdmaCompChEpollFd = epoll_create1(EPOLL_CLOEXEC);
    assert(rdmaCompChEpollFd >= 0);
    rdmaPollThd = std::thread([this] { RdmaPollLoop(); });

    // Allocate notification buffer
    SYSCALL_RETURN_ZERO(posix_memalign(reinterpret_cast<void**>(&rdmaTrsfUidBuf), PAGESIZE,
                                       rdmaTrsfUidNum * sizeof(TransferUniqueId)));
    rdmaTrsfUidMr = rdmaDeviceContext->RegisterRdmaMemoryRegion(
        rdmaTrsfUidBuf, rdmaTrsfUidNum * sizeof(TransferUniqueId));
    // Pre post notification receive wr
    for (uint64_t i = 0; i < rdmaTrsfUidNum; i++) {
      ibv_sge sge = {.addr = reinterpret_cast<uintptr_t>(rdmaTrsfUidBuf + i),
                     .length = sizeof(TransferUniqueId),
                     .lkey = rdmaTrsfUidMr.lkey};
      ibv_recv_wr wr = {.wr_id = i, .sg_list = &sge, .num_sge = 1};
      ibv_recv_wr* bad;
      SYSCALL_RETURN_ZERO(ibv_post_srq_recv(srq, &wr, &bad));
    }
  }

  if (desc.backends.IsAvailableBackend(BackendType::XGMI)) {
    assert(false && "not implemented");
  }

  if (desc.backends.IsAvailableBackend(BackendType::TCP)) {
    assert(false && "not implemented");
  }
}

void IOEngine::ShutdownDataPlane() {
  running.store(false);
  if (rdmaPollThd.joinable()) rdmaPollThd.join();
  free(rdmaTrsfUidBuf);
}

}  // namespace io
}  // namespace mori
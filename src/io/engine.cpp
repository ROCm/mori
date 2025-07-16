#include "mori/io/engine.hpp"

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
  InitDataPlane();
}

IOEngine::~IOEngine() { ShutdownControlPlane(); }

EngineDesc IOEngine::GetEngineDesc() {
  printf("before return host %s key %s %d\n", desc.tcpHandle.host.c_str(), desc.key.c_str(),
         desc.backends.bits);
  return desc;
}

application::RdmaEndpoint IOEngine::CreateRdmaEndpoint() {
  application::RdmaEndpointConfig config;
  config.portId = devicePort.second;
  config.gidIdx = 1;
  config.maxMsgsNum = 1024;
  config.maxCqeNum = 1024;
  config.alignment = 4096;
  application::RdmaEndpoint rdmaEp = rdmaDeviceContext->CreateRdmaEndpoint(config);
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
  }

  tcpContext->CloseEndpoint(tcpEph);
}

void IOEngine::DeRegisterRemoteEngine(EngineDesc remote) {
  engineKV.erase(remote.key);
  rdmaEpKV.erase(remote.key);
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
  rdmaDeviceContext->DeRegisterRdmaMemoryRegion(desc.data);
}

void IOEngine::Read(MemoryDesc local, size_t localOffset, MemoryDesc remote, size_t remoteOffset,
                    size_t size) {
  assert((engineKV.find(remote.engineKey) != engineKV.end()) && "register remote engine first");
  assert((memPool.find(local.id) != memPool.end()) && "register local memory first");

  if (rdmaEpKV.find(remote.engineKey) == rdmaEpKV.end()) {
    // TODO make connection
    assert(false && "lazy connection built up has not yet implemented");
  }

  // TODO: add selection logics when qp pool feature is ready
  application::RdmaEndpoint ep = rdmaEpKV[remote.engineKey][0].first;

  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(local.data) + localOffset;
  sge.length = size;
  sge.lkey = local.backendDesc.rdmaMr.lkey;

  ibv_send_wr wr{};
  ibv_send_wr* bad_wr = nullptr;
  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remote.data) + remoteOffset;
  wr.wr.rdma.rkey = remote.backendDesc.rdmaMr.lkey;

  assert(!ibv_post_send(ep.ibvHandle.qp, &wr, &bad_wr) && "ibv_post_send RDMA READ");
  ibv_wc wc{};
  while (ibv_poll_cq(ep.ibvHandle.cq, 1, &wc) == 0) {
    usleep(1000);
  }
  std::string errStr = ibv_wc_status_str(wc.status);
  printf("%s\n", errStr.c_str());
  assert(wc.status == IBV_WC_SUCCESS);
}

void IOEngine::StartControlPlane() {
  if (running.load()) return;

  tcpContext->Listen();
  printf("host %s port %u fd %d\n", tcpContext->GetHost().c_str(), tcpContext->GetPort(),
         tcpContext->GetListenFd());
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

void IOEngine::InitDataPlane() {
  if (desc.backends.IsAvailableBackend(BackendType::RDMA)) {
    // TODO: add topology detection
    rdmaContext.reset(new application::RdmaContext(application::RdmaBackendType::IBVerbs));
    application::RdmaDeviceList devices = rdmaContext->GetRdmaDeviceList();
    application::ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(devices);
    assert(activeDevicePortList.size() > 0);
    devicePort = activeDevicePortList[desc.gpuId % activeDevicePortList.size()];
    application::RdmaDevice* device = devicePort.first;
    printf("%d nic id\n", desc.gpuId % activeDevicePortList.size());
    rdmaDeviceContext = device->CreateRdmaDeviceContext();
  }

  if (desc.backends.IsAvailableBackend(BackendType::XGMI)) {
    assert(false && "not implemented");
  }

  if (desc.backends.IsAvailableBackend(BackendType::TCP)) {
    assert(false && "not implemented");
  }
}

}  // namespace io
}  // namespace mori
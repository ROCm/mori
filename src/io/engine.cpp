#include "mori/io/engine.hpp"

#include <sys/epoll.h>

#include "mori/application/utils/check.hpp"
#include "mori/application/utils/serdes.hpp"
#include "mori/io/meta_data.hpp"

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
  printf("before host %s\n", tcpContext->handle.host.c_str());
  desc.tcpHandle = tcpContext->handle;
  printf("after host %s %d\n", desc.tcpHandle.host.c_str(), desc.backends.bits);

  // Initialize data plane
  InitDataPlane();
}

IOEngine::~IOEngine() { ShutdownControlPlane(); }

EngineDesc IOEngine::GetEngineDesc() {
  printf("before return host %s key %s %d\n", desc.tcpHandle.host.c_str(), desc.key.c_str(),
         desc.backends.bits);
  return desc;
}

void IOEngine::BuildRdmaConnection(const application::TCPEndpointHandle& tcpEph, bool isInitiator) {
  // exchange meta data
  application::RdmaEndpointConfig config;
  config.portId = rdmaPortId;
  config.gidIdx = 1;
  config.maxMsgsNum = 1024;
  config.maxCqeNum = 1024;
  config.alignment = 4096;
  application::RdmaEndpoint rdmaEp = rdmaDeviceContext->CreateRdmaEndpoint(config);

  // Exchange rdma endpoint
  application::TCPEndpoint tcpEp(tcpEph);
  application::RdmaEndpointHandle localRdmaEph = rdmaEp.handle;
  application::RdmaEndpointHandle remoteRdmaEph;
  application::RdmaEndpointHandlePacker packer;

  size_t packedSize = packer.PackedSizeCompact();
  std::vector<char> packed(packedSize);

  // send rdma endpoint
  if (isInitiator) {
    packer.PackCompact(localRdmaEph, packed.data());
    SYSCALL_RETURN_ZERO(tcpEp.Send(packed.data(), packedSize));
    SYSCALL_RETURN_ZERO(tcpEp.Recv(packed.data(), packedSize));
    packer.UnpackCompact(remoteRdmaEph, packed.data());
  } else {
    SYSCALL_RETURN_ZERO(tcpEp.Recv(packed.data(), packedSize));
    packer.UnpackCompact(remoteRdmaEph, packed.data());
    packer.PackCompact(localRdmaEph, packed.data());
    SYSCALL_RETURN_ZERO(tcpEp.Send(packed.data(), packedSize));
  }

  // Connect
  rdmaDeviceContext->ConnectEndpoint(localRdmaEph, remoteRdmaEph);
}

void IOEngine::RegisterRemoteEngine(EngineDesc remote) {
  assert(engineKV.find(remote.key) == engineKV.end());  // duplicate engine
  application::TCPEndpointHandle tcpEph =
      tcpContext->Connect(remote.tcpHandle.host, remote.tcpHandle.port);
  engineKV.insert({remote.key, remote});
  BuildRdmaConnection(tcpEph, true /*isInitiator*/);
  tcpContext->CloseEndpoint(tcpEph);
}

void IOEngine::DeRegisterRemoteEngine(EngineDesc remote) {
  // TODO: cleanup other resources such as qp pool and tcp conn
  engineKV.erase(remote.key);
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

void IOEngine::ControlPlaneLoop() {
  int maxEvents = 128;
  epoll_event events[maxEvents];
  while (running.load()) {
    int nfds = epoll_wait(epollFd, events, maxEvents, 5 /*ms*/);

    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;

      // Add new endpoints into epoll list
      if (fd == tcpContext->GetListenFd()) {
        application::TCPEndpointHandleVec eps = tcpContext->Accept();
        for (auto& ep : eps) {
          epoll_event ev{};
          ev.events = EPOLLIN | EPOLLET;
          ev.data.fd = ep.fd;
          SYSCALL_RETURN_ZERO(epoll_ctl(epollFd, EPOLL_CTL_ADD, ep.fd, &ev));
          tcpEpKV.insert({ep.fd, ep});
        }
      } else {  // for other endpoints, build RDMA connections
        assert(tcpEpKV.find(fd) != tcpEpKV.end());
        application::TCPEndpointHandle eph = tcpEpKV[fd];
        BuildRdmaConnection(eph, false /*isInitiator*/);
        SYSCALL_RETURN_ZERO(epoll_ctl(epollFd, EPOLL_CTL_DEL, fd, NULL));
        tcpContext->CloseEndpoint(eph);
        tcpEpKV.erase(fd);
      }
    }
  }
  printf("exit\n");
}

void IOEngine::InitDataPlane() {
  if (desc.backends.IsAvailableBackend(BackendType::RDMA)) {
    // TODO: add topology detection
    rdmaContext.reset(new application::RdmaContext(application::RdmaBackendType::IBVerbs));
    application::RdmaDeviceList devices = rdmaContext->GetRdmaDeviceList();
    application::ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(devices);
    assert(activeDevicePortList.size() > 0);
    application::ActiveDevicePort devicePort =
        activeDevicePortList[desc.gpuId % activeDevicePortList.size()];
    application::RdmaDevice* device = devicePort.first;
    rdmaPortId = devicePort.second;
    rdmaDeviceContext = device->CreateRdmaDeviceContext();
  }
}

}  // namespace io
}  // namespace mori
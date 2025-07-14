#include "mori/io/engine.hpp"

#include <sys/epoll.h>

#include "mori/application/utils/check.hpp"
#include "mori/io/meta_data.hpp"

namespace mori {
namespace ioengine {

IOEngine::IOEngine(EngineKey key, IOEngineConfig config) : config(config) {
  // Initialize descriptor
  EngineDesc desc;
  desc.key = key;
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  desc.hostname = std::string(hostname);
  desc.backends = BackendBitmap(config.backends);
}

IOEngine::~IOEngine() {
  ShutdownControlPlane();
  if (ctrlPlaneThd.joinable()) ctrlPlaneThd.join();
}

EngineDesc IOEngine::GetEngineDesc() { return desc; }

void IOEngine::RegisterRemoteEngine(EngineDesc) {}

void IOEngine::DeRegisterRemoteEngine(EngineDesc) {}

void IOEngine::StartControlPlane() {
  if (running.load()) return;

  // Create epoll fd
  epfd = epoll_create1(EPOLL_CLOEXEC);
  assert(epfd >= 0);

  // Add TCP listen fd
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ev.data.fd = epfd;
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, tcpContext->GetListenFd(), &ev));

  running.store(true);
  // ctrlPlaneThd = std::thread([this] { loop(); });
}

void IOEngine::ShutdownControlPlane() {}

void IOEngine::ControlPlaneLoop() {
  int maxEvents = 128;
  epoll_event events[maxEvents];
  while (running.load()) {
    int nfds = epoll_wait(epfd, events, maxEvents, 5 /*ms*/);
    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;

      // TODO: handle accepted tcp endpoint
      if (fd == tcpContext->GetListenFd()) {
        application::TCPEndpointHandleVec eps = tcpContext->Accept();
      }
    }
  }
}

}  // namespace ioengine
}  // namespace mori
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
#include "src/ras/link.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "src/ras/fd_registry.hpp"
#include "src/ras/identity.hpp"
#include "src/ras/logging.hpp"

namespace mori {
namespace ras {
namespace {

constexpr uint64_t kTagListener = ~0ull;
constexpr uint64_t kTagEvents = ~0ull - 1;
constexpr uint64_t kPublishIntervalNs = 50ull * 1000 * 1000;
constexpr size_t kEventsPerIteration = 1024;
constexpr int kListenBacklog = 128;

template <class... Fs>
struct Overloaded : Fs... {
  using Fs::operator()...;
};
template <class... Fs>
Overloaded(Fs...) -> Overloaded<Fs...>;

}  // namespace

uint64_t MonoNowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

int64_t WallNowNs() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}

LinkManager::LinkManager(Options options) : opts_(std::move(options)) {}

LinkManager::~LinkManager() {
  fdreg::Close(listenFd_);
  fdreg::Close(epollFd_);
}

bool LinkManager::Listen(const std::string& bindHost, std::string* error) {
  sockaddr_storage addr{};
  socklen_t addrLen = 0;
  auto* v4 = reinterpret_cast<sockaddr_in*>(&addr);
  auto* v6 = reinterpret_cast<sockaddr_in6*>(&addr);
  if (bindHost.empty()) {
    v4->sin_family = AF_INET;
    v4->sin_addr.s_addr = htonl(INADDR_ANY);
    addrLen = sizeof(*v4);
  } else if (inet_pton(AF_INET, bindHost.c_str(), &v4->sin_addr) == 1) {
    v4->sin_family = AF_INET;
    addrLen = sizeof(*v4);
  } else if (inet_pton(AF_INET6, bindHost.c_str(), &v6->sin6_addr) == 1) {
    v6->sin6_family = AF_INET6;
    addrLen = sizeof(*v6);
  } else {
    *error = "the RAS listener can't bind '" + bindHost + "': not an IP address";
    return false;
  }
  const uint16_t want = opts_.config.port;
  // An ephemeral pick can land on RCCL RAS's port if the host's range includes it.
  for (int attempt = 0; attempt < 8; ++attempt) {
    int fd = fdreg::Socket(addr.ss_family, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (fd < 0) {
      *error = std::string("socket: ") + std::strerror(errno);
      return false;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (addr.ss_family == AF_INET) {
      v4->sin_port = htons(want);
    } else {
      v6->sin6_port = htons(want);
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), addrLen) != 0) {
      int err = errno;
      fdreg::Close(fd);
      *error = err == EADDRINUSE ? "port " + std::to_string(want) + " on " + bindHost + " is in use"
                                 : std::string("bind: ") + std::strerror(err);
      return false;
    }
    if (listen(fd, kListenBacklog) != 0) {
      *error = std::string("listen: ") + std::strerror(errno);
      fdreg::Close(fd);
      return false;
    }
    sockaddr_storage bound{};
    socklen_t len = sizeof(bound);
    getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &len);
    uint16_t port =
        ntohs(bound.ss_family == AF_INET ? reinterpret_cast<sockaddr_in*>(&bound)->sin_port
                                         : reinterpret_cast<sockaddr_in6*>(&bound)->sin6_port);
    if (want == 0 && port == kRcclRasPort) {
      fdreg::Close(fd);
      continue;
    }
    listenFd_ = fd;
    port_ = port;
    return true;
  }
  *error = "every ephemeral port picked was 28028";
  return false;
}

bool LinkManager::InitLoop(std::string* error) {
  epollFd_ = fdreg::EpollCreate();
  if (epollFd_ < 0) {
    *error = std::string("epoll_create1: ") + std::strerror(errno);
    return false;
  }
  epoll_event ev{};
  ev.events = EPOLLIN;
  ev.data.u64 = kTagListener;
  if (epoll_ctl(epollFd_, EPOLL_CTL_ADD, listenFd_, &ev) != 0) {
    *error = std::string("epoll_ctl: ") + std::strerror(errno);
    return false;
  }
  if (opts_.events != nullptr && opts_.events->HasWakeFd()) {
    ev.data.u64 = kTagEvents;
    if (epoll_ctl(epollFd_, EPOLL_CTL_ADD, opts_.events->WakeFd(), &ev) != 0) {
      *error = std::string("epoll_ctl: ") + std::strerror(errno);
      return false;
    }
  }
  return true;
}

void LinkManager::SetIdentity(const RasProcessId& self) {
  self_ = self;
  {
    std::lock_guard<std::mutex> lock(labelsMu_);
    self_.labels = labels_;
  }
  MarkDirty();
}

void LinkManager::RunOnce(int maxWaitMs) {
  uint64_t now = MonoNowNs();
  int timeout = maxWaitMs;
  if (dirty_) {
    uint64_t due = lastPublishNs_ + kPublishIntervalNs;
    timeout = std::min<int>(timeout, due <= now ? 0 : static_cast<int>((due - now) / 1000000) + 1);
  }
  if (moreEvents_) timeout = 0;

  epoll_event evs[64];
  int n = epoll_wait(epollFd_, evs, 64, timeout);
  if (n < 0) {
    if (errno != EINTR && !(Degraded() & kDegradedLoopError)) {
      AddDegraded(kDegradedLoopError);
      RAS_LOG_WARN("RAS epoll_wait failed: {}; RAS is degraded", std::strerror(errno));
    }
    n = 0;
  }
  for (int i = 0; i < n; ++i) {
    if (evs[i].data.u64 == kTagListener) AcceptAll();
  }
  // Drained every pass, not only on a wakeup, so a missed wakeup only delays events.
  if (opts_.events != nullptr) {
    opts_.events->ConsumeWakeup();
    size_t handled =
        opts_.events->Drain([this](RasEvent& ev) { HandleEvent(ev); }, kEventsPerIteration);
    moreEvents_ = handled == kEventsPerIteration;
  }
  if (labelsChanged_.exchange(false, std::memory_order_acq_rel)) {
    std::lock_guard<std::mutex> lock(labelsMu_);
    self_.labels = labels_;
    MarkDirty();
  }
  now = MonoNowNs();
  if (dirty_ && now >= lastPublishNs_ + kPublishIntervalNs) Publish(now);
}

void LinkManager::AcceptAll() {
  for (;;) {
    int fd = fdreg::Accept(listenFd_, SOCK_NONBLOCK);
    if (fd < 0) return;
    ++inboundAccepted_;
    fdreg::Close(fd);
    MarkDirty();
  }
}

void LinkManager::HandleEvent(RasEvent& ev) {
  std::visit(Overloaded{
                 [&](EngineRegistered& e) { localEngines_.insert(e.engineKey); },
                 [&](EngineUnregistered& e) { localEngines_.erase(e.engineKey); },
                 [&](PeerAdded& e) {
                   if (!e.endpoint.Valid()) {
                     ++peersWithoutRas_;
                     RAS_LOG_DEBUG("engine {} registered {}, which advertises no RAS endpoint",
                                   e.localEngine, e.remoteEngine);
                   } else if (e.endpoint.uuid == opts_.uuid) {
                     ++sameProcessPeers_;
                   }
                 },
                 [&](PeerRemoved&) {},
             },
             ev);
  MarkDirty();
}

void LinkManager::Publish(uint64_t nowNs) {
  auto snap = std::make_shared<RuntimeSnapshot>();
  snap->self = self_;
  snap->listenPort = port_;
  snap->degraded = Degraded();
  if (opts_.events != nullptr) {
    EventQueueStats q = opts_.events->Stats();
    snap->events.capacity = q.capacity;
    snap->events.enqueued = q.enqueued;
    snap->events.dropped = q.dropped;
    snap->events.droppedByKind.assign(q.droppedByKind.begin(), q.droppedByKind.end());
    snap->events.depthHighWater = q.depthHighWater;
  }
  snap->inboundAccepted = inboundAccepted_;
  snap->inboundRejected = inboundRejected_;
  snap->protocolErrors = protocolErrors_;
  snap->peersWithoutRas = peersWithoutRas_;
  snap->sameProcessPeers = sameProcessPeers_;
  {
    std::lock_guard<std::mutex> lock(snapshotMu_);
    published_ = std::move(snap);
  }
  lastPublishNs_ = nowNs;
  dirty_ = false;
}

void LinkManager::Shutdown(bool) {
  fdreg::Close(listenFd_);
  listenFd_ = -1;
  fdreg::Close(epollFd_);
  epollFd_ = -1;
}

void LinkManager::RequestStop() {
  stop_.store(true, std::memory_order_release);
  if (opts_.events != nullptr) opts_.events->Wake();
}

RuntimeSnapshot LinkManager::Snapshot() const {
  RuntimeSnapshot s;
  {
    std::lock_guard<std::mutex> lock(snapshotMu_);
    if (published_) s = *published_;
  }
  s.listenPort = port_;
  s.degraded = Degraded();  // any thread may set a bit between publications
  return s;
}

bool LinkManager::SetLabels(const Labels& labels) {
  if (!ValidLabels(labels)) return false;
  {
    std::lock_guard<std::mutex> lock(labelsMu_);
    labels_ = labels;
  }
  labelsChanged_.store(true, std::memory_order_release);
  if (opts_.events != nullptr) opts_.events->Wake();
  return true;
}

void LinkManager::AddDegraded(uint32_t reasons) {
  degraded_.fetch_or(reasons, std::memory_order_acq_rel);
}

void LinkManager::ForgetFds() {
  listenFd_ = -1;
  epollFd_ = -1;
}

}  // namespace ras
}  // namespace mori

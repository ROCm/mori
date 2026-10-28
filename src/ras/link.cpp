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
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

#include "src/ras/fd_registry.hpp"
#include "src/ras/identity.hpp"
#include "src/ras/logging.hpp"

namespace mori {
namespace ras {
namespace {

constexpr uint64_t kTagListener = ~0ull;
constexpr uint64_t kTagEvents = ~0ull - 1;
constexpr uint64_t kMs = 1000ull * 1000;
constexpr uint64_t kPublishIntervalNs = 50 * kMs;
constexpr uint64_t kAcceptPauseNs = 100 * kMs;
constexpr uint64_t kShutdownLingerNs = 200 * kMs;
constexpr uint64_t kCloseLingerNs = 1000 * kMs;
constexpr size_t kEventsPerIteration = 1024;
constexpr int kListenBacklog = 128;
constexpr size_t kMaxPendingInbound = 64;
constexpr size_t kMaxSendBuffer = 256 * 1024;
constexpr size_t kMaxReadPerWakeup = 64 * 1024;
constexpr size_t kMaxTransitions = 8;

template <class... Fs>
struct Overloaded : Fs... {
  using Fs::operator()...;
};
template <class... Fs>
Overloaded(Fs...) -> Overloaded<Fs...>;

bool ToSockaddr(const std::string& host, uint16_t port, sockaddr_storage* addr, socklen_t* len) {
  *addr = sockaddr_storage{};
  auto* v4 = reinterpret_cast<sockaddr_in*>(addr);
  auto* v6 = reinterpret_cast<sockaddr_in6*>(addr);
  if (inet_pton(AF_INET, host.c_str(), &v4->sin_addr) == 1) {
    v4->sin_family = AF_INET;
    v4->sin_port = htons(port);
    *len = sizeof(*v4);
    return true;
  }
  if (inet_pton(AF_INET6, host.c_str(), &v6->sin6_addr) == 1) {
    v6->sin6_family = AF_INET6;
    v6->sin6_port = htons(port);
    *len = sizeof(*v6);
    return true;
  }
  return false;
}

LinkError ErrorForConnect(int err) {
  switch (err) {
    case ECONNREFUSED:
      return LinkError::kRasEndpointUnavailable;
    case ETIMEDOUT:
      return LinkError::kConnectTimeout;
    default:
      return LinkError::kConnectFailed;
  }
}

LinkError ErrorForRejection(proto::HelloStatus s) {
  switch (s) {
    case proto::HelloStatus::kRejectedVersion:
    case proto::HelloStatus::kRejectedCapability:
      return LinkError::kVersionMismatch;
    case proto::HelloStatus::kRejectedToken:
      return LinkError::kTokenMismatch;
    case proto::HelloStatus::kRejectedNotMe:
      return LinkError::kEndpointReused;
    case proto::HelloStatus::kRejectedDuplicate:
      return LinkError::kDuplicate;
    default:
      return LinkError::kProtocolError;
  }
}

const char* StatusName(proto::HelloStatus s) {
  switch (s) {
    case proto::HelloStatus::kOk:
      return "ok";
    case proto::HelloStatus::kDegraded:
      return "degraded";
    case proto::HelloStatus::kRejectedVersion:
      return "no common protocol version";
    case proto::HelloStatus::kRejectedToken:
      return "token mismatch";
    case proto::HelloStatus::kRejectedNotMe:
      return "a different runtime";
    case proto::HelloStatus::kRejectedDuplicate:
      return "a duplicate connection";
    case proto::HelloStatus::kRejectedSelf:
      return "its own runtime";
    case proto::HelloStatus::kRejectedCapability:
      return "no common security capability";
  }
  return "unknown";
}

// Errors that say something is misconfigured rather than shutting down or racing.
bool Noteworthy(LinkError e) {
  return e == LinkError::kTokenMismatch || e == LinkError::kVersionMismatch ||
         e == LinkError::kEndpointReused;
}

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
  for (auto& entry : conns_) fdreg::Close(entry.second->fd);
  fdreg::Close(listenFd_);
  fdreg::Close(epollFd_);
}

// ------------------------------------------------------------------------------------------
// Setup, on the starting thread.

bool LinkManager::Listen(const std::string& bindHost, std::string* error) {
  sockaddr_storage addr{};
  socklen_t addrLen = 0;
  if (bindHost.empty()) {
    auto* v4 = reinterpret_cast<sockaddr_in*>(&addr);
    v4->sin_family = AF_INET;
    v4->sin_addr.s_addr = htonl(INADDR_ANY);
    addrLen = sizeof(*v4);
  } else if (!ToSockaddr(bindHost, 0, &addr, &addrLen)) {
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
      reinterpret_cast<sockaddr_in*>(&addr)->sin_port = htons(want);
    } else {
      reinterpret_cast<sockaddr_in6*>(&addr)->sin6_port = htons(want);
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

// ------------------------------------------------------------------------------------------
// The loop.

void LinkManager::RunOnce(int maxWaitMs) {
  uint64_t now = MonoNowNs();
  int timeout = maxWaitMs;
  uint64_t due = NextDeadlineNs(now);
  if (due != std::numeric_limits<uint64_t>::max()) {
    uint64_t waitMs = due <= now ? 0 : (due - now + kMs - 1) / kMs;
    timeout = static_cast<int>(std::min<uint64_t>(static_cast<uint64_t>(timeout), waitMs));
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
  now = MonoNowNs();
  for (int i = 0; i < n; ++i) {
    uint64_t tag = evs[i].data.u64;
    if (tag == kTagListener) {
      AcceptAll(now);
    } else if (tag != kTagEvents) {
      // A connection closed earlier in this batch is gone from the map, so its stale event
      // is skipped rather than applied to whatever reused its fd.
      auto it = conns_.find(tag);
      if (it != conns_.end()) HandleConnEvent(it->second.get(), evs[i].events, now);
    }
  }
  // Drained every pass, not only on a wakeup, so a missed wakeup only delays events.
  if (opts_.events != nullptr) {
    opts_.events->ConsumeWakeup();
    size_t handled =
        opts_.events->Drain([&](RasEvent& ev) { HandleEvent(ev, now); }, kEventsPerIteration);
    moreEvents_ = handled == kEventsPerIteration;
  }
  if (labelsChanged_.exchange(false, std::memory_order_acq_rel)) {
    std::lock_guard<std::mutex> lock(labelsMu_);
    self_.labels = labels_;
    MarkDirty();
  }
  now = MonoNowNs();
  RunTimers(now);
  SweepBroken(now);
  if (dirty_ && now >= lastPublishNs_ + kPublishIntervalNs) Publish(now);
}

uint64_t LinkManager::NextDeadlineNs(uint64_t now) const {
  uint64_t due = std::numeric_limits<uint64_t>::max();
  for (const auto& entry : conns_) {
    if (entry.second->phase != Phase::kEstablished) due = std::min(due, entry.second->deadlineNs);
  }
  for (const auto& entry : links_) {
    const Link& l = entry.second;
    if (l.connectScheduled) due = std::min(due, l.connectDueNs);
    if (l.conn != nullptr && l.conn->phase == Phase::kEstablished) {
      due = std::min(due, l.nextHeartbeatNs);
    }
  }
  if (acceptPaused_) due = std::min(due, acceptResumeNs_);
  if (dirty_) due = std::min(due, lastPublishNs_ + kPublishIntervalNs);
  (void)now;
  return due;
}

void LinkManager::RunTimers(uint64_t now) {
  if (acceptPaused_ && now >= acceptResumeNs_) {
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u64 = kTagListener;
    epoll_ctl(epollFd_, EPOLL_CTL_MOD, listenFd_, &ev);
    acceptPaused_ = false;
  }
  // Handshakes and closes that ran out of time.
  std::vector<uint64_t> expired;
  for (const auto& entry : conns_) {
    if (entry.second->phase != Phase::kEstablished && now >= entry.second->deadlineNs) {
      expired.push_back(entry.first);
    }
  }
  for (uint64_t id : expired) {
    auto cit = conns_.find(id);
    if (cit == conns_.end()) continue;
    Conn* c = cit->second.get();
    Link* l = c->phase == Phase::kClosing ? nullptr : LinkOf(c);
    if (l == nullptr) {
      CloseConn(c);
    } else {
      LinkDown(links_.find(l->peer), LinkError::kConnectTimeout,
               "no HELLO within the handshake timeout", now, false);
    }
  }
  for (auto& entry : links_) {
    Link& l = entry.second;
    if (l.connectScheduled && now >= l.connectDueNs && l.conn == nullptr) StartConnect(l, now);
    if (l.conn != nullptr && l.conn->phase == Phase::kEstablished && now >= l.nextHeartbeatNs) {
      SendHeartbeat(l, now);
    }
  }
}

void LinkManager::SweepBroken(uint64_t now) {
  std::vector<uint64_t> broken;
  for (const auto& entry : conns_) {
    if (entry.second->broken) broken.push_back(entry.first);
  }
  for (uint64_t id : broken) {
    auto cit = conns_.find(id);
    if (cit == conns_.end()) continue;
    Conn* c = cit->second.get();
    Link* l = LinkOf(c);
    if (l == nullptr) {
      CloseConn(c);
      continue;
    }
    bool wasUp = c->phase == Phase::kEstablished;
    LinkDown(links_.find(l->peer), LinkError::kLinkClosed, "a send failed", now, wasUp);
  }
}

void LinkManager::AcceptAll(uint64_t now) {
  for (;;) {
    int fd = fdreg::Accept(listenFd_, SOCK_NONBLOCK);
    if (fd < 0) {
      if (errno == EINTR) continue;
      if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS || errno == ENOMEM) {
        epoll_event ev{};
        ev.data.u64 = kTagListener;
        epoll_ctl(epollFd_, EPOLL_CTL_MOD, listenFd_, &ev);
        acceptPaused_ = true;
        acceptResumeNs_ = now + kAcceptPauseNs;
        RAS_LOG_DEBUG("RAS accept paused: {}", std::strerror(errno));
      }
      return;
    }
    ++inboundAccepted_;
    MarkDirty();
    if (pendingInbound_ >= kMaxPendingInbound) {
      ++inboundRejected_;
      fdreg::Close(fd);
      continue;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    Conn* c = NewConn(fd, /*outbound=*/false, now);
    c->phase = Phase::kAwaitingHello;
    c->pending = true;
    ++pendingInbound_;
    c->deadlineNs = now + opts_.config.HandshakeTimeoutNs();
    Watch(c, EPOLLIN);
  }
}

void LinkManager::HandleEvent(RasEvent& ev, uint64_t now) {
  std::visit(
      Overloaded{
          [&](EngineRegistered& e) { localEngines_.insert(e.engineKey); },
          [&](EngineUnregistered& e) {
            localEngines_.erase(e.engineKey);
            DetachOwner(e.engineKey, &self_);
            std::vector<std::pair<std::string, std::string>> gone;
            for (const auto& entry : relations_) {
              if (entry.first.first == e.engineKey) gone.push_back(entry.first);
            }
            for (const auto& key : gone) RemoveRelation(key.first, key.second, now);
          },
          [&](PeerAdded& e) {
            if (!e.endpoint.Valid()) {
              ++peersWithoutRas_;
              RAS_LOG_DEBUG("engine {} registered {}, which advertises no RAS endpoint",
                            e.localEngine, e.remoteEngine);
            } else if (e.endpoint.uuid == opts_.uuid) {
              ++sameProcessPeers_;
            } else {
              AddRelation(e, now);
            }
          },
          [&](PeerRemoved& e) { RemoveRelation(e.localEngine, e.remoteEngine, now); },
          [&](EngineGpuAttached& e) { AttachGpu(e.engineKey, e.gpu, e.pciBusId, e.nics, &self_); },
      },
      ev);
  MarkDirty();
}

void LinkManager::HandleConnEvent(Conn* c, uint32_t events, uint64_t now) {
  if (c->phase == Phase::kTcpConnecting) {
    if (!(events & (EPOLLOUT | EPOLLERR | EPOLLHUP))) return;
    int err = 0;
    socklen_t len = sizeof(err);
    getsockopt(c->fd, SOL_SOCKET, SO_ERROR, &err, &len);
    Link* l = LinkOf(c);
    if (l == nullptr) {
      CloseConn(c);
    } else if (err == 0) {
      OnConnected(c, now);
    } else {
      LinkDown(links_.find(l->peer), ErrorForConnect(err), std::strerror(err), now, false);
    }
    return;
  }
  if (events & EPOLLOUT) Flush(c);
  if (events & (EPOLLIN | EPOLLHUP | EPOLLERR)) OnReadable(c, now);
}

// ------------------------------------------------------------------------------------------
// Connections.

LinkManager::Conn* LinkManager::NewConn(int fd, bool outbound, uint64_t now) {
  auto conn = std::make_unique<Conn>();
  conn->id = nextConnId_++;
  conn->fd = fd;
  conn->outbound = outbound;
  conn->deadlineNs = now + opts_.config.HandshakeTimeoutNs();
  Conn* c = conn.get();
  conns_.emplace(c->id, std::move(conn));
  return c;
}

void LinkManager::Watch(Conn* c, uint32_t events) {
  epoll_event ev{};
  ev.events = events;
  ev.data.u64 = c->id;
  if (epoll_ctl(epollFd_, c->inEpoll ? EPOLL_CTL_MOD : EPOLL_CTL_ADD, c->fd, &ev) == 0) {
    c->inEpoll = true;
    c->epollEvents = events;
  } else {
    c->broken = true;
  }
}

void LinkManager::CloseConn(Conn* c) {
  if (Link* l = LinkOf(c)) l->conn = nullptr;
  if (c->pending) --pendingInbound_;
  if (c->fd >= 0) {
    if (c->inEpoll) epoll_ctl(epollFd_, EPOLL_CTL_DEL, c->fd, nullptr);
    fdreg::Close(c->fd);
  }
  MarkDirty();
  conns_.erase(c->id);
}

// Delivers what is queued, half-closes, and waits for the peer's close, so the last message
// (a HELLO rejection or LEAVE) isn't lost to a reset.
void LinkManager::BeginClose(Conn* c, uint64_t now) {
  if (Link* l = LinkOf(c)) l->conn = nullptr;
  if (c->pending) {
    --pendingInbound_;
    c->pending = false;
  }
  Flush(c);
  shutdown(c->fd, SHUT_WR);
  c->phase = Phase::kClosing;
  c->deadlineNs = now + std::min(kCloseLingerNs, opts_.config.HandshakeTimeoutNs());
  c->in.clear();
  c->inStart = 0;
  Watch(c, EPOLLIN);
  MarkDirty();
}

LinkManager::Link* LinkManager::LinkOf(const Conn* c) {
  auto it = links_.find(c->peer);
  if (it == links_.end() || it->second.conn != c) return nullptr;
  return &it->second;
}

bool LinkManager::Send(Conn* c, const proto::Message& m) {
  uint16_t version = c->version != 0 ? c->version : proto::kVersionMin;
  std::vector<uint8_t> frame =
      proto::EncodeFrame(proto::LinkHeader(version, opts_.uuid.lo, c->txSeq + 1), m);
  // A peer that stops reading fills the buffer; later messages are dropped, not queued.
  if (c->out.size() - c->outStart + frame.size() > kMaxSendBuffer) return false;
  ++c->txSeq;
  if (c->outStart == c->out.size()) {
    c->out.clear();
    c->outStart = 0;
  }
  c->out.insert(c->out.end(), frame.begin(), frame.end());
  Flush(c);
  return true;
}

void LinkManager::Flush(Conn* c) {
  while (c->outStart < c->out.size()) {
    ssize_t n = send(c->fd, c->out.data() + c->outStart, c->out.size() - c->outStart,
                     MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n > 0) {
      c->outStart += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      if (!(c->epollEvents & EPOLLOUT)) Watch(c, c->epollEvents | EPOLLOUT);
      return;
    }
    c->broken = true;
    return;
  }
  c->out.clear();
  c->outStart = 0;
  if ((c->epollEvents & EPOLLOUT) && c->phase != Phase::kTcpConnecting) {
    Watch(c, c->epollEvents & ~static_cast<uint32_t>(EPOLLOUT));
  }
}

void LinkManager::SendHello(Conn* c, proto::HelloRole role, proto::HelloStatus status,
                            const Link* l) {
  proto::Hello h;
  h.minVersion = proto::kVersionMin;
  h.maxVersion = proto::kVersionMax;
  h.role = role;
  h.status = status;
  h.capabilities = proto::kCapToken | proto::kCapReadOnly;
  if (role == proto::HelloRole::kRequest && l != nullptr) {
    h.token = l->endpoint.token;
    h.expectedPeer = l->peer;
  }
  h.identity = self_;
  Send(c, h);
}

// ------------------------------------------------------------------------------------------
// Reading and the protocol.

void LinkManager::OnReadable(Conn* c, uint64_t now) {
  size_t total = 0;
  bool eof = false;
  int err = 0;
  uint64_t rxMono = 0;
  int64_t rxWall = 0;
  uint8_t buf[16384];
  while (total < kMaxReadPerWakeup) {
    ssize_t n = recv(c->fd, buf, sizeof(buf), MSG_DONTWAIT);
    if (n > 0) {
      if (rxMono == 0) {
        rxMono = MonoNowNs();
        rxWall = WallNowNs();
      }
      if (c->phase != Phase::kClosing) c->in.insert(c->in.end(), buf, buf + n);
      total += static_cast<size_t>(n);
      continue;
    }
    if (n == 0) {
      eof = true;
    } else if (errno == EINTR) {
      continue;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
      err = errno;
      eof = true;
    }
    break;
  }
  if (c->phase == Phase::kClosing) {
    if (eof) CloseConn(c);
    return;
  }
  if (rxMono != 0 && !ProcessFrames(c, rxMono, rxWall)) return;
  if (!eof) return;
  Link* l = LinkOf(c);
  if (l == nullptr) {
    CloseConn(c);
    return;
  }
  bool wasUp = c->phase == Phase::kEstablished;
  std::string detail = err != 0 ? std::strerror(err) : "closed by the peer";
  LinkDown(links_.find(l->peer), wasUp ? LinkError::kLinkClosed : LinkError::kConnectFailed,
           wasUp ? detail : detail + " before its HELLO", now, wasUp);
}

bool LinkManager::ProcessFrames(Conn* c, uint64_t rxMonoNs, int64_t rxWallNs) {
  for (;;) {
    size_t avail = c->in.size() - c->inStart;
    if (avail < proto::kHeaderSize) break;
    const uint8_t* p = c->in.data() + c->inStart;
    proto::VersionRange accept =
        c->version != 0 ? proto::VersionRange{c->version, c->version} : proto::kSupportedVersions;
    proto::RasHeader h;
    proto::DecodeStatus st = proto::DecodeHeader(p, avail, accept, &h);
    uint64_t now = MonoNowNs();
    if (st != proto::DecodeStatus::kOk) {
      proto::RasHeader raw = proto::ReadHeaderFields(p);
      bool helloInHandshake =
          (c->phase == Phase::kAwaitingHello || c->phase == Phase::kHelloSent) &&
          raw.type == static_cast<uint16_t>(proto::MessageType::kHello);
      if (st == proto::DecodeStatus::kBadVersion && helloInHandshake) {
        // A peer whose versions are all outside ours: say so in a v1 HELLO it can read.
        std::string detail = "the peer speaks protocol version " + std::to_string(raw.version);
        if (c->phase == Phase::kAwaitingHello) {
          ++inboundRejected_;
          RAS_LOG_DEBUG("RAS refused a HELLO: {}", detail);
          RejectHello(c, proto::HelloStatus::kRejectedVersion, now);
        } else if (Link* l = LinkOf(c)) {
          LinkDown(links_.find(l->peer), LinkError::kVersionMismatch, detail, now, false);
        } else {
          CloseConn(c);
        }
        return false;
      }
      ProtocolError(c, std::string("bad header: ") + proto::ToString(st), now);
      return false;
    }
    if (avail < proto::kHeaderSize + h.length) break;
    proto::Message m;
    st = proto::DecodePayload(h, p + proto::kHeaderSize, h.length, &m);
    if (st != proto::DecodeStatus::kOk) {
      ProtocolError(c, std::string("bad payload: ") + proto::ToString(st), now);
      return false;
    }
    c->inStart += proto::kHeaderSize + h.length;
    if (h.sequence != c->rxSeq + 1) {
      ProtocolError(
          c, "sequence " + std::to_string(h.sequence) + " after " + std::to_string(c->rxSeq), now);
      return false;
    }
    c->rxSeq = h.sequence;

    bool alive = false;
    const proto::Hello* hello = std::get_if<proto::Hello>(&m);
    switch (c->phase) {
      case Phase::kAwaitingHello:
        if (hello == nullptr) {
          ProtocolError(c, "expected a HELLO request", now);
        } else {
          alive = OnHelloRequest(c, h, *hello, now);
        }
        break;
      case Phase::kHelloSent:
        if (hello == nullptr) {
          ProtocolError(c, "expected a HELLO reply", now);
        } else {
          alive = OnHelloReply(c, h, *hello, now);
        }
        break;
      case Phase::kEstablished: {
        Link* l = LinkOf(c);
        if (l == nullptr) {
          CloseConn(c);
        } else if (h.sender != c->peer.lo) {
          ProtocolError(c, "the sender changed", now);
        } else {
          alive = OnLinkMessage(c, *l, m, rxMonoNs, rxWallNs);
        }
        break;
      }
      default:
        alive = true;
    }
    if (!alive) return false;
  }
  if (c->inStart == c->in.size()) {
    c->in.clear();
    c->inStart = 0;
  } else if (c->inStart > kMaxReadPerWakeup) {
    c->in.erase(c->in.begin(), c->in.begin() + static_cast<std::ptrdiff_t>(c->inStart));
    c->inStart = 0;
  }
  return true;
}

bool LinkManager::OnHelloRequest(Conn* c, const proto::RasHeader& h, const proto::Hello& req,
                                 uint64_t now) {
  if (req.role != proto::HelloRole::kRequest || h.sender != req.identity.uuid.lo) {
    ProtocolError(c, "malformed HELLO request", now);
    return false;
  }
  const RuntimeUuid peer = req.identity.uuid;
  std::optional<uint16_t> version =
      proto::NegotiateVersion(proto::kSupportedVersions, {req.minVersion, req.maxVersion});
  proto::HelloStatus reject = proto::HelloStatus::kOk;
  if (!version) {
    reject = proto::HelloStatus::kRejectedVersion;
  } else if (!(req.capabilities & proto::kCapToken)) {
    reject = proto::HelloStatus::kRejectedCapability;
  } else if (peer == opts_.uuid) {
    reject = proto::HelloStatus::kRejectedSelf;
  } else if (!req.expectedPeer.IsZero() && req.expectedPeer != opts_.uuid) {
    reject = proto::HelloStatus::kRejectedNotMe;
  } else if (req.token != opts_.token) {
    reject = proto::HelloStatus::kRejectedToken;
  }
  LinkMap::iterator it = links_.find(peer);
  if (reject == proto::HelloStatus::kOk && it != links_.end()) {
    const Link& l = it->second;
    if (l.state == LinkState::kPeerLeft) {
      reject = proto::HelloStatus::kRejectedDuplicate;  // that runtime announced its exit
    } else if (l.conn != nullptr && l.conn->outbound && opts_.uuid < peer) {
      reject = proto::HelloStatus::kRejectedDuplicate;  // our crossing connection wins
    }
  }
  if (reject != proto::HelloStatus::kOk) {
    ++inboundRejected_;
    RAS_LOG_DEBUG("RAS refused a HELLO from runtime {} on {}: {}", peer.ToString(),
                  req.identity.hostname, StatusName(reject));
    RejectHello(c, reject, now);
    return false;
  }
  if (it == links_.end()) {
    it = links_.emplace(peer, Link{}).first;
    it->second.peer = peer;
  }
  Link& l = it->second;
  // The peer's newer connection, or its winning crossing one, replaces whatever we hold.
  if (l.conn != nullptr) CloseConn(l.conn);
  if (c->pending) {
    c->pending = false;
    --pendingInbound_;
  }
  c->peer = peer;
  SendHello(c, proto::HelloRole::kReply,
            Degraded() != 0 ? proto::HelloStatus::kDegraded : proto::HelloStatus::kOk, &l);
  Establish(l, c, *version, req, now);
  return true;
}

bool LinkManager::OnHelloReply(Conn* c, const proto::RasHeader& h, const proto::Hello& rep,
                               uint64_t now) {
  LinkMap::iterator it = links_.find(c->peer);
  if (it == links_.end() || it->second.conn != c) {
    CloseConn(c);
    return false;
  }
  if (rep.role != proto::HelloRole::kReply || h.sender != rep.identity.uuid.lo) {
    ProtocolError(c, "malformed HELLO reply", now);
    return false;
  }
  Link& l = it->second;
  if (rep.identity.uuid != l.peer) {
    LinkDown(it, LinkError::kEndpointReused,
             "runtime " + rep.identity.uuid.ToString() + " answers at " + l.endpoint.host + ":" +
                 std::to_string(l.endpoint.port),
             now, false);
    return false;
  }
  if (rep.status != proto::HelloStatus::kOk && rep.status != proto::HelloStatus::kDegraded) {
    LinkDown(it, ErrorForRejection(rep.status),
             std::string("the peer refused the HELLO: ") + StatusName(rep.status), now, false);
    return false;
  }
  std::optional<uint16_t> version =
      proto::NegotiateVersion(proto::kSupportedVersions, {rep.minVersion, rep.maxVersion});
  if (!version) {
    LinkDown(it, LinkError::kVersionMismatch, "the peer accepted with no common version", now,
             false);
    return false;
  }
  Establish(l, c, *version, rep, now);
  return true;
}

bool LinkManager::OnLinkMessage(Conn* c, Link& l, const proto::Message& m, uint64_t rxMonoNs,
                                int64_t rxWallNs) {
  l.lastRxNs = rxMonoNs;
  MarkDirty();
  return std::visit(
      Overloaded{
          [&](const proto::Heartbeat& hb) {
            ++l.heartbeatsReceived;
            l.peerDegraded = (hb.flags & proto::kHeartbeatFlagDegraded) != 0;
            // Answered at once: the hold the peer subtracts is this function's run time.
            if (Send(c, MakeAck(hb, rxMonoNs, MonoNowNs(), WallNowNs()))) ++l.acksSent;
            return true;
          },
          [&](const proto::HeartbeatAck& ack) {
            ++l.acksReceived;
            l.heartbeat.OnAck(ack, rxMonoNs, rxWallNs);
            return true;
          },
          [&](const proto::Leave& leave) {
            uint64_t now = MonoNowNs();
            LinkMap::iterator it = links_.find(l.peer);
            if (leave.reason == proto::LeaveReason::kProcessExit) {
              CloseConn(c);
              ++l.disconnects;
              l.connectScheduled = false;
              SetState(l, LinkState::kPeerLeft, LinkError::kNone, now);
              RAS_LOG_DEBUG("RAS peer {} ({}) left", l.peer.ToString(), l.peerHostname);
              if (l.relations.empty()) links_.erase(it);
            } else {
              LinkDown(it, LinkError::kLinkReleased, "the peer released the link", now, true);
            }
            return false;
          },
          [&](const proto::Hello&) {
            ProtocolError(c, "HELLO on an established link", MonoNowNs());
            return false;
          },
      },
      m);
}

void LinkManager::RejectHello(Conn* c, proto::HelloStatus status, uint64_t now) {
  SendHello(c, proto::HelloRole::kReply, status, nullptr);
  BeginClose(c, now);
}

void LinkManager::ProtocolError(Conn* c, const std::string& detail, uint64_t now) {
  ++protocolErrors_;
  MarkDirty();
  RAS_LOG_DEBUG("RAS protocol error on a connection with {}: {}",
                c->peer.IsZero() ? std::string("an unidentified peer") : c->peer.ToString(),
                detail);
  if (Link* l = LinkOf(c)) {
    LinkDown(links_.find(l->peer), LinkError::kProtocolError, detail, now, false);
  } else {
    CloseConn(c);
  }
}

// ------------------------------------------------------------------------------------------
// Links.

void LinkManager::AddRelation(const PeerAdded& e, uint64_t now) {
  std::pair<std::string, std::string> key{e.localEngine, e.remoteEngine};
  auto existing = relations_.find(key);
  // The same remote engine key re-registered from another runtime: the old relation ends.
  if (existing != relations_.end() && existing->second != e.endpoint.uuid) {
    RemoveRelation(e.localEngine, e.remoteEngine, now);
  }
  relations_[key] = e.endpoint.uuid;
  LinkMap::iterator it = links_.find(e.endpoint.uuid);
  if (it == links_.end()) {
    it = links_.emplace(e.endpoint.uuid, Link{}).first;
    it->second.peer = e.endpoint.uuid;
  }
  Link& l = it->second;
  l.relations.insert(key);
  l.endpoint = e.endpoint;
  if (l.peerHostname.empty()) l.peerHostname = e.remoteHostname;
  if (l.peerPid == 0) l.peerPid = e.remotePid;
  if (l.state != LinkState::kPeerLeft && l.conn == nullptr && !l.connectScheduled) {
    ScheduleConnect(l, now, /*immediate=*/true);
  }
  MarkDirty();
}

void LinkManager::RemoveRelation(const std::string& local, const std::string& remote,
                                 uint64_t now) {
  auto rit = relations_.find({local, remote});
  if (rit == relations_.end()) return;
  RuntimeUuid peer = rit->second;
  relations_.erase(rit);
  LinkMap::iterator it = links_.find(peer);
  if (it == links_.end()) return;
  Link& l = it->second;
  l.relations.erase({local, remote});
  MarkDirty();
  if (!l.relations.empty()) return;
  l.connectScheduled = false;
  Conn* c = l.conn;
  // The peer opened this connection, so it holds a relation of its own; keep the link.
  if (c != nullptr && !c->outbound) return;
  if (c != nullptr) {
    if (c->phase == Phase::kEstablished) {
      Send(c, proto::Leave{proto::LeaveReason::kLinkReleased});
    }
    BeginClose(c, now);
  }
  links_.erase(it);
}

void LinkManager::StartConnect(Link& l, uint64_t now) {
  l.connectScheduled = false;
  ++l.connectAttempts;
  MarkDirty();
  sockaddr_storage addr;
  socklen_t len;
  bool parsed = ToSockaddr(l.endpoint.host, l.endpoint.port, &addr, &len);
  int fd = parsed ? fdreg::Socket(addr.ss_family, SOCK_STREAM | SOCK_NONBLOCK, 0) : -1;
  if (fd < 0) {
    l.lastErrorDetail =
        parsed ? std::strerror(errno) : "'" + l.endpoint.host + "' isn't an IP address";
    SetState(l, LinkState::kDisconnected, LinkError::kConnectFailed, now);
    ScheduleConnect(l, now, false);
    return;
  }
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  Conn* c = NewConn(fd, /*outbound=*/true, now);
  c->peer = l.peer;
  l.conn = c;
  l.lastOutbound = true;
  SetState(l, LinkState::kConnecting, l.lastError, now);
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), len) == 0) {
    OnConnected(c, now);
  } else if (errno == EINPROGRESS) {
    c->phase = Phase::kTcpConnecting;
    Watch(c, EPOLLOUT);
  } else {
    int err = errno;
    LinkDown(links_.find(l.peer), ErrorForConnect(err), std::strerror(err), now, false);
  }
}

void LinkManager::OnConnected(Conn* c, uint64_t now) {
  c->phase = Phase::kHelloSent;
  c->deadlineNs = now + opts_.config.HandshakeTimeoutNs();
  Watch(c, EPOLLIN);
  SendHello(c, proto::HelloRole::kRequest,
            Degraded() != 0 ? proto::HelloStatus::kDegraded : proto::HelloStatus::kOk, LinkOf(c));
}

void LinkManager::Establish(Link& l, Conn* c, uint16_t version, const proto::Hello& hello,
                            uint64_t now) {
  c->phase = Phase::kEstablished;
  c->version = version;
  l.conn = c;
  l.lastOutbound = c->outbound;
  l.version = version;
  l.peerHostname = hello.identity.hostname;
  l.peerPid = hello.identity.pid;
  l.peerDegraded = hello.status == proto::HelloStatus::kDegraded;
  l.connectScheduled = false;
  l.backoffNs = 0;
  ++l.connects;
  l.lastRxNs = now;
  l.nextHeartbeatNs = now;
  l.lastErrorDetail.clear();
  SetState(l, LinkState::kConnected, LinkError::kNone, now);
  RAS_LOG_DEBUG("RAS link to runtime {} ({} pid {}) is up, {}", l.peer.ToString(), l.peerHostname,
                l.peerPid, c->outbound ? "opened here" : "opened by the peer");
}

bool LinkManager::LinkDown(LinkMap::iterator it, LinkError error, const std::string& detail,
                           uint64_t now, bool retryNow) {
  Link& l = it->second;
  if (l.conn != nullptr) {
    if (l.conn->phase == Phase::kEstablished) ++l.disconnects;
    CloseConn(l.conn);
  }
  bool changed = l.lastError != error;
  l.lastErrorDetail = detail;
  SetState(l, LinkState::kDisconnected, error, now);
  if (changed && Noteworthy(error)) {
    RAS_LOG_INFO("RAS link to runtime {} ({}): {}: {}", l.peer.ToString(), l.peerHostname,
                 ToString(error), detail);
  } else {
    RAS_LOG_DEBUG("RAS link to runtime {} ({}): {}: {}", l.peer.ToString(), l.peerHostname,
                  ToString(error), detail);
  }
  if (l.relations.empty()) {
    // Nothing here needs the link; the peer reconnects if it does.
    links_.erase(it);
    return false;
  }
  ScheduleConnect(l, now, retryNow);
  return true;
}

void LinkManager::ScheduleConnect(Link& l, uint64_t now, bool immediate) {
  if (l.state == LinkState::kPeerLeft || l.relations.empty() || !l.endpoint.Valid()) {
    l.connectScheduled = false;
    return;
  }
  if (immediate) {
    l.backoffNs = 0;
    l.connectDueNs = now;
  } else {
    l.backoffNs = l.backoffNs == 0
                      ? opts_.config.ReconnectBackoffInitialNs()
                      : std::min(l.backoffNs * 2, opts_.config.ReconnectBackoffMaxNs());
    l.connectDueNs = now + l.backoffNs;
  }
  l.connectScheduled = true;
}

void LinkManager::SendHeartbeat(Link& l, uint64_t now) {
  proto::Heartbeat hb = l.heartbeat.Next(now, Degraded() != 0 ? proto::kHeartbeatFlagDegraded : 0);
  if (Send(l.conn, hb)) ++l.heartbeatsSent;
  l.nextHeartbeatNs = now + opts_.config.HeartbeatNs();
}

void LinkManager::SetState(Link& l, LinkState state, LinkError error, uint64_t now) {
  if (l.state == state && l.lastError == error && !l.transitions.empty()) return;
  l.state = state;
  l.lastError = error;
  l.transitions.push_back(Link::Transition{now, state, error});
  if (l.transitions.size() > kMaxTransitions) l.transitions.pop_front();
  MarkDirty();
}

LinkInfo LinkManager::Describe(const Link& l, uint64_t now) const {
  LinkInfo i;
  i.peer = l.peer;
  i.peerHost = l.endpoint.host;
  i.peerPort = l.endpoint.port;
  i.peerHostname = l.peerHostname;
  i.peerPid = l.peerPid;
  i.state = l.state;
  i.lastError = l.lastError;
  i.lastErrorDetail = l.lastErrorDetail;
  i.outbound = l.conn != nullptr ? l.conn->outbound : l.lastOutbound;
  i.peerDegraded = l.peerDegraded;
  i.version = l.version;
  i.relations = static_cast<uint32_t>(l.relations.size());
  i.connectAttempts = l.connectAttempts;
  i.connects = l.connects;
  i.disconnects = l.disconnects;
  i.heartbeatsSent = l.heartbeatsSent;
  i.heartbeatsReceived = l.heartbeatsReceived;
  i.acksSent = l.acksSent;
  i.acksReceived = l.acksReceived;
  i.lastRxAgeMs = l.lastRxNs != 0 ? static_cast<int64_t>((now - l.lastRxNs) / kMs) : -1;
  if (const TimingSample* last = l.heartbeat.Last()) {
    const TimingSample* best = l.heartbeat.Best();
    i.hasTiming = true;
    i.rttNs = last->rttNs;
    i.offsetNs = best->offsetNs;
    i.offsetBoundNs = best->errorBoundNs;
  }
  for (const Link::Transition& t : l.transitions) {
    i.transitions.push_back(
        LinkTransition{static_cast<int64_t>((now - t.atNs) / kMs), t.state, t.error});
  }
  return i;
}

void LinkManager::Publish(uint64_t nowNs) {
  auto snap = std::make_shared<RuntimeSnapshot>();
  snap->self = self_;
  snap->listenPort = port_;
  snap->degraded = Degraded();
  for (const auto& entry : links_) snap->links.push_back(Describe(entry.second, nowNs));
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

// ------------------------------------------------------------------------------------------
// Shutdown and cross-thread calls.

void LinkManager::Shutdown(bool sendLeave) {
  // The listener goes first, so a peer that reconnects at once is refused, not left hanging
  // in the backlog of a runtime that will never answer it.
  fdreg::Close(listenFd_);
  listenFd_ = -1;
  // LEAVE on every established link, then FIN after it, so each peer reads LEAVE before EOF.
  for (auto& entry : links_) {
    Conn* c = entry.second.conn;
    if (sendLeave && c != nullptr && c->phase == Phase::kEstablished) {
      Send(c, proto::Leave{proto::LeaveReason::kProcessExit});
    }
  }
  std::vector<pollfd> waiting;
  for (auto& entry : conns_) {
    Conn* c = entry.second.get();
    Flush(c);
    shutdown(c->fd, SHUT_WR);
    waiting.push_back(pollfd{c->fd, POLLIN, 0});
  }
  // Closing with unread input would reset the connection and could cost the peer its LEAVE,
  // so wait briefly for each peer to close first.
  uint64_t deadline = MonoNowNs() + kShutdownLingerNs;
  while (!waiting.empty()) {
    uint64_t now = MonoNowNs();
    if (now >= deadline) break;
    int timeoutMs = static_cast<int>((deadline - now + kMs - 1) / kMs);
    if (poll(waiting.data(), waiting.size(), timeoutMs) <= 0) break;
    for (size_t i = 0; i < waiting.size();) {
      bool done = false;
      if (waiting[i].revents != 0) {
        char buf[4096];
        ssize_t n = recv(waiting[i].fd, buf, sizeof(buf), MSG_DONTWAIT);
        done = n == 0 || (n < 0 && errno != EAGAIN && errno != EINTR);
      }
      if (done) {
        waiting[i] = waiting.back();
        waiting.pop_back();
      } else {
        waiting[i].revents = 0;
        ++i;
      }
    }
  }
  for (auto& entry : conns_) fdreg::Close(entry.second->fd);
  conns_.clear();
  links_.clear();
  relations_.clear();
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
  if (opts_.events != nullptr) opts_.events->Wake();
}

void LinkManager::ForgetFds() {
  for (auto& entry : conns_) entry.second->fd = -1;
  listenFd_ = -1;
  epollFd_ = -1;
}

}  // namespace ras
}  // namespace mori

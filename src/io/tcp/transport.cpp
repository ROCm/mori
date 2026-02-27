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
#include "src/io/tcp/transport.hpp"

#include <limits>

namespace mori {
namespace io {

namespace {

struct LinearReqView {
  uint64_t opId{0};
  uint32_t memId{0};
  uint64_t remoteOff{0};
  uint64_t size{0};
  uint8_t lanesTotal{1};
};

bool ParseLinearReq(const uint8_t* body, size_t len, LinearReqView* out) {
  if (out == nullptr) return false;
  size_t off = 0;
  if (!tcp::ReadU64BE(body, len, &off, &out->opId) ||
      !tcp::ReadU32BE(body, len, &off, &out->memId) ||
      !tcp::ReadU64BE(body, len, &off, &out->remoteOff) ||
      !tcp::ReadU64BE(body, len, &off, &out->size)) {
    return false;
  }
  if (off < len) out->lanesTotal = body[off];
  out->lanesTotal = ClampLanesTotal(out->lanesTotal);
  return true;
}

struct BatchReqView {
  uint64_t opId{0};
  uint32_t memId{0};
  std::vector<Segment> segs;
  uint8_t lanesTotal{1};
};

bool ParseBatchReq(const uint8_t* body, size_t len, BatchReqView* out) {
  if (out == nullptr) return false;
  size_t off = 0;
  uint32_t n = 0;
  if (!tcp::ReadU64BE(body, len, &off, &out->opId) ||
      !tcp::ReadU32BE(body, len, &off, &out->memId) || !tcp::ReadU32BE(body, len, &off, &n)) {
    return false;
  }

  out->segs.clear();
  out->segs.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    uint64_t remoteOff = 0;
    uint64_t size = 0;
    if (!tcp::ReadU64BE(body, len, &off, &remoteOff) || !tcp::ReadU64BE(body, len, &off, &size)) {
      return false;
    }
    if (size > 0) out->segs.push_back({remoteOff, size});
  }

  if (off < len) out->lanesTotal = body[off];
  out->lanesTotal = ClampLanesTotal(out->lanesTotal);
  return true;
}

struct CompletionView {
  uint64_t opId{0};
  uint32_t statusCode{0};
  std::string msg;
};

bool ParseCompletion(const uint8_t* body, size_t len, CompletionView* out) {
  if (out == nullptr) return false;
  size_t off = 0;
  uint32_t msgLen = 0;
  if (!tcp::ReadU64BE(body, len, &off, &out->opId) ||
      !tcp::ReadU32BE(body, len, &off, &out->statusCode) ||
      !tcp::ReadU32BE(body, len, &off, &msgLen)) {
    return false;
  }
  if (off + msgLen > len) return false;
  out->msg.assign(reinterpret_cast<const char*>(body + off), msgLen);
  return true;
}

bool SegmentsInRange(const std::vector<Segment>& segs, uint64_t memSize) {
  for (const auto& seg : segs) {
    if (seg.off + seg.len > memSize) return false;
  }
  return true;
}

}  // namespace

TcpTransport::TcpTransport(EngineKey myKey, const IOEngineConfig& engCfg,
                           const TcpBackendConfig& cfg)
    : myEngKey(std::move(myKey)), engConfig(engCfg), config(cfg) {}

TcpTransport::~TcpTransport() { Shutdown(); }

void TcpTransport::Start() {
  if (running.load()) return;

  epfd = epoll_create1(EPOLL_CLOEXEC);
  assert(epfd >= 0);

  listenFd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  assert(listenFd >= 0);

  int one = 1;
  SetSockOptOrLog(listenFd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one), "SO_REUSEADDR");

  auto addrOpt =
      ParseIpv4(engConfig.host.empty() ? std::string("0.0.0.0") : engConfig.host, engConfig.port);
  assert(addrOpt.has_value());
  sockaddr_in addr = addrOpt.value();
  if (bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    MORI_IO_ERROR("TCP: bind {}:{} failed: {}", engConfig.host, engConfig.port, strerror(errno));
    assert(false && "bind failed");
  }
  if (listen(listenFd, 256) != 0) {
    MORI_IO_ERROR("TCP: listen failed: {}", strerror(errno));
    assert(false && "listen failed");
  }
  listenPort = GetBoundPort(listenFd);
  MORI_IO_INFO("TCP: listen on {}:{} (port={})", engConfig.host, engConfig.port, listenPort);

  wakeFd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  assert(wakeFd >= 0);

  AddEpoll(listenFd, true, false);
  AddEpoll(wakeFd, true, false);

  running.store(true);
  ioThread = std::thread([this] { this->IoLoop(); });
}

void TcpTransport::Shutdown() {
  bool wasRunning = running.exchange(false);
  if (!wasRunning) return;

  if (wakeFd >= 0) {
    uint64_t one = 1;
    ::write(wakeFd, &one, sizeof(one));
  }
  if (ioThread.joinable()) ioThread.join();

  for (auto& kv : dataWorkers) kv.second->Stop();
  dataWorkers.clear();
  workerNotifyMap.clear();

  for (auto& kv : conns) CloseConnInternal(kv.second.get());
  conns.clear();
  peers.clear();

  if (listenFd >= 0) {
    close(listenFd);
    listenFd = -1;
  }
  if (wakeFd >= 0) {
    close(wakeFd);
    wakeFd = -1;
  }
  if (epfd >= 0) {
    close(epfd);
    epfd = -1;
  }
}

std::optional<uint16_t> TcpTransport::GetListenPort() const {
  if (listenPort == 0) return std::nullopt;
  return listenPort;
}

void TcpTransport::RegisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lock(remoteMu);
  remoteEngines[desc.key] = desc;
}

void TcpTransport::DeregisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lock(remoteMu);
  remoteEngines.erase(desc.key);
}

void TcpTransport::RegisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems[desc.id] = desc;
}

void TcpTransport::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lock(memMu);
  localMems.erase(desc.id);
}

bool TcpTransport::PopInboundTransferStatus(const EngineKey& remote, TransferUniqueId id,
                                            TransferStatus* status) {
  std::lock_guard<std::mutex> lock(inboundMu);
  auto it = inboundStatus.find(remote);
  if (it == inboundStatus.end()) return false;
  auto it2 = it->second.find(id);
  if (it2 == it->second.end()) return false;
  status->Update(it2->second.code, it2->second.msg);
  it->second.erase(it2);
  return true;
}

void TcpTransport::SubmitReadWrite(const MemoryDesc& local, size_t localOffset,
                                   const MemoryDesc& remote, size_t remoteOffset, size_t size,
                                   TransferStatus* status, TransferUniqueId id, bool isRead) {
  if (status == nullptr) return;
  if (size == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  if ((localOffset + size) > local.size || (remoteOffset + size) > remote.size) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: offset+size out of range");
    return;
  }

  auto op = std::make_unique<OutboundOpState>();
  op->peer = remote.engineKey;
  op->id = id;
  op->isRead = isRead;
  op->status = status;
  op->local = local;
  op->remote = remote;
  op->localSegs = {{static_cast<uint64_t>(localOffset), static_cast<uint64_t>(size)}};
  op->remoteSegs = {{static_cast<uint64_t>(remoteOffset), static_cast<uint64_t>(size)}};
  op->expectedRxBytes = isRead ? static_cast<uint64_t>(size) : 0;
  op->startTs = Clock::now();

  status->SetCode(StatusCode::IN_PROGRESS);
  EnqueueOp(std::move(op));
}

void TcpTransport::SubmitBatchReadWrite(const MemoryDesc& local, const SizeVec& localOffsets,
                                        const MemoryDesc& remote, const SizeVec& remoteOffsets,
                                        const SizeVec& sizes, TransferStatus* status,
                                        TransferUniqueId id, bool isRead) {
  if (status == nullptr) return;
  const size_t n = sizes.size();
  if (n == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  if (localOffsets.size() != n || remoteOffsets.size() != n) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: batch vector size mismatch");
    return;
  }

  std::vector<Segment> localSegs, remoteSegs;
  localSegs.reserve(n);
  remoteSegs.reserve(n);
  uint64_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    const size_t lo = localOffsets[i], ro = remoteOffsets[i], sz = sizes[i];
    if (sz == 0) continue;
    if ((lo + sz) > local.size || (ro + sz) > remote.size) {
      status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: batch offset+size out of range");
      return;
    }
    localSegs.push_back({static_cast<uint64_t>(lo), static_cast<uint64_t>(sz)});
    remoteSegs.push_back({static_cast<uint64_t>(ro), static_cast<uint64_t>(sz)});
    total += static_cast<uint64_t>(sz);
  }

  if (localSegs.size() > 1) {
    std::vector<Segment> newLocal, newRemote;
    newLocal.reserve(localSegs.size());
    newRemote.reserve(remoteSegs.size());
    Segment curL = localSegs[0], curR = remoteSegs[0];
    for (size_t i = 1; i < localSegs.size(); ++i) {
      const Segment& l = localSegs[i];
      const Segment& r = remoteSegs[i];
      if ((curL.off + curL.len == l.off) && (curR.off + curR.len == r.off) &&
          (curL.len == curR.len) && (l.len == r.len)) {
        curL.len += l.len;
        curR.len += r.len;
      } else {
        newLocal.push_back(curL);
        newRemote.push_back(curR);
        curL = l;
        curR = r;
      }
    }
    newLocal.push_back(curL);
    newRemote.push_back(curR);
    localSegs = std::move(newLocal);
    remoteSegs = std::move(newRemote);
  }

  auto op = std::make_unique<OutboundOpState>();
  op->peer = remote.engineKey;
  op->id = id;
  op->isRead = isRead;
  op->status = status;
  op->local = local;
  op->remote = remote;
  op->localSegs = std::move(localSegs);
  op->remoteSegs = std::move(remoteSegs);
  op->expectedRxBytes = isRead ? total : 0;
  op->startTs = Clock::now();

  status->SetCode(StatusCode::IN_PROGRESS);
  EnqueueOp(std::move(op));
}

void TcpTransport::EnqueueOp(std::unique_ptr<OutboundOpState> op) {
  {
    std::lock_guard<std::mutex> lock(submitMu);
    submitQ.push_back(std::move(op));
  }
  uint64_t one = 1;
  ::write(wakeFd, &one, sizeof(one));
}

void TcpTransport::AddEpoll(int fd, bool wantRead, bool wantWrite) {
  epoll_event ev{};
  ev.data.fd = fd;
  ev.events = EPOLLET | (wantRead ? EPOLLIN : 0) | (wantWrite ? EPOLLOUT : 0);
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev));
}

void TcpTransport::ModEpoll(int fd, bool wantRead, bool wantWrite) {
  epoll_event ev{};
  ev.data.fd = fd;
  ev.events = EPOLLET | (wantRead ? EPOLLIN : 0) | (wantWrite ? EPOLLOUT : 0);
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_MOD, fd, &ev));
}

void TcpTransport::DelEpoll(int fd) { epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr); }

void TcpTransport::CloseConnInternal(Connection* c) {
  if (c == nullptr) return;
  if (c->fd >= 0) {
    DelEpoll(c->fd);
    shutdown(c->fd, SHUT_RDWR);
    close(c->fd);
    c->fd = -1;
  }
}

bool TcpTransport::PreferOutgoingFor(const EngineKey& peerKey) const { return myEngKey < peerKey; }

void TcpTransport::AssignConnToPeer(Connection* c) {
  assert(c && c->helloReceived);
  PeerLinks& link = peers[c->peerKey];
  const bool preferOutgoing = PreferOutgoingFor(c->peerKey);
  const bool wasOutgoing = c->isOutgoing;
  const tcp::Channel ch = c->ch;

  if (wasOutgoing) {
    if (ch == tcp::Channel::CTRL) {
      if (link.ctrlPending > 0) link.ctrlPending--;
    } else {
      if (link.dataPending > 0) link.dataPending--;
    }
  }

  auto replace_if_needed = [&](int& slotFd) {
    if (slotFd < 0) {
      slotFd = c->fd;
      return;
    }
    const int existingFd = slotFd;
    const int newFd = c->fd;
    Connection* existing = conns[existingFd].get();
    if (!existing) {
      slotFd = newFd;
      return;
    }
    const bool keepNew = (preferOutgoing && c->isOutgoing) || (!preferOutgoing && !c->isOutgoing);
    if (keepNew) {
      MORI_IO_WARN("TCP: peer {} channel {} replacing fd {} with fd {}", c->peerKey,
                   static_cast<int>(c->ch), existing->fd, c->fd);
      CloseConnInternal(existing);
      conns.erase(existingFd);
      slotFd = newFd;
    } else {
      MORI_IO_WARN("TCP: peer {} channel {} dropping duplicate fd {}", c->peerKey,
                   static_cast<int>(c->ch), c->fd);
      CloseConnInternal(c);
      conns.erase(newFd);
    }
  };

  if (ch == tcp::Channel::CTRL) {
    replace_if_needed(link.ctrlFd);
    return;
  }

  const bool keepPreferred =
      (preferOutgoing && c->isOutgoing) || (!preferOutgoing && !c->isOutgoing);
  if (!keepPreferred) {
    MORI_IO_TRACE("TCP: peer {} dropping non-preferred DATA fd {} outgoing={}", c->peerKey, c->fd,
                  c->isOutgoing);
    const int fd = c->fd;
    CloseConnInternal(c);
    conns.erase(fd);
    return;
  }

  const size_t want = static_cast<size_t>(std::max(1, config.numDataConns));
  if (link.dataFds.size() >= want) {
    MORI_IO_TRACE("TCP: peer {} dropping extra DATA fd {} (have {} want {})", c->peerKey, c->fd,
                  link.dataFds.size(), want);
    const int fd = c->fd;
    CloseConnInternal(c);
    conns.erase(fd);
    return;
  }

  const int dataFd = c->fd;
  link.dataFds.push_back(dataFd);
  MORI_IO_TRACE("TCP: peer {} DATA conn up {}/{}", c->peerKey.c_str(), link.dataFds.size(), want);

  DelEpoll(dataFd);
  SetNonBlocking(dataFd);
  ConfigureDataSocket(dataFd, config);

  auto worker = std::make_unique<DataConnectionWorker>(dataFd, c->peerKey, &staging);
  worker->Start();
  AddEpoll(worker->NotifyFd(), true, false);
  workerNotifyMap[worker->NotifyFd()] = worker.get();
  link.workers.push_back(worker.get());
  dataWorkers[dataFd] = std::move(worker);
}

void TcpTransport::MaybeDispatchQueuedOps(const EngineKey& peerKey) {
  auto it = peers.find(peerKey);
  if (it == peers.end()) return;
  if (!it->second.CtrlUp() || !it->second.DataUp()) return;
  Connection* ctrl = conns[it->second.ctrlFd].get();
  if (!ctrl || !ctrl->helloReceived) return;
  if (it->second.workers.empty()) return;

  auto qit = waitingOps.find(peerKey);
  if (qit == waitingOps.end()) return;

  auto ops = std::move(qit->second);
  waitingOps.erase(qit);
  MORI_IO_TRACE("TCP: peer {} ready, dispatch {} queued ops", peerKey.c_str(), ops.size());
  for (auto& op : ops) DispatchOp(std::move(op));
}

void TcpTransport::EnsurePeerChannels(const EngineKey& peerKey) {
  PeerLinks& link = peers[peerKey];
  if (!link.CtrlUp() && link.ctrlPending == 0) ConnectChannel(peerKey, tcp::Channel::CTRL);
  const int want = std::max(1, config.numDataConns);
  while (static_cast<int>(link.dataFds.size()) + link.dataPending < want)
    ConnectChannel(peerKey, tcp::Channel::DATA);
}

void TcpTransport::ConnectChannel(const EngineKey& peerKey, tcp::Channel ch) {
  EngineDesc desc;
  {
    std::lock_guard<std::mutex> lock(remoteMu);
    auto it = remoteEngines.find(peerKey);
    if (it == remoteEngines.end()) {
      MORI_IO_ERROR("TCP: remote engine {} not registered", peerKey.c_str());
      return;
    }
    desc = it->second;
  }

  auto peerAddrOpt = ParseIpv4(desc.host, static_cast<uint16_t>(desc.port));
  if (!peerAddrOpt.has_value()) {
    MORI_IO_ERROR("TCP: invalid remote host {}:{}", desc.host, desc.port);
    return;
  }
  sockaddr_in peerAddr = peerAddrOpt.value();

  int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    MORI_IO_ERROR("TCP: socket() failed: {}", strerror(errno));
    return;
  }
  MORI_IO_TRACE("TCP: connect start peer={} ch={} fd={}", peerKey.c_str(), static_cast<int>(ch),
                fd);

  if (!engConfig.host.empty()) {
    auto localAddrOpt = ParseIpv4(engConfig.host, 0);
    if (localAddrOpt.has_value()) {
      sockaddr_in localAddr = localAddrOpt.value();
      if (bind(fd, reinterpret_cast<sockaddr*>(&localAddr), sizeof(localAddr)) != 0) {
        MORI_IO_WARN("TCP: bind(local) {} failed: {}", engConfig.host, strerror(errno));
      }
    }
  }

  int rc = connect(fd, reinterpret_cast<sockaddr*>(&peerAddr), sizeof(peerAddr));
  bool connecting = false;
  if (rc != 0) {
    if (errno == EINPROGRESS) {
      connecting = true;
    } else {
      MORI_IO_ERROR("TCP: connect to {}:{} failed: {}", desc.host, desc.port, strerror(errno));
      close(fd);
      return;
    }
  }

  auto conn = std::make_unique<Connection>();
  conn->fd = fd;
  conn->isOutgoing = true;
  conn->connecting = connecting;
  conn->peerKey = peerKey;
  conn->ch = ch;
  conn->inbuf.reserve(4096);

  if (ch == tcp::Channel::CTRL) ConfigureCtrlSocket(fd, config);

  const bool wantWrite = connecting || !conn->sendq.empty();
  AddEpoll(fd, true, wantWrite);
  conns[fd] = std::move(conn);

  PeerLinks& link = peers[peerKey];
  if (ch == tcp::Channel::CTRL)
    link.ctrlPending++;
  else
    link.dataPending++;

  if (!connecting) {
    QueueHello(fd);
    ModEpoll(fd, true, true);
  }
}

void TcpTransport::QueueHello(int fd) {
  Connection* c = conns[fd].get();
  if (!c || c->helloSent) return;
  c->helloSent = true;
  auto hello = tcp::BuildHello(c->ch, myEngKey);
  MORI_IO_TRACE("TCP: queue HELLO fd={} ch={} peer={}", fd, static_cast<int>(c->ch),
                c->peerKey.c_str());

  SendItem item;
  item.header = std::move(hello);
  item.iov.resize(1);
  item.iov[0].iov_base = item.header.data();
  item.iov[0].iov_len = item.header.size();
  c->sendq.push_back(std::move(item));
}

void TcpTransport::AcceptNew() {
  while (true) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    int fd =
        accept4(listenFd, reinterpret_cast<sockaddr*>(&peer), &len, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      if (IsWouldBlock(errno)) break;
      MORI_IO_WARN("TCP: accept failed: {}", strerror(errno));
      break;
    }
    MORI_IO_TRACE("TCP: accept fd={}", fd);

    auto conn = std::make_unique<Connection>();
    conn->fd = fd;
    conn->isOutgoing = false;
    conn->inbuf.reserve(4096);
    AddEpoll(fd, true, false);
    conns[fd] = std::move(conn);
  }
}

void TcpTransport::DrainWakeFd() {
  uint64_t v = 0;
  while (true) {
    ssize_t n = ::read(wakeFd, &v, sizeof(v));
    if (n <= 0) break;
  }

  std::deque<std::unique_ptr<OutboundOpState>> ops;
  {
    std::lock_guard<std::mutex> lock(submitMu);
    ops.swap(submitQ);
  }

  for (auto& op : ops) {
    EnsurePeerChannels(op->peer);
    if (!IsPeerReady(op->peer)) {
      waitingOps[op->peer].push_back(std::move(op));
      continue;
    }
    DispatchOp(std::move(op));
  }
}

bool TcpTransport::IsPeerReady(const EngineKey& peerKey) {
  auto it = peers.find(peerKey);
  if (it == peers.end()) return false;
  if (!it->second.CtrlUp() || !it->second.DataUp()) return false;
  Connection* ctrl = conns[it->second.ctrlFd].get();
  if (!ctrl || !ctrl->helloReceived) return false;
  return !it->second.workers.empty();
}

void TcpTransport::RegisterRecvTargetWithWorkers(const EngineKey& peerKey, TransferUniqueId opId,
                                                 const WorkerRecvTarget& target) {
  auto pit = peers.find(peerKey);
  if (pit == peers.end()) return;
  for (auto* w : pit->second.workers) w->RegisterRecvTarget(opId, target);
}

void TcpTransport::RemoveRecvTargetFromWorkers(const EngineKey& peerKey, TransferUniqueId opId) {
  auto pit = peers.find(peerKey);
  if (pit == peers.end()) return;
  for (auto* w : pit->second.workers) w->RemoveRecvTarget(opId);
}

void TcpTransport::DispatchOp(std::unique_ptr<OutboundOpState> op) {
  if (!op) {
    MORI_IO_ERROR("TCP: DispatchOp got null op");
    return;
  }
  const EngineKey peerKey = op->peer;
  auto it = peers.find(peerKey);
  if (it == peers.end() || !it->second.CtrlUp() || !it->second.DataUp()) {
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: peer not connected");
    return;
  }
  Connection* ctrl = conns[it->second.ctrlFd].get();
  if (!ctrl) {
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: peer ctrl connection missing");
    return;
  }
  auto& workerList = it->second.workers;
  if (workerList.empty()) {
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: no data workers for peer");
    return;
  }

  const TransferUniqueId opId = op->id;
  auto [itIns, inserted] = pendingOutbound.emplace(opId, std::move(op));
  if (!inserted) {
    MORI_IO_ERROR("TCP: duplicate outbound op id={} for peer={}", opId, peerKey.c_str());
    itIns->second->status->Update(StatusCode::ERR_BAD_STATE, "TCP: duplicate op id");
    pendingOutbound.erase(itIns);
    return;
  }
  OutboundOpState* st = itIns->second.get();
  if (!st) {
    MORI_IO_ERROR("TCP: failed to store outbound op id={}", opId);
    pendingOutbound.erase(itIns);
    return;
  }

  const uint64_t totalBytes = SumLens(st->localSegs);
  int wantLanes = std::max(1, config.numDataConns);
  wantLanes = std::min<int>(wantLanes, (1U << kLaneBits));
  uint8_t lanesTotal = 1;
  const bool canStripe = (wantLanes > 1) && (config.stripingThresholdBytes > 0) &&
                         (totalBytes >= static_cast<uint64_t>(config.stripingThresholdBytes)) &&
                         (st->localSegs.size() == 1) && (st->remoteSegs.size() == 1) &&
                         (workerList.size() >= 2);
  if (canStripe) {
    lanesTotal =
        static_cast<uint8_t>(std::min<size_t>(static_cast<size_t>(wantLanes), workerList.size()));
  }
  st->lanesTotal = lanesTotal;

  if (st->isRead && st->local.loc == MemoryLocationType::GPU) {
    st->pinned = staging.Acquire(static_cast<size_t>(totalBytes));
    if (!st->pinned) {
      st->status->Update(StatusCode::ERR_BAD_STATE, "TCP: failed to allocate pinned staging");
      pendingOutbound.erase(opId);
      return;
    }
  }

  if (st->isRead) {
    WorkerRecvTarget target;
    target.lanesTotal = lanesTotal;
    target.totalLen = totalBytes;
    target.discard = false;
    if (st->local.loc == MemoryLocationType::GPU) {
      target.toGpu = true;
      target.pinned = st->pinned;
    } else {
      target.toGpu = false;
      target.cpuBase = reinterpret_cast<void*>(st->local.data);
      target.segs = st->localSegs;
    }
    RegisterRecvTargetWithWorkers(peerKey, opId, target);
  }

  std::vector<uint8_t> ctrlFrame;
  if (st->localSegs.size() == 1) {
    if (st->isRead)
      ctrlFrame = tcp::BuildReadReq(st->id, st->remote.id, st->remoteSegs[0].off,
                                    st->remoteSegs[0].len, lanesTotal);
    else
      ctrlFrame = tcp::BuildWriteReq(st->id, st->remote.id, st->remoteSegs[0].off,
                                     st->remoteSegs[0].len, lanesTotal);
  } else {
    std::vector<uint64_t> roffs, szs;
    roffs.reserve(st->remoteSegs.size());
    szs.reserve(st->remoteSegs.size());
    for (const auto& s : st->remoteSegs) {
      roffs.push_back(s.off);
      szs.push_back(s.len);
    }
    if (st->isRead)
      ctrlFrame = tcp::BuildBatchReadReq(st->id, st->remote.id, roffs, szs, lanesTotal);
    else
      ctrlFrame = tcp::BuildBatchWriteReq(st->id, st->remote.id, roffs, szs, lanesTotal);
  }

  QueueSend(ctrl->fd, std::move(ctrlFrame));
  if (!st->isRead) QueueDataSendForWrite(workerList, *st);
  UpdateWriteInterest(ctrl->fd);
}

void TcpTransport::QueueSend(int fd, std::vector<uint8_t> bytes, std::function<void()> onDone) {
  Connection* c = conns[fd].get();
  if (!c) return;
  SendItem item;
  item.header = std::move(bytes);
  item.iov.resize(1);
  item.iov[0].iov_base = item.header.data();
  item.iov[0].iov_len = item.header.size();
  item.onDone = std::move(onDone);
  c->sendq.push_back(std::move(item));
}

void TcpTransport::QueueSegmentSend(DataConnectionWorker* worker, uint64_t wireOpId, uint8_t* base,
                                    const std::vector<Segment>& segs, uint64_t totalLen,
                                    std::function<void()> onDone) {
  SendItem item;
  item.header = tcp::BuildDataHeader(wireOpId, totalLen, 0);
  item.iov.reserve(1 + segs.size());
  item.iov.push_back({item.header.data(), item.header.size()});
  for (const auto& s : segs) item.iov.push_back({base + s.off, static_cast<size_t>(s.len)});
  item.onDone = std::move(onDone);
  worker->SubmitSend(std::move(item));
}

void TcpTransport::QueueStripedCpuSend(const std::vector<DataConnectionWorker*>& workers,
                                       uint64_t opId, uint8_t lanesTotal, uint8_t* base,
                                       uint64_t baseOff, uint64_t total,
                                       std::function<void()> onLaneDone) {
  for (uint8_t lane = 0; lane < lanesTotal; ++lane) {
    const LaneSpan span = ComputeLaneSpan(total, lanesTotal, lane);
    auto* worker = workers[lane % workers.size()];
    SendItem item;
    item.header = tcp::BuildDataHeader(ToWireOpId(opId, lane), span.len, 0);
    item.iov.resize(2);
    item.iov[0].iov_base = item.header.data();
    item.iov[0].iov_len = item.header.size();
    item.iov[1].iov_base = base + baseOff + span.off;
    item.iov[1].iov_len = static_cast<size_t>(span.len);
    item.onDone = onLaneDone;
    worker->SubmitSend(std::move(item));
  }
}

bool TcpTransport::ScheduleGpuDtoH(int deviceId, const MemoryDesc& src,
                                   const std::vector<Segment>& srcSegs,
                                   std::shared_ptr<PinnedBuf> pinned,
                                   std::function<void()> onComplete) {
  const uint64_t total = SumLens(srcSegs);
  hipStream_t stream = streamPool.GetNextStream(deviceId);
  hipEvent_t ev = eventPool.GetEvent(deviceId);
  if (stream == nullptr || ev == nullptr) {
    MORI_IO_ERROR("TCP: failed to get HIP stream/event for GPU DtoH");
    if (ev) eventPool.PutEvent(ev, deviceId);
    return false;
  }

  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));
  uint8_t* dst = reinterpret_cast<uint8_t*>(pinned->ptr);
  uint64_t spanOff = 0, spanLen = 0;
  if (IsSingleContiguousSpan(srcSegs, &spanOff, &spanLen) && spanLen == total) {
    hipDeviceptr_t gpuPtr = reinterpret_cast<hipDeviceptr_t>(src.data + spanOff);
    HIP_RUNTIME_CHECK(hipMemcpyDtoHAsync(dst, gpuPtr, static_cast<size_t>(total), stream));
  } else {
    uint64_t off = 0;
    for (const auto& s : srcSegs) {
      hipDeviceptr_t gpuPtr = reinterpret_cast<hipDeviceptr_t>(src.data + s.off);
      HIP_RUNTIME_CHECK(hipMemcpyDtoHAsync(dst + off, gpuPtr, static_cast<size_t>(s.len), stream));
      off += s.len;
    }
  }
  HIP_RUNTIME_CHECK(hipEventRecord(ev, stream));
  gpuTasks.push_back({deviceId, ev, std::move(onComplete)});
  return true;
}

bool TcpTransport::ScheduleGpuHtoD(int deviceId, const MemoryDesc& dst,
                                   const std::vector<Segment>& dstSegs,
                                   std::shared_ptr<PinnedBuf> pinned,
                                   std::function<void()> onComplete) {
  const uint64_t total = SumLens(dstSegs);
  hipStream_t stream = streamPool.GetNextStream(deviceId);
  hipEvent_t ev = eventPool.GetEvent(deviceId);
  if (stream == nullptr || ev == nullptr) {
    MORI_IO_ERROR("TCP: failed to get HIP stream/event for GPU HtoD");
    if (ev) eventPool.PutEvent(ev, deviceId);
    return false;
  }

  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));
  uint8_t* src = reinterpret_cast<uint8_t*>(pinned->ptr);
  uint64_t spanOff = 0, spanLen = 0;
  if (IsSingleContiguousSpan(dstSegs, &spanOff, &spanLen) && spanLen == total) {
    void* gpuPtr = reinterpret_cast<void*>(dst.data + spanOff);
    HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpuPtr, src, static_cast<size_t>(total), stream));
  } else {
    uint64_t off = 0;
    for (const auto& s : dstSegs) {
      void* gpuPtr = reinterpret_cast<void*>(dst.data + s.off);
      HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpuPtr, src + off, static_cast<size_t>(s.len), stream));
      off += s.len;
    }
  }
  HIP_RUNTIME_CHECK(hipEventRecord(ev, stream));
  gpuTasks.push_back({deviceId, ev, std::move(onComplete)});
  return true;
}

void TcpTransport::QueueGpuSend(const std::vector<DataConnectionWorker*>& workers, uint64_t opId,
                                uint8_t lanesTotal, const MemoryDesc& src,
                                const std::vector<Segment>& srcSegs,
                                std::function<void()> onLaneDone) {
  const uint64_t total = SumLens(srcSegs);
  auto pinned = staging.Acquire(static_cast<size_t>(total));
  if (!pinned) {
    MORI_IO_ERROR("TCP: failed to allocate pinned staging for GPU send");
    return;
  }

  auto sendCallback = [workers, pinned, opId, lanesTotal, total,
                       onLaneDone = std::move(onLaneDone)]() {
    for (uint8_t lane = 0; lane < lanesTotal; ++lane) {
      const LaneSpan span = ComputeLaneSpan(total, lanesTotal, lane);
      auto* worker = workers[lane % workers.size()];
      SendItem item;
      item.header = tcp::BuildDataHeader(ToWireOpId(opId, lane), span.len, 0);
      item.iov.resize(2);
      item.iov[0].iov_base = item.header.data();
      item.iov[0].iov_len = item.header.size();
      item.iov[1].iov_base = static_cast<uint8_t*>(pinned->ptr) + static_cast<size_t>(span.off);
      item.iov[1].iov_len = static_cast<size_t>(span.len);
      item.keepalive = pinned;
      item.onDone = onLaneDone;
      worker->SubmitSend(std::move(item));
    }
  };

  ScheduleGpuDtoH(src.deviceId, src, srcSegs, pinned, std::move(sendCallback));
}

void TcpTransport::QueueDataSendForWrite(const std::vector<DataConnectionWorker*>& workerList,
                                         OutboundOpState& st) {
  if (workerList.empty()) return;
  uint8_t lanesTotal = std::max<uint8_t>(1, st.lanesTotal);

  if (lanesTotal > 1 && st.localSegs.size() != 1) {
    MORI_IO_WARN("TCP: striping requested but localSegs.size={}, fallback to 1 lane",
                 st.localSegs.size());
    lanesTotal = 1;
    st.lanesTotal = 1;
  }

  QueueDataSendCommon(workerList, st.local, st.localSegs, st.id, lanesTotal);
}

void TcpTransport::QueueDataSendForRead(const EngineKey& peer, uint64_t opId, const MemoryDesc& src,
                                        const std::vector<Segment>& srcSegs, uint8_t lanesTotal) {
  auto pit = peers.find(peer);
  if (pit == peers.end()) return;
  auto& workerList = pit->second.workers;
  if (workerList.empty()) return;

  struct DoneState {
    EngineKey peer;
    uint64_t opId{0};
    std::atomic<int> remaining{0};
  };
  auto done = std::make_shared<DoneState>();
  done->peer = peer;
  done->opId = opId;
  uint8_t useLanes = std::min<uint8_t>(
      ClampLanesTotal(lanesTotal), static_cast<uint8_t>(std::max<size_t>(1, workerList.size())));
  if (useLanes > 1 && srcSegs.size() != 1) useLanes = 1;
  done->remaining.store(useLanes);
  auto laneDone = [this, done]() mutable {
    if (done->remaining.fetch_sub(1) > 1) return;
    SendCompletionAndRecord(done->peer, done->opId, StatusCode::SUCCESS, "");
  };

  QueueDataSendCommon(workerList, src, srcSegs, opId, useLanes, std::move(laneDone));
}

void TcpTransport::QueueDataSendCommon(const std::vector<DataConnectionWorker*>& workerList,
                                       const MemoryDesc& src, const std::vector<Segment>& srcSegs,
                                       uint64_t opId, uint8_t lanesTotal,
                                       std::function<void()> onLaneDone) {
  if (workerList.empty()) return;
  const uint64_t total = SumLens(srcSegs);
  lanesTotal = ClampLanesTotal(lanesTotal);
  lanesTotal = std::min<uint8_t>(lanesTotal, static_cast<uint8_t>(workerList.size()));
  if (lanesTotal > 1 && srcSegs.size() != 1) {
    MORI_IO_WARN("TCP: striping requested for {} segments, fallback to 1 lane", srcSegs.size());
    lanesTotal = 1;
  }

  if (src.loc == MemoryLocationType::GPU) {
    std::vector<DataConnectionWorker*> workers(workerList.begin(), workerList.begin() + lanesTotal);
    QueueGpuSend(workers, opId, lanesTotal, src, srcSegs, std::move(onLaneDone));
    return;
  }

  uint8_t* base = reinterpret_cast<uint8_t*>(src.data);
  if (lanesTotal == 1) {
    QueueSegmentSend(workerList[0], ToWireOpId(opId, 0), base, srcSegs, total,
                     std::move(onLaneDone));
  } else {
    QueueStripedCpuSend(workerList, opId, lanesTotal, base, srcSegs[0].off, total,
                        std::move(onLaneDone));
  }
}

void TcpTransport::PollGpuTasks() {
  for (auto it = gpuTasks.begin(); it != gpuTasks.end();) {
    hipError_t st = hipEventQuery(it->ev);
    if (st == hipSuccess) {
      eventPool.PutEvent(it->ev, it->deviceId);
      if (it->onReady) it->onReady();
      it = gpuTasks.erase(it);
    } else if (st == hipErrorNotReady) {
      ++it;
    } else {
      MORI_IO_ERROR("TCP: hipEventQuery failed: {}", hipGetErrorString(st));
      eventPool.PutEvent(it->ev, it->deviceId);
      it = gpuTasks.erase(it);
    }
  }
}

void TcpTransport::UpdateWriteInterest(int fd) {
  auto it = conns.find(fd);
  if (it == conns.end()) return;
  Connection* c = it->second.get();
  if (!c || c->fd < 0) return;

  if (!c->connecting && !c->sendq.empty()) {
    FlushSend(c);
    it = conns.find(fd);
    if (it == conns.end()) return;
    c = it->second.get();
    if (!c || c->fd < 0) return;
  }
  ModEpoll(fd, true, c->connecting || !c->sendq.empty());
}

void TcpTransport::HandleConnWritable(Connection* c) {
  if (c->connecting) {
    int err = 0;
    socklen_t len = sizeof(err);
    if (getsockopt(c->fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0) {
      MORI_IO_ERROR("TCP: connect failed fd {}: {}", c->fd, strerror(err == 0 ? errno : err));
      ClosePeerByFd(c->fd);
      return;
    }
    c->connecting = false;
    QueueHello(c->fd);
  }
  UpdateWriteInterest(c->fd);
}

void TcpTransport::FlushSend(Connection* c) {
  constexpr size_t kMaxIov = 64;
  while (!c->sendq.empty()) {
    SendItem& item = c->sendq.front();
    if (item.Done()) {
      auto cb = std::move(item.onDone);
      c->sendq.pop_front();
      if (cb) cb();
      continue;
    }

    iovec iov[kMaxIov];
    size_t cnt = 0;
    for (size_t i = item.idx; i < item.iov.size() && cnt < kMaxIov; ++i) {
      iov[cnt] = item.iov[i];
      if (i == item.idx && item.off > 0) {
        iov[cnt].iov_base = static_cast<uint8_t*>(iov[cnt].iov_base) + item.off;
        iov[cnt].iov_len -= item.off;
      }
      cnt++;
    }
    msghdr msg{};
    msg.msg_iov = iov;
    msg.msg_iovlen = cnt;
    ssize_t n = sendmsg(c->fd, &msg, MSG_NOSIGNAL | item.flags);
    if (n < 0) {
      if (IsWouldBlock(errno)) break;
      MORI_IO_ERROR("TCP: sendmsg fd {} failed: {}", c->fd, strerror(errno));
      ClosePeerByFd(c->fd);
      break;
    }
    if (n == 0) break;
    item.Advance(static_cast<size_t>(n));
  }
}

void TcpTransport::CloseAndRemoveFd(int fd) {
  if (fd < 0) return;
  auto wit = dataWorkers.find(fd);
  if (wit != dataWorkers.end()) {
    wit->second->Stop();
    auto nit = workerNotifyMap.find(wit->second->NotifyFd());
    if (nit != workerNotifyMap.end()) {
      DelEpoll(nit->first);
      workerNotifyMap.erase(nit);
    }
    dataWorkers.erase(wit);
  }
  auto it = conns.find(fd);
  if (it != conns.end()) {
    CloseConnInternal(it->second.get());
    conns.erase(it);
  }
}

EngineKey TcpTransport::FindPeerByFd(int fd) {
  for (auto& kv : peers) {
    if (kv.second.ctrlFd == fd) return kv.first;
    for (int dfd : kv.second.dataFds) {
      if (dfd == fd) return kv.first;
    }
  }
  return {};
}

void TcpTransport::ClosePeerByFd(int fd) {
  MORI_IO_TRACE("TCP: close fd={}", fd);
  EngineKey peer = FindPeerByFd(fd);
  if (!peer.empty()) {
    ClosePeerByKey(peer, "TCP: connection lost");
  } else {
    CloseAndRemoveFd(fd);
  }
}

void TcpTransport::ClosePeerByKey(const EngineKey& peer, const std::string& reason) {
  auto pit = peers.find(peer);
  if (pit == peers.end()) return;
  auto link = pit->second;
  CloseAndRemoveFd(link.ctrlFd);
  for (int dfd : link.dataFds) CloseAndRemoveFd(dfd);
  peers.erase(peer);
  FailPendingOpsForPeer(peer, reason);
}

void TcpTransport::FailPendingOpsForPeer(const EngineKey& peer, const std::string& msg) {
  for (auto it = pendingOutbound.begin(); it != pendingOutbound.end();) {
    if (it->second->peer == peer) {
      it->second->status->Update(StatusCode::ERR_BAD_STATE, msg);
      it = pendingOutbound.erase(it);
    } else {
      ++it;
    }
  }
  waitingOps.erase(peer);
  inboundWrites.erase(peer);
  earlyWrites.erase(peer);
}

void TcpTransport::HandleCtrlReadable(Connection* c) {
  while (true) {
    uint8_t tmp[16384];
    ssize_t n = ::recv(c->fd, tmp, sizeof(tmp), 0);
    if (n < 0) {
      if (IsWouldBlock(errno)) break;
      MORI_IO_ERROR("TCP: recv(ctrl) fd {} failed: {}", c->fd, strerror(errno));
      ClosePeerByFd(c->fd);
      return;
    }
    if (n == 0) {
      ClosePeerByFd(c->fd);
      return;
    }
    c->inbuf.insert(c->inbuf.end(), tmp, tmp + n);
  }

  while (true) {
    tcp::CtrlHeaderView hv;
    if (!tcp::TryParseCtrlHeader(c->inbuf.data(), c->inbuf.size(), &hv)) {
      if (c->inbuf.size() >= tcp::kCtrlHeaderSize) {
        MORI_IO_ERROR("TCP: bad ctrl header on fd {}, closing", c->fd);
        ClosePeerByFd(c->fd);
      }
      break;
    }
    if (c->inbuf.size() < tcp::kCtrlHeaderSize + hv.bodyLen) break;

    const uint8_t* body = c->inbuf.data() + tcp::kCtrlHeaderSize;
    HandleCtrlFrame(c, hv.type, body, hv.bodyLen);
    c->inbuf.erase(c->inbuf.begin(), c->inbuf.begin() + tcp::kCtrlHeaderSize + hv.bodyLen);

    if (c->helloReceived && c->ch == tcp::Channel::DATA) return;
  }
}

void TcpTransport::HandleCtrlFrame(Connection* c, tcp::CtrlMsgType type, const uint8_t* body,
                                   size_t len) {
  if (type == tcp::CtrlMsgType::HELLO) {
    HandleHello(c, body, len);
    return;
  }
  if (!c->helloReceived) {
    MORI_IO_WARN("TCP: received ctrl message before HELLO, dropping");
    return;
  }
  switch (type) {
    case tcp::CtrlMsgType::WRITE_REQ:
      HandleWriteReq(c->peerKey, body, len);
      break;
    case tcp::CtrlMsgType::READ_REQ:
      HandleReadReq(c->peerKey, body, len);
      break;
    case tcp::CtrlMsgType::BATCH_WRITE_REQ:
      HandleBatchWriteReq(c->peerKey, body, len);
      break;
    case tcp::CtrlMsgType::BATCH_READ_REQ:
      HandleBatchReadReq(c->peerKey, body, len);
      break;
    case tcp::CtrlMsgType::COMPLETION:
      HandleCompletion(c->peerKey, body, len);
      break;
    default:
      MORI_IO_WARN("TCP: unknown ctrl msg type {}", static_cast<uint32_t>(type));
  }
}

void TcpTransport::HandleHello(Connection* c, const uint8_t* body, size_t len) {
  if (len < 1 + 4) {
    MORI_IO_WARN("TCP: bad HELLO len {}", len);
    ClosePeerByFd(c->fd);
    return;
  }
  size_t off = 0;
  const uint8_t chRaw = body[off++];
  uint32_t keyLen = 0;
  if (!tcp::ReadU32BE(body, len, &off, &keyLen)) {
    ClosePeerByFd(c->fd);
    return;
  }
  if (off + keyLen > len) {
    ClosePeerByFd(c->fd);
    return;
  }
  EngineKey peerKey(reinterpret_cast<const char*>(body + off), keyLen);
  off += keyLen;

  c->peerKey = peerKey;
  c->ch =
      (chRaw == static_cast<uint8_t>(tcp::Channel::DATA)) ? tcp::Channel::DATA : tcp::Channel::CTRL;
  c->helloReceived = true;
  MORI_IO_TRACE("TCP: recv HELLO fd={} peer={} ch={} outgoing={}", c->fd, c->peerKey.c_str(),
                static_cast<int>(c->ch), c->isOutgoing);

  if (!c->helloSent) {
    QueueHello(c->fd);
    UpdateWriteInterest(c->fd);
  }
  if (c->ch == tcp::Channel::CTRL) ConfigureCtrlSocket(c->fd, config);
  AssignConnToPeer(c);
  MaybeDispatchQueuedOps(peerKey);
}

std::optional<MemoryDesc> TcpTransport::LookupLocalMem(MemoryUniqueId id) {
  std::lock_guard<std::mutex> lock(memMu);
  auto it = localMems.find(id);
  if (it == localMems.end()) return std::nullopt;
  return it->second;
}

void TcpTransport::RecordInboundStatus(const EngineKey& peer, TransferUniqueId id, StatusCode code,
                                       const std::string& msg) {
  std::lock_guard<std::mutex> lock(inboundMu);
  inboundStatus[peer][id] = InboundStatusEntry{code, msg};
}

void TcpTransport::SendCompletionAndRecord(const EngineKey& peer, TransferUniqueId opId,
                                           StatusCode code, const std::string& msg) {
  Connection* ctrl = PeerCtrl(peer);
  if (ctrl != nullptr) {
    QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(code), msg));
    UpdateWriteInterest(ctrl->fd);
  }
  RecordInboundStatus(peer, opId, code, msg);
}

Connection* TcpTransport::PeerCtrl(const EngineKey& peer) {
  auto it = peers.find(peer);
  if (it == peers.end() || !it->second.CtrlUp()) return nullptr;
  return conns[it->second.ctrlFd].get();
}

void TcpTransport::FinalizeInboundWriteSetup(const EngineKey& peer, TransferUniqueId opId,
                                             InboundWriteState& ws) {
  if (!ws.discard && ws.dst.loc == MemoryLocationType::GPU) {
    const uint64_t total = SumLens(ws.dstSegs);
    ws.pinned = staging.Acquire(static_cast<size_t>(total));
    if (!ws.pinned) ws.discard = true;
  }
  inboundWrites[peer][opId] = ws;
  SetupInboundWriteWorkerTarget(peer, opId, ws);
  TryConsumeEarlyWriteLanes(peer, opId);
}

void TcpTransport::SetupInboundWriteWorkerTarget(const EngineKey& peer, TransferUniqueId opId,
                                                 const InboundWriteState& ws) {
  WorkerRecvTarget target;
  target.lanesTotal = ws.lanesTotal;
  target.totalLen = SumLens(ws.dstSegs);
  target.discard = ws.discard;
  if (!ws.discard && ws.dst.loc == MemoryLocationType::GPU) {
    target.toGpu = true;
    target.pinned = ws.pinned;
  } else if (!ws.discard) {
    target.toGpu = false;
    target.cpuBase = reinterpret_cast<void*>(ws.dst.data);
    target.segs = ws.dstSegs;
  }
  RegisterRecvTargetWithWorkers(peer, opId, target);
}

void TcpTransport::HandleWriteReq(const EngineKey& peer, const uint8_t* body, size_t len) {
  LinearReqView req;
  if (!ParseLinearReq(body, len, &req)) {
    MORI_IO_WARN("TCP: malformed WRITE_REQ");
    return;
  }
  HandleWriteReqSegments(peer, req.opId, req.memId, {{req.remoteOff, req.size}}, req.lanesTotal);
}

void TcpTransport::HandleWriteReqSegments(const EngineKey& peer, TransferUniqueId opId,
                                          MemoryUniqueId memId, std::vector<Segment> segs,
                                          uint8_t lanesTotal) {
  auto memOpt = LookupLocalMem(memId);
  InboundWriteState ws;
  ws.peer = peer;
  ws.id = opId;
  ws.lanesTotal = lanesTotal;
  ws.discard = true;
  if (memOpt.has_value()) {
    ws.dst = memOpt.value();
    if (SegmentsInRange(segs, ws.dst.size)) {
      ws.discard = false;
      ws.dstSegs = std::move(segs);
    }
  }
  FinalizeInboundWriteSetup(peer, opId, ws);
}

void TcpTransport::HandleBatchWriteReq(const EngineKey& peer, const uint8_t* body, size_t len) {
  BatchReqView req;
  if (!ParseBatchReq(body, len, &req)) {
    MORI_IO_WARN("TCP: malformed BATCH_WRITE_REQ");
    return;
  }
  HandleWriteReqSegments(peer, req.opId, req.memId, std::move(req.segs), req.lanesTotal);
}

void TcpTransport::HandleReadReq(const EngineKey& peer, const uint8_t* body, size_t len) {
  LinearReqView req;
  if (!ParseLinearReq(body, len, &req)) {
    MORI_IO_WARN("TCP: malformed READ_REQ");
    return;
  }
  HandleReadReqSegments(peer, req.opId, req.memId, {{req.remoteOff, req.size}}, req.lanesTotal,
                        false);
}

void TcpTransport::HandleReadReqSegments(const EngineKey& peer, TransferUniqueId opId,
                                         MemoryUniqueId memId, std::vector<Segment> segs,
                                         uint8_t lanesTotal, bool batchReq) {
  const StatusCode badRangeCode =
      batchReq ? StatusCode::ERR_INVALID_ARGS : StatusCode::ERR_NOT_FOUND;
  const char* badRangeMsg =
      batchReq ? "TCP: batch read out of range" : "TCP: remote mem not found/out of range";
  const char* notFoundMsg =
      batchReq ? "TCP: remote mem not found" : "TCP: remote mem not found/out of range";
  auto memOpt = LookupLocalMem(memId);
  if (!memOpt.has_value()) {
    SendCompletionAndRecord(peer, opId, StatusCode::ERR_NOT_FOUND, notFoundMsg);
    return;
  }
  MemoryDesc src = memOpt.value();
  if (!SegmentsInRange(segs, src.size)) {
    SendCompletionAndRecord(peer, opId, badRangeCode, badRangeMsg);
    return;
  }
  QueueDataSendForRead(peer, opId, src, segs, lanesTotal);
}

void TcpTransport::HandleBatchReadReq(const EngineKey& peer, const uint8_t* body, size_t len) {
  BatchReqView req;
  if (!ParseBatchReq(body, len, &req)) {
    MORI_IO_WARN("TCP: malformed BATCH_READ_REQ");
    return;
  }
  HandleReadReqSegments(peer, req.opId, req.memId, std::move(req.segs), req.lanesTotal, true);
}

void TcpTransport::HandleCompletion(const EngineKey& peer, const uint8_t* body, size_t len) {
  CompletionView msg;
  if (!ParseCompletion(body, len, &msg)) {
    MORI_IO_WARN("TCP: malformed COMPLETION");
    return;
  }

  auto it = pendingOutbound.find(msg.opId);
  if (it == pendingOutbound.end()) return;
  OutboundOpState& st = *it->second;
  st.completionReceived = true;
  st.completionCode = static_cast<StatusCode>(msg.statusCode);
  st.completionMsg = std::move(msg.msg);

  if (st.completionCode != StatusCode::SUCCESS) {
    RemoveRecvTargetFromWorkers(st.peer, msg.opId);
    st.status->Update(st.completionCode, st.completionMsg);
    pendingOutbound.erase(it);
    return;
  }
  MaybeCompleteOutbound(st);
}

void TcpTransport::MaybeCompleteOutbound(OutboundOpState& st) {
  if (!st.completionReceived) return;
  if (st.isRead) {
    const uint8_t allMask = LanesAllMask(st.lanesTotal);
    if (st.lanesDoneMask != allMask) return;
    if (st.rxBytes != st.expectedRxBytes) return;
    if (st.gpuCopyPending) return;
  }
  RemoveRecvTargetFromWorkers(st.peer, st.id);
  st.status->Update(StatusCode::SUCCESS, "");
  pendingOutbound.erase(st.id);
}

void TcpTransport::MaybeFinalizeInboundWrite(const EngineKey& peer, TransferUniqueId opId) {
  auto iwPeerIt = inboundWrites.find(peer);
  if (iwPeerIt == inboundWrites.end()) return;
  auto wsIt = iwPeerIt->second.find(opId);
  if (wsIt == iwPeerIt->second.end()) return;

  InboundWriteState& ws = wsIt->second;
  ws.lanesTotal = ClampLanesTotal(ws.lanesTotal);
  const uint8_t allMask = LanesAllMask(ws.lanesTotal);
  if ((ws.lanesDoneMask & allMask) != allMask) return;

  RemoveRecvTargetFromWorkers(peer, opId);

  if (ws.discard) {
    SendCompletionAndRecord(peer, opId, StatusCode::ERR_INVALID_ARGS, "TCP: write discarded");
  } else if (ws.dst.loc == MemoryLocationType::GPU) {
    if (!ws.pinned) {
      SendCompletionAndRecord(peer, opId, StatusCode::ERR_BAD_STATE,
                              "TCP: missing pinned staging (write)");
    } else {
      auto pinnedRef = ws.pinned;
      bool ok = ScheduleGpuHtoD(ws.dst.deviceId, ws.dst, ws.dstSegs, pinnedRef,
                                [this, peer, opId, pinnedRef]() {
                                  SendCompletionAndRecord(peer, opId, StatusCode::SUCCESS, "");
                                });
      if (!ok) {
        SendCompletionAndRecord(peer, opId, StatusCode::ERR_BAD_STATE,
                                "TCP: failed HIP stream/event");
      }
    }
  } else {
    SendCompletionAndRecord(peer, opId, StatusCode::SUCCESS, "");
  }

  iwPeerIt->second.erase(wsIt);
  if (iwPeerIt->second.empty()) inboundWrites.erase(iwPeerIt);
  auto ewPeerIt = earlyWrites.find(peer);
  if (ewPeerIt != earlyWrites.end()) {
    ewPeerIt->second.erase(opId);
    if (ewPeerIt->second.empty()) earlyWrites.erase(ewPeerIt);
  }
}

void TcpTransport::TryConsumeEarlyWriteLanes(const EngineKey& peer, TransferUniqueId opId) {
  auto iwPeerIt = inboundWrites.find(peer);
  if (iwPeerIt == inboundWrites.end()) return;
  auto wsIt = iwPeerIt->second.find(opId);
  if (wsIt == iwPeerIt->second.end()) return;
  InboundWriteState& ws = wsIt->second;
  ws.lanesTotal = ClampLanesTotal(ws.lanesTotal);

  auto ewPeerIt = earlyWrites.find(peer);
  if (ewPeerIt == earlyWrites.end()) return;
  auto ewIt = ewPeerIt->second.find(opId);
  if (ewIt == ewPeerIt->second.end()) return;

  EarlyWriteState& early = ewIt->second;
  const uint64_t total = SumLens(ws.dstSegs);
  uint8_t* dstBase = reinterpret_cast<uint8_t*>(ws.dst.data);

  for (auto it = early.lanes.begin(); it != early.lanes.end();) {
    const uint8_t lane = it->first;
    EarlyWriteLaneState& laneState = it->second;
    if (!laneState.complete) {
      ++it;
      continue;
    }

    if (lane >= ws.lanesTotal) ws.discard = true;
    const LaneSpan span = ComputeLaneSpan(total, ws.lanesTotal, lane);
    if (span.len != laneState.payloadLen) ws.discard = true;

    if (!ws.discard && laneState.pinned) {
      uint8_t* src = reinterpret_cast<uint8_t*>(laneState.pinned->ptr);
      if (ws.dst.loc == MemoryLocationType::GPU) {
        if (!ws.pinned) {
          ws.pinned = staging.Acquire(static_cast<size_t>(total));
          if (!ws.pinned) ws.discard = true;
        }
        if (!ws.discard && ws.pinned) {
          std::memcpy(reinterpret_cast<uint8_t*>(ws.pinned->ptr) + span.off, src,
                      static_cast<size_t>(span.len));
        }
      } else {
        const auto segs = SliceSegments(ws.dstSegs, span.off, span.len);
        uint64_t copied = 0;
        for (const auto& s : segs) {
          std::memcpy(dstBase + s.off, src + copied, static_cast<size_t>(s.len));
          copied += s.len;
        }
      }
    }

    if (lane < 8) ws.lanesDoneMask |= static_cast<uint8_t>(1U << lane);
    it = early.lanes.erase(it);
  }

  if (early.lanes.empty()) {
    ewPeerIt->second.erase(ewIt);
    if (ewPeerIt->second.empty()) earlyWrites.erase(ewPeerIt);
  }
  MaybeFinalizeInboundWrite(peer, opId);
}

void TcpTransport::ProcessEventsFrom(DataConnectionWorker* worker) {
  std::deque<WorkerEvent> events;
  worker->DrainEvents(events);
  for (auto& ev : events) {
    switch (ev.type) {
      case WorkerEventType::RECV_DONE:
        HandleWorkerRecvDone(ev);
        break;
      case WorkerEventType::EARLY_DATA:
        HandleWorkerEarlyData(ev);
        break;
      case WorkerEventType::SEND_CALLBACK:
        if (ev.callback) ev.callback();
        break;
      case WorkerEventType::CONN_ERROR:
        MORI_IO_WARN("TCP: worker error for peer {}: {}", ev.peerKey, ev.errorMsg);
        ClosePeerByKey(ev.peerKey, ev.errorMsg);
        break;
    }
  }
}

void TcpTransport::ProcessWorkerEvents() {
  std::vector<int> notifyFds;
  notifyFds.reserve(workerNotifyMap.size());
  for (const auto& kv : workerNotifyMap) notifyFds.push_back(kv.first);
  for (int notifyFd : notifyFds) {
    auto it = workerNotifyMap.find(notifyFd);
    if (it == workerNotifyMap.end()) continue;
    ProcessEventsFrom(it->second);
  }
}

void TcpTransport::HandleWorkerRecvDone(const WorkerEvent& ev) {
  const EngineKey& peer = ev.peerKey;
  const TransferUniqueId opId = ev.opId;
  const uint8_t lane = ev.lane;
  const uint64_t laneLen = ev.laneLen;

  auto iwPeerIt = inboundWrites.find(peer);
  if (iwPeerIt != inboundWrites.end()) {
    auto wsIt = iwPeerIt->second.find(opId);
    if (wsIt != iwPeerIt->second.end()) {
      InboundWriteState& ws = wsIt->second;
      ws.lanesTotal = ClampLanesTotal(ws.lanesTotal);
      if (lane < 8) ws.lanesDoneMask |= static_cast<uint8_t>(1U << lane);
      MaybeFinalizeInboundWrite(peer, opId);
      return;
    }
  }

  auto obIt = pendingOutbound.find(opId);
  if (obIt == pendingOutbound.end()) return;
  OutboundOpState& st = *obIt->second;
  st.lanesTotal = ClampLanesTotal(st.lanesTotal);
  const uint8_t bit = static_cast<uint8_t>(1U << lane);
  if ((st.lanesDoneMask & bit) == 0) {
    st.lanesDoneMask |= bit;
    st.rxBytes += laneLen;
  }

  if (st.local.loc == MemoryLocationType::GPU) {
    const uint8_t allMask = LanesAllMask(st.lanesTotal);
    if ((st.lanesDoneMask & allMask) != allMask) {
      MaybeCompleteOutbound(st);
      return;
    }
    if (st.gpuCopyPending) return;
    if (!st.pinned) {
      st.status->Update(StatusCode::ERR_BAD_STATE, "TCP: missing pinned staging (read)");
      RemoveRecvTargetFromWorkers(st.peer, opId);
      pendingOutbound.erase(obIt);
      return;
    }

    st.gpuCopyPending = true;
    auto pinnedRef = st.pinned;
    bool ok = ScheduleGpuHtoD(st.local.deviceId, st.local, st.localSegs, pinnedRef,
                              [this, opId, pinnedRef]() {
                                auto it2 = pendingOutbound.find(opId);
                                if (it2 == pendingOutbound.end()) return;
                                it2->second->gpuCopyPending = false;
                                MaybeCompleteOutbound(*it2->second);
                              });
    if (!ok) {
      st.status->Update(StatusCode::ERR_BAD_STATE, "TCP: failed HIP stream/event (read)");
      RemoveRecvTargetFromWorkers(st.peer, opId);
      pendingOutbound.erase(obIt);
    }
    return;
  }

  MaybeCompleteOutbound(st);
}

void TcpTransport::HandleWorkerEarlyData(const WorkerEvent& ev) {
  const EngineKey& peer = ev.peerKey;
  const TransferUniqueId opId = ev.opId;
  const uint8_t lane = ev.lane;

  auto& perPeer = earlyWrites[peer];
  EarlyWriteState& early = perPeer[opId];
  if (early.lanes.find(lane) != early.lanes.end()) {
    MORI_IO_WARN("TCP: duplicate early data for op {} lane {} from peer {}", opId, (uint32_t)lane,
                 peer);
    return;
  }
  early.lanes.emplace(lane, EarlyWriteLaneState{ev.laneLen, ev.earlyBuf, true});
  TryConsumeEarlyWriteLanes(peer, opId);
}

void TcpTransport::ScanTimeouts() {
  if (config.opTimeoutMs <= 0) return;
  const auto now = Clock::now();
  const auto timeout = std::chrono::milliseconds(config.opTimeoutMs);
  for (auto it = pendingOutbound.begin(); it != pendingOutbound.end();) {
    if ((now - it->second->startTs) > timeout) {
      RemoveRecvTargetFromWorkers(it->second->peer, it->first);
      it->second->status->Update(StatusCode::ERR_BAD_STATE, "TCP: op timeout");
      it = pendingOutbound.erase(it);
    } else {
      ++it;
    }
  }
}

void TcpTransport::IoLoop() {
  constexpr int kMaxEvents = 128;
  epoll_event events[kMaxEvents];

  while (running.load()) {
    PollGpuTasks();
    ProcessWorkerEvents();
    ScanTimeouts();

    const bool hasActive =
        !gpuTasks.empty() || !pendingOutbound.empty() || !workerNotifyMap.empty();
    const int timeoutMs = hasActive ? 0 : 2;
    int nfds = epoll_wait(epfd, events, kMaxEvents, timeoutMs);
    if (nfds < 0) {
      if (errno == EINTR) continue;
      MORI_IO_ERROR("TCP: epoll_wait failed: {}", strerror(errno));
      break;
    }

    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;
      uint32_t evMask = events[i].events;

      if (fd == listenFd) {
        AcceptNew();
        continue;
      }
      if (fd == wakeFd) {
        DrainWakeFd();
        continue;
      }

      auto wnit = workerNotifyMap.find(fd);
      if (wnit != workerNotifyMap.end()) {
        ProcessEventsFrom(wnit->second);
        continue;
      }

      Connection* c = nullptr;
      auto it = conns.find(fd);
      if (it != conns.end()) c = it->second.get();
      if (!c) continue;

      if (evMask & (EPOLLERR | EPOLLHUP)) {
        ClosePeerByFd(fd);
        continue;
      }

      if (evMask & EPOLLIN) {
        HandleCtrlReadable(c);
        auto it2 = conns.find(fd);
        if (it2 == conns.end()) continue;
        c = it2->second.get();
        if (!c) continue;
      }
      if (evMask & EPOLLOUT) HandleConnWritable(c);
    }
  }
}

}  // namespace io
}  // namespace mori

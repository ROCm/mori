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

// ===========================================================================
// PinnedStagingPool
// ===========================================================================
std::shared_ptr<PinnedBuf> PinnedStagingPool::Acquire(size_t size) {
  const size_t cap = RoundUp(size);
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = free_.find(cap);
    if (it != free_.end() && !it->second.empty()) {
      void* p = it->second.back();
      it->second.pop_back();
      return std::shared_ptr<PinnedBuf>(new PinnedBuf{p, cap},
                                        [this](PinnedBuf* b) { Release(b); });
    }
  }
  void* p = nullptr;
  if (hipHostMalloc(&p, cap, hipHostMallocDefault) != hipSuccess) {
    MORI_IO_ERROR("TCP: hipHostMalloc({}) failed", cap);
    return nullptr;
  }
  return std::shared_ptr<PinnedBuf>(new PinnedBuf{p, cap}, [this](PinnedBuf* b) { Release(b); });
}

void PinnedStagingPool::Clear() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& kv : free_)
    for (void* p : kv.second) hipHostFree(p);
  free_.clear();
}

void PinnedStagingPool::Release(PinnedBuf* b) {
  if (!b) return;
  constexpr size_t kMaxCached = 8;
  size_t cap = b->cap;
  void* p = b->ptr;
  delete b;
  std::lock_guard<std::mutex> lk(mu_);
  auto& vec = free_[cap];
  if (vec.size() < kMaxCached)
    vec.push_back(p);
  else
    hipHostFree(p);
}

// ===========================================================================
// SendItem
// ===========================================================================
void SendItem::Advance(size_t n) {
  while (n > 0 && idx < iov.size()) {
    size_t avail = iov[idx].iov_len - off;
    if (n < avail) {
      off += n;
      return;
    }
    n -= avail;
    idx++;
    off = 0;
  }
}

// ===========================================================================
// DataConnectionWorker
// ===========================================================================
DataConnectionWorker::DataConnectionWorker(int fd, EngineKey peer, PinnedStagingPool* staging)
    : fd_(fd), peerKey_(std::move(peer)), staging_(staging) {
  notifyFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  wakeFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
}

DataConnectionWorker::~DataConnectionWorker() {
  Stop();
  if (notifyFd_ >= 0) close(notifyFd_);
  if (wakeFd_ >= 0) close(wakeFd_);
}

void DataConnectionWorker::Start() {
  if (running_.load()) return;
  running_.store(true);
  thread_ = std::thread(&DataConnectionWorker::Run, this);
}

void DataConnectionWorker::Stop() {
  if (!running_.exchange(false)) return;
  WakeWorker();
  if (thread_.joinable()) thread_.join();
}

void DataConnectionWorker::SubmitSend(SendItem item) {
  {
    std::lock_guard<std::mutex> lk(sendMu_);
    sendQ_.push_back(std::move(item));
  }
  WakeWorker();
}

void DataConnectionWorker::RegisterRecvTarget(TransferUniqueId opId,
                                              const WorkerRecvTarget& target) {
  std::lock_guard<std::mutex> lk(targetMu_);
  recvTargets_[opId] = target;
}

void DataConnectionWorker::RemoveRecvTarget(TransferUniqueId opId) {
  std::lock_guard<std::mutex> lk(targetMu_);
  recvTargets_.erase(opId);
}

void DataConnectionWorker::DrainEvents(std::deque<WorkerEvent>& out) {
  uint64_t v;
  while (::read(notifyFd_, &v, sizeof(v)) > 0) {
  }
  std::lock_guard<std::mutex> lk(eventMu_);
  while (!eventQ_.empty()) {
    out.push_back(std::move(eventQ_.front()));
    eventQ_.pop_front();
  }
}

void DataConnectionWorker::WakeWorker() {
  uint64_t one = 1;
  ::write(wakeFd_, &one, sizeof(one));
}

void DataConnectionWorker::NotifyMain() {
  uint64_t one = 1;
  ::write(notifyFd_, &one, sizeof(one));
}

void DataConnectionWorker::PostEvent(WorkerEvent ev) {
  {
    std::lock_guard<std::mutex> lk(eventMu_);
    eventQ_.push_back(std::move(ev));
  }
  NotifyMain();
}

void DataConnectionWorker::Run() {
  MORI_IO_TRACE("TCP: DataWorker fd={} peer={} started", fd_, peerKey_);
  pollfd pfds[2];
  pfds[0].fd = fd_;
  pfds[1].fd = wakeFd_;
  pfds[1].events = POLLIN;

  while (running_.load()) {
    bool hasSend;
    {
      std::lock_guard<std::mutex> lk(sendMu_);
      hasSend = !sendQ_.empty();
    }

    pfds[0].events = POLLIN | (hasSend ? POLLOUT : 0);
    pfds[0].revents = pfds[1].revents = 0;

    int n = ::poll(pfds, 2, hasSend ? 0 : 1);
    if (n < 0) {
      if (errno == EINTR) continue;
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 std::string("poll failed: ") + strerror(errno)});
      break;
    }

    if (pfds[1].revents & POLLIN) {
      uint64_t v;
      while (::read(wakeFd_, &v, sizeof(v)) > 0) {
      }
    }
    if (pfds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 "data connection error/hangup"});
      break;
    }
    if ((pfds[0].revents & POLLOUT) && !ProcessSend()) break;
    if ((pfds[0].revents & POLLIN) && !ProcessRecv()) break;
  }
  MORI_IO_TRACE("TCP: DataWorker fd={} peer={} exiting", fd_, peerKey_);
}

bool DataConnectionWorker::ProcessSend() {
  std::deque<SendItem> batch;
  {
    std::lock_guard<std::mutex> lk(sendMu_);
    batch.swap(sendQ_);
  }

  for (auto& item : batch) {
    while (!item.Done()) {
      constexpr size_t kMaxIov = 64;
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
      ssize_t n = ::sendmsg(fd_, &msg, MSG_NOSIGNAL | item.flags);
      if (n < 0) {
        if (IsWouldBlock(errno)) goto requeue;
        PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                   std::string("sendmsg failed: ") + strerror(errno)});
        return false;
      }
      if (n == 0) goto requeue;
      item.Advance(static_cast<size_t>(n));
    }
    if (item.onDone) {
      WorkerEvent ev;
      ev.type = WorkerEventType::SEND_CALLBACK;
      ev.callback = std::move(item.onDone);
      PostEvent(std::move(ev));
    }
  }
  return true;

requeue: {
  std::lock_guard<std::mutex> lk(sendMu_);
  for (auto rit = batch.rbegin(); rit != batch.rend(); ++rit)
    if (!rit->Done()) sendQ_.push_front(std::move(*rit));
}
  return true;
}

bool DataConnectionWorker::ProcessRecv() {
  while (true) {
    // Read data header
    while (hdrGot_ < tcp::kDataHeaderSize) {
      ssize_t n = ::recv(fd_, hdrBuf_ + hdrGot_, tcp::kDataHeaderSize - hdrGot_, 0);
      if (n < 0) {
        if (IsWouldBlock(errno)) return true;
        PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                   std::string("recv header failed: ") + strerror(errno)});
        return false;
      }
      if (n == 0) {
        PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                   "data connection closed by peer"});
        return false;
      }
      hdrGot_ += static_cast<size_t>(n);
    }
    hdrGot_ = 0;

    tcp::DataHeaderView hv;
    if (!tcp::TryParseDataHeader(hdrBuf_, tcp::kDataHeaderSize, &hv)) {
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 "bad data header"});
      return false;
    }

    const uint8_t lane = static_cast<uint8_t>(hv.opId & kLaneMask);
    const TransferUniqueId userOpId = static_cast<TransferUniqueId>(ToUserOpId(hv.opId));
    const uint64_t payloadLen = hv.payloadLen;

    // Look up recv target
    WorkerRecvTarget target;
    bool hasTarget = false;
    {
      std::lock_guard<std::mutex> lk(targetMu_);
      auto it = recvTargets_.find(userOpId);
      if (it != recvTargets_.end()) {
        target = it->second;
        hasTarget = true;
      }
    }

    auto postRecvDone = [&](bool discarded = false) {
      WorkerEvent ev;
      ev.type = WorkerEventType::RECV_DONE;
      ev.peerKey = peerKey_;
      ev.opId = userOpId;
      ev.lane = lane;
      ev.laneLen = payloadLen;
      ev.discarded = discarded;
      PostEvent(std::move(ev));
    };

    if (hasTarget && !target.discard) {
      const LaneSpan span = ComputeLaneSpan(target.totalLen, target.lanesTotal, lane);
      if (span.len != payloadLen) {
        MORI_IO_WARN("TCP: worker recv op {} lane {} len mismatch expected={} got={}", userOpId,
                     (uint32_t)lane, span.len, payloadLen);
        if (!DiscardPayload(payloadLen)) return false;
        postRecvDone(true);
      } else if (target.toGpu) {
        if (!RecvExact(reinterpret_cast<uint8_t*>(target.pinned->ptr) + span.off, payloadLen))
          return false;
        postRecvDone();
      } else {
        if (!RecvIntoSegments(reinterpret_cast<uint8_t*>(target.cpuBase),
                              SliceSegments(target.segs, span.off, span.len), payloadLen))
          return false;
        postRecvDone();
      }
    } else if (hasTarget) {
      if (!DiscardPayload(payloadLen)) return false;
      postRecvDone(true);
    } else {
      // Early data: no target registered yet
      if (payloadLen == 0) {
        WorkerEvent ev;
        ev.type = WorkerEventType::EARLY_DATA;
        ev.peerKey = peerKey_;
        ev.opId = userOpId;
        ev.lane = lane;
        ev.laneLen = 0;
        PostEvent(std::move(ev));
      } else {
        auto buf = staging_->Acquire(static_cast<size_t>(payloadLen));
        if (!buf) {
          if (!DiscardPayload(payloadLen)) return false;
          postRecvDone(true);
        } else {
          if (!RecvExact(reinterpret_cast<uint8_t*>(buf->ptr), payloadLen)) return false;
          WorkerEvent ev;
          ev.type = WorkerEventType::EARLY_DATA;
          ev.peerKey = peerKey_;
          ev.opId = userOpId;
          ev.lane = lane;
          ev.laneLen = payloadLen;
          ev.earlyBuf = std::move(buf);
          PostEvent(std::move(ev));
        }
      }
    }
  }
  return true;
}

bool DataConnectionWorker::RecvExact(uint8_t* dst, uint64_t len) {
  uint64_t got = 0;
  while (got < len) {
    ssize_t n =
        ::recv(fd_, dst + got, static_cast<size_t>(std::min<uint64_t>(len - got, 16ULL << 20)), 0);
    if (n < 0) {
      if (IsWouldBlock(errno)) continue;
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 std::string("recv payload failed: ") + strerror(errno)});
      return false;
    }
    if (n == 0) {
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 "data connection closed during recv"});
      return false;
    }
    got += static_cast<uint64_t>(n);
  }
  return true;
}

bool DataConnectionWorker::RecvIntoSegments(uint8_t* base, const std::vector<Segment>& segs,
                                            uint64_t totalLen) {
  uint64_t remaining = totalLen;
  size_t segIdx = 0;
  uint64_t segOff = 0;
  while (remaining > 0 && segIdx < segs.size()) {
    const Segment& seg = segs[segIdx];
    size_t want = static_cast<size_t>(
        std::min<uint64_t>(remaining, std::min<uint64_t>(seg.len - segOff, 16ULL << 20)));
    ssize_t n = ::recv(fd_, base + seg.off + segOff, want, 0);
    if (n < 0) {
      if (IsWouldBlock(errno)) continue;
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 std::string("recv seg failed: ") + strerror(errno)});
      return false;
    }
    if (n == 0) {
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 "data connection closed during seg recv"});
      return false;
    }
    remaining -= static_cast<uint64_t>(n);
    segOff += static_cast<uint64_t>(n);
    if (segOff >= seg.len) {
      segIdx++;
      segOff = 0;
    }
  }
  return (remaining == 0);
}

bool DataConnectionWorker::DiscardPayload(uint64_t len) {
  uint8_t tmp[65536];
  uint64_t remaining = len;
  while (remaining > 0) {
    ssize_t n =
        ::recv(fd_, tmp, static_cast<size_t>(std::min<uint64_t>(remaining, sizeof(tmp))), 0);
    if (n < 0) {
      if (IsWouldBlock(errno)) continue;
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 std::string("recv discard failed: ") + strerror(errno)});
      return false;
    }
    if (n == 0) {
      PostEvent({WorkerEventType::CONN_ERROR, peerKey_, 0, 0, 0, false, nullptr, nullptr,
                 "data connection closed during discard"});
      return false;
    }
    remaining -= static_cast<uint64_t>(n);
  }
  return true;
}

// ===========================================================================
// Request parsing helpers (anonymous namespace)
// ===========================================================================
namespace {

struct RequestView {
  uint64_t opId{0};
  uint32_t memId{0};
  std::vector<Segment> segs;
  uint8_t lanesTotal{1};
};

// Parse either a linear or batch request into a uniform RequestView
bool ParseRequest(tcp::CtrlMsgType type, const uint8_t* body, size_t len, RequestView* out) {
  tcp::WireReader r{body, len};
  if (!r.u64(&out->opId) || !r.u32(&out->memId)) return false;

  bool isBatch =
      (type == tcp::CtrlMsgType::BATCH_WRITE_REQ || type == tcp::CtrlMsgType::BATCH_READ_REQ);
  if (isBatch) {
    uint32_t n = 0;
    if (!r.u32(&n)) return false;
    out->segs.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      uint64_t off, sz;
      if (!r.u64(&off) || !r.u64(&sz)) return false;
      if (sz > 0) out->segs.push_back({off, sz});
    }
  } else {
    uint64_t off, sz;
    if (!r.u64(&off) || !r.u64(&sz)) return false;
    out->segs.push_back({off, sz});
  }
  if (r.off < r.len) {
    uint8_t lt;
    if (r.u8(&lt)) out->lanesTotal = lt;
  }
  out->lanesTotal = ClampLanesTotal(out->lanesTotal);
  return true;
}

struct CompletionView {
  uint64_t opId{0};
  uint32_t statusCode{0};
  std::string msg;
};

bool ParseCompletion(const uint8_t* body, size_t len, CompletionView* out) {
  tcp::WireReader r{body, len};
  uint32_t msgLen = 0;
  if (!r.u64(&out->opId) || !r.u32(&out->statusCode) || !r.u32(&msgLen)) return false;
  if (r.off + msgLen > r.len) return false;
  out->msg.assign(reinterpret_cast<const char*>(body + r.off), msgLen);
  return true;
}

}  // namespace

// ===========================================================================
// TcpTransport
// ===========================================================================
TcpTransport::TcpTransport(EngineKey myKey, const IOEngineConfig& engCfg,
                           const TcpBackendConfig& cfg)
    : myEngKey_(std::move(myKey)), engConfig_(engCfg), config_(cfg) {}

TcpTransport::~TcpTransport() { Shutdown(); }

void TcpTransport::Start() {
  if (running_.load()) return;

  epfd_ = epoll_create1(EPOLL_CLOEXEC);
  assert(epfd_ >= 0);

  listenFd_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  assert(listenFd_ >= 0);

  int one = 1;
  SetSockOpt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one), "SO_REUSEADDR");

  auto addrOpt = ParseIpv4(engConfig_.host.empty() ? "0.0.0.0" : engConfig_.host, engConfig_.port);
  assert(addrOpt.has_value());
  sockaddr_in addr = *addrOpt;
  if (bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    MORI_IO_ERROR("TCP: bind {}:{} failed: {}", engConfig_.host, engConfig_.port, strerror(errno));
    assert(false && "bind failed");
  }
  if (listen(listenFd_, 256) != 0) {
    MORI_IO_ERROR("TCP: listen failed: {}", strerror(errno));
    assert(false && "listen failed");
  }
  listenPort_ = GetBoundPort(listenFd_);
  MORI_IO_INFO("TCP: listen on {}:{} (port={})", engConfig_.host, engConfig_.port, listenPort_);

  wakeFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  assert(wakeFd_ >= 0);

  AddEpoll(listenFd_, true, false);
  AddEpoll(wakeFd_, true, false);

  running_.store(true);
  ioThread_ = std::thread([this] { IoLoop(); });
}

void TcpTransport::Shutdown() {
  if (!running_.exchange(false)) return;
  if (wakeFd_ >= 0) {
    uint64_t one = 1;
    ::write(wakeFd_, &one, sizeof(one));
  }
  if (ioThread_.joinable()) ioThread_.join();

  for (auto& kv : dataWorkers_) kv.second->Stop();
  dataWorkers_.clear();
  workerNotifyMap_.clear();

  for (auto& kv : conns_) CloseConnInternal(kv.second.get());
  conns_.clear();
  peers_.clear();

  auto closeFd = [](int& fd) {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  };
  closeFd(listenFd_);
  closeFd(wakeFd_);
  closeFd(epfd_);
}

std::optional<uint16_t> TcpTransport::GetListenPort() const {
  return listenPort_ ? std::optional<uint16_t>(listenPort_) : std::nullopt;
}

void TcpTransport::RegisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lk(remoteMu_);
  remoteEngines_[desc.key] = desc;
}
void TcpTransport::DeregisterRemoteEngine(const EngineDesc& desc) {
  std::lock_guard<std::mutex> lk(remoteMu_);
  remoteEngines_.erase(desc.key);
}
void TcpTransport::RegisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lk(memMu_);
  localMems_[desc.id] = desc;
}
void TcpTransport::DeregisterMemory(const MemoryDesc& desc) {
  std::lock_guard<std::mutex> lk(memMu_);
  localMems_.erase(desc.id);
}

bool TcpTransport::PopInboundTransferStatus(const EngineKey& remote, TransferUniqueId id,
                                            TransferStatus* status) {
  std::lock_guard<std::mutex> lk(inboundMu_);
  auto it = inboundStatus_.find(remote);
  if (it == inboundStatus_.end()) return false;
  auto it2 = it->second.find(id);
  if (it2 == it->second.end()) return false;
  status->Update(it2->second.code, it2->second.msg);
  it->second.erase(it2);
  return true;
}

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------
void TcpTransport::SubmitReadWrite(const MemoryDesc& local, size_t localOffset,
                                   const MemoryDesc& remote, size_t remoteOffset, size_t size,
                                   TransferStatus* status, TransferUniqueId id, bool isRead) {
  if (!status) return;
  if (size == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  if (localOffset + size > local.size || remoteOffset + size > remote.size) {
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
  op->localSegs = {{uint64_t(localOffset), uint64_t(size)}};
  op->remoteSegs = {{uint64_t(remoteOffset), uint64_t(size)}};
  op->expectedRxBytes = isRead ? uint64_t(size) : 0;
  status->SetCode(StatusCode::IN_PROGRESS);
  EnqueueOp(std::move(op));
}

void TcpTransport::SubmitBatchReadWrite(const MemoryDesc& local, const SizeVec& localOffsets,
                                        const MemoryDesc& remote, const SizeVec& remoteOffsets,
                                        const SizeVec& sizes, TransferStatus* status,
                                        TransferUniqueId id, bool isRead) {
  if (!status) return;
  const size_t n = sizes.size();
  if (n == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }
  if (localOffsets.size() != n || remoteOffsets.size() != n) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: batch vector size mismatch");
    return;
  }

  std::vector<Segment> lSegs, rSegs;
  lSegs.reserve(n);
  rSegs.reserve(n);
  uint64_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    if (sizes[i] == 0) continue;
    if (localOffsets[i] + sizes[i] > local.size || remoteOffsets[i] + sizes[i] > remote.size) {
      status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: batch offset+size out of range");
      return;
    }
    lSegs.push_back({uint64_t(localOffsets[i]), uint64_t(sizes[i])});
    rSegs.push_back({uint64_t(remoteOffsets[i]), uint64_t(sizes[i])});
    total += sizes[i];
  }

  // Merge adjacent contiguous segments
  if (lSegs.size() > 1) {
    std::vector<Segment> ml, mr;
    ml.reserve(lSegs.size());
    mr.reserve(rSegs.size());
    Segment cl = lSegs[0], cr = rSegs[0];
    for (size_t i = 1; i < lSegs.size(); ++i) {
      if (cl.off + cl.len == lSegs[i].off && cr.off + cr.len == rSegs[i].off && cl.len == cr.len &&
          lSegs[i].len == rSegs[i].len) {
        cl.len += lSegs[i].len;
        cr.len += rSegs[i].len;
      } else {
        ml.push_back(cl);
        mr.push_back(cr);
        cl = lSegs[i];
        cr = rSegs[i];
      }
    }
    ml.push_back(cl);
    mr.push_back(cr);
    lSegs = std::move(ml);
    rSegs = std::move(mr);
  }

  auto op = std::make_unique<OutboundOpState>();
  op->peer = remote.engineKey;
  op->id = id;
  op->isRead = isRead;
  op->status = status;
  op->local = local;
  op->remote = remote;
  op->localSegs = std::move(lSegs);
  op->remoteSegs = std::move(rSegs);
  op->expectedRxBytes = isRead ? total : 0;
  status->SetCode(StatusCode::IN_PROGRESS);
  EnqueueOp(std::move(op));
}

void TcpTransport::EnqueueOp(std::unique_ptr<OutboundOpState> op) {
  {
    std::lock_guard<std::mutex> lk(submitMu_);
    submitQ_.push_back(std::move(op));
  }
  uint64_t one = 1;
  ::write(wakeFd_, &one, sizeof(one));
}

// ---------------------------------------------------------------------------
// Epoll helpers
// ---------------------------------------------------------------------------
void TcpTransport::AddEpoll(int fd, bool rd, bool wr) {
  epoll_event ev{};
  ev.data.fd = fd;
  ev.events = EPOLLET | (rd ? EPOLLIN : 0) | (wr ? EPOLLOUT : 0);
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, &ev));
}

void TcpTransport::ModEpoll(int fd, bool rd, bool wr) {
  epoll_event ev{};
  ev.data.fd = fd;
  ev.events = EPOLLET | (rd ? EPOLLIN : 0) | (wr ? EPOLLOUT : 0);
  SYSCALL_RETURN_ZERO(epoll_ctl(epfd_, EPOLL_CTL_MOD, fd, &ev));
}

void TcpTransport::DelEpoll(int fd) { epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, nullptr); }

void TcpTransport::CloseConnInternal(Connection* c) {
  if (!c || c->fd < 0) return;
  DelEpoll(c->fd);
  shutdown(c->fd, SHUT_RDWR);
  close(c->fd);
  c->fd = -1;
}

// ---------------------------------------------------------------------------
// Connection management
// ---------------------------------------------------------------------------
void TcpTransport::AssignConnToPeer(Connection* c) {
  assert(c && c->helloReceived);
  PeerLinks& link = peers_[c->peerKey];
  const bool preferOut = myEngKey_ < c->peerKey;

  if (c->isOutgoing) {
    if (c->ch == tcp::Channel::CTRL) {
      if (link.ctrlPending > 0) link.ctrlPending--;
    } else {
      if (link.dataPending > 0) link.dataPending--;
    }
  }

  if (c->ch == tcp::Channel::CTRL) {
    // Replace ctrl connection if needed
    if (link.ctrlFd < 0) {
      link.ctrlFd = c->fd;
      return;
    }
    int existFd = link.ctrlFd;
    auto eIt = conns_.find(existFd);
    if (eIt == conns_.end()) {
      link.ctrlFd = c->fd;
      return;
    }
    bool keepNew = (preferOut && c->isOutgoing) || (!preferOut && !c->isOutgoing);
    if (keepNew) {
      MORI_IO_WARN("TCP: peer {} CTRL replacing fd {} with {}", c->peerKey, existFd, c->fd);
      CloseConnInternal(eIt->second.get());
      conns_.erase(existFd);
      link.ctrlFd = c->fd;
    } else {
      MORI_IO_WARN("TCP: peer {} CTRL dropping duplicate fd {}", c->peerKey, c->fd);
      int fd = c->fd;
      CloseConnInternal(c);
      conns_.erase(fd);
    }
    return;
  }

  // DATA channel
  bool keepPref = (preferOut && c->isOutgoing) || (!preferOut && !c->isOutgoing);
  if (!keepPref) {
    MORI_IO_TRACE("TCP: peer {} dropping non-preferred DATA fd {}", c->peerKey, c->fd);
    int fd = c->fd;
    CloseConnInternal(c);
    conns_.erase(fd);
    return;
  }
  size_t want = static_cast<size_t>(std::max(1, config_.numDataConns));
  if (link.dataFds.size() >= want) {
    MORI_IO_TRACE("TCP: peer {} dropping extra DATA fd {}", c->peerKey, c->fd);
    int fd = c->fd;
    CloseConnInternal(c);
    conns_.erase(fd);
    return;
  }

  int dataFd = c->fd;
  link.dataFds.push_back(dataFd);
  MORI_IO_TRACE("TCP: peer {} DATA conn up {}/{}", c->peerKey, link.dataFds.size(), want);

  DelEpoll(dataFd);
  SetNonBlocking(dataFd);
  ConfigureDataSocket(dataFd, config_);

  auto worker = std::make_unique<DataConnectionWorker>(dataFd, c->peerKey, &staging_);
  worker->Start();
  AddEpoll(worker->NotifyFd(), true, false);
  workerNotifyMap_[worker->NotifyFd()] = worker.get();
  link.workers.push_back(worker.get());
  dataWorkers_[dataFd] = std::move(worker);
}

void TcpTransport::MaybeDispatchQueuedOps(const EngineKey& peer) {
  auto it = peers_.find(peer);
  if (it == peers_.end() || !it->second.CtrlUp() || !it->second.DataUp()) return;
  Connection* ctrl = conns_[it->second.ctrlFd].get();
  if (!ctrl || !ctrl->helloReceived || it->second.workers.empty()) return;

  auto qit = waitingOps_.find(peer);
  if (qit == waitingOps_.end()) return;
  auto ops = std::move(qit->second);
  waitingOps_.erase(qit);
  MORI_IO_TRACE("TCP: peer {} ready, dispatch {} queued ops", peer, ops.size());
  for (auto& op : ops) DispatchOp(std::move(op));
}

void TcpTransport::EnsurePeerChannels(const EngineKey& peer) {
  PeerLinks& link = peers_[peer];
  if (!link.CtrlUp() && link.ctrlPending == 0) ConnectChannel(peer, tcp::Channel::CTRL);
  int want = std::max(1, config_.numDataConns);
  while (int(link.dataFds.size()) + link.dataPending < want)
    ConnectChannel(peer, tcp::Channel::DATA);
}

void TcpTransport::ConnectChannel(const EngineKey& peer, tcp::Channel ch) {
  EngineDesc desc;
  {
    std::lock_guard<std::mutex> lk(remoteMu_);
    auto it = remoteEngines_.find(peer);
    if (it == remoteEngines_.end()) {
      MORI_IO_ERROR("TCP: remote engine {} not registered", peer);
      return;
    }
    desc = it->second;
  }

  auto peerAddr = ParseIpv4(desc.host, static_cast<uint16_t>(desc.port));
  if (!peerAddr) {
    MORI_IO_ERROR("TCP: invalid remote host {}:{}", desc.host, desc.port);
    return;
  }

  int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    MORI_IO_ERROR("TCP: socket() failed: {}", strerror(errno));
    return;
  }
  MORI_IO_TRACE("TCP: connect start peer={} ch={} fd={}", peer, int(ch), fd);

  if (!engConfig_.host.empty()) {
    auto la = ParseIpv4(engConfig_.host, 0);
    if (la) {
      sockaddr_in localAddr = *la;
      if (bind(fd, reinterpret_cast<sockaddr*>(&localAddr), sizeof(localAddr)) != 0)
        MORI_IO_WARN("TCP: bind(local) {} failed: {}", engConfig_.host, strerror(errno));
    }
  }

  sockaddr_in pa = *peerAddr;
  int rc = connect(fd, reinterpret_cast<sockaddr*>(&pa), sizeof(pa));
  bool connecting = false;
  if (rc != 0) {
    if (errno == EINPROGRESS)
      connecting = true;
    else {
      MORI_IO_ERROR("TCP: connect failed: {}", strerror(errno));
      close(fd);
      return;
    }
  }

  auto conn = std::make_unique<Connection>();
  conn->fd = fd;
  conn->isOutgoing = true;
  conn->connecting = connecting;
  conn->peerKey = peer;
  conn->ch = ch;
  conn->inbuf.reserve(4096);
  if (ch == tcp::Channel::CTRL) ConfigureCtrlSocket(fd, config_);
  AddEpoll(fd, true, connecting || !conn->sendq.empty());
  conns_[fd] = std::move(conn);

  PeerLinks& link = peers_[peer];
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
  auto it = conns_.find(fd);
  if (it == conns_.end()) return;
  Connection* c = it->second.get();
  if (!c || c->helloSent) return;
  c->helloSent = true;
  MORI_IO_TRACE("TCP: queue HELLO fd={} ch={}", fd, int(c->ch));
  SendItem item;
  item.header = tcp::BuildHello(c->ch, myEngKey_);
  item.iov = {{item.header.data(), item.header.size()}};
  c->sendq.push_back(std::move(item));
}

void TcpTransport::AcceptNew() {
  while (true) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    int fd =
        accept4(listenFd_, reinterpret_cast<sockaddr*>(&peer), &len, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      if (IsWouldBlock(errno)) break;
      MORI_IO_WARN("TCP: accept failed: {}", strerror(errno));
      break;
    }
    MORI_IO_TRACE("TCP: accept fd={}", fd);
    auto conn = std::make_unique<Connection>();
    conn->fd = fd;
    conn->inbuf.reserve(4096);
    AddEpoll(fd, true, false);
    conns_[fd] = std::move(conn);
  }
}

void TcpTransport::DrainWakeFd() {
  uint64_t v;
  while (::read(wakeFd_, &v, sizeof(v)) > 0) {
  }
  std::deque<std::unique_ptr<OutboundOpState>> ops;
  {
    std::lock_guard<std::mutex> lk(submitMu_);
    ops.swap(submitQ_);
  }
  for (auto& op : ops) {
    EnsurePeerChannels(op->peer);
    if (IsPeerReady(op->peer))
      DispatchOp(std::move(op));
    else
      waitingOps_[op->peer].push_back(std::move(op));
  }
}

bool TcpTransport::IsPeerReady(const EngineKey& peer) {
  auto it = peers_.find(peer);
  if (it == peers_.end() || !it->second.CtrlUp() || !it->second.DataUp()) return false;
  auto cit = conns_.find(it->second.ctrlFd);
  if (cit == conns_.end() || !cit->second->helloReceived) return false;
  return !it->second.workers.empty();
}

void TcpTransport::RegisterRecvTargetWithWorkers(const EngineKey& peer, TransferUniqueId opId,
                                                 const WorkerRecvTarget& target) {
  auto pit = peers_.find(peer);
  if (pit == peers_.end()) return;
  for (auto* w : pit->second.workers) w->RegisterRecvTarget(opId, target);
}

void TcpTransport::RemoveRecvTargetFromWorkers(const EngineKey& peer, TransferUniqueId opId) {
  auto pit = peers_.find(peer);
  if (pit == peers_.end()) return;
  for (auto* w : pit->second.workers) w->RemoveRecvTarget(opId);
}

// ---------------------------------------------------------------------------
// DispatchOp - initiate an outbound operation
// ---------------------------------------------------------------------------
void TcpTransport::DispatchOp(std::unique_ptr<OutboundOpState> op) {
  if (!op) {
    MORI_IO_ERROR("TCP: DispatchOp got null op");
    return;
  }
  const EngineKey peerKey = op->peer;
  auto pit = peers_.find(peerKey);
  if (pit == peers_.end() || !pit->second.CtrlUp() || !pit->second.DataUp()) {
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: peer not connected");
    return;
  }
  Connection* ctrl = conns_[pit->second.ctrlFd].get();
  if (!ctrl) {
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: ctrl missing");
    return;
  }
  auto& workerList = pit->second.workers;
  if (workerList.empty()) {
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: no data workers");
    return;
  }

  const TransferUniqueId opId = op->id;
  auto [itIns, inserted] = pendingOutbound_.emplace(opId, std::move(op));
  if (!inserted) {
    MORI_IO_ERROR("TCP: duplicate op id={}", opId);
    itIns->second->status->Update(StatusCode::ERR_BAD_STATE, "TCP: duplicate op id");
    pendingOutbound_.erase(itIns);
    return;
  }
  OutboundOpState* st = itIns->second.get();

  // Decide lane count for striping
  const uint64_t totalBytes = SumLens(st->localSegs);
  int wantLanes = std::min<int>(std::max(1, config_.numDataConns), 1U << kLaneBits);
  uint8_t lanesTotal = 1;
  if (wantLanes > 1 && config_.stripingThresholdBytes > 0 &&
      totalBytes >= uint64_t(config_.stripingThresholdBytes) && st->localSegs.size() == 1 &&
      st->remoteSegs.size() == 1 && workerList.size() >= 2) {
    lanesTotal = uint8_t(std::min<size_t>(wantLanes, workerList.size()));
  }
  st->lanesTotal = lanesTotal;

  // Allocate pinned staging for GPU reads
  if (st->isRead && st->local.loc == MemoryLocationType::GPU) {
    st->pinned = staging_.Acquire(static_cast<size_t>(totalBytes));
    if (!st->pinned) {
      st->status->Update(StatusCode::ERR_BAD_STATE, "TCP: staging alloc failed");
      pendingOutbound_.erase(opId);
      return;
    }
  }

  // Set up recv targets for reads
  if (st->isRead) {
    WorkerRecvTarget target;
    target.lanesTotal = lanesTotal;
    target.totalLen = totalBytes;
    if (st->local.loc == MemoryLocationType::GPU) {
      target.toGpu = true;
      target.pinned = st->pinned;
    } else {
      target.cpuBase = reinterpret_cast<void*>(st->local.data);
      target.segs = st->localSegs;
    }
    RegisterRecvTargetWithWorkers(peerKey, opId, target);
  }

  // Build and send ctrl frame
  std::vector<uint8_t> ctrlFrame;
  if (st->localSegs.size() == 1) {
    auto type = st->isRead ? tcp::CtrlMsgType::READ_REQ : tcp::CtrlMsgType::WRITE_REQ;
    ctrlFrame = tcp::BuildLinearReq(type, st->id, st->remote.id, st->remoteSegs[0].off,
                                    st->remoteSegs[0].len, lanesTotal);
  } else {
    auto type = st->isRead ? tcp::CtrlMsgType::BATCH_READ_REQ : tcp::CtrlMsgType::BATCH_WRITE_REQ;
    std::vector<uint64_t> roffs, szs;
    roffs.reserve(st->remoteSegs.size());
    szs.reserve(st->remoteSegs.size());
    for (auto& s : st->remoteSegs) {
      roffs.push_back(s.off);
      szs.push_back(s.len);
    }
    ctrlFrame = tcp::BuildBatchReq(type, st->id, st->remote.id, roffs, szs, lanesTotal);
  }
  QueueSend(ctrl->fd, std::move(ctrlFrame));

  if (!st->isRead) QueueDataSend(workerList, st->local, st->localSegs, st->id, lanesTotal);
  UpdateWriteInterest(ctrl->fd);
}

void TcpTransport::QueueSend(int fd, std::vector<uint8_t> bytes, std::function<void()> onDone) {
  auto it = conns_.find(fd);
  if (it == conns_.end()) return;
  SendItem item;
  item.header = std::move(bytes);
  item.iov = {{item.header.data(), item.header.size()}};
  item.onDone = std::move(onDone);
  it->second->sendq.push_back(std::move(item));
}

// ---------------------------------------------------------------------------
// Data send (unified for write-send and read-response)
// ---------------------------------------------------------------------------
void TcpTransport::QueueDataSend(const std::vector<DataConnectionWorker*>& workers,
                                 const MemoryDesc& src, const std::vector<Segment>& srcSegs,
                                 uint64_t opId, uint8_t lanesTotal,
                                 std::function<void()> onLaneDone) {
  if (workers.empty()) return;
  const uint64_t total = SumLens(srcSegs);
  lanesTotal = ClampLanesTotal(lanesTotal);
  lanesTotal = std::min<uint8_t>(lanesTotal, static_cast<uint8_t>(workers.size()));
  if (lanesTotal > 1 && srcSegs.size() != 1) {
    MORI_IO_WARN("TCP: striping requires 1 segment, fallback to 1 lane");
    lanesTotal = 1;
  }

  if (src.loc == MemoryLocationType::GPU) {
    // GPU path: DtoH copy, then send from pinned buffer
    auto pinned = staging_.Acquire(static_cast<size_t>(total));
    if (!pinned) {
      MORI_IO_ERROR("TCP: staging alloc failed for GPU send");
      return;
    }

    auto workersCopy =
        std::vector<DataConnectionWorker*>(workers.begin(), workers.begin() + lanesTotal);
    auto sendCb = [workersCopy, pinned, opId, lanesTotal, total,
                   onLaneDone = std::move(onLaneDone)]() {
      for (uint8_t lane = 0; lane < lanesTotal; ++lane) {
        LaneSpan span = ComputeLaneSpan(total, lanesTotal, lane);
        SendItem item;
        item.header = tcp::BuildDataHeader(ToWireOpId(opId, lane), span.len, 0);
        item.iov = {{item.header.data(), item.header.size()},
                    {static_cast<uint8_t*>(pinned->ptr) + span.off, size_t(span.len)}};
        item.keepalive = pinned;
        item.onDone = onLaneDone;
        workersCopy[lane % workersCopy.size()]->SubmitSend(std::move(item));
      }
    };
    ScheduleGpuCopy(src.deviceId, false, src, srcSegs, pinned, std::move(sendCb));
    return;
  }

  // CPU path
  uint8_t* base = reinterpret_cast<uint8_t*>(src.data);
  if (lanesTotal == 1) {
    SendItem item;
    item.header = tcp::BuildDataHeader(ToWireOpId(opId, 0), total, 0);
    item.iov.reserve(1 + srcSegs.size());
    item.iov.push_back({item.header.data(), item.header.size()});
    for (auto& s : srcSegs) item.iov.push_back({base + s.off, size_t(s.len)});
    item.onDone = std::move(onLaneDone);
    workers[0]->SubmitSend(std::move(item));
  } else {
    // Multi-lane striping (contiguous single-segment)
    for (uint8_t lane = 0; lane < lanesTotal; ++lane) {
      LaneSpan span = ComputeLaneSpan(total, lanesTotal, lane);
      SendItem item;
      item.header = tcp::BuildDataHeader(ToWireOpId(opId, lane), span.len, 0);
      item.iov = {{item.header.data(), item.header.size()},
                  {base + srcSegs[0].off + span.off, size_t(span.len)}};
      item.onDone = onLaneDone;
      workers[lane % workers.size()]->SubmitSend(std::move(item));
    }
  }
}

// ---------------------------------------------------------------------------
// GPU copy (unified DtoH / HtoD)
// ---------------------------------------------------------------------------
bool TcpTransport::ScheduleGpuCopy(int deviceId, bool toDevice, const MemoryDesc& mem,
                                   const std::vector<Segment>& segs,
                                   std::shared_ptr<PinnedBuf> pinned,
                                   std::function<void()> onComplete) {
  const uint64_t total = SumLens(segs);
  hipStream_t stream = streamPool_.GetNextStream(deviceId);
  hipEvent_t ev = eventPool_.GetEvent(deviceId);
  if (!stream || !ev) {
    MORI_IO_ERROR("TCP: failed to get HIP stream/event");
    if (ev) eventPool_.PutEvent(ev, deviceId);
    return false;
  }

  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));
  uint8_t* hostPtr = reinterpret_cast<uint8_t*>(pinned->ptr);

  uint64_t spanOff = 0, spanLen = 0;
  if (IsSingleContiguousSpan(segs, &spanOff, &spanLen) && spanLen == total) {
    if (toDevice) {
      void* gpu = reinterpret_cast<void*>(mem.data + spanOff);
      HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpu, hostPtr, size_t(total), stream));
    } else {
      hipDeviceptr_t gpu = reinterpret_cast<hipDeviceptr_t>(mem.data + spanOff);
      HIP_RUNTIME_CHECK(hipMemcpyDtoHAsync(hostPtr, gpu, size_t(total), stream));
    }
  } else {
    uint64_t off = 0;
    for (auto& s : segs) {
      if (toDevice) {
        void* gpu = reinterpret_cast<void*>(mem.data + s.off);
        HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpu, hostPtr + off, size_t(s.len), stream));
      } else {
        hipDeviceptr_t gpu = reinterpret_cast<hipDeviceptr_t>(mem.data + s.off);
        HIP_RUNTIME_CHECK(hipMemcpyDtoHAsync(hostPtr + off, gpu, size_t(s.len), stream));
      }
      off += s.len;
    }
  }
  HIP_RUNTIME_CHECK(hipEventRecord(ev, stream));
  gpuTasks_.push_back({deviceId, ev, std::move(onComplete)});
  return true;
}

void TcpTransport::PollGpuTasks() {
  for (auto it = gpuTasks_.begin(); it != gpuTasks_.end();) {
    hipError_t st = hipEventQuery(it->ev);
    if (st == hipSuccess) {
      eventPool_.PutEvent(it->ev, it->deviceId);
      if (it->onReady) it->onReady();
      it = gpuTasks_.erase(it);
    } else if (st == hipErrorNotReady) {
      ++it;
    } else {
      MORI_IO_ERROR("TCP: hipEventQuery failed: {}", hipGetErrorString(st));
      eventPool_.PutEvent(it->ev, it->deviceId);
      it = gpuTasks_.erase(it);
    }
  }
}

// ---------------------------------------------------------------------------
// Ctrl-connection I/O
// ---------------------------------------------------------------------------
void TcpTransport::UpdateWriteInterest(int fd) {
  auto it = conns_.find(fd);
  if (it == conns_.end()) return;
  Connection* c = it->second.get();
  if (!c || c->fd < 0) return;
  if (!c->connecting && !c->sendq.empty()) {
    FlushSend(c);
    it = conns_.find(fd);
    if (it == conns_.end()) return;
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
      MORI_IO_ERROR("TCP: connect failed fd {}: {}", c->fd, strerror(err ? err : errno));
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
    ssize_t n = ::sendmsg(c->fd, &msg, MSG_NOSIGNAL);
    if (n < 0) {
      if (IsWouldBlock(errno)) return;
      MORI_IO_ERROR("TCP: sendmsg ctrl fd {} failed: {}", c->fd, strerror(errno));
      ClosePeerByFd(c->fd);
      return;
    }
    if (n == 0) return;
    item.Advance(static_cast<size_t>(n));
  }
}

// ---------------------------------------------------------------------------
// Peer lifecycle
// ---------------------------------------------------------------------------
void TcpTransport::CloseAndRemoveFd(int fd) {
  auto wit = dataWorkers_.find(fd);
  if (wit != dataWorkers_.end()) {
    wit->second->Stop();
    int nfd = wit->second->NotifyFd();
    DelEpoll(nfd);
    workerNotifyMap_.erase(nfd);
    dataWorkers_.erase(wit);
  }
  auto cit = conns_.find(fd);
  if (cit != conns_.end()) {
    CloseConnInternal(cit->second.get());
    conns_.erase(cit);
  }
}

EngineKey TcpTransport::FindPeerByFd(int fd) {
  for (auto& [key, link] : peers_) {
    if (link.ctrlFd == fd) return key;
    for (int dfd : link.dataFds)
      if (dfd == fd) return key;
  }
  return {};
}

void TcpTransport::ClosePeerByFd(int fd) {
  EngineKey peer = FindPeerByFd(fd);
  if (!peer.empty())
    ClosePeerByKey(peer, "TCP: connection lost");
  else
    CloseAndRemoveFd(fd);
}

void TcpTransport::ClosePeerByKey(const EngineKey& peer, const std::string& reason) {
  auto pit = peers_.find(peer);
  if (pit == peers_.end()) return;
  auto link = pit->second;
  CloseAndRemoveFd(link.ctrlFd);
  for (int dfd : link.dataFds) CloseAndRemoveFd(dfd);
  peers_.erase(peer);
  FailPendingOpsForPeer(peer, reason);
}

void TcpTransport::FailPendingOpsForPeer(const EngineKey& peer, const std::string& msg) {
  for (auto it = pendingOutbound_.begin(); it != pendingOutbound_.end();) {
    if (it->second->peer == peer) {
      it->second->status->Update(StatusCode::ERR_BAD_STATE, msg);
      it = pendingOutbound_.erase(it);
    } else
      ++it;
  }
  waitingOps_.erase(peer);
  inboundWrites_.erase(peer);
  earlyWrites_.erase(peer);
}

// ---------------------------------------------------------------------------
// Ctrl message handling
// ---------------------------------------------------------------------------
void TcpTransport::HandleCtrlReadable(Connection* c) {
  while (true) {
    uint8_t tmp[65536];
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
        MORI_IO_ERROR("TCP: bad ctrl header fd {}", c->fd);
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
    MORI_IO_WARN("TCP: ctrl message before HELLO, dropping");
    return;
  }

  switch (type) {
    case tcp::CtrlMsgType::WRITE_REQ:
    case tcp::CtrlMsgType::READ_REQ:
    case tcp::CtrlMsgType::BATCH_WRITE_REQ:
    case tcp::CtrlMsgType::BATCH_READ_REQ:
      HandleRequest(c->peerKey, type, body, len);
      break;
    case tcp::CtrlMsgType::COMPLETION:
      HandleCompletion(c->peerKey, body, len);
      break;
    default:
      MORI_IO_WARN("TCP: unknown ctrl msg type {}", uint32_t(type));
  }
}

void TcpTransport::HandleHello(Connection* c, const uint8_t* body, size_t len) {
  if (len < 5) {
    MORI_IO_WARN("TCP: bad HELLO len {}", len);
    ClosePeerByFd(c->fd);
    return;
  }
  tcp::WireReader r{body, len};
  uint8_t chRaw;
  uint32_t keyLen;
  if (!r.u8(&chRaw) || !r.u32(&keyLen) || r.off + keyLen > len) {
    ClosePeerByFd(c->fd);
    return;
  }

  c->peerKey.assign(reinterpret_cast<const char*>(body + r.off), keyLen);
  c->ch = (chRaw == uint8_t(tcp::Channel::DATA)) ? tcp::Channel::DATA : tcp::Channel::CTRL;
  c->helloReceived = true;
  MORI_IO_TRACE("TCP: recv HELLO fd={} peer={} ch={} out={}", c->fd, c->peerKey, int(c->ch),
                c->isOutgoing);

  if (!c->helloSent) {
    QueueHello(c->fd);
    UpdateWriteInterest(c->fd);
  }
  if (c->ch == tcp::Channel::CTRL) ConfigureCtrlSocket(c->fd, config_);
  AssignConnToPeer(c);
  MaybeDispatchQueuedOps(c->peerKey);
}

// Unified handler for WRITE_REQ, READ_REQ, BATCH_WRITE_REQ, BATCH_READ_REQ
void TcpTransport::HandleRequest(const EngineKey& peer, tcp::CtrlMsgType type, const uint8_t* body,
                                 size_t len) {
  RequestView req;
  if (!ParseRequest(type, body, len, &req)) {
    MORI_IO_WARN("TCP: malformed request type={}", uint8_t(type));
    return;
  }

  bool isWrite = (type == tcp::CtrlMsgType::WRITE_REQ || type == tcp::CtrlMsgType::BATCH_WRITE_REQ);
  bool isBatch =
      (type == tcp::CtrlMsgType::BATCH_WRITE_REQ || type == tcp::CtrlMsgType::BATCH_READ_REQ);

  auto memOpt = LookupLocalMem(req.memId);

  if (isWrite) {
    // Inbound write: remote is sending data to us
    InboundWriteState ws;
    ws.peer = peer;
    ws.id = req.opId;
    ws.lanesTotal = req.lanesTotal;
    ws.discard = true;
    if (memOpt && SegmentsInRange(req.segs, memOpt->size)) {
      ws.dst = *memOpt;
      ws.dstSegs = std::move(req.segs);
      ws.discard = false;
    }
    FinalizeInboundWriteSetup(peer, req.opId, ws);
  } else {
    // Inbound read: remote wants data from us
    if (!memOpt) {
      SendCompletionAndRecord(
          peer, req.opId, StatusCode::ERR_NOT_FOUND,
          isBatch ? "TCP: remote mem not found" : "TCP: remote mem not found/out of range");
      return;
    }
    if (!SegmentsInRange(req.segs, memOpt->size)) {
      auto code = isBatch ? StatusCode::ERR_INVALID_ARGS : StatusCode::ERR_NOT_FOUND;
      SendCompletionAndRecord(
          peer, req.opId, code,
          isBatch ? "TCP: batch read out of range" : "TCP: remote mem not found/out of range");
      return;
    }

    // Send data back to requester
    auto pit = peers_.find(peer);
    if (pit == peers_.end() || pit->second.workers.empty()) return;
    auto& workerList = pit->second.workers;
    uint8_t useLanes = std::min<uint8_t>(ClampLanesTotal(req.lanesTotal),
                                         uint8_t(std::max<size_t>(1, workerList.size())));
    if (useLanes > 1 && req.segs.size() != 1) useLanes = 1;

    struct DoneState {
      EngineKey peer;
      uint64_t opId;
      std::atomic<int> remaining{0};
    };
    auto done = std::make_shared<DoneState>();
    done->peer = peer;
    done->opId = req.opId;
    done->remaining.store(useLanes);
    auto laneDone = [this, done]() {
      if (done->remaining.fetch_sub(1) > 1) return;
      SendCompletionAndRecord(done->peer, done->opId, StatusCode::SUCCESS, "");
    };
    QueueDataSend(workerList, *memOpt, req.segs, req.opId, useLanes, std::move(laneDone));
  }
}

void TcpTransport::HandleCompletion(const EngineKey& peer, const uint8_t* body, size_t len) {
  CompletionView msg;
  if (!ParseCompletion(body, len, &msg)) {
    MORI_IO_WARN("TCP: malformed COMPLETION");
    return;
  }

  auto it = pendingOutbound_.find(msg.opId);
  if (it == pendingOutbound_.end()) return;
  OutboundOpState& st = *it->second;
  st.completionReceived = true;
  st.completionCode = static_cast<StatusCode>(msg.statusCode);
  st.completionMsg = std::move(msg.msg);
  if (st.completionCode != StatusCode::SUCCESS) {
    RemoveRecvTargetFromWorkers(st.peer, msg.opId);
    st.status->Update(st.completionCode, st.completionMsg);
    pendingOutbound_.erase(it);
    return;
  }
  MaybeCompleteOutbound(st);
}

// ---------------------------------------------------------------------------
// Inbound / Outbound state machines
// ---------------------------------------------------------------------------
std::optional<MemoryDesc> TcpTransport::LookupLocalMem(MemoryUniqueId id) {
  std::lock_guard<std::mutex> lk(memMu_);
  auto it = localMems_.find(id);
  return (it != localMems_.end()) ? std::optional(it->second) : std::nullopt;
}

void TcpTransport::RecordInboundStatus(const EngineKey& peer, TransferUniqueId id, StatusCode code,
                                       const std::string& msg) {
  std::lock_guard<std::mutex> lk(inboundMu_);
  inboundStatus_[peer][id] = {code, msg};
}

void TcpTransport::SendCompletionAndRecord(const EngineKey& peer, TransferUniqueId opId,
                                           StatusCode code, const std::string& msg) {
  Connection* ctrl = PeerCtrl(peer);
  if (ctrl) {
    QueueSend(ctrl->fd, tcp::BuildCompletion(opId, uint32_t(code), msg));
    UpdateWriteInterest(ctrl->fd);
  }
  RecordInboundStatus(peer, opId, code, msg);
}

Connection* TcpTransport::PeerCtrl(const EngineKey& peer) {
  auto it = peers_.find(peer);
  if (it == peers_.end() || !it->second.CtrlUp()) return nullptr;
  auto cit = conns_.find(it->second.ctrlFd);
  return (cit != conns_.end()) ? cit->second.get() : nullptr;
}

void TcpTransport::FinalizeInboundWriteSetup(const EngineKey& peer, TransferUniqueId opId,
                                             InboundWriteState& ws) {
  if (!ws.discard && ws.dst.loc == MemoryLocationType::GPU) {
    ws.pinned = staging_.Acquire(static_cast<size_t>(SumLens(ws.dstSegs)));
    if (!ws.pinned) ws.discard = true;
  }
  inboundWrites_[peer][opId] = ws;

  // Set up worker recv targets
  WorkerRecvTarget target;
  target.lanesTotal = ws.lanesTotal;
  target.totalLen = SumLens(ws.dstSegs);
  target.discard = ws.discard;
  if (!ws.discard && ws.dst.loc == MemoryLocationType::GPU) {
    target.toGpu = true;
    target.pinned = ws.pinned;
  } else if (!ws.discard) {
    target.cpuBase = reinterpret_cast<void*>(ws.dst.data);
    target.segs = ws.dstSegs;
  }
  RegisterRecvTargetWithWorkers(peer, opId, target);

  TryConsumeEarlyWriteLanes(peer, opId);
}

void TcpTransport::MaybeFinalizeInboundWrite(const EngineKey& peer, TransferUniqueId opId) {
  auto iwIt = inboundWrites_.find(peer);
  if (iwIt == inboundWrites_.end()) return;
  auto wsIt = iwIt->second.find(opId);
  if (wsIt == iwIt->second.end()) return;

  InboundWriteState& ws = wsIt->second;
  ws.lanesTotal = ClampLanesTotal(ws.lanesTotal);
  if ((ws.lanesDoneMask & LanesAllMask(ws.lanesTotal)) != LanesAllMask(ws.lanesTotal)) return;

  RemoveRecvTargetFromWorkers(peer, opId);

  if (ws.discard) {
    SendCompletionAndRecord(peer, opId, StatusCode::ERR_INVALID_ARGS, "TCP: write discarded");
  } else if (ws.dst.loc == MemoryLocationType::GPU) {
    if (!ws.pinned) {
      SendCompletionAndRecord(peer, opId, StatusCode::ERR_BAD_STATE,
                              "TCP: missing staging (write)");
    } else {
      auto pinnedRef = ws.pinned;
      bool ok = ScheduleGpuCopy(ws.dst.deviceId, true, ws.dst, ws.dstSegs, pinnedRef,
                                [this, peer, opId, pinnedRef]() {
                                  SendCompletionAndRecord(peer, opId, StatusCode::SUCCESS, "");
                                });
      if (!ok)
        SendCompletionAndRecord(peer, opId, StatusCode::ERR_BAD_STATE, "TCP: HIP copy failed");
    }
  } else {
    SendCompletionAndRecord(peer, opId, StatusCode::SUCCESS, "");
  }

  iwIt->second.erase(wsIt);
  if (iwIt->second.empty()) inboundWrites_.erase(iwIt);
  auto ewIt = earlyWrites_.find(peer);
  if (ewIt != earlyWrites_.end()) {
    ewIt->second.erase(opId);
    if (ewIt->second.empty()) earlyWrites_.erase(ewIt);
  }
}

void TcpTransport::TryConsumeEarlyWriteLanes(const EngineKey& peer, TransferUniqueId opId) {
  auto iwIt = inboundWrites_.find(peer);
  if (iwIt == inboundWrites_.end()) return;
  auto wsIt = iwIt->second.find(opId);
  if (wsIt == iwIt->second.end()) return;
  InboundWriteState& ws = wsIt->second;
  ws.lanesTotal = ClampLanesTotal(ws.lanesTotal);

  auto ewIt = earlyWrites_.find(peer);
  if (ewIt == earlyWrites_.end()) return;
  auto elIt = ewIt->second.find(opId);
  if (elIt == ewIt->second.end()) return;

  EarlyWriteState& early = elIt->second;
  const uint64_t total = SumLens(ws.dstSegs);
  uint8_t* dstBase = reinterpret_cast<uint8_t*>(ws.dst.data);

  for (auto it = early.lanes.begin(); it != early.lanes.end();) {
    uint8_t lane = it->first;
    EarlyWriteLaneState& ls = it->second;
    if (!ls.complete) {
      ++it;
      continue;
    }

    if (lane >= ws.lanesTotal) ws.discard = true;
    LaneSpan span = ComputeLaneSpan(total, ws.lanesTotal, lane);
    if (span.len != ls.payloadLen) ws.discard = true;

    if (!ws.discard && ls.pinned) {
      uint8_t* src = reinterpret_cast<uint8_t*>(ls.pinned->ptr);
      if (ws.dst.loc == MemoryLocationType::GPU) {
        if (!ws.pinned) {
          ws.pinned = staging_.Acquire(size_t(total));
          if (!ws.pinned) ws.discard = true;
        }
        if (!ws.discard && ws.pinned)
          std::memcpy(reinterpret_cast<uint8_t*>(ws.pinned->ptr) + span.off, src, size_t(span.len));
      } else {
        auto segs = SliceSegments(ws.dstSegs, span.off, span.len);
        uint64_t copied = 0;
        for (auto& s : segs) {
          std::memcpy(dstBase + s.off, src + copied, size_t(s.len));
          copied += s.len;
        }
      }
    }
    if (lane < 8) ws.lanesDoneMask |= uint8_t(1U << lane);
    it = early.lanes.erase(it);
  }

  if (early.lanes.empty()) {
    ewIt->second.erase(elIt);
    if (ewIt->second.empty()) earlyWrites_.erase(ewIt);
  }
  MaybeFinalizeInboundWrite(peer, opId);
}

void TcpTransport::MaybeCompleteOutbound(OutboundOpState& st) {
  if (!st.completionReceived) return;
  if (st.isRead) {
    uint16_t allMask = LanesAllMask(st.lanesTotal);
    if (st.lanesDoneMask != allMask || st.rxBytes != st.expectedRxBytes || st.gpuCopyPending)
      return;
  }
  RemoveRecvTargetFromWorkers(st.peer, st.id);
  st.status->Update(StatusCode::SUCCESS, "");
  pendingOutbound_.erase(st.id);
}

// ---------------------------------------------------------------------------
// Worker event processing
// ---------------------------------------------------------------------------
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
        MORI_IO_WARN("TCP: worker error peer {}: {}", ev.peerKey, ev.errorMsg);
        ClosePeerByKey(ev.peerKey, ev.errorMsg);
        break;
    }
  }
}

void TcpTransport::ProcessWorkerEvents() {
  // Snapshot notify fds to avoid invalidation during iteration
  std::vector<int> fds;
  fds.reserve(workerNotifyMap_.size());
  for (auto& kv : workerNotifyMap_) fds.push_back(kv.first);
  for (int nfd : fds) {
    auto it = workerNotifyMap_.find(nfd);
    if (it != workerNotifyMap_.end()) ProcessEventsFrom(it->second);
  }
}

void TcpTransport::HandleWorkerRecvDone(const WorkerEvent& ev) {
  const EngineKey& peer = ev.peerKey;
  const TransferUniqueId opId = ev.opId;

  // Check inbound writes first
  auto iwIt = inboundWrites_.find(peer);
  if (iwIt != inboundWrites_.end()) {
    auto wsIt = iwIt->second.find(opId);
    if (wsIt != iwIt->second.end()) {
      wsIt->second.lanesTotal = ClampLanesTotal(wsIt->second.lanesTotal);
      if (ev.lane < 8) wsIt->second.lanesDoneMask |= uint8_t(1U << ev.lane);
      MaybeFinalizeInboundWrite(peer, opId);
      return;
    }
  }

  // Check outbound reads
  auto obIt = pendingOutbound_.find(opId);
  if (obIt == pendingOutbound_.end()) return;
  OutboundOpState& st = *obIt->second;
  st.lanesTotal = ClampLanesTotal(st.lanesTotal);
  uint8_t bit = uint8_t(1U << ev.lane);
  if (!(st.lanesDoneMask & bit)) {
    st.lanesDoneMask |= bit;
    st.rxBytes += ev.laneLen;
  }

  if (st.local.loc == MemoryLocationType::GPU) {
    uint16_t allMask = LanesAllMask(st.lanesTotal);
    if ((st.lanesDoneMask & allMask) != allMask) {
      MaybeCompleteOutbound(st);
      return;
    }
    if (st.gpuCopyPending) return;
    if (!st.pinned) {
      st.status->Update(StatusCode::ERR_BAD_STATE, "TCP: missing staging (read)");
      RemoveRecvTargetFromWorkers(st.peer, opId);
      pendingOutbound_.erase(obIt);
      return;
    }
    st.gpuCopyPending = true;
    auto pinnedRef = st.pinned;
    bool ok = ScheduleGpuCopy(st.local.deviceId, true, st.local, st.localSegs, pinnedRef,
                              [this, opId, pinnedRef]() {
                                auto it2 = pendingOutbound_.find(opId);
                                if (it2 == pendingOutbound_.end()) return;
                                it2->second->gpuCopyPending = false;
                                MaybeCompleteOutbound(*it2->second);
                              });
    if (!ok) {
      st.status->Update(StatusCode::ERR_BAD_STATE, "TCP: HIP copy failed (read)");
      RemoveRecvTargetFromWorkers(st.peer, opId);
      pendingOutbound_.erase(obIt);
    }
    return;
  }
  MaybeCompleteOutbound(st);
}

void TcpTransport::HandleWorkerEarlyData(const WorkerEvent& ev) {
  auto& perPeer = earlyWrites_[ev.peerKey];
  auto& early = perPeer[ev.opId];
  if (early.lanes.count(ev.lane)) {
    MORI_IO_WARN("TCP: duplicate early data op {} lane {} peer {}", ev.opId, uint32_t(ev.lane),
                 ev.peerKey);
    return;
  }
  early.lanes.emplace(ev.lane, EarlyWriteLaneState{ev.laneLen, ev.earlyBuf, true});
  TryConsumeEarlyWriteLanes(ev.peerKey, ev.opId);
}

void TcpTransport::ScanTimeouts() {
  if (config_.opTimeoutMs <= 0) return;
  auto now = Clock::now();
  auto timeout = std::chrono::milliseconds(config_.opTimeoutMs);
  for (auto it = pendingOutbound_.begin(); it != pendingOutbound_.end();) {
    if ((now - it->second->startTs) > timeout) {
      RemoveRecvTargetFromWorkers(it->second->peer, it->first);
      it->second->status->Update(StatusCode::ERR_BAD_STATE, "TCP: op timeout");
      it = pendingOutbound_.erase(it);
    } else
      ++it;
  }
}

// ---------------------------------------------------------------------------
// Main I/O loop
// ---------------------------------------------------------------------------
void TcpTransport::IoLoop() {
  constexpr int kMaxEvents = 128;
  epoll_event events[kMaxEvents];

  while (running_.load()) {
    PollGpuTasks();
    ProcessWorkerEvents();
    ScanTimeouts();

    bool hasActive = !gpuTasks_.empty() || !pendingOutbound_.empty() || !workerNotifyMap_.empty();
    int nfds = epoll_wait(epfd_, events, kMaxEvents, hasActive ? 0 : 2);
    if (nfds < 0) {
      if (errno == EINTR) continue;
      MORI_IO_ERROR("TCP: epoll_wait: {}", strerror(errno));
      break;
    }

    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;
      uint32_t ev = events[i].events;

      if (fd == listenFd_) {
        AcceptNew();
        continue;
      }
      if (fd == wakeFd_) {
        DrainWakeFd();
        continue;
      }

      auto wnit = workerNotifyMap_.find(fd);
      if (wnit != workerNotifyMap_.end()) {
        ProcessEventsFrom(wnit->second);
        continue;
      }

      auto cit = conns_.find(fd);
      if (cit == conns_.end()) continue;
      Connection* c = cit->second.get();
      if (!c) continue;

      if (ev & (EPOLLERR | EPOLLHUP)) {
        ClosePeerByFd(fd);
        continue;
      }
      if (ev & EPOLLIN) {
        HandleCtrlReadable(c);
        cit = conns_.find(fd);
        if (cit == conns_.end()) continue;
        c = cit->second.get();
        if (!c) continue;
      }
      if (ev & EPOLLOUT) HandleConnWritable(c);
    }
  }
}

}  // namespace io
}  // namespace mori

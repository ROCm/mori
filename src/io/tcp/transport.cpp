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

#include <climits>
#include <limits>
#include <stdexcept>

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
DataConnectionWorker::DataConnectionWorker(int fd, EngineKey peer,
                                           std::shared_ptr<PeerRecvTargets> recvTable,
                                           std::vector<uint8_t> initialBytes)
    : fd_(fd),
      peerKey_(std::move(peer)),
      recvTable_(std::move(recvTable)),
      initialBytes_(std::move(initialBytes)) {
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
  bool wasRunning = running_.exchange(false);
  if (wasRunning) WakeWorker();
  if (thread_.joinable()) thread_.join();
}

void DataConnectionWorker::SubmitSend(SendItem item) {
  {
    std::lock_guard<std::mutex> lk(sendMu_);
    sendQ_.push_back(std::move(item));
  }
  WakeWorker();
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
    if (initialOffset_ < initialBytes_.size() && recv_.state != RecvState::AwaitTarget) {
      if (!ProcessRecv()) break;
      if (!running_.load()) break;
    }
    bool hasSend;
    {
      std::lock_guard<std::mutex> lk(sendMu_);
      hasSend = !sendQ_.empty();
    }

    const bool awaitingTarget = recv_.state == RecvState::AwaitTarget;
    pfds[0].events = (awaitingTarget ? 0 : POLLIN) | (hasSend ? POLLOUT : 0);
    pfds[0].revents = pfds[1].revents = 0;

    int n = ::poll(pfds, 2, -1);
    if (n < 0) {
      if (errno == EINTR) continue;
      WorkerEvent ev;
      ev.type = WorkerEventType::CONN_ERROR;
      ev.peerKey = peerKey_;
      ev.errorMsg = std::string("poll failed: ") + strerror(errno);
      PostEvent(std::move(ev));
      break;
    }

    if (pfds[1].revents & POLLIN) {
      uint64_t v;
      while (::read(wakeFd_, &v, sizeof(v)) > 0) {
      }
      if (recv_.state == RecvState::AwaitTarget && !ProcessRecv()) break;
    }
    if (pfds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
      WorkerEvent ev;
      ev.type = WorkerEventType::CONN_ERROR;
      ev.peerKey = peerKey_;
      ev.errorMsg = "data connection error/hangup";
      PostEvent(std::move(ev));
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
        WorkerEvent ev;
        ev.type = WorkerEventType::CONN_ERROR;
        ev.peerKey = peerKey_;
        ev.errorMsg = std::string("sendmsg failed: ") + strerror(errno);
        PostEvent(std::move(ev));
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
    if (recv_.state == RecvState::ReadHeader) {
      ssize_t n = RecvSome(recv_.hdr.data() + recv_.hdrGot, tcp::kDataHeaderSize - recv_.hdrGot);
      if (n < 0) {
        if (IsWouldBlock(errno)) return true;
        WorkerEvent ev;
        ev.type = WorkerEventType::CONN_ERROR;
        ev.peerKey = peerKey_;
        ev.errorMsg = std::string("recv header failed: ") + strerror(errno);
        PostEvent(std::move(ev));
        return false;
      }
      if (n == 0) {
        WorkerEvent ev;
        ev.type = WorkerEventType::CONN_ERROR;
        ev.peerKey = peerKey_;
        ev.errorMsg = "data connection closed by peer";
        PostEvent(std::move(ev));
        return false;
      }
      recv_.hdrGot += static_cast<size_t>(n);
      if (recv_.hdrGot < tcp::kDataHeaderSize) return true;
      recv_.hdrGot = 0;
      if (tcp::TryParseDataHeader(recv_.hdr.data(), recv_.hdr.size(), &recv_.parsed) !=
          tcp::ParseError::Ok) {
        WorkerEvent ev;
        ev.type = WorkerEventType::CONN_ERROR;
        ev.peerKey = peerKey_;
        ev.errorMsg = "bad data header";
        PostEvent(std::move(ev));
        return false;
      }
      recv_.state = RecvState::AwaitTarget;
    }

    if (recv_.state == RecvState::AwaitTarget && !BeginPayload()) return true;
    if ((recv_.state == RecvState::RecvPayload || recv_.state == RecvState::DiscardPayload) &&
        !ProcessPayload())
      return false;
    if (recv_.state != RecvState::ReadHeader) return true;
  }
}

ssize_t DataConnectionWorker::RecvSome(void* dst, size_t len) {
  if (initialOffset_ < initialBytes_.size()) {
    size_t n = std::min(len, initialBytes_.size() - initialOffset_);
    std::memcpy(dst, initialBytes_.data() + initialOffset_, n);
    initialOffset_ += n;
    if (initialOffset_ == initialBytes_.size()) {
      initialBytes_.clear();
      initialOffset_ = 0;
    }
    return static_cast<ssize_t>(n);
  }
  return ::recv(fd_, dst, len, 0);
}

bool DataConnectionWorker::BeginPayload() {
  const RecvTargetKey key{recv_.parsed.kind, recv_.parsed.opId};
  {
    std::lock_guard<std::mutex> lk(recvTable_->mu);
    auto it = recvTable_->entries.find(key);
    if (it == recvTable_->entries.end()) {
      if (!recv_.awaitNotified) {
        WorkerEvent ev;
        ev.type = WorkerEventType::AWAIT_TARGET;
        ev.peerKey = peerKey_;
        ev.dataKind = recv_.parsed.kind;
        ev.opId = recv_.parsed.opId;
        ev.lane = recv_.parsed.lane;
        PostEvent(std::move(ev));
        recv_.awaitNotified = true;
      }
      return false;
    }
    recv_.activeTarget = it->second;
  }

  if (recv_.parsed.lane >= recv_.activeTarget.lanesTotal) {
    WorkerEvent ev;
    ev.type = WorkerEventType::CONN_ERROR;
    ev.peerKey = peerKey_;
    ev.errorMsg = "DATA lane exceeds request lanesTotal";
    PostEvent(std::move(ev));
    running_.store(false);
    return false;
  }

  const LaneSpan span = ComputeLaneSpan(recv_.activeTarget.totalLen, recv_.activeTarget.lanesTotal,
                                        recv_.parsed.lane);
  recv_.laneOffset = span.off;
  recv_.payloadOff = 0;
  recv_.payloadRemaining = recv_.parsed.payloadLen;
  recv_.segIndex = 0;
  recv_.segOffset = 0;
  recv_.laneSegs = SliceSegments(recv_.activeTarget.segs, span.off, span.len);
  const bool badLength = span.len != recv_.parsed.payloadLen;
  recv_.state = (recv_.activeTarget.discard || badLength) ? RecvState::DiscardPayload
                                                          : RecvState::RecvPayload;
  if (recv_.payloadRemaining == 0) FinishPayload();
  return true;
}

bool DataConnectionWorker::ProcessPayload() {
  while (recv_.payloadRemaining > 0) {
    ssize_t n = -1;
    if (recv_.state == RecvState::DiscardPayload) {
      uint8_t discard[65536];
      size_t want =
          static_cast<size_t>(std::min<uint64_t>(recv_.payloadRemaining, sizeof(discard)));
      n = RecvSome(discard, want);
    } else {
      assert(recv_.state == RecvState::RecvPayload);
      constexpr uint64_t kMaxDirectRecvBytes = 16ULL << 20;
      size_t want =
          static_cast<size_t>(std::min<uint64_t>(recv_.payloadRemaining, kMaxDirectRecvBytes));
      void* dst = nullptr;
      if (recv_.activeTarget.toGpu) {
        dst = static_cast<uint8_t*>(recv_.activeTarget.pinned->ptr) + recv_.laneOffset +
              recv_.payloadOff;
      } else {
        if (recv_.segIndex >= recv_.laneSegs.size()) {
          recv_.state = RecvState::DiscardPayload;
          continue;
        }
        const Segment& seg = recv_.laneSegs[recv_.segIndex];
        want = static_cast<size_t>(std::min<uint64_t>(want, seg.len - recv_.segOffset));
        dst = static_cast<uint8_t*>(recv_.activeTarget.cpuBase) + seg.off + recv_.segOffset;
      }
      n = RecvSome(dst, want);
    }

    if (n < 0) {
      if (IsWouldBlock(errno)) return true;
      WorkerEvent ev;
      ev.type = WorkerEventType::CONN_ERROR;
      ev.peerKey = peerKey_;
      ev.errorMsg = std::string("recv payload failed: ") + strerror(errno);
      PostEvent(std::move(ev));
      return false;
    }
    if (n == 0) {
      WorkerEvent ev;
      ev.type = WorkerEventType::CONN_ERROR;
      ev.peerKey = peerKey_;
      ev.errorMsg = "data connection closed during payload";
      PostEvent(std::move(ev));
      return false;
    }
    recv_.payloadRemaining -= static_cast<uint64_t>(n);
    recv_.payloadOff += static_cast<uint64_t>(n);
    if (recv_.state == RecvState::RecvPayload && !recv_.activeTarget.toGpu) {
      recv_.segOffset += static_cast<uint64_t>(n);
      if (recv_.segOffset == recv_.laneSegs[recv_.segIndex].len) {
        ++recv_.segIndex;
        recv_.segOffset = 0;
      }
    }
  }
  FinishPayload();
  return true;
}

void DataConnectionWorker::FinishPayload() {
  WorkerEvent ev;
  ev.type = WorkerEventType::RECV_DONE;
  ev.peerKey = peerKey_;
  ev.dataKind = recv_.parsed.kind;
  ev.opId = recv_.parsed.opId;
  ev.lane = recv_.parsed.lane;
  ev.laneLen = recv_.parsed.payloadLen;
  ev.discarded = recv_.state == RecvState::DiscardPayload;
  PostEvent(std::move(ev));
  recv_ = RecvCursor{};
}

// ===========================================================================
// Request parsing helpers (anonymous namespace)
// ===========================================================================
namespace {

constexpr size_t kMaxInboundWireOpsPerPeer = 1024;
constexpr size_t kMaxInboundStatusesPerPeer = 4096;
// GPU copies generally complete much faster than a network round trip. A
// coarse poll interval is paid once for D2H and again for H2D, directly adding
// twice that interval to every GPU transfer. Busy-poll only while GPU work is
// active to avoid adding scheduler-granularity latency; the reactor still
// blocks indefinitely when gpuTasks_ is empty.
constexpr int kGpuPollIntervalMs = 0;

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
    if (n == 0 || n > (r.len - r.off) / 16) return false;
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
  if (!r.u8(&out->lanesTotal) || r.off != r.len) return false;
  return out->lanesTotal >= 1 && out->lanesTotal <= tcp::kMaxLanes;
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
  if (msgLen > r.len - r.off || r.off + msgLen != r.len) return false;
  out->msg.assign(reinterpret_cast<const char*>(body + r.off), msgLen);
  return true;
}

bool IsTerminalStatusCode(uint32_t code) {
  switch (static_cast<StatusCode>(code)) {
    case StatusCode::SUCCESS:
    case StatusCode::ERR_INVALID_ARGS:
    case StatusCode::ERR_NOT_FOUND:
    case StatusCode::ERR_RDMA_OP:
    case StatusCode::ERR_BAD_STATE:
    case StatusCode::ERR_GPU_OP:
      return true;
    default:
      return false;
  }
}

}  // namespace

// ===========================================================================
// TcpTransport
// ===========================================================================
TcpTransport::TcpTransport(EngineKey myKey, const IOEngineConfig& engCfg,
                           const TcpBackendConfig& cfg)
    : myEngKey_(std::move(myKey)), engConfig_(engCfg), config_(cfg) {
  if (config_.numDataConns < 1 || config_.numDataConns > tcp::kMaxLanes)
    throw std::invalid_argument("TCP numDataConns must be in [1, 16]");
}

TcpTransport::~TcpTransport() { Shutdown(); }

void TcpTransport::Start() {
  if (running_.load()) return;

  epfd_ = epoll_create1(EPOLL_CLOEXEC);
  if (epfd_ < 0)
    throw std::runtime_error(std::string("TCP: epoll_create1 failed: ") + strerror(errno));

  auto failStart = [this](const std::string& message) {
    if (listenFd_ >= 0) close(listenFd_);
    if (wakeFd_ >= 0) close(wakeFd_);
    if (epfd_ >= 0) close(epfd_);
    listenFd_ = wakeFd_ = epfd_ = -1;
    listenPort_ = 0;
    throw std::runtime_error(message);
  };

  listenFd_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (listenFd_ < 0) failStart(std::string("TCP: socket failed: ") + strerror(errno));

  int one = 1;
  SetSockOpt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one), "SO_REUSEADDR");

  auto addrOpt = ParseIpv4(engConfig_.host.empty() ? "0.0.0.0" : engConfig_.host, engConfig_.port);
  if (!addrOpt) failStart("TCP: invalid listen IPv4 address: " + engConfig_.host);
  sockaddr_in addr = *addrOpt;
  if (bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::string error = strerror(errno);
    failStart("TCP: bind " + engConfig_.host + ":" + std::to_string(engConfig_.port) +
              " failed: " + error);
  }
  if (listen(listenFd_, 256) != 0) {
    std::string error = strerror(errno);
    failStart("TCP: listen failed: " + error);
  }
  listenPort_ = GetBoundPort(listenFd_);
  MORI_IO_INFO("TCP: listen on {}:{} (port={})", engConfig_.host, engConfig_.port, listenPort_);

  wakeFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (wakeFd_ < 0) failStart(std::string("TCP: eventfd failed: ") + strerror(errno));

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
  if (it->second.empty()) inboundStatus_.erase(it);
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
  if (localOffset > local.size || size > local.size - localOffset || remoteOffset > remote.size ||
      size > remote.size - remoteOffset) {
    status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: offset+size out of range");
    return;
  }

  auto op = std::make_unique<OutboundOpState>();
  op->key = {remote.engineKey, id};
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
    if (localOffsets[i] > local.size || sizes[i] > local.size - localOffsets[i] ||
        remoteOffsets[i] > remote.size || sizes[i] > remote.size - remoteOffsets[i] ||
        sizes[i] > std::numeric_limits<uint64_t>::max() - total) {
      status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: batch offset+size out of range");
      return;
    }
    lSegs.push_back({uint64_t(localOffsets[i]), uint64_t(sizes[i])});
    rSegs.push_back({uint64_t(remoteOffsets[i]), uint64_t(sizes[i])});
    total += sizes[i];
  }
  if (total == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
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
  op->key = {remote.engineKey, id};
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
  if (closedThisBatch_) closedThisBatch_->insert(c->fd);
  DelEpoll(c->fd);
  shutdown(c->fd, SHUT_RDWR);
  close(c->fd);
  c->fd = -1;
}

// ---------------------------------------------------------------------------
// Connection management
// ---------------------------------------------------------------------------
TcpTransport::ConnResult TcpTransport::AssignConnToPeer(Connection* c) {
  assert(c && c->helloReceived);
  PeerLinks& link = peers_[c->peerKey];

  auto handoffData = [&]() -> ConnResult {
    if (c->handedOff || !c->sendq.empty()) return ConnResult::Alive;
    const int dataFd = c->fd;
    const EngineKey peer = c->peerKey;
    c->handedOff = true;
    link.dataFds.push_back(dataFd);
    MORI_IO_TRACE("TCP: peer {} DATA conn up {}/{}", peer, link.dataFds.size(),
                  config_.numDataConns);
    DelEpoll(dataFd);
    SetNonBlocking(dataFd);
    ConfigureDataSocket(dataFd, config_);
    auto worker =
        std::make_unique<DataConnectionWorker>(dataFd, peer, link.recvTargets, std::move(c->inbuf));
    worker->Start();
    AddEpoll(worker->NotifyFd(), true, false);
    workerNotifyMap_[worker->NotifyFd()] = worker.get();
    link.workers.push_back(worker.get());
    dataWorkers_[dataFd] = std::move(worker);
    // DATA fd ownership is now exclusively held by dataWorkers_. Keeping a
    // Connection alias would let reactor teardown close a live worker fd.
    conns_.erase(dataFd);
    MaybeDispatchInboundReads(peer);
    MaybeDispatchQueuedOps(peer);
    return ConnResult::Gone;
  };

  if (c->assigned) {
    return c->ch == tcp::Channel::DATA ? handoffData() : ConnResult::Alive;
  }

  if (c->isOutgoing) {
    if (c->ch == tcp::Channel::CTRL) {
      if (link.ctrlPending > 0) link.ctrlPending--;
    } else {
      if (link.dataPending > 0) link.dataPending--;
    }
  }

  if (c->ch == tcp::Channel::CTRL) {
    // The first completed CTRL connection remains primary. Staggered dialing
    // below makes both peers converge on the same physical connection.
    if (link.ctrlFd < 0) {
      link.ctrlFd = c->fd;
      c->assigned = true;
      return ConnResult::Alive;
    }
    int existFd = link.ctrlFd;
    auto eIt = conns_.find(existFd);
    if (eIt == conns_.end()) {
      link.ctrlFd = c->fd;
      c->assigned = true;
      return ConnResult::Alive;
    }
    MORI_IO_TRACE("TCP: peer {} CTRL dropping duplicate fd {}", c->peerKey, c->fd);
    int fd = c->fd;
    CloseConnInternal(c);
    conns_.erase(fd);
    return ConnResult::Gone;
  }

  // DATA channel
  size_t want = static_cast<size_t>(config_.numDataConns);
  if (link.dataFds.size() >= want) {
    MORI_IO_TRACE("TCP: peer {} dropping extra DATA fd {}", c->peerKey, c->fd);
    int fd = c->fd;
    CloseConnInternal(c);
    conns_.erase(fd);
    return ConnResult::Gone;
  }
  c->assigned = true;
  return handoffData();
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
  if (myEngKey_ > peer && !IsPeerReady(peer)) {
    auto now = Clock::now();
    if (link.connectNotBefore == Clock::time_point::max())
      link.connectNotBefore = now + std::chrono::milliseconds(20);
    NoteAuxDeadline(link.connectNotBefore);
    if (now < link.connectNotBefore) return;
  }
  link.connectNotBefore = Clock::time_point::max();
  if (!link.CtrlUp() && link.ctrlPending == 0) ConnectChannel(peer, tcp::Channel::CTRL);
  int want = config_.numDataConns;
  while (int(link.dataFds.size()) + link.dataPending < want)
    if (!ConnectChannel(peer, tcp::Channel::DATA)) break;
}

bool TcpTransport::ConnectChannel(const EngineKey& peer, tcp::Channel ch) {
  EngineDesc desc;
  {
    std::lock_guard<std::mutex> lk(remoteMu_);
    auto it = remoteEngines_.find(peer);
    if (it == remoteEngines_.end()) {
      MORI_IO_ERROR("TCP: remote engine {} not registered", peer);
      return false;
    }
    desc = it->second;
  }

  auto peerAddr = ParseIpv4(desc.host, static_cast<uint16_t>(desc.port));
  if (!peerAddr) {
    MORI_IO_ERROR("TCP: invalid remote host {}:{}", desc.host, desc.port);
    return false;
  }

  int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    MORI_IO_ERROR("TCP: socket() failed: {}", strerror(errno));
    return false;
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
      return false;
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
  return true;
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
  DrainSubmissions();
}

bool TcpTransport::IsPeerReady(const EngineKey& peer) {
  auto it = peers_.find(peer);
  if (it == peers_.end() || !it->second.CtrlUp() || !it->second.DataUp()) return false;
  auto cit = conns_.find(it->second.ctrlFd);
  if (cit == conns_.end() || !cit->second->helloReceived) return false;
  return !it->second.workers.empty();
}

bool TcpTransport::RegisterRecvTarget(const EngineKey& peer, const RecvTargetKey& key,
                                      const WorkerRecvTarget& target) {
  auto pit = peers_.find(peer);
  if (pit == peers_.end()) return false;
  {
    std::lock_guard<std::mutex> lk(pit->second.recvTargets->mu);
    if (!pit->second.recvTargets->entries.emplace(key, target).second) return false;
  }
  auto awaitingPeer = awaitingTargets_.find(peer);
  if (awaitingPeer != awaitingTargets_.end()) {
    awaitingPeer->second.erase(key);
    if (awaitingPeer->second.empty()) awaitingTargets_.erase(awaitingPeer);
  }
  for (auto* worker : pit->second.workers) worker->WakeWorker();
  return true;
}

void TcpTransport::RemoveRecvTarget(const EngineKey& peer, const RecvTargetKey& key) {
  auto pit = peers_.find(peer);
  if (pit == peers_.end()) return;
  std::lock_guard<std::mutex> lk(pit->second.recvTargets->mu);
  pit->second.recvTargets->entries.erase(key);
}

bool TcpTransport::HasRecvTarget(const EngineKey& peer, const RecvTargetKey& key) {
  auto pit = peers_.find(peer);
  if (pit == peers_.end()) return false;
  std::lock_guard<std::mutex> lk(pit->second.recvTargets->mu);
  return pit->second.recvTargets->entries.count(key) != 0;
}

std::vector<DataConnectionWorker*> TcpTransport::SelectWorkers(PeerLinks& links,
                                                               uint8_t lanesTotal) {
  std::vector<DataConnectionWorker*> selected;
  if (links.workers.empty()) return selected;
  lanesTotal = std::min<uint8_t>(lanesTotal, static_cast<uint8_t>(links.workers.size()));
  selected.reserve(lanesTotal);
  for (uint8_t lane = 0; lane < lanesTotal; ++lane)
    selected.push_back(links.workers[(links.nextWorker + lane) % links.workers.size()]);
  links.nextWorker = (links.nextWorker + lanesTotal) % links.workers.size();
  return selected;
}

// ---------------------------------------------------------------------------
// DispatchOp - initiate an outbound operation
// ---------------------------------------------------------------------------
void TcpTransport::DispatchOp(std::unique_ptr<OutboundOpState> op) {
  if (!op) {
    MORI_IO_ERROR("TCP: DispatchOp got null op");
    return;
  }
  const EngineKey peerKey = op->key.peer;
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

  const OpKey key = op->key;
  const TransferUniqueId opId = key.id;
  if (pendingOutbound_.count(key)) {
    MORI_IO_ERROR("TCP: duplicate op peer={} id={}", peerKey, opId);
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: duplicate peer/op id");
    return;
  }
  auto [itIns, inserted] = pendingOutbound_.emplace(key, std::move(op));
  assert(inserted);
  OutboundOpState* st = itIns->second.get();

  // Decide lane count for striping
  const uint64_t totalBytes = SumLens(st->localSegs);
  int wantLanes = std::min<int>(config_.numDataConns, workerList.size());
  uint8_t lanesTotal = 1;
  if (wantLanes > 1 && config_.stripingThresholdBytes > 0 &&
      totalBytes >= uint64_t(config_.stripingThresholdBytes)) {
    lanesTotal = static_cast<uint8_t>(wantLanes);
  }
  const auto& destinationSegs = st->isRead ? st->localSegs : st->remoteSegs;
  if (lanesTotal > 1 && SegmentsOverlap(destinationSegs)) {
    FinalizeOutbound(key, StatusCode::ERR_INVALID_ARGS,
                     "TCP: overlapping destination segments cannot be striped");
    return;
  }
  st->lanesTotal = lanesTotal;

  // Allocate pinned staging for GPU reads
  if (st->isRead && st->local.loc == MemoryLocationType::GPU) {
    st->pinned = staging_.Acquire(static_cast<size_t>(totalBytes));
    if (!st->pinned) {
      FinalizeOutbound(key, StatusCode::ERR_BAD_STATE, "TCP: staging alloc failed");
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
    if (!RegisterRecvTarget(peerKey, {tcp::DataKind::READ_RESPONSE, opId}, target)) {
      FinalizeOutbound(key, StatusCode::ERR_BAD_STATE, "TCP: duplicate receive target");
      return;
    }
  }

  // Build and send ctrl frame
  std::vector<uint8_t> ctrlFrame;
  if (st->localSegs.size() == 1) {
    auto type = st->isRead ? tcp::CtrlMsgType::READ_REQ : tcp::CtrlMsgType::WRITE_REQ;
    ctrlFrame = tcp::BuildLinearReq(type, st->key.id, st->remote.id, st->remoteSegs[0].off,
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
    ctrlFrame = tcp::BuildBatchReq(type, st->key.id, st->remote.id, roffs, szs, lanesTotal);
  }
  QueueSend(ctrl->fd, std::move(ctrlFrame));

  if (!st->isRead) {
    auto selected = SelectWorkers(pit->second, lanesTotal);
    auto sent = std::make_shared<uint8_t>(0);
    if (!QueueDataSend(peerKey, selected, st->local, st->localSegs, tcp::DataKind::WRITE_PAYLOAD,
                       st->key.id, lanesTotal, [this, key, sent]() {
                         auto it = pendingOutbound_.find(key);
                         if (it == pendingOutbound_.end()) return;
                         uint8_t completed = (*sent)++;
                         if (completed < tcp::kMaxLanes)
                           it->second->lanesSentMask |= uint16_t(1U << completed);
                         MaybeCompleteOutbound(*it->second);
                       })) {
      AbortPeer(peerKey, StatusCode::ERR_GPU_OP, "TCP: failed to prepare GPU payload");
      return;
    }
  }
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
bool TcpTransport::QueueDataSend(const EngineKey& peer,
                                 const std::vector<DataConnectionWorker*>& workers,
                                 const MemoryDesc& src, const std::vector<Segment>& srcSegs,
                                 tcp::DataKind kind, uint64_t opId, uint8_t lanesTotal,
                                 std::function<void()> onLaneDone) {
  if (workers.empty() || lanesTotal == 0 || lanesTotal > tcp::kMaxLanes) return false;
  const uint64_t total = SumLens(srcSegs);

  if (src.loc == MemoryLocationType::GPU) {
    // GPU path: DtoH copy, then send from pinned buffer
    auto pinned = staging_.Acquire(static_cast<size_t>(total));
    if (!pinned) {
      MORI_IO_ERROR("TCP: staging alloc failed for GPU send");
      return false;
    }

    auto workersCopy = workers;
    auto sendCb = [workersCopy, pinned, kind, opId, lanesTotal, total,
                   onLaneDone = std::move(onLaneDone)]() {
      for (uint8_t lane = 0; lane < lanesTotal; ++lane) {
        LaneSpan span = ComputeLaneSpan(total, lanesTotal, lane);
        SendItem item;
        auto header = tcp::BuildDataHeader(kind, lane, opId, span.len);
        item.header.assign(header.begin(), header.end());
        item.iov = {{item.header.data(), item.header.size()},
                    {static_cast<uint8_t*>(pinned->ptr) + span.off, size_t(span.len)}};
        item.keepalive = pinned;
        item.onDone = onLaneDone;
        workersCopy[lane % workersCopy.size()]->SubmitSend(std::move(item));
      }
    };
    return ScheduleGpuCopy(src.deviceId, false, src, srcSegs, pinned, peer, opId,
                           std::move(sendCb));
  }

  // CPU path
  uint8_t* base = reinterpret_cast<uint8_t*>(src.data);
  for (uint8_t lane = 0; lane < lanesTotal; ++lane) {
    LaneSpan span = ComputeLaneSpan(total, lanesTotal, lane);
    auto laneSegs = SliceSegments(srcSegs, span.off, span.len);
    SendItem item;
    auto header = tcp::BuildDataHeader(kind, lane, opId, span.len);
    item.header.assign(header.begin(), header.end());
    item.iov.reserve(1 + laneSegs.size());
    item.iov.push_back({item.header.data(), item.header.size()});
    for (auto& seg : laneSegs) item.iov.push_back({base + seg.off, size_t(seg.len)});
    item.onDone = onLaneDone;
    workers[lane % workers.size()]->SubmitSend(std::move(item));
  }
  return true;
}

// ---------------------------------------------------------------------------
// GPU copy (unified DtoH / HtoD)
// ---------------------------------------------------------------------------
bool TcpTransport::ScheduleGpuCopy(int deviceId, bool toDevice, const MemoryDesc& mem,
                                   const std::vector<Segment>& segs,
                                   std::shared_ptr<PinnedBuf> pinned, const EngineKey& peer,
                                   TransferUniqueId opId, std::function<void()> onComplete) {
  const uint64_t total = SumLens(segs);
  hipStream_t stream = streamPool_.GetNextStream(deviceId);
  hipEvent_t ev = eventPool_.GetEvent(deviceId);
  if (!stream || !ev) {
    MORI_IO_ERROR("TCP: failed to get HIP stream/event");
    if (ev) eventPool_.PutEvent(ev, deviceId);
    return false;
  }

  hipError_t hipStatus = hipSetDevice(deviceId);
  if (hipStatus != hipSuccess) {
    MORI_IO_ERROR("TCP: hipSetDevice({}) failed: {}", deviceId, hipGetErrorString(hipStatus));
    eventPool_.PutEvent(ev, deviceId);
    return false;
  }
  uint8_t* hostPtr = reinterpret_cast<uint8_t*>(pinned->ptr);
  bool copySubmitted = false;
  auto submitCopy = [&](hipError_t result, const char* operation) {
    if (result == hipSuccess) {
      copySubmitted = true;
      return true;
    }
    MORI_IO_ERROR("TCP: {} failed: {}", operation, hipGetErrorString(result));
    if (copySubmitted) {
      hipError_t syncStatus = hipStreamSynchronize(stream);
      if (syncStatus != hipSuccess)
        MORI_IO_WARN("TCP: hipStreamSynchronize after copy failure failed: {}",
                     hipGetErrorString(syncStatus));
    }
    eventPool_.PutEvent(ev, deviceId);
    return false;
  };

  uint64_t spanOff = 0, spanLen = 0;
  if (IsSingleContiguousSpan(segs, &spanOff, &spanLen) && spanLen == total) {
    if (toDevice) {
      void* gpu = reinterpret_cast<void*>(mem.data + spanOff);
      if (!submitCopy(hipMemcpyHtoDAsync(gpu, hostPtr, size_t(total), stream),
                      "hipMemcpyHtoDAsync"))
        return false;
    } else {
      hipDeviceptr_t gpu = reinterpret_cast<hipDeviceptr_t>(mem.data + spanOff);
      if (!submitCopy(hipMemcpyDtoHAsync(hostPtr, gpu, size_t(total), stream),
                      "hipMemcpyDtoHAsync"))
        return false;
    }
  } else {
    uint64_t off = 0;
    for (auto& s : segs) {
      if (toDevice) {
        void* gpu = reinterpret_cast<void*>(mem.data + s.off);
        if (!submitCopy(hipMemcpyHtoDAsync(gpu, hostPtr + off, size_t(s.len), stream),
                        "hipMemcpyHtoDAsync"))
          return false;
      } else {
        hipDeviceptr_t gpu = reinterpret_cast<hipDeviceptr_t>(mem.data + s.off);
        if (!submitCopy(hipMemcpyDtoHAsync(hostPtr + off, gpu, size_t(s.len), stream),
                        "hipMemcpyDtoHAsync"))
          return false;
      }
      off += s.len;
    }
  }
  hipStatus = hipEventRecord(ev, stream);
  if (hipStatus != hipSuccess) {
    MORI_IO_ERROR("TCP: hipEventRecord failed: {}", hipGetErrorString(hipStatus));
    hipError_t syncStatus = hipStreamSynchronize(stream);
    if (syncStatus != hipSuccess)
      MORI_IO_WARN("TCP: hipStreamSynchronize after event failure failed: {}",
                   hipGetErrorString(syncStatus));
    eventPool_.PutEvent(ev, deviceId);
    return false;
  }
  gpuTasks_.push_back({peer, opId, deviceId, ev, std::move(pinned), std::move(onComplete)});
  return true;
}

void TcpTransport::PollGpuTasks() {
  for (auto it = gpuTasks_.begin(); it != gpuTasks_.end();) {
    hipError_t setStatus = hipSetDevice(it->deviceId);
    if (setStatus != hipSuccess) {
      EngineKey peer = it->peer;
      MORI_IO_ERROR("TCP: hipSetDevice({}) failed while polling: {}", it->deviceId,
                    hipGetErrorString(setStatus));
      eventPool_.PutEvent(it->ev, it->deviceId);
      it = gpuTasks_.erase(it);
      if (!peer.empty()) {
        AbortPeer(std::move(peer), StatusCode::ERR_GPU_OP, "TCP: HIP device selection failed");
        it = gpuTasks_.begin();
      }
      continue;
    }
    hipError_t st = hipEventQuery(it->ev);
    if (st == hipSuccess) {
      // Remove the task before invoking user-visible transport callbacks. A
      // callback may synchronously AbortPeer() and mutate gpuTasks_; keeping
      // the current element in the deque would invalidate `it` and return the
      // same event to eventPool_ twice.
      GpuTask task = std::move(*it);
      it = gpuTasks_.erase(it);
      eventPool_.PutEvent(task.ev, task.deviceId);
      if (task.onReady) task.onReady();
      it = gpuTasks_.begin();
    } else if (st == hipErrorNotReady) {
      ++it;
    } else {
      MORI_IO_ERROR("TCP: hipEventQuery failed: {}", hipGetErrorString(st));
      EngineKey peer = it->peer;
      eventPool_.PutEvent(it->ev, it->deviceId);
      it = gpuTasks_.erase(it);
      if (!peer.empty()) {
        AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: HIP event query failed");
        it = gpuTasks_.begin();
      }
    }
  }
}

void TcpTransport::DrainGpuTasksForPeer(const EngineKey& peer) {
  for (auto it = gpuTasks_.begin(); it != gpuTasks_.end();) {
    if (it->peer != peer) {
      ++it;
      continue;
    }
    hipError_t setStatus = hipSetDevice(it->deviceId);
    if (setStatus != hipSuccess)
      MORI_IO_WARN("TCP: hipSetDevice({}) during abort failed: {}", it->deviceId,
                   hipGetErrorString(setStatus));
    hipError_t status = hipEventSynchronize(it->ev);
    if (status != hipSuccess)
      MORI_IO_WARN("TCP: hipEventSynchronize during abort failed: {}", hipGetErrorString(status));
    eventPool_.PutEvent(it->ev, it->deviceId);
    it = gpuTasks_.erase(it);
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
  const int fd = c->fd;
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
  UpdateWriteInterest(fd);
  auto it = conns_.find(fd);
  if (it != conns_.end() && it->second->helloReceived && it->second->ch == tcp::Channel::DATA &&
      it->second->sendq.empty())
    AssignConnToPeer(it->second.get());
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
    if (closedThisBatch_) closedThisBatch_->insert(fd);
    shutdown(fd, SHUT_RDWR);
    close(fd);
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
  if (peer.empty()) {
    auto it = conns_.find(fd);
    if (it != conns_.end()) peer = it->second->peerKey;
  }
  if (!peer.empty())
    AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: connection lost");
  else
    CloseAndRemoveFd(fd);
}

void TcpTransport::AbortPeer(EngineKey peer, StatusCode code, std::string reason) {
  MORI_IO_WARN("TCP: abort peer={} code={} reason={}", peer, static_cast<uint32_t>(code), reason);
  auto pit = peers_.find(peer);
  std::vector<int> peerFds;
  if (pit != peers_.end()) {
    if (pit->second.ctrlFd >= 0) peerFds.push_back(pit->second.ctrlFd);
    peerFds.insert(peerFds.end(), pit->second.dataFds.begin(), pit->second.dataFds.end());
  }
  for (auto& [fd, conn] : conns_)
    if (conn->peerKey == peer && std::find(peerFds.begin(), peerFds.end(), fd) == peerFds.end())
      peerFds.push_back(fd);

  // shutdown() wakes worker poll/recv without releasing the fd number. close()
  // happens only after the worker has joined, so it cannot hit a reused fd.
  for (int fd : peerFds) {
    if (closedThisBatch_) closedThisBatch_->insert(fd);
    auto cit = conns_.find(fd);
    if (cit != conns_.end()) DelEpoll(fd);
    shutdown(fd, SHUT_RDWR);
  }
  // Then join every worker before publishing any caller-visible terminal status.
  for (int fd : peerFds) {
    auto wit = dataWorkers_.find(fd);
    if (wit == dataWorkers_.end()) continue;
    int notifyFd = wit->second->NotifyFd();
    wit->second->Stop();
    DelEpoll(notifyFd);
    workerNotifyMap_.erase(notifyFd);
    dataWorkers_.erase(wit);
    close(fd);
  }
  DrainGpuTasksForPeer(peer);
  for (int fd : peerFds) {
    auto cit = conns_.find(fd);
    if (cit == conns_.end()) continue;
    close(fd);
    cit->second->fd = -1;
    conns_.erase(cit);
  }

  if (pit != peers_.end()) {
    std::lock_guard<std::mutex> lk(pit->second.recvTargets->mu);
    pit->second.recvTargets->entries.clear();
  }
  peers_.erase(peer);
  inboundWrites_.erase(peer);
  pendingInboundReads_.erase(peer);
  awaitingTargets_.erase(peer);

  std::vector<TransferStatus*> statuses;
  for (auto it = pendingOutbound_.begin(); it != pendingOutbound_.end();) {
    if (it->first.peer == peer) {
      statuses.push_back(it->second->status);
      it = pendingOutbound_.erase(it);
    } else
      ++it;
  }
  auto waiting = waitingOps_.find(peer);
  if (waiting != waitingOps_.end()) {
    for (auto& op : waiting->second) statuses.push_back(op->status);
    waitingOps_.erase(waiting);
  }
  for (TransferStatus* status : statuses) status->Update(code, reason);
}

void TcpTransport::FinalizeOutbound(OpKey key, StatusCode code, std::string msg) {
  auto it = pendingOutbound_.find(key);
  if (it == pendingOutbound_.end()) return;
  OutboundOpState& op = *it->second;
  if (op.finished) return;
  op.finished = true;
  if (op.isRead) RemoveRecvTarget(key.peer, {tcp::DataKind::READ_RESPONSE, key.id});
  TransferStatus* status = op.status;
  pendingOutbound_.erase(it);
  status->Update(code, msg);
}

// ---------------------------------------------------------------------------
// Ctrl message handling
// ---------------------------------------------------------------------------
TcpTransport::ConnResult TcpTransport::HandleCtrlReadable(Connection* c) {
  const int fd = c->fd;
  while (true) {
    uint8_t tmp[65536];
    ssize_t n = ::recv(c->fd, tmp, sizeof(tmp), 0);
    if (n < 0) {
      if (IsWouldBlock(errno)) break;
      MORI_IO_ERROR("TCP: recv(ctrl) fd {} failed: {}", c->fd, strerror(errno));
      ClosePeerByFd(c->fd);
      return ConnResult::Gone;
    }
    if (n == 0) {
      ClosePeerByFd(c->fd);
      return ConnResult::Gone;
    }
    c->inbuf.insert(c->inbuf.end(), tmp, tmp + n);
  }

  while (true) {
    tcp::CtrlHeaderView hv;
    tcp::ParseError parse = tcp::TryParseCtrlHeader(c->inbuf.data(), c->inbuf.size(), &hv);
    if (parse != tcp::ParseError::Ok) {
      if (parse != tcp::ParseError::Truncated || c->inbuf.size() >= tcp::kCtrlHeaderSize) {
        MORI_IO_ERROR("TCP: bad ctrl header fd {}", c->fd);
        ClosePeerByFd(c->fd);
        return ConnResult::Gone;
      }
      break;
    }
    if (c->inbuf.size() < tcp::kCtrlHeaderSize + hv.bodyLen) break;

    std::vector<uint8_t> body(c->inbuf.begin() + tcp::kCtrlHeaderSize,
                              c->inbuf.begin() + tcp::kCtrlHeaderSize + hv.bodyLen);
    c->inbuf.erase(c->inbuf.begin(), c->inbuf.begin() + tcp::kCtrlHeaderSize + hv.bodyLen);
    if (HandleCtrlFrame(c, hv.type, body.data(), body.size()) == ConnResult::Gone)
      return ConnResult::Gone;
    auto it = conns_.find(fd);
    if (it == conns_.end()) return ConnResult::Gone;
    c = it->second.get();
    if (c->handedOff) return ConnResult::Alive;
  }
  return ConnResult::Alive;
}

TcpTransport::ConnResult TcpTransport::HandleCtrlFrame(Connection* c, tcp::CtrlMsgType type,
                                                       const uint8_t* body, size_t len) {
  if (type == tcp::CtrlMsgType::HELLO) {
    return HandleHello(c, body, len);
  }
  if (!c->helloReceived) {
    ClosePeerByFd(c->fd);
    return ConnResult::Gone;
  }
  const EngineKey peer = c->peerKey;

  switch (type) {
    case tcp::CtrlMsgType::WRITE_REQ:
    case tcp::CtrlMsgType::READ_REQ:
    case tcp::CtrlMsgType::BATCH_WRITE_REQ:
    case tcp::CtrlMsgType::BATCH_READ_REQ:
      HandleRequest(peer, type, body, len);
      break;
    case tcp::CtrlMsgType::COMPLETION:
      HandleCompletion(peer, body, len);
      break;
    default:
      AbortPeer(peer, StatusCode::ERR_INVALID_ARGS, "TCP: unknown ctrl message");
      return ConnResult::Gone;
  }
  return peers_.count(peer) ? ConnResult::Alive : ConnResult::Gone;
}

TcpTransport::ConnResult TcpTransport::HandleHello(Connection* c, const uint8_t* body, size_t len) {
  if (c->helloReceived) {
    EngineKey peer = c->peerKey;
    if (peer.empty())
      CloseAndRemoveFd(c->fd);
    else
      AbortPeer(std::move(peer), StatusCode::ERR_INVALID_ARGS, "TCP: duplicate HELLO");
    return ConnResult::Gone;
  }
  if (len < 5) {
    MORI_IO_WARN("TCP: bad HELLO len {}", len);
    ClosePeerByFd(c->fd);
    return ConnResult::Gone;
  }
  tcp::WireReader r{body, len};
  uint8_t chRaw;
  uint32_t keyLen;
  if (!r.u8(&chRaw) || !r.u32(&keyLen) || keyLen == 0 || r.off + keyLen != len ||
      (chRaw != uint8_t(tcp::Channel::CTRL) && chRaw != uint8_t(tcp::Channel::DATA))) {
    ClosePeerByFd(c->fd);
    return ConnResult::Gone;
  }

  EngineKey helloPeer(reinterpret_cast<const char*>(body + r.off), keyLen);
  if (c->isOutgoing && !c->peerKey.empty() && c->peerKey != helloPeer) {
    AbortPeer(c->peerKey, StatusCode::ERR_INVALID_ARGS, "TCP: HELLO peer mismatch");
    return ConnResult::Gone;
  }
  c->peerKey = std::move(helloPeer);
  c->ch = static_cast<tcp::Channel>(chRaw);
  c->helloReceived = true;
  MORI_IO_TRACE("TCP: recv HELLO fd={} peer={} ch={} out={}", c->fd, c->peerKey, int(c->ch),
                c->isOutgoing);

  const int fd = c->fd;
  const EngineKey peer = c->peerKey;
  if (!c->helloSent) {
    QueueHello(fd);
    UpdateWriteInterest(fd);
  }
  auto it = conns_.find(fd);
  if (it == conns_.end()) return ConnResult::Gone;
  c = it->second.get();
  if (c->ch == tcp::Channel::CTRL) ConfigureCtrlSocket(c->fd, config_);
  if (AssignConnToPeer(c) == ConnResult::Gone) return ConnResult::Gone;
  MaybeDispatchQueuedOps(peer);
  return ConnResult::Alive;
}

// Unified handler for WRITE_REQ, READ_REQ, BATCH_WRITE_REQ, BATCH_READ_REQ
void TcpTransport::HandleRequest(const EngineKey& peer, tcp::CtrlMsgType type, const uint8_t* body,
                                 size_t len) {
  RequestView req;
  if (!ParseRequest(type, body, len, &req)) {
    MORI_IO_WARN("TCP: malformed request type={}", uint8_t(type));
    AbortPeer(peer, StatusCode::ERR_INVALID_ARGS, "TCP: malformed request");
    return;
  }

  bool isWrite = (type == tcp::CtrlMsgType::WRITE_REQ || type == tcp::CtrlMsgType::BATCH_WRITE_REQ);
  bool isBatch =
      (type == tcp::CtrlMsgType::BATCH_WRITE_REQ || type == tcp::CtrlMsgType::BATCH_READ_REQ);

  uint64_t requestBytes = 0;
  if (!TrySumLens(req.segs, &requestBytes)) {
    AbortPeer(peer, StatusCode::ERR_INVALID_ARGS, "TCP: request length overflow");
    return;
  }

  auto memOpt = LookupLocalMem(req.memId);

  if (isWrite) {
    RecvTargetKey targetKey{tcp::DataKind::WRITE_PAYLOAD, req.opId};
    auto peerWrites = inboundWrites_.find(peer);
    if (HasRecvTarget(peer, targetKey) ||
        (peerWrites != inboundWrites_.end() && peerWrites->second.count(req.opId))) {
      AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: duplicate inbound write op id");
      return;
    }
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
    if (ws.discard && peers_.count(peer)) {
      auto iw = inboundWrites_.find(peer);
      if (iw != inboundWrites_.end()) {
        auto state = iw->second.find(req.opId);
        if (state != iw->second.end()) state->second.completionSent = true;
      }
      SendCompletionAndRecord(peer, req.opId, StatusCode::ERR_INVALID_ARGS,
                              "TCP: invalid write request");
    }
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

    InboundReadState read;
    read.peer = peer;
    read.id = req.opId;
    read.src = *memOpt;
    read.srcSegs = std::move(req.segs);
    read.lanesTotal = req.lanesTotal;
    if (config_.opTimeoutMs > 0)
      read.deadline = Clock::now() + std::chrono::milliseconds(config_.opTimeoutMs);

    auto pit = peers_.find(peer);
    if (pit == peers_.end() || pit->second.workers.empty()) {
      if (pendingInboundReads_[peer].size() >= kMaxInboundWireOpsPerPeer) {
        AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: inbound read quota exceeded");
        return;
      }
      NoteAuxDeadline(read.deadline);
      pendingInboundReads_[peer].push_back(std::move(read));
      return;
    }
    DispatchInboundRead(std::move(read));
  }
}

void TcpTransport::HandleCompletion(const EngineKey& peer, const uint8_t* body, size_t len) {
  CompletionView msg;
  if (!ParseCompletion(body, len, &msg)) {
    MORI_IO_WARN("TCP: malformed COMPLETION");
    AbortPeer(peer, StatusCode::ERR_INVALID_ARGS, "TCP: malformed completion");
    return;
  }
  if (!IsTerminalStatusCode(msg.statusCode)) {
    AbortPeer(peer, StatusCode::ERR_INVALID_ARGS, "TCP: invalid completion status code");
    return;
  }

  OpKey key{peer, msg.opId};
  auto it = pendingOutbound_.find(key);
  if (it == pendingOutbound_.end()) return;
  OutboundOpState& st = *it->second;
  st.completionReceived = true;
  st.completionCode = static_cast<StatusCode>(msg.statusCode);
  st.completionMsg = std::move(msg.msg);
  if (st.completionCode != StatusCode::SUCCESS) {
    uint16_t all = tcp::LanesAllMask(st.lanesTotal);
    if ((st.isRead && (st.lanesDoneMask & all) != all) ||
        (!st.isRead && (st.lanesSentMask & all) != all)) {
      AbortPeer(peer, st.completionCode, st.completionMsg);
      return;
    }
    FinalizeOutbound(key, st.completionCode, st.completionMsg);
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
  auto now = Clock::now();
  std::lock_guard<std::mutex> lk(inboundMu_);
  auto& statuses = inboundStatus_[peer];
  statuses[id] = {code, msg};
  Clock::time_point pruneAt =
      statuses.size() > kMaxInboundStatusesPerPeer ? now : now + std::chrono::seconds(1);
  nextStatusPrune_ = std::min(nextStatusPrune_, pruneAt);
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

void TcpTransport::DispatchInboundRead(InboundReadState read) {
  auto pit = peers_.find(read.peer);
  if (pit == peers_.end() || pit->second.workers.empty()) {
    if (pendingInboundReads_[read.peer].size() >= kMaxInboundWireOpsPerPeer) {
      AbortPeer(std::move(read.peer), StatusCode::ERR_BAD_STATE,
                "TCP: inbound read quota exceeded");
      return;
    }
    NoteAuxDeadline(read.deadline);
    pendingInboundReads_[read.peer].push_back(std::move(read));
    return;
  }

  auto selected = SelectWorkers(pit->second, read.lanesTotal);
  struct DoneState {
    EngineKey peer;
    TransferUniqueId opId{0};
    std::atomic<int> remaining{0};
  };
  auto done = std::make_shared<DoneState>();
  done->peer = read.peer;
  done->opId = read.id;
  done->remaining.store(read.lanesTotal);
  auto laneDone = [this, done]() {
    if (done->remaining.fetch_sub(1) > 1) return;
    SendCompletionAndRecord(done->peer, done->opId, StatusCode::SUCCESS, "");
  };
  if (!QueueDataSend(read.peer, selected, read.src, read.srcSegs, tcp::DataKind::READ_RESPONSE,
                     read.id, read.lanesTotal, std::move(laneDone)))
    AbortPeer(std::move(read.peer), StatusCode::ERR_GPU_OP, "TCP: failed to prepare read response");
}

void TcpTransport::MaybeDispatchInboundReads(const EngineKey& peer) {
  auto it = pendingInboundReads_.find(peer);
  if (it == pendingInboundReads_.end()) return;
  auto reads = std::move(it->second);
  pendingInboundReads_.erase(it);
  for (auto& read : reads) {
    if (!peers_.count(peer)) return;
    DispatchInboundRead(std::move(read));
  }
}

void TcpTransport::FinalizeInboundWriteSetup(const EngineKey& peer, TransferUniqueId opId,
                                             InboundWriteState& ws) {
  if (!ws.discard && ws.dst.loc == MemoryLocationType::GPU) {
    ws.pinned = staging_.Acquire(static_cast<size_t>(SumLens(ws.dstSegs)));
    if (!ws.pinned) ws.discard = true;
  }
  if (config_.opTimeoutMs > 0)
    ws.deadline = Clock::now() + std::chrono::milliseconds(config_.opTimeoutMs);
  NoteAuxDeadline(ws.deadline);
  auto& writes = inboundWrites_[peer];
  writes[opId] = ws;
  if (writes.size() > kMaxInboundWireOpsPerPeer) {
    AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: inbound write quota exceeded");
    return;
  }

  // Set up worker recv targets
  WorkerRecvTarget target;
  target.lanesTotal = ws.lanesTotal;
  target.totalLen = ws.discard ? 0 : SumLens(ws.dstSegs);
  target.discard = ws.discard;
  if (!ws.discard && ws.dst.loc == MemoryLocationType::GPU) {
    target.toGpu = true;
    target.pinned = ws.pinned;
  } else if (!ws.discard) {
    target.cpuBase = reinterpret_cast<void*>(ws.dst.data);
    target.segs = ws.dstSegs;
  }
  if (!RegisterRecvTarget(peer, {tcp::DataKind::WRITE_PAYLOAD, opId}, target))
    AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: duplicate inbound receive target");
}

void TcpTransport::MaybeFinalizeInboundWrite(const EngineKey& peer, TransferUniqueId opId) {
  auto iwIt = inboundWrites_.find(peer);
  if (iwIt == inboundWrites_.end()) return;
  auto wsIt = iwIt->second.find(opId);
  if (wsIt == iwIt->second.end()) return;

  InboundWriteState& ws = wsIt->second;
  if ((ws.lanesDoneMask & tcp::LanesAllMask(ws.lanesTotal)) != tcp::LanesAllMask(ws.lanesTotal))
    return;

  RemoveRecvTarget(peer, {tcp::DataKind::WRITE_PAYLOAD, opId});

  if (ws.discard) {
    if (!ws.completionSent)
      SendCompletionAndRecord(peer, opId, StatusCode::ERR_INVALID_ARGS, "TCP: write discarded");
  } else if (ws.dst.loc == MemoryLocationType::GPU) {
    if (!ws.pinned) {
      SendCompletionAndRecord(peer, opId, StatusCode::ERR_BAD_STATE,
                              "TCP: missing staging (write)");
    } else {
      auto pinnedRef = ws.pinned;
      bool ok = ScheduleGpuCopy(ws.dst.deviceId, true, ws.dst, ws.dstSegs, pinnedRef, peer, opId,
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
}

void TcpTransport::MaybeCompleteOutbound(OutboundOpState& st) {
  if (!st.completionReceived) return;
  uint16_t allMask = tcp::LanesAllMask(st.lanesTotal);
  if (st.isRead) {
    if ((st.lanesDoneMask & allMask) != allMask || st.rxBytes != st.expectedRxBytes ||
        st.gpuCopyPending)
      return;
  } else if ((st.lanesSentMask & allMask) != allMask) {
    return;
  }
  FinalizeOutbound(st.key, st.completionCode, st.completionMsg);
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
      case WorkerEventType::AWAIT_TARGET:
        if (!HasRecvTarget(ev.peerKey, {ev.dataKind, ev.opId})) {
          auto& targets = awaitingTargets_[ev.peerKey];
          RecvTargetKey targetKey{ev.dataKind, ev.opId};
          if (!targets.count(targetKey) && targets.size() >= kMaxInboundWireOpsPerPeer) {
            AbortPeer(ev.peerKey, StatusCode::ERR_BAD_STATE, "TCP: awaiting target quota exceeded");
            return;
          }
          auto deadline = config_.opTimeoutMs > 0
                              ? Clock::now() + std::chrono::milliseconds(config_.opTimeoutMs)
                              : Clock::time_point::max();
          targets[targetKey] = deadline;
          NoteAuxDeadline(deadline);
        }
        break;
      case WorkerEventType::SEND_CALLBACK:
        if (ev.callback) ev.callback();
        break;
      case WorkerEventType::CONN_ERROR:
        MORI_IO_WARN("TCP: worker error peer {}: {}", ev.peerKey, ev.errorMsg);
        AbortPeer(ev.peerKey, StatusCode::ERR_BAD_STATE, ev.errorMsg);
        return;
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
  auto awaitingPeer = awaitingTargets_.find(peer);
  if (awaitingPeer != awaitingTargets_.end()) {
    awaitingPeer->second.erase({ev.dataKind, opId});
    if (awaitingPeer->second.empty()) awaitingTargets_.erase(awaitingPeer);
  }

  if (ev.dataKind == tcp::DataKind::WRITE_PAYLOAD) {
    auto iwIt = inboundWrites_.find(peer);
    if (iwIt == inboundWrites_.end() || !iwIt->second.count(opId)) {
      AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: unknown inbound write payload");
      return;
    }
    InboundWriteState& state = iwIt->second.at(opId);
    if (ev.lane >= state.lanesTotal || (state.lanesDoneMask & uint16_t(1U << ev.lane))) {
      AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: duplicate or invalid write lane");
      return;
    }
    if (ev.discarded && !state.discard) {
      AbortPeer(peer, StatusCode::ERR_INVALID_ARGS, "TCP: invalid write payload length");
      return;
    }
    state.lanesDoneMask |= uint16_t(1U << ev.lane);
    MaybeFinalizeInboundWrite(peer, opId);
    return;
  }

  OpKey key{peer, opId};
  auto obIt = pendingOutbound_.find(key);
  if (obIt == pendingOutbound_.end()) return;
  OutboundOpState& st = *obIt->second;
  if (!st.isRead || ev.lane >= st.lanesTotal || (st.lanesDoneMask & uint16_t(1U << ev.lane)) ||
      ev.discarded) {
    AbortPeer(peer, StatusCode::ERR_INVALID_ARGS, "TCP: invalid read response lane");
    return;
  }
  st.lanesDoneMask |= uint16_t(1U << ev.lane);
  st.rxBytes += ev.laneLen;

  if (st.local.loc == MemoryLocationType::GPU) {
    uint16_t allMask = tcp::LanesAllMask(st.lanesTotal);
    if ((st.lanesDoneMask & allMask) != allMask) {
      MaybeCompleteOutbound(st);
      return;
    }
    if (st.gpuCopyPending) return;
    if (!st.pinned) {
      AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: missing staging (read)");
      return;
    }
    st.gpuCopyPending = true;
    auto pinnedRef = st.pinned;
    bool ok = ScheduleGpuCopy(st.local.deviceId, true, st.local, st.localSegs, pinnedRef, peer,
                              opId, [this, key, pinnedRef]() {
                                auto it2 = pendingOutbound_.find(key);
                                if (it2 == pendingOutbound_.end()) return;
                                it2->second->gpuCopyPending = false;
                                MaybeCompleteOutbound(*it2->second);
                              });
    if (!ok) {
      AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: HIP copy failed (read)");
    }
    return;
  }
  MaybeCompleteOutbound(st);
}

void TcpTransport::DrainSubmissions() {
  std::deque<std::unique_ptr<OutboundOpState>> ops;
  {
    std::lock_guard<std::mutex> lk(submitMu_);
    ops.swap(submitQ_);
  }
  for (auto& op : ops) {
    bool duplicate = pendingOutbound_.count(op->key) != 0;
    auto waiting = waitingOps_.find(op->key.peer);
    if (!duplicate && waiting != waitingOps_.end()) {
      duplicate = std::any_of(waiting->second.begin(), waiting->second.end(),
                              [&](const auto& old) { return old->key == op->key; });
    }
    if (duplicate) {
      op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: duplicate peer/op id");
      continue;
    }
    if (config_.opTimeoutMs > 0) {
      op->deadline = Clock::now() + std::chrono::milliseconds(config_.opTimeoutMs);
      deadlineDeque_.push_back({op->deadline, op->key});
    }
    const EngineKey peer = op->key.peer;
    EnsurePeerChannels(peer);
    if (IsPeerReady(peer))
      DispatchOp(std::move(op));
    else
      waitingOps_[peer].push_back(std::move(op));
  }
}

void TcpTransport::RetryWaitingConnections() {
  if (Clock::now() < nextAuxDeadline_) return;
  std::vector<EngineKey> peers;
  peers.reserve(waitingOps_.size());
  for (const auto& [peer, ops] : waitingOps_) peers.push_back(peer);
  for (const auto& peer : peers)
    if (!IsPeerReady(peer)) EnsurePeerChannels(peer);
}

void TcpTransport::NoteAuxDeadline(Clock::time_point deadline) {
  nextAuxDeadline_ = std::min(nextAuxDeadline_, deadline);
}

void TcpTransport::RecomputeAuxDeadline() {
  nextAuxDeadline_ = Clock::time_point::max();
  for (const auto& [peer, writes] : inboundWrites_)
    for (const auto& [id, state] : writes) NoteAuxDeadline(state.deadline);
  for (const auto& [peer, reads] : pendingInboundReads_)
    for (const auto& read : reads) NoteAuxDeadline(read.deadline);
  for (const auto& [peer, targets] : awaitingTargets_)
    for (const auto& [key, deadline] : targets) NoteAuxDeadline(deadline);
  for (const auto& [peer, links] : peers_) NoteAuxDeadline(links.connectNotBefore);
}

bool TcpTransport::IsLiveDeadline(const OpKey& key, Clock::time_point deadline) const {
  auto pending = pendingOutbound_.find(key);
  if (pending != pendingOutbound_.end()) return pending->second->deadline == deadline;
  auto waiting = waitingOps_.find(key.peer);
  if (waiting == waitingOps_.end()) return false;
  for (const auto& op : waiting->second)
    if (op->key == key && op->deadline == deadline) return true;
  return false;
}

int TcpTransport::ComputeEpollTimeoutMs() {
  while (!deadlineDeque_.empty() &&
         !IsLiveDeadline(deadlineDeque_.front().second, deadlineDeque_.front().first))
    deadlineDeque_.pop_front();
  int timeout = -1;
  if (!deadlineDeque_.empty()) {
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                         deadlineDeque_.front().first - Clock::now())
                         .count();
    timeout = remaining <= 0 ? 0 : static_cast<int>(std::min<int64_t>(remaining, INT_MAX));
  }
  for (Clock::time_point deadline : {nextAuxDeadline_, nextStatusPrune_}) {
    if (deadline == Clock::time_point::max()) continue;
    auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    int auxiliaryTimeout =
        remaining <= 0 ? 0 : static_cast<int>(std::min<int64_t>(remaining, INT_MAX));
    timeout = timeout < 0 ? auxiliaryTimeout : std::min(timeout, auxiliaryTimeout);
  }
  if (!gpuTasks_.empty())
    timeout = timeout < 0 ? kGpuPollIntervalMs : std::min(timeout, kGpuPollIntervalMs);
  return timeout;
}

void TcpTransport::PruneInboundStatuses() {
  const auto now = Clock::now();
  if (now < nextStatusPrune_) return;
  const auto maxAge = std::chrono::milliseconds(std::max(config_.opTimeoutMs * 4, 60000));
  std::lock_guard<std::mutex> lk(inboundMu_);
  for (auto peerIt = inboundStatus_.begin(); peerIt != inboundStatus_.end();) {
    auto& entries = peerIt->second;
    for (auto it = entries.begin(); it != entries.end();) {
      if (now - it->second.created > maxAge) {
        MORI_IO_WARN("TCP: evicting stale inbound status peer={} op={}", peerIt->first, it->first);
        it = entries.erase(it);
      } else {
        ++it;
      }
    }
    if (entries.size() > kMaxInboundStatusesPerPeer) {
      std::vector<std::pair<Clock::time_point, TransferUniqueId>> byAge;
      byAge.reserve(entries.size());
      for (const auto& [id, entry] : entries) byAge.push_back({entry.created, id});
      std::sort(byAge.begin(), byAge.end());
      size_t excess = entries.size() - kMaxInboundStatusesPerPeer;
      for (size_t i = 0; i < excess; ++i) {
        MORI_IO_WARN("TCP: evicting excess inbound status peer={} op={}", peerIt->first,
                     byAge[i].second);
        entries.erase(byAge[i].second);
      }
    }
    if (entries.empty())
      peerIt = inboundStatus_.erase(peerIt);
    else
      ++peerIt;
  }
  nextStatusPrune_ =
      inboundStatus_.empty() ? Clock::time_point::max() : now + std::chrono::seconds(1);
}

void TcpTransport::ScanTimeouts() {
  const auto now = Clock::now();
  while (!deadlineDeque_.empty() && deadlineDeque_.front().first <= now) {
    auto [deadline, key] = deadlineDeque_.front();
    deadlineDeque_.pop_front();
    if (!IsLiveDeadline(key, deadline)) continue;
    if (pendingOutbound_.count(key)) {
      AbortPeer(key.peer, StatusCode::ERR_BAD_STATE, "TCP: operation timeout");
      continue;
    }
    auto waiting = waitingOps_.find(key.peer);
    if (waiting == waitingOps_.end()) continue;
    if (!IsPeerReady(key.peer)) {
      AbortPeer(key.peer, StatusCode::ERR_BAD_STATE, "TCP: connection timeout");
      continue;
    }
    for (auto it = waiting->second.begin(); it != waiting->second.end(); ++it) {
      if ((*it)->key != key) continue;
      TransferStatus* status = (*it)->status;
      waiting->second.erase(it);
      status->Update(StatusCode::ERR_BAD_STATE, "TCP: no DATA lane available");
      break;
    }
    if (waiting->second.empty()) waitingOps_.erase(waiting);
  }

  std::vector<EngineKey> abortPeers;
  const bool scanAuxiliary = now >= nextAuxDeadline_;
  if (scanAuxiliary) {
    for (const auto& [peer, writes] : inboundWrites_) {
      bool expired = false;
      for (const auto& [id, state] : writes)
        expired = expired || (state.deadline != Clock::time_point::max() && state.deadline <= now);
      if (expired) abortPeers.push_back(peer);
    }
    for (const auto& [peer, reads] : pendingInboundReads_) {
      bool expired = false;
      for (const auto& read : reads)
        expired = expired || (read.deadline != Clock::time_point::max() && read.deadline <= now);
      if (expired && std::find(abortPeers.begin(), abortPeers.end(), peer) == abortPeers.end())
        abortPeers.push_back(peer);
    }
    for (const auto& [peer, targets] : awaitingTargets_) {
      bool expired = false;
      for (const auto& [key, deadline] : targets)
        expired = expired || (deadline != Clock::time_point::max() && deadline <= now);
      if (expired && std::find(abortPeers.begin(), abortPeers.end(), peer) == abortPeers.end())
        abortPeers.push_back(peer);
    }
  }
  for (const auto& peer : abortPeers)
    AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: inbound wire timeout or quota exceeded");
  if (scanAuxiliary) RecomputeAuxDeadline();
  PruneInboundStatuses();
}

void TcpTransport::ShutdownDrain() {
  std::unordered_set<EngineKey> peerKeys;
  for (const auto& [peer, links] : peers_) peerKeys.insert(peer);
  for (const auto& [peer, ops] : waitingOps_) peerKeys.insert(peer);
  for (const auto& [key, op] : pendingOutbound_) peerKeys.insert(key.peer);
  for (const auto& peer : peerKeys)
    AbortPeer(peer, StatusCode::ERR_BAD_STATE, "TCP: transport shutdown");

  std::deque<std::unique_ptr<OutboundOpState>> submitted;
  {
    std::lock_guard<std::mutex> lk(submitMu_);
    submitted.swap(submitQ_);
  }
  for (auto& op : submitted)
    op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: transport shutdown");

  for (auto& [fd, worker] : dataWorkers_) shutdown(fd, SHUT_RDWR);
  for (auto& [fd, worker] : dataWorkers_) {
    worker->Stop();
    close(fd);
  }
  dataWorkers_.clear();
  workerNotifyMap_.clear();
  for (auto& [fd, conn] : conns_) CloseConnInternal(conn.get());
  conns_.clear();
  peers_.clear();

  for (auto& task : gpuTasks_) {
    hipError_t setStatus = hipSetDevice(task.deviceId);
    if (setStatus != hipSuccess)
      MORI_IO_WARN("TCP: hipSetDevice({}) during shutdown failed: {}", task.deviceId,
                   hipGetErrorString(setStatus));
    hipError_t status = hipEventSynchronize(task.ev);
    if (status != hipSuccess)
      MORI_IO_WARN("TCP: hipEventSynchronize during shutdown failed: {}",
                   hipGetErrorString(status));
    eventPool_.PutEvent(task.ev, task.deviceId);
  }
  gpuTasks_.clear();
}

// ---------------------------------------------------------------------------
// Main I/O loop
// ---------------------------------------------------------------------------
void TcpTransport::IoLoop() {
  constexpr int kMaxEvents = 128;
  epoll_event events[kMaxEvents];

  while (running_.load()) {
    DrainSubmissions();
    RetryWaitingConnections();
    PollGpuTasks();
    ProcessWorkerEvents();
    ScanTimeouts();

    int nfds = epoll_wait(epfd_, events, kMaxEvents, ComputeEpollTimeoutMs());
    if (nfds < 0) {
      if (errno == EINTR) continue;
      MORI_IO_ERROR("TCP: epoll_wait: {}", strerror(errno));
      break;
    }

    std::unordered_set<int> closedThisBatch;
    closedThisBatch_ = &closedThisBatch;
    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;
      uint32_t ev = events[i].events;
      if (closedThisBatch.count(fd)) continue;

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
        if (HandleCtrlReadable(c) == ConnResult::Gone) continue;
        cit = conns_.find(fd);
        if (cit == conns_.end()) continue;
        c = cit->second.get();
        if (!c) continue;
        if (c->handedOff) continue;
      }
      if (ev & EPOLLOUT) {
        cit = conns_.find(fd);
        if (cit != conns_.end()) HandleConnWritable(cit->second.get());
      }
    }
    closedThisBatch_ = nullptr;
  }
  closedThisBatch_ = nullptr;
  ShutdownDrain();
}

}  // namespace io
}  // namespace mori

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
#include "src/io/tcp/backend_impl.hpp"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/logging.hpp"
#include "src/io/tcp/protocol.hpp"
#include "src/io/xgmi/hip_resource_pool.hpp"

namespace mori {
namespace io {

namespace {

using Clock = std::chrono::steady_clock;

inline bool IsWouldBlock(int err) { return (err == EAGAIN) || (err == EWOULDBLOCK); }

inline int SetNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return -1;
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return -1;
  return 0;
}

inline void SetSockOptOrLog(int fd, int level, int optname, const void* optval, socklen_t optlen,
                            const char* name) {
  if (setsockopt(fd, level, optname, optval, optlen) != 0) {
    MORI_IO_WARN("TCP: setsockopt {} failed: {}", name, strerror(errno));
  }
}

inline void ConfigureSocketCommon(int fd, const TcpBackendConfig& cfg) {
  if (cfg.enableKeepalive) {
    int on = 1;
    SetSockOptOrLog(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on), "SO_KEEPALIVE");

    SetSockOptOrLog(fd, IPPROTO_TCP, TCP_KEEPIDLE, &cfg.keepaliveIdleSec,
                    sizeof(cfg.keepaliveIdleSec), "TCP_KEEPIDLE");
    SetSockOptOrLog(fd, IPPROTO_TCP, TCP_KEEPINTVL, &cfg.keepaliveIntvlSec,
                    sizeof(cfg.keepaliveIntvlSec), "TCP_KEEPINTVL");
    SetSockOptOrLog(fd, IPPROTO_TCP, TCP_KEEPCNT, &cfg.keepaliveCnt, sizeof(cfg.keepaliveCnt),
                    "TCP_KEEPCNT");
  }
}

inline void ConfigureCtrlSocket(int fd, const TcpBackendConfig& cfg) {
  ConfigureSocketCommon(fd, cfg);
  if (cfg.enableCtrlNodelay) {
    int on = 1;
    SetSockOptOrLog(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on), "TCP_NODELAY(ctrl)");
  }
}

inline void ConfigureDataSocket(int fd, const TcpBackendConfig& cfg) {
  ConfigureSocketCommon(fd, cfg);
  if (cfg.enableDataNodelay) {
    int on = 1;
    SetSockOptOrLog(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on), "TCP_NODELAY(data)");
  }
  if (cfg.sockSndbufBytes > 0) {
    SetSockOptOrLog(fd, SOL_SOCKET, SO_SNDBUF, &cfg.sockSndbufBytes, sizeof(cfg.sockSndbufBytes),
                    "SO_SNDBUF");
  }
  if (cfg.sockRcvbufBytes > 0) {
    SetSockOptOrLog(fd, SOL_SOCKET, SO_RCVBUF, &cfg.sockRcvbufBytes, sizeof(cfg.sockRcvbufBytes),
                    "SO_RCVBUF");
  }
}

inline std::optional<sockaddr_in> ParseIpv4(const std::string& host, uint16_t port) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    return std::nullopt;
  }
  return addr;
}

inline uint16_t GetBoundPort(int fd) {
  sockaddr_in addr{};
  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    return 0;
  }
  return ntohs(addr.sin_port);
}

inline size_t RoundUpPow2(size_t v) {
  if (v <= 1) return 1;
  size_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

struct PinnedBuf {
  void* ptr{nullptr};
  size_t cap{0};
};

class PinnedStagingPool {
 public:
  PinnedStagingPool() = default;
  ~PinnedStagingPool() { Clear(); }

  PinnedStagingPool(const PinnedStagingPool&) = delete;
  PinnedStagingPool& operator=(const PinnedStagingPool&) = delete;

  std::shared_ptr<PinnedBuf> Acquire(size_t size) {
    const size_t cap = RoundUpPow2(size);
    {
      std::lock_guard<std::mutex> lock(mu);
      auto it = free.find(cap);
      if (it != free.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        return std::shared_ptr<PinnedBuf>(new PinnedBuf{p, cap}, [this](PinnedBuf* b) {
          this->Release(b);
        });
      }
    }

    void* p = nullptr;
    hipError_t err = hipHostMalloc(&p, cap, hipHostMallocDefault);
    if (err != hipSuccess) {
      MORI_IO_ERROR("TCP: hipHostMalloc({}) failed: {}", cap, hipGetErrorString(err));
      return nullptr;
    }
    return std::shared_ptr<PinnedBuf>(new PinnedBuf{p, cap}, [this](PinnedBuf* b) { this->Release(b); });
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mu);
    for (auto& kv : free) {
      for (void* p : kv.second) {
        hipError_t err = hipHostFree(p);
        if (err != hipSuccess) {
          MORI_IO_WARN("TCP: hipHostFree failed: {}", hipGetErrorString(err));
        }
      }
      kv.second.clear();
    }
    free.clear();
  }

 private:
  void Release(PinnedBuf* b) {
    if (b == nullptr) return;
    const size_t cap = b->cap;
    void* p = b->ptr;
    delete b;

    constexpr size_t kMaxCachedPerClass = 8;
    std::lock_guard<std::mutex> lock(mu);
    auto& vec = free[cap];
    if (vec.size() < kMaxCachedPerClass) {
      vec.push_back(p);
    } else {
      hipError_t err = hipHostFree(p);
      if (err != hipSuccess) {
        MORI_IO_WARN("TCP: hipHostFree failed: {}", hipGetErrorString(err));
      }
    }
  }

 private:
  std::mutex mu;
  std::unordered_map<size_t, std::vector<void*>> free;
};

struct Segment {
  uint64_t off{0};
  uint64_t len{0};
};

inline bool IsSingleContiguousSpan(const std::vector<Segment>& segs, uint64_t* outOff,
                                   uint64_t* outLen) {
  if (!outOff || !outLen) return false;
  if (segs.empty()) return false;
  uint64_t off = segs[0].off;
  uint64_t end = off + segs[0].len;
  for (size_t i = 1; i < segs.size(); ++i) {
    if (segs[i].off != end) return false;
    end += segs[i].len;
  }
  *outOff = off;
  *outLen = end - off;
  return true;
}

inline uint64_t SumLens(const std::vector<Segment>& segs) {
  uint64_t total = 0;
  for (const auto& s : segs) total += s.len;
  return total;
}

struct InboundStatusEntry {
  StatusCode code{StatusCode::INIT};
  std::string msg;
};

struct OutboundOpState {
  EngineKey peer;
  TransferUniqueId id{0};
  bool isRead{false};
  TransferStatus* status{nullptr};

  MemoryDesc local{};
  std::vector<Segment> localSegs;

  MemoryDesc remote{};
  std::vector<Segment> remoteSegs;

  uint64_t expectedRxBytes{0};
  uint64_t rxBytes{0};
  bool completionReceived{false};
  StatusCode completionCode{StatusCode::INIT};
  std::string completionMsg;

  bool gpuCopyPending{false};

  Clock::time_point startTs{Clock::now()};
};

struct InboundWriteState {
  EngineKey peer;
  TransferUniqueId id{0};

  MemoryDesc dst{};
  std::vector<Segment> dstSegs;

  bool discard{false};
};

struct EarlyWriteState {
  uint64_t payloadLen{0};
  std::shared_ptr<PinnedBuf> pinned;
  bool complete{false};
};

struct ActiveDataRx {
  bool active{false};
  TransferUniqueId id{0};
  uint64_t remaining{0};

  bool discard{false};
  // CPU scatter target:
  void* base{nullptr};
  std::vector<Segment> segs;
  size_t segIdx{0};
  uint64_t segOff{0};

  // GPU staging target:
  bool toGpu{false};
  int gpuDevice{-1};
  void* gpuBase{nullptr};
  std::shared_ptr<PinnedBuf> pinned;
  uint64_t pinnedWriteOff{0};
};

struct SendItem {
  std::vector<uint8_t> header;
  std::vector<iovec> iov;
  size_t idx{0};
  size_t off{0};
  int flags{0};
  std::shared_ptr<void> keepalive;
  std::function<void()> onDone;

  bool Done() const { return idx >= iov.size(); }

  void Advance(size_t n) {
    while (n > 0 && idx < iov.size()) {
      size_t avail = iov[idx].iov_len - off;
      if (n < avail) {
        off += n;
        n = 0;
        break;
      }
      n -= avail;
      idx++;
      off = 0;
    }
  }
};

struct Connection {
  int fd{-1};
  bool isOutgoing{false};
  bool connecting{false};
  bool helloSent{false};
  bool helloReceived{false};
  tcp::Channel ch{tcp::Channel::CTRL};
  EngineKey peerKey{};

  // Control parsing buffer (also used for HELLO on both channels).
  std::vector<uint8_t> inbuf;

  // Data RX state (after HELLO on data channel).
  std::array<uint8_t, tcp::kDataHeaderSize> dataHdrBuf{};
  size_t dataHdrGot{0};
  tcp::DataHeaderView curDataHdr{};
  ActiveDataRx rx{};

  std::deque<SendItem> sendq;
};

struct PeerLinks {
  int ctrlFd{-1};
  int dataFd{-1};
  bool CtrlUp() const { return ctrlFd >= 0; }
  bool DataUp() const { return dataFd >= 0; }
};

}  // namespace

class TcpTransport {
 public:
  TcpTransport(EngineKey myKey, const IOEngineConfig& engCfg, const TcpBackendConfig& cfg)
      : myEngKey(std::move(myKey)), engConfig(engCfg), config(cfg) {}

  ~TcpTransport() { Shutdown(); }

  TcpTransport(const TcpTransport&) = delete;
  TcpTransport& operator=(const TcpTransport&) = delete;

  void Start() {
    if (running.load()) return;

    if (config.numIoThreads != 1) {
      MORI_IO_WARN("TCP: numIoThreads={} requested but only 1 is supported; forcing 1",
                   config.numIoThreads);
    }

    epfd = epoll_create1(EPOLL_CLOEXEC);
    assert(epfd >= 0);

    // listener
    listenFd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    assert(listenFd >= 0);

    int one = 1;
    SetSockOptOrLog(listenFd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one), "SO_REUSEADDR");

    auto addrOpt = ParseIpv4(engConfig.host.empty() ? std::string("0.0.0.0") : engConfig.host,
                             engConfig.port);
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

    // eventfd for submit queue
    wakeFd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    assert(wakeFd >= 0);

    AddEpoll(listenFd, true, false);
    AddEpoll(wakeFd, true, false);

    running.store(true);
    ioThread = std::thread([this] { this->IoLoop(); });
  }

  void Shutdown() {
    bool wasRunning = running.exchange(false);
    if (!wasRunning) return;

    // Wake epoll.
    if (wakeFd >= 0) {
      uint64_t one = 1;
      ::write(wakeFd, &one, sizeof(one));
    }

    if (ioThread.joinable()) ioThread.join();

    // Close all connections.
    for (auto& kv : conns) {
      CloseConnInternal(kv.second.get());
    }
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

  std::optional<uint16_t> GetListenPort() const {
    if (listenPort == 0) return std::nullopt;
    return listenPort;
  }

  void RegisterRemoteEngine(const EngineDesc& desc) {
    std::lock_guard<std::mutex> lock(remoteMu);
    remoteEngines[desc.key] = desc;
  }

  void DeregisterRemoteEngine(const EngineDesc& desc) {
    std::lock_guard<std::mutex> lock(remoteMu);
    remoteEngines.erase(desc.key);
  }

  void RegisterMemory(const MemoryDesc& desc) {
    std::lock_guard<std::mutex> lock(memMu);
    localMems[desc.id] = desc;
  }

  void DeregisterMemory(const MemoryDesc& desc) {
    std::lock_guard<std::mutex> lock(memMu);
    localMems.erase(desc.id);
  }

  bool PopInboundTransferStatus(const EngineKey& remote, TransferUniqueId id, TransferStatus* status) {
    std::lock_guard<std::mutex> lock(inboundMu);
    auto it = inboundStatus.find(remote);
    if (it == inboundStatus.end()) return false;
    auto it2 = it->second.find(id);
    if (it2 == it->second.end()) return false;
    status->Update(it2->second.code, it2->second.msg);
    it->second.erase(it2);
    return true;
  }

  void SubmitReadWrite(const MemoryDesc& local, size_t localOffset, const MemoryDesc& remote,
                       size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                       bool isRead) {
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

  void SubmitBatchReadWrite(const MemoryDesc& local, const SizeVec& localOffsets,
                            const MemoryDesc& remote, const SizeVec& remoteOffsets,
                            const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                            bool isRead) {
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

    std::vector<Segment> localSegs;
    std::vector<Segment> remoteSegs;
    localSegs.reserve(n);
    remoteSegs.reserve(n);

    uint64_t total = 0;
    for (size_t i = 0; i < n; ++i) {
      const size_t lo = localOffsets[i];
      const size_t ro = remoteOffsets[i];
      const size_t sz = sizes[i];
      if (sz == 0) continue;
      if ((lo + sz) > local.size || (ro + sz) > remote.size) {
        status->Update(StatusCode::ERR_INVALID_ARGS, "TCP: batch offset+size out of range");
        return;
      }
      localSegs.push_back({static_cast<uint64_t>(lo), static_cast<uint64_t>(sz)});
      remoteSegs.push_back({static_cast<uint64_t>(ro), static_cast<uint64_t>(sz)});
      total += static_cast<uint64_t>(sz);
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

 private:
  void EnqueueOp(std::unique_ptr<OutboundOpState> op) {
    {
      std::lock_guard<std::mutex> lock(submitMu);
      submitQ.push_back(std::move(op));
    }
    uint64_t one = 1;
    ::write(wakeFd, &one, sizeof(one));
  }

  void AddEpoll(int fd, bool wantRead, bool wantWrite) {
    epoll_event ev{};
    ev.data.fd = fd;
    ev.events = EPOLLET | (wantRead ? EPOLLIN : 0) | (wantWrite ? EPOLLOUT : 0);
    SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev));
  }

  void ModEpoll(int fd, bool wantRead, bool wantWrite) {
    epoll_event ev{};
    ev.data.fd = fd;
    ev.events = EPOLLET | (wantRead ? EPOLLIN : 0) | (wantWrite ? EPOLLOUT : 0);
    SYSCALL_RETURN_ZERO(epoll_ctl(epfd, EPOLL_CTL_MOD, fd, &ev));
  }

  void DelEpoll(int fd) { epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr); }

  void CloseConnInternal(Connection* c) {
    if (c == nullptr) return;
    if (c->fd >= 0) {
      DelEpoll(c->fd);
      shutdown(c->fd, SHUT_RDWR);
      close(c->fd);
      c->fd = -1;
    }
  }

  bool PreferOutgoingFor(const EngineKey& peerKey) const { return myEngKey < peerKey; }

  void AssignConnToPeer(Connection* c) {
    assert(c && c->helloReceived);
    PeerLinks& link = peers[c->peerKey];
    const bool preferOutgoing = PreferOutgoingFor(c->peerKey);

    auto replace_if_needed = [&](int& slotFd) {
      if (slotFd < 0) {
        slotFd = c->fd;
        return;
      }
      // Collision: choose deterministically by key + direction so both sides keep the same TCP conn.
      const int existingFd = slotFd;
      const int newFd = c->fd;
      Connection* existing = conns[existingFd].get();
      if (!existing) {
        slotFd = newFd;
        return;
      }
      const bool keepNew = (preferOutgoing && c->isOutgoing) || (!preferOutgoing && !c->isOutgoing);
      if (keepNew) {
        MORI_IO_WARN("TCP: peer {} channel {} replacing existing fd {} with fd {}", c->peerKey,
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

    if (c->ch == tcp::Channel::CTRL) {
      replace_if_needed(link.ctrlFd);
    } else {
      replace_if_needed(link.dataFd);
    }
  }

  void MaybeDispatchQueuedOps(const EngineKey& peerKey) {
    auto it = peers.find(peerKey);
    if (it == peers.end()) return;
    if (!it->second.CtrlUp() || !it->second.DataUp()) return;
    Connection* ctrl = conns[it->second.ctrlFd].get();
    Connection* data = conns[it->second.dataFd].get();
    if (!ctrl || !data) return;
    if (!ctrl->helloReceived || !data->helloReceived) return;

    auto qit = waitingOps.find(peerKey);
    if (qit == waitingOps.end()) return;

    auto ops = std::move(qit->second);
    waitingOps.erase(qit);
    MORI_IO_TRACE("TCP: peer {} ready, dispatch {} queued ops", peerKey.c_str(), ops.size());
    for (auto& op : ops) {
      DispatchOp(std::move(op));
    }
  }

  void EnsurePeerChannels(const EngineKey& peerKey) {
    PeerLinks& link = peers[peerKey];
    if (!link.CtrlUp()) ConnectChannel(peerKey, tcp::Channel::CTRL);
    if (!link.DataUp()) ConnectChannel(peerKey, tcp::Channel::DATA);
  }

  void ConnectChannel(const EngineKey& peerKey, tcp::Channel ch) {
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

    // Bind to our selected host to pin NIC choice when multiple interfaces exist.
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

    // socket opts
    if (ch == tcp::Channel::CTRL) {
      ConfigureCtrlSocket(fd, config);
    } else {
      ConfigureDataSocket(fd, config);
    }

    const bool wantWrite = connecting || !conn->sendq.empty();
    AddEpoll(fd, true, wantWrite);
    conns[fd] = std::move(conn);

    if (!connecting) {
      QueueHello(fd);
      ModEpoll(fd, true, true);
    }
  }

  void QueueHello(int fd) {
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

  void AcceptNew() {
    while (true) {
      sockaddr_in peer{};
      socklen_t len = sizeof(peer);
      int fd = accept4(listenFd, reinterpret_cast<sockaddr*>(&peer), &len, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0) {
        if (IsWouldBlock(errno)) break;
        MORI_IO_WARN("TCP: accept failed: {}", strerror(errno));
        break;
      }
      MORI_IO_TRACE("TCP: accept fd={}", fd);

      auto conn = std::make_unique<Connection>();
      conn->fd = fd;
      conn->isOutgoing = false;
      conn->connecting = false;
      conn->helloSent = false;
      conn->helloReceived = false;
      conn->inbuf.reserve(4096);

      AddEpoll(fd, true, false);
      conns[fd] = std::move(conn);
    }
  }

  void DrainWakeFd() {
    uint64_t v = 0;
    while (true) {
      ssize_t n = ::read(wakeFd, &v, sizeof(v));
      if (n < 0) {
        if (IsWouldBlock(errno)) break;
        break;
      }
      if (n == 0) break;
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

  bool IsPeerReady(const EngineKey& peerKey) {
    auto it = peers.find(peerKey);
    if (it == peers.end()) return false;
    if (!it->second.CtrlUp() || !it->second.DataUp()) return false;
    Connection* ctrl = conns[it->second.ctrlFd].get();
    Connection* data = conns[it->second.dataFd].get();
    return ctrl && data && ctrl->helloReceived && data->helloReceived;
  }

  void DispatchOp(std::unique_ptr<OutboundOpState> op) {
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
    Connection* data = conns[it->second.dataFd].get();
    if (!ctrl || !data) {
      op->status->Update(StatusCode::ERR_BAD_STATE, "TCP: peer connection missing");
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
      MORI_IO_ERROR("TCP: failed to store outbound op id={} (nullptr)", opId);
      pendingOutbound.erase(itIns);
      return;
    }
    MORI_IO_TRACE("TCP: dispatch op id={} peer={} isRead={} segs={}", st->id, peerKey.c_str(),
                  st->isRead, st->localSegs.size());

    // CTRL request
    std::vector<uint8_t> ctrlFrame;
    if (st->localSegs.size() == 1) {
      if (st->isRead) {
        ctrlFrame =
            tcp::BuildReadReq(st->id, st->remote.id, st->remoteSegs[0].off, st->remoteSegs[0].len);
      } else {
        ctrlFrame =
            tcp::BuildWriteReq(st->id, st->remote.id, st->remoteSegs[0].off, st->remoteSegs[0].len);
      }
    } else {
      std::vector<uint64_t> roffs;
      std::vector<uint64_t> szs;
      roffs.reserve(st->remoteSegs.size());
      szs.reserve(st->remoteSegs.size());
      for (const auto& s : st->remoteSegs) {
        roffs.push_back(s.off);
        szs.push_back(s.len);
      }
      if (st->isRead) {
        ctrlFrame = tcp::BuildBatchReadReq(st->id, st->remote.id, roffs, szs);
      } else {
        ctrlFrame = tcp::BuildBatchWriteReq(st->id, st->remote.id, roffs, szs);
      }
    }

    QueueSend(ctrl->fd, std::move(ctrlFrame));

    // DATA payload (writes only)
    if (!st->isRead) {
      QueueDataSendForWrite(data, *st);
    }

    UpdateWriteInterest(ctrl->fd);
    UpdateWriteInterest(data->fd);
  }

  void QueueSend(int fd, std::vector<uint8_t> bytes, std::function<void()> onDone = nullptr) {
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

  void QueueDataSendForWrite(Connection* data, OutboundOpState& st) {
    const uint64_t total = SumLens(st.localSegs);
    auto hdr = tcp::BuildDataHeader(st.id, total, 0);

    if (st.local.loc == MemoryLocationType::GPU) {
      QueueGpuToNetSend(data, st.local, st.localSegs, std::move(hdr), /*onDone*/ nullptr);
      return;
    }

    SendItem item;
    item.header = std::move(hdr);
    item.iov.reserve(1 + st.localSegs.size());
    item.iov.push_back({item.header.data(), item.header.size()});

    uint8_t* base = reinterpret_cast<uint8_t*>(st.local.data);
    for (const auto& s : st.localSegs) {
      item.iov.push_back({base + s.off, static_cast<size_t>(s.len)});
    }
    data->sendq.push_back(std::move(item));
  }

  void QueueGpuToNetSend(Connection* data, const MemoryDesc& src, const std::vector<Segment>& srcSegs,
                         std::vector<uint8_t> dataHdr, std::function<void()> onDone) {
    const uint64_t total = SumLens(srcSegs);
    auto pinned = staging.Acquire(static_cast<size_t>(total));
    if (!pinned) {
      MORI_IO_ERROR("TCP: failed to allocate pinned staging for GPU send");
      return;
    }

    hipStream_t stream = streamPool.GetNextStream(src.deviceId);
    hipEvent_t ev = eventPool.GetEvent(src.deviceId);
    if (stream == nullptr || ev == nullptr) {
      MORI_IO_ERROR("TCP: failed to get HIP stream/event for GPU send");
      if (ev) eventPool.PutEvent(ev, src.deviceId);
      return;
    }

    HIP_RUNTIME_CHECK(hipSetDevice(src.deviceId));
    uint8_t* dst = reinterpret_cast<uint8_t*>(pinned->ptr);
    uint64_t spanOff = 0;
    uint64_t spanLen = 0;
    if (IsSingleContiguousSpan(srcSegs, &spanOff, &spanLen) && spanLen == total) {
      hipDeviceptr_t gpuPtr = reinterpret_cast<hipDeviceptr_t>(src.data + spanOff);
      HIP_RUNTIME_CHECK(hipMemcpyDtoHAsync(dst, gpuPtr, static_cast<size_t>(total), stream));
    } else {
      uint64_t off = 0;
      for (const auto& s : srcSegs) {
        hipDeviceptr_t gpuPtr = reinterpret_cast<hipDeviceptr_t>(src.data + s.off);
        HIP_RUNTIME_CHECK(
            hipMemcpyDtoHAsync(dst + off, gpuPtr, static_cast<size_t>(s.len), stream));
        off += s.len;
      }
    }
    HIP_RUNTIME_CHECK(hipEventRecord(ev, stream));

    gpuTasks.push_back({src.deviceId, ev, [this, dataFd = data->fd, pinned, hdr = std::move(dataHdr),
                                          total, onDone = std::move(onDone)]() mutable {
                          Connection* c = conns[dataFd].get();
                          if (!c || c->fd < 0) return;
                          SendItem item;
                          item.header = std::move(hdr);
                          item.iov.reserve(2);
                          item.iov.push_back({item.header.data(), item.header.size()});
                          item.iov.push_back({pinned->ptr, static_cast<size_t>(total)});
                          item.keepalive = pinned;
                          item.onDone = std::move(onDone);
                          c->sendq.push_back(std::move(item));
                          UpdateWriteInterest(dataFd);
                        }});
  }

  struct GpuTask {
    int deviceId{-1};
    hipEvent_t ev{nullptr};
    std::function<void()> onReady;
  };

  void PollGpuTasks() {
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

  void UpdateWriteInterest(int fd) {
    auto it = conns.find(fd);
    if (it == conns.end()) return;
    Connection* c = it->second.get();
    if (!c || c->fd < 0) return;

    // With edge-triggered epoll, don't rely solely on EPOLLOUT transitions to make progress.
    // If the socket is already writable, we may not get another EPOLLOUT edge after enqueueing.
    if (!c->connecting && !c->sendq.empty()) {
      FlushSend(c);
      it = conns.find(fd);
      if (it == conns.end()) return;
      c = it->second.get();
      if (!c || c->fd < 0) return;
    }

    bool wantWrite = c->connecting || !c->sendq.empty();
    ModEpoll(fd, true, wantWrite);
  }

  void HandleConnWritable(Connection* c) {
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

  void FlushSend(Connection* c) {
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

  void ClosePeerByFd(int fd) {
    MORI_IO_TRACE("TCP: close fd={}", fd);
    // Find which peer+channel this fd belongs to, close both channels and fail pending ops.
    EngineKey peer{};
    for (auto& kv : peers) {
      if (kv.second.ctrlFd == fd || kv.second.dataFd == fd) {
        peer = kv.first;
        break;
      }
    }

    Connection* c = conns[fd].get();
    if (c) {
      CloseConnInternal(c);
      conns.erase(fd);
    }

    if (!peer.empty()) {
      PeerLinks& link = peers[peer];
      if (link.ctrlFd >= 0) {
        Connection* cc = conns[link.ctrlFd].get();
        if (cc) {
          CloseConnInternal(cc);
          conns.erase(link.ctrlFd);
        }
        link.ctrlFd = -1;
      }
      if (link.dataFd >= 0) {
        Connection* dc = conns[link.dataFd].get();
        if (dc) {
          CloseConnInternal(dc);
          conns.erase(link.dataFd);
        }
        link.dataFd = -1;
      }
      peers.erase(peer);

      FailPendingOpsForPeer(peer, "TCP: connection lost");
    }
  }

  void FailPendingOpsForPeer(const EngineKey& peer, const std::string& msg) {
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

  void HandleCtrlReadable(Connection* c) {
    while (true) {
      uint8_t tmp[4096];
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

    // Parse frames.
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

      // Data channel transitions to a different framing after HELLO. Any bytes already read into
      // inbuf may include the beginning of the data stream; handle them immediately to avoid
      // edge-triggered epoll stalls.
      if (c->helloReceived && c->ch == tcp::Channel::DATA) {
        HandleDataReadable(c);
        return;
      }
    }
  }

  void HandleCtrlFrame(Connection* c, tcp::CtrlMsgType type, const uint8_t* body, size_t len) {
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

  void HandleHello(Connection* c, const uint8_t* body, size_t len) {
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
    c->ch = (chRaw == static_cast<uint8_t>(tcp::Channel::DATA)) ? tcp::Channel::DATA
                                                                : tcp::Channel::CTRL;
    c->helloReceived = true;
    MORI_IO_TRACE("TCP: recv HELLO fd={} peer={} ch={} outgoing={}", c->fd, c->peerKey.c_str(),
                  static_cast<int>(c->ch), c->isOutgoing);

    // Respond with our hello if we haven't sent it yet.
    if (!c->helloSent) {
      QueueHello(c->fd);
      UpdateWriteInterest(c->fd);
    }

    // Post-handshake: configure socket by channel type.
    if (c->ch == tcp::Channel::CTRL) {
      ConfigureCtrlSocket(c->fd, config);
    } else {
      ConfigureDataSocket(c->fd, config);
    }

    AssignConnToPeer(c);
    MaybeDispatchQueuedOps(peerKey);
  }

  std::optional<MemoryDesc> LookupLocalMem(MemoryUniqueId id) {
    std::lock_guard<std::mutex> lock(memMu);
    auto it = localMems.find(id);
    if (it == localMems.end()) return std::nullopt;
    return it->second;
  }

  void RecordInboundStatus(const EngineKey& peer, TransferUniqueId id, StatusCode code,
                           const std::string& msg) {
    std::lock_guard<std::mutex> lock(inboundMu);
    inboundStatus[peer][id] = InboundStatusEntry{code, msg};
  }

  Connection* PeerCtrl(const EngineKey& peer) {
    auto it = peers.find(peer);
    if (it == peers.end()) return nullptr;
    if (!it->second.CtrlUp()) return nullptr;
    return conns[it->second.ctrlFd].get();
  }

  Connection* PeerData(const EngineKey& peer) {
    auto it = peers.find(peer);
    if (it == peers.end()) return nullptr;
    if (!it->second.DataUp()) return nullptr;
    return conns[it->second.dataFd].get();
  }

  void FinalizeBufferedWrite(const EngineKey& peer, TransferUniqueId opId, InboundWriteState ws,
                             uint64_t payloadLen, const std::shared_ptr<PinnedBuf>& pinned) {
    auto ctrl = PeerCtrl(peer);
    if (!ctrl) return;

    if (ws.discard) {
      QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_INVALID_ARGS),
                                               "TCP: write discarded"));
      UpdateWriteInterest(ctrl->fd);
      RecordInboundStatus(peer, opId, StatusCode::ERR_INVALID_ARGS, "TCP: write discarded");
      return;
    }
    if (!pinned) {
      QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_BAD_STATE),
                                               "TCP: missing pinned staging (early write)"));
      UpdateWriteInterest(ctrl->fd);
      RecordInboundStatus(peer, opId, StatusCode::ERR_BAD_STATE, "TCP: missing pinned staging");
      return;
    }

    const uint64_t expected = SumLens(ws.dstSegs);
    if (expected != payloadLen) {
      QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_BAD_STATE),
                                               "TCP: early payload length mismatch"));
      UpdateWriteInterest(ctrl->fd);
      RecordInboundStatus(peer, opId, StatusCode::ERR_BAD_STATE, "TCP: early payload length mismatch");
      return;
    }

    if (ws.dst.loc == MemoryLocationType::GPU) {
      hipStream_t stream = streamPool.GetNextStream(ws.dst.deviceId);
      hipEvent_t ev = eventPool.GetEvent(ws.dst.deviceId);
      if (stream == nullptr || ev == nullptr) {
        QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_BAD_STATE),
                                                 "TCP: failed to get HIP stream/event"));
        UpdateWriteInterest(ctrl->fd);
        RecordInboundStatus(peer, opId, StatusCode::ERR_BAD_STATE, "TCP: failed to get HIP stream/event");
        if (ev) eventPool.PutEvent(ev, ws.dst.deviceId);
        return;
      }

      HIP_RUNTIME_CHECK(hipSetDevice(ws.dst.deviceId));
      uint8_t* src = reinterpret_cast<uint8_t*>(pinned->ptr);
      const uint64_t total = payloadLen;
      uint64_t spanOff = 0;
      uint64_t spanLen = 0;
      if (IsSingleContiguousSpan(ws.dstSegs, &spanOff, &spanLen) && spanLen == total) {
        void* gpuPtr = reinterpret_cast<void*>(ws.dst.data + spanOff);
        HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpuPtr, src, static_cast<size_t>(total), stream));
      } else {
        uint64_t off = 0;
        for (const auto& s : ws.dstSegs) {
          void* gpuPtr = reinterpret_cast<void*>(ws.dst.data + s.off);
          HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpuPtr, src + off, static_cast<size_t>(s.len), stream));
          off += s.len;
        }
      }
      HIP_RUNTIME_CHECK(hipEventRecord(ev, stream));

      gpuTasks.push_back({ws.dst.deviceId, ev, [this, peer, opId, ctrlFd = ctrl->fd, pinned]() {
                            QueueSend(ctrlFd,
                                      tcp::BuildCompletion(opId,
                                                           static_cast<uint32_t>(StatusCode::SUCCESS),
                                                           ""));
                            UpdateWriteInterest(ctrlFd);
                            RecordInboundStatus(peer, opId, StatusCode::SUCCESS, "");
                          }});
      return;
    }

    // CPU destination.
    uint8_t* src = reinterpret_cast<uint8_t*>(pinned->ptr);
    uint8_t* dstBase = reinterpret_cast<uint8_t*>(ws.dst.data);
    const uint64_t total = payloadLen;
    uint64_t spanOff = 0;
    uint64_t spanLen = 0;
    if (IsSingleContiguousSpan(ws.dstSegs, &spanOff, &spanLen) && spanLen == total) {
      std::memcpy(dstBase + spanOff, src, static_cast<size_t>(total));
    } else {
      uint64_t off = 0;
      for (const auto& s : ws.dstSegs) {
        std::memcpy(dstBase + s.off, src + off, static_cast<size_t>(s.len));
        off += s.len;
      }
    }

    QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::SUCCESS), ""));
    UpdateWriteInterest(ctrl->fd);
    RecordInboundStatus(peer, opId, StatusCode::SUCCESS, "");
  }

  void HandleWriteReq(const EngineKey& peer, const uint8_t* body, size_t len) {
    size_t off = 0;
    uint64_t opId = 0;
    uint32_t memId = 0;
    uint64_t remoteOff = 0;
    uint64_t size = 0;
    if (!tcp::ReadU64BE(body, len, &off, &opId) || !tcp::ReadU32BE(body, len, &off, &memId) ||
        !tcp::ReadU64BE(body, len, &off, &remoteOff) || !tcp::ReadU64BE(body, len, &off, &size)) {
      MORI_IO_WARN("TCP: malformed WRITE_REQ");
      return;
    }

    auto memOpt = LookupLocalMem(memId);
    InboundWriteState ws;
    ws.peer = peer;
    ws.id = opId;
    ws.discard = true;

    if (memOpt.has_value()) {
      ws.dst = memOpt.value();
      if (remoteOff + size <= ws.dst.size) {
        ws.discard = false;
        ws.dstSegs = {{remoteOff, size}};
      }
    }

    auto ewPeerIt = earlyWrites.find(peer);
    if (ewPeerIt != earlyWrites.end()) {
      auto ewIt = ewPeerIt->second.find(opId);
      if (ewIt != ewPeerIt->second.end()) {
        EarlyWriteState early = std::move(ewIt->second);
        ewPeerIt->second.erase(ewIt);
        if (ewPeerIt->second.empty()) earlyWrites.erase(ewPeerIt);

        if (early.complete) {
          FinalizeBufferedWrite(peer, opId, std::move(ws), early.payloadLen, early.pinned);
          return;
        }
        // Data is still in flight; keep the write state so data RX can finalize it.
        inboundWrites[peer][opId] = std::move(ws);
        return;
      }
    }

    inboundWrites[peer][opId] = std::move(ws);
  }

  void HandleBatchWriteReq(const EngineKey& peer, const uint8_t* body, size_t len) {
    size_t off = 0;
    uint64_t opId = 0;
    uint32_t memId = 0;
    uint32_t n = 0;
    if (!tcp::ReadU64BE(body, len, &off, &opId) || !tcp::ReadU32BE(body, len, &off, &memId) ||
        !tcp::ReadU32BE(body, len, &off, &n)) {
      MORI_IO_WARN("TCP: malformed BATCH_WRITE_REQ");
      return;
    }

    auto memOpt = LookupLocalMem(memId);
    InboundWriteState ws;
    ws.peer = peer;
    ws.id = opId;
    ws.discard = true;

    if (memOpt.has_value()) {
      ws.dst = memOpt.value();
      ws.dstSegs.reserve(n);
      bool ok = true;
      for (uint32_t i = 0; i < n; ++i) {
        uint64_t ro = 0, sz = 0;
        if (!tcp::ReadU64BE(body, len, &off, &ro) || !tcp::ReadU64BE(body, len, &off, &sz)) {
          ok = false;
          break;
        }
        if (ro + sz > ws.dst.size) ok = false;
        if (sz > 0) ws.dstSegs.push_back({ro, sz});
      }
      if (ok) ws.discard = false;
    }

    auto ewPeerIt = earlyWrites.find(peer);
    if (ewPeerIt != earlyWrites.end()) {
      auto ewIt = ewPeerIt->second.find(opId);
      if (ewIt != ewPeerIt->second.end()) {
        EarlyWriteState early = std::move(ewIt->second);
        ewPeerIt->second.erase(ewIt);
        if (ewPeerIt->second.empty()) earlyWrites.erase(ewPeerIt);

        if (early.complete) {
          FinalizeBufferedWrite(peer, opId, std::move(ws), early.payloadLen, early.pinned);
          return;
        }
        inboundWrites[peer][opId] = std::move(ws);
        return;
      }
    }

    inboundWrites[peer][opId] = std::move(ws);
  }

  void HandleReadReq(const EngineKey& peer, const uint8_t* body, size_t len) {
    size_t off = 0;
    uint64_t opId = 0;
    uint32_t memId = 0;
    uint64_t srcOff = 0;
    uint64_t size = 0;
    if (!tcp::ReadU64BE(body, len, &off, &opId) || !tcp::ReadU32BE(body, len, &off, &memId) ||
        !tcp::ReadU64BE(body, len, &off, &srcOff) || !tcp::ReadU64BE(body, len, &off, &size)) {
      MORI_IO_WARN("TCP: malformed READ_REQ");
      return;
    }

    auto memOpt = LookupLocalMem(memId);
    if (!memOpt.has_value() || (srcOff + size > memOpt->size)) {
      // No data; completion with error.
      auto ctrl = PeerCtrl(peer);
      if (ctrl) {
        QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_NOT_FOUND),
                                                 "TCP: remote mem not found/out of range"));
        UpdateWriteInterest(ctrl->fd);
      }
      RecordInboundStatus(peer, opId, StatusCode::ERR_NOT_FOUND, "TCP: read mem not found");
      return;
    }

    MemoryDesc src = memOpt.value();
    std::vector<Segment> segs = {{srcOff, size}};
    QueueDataSendForRead(peer, opId, src, segs);
  }

  void HandleBatchReadReq(const EngineKey& peer, const uint8_t* body, size_t len) {
    size_t off = 0;
    uint64_t opId = 0;
    uint32_t memId = 0;
    uint32_t n = 0;
    if (!tcp::ReadU64BE(body, len, &off, &opId) || !tcp::ReadU32BE(body, len, &off, &memId) ||
        !tcp::ReadU32BE(body, len, &off, &n)) {
      MORI_IO_WARN("TCP: malformed BATCH_READ_REQ");
      return;
    }

    auto memOpt = LookupLocalMem(memId);
    if (!memOpt.has_value()) {
      auto ctrl = PeerCtrl(peer);
      if (ctrl) {
        QueueSend(ctrl->fd,
                  tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_NOT_FOUND),
                                       "TCP: remote mem not found"));
        UpdateWriteInterest(ctrl->fd);
      }
      RecordInboundStatus(peer, opId, StatusCode::ERR_NOT_FOUND, "TCP: read mem not found");
      return;
    }

    MemoryDesc src = memOpt.value();
    std::vector<Segment> segs;
    segs.reserve(n);
    bool ok = true;
    for (uint32_t i = 0; i < n; ++i) {
      uint64_t ro = 0, sz = 0;
      if (!tcp::ReadU64BE(body, len, &off, &ro) || !tcp::ReadU64BE(body, len, &off, &sz)) {
        ok = false;
        break;
      }
      if (ro + sz > src.size) ok = false;
      if (sz > 0) segs.push_back({ro, sz});
    }
    if (!ok) {
      auto ctrl = PeerCtrl(peer);
      if (ctrl) {
        QueueSend(ctrl->fd,
                  tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_INVALID_ARGS),
                                       "TCP: batch read out of range"));
        UpdateWriteInterest(ctrl->fd);
      }
      RecordInboundStatus(peer, opId, StatusCode::ERR_INVALID_ARGS, "TCP: batch read bad args");
      return;
    }

    QueueDataSendForRead(peer, opId, src, segs);
  }

  void QueueDataSendForRead(const EngineKey& peer, uint64_t opId, const MemoryDesc& src,
                            const std::vector<Segment>& srcSegs) {
    Connection* data = PeerData(peer);
    Connection* ctrl = PeerCtrl(peer);
    if (!data || !ctrl) return;

    const uint64_t total = SumLens(srcSegs);
    auto hdr = tcp::BuildDataHeader(opId, total, 0);
    auto onDone = [this, peer, opId, ctrlFd = ctrl->fd]() {
      QueueSend(ctrlFd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::SUCCESS), ""));
      UpdateWriteInterest(ctrlFd);
      RecordInboundStatus(peer, opId, StatusCode::SUCCESS, "");
    };

    if (src.loc == MemoryLocationType::GPU) {
      QueueGpuToNetSend(data, src, srcSegs, std::move(hdr), std::move(onDone));
      return;
    }

    SendItem item;
    item.header = std::move(hdr);
    item.iov.reserve(1 + srcSegs.size());
    item.iov.push_back({item.header.data(), item.header.size()});
    uint8_t* base = reinterpret_cast<uint8_t*>(src.data);
    for (const auto& s : srcSegs) {
      item.iov.push_back({base + s.off, static_cast<size_t>(s.len)});
    }
    item.onDone = std::move(onDone);
    data->sendq.push_back(std::move(item));
    UpdateWriteInterest(data->fd);
  }

  void HandleCompletion(const EngineKey& peer, const uint8_t* body, size_t len) {
    size_t off = 0;
    uint64_t opId = 0;
    uint32_t code = 0;
    uint32_t msgLen = 0;
    if (!tcp::ReadU64BE(body, len, &off, &opId) || !tcp::ReadU32BE(body, len, &off, &code) ||
        !tcp::ReadU32BE(body, len, &off, &msgLen)) {
      MORI_IO_WARN("TCP: malformed COMPLETION");
      return;
    }
    if (off + msgLen > len) {
      MORI_IO_WARN("TCP: malformed COMPLETION msg len");
      return;
    }
    std::string msg(reinterpret_cast<const char*>(body + off), msgLen);

    auto it = pendingOutbound.find(opId);
    if (it == pendingOutbound.end()) return;
    OutboundOpState& st = *it->second;
    st.completionReceived = true;
    st.completionCode = static_cast<StatusCode>(code);
    st.completionMsg = std::move(msg);

    // Fast fail on remote error.
    if (st.completionCode != StatusCode::SUCCESS) {
      st.status->Update(st.completionCode, st.completionMsg);
      pendingOutbound.erase(it);
      return;
    }

    MaybeCompleteOutbound(st);
  }

  void MaybeCompleteOutbound(OutboundOpState& st) {
    if (!st.completionReceived) return;
    if (st.isRead) {
      if (st.rxBytes != st.expectedRxBytes) return;
      if (st.gpuCopyPending) return;
    }
    st.status->Update(StatusCode::SUCCESS, "");
    pendingOutbound.erase(st.id);
  }

  void ConsumeBufferedData(Connection* c) {
    while (!c->inbuf.empty()) {
      if (!c->rx.active) {
        const size_t need = tcp::kDataHeaderSize - c->dataHdrGot;
        const size_t take = std::min(need, c->inbuf.size());
        std::memcpy(c->dataHdrBuf.data() + c->dataHdrGot, c->inbuf.data(), take);
        c->dataHdrGot += take;
        c->inbuf.erase(c->inbuf.begin(), c->inbuf.begin() + take);
        if (c->dataHdrGot < tcp::kDataHeaderSize) return;

        tcp::DataHeaderView hv;
        if (!tcp::TryParseDataHeader(c->dataHdrBuf.data(), tcp::kDataHeaderSize, &hv)) {
          MORI_IO_ERROR("TCP: bad data header during handoff, closing");
          ClosePeerByFd(c->fd);
          return;
        }
        c->dataHdrGot = 0;
        BeginDataRx(c, hv.opId, hv.payloadLen);
        continue;
      }

      if (c->rx.remaining == 0) {
        FinishDataRx(c);
        continue;
      }

      const size_t take = static_cast<size_t>(std::min<uint64_t>(c->rx.remaining, c->inbuf.size()));
      if (take == 0) return;
      const uint8_t* src = c->inbuf.data();

      if (c->rx.discard) {
        // nothing
      } else if (c->rx.toGpu) {
        if (!c->rx.pinned) {
          MORI_IO_ERROR("TCP: missing pinned buffer for GPU recv");
          ClosePeerByFd(c->fd);
          return;
        }
        uint8_t* dst = reinterpret_cast<uint8_t*>(c->rx.pinned->ptr) + c->rx.pinnedWriteOff;
        std::memcpy(dst, src, take);
        c->rx.pinnedWriteOff += take;
      } else {
        size_t copied = 0;
        while (copied < take) {
          if (c->rx.segIdx >= c->rx.segs.size()) {
            MORI_IO_ERROR("TCP: cpu scatter overflow during buffered consume");
            ClosePeerByFd(c->fd);
            return;
          }
          Segment& seg = c->rx.segs[c->rx.segIdx];
          const uint64_t segRemain = seg.len - c->rx.segOff;
          const size_t chunk =
              static_cast<size_t>(std::min<uint64_t>(segRemain, static_cast<uint64_t>(take - copied)));
          uint8_t* dst = reinterpret_cast<uint8_t*>(c->rx.base) + seg.off + c->rx.segOff;
          std::memcpy(dst, src + copied, chunk);
          c->rx.segOff += chunk;
          copied += chunk;
          if (c->rx.segOff >= seg.len) {
            c->rx.segIdx++;
            c->rx.segOff = 0;
          }
        }
      }

      c->rx.remaining -= static_cast<uint64_t>(take);
      c->inbuf.erase(c->inbuf.begin(), c->inbuf.begin() + take);
    }
  }

  void HandleDataReadable(Connection* c) {
    // If we haven't received HELLO on this channel, treat data as ctrl for handshake.
    if (!c->helloReceived) {
      HandleCtrlReadable(c);
      return;
    }

    if (!c->inbuf.empty()) {
      ConsumeBufferedData(c);
      if (c->fd < 0) return;
    }

    while (true) {
      if (!c->rx.active) {
        // Need header.
        while (c->dataHdrGot < tcp::kDataHeaderSize) {
          ssize_t n = ::recv(c->fd, c->dataHdrBuf.data() + c->dataHdrGot,
                             tcp::kDataHeaderSize - c->dataHdrGot, 0);
          if (n < 0) {
            if (IsWouldBlock(errno)) return;
            MORI_IO_ERROR("TCP: recv(data hdr) failed: {}", strerror(errno));
            ClosePeerByFd(c->fd);
            return;
          }
          if (n == 0) {
            ClosePeerByFd(c->fd);
            return;
          }
          c->dataHdrGot += static_cast<size_t>(n);
        }
        tcp::DataHeaderView hv;
        if (!tcp::TryParseDataHeader(c->dataHdrBuf.data(), tcp::kDataHeaderSize, &hv)) {
          MORI_IO_ERROR("TCP: bad data header, closing");
          ClosePeerByFd(c->fd);
          return;
        }
        c->dataHdrGot = 0;
        BeginDataRx(c, hv.opId, hv.payloadLen);
        continue;
      }

      if (c->rx.remaining == 0) {
        FinishDataRx(c);
        continue;
      }

      ssize_t n = 0;
      if (c->rx.discard) {
        uint8_t tmp[8192];
        const size_t want = std::min<uint64_t>(sizeof(tmp), c->rx.remaining);
        n = ::recv(c->fd, tmp, want, 0);
      } else if (c->rx.toGpu) {
        uint8_t* dst = reinterpret_cast<uint8_t*>(c->rx.pinned->ptr) + c->rx.pinnedWriteOff;
        const size_t want = static_cast<size_t>(std::min<uint64_t>(c->rx.remaining, 1ULL << 20));
        n = ::recv(c->fd, dst, want, 0);
      } else {
        // CPU scatter: recv into current segment.
        Segment& seg = c->rx.segs[c->rx.segIdx];
        uint8_t* dst = reinterpret_cast<uint8_t*>(c->rx.base) + seg.off + c->rx.segOff;
        uint64_t segRemain = seg.len - c->rx.segOff;
        const size_t want = static_cast<size_t>(std::min<uint64_t>(c->rx.remaining, segRemain));
        n = ::recv(c->fd, dst, want, 0);
      }

      if (n < 0) {
        if (IsWouldBlock(errno)) return;
        MORI_IO_ERROR("TCP: recv(data) failed: {}", strerror(errno));
        ClosePeerByFd(c->fd);
        return;
      }
      if (n == 0) {
        ClosePeerByFd(c->fd);
        return;
      }

      const uint64_t got = static_cast<uint64_t>(n);
      c->rx.remaining -= got;
      if (c->rx.discard) {
        continue;
      }
      if (c->rx.toGpu) {
        c->rx.pinnedWriteOff += got;
        continue;
      }
      // CPU seg advance
      c->rx.segOff += got;
      Segment& seg = c->rx.segs[c->rx.segIdx];
      if (c->rx.segOff >= seg.len) {
        c->rx.segIdx++;
        c->rx.segOff = 0;
      }
    }
  }

  void BeginDataRx(Connection* c, TransferUniqueId opId, uint64_t payloadLen) {
    c->rx = ActiveDataRx{};
    c->rx.active = true;
    c->rx.id = opId;
    c->rx.remaining = payloadLen;

    // 1) inbound write: peer->me, stored by peer key.
    auto iwIt = inboundWrites.find(c->peerKey);
    if (iwIt != inboundWrites.end()) {
      auto it = iwIt->second.find(opId);
      if (it != iwIt->second.end()) {
        InboundWriteState& ws = it->second;
        if (!ws.discard) {
          const uint64_t expected = SumLens(ws.dstSegs);
          if (expected != payloadLen) {
            MORI_IO_WARN("TCP: inbound write op {} payloadLen mismatch expected={} got={}", opId,
                         expected, payloadLen);
            ws.discard = true;
          }
        }
        c->rx.discard = ws.discard;
        if (!ws.discard) {
          if (ws.dst.loc == MemoryLocationType::GPU) {
            c->rx.toGpu = true;
            c->rx.gpuDevice = ws.dst.deviceId;
            c->rx.gpuBase = reinterpret_cast<void*>(ws.dst.data);
            c->rx.segs = ws.dstSegs;
            const uint64_t total = SumLens(ws.dstSegs);
            c->rx.pinned = staging.Acquire(static_cast<size_t>(total));
            c->rx.pinnedWriteOff = 0;
          } else {
            c->rx.toGpu = false;
            c->rx.base = reinterpret_cast<void*>(ws.dst.data);
            c->rx.segs = ws.dstSegs;
            c->rx.segIdx = 0;
            c->rx.segOff = 0;
          }
        }
        return;
      }
    }

    // 2) outbound read response: peer->me for my pending outbound read.
    auto obIt = pendingOutbound.find(opId);
    if (obIt != pendingOutbound.end()) {
      OutboundOpState& st = *obIt->second;
      if (!st.isRead) {
        c->rx.discard = true;
        return;
      }
      if (payloadLen != st.expectedRxBytes) {
        MORI_IO_ERROR("TCP: outbound read op {} payloadLen mismatch expected={} got={}", opId,
                      st.expectedRxBytes, payloadLen);
        st.status->Update(StatusCode::ERR_BAD_STATE, "TCP: read payload length mismatch");
        pendingOutbound.erase(obIt);
        c->rx.discard = true;
        return;
      }
      c->rx.discard = false;
      if (st.local.loc == MemoryLocationType::GPU) {
        c->rx.toGpu = true;
        c->rx.gpuDevice = st.local.deviceId;
        c->rx.gpuBase = reinterpret_cast<void*>(st.local.data);
        c->rx.segs = st.localSegs;
        c->rx.pinned = staging.Acquire(static_cast<size_t>(payloadLen));
        c->rx.pinnedWriteOff = 0;
      } else {
        c->rx.toGpu = false;
        c->rx.base = reinterpret_cast<void*>(st.local.data);
        c->rx.segs = st.localSegs;
        c->rx.segIdx = 0;
        c->rx.segOff = 0;
      }
      return;
    }

    // Control+data are on separate TCP connections; data may arrive before its CTRL write request.
    // Buffer it in pinned host memory, and finalize once CTRL arrives.
    if (payloadLen > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      MORI_IO_WARN("TCP: data payloadLen too large for op {} from peer {}, discarding", opId,
                   c->peerKey.c_str());
      c->rx.discard = true;
      return;
    }
    auto& perPeer = earlyWrites[c->peerKey];
    if (perPeer.find(opId) != perPeer.end()) {
      MORI_IO_WARN("TCP: duplicate early data for op {} from peer {}, discarding", opId,
                   c->peerKey.c_str());
      c->rx.discard = true;
      return;
    }

    const size_t alloc = payloadLen == 0 ? 1 : static_cast<size_t>(payloadLen);
    auto pinned = staging.Acquire(alloc);
    if (!pinned) {
      MORI_IO_WARN("TCP: failed to allocate pinned buffer for early data op {} from peer {}, discarding",
                   opId, c->peerKey.c_str());
      c->rx.discard = true;
      return;
    }

    perPeer.emplace(opId, EarlyWriteState{payloadLen, pinned, false});
    c->rx.discard = false;
    c->rx.toGpu = true;  // receive into pinned
    c->rx.pinned = pinned;
    c->rx.pinnedWriteOff = 0;
  }

  void FinishDataRx(Connection* c) {
    const TransferUniqueId opId = c->rx.id;
    const EngineKey peer = c->peerKey;

    // Case A: inbound write complete -> copy to GPU if needed, then send completion.
    auto iwIt = inboundWrites.find(peer);
	    if (iwIt != inboundWrites.end()) {
	      auto it = iwIt->second.find(opId);
	      if (it != iwIt->second.end()) {
	        InboundWriteState ws = std::move(it->second);
	        iwIt->second.erase(it);
	        auto ewPeerIt = earlyWrites.find(peer);
	        if (ewPeerIt != earlyWrites.end()) {
	          ewPeerIt->second.erase(opId);
	          if (ewPeerIt->second.empty()) earlyWrites.erase(ewPeerIt);
	        }

	        if (ws.discard) {
	          auto ctrl = PeerCtrl(peer);
	          if (ctrl) {
            QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_INVALID_ARGS),
                                                     "TCP: write discarded"));
            UpdateWriteInterest(ctrl->fd);
          }
          RecordInboundStatus(peer, opId, StatusCode::ERR_INVALID_ARGS, "TCP: write discarded");
          c->rx = ActiveDataRx{};
          return;
        }

        auto ctrl = PeerCtrl(peer);
        if (!ctrl) {
          c->rx = ActiveDataRx{};
          return;
        }

        if (ws.dst.loc == MemoryLocationType::GPU) {
          hipStream_t stream = streamPool.GetNextStream(ws.dst.deviceId);
          hipEvent_t ev = eventPool.GetEvent(ws.dst.deviceId);
          if (stream == nullptr || ev == nullptr || !c->rx.pinned) {
            QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::ERR_BAD_STATE),
                                                     "TCP: GPU staging missing"));
            UpdateWriteInterest(ctrl->fd);
            RecordInboundStatus(peer, opId, StatusCode::ERR_BAD_STATE, "TCP: GPU staging missing");
            c->rx = ActiveDataRx{};
            return;
          }

          HIP_RUNTIME_CHECK(hipSetDevice(ws.dst.deviceId));
          uint8_t* src = reinterpret_cast<uint8_t*>(c->rx.pinned->ptr);
          const uint64_t total = SumLens(ws.dstSegs);
          uint64_t spanOff = 0;
          uint64_t spanLen = 0;
          if (IsSingleContiguousSpan(ws.dstSegs, &spanOff, &spanLen) && spanLen == total) {
            void* gpuPtr = reinterpret_cast<void*>(ws.dst.data + spanOff);
            HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpuPtr, src, static_cast<size_t>(total), stream));
          } else {
            uint64_t off = 0;
            for (const auto& s : ws.dstSegs) {
              void* gpuPtr = reinterpret_cast<void*>(ws.dst.data + s.off);
              HIP_RUNTIME_CHECK(
                  hipMemcpyHtoDAsync(gpuPtr, src + off, static_cast<size_t>(s.len), stream));
              off += s.len;
            }
          }
          HIP_RUNTIME_CHECK(hipEventRecord(ev, stream));

          auto pinned = c->rx.pinned;
          c->rx = ActiveDataRx{};

          gpuTasks.push_back({ws.dst.deviceId, ev, [this, peer, opId, ctrlFd = ctrl->fd, pinned]() {
                                QueueSend(ctrlFd,
                                          tcp::BuildCompletion(opId,
                                                               static_cast<uint32_t>(StatusCode::SUCCESS),
                                                               ""));
                                UpdateWriteInterest(ctrlFd);
                                RecordInboundStatus(peer, opId, StatusCode::SUCCESS, "");
                              }});
          return;
	        }

	        if (c->rx.pinned) {
	          uint8_t* src = reinterpret_cast<uint8_t*>(c->rx.pinned->ptr);
	          uint8_t* dstBase = reinterpret_cast<uint8_t*>(ws.dst.data);
	          const uint64_t total = SumLens(ws.dstSegs);
	          uint64_t spanOff = 0;
	          uint64_t spanLen = 0;
	          if (IsSingleContiguousSpan(ws.dstSegs, &spanOff, &spanLen) && spanLen == total) {
	            std::memcpy(dstBase + spanOff, src, static_cast<size_t>(total));
	          } else {
	            uint64_t off = 0;
	            for (const auto& s : ws.dstSegs) {
	              std::memcpy(dstBase + s.off, src + off, static_cast<size_t>(s.len));
	              off += s.len;
	            }
	          }
	        }

	        QueueSend(ctrl->fd, tcp::BuildCompletion(opId, static_cast<uint32_t>(StatusCode::SUCCESS), ""));
	        UpdateWriteInterest(ctrl->fd);
	        RecordInboundStatus(peer, opId, StatusCode::SUCCESS, "");
	        c->rx = ActiveDataRx{};
	        return;
      }
    }

    // Case B: outbound read response received -> copy to GPU if needed and update status.
    auto obIt = pendingOutbound.find(opId);
	    if (obIt != pendingOutbound.end()) {
	      OutboundOpState& st = *obIt->second;
	      st.rxBytes = st.expectedRxBytes;
	      if (st.local.loc == MemoryLocationType::GPU) {
        hipStream_t stream = streamPool.GetNextStream(st.local.deviceId);
        hipEvent_t ev = eventPool.GetEvent(st.local.deviceId);
        if (stream == nullptr || ev == nullptr || !c->rx.pinned) {
          st.status->Update(StatusCode::ERR_BAD_STATE, "TCP: GPU staging missing (read)");
          pendingOutbound.erase(obIt);
          c->rx = ActiveDataRx{};
          return;
        }

        HIP_RUNTIME_CHECK(hipSetDevice(st.local.deviceId));
        uint8_t* src = reinterpret_cast<uint8_t*>(c->rx.pinned->ptr);
        st.gpuCopyPending = true;
        const uint64_t total = st.expectedRxBytes;
        uint64_t spanOff = 0;
        uint64_t spanLen = 0;
        if (IsSingleContiguousSpan(st.localSegs, &spanOff, &spanLen) && spanLen == total) {
          void* gpuPtr = reinterpret_cast<void*>(st.local.data + spanOff);
          HIP_RUNTIME_CHECK(hipMemcpyHtoDAsync(gpuPtr, src, static_cast<size_t>(total), stream));
        } else {
          uint64_t off = 0;
          for (const auto& s : st.localSegs) {
            void* gpuPtr = reinterpret_cast<void*>(st.local.data + s.off);
            HIP_RUNTIME_CHECK(
                hipMemcpyHtoDAsync(gpuPtr, src + off, static_cast<size_t>(s.len), stream));
            off += s.len;
          }
        }
        HIP_RUNTIME_CHECK(hipEventRecord(ev, stream));

        auto pinned = c->rx.pinned;
        c->rx = ActiveDataRx{};

        gpuTasks.push_back({st.local.deviceId, ev, [this, opId, pinned]() {
                              auto it = pendingOutbound.find(opId);
                              if (it == pendingOutbound.end()) return;
                              it->second->gpuCopyPending = false;
                              MaybeCompleteOutbound(*it->second);
                            }});
        return;
      }

      c->rx = ActiveDataRx{};
	      MaybeCompleteOutbound(st);
	      return;
	    }

	    // Case C: early-arrived write payload (data before CTRL). Mark it complete and finalize once
	    // the corresponding CTRL write request arrives.
	    auto ewPeerIt = earlyWrites.find(peer);
	    if (ewPeerIt != earlyWrites.end()) {
	      auto ewIt = ewPeerIt->second.find(opId);
	      if (ewIt != ewPeerIt->second.end()) {
	        ewIt->second.complete = true;
	        c->rx = ActiveDataRx{};
	        return;
	      }
	    }

	    c->rx = ActiveDataRx{};
	  }

  void HandleReadable(Connection* c) {
    if (!c->helloReceived) {
      HandleCtrlReadable(c);
      return;
    }
    if (c->ch == tcp::Channel::CTRL) {
      HandleCtrlReadable(c);
    } else {
      HandleDataReadable(c);
    }
  }

  void ScanTimeouts() {
    if (config.opTimeoutMs <= 0) return;
    const auto now = Clock::now();
    const auto timeout = std::chrono::milliseconds(config.opTimeoutMs);
    for (auto it = pendingOutbound.begin(); it != pendingOutbound.end();) {
      if ((now - it->second->startTs) > timeout) {
        it->second->status->Update(StatusCode::ERR_BAD_STATE, "TCP: op timeout");
        it = pendingOutbound.erase(it);
      } else {
        ++it;
      }
    }
  }

  void IoLoop() {
    constexpr int kMaxEvents = 128;
    epoll_event events[kMaxEvents];

	    while (running.load()) {
	      PollGpuTasks();
	      ScanTimeouts();

	      // When GPU staging is in flight, prefer low-latency polling so we can arm the next
	      // network send / completion promptly after HIP events complete.
	      const int timeoutMs = gpuTasks.empty() ? 5 /*ms*/ : 0 /*busy*/;
	      int nfds = epoll_wait(epfd, events, kMaxEvents, timeoutMs);
	      if (nfds < 0) {
	        if (errno == EINTR) continue;
	        MORI_IO_ERROR("TCP: epoll_wait failed: {}", strerror(errno));
	        break;
	      }

      for (int i = 0; i < nfds; ++i) {
        int fd = events[i].data.fd;
        uint32_t ev = events[i].events;

        if (fd == listenFd) {
          AcceptNew();
          continue;
        }
        if (fd == wakeFd) {
          DrainWakeFd();
          continue;
        }

        Connection* c = nullptr;
        auto it = conns.find(fd);
        if (it != conns.end()) c = it->second.get();
        if (!c) continue;

        if (ev & (EPOLLERR | EPOLLHUP)) {
          ClosePeerByFd(fd);
          continue;
        }

        if (ev & EPOLLIN) {
          HandleReadable(c);
          auto it2 = conns.find(fd);
          if (it2 == conns.end()) {
            continue;
          }
          c = it2->second.get();
          if (!c) continue;
        }
        if (ev & EPOLLOUT) {
          HandleConnWritable(c);
        }
      }
    }
  }

 private:
  EngineKey myEngKey;
  IOEngineConfig engConfig;
  TcpBackendConfig config;

  int epfd{-1};
  int listenFd{-1};
  int wakeFd{-1};
  uint16_t listenPort{0};

  std::atomic<bool> running{false};
  std::thread ioThread;

  std::mutex submitMu;
  std::deque<std::unique_ptr<OutboundOpState>> submitQ;

  std::mutex remoteMu;
  std::unordered_map<EngineKey, EngineDesc> remoteEngines;

  std::mutex memMu;
  std::unordered_map<MemoryUniqueId, MemoryDesc> localMems;

  std::mutex inboundMu;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, InboundStatusEntry>> inboundStatus;

  std::unordered_map<int, std::unique_ptr<Connection>> conns;
  std::unordered_map<EngineKey, PeerLinks> peers;

  std::unordered_map<EngineKey, std::vector<std::unique_ptr<OutboundOpState>>> waitingOps;

	  std::unordered_map<TransferUniqueId, std::unique_ptr<OutboundOpState>> pendingOutbound;

	  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, InboundWriteState>> inboundWrites;
	  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, EarlyWriteState>> earlyWrites;

	  PinnedStagingPool staging;
	  StreamPool streamPool{8};
	  EventPool eventPool{64};
	  std::deque<GpuTask> gpuTasks;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                       TcpBackendSession                                        */
/* ---------------------------------------------------------------------------------------------- */
TcpBackendSession::TcpBackendSession(const TcpBackendConfig& cfg, const MemoryDesc& l, const MemoryDesc& r,
                                     TcpTransport* t)
    : config(cfg), local(l), remote(r), transport(t) {}

void TcpBackendSession::ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                                  TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitReadWrite(local, localOffset, remote, remoteOffset, size, status, id, isRead);
}

void TcpBackendSession::BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                                       const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                                       bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitBatchReadWrite(local, localOffsets, remote, remoteOffsets, sizes, status, id, isRead);
}

bool TcpBackendSession::Alive() const { return true; }

/* ---------------------------------------------------------------------------------------------- */
/*                                           TcpBackend                                           */
/* ---------------------------------------------------------------------------------------------- */
TcpBackend::TcpBackend(EngineKey k, const IOEngineConfig& engCfg, const TcpBackendConfig& cfg)
    : myEngKey(std::move(k)), config(cfg) {
  transport = std::make_unique<TcpTransport>(myEngKey, engCfg, cfg);
  transport->Start();
  MORI_IO_INFO("TcpBackend created key={}", myEngKey.c_str());
}

TcpBackend::~TcpBackend() { transport->Shutdown(); }

std::optional<uint16_t> TcpBackend::GetListenPort() const { return transport->GetListenPort(); }

void TcpBackend::RegisterRemoteEngine(const EngineDesc& desc) { transport->RegisterRemoteEngine(desc); }

void TcpBackend::DeregisterRemoteEngine(const EngineDesc& desc) { transport->DeregisterRemoteEngine(desc); }

void TcpBackend::RegisterMemory(MemoryDesc& desc) { transport->RegisterMemory(desc); }

void TcpBackend::DeregisterMemory(const MemoryDesc& desc) { transport->DeregisterMemory(desc); }

void TcpBackend::ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                           size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                           bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitReadWrite(localDest, localOffset, remoteSrc, remoteOffset, size, status, id, isRead);
}

void TcpBackend::BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets, const MemoryDesc& remoteSrc,
                                const SizeVec& remoteOffsets, const SizeVec& sizes, TransferStatus* status,
                                TransferUniqueId id, bool isRead) {
  MORI_IO_FUNCTION_TIMER;
  transport->SubmitBatchReadWrite(localDest, localOffsets, remoteSrc, remoteOffsets, sizes, status, id, isRead);
}

BackendSession* TcpBackend::CreateSession(const MemoryDesc& local, const MemoryDesc& remote) {
  auto* sess = new TcpBackendSession(config, local, remote, transport.get());
  return sess;
}

bool TcpBackend::PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status) {
  return transport->PopInboundTransferStatus(remote, id, status);
}

}  // namespace io
}  // namespace mori

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
#pragma once

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/logging.hpp"
#include "src/io/tcp/protocol.hpp"

namespace mori {
namespace io {

class DataConnectionWorker;

using Clock = std::chrono::steady_clock;

inline bool IsWouldBlock(int err) { return (err == EAGAIN) || (err == EWOULDBLOCK); }

inline int SetNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return -1;
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return -1;
  return 0;
}

inline int SetBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return -1;
  if (fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0) return -1;
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
  {
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
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) return std::nullopt;
  return addr;
}

inline uint16_t GetBoundPort(int fd) {
  sockaddr_in addr{};
  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) return 0;
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
        return std::shared_ptr<PinnedBuf>(new PinnedBuf{p, cap},
                                          [this](PinnedBuf* b) { this->Release(b); });
      }
    }
    void* p = nullptr;
    hipError_t err = hipHostMalloc(&p, cap, hipHostMallocDefault);
    if (err != hipSuccess) {
      MORI_IO_ERROR("TCP: hipHostMalloc({}) failed: {}", cap, hipGetErrorString(err));
      return nullptr;
    }
    return std::shared_ptr<PinnedBuf>(new PinnedBuf{p, cap},
                                      [this](PinnedBuf* b) { this->Release(b); });
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

  std::mutex mu;
  std::unordered_map<size_t, std::vector<void*>> free;
};

struct Segment {
  uint64_t off{0};
  uint64_t len{0};
};

constexpr uint8_t kLaneBits = 3;
constexpr uint64_t kLaneMask = (1ULL << kLaneBits) - 1ULL;

inline uint64_t ToWireOpId(uint64_t userOpId, uint8_t lane) {
  return (userOpId << kLaneBits) | lane;
}
inline uint64_t ToUserOpId(uint64_t wireOpId) { return wireOpId >> kLaneBits; }

struct LaneSpan {
  uint64_t off{0};
  uint64_t len{0};
};

inline LaneSpan ComputeLaneSpan(uint64_t total, uint8_t lanesTotal, uint8_t lane) {
  if (lanesTotal <= 1) return LaneSpan{0, total};
  const uint64_t base = total / lanesTotal;
  const uint64_t rem = total % lanesTotal;
  return LaneSpan{static_cast<uint64_t>(lane) * base + std::min<uint64_t>(lane, rem),
                  base + (lane < rem ? 1 : 0)};
}

inline uint8_t LanesAllMask(uint8_t lanesTotal) {
  if (lanesTotal >= (1U << kLaneBits)) return 0xFF;
  return static_cast<uint8_t>((1U << lanesTotal) - 1U);
}

inline uint8_t ClampLanesTotal(uint8_t lanesTotal) {
  if (lanesTotal == 0) return 1;
  return std::min<uint8_t>(lanesTotal, static_cast<uint8_t>(1U << kLaneBits));
}

inline std::vector<Segment> SliceSegments(const std::vector<Segment>& segs, uint64_t start,
                                          uint64_t len) {
  std::vector<Segment> out;
  if (len == 0) return out;
  uint64_t skip = start;
  uint64_t remaining = len;
  for (const auto& s : segs) {
    if (remaining == 0) break;
    if (skip >= s.len) {
      skip -= s.len;
      continue;
    }
    const uint64_t take = std::min<uint64_t>(s.len - skip, remaining);
    out.push_back({s.off + skip, take});
    remaining -= take;
    skip = 0;
  }
  return out;
}

inline bool IsSingleContiguousSpan(const std::vector<Segment>& segs, uint64_t* outOff,
                                   uint64_t* outLen) {
  if (!outOff || !outLen || segs.empty()) return false;
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
  std::vector<uint8_t> inbuf;
  std::deque<SendItem> sendq;
};

struct PeerLinks {
  int ctrlFd{-1};
  std::vector<int> dataFds;
  std::vector<DataConnectionWorker*> workers;
  int ctrlPending{0};
  int dataPending{0};
  size_t rr{0};
  bool CtrlUp() const { return ctrlFd >= 0; }
  bool DataUp() const { return !dataFds.empty(); }
};

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
  uint8_t lanesTotal{1};
  uint8_t lanesDoneMask{0};
  StatusCode completionCode{StatusCode::SUCCESS};
  std::string completionMsg;
  bool gpuCopyPending{false};
  std::shared_ptr<PinnedBuf> pinned;
  Clock::time_point startTs{Clock::now()};
};

struct InboundWriteState {
  EngineKey peer;
  TransferUniqueId id{0};
  MemoryDesc dst{};
  std::vector<Segment> dstSegs;
  bool discard{false};
  uint8_t lanesTotal{1};
  uint8_t lanesDoneMask{0};
  std::shared_ptr<PinnedBuf> pinned;
};

struct EarlyWriteLaneState {
  uint64_t payloadLen{0};
  std::shared_ptr<PinnedBuf> pinned;
  bool complete{false};
};

struct EarlyWriteState {
  std::unordered_map<uint8_t, EarlyWriteLaneState> lanes;
};

struct WorkerRecvTarget {
  uint8_t lanesTotal{1};
  uint64_t totalLen{0};
  bool discard{false};
  bool toGpu{false};
  void* cpuBase{nullptr};
  std::vector<Segment> segs;
  std::shared_ptr<PinnedBuf> pinned;
};

enum class WorkerEventType : uint8_t {
  RECV_DONE = 0,
  EARLY_DATA = 1,
  SEND_CALLBACK = 2,
  CONN_ERROR = 3,
};

struct WorkerEvent {
  WorkerEventType type{WorkerEventType::RECV_DONE};
  EngineKey peerKey;
  TransferUniqueId opId{0};
  uint8_t lane{0};
  uint64_t laneLen{0};
  bool discarded{false};
  std::shared_ptr<PinnedBuf> earlyBuf;
  std::function<void()> callback;
  std::string errorMsg;
};

struct GpuTask {
  int deviceId{-1};
  hipEvent_t ev{nullptr};
  std::function<void()> onReady;
};

}  // namespace io
}  // namespace mori

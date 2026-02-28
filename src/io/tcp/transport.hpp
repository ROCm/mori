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
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/logging.hpp"
#include "src/io/xgmi/hip_resource_pool.hpp"

namespace mori {
namespace io {

// ---------------------------------------------------------------------------
// Socket utilities
// ---------------------------------------------------------------------------
inline bool IsWouldBlock(int err) { return err == EAGAIN || err == EWOULDBLOCK; }

inline int SetNonBlocking(int fd) {
  int f = fcntl(fd, F_GETFL, 0);
  return (f < 0 || fcntl(fd, F_SETFL, f | O_NONBLOCK) < 0) ? -1 : 0;
}

inline void SetSockOpt(int fd, int level, int opt, const void* val, socklen_t len, const char* nm) {
  if (setsockopt(fd, level, opt, val, len) != 0)
    MORI_IO_WARN("TCP: setsockopt {} failed: {}", nm, strerror(errno));
}

inline void ConfigureSocketCommon(int fd, const TcpBackendConfig& cfg) {
  if (!cfg.enableKeepalive) return;
  int on = 1;
  SetSockOpt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on), "SO_KEEPALIVE");
  SetSockOpt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &cfg.keepaliveIdleSec, sizeof(cfg.keepaliveIdleSec),
             "TCP_KEEPIDLE");
  SetSockOpt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &cfg.keepaliveIntvlSec, sizeof(cfg.keepaliveIntvlSec),
             "TCP_KEEPINTVL");
  SetSockOpt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cfg.keepaliveCnt, sizeof(cfg.keepaliveCnt),
             "TCP_KEEPCNT");
}

inline void ConfigureCtrlSocket(int fd, const TcpBackendConfig& cfg) {
  ConfigureSocketCommon(fd, cfg);
  if (cfg.enableCtrlNodelay) {
    int on = 1;
    SetSockOpt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on), "TCP_NODELAY(ctrl)");
  }
}

inline void ConfigureDataSocket(int fd, const TcpBackendConfig& cfg) {
  ConfigureSocketCommon(fd, cfg);
  int on = 1;
  SetSockOpt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on), "TCP_NODELAY(data)");
  if (cfg.sockSndbufBytes > 0)
    SetSockOpt(fd, SOL_SOCKET, SO_SNDBUF, &cfg.sockSndbufBytes, sizeof(cfg.sockSndbufBytes),
               "SO_SNDBUF");
  if (cfg.sockRcvbufBytes > 0)
    SetSockOpt(fd, SOL_SOCKET, SO_RCVBUF, &cfg.sockRcvbufBytes, sizeof(cfg.sockRcvbufBytes),
               "SO_RCVBUF");
}

inline std::optional<sockaddr_in> ParseIpv4(const std::string& host, uint16_t port) {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) return std::nullopt;
  return a;
}

inline uint16_t GetBoundPort(int fd) {
  sockaddr_in a{};
  socklen_t l = sizeof(a);
  return (getsockname(fd, reinterpret_cast<sockaddr*>(&a), &l) == 0) ? ntohs(a.sin_port) : 0;
}

// ---------------------------------------------------------------------------
// Wire protocol constants and helpers
// ---------------------------------------------------------------------------
namespace tcp {

constexpr uint32_t kCtrlMagic = 0x4D544330;  // "MTC0"
constexpr uint32_t kDataMagic = 0x4D544430;  // "MTD0"
constexpr uint16_t kProtoVersion = 2;

enum class Channel : uint8_t { CTRL = 1, DATA = 2 };

enum class CtrlMsgType : uint8_t {
  HELLO = 1,
  WRITE_REQ = 2,
  READ_REQ = 3,
  BATCH_WRITE_REQ = 4,
  BATCH_READ_REQ = 5,
  COMPLETION = 6,
};

constexpr size_t kCtrlHeaderSize = 12;
constexpr size_t kDataHeaderSize = 24;

struct CtrlHeaderView {
  CtrlMsgType type{CtrlMsgType::HELLO};
  uint32_t bodyLen{0};
};
struct DataHeaderView {
  uint16_t flags{0};
  uint64_t opId{0};
  uint64_t payloadLen{0};
};

// Compact wire writer (appends big-endian values)
struct WireWriter {
  std::vector<uint8_t> buf;
  void reserve(size_t n) { buf.reserve(n); }
  void u8(uint8_t v) { buf.push_back(v); }
  void u16(uint16_t v) {
    v = htons(v);
    auto* p = reinterpret_cast<uint8_t*>(&v);
    buf.insert(buf.end(), p, p + 2);
  }
  void u32(uint32_t v) {
    v = htonl(v);
    auto* p = reinterpret_cast<uint8_t*>(&v);
    buf.insert(buf.end(), p, p + 4);
  }
  void u64(uint64_t v) {
    v = htobe64(v);
    auto* p = reinterpret_cast<uint8_t*>(&v);
    buf.insert(buf.end(), p, p + 8);
  }
  void bytes(const void* d, size_t n) {
    auto* p = static_cast<const uint8_t*>(d);
    buf.insert(buf.end(), p, p + n);
  }
};

// Compact wire reader (reads big-endian values from buffer)
struct WireReader {
  const uint8_t* data;
  size_t len;
  size_t off{0};
  bool u8(uint8_t* o) {
    if (off + 1 > len) return false;
    *o = data[off++];
    return true;
  }
  bool u16(uint16_t* o) {
    if (off + 2 > len) return false;
    uint16_t v;
    memcpy(&v, data + off, 2);
    *o = ntohs(v);
    off += 2;
    return true;
  }
  bool u32(uint32_t* o) {
    if (off + 4 > len) return false;
    uint32_t v;
    memcpy(&v, data + off, 4);
    *o = ntohl(v);
    off += 4;
    return true;
  }
  bool u64(uint64_t* o) {
    if (off + 8 > len) return false;
    uint64_t v;
    memcpy(&v, data + off, 8);
    *o = be64toh(v);
    off += 8;
    return true;
  }
};

inline bool TryParseCtrlHeader(const uint8_t* buf, size_t len, CtrlHeaderView* h) {
  if (len < kCtrlHeaderSize) return false;
  WireReader r{buf, len};
  uint32_t magic;
  uint16_t ver;
  uint8_t type, reserved;
  if (!r.u32(&magic) || !r.u16(&ver) || !r.u8(&type) || !r.u8(&reserved) || !r.u32(&h->bodyLen))
    return false;
  if (magic != kCtrlMagic || ver != kProtoVersion) return false;
  h->type = static_cast<CtrlMsgType>(type);
  return true;
}

inline bool TryParseDataHeader(const uint8_t* buf, size_t len, DataHeaderView* h) {
  if (len < kDataHeaderSize) return false;
  WireReader r{buf, len};
  uint32_t magic;
  uint16_t ver;
  if (!r.u32(&magic) || !r.u16(&ver) || !r.u16(&h->flags) || !r.u64(&h->opId) ||
      !r.u64(&h->payloadLen))
    return false;
  return (magic == kDataMagic && ver == kProtoVersion);
}

// Build a ctrl frame: header(12B) + body
inline std::vector<uint8_t> BuildCtrlFrame(CtrlMsgType type,
                                           const std::function<void(WireWriter&)>& writeBody) {
  WireWriter body;
  writeBody(body);
  WireWriter frame;
  frame.reserve(kCtrlHeaderSize + body.buf.size());
  frame.u32(kCtrlMagic);
  frame.u16(kProtoVersion);
  frame.u8(static_cast<uint8_t>(type));
  frame.u8(0);
  frame.u32(static_cast<uint32_t>(body.buf.size()));
  frame.bytes(body.buf.data(), body.buf.size());
  return std::move(frame.buf);
}

inline std::vector<uint8_t> BuildHello(Channel ch, const EngineKey& key) {
  return BuildCtrlFrame(CtrlMsgType::HELLO, [&](WireWriter& w) {
    w.u8(static_cast<uint8_t>(ch));
    w.u32(static_cast<uint32_t>(key.size()));
    w.bytes(key.data(), key.size());
  });
}

// Unified request builder for WRITE_REQ, READ_REQ (single segment)
inline std::vector<uint8_t> BuildLinearReq(CtrlMsgType type, uint64_t opId, uint32_t memId,
                                           uint64_t off, uint64_t size, uint8_t lanes) {
  return BuildCtrlFrame(type, [&](WireWriter& w) {
    w.u64(opId);
    w.u32(memId);
    w.u64(off);
    w.u64(size);
    w.u8(lanes);
  });
}

// Unified request builder for BATCH_WRITE_REQ, BATCH_READ_REQ
inline std::vector<uint8_t> BuildBatchReq(CtrlMsgType type, uint64_t opId, uint32_t memId,
                                          const std::vector<uint64_t>& offs,
                                          const std::vector<uint64_t>& sizes, uint8_t lanes) {
  return BuildCtrlFrame(type, [&](WireWriter& w) {
    w.u64(opId);
    w.u32(memId);
    w.u32(static_cast<uint32_t>(offs.size()));
    for (size_t i = 0; i < offs.size(); ++i) {
      w.u64(offs[i]);
      w.u64(sizes[i]);
    }
    w.u8(lanes);
  });
}

inline std::vector<uint8_t> BuildCompletion(uint64_t opId, uint32_t code, const std::string& msg) {
  return BuildCtrlFrame(CtrlMsgType::COMPLETION, [&](WireWriter& w) {
    w.u64(opId);
    w.u32(code);
    w.u32(static_cast<uint32_t>(msg.size()));
    w.bytes(msg.data(), msg.size());
  });
}

inline std::vector<uint8_t> BuildDataHeader(uint64_t opId, uint64_t payloadLen, uint16_t flags) {
  WireWriter w;
  w.reserve(kDataHeaderSize);
  w.u32(kDataMagic);
  w.u16(kProtoVersion);
  w.u16(flags);
  w.u64(opId);
  w.u64(payloadLen);
  return std::move(w.buf);
}

}  // namespace tcp

// ---------------------------------------------------------------------------
// Segment and lane helpers
// ---------------------------------------------------------------------------
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

inline LaneSpan ComputeLaneSpan(uint64_t total, uint8_t lanes, uint8_t lane) {
  if (lanes <= 1) return {0, total};
  uint64_t base = total / lanes, rem = total % lanes;
  return {uint64_t(lane) * base + std::min<uint64_t>(lane, rem), base + (lane < rem ? 1 : 0)};
}

inline uint8_t LanesAllMask(uint8_t n) {
  return (n >= (1U << kLaneBits)) ? 0xFF : uint8_t((1U << n) - 1);
}
inline uint8_t ClampLanesTotal(uint8_t n) {
  return n == 0 ? 1 : std::min<uint8_t>(n, 1U << kLaneBits);
}

inline uint64_t SumLens(const std::vector<Segment>& segs) {
  uint64_t t = 0;
  for (auto& s : segs) t += s.len;
  return t;
}

inline std::vector<Segment> SliceSegments(const std::vector<Segment>& segs, uint64_t start,
                                          uint64_t len) {
  std::vector<Segment> out;
  if (len == 0) return out;
  uint64_t skip = start, remaining = len;
  for (auto& s : segs) {
    if (remaining == 0) break;
    if (skip >= s.len) {
      skip -= s.len;
      continue;
    }
    uint64_t take = std::min(s.len - skip, remaining);
    out.push_back({s.off + skip, take});
    remaining -= take;
    skip = 0;
  }
  return out;
}

inline bool IsSingleContiguousSpan(const std::vector<Segment>& segs, uint64_t* outOff,
                                   uint64_t* outLen) {
  if (segs.empty()) return false;
  uint64_t off = segs[0].off, end = off + segs[0].len;
  for (size_t i = 1; i < segs.size(); ++i) {
    if (segs[i].off != end) return false;
    end += segs[i].len;
  }
  *outOff = off;
  *outLen = end - off;
  return true;
}

inline bool SegmentsInRange(const std::vector<Segment>& segs, uint64_t memSize) {
  for (auto& s : segs)
    if (s.off + s.len > memSize) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Pinned staging pool (HIP host memory)
// ---------------------------------------------------------------------------
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

  std::shared_ptr<PinnedBuf> Acquire(size_t size);
  void Clear();

 private:
  void Release(PinnedBuf* b);
  static size_t RoundUp(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
  }

  std::mutex mu_;
  std::unordered_map<size_t, std::vector<void*>> free_;
};

// ---------------------------------------------------------------------------
// Send / Connection / Peer state
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

struct SendItem {
  std::vector<uint8_t> header;
  std::vector<iovec> iov;
  size_t idx{0}, off{0};
  int flags{0};
  std::shared_ptr<void> keepalive;
  std::function<void()> onDone;
  bool Done() const { return idx >= iov.size(); }
  void Advance(size_t n);
};

struct Connection {
  int fd{-1};
  bool isOutgoing{false}, connecting{false}, helloSent{false}, helloReceived{false};
  tcp::Channel ch{tcp::Channel::CTRL};
  EngineKey peerKey;
  std::vector<uint8_t> inbuf;
  std::deque<SendItem> sendq;
};

class DataConnectionWorker;

struct PeerLinks {
  int ctrlFd{-1};
  std::vector<int> dataFds;
  std::vector<DataConnectionWorker*> workers;
  int ctrlPending{0}, dataPending{0};
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
  MemoryDesc local{}, remote{};
  std::vector<Segment> localSegs, remoteSegs;
  uint64_t expectedRxBytes{0}, rxBytes{0};
  bool completionReceived{false}, gpuCopyPending{false};
  uint8_t lanesTotal{1}, lanesDoneMask{0};
  StatusCode completionCode{StatusCode::SUCCESS};
  std::string completionMsg;
  std::shared_ptr<PinnedBuf> pinned;
  Clock::time_point startTs{Clock::now()};
};

struct InboundWriteState {
  EngineKey peer;
  TransferUniqueId id{0};
  MemoryDesc dst{};
  std::vector<Segment> dstSegs;
  bool discard{false};
  uint8_t lanesTotal{1}, lanesDoneMask{0};
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

// Worker ←→ IO thread communication
struct WorkerRecvTarget {
  uint8_t lanesTotal{1};
  uint64_t totalLen{0};
  bool discard{false}, toGpu{false};
  void* cpuBase{nullptr};
  std::vector<Segment> segs;
  std::shared_ptr<PinnedBuf> pinned;
};

enum class WorkerEventType : uint8_t {
  RECV_DONE = 0,
  EARLY_DATA = 1,
  SEND_CALLBACK = 2,
  CONN_ERROR = 3
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

// ---------------------------------------------------------------------------
// DataConnectionWorker — runs one thread per data connection
// ---------------------------------------------------------------------------
class DataConnectionWorker {
 public:
  DataConnectionWorker(int fd, EngineKey peer, PinnedStagingPool* staging);
  ~DataConnectionWorker();
  DataConnectionWorker(const DataConnectionWorker&) = delete;
  DataConnectionWorker& operator=(const DataConnectionWorker&) = delete;

  void Start();
  void Stop();
  int NotifyFd() const { return notifyFd_; }
  int Fd() const { return fd_; }

  void SubmitSend(SendItem item);
  void RegisterRecvTarget(TransferUniqueId opId, const WorkerRecvTarget& target);
  void RemoveRecvTarget(TransferUniqueId opId);
  void DrainEvents(std::deque<WorkerEvent>& out);

 private:
  void WakeWorker();
  void NotifyMain();
  void PostEvent(WorkerEvent ev);
  void Run();
  bool ProcessSend();
  bool ProcessRecv();
  bool RecvExact(uint8_t* dst, uint64_t len);
  bool RecvIntoSegments(uint8_t* base, const std::vector<Segment>& segs, uint64_t totalLen);
  bool DiscardPayload(uint64_t len);

  int fd_;
  EngineKey peerKey_;
  PinnedStagingPool* staging_;
  std::atomic<bool> running_{false};
  std::thread thread_;
  int notifyFd_{-1}, wakeFd_{-1};

  std::mutex sendMu_;
  std::deque<SendItem> sendQ_;

  std::mutex targetMu_;
  std::unordered_map<TransferUniqueId, WorkerRecvTarget> recvTargets_;

  std::mutex eventMu_;
  std::deque<WorkerEvent> eventQ_;

  uint8_t hdrBuf_[tcp::kDataHeaderSize]{};
  size_t hdrGot_{0};
};

// ---------------------------------------------------------------------------
// TcpTransport — main transport layer
// ---------------------------------------------------------------------------
class TcpTransport {
 public:
  TcpTransport(EngineKey myKey, const IOEngineConfig& engCfg, const TcpBackendConfig& cfg);
  ~TcpTransport();
  TcpTransport(const TcpTransport&) = delete;
  TcpTransport& operator=(const TcpTransport&) = delete;

  void Start();
  void Shutdown();
  std::optional<uint16_t> GetListenPort() const;

  void RegisterRemoteEngine(const EngineDesc& desc);
  void DeregisterRemoteEngine(const EngineDesc& desc);
  void RegisterMemory(const MemoryDesc& desc);
  void DeregisterMemory(const MemoryDesc& desc);

  bool PopInboundTransferStatus(const EngineKey& remote, TransferUniqueId id,
                                TransferStatus* status);

  void SubmitReadWrite(const MemoryDesc& local, size_t localOffset, const MemoryDesc& remote,
                       size_t remoteOffset, size_t size, TransferStatus* status,
                       TransferUniqueId id, bool isRead);
  void SubmitBatchReadWrite(const MemoryDesc& local, const SizeVec& localOffsets,
                            const MemoryDesc& remote, const SizeVec& remoteOffsets,
                            const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                            bool isRead);

 private:
  // Operation submission
  void EnqueueOp(std::unique_ptr<OutboundOpState> op);

  // Epoll helpers
  void AddEpoll(int fd, bool rd, bool wr);
  void ModEpoll(int fd, bool rd, bool wr);
  void DelEpoll(int fd);
  void CloseConnInternal(Connection* c);

  // Connection management
  void AssignConnToPeer(Connection* c);
  void MaybeDispatchQueuedOps(const EngineKey& peer);
  void EnsurePeerChannels(const EngineKey& peer);
  void ConnectChannel(const EngineKey& peer, tcp::Channel ch);
  void QueueHello(int fd);
  void AcceptNew();
  void DrainWakeFd();
  bool IsPeerReady(const EngineKey& peer);

  // Worker coordination
  void RegisterRecvTargetWithWorkers(const EngineKey& peer, TransferUniqueId opId,
                                     const WorkerRecvTarget& target);
  void RemoveRecvTargetFromWorkers(const EngineKey& peer, TransferUniqueId opId);

  // Data transfer
  void DispatchOp(std::unique_ptr<OutboundOpState> op);
  void QueueSend(int fd, std::vector<uint8_t> bytes, std::function<void()> onDone = nullptr);
  void QueueDataSend(const std::vector<DataConnectionWorker*>& workers, const MemoryDesc& src,
                     const std::vector<Segment>& srcSegs, uint64_t opId, uint8_t lanesTotal,
                     std::function<void()> onLaneDone = nullptr);

  // GPU memory transfers
  bool ScheduleGpuCopy(int deviceId, bool toDevice, const MemoryDesc& mem,
                       const std::vector<Segment>& segs, std::shared_ptr<PinnedBuf> pinned,
                       std::function<void()> onComplete);
  void PollGpuTasks();

  // Ctrl-connection I/O
  void UpdateWriteInterest(int fd);
  void HandleConnWritable(Connection* c);
  void FlushSend(Connection* c);

  // Peer lifecycle
  void CloseAndRemoveFd(int fd);
  EngineKey FindPeerByFd(int fd);
  void ClosePeerByFd(int fd);
  void ClosePeerByKey(const EngineKey& peer, const std::string& reason);
  void FailPendingOpsForPeer(const EngineKey& peer, const std::string& msg);

  // Ctrl message handling
  void HandleCtrlReadable(Connection* c);
  void HandleCtrlFrame(Connection* c, tcp::CtrlMsgType type, const uint8_t* body, size_t len);
  void HandleHello(Connection* c, const uint8_t* body, size_t len);
  void HandleRequest(const EngineKey& peer, tcp::CtrlMsgType type, const uint8_t* body, size_t len);
  void HandleCompletion(const EngineKey& peer, const uint8_t* body, size_t len);

  // Inbound / outbound state machines
  std::optional<MemoryDesc> LookupLocalMem(MemoryUniqueId id);
  void RecordInboundStatus(const EngineKey& peer, TransferUniqueId id, StatusCode code,
                           const std::string& msg);
  void SendCompletionAndRecord(const EngineKey& peer, TransferUniqueId opId, StatusCode code,
                               const std::string& msg);
  Connection* PeerCtrl(const EngineKey& peer);

  void FinalizeInboundWriteSetup(const EngineKey& peer, TransferUniqueId opId,
                                 InboundWriteState& ws);
  void MaybeFinalizeInboundWrite(const EngineKey& peer, TransferUniqueId opId);
  void TryConsumeEarlyWriteLanes(const EngineKey& peer, TransferUniqueId opId);
  void MaybeCompleteOutbound(OutboundOpState& st);

  // Worker event processing
  void ProcessEventsFrom(DataConnectionWorker* worker);
  void ProcessWorkerEvents();
  void HandleWorkerRecvDone(const WorkerEvent& ev);
  void HandleWorkerEarlyData(const WorkerEvent& ev);
  void ScanTimeouts();

  void IoLoop();

 private:
  EngineKey myEngKey_;
  IOEngineConfig engConfig_;
  TcpBackendConfig config_;

  int epfd_{-1}, listenFd_{-1}, wakeFd_{-1};
  uint16_t listenPort_{0};

  std::atomic<bool> running_{false};
  std::thread ioThread_;

  std::mutex submitMu_;
  std::deque<std::unique_ptr<OutboundOpState>> submitQ_;

  std::mutex remoteMu_;
  std::unordered_map<EngineKey, EngineDesc> remoteEngines_;

  std::mutex memMu_;
  std::unordered_map<MemoryUniqueId, MemoryDesc> localMems_;

  std::mutex inboundMu_;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, InboundStatusEntry>>
      inboundStatus_;

  std::unordered_map<int, std::unique_ptr<Connection>> conns_;
  std::unordered_map<EngineKey, PeerLinks> peers_;
  std::unordered_map<EngineKey, std::vector<std::unique_ptr<OutboundOpState>>> waitingOps_;
  std::unordered_map<TransferUniqueId, std::unique_ptr<OutboundOpState>> pendingOutbound_;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, InboundWriteState>>
      inboundWrites_;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, EarlyWriteState>> earlyWrites_;

  std::unordered_map<int, std::unique_ptr<DataConnectionWorker>> dataWorkers_;
  std::unordered_map<int, DataConnectionWorker*> workerNotifyMap_;

  PinnedStagingPool staging_;
  StreamPool streamPool_{8};
  EventPool eventPool_{64};
  std::deque<GpuTask> gpuTasks_;
};

}  // namespace io
}  // namespace mori

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
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/logging.hpp"
#include "src/io/tcp/protocol.hpp"
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
// Segment and lane helpers
// ---------------------------------------------------------------------------
struct Segment {
  uint64_t off{0};
  uint64_t len{0};
};

struct LaneSpan {
  uint64_t off{0};
  uint64_t len{0};
};

inline LaneSpan ComputeLaneSpan(uint64_t total, uint8_t lanes, uint8_t lane) {
  if (lanes <= 1) return {0, total};
  uint64_t base = total / lanes, rem = total % lanes;
  return {uint64_t(lane) * base + std::min<uint64_t>(lane, rem), base + (lane < rem ? 1 : 0)};
}

inline uint64_t SumLens(const std::vector<Segment>& segs) {
  uint64_t t = 0;
  for (auto& s : segs) t += s.len;
  return t;
}

inline bool TrySumLens(const std::vector<Segment>& segs, uint64_t* out) {
  uint64_t total = 0;
  for (const auto& seg : segs) {
    if (seg.len > std::numeric_limits<uint64_t>::max() - total) return false;
    total += seg.len;
  }
  *out = total;
  return true;
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
    if (s.off > memSize || s.len > memSize - s.off) return false;
  return true;
}

inline bool SegmentsOverlap(const std::vector<Segment>& segs) {
  std::vector<Segment> sorted = segs;
  std::sort(sorted.begin(), sorted.end(),
            [](const Segment& a, const Segment& b) { return a.off < b.off; });
  for (size_t i = 1; i < sorted.size(); ++i)
    if (sorted[i - 1].off + sorted[i - 1].len > sorted[i].off) return true;
  return false;
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
    while (p < v) {
      if (p > std::numeric_limits<size_t>::max() / 2) return v;
      p <<= 1;
    }
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
  bool assigned{false}, handedOff{false};
  tcp::Channel ch{tcp::Channel::CTRL};
  EngineKey peerKey;
  std::vector<uint8_t> inbuf;
  std::deque<SendItem> sendq;
};

class DataConnectionWorker;

struct RecvTargetKey {
  tcp::DataKind kind{tcp::DataKind::WRITE_PAYLOAD};
  TransferUniqueId opId{0};
  bool operator==(const RecvTargetKey& other) const {
    return kind == other.kind && opId == other.opId;
  }
};

struct RecvTargetKeyHash {
  size_t operator()(const RecvTargetKey& key) const {
    size_t seed = std::hash<TransferUniqueId>{}(key.opId);
    return seed ^ (std::hash<uint8_t>{}(static_cast<uint8_t>(key.kind)) + 0x9e3779b9 + (seed << 6) +
                   (seed >> 2));
  }
};

struct WorkerRecvTarget {
  uint8_t lanesTotal{1};
  uint64_t totalLen{0};
  bool discard{false}, toGpu{false};
  void* cpuBase{nullptr};
  std::vector<Segment> segs;
  std::shared_ptr<PinnedBuf> pinned;
};

struct PeerRecvTargets {
  std::mutex mu;
  std::unordered_map<RecvTargetKey, WorkerRecvTarget, RecvTargetKeyHash> entries;
};

struct PeerLinks {
  int ctrlFd{-1};
  std::vector<int> dataFds;
  std::vector<DataConnectionWorker*> workers;
  std::shared_ptr<PeerRecvTargets> recvTargets{std::make_shared<PeerRecvTargets>()};
  size_t nextWorker{0};
  Clock::time_point connectNotBefore{Clock::time_point::max()};
  int ctrlPending{0}, dataPending{0};
  bool CtrlUp() const { return ctrlFd >= 0; }
  bool DataUp() const { return !dataFds.empty(); }
};

struct InboundStatusEntry {
  StatusCode code{StatusCode::INIT};
  std::string msg;
  Clock::time_point created{Clock::now()};
};

struct OpKey {
  EngineKey peer;
  TransferUniqueId id{0};
  bool operator==(const OpKey& other) const { return peer == other.peer && id == other.id; }
  bool operator!=(const OpKey& other) const { return !(*this == other); }
};

struct OpKeyHash {
  size_t operator()(const OpKey& key) const {
    size_t seed = std::hash<EngineKey>{}(key.peer);
    return seed ^ (std::hash<TransferUniqueId>{}(key.id) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
  }
};

struct OutboundOpState {
  OpKey key;
  bool isRead{false};
  bool finished{false};
  TransferStatus* status{nullptr};
  MemoryDesc local{}, remote{};
  std::vector<Segment> localSegs, remoteSegs;
  uint64_t expectedRxBytes{0}, rxBytes{0};
  bool completionReceived{false}, gpuCopyPending{false};
  uint8_t lanesTotal{1};
  uint16_t lanesDoneMask{0};
  uint16_t lanesSentMask{0};
  StatusCode completionCode{StatusCode::SUCCESS};
  std::string completionMsg;
  std::shared_ptr<PinnedBuf> pinned;
  Clock::time_point deadline{Clock::time_point::max()};
};

struct InboundWriteState {
  EngineKey peer;
  TransferUniqueId id{0};
  MemoryDesc dst{};
  std::vector<Segment> dstSegs;
  bool discard{false};
  bool completionSent{false};
  uint8_t lanesTotal{1};
  uint16_t lanesDoneMask{0};
  std::shared_ptr<PinnedBuf> pinned;
  Clock::time_point deadline{Clock::time_point::max()};
};

struct InboundReadState {
  EngineKey peer;
  TransferUniqueId id{0};
  MemoryDesc src{};
  std::vector<Segment> srcSegs;
  uint8_t lanesTotal{1};
  Clock::time_point deadline{Clock::time_point::max()};
};

// Worker ←→ IO thread communication
enum class WorkerEventType : uint8_t {
  RECV_DONE = 0,
  AWAIT_TARGET = 1,
  SEND_CALLBACK = 2,
  CONN_ERROR = 3
};

struct WorkerEvent {
  WorkerEventType type{WorkerEventType::RECV_DONE};
  EngineKey peerKey;
  tcp::DataKind dataKind{tcp::DataKind::WRITE_PAYLOAD};
  TransferUniqueId opId{0};
  uint8_t lane{0};
  uint64_t laneLen{0};
  bool discarded{false};
  std::function<void()> callback;
  std::string errorMsg;
};

struct GpuTask {
  EngineKey peer;
  TransferUniqueId opId{0};
  int deviceId{-1};
  hipEvent_t ev{nullptr};
  std::shared_ptr<PinnedBuf> staging;
  std::function<void()> onReady;
};

enum class RecvState : uint8_t { ReadHeader, AwaitTarget, RecvPayload, DiscardPayload };

struct RecvCursor {
  RecvState state{RecvState::ReadHeader};
  std::array<uint8_t, tcp::kDataHeaderSize> hdr{};
  size_t hdrGot{0};
  tcp::ParsedDataHeader parsed{};
  WorkerRecvTarget activeTarget{};
  std::vector<Segment> laneSegs;
  uint64_t laneOffset{0};
  uint64_t payloadOff{0};
  uint64_t payloadRemaining{0};
  bool awaitNotified{false};
  size_t segIndex{0};
  uint64_t segOffset{0};
};

// ---------------------------------------------------------------------------
// DataConnectionWorker — runs one thread per data connection
// ---------------------------------------------------------------------------
class DataConnectionWorker {
 public:
  DataConnectionWorker(int fd, EngineKey peer, std::shared_ptr<PeerRecvTargets> recvTable,
                       std::vector<uint8_t> initialBytes = {});
  ~DataConnectionWorker();
  DataConnectionWorker(const DataConnectionWorker&) = delete;
  DataConnectionWorker& operator=(const DataConnectionWorker&) = delete;

  void Start();
  void Stop();
  int NotifyFd() const { return notifyFd_; }
  int Fd() const { return fd_; }

  void SubmitSend(SendItem item);
  void WakeWorker();
  void DrainEvents(std::deque<WorkerEvent>& out);

 private:
  void NotifyMain();
  void PostEvent(WorkerEvent ev);
  void Run();
  bool ProcessSend();
  bool ProcessRecv();
  ssize_t RecvSome(void* dst, size_t len);
  bool BeginPayload();
  bool ProcessPayload();
  void FinishPayload();

  int fd_;
  EngineKey peerKey_;
  std::shared_ptr<PeerRecvTargets> recvTable_;
  std::atomic<bool> running_{false};
  std::thread thread_;
  int notifyFd_{-1}, wakeFd_{-1};

  std::mutex sendMu_;
  std::deque<SendItem> sendQ_;

  std::mutex eventMu_;
  std::deque<WorkerEvent> eventQ_;
  std::vector<uint8_t> initialBytes_;
  size_t initialOffset_{0};
  RecvCursor recv_;
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
  enum class ConnResult : uint8_t { Alive, Gone };
  ConnResult AssignConnToPeer(Connection* c);
  void MaybeDispatchQueuedOps(const EngineKey& peer);
  void EnsurePeerChannels(const EngineKey& peer);
  bool ConnectChannel(const EngineKey& peer, tcp::Channel ch);
  void QueueHello(int fd);
  void AcceptNew();
  void DrainWakeFd();
  bool IsPeerReady(const EngineKey& peer);

  // Worker coordination
  bool RegisterRecvTarget(const EngineKey& peer, const RecvTargetKey& key,
                          const WorkerRecvTarget& target);
  void RemoveRecvTarget(const EngineKey& peer, const RecvTargetKey& key);
  bool HasRecvTarget(const EngineKey& peer, const RecvTargetKey& key);
  std::vector<DataConnectionWorker*> SelectWorkers(PeerLinks& links, uint8_t lanesTotal);

  // Data transfer
  void DispatchOp(std::unique_ptr<OutboundOpState> op);
  void QueueSend(int fd, std::vector<uint8_t> bytes, std::function<void()> onDone = nullptr);
  bool QueueDataSend(const EngineKey& peer, const std::vector<DataConnectionWorker*>& workers,
                     const MemoryDesc& src, const std::vector<Segment>& srcSegs, tcp::DataKind kind,
                     uint64_t opId, uint8_t lanesTotal, std::function<void()> onLaneDone = nullptr);

  // GPU memory transfers
  bool ScheduleGpuCopy(int deviceId, bool toDevice, const MemoryDesc& mem,
                       const std::vector<Segment>& segs, std::shared_ptr<PinnedBuf> pinned,
                       const EngineKey& peer, TransferUniqueId opId,
                       std::function<void()> onComplete);
  void PollGpuTasks();
  void DrainGpuTasksForPeer(const EngineKey& peer);

  // Ctrl-connection I/O
  void UpdateWriteInterest(int fd);
  void HandleConnWritable(Connection* c);
  void FlushSend(Connection* c);

  // Peer lifecycle
  void CloseAndRemoveFd(int fd);
  EngineKey FindPeerByFd(int fd);
  void ClosePeerByFd(int fd);
  void AbortPeer(EngineKey peer, StatusCode code, std::string reason);
  void FinalizeOutbound(OpKey key, StatusCode code, std::string msg);

  // Ctrl message handling
  ConnResult HandleCtrlReadable(Connection* c);
  ConnResult HandleCtrlFrame(Connection* c, tcp::CtrlMsgType type, const uint8_t* body, size_t len);
  ConnResult HandleHello(Connection* c, const uint8_t* body, size_t len);
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
  void DispatchInboundRead(InboundReadState read);
  void MaybeDispatchInboundReads(const EngineKey& peer);
  void MaybeCompleteOutbound(OutboundOpState& st);

  // Worker event processing
  void ProcessEventsFrom(DataConnectionWorker* worker);
  void ProcessWorkerEvents();
  void HandleWorkerRecvDone(const WorkerEvent& ev);
  void DrainSubmissions();
  void RetryWaitingConnections();
  void NoteAuxDeadline(Clock::time_point deadline);
  void RecomputeAuxDeadline();
  int ComputeEpollTimeoutMs();
  bool IsLiveDeadline(const OpKey& key, Clock::time_point deadline) const;
  void ScanTimeouts();
  void PruneInboundStatuses();
  void ShutdownDrain();

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
  std::unordered_map<OpKey, std::unique_ptr<OutboundOpState>, OpKeyHash> pendingOutbound_;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, InboundWriteState>>
      inboundWrites_;
  std::unordered_map<EngineKey, std::vector<InboundReadState>> pendingInboundReads_;
  std::unordered_map<EngineKey,
                     std::unordered_map<RecvTargetKey, Clock::time_point, RecvTargetKeyHash>>
      awaitingTargets_;
  std::deque<std::pair<Clock::time_point, OpKey>> deadlineDeque_;

  std::unordered_map<int, std::unique_ptr<DataConnectionWorker>> dataWorkers_;
  std::unordered_map<int, DataConnectionWorker*> workerNotifyMap_;

  PinnedStagingPool staging_;
  StreamPool streamPool_{8};
  EventPool eventPool_{64};
  std::deque<GpuTask> gpuTasks_;
  Clock::time_point nextAuxDeadline_{Clock::time_point::max()};
  Clock::time_point nextStatusPrune_{Clock::time_point::max()};
  std::unordered_set<int>* closedThisBatch_{nullptr};
};

}  // namespace io
}  // namespace mori

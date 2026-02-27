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

#include <sys/epoll.h>
#include <sys/eventfd.h>

#include <cassert>
#include <thread>

#include "mori/application/utils/check.hpp"
#include "mori/io/engine.hpp"
#include "src/io/tcp/data_worker.hpp"
#include "src/io/tcp/tcp_types.hpp"
#include "src/io/xgmi/hip_resource_pool.hpp"

namespace mori {
namespace io {

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
  void EnqueueOp(std::unique_ptr<OutboundOpState> op);

  void AddEpoll(int fd, bool wantRead, bool wantWrite);
  void ModEpoll(int fd, bool wantRead, bool wantWrite);
  void DelEpoll(int fd);
  void CloseConnInternal(Connection* c);

  bool PreferOutgoingFor(const EngineKey& peerKey) const;
  void AssignConnToPeer(Connection* c);
  void MaybeDispatchQueuedOps(const EngineKey& peerKey);
  void EnsurePeerChannels(const EngineKey& peerKey);
  void ConnectChannel(const EngineKey& peerKey, tcp::Channel ch);
  void QueueHello(int fd);
  void AcceptNew();
  void DrainWakeFd();
  bool IsPeerReady(const EngineKey& peerKey);

  void RegisterRecvTargetWithWorkers(const EngineKey& peerKey, TransferUniqueId opId,
                                     const WorkerRecvTarget& target);
  void RemoveRecvTargetFromWorkers(const EngineKey& peerKey, TransferUniqueId opId);

  void DispatchOp(std::unique_ptr<OutboundOpState> op);
  void QueueSend(int fd, std::vector<uint8_t> bytes, std::function<void()> onDone = nullptr);

  void QueueSegmentSend(DataConnectionWorker* worker, uint64_t wireOpId, uint8_t* base,
                        const std::vector<Segment>& segs, uint64_t totalLen,
                        std::function<void()> onDone = nullptr);
  void QueueStripedCpuSend(const std::vector<DataConnectionWorker*>& workers, uint64_t opId,
                           uint8_t lanesTotal, uint8_t* base, uint64_t baseOff, uint64_t total,
                           std::function<void()> onLaneDone = nullptr);
  void QueueGpuSend(const std::vector<DataConnectionWorker*>& workers, uint64_t opId,
                    uint8_t lanesTotal, const MemoryDesc& src, const std::vector<Segment>& srcSegs,
                    std::function<void()> onLaneDone = nullptr);

  void QueueDataSendForWrite(const std::vector<DataConnectionWorker*>& workerList,
                             OutboundOpState& st);
  void QueueDataSendForRead(const EngineKey& peer, uint64_t opId, const MemoryDesc& src,
                            const std::vector<Segment>& srcSegs, uint8_t lanesTotal);
  void QueueDataSendCommon(const std::vector<DataConnectionWorker*>& workerList,
                           const MemoryDesc& src, const std::vector<Segment>& srcSegs,
                           uint64_t opId, uint8_t lanesTotal,
                           std::function<void()> onLaneDone = nullptr);

  bool ScheduleGpuDtoH(int deviceId, const MemoryDesc& src, const std::vector<Segment>& srcSegs,
                       std::shared_ptr<PinnedBuf> pinned, std::function<void()> onComplete);
  bool ScheduleGpuHtoD(int deviceId, const MemoryDesc& dst, const std::vector<Segment>& dstSegs,
                       std::shared_ptr<PinnedBuf> pinned, std::function<void()> onComplete);

  void PollGpuTasks();
  void UpdateWriteInterest(int fd);
  void HandleConnWritable(Connection* c);
  void FlushSend(Connection* c);

  void CloseAndRemoveFd(int fd);
  EngineKey FindPeerByFd(int fd);
  void ClosePeerByFd(int fd);
  void ClosePeerByKey(const EngineKey& peer, const std::string& reason);
  void FailPendingOpsForPeer(const EngineKey& peer, const std::string& msg);

  void HandleCtrlReadable(Connection* c);
  void HandleCtrlFrame(Connection* c, tcp::CtrlMsgType type, const uint8_t* body, size_t len);
  void HandleHello(Connection* c, const uint8_t* body, size_t len);

  std::optional<MemoryDesc> LookupLocalMem(MemoryUniqueId id);
  void RecordInboundStatus(const EngineKey& peer, TransferUniqueId id, StatusCode code,
                           const std::string& msg);
  void SendCompletionAndRecord(const EngineKey& peer, TransferUniqueId opId, StatusCode code,
                               const std::string& msg);
  Connection* PeerCtrl(const EngineKey& peer);

  void FinalizeInboundWriteSetup(const EngineKey& peer, TransferUniqueId opId,
                                 InboundWriteState& ws);
  void SetupInboundWriteWorkerTarget(const EngineKey& peer, TransferUniqueId opId,
                                     const InboundWriteState& ws);
  void HandleWriteReq(const EngineKey& peer, const uint8_t* body, size_t len);
  void HandleBatchWriteReq(const EngineKey& peer, const uint8_t* body, size_t len);
  void HandleWriteReqSegments(const EngineKey& peer, TransferUniqueId opId, MemoryUniqueId memId,
                              std::vector<Segment> segs, uint8_t lanesTotal);
  void HandleReadReq(const EngineKey& peer, const uint8_t* body, size_t len);
  void HandleBatchReadReq(const EngineKey& peer, const uint8_t* body, size_t len);
  void HandleReadReqSegments(const EngineKey& peer, TransferUniqueId opId, MemoryUniqueId memId,
                             std::vector<Segment> segs, uint8_t lanesTotal, bool batchReq);
  void HandleCompletion(const EngineKey& peer, const uint8_t* body, size_t len);

  void MaybeFinalizeInboundWrite(const EngineKey& peer, TransferUniqueId opId);
  void TryConsumeEarlyWriteLanes(const EngineKey& peer, TransferUniqueId opId);

  void MaybeCompleteOutbound(OutboundOpState& st);
  void ProcessEventsFrom(DataConnectionWorker* worker);
  void ProcessWorkerEvents();
  void HandleWorkerRecvDone(const WorkerEvent& ev);
  void HandleWorkerEarlyData(const WorkerEvent& ev);
  void ScanTimeouts();

  void IoLoop();

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
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, InboundStatusEntry>>
      inboundStatus;

  std::unordered_map<int, std::unique_ptr<Connection>> conns;
  std::unordered_map<EngineKey, PeerLinks> peers;
  std::unordered_map<EngineKey, std::vector<std::unique_ptr<OutboundOpState>>> waitingOps;
  std::unordered_map<TransferUniqueId, std::unique_ptr<OutboundOpState>> pendingOutbound;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, InboundWriteState>>
      inboundWrites;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, EarlyWriteState>> earlyWrites;

  std::unordered_map<int, std::unique_ptr<DataConnectionWorker>> dataWorkers;
  std::unordered_map<int, DataConnectionWorker*> workerNotifyMap;

  PinnedStagingPool staging;
  StreamPool streamPool{8};
  EventPool eventPool{64};
  std::deque<GpuTask> gpuTasks;
};

}  // namespace io
}  // namespace mori

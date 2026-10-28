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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"

namespace mori {
namespace io {

// Hardware-free stand-ins for the RDMA / TCP / XGMI backends (MORI_IO_SIM_ENABLE=1).
// Registration, routing, sessions, the TCP control plane and notifications run
// for real; only the data movement is replaced by a modelled delay.
namespace sim {

bool IsEnabled();
// The type a backend request is simulated as; MORI_IO_SIM_BACKEND remaps it.
BackendType ResolveBackendType(BackendType requested);
std::string BackendTypeName(BackendType type);

struct LinkProfile {
  double bytesPerNs{1.0};  // GB/s
  std::chrono::nanoseconds latency{0};
  std::chrono::nanoseconds perSegment{0};
  std::chrono::nanoseconds perOp{0};  // link occupancy per transfer call
};

LinkProfile LoadLinkProfile(BackendType type);

}  // namespace sim

class SimBackend;

class SimBackendSession : public BackendSession {
 public:
  SimBackendSession(SimBackend* backend, const MemoryDesc& local, const MemoryDesc& remote)
      : backend_(backend), local_(local), remote_(remote) {}

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead) override;
  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead) override;
  bool Alive() const override;

 private:
  SimBackend* backend_;
  MemoryDesc local_;
  MemoryDesc remote_;
};

class SimBackend : public Backend {
 public:
  SimBackend(BackendType type, const EngineDesc& self, const IOEngineConfig& engConfig,
             bool enableNotification);
  ~SimBackend() override;

  BackendType Type() const { return type_; }
  std::optional<uint16_t> GetListenPort() const;

  void RegisterRemoteEngine(const EngineDesc&) override;
  void DeregisterRemoteEngine(const EngineDesc&) override;
  void RegisterMemory(MemoryDesc& desc) override;
  void DeregisterMemory(const MemoryDesc& desc) override;
  void ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                 size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                 bool isRead) override;
  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead) override;
  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) override;
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                TransferStatus* status) override;
  bool PopPeerFailure(PeerFailureEvent* out) override;
  bool CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const override;

  void Submit(const MemoryDesc& local, const SizeVec& localOffsets, const MemoryDesc& remote,
              const SizeVec& remoteOffsets, const SizeVec& sizes, TransferStatus* status,
              TransferUniqueId id, bool isRead);
  bool PeerAlive(const EngineKey& remote) const;

 private:
  struct Peer {
    EngineDesc desc;
    std::mutex mu;  // serializes this peer's control-plane socket
    std::optional<application::TCPEndpointHandle> ep;
    std::atomic<bool> dead{false};
    std::unordered_map<MemoryUniqueId, size_t> verifiedMems;
  };

  struct Completion {
    std::chrono::steady_clock::time_point deadline;
    uint64_t seq;
    TransferStatus* status;
    TransferUniqueId id;
    std::shared_ptr<Peer> peer;  // null for XGMI, which has no control plane
    bool operator>(const Completion& o) const {
      return deadline != o.deadline ? deadline > o.deadline : seq > o.seq;
    }
  };

  bool HasControlPlane() const { return type_ != BackendType::XGMI; }
  bool IsSameNode(const EngineKey& remote) const;
  std::shared_ptr<Peer> GetPeer(const EngineKey& remote) const;
  // Connects and checks the remote memory once, as RdmaBackend does when it builds QPs and MRs.
  void EnsurePeerReady(const std::shared_ptr<Peer>& peer, const MemoryDesc& remote);
  bool PeerSocketClosedLocked(Peer& peer);
  void MarkPeerDead(Peer& peer, const std::string& detail);
  std::chrono::steady_clock::time_point ReserveLink(bool isRead, size_t bytes, size_t segments);

  void TimerLoop();
  void Complete(Completion& c);

  void ServerLoop();
  bool HandleServerMessage(int fd);
  void DropServerConn(int fd) noexcept;

 private:
  BackendType type_;
  EngineDesc self_;
  bool enableNotification_;
  sim::LinkProfile profile_;
  double timeScale_{1.0};

  mutable std::mutex peersMu_;
  std::unordered_map<EngineKey, std::shared_ptr<Peer>> peers_;

  std::mutex memsMu_;
  std::unordered_map<MemoryUniqueId, MemoryDesc> mems_;

  // Each direction of the link is a FIFO pipe: transfers queue behind each other.
  std::mutex linkMu_;
  std::chrono::steady_clock::time_point txFreeAt_{};
  std::chrono::steady_clock::time_point rxFreeAt_{};

  std::mutex timerMu_;
  std::condition_variable timerCv_;
  std::priority_queue<Completion, std::vector<Completion>, std::greater<Completion>> timerQ_;
  uint64_t timerSeq_{0};
  bool stopping_{false};
  std::thread timerThd_;

  std::mutex notifMu_;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, int>> notifPool_;

  std::mutex failuresMu_;
  std::deque<PeerFailureEvent> failures_;

  std::unique_ptr<application::TCPContext> ctx_;
  int epfd_{-1};
  std::atomic<bool> serverRunning_{false};
  std::unordered_map<int, application::TCPEndpointHandle> serverConns_;
  std::thread serverThd_;
};

}  // namespace io
}  // namespace mori

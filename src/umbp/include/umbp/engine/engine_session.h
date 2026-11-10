// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/engine/engine_control.h"

namespace umbp::engine {
class EngineFrame;
}

namespace mori::umbp::engine {

#ifdef MORI_UMBP_TESTING
std::vector<PeerVmmRegion> TakePeerRegionBatchForTest(std::vector<PeerVmmRegion>* pending);
#endif

// One rank's ordered node-side state. Handle() never waits for a transfer or a
// retire to become ready; it only accepts one frame into the existing PeerPool
// serialization domain and returns an immediate response.
class EngineSession {
 public:
  using Clock = std::chrono::steady_clock;
  using NowFn = std::function<Clock::time_point()>;

  explicit EngineSession(EngineControl* control, NowFn now = Clock::now,
                         std::array<uint8_t, 16> node_boot_id = {});
  ~EngineSession();

  bool Handle(const ::umbp::engine::EngineFrame& request, const std::vector<int>& request_fds,
              ::umbp::engine::EngineFrame* response, std::vector<int>* response_fds,
              std::string* error);
  bool Expired() const;
  bool registered() const { return registered_; }
  RankId rank() const { return rank_; }
  uint64_t last_received_seq() const { return last_received_seq_; }
  void Fail();
  bool PopWorkFrame(::umbp::engine::EngineFrame* response);
  int WorkEventFd() const;
  void FailPushedWork(uint64_t transfer_id);

 private:
  EngineControl* control_ = nullptr;
  NowFn now_;
  Clock::time_point last_activity_;
  std::chrono::milliseconds heartbeat_timeout_{5000};
  std::array<uint8_t, 16> session_id_{};
  PoolEpoch epoch_;
  RankId rank_ = 0;
  uint32_t backend_id_ = 0;
  std::string executor_id_;
  uint64_t last_received_seq_ = 0;
  bool registered_ = false;
  bool failed_ = false;
  std::array<uint8_t, 16> node_boot_id_{};
  std::unordered_map<uint64_t, std::vector<PeerVmmRegion>> pending_peer_regions_;
  std::unordered_set<std::string> sent_peer_regions_;
};

// Dedicated full-duplex UDS listener used only when engine_hbm_ranks is
// configured. Payloads are protobuf EngineFrame values and all descriptors are
// SCM_RIGHTS attachments to the same ordered frame.
class EngineSessionServer {
 public:
  explicit EngineSessionServer(EngineControl* control);
  ~EngineSessionServer();

  bool Start(const std::string& path, std::string* error = nullptr);
  void Shutdown();
  const std::string& path() const { return path_; }

 private:
  void AcceptLoop();
  void Serve(int fd);

  EngineControl* control_ = nullptr;
  std::string path_;
  int listen_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::thread accept_thread_;
  std::mutex clients_mutex_;
  std::condition_variable clients_cv_;
  std::vector<int> clients_;
  std::array<uint8_t, 16> node_boot_id_{};
};

}  // namespace mori::umbp::engine

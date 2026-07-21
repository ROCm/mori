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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "umbp/umbp_client.h"

namespace mori::umbp::standalone {

// Lightweight Master identity used only for external-KV reports on behalf of
// one worker attached to a distributed-backed standalone server. It never
// reports storage capacities or owned-KV event bundles.
class ExternalKvIdentityClient {
 public:
  struct Config {
    std::string master_address;
    std::string node_id;
    std::string node_address;
    std::string peer_address;
    std::vector<uint8_t> engine_desc_bytes;
    std::vector<std::string> tags;
  };

  explicit ExternalKvIdentityClient(Config config);
  ~ExternalKvIdentityClient();

  ExternalKvIdentityClient(const ExternalKvIdentityClient&) = delete;
  ExternalKvIdentityClient& operator=(const ExternalKvIdentityClient&) = delete;

  bool Start();
  void Stop();
  const std::string& node_id() const { return config_.node_id; }

  bool ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier);
  bool RevokeExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier);
  bool RevokeAllExternalKvBlocksAtTier(TierType tier);
  std::vector<IUMBPClient::ExternalKvMatch> MatchExternalKv(const std::vector<std::string>& hashes,
                                                            bool count_as_hit = false);
  std::vector<IUMBPClient::ExternalKvHitCountEntry> GetExternalKvHitCounts(
      const std::vector<std::string>& hashes);

 private:
  bool RegisterLocked();
  void UnregisterLocked();
  bool SendHeartbeatOnceLocked();
  void HeartbeatLoop();

  Config config_;
  std::shared_ptr<void> channel_;
  std::unique_ptr<void, void (*)(void*)> stub_;

  std::mutex rpc_mu_;
  std::mutex cv_mu_;
  std::condition_variable cv_;
  std::thread heartbeat_thread_;
  std::atomic<bool> running_{false};
  bool registered_ = false;
  uint64_t heartbeat_interval_ms_ = 1000;
};

}  // namespace mori::umbp::standalone

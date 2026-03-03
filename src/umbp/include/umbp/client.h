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
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "umbp/types.h"

namespace grpc_impl {
class Channel;
}

namespace mori::umbp {

struct UMBPClientConfig {
  std::string master_address;
  std::string client_id;
  std::string node_address;
  bool auto_heartbeat = true;
};

class UMBPClient {
 public:
  explicit UMBPClient(const UMBPClientConfig& config);
  ~UMBPClient();

  UMBPClient(const UMBPClient&) = delete;
  UMBPClient& operator=(const UMBPClient&) = delete;

  // --- Client lifecycle ---
  // Register with master. If auto_heartbeat, starts heartbeat thread.
  void RegisterSelf(const std::map<TierType, TierCapacity>& tier_capacities);
  void UnregisterSelf();

  // --- Heartbeat ---
  void StartHeartbeat();
  void StopHeartbeat();

  bool IsRegistered() const { return registered_; }

 private:
  UMBPClientConfig config_;

  std::shared_ptr<grpc_impl::Channel> channel_;
  // Use void* to avoid exposing generated stub type in header.
  // Cast to UMBPMaster::Stub* in the .cpp file.
  std::unique_ptr<void, void (*)(void*)> stub_;

  std::thread heartbeat_thread_;
  std::atomic<bool> heartbeat_running_{false};
  std::atomic<bool> registered_{false};
  uint64_t heartbeat_interval_ms_ = 5000;

  std::mutex hb_cv_mutex_;
  std::condition_variable hb_cv_;

  // Cached tier capacities for heartbeat reporting
  std::mutex caps_mutex_;
  std::map<TierType, TierCapacity> current_capacities_;

  void HeartbeatLoop();
};

}  // namespace mori::umbp

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

#include <grpcpp/support/status.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/common/types.h"
#include "umbp/distributed/routing/route_put_strategy.h"

namespace grpc_impl {
class Channel;
}

namespace mori::umbp {

struct RouteGetResult {
  Location location;
  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;
  std::vector<uint8_t> dram_memory_desc_bytes;
};

class MasterClient {
 public:
  explicit MasterClient(const MasterClientConfig& config);
  ~MasterClient();

  MasterClient(const MasterClient&) = delete;
  MasterClient& operator=(const MasterClient&) = delete;

  // --- Client lifecycle ---
  // Register with master. If auto_heartbeat, starts heartbeat thread.
  grpc::Status RegisterSelf(
      const std::map<TierType, TierCapacity>& tier_capacities, const std::string& peer_address = "",
      const std::vector<uint8_t>& engine_desc_bytes = {},
      const std::vector<std::vector<uint8_t>>& dram_memory_desc_bytes_list = {},
      const std::vector<uint64_t>& dram_buffer_sizes = {},
      const std::vector<uint64_t>& ssd_store_capacities = {});
  grpc::Status UnregisterSelf();

  // --- Block index ---
  // Register a block key owned by this node in the master index.
  grpc::Status Register(const std::string& key, const Location& location);
  // Unregister a block key location owned by this node.
  // If removed is non-null, returns 1 when removed, otherwise 0.
  grpc::Status Unregister(const std::string& key, const Location& location,
                          uint32_t* removed = nullptr);
  grpc::Status FinalizeAllocation(const std::string& key, const Location& location,
                                  const std::string& allocation_id);
  grpc::Status PublishLocalBlock(const std::string& key, const Location& location);
  grpc::Status AbortAllocation(const std::string& node_id, const std::string& allocation_id,
                               uint64_t size);

  // --- Router ---
  /// Pick an existing replica to read from.
  /// Returns the Location via @p out_location (if found).
  grpc::Status RouteGet(const std::string& key, std::optional<RouteGetResult>* out_result);

  /// Pick a target node to write to.
  /// After receiving the result, write via MORI-IO, then call FinalizeAllocation()
  /// or AbortAllocation().
  grpc::Status RoutePut(const std::string& key, uint64_t block_size,
                        std::optional<RoutePutResult>* out_result);

  // --- Heartbeat ---
  void StartHeartbeat();
  void StopHeartbeat();

  bool IsRegistered() const { return registered_; }

 private:
  MasterClientConfig config_;

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

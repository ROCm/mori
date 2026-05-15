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

#include <grpcpp/channel.h>
#include <grpcpp/support/status.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/routing/route_put_strategy.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class PeerDramAllocator;

inline constexpr std::size_t kMasterClientMaxPendingHistograms = 15000;

// Result of RouteGet — pure routing advisory.  The reader follows up
// with peer.ResolveKey to fetch pages/descs/page_size.  `size` is
// carried so the reader can preflight its destination buffer without
// a separate round trip.
struct RouteGetResult {
  std::string node_id;
  TierType tier = TierType::UNKNOWN;
  uint64_t size = 0;
  std::string peer_address;
};

class MasterClient {
 public:
  using Labels = std::vector<std::pair<std::string, std::string>>;

  explicit MasterClient(const UMBPMasterClientConfig& config);
  ~MasterClient();

  MasterClient(const MasterClient&) = delete;
  MasterClient& operator=(const MasterClient&) = delete;

  // --- Client lifecycle ---

  // Register with master.  In the master-as-advisor design only
  // membership + capacity-snapshot metadata is shipped — DRAM/HBM
  // descriptors are peer-internal now.
  grpc::Status RegisterSelf(const std::map<TierType, TierCapacity>& tier_capacities,
                            const std::string& peer_address = "",
                            const std::vector<uint8_t>& engine_desc_bytes = {},
                            const std::vector<uint64_t>& ssd_store_capacities = {});
  grpc::Status UnregisterSelf();

  // --- Router ---

  // Pick a target node for a Put.  `exclude_nodes` lets the writer
  // steer master past nodes that already returned ENOSPC at peer
  // level.
  grpc::Status RoutePut(const std::string& key, uint64_t block_size,
                        const std::unordered_set<std::string>& exclude_nodes,
                        std::optional<RoutePutResult>* out_result);

  // Pick a replica for a Get.  Same exclude semantics as RoutePut,
  // used to retry past peers that report `found=false` on Resolve.
  grpc::Status RouteGet(const std::string& key,
                        const std::unordered_set<std::string>& exclude_nodes,
                        std::optional<RouteGetResult>* out_result);

  // --- Batch RPCs ---
  grpc::Status BatchRoutePut(const std::vector<std::string>& keys,
                             const std::vector<uint64_t>& block_sizes,
                             const std::unordered_set<std::string>& exclude_nodes,
                             std::vector<std::optional<RoutePutResult>>* out);
  grpc::Status BatchRouteGet(const std::vector<std::string>& keys,
                             const std::unordered_set<std::string>& exclude_nodes,
                             std::vector<std::optional<RouteGetResult>>* out);

  // --- Heartbeat (event-driven) ---
  // Bind a PeerDramAllocator whose outbox the heartbeat thread will
  // drain.  Pass nullptr for SSD-only peers.  Must be set before
  // StartHeartbeat() — the heartbeat thread reads it once per tick.
  void SetPeerDramAllocator(PeerDramAllocator* alloc);

  void StartHeartbeat();
  void StopHeartbeat();

  // --- Client-side metrics ---
  void AddCounter(std::string name, std::string help, Labels labels, double delta);
  void SetGauge(std::string name, std::string help, Labels labels, double value);
  void Observe(std::string name, std::string help, Labels labels, const std::vector<double>& bounds,
               double value);

  bool IsRegistered() const { return registered_; }

  // --- External KV block events ---
  grpc::Status ReportExternalKvBlocks(const std::string& node_id,
                                      const std::vector<std::string>& hashes, TierType tier);
  // Revoke specific hashes from a single tier on this node.
  grpc::Status RevokeExternalKvBlocks(const std::string& node_id,
                                      const std::vector<std::string>& hashes, TierType tier);
  // Bulk: revoke every hash registered by this node at the given tier.
  grpc::Status RevokeAllExternalKvBlocksAtTier(const std::string& node_id, TierType tier);

  struct ExternalKvNodeMatch {
    std::string node_id;
    std::string peer_address;
    // Matched hashes grouped by every tier they currently live on for this
    // node.  A single hash MAY appear in multiple tier buckets when the
    // node holds physical copies on more than one tier (e.g. write_through
    // created a CPU mirror while the GPU copy is still alive).  std::map
    // keys iterate in sorted TierType order, so the first non-empty bucket
    // is the fastest tier currently available on this node.
    std::map<TierType, std::vector<std::string>> hashes_by_tier;

    // Number of *distinct* matched hashes (size of the union across tiers).
    // A hash present on HBM+DRAM still counts once.
    size_t MatchedHashCount() const {
      std::unordered_set<std::string_view> seen;
      for (const auto& [tier, hashes] : hashes_by_tier) {
        for (const auto& h : hashes) seen.insert(h);
      }
      return seen.size();
    }
  };
  grpc::Status MatchExternalKv(const std::vector<std::string>& hashes,
                               std::vector<ExternalKvNodeMatch>* out_matches);

 private:
  UMBPMasterClientConfig config_;

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<void, void (*)(void*)> stub_;

  std::thread heartbeat_thread_;
  std::atomic<bool> heartbeat_running_{false};
  std::atomic<bool> registered_{false};
  uint64_t heartbeat_interval_ms_ = 5000;

  std::mutex hb_cv_mutex_;
  std::condition_variable hb_cv_;

  // Cached tier capacities — heartbeat reports the latest peer
  // allocator snapshot when the bound PeerDramAllocator is non-null,
  // else falls back to whatever was last set here.
  std::mutex caps_mutex_;
  std::map<TierType, TierCapacity> current_capacities_;

  // Peer-event source for the heartbeat thread.  Non-owning; lifetime
  // is managed by PoolClient.
  PeerDramAllocator* peer_alloc_ = nullptr;

  // Heartbeat seq state — the wire protocol's gap-recovery channel.
  uint64_t hb_seq_ = 0;
  uint64_t hb_last_acked_seq_ = 0;

  void HeartbeatLoop();

  // --- Metrics buffering ---
  struct PendingSample {
    std::string name;
    std::string help;
    Labels labels;
    double value = 0.0;
  };
  // Per-series histogram state.  bucket_counts is CUMULATIVE
  // (bucket_counts[i] = #observations with value <= bounds[i]) so the master
  // can merge by per-bucket addition without any encoding conversion.
  // warned_mismatch dedups the "first-write-wins" WARN per series, so a
  // single misconfigured caller does not silence every other series' WARN.
  struct HistogramAccumulator {
    std::string name;
    std::string help;
    Labels labels;
    std::vector<double> bounds;
    std::vector<uint64_t> bucket_counts;
    uint64_t count = 0;
    double sum = 0.0;
    bool warned_mismatch = false;
  };

  std::mutex metrics_mutex_;
  std::unordered_map<std::string, PendingSample> pending_counters_;
  std::unordered_map<std::string, PendingSample> pending_gauges_;
  std::unordered_map<std::string, HistogramAccumulator> pending_histogram_aggregates_;
  std::atomic<uint64_t> metrics_dropped_count_{0};

  // Non-const so a friend test can shrink the cap to exercise the cold drop
  // path without env vars or config plumbing.  Production reads the constant
  // default; tests that want to verify the cap install a small override.
  std::size_t pending_histogram_series_cap_ = kMasterClientMaxPendingHistograms;

  uint64_t metrics_interval_ms_ = 1000;

  std::thread metrics_thread_;
  std::atomic<bool> metrics_running_{false};
  std::mutex metrics_cv_mutex_;
  std::condition_variable metrics_cv_;

  void StartMetricsReporting();
  void StopMetricsReporting();
  void MetricsLoop();

  // --- ScopedRpcTimer integration ---
  // Called by ScopedRpcTimer at the end of every monitored MasterClient RPC.
  // Both methods short-circuit when metrics_running_ is false to avoid
  // unbounded buffer growth on never-registered (Python read-only) clients
  // and during destructor windows.
  friend class ScopedRpcTimer;
  void RecordRpcLatency(std::string_view method, bool ok, double seconds);
  void RecordRpcError(std::string_view method, std::string_view code);

  // Test-only access: lets the cap-exercise test in
  // tests/cpp/umbp/distributed/test_master_client_rpc_latency.cpp shrink
  // pending_histogram_series_cap_ and inspect pending_histogram_aggregates_
  // / metrics_dropped_count_.  Production code never touches these fields
  // through this friend.
  friend class MasterClientRpcLatencyTest;
};

}  // namespace mori::umbp

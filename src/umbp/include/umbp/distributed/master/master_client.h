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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/routing/route_put_strategy.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Buckets for mori_umbp_master_client_rpc_latency_seconds.  Spans 0.1 ms ~ 5 s
// (14 finite bounds + implicit +Inf).  Bucket layout is locked on the first
// observation client-side and again on the master side; the layout is
// silently first-write-wins, so editing in place corrupts existing series.
// To change buckets bump the metric name suffix (e.g. _v2) instead.
inline constexpr double kMasterClientRpcLatencyBucketsArr[] = {
    1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1.0, 2.5, 5.0,
};

// Series-cardinality cap for MasterClient::pending_histogram_aggregates_.
// With client-side bucket aggregation the map is keyed by (name, labels) so
// its size is bounded by the number of distinct series, not by QPS.  Healthy
// operation with the current label set peaks at ~50 series; reaching this cap
// indicates a label-cardinality leak (e.g. a per-key/per-allocation_id label
// was accidentally introduced).  Excess series are dropped at insert time and
// accounted in metrics_dropped_count_.
inline constexpr std::size_t kMasterClientMaxPendingHistograms = 15000;

struct RouteGetResult {
  Location location;
  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;

  // DRAM/HBM only; empty for SSD tier.
  std::vector<BufferMemoryDescBytes> dram_memory_descs;
  uint64_t page_size = 0;

  // Structured page set populated by Master (parallel to RoutePutResult.pages).
  // Master parses location.location_id and sends the result here so the Client
  // can build scatter-gather RDMA descriptors without re-parsing the string.
  std::vector<PageLocation> pages;
};

class MasterClient {
 public:
  using Labels = std::vector<std::pair<std::string, std::string>>;

  explicit MasterClient(const UMBPMasterClientConfig& config);
  ~MasterClient();

  MasterClient(const MasterClient&) = delete;
  MasterClient& operator=(const MasterClient&) = delete;

  // --- Client lifecycle ---
  // Register with master. If auto_heartbeat, starts heartbeat thread.
  // `dram_page_size` is the page size the Client wants Master's
  // PageBitmapAllocator to use for this node's DRAM/HBM tier (per Q6, the
  // same value applies to both tiers).  0 means "use Master's
  // ClientRegistryConfig.default_dram_page_size".  Set from
  // PoolClientConfig.dram_page_size at the call site.
  grpc::Status RegisterSelf(
      const std::map<TierType, TierCapacity>& tier_capacities, const std::string& peer_address = "",
      const std::vector<uint8_t>& engine_desc_bytes = {},
      const std::vector<std::vector<uint8_t>>& dram_memory_desc_bytes_list = {},
      const std::vector<uint64_t>& dram_buffer_sizes = {},
      const std::vector<uint64_t>& ssd_store_capacities = {}, uint64_t dram_page_size = 0);
  grpc::Status UnregisterSelf();

  // --- Block index ---
  // Register a block key owned by this node in the master index.
  grpc::Status Register(const std::string& key, const Location& location);
  // Unregister a block key location owned by this node.
  // If removed is non-null, returns 1 when removed, otherwise 0.
  grpc::Status Unregister(const std::string& key, const Location& location,
                          uint32_t* removed = nullptr);
  // Read-only existence check (no access count side-effects).
  grpc::Status Lookup(const std::string& key, bool* found);
  grpc::Status FinalizeAllocation(const std::string& key, const Location& location,
                                  const std::string& allocation_id, int32_t depth = -1);
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

  // --- Batch RPCs ---
  grpc::Status BatchRoutePut(const std::vector<std::string>& keys,
                             const std::vector<uint64_t>& block_sizes,
                             std::vector<std::optional<RoutePutResult>>* out);
  grpc::Status BatchRouteGet(const std::vector<std::string>& keys,
                             std::vector<std::optional<RouteGetResult>>* out);
  grpc::Status BatchFinalizeAllocation(const std::vector<std::string>& keys,
                                       const std::vector<Location>& locations,
                                       const std::vector<std::string>& allocation_ids,
                                       std::vector<bool>* out,
                                       const std::vector<int32_t>& depths = {});
  // Read-only batch existence check (no access-count / lease side-effects).
  // `out` is cleared on entry; on wire error it remains empty and the
  // returned Status carries the failure.  On success, `*out` is resized to
  // keys.size() and populated parallel to keys.
  grpc::Status BatchLookup(const std::vector<std::string>& keys, std::vector<bool>* out);

  // One entry of a BatchAbortAllocation request.  `node_id` is the target
  // node owning the pending lease (not the caller); matches the single-item
  // AbortAllocation semantics.
  struct BatchAbortEntry {
    std::string node_id;
    std::string allocation_id;
    uint64_t size;
  };
  // Batched rollback of pending allocations.  Semantics align with
  // BatchFinalizeAllocation: wire success returns Status::OK and fills
  // *out parallel to `entries`; per-entry false is not an error (racy
  // reap / double-abort / EXPIRED node).  Callers use it as best-effort
  // cleanup; TTL reaper covers the rest.
  grpc::Status BatchAbortAllocation(const std::vector<BatchAbortEntry>& entries,
                                    std::vector<bool>* out);

  // --- Heartbeat ---
  void StartHeartbeat();
  void StopHeartbeat();

  // --- Client-side metrics ---
  // Buffer a counter delta. Flushed to master via ReportMetrics RPC periodically.
  void AddCounter(std::string name, std::string help, Labels labels, double delta);
  // Buffer a gauge value (last write wins between flushes).
  void SetGauge(std::string name, std::string help, Labels labels, double value);
  // Buffer a histogram observation. `bounds` is taken by const-ref so the
  // 14-double rpc-latency layout (and the 3072-double bandwidth layout in
  // pool_client.cpp) avoids a per-call heap allocation.  The observation is
  // accumulated into a per-series HistogramAccumulator; only one MetricSample
  // per (name, labels) series is sent on the wire per flush.
  void Observe(std::string name, std::string help, Labels labels, const std::vector<double>& bounds,
               double value);

  bool IsRegistered() const { return registered_; }

  // --- External KV block events ---
  grpc::Status ReportExternalKvBlocks(const std::string& node_id,
                                      const std::vector<std::string>& hashes, TierType tier);
  grpc::Status RevokeExternalKvBlocks(const std::string& node_id,
                                      const std::vector<std::string>& hashes);

  struct ExternalKvNodeMatch {
    std::string node_id;
    std::string peer_address;
    std::vector<std::string> matched_hashes;
    TierType tier = TierType::UNKNOWN;
  };
  grpc::Status MatchExternalKv(const std::vector<std::string>& hashes,
                               std::vector<ExternalKvNodeMatch>* out_matches);

 private:
  UMBPMasterClientConfig config_;

  std::shared_ptr<grpc::Channel> channel_;
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

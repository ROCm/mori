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
#include "umbp/distributed/master/master_server.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp.grpc.pb.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/external_kv_block_index.h"
#include "umbp/distributed/master/external_kv_hit_index.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/routing/router.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {

uint32_t HeartbeatIntervalDivisor() {
  static const uint32_t v = GetEnvUint32("UMBP_HEARTBEAT_INTERVAL_DIVISOR", 2, /*min_allowed=*/1);
  return v;
}

std::chrono::seconds GrpcShutdownDeadline() {
  static const auto v =
      GetEnvSeconds("UMBP_GRPC_SHUTDOWN_DEADLINE_SEC", std::chrono::seconds(3), /*min_allowed=*/1);
  return v;
}

std::chrono::seconds HitIndexTtl() {
  static const auto v =
      GetEnvSeconds("UMBP_HIT_INDEX_TTL_SEC", std::chrono::seconds(7200), /*min_allowed=*/1);
  return v;
}

std::chrono::seconds HitIndexGcInterval() {
  static const auto v =
      GetEnvSeconds("UMBP_HIT_INDEX_GC_INTERVAL_SEC", std::chrono::seconds(60), /*min_allowed=*/1);
  return v;
}

uint32_t HitQueryMaxBatch() {
  static const uint32_t v = GetEnvUint32("UMBP_HIT_QUERY_MAX_BATCH", 4096, /*min_allowed=*/1);
  return v;
}

// GetTierCoeffs fallbacks, returned with epoch_ms=0 when there is no fresh
// online data.  Mirror the router's static defaults so behavior is sane even if
// a caller ignores epoch_ms.
constexpr float kDefaultY = 0.9f;
constexpr float kDefaultZ = 0.0f;
constexpr float kDefaultSsd = 0.0f;
constexpr double kBytesPerGiB = 1073741824.0;  // 1 << 30

// Prometheus-style histogram_quantile over an explicit-bound histogram with
// CUMULATIVE bucket counts.  `bounds` are finite upper bounds (ascending); the
// implicit +Inf bucket holds (total - cum.back()).  Linear interpolation within
// the bucket the requested rank lands in.  Returns NaN when there is no data.
double QuantileFromBuckets(const std::vector<double>& bounds, const std::vector<uint64_t>& cum,
                           double total, double q) {
  if (total <= 0.0 || bounds.empty() || cum.size() != bounds.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double rank = q * total;
  std::size_t i = 0;
  while (i < cum.size() && static_cast<double>(cum[i]) < rank) ++i;
  if (i == cum.size()) {
    // Rank falls in the implicit +Inf bucket; best finite estimate is the last
    // explicit bound.
    return bounds.back();
  }
  const double upper = bounds[i];
  const double lower = (i == 0) ? 0.0 : bounds[i - 1];
  const double cum_lower = (i == 0) ? 0.0 : static_cast<double>(cum[i - 1]);
  const double bucket_count = static_cast<double>(cum[i]) - cum_lower;
  if (bucket_count <= 0.0) return upper;
  const double frac = (rank - cum_lower) / bucket_count;
  return lower + frac * (upper - lower);
}

uint64_t NowNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

uint64_t ToNs(std::chrono::seconds value) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(value).count());
}

KvEvent FromProtoEvent(const ::umbp::KvEvent& pe) {
  KvEvent ev;
  switch (pe.kind()) {
    case ::umbp::KvEvent::ADD:
      ev.kind = KvEvent::Kind::ADD;
      break;
    case ::umbp::KvEvent::REMOVE:
      ev.kind = KvEvent::Kind::REMOVE;
      break;
    case ::umbp::KvEvent::CLEAR_AT_TIER:
      ev.kind = KvEvent::Kind::CLEAR_AT_TIER;
      break;
    default:
      ev.kind = KvEvent::Kind::ADD;
      break;
  }
  ev.key = pe.key();
  ev.tier = static_cast<TierType>(pe.tier());
  ev.size = pe.size();
  return ev;
}

int EvictKeyDeadlineMs() {
  static const int v =
      static_cast<int>(GetEnvMilliseconds("UMBP_EVICTKEY_DEADLINE_MS",
                                          std::chrono::milliseconds(1000), /*min_allowed=*/1)
                           .count());
  return v;
}

// Master-owned outbound stub pool.  EvictionManager calls into this to
// ship EvictKey to peers; the pool keeps one stub per node_id and
// drops it when the peer's address changes or the RPC fails (so the
// next dispatch rebuilds against a fresh channel).
class MasterPeerStubPool : public EvictKeyDispatcher {
 public:
  void DispatchEvictKey(const std::string& node_id, const std::string& peer_address,
                        std::vector<std::string> keys) override {
    if (keys.empty() || peer_address.empty()) return;

    auto stub = GetOrCreateStub(node_id, peer_address);
    if (stub == nullptr) return;

    ::umbp::EvictKeyRequest req;
    for (auto& k : keys) req.add_keys(std::move(k));
    ::umbp::EvictKeyResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() +
                     std::chrono::milliseconds(EvictKeyDeadlineMs()));

    auto status = stub->EvictKey(&ctx, req, &resp);
    if (!status.ok()) {
      MORI_UMBP_WARN("[Master] EvictKey to {} failed: {} (will retry next round)", node_id,
                     status.error_message());
      // Drop the cached stub so the next dispatch picks up a fresh
      // channel (covers transient gRPC channel teardown / restart).
      DropStub(node_id);
      return;
    }
    MORI_UMBP_DEBUG("[Master] EvictKey to {}: {} entries acked", node_id, resp.evicted_size());
  }

  // Called when ClientRegistry sees a node leave (Unregister or
  // expiry) so we don't sit on a stale stub for a node that has gone.
  void DropStub(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(node_id);
  }

 private:
  struct Entry {
    std::string peer_address;
    std::shared_ptr<::umbp::UMBPPeer::Stub> stub;
  };

  std::shared_ptr<::umbp::UMBPPeer::Stub> GetOrCreateStub(const std::string& node_id,
                                                          const std::string& peer_address) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(node_id);
    if (it != entries_.end() && it->second.peer_address == peer_address) return it->second.stub;

    auto channel = grpc::CreateChannel(peer_address, grpc::InsecureChannelCredentials());
    if (channel == nullptr) return nullptr;
    std::shared_ptr<::umbp::UMBPPeer::Stub> stub(::umbp::UMBPPeer::NewStub(channel).release());
    entries_[node_id] = Entry{peer_address, stub};
    return stub;
  }

  std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

}  // namespace

MasterServerConfig MasterServerConfig::FromEnvironment() {
  MasterServerConfig cfg;
  cfg.registry_config = ClientRegistryConfig::FromEnvironment();
  cfg.eviction_config = EvictionConfig::FromEnvironment();
  cfg.tier_coeff_config = TierCoeffConfig::FromEnvironment();
  return cfg;
}

// ---------------------------------------------------------------------------
//  gRPC service implementation
// ---------------------------------------------------------------------------
class MasterServer::UMBPMasterServiceImpl final : public ::umbp::UMBPMaster::Service {
 public:
  UMBPMasterServiceImpl(ClientRegistry& registry, GlobalBlockIndex& index,
                        ExternalKvBlockIndex& external_kv_index,
                        ExternalKvHitIndex& external_kv_hit_index, Router& router,
                        const ClientRegistryConfig& config, mori::metrics::MetricsServer* metrics,
                        const TierCoeffConfig& tier_cfg = {})
      : registry_(registry),
        index_(index),
        external_kv_index_(external_kv_index),
        external_kv_hit_index_(external_kv_hit_index),
        router_(router),
        config_(config),
        metrics_(metrics),
        tier_cfg_(tier_cfg) {}

  // -------- Client lifecycle --------

  grpc::Status RegisterClient(grpc::ServerContext* /*ctx*/,
                              const ::umbp::RegisterClientRequest* request,
                              ::umbp::RegisterClientResponse* response) override {
    std::map<TierType, TierCapacity> caps;
    for (const auto& tc : request->tier_capacities()) {
      TierCapacity c;
      c.total_bytes = tc.total_capacity_bytes();
      c.available_bytes = tc.available_capacity_bytes();
      caps[static_cast<TierType>(tc.tier())] = c;
    }

    const auto& engine_desc_str = request->engine_desc();
    std::vector<uint8_t> engine_desc_bytes(engine_desc_str.begin(), engine_desc_str.end());

    std::vector<std::string> tags(request->tags().begin(), request->tags().end());

    const bool registered =
        registry_.RegisterClient(request->node_id(), request->node_address(), caps,
                                 request->peer_address(), engine_desc_bytes, tags);
    if (!registered) {
      return grpc::Status(grpc::StatusCode::ALREADY_EXISTS,
                          "node is already alive and cannot be re-registered");
    }

    UpdateClientCountMetric();
    UpdateClientCapacityMetrics(request->node_id(), caps);

    auto interval_ms =
        static_cast<uint64_t>(config_.heartbeat_ttl.count() * 1000) / HeartbeatIntervalDivisor();
    response->set_heartbeat_interval_ms(interval_ms);
    response->set_ack_seq(0);
    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext* /*ctx*/,
                                const ::umbp::UnregisterClientRequest* request,
                                ::umbp::UnregisterClientResponse* /*response*/) override {
    registry_.UnregisterClient(request->node_id());
    UpdateClientCountMetric();
    return grpc::Status::OK;
  }

  // -------- Heartbeat (event-driven) --------

  grpc::Status Heartbeat(grpc::ServerContext* /*ctx*/, const ::umbp::HeartbeatRequest* request,
                         ::umbp::HeartbeatResponse* response) override {
    std::map<TierType, TierCapacity> caps;
    for (const auto& tc : request->tier_capacities()) {
      TierCapacity c;
      c.total_bytes = tc.total_capacity_bytes();
      c.available_bytes = tc.available_capacity_bytes();
      caps[static_cast<TierType>(tc.tier())] = c;
    }

    std::vector<EventBundle> bundles;
    bundles.reserve(static_cast<size_t>(request->bundles_size()));
    for (const auto& pb : request->bundles()) {
      EventBundle bundle;
      bundle.seq = pb.seq();
      bundle.events.reserve(static_cast<size_t>(pb.events_size()));
      for (const auto& pe : pb.events()) bundle.events.push_back(FromProtoEvent(pe));
      bundles.push_back(std::move(bundle));
    }

    uint64_t acked_seq = 0;
    bool request_full_sync = false;
    auto status =
        registry_.Heartbeat(request->node_id(), caps, bundles, request->is_full_sync(),
                            request->delta_seq_baseline(), &acked_seq, &request_full_sync);

    response->set_status(static_cast<::umbp::ClientStatus>(status));
    response->set_acked_seq(acked_seq);
    response->set_request_full_sync(request_full_sync);

    UpdateClientCapacityMetrics(request->node_id(), caps);

    if (metrics_ != nullptr && request->tier_kv_counts_size() > 0) {
      mori::metrics::MetricsServer::Labels base = {{"node", request->node_id()}};
      for (const auto& tag : registry_.GetClientTags(request->node_id())) {
        const auto sep = tag.find('=');
        if (sep != std::string::npos) {
          base.push_back({tag.substr(0, sep), tag.substr(sep + 1)});
        }
      }
      uint64_t total = 0;
      for (const auto& tkc : request->tier_kv_counts()) {
        total += tkc.count();
        auto labels = base;
        labels.push_back({"tier", TierTypeName(static_cast<TierType>(tkc.tier()))});
        metrics_->setGauge(MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT,
                           MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT_HELP, labels,
                           static_cast<double>(tkc.count()));
      }
      metrics_->setGauge(MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT_TOTAL,
                         MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT_TOTAL_HELP, base,
                         static_cast<double>(total));
    }

    if (request_full_sync && metrics_ != nullptr) {
      metrics_->addCounter("mori_umbp_heartbeat_seq_gap_total",
                           "Heartbeats rejected due to seq gap (full sync requested)",
                           {{"node", request->node_id()}});
    }
    size_t event_count = 0;
    for (const auto& bundle : bundles) event_count += bundle.events.size();
    if (metrics_ != nullptr && event_count > 0) {
      metrics_->addCounter("mori_umbp_heartbeat_events_applied_total",
                           "KvEvents applied to GlobalBlockIndex via heartbeat",
                           {{"node", request->node_id()}}, static_cast<uint64_t>(event_count));
    }
    return grpc::Status::OK;
  }

  // -------- Routing (read-only) --------

  grpc::Status RoutePut(grpc::ServerContext* /*ctx*/, const ::umbp::RoutePutRequest* request,
                        ::umbp::RoutePutResponse* response) override {
    if (request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
    }
    std::unordered_set<std::string> excludes(request->exclude_nodes().begin(),
                                             request->exclude_nodes().end());
    auto result =
        router_.RoutePut(request->key(), request->node_id(), request->block_size(), excludes);
    if (!result.has_value()) {
      response->set_found(false);
      return grpc::Status::OK;
    }
    response->set_found(true);
    response->set_node_id(result->node_id);
    response->set_tier(static_cast<::umbp::TierType>(result->tier));
    response->set_peer_address(result->peer_address);

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_ROUTE_PUT,
                           MORI_UMBP_METRIC_CLIENT_ROUTE_PUT_HELP, {{"node", result->node_id}});
    }
    return grpc::Status::OK;
  }

  grpc::Status RouteGet(grpc::ServerContext* /*ctx*/, const ::umbp::RouteGetRequest* request,
                        ::umbp::RouteGetResponse* response) override {
    if (request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
    }
    std::unordered_set<std::string> excludes(request->exclude_nodes().begin(),
                                             request->exclude_nodes().end());
    auto result = router_.RouteGet(request->key(), request->node_id(), excludes);
    if (!result.has_value()) {
      response->set_found(false);
      return grpc::Status::OK;
    }
    response->set_found(true);
    response->set_node_id(result->location.node_id);
    response->set_tier(static_cast<::umbp::TierType>(result->location.tier));
    response->set_size(result->location.size);
    response->set_peer_address(result->peer_address);

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_ROUTE_GET,
                           MORI_UMBP_METRIC_CLIENT_ROUTE_GET_HELP,
                           {{"node", result->location.node_id}});
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchRoutePut(grpc::ServerContext* /*ctx*/,
                             const ::umbp::BatchRoutePutRequest* request,
                             ::umbp::BatchRoutePutResponse* response) override {
    if (request->keys_size() != request->block_sizes_size()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "keys and block_sizes must have the same length");
    }
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::vector<uint64_t> block_sizes(request->block_sizes().begin(), request->block_sizes().end());
    std::unordered_set<std::string> excludes(request->exclude_nodes().begin(),
                                             request->exclude_nodes().end());

    auto results = router_.BatchRoutePut(keys, request->node_id(), block_sizes, excludes);
    for (auto& opt : results) {
      auto* entry = response->add_entries();
      if (!opt.has_value()) continue;  // default UNAVAILABLE
      if (opt->outcome == RoutePutOutcome::kAlreadyExists) {
        entry->set_outcome(::umbp::ROUTE_PUT_OUTCOME_ALREADY_EXISTS);
        continue;
      }
      entry->set_outcome(::umbp::ROUTE_PUT_OUTCOME_ROUTED);
      entry->set_node_id(opt->node_id);
      entry->set_tier(static_cast<::umbp::TierType>(opt->tier));
      entry->set_peer_address(opt->peer_address);
      if (metrics_) {
        metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT,
                             MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT_HELP,
                             {{"node", opt->node_id}});
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchRouteGet(grpc::ServerContext* /*ctx*/,
                             const ::umbp::BatchRouteGetRequest* request,
                             ::umbp::BatchRouteGetResponse* response) override {
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::unordered_set<std::string> excludes(request->exclude_nodes().begin(),
                                             request->exclude_nodes().end());
    auto results = router_.BatchRouteGet(keys, request->node_id(), excludes);
    for (auto& opt : results) {
      auto* entry = response->add_entries();
      if (!opt.has_value()) {
        entry->set_found(false);
        continue;
      }
      entry->set_found(true);
      entry->set_node_id(opt->location.node_id);
      entry->set_tier(static_cast<::umbp::TierType>(opt->location.tier));
      entry->set_size(opt->location.size);
      entry->set_peer_address(opt->peer_address);
      if (metrics_) {
        metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET,
                             MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET_HELP,
                             {{"node", opt->location.node_id}});
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status BatchLookup(grpc::ServerContext* /*ctx*/, const ::umbp::BatchLookupRequest* request,
                           ::umbp::BatchLookupResponse* response) override {
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    auto found = index_.BatchLookupExists(keys);
    for (bool b : found) response->add_found(b);
    return grpc::Status::OK;
  }

  // -------- External KV mutation/query --------

  grpc::Status ReportExternalKvBlocks(
      grpc::ServerContext* /*ctx*/, const ::umbp::ReportExternalKvBlocksRequest* request,
      ::umbp::ReportExternalKvBlocksResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id must not be empty");
    }
    if (request->hashes_size() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "hashes must not be empty");
    }

    const TierType tier = static_cast<TierType>(request->tier());
    if (!registry_.IsClientAlive(request->node_id())) {
      MORI_UMBP_WARN("[Server] ReportExternalKvBlocks rejected: node not alive: {}",
                     request->node_id());
      if (metrics_) {
        metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL,
                             MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL_HELP,
                             {{"node", request->node_id()},
                              {"tier", TierTypeName(tier)},
                              {"result", "rejected_not_alive"}});
      }
      return grpc::Status::OK;
    }

    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    const size_t mutated = external_kv_index_.Register(request->node_id(), hashes, tier);
    if (metrics_) {
      const mori::metrics::MetricsServer::Labels labels = {{"node", request->node_id()},
                                                           {"tier", TierTypeName(tier)}};
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL_HELP, labels,
                           static_cast<uint64_t>(mutated));
      metrics_->addCounter(
          MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL, MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL_HELP,
          {{"node", request->node_id()}, {"tier", TierTypeName(tier)}, {"result", "ok"}});
    }
    return grpc::Status::OK;
  }

  grpc::Status RevokeExternalKvBlocks(
      grpc::ServerContext* /*ctx*/, const ::umbp::RevokeExternalKvBlocksRequest* request,
      ::umbp::RevokeExternalKvBlocksResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id must not be empty");
    }
    if (request->hashes_size() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "hashes must not be empty");
    }

    const TierType tier = static_cast<TierType>(request->tier());
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    const size_t mutated = external_kv_index_.Unregister(request->node_id(), hashes, tier);
    if (metrics_) {
      const mori::metrics::MetricsServer::Labels labels = {{"node", request->node_id()},
                                                           {"tier", TierTypeName(tier)}};
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL_HELP, labels,
                           static_cast<uint64_t>(mutated));
      metrics_->addCounter(
          MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL, MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL_HELP,
          {{"node", request->node_id()}, {"tier", TierTypeName(tier)}, {"result", "ok"}});
    }
    return grpc::Status::OK;
  }

  grpc::Status RevokeAllExternalKvBlocksAtTier(
      grpc::ServerContext* /*ctx*/, const ::umbp::RevokeAllExternalKvBlocksAtTierRequest* request,
      ::umbp::RevokeAllExternalKvBlocksAtTierResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id must not be empty");
    }

    const TierType tier = static_cast<TierType>(request->tier());
    const size_t mutated = external_kv_index_.UnregisterByNodeAtTier(request->node_id(), tier);
    if (metrics_) {
      const mori::metrics::MetricsServer::Labels labels = {{"node", request->node_id()},
                                                           {"tier", TierTypeName(tier)}};
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL_HELP, labels,
                           static_cast<uint64_t>(mutated));
      metrics_->addCounter(
          MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL, MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL_HELP,
          {{"node", request->node_id()}, {"tier", TierTypeName(tier)}, {"result", "ok"}});
    }
    return grpc::Status::OK;
  }

  grpc::Status RevokeAllExternalKvBlocksForNode(
      grpc::ServerContext* /*ctx*/, const ::umbp::RevokeAllExternalKvBlocksForNodeRequest* request,
      ::umbp::RevokeAllExternalKvBlocksForNodeResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id must not be empty");
    }

    const size_t mutated = external_kv_index_.UnregisterByNode(request->node_id());
    if (metrics_) {
      const mori::metrics::MetricsServer::Labels labels = {{"node", request->node_id()},
                                                           {"tier", "ALL"}};
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL_HELP, labels,
                           static_cast<uint64_t>(mutated));
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL_HELP,
                           {{"node", request->node_id()}, {"tier", "ALL"}, {"result", "ok"}});
    }
    return grpc::Status::OK;
  }

  grpc::Status MatchExternalKv(grpc::ServerContext* /*ctx*/,
                               const ::umbp::MatchExternalKvRequest* request,
                               ::umbp::MatchExternalKvResponse* response) override {
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    auto matches = external_kv_index_.Match(hashes);

    std::unordered_map<std::string, std::string> peer_map;
    for (const auto& record : registry_.GetAliveClients()) {
      peer_map[record.node_id] = record.peer_address;
    }
    for (auto& m : matches) {
      auto* proto_match = response->add_matches();
      proto_match->set_node_id(m.node_id);
      auto peer_it = peer_map.find(m.node_id);
      if (peer_it != peer_map.end()) proto_match->set_peer_address(peer_it->second);
      for (const auto& [tier, hashes] : m.hashes_by_tier) {
        auto* proto_bucket = proto_match->add_hashes_by_tier();
        proto_bucket->set_tier(static_cast<::umbp::TierType>(tier));
        for (const auto& hash : hashes) proto_bucket->add_hashes(hash);
      }
    }

    if (request->count_as_hit() && !matches.empty()) {
      std::unordered_set<std::string> matched_hashes;
      for (const auto& m : matches) {
        for (const auto& [tier, hashes_in_tier] : m.hashes_by_tier) {
          for (const auto& hash : hashes_in_tier) matched_hashes.insert(hash);
        }
      }
      if (!matched_hashes.empty()) {
        std::vector<std::string> unique_matched;
        unique_matched.reserve(matched_hashes.size());
        for (const auto& hash : matched_hashes) unique_matched.push_back(hash);
        external_kv_hit_index_.IncrementHits(unique_matched, NowNs());
      }
    }

    size_t total_matched = 0;
    for (const auto& m : matches) total_matched += m.MatchedHashCount();
    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_MATCH_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_MATCH_TOTAL_HELP);
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_MATCH_QUERIED_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_MATCH_QUERIED_BLOCKS_TOTAL_HELP,
                           static_cast<uint64_t>(hashes.size()));
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_MATCH_MATCHED_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_MATCH_MATCHED_BLOCKS_TOTAL_HELP,
                           static_cast<uint64_t>(total_matched));
    }
    return grpc::Status::OK;
  }

  grpc::Status GetExternalKvHitCounts(grpc::ServerContext* /*ctx*/,
                                      const ::umbp::GetExternalKvHitCountsRequest* request,
                                      ::umbp::GetExternalKvHitCountsResponse* response) override {
    const size_t max_batch = static_cast<size_t>(HitQueryMaxBatch());
    if (static_cast<size_t>(request->hashes_size()) > max_batch) {
      return grpc::Status(
          grpc::StatusCode::INVALID_ARGUMENT,
          "hashes size exceeds UMBP_HIT_QUERY_MAX_BATCH=" + std::to_string(max_batch));
    }

    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    std::vector<std::pair<std::string, uint64_t>> entries;
    entries.reserve(hashes.size());
    external_kv_hit_index_.Lookup(hashes, &entries);
    for (const auto& [hash, total] : entries) {
      auto* entry = response->add_entries();
      entry->set_hash(hash);
      entry->set_hit_count_total(total);
    }
    return grpc::Status::OK;
  }

  grpc::Status ReportMetrics(grpc::ServerContext* /*ctx*/,
                             const ::umbp::ReportMetricsRequest* request,
                             ::umbp::ReportMetricsResponse* /*response*/) override {
    if (!metrics_) return grpc::Status::OK;

    mori::metrics::MetricsServer::Labels base = {{"node", request->node_id()}};
    for (const auto& tag : registry_.GetClientTags(request->node_id())) {
      const auto sep = tag.find('=');
      if (sep != std::string::npos) {
        base.push_back({tag.substr(0, sep), tag.substr(sep + 1)});
      }
    }
    for (const auto& s : request->metrics()) {
      mori::metrics::MetricsServer::Labels labels = base;
      for (const auto& l : s.labels()) labels.push_back({l.name(), l.value()});
      switch (s.value_case()) {
        case ::umbp::MetricSample::kCounterDelta:
          metrics_->addCounter(s.name(), s.help(), labels,
                               static_cast<uint64_t>(s.counter_delta()));
          break;
        case ::umbp::MetricSample::kGaugeValue:
          metrics_->setGauge(s.name(), s.help(), labels, s.gauge_value());
          break;
        case ::umbp::MetricSample::kHistogramAggregate: {
          const auto& a = s.histogram_aggregate();
          std::vector<double> bounds(a.bounds().begin(), a.bounds().end());
          std::vector<uint64_t> counts(a.bucket_counts().begin(), a.bucket_counts().end());
          metrics_->observeAggregated(s.name(), s.help(), labels, bounds, counts, a.count(),
                                      a.sum());
          break;
        }
        default:
          break;
      }
    }
    return grpc::Status::OK;
  }

  // -------- Online tier hit-quality coefficients --------
  // Read-only.  Reads the in-memory metrics histograms, differences them over a
  // fixed wall-clock window, and applies the saturating formula.  On
  // missing/cold/stale data it returns fallbacks with epoch_ms=0, which the
  // router treats as "use my config".
  grpc::Status GetTierCoeffs(grpc::ServerContext* /*ctx*/,
                             const ::umbp::GetTierCoeffsRequest* request,
                             ::umbp::GetTierCoeffsResponse* response) override {
    // Resolve the candidate set: explicit list, or all alive clients when empty.
    std::vector<std::string> nodes(request->candidate_node_ids().begin(),
                                   request->candidate_node_ids().end());
    if (nodes.empty()) {
      for (const auto& record : registry_.GetAliveClients()) nodes.push_back(record.node_id);
    }

    const uint64_t now_ms =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count());

    for (const auto& node : nodes) {
      auto* out = response->add_coeffs();
      out->set_node_id(node);
      out->set_x(1.0f);           // HBM, c_fetch == 0
      out->set_ssd(kDefaultSsd);  // reserved; SSD is never matched on this path

      // c_gather = p50 of the per-node gather histogram (already seconds/token).
      const auto c_gather = WindowedQuantile(MORI_UMBP_METRIC_CLIENT_KV_GATHER_DURATION,
                                             {{"node", node}}, tier_cfg_.gather_quantile);

      // c_remote = bytes_per_token / bandwidth_lowquantile.  Low bandwidth
      // quantile -> high latency (conservative).
      const auto bw_remote =
          WindowedQuantile(MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH,
                           {{"node", node}, {"traffic", "remote"}}, tier_cfg_.remote_bw_quantile);

      const double c_half = tier_cfg_.c_half_seconds;
      bool fresh = false;

      // y depends only on gather.
      float y = kDefaultY;
      if (c_gather && std::isfinite(*c_gather) && *c_gather >= 0.0) {
        y = static_cast<float>(c_half / (c_half + *c_gather));
        fresh = true;
      }
      out->set_y(y);

      // z needs both remote and gather.
      float z = kDefaultZ;
      if (c_gather && std::isfinite(*c_gather) && *c_gather >= 0.0 && bw_remote &&
          std::isfinite(*bw_remote) && *bw_remote > 0.0) {
        const double c_remote = tier_cfg_.bytes_per_token / (*bw_remote * kBytesPerGiB);
        z = static_cast<float>(c_half / (c_half + c_remote + *c_gather));
        if (tier_cfg_.clamp_z_le_y && z > y) z = y;  // start-up safety
        fresh = true;
      }
      out->set_z(z);

      out->set_epoch_ms(fresh ? now_ms : 0u);
    }
    return grpc::Status::OK;
  }

  void SetMetrics(mori::metrics::MetricsServer* metrics) { metrics_ = metrics; }

 private:
  void UpdateClientCountMetric() {
    if (!metrics_) return;
    metrics_->setGauge(MORI_UMBP_METRIC_CLIENT_COUNT, MORI_UMBP_METRIC_CLIENT_COUNT_HELP,
                       static_cast<double>(registry_.GetAliveClients().size()));
  }

  void UpdateClientCapacityMetrics(const std::string& node_id,
                                   const std::map<TierType, TierCapacity>& caps) {
    if (!metrics_) return;
    for (const auto& [tier, cap] : caps) {
      const char* tier_name = TierTypeName(tier);
      mori::metrics::MetricsServer::Labels labels = {{"node", node_id}, {"tier", tier_name}};
      metrics_->setGauge(MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL,
                         MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL_HELP, labels,
                         static_cast<double>(cap.total_bytes));
      metrics_->setGauge(MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL,
                         MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL_HELP, labels,
                         static_cast<double>(cap.available_bytes));
      const uint64_t used_bytes =
          cap.total_bytes >= cap.available_bytes ? cap.total_bytes - cap.available_bytes : 0;
      metrics_->setGauge(MORI_UMBP_METRIC_CLIENT_CAPACITY_USED,
                         MORI_UMBP_METRIC_CLIENT_CAPACITY_USED_HELP, labels,
                         static_cast<double>(used_bytes));
      const double utilization = cap.total_bytes > 0 ? static_cast<double>(used_bytes) /
                                                           static_cast<double>(cap.total_bytes)
                                                     : 0.0;
      metrics_->setGauge(MORI_UMBP_METRIC_CLIENT_CAPACITY_UTILIZATION,
                         MORI_UMBP_METRIC_CLIENT_CAPACITY_UTILIZATION_HELP, labels, utilization);
    }
  }

  // Snapshot the named labeled histogram now, difference it against the snapshot
  // taken ~window ago (per-series ring), and return the requested quantile over
  // the DELTA histogram, so the result reflects recent cost rather than all-time
  // accumulation.  nullopt when metrics are disabled, the series is absent, the
  // ring has no window-old baseline yet (cold start), or no fresh samples landed
  // in the window.
  std::optional<double> WindowedQuantile(const char* metric_name,
                                         const mori::metrics::MetricsServer::Labels& labels,
                                         double quantile) {
    if (!metrics_) return std::nullopt;
    auto now_snap = metrics_->SnapshotLabeledHistogram(metric_name, labels);
    if (!now_snap) return std::nullopt;

    const auto now = std::chrono::steady_clock::now();
    const auto window = tier_cfg_.window;

    // Ring key = metric name + label signature (caller passes labels in a fixed
    // order, so concatenation is stable).
    std::string key = metric_name;
    for (const auto& kv : labels) {
      key += '|';
      key += kv.first;
      key += '=';
      key += kv.second;
    }

    std::vector<uint64_t> baseline_counts;
    uint64_t baseline_total = 0;
    bool have_baseline = false;
    {
      std::lock_guard<std::mutex> lk(coeff_ring_mutex_);
      auto& ring = coeff_ring_[key];
      // Pick the newest snapshot that is at least `window` old as the baseline.
      for (auto it = ring.rbegin(); it != ring.rend(); ++it) {
        if (now - it->t >= window) {
          if (it->snap.bucket_counts.size() == now_snap->bucket_counts.size()) {
            baseline_counts = it->snap.bucket_counts;
            baseline_total = it->snap.count;
            have_baseline = true;
          }
          break;
        }
      }
      // Record the current snapshot, then prune anything older than 2*window so
      // the ring stays bounded regardless of polling cadence.
      ring.push_back(TimedSnapshot{now, *now_snap});
      while (!ring.empty() && now - ring.front().t > 2 * window) ring.pop_front();
    }

    if (!have_baseline) return std::nullopt;  // cold start: no window-old baseline

    std::vector<uint64_t> delta(now_snap->bucket_counts.size());
    for (std::size_t i = 0; i < delta.size(); ++i) {
      delta[i] = now_snap->bucket_counts[i] - baseline_counts[i];
    }
    const double delta_total = static_cast<double>(now_snap->count - baseline_total);
    if (delta_total <= 0.0) return std::nullopt;  // no fresh samples in window

    const double q = QuantileFromBuckets(now_snap->bounds, delta, delta_total, quantile);
    if (!std::isfinite(q)) return std::nullopt;
    return q;
  }

  ClientRegistry& registry_;
  GlobalBlockIndex& index_;
  ExternalKvBlockIndex& external_kv_index_;
  ExternalKvHitIndex& external_kv_hit_index_;
  Router& router_;
  ClientRegistryConfig config_;
  mori::metrics::MetricsServer* metrics_ = nullptr;
  TierCoeffConfig tier_cfg_;

  // Per-series ring of timestamped cumulative snapshots, used to difference the
  // monotonic in-memory histograms over a fixed wall-clock window.  Keyed by the
  // lookup signature (metric name + required labels) and guarded by its own
  // mutex; only touched on the GetTierCoeffs path.
  struct TimedSnapshot {
    std::chrono::steady_clock::time_point t;
    mori::metrics::MetricsServer::HistogramSnapshot snap;
  };
  std::mutex coeff_ring_mutex_;
  std::map<std::string, std::deque<TimedSnapshot>> coeff_ring_;
};

// ---------------------------------------------------------------------------
//  MasterServer
// ---------------------------------------------------------------------------
MasterServer::MasterServer(MasterServerConfig config)
    : config_(std::move(config)),
      index_(),
      external_kv_index_(),
      external_kv_hit_index_(),
      registry_(config_.registry_config, index_, &external_kv_index_),
      router_(index_, registry_, std::move(config_.get_strategy), std::move(config_.put_strategy)),
      service_(std::make_unique<UMBPMasterServiceImpl>(
          registry_, index_, external_kv_index_, external_kv_hit_index_, router_,
          config_.registry_config, nullptr, config_.tier_coeff_config)),
      peer_stub_pool_(std::make_unique<MasterPeerStubPool>()),
      eviction_manager_(std::make_unique<EvictionManager>(
          index_, registry_, config_.eviction_config, peer_stub_pool_.get())) {
  router_.SetLeaseDuration(config_.eviction_config.lease_duration);
}

MasterServer::~MasterServer() {
  Shutdown();
  server_.reset();
}

void MasterServer::Run() {
  if (config_.metrics_port > 0) {
    metrics_server_ = std::make_unique<mori::metrics::MetricsServer>(config_.metrics_port);
    service_->SetMetrics(metrics_server_.get());
    metrics_server_->setGauge(MORI_UMBP_METRIC_CLIENT_COUNT, MORI_UMBP_METRIC_CLIENT_COUNT_HELP,
                              0.0);
    MORI_UMBP_INFO("[Master] Metrics server listening on port {}", config_.metrics_port);
  }

  registry_.StartReaper();
  eviction_manager_->Start();
  StartHitIndexGc();

  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  builder.SetMaxSendMessageSize(64 * 1024 * 1024);
  int selected_port = 0;
  builder.AddListeningPort(config_.listen_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  bound_port_.store(static_cast<uint16_t>(selected_port));

  MORI_UMBP_INFO("[Master] Listening on {}", config_.listen_address);
  server_->Wait();
}

void MasterServer::Shutdown() {
  if (eviction_manager_) eviction_manager_->Stop();
  if (server_) {
    const auto deadline = std::chrono::system_clock::now() + GrpcShutdownDeadline();
    MORI_UMBP_INFO("[Master] Shutting down");
    server_->Shutdown(deadline);
  }
  registry_.StopReaper();
  StopHitIndexGc();
}

void MasterServer::StartHitIndexGc() {
  bool expected = false;
  if (!hit_index_gc_running_.compare_exchange_strong(expected, true)) return;
  hit_index_gc_thread_ = std::thread(&MasterServer::HitIndexGcLoop, this);
  MORI_UMBP_INFO("[Master] External KV hit index GC started (ttl={}s, interval={}s)",
                 HitIndexTtl().count(), HitIndexGcInterval().count());
}

void MasterServer::StopHitIndexGc() {
  bool expected = true;
  if (!hit_index_gc_running_.compare_exchange_strong(expected, false)) return;
  hit_index_gc_cv_.notify_one();
  if (hit_index_gc_thread_.joinable()) hit_index_gc_thread_.join();
  MORI_UMBP_INFO("[Master] External KV hit index GC stopped");
}

void MasterServer::HitIndexGcLoop() {
  const uint64_t ttl_ns = ToNs(HitIndexTtl());
  while (hit_index_gc_running_) {
    {
      std::unique_lock lock(hit_index_gc_cv_mutex_);
      hit_index_gc_cv_.wait_for(lock, HitIndexGcInterval(),
                                [this] { return !hit_index_gc_running_.load(); });
    }
    if (!hit_index_gc_running_) break;

    const uint64_t now_ns = NowNs();
    const uint64_t cutoff_ns = now_ns > ttl_ns ? now_ns - ttl_ns : 0;
    if (cutoff_ns == 0) continue;
    const size_t dropped = external_kv_hit_index_.GarbageCollect(cutoff_ns);
    if (dropped > 0) {
      MORI_UMBP_DEBUG("[Master] External KV hit index GC dropped {} entries", dropped);
    }
  }
}

}  // namespace mori::umbp

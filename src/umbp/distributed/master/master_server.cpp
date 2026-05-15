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
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp.grpc.pb.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/external_kv_block_index.h"
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

KvEvent FromProtoEvent(const ::umbp::KvEvent& pe) {
  KvEvent ev;
  ev.kind = (pe.kind() == ::umbp::KvEvent::ADD) ? KvEvent::Kind::ADD : KvEvent::Kind::REMOVE;
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
  return cfg;
}

// ---------------------------------------------------------------------------
//  gRPC service implementation
// ---------------------------------------------------------------------------
class MasterServer::UMBPMasterServiceImpl final : public ::umbp::UMBPMaster::Service {
 public:
  UMBPMasterServiceImpl(ClientRegistry& registry, GlobalBlockIndex& index,
                        ExternalKvBlockIndex& external_kv_index, Router& router,
                        const ClientRegistryConfig& config, mori::metrics::MetricsServer* metrics)
      : registry_(registry),
        index_(index),
        external_kv_index_(external_kv_index),
        router_(router),
        config_(config),
        metrics_(metrics) {}

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

    std::vector<KvEvent> events;
    events.reserve(static_cast<size_t>(request->events_size()));
    for (const auto& pe : request->events()) events.push_back(FromProtoEvent(pe));

    uint64_t acked_seq = 0;
    bool request_full_sync = false;
    auto status =
        registry_.Heartbeat(request->node_id(), request->seq(), request->last_acked_seq(), caps,
                            events, request->is_full_sync(), &acked_seq, &request_full_sync);

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
    if (metrics_ != nullptr && !events.empty()) {
      metrics_->addCounter("mori_umbp_heartbeat_events_applied_total",
                           "KvEvents applied to GlobalBlockIndex via heartbeat",
                           {{"node", request->node_id()}}, static_cast<uint64_t>(events.size()));
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

  // -------- External KV (unchanged from prior design) --------

  grpc::Status ReportExternalKvBlocks(
      grpc::ServerContext* /*ctx*/, const ::umbp::ReportExternalKvBlocksRequest* request,
      ::umbp::ReportExternalKvBlocksResponse* /*response*/) override {
    if (request->node_id().empty() || request->hashes_size() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id/hashes cannot be empty");
    }
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    TierType tier = static_cast<TierType>(request->tier());
    registry_.RegisterExternalKvBlocks(request->node_id(), hashes, tier);

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL_HELP,
                           {{"node", request->node_id()}});
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL_HELP,
                           {{"node", request->node_id()}}, static_cast<uint64_t>(hashes.size()));
      const size_t kv_count = external_kv_index_.GetKvCount(request->node_id());
      metrics_->setGauge(MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT,
                         MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT_HELP, {{"node", request->node_id()}},
                         static_cast<double>(kv_count));
    }
    return grpc::Status::OK;
  }

  grpc::Status RevokeExternalKvBlocks(
      grpc::ServerContext* /*ctx*/, const ::umbp::RevokeExternalKvBlocksRequest* request,
      ::umbp::RevokeExternalKvBlocksResponse* /*response*/) override {
    if (request->node_id().empty() || request->hashes_size() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id/hashes cannot be empty");
    }
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    TierType tier = static_cast<TierType>(request->tier());
    registry_.UnregisterExternalKvBlocks(request->node_id(), hashes, tier);

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL_HELP,
                           {{"node", request->node_id()}});
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL_HELP,
                           {{"node", request->node_id()}}, static_cast<uint64_t>(hashes.size()));
      const size_t kv_count = external_kv_index_.GetKvCount(request->node_id());
      metrics_->setGauge(MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT,
                         MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT_HELP, {{"node", request->node_id()}},
                         static_cast<double>(kv_count));
    }
    return grpc::Status::OK;
  }

  grpc::Status RevokeAllExternalKvBlocksAtTier(
      grpc::ServerContext* /*ctx*/, const ::umbp::RevokeAllExternalKvBlocksAtTierRequest* request,
      ::umbp::RevokeAllExternalKvBlocksAtTierResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id cannot be empty");
    }
    TierType tier = static_cast<TierType>(request->tier());
    registry_.UnregisterExternalKvBlocksByTier(request->node_id(), tier);

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL_HELP,
                           {{"node", request->node_id()}});
      const size_t kv_count = external_kv_index_.GetKvCount(request->node_id());
      metrics_->setGauge(MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT,
                         MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT_HELP, {{"node", request->node_id()}},
                         static_cast<double>(kv_count));
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

  ClientRegistry& registry_;
  GlobalBlockIndex& index_;
  ExternalKvBlockIndex& external_kv_index_;
  Router& router_;
  ClientRegistryConfig config_;
  mori::metrics::MetricsServer* metrics_ = nullptr;
};

// ---------------------------------------------------------------------------
//  MasterServer
// ---------------------------------------------------------------------------
MasterServer::MasterServer(MasterServerConfig config)
    : config_(std::move(config)),
      index_(),
      external_kv_index_(),
      registry_(config_.registry_config, index_),
      router_(index_, registry_, std::move(config_.get_strategy), std::move(config_.put_strategy)),
      service_(std::make_unique<UMBPMasterServiceImpl>(registry_, index_, external_kv_index_,
                                                       router_, config_.registry_config, nullptr)),
      peer_stub_pool_(std::make_unique<MasterPeerStubPool>()),
      eviction_manager_(std::make_unique<EvictionManager>(
          index_, registry_, config_.eviction_config, peer_stub_pool_.get())) {
  router_.SetLeaseDuration(config_.eviction_config.lease_duration);
  registry_.SetExternalKvBlockIndex(&external_kv_index_);
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

  grpc::ServerBuilder builder;
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
}

}  // namespace mori::umbp

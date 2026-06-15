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
#include "umbp/distributed/master/master_client.h"

#include <grpcpp/grpcpp.h>

#include <array>
#include <chrono>
#include <iterator>
#include <set>
#include <system_error>

#include "mori/utils/mori_log.hpp"
#include "umbp.grpc.pb.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/master/rpc_latency_timer.h"
#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"

namespace mori::umbp {

namespace {

constexpr std::array<double, 14> kMasterClientRpcLatencyBucketsArr = {
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0};

int RpcShutdownTimeoutMs() {
  static const int v =
      static_cast<int>(GetEnvMilliseconds("UMBP_RPC_SHUTDOWN_TIMEOUT_MS",
                                          std::chrono::milliseconds(3000), /*min_allowed=*/1)
                           .count());
  return v;
}

uint64_t MetricsReportIntervalMs() {
  static const uint64_t v =
      static_cast<uint64_t>(GetEnvMilliseconds("UMBP_METRICS_REPORT_INTERVAL_MS",
                                               std::chrono::milliseconds(1000), /*min_allowed=*/1)
                                .count());
  return v;
}

::umbp::UMBPMaster::Stub* GetStub(void* ptr) { return static_cast<::umbp::UMBPMaster::Stub*>(ptr); }

::umbp::TierType ToProtoTier(TierType t) { return static_cast<::umbp::TierType>(t); }
TierType FromProtoTier(::umbp::TierType t) { return static_cast<TierType>(t); }

::umbp::KvEvent::Kind ToProtoEventKind(KvEvent::Kind kind) {
  switch (kind) {
    case KvEvent::Kind::ADD:
      return ::umbp::KvEvent::ADD;
    case KvEvent::Kind::REMOVE:
      return ::umbp::KvEvent::REMOVE;
    case KvEvent::Kind::CLEAR_AT_TIER:
      return ::umbp::KvEvent::CLEAR_AT_TIER;
  }
  return ::umbp::KvEvent::ADD;
}

void FillTierCapacities(::google::protobuf::RepeatedPtrField<::umbp::TierCapacity>* dst,
                        const std::map<TierType, TierCapacity>& src) {
  for (const auto& [tier, cap] : src) {
    auto* tc = dst->Add();
    tc->set_tier(ToProtoTier(tier));
    tc->set_total_capacity_bytes(cap.total_bytes);
    tc->set_available_capacity_bytes(cap.available_bytes);
  }
}

void FillExcludeNodes(::google::protobuf::RepeatedPtrField<std::string>* dst,
                      const std::unordered_set<std::string>& excludes) {
  for (const auto& n : excludes) dst->Add()->assign(n);
}

void FillTierKvCounts(::google::protobuf::RepeatedPtrField<::umbp::TierKvCount>* dst,
                      const std::map<TierType, uint64_t>& counts) {
  for (const auto& [tier, count] : counts) {
    auto* tkc = dst->Add();
    tkc->set_tier(ToProtoTier(tier));
    tkc->set_count(count);
  }
}

void FillBundle(::umbp::EventBundle* dst, const EventBundle& src) {
  dst->set_seq(src.seq);
  for (const auto& ev : src.events) {
    auto* pe = dst->add_events();
    pe->set_kind(ToProtoEventKind(ev.kind));
    pe->set_key(ev.key);
    pe->set_tier(ToProtoTier(ev.tier));
    pe->set_size(ev.size);
  }
}

}  // namespace

MasterClient::MasterClient(const UMBPMasterClientConfig& config)
    : config_(config),
      stub_(nullptr, [](void* p) { delete static_cast<::umbp::UMBPMaster::Stub*>(p); }) {
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  args.SetMaxSendMessageSize(64 * 1024 * 1024);
  channel_ =
      grpc::CreateCustomChannel(config.master_address, grpc::InsecureChannelCredentials(), args);
  stub_.reset(::umbp::UMBPMaster::NewStub(channel_).release());
  metrics_interval_ms_ = MetricsReportIntervalMs();
  MORI_UMBP_INFO("[Client] Created, master={}", config.master_address);
}

MasterClient::~MasterClient() {
  StopMetricsReporting();
  StopHeartbeat();
  if (registered_) {
    UnregisterSelf();
  }
}

grpc::Status MasterClient::RegisterSelf(const std::map<TierType, TierCapacity>& tier_capacities,
                                        const std::string& peer_address,
                                        const std::vector<uint8_t>& engine_desc_bytes) {
  ScopedRpcTimer _rpc_timer(this, "RegisterClient");
  if (registered_) {
    return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "node is already registered");
  }

  ::umbp::RegisterClientRequest req;
  req.set_node_id(config_.node_id);
  req.set_node_address(config_.node_address);
  req.set_peer_address(peer_address);
  req.set_engine_desc(engine_desc_bytes.data(), engine_desc_bytes.size());
  FillTierCapacities(req.mutable_tier_capacities(), tier_capacities);
  for (const auto& tag : config_.tags) req.add_tags(tag);

  ::umbp::RegisterClientResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->RegisterClient(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) {
    MORI_UMBP_WARN("[Client] RegisterSelf failed: {}", status.error_message());
    return status;
  }

  if (resp.heartbeat_interval_ms() > 0) heartbeat_interval_ms_ = resp.heartbeat_interval_ms();
  {
    std::lock_guard lock(hb_state_mutex_);
    hb_last_acked_seq_ = 0;
    next_bundle_seq_ = 1;
    outbox_.clear();
    full_sync_pending_ = false;
  }
  {
    std::lock_guard lock(caps_mutex_);
    current_capacities_ = tier_capacities;
  }
  registered_ = true;
  MORI_UMBP_INFO("[Client] Registered with master (heartbeat={}ms)", heartbeat_interval_ms_);
  StartMetricsReporting();
  return grpc::Status::OK;
}

grpc::Status MasterClient::UnregisterSelf() {
  if (!registered_) return grpc::Status::OK;
  ScopedRpcTimer _rpc_timer(this, "UnregisterClient");
  ::umbp::UnregisterClientRequest req;
  req.set_node_id(config_.node_id);
  ::umbp::UnregisterClientResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(RpcShutdownTimeoutMs()));
  auto status = GetStub(stub_.get())->UnregisterClient(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  registered_ = false;
  return status;
}

grpc::Status MasterClient::RoutePut(const std::string& key, uint64_t block_size,
                                    const std::unordered_set<std::string>& exclude_nodes,
                                    std::optional<RoutePutResult>* out_result) {
  ScopedRpcTimer _rpc_timer(this, "RoutePut");
  if (out_result == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "out_result is null");
  }
  out_result->reset();

  ::umbp::RoutePutRequest req;
  req.set_key(key);
  req.set_node_id(config_.node_id);
  req.set_block_size(block_size);
  FillExcludeNodes(req.mutable_exclude_nodes(), exclude_nodes);

  ::umbp::RoutePutResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->RoutePut(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) return status;

  switch (resp.outcome()) {
    case ::umbp::ROUTE_PUT_OUTCOME_ROUTED:
      *out_result = RoutePutResult{
          .outcome = RoutePutOutcome::kRouted,
          .node_id = resp.node_id(),
          .peer_address = resp.peer_address(),
          .tier = FromProtoTier(resp.tier()),
      };
      break;
    case ::umbp::ROUTE_PUT_OUTCOME_ALREADY_EXISTS:
      // Non-nullopt so caller distinguishes dedup from unavailable (= nullopt).
      *out_result = RoutePutResult{.outcome = RoutePutOutcome::kAlreadyExists};
      break;
    case ::umbp::ROUTE_PUT_OUTCOME_UNAVAILABLE:
    default:
      break;  // leave as nullopt
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::RouteGet(const std::string& key,
                                    const std::unordered_set<std::string>& exclude_nodes,
                                    std::optional<RouteGetResult>* out_result) {
  ScopedRpcTimer _rpc_timer(this, "RouteGet");
  if (out_result == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "out_result is null");
  }
  out_result->reset();

  ::umbp::RouteGetRequest req;
  req.set_key(key);
  req.set_node_id(config_.node_id);
  FillExcludeNodes(req.mutable_exclude_nodes(), exclude_nodes);

  ::umbp::RouteGetResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->RouteGet(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) return status;
  if (!resp.found()) return grpc::Status::OK;

  RouteGetResult r;
  r.node_id = resp.node_id();
  r.tier = FromProtoTier(resp.tier());
  r.size = resp.size();
  r.peer_address = resp.peer_address();
  *out_result = std::move(r);
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchRoutePut(const std::vector<std::string>& keys,
                                         const std::vector<uint64_t>& block_sizes,
                                         const std::unordered_set<std::string>& exclude_nodes,
                                         std::vector<std::optional<RoutePutResult>>* out) {
  ScopedRpcTimer _rpc_timer(this, "BatchRoutePut");
  if (out == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "out is null");
  }
  out->clear();
  if (keys.size() != block_sizes.size()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "keys and block_sizes length mismatch");
  }
  ::umbp::BatchRoutePutRequest req;
  req.set_node_id(config_.node_id);
  for (const auto& k : keys) req.add_keys(k);
  for (uint64_t s : block_sizes) req.add_block_sizes(s);
  FillExcludeNodes(req.mutable_exclude_nodes(), exclude_nodes);

  ::umbp::BatchRoutePutResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchRoutePut(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) return status;

  out->resize(static_cast<size_t>(resp.entries_size()));
  for (int i = 0; i < resp.entries_size(); ++i) {
    const auto& e = resp.entries(i);
    switch (e.outcome()) {
      case ::umbp::ROUTE_PUT_OUTCOME_ROUTED:
        (*out)[i] = RoutePutResult{
            .outcome = RoutePutOutcome::kRouted,
            .node_id = e.node_id(),
            .peer_address = e.peer_address(),
            .tier = FromProtoTier(e.tier()),
        };
        break;
      case ::umbp::ROUTE_PUT_OUTCOME_ALREADY_EXISTS:
        // Non-nullopt so caller distinguishes dedup from unavailable (= nullopt).
        (*out)[i] = RoutePutResult{.outcome = RoutePutOutcome::kAlreadyExists};
        break;
      case ::umbp::ROUTE_PUT_OUTCOME_UNAVAILABLE:
      default:
        break;  // leave as nullopt
    }
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchRouteGet(const std::vector<std::string>& keys,
                                         const std::unordered_set<std::string>& exclude_nodes,
                                         std::vector<std::optional<RouteGetResult>>* out) {
  ScopedRpcTimer _rpc_timer(this, "BatchRouteGet");
  if (out == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "out is null");
  }
  out->clear();
  ::umbp::BatchRouteGetRequest req;
  req.set_node_id(config_.node_id);
  for (const auto& k : keys) req.add_keys(k);
  FillExcludeNodes(req.mutable_exclude_nodes(), exclude_nodes);

  ::umbp::BatchRouteGetResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchRouteGet(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) return status;

  out->resize(static_cast<size_t>(resp.entries_size()));
  for (int i = 0; i < resp.entries_size(); ++i) {
    const auto& e = resp.entries(i);
    if (!e.found()) continue;
    RouteGetResult r;
    r.node_id = e.node_id();
    r.tier = FromProtoTier(e.tier());
    r.size = e.size();
    r.peer_address = e.peer_address();
    (*out)[i] = std::move(r);
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchLookup(const std::vector<std::string>& keys,
                                       std::vector<bool>* out) {
  ScopedRpcTimer _rpc_timer(this, "BatchLookup");
  if (out == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "out is null");
  }
  out->clear();

  ::umbp::BatchLookupRequest req;
  req.set_node_id(config_.node_id);
  for (const auto& k : keys) req.add_keys(k);

  ::umbp::BatchLookupResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchLookup(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) return status;

  out->reserve(static_cast<size_t>(resp.found_size()));
  for (int i = 0; i < resp.found_size(); ++i) out->push_back(resp.found(i));
  return grpc::Status::OK;
}

void MasterClient::SetPeerDramAllocator(PeerDramAllocator* dram_alloc) {
  peer_alloc_ = dram_alloc;
  AddOwnedLocationSource(dram_alloc);
}

void MasterClient::SetPeerSsdManager(PeerSsdManager* ssd_manager) {
  ssd_manager_ = ssd_manager;
  AddOwnedLocationSource(ssd_manager);
}

void MasterClient::AddOwnedLocationSource(OwnedLocationSource* source) {
  if (source == nullptr) return;
  owned_sources_.push_back(source);
}

bool MasterClient::ClearFullSync() {
  std::lock_guard send_lock(hb_send_mutex_);
  if (!registered_) return false;

  auto caps = SnapshotAndCacheTierCapacities();
  std::map<TierType, uint64_t> kv_counts;
  if (peer_alloc_ != nullptr) kv_counts = peer_alloc_->OwnedKeyCountByTier();

  ::umbp::HeartbeatRequest req;
  req.set_node_id(config_.node_id);
  req.set_is_full_sync(true);
  FillTierCapacities(req.mutable_tier_capacities(), caps);
  FillTierKvCounts(req.mutable_tier_kv_counts(), kv_counts);
  {
    std::lock_guard state_lock(hb_state_mutex_);
    req.set_delta_seq_baseline(next_bundle_seq_ - 1);
  }
  // No bundles: ReplaceNodeLocations({}) drops every prior placement for this node.

  ::umbp::HeartbeatResponse resp;
  grpc::Status status = SendHeartbeatRpcLocked(req, &resp);
  if (!status.ok()) {
    MORI_UMBP_WARN("[Client] Clear full-sync RPC failed: {}", status.error_message());
    return false;
  }

  auto external_status = RevokeAllExternalKvBlocksForNode(config_.node_id);
  if (!external_status.ok()) {
    MORI_UMBP_WARN("[Client] Clear external KV revoke failed: {}", external_status.error_message());
    return false;
  }

  if (peer_alloc_ != nullptr) {
    peer_alloc_->ClearFullSyncAcked();
    MORI_UMBP_INFO("[Client] Clear full-sync acked by master; allocator writes re-enabled");
  }
  return true;
}

void MasterClient::StartHeartbeat() {
  if (!registered_) {
    MORI_UMBP_WARN("[Client] StartHeartbeat ignored: not registered");
    return;
  }
  if (heartbeat_running_) return;
  heartbeat_running_ = true;
  try {
    heartbeat_thread_ = std::thread(&MasterClient::HeartbeatLoop, this);
  } catch (const std::system_error& e) {
    heartbeat_running_ = false;
    MORI_UMBP_ERROR("[Client] Failed to start heartbeat thread: {}", e.what());
    return;
  }
  MORI_UMBP_INFO("[Client] Heartbeat thread started (interval={}ms)", heartbeat_interval_ms_);
}

void MasterClient::StopHeartbeat() {
  if (!heartbeat_running_) return;
  heartbeat_running_ = false;
  hb_cv_.notify_one();
  if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
  MORI_UMBP_INFO("[Client] Heartbeat thread stopped");
}

void MasterClient::HeartbeatLoop() {
  while (heartbeat_running_) {
    {
      std::unique_lock lock(hb_cv_mutex_);
      hb_cv_.wait_for(lock, std::chrono::milliseconds(heartbeat_interval_ms_),
                      [this] { return !heartbeat_running_.load(); });
    }
    if (!heartbeat_running_) break;
    SendHeartbeatOnce();
  }
}

std::map<TierType, TierCapacity> MasterClient::SnapshotAndCacheTierCapacities() {
  std::map<TierType, TierCapacity> caps;
  bool have_live = false;
  if (peer_alloc_ != nullptr) {
    caps = peer_alloc_->TierCapacitiesSnapshot();  // DRAM/HBM, bitmap-derived
    have_live = true;
  }
  if (ssd_manager_ != nullptr) {
    auto [used, total] = ssd_manager_->Capacity();
    if (total > 0) {
      const uint64_t avail = used < total ? total - used : 0;
      caps[TierType::SSD] = TierCapacity{total, avail};
      have_live = true;
    }
  }
  std::lock_guard lock(caps_mutex_);
  if (!have_live) return current_capacities_;
  // Fill tiers we have no live source for from the cached snapshot.
  for (auto& [t, c] : current_capacities_) {
    if (caps.find(t) == caps.end()) caps[t] = c;
  }
  current_capacities_ = caps;
  return caps;
}

bool MasterClient::SendHeartbeatOnce() {
  std::lock_guard send_lock(hb_send_mutex_);
  if (!registered_) return false;

  auto caps = SnapshotAndCacheTierCapacities();
  std::map<TierType, uint64_t> kv_counts;
  if (peer_alloc_ != nullptr) kv_counts = peer_alloc_->OwnedKeyCountByTier();

  bool do_full_sync;
  {
    std::lock_guard state_lock(hb_state_mutex_);
    do_full_sync = std::exchange(full_sync_pending_, false);
  }

  if (do_full_sync) {
    if (SendFullSyncHeartbeatLocked(caps, kv_counts)) return true;
    std::lock_guard state_lock(hb_state_mutex_);
    full_sync_pending_ = true;
    return false;
  }
  return SendDeltaHeartbeatLocked(caps, kv_counts);
}

bool MasterClient::SendFullSyncHeartbeatLocked(const std::map<TierType, TierCapacity>& caps,
                                               const std::map<TierType, uint64_t>& kv_counts) {
  ::umbp::HeartbeatRequest req;
  req.set_node_id(config_.node_id);
  req.set_is_full_sync(true);
  FillTierCapacities(req.mutable_tier_capacities(), caps);
  FillTierKvCounts(req.mutable_tier_kv_counts(), kv_counts);

  EventBundle snapshot;
  {
    std::lock_guard state_lock(hb_state_mutex_);
    snapshot.seq = next_bundle_seq_ - 1;
    req.set_delta_seq_baseline(snapshot.seq);
  }
  snapshot.events = SnapshotAllSources(owned_sources_);
  FillBundle(req.add_bundles(), snapshot);

  ::umbp::HeartbeatResponse resp;
  grpc::Status status = SendHeartbeatRpcLocked(req, &resp);
  if (!status.ok()) {
    MORI_UMBP_WARN("[Client] Full-sync heartbeat failed: error={}", status.error_message());
    return false;
  }
  return true;
}

bool MasterClient::SendDeltaHeartbeatLocked(const std::map<TierType, TierCapacity>& caps,
                                            const std::map<TierType, uint64_t>& kv_counts) {
  // Drain every owned-location source (DRAM allocator + SSD manager) and
  // concat into ONE bundle under ONE monotonic seq — never one seq per source,
  // which would break ack / seq-gap full-sync recovery.
  auto new_events = DrainAllSources(owned_sources_);
  if (!new_events.empty()) {
    std::lock_guard state_lock(hb_state_mutex_);
    outbox_.push_back(EventBundle{next_bundle_seq_++, std::move(new_events)});
  }

  ::umbp::HeartbeatRequest req;
  req.set_node_id(config_.node_id);
  req.set_is_full_sync(false);
  FillTierCapacities(req.mutable_tier_capacities(), caps);
  FillTierKvCounts(req.mutable_tier_kv_counts(), kv_counts);
  {
    std::lock_guard state_lock(hb_state_mutex_);
    for (const auto& bundle : outbox_) {
      if (bundle.seq > hb_last_acked_seq_) FillBundle(req.add_bundles(), bundle);
    }
  }

  ::umbp::HeartbeatResponse resp;
  grpc::Status status = SendHeartbeatRpcLocked(req, &resp);
  if (!status.ok()) {
    MORI_UMBP_WARN("[Client] Heartbeat failed: node_id={}, error={}", config_.node_id,
                   status.error_message());
    return false;
  }

  if (resp.status() == ::umbp::CLIENT_STATUS_UNKNOWN) {
    MORI_UMBP_WARN("[Client] Master does not recognize us; re-registering...");
    registered_ = false;
    ::umbp::RegisterClientRequest re_req;
    re_req.set_node_id(config_.node_id);
    re_req.set_node_address(config_.node_address);
    FillTierCapacities(re_req.mutable_tier_capacities(), caps);
    for (const auto& tag : config_.tags) re_req.add_tags(tag);
    ::umbp::RegisterClientResponse re_resp;
    grpc::ClientContext re_ctx;
    grpc::Status re_status;
    {
      ScopedRpcTimer _rpc_timer(this, "RegisterClient");
      re_status = GetStub(stub_.get())->RegisterClient(&re_ctx, re_req, &re_resp);
      _rpc_timer.SetStatus(re_status);
    }
    if (re_status.ok() || re_status.error_code() == grpc::StatusCode::ALREADY_EXISTS) {
      registered_ = true;
      std::lock_guard state_lock(hb_state_mutex_);
      hb_last_acked_seq_ = 0;
      full_sync_pending_ = true;
      MORI_UMBP_INFO("[Client] Re-registered with master after UNKNOWN status");
      return true;
    }
    MORI_UMBP_WARN("[Client] Re-registration failed: {}", re_status.error_message());
    return false;
  }

  // Recover from a master-reported seq gap within the same tick.
  if (resp.request_full_sync()) {
    if (!SendFullSyncHeartbeatLocked(caps, kv_counts)) {
      std::lock_guard state_lock(hb_state_mutex_);
      full_sync_pending_ = true;
      return false;
    }
  }
  return true;
}

grpc::Status MasterClient::SendHeartbeatRpcLocked(::umbp::HeartbeatRequest& req,
                                                  ::umbp::HeartbeatResponse* resp) {
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(RpcShutdownTimeoutMs()));
  grpc::Status status;
  {
    ScopedRpcTimer _rpc_timer(this, "Heartbeat");
    status = GetStub(stub_.get())->Heartbeat(&ctx, req, resp);
    _rpc_timer.SetStatus(status);
  }
  if (!status.ok()) return status;
  std::lock_guard state_lock(hb_state_mutex_);
  hb_last_acked_seq_ = resp->acked_seq();
  while (!outbox_.empty() && outbox_.front().seq <= hb_last_acked_seq_) {
    outbox_.pop_front();
  }
  return status;
}

// ---------------------------------------------------------------------------
//  Client-side metrics: buffering API
// ---------------------------------------------------------------------------

namespace {
// Build a stable per-series key as "name|k1=v1|k2=v2...".  Pre-reserve the
// final length so the per-Observe hot path does at most one allocation —
// repeated += without reserve costs ~2 reallocs + memcpy on a typical
// 65-char rpc-latency key, which adds up at 100k RPC/s.
std::string MetricKey(const std::string& name, const MasterClient::Labels& labels) {
  std::size_t needed = name.size();
  for (const auto& [k, v] : labels) {
    needed += 2 + k.size() + v.size();  // '|' + k + '=' + v
  }
  std::string key;
  key.reserve(needed);
  key.append(name);
  for (const auto& [k, v] : labels) {
    key.push_back('|');
    key.append(k);
    key.push_back('=');
    key.append(v);
  }
  return key;
}
}  // namespace

void MasterClient::AddCounter(std::string name, std::string help, Labels labels, double delta) {
  std::lock_guard lock(metrics_mutex_);
  auto key = MetricKey(name, labels);
  auto& s = pending_counters_[key];
  s.name = std::move(name);
  s.help = std::move(help);
  s.labels = std::move(labels);
  s.value += delta;
}

void MasterClient::SetGauge(std::string name, std::string help, Labels labels, double value) {
  std::lock_guard lock(metrics_mutex_);
  auto key = MetricKey(name, labels);
  auto& s = pending_gauges_[key];
  s.name = std::move(name);
  s.help = std::move(help);
  s.labels = std::move(labels);
  s.value = value;
}

void MasterClient::Observe(std::string name, std::string help, Labels labels,
                           const std::vector<double>& bounds, double value) {
  std::lock_guard lock(metrics_mutex_);
  auto key = MetricKey(name, labels);
  auto [it, inserted] = pending_histogram_aggregates_.try_emplace(std::move(key));
  auto& h = it->second;
  if (inserted) {
    // Series-cardinality cap.  Bounds the map by # distinct (name, labels)
    // series, not by QPS — hitting it indicates a label-cardinality leak.
    if (pending_histogram_aggregates_.size() > pending_histogram_series_cap_) {
      pending_histogram_aggregates_.erase(it);
      metrics_dropped_count_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    h.name = std::move(name);
    h.help = std::move(help);
    h.labels = std::move(labels);
    h.bounds = bounds;
    h.bucket_counts.assign(bounds.size(), 0);
  } else if (h.bounds.size() != bounds.size() && !h.warned_mismatch) {
    // Per-accumulator dedup: each series may warn once on its own bounds
    // mismatch.  A process-wide once_flag would silence every other series
    // after the first WARN, defeating the point of surfacing the bug.
    h.warned_mismatch = true;
    MORI_UMBP_WARN("[Client] Observe: bounds mismatch on '{}' — first write wins", h.name);
  }
  // Cumulative bucket increments — every bucket whose upper bound is >= value
  // gets +1.  Mirrors MetricsServer::observe() so master merge is a plain
  // per-bucket add (no encoding conversion).
  for (std::size_t i = 0; i < h.bounds.size(); ++i) {
    if (value <= h.bounds[i]) ++h.bucket_counts[i];
  }
  ++h.count;
  h.sum += value;
}

void MasterClient::RecordRpcLatency(std::string_view method, bool ok, double seconds) {
  // Short-circuit on Python read-only clients (never registered) and during
  // teardown after StopMetricsReporting().  Avoids unbounded buffer growth
  // and any UAF window after the flush thread joins.
  if (!metrics_running_.load(std::memory_order_relaxed)) return;
  // Built once per process: avoids constructing a 14-double vector on every
  // monitored RPC.  Observe takes bounds by const-ref so this is alloc-free.
  static const std::vector<double> kBounds(std::begin(kMasterClientRpcLatencyBucketsArr),
                                           std::end(kMasterClientRpcLatencyBucketsArr));
  Labels labels = {{"rpc", std::string(method)}, {"status", ok ? "ok" : "error"}};
  Observe(MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY,
          MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY_HELP, std::move(labels), kBounds, seconds);
}

void MasterClient::RecordRpcError(std::string_view method, std::string_view code) {
  if (!metrics_running_.load(std::memory_order_relaxed)) return;
  Labels labels = {{"rpc", std::string(method)}, {"code", std::string(code)}};
  AddCounter(MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL,
             MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL_HELP, std::move(labels), 1.0);
}

void MasterClient::AddMetricsProvider(std::function<void()> provider) {
  // Reject late registration: MetricsLoop reads metrics_providers_ lock-free, so
  // adding after the thread starts would race the reader (see header).
  if (metrics_running_.load(std::memory_order_relaxed)) {
    MORI_UMBP_ERROR(
        "[Client] AddMetricsProvider called after the metrics thread started; ignoring "
        "(providers must be registered before RegisterSelf)");
    return;
  }
  if (provider) metrics_providers_.push_back(std::move(provider));
}

void MasterClient::StartMetricsReporting() {
  if (!registered_) return;
  if (metrics_running_) return;
  metrics_running_ = true;
  try {
    metrics_thread_ = std::thread(&MasterClient::MetricsLoop, this);
  } catch (const std::system_error& e) {
    metrics_running_ = false;
    MORI_UMBP_ERROR("[Client] Failed to start metrics thread: {}", e.what());
    return;
  }
  MORI_UMBP_INFO("[Client] Metrics reporting thread started (interval={}ms)", metrics_interval_ms_);
}

void MasterClient::StopMetricsReporting() {
  if (!metrics_running_) return;
  metrics_running_ = false;
  metrics_cv_.notify_one();
  if (metrics_thread_.joinable()) metrics_thread_.join();
  MORI_UMBP_INFO("[Client] Metrics reporting thread stopped");
}

void MasterClient::MetricsLoop() {
  while (metrics_running_) {
    {
      std::unique_lock lock(metrics_cv_mutex_);
      metrics_cv_.wait_for(lock, std::chrono::milliseconds(metrics_interval_ms_),
                           [this] { return !metrics_running_.load(); });
    }
    if (!metrics_running_) break;
    FlushMetricsOnce();
  }
  // Final flush: ship the last sub-interval of provider deltas before the
  // thread exits.  PoolClient::Shutdown calls StopMetricsReporting() BEFORE
  // UnregisterSelf, so the master is still reachable here; without this final
  // flush the last (<metrics_interval_ms_) of SSD counter deltas would be
  // dropped at shutdown.
  FlushMetricsOnce();
}

void MasterClient::FlushMetricsOnce() {
  // Let registered providers publish their latest counters/gauges into the
  // pending buffers BEFORE we swap them out.  Runs in the metrics thread (no
  // extra thread); providers are set before the thread starts.
  for (const auto& provider : metrics_providers_) {
    if (provider) provider();
  }

  std::unordered_map<std::string, PendingSample> counters;
  std::unordered_map<std::string, PendingSample> gauges;
  std::unordered_map<std::string, HistogramAccumulator> histogram_aggregates;
  {
    std::lock_guard lock(metrics_mutex_);
    counters.swap(pending_counters_);
    gauges.swap(pending_gauges_);
    histogram_aggregates.swap(pending_histogram_aggregates_);
  }
  auto dropped_delta = metrics_dropped_count_.exchange(0, std::memory_order_relaxed);
  if (counters.empty() && gauges.empty() && histogram_aggregates.empty() && dropped_delta == 0)
    return;

  ::umbp::ReportMetricsRequest req;
  req.set_node_id(config_.node_id);

  for (const auto& [key, s] : counters) {
    auto* sample = req.add_metrics();
    sample->set_name(s.name);
    sample->set_help(s.help);
    for (const auto& [k, v] : s.labels) {
      auto* l = sample->add_labels();
      l->set_name(k);
      l->set_value(v);
    }
    sample->set_counter_delta(s.value);
  }
  for (const auto& [key, s] : gauges) {
    auto* sample = req.add_metrics();
    sample->set_name(s.name);
    sample->set_help(s.help);
    for (const auto& [k, v] : s.labels) {
      auto* l = sample->add_labels();
      l->set_name(k);
      l->set_value(v);
    }
    sample->set_gauge_value(s.value);
  }
  for (const auto& [key, h] : histogram_aggregates) {
    auto* sample = req.add_metrics();
    sample->set_name(h.name);
    sample->set_help(h.help);
    for (const auto& [k, v] : h.labels) {
      auto* l = sample->add_labels();
      l->set_name(k);
      l->set_value(v);
    }
    auto* agg = sample->mutable_histogram_aggregate();
    for (double b : h.bounds) agg->add_bounds(b);
    for (uint64_t c : h.bucket_counts) agg->add_bucket_counts(c);
    agg->set_count(h.count);
    agg->set_sum(h.sum);
  }
  if (dropped_delta > 0) {
    auto* sample = req.add_metrics();
    sample->set_name(MORI_UMBP_METRIC_MASTER_CLIENT_METRICS_DROPPED_TOTAL);
    sample->set_help(MORI_UMBP_METRIC_MASTER_CLIENT_METRICS_DROPPED_TOTAL_HELP);
    sample->set_counter_delta(static_cast<double>(dropped_delta));
  }

  ::umbp::ReportMetricsResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(RpcShutdownTimeoutMs()));
  auto status = GetStub(stub_.get())->ReportMetrics(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_WARN("[Client] ReportMetrics RPC failed: node_id={}, error={}", config_.node_id,
                   status.error_message());
  }
}

// ---------------------------------------------------------------------------
//  External KV block events
// ---------------------------------------------------------------------------

grpc::Status MasterClient::ReportExternalKvBlocks(const std::string& node_id,
                                                  const std::vector<std::string>& hashes,
                                                  TierType tier) {
  ScopedRpcTimer _rpc_timer(this, "ReportExternalKvBlocks");
  ::umbp::ReportExternalKvBlocksRequest req;
  req.set_node_id(node_id);
  req.set_tier(ToProtoTier(tier));
  for (const auto& hash : hashes) req.add_hashes(hash);

  ::umbp::ReportExternalKvBlocksResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(RpcShutdownTimeoutMs()));
  auto status = GetStub(stub_.get())->ReportExternalKvBlocks(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  return status;
}

grpc::Status MasterClient::RevokeExternalKvBlocks(const std::string& node_id,
                                                  const std::vector<std::string>& hashes,
                                                  TierType tier) {
  ScopedRpcTimer _rpc_timer(this, "RevokeExternalKvBlocks");
  ::umbp::RevokeExternalKvBlocksRequest req;
  req.set_node_id(node_id);
  req.set_tier(ToProtoTier(tier));
  for (const auto& hash : hashes) req.add_hashes(hash);

  ::umbp::RevokeExternalKvBlocksResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(RpcShutdownTimeoutMs()));
  auto status = GetStub(stub_.get())->RevokeExternalKvBlocks(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  return status;
}

grpc::Status MasterClient::RevokeAllExternalKvBlocksAtTier(const std::string& node_id,
                                                           TierType tier) {
  ScopedRpcTimer _rpc_timer(this, "RevokeAllExternalKvBlocksAtTier");
  ::umbp::RevokeAllExternalKvBlocksAtTierRequest req;
  req.set_node_id(node_id);
  req.set_tier(ToProtoTier(tier));

  ::umbp::RevokeAllExternalKvBlocksAtTierResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(RpcShutdownTimeoutMs()));
  auto status = GetStub(stub_.get())->RevokeAllExternalKvBlocksAtTier(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  return status;
}

grpc::Status MasterClient::RevokeAllExternalKvBlocksForNode(const std::string& node_id) {
  ScopedRpcTimer _rpc_timer(this, "RevokeAllExternalKvBlocksForNode");
  ::umbp::RevokeAllExternalKvBlocksForNodeRequest req;
  req.set_node_id(node_id);

  ::umbp::RevokeAllExternalKvBlocksForNodeResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(RpcShutdownTimeoutMs()));
  auto status = GetStub(stub_.get())->RevokeAllExternalKvBlocksForNode(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  return status;
}

grpc::Status MasterClient::MatchExternalKv(const std::vector<std::string>& hashes,
                                           std::vector<ExternalKvNodeMatch>* out_matches,
                                           bool count_as_hit) {
  ScopedRpcTimer _rpc_timer(this, "MatchExternalKv");
  ::umbp::MatchExternalKvRequest req;
  for (const auto& h : hashes) req.add_hashes(h);
  req.set_count_as_hit(count_as_hit);
  ::umbp::MatchExternalKvResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->MatchExternalKv(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) return status;
  if (out_matches != nullptr) {
    for (const auto& m : resp.matches()) {
      ExternalKvNodeMatch out;
      out.node_id = m.node_id();
      out.peer_address = m.peer_address();
      for (const auto& bucket : m.hashes_by_tier()) {
        auto& vec = out.hashes_by_tier[FromProtoTier(bucket.tier())];
        vec.assign(bucket.hashes().begin(), bucket.hashes().end());
      }
      out_matches->push_back(std::move(out));
    }
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes, std::vector<ExternalKvHitCountEntry>* out_entries) {
  ScopedRpcTimer _rpc_timer(this, "GetExternalKvHitCounts");
  ::umbp::GetExternalKvHitCountsRequest req;
  for (const auto& h : hashes) req.add_hashes(h);
  ::umbp::GetExternalKvHitCountsResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->GetExternalKvHitCounts(&ctx, req, &resp);
  _rpc_timer.SetStatus(status);
  if (!status.ok()) return status;
  if (out_entries != nullptr) {
    out_entries->clear();
    out_entries->reserve(static_cast<size_t>(resp.entries_size()));
    for (const auto& entry : resp.entries()) {
      ExternalKvHitCountEntry out;
      out.hash = entry.hash();
      out.hit_count_total = entry.hit_count_total();
      out_entries->push_back(std::move(out));
    }
  }
  return grpc::Status::OK;
}

}  // namespace mori::umbp

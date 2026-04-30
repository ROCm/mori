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
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp.grpc.pb.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/external_kv_block_index.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/routing/router.h"

namespace mori::umbp {

namespace {
// Recommended-heartbeat interval = heartbeat_ttl / divisor.  Kept at 2 by
// default (interval = half the TTL).  Overridable via
// UMBP_HEARTBEAT_INTERVAL_DIVISOR; min_allowed=1 guards against div-by-zero.
uint32_t HeartbeatIntervalDivisor() {
  static const uint32_t v = GetEnvUint32("UMBP_HEARTBEAT_INTERVAL_DIVISOR", 2, /*min_allowed=*/1);
  return v;
}

std::chrono::seconds GrpcShutdownDeadline() {
  static const auto v =
      GetEnvSeconds("UMBP_GRPC_SHUTDOWN_DEADLINE_SEC", std::chrono::seconds(3), /*min_allowed=*/1);
  return v;
}
}  // namespace

// Out-of-line to keep config.h free of RouteGetStrategy / RoutePutStrategy
// includes; see the declaration in umbp/distributed/config.h.
MasterServerConfig MasterServerConfig::FromEnvironment() {
  MasterServerConfig cfg;
  cfg.registry_config = ClientRegistryConfig::FromEnvironment();
  cfg.eviction_config = EvictionConfig::FromEnvironment();
  return cfg;
}

static Location ToLocation(const ::umbp::Location& proto_location) {
  Location location;
  location.node_id = proto_location.node_id();
  location.location_id = proto_location.location_id();
  location.size = proto_location.size();
  location.tier = static_cast<TierType>(proto_location.tier());
  return location;
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

  grpc::Status RegisterClient(grpc::ServerContext* /*context*/,
                              const ::umbp::RegisterClientRequest* request,
                              ::umbp::RegisterClientResponse* response) override {
    // Convert proto TierCapacity → C++ types
    std::map<TierType, TierCapacity> caps;
    for (const auto& tc : request->tier_capacities()) {
      TierCapacity c;
      c.total_bytes = tc.total_capacity_bytes();
      c.available_bytes = tc.available_capacity_bytes();
      caps[static_cast<TierType>(tc.tier())] = c;
    }

    const auto& engine_desc_str = request->engine_desc();
    std::vector<uint8_t> engine_desc_bytes(engine_desc_str.begin(), engine_desc_str.end());

    std::vector<std::vector<uint8_t>> dram_memory_desc_bytes_list;
    for (const auto& desc : request->dram_memory_descs()) {
      dram_memory_desc_bytes_list.emplace_back(desc.begin(), desc.end());
    }

    std::vector<uint64_t> dram_buffer_sizes(request->dram_buffer_sizes().begin(),
                                            request->dram_buffer_sizes().end());

    std::vector<uint64_t> ssd_store_capacities(request->ssd_store_capacities().begin(),
                                               request->ssd_store_capacities().end());

    const bool registered = registry_.RegisterClient(
        request->node_id(), request->node_address(), caps, request->peer_address(),
        engine_desc_bytes, dram_memory_desc_bytes_list, dram_buffer_sizes, ssd_store_capacities,
        request->dram_page_size());
    if (!registered) {
      return grpc::Status(grpc::StatusCode::ALREADY_EXISTS,
                          "node is already alive and cannot be re-registered");
    }

    UpdateClientCountMetric();
    UpdateClientCapacityMetrics(request->node_id(), caps);

    // Recommend heartbeat = heartbeat_ttl / divisor (default 2 => half TTL).
    auto interval_ms =
        static_cast<uint64_t>(config_.heartbeat_ttl.count() * 1000) / HeartbeatIntervalDivisor();
    response->set_heartbeat_interval_ms(interval_ms);

    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext* /*context*/,
                                const ::umbp::UnregisterClientRequest* request,
                                ::umbp::UnregisterClientResponse* response) override {
    size_t removed = registry_.UnregisterClient(request->node_id());
    response->set_keys_removed(static_cast<uint32_t>(removed));
    UpdateClientCountMetric();
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext* /*context*/, const ::umbp::HeartbeatRequest* request,
                         ::umbp::HeartbeatResponse* response) override {
    std::map<TierType, TierCapacity> caps;
    for (const auto& tc : request->tier_capacities()) {
      TierCapacity c;
      c.total_bytes = tc.total_capacity_bytes();
      c.available_bytes = tc.available_capacity_bytes();
      caps[static_cast<TierType>(tc.tier())] = c;
    }

    ClientStatus status = registry_.Heartbeat(request->node_id(), caps);
    response->set_status(static_cast<::umbp::ClientStatus>(status));

    UpdateClientCapacityMetrics(request->node_id(), caps);

    MORI_UMBP_INFO("[Master] Heartbeat received: node_id={}, tiers={}, status={}",
                   request->node_id(), request->tier_capacities_size(), ClientStatusName(status));

    return grpc::Status::OK;
  }

  grpc::Status Register(grpc::ServerContext* /*context*/, const ::umbp::RegisterRequest* request,
                        ::umbp::RegisterResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id cannot be empty");
    }
    if (request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
    }
    if (!registry_.IsClientAlive(request->node_id())) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "node is not registered/alive");
    }

    Location location = ToLocation(request->location());
    if (location.node_id.empty()) {
      location.node_id = request->node_id();
    }

    index_.Register(request->node_id(), request->key(), location);
    MORI_UMBP_INFO("[Master] Register key: node_id={}, key={}, location_id={}, size={}, tier={}",
                   request->node_id(), request->key(), location.location_id, location.size,
                   TierTypeName(location.tier));
    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_REGISTER_TOTAL, MORI_UMBP_METRIC_REGISTER_TOTAL_HELP);
    }
    return grpc::Status::OK;
  }

  grpc::Status Unregister(grpc::ServerContext* /*context*/,
                          const ::umbp::UnregisterRequest* request,
                          ::umbp::UnregisterResponse* response) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id cannot be empty");
    }
    if (request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
    }

    Location location = ToLocation(request->location());
    if (location.node_id.empty()) {
      location.node_id = request->node_id();
    }

    const bool removed = index_.Unregister(request->node_id(), request->key(), location);
    response->set_removed(removed ? 1u : 0u);

    if (removed && location.size > 0) {
      // Registry parses both the page-bitmap "0:p3,4;1:p0" DRAM/HBM format
      // and the SSD format from location.location_id, so pass it through.
      registry_.DeallocateForUnregister(location.node_id, location);
    }

    MORI_UMBP_INFO("[Master] Unregister key: node_id={}, key={}, location_id={}, removed={}",
                   request->node_id(), request->key(), location.location_id, response->removed());
    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_UNREGISTER_TOTAL,
                           MORI_UMBP_METRIC_UNREGISTER_TOTAL_HELP);
    }
    return grpc::Status::OK;
  }

  grpc::Status Lookup(grpc::ServerContext* /*context*/, const ::umbp::LookupRequest* request,
                      ::umbp::LookupResponse* response) override {
    if (request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
    }

    auto locations = index_.Lookup(request->key());
    response->set_found(!locations.empty());
    if (metrics_) {
      for (const auto& loc : locations) {
        metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_LOOKUP, MORI_UMBP_METRIC_CLIENT_LOOKUP_HELP,
                             {{"node", loc.node_id}});
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status FinalizeAllocation(grpc::ServerContext* /*context*/,
                                  const ::umbp::FinalizeRequest* request,
                                  ::umbp::FinalizeResponse* response) override {
    if (request->node_id().empty() || request->key().empty() || request->allocation_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "node_id/key/allocation_id cannot be empty");
    }

    Location location = ToLocation(request->location());
    if (location.node_id.empty()) {
      location.node_id = request->node_id();
    }

    const bool finalized = registry_.FinalizeAllocation(location.node_id, request->key(), location,
                                                        request->allocation_id());
    if (finalized && request->depth() >= 0) {
      index_.SetDepth(request->key(), request->depth());
    }
    response->set_finalized(finalized);
    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_FINALIZE_ALLOCATION_TOTAL,
                           MORI_UMBP_METRIC_FINALIZE_ALLOCATION_TOTAL_HELP);
    }
    return grpc::Status::OK;
  }

  grpc::Status PublishLocalBlock(grpc::ServerContext* /*context*/,
                                 const ::umbp::PublishRequest* request,
                                 ::umbp::PublishResponse* response) override {
    if (request->node_id().empty() || request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id/key cannot be empty");
    }

    Location location = ToLocation(request->location());
    if (location.node_id.empty()) {
      location.node_id = request->node_id();
    }

    const bool published =
        registry_.PublishLocalBlock(request->node_id(), request->key(), location);
    response->set_published(published);
    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_PUBLISH_LOCAL_BLOCK_TOTAL,
                           MORI_UMBP_METRIC_PUBLISH_LOCAL_BLOCK_TOTAL_HELP);
    }
    return grpc::Status::OK;
  }

  grpc::Status AbortAllocation(grpc::ServerContext* /*context*/,
                               const ::umbp::AbortAllocationRequest* request,
                               ::umbp::AbortAllocationResponse* response) override {
    if (request->node_id().empty() || request->allocation_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "node_id/allocation_id cannot be empty");
    }

    const bool aborted =
        registry_.AbortAllocation(request->node_id(), request->allocation_id(), request->size());
    response->set_aborted(aborted);
    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_ABORT_ALLOCATION_TOTAL,
                           MORI_UMBP_METRIC_ABORT_ALLOCATION_TOTAL_HELP);
    }
    return grpc::Status::OK;
  }

  grpc::Status RouteGet(grpc::ServerContext* /*context*/, const ::umbp::RouteGetRequest* request,
                        ::umbp::RouteGetResponse* response) override {
    if (request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
    }

    auto result = router_.RouteGet(request->key(), request->node_id());
    if (!result.has_value()) {
      response->set_found(false);
      return grpc::Status::OK;
    }

    response->set_found(true);
    auto* source = response->mutable_source();
    source->set_node_id(result->node_id);
    source->set_location_id(result->location_id);
    source->set_size(result->size);
    source->set_tier(static_cast<::umbp::TierType>(result->tier));

    auto io_info = registry_.GetClientIOInfo(result->node_id, /*buffer_index=*/0);
    if (io_info) {
      response->set_peer_address(io_info->peer_address);
      response->set_engine_desc(io_info->engine_desc_bytes.data(),
                                io_info->engine_desc_bytes.size());
    }

    if (result->tier == TierType::DRAM || result->tier == TierType::HBM) {
      auto parsed = ParseDramLocationId(result->location_id);
      if (parsed) {
        auto descs = registry_.GetDramMemoryDescsForPages(result->node_id, parsed->pages);
        if (descs) {
          for (const auto& bd : *descs) {
            auto* proto_bd = response->add_dram_memory_descs();
            proto_bd->set_buffer_index(bd.buffer_index);
            proto_bd->set_desc(bd.desc_bytes.data(), bd.desc_bytes.size());
          }
        }
        // page_size of the source node's DRAM/HBM allocator.  Falls back to
        // the registry-wide default if the source node has no live allocator
        // (e.g. it expired between RouteGet's index lookup and this query).
        auto src_ps = registry_.GetNodeDramPageSize(result->node_id, result->tier);
        response->set_page_size(src_ps.value_or(config_.default_dram_page_size));
      } else {
        MORI_UMBP_ERROR("[Master] RouteGet: malformed DRAM/HBM location_id '{}' for key='{}'",
                        result->location_id, request->key());
      }
    }

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_ROUTE_GET,
                           MORI_UMBP_METRIC_CLIENT_ROUTE_GET_HELP, {{"node", result->node_id}});
    }
    MORI_UMBP_INFO("[Master] RouteGet key='{}': node={}, location={}", request->key(),
                   result->node_id, result->location_id);
    return grpc::Status::OK;
  }

  grpc::Status RoutePut(grpc::ServerContext* /*context*/, const ::umbp::RoutePutRequest* request,
                        ::umbp::RoutePutResponse* response) override {
    if (request->key().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
    }

    auto result = router_.RoutePut(request->key(), request->node_id(), request->block_size());
    if (!result.has_value()) {
      response->set_found(false);
      return grpc::Status::OK;
    }

    response->set_found(true);
    response->set_node_id(result->node_id);
    response->set_node_address(result->node_address);
    response->set_tier(static_cast<::umbp::TierType>(result->tier));
    response->set_peer_address(result->peer_address);
    response->set_engine_desc(result->engine_desc_bytes.data(), result->engine_desc_bytes.size());
    response->set_allocation_id(result->allocation_id);

    response->set_location_id(result->location_id);
    for (const auto& p : result->pages) {
      auto* proto_p = response->add_pages();
      proto_p->set_buffer_index(p.buffer_index);
      proto_p->set_page_index(p.page_index);
    }
    for (const auto& bd : result->dram_memory_descs) {
      auto* proto_bd = response->add_dram_memory_descs();
      proto_bd->set_buffer_index(bd.buffer_index);
      proto_bd->set_desc(bd.desc_bytes.data(), bd.desc_bytes.size());
    }
    response->set_page_size(result->page_size);

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_ROUTE_PUT,
                           MORI_UMBP_METRIC_CLIENT_ROUTE_PUT_HELP, {{"node", result->node_id}});
    }
    MORI_UMBP_INFO("[Master] RoutePut key='{}': target_node={}, tier={}, location='{}', pages={}",
                   request->key(), result->node_id, TierTypeName(result->tier), result->location_id,
                   result->pages.size());
    return grpc::Status::OK;
  }

  grpc::Status BatchRoutePut(grpc::ServerContext* /*context*/,
                             const ::umbp::BatchRoutePutRequest* request,
                             ::umbp::BatchRoutePutResponse* response) override {
    if (request->keys_size() != request->block_sizes_size()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "keys and block_sizes must have the same length");
    }

    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    std::vector<uint64_t> block_sizes(request->block_sizes().begin(), request->block_sizes().end());

    auto results = router_.BatchRoutePut(keys, request->node_id(), block_sizes);

    for (size_t i = 0; i < results.size(); ++i) {
      auto* entry = response->add_entries();
      if (!results[i].has_value()) {
        entry->set_found(false);
        continue;
      }
      auto& r = *results[i];
      entry->set_found(true);
      entry->set_node_id(r.node_id);
      entry->set_node_address(r.node_address);
      entry->set_tier(static_cast<::umbp::TierType>(r.tier));
      entry->set_peer_address(r.peer_address);
      entry->set_engine_desc(r.engine_desc_bytes.data(), r.engine_desc_bytes.size());
      entry->set_allocation_id(r.allocation_id);

      entry->set_location_id(r.location_id);
      for (const auto& p : r.pages) {
        auto* proto_p = entry->add_pages();
        proto_p->set_buffer_index(p.buffer_index);
        proto_p->set_page_index(p.page_index);
      }
      for (const auto& bd : r.dram_memory_descs) {
        auto* proto_bd = entry->add_dram_memory_descs();
        proto_bd->set_buffer_index(bd.buffer_index);
        proto_bd->set_desc(bd.desc_bytes.data(), bd.desc_bytes.size());
      }
      entry->set_page_size(r.page_size);
    }

    if (metrics_) {
      for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].has_value()) {
          metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT,
                               MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT_HELP,
                               {{"node", results[i]->node_id}});
        }
      }
    }
    MORI_UMBP_INFO("[Master] BatchRoutePut: {} keys from node={}", keys.size(), request->node_id());
    return grpc::Status::OK;
  }

  grpc::Status BatchRouteGet(grpc::ServerContext* /*context*/,
                             const ::umbp::BatchRouteGetRequest* request,
                             ::umbp::BatchRouteGetResponse* response) override {
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());

    auto results = router_.BatchRouteGet(keys, request->node_id());

    for (size_t i = 0; i < results.size(); ++i) {
      auto* entry = response->add_entries();
      if (!results[i].has_value()) {
        entry->set_found(false);
        continue;
      }
      const auto& loc = *results[i];
      entry->set_found(true);
      auto* source = entry->mutable_source();
      source->set_node_id(loc.node_id);
      source->set_location_id(loc.location_id);
      source->set_size(loc.size);
      source->set_tier(static_cast<::umbp::TierType>(loc.tier));

      auto io_info = registry_.GetClientIOInfo(loc.node_id, /*buffer_index=*/0);
      if (io_info) {
        entry->set_peer_address(io_info->peer_address);
        entry->set_engine_desc(io_info->engine_desc_bytes.data(),
                               io_info->engine_desc_bytes.size());
      }

      if (loc.tier == TierType::DRAM || loc.tier == TierType::HBM) {
        auto parsed = ParseDramLocationId(loc.location_id);
        if (parsed) {
          auto descs = registry_.GetDramMemoryDescsForPages(loc.node_id, parsed->pages);
          if (descs) {
            for (const auto& bd : *descs) {
              auto* proto_bd = entry->add_dram_memory_descs();
              proto_bd->set_buffer_index(bd.buffer_index);
              proto_bd->set_desc(bd.desc_bytes.data(), bd.desc_bytes.size());
            }
          }
          auto src_ps = registry_.GetNodeDramPageSize(loc.node_id, loc.tier);
          entry->set_page_size(src_ps.value_or(config_.default_dram_page_size));
        } else {
          MORI_UMBP_ERROR(
              "[Master] BatchRouteGet: malformed DRAM/HBM location_id '{}' for key='{}'",
              loc.location_id, request->keys(static_cast<int>(i)));
        }
      }
    }

    if (metrics_) {
      for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].has_value()) {
          metrics_->addCounter(MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET,
                               MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET_HELP,
                               {{"node", results[i]->node_id}});
        }
      }
    }
    MORI_UMBP_INFO("[Master] BatchRouteGet: {} keys from node={}", keys.size(), request->node_id());
    return grpc::Status::OK;
  }

  grpc::Status BatchLookup(grpc::ServerContext* /*context*/,
                           const ::umbp::BatchLookupRequest* request,
                           ::umbp::BatchLookupResponse* response) override {
    std::vector<std::string> keys(request->keys().begin(), request->keys().end());
    // Single shared_lock on the index for the whole batch.  Read-only:
    // no access-count bump and no lease grant (same semantics as Lookup).
    auto found = index_.BatchLookupExists(keys);
    response->mutable_found()->Reserve(static_cast<int>(found.size()));
    for (bool f : found) {
      response->add_found(f);
    }

    if (metrics_) {
      uint64_t found_count = 0;
      for (bool f : found) {
        if (f) ++found_count;
      }
      metrics_->addCounter(MORI_UMBP_METRIC_BATCH_LOOKUP_TOTAL,
                           MORI_UMBP_METRIC_BATCH_LOOKUP_TOTAL_HELP);
      metrics_->addCounter(MORI_UMBP_METRIC_BATCH_LOOKUP_KEYS_TOTAL,
                           MORI_UMBP_METRIC_BATCH_LOOKUP_KEYS_TOTAL_HELP,
                           static_cast<uint64_t>(keys.size()));
      metrics_->addCounter(MORI_UMBP_METRIC_BATCH_LOOKUP_FOUND_TOTAL,
                           MORI_UMBP_METRIC_BATCH_LOOKUP_FOUND_TOTAL_HELP, found_count);
    }
    MORI_UMBP_DEBUG("[Master] BatchLookup: {} keys from node={}", keys.size(), request->node_id());
    return grpc::Status::OK;
  }

  grpc::Status BatchFinalizeAllocation(grpc::ServerContext* /*context*/,
                                       const ::umbp::BatchFinalizeRequest* request,
                                       ::umbp::BatchFinalizeResponse* response) override {
    int n = request->keys_size();
    if (request->locations_size() != n || request->allocation_ids_size() != n) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "keys, locations, and allocation_ids must have the same length");
    }

    for (int i = 0; i < n; ++i) {
      Location location = ToLocation(request->locations(i));
      if (location.node_id.empty()) {
        location.node_id = request->node_id();
      }
      bool finalized = registry_.FinalizeAllocation(location.node_id, request->keys(i), location,
                                                    request->allocation_ids(i));
      if (finalized && i < request->depths_size() && request->depths(i) >= 0) {
        index_.SetDepth(request->keys(i), request->depths(i));
      }
      response->add_finalized(finalized);
    }

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_TOTAL,
                           MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_TOTAL_HELP);
      metrics_->addCounter(MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_KEYS_TOTAL,
                           MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_KEYS_TOTAL_HELP,
                           static_cast<uint64_t>(n));
    }
    MORI_UMBP_INFO("[Master] BatchFinalizeAllocation: {} keys from node={}", n, request->node_id());
    return grpc::Status::OK;
  }

  grpc::Status BatchAbortAllocation(grpc::ServerContext* /*context*/,
                                    const ::umbp::BatchAbortAllocationRequest* request,
                                    ::umbp::BatchAbortAllocationResponse* response) override {
    // Per-entry loop mirrors BatchFinalizeAllocation: each entry carries
    // its own target node_id, and registry_.AbortAllocation is idempotent
    // on racy failure (not-found / already-reaped returns false, which is
    // not an error at the RPC layer).
    const int n = request->entries_size();
    response->mutable_aborted()->Reserve(n);
    for (const auto& e : request->entries()) {
      if (e.node_id().empty() || e.allocation_id().empty()) {
        response->add_aborted(false);
        continue;
      }
      const bool aborted = registry_.AbortAllocation(e.node_id(), e.allocation_id(), e.size());
      response->add_aborted(aborted);
    }
    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_TOTAL,
                           MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_TOTAL_HELP);
      metrics_->addCounter(MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_ENTRIES_TOTAL,
                           MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_ENTRIES_TOTAL_HELP,
                           static_cast<uint64_t>(n));
    }
    MORI_UMBP_INFO("[Master] BatchAbortAllocation: {} entries", n);
    return grpc::Status::OK;
  }

  grpc::Status ReportExternalKvBlocks(
      grpc::ServerContext* /*context*/, const ::umbp::ReportExternalKvBlocksRequest* request,
      ::umbp::ReportExternalKvBlocksResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id cannot be empty");
    }
    if (request->hashes_size() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "hashes cannot be empty");
    }

    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    TierType tier = static_cast<TierType>(request->tier());

    registry_.RegisterExternalKvBlocks(request->node_id(), hashes, tier);
    MORI_UMBP_INFO("[Master] ReportExternalKvBlocks: node_id={}, hashes={}, tier={}",
                   request->node_id(), hashes.size(), TierTypeName(tier));

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
      grpc::ServerContext* /*context*/, const ::umbp::RevokeExternalKvBlocksRequest* request,
      ::umbp::RevokeExternalKvBlocksResponse* /*response*/) override {
    if (request->node_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "node_id cannot be empty");
    }
    if (request->hashes_size() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "hashes cannot be empty");
    }

    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());
    registry_.UnregisterExternalKvBlocks(request->node_id(), hashes);
    MORI_UMBP_INFO("[Master] RevokeExternalKvBlocks: node_id={}, hashes={}", request->node_id(),
                   hashes.size());

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

  grpc::Status MatchExternalKv(grpc::ServerContext* /*context*/,
                               const ::umbp::MatchExternalKvRequest* request,
                               ::umbp::MatchExternalKvResponse* response) override {
    std::vector<std::string> hashes(request->hashes().begin(), request->hashes().end());

    auto matches = external_kv_index_.Match(hashes);

    // Build a node_id -> peer_address lookup from alive clients.
    std::unordered_map<std::string, std::string> peer_map;
    for (const auto& record : registry_.GetAliveClients()) {
      peer_map[record.node_id] = record.peer_address;
    }

    for (auto& m : matches) {
      auto* proto_match = response->add_matches();
      proto_match->set_node_id(m.node_id);
      auto peer_it = peer_map.find(m.node_id);
      if (peer_it != peer_map.end()) {
        proto_match->set_peer_address(peer_it->second);
      }
      for (const auto& hash : m.matched_hashes) {
        proto_match->add_matched_hashes(hash);
      }
      proto_match->set_tier(static_cast<::umbp::TierType>(m.tier));
    }

    size_t total_matched_blocks = 0;
    for (const auto& m : matches) {
      total_matched_blocks += m.matched_hashes.size();
    }

    MORI_UMBP_INFO(
        "[Master] MatchExternalKv: queried_hashes={}, matched_nodes={}, matched_blocks={}",
        hashes.size(), matches.size(), total_matched_blocks);

    if (metrics_) {
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_MATCH_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_MATCH_TOTAL_HELP);
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_MATCH_QUERIED_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_MATCH_QUERIED_BLOCKS_TOTAL_HELP,
                           static_cast<uint64_t>(hashes.size()));
      metrics_->addCounter(MORI_UMBP_METRIC_EXT_KV_MATCH_MATCHED_BLOCKS_TOTAL,
                           MORI_UMBP_METRIC_EXT_KV_MATCH_MATCHED_BLOCKS_TOTAL_HELP,
                           static_cast<uint64_t>(total_matched_blocks));
    }

    return grpc::Status::OK;
  }

  grpc::Status ReportMetrics(grpc::ServerContext* /*context*/,
                             const ::umbp::ReportMetricsRequest* request,
                             ::umbp::ReportMetricsResponse* /*response*/) override {
    if (!metrics_) return grpc::Status::OK;

    // Log each batch so we can confirm receipt and see put/get byte deltas.
    {
      double put_bytes = 0, get_bytes = 0;
      for (const auto& s : request->metrics()) {
        if (s.name() == MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL &&
            s.value_case() == ::umbp::MetricSample::kCounterDelta)
          put_bytes += s.counter_delta();
        if (s.name() == MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL &&
            s.value_case() == ::umbp::MetricSample::kCounterDelta)
          get_bytes += s.counter_delta();
      }
      MORI_UMBP_INFO(
          "[Master] ReportMetrics: node={} samples={} put_bytes_delta={:.0f} "
          "get_bytes_delta={:.0f}",
          request->node_id(), request->metrics_size(), put_bytes, get_bytes);
    }

    mori::metrics::MetricsServer::Labels base = {{"node", request->node_id()}};

    for (const auto& s : request->metrics()) {
      mori::metrics::MetricsServer::Labels labels = base;
      for (const auto& l : s.labels()) {
        labels.push_back({l.name(), l.value()});
      }
      switch (s.value_case()) {
        case ::umbp::MetricSample::kCounterDelta:
          metrics_->addCounter(s.name(), s.help(), labels,
                               static_cast<uint64_t>(s.counter_delta()));
          break;
        case ::umbp::MetricSample::kGaugeValue:
          metrics_->setGauge(s.name(), s.help(), labels, s.gauge_value());
          break;
        case ::umbp::MetricSample::kHistogram: {
          std::vector<double> bounds(s.histogram().bounds().begin(), s.histogram().bounds().end());
          metrics_->observe(s.name(), s.help(), labels, bounds, s.histogram().value());
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
      eviction_manager_(
          std::make_unique<EvictionManager>(index_, registry_, config_.eviction_config)) {
  index_.SetClientRegistry(&registry_);
  router_.SetLeaseDuration(config_.eviction_config.lease_duration);
  registry_.SetExternalKvBlockIndex(&external_kv_index_);
}

MasterServer::~MasterServer() {
  // Shutdown() signals gRPC to stop but does not reset server_.  The
  // destructor is the single site that calls server_.reset(), ensuring the
  // grpc::Server object outlives any thread that is still inside Wait().
  // Callers that start Run() in a separate thread MUST join that thread
  // before destroying MasterServer (i.e., before reaching this destructor).
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
  if (eviction_manager_) {
    eviction_manager_->Stop();
  }
  if (server_) {
    const auto deadline = std::chrono::system_clock::now() + GrpcShutdownDeadline();
    MORI_UMBP_INFO("[Master] Shutting down");
    server_->Shutdown(deadline);
    // Intentionally do NOT reset server_ here — the thread running Run()/Wait()
    // may not have returned yet.  server_.reset() is deferred to ~MasterServer(),
    // which callers must invoke only after joining the server thread.
  }
  registry_.StopReaper();
}

}  // namespace mori::umbp

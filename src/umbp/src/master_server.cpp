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
#include "umbp/master_server.h"

#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "umbp.grpc.pb.h"
#include "umbp/router.h"

namespace mori::umbp {

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
  UMBPMasterServiceImpl(ClientRegistry& registry, BlockIndex& index, Router& router,
                        const ClientRegistryConfig& config)
      : registry_(registry), index_(index), router_(router), config_(config) {}

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

    const auto& dram_memory_desc_str = request->dram_memory_desc();
    std::vector<uint8_t> dram_memory_desc_bytes(dram_memory_desc_str.begin(),
                                                dram_memory_desc_str.end());

    const bool registered = registry_.RegisterClient(
        request->node_id(), request->node_address(), caps, request->peer_address(),
        engine_desc_bytes, dram_memory_desc_bytes);
    if (!registered) {
      return grpc::Status(grpc::StatusCode::ALREADY_EXISTS,
                          "node is already alive and cannot be re-registered");
    }

    // Recommend heartbeat at half the TTL
    auto interval_ms = static_cast<uint64_t>(config_.heartbeat_ttl.count() * 1000) / 2;
    response->set_heartbeat_interval_ms(interval_ms);

    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext* /*context*/,
                                const ::umbp::UnregisterClientRequest* request,
                                ::umbp::UnregisterClientResponse* response) override {
    size_t removed = registry_.UnregisterClient(request->node_id());
    response->set_keys_removed(static_cast<uint32_t>(removed));
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

    spdlog::info("[Master] Heartbeat received: node_id={}, tiers={}, status={}", request->node_id(),
                 request->tier_capacities_size(), ClientStatusName(status));

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
    spdlog::info("[Master] Register key: node_id={}, key={}, location_id={}, size={}, tier={}",
                 request->node_id(), request->key(), location.location_id, location.size,
                 TierTypeName(location.tier));
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
      uint64_t offset = 0;
      if (!location.location_id.empty()) {
        try {
          offset = std::stoull(location.location_id);
        } catch (...) {
          // location_id is not a numeric offset (e.g. SSD file path); use 0
        }
      }
      registry_.DeallocateForUnregister(location.node_id, location.tier, offset, location.size);
    }

    spdlog::info("[Master] Unregister key: node_id={}, key={}, location_id={}, removed={}",
                 request->node_id(), request->key(), location.location_id, response->removed());
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

    auto io_info = registry_.GetClientIOInfo(result->node_id);
    if (io_info) {
      response->set_peer_address(io_info->peer_address);
      response->set_engine_desc(io_info->engine_desc_bytes.data(),
                                io_info->engine_desc_bytes.size());
      response->set_dram_memory_desc(io_info->dram_memory_desc_bytes.data(),
                                     io_info->dram_memory_desc_bytes.size());
    }

    spdlog::info("[Master] RouteGet key='{}': node={}, location={}", request->key(),
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

    auto alloc_result =
        registry_.AllocateForPut(result->node_id, result->tier, request->block_size());
    if (!alloc_result) {
      response->set_found(false);
      return grpc::Status::OK;
    }

    response->set_found(true);
    response->set_node_id(result->node_id);
    response->set_node_address(result->node_address);
    response->set_tier(static_cast<::umbp::TierType>(result->tier));
    response->set_peer_address(alloc_result->peer_address);
    response->set_engine_desc(alloc_result->engine_desc_bytes.data(),
                              alloc_result->engine_desc_bytes.size());
    response->set_dram_memory_desc(alloc_result->dram_memory_desc_bytes.data(),
                                   alloc_result->dram_memory_desc_bytes.size());
    response->set_allocated_offset(alloc_result->allocated_offset);

    spdlog::info("[Master] RoutePut key='{}': target_node={}, tier={}, offset={}",
                 request->key(), result->node_id, TierTypeName(result->tier),
                 alloc_result->allocated_offset);
    return grpc::Status::OK;
  }

 private:
  ClientRegistry& registry_;
  BlockIndex& index_;
  Router& router_;
  ClientRegistryConfig config_;
};

// ---------------------------------------------------------------------------
//  MasterServer
// ---------------------------------------------------------------------------
MasterServer::MasterServer(MasterServerConfig config)
    : config_(std::move(config)),
      index_(),
      registry_(config_.registry_config, index_),
      router_(index_, registry_, std::move(config_.get_strategy), std::move(config_.put_strategy)),
      service_(std::make_unique<UMBPMasterServiceImpl>(registry_, index_, router_,
                                                       config_.registry_config)) {
  index_.SetClientRegistry(&registry_);
}

MasterServer::~MasterServer() { Shutdown(); }

void MasterServer::Run() {
  registry_.StartReaper();

  grpc::ServerBuilder builder;
  builder.AddListeningPort(config_.listen_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();

  spdlog::info("[Master] Listening on {}", config_.listen_address);
  server_->Wait();
}

void MasterServer::Shutdown() {
  if (server_) {
    // Use a deadline so Wait() unblocks even if RPCs do not drain quickly.
    const auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(3);
    spdlog::info("[Master] Shutting down");
    server_->Shutdown(deadline);
    server_.reset();
  }
  registry_.StopReaper();
}

}  // namespace mori::umbp

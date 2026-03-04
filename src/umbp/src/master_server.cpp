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

#include "umbp.grpc.pb.h"

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  gRPC service implementation
// ---------------------------------------------------------------------------
class MasterServer::UMBPMasterServiceImpl final : public ::umbp::UMBPMaster::Service {
 public:
  UMBPMasterServiceImpl(ClientRegistry& registry, const ClientRegistryConfig& config)
      : registry_(registry), config_(config) {}

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

    const bool registered =
        registry_.RegisterClient(request->client_id(), request->node_address(), caps);
    if (!registered) {
      return grpc::Status(grpc::StatusCode::ALREADY_EXISTS,
                          "client is already alive and cannot be re-registered");
    }

    // Recommend heartbeat at half the TTL
    auto interval_ms = static_cast<uint64_t>(config_.heartbeat_ttl.count() * 1000) / 2;
    response->set_heartbeat_interval_ms(interval_ms);

    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext* /*context*/,
                                const ::umbp::UnregisterClientRequest* request,
                                ::umbp::UnregisterClientResponse* response) override {
    size_t removed = registry_.UnregisterClient(request->client_id());
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

    ClientStatus status = registry_.Heartbeat(request->client_id(), caps);
    response->set_status(static_cast<::umbp::ClientStatus>(status));

    spdlog::info("[Master] Heartbeat received: client_id={}, tiers={}, status={}",
                 request->client_id(), request->tier_capacities_size(), ClientStatusName(status));

    return grpc::Status::OK;
  }

 private:
  ClientRegistry& registry_;
  ClientRegistryConfig config_;
};

// ---------------------------------------------------------------------------
//  MasterServer
// ---------------------------------------------------------------------------
MasterServer::MasterServer(MasterServerConfig config)
    : config_(std::move(config)),
      registry_(config_.registry_config),
      service_(std::make_unique<UMBPMasterServiceImpl>(registry_, config_.registry_config)) {}

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

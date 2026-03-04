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
#include "umbp/client.h"

#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>

#include <system_error>

#include "umbp.grpc.pb.h"

namespace mori::umbp {

// Helper: get the typed stub from the opaque pointer
static ::umbp::UMBPMaster::Stub* GetStub(void* ptr) {
  return static_cast<::umbp::UMBPMaster::Stub*>(ptr);
}

static void FillProtoLocation(const Location& location, ::umbp::Location* proto_location) {
  proto_location->set_node_id(location.node_id);
  proto_location->set_location_id(location.location_id);
  proto_location->set_size(location.size);
  proto_location->set_tier(static_cast<::umbp::TierType>(location.tier));
}

UMBPClient::UMBPClient(const UMBPClientConfig& config)
    : config_(config),
      stub_(nullptr, [](void* p) { delete static_cast<::umbp::UMBPMaster::Stub*>(p); }) {
  channel_ = grpc::CreateChannel(config.master_address, grpc::InsecureChannelCredentials());
  stub_.reset(::umbp::UMBPMaster::NewStub(channel_).release());
  spdlog::info("[Client] Created, master={}", config.master_address);
}

UMBPClient::~UMBPClient() {
  StopHeartbeat();
  if (registered_) {
    UnregisterSelf();
  }
}

grpc::Status UMBPClient::RegisterSelf(const std::map<TierType, TierCapacity>& tier_capacities) {
  if (registered_) {
    return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "node is already registered");
  }

  ::umbp::RegisterClientRequest req;
  req.set_node_id(config_.node_id);
  req.set_node_address(config_.node_address);
  for (const auto& [tier, cap] : tier_capacities) {
    auto* tc = req.add_tier_capacities();
    tc->set_tier(static_cast<::umbp::TierType>(tier));
    tc->set_total_capacity_bytes(cap.total_bytes);
    tc->set_available_capacity_bytes(cap.available_bytes);
  }

  ::umbp::RegisterClientResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->RegisterClient(&ctx, req, &resp);

  if (!status.ok()) {
    spdlog::error("[Client] RegisterClient failed: {}", status.error_message());
    return status;
  }

  heartbeat_interval_ms_ = resp.heartbeat_interval_ms();
  registered_ = true;

  {
    std::lock_guard lock(caps_mutex_);
    current_capacities_ = tier_capacities;
  }

  spdlog::info("[Client] Registered with master (heartbeat_interval={}ms)", heartbeat_interval_ms_);

  if (config_.auto_heartbeat) {
    StartHeartbeat();
  }

  return grpc::Status::OK;
}

grpc::Status UMBPClient::UnregisterSelf() {
  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "node is not registered");
  }

  StopHeartbeat();

  ::umbp::UnregisterClientRequest req;
  req.set_node_id(config_.node_id);

  ::umbp::UnregisterClientResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->UnregisterClient(&ctx, req, &resp);

  if (status.ok()) {
    spdlog::info("[Client] Unregistered from master (keys_removed={})", resp.keys_removed());
  } else {
    spdlog::error("[Client] UnregisterClient failed: {}", status.error_message());
  }
  registered_ = false;
  return status;
}

grpc::Status UMBPClient::Register(const std::string& key, const Location& location) {
  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before block registration");
  }

  if (key.empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
  }

  Location normalized_location = location;
  if (normalized_location.node_id.empty()) {
    normalized_location.node_id = config_.node_id;
  }
  if (normalized_location.node_id != config_.node_id) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "location.node_id must match client node_id");
  }

  ::umbp::RegisterRequest req;
  req.set_node_id(config_.node_id);
  req.set_key(key);
  FillProtoLocation(normalized_location, req.mutable_location());

  ::umbp::RegisterResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->Register(&ctx, req, &resp);
  if (!status.ok()) {
    spdlog::error("[Client] Register(key={}) failed: {}", key, status.error_message());
    return status;
  }

  spdlog::info("[Client] Registered key='{}' location='{}'", key, normalized_location.location_id);
  return grpc::Status::OK;
}

grpc::Status UMBPClient::Unregister(const std::string& key, const Location& location,
                                    uint32_t* removed) {
  if (removed != nullptr) {
    *removed = 0;
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before block unregistration");
  }

  if (key.empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
  }

  Location normalized_location = location;
  if (normalized_location.node_id.empty()) {
    normalized_location.node_id = config_.node_id;
  }
  if (normalized_location.node_id != config_.node_id) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "location.node_id must match client node_id");
  }

  ::umbp::UnregisterRequest req;
  req.set_node_id(config_.node_id);
  req.set_key(key);
  FillProtoLocation(normalized_location, req.mutable_location());

  ::umbp::UnregisterResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->Unregister(&ctx, req, &resp);
  if (!status.ok()) {
    spdlog::error("[Client] Unregister(key={}) failed: {}", key, status.error_message());
    return status;
  }

  if (removed != nullptr) {
    *removed = resp.removed();
  }

  spdlog::info("[Client] Unregistered key='{}' location='{}' (removed={})", key,
               normalized_location.location_id, resp.removed());
  return grpc::Status::OK;
}

void UMBPClient::StartHeartbeat() {
  if (!registered_) {
    spdlog::warn("[Client] StartHeartbeat ignored: not registered");
    return;
  }

  if (heartbeat_running_) {
    return;
  }

  heartbeat_running_ = true;
  try {
    heartbeat_thread_ = std::thread(&UMBPClient::HeartbeatLoop, this);
  } catch (const std::system_error& e) {
    heartbeat_running_ = false;
    spdlog::error("[Client] Failed to start heartbeat thread: {}", e.what());
    return;
  }

  spdlog::info("[Client] Heartbeat thread started (interval={}ms)", heartbeat_interval_ms_);
}

void UMBPClient::StopHeartbeat() {
  if (!heartbeat_running_) {
    return;
  }

  heartbeat_running_ = false;
  hb_cv_.notify_one();
  if (heartbeat_thread_.joinable()) {
    heartbeat_thread_.join();
  }
  spdlog::info("[Client] Heartbeat thread stopped");
}

void UMBPClient::HeartbeatLoop() {
  while (heartbeat_running_) {
    {
      std::unique_lock lock(hb_cv_mutex_);
      hb_cv_.wait_for(lock, std::chrono::milliseconds(heartbeat_interval_ms_),
                      [this] { return !heartbeat_running_.load(); });
    }
    if (!heartbeat_running_) {
      break;
    }

    ::umbp::HeartbeatRequest req;
    req.set_node_id(config_.node_id);

    {
      std::lock_guard lock(caps_mutex_);
      for (const auto& [tier, cap] : current_capacities_) {
        auto* tc = req.add_tier_capacities();
        tc->set_tier(static_cast<::umbp::TierType>(tier));
        tc->set_total_capacity_bytes(cap.total_bytes);
        tc->set_available_capacity_bytes(cap.available_bytes);
      }
    }

    spdlog::info("[Client] Heartbeat sending: node_id={}, tiers={}", config_.node_id,
                 req.tier_capacities_size());

    ::umbp::HeartbeatResponse resp;
    grpc::ClientContext ctx;
    auto status = GetStub(stub_.get())->Heartbeat(&ctx, req, &resp);

    if (!status.ok()) {
      spdlog::warn("[Client] Heartbeat failed: node_id={}, error={}", config_.node_id,
                   status.error_message());
    } else {
      auto server_status = static_cast<ClientStatus>(resp.status());
      spdlog::info("[Client] Heartbeat ack: node_id={}, status={}", config_.node_id,
                   ClientStatusName(server_status));

      if (resp.status() == ::umbp::CLIENT_STATUS_UNKNOWN) {
        spdlog::warn(
            "[Client] Master does not recognize us; "
            "re-registration needed");
      }
    }
  }
}

}  // namespace mori::umbp

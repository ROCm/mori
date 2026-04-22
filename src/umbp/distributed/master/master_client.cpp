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

#include <system_error>

#include "mori/utils/mori_log.hpp"
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

MasterClient::MasterClient(const UMBPMasterClientConfig& config)
    : config_(config),
      stub_(nullptr, [](void* p) { delete static_cast<::umbp::UMBPMaster::Stub*>(p); }) {
  channel_ = grpc::CreateChannel(config.master_address, grpc::InsecureChannelCredentials());
  stub_.reset(::umbp::UMBPMaster::NewStub(channel_).release());
  MORI_UMBP_INFO("[Client] Created, master={}", config.master_address);
}

MasterClient::~MasterClient() {
  StopHeartbeat();
  if (registered_) {
    UnregisterSelf();
  }
}

grpc::Status MasterClient::RegisterSelf(
    const std::map<TierType, TierCapacity>& tier_capacities, const std::string& peer_address,
    const std::vector<uint8_t>& engine_desc_bytes,
    const std::vector<std::vector<uint8_t>>& dram_memory_desc_bytes_list,
    const std::vector<uint64_t>& dram_buffer_sizes,
    const std::vector<uint64_t>& ssd_store_capacities, uint64_t dram_page_size) {
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

  req.set_peer_address(peer_address);
  req.set_engine_desc(engine_desc_bytes.data(), engine_desc_bytes.size());

  // Multi-buffer: populate repeated fields
  for (const auto& desc : dram_memory_desc_bytes_list) {
    req.add_dram_memory_descs(desc.data(), desc.size());
  }
  for (uint64_t sz : dram_buffer_sizes) {
    req.add_dram_buffer_sizes(sz);
  }
  for (uint64_t cap : ssd_store_capacities) {
    req.add_ssd_store_capacities(cap);
  }

  // Per-node DRAM/HBM page_size override.  Master treats 0 as "fall back to
  // ClientRegistryConfig.default_dram_page_size".
  req.set_dram_page_size(dram_page_size);

  ::umbp::RegisterClientResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->RegisterClient(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] RegisterClient failed: {}", status.error_message());
    return status;
  }

  heartbeat_interval_ms_ = resp.heartbeat_interval_ms();
  registered_ = true;

  {
    std::lock_guard lock(caps_mutex_);
    current_capacities_ = tier_capacities;
  }

  MORI_UMBP_INFO("[Client] Registered with master (heartbeat_interval={}ms, dram_buffers={})",
                 heartbeat_interval_ms_, dram_memory_desc_bytes_list.size());

  if (config_.auto_heartbeat) {
    StartHeartbeat();
  }

  return grpc::Status::OK;
}

grpc::Status MasterClient::UnregisterSelf() {
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
    MORI_UMBP_INFO("[Client] Unregistered from master (keys_removed={})", resp.keys_removed());
  } else {
    MORI_UMBP_ERROR("[Client] UnregisterClient failed: {}", status.error_message());
  }
  registered_ = false;
  return status;
}

grpc::Status MasterClient::Register(const std::string& key, const Location& location) {
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

  ::umbp::RegisterRequest req;
  req.set_node_id(config_.node_id);
  req.set_key(key);
  FillProtoLocation(normalized_location, req.mutable_location());

  ::umbp::RegisterResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->Register(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] Register(key={}) failed: {}", key, status.error_message());
    return status;
  }

  MORI_UMBP_INFO("[Client] Registered key='{}' location='{}'", key,
                 normalized_location.location_id);
  return grpc::Status::OK;
}

grpc::Status MasterClient::Unregister(const std::string& key, const Location& location,
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

  ::umbp::UnregisterRequest req;
  req.set_node_id(config_.node_id);
  req.set_key(key);
  FillProtoLocation(normalized_location, req.mutable_location());

  ::umbp::UnregisterResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->Unregister(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] Unregister(key={}) failed: {}", key, status.error_message());
    return status;
  }

  if (removed != nullptr) {
    *removed = resp.removed();
  }

  MORI_UMBP_INFO("[Client] Unregistered key='{}' location='{}' (removed={})", key,
                 normalized_location.location_id, resp.removed());
  return grpc::Status::OK;
}

grpc::Status MasterClient::Lookup(const std::string& key, bool* found) {
  if (found != nullptr) {
    *found = false;
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before Lookup");
  }

  if (key.empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "key cannot be empty");
  }

  ::umbp::LookupRequest req;
  req.set_key(key);
  req.set_node_id(config_.node_id);

  ::umbp::LookupResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->Lookup(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] Lookup(key={}) failed: {}", key, status.error_message());
    return status;
  }

  if (found != nullptr) {
    *found = resp.found();
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::FinalizeAllocation(const std::string& key, const Location& location,
                                              const std::string& allocation_id, int32_t depth) {
  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before finalization");
  }

  Location normalized_location = location;
  if (normalized_location.node_id.empty()) {
    normalized_location.node_id = config_.node_id;
  }

  ::umbp::FinalizeRequest req;
  req.set_node_id(normalized_location.node_id);
  req.set_key(key);
  req.set_allocation_id(allocation_id);
  req.set_depth(depth);
  FillProtoLocation(normalized_location, req.mutable_location());

  ::umbp::FinalizeResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->FinalizeAllocation(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] FinalizeAllocation(key={}) failed: {}", key, status.error_message());
    return status;
  }
  if (!resp.finalized()) {
    return grpc::Status(grpc::StatusCode::UNKNOWN, "FinalizeAllocation rejected by master");
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::PublishLocalBlock(const std::string& key, const Location& location) {
  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before publishing");
  }

  Location normalized_location = location;
  if (normalized_location.node_id.empty()) {
    normalized_location.node_id = config_.node_id;
  }

  ::umbp::PublishRequest req;
  req.set_node_id(config_.node_id);
  req.set_key(key);
  FillProtoLocation(normalized_location, req.mutable_location());

  ::umbp::PublishResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->PublishLocalBlock(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] PublishLocalBlock(key={}) failed: {}", key, status.error_message());
    return status;
  }
  if (!resp.published()) {
    return grpc::Status(grpc::StatusCode::UNKNOWN, "PublishLocalBlock rejected by master");
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::AbortAllocation(const std::string& node_id,
                                           const std::string& allocation_id, uint64_t size) {
  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before aborting allocation");
  }

  ::umbp::AbortAllocationRequest req;
  req.set_node_id(node_id);
  req.set_allocation_id(allocation_id);
  req.set_size(size);

  ::umbp::AbortAllocationResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->AbortAllocation(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] AbortAllocation(node={}, id={}) failed: {}", node_id, allocation_id,
                    status.error_message());
    return status;
  }
  if (!resp.aborted()) {
    return grpc::Status(grpc::StatusCode::UNKNOWN, "AbortAllocation rejected by master");
  }
  return grpc::Status::OK;
}

grpc::Status MasterClient::RouteGet(const std::string& key,
                                    std::optional<RouteGetResult>* out_result) {
  if (out_result != nullptr) {
    *out_result = std::nullopt;
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before RouteGet");
  }

  ::umbp::RouteGetRequest req;
  req.set_key(key);
  req.set_node_id(config_.node_id);

  ::umbp::RouteGetResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->RouteGet(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] RouteGet(key={}) failed: {}", key, status.error_message());
    return status;
  }

  if (resp.found() && out_result != nullptr) {
    RouteGetResult result;
    result.location.node_id = resp.source().node_id();
    result.location.location_id = resp.source().location_id();
    result.location.size = resp.source().size();
    result.location.tier = static_cast<TierType>(resp.source().tier());
    result.peer_address = resp.peer_address();
    const auto& ed = resp.engine_desc();
    result.engine_desc_bytes.assign(ed.begin(), ed.end());
    result.dram_memory_descs.reserve(resp.dram_memory_descs_size());
    for (const auto& bd : resp.dram_memory_descs()) {
      BufferMemoryDescBytes b;
      b.buffer_index = bd.buffer_index();
      b.desc_bytes.assign(bd.desc().begin(), bd.desc().end());
      result.dram_memory_descs.push_back(std::move(b));
    }
    result.page_size = resp.page_size();
    *out_result = result;
  }

  MORI_UMBP_INFO("[Client] RouteGet key='{}': found={}", key, resp.found());
  return grpc::Status::OK;
}

grpc::Status MasterClient::RoutePut(const std::string& key, uint64_t block_size,
                                    std::optional<RoutePutResult>* out_result) {
  if (out_result != nullptr) {
    *out_result = std::nullopt;
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before RoutePut");
  }

  ::umbp::RoutePutRequest req;
  req.set_key(key);
  req.set_node_id(config_.node_id);
  req.set_block_size(block_size);

  ::umbp::RoutePutResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->RoutePut(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] RoutePut(key={}) failed: {}", key, status.error_message());
    return status;
  }

  if (resp.found() && out_result != nullptr) {
    RoutePutResult result;
    result.node_id = resp.node_id();
    result.node_address = resp.node_address();
    result.tier = static_cast<TierType>(resp.tier());
    result.peer_address = resp.peer_address();
    const auto& ed = resp.engine_desc();
    result.engine_desc_bytes.assign(ed.begin(), ed.end());
    result.allocation_id = resp.allocation_id();
    result.location_id = resp.location_id();
    result.pages.reserve(resp.pages_size());
    for (const auto& p : resp.pages()) {
      result.pages.push_back({p.buffer_index(), p.page_index()});
    }
    result.dram_memory_descs.reserve(resp.dram_memory_descs_size());
    for (const auto& bd : resp.dram_memory_descs()) {
      BufferMemoryDescBytes b;
      b.buffer_index = bd.buffer_index();
      b.desc_bytes.assign(bd.desc().begin(), bd.desc().end());
      result.dram_memory_descs.push_back(std::move(b));
    }
    result.page_size = resp.page_size();
    *out_result = result;
  }

  MORI_UMBP_INFO("[Client] RoutePut key='{}': found={}", key, resp.found());
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchRoutePut(const std::vector<std::string>& keys,
                                         const std::vector<uint64_t>& block_sizes,
                                         std::vector<std::optional<RoutePutResult>>* out) {
  if (out != nullptr) {
    out->clear();
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before BatchRoutePut");
  }

  ::umbp::BatchRoutePutRequest req;
  req.set_node_id(config_.node_id);
  for (const auto& k : keys) {
    req.add_keys(k);
  }
  for (uint64_t sz : block_sizes) {
    req.add_block_sizes(sz);
  }

  ::umbp::BatchRoutePutResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchRoutePut(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] BatchRoutePut failed: {}", status.error_message());
    return status;
  }

  if (out != nullptr) {
    out->resize(resp.entries_size());
    for (int i = 0; i < resp.entries_size(); ++i) {
      const auto& entry = resp.entries(i);
      if (entry.found()) {
        RoutePutResult result;
        result.node_id = entry.node_id();
        result.node_address = entry.node_address();
        result.tier = static_cast<TierType>(entry.tier());
        result.peer_address = entry.peer_address();
        const auto& ed = entry.engine_desc();
        result.engine_desc_bytes.assign(ed.begin(), ed.end());
        result.allocation_id = entry.allocation_id();
        result.location_id = entry.location_id();
        result.pages.reserve(entry.pages_size());
        for (const auto& p : entry.pages()) {
          result.pages.push_back({p.buffer_index(), p.page_index()});
        }
        result.dram_memory_descs.reserve(entry.dram_memory_descs_size());
        for (const auto& bd : entry.dram_memory_descs()) {
          BufferMemoryDescBytes b;
          b.buffer_index = bd.buffer_index();
          b.desc_bytes.assign(bd.desc().begin(), bd.desc().end());
          result.dram_memory_descs.push_back(std::move(b));
        }
        result.page_size = entry.page_size();
        (*out)[i] = std::move(result);
      }
    }
  }

  MORI_UMBP_INFO("[Client] BatchRoutePut: {} keys", keys.size());
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchRouteGet(const std::vector<std::string>& keys,
                                         std::vector<std::optional<RouteGetResult>>* out) {
  if (out != nullptr) {
    out->clear();
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before BatchRouteGet");
  }

  ::umbp::BatchRouteGetRequest req;
  req.set_node_id(config_.node_id);
  for (const auto& k : keys) {
    req.add_keys(k);
  }

  ::umbp::BatchRouteGetResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchRouteGet(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] BatchRouteGet failed: {}", status.error_message());
    return status;
  }

  if (out != nullptr) {
    out->resize(resp.entries_size());
    for (int i = 0; i < resp.entries_size(); ++i) {
      const auto& entry = resp.entries(i);
      if (entry.found()) {
        RouteGetResult result;
        result.location.node_id = entry.source().node_id();
        result.location.location_id = entry.source().location_id();
        result.location.size = entry.source().size();
        result.location.tier = static_cast<TierType>(entry.source().tier());
        result.peer_address = entry.peer_address();
        const auto& ed = entry.engine_desc();
        result.engine_desc_bytes.assign(ed.begin(), ed.end());
        result.dram_memory_descs.reserve(entry.dram_memory_descs_size());
        for (const auto& bd : entry.dram_memory_descs()) {
          BufferMemoryDescBytes b;
          b.buffer_index = bd.buffer_index();
          b.desc_bytes.assign(bd.desc().begin(), bd.desc().end());
          result.dram_memory_descs.push_back(std::move(b));
        }
        result.page_size = entry.page_size();
        (*out)[i] = std::move(result);
      }
    }
  }

  MORI_UMBP_INFO("[Client] BatchRouteGet: {} keys", keys.size());
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchFinalizeAllocation(const std::vector<std::string>& keys,
                                                   const std::vector<Location>& locations,
                                                   const std::vector<std::string>& allocation_ids,
                                                   std::vector<bool>* out,
                                                   const std::vector<int32_t>& depths) {
  if (out != nullptr) {
    out->clear();
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before BatchFinalizeAllocation");
  }

  ::umbp::BatchFinalizeRequest req;
  req.set_node_id(config_.node_id);
  for (const auto& k : keys) {
    req.add_keys(k);
  }
  for (const auto& loc : locations) {
    FillProtoLocation(loc, req.add_locations());
  }
  for (const auto& id : allocation_ids) {
    req.add_allocation_ids(id);
  }
  for (auto d : depths) {
    req.add_depths(d);
  }

  ::umbp::BatchFinalizeResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchFinalizeAllocation(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] BatchFinalizeAllocation failed: {}", status.error_message());
    return status;
  }

  if (out != nullptr) {
    out->resize(resp.finalized_size());
    for (int i = 0; i < resp.finalized_size(); ++i) {
      (*out)[i] = resp.finalized(i);
    }
  }

  MORI_UMBP_INFO("[Client] BatchFinalizeAllocation: {} keys", keys.size());
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchAbortAllocation(const std::vector<BatchAbortEntry>& entries,
                                                std::vector<bool>* out) {
  if (out != nullptr) {
    out->clear();
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before BatchAbortAllocation");
  }

  if (entries.empty()) {
    return grpc::Status::OK;
  }

  ::umbp::BatchAbortAllocationRequest req;
  for (const auto& e : entries) {
    auto* proto_e = req.add_entries();
    proto_e->set_node_id(e.node_id);
    proto_e->set_allocation_id(e.allocation_id);
    proto_e->set_size(e.size);
  }

  ::umbp::BatchAbortAllocationResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchAbortAllocation(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] BatchAbortAllocation ({} entries) failed: {}", entries.size(),
                    status.error_message());
    return status;
  }

  // Semantics align with BatchFinalizeAllocation: wire OK → fill out,
  // do NOT promote any per-entry false to an RPC-level error.  Per-entry
  // false is a normal race (already reaped / double-abort / EXPIRED).
  if (out != nullptr) {
    out->resize(resp.aborted_size());
    for (int i = 0; i < resp.aborted_size(); ++i) {
      (*out)[i] = resp.aborted(i);
    }
  }

  MORI_UMBP_INFO("[Client] BatchAbortAllocation: {} entries", entries.size());
  return grpc::Status::OK;
}

grpc::Status MasterClient::BatchLookup(const std::vector<std::string>& keys,
                                       std::vector<bool>* out) {
  if (out != nullptr) {
    out->clear();
  }

  if (!registered_) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "node must be registered before BatchLookup");
  }

  if (keys.empty()) {
    return grpc::Status::OK;
  }

  ::umbp::BatchLookupRequest req;
  req.set_node_id(config_.node_id);
  for (const auto& k : keys) {
    req.add_keys(k);
  }

  ::umbp::BatchLookupResponse resp;
  grpc::ClientContext ctx;
  auto status = GetStub(stub_.get())->BatchLookup(&ctx, req, &resp);

  if (!status.ok()) {
    MORI_UMBP_ERROR("[Client] BatchLookup failed: {}", status.error_message());
    return status;
  }

  if (out != nullptr) {
    out->resize(resp.found_size());
    for (int i = 0; i < resp.found_size(); ++i) {
      (*out)[i] = resp.found(i);
    }
  }

  MORI_UMBP_DEBUG("[Client] BatchLookup: {} keys", keys.size());
  return grpc::Status::OK;
}

void MasterClient::StartHeartbeat() {
  if (!registered_) {
    MORI_UMBP_WARN("[Client] StartHeartbeat ignored: not registered");
    return;
  }

  if (heartbeat_running_) {
    return;
  }

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
  if (!heartbeat_running_) {
    return;
  }

  heartbeat_running_ = false;
  hb_cv_.notify_one();
  if (heartbeat_thread_.joinable()) {
    heartbeat_thread_.join();
  }
  MORI_UMBP_INFO("[Client] Heartbeat thread stopped");
}

void MasterClient::HeartbeatLoop() {
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

    MORI_UMBP_INFO("[Client] Heartbeat sending: node_id={}, tiers={}", config_.node_id,
                   req.tier_capacities_size());

    ::umbp::HeartbeatResponse resp;
    grpc::ClientContext ctx;
    auto status = GetStub(stub_.get())->Heartbeat(&ctx, req, &resp);

    if (!status.ok()) {
      MORI_UMBP_WARN("[Client] Heartbeat failed: node_id={}, error={}", config_.node_id,
                     status.error_message());
    } else {
      auto server_status = static_cast<ClientStatus>(resp.status());
      MORI_UMBP_INFO("[Client] Heartbeat ack: node_id={}, status={}", config_.node_id,
                     ClientStatusName(server_status));

      if (resp.status() == ::umbp::CLIENT_STATUS_UNKNOWN) {
        MORI_UMBP_WARN(
            "[Client] Master does not recognize us; "
            "re-registration needed");
      }
    }
  }
}

}  // namespace mori::umbp

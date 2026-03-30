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
#include "umbp/distributed/pool_client.h"

#include <fcntl.h>
#include <grpcpp/grpcpp.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <msgpack.hpp>

#include "mori/io/backend.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {
struct ParsedLocationId {
  uint32_t buffer_index = 0;
  uint64_t offset = 0;
};

std::optional<ParsedLocationId> ParseLocationId(const std::string& location_id) {
  auto colon = location_id.find(':');
  if (colon == std::string::npos) {
    MORI_UMBP_ERROR("[PoolClient] Invalid location_id format (expected 'index:value'): {}",
                    location_id);
    return std::nullopt;
  }
  try {
    ParsedLocationId result;
    result.buffer_index = static_cast<uint32_t>(std::stoul(location_id.substr(0, colon)));
    result.offset = std::stoull(location_id.substr(colon + 1));
    return result;
  } catch (...) {
    MORI_UMBP_ERROR("[PoolClient] Failed to parse location_id: {}", location_id);
    return std::nullopt;
  }
}

bool IsValidMemoryDesc(const mori::io::MemoryDesc& desc) { return desc.size > 0; }
}  // namespace

PoolClient::PoolClient(PoolClientConfig config) : config_(std::move(config)) {}

PoolClient::~PoolClient() { Shutdown(); }

bool PoolClient::Init() {
  if (initialized_) return true;

  master_client_ = std::make_unique<MasterClient>(config_.master_config);

  // Initialize IO Engine for RDMA data plane
  if (config_.io_engine_port > 0) {
    mori::io::IOEngineConfig io_cfg;
    io_cfg.host = config_.io_engine_host;
    io_cfg.port = config_.io_engine_port;

    io_engine_ = std::make_unique<mori::io::IOEngine>(config_.master_config.node_id, io_cfg);

    mori::io::RdmaBackendConfig rdma_cfg;
    io_engine_->CreateBackend(mori::io::BackendType::RDMA, rdma_cfg);

    staging_buffer_ = std::make_unique<char[]>(config_.staging_buffer_size);
    std::memset(staging_buffer_.get(), 0, config_.staging_buffer_size);
    staging_mem_ = io_engine_->RegisterMemory(staging_buffer_.get(), config_.staging_buffer_size,
                                              -1, mori::io::MemoryLocationType::CPU);

    for (const auto& dram : config_.dram_buffers) {
      if (dram.buffer && dram.size > 0) {
        auto mem = io_engine_->RegisterMemory(dram.buffer, dram.size, -1,
                                              mori::io::MemoryLocationType::CPU);
        export_dram_mems_.push_back(mem);
      }
    }

    MORI_UMBP_INFO("[PoolClient] IOEngine initialized on {}:{} ({} DRAM buffers)",
                   config_.io_engine_host, config_.io_engine_port, export_dram_mems_.size());
  }

  // Pack EngineDesc and per-buffer MemoryDesc for registration
  std::vector<uint8_t> engine_desc_bytes;
  std::vector<std::vector<uint8_t>> dram_memory_desc_bytes_list;
  std::vector<uint64_t> dram_buffer_sizes;
  if (io_engine_) {
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, io_engine_->GetEngineDesc());
    engine_desc_bytes.assign(sbuf.data(), sbuf.data() + sbuf.size());

    for (size_t i = 0; i < export_dram_mems_.size(); ++i) {
      msgpack::sbuffer mbuf;
      msgpack::pack(mbuf, export_dram_mems_[i]);
      dram_memory_desc_bytes_list.emplace_back(mbuf.data(), mbuf.data() + mbuf.size());
      dram_buffer_sizes.push_back(config_.dram_buffers[i].size);
    }
  }

  // Allocate a dedicated SSD staging buffer, independent of DRAM exportable
  // buffers, so that SSD staging RDMA traffic cannot conflict with
  // Master-managed DRAM tier offset allocations.
  if (!config_.ssd_stores.empty()) {
    ssd_staging_buffer_ = std::make_unique<char[]>(config_.staging_buffer_size);
    std::memset(ssd_staging_buffer_.get(), 0, config_.staging_buffer_size);
    if (io_engine_) {
      ssd_staging_mem_ =
          io_engine_->RegisterMemory(ssd_staging_buffer_.get(), config_.staging_buffer_size, -1,
                                     mori::io::MemoryLocationType::CPU);
      msgpack::sbuffer sbuf;
      msgpack::pack(sbuf, ssd_staging_mem_);
      ssd_staging_mem_desc_bytes_.assign(sbuf.data(), sbuf.data() + sbuf.size());
    }
  }

  // PeerService is started by UMBPClient after PoolClient init. Advertise the
  // configured address here so the Master can route peer SSD traffic.
  std::string peer_address;
  if (config_.peer_service_port > 0 && !config_.ssd_stores.empty()) {
    std::string host = config_.io_engine_host.empty() ? config_.master_config.node_address
                                                      : config_.io_engine_host;
    peer_address = host + ":" + std::to_string(config_.peer_service_port);
  }

  std::vector<uint64_t> ssd_store_capacities;
  for (const auto& store : config_.ssd_stores) {
    ssd_store_capacities.push_back(store.capacity);
  }

  auto status = master_client_->RegisterSelf(config_.tier_capacities, peer_address,
                                             engine_desc_bytes, dram_memory_desc_bytes_list,
                                             dram_buffer_sizes, ssd_store_capacities);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] RegisterSelf failed: {}", status.error_message());
    return false;
  }

  if (config_.master_config.auto_heartbeat) {
    master_client_->StartHeartbeat();
  }

  initialized_ = true;
  MORI_UMBP_INFO("[PoolClient] Initialized node_id='{}'", config_.master_config.node_id);
  return true;
}

void PoolClient::Shutdown() {
  if (!initialized_) return;
  initialized_ = false;

  if (master_client_) {
    master_client_->StopHeartbeat();
    auto status = master_client_->UnregisterSelf();
    if (!status.ok()) {
      MORI_UMBP_WARN("[PoolClient] UnregisterSelf failed: {}", status.error_message());
    }
  }

  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    peers_.clear();
  }

  if (io_engine_) {
    {
      std::lock_guard<std::mutex> lock(registered_mem_mutex_);
      for (auto& reg : registered_regions_) {
        io_engine_->DeregisterMemory(reg.mem_desc);
      }
      registered_regions_.clear();
    }
    if (staging_buffer_) {
      io_engine_->DeregisterMemory(staging_mem_);
    }
    if (ssd_staging_buffer_) {
      io_engine_->DeregisterMemory(ssd_staging_mem_);
      ssd_staging_buffer_.reset();
    }
    for (auto& mem : export_dram_mems_) {
      io_engine_->DeregisterMemory(mem);
    }
    export_dram_mems_.clear();
    io_engine_.reset();
    staging_buffer_.reset();
  }

  master_client_.reset();

  std::lock_guard<std::mutex> lock(cache_mutex_);
  cluster_locations_.clear();
}

bool PoolClient::RegisterMemory(void* ptr, size_t size) {
  if (!io_engine_) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory: IOEngine not available");
    return false;
  }
  auto mem_desc = io_engine_->RegisterMemory(ptr, size, -1, mori::io::MemoryLocationType::CPU);
  std::lock_guard<std::mutex> lock(registered_mem_mutex_);
  registered_regions_.push_back({ptr, size, mem_desc});
  MORI_UMBP_INFO("[PoolClient] RegisterMemory: ptr={}, size={}", ptr, size);
  return true;
}

void PoolClient::DeregisterMemory(void* ptr) {
  std::lock_guard<std::mutex> lock(registered_mem_mutex_);
  auto it = std::find_if(registered_regions_.begin(), registered_regions_.end(),
                         [ptr](const RegisteredRegion& r) { return r.base == ptr; });
  if (it != registered_regions_.end()) {
    if (io_engine_) io_engine_->DeregisterMemory(it->mem_desc);
    registered_regions_.erase(it);
  }
}

std::optional<std::pair<mori::io::MemoryDesc, size_t>> PoolClient::FindRegisteredMemory(
    const void* ptr, size_t size) {
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(registered_mem_mutex_);
  for (auto& reg : registered_regions_) {
    auto base = reinterpret_cast<uintptr_t>(reg.base);
    if (addr >= base && addr + size <= base + reg.size) {
      return std::pair{reg.mem_desc, static_cast<size_t>(addr - base)};
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Phase 2: DRAM-only methods for UMBPClient integration
// ---------------------------------------------------------------------------

bool PoolClient::RegisterWithMaster(const std::string& key, size_t size,
                                    const std::string& location_id, TierType tier) {
  return PublishLocalBlock(key, size, location_id, tier);
}

bool PoolClient::FinalizeAllocation(const std::string& key, size_t size,
                                    const std::string& location_id, TierType tier,
                                    const std::string& allocation_id) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  Location location;
  location.node_id = config_.master_config.node_id;
  location.location_id = location_id;
  location.size = size;
  location.tier = tier;

  auto status = master_client_->FinalizeAllocation(key, location, allocation_id);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] FinalizeAllocation failed for key '{}': {}", key,
                    status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cluster_locations_[key] = location;
  }

  return true;
}

bool PoolClient::PublishLocalBlock(const std::string& key, size_t size,
                                   const std::string& location_id, TierType tier) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  Location location;
  location.node_id = config_.master_config.node_id;
  location.location_id = location_id;
  location.size = size;
  location.tier = tier;

  auto status = master_client_->PublishLocalBlock(key, location);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] PublishLocalBlock failed for key '{}': {}", key,
                    status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cluster_locations_[key] = location;
  }

  return true;
}

bool PoolClient::AbortAllocation(const std::string& node_id, TierType /*tier*/,
                                 const std::string& allocation_id, uint64_t size) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  auto status = master_client_->AbortAllocation(node_id, allocation_id, size);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] AbortAllocation failed for node '{}' allocation '{}': {}",
                    node_id, allocation_id, status.error_message());
    return false;
  }
  return true;
}

bool PoolClient::UnregisterFromMaster(const std::string& key) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  Location location;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = cluster_locations_.find(key);
    if (it == cluster_locations_.end()) {
      MORI_UMBP_WARN("[PoolClient] UnregisterFromMaster: key '{}' not in local cache", key);
      return false;
    }
    location = it->second;
  }

  uint32_t removed = 0;
  auto status = master_client_->Unregister(key, location, &removed);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] UnregisterFromMaster failed for key '{}': {}", key,
                    status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cluster_locations_.erase(key);
  }

  return removed > 0;
}

bool PoolClient::IsRegistered(const std::string& key) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cluster_locations_.find(key) != cluster_locations_.end();
}

bool PoolClient::ExistsRemote(const std::string& key) {
  if (!initialized_) return false;

  std::optional<RouteGetResult> result;
  auto status = master_client_->RouteGet(key, &result);
  if (!status.ok()) return false;
  return result.has_value();
}

bool PoolClient::GetRemote(const std::string& key, void* dst, size_t size) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  std::optional<RouteGetResult> result;
  auto status = master_client_->RouteGet(key, &result);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] GetRemote RouteGet failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) return false;

  const auto& loc = result->location;

  bool is_local = (loc.node_id == config_.master_config.node_id);
  if (is_local) {
    MORI_UMBP_WARN("[PoolClient] GetRemote: key '{}' is on local node", key);
    return false;
  }

  if (loc.tier == TierType::DRAM) {
    auto parsed = ParseLocationId(loc.location_id);
    if (!parsed) return false;
    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address, result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes, parsed->buffer_index);
    return RemoteDramRead(peer, parsed->buffer_index, dst, size, parsed->offset, false);
  }

  if (loc.tier == TierType::SSD) {
    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address, result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes);
    return RemoteSsdRead(peer, key, loc.location_id, dst, size, false);
  }

  MORI_UMBP_WARN("[PoolClient] GetRemote: key '{}' is on unsupported tier {}", key,
                 TierTypeName(loc.tier));
  return false;
}

bool PoolClient::PutRemote(const std::string& key, const void* src, size_t size) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  std::optional<RoutePutResult> result;
  auto status = master_client_->RoutePut(key, size, &result);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] PutRemote RoutePut failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) {
    MORI_UMBP_ERROR("[PoolClient] PutRemote: no suitable target");
    return false;
  }

  // DRAM-only: reject SSD targets (Phase 6 will add SSD support)
  if (result->tier != TierType::DRAM) {
    MORI_UMBP_WARN("[PoolClient] PutRemote: target tier is {} (DRAM-only supported)",
                   TierTypeName(result->tier));
    return false;
  }

  bool is_local = (result->node_id == config_.master_config.node_id);
  if (is_local) {
    // UMBPClient should handle local writes via storage_ directly
    MORI_UMBP_WARN("[PoolClient] PutRemote: target is local node");
    return false;
  }

  auto& peer = GetOrConnectPeer(result->node_id, result->peer_address, result->engine_desc_bytes,
                                result->dram_memory_desc_bytes, result->buffer_index);
  bool ok = RemoteDramWrite(peer, result->buffer_index, src, size, result->allocated_offset, false);
  if (!ok) {
    master_client_->AbortAllocation(result->node_id, result->allocation_id, size);
    return false;
  }

  // Register with Master so the block is discoverable
  Location location;
  location.node_id = result->node_id;
  location.location_id =
      std::to_string(result->buffer_index) + ":" + std::to_string(result->allocated_offset);
  location.size = size;
  location.tier = result->tier;

  status = master_client_->FinalizeAllocation(key, location, result->allocation_id);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] PutRemote FinalizeAllocation failed: {}", status.error_message());
    master_client_->AbortAllocation(result->node_id, result->allocation_id, size);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cluster_locations_[key] = location;
  }

  return true;
}

MasterClient& PoolClient::Master() { return *master_client_; }

bool PoolClient::IsInitialized() const { return initialized_; }

// ---------------------------------------------------------------------------
// Peer connection management
// ---------------------------------------------------------------------------

PoolClient::PeerConnection& PoolClient::GetOrConnectPeer(
    const std::string& node_id, const std::string& peer_address,
    const std::vector<uint8_t>& engine_desc_bytes,
    const std::vector<uint8_t>& dram_memory_desc_bytes, uint32_t buffer_index) {
  std::lock_guard<std::mutex> lock(peers_mutex_);
  auto it = peers_.find(node_id);
  if (it != peers_.end()) {
    // Ensure dram_memories vector has the requested index populated
    auto& peer = *it->second;
    if (buffer_index >= peer.dram_memories.size() && !dram_memory_desc_bytes.empty()) {
      peer.dram_memories.resize(buffer_index + 1);
      auto handle = msgpack::unpack(reinterpret_cast<const char*>(dram_memory_desc_bytes.data()),
                                    dram_memory_desc_bytes.size());
      peer.dram_memories[buffer_index] = handle.get().as<mori::io::MemoryDesc>();
    }
    return peer;
  }

  auto peer = std::make_unique<PeerConnection>();
  peer->peer_address = peer_address;

  if (io_engine_ && !engine_desc_bytes.empty()) {
    auto handle = msgpack::unpack(reinterpret_cast<const char*>(engine_desc_bytes.data()),
                                  engine_desc_bytes.size());
    peer->engine_desc = handle.get().as<mori::io::EngineDesc>();
    io_engine_->RegisterRemoteEngine(peer->engine_desc);
    peer->engine_registered = true;
    MORI_UMBP_INFO("[PoolClient] Registered remote engine for node '{}'", node_id);
  }

  if (!dram_memory_desc_bytes.empty()) {
    peer->dram_memories.resize(buffer_index + 1);
    auto handle = msgpack::unpack(reinterpret_cast<const char*>(dram_memory_desc_bytes.data()),
                                  dram_memory_desc_bytes.size());
    peer->dram_memories[buffer_index] = handle.get().as<mori::io::MemoryDesc>();
  }

  // PeerService connection (stub + staging MemoryDesc) is lazy-initialized
  // on first SSD operation. DRAM path doesn't need PeerService.

  auto* raw = peer.get();
  peers_.emplace(node_id, std::move(peer));
  return *raw;
}

// ---------------------------------------------------------------------------
// Remote DRAM path (pure RDMA)
// ---------------------------------------------------------------------------

bool PoolClient::RemoteDramWrite(PeerConnection& peer, uint32_t buffer_index, const void* src,
                                 size_t size, uint64_t offset, bool zero_copy) {
  if (!io_engine_) return false;
  if (!zero_copy && size > config_.staging_buffer_size) {
    MORI_UMBP_ERROR("[PoolClient] RemoteDramWrite: size {} exceeds staging_buffer_size {}", size,
                    config_.staging_buffer_size);
    return false;
  }
  if (buffer_index >= peer.dram_memories.size() ||
      !IsValidMemoryDesc(peer.dram_memories[buffer_index])) {
    MORI_UMBP_ERROR("[PoolClient] RemoteDramWrite: invalid buffer_index {} (size={}, valid={})",
                    buffer_index, peer.dram_memories.size(),
                    buffer_index < peer.dram_memories.size()
                        ? IsValidMemoryDesc(peer.dram_memories[buffer_index])
                        : false);
    return false;
  }
  auto& remote_mem = peer.dram_memories[buffer_index];

  if (zero_copy) {
    auto reg = FindRegisteredMemory(src, size);
    if (reg) {
      auto uid = io_engine_->AllocateTransferUniqueId();
      MORI_UMBP_DEBUG(
          "[PoolClient] RemoteDramWrite (zero-copy) start: uid={}, buf={}, offset={}, size={}", uid,
          buffer_index, offset, size);
      mori::io::TransferStatus status;
      io_engine_->Write(reg->first, reg->second, remote_mem, offset, size, &status, uid);
      status.Wait();
      if (!status.Succeeded()) {
        MORI_UMBP_ERROR("[PoolClient] RemoteDramWrite (zero-copy) failed: uid={}, {}", uid,
                        status.Message());
        return false;
      }
      MORI_UMBP_DEBUG("[PoolClient] RemoteDramWrite (zero-copy) done: uid={}", uid);
      return true;
    }
    MORI_UMBP_WARN(
        "[PoolClient] zero_copy=true but pointer not registered, "
        "falling back to staging");
  }

  std::lock_guard<std::mutex> lock(staging_mutex_);
  std::memcpy(staging_buffer_.get(), src, size);

  auto uid = io_engine_->AllocateTransferUniqueId();
  MORI_UMBP_DEBUG("[PoolClient] RemoteDramWrite start: uid={}, buf={}, offset={}, size={}", uid,
                  buffer_index, offset, size);
  mori::io::TransferStatus status;
  io_engine_->Write(staging_mem_, 0, remote_mem, offset, size, &status, uid);
  status.Wait();
  if (!status.Succeeded()) {
    MORI_UMBP_ERROR("[PoolClient] RemoteDramWrite failed: uid={}, {}", uid, status.Message());
    return false;
  }
  MORI_UMBP_DEBUG("[PoolClient] RemoteDramWrite done: uid={}", uid);
  return true;
}

bool PoolClient::RemoteDramRead(PeerConnection& peer, uint32_t buffer_index, void* dst, size_t size,
                                uint64_t offset, bool zero_copy) {
  if (!io_engine_) return false;
  if (!zero_copy && size > config_.staging_buffer_size) {
    MORI_UMBP_ERROR("[PoolClient] RemoteDramRead: size {} exceeds staging_buffer_size {}", size,
                    config_.staging_buffer_size);
    return false;
  }
  if (buffer_index >= peer.dram_memories.size() ||
      !IsValidMemoryDesc(peer.dram_memories[buffer_index])) {
    MORI_UMBP_ERROR("[PoolClient] RemoteDramRead: invalid buffer_index {} (size={}, valid={})",
                    buffer_index, peer.dram_memories.size(),
                    buffer_index < peer.dram_memories.size()
                        ? IsValidMemoryDesc(peer.dram_memories[buffer_index])
                        : false);
    return false;
  }
  auto& remote_mem = peer.dram_memories[buffer_index];

  if (zero_copy) {
    auto reg = FindRegisteredMemory(dst, size);
    if (reg) {
      auto uid = io_engine_->AllocateTransferUniqueId();
      MORI_UMBP_DEBUG(
          "[PoolClient] RemoteDramRead (zero-copy) start: uid={}, buf={}, offset={}, size={}", uid,
          buffer_index, offset, size);
      mori::io::TransferStatus status;
      io_engine_->Read(reg->first, reg->second, remote_mem, offset, size, &status, uid);
      status.Wait();
      if (!status.Succeeded()) {
        MORI_UMBP_ERROR("[PoolClient] RemoteDramRead (zero-copy) failed: uid={}, {}", uid,
                        status.Message());
        return false;
      }
      MORI_UMBP_DEBUG("[PoolClient] RemoteDramRead (zero-copy) done: uid={}", uid);
      return true;
    }
    MORI_UMBP_WARN(
        "[PoolClient] zero_copy=true but pointer not registered, "
        "falling back to staging");
  }

  std::lock_guard<std::mutex> lock(staging_mutex_);

  auto uid = io_engine_->AllocateTransferUniqueId();
  MORI_UMBP_DEBUG("[PoolClient] RemoteDramRead start: uid={}, buf={}, offset={}, size={}", uid,
                  buffer_index, offset, size);
  mori::io::TransferStatus status;
  io_engine_->Read(staging_mem_, 0, remote_mem, offset, size, &status, uid);
  status.Wait();
  if (!status.Succeeded()) {
    MORI_UMBP_ERROR("[PoolClient] RemoteDramRead failed: uid={}, {}", uid, status.Message());
    return false;
  }
  MORI_UMBP_DEBUG("[PoolClient] RemoteDramRead done: uid={}", uid);

  std::memcpy(dst, staging_buffer_.get(), size);
  return true;
}

// ---------------------------------------------------------------------------
// Remote SSD path (RDMA + PeerService gRPC coordination)
// ---------------------------------------------------------------------------

bool PoolClient::EnsurePeerServiceConnection(PeerConnection& peer) {
  if (peer.peer_stub) return true;
  if (peer.peer_address.empty()) {
    MORI_UMBP_ERROR("[PoolClient] No peer_address for PeerService connection");
    return false;
  }

  auto channel = grpc::CreateChannel(peer.peer_address, grpc::InsecureChannelCredentials());
  auto stub = ::umbp::UMBPPeer::NewStub(channel);

  ::umbp::GetPeerInfoRequest req;
  ::umbp::GetPeerInfoResponse resp;
  grpc::ClientContext ctx;
  auto status = stub->GetPeerInfo(&ctx, req, &resp);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] GetPeerInfo failed for '{}': {}", peer.peer_address,
                    status.error_message());
    return false;
  }

  if (!resp.ssd_staging_mem_desc().empty()) {
    auto handle = msgpack::unpack(reinterpret_cast<const char*>(resp.ssd_staging_mem_desc().data()),
                                  resp.ssd_staging_mem_desc().size());
    peer.ssd_staging_mem = handle.get().as<mori::io::MemoryDesc>();
    peer.ssd_staging_size = resp.ssd_staging_size();
  }

  peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
      stub.release(), +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
  return true;
}

bool PoolClient::RemoteSsdWrite(PeerConnection& peer, const std::string& key, const void* src,
                                size_t size, bool zero_copy, uint32_t store_index,
                                const std::string& allocation_id) {
  if (!io_engine_) return false;
  if (!EnsurePeerServiceConnection(peer)) return false;
  if (!zero_copy && size > config_.staging_buffer_size) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite: size {} exceeds local staging_buffer_size {}",
                    size, config_.staging_buffer_size);
    return false;
  }
  if (!IsValidMemoryDesc(peer.ssd_staging_mem)) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite: no SSD staging MemoryDesc");
    return false;
  }
  auto& staging_remote_mem = peer.ssd_staging_mem;

  {
    std::lock_guard<std::mutex> lock(peer.ssd_op_mutex);
    if (!peer.peer_stub) {
      auto channel = grpc::CreateChannel(peer.peer_address, grpc::InsecureChannelCredentials());
      auto s = ::umbp::UMBPPeer::NewStub(channel);
      peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
          s.release(), +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
    }
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  // Phase 0: Pre-allocate a write slot on the remote peer
  ::umbp::AllocateWriteSlotRequest alloc_req;
  alloc_req.set_size(size);
  ::umbp::AllocateWriteSlotResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  auto alloc_status = stub->AllocateWriteSlot(&alloc_ctx, alloc_req, &alloc_resp);
  if (!alloc_status.ok() || !alloc_resp.success()) {
    MORI_UMBP_ERROR("[PoolClient] AllocateWriteSlot failed for key={}", key);
    return false;
  }
  uint64_t write_offset = alloc_resp.staging_offset();

  // Phase 1: RDMA write data into the allocated staging slot
  {
    bool used_zero_copy = false;
    if (zero_copy) {
      auto reg = FindRegisteredMemory(src, size);
      if (reg) {
        auto uid = io_engine_->AllocateTransferUniqueId();
        MORI_UMBP_DEBUG("[PoolClient] RemoteSsdWrite RDMA (zero-copy) start: uid={}, size={}", uid,
                        size);
        mori::io::TransferStatus status;
        io_engine_->Write(reg->first, reg->second, staging_remote_mem, write_offset, size, &status,
                          uid);
        status.Wait();
        if (!status.Succeeded()) {
          MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite RDMA (zero-copy) failed: uid={}, {}", uid,
                          status.Message());
          return false;
        }
        MORI_UMBP_DEBUG("[PoolClient] RemoteSsdWrite RDMA (zero-copy) done: uid={}", uid);
        used_zero_copy = true;
      } else {
        MORI_UMBP_WARN("[PoolClient] zero_copy=true but pointer not registered, falling back");
      }
    }

    if (!used_zero_copy) {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      std::memcpy(staging_buffer_.get(), src, size);

      auto uid = io_engine_->AllocateTransferUniqueId();
      MORI_UMBP_DEBUG("[PoolClient] RemoteSsdWrite RDMA start: uid={}, size={}", uid, size);
      mori::io::TransferStatus status;
      io_engine_->Write(staging_mem_, 0, staging_remote_mem, write_offset, size, &status, uid);
      status.Wait();
      if (!status.Succeeded()) {
        MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite RDMA failed: uid={}, {}", uid,
                        status.Message());
        return false;
      }
      MORI_UMBP_DEBUG("[PoolClient] RemoteSsdWrite RDMA done: uid={}", uid);
    }
  }

  // Phase 2: CommitSsdWrite with lease_id (slot is released by server on completion)
  ::umbp::CommitSsdWriteRequest req;
  req.set_key(key);
  req.set_staging_offset(write_offset);
  req.set_size(size);
  req.set_store_index(store_index);
  req.set_allocation_id(allocation_id);
  req.set_lease_id(alloc_resp.lease_id());

  ::umbp::CommitSsdWriteResponse resp;
  grpc::ClientContext ctx;
  auto grpc_status = stub->CommitSsdWrite(&ctx, req, &resp);
  if (!grpc_status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] CommitSsdWrite RPC failed: {}", grpc_status.error_message());
    return false;
  }
  if (!resp.success()) {
    MORI_UMBP_ERROR("[PoolClient] CommitSsdWrite rejected by peer for key={}", key);
    return false;
  }

  return true;
}

bool PoolClient::RemoteSsdRead(PeerConnection& peer, const std::string& key,
                               const std::string& location_id, void* dst, size_t size,
                               bool zero_copy) {
  if (!io_engine_) return false;
  if (!EnsurePeerServiceConnection(peer)) return false;
  if (!zero_copy && size > config_.staging_buffer_size) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead: size {} exceeds local staging_buffer_size {}",
                    size, config_.staging_buffer_size);
    return false;
  }
  if (!IsValidMemoryDesc(peer.ssd_staging_mem)) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead: no SSD staging MemoryDesc");
    return false;
  }
  auto& staging_remote_mem = peer.ssd_staging_mem;

  {
    std::lock_guard<std::mutex> lock(peer.ssd_op_mutex);
    if (!peer.peer_stub) {
      auto channel = grpc::CreateChannel(peer.peer_address, grpc::InsecureChannelCredentials());
      auto s = ::umbp::UMBPPeer::NewStub(channel);
      peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
          s.release(), +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
    }
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  // Phase 1: PrepareSsdRead — server allocates a slot and loads SSD data
  ::umbp::PrepareSsdReadRequest req;
  req.set_key(key);
  req.set_ssd_location_id(location_id);
  req.set_size(size);

  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;
  auto grpc_status = stub->PrepareSsdRead(&ctx, req, &resp);
  if (!grpc_status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] PrepareSsdRead RPC failed: {}", grpc_status.error_message());
    return false;
  }
  if (!resp.success()) {
    MORI_UMBP_ERROR("[PoolClient] PrepareSsdRead failed for key={}", key);
    return false;
  }

  // Phase 2: RDMA read from the allocated staging slot
  bool rdma_ok = false;
  if (zero_copy) {
    auto reg = FindRegisteredMemory(dst, size);
    if (reg) {
      auto uid = io_engine_->AllocateTransferUniqueId();
      MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead RDMA (zero-copy) start: uid={}, size={}", uid,
                      size);
      mori::io::TransferStatus status;
      io_engine_->Read(reg->first, reg->second, staging_remote_mem, resp.staging_offset(), size,
                       &status, uid);
      status.Wait();
      rdma_ok = status.Succeeded();
      if (!rdma_ok) {
        MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead RDMA (zero-copy) failed: uid={}, {}", uid,
                        status.Message());
      } else {
        MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead RDMA (zero-copy) done: uid={}", uid);
      }
    } else {
      MORI_UMBP_WARN("[PoolClient] zero_copy=true but pointer not registered, falling back");
    }
  }

  if (!rdma_ok) {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    auto uid = io_engine_->AllocateTransferUniqueId();
    MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead RDMA start: uid={}, size={}", uid, size);
    mori::io::TransferStatus status;
    io_engine_->Read(staging_mem_, 0, staging_remote_mem, resp.staging_offset(), size, &status,
                     uid);
    status.Wait();
    if (!status.Succeeded()) {
      MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead RDMA failed: uid={}, {}", uid, status.Message());
      // Release slot even on failure
      if (resp.lease_id() > 0) {
        for (int attempt = 0; attempt < 2; ++attempt) {
          ::umbp::ReleaseSsdLeaseRequest rel_req;
          rel_req.set_lease_id(resp.lease_id());
          ::umbp::ReleaseSsdLeaseResponse rel_resp;
          grpc::ClientContext rel_ctx;
          if (stub->ReleaseSsdLease(&rel_ctx, rel_req, &rel_resp).ok()) break;
        }
      }
      return false;
    }
    MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead RDMA done: uid={}", uid);
    std::memcpy(dst, staging_buffer_.get(), size);
    rdma_ok = true;
  }

  // Phase 3: Release staging slot (with lightweight retry)
  if (resp.lease_id() > 0) {
    for (int attempt = 0; attempt < 2; ++attempt) {
      ::umbp::ReleaseSsdLeaseRequest rel_req;
      rel_req.set_lease_id(resp.lease_id());
      ::umbp::ReleaseSsdLeaseResponse rel_resp;
      grpc::ClientContext rel_ctx;
      if (stub->ReleaseSsdLease(&rel_ctx, rel_req, &rel_resp).ok()) break;
    }
  }

  return rdma_ok;
}

}  // namespace mori::umbp

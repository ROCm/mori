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

struct ParsedSsdLocationId {
  uint32_t store_index = 0;
  std::string filename;
};

ParsedSsdLocationId ParseSsdLocationId(const std::string& location_id) {
  ParsedSsdLocationId result;
  auto colon = location_id.find(':');
  if (colon == std::string::npos) {
    result.filename = location_id;
    return result;
  }
  try {
    result.store_index = static_cast<uint32_t>(std::stoul(location_id.substr(0, colon)));
    result.filename = location_id.substr(colon + 1);
  } catch (...) {
    MORI_UMBP_ERROR("[PoolClient] Failed to parse SSD location_id: {}", location_id);
    result.filename = location_id;
  }
  return result;
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
  std::vector<std::string> ssd_dirs;
  std::vector<size_t> ssd_capacities;
  for (const auto& store : config_.ssd_stores) {
    ssd_dirs.push_back(store.dir);
    ssd_capacities.push_back(store.capacity);
  }

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

  // Start PeerService (for remote SSD coordination)
  if (config_.peer_service_port > 0 && !ssd_dirs.empty()) {
    peer_service_ =
        std::make_unique<PeerServiceServer>(ssd_staging_buffer_.get(), config_.staging_buffer_size,
                                            ssd_staging_mem_desc_bytes_, ssd_dirs, ssd_capacities);
    if (!peer_service_->Start(config_.peer_service_port)) {
      MORI_UMBP_ERROR("[PoolClient] PeerService failed to start on port {}",
                      config_.peer_service_port);
      peer_service_.reset();
    } else {
      MORI_UMBP_INFO("[PoolClient] PeerService started on port {}", config_.peer_service_port);
    }
  }

  // Only advertise peer_address if PeerService was actually started
  std::string peer_address;
  if (peer_service_) {
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

  if (peer_service_) {
    peer_service_->Stop();
    peer_service_.reset();
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
  location_cache_.clear();
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

bool PoolClient::Put(const std::string& key, const void* src, size_t size, bool zero_copy) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  // If key already exists, remove the old entry first to avoid resource leaks
  {
    Location old_loc;
    bool has_old = false;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      auto it = location_cache_.find(key);
      if (it != location_cache_.end()) {
        old_loc = it->second;
        has_old = true;
        location_cache_.erase(it);
      }
    }
    if (has_old) {
      master_client_->Unregister(key, old_loc);
    }
  }

  std::optional<RoutePutResult> result;
  auto status = master_client_->RoutePut(key, size, &result);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] RoutePut failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) {
    MORI_UMBP_ERROR("[PoolClient] RoutePut: no suitable target");
    return false;
  }

  bool is_local = (result->node_id == config_.master_config.node_id);

  bool ok = false;
  Location location;
  location.node_id = result->node_id;
  location.size = size;
  location.tier = result->tier;

  if (is_local && result->tier == TierType::DRAM) {
    ok = PutLocalDram(result->buffer_index, src, size, result->allocated_offset);
    location.location_id =
        std::to_string(result->buffer_index) + ":" + std::to_string(result->allocated_offset);
  } else if (is_local && result->tier == TierType::SSD) {
    ok = PutLocalSsd(key, src, size, result->buffer_index);
    location.location_id = std::to_string(result->buffer_index) + ":" + key + ".bin";
  } else if (!is_local && result->tier == TierType::DRAM) {
    auto& peer = GetOrConnectPeer(result->node_id, result->peer_address, result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes, result->buffer_index);
    ok =
        RemoteDramWrite(peer, result->buffer_index, src, size, result->allocated_offset, zero_copy);
    location.location_id =
        std::to_string(result->buffer_index) + ":" + std::to_string(result->allocated_offset);
  } else if (!is_local && result->tier == TierType::SSD) {
    auto& peer = GetOrConnectPeer(result->node_id, result->peer_address, result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes);
    ok = RemoteSsdWrite(peer, key, src, size, zero_copy, result->buffer_index);
    location.location_id = std::to_string(result->buffer_index) + ":" + key + ".bin";
  } else {
    MORI_UMBP_ERROR("[PoolClient] Unsupported Put path: node={}, tier={}", result->node_id,
                    TierTypeName(result->tier));
    return false;
  }

  if (!ok) return false;

  status = master_client_->Register(key, location);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] Register failed: {}, attempting rollback",
                    status.error_message());
    master_client_->Unregister(key, location);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    location_cache_[key] = location;
  }

  return true;
}

bool PoolClient::Get(const std::string& key, void* dst, size_t size, bool zero_copy) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  std::optional<RouteGetResult> result;
  auto status = master_client_->RouteGet(key, &result);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] RouteGet failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) {
    MORI_UMBP_ERROR("[PoolClient] RouteGet: key '{}' not found", key);
    return false;
  }

  const auto& loc = result->location;
  bool is_local = (loc.node_id == config_.master_config.node_id);

  if (is_local && loc.tier == TierType::DRAM) {
    auto parsed = ParseLocationId(loc.location_id);
    if (!parsed) return false;
    return GetLocalDram(parsed->buffer_index, dst, size, parsed->offset);
  } else if (is_local && loc.tier == TierType::SSD) {
    auto parsed = ParseSsdLocationId(loc.location_id);
    return GetLocalSsd(parsed.filename, dst, size, parsed.store_index);
  } else if (!is_local && loc.tier == TierType::DRAM) {
    auto parsed = ParseLocationId(loc.location_id);
    if (!parsed) return false;
    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address, result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes, parsed->buffer_index);
    return RemoteDramRead(peer, parsed->buffer_index, dst, size, parsed->offset, zero_copy);
  } else if (!is_local && loc.tier == TierType::SSD) {
    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address, result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes);
    auto parsed_ssd = ParseSsdLocationId(loc.location_id);
    return RemoteSsdRead(peer, key, parsed_ssd.filename, dst, size, zero_copy);
  } else {
    MORI_UMBP_ERROR("[PoolClient] Unsupported Get path: node={}, tier={}", loc.node_id,
                    TierTypeName(loc.tier));
    return false;
  }
}

bool PoolClient::Remove(const std::string& key) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  Location location;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = location_cache_.find(key);
    if (it == location_cache_.end()) {
      MORI_UMBP_WARN("[PoolClient] Remove: key '{}' not in local cache", key);
      return false;
    }
    location = it->second;
  }

  uint32_t removed = 0;
  auto status = master_client_->Unregister(key, location, &removed);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] Unregister failed: {}", status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    location_cache_.erase(key);
  }

  return removed > 0;
}

// ---------------------------------------------------------------------------
// Phase 2: DRAM-only methods for UMBPClient integration
// ---------------------------------------------------------------------------

bool PoolClient::RegisterWithMaster(const std::string& key, size_t size,
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

  auto status = master_client_->Register(key, location);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] RegisterWithMaster failed for key '{}': {}", key,
                    status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    location_cache_[key] = location;
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
    auto it = location_cache_.find(key);
    if (it == location_cache_.end()) {
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
    location_cache_.erase(key);
  }

  return removed > 0;
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

  // DRAM-only: reject SSD-resident remote blocks (Phase 6 will add SSD support)
  if (loc.tier != TierType::DRAM) {
    MORI_UMBP_WARN("[PoolClient] GetRemote: key '{}' is on {} (DRAM-only supported)", key,
                   TierTypeName(loc.tier));
    return false;
  }

  bool is_local = (loc.node_id == config_.master_config.node_id);
  if (is_local) {
    // UMBPClient should handle local reads via storage_ directly
    MORI_UMBP_WARN("[PoolClient] GetRemote: key '{}' is on local node", key);
    return false;
  }

  auto parsed = ParseLocationId(loc.location_id);
  if (!parsed) return false;

  auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address, result->engine_desc_bytes,
                                result->dram_memory_desc_bytes, parsed->buffer_index);
  return RemoteDramRead(peer, parsed->buffer_index, dst, size, parsed->offset, false);
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
  if (!ok) return false;

  // Register with Master so the block is discoverable
  Location location;
  location.node_id = result->node_id;
  location.location_id =
      std::to_string(result->buffer_index) + ":" + std::to_string(result->allocated_offset);
  location.size = size;
  location.tier = result->tier;

  status = master_client_->Register(key, location);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] PutRemote Register failed: {}", status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    location_cache_[key] = location;
  }

  return true;
}

MasterClient& PoolClient::Master() { return *master_client_; }

bool PoolClient::IsInitialized() const { return initialized_; }

bool PoolClient::PutLocalDram(uint32_t buffer_index, const void* src, size_t size,
                              uint64_t offset) {
  if (buffer_index >= config_.dram_buffers.size()) {
    MORI_UMBP_ERROR("[PoolClient] DRAM buffer_index {} out of range (have {})", buffer_index,
                    config_.dram_buffers.size());
    return false;
  }
  auto& buf = config_.dram_buffers[buffer_index];
  if (!buf.buffer) {
    MORI_UMBP_ERROR("[PoolClient] No exportable DRAM buffer at index {}", buffer_index);
    return false;
  }
  if (offset + size > buf.size) {
    MORI_UMBP_ERROR("[PoolClient] DRAM write out of bounds: buf={} offset={} size={} total={}",
                    buffer_index, offset, size, buf.size);
    return false;
  }
  std::memcpy(static_cast<char*>(buf.buffer) + offset, src, size);
  return true;
}

bool PoolClient::GetLocalDram(uint32_t buffer_index, void* dst, size_t size, uint64_t offset) {
  if (buffer_index >= config_.dram_buffers.size()) {
    MORI_UMBP_ERROR("[PoolClient] DRAM buffer_index {} out of range (have {})", buffer_index,
                    config_.dram_buffers.size());
    return false;
  }
  auto& buf = config_.dram_buffers[buffer_index];
  if (!buf.buffer) {
    MORI_UMBP_ERROR("[PoolClient] No exportable DRAM buffer at index {}", buffer_index);
    return false;
  }
  if (offset + size > buf.size) {
    MORI_UMBP_ERROR("[PoolClient] DRAM read out of bounds: buf={} offset={} size={} total={}",
                    buffer_index, offset, size, buf.size);
    return false;
  }
  std::memcpy(dst, static_cast<const char*>(buf.buffer) + offset, size);
  return true;
}

bool PoolClient::PutLocalSsd(const std::string& key, const void* src, size_t size,
                             uint32_t store_index) {
  if (store_index >= config_.ssd_stores.size()) {
    MORI_UMBP_ERROR("[PoolClient] SSD store_index {} out of range (have {})", store_index,
                    config_.ssd_stores.size());
    return false;
  }

  const auto& ssd_dir = config_.ssd_stores[store_index].dir;
  std::filesystem::create_directories(ssd_dir);
  std::string path = ssd_dir + "/" + key + ".bin";

  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    MORI_UMBP_ERROR("[PoolClient] Failed to open SSD file for write: {}", path);
    return false;
  }

  ssize_t written = ::write(fd, src, size);
  if (written < 0 || static_cast<size_t>(written) != size) {
    MORI_UMBP_ERROR("[PoolClient] SSD write incomplete: {} of {} bytes", written, size);
    ::close(fd);
    return false;
  }

  ::fsync(fd);
  ::close(fd);
  return true;
}

bool PoolClient::GetLocalSsd(const std::string& filename, void* dst, size_t size,
                             uint32_t store_index) {
  if (store_index >= config_.ssd_stores.size()) {
    MORI_UMBP_ERROR("[PoolClient] SSD store_index {} out of range (have {})", store_index,
                    config_.ssd_stores.size());
    return false;
  }

  std::string path = config_.ssd_stores[store_index].dir + "/" + filename;
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd >= 0) {
    ssize_t bytes_read = ::read(fd, dst, size);
    ::close(fd);
    if (bytes_read >= 0 && static_cast<size_t>(bytes_read) == size) {
      return true;
    }
  }

  MORI_UMBP_ERROR("[PoolClient] Failed to read SSD file: {}", path);
  return false;
}

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
                                size_t size, bool zero_copy, uint32_t store_index) {
  if (!io_engine_) return false;
  if (!EnsurePeerServiceConnection(peer)) return false;
  // SSD staging is split in half: write region = [0, size/2)
  // Use min of local and remote staging size for bounds check
  size_t effective_staging = peer.ssd_staging_size > 0
                                 ? std::min(config_.staging_buffer_size, peer.ssd_staging_size)
                                 : config_.staging_buffer_size;
  size_t max_ssd_staging = effective_staging / 2;
  if (!zero_copy && size > max_ssd_staging) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite: size {} exceeds SSD staging write region {}",
                    size, max_ssd_staging);
    return false;
  }
  if (!IsValidMemoryDesc(peer.ssd_staging_mem)) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite: no SSD staging MemoryDesc");
    return false;
  }
  auto& staging_remote_mem = peer.ssd_staging_mem;
  // Write region: first half of the dedicated SSD staging buffer [0, size/2)
  constexpr uint64_t write_offset = 0;

  std::lock_guard<std::mutex> ssd_lock(peer.ssd_op_mutex);

  // Phase 1: RDMA write data into remote SSD staging write region
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
          MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite RDMA phase (zero-copy) failed: uid={}, {}",
                          uid, status.Message());
          return false;
        }
        MORI_UMBP_DEBUG("[PoolClient] RemoteSsdWrite RDMA (zero-copy) done: uid={}", uid);
        used_zero_copy = true;
      } else {
        MORI_UMBP_WARN(
            "[PoolClient] zero_copy=true but pointer not registered, "
            "falling back to staging");
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
        MORI_UMBP_ERROR("[PoolClient] RemoteSsdWrite RDMA phase failed: uid={}, {}", uid,
                        status.Message());
        return false;
      }
      MORI_UMBP_DEBUG("[PoolClient] RemoteSsdWrite RDMA done: uid={}", uid);
    }
  }

  // Phase 2: Ask PeerService to persist staging data to SSD
  if (!peer.peer_stub) {
    auto channel = grpc::CreateChannel(peer.peer_address, grpc::InsecureChannelCredentials());
    auto stub = ::umbp::UMBPPeer::NewStub(channel);
    peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
        stub.release(), +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  ::umbp::CommitSsdWriteRequest req;
  req.set_key(key);
  req.set_staging_offset(write_offset);
  req.set_size(size);
  req.set_store_index(store_index);

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
  // SSD staging is split in half: read region = [size/2, size)
  size_t effective_staging = peer.ssd_staging_size > 0
                                 ? std::min(config_.staging_buffer_size, peer.ssd_staging_size)
                                 : config_.staging_buffer_size;
  size_t max_ssd_staging = effective_staging / 2;
  if (!zero_copy && size > max_ssd_staging) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead: size {} exceeds SSD staging read region {}", size,
                    max_ssd_staging);
    return false;
  }
  if (!IsValidMemoryDesc(peer.ssd_staging_mem)) {
    MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead: no SSD staging MemoryDesc");
    return false;
  }
  // Read region: second half of the dedicated SSD staging buffer [size/2, size)
  auto& staging_remote_mem = peer.ssd_staging_mem;

  std::lock_guard<std::mutex> ssd_lock(peer.ssd_op_mutex);

  // Phase 1: Ask PeerService to load SSD data into staging (unaffected by zero_copy)
  if (!peer.peer_stub) {
    auto channel = grpc::CreateChannel(peer.peer_address, grpc::InsecureChannelCredentials());
    auto s = ::umbp::UMBPPeer::NewStub(channel);
    peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
        s.release(), +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

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

  // Phase 2: RDMA read from remote staging area
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
      if (!status.Succeeded()) {
        MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead RDMA phase (zero-copy) failed: uid={}, {}", uid,
                        status.Message());
        return false;
      }
      MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead RDMA (zero-copy) done: uid={}", uid);
      return true;
    }
    MORI_UMBP_WARN(
        "[PoolClient] zero_copy=true but pointer not registered, "
        "falling back to staging");
  }

  {
    std::lock_guard<std::mutex> lock(staging_mutex_);

    auto uid = io_engine_->AllocateTransferUniqueId();
    MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead RDMA start: uid={}, size={}", uid, size);
    mori::io::TransferStatus status;
    io_engine_->Read(staging_mem_, 0, staging_remote_mem, resp.staging_offset(), size, &status,
                     uid);
    status.Wait();
    if (!status.Succeeded()) {
      MORI_UMBP_ERROR("[PoolClient] RemoteSsdRead RDMA phase failed: uid={}, {}", uid,
                      status.Message());
      return false;
    }
    MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead RDMA done: uid={}", uid);

    std::memcpy(dst, staging_buffer_.get(), size);
  }

  return true;
}

}  // namespace mori::umbp

#include "umbp/pool_client.h"

#include <fcntl.h>
#include <grpcpp/grpcpp.h>
#include <msgpack.hpp>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>

#include "mori/io/backend.hpp"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

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

    io_engine_ = std::make_unique<mori::io::IOEngine>(
        config_.master_config.node_id, io_cfg);

    mori::io::RdmaBackendConfig rdma_cfg;
    io_engine_->CreateBackend(mori::io::BackendType::RDMA, rdma_cfg);

    staging_buffer_ = std::make_unique<char[]>(config_.staging_buffer_size);
    std::memset(staging_buffer_.get(), 0, config_.staging_buffer_size);
    staging_mem_ = io_engine_->RegisterMemory(
        staging_buffer_.get(), config_.staging_buffer_size, -1,
        mori::io::MemoryLocationType::CPU);

    if (config_.exportable_dram_buffer &&
        config_.exportable_dram_buffer_size > 0) {
      export_dram_mem_ = io_engine_->RegisterMemory(
          config_.exportable_dram_buffer, config_.exportable_dram_buffer_size,
          -1, mori::io::MemoryLocationType::CPU);
    }

    spdlog::info("[PoolClient] IOEngine initialized on {}:{}",
                 config_.io_engine_host, config_.io_engine_port);
  }

  // Pack EngineDesc and MemoryDesc for registration
  std::vector<uint8_t> engine_desc_bytes;
  std::vector<uint8_t> dram_memory_desc_bytes;
  if (io_engine_) {
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, io_engine_->GetEngineDesc());
    engine_desc_bytes.assign(sbuf.data(), sbuf.data() + sbuf.size());

    if (config_.exportable_dram_buffer) {
      msgpack::sbuffer mbuf;
      msgpack::pack(mbuf, export_dram_mem_);
      dram_memory_desc_bytes.assign(mbuf.data(), mbuf.data() + mbuf.size());
    }
  }

  // Start PeerService (for remote SSD coordination)
  uint64_t staging_base_offset = 0;
  if (config_.peer_service_port > 0 && !config_.exportable_ssd_dir.empty()) {
    void* ssd_staging_base = config_.exportable_dram_buffer;
    size_t ssd_staging_size = config_.exportable_dram_buffer_size;

    peer_service_ = std::make_unique<PeerServiceServer>(
        ssd_staging_base, ssd_staging_size, engine_desc_bytes,
        dram_memory_desc_bytes, config_.exportable_ssd_dir,
        config_.exportable_ssd_capacity, staging_base_offset);
    peer_service_->Start(config_.peer_service_port);
    spdlog::info("[PoolClient] PeerService started on port {}",
                 config_.peer_service_port);
  }

  std::string peer_address;
  if (config_.peer_service_port > 0) {
    peer_address = config_.master_config.node_address + ":" +
                   std::to_string(config_.peer_service_port);
  }

  auto status = master_client_->RegisterSelf(
      config_.tier_capacities, peer_address, engine_desc_bytes,
      dram_memory_desc_bytes);
  if (!status.ok()) {
    spdlog::error("[PoolClient] RegisterSelf failed: {}",
                  status.error_message());
    return false;
  }

  if (config_.master_config.auto_heartbeat) {
    master_client_->StartHeartbeat();
  }

  initialized_ = true;
  spdlog::info("[PoolClient] Initialized node_id='{}'",
               config_.master_config.node_id);
  return true;
}

void PoolClient::Shutdown() {
  if (!initialized_) return;
  initialized_ = false;

  if (master_client_) {
    master_client_->StopHeartbeat();
    auto status = master_client_->UnregisterSelf();
    if (!status.ok()) {
      spdlog::warn("[PoolClient] UnregisterSelf failed: {}",
                   status.error_message());
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
    if (staging_buffer_) {
      io_engine_->DeregisterMemory(staging_mem_);
    }
    if (config_.exportable_dram_buffer) {
      io_engine_->DeregisterMemory(export_dram_mem_);
    }
    io_engine_.reset();
    staging_buffer_.reset();
  }

  master_client_.reset();

  std::lock_guard<std::mutex> lock(cache_mutex_);
  location_cache_.clear();
}

bool PoolClient::Put(const std::string& key, const void* src, size_t size) {
  if (!initialized_) {
    spdlog::error("[PoolClient] Not initialized");
    return false;
  }

  std::optional<RoutePutResult> result;
  auto status = master_client_->RoutePut(key, size, &result);
  if (!status.ok()) {
    spdlog::error("[PoolClient] RoutePut failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) {
    spdlog::error("[PoolClient] RoutePut: no suitable target");
    return false;
  }

  bool is_local =
      (result->node_id == config_.master_config.node_id);

  bool ok = false;
  Location location;
  location.node_id = result->node_id;
  location.size = size;
  location.tier = result->tier;

  if (is_local && result->tier == TierType::DRAM) {
    ok = PutLocalDram(src, size, result->allocated_offset);
    location.location_id = std::to_string(result->allocated_offset);
  } else if (is_local && result->tier == TierType::SSD) {
    ok = PutLocalSsd(key, src, size);
    location.location_id = key + ".bin";
  } else if (!is_local && result->tier == TierType::DRAM) {
    auto& peer = GetOrConnectPeer(result->node_id, result->peer_address,
                                  result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes);
    ok = RemoteDramWrite(peer, src, size, result->allocated_offset);
    location.location_id = std::to_string(result->allocated_offset);
  } else if (!is_local && result->tier == TierType::SSD) {
    auto& peer = GetOrConnectPeer(result->node_id, result->peer_address,
                                  result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes);
    ok = RemoteSsdWrite(peer, key, src, size);
    location.location_id = key + ".bin";
  } else {
    spdlog::error("[PoolClient] Unsupported Put path: node={}, tier={}",
                  result->node_id, TierTypeName(result->tier));
    return false;
  }

  if (!ok) return false;

  status = master_client_->Register(key, location);
  if (!status.ok()) {
    spdlog::error("[PoolClient] Register failed: {}", status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    location_cache_[key] = location;
  }

  return true;
}

bool PoolClient::Get(const std::string& key, void* dst, size_t size) {
  if (!initialized_) {
    spdlog::error("[PoolClient] Not initialized");
    return false;
  }

  std::optional<RouteGetResult> result;
  auto status = master_client_->RouteGet(key, &result);
  if (!status.ok()) {
    spdlog::error("[PoolClient] RouteGet failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) {
    spdlog::error("[PoolClient] RouteGet: key '{}' not found", key);
    return false;
  }

  const auto& loc = result->location;
  bool is_local = (loc.node_id == config_.master_config.node_id);

  if (is_local && loc.tier == TierType::DRAM) {
    uint64_t offset = std::stoull(loc.location_id);
    return GetLocalDram(dst, size, offset);
  } else if (is_local && loc.tier == TierType::SSD) {
    return GetLocalSsd(key, dst, size);
  } else if (!is_local && loc.tier == TierType::DRAM) {
    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address,
                                  result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes);
    uint64_t offset = std::stoull(loc.location_id);
    return RemoteDramRead(peer, dst, size, offset);
  } else if (!is_local && loc.tier == TierType::SSD) {
    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address,
                                  result->engine_desc_bytes,
                                  result->dram_memory_desc_bytes);
    return RemoteSsdRead(peer, key, loc.location_id, dst, size);
  } else {
    spdlog::error("[PoolClient] Unsupported Get path: node={}, tier={}",
                  loc.node_id, TierTypeName(loc.tier));
    return false;
  }
}

bool PoolClient::Remove(const std::string& key) {
  if (!initialized_) {
    spdlog::error("[PoolClient] Not initialized");
    return false;
  }

  Location location;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = location_cache_.find(key);
    if (it == location_cache_.end()) {
      spdlog::warn("[PoolClient] Remove: key '{}' not in local cache", key);
      return false;
    }
    location = it->second;
  }

  uint32_t removed = 0;
  auto status = master_client_->Unregister(key, location, &removed);
  if (!status.ok()) {
    spdlog::error("[PoolClient] Unregister failed: {}", status.error_message());
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    location_cache_.erase(key);
  }

  return removed > 0;
}

MasterClient& PoolClient::Master() { return *master_client_; }

bool PoolClient::IsInitialized() const { return initialized_; }

bool PoolClient::PutLocalDram(const void* src, size_t size, uint64_t offset) {
  if (!config_.exportable_dram_buffer) {
    spdlog::error("[PoolClient] No exportable DRAM buffer configured");
    return false;
  }
  if (offset + size > config_.exportable_dram_buffer_size) {
    spdlog::error("[PoolClient] DRAM write out of bounds: offset={} size={} total={}",
                  offset, size, config_.exportable_dram_buffer_size);
    return false;
  }
  std::memcpy(static_cast<char*>(config_.exportable_dram_buffer) + offset, src,
              size);
  return true;
}

bool PoolClient::GetLocalDram(void* dst, size_t size, uint64_t offset) {
  if (!config_.exportable_dram_buffer) {
    spdlog::error("[PoolClient] No exportable DRAM buffer configured");
    return false;
  }
  if (offset + size > config_.exportable_dram_buffer_size) {
    spdlog::error("[PoolClient] DRAM read out of bounds: offset={} size={} total={}",
                  offset, size, config_.exportable_dram_buffer_size);
    return false;
  }
  std::memcpy(dst,
              static_cast<const char*>(config_.exportable_dram_buffer) + offset,
              size);
  return true;
}

bool PoolClient::PutLocalSsd(const std::string& key, const void* src,
                             size_t size) {
  if (config_.exportable_ssd_dir.empty()) {
    spdlog::error("[PoolClient] No SSD directory configured");
    return false;
  }

  std::filesystem::create_directories(config_.exportable_ssd_dir);
  std::string path = config_.exportable_ssd_dir + "/" + key + ".bin";

  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    spdlog::error("[PoolClient] Failed to open SSD file for write: {}", path);
    return false;
  }

  ssize_t written = ::write(fd, src, size);
  if (written < 0 || static_cast<size_t>(written) != size) {
    spdlog::error("[PoolClient] SSD write incomplete: {} of {} bytes", written,
                  size);
    ::close(fd);
    return false;
  }

  ::fsync(fd);
  ::close(fd);
  return true;
}

bool PoolClient::GetLocalSsd(const std::string& key, void* dst, size_t size) {
  if (config_.exportable_ssd_dir.empty()) {
    spdlog::error("[PoolClient] No SSD directory configured");
    return false;
  }

  std::string path = config_.exportable_ssd_dir + "/" + key + ".bin";

  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    spdlog::error("[PoolClient] Failed to open SSD file for read: {}", path);
    return false;
  }

  ssize_t bytes_read = ::read(fd, dst, size);
  if (bytes_read < 0 || static_cast<size_t>(bytes_read) != size) {
    spdlog::error("[PoolClient] SSD read incomplete: {} of {} bytes",
                  bytes_read, size);
    ::close(fd);
    return false;
  }

  ::close(fd);
  return true;
}

// ---------------------------------------------------------------------------
// Peer connection management
// ---------------------------------------------------------------------------

PoolClient::PeerConnection& PoolClient::GetOrConnectPeer(
    const std::string& node_id, const std::string& peer_address,
    const std::vector<uint8_t>& engine_desc_bytes,
    const std::vector<uint8_t>& dram_memory_desc_bytes) {
  std::lock_guard<std::mutex> lock(peers_mutex_);
  auto it = peers_.find(node_id);
  if (it != peers_.end()) return *it->second;

  auto peer = std::make_unique<PeerConnection>();
  peer->peer_address = peer_address;

  if (io_engine_ && !engine_desc_bytes.empty()) {
    auto handle = msgpack::unpack(
        reinterpret_cast<const char*>(engine_desc_bytes.data()),
        engine_desc_bytes.size());
    peer->engine_desc = handle.get().as<mori::io::EngineDesc>();
    io_engine_->RegisterRemoteEngine(peer->engine_desc);
    peer->engine_registered = true;
    spdlog::info("[PoolClient] Registered remote engine for node '{}'",
                 node_id);
  }

  if (!dram_memory_desc_bytes.empty()) {
    auto handle = msgpack::unpack(
        reinterpret_cast<const char*>(dram_memory_desc_bytes.data()),
        dram_memory_desc_bytes.size());
    peer->dram_memory = handle.get().as<mori::io::MemoryDesc>();
  }

  // Fetch staging_base_offset from PeerService for SSD ops
  if (!peer_address.empty()) {
    auto channel =
        grpc::CreateChannel(peer_address, grpc::InsecureChannelCredentials());
    auto stub = ::umbp::UMBPPeer::NewStub(channel);

    ::umbp::GetPeerInfoRequest req;
    ::umbp::GetPeerInfoResponse resp;
    grpc::ClientContext ctx;
    auto grpc_status = stub->GetPeerInfo(&ctx, req, &resp);
    if (grpc_status.ok()) {
      peer->staging_base_offset = resp.staging_base_offset();
      peer->peer_stub = std::unique_ptr<void, void (*)(void*)>(
          stub.release(), +[](void* p) {
            delete static_cast<::umbp::UMBPPeer::Stub*>(p);
          });
    } else {
      spdlog::warn(
          "[PoolClient] GetPeerInfo failed for node '{}': {}", node_id,
          grpc_status.error_message());
    }
  }

  auto* raw = peer.get();
  peers_.emplace(node_id, std::move(peer));
  return *raw;
}

// ---------------------------------------------------------------------------
// Remote DRAM path (pure RDMA)
// ---------------------------------------------------------------------------

bool PoolClient::RemoteDramWrite(PeerConnection& peer, const void* src,
                                 size_t size, uint64_t offset) {
  if (!io_engine_) return false;

  std::lock_guard<std::mutex> lock(staging_mutex_);
  std::memcpy(staging_buffer_.get(), src, size);

  auto uid = io_engine_->AllocateTransferUniqueId();
  mori::io::TransferStatus status;
  io_engine_->Write(staging_mem_, 0, peer.dram_memory, offset, size, &status,
                    uid);
  status.Wait();
  if (!status.Succeeded()) {
    spdlog::error("[PoolClient] RemoteDramWrite failed: {}", status.Message());
    return false;
  }
  return true;
}

bool PoolClient::RemoteDramRead(PeerConnection& peer, void* dst, size_t size,
                                uint64_t offset) {
  if (!io_engine_) return false;

  std::lock_guard<std::mutex> lock(staging_mutex_);

  auto uid = io_engine_->AllocateTransferUniqueId();
  mori::io::TransferStatus status;
  io_engine_->Read(staging_mem_, 0, peer.dram_memory, offset, size, &status,
                   uid);
  status.Wait();
  if (!status.Succeeded()) {
    spdlog::error("[PoolClient] RemoteDramRead failed: {}", status.Message());
    return false;
  }

  std::memcpy(dst, staging_buffer_.get(), size);
  return true;
}

// ---------------------------------------------------------------------------
// Remote SSD path (RDMA + PeerService gRPC coordination)
// ---------------------------------------------------------------------------

bool PoolClient::RemoteSsdWrite(PeerConnection& peer, const std::string& key,
                                const void* src, size_t size) {
  if (!io_engine_) return false;

  std::lock_guard<std::mutex> ssd_lock(peer.ssd_op_mutex);

  // Phase 1: RDMA write data into remote staging area
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    std::memcpy(staging_buffer_.get(), src, size);

    auto uid = io_engine_->AllocateTransferUniqueId();
    mori::io::TransferStatus status;
    io_engine_->Write(staging_mem_, 0, peer.dram_memory,
                      peer.staging_base_offset, size, &status, uid);
    status.Wait();
    if (!status.Succeeded()) {
      spdlog::error("[PoolClient] RemoteSsdWrite RDMA phase failed: {}",
                    status.Message());
      return false;
    }
  }

  // Phase 2: Ask PeerService to persist staging data to SSD
  if (!peer.peer_stub) {
    auto channel = grpc::CreateChannel(peer.peer_address,
                                       grpc::InsecureChannelCredentials());
    auto stub = ::umbp::UMBPPeer::NewStub(channel);
    peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
        stub.release(),
        +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  ::umbp::CommitSsdWriteRequest req;
  req.set_key(key);
  req.set_staging_offset(peer.staging_base_offset);
  req.set_size(size);

  ::umbp::CommitSsdWriteResponse resp;
  grpc::ClientContext ctx;
  auto grpc_status = stub->CommitSsdWrite(&ctx, req, &resp);
  if (!grpc_status.ok()) {
    spdlog::error("[PoolClient] CommitSsdWrite RPC failed: {}",
                  grpc_status.error_message());
    return false;
  }
  if (!resp.success()) {
    spdlog::error("[PoolClient] CommitSsdWrite rejected by peer for key={}",
                  key);
    return false;
  }

  return true;
}

bool PoolClient::RemoteSsdRead(PeerConnection& peer, const std::string& key,
                               const std::string& location_id, void* dst,
                               size_t size) {
  if (!io_engine_) return false;

  std::lock_guard<std::mutex> ssd_lock(peer.ssd_op_mutex);

  // Phase 1: Ask PeerService to load SSD data into staging
  if (!peer.peer_stub) {
    auto channel = grpc::CreateChannel(peer.peer_address,
                                       grpc::InsecureChannelCredentials());
    auto s = ::umbp::UMBPPeer::NewStub(channel);
    peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
        s.release(),
        +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
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
    spdlog::error("[PoolClient] PrepareSsdRead RPC failed: {}",
                  grpc_status.error_message());
    return false;
  }
  if (!resp.success()) {
    spdlog::error("[PoolClient] PrepareSsdRead failed for key={}", key);
    return false;
  }

  // Phase 2: RDMA read from remote staging area
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);

    auto uid = io_engine_->AllocateTransferUniqueId();
    mori::io::TransferStatus status;
    io_engine_->Read(staging_mem_, 0, peer.dram_memory, resp.staging_offset(),
                     size, &status, uid);
    status.Wait();
    if (!status.Succeeded()) {
      spdlog::error("[PoolClient] RemoteSsdRead RDMA phase failed: {}",
                    status.Message());
      return false;
    }

    std::memcpy(dst, staging_buffer_.get(), size);
  }

  return true;
}

}  // namespace mori::umbp

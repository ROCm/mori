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

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <msgpack.hpp>
#include <unordered_map>

#include "mori/io/backend.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {

bool IsValidMemoryDesc(const mori::io::MemoryDesc& desc) { return desc.size > 0; }

// Bytes belonging to the i-th logical page of a Put/Get spread across
// `num_pages` pages of `page_size` bytes.  Last page may be partial.
inline uint64_t LogicalPageBytes(size_t i, size_t num_pages, uint64_t page_size,
                                 size_t total_size) {
  return (i + 1 == num_pages) ? (total_size - i * page_size) : page_size;
}

bool SizeMatchesAllocation(uint64_t size, size_t num_pages, uint64_t page_size) {
  if (page_size == 0 || num_pages == 0 || size == 0) return false;
  if (size > num_pages * page_size) return false;
  if (size <= (num_pages - 1) * page_size) return false;
  return true;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  if (alignment == 0) return value;
  uint64_t remainder = value % alignment;
  return remainder == 0 ? value : (value + alignment - remainder);
}

// Group `pages` by buffer_index, preserving first-seen ordering so
// IOEngine BatchWrite/BatchRead pair indexing stays predictable.
struct ScatterGroup {
  uint32_t buffer_index;
  std::vector<size_t> src_page_indices;
};
std::vector<ScatterGroup> GroupPagesByBuffer(const std::vector<PageLocation>& pages) {
  std::vector<ScatterGroup> groups;
  std::unordered_map<uint32_t, size_t> buf_to_group;
  groups.reserve(pages.size());
  for (size_t i = 0; i < pages.size(); ++i) {
    uint32_t bi = pages[i].buffer_index;
    auto it = buf_to_group.find(bi);
    if (it == buf_to_group.end()) {
      buf_to_group.emplace(bi, groups.size());
      groups.push_back({bi, {}});
      it = buf_to_group.find(bi);
    }
    groups[it->second].src_page_indices.push_back(i);
  }
  return groups;
}

uint32_t ReleaseLeaseMaxRetries() {
  static const uint32_t v = GetEnvUint32("UMBP_RELEASE_LEASE_MAX_RETRIES", 2, /*min_allowed=*/1);
  return v;
}

// Build a TierConfig from the PoolClientConfig (DRAM only — HBM
// support requires per-tier buffer plumbing the upper layers don't
// currently provide).  Returns an empty config when the engine has
// no DRAM buffers, signalling that no DRAM allocator should be built.
PeerDramAllocator::TierConfig BuildDramTierConfig(const std::vector<ExportableDram>& bufs,
                                                  const std::vector<mori::io::MemoryDesc>& mems) {
  PeerDramAllocator::TierConfig cfg;
  if (bufs.size() != mems.size()) return cfg;
  for (size_t i = 0; i < bufs.size(); ++i) {
    cfg.buffer_sizes.push_back(bufs[i].size);
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, mems[i]);
    cfg.buffer_descs.emplace_back(sbuf.data(), sbuf.data() + sbuf.size());
  }
  return cfg;
}

// Translate a peer-side ::umbp::AllocateSlotResponse / ResolveKeyResponse
// into the C++ shapes our code consumes.
PoolClient::SlotPlan FromAllocateSlotResponse(const ::umbp::AllocateSlotResponse& resp) {
  PoolClient::SlotPlan p;
  p.slot_id = resp.slot_id();
  p.page_size = resp.page_size();
  p.pages.reserve(resp.pages_size());
  for (const auto& pp : resp.pages()) p.pages.push_back({pp.buffer_index(), pp.page_index()});
  p.descs.reserve(resp.descs_size());
  for (const auto& d : resp.descs()) {
    BufferMemoryDescBytes b;
    b.buffer_index = d.buffer_index();
    b.desc_bytes.assign(d.desc().begin(), d.desc().end());
    p.descs.push_back(std::move(b));
  }
  return p;
}

PoolClient::SlotPlan FromResolveKeyResponse(const ::umbp::ResolveKeyResponse& resp) {
  PoolClient::SlotPlan p;
  p.page_size = resp.page_size();
  p.pages.reserve(resp.pages_size());
  for (const auto& pp : resp.pages()) p.pages.push_back({pp.buffer_index(), pp.page_index()});
  p.descs.reserve(resp.descs_size());
  for (const auto& d : resp.descs()) {
    BufferMemoryDescBytes b;
    b.buffer_index = d.buffer_index();
    b.desc_bytes.assign(d.desc().begin(), d.desc().end());
    p.descs.push_back(std::move(b));
  }
  return p;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

PoolClient::PoolClient(PoolClientConfig config) : config_(std::move(config)) {}
PoolClient::~PoolClient() { Shutdown(); }

bool PoolClient::Init() {
  bool expected = false;
  if (!initialized_.compare_exchange_strong(expected, true)) return true;

  master_client_ = std::make_unique<MasterClient>(config_.master_config);

  // IO Engine setup (RDMA data plane).
  if (!config_.io_engine.host.empty()) {
    mori::io::IOEngineConfig io_cfg;
    io_cfg.host = config_.io_engine.host;
    io_cfg.port = config_.io_engine.port;
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
                   config_.io_engine.host, config_.io_engine.port, export_dram_mems_.size());
  }

  // Peer-side allocator: even SSD-only deployments build one (with
  // empty DRAM/HBM tiers) so the SSD CommitSsdWrite path has an event
  // outbox to push ADD events into.
  const uint64_t page_size =
      config_.dram_page_size > 0 ? config_.dram_page_size : 2ULL * 1024 * 1024;
  PeerDramAllocator::TierConfig dram_cfg =
      io_engine_ ? BuildDramTierConfig(config_.dram_buffers, export_dram_mems_)
                 : PeerDramAllocator::TierConfig{};
  PeerDramAllocator::TierConfig hbm_cfg;  // HBM not currently exposed via PoolClientConfig
  peer_alloc_ =
      std::make_unique<PeerDramAllocator>(page_size, std::move(dram_cfg), std::move(hbm_cfg),
                                          /*pending_ttl=*/std::chrono::milliseconds{30000});
  peer_alloc_->StartReaper();
  master_client_->SetPeerDramAllocator(peer_alloc_.get());

  // Pack engine_desc for master registration.
  std::vector<uint8_t> engine_desc_bytes;
  if (io_engine_) {
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, io_engine_->GetEngineDesc());
    engine_desc_bytes.assign(sbuf.data(), sbuf.data() + sbuf.size());
  }

  // SSD staging buffer (one per process; not part of DRAM exports).
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

  if (config_.peer_service_port > 0) {
    peer_service_ =
        std::make_unique<PeerServiceServer>(peer_alloc_.get(), 8, 8, 10, engine_desc_bytes);
    if (!peer_service_->Start(config_.peer_service_port)) {
      MORI_UMBP_ERROR("[PoolClient] PeerService failed to start on port {}",
                      config_.peer_service_port);
      peer_service_.reset();
      initialized_ = false;
      return false;
    }
  }

  std::string peer_address;
  if (config_.peer_service_port > 0) {
    std::string host = config_.master_config.node_address;
    peer_address = host + ":" + std::to_string(config_.peer_service_port);
  }

  std::vector<uint64_t> ssd_store_capacities;
  for (const auto& store : config_.ssd_stores) ssd_store_capacities.push_back(store.capacity);

  // Master register.  In the new design master holds no DRAM-side
  // metadata; only membership + capacity-snapshot.
  auto status = master_client_->RegisterSelf(config_.tier_capacities, peer_address,
                                             engine_desc_bytes, ssd_store_capacities);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] RegisterSelf failed: {}", status.error_message());
    initialized_ = false;
    return false;
  }

  if (config_.master_config.auto_heartbeat) master_client_->StartHeartbeat();

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

  peer_service_.reset();

  if (peer_alloc_) {
    peer_alloc_->StopReaper();
    peer_alloc_.reset();
  }

  if (io_engine_) {
    {
      std::lock_guard<std::mutex> lock(registered_mem_mutex_);
      for (auto& reg : registered_regions_) io_engine_->DeregisterMemory(reg.mem_desc);
      registered_regions_.clear();
    }
    if (staging_buffer_) io_engine_->DeregisterMemory(staging_mem_);
    if (ssd_staging_buffer_) {
      io_engine_->DeregisterMemory(ssd_staging_mem_);
      ssd_staging_buffer_.reset();
    }
    for (auto& mem : export_dram_mems_) io_engine_->DeregisterMemory(mem);
    export_dram_mems_.clear();
    io_engine_.reset();
    staging_buffer_.reset();
  }

  master_client_.reset();
}

bool PoolClient::IsInitialized() const { return initialized_; }
MasterClient& PoolClient::Master() { return *master_client_; }
PeerDramAllocator* PoolClient::DramAllocator() { return peer_alloc_.get(); }

// ---------------------------------------------------------------------------
//  Memory registration
// ---------------------------------------------------------------------------

bool PoolClient::RegisterMemory(void* ptr, size_t size) {
  if (!io_engine_) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory: IOEngine not available");
    return false;
  }
  if (ptr == nullptr || size == 0) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory: invalid args ptr={}, size={}", ptr, size);
    return false;
  }
  std::lock_guard<std::mutex> lock(registered_mem_mutex_);
  for (auto& reg : registered_regions_) {
    if (reg.base == ptr) return true;
  }
  auto mem_desc = io_engine_->RegisterMemory(ptr, size, -1, mori::io::MemoryLocationType::CPU);
  registered_regions_.push_back({ptr, size, mem_desc});
  return true;
}

void PoolClient::DeregisterMemory(void* ptr) {
  if (ptr == nullptr) return;
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
    if (addr >= base && size <= reg.size && (addr - base) <= reg.size - size) {
      return std::pair{reg.mem_desc, static_cast<size_t>(addr - base)};
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
//  Self-target fast paths
// ---------------------------------------------------------------------------

bool PoolClient::LocalPutPages(const std::vector<PageLocation>& pages, uint64_t page_size,
                               const void* src, size_t size) {
  const char* src_bytes = static_cast<const char*>(src);
  for (size_t i = 0; i < pages.size(); ++i) {
    const auto& p = pages[i];
    if (p.buffer_index >= config_.dram_buffers.size()) {
      MORI_UMBP_ERROR("[PoolClient] local Put: invalid buffer_index {}", p.buffer_index);
      return false;
    }
    auto& dram = config_.dram_buffers[p.buffer_index];
    const uint64_t off = static_cast<uint64_t>(p.page_index) * page_size;
    if (!dram.buffer || page_size > dram.size || off > dram.size - page_size) {
      MORI_UMBP_ERROR("[PoolClient] local Put: OOB buf={} off={}", p.buffer_index, off);
      return false;
    }
    const uint64_t bytes = LogicalPageBytes(i, pages.size(), page_size, size);
    std::memcpy(static_cast<char*>(dram.buffer) + off, src_bytes + i * page_size, bytes);
  }
  return true;
}

bool PoolClient::LocalGetPages(const std::vector<PageLocation>& pages, uint64_t page_size,
                               void* dst, size_t size) {
  char* dst_bytes = static_cast<char*>(dst);
  for (size_t i = 0; i < pages.size(); ++i) {
    const auto& p = pages[i];
    if (p.buffer_index >= config_.dram_buffers.size()) {
      MORI_UMBP_ERROR("[PoolClient] local Get: invalid buffer_index {}", p.buffer_index);
      return false;
    }
    auto& dram = config_.dram_buffers[p.buffer_index];
    const uint64_t off = static_cast<uint64_t>(p.page_index) * page_size;
    if (!dram.buffer || page_size > dram.size || off > dram.size - page_size) {
      MORI_UMBP_ERROR("[PoolClient] local Get: OOB buf={} off={}", p.buffer_index, off);
      return false;
    }
    const uint64_t bytes = LogicalPageBytes(i, pages.size(), page_size, size);
    std::memcpy(dst_bytes + i * page_size, static_cast<const char*>(dram.buffer) + off, bytes);
  }
  return true;
}

PoolClient::PutAttemptOutcome PoolClient::ExecuteLocalPut(const std::string& key, const void* src,
                                                          size_t size, TierType tier) {
  if (!peer_alloc_) {
    MORI_UMBP_ERROR("[PoolClient] Local Put requested but peer allocator unavailable");
    return PutAttemptOutcome::kFatal;
  }
  auto pending = peer_alloc_->Allocate(size, tier);
  if (!pending) {
    return PutAttemptOutcome::kRetry;
  }
  if (!LocalPutPages(pending->pages, peer_alloc_->PageSize(), src, size)) {
    peer_alloc_->Abort(pending->slot_id);
    return PutAttemptOutcome::kFatal;
  }
  if (!peer_alloc_->Commit(pending->slot_id, key)) {
    peer_alloc_->Abort(pending->slot_id);
    return PutAttemptOutcome::kFatal;
  }
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                             static_cast<double>(size));
  return PutAttemptOutcome::kSuccess;
}

PoolClient::GetAttemptOutcome PoolClient::ExecuteLocalGet(const std::string& key, void* dst,
                                                          size_t size) {
  if (!peer_alloc_) {
    MORI_UMBP_ERROR("[PoolClient] Local Get requested but peer allocator unavailable");
    return GetAttemptOutcome::kFatal;
  }
  auto resolved = peer_alloc_->Resolve(key);
  if (!resolved.found) {
    return GetAttemptOutcome::kRetry;
  }
  if (!LocalGetPages(resolved.pages, peer_alloc_->PageSize(), dst, size)) {
    return GetAttemptOutcome::kFatal;
  }
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP, {{"traffic", "local"}},
                             static_cast<double>(size));
  return GetAttemptOutcome::kSuccess;
}

std::unordered_map<std::string, std::vector<PoolClient::BatchPutItem>>
PoolClient::PartitionBatchPutTargets(const std::vector<std::string>& keys,
                                     const std::vector<const void*>& srcs,
                                     const std::vector<size_t>& sizes,
                                     const std::vector<std::optional<RoutePutResult>>& routes,
                                     std::vector<bool>* results) {
  std::unordered_map<std::string, std::vector<BatchPutItem>> remote_groups;
  const size_t count = keys.size();
  for (size_t i = 0; i < count; ++i) {
    if (i >= routes.size() || !routes[i].has_value()) {
      continue;
    }
    const auto& route = routes[i].value();
    if (route.node_id == config_.master_config.node_id) {
      auto outcome = ExecuteLocalPut(keys[i], srcs[i], sizes[i], route.tier);
      if (outcome == PutAttemptOutcome::kSuccess) {
        (*results)[i] = true;
      }
      continue;
    }
    if (route.tier != TierType::DRAM && route.tier != TierType::HBM) {
      (*results)[i] = false;
      continue;
    }
    BatchPutItem item{
        .index = i, .key = &keys[i], .src = srcs[i], .size = sizes[i], .route = route};
    remote_groups[route.node_id].push_back(std::move(item));
  }
  return remote_groups;
}

// ---------------------------------------------------------------------------
//  Put / Get hot paths
// ---------------------------------------------------------------------------

bool PoolClient::Put(const std::string& key, const void* src, size_t size) {
  std::vector<std::string> keys{key};
  std::vector<const void*> srcs{src};
  std::vector<size_t> sizes{size};
  auto results = BatchPut(keys, srcs, sizes);
  return !results.empty() && results[0];
}

bool PoolClient::Get(const std::string& key, void* dst, size_t size) {
  std::vector<std::string> keys{key};
  std::vector<void*> dsts{dst};
  std::vector<size_t> sizes{size};
  auto results = BatchGet(keys, dsts, sizes);
  return !results.empty() && results[0];
}

std::vector<bool> PoolClient::BatchPut(const std::vector<std::string>& keys,
                                       const std::vector<const void*>& srcs,
                                       const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.size() != srcs.size() || keys.size() != sizes.size()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: vector length mismatch");
    return results;
  }
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: client not initialized");
    return results;
  }

  std::vector<uint64_t> block_sizes(keys.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    block_sizes[i] = static_cast<uint64_t>(sizes[i]);
  }
  std::vector<std::optional<RoutePutResult>> routes;
  std::unordered_set<std::string> excludes;
  auto status = master_client_->BatchRoutePut(keys, block_sizes, excludes, &routes);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: BatchRoutePut failed: {}", status.error_message());
    return results;
  }
  if (routes.size() < keys.size()) {
    routes.resize(keys.size());
  }

  auto remote_groups = PartitionBatchPutTargets(keys, srcs, sizes, routes, &results);

  for (auto& [node_id, items] : remote_groups) {
    ProcessRemoteBatchPut(items, &results);
  }

  // Entries without routing advice or still false fall back to individual attempts.
  for (size_t i = 0; i < keys.size(); ++i) {
    if (results[i]) continue;
    results[i] = false;
  }
  return results;
}

std::vector<bool> PoolClient::BatchGet(const std::vector<std::string>& keys,
                                       const std::vector<void*>& dsts,
                                       const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.size() != dsts.size() || keys.size() != sizes.size()) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: vector length mismatch");
    return results;
  }
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: client not initialized");
    return results;
  }

  std::vector<std::optional<RouteGetResult>> routes;
  std::unordered_set<std::string> excludes;
  auto status = master_client_->BatchRouteGet(keys, excludes, &routes);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: BatchRouteGet failed: {}", status.error_message());
    return results;
  }
  if (routes.size() < keys.size()) {
    routes.resize(keys.size());
  }

  std::unordered_map<std::string, std::vector<BatchGetItem>> remote_groups;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (i >= routes.size() || !routes[i].has_value()) {
      continue;
    }
    const auto& route = routes[i].value();
    if (route.node_id == config_.master_config.node_id) {
      auto outcome = ExecuteLocalGet(keys[i], const_cast<void*>(dsts[i]), sizes[i]);
      if (outcome == GetAttemptOutcome::kSuccess) {
        results[i] = true;
      }
      continue;
    }
    if (route.tier != TierType::DRAM && route.tier != TierType::HBM) {
      results[i] = false;
      continue;
    }
    BatchGetItem item{.index = i,
                      .key = &keys[i],
                      .dst = const_cast<void*>(dsts[i]),
                      .size = sizes[i],
                      .route = route};
    remote_groups[route.node_id].push_back(std::move(item));
  }

  for (auto& [node_id, items] : remote_groups) {
    ProcessRemoteBatchGet(items, &results);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    if (results[i]) continue;
    results[i] = false;
  }
  return results;
}

void PoolClient::ProcessRemoteBatchPut(const std::vector<BatchPutItem>& items,
                                       std::vector<bool>* results) {
  if (items.empty()) return;
  if (!io_engine_) {
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return;
  }

  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return;
  }

  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());
  ExecuteRemoteBatchPut(items, results, peer, stub);
}

void PoolClient::ExecuteRemoteBatchPut(const std::vector<BatchPutItem>& items,
                                       std::vector<bool>* results, PeerConnection& peer,
                                       ::umbp::UMBPPeer::Stub* stub) {
  if (items.empty()) return;
  if (!io_engine_) {
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return;
  }

  std::vector<RemotePutEntry> entries;
  std::vector<uint64_t> abort_slots;
  if (!AllocateRemotePutEntries(items, stub, &entries, &abort_slots, results)) {
    return;
  }

  std::vector<TransferInstruction> transfers;
  uint64_t staging_bytes = 0;
  if (!BuildRemotePutTransfers(entries, peer, &transfers, &staging_bytes)) {
    for (auto& entry : entries) {
      abort_slots.push_back(entry.slot_id);
      (*results)[entry.result_index] = false;
    }
    if (!abort_slots.empty()) {
      ::umbp::BatchAbortSlotsRequest abort_req;
      for (uint64_t slot_id : abort_slots) abort_req.add_slot_ids(slot_id);
      ::umbp::BatchAbortSlotsResponse abort_resp;
      grpc::ClientContext abort_ctx;
      stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
    }
    return;
  }

  ExecuteRemotePutTransfers(entries, transfers, staging_bytes);
  FinalizeRemotePutEntries(entries, abort_slots, results, stub);
}

bool PoolClient::AllocateRemotePutEntries(const std::vector<BatchPutItem>& items,
                                          ::umbp::UMBPPeer::Stub* stub,
                                          std::vector<RemotePutEntry>* entries,
                                          std::vector<uint64_t>* abort_slots,
                                          std::vector<bool>* results) {
  entries->clear();
  ::umbp::BatchAllocateSlotsRequest alloc_req;
  for (const auto& item : items) {
    auto* entry = alloc_req.add_entries();
    entry->set_size(item.size);
    entry->set_tier(static_cast<::umbp::TierType>(item.route.tier));
  }

  ::umbp::BatchAllocateSlotsResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  auto alloc_status = stub->BatchAllocateSlots(&alloc_ctx, alloc_req, &alloc_resp);
  if (!alloc_status.ok() || alloc_resp.entries_size() != static_cast<int>(items.size())) {
    MORI_UMBP_WARN("[PoolClient] BatchAllocateSlots failed on {}: {}", items.front().route.node_id,
                   alloc_status.error_message());
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return false;
  }

  entries->reserve(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& resp_entry = alloc_resp.entries(static_cast<int>(i));
    if (!resp_entry.success()) {
      (*results)[item.index] = false;
      continue;
    }

    PoolClient::SlotPlan plan = FromAllocateSlotResponse(resp_entry);
    if (!SizeMatchesAllocation(item.size, plan.pages.size(), plan.page_size)) {
      MORI_UMBP_ERROR("[PoolClient] BatchPut: malformed slot for key='{}'", *item.key);
      abort_slots->push_back(plan.slot_id);
      (*results)[item.index] = false;
      continue;
    }

    RemotePutEntry entry;
    entry.result_index = item.index;
    entry.item = &item;
    entry.slot_id = plan.slot_id;
    entry.plan = std::move(plan);
    entries->push_back(std::move(entry));
  }

  if (entries->empty()) {
    if (!abort_slots->empty()) {
      ::umbp::BatchAbortSlotsRequest abort_req;
      for (uint64_t slot_id : *abort_slots) abort_req.add_slot_ids(slot_id);
      ::umbp::BatchAbortSlotsResponse abort_resp;
      grpc::ClientContext abort_ctx;
      stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
      abort_slots->clear();
    }
    return false;
  }
  return true;
}

bool PoolClient::BuildRemotePutTransfers(std::vector<RemotePutEntry>& entries, PeerConnection& peer,
                                         std::vector<TransferInstruction>* transfers,
                                         uint64_t* staging_bytes) {
  transfers->clear();
  uint64_t cursor = 0;

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    EnsureBufferDescsCached(peer, entry.plan.descs);

    auto zero_copy = FindRegisteredMemory(entry.item->src, entry.item->size);
    if (zero_copy.has_value()) {
      entry.zero_copy = zero_copy;
      entry.use_staging = false;
      entry.staging_offset = 0;
    } else {
      entry.zero_copy.reset();
      entry.use_staging = true;
      cursor = AlignUp(cursor, entry.plan.page_size);
      const uint64_t aligned_size = AlignUp(entry.item->size, entry.plan.page_size);
      if (cursor + aligned_size > config_.staging_buffer_size) return false;
      entry.staging_offset = cursor;
      cursor += aligned_size;
    }

    mori::io::MemoryDesc local_desc{};
    uint64_t base_offset = 0;
    if (entry.use_staging) {
      local_desc = staging_mem_;
      base_offset = entry.staging_offset;
    } else {
      local_desc = entry.zero_copy->first;
      base_offset = entry.zero_copy->second;
    }

    std::vector<TransferInstruction> entry_transfers;
    entry_transfers.reserve(entry.plan.pages.size());
    for (size_t p = 0; p < entry.plan.pages.size(); ++p) {
      const auto& page = entry.plan.pages[p];
      if (page.buffer_index >= peer.dram_memories.size() ||
          !IsValidMemoryDesc(peer.dram_memories[page.buffer_index])) {
        entry.failed = true;
        entry_transfers.clear();
        break;
      }
      TransferInstruction instr;
      instr.entry_index = idx;
      instr.local_desc = local_desc;
      instr.local_offset = base_offset + static_cast<uint64_t>(p) * entry.plan.page_size;
      instr.remote_desc = peer.dram_memories[page.buffer_index];
      instr.remote_offset = static_cast<uint64_t>(page.page_index) * entry.plan.page_size;
      instr.size =
          LogicalPageBytes(p, entry.plan.pages.size(), entry.plan.page_size, entry.item->size);
      entry_transfers.push_back(std::move(instr));
    }

    if (!entry_transfers.empty()) {
      transfers->insert(transfers->end(), entry_transfers.begin(), entry_transfers.end());
    }
  }

  *staging_bytes = cursor;
  return true;
}

void PoolClient::ExecuteRemotePutTransfers(std::vector<RemotePutEntry>& entries,
                                           std::vector<TransferInstruction>& transfers,
                                           uint64_t staging_bytes) {
  std::unique_lock<std::mutex> staging_lock(staging_mutex_, std::defer_lock);
  if (staging_bytes > 0) staging_lock.lock();

  for (auto& entry : entries) {
    if (entry.failed) continue;
    if (entry.use_staging) {
      if (entry.staging_offset + entry.item->size > config_.staging_buffer_size) {
        entry.failed = true;
        continue;
      }
      std::memcpy(staging_buffer_.get() + entry.staging_offset, entry.item->src, entry.item->size);
    }
  }

  std::vector<TransferInstruction> active_transfers;
  active_transfers.reserve(transfers.size());
  for (const auto& transfer : transfers) {
    if (!entries[transfer.entry_index].failed) {
      active_transfers.push_back(transfer);
    }
  }
  if (active_transfers.empty()) {
    if (staging_lock.owns_lock()) staging_lock.unlock();
    return;
  }

  const size_t N = active_transfers.size();
  mori::io::MemDescVec local_descs(N), remote_descs(N);
  mori::io::BatchSizeVec local_offsets(N), remote_offsets(N), sizes_v(N);
  std::vector<mori::io::TransferStatus> statuses(N);
  mori::io::TransferStatusPtrVec status_ptrs(N);
  mori::io::TransferUniqueIdVec ids(N);
  for (size_t i = 0; i < N; ++i) {
    const auto& transfer = active_transfers[i];
    local_descs[i] = transfer.local_desc;
    remote_descs[i] = transfer.remote_desc;
    local_offsets[i].push_back(transfer.local_offset);
    remote_offsets[i].push_back(transfer.remote_offset);
    sizes_v[i].push_back(transfer.size);
    status_ptrs[i] = &statuses[i];
    ids[i] = io_engine_->AllocateTransferUniqueId();
  }

  io_engine_->BatchWrite(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v,
                         status_ptrs, ids);
  for (size_t i = 0; i < active_transfers.size(); ++i) {
    statuses[i].Wait();
    if (!statuses[i].Succeeded()) {
      entries[active_transfers[i].entry_index].failed = true;
    }
  }

  if (staging_lock.owns_lock()) staging_lock.unlock();
}

void PoolClient::FinalizeRemotePutEntries(std::vector<RemotePutEntry>& entries,
                                          std::vector<uint64_t>& abort_slots,
                                          std::vector<bool>* results,
                                          ::umbp::UMBPPeer::Stub* stub) {
  ::umbp::BatchCommitSlotsRequest commit_req;
  std::vector<size_t> commit_indices;
  commit_indices.reserve(entries.size());

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    if (entry.failed) {
      abort_slots.push_back(entry.slot_id);
      (*results)[entry.result_index] = false;
      continue;
    }
    auto* commit = commit_req.add_entries();
    commit->set_slot_id(entry.slot_id);
    commit->set_key(*entry.item->key);
    commit_indices.push_back(idx);
  }

  if (!commit_indices.empty()) {
    ::umbp::BatchCommitSlotsResponse commit_resp;
    grpc::ClientContext commit_ctx;
    auto commit_status = stub->BatchCommitSlots(&commit_ctx, commit_req, &commit_resp);
    if (!commit_status.ok() ||
        commit_resp.success_size() != static_cast<int>(commit_indices.size())) {
      const std::string& node_id = entries[commit_indices.front()].item->route.node_id;
      MORI_UMBP_WARN("[PoolClient] BatchCommitSlots failed on {}: {}", node_id,
                     commit_status.error_message());
      for (size_t idx : commit_indices) {
        abort_slots.push_back(entries[idx].slot_id);
        (*results)[entries[idx].result_index] = false;
        entries[idx].failed = true;
      }
    } else {
      for (size_t i = 0; i < commit_indices.size(); ++i) {
        auto idx = commit_indices[i];
        auto& entry = entries[idx];
        if (commit_resp.success(static_cast<int>(i))) {
          master_client_->AddCounter(
              MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL, MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL_HELP,
              {{"traffic", "remote"}}, static_cast<double>(entry.item->size));
          (*results)[entry.result_index] = true;
        } else {
          abort_slots.push_back(entry.slot_id);
          (*results)[entry.result_index] = false;
          entry.failed = true;
        }
      }
    }
  }

  if (!abort_slots.empty()) {
    ::umbp::BatchAbortSlotsRequest abort_req;
    for (uint64_t slot_id : abort_slots) abort_req.add_slot_ids(slot_id);
    ::umbp::BatchAbortSlotsResponse abort_resp;
    grpc::ClientContext abort_ctx;
    stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
    abort_slots.clear();
  }
}

void PoolClient::ProcessRemoteBatchGet(const std::vector<BatchGetItem>& items,
                                       std::vector<bool>* results) {
  if (items.empty()) return;
  if (!io_engine_) {
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return;
  }

  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return;
  }

  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());
  ExecuteRemoteBatchGet(items, results, peer, stub);
}

void PoolClient::ExecuteRemoteBatchGet(const std::vector<BatchGetItem>& items,
                                       std::vector<bool>* results, PeerConnection& peer,
                                       ::umbp::UMBPPeer::Stub* stub) {
  if (items.empty()) return;
  if (!io_engine_) {
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return;
  }

  std::vector<RemoteGetEntry> entries;
  if (!PrepareRemoteGetEntries(items, stub, &entries, results)) {
    return;
  }

  std::vector<TransferInstruction> transfers;
  uint64_t staging_bytes = 0;
  if (!BuildRemoteGetTransfers(entries, peer, &transfers, &staging_bytes)) {
    for (auto& entry : entries) {
      (*results)[entry.result_index] = false;
    }
    return;
  }

  ExecuteRemoteGetTransfers(entries, transfers, staging_bytes);
  FinalizeRemoteGetEntries(entries, results);
}

bool PoolClient::PrepareRemoteGetEntries(const std::vector<BatchGetItem>& items,
                                         ::umbp::UMBPPeer::Stub* stub,
                                         std::vector<RemoteGetEntry>* entries,
                                         std::vector<bool>* results) {
  entries->clear();
  ::umbp::BatchResolveKeysRequest resolve_req;
  for (const auto& item : items) resolve_req.add_keys(*item.key);

  ::umbp::BatchResolveKeysResponse resolve_resp;
  grpc::ClientContext resolve_ctx;
  auto resolve_status = stub->BatchResolveKeys(&resolve_ctx, resolve_req, &resolve_resp);
  if (!resolve_status.ok() || resolve_resp.entries_size() != static_cast<int>(items.size())) {
    MORI_UMBP_WARN("[PoolClient] BatchResolveKeys failed on {}: {}", items.front().route.node_id,
                   resolve_status.error_message());
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return false;
  }

  entries->reserve(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& resp_entry = resolve_resp.entries(static_cast<int>(i));
    if (!resp_entry.found()) {
      (*results)[item.index] = false;
      continue;
    }
    if (resp_entry.size() != item.size) {
      MORI_UMBP_WARN("[PoolClient] BatchGet: size mismatch for key='{}' (wanted {}, got {})",
                     *item.key, item.size, resp_entry.size());
      (*results)[item.index] = false;
      continue;
    }
    PoolClient::SlotPlan plan = FromResolveKeyResponse(resp_entry);
    if (!SizeMatchesAllocation(item.size, plan.pages.size(), plan.page_size)) {
      MORI_UMBP_ERROR("[PoolClient] BatchGet: malformed slot for key='{}'", *item.key);
      (*results)[item.index] = false;
      continue;
    }

    RemoteGetEntry entry;
    entry.result_index = item.index;
    entry.item = &item;
    entry.plan = std::move(plan);
    entries->push_back(std::move(entry));
  }

  return !entries->empty();
}

bool PoolClient::BuildRemoteGetTransfers(std::vector<RemoteGetEntry>& entries, PeerConnection& peer,
                                         std::vector<TransferInstruction>* transfers,
                                         uint64_t* staging_bytes) {
  transfers->clear();
  uint64_t cursor = 0;

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    EnsureBufferDescsCached(peer, entry.plan.descs);

    auto zero_copy = FindRegisteredMemory(entry.item->dst, entry.item->size);
    if (zero_copy.has_value()) {
      entry.zero_copy = zero_copy;
      entry.use_staging = false;
      entry.staging_offset = 0;
    } else {
      entry.zero_copy.reset();
      entry.use_staging = true;
      cursor = AlignUp(cursor, entry.plan.page_size);
      const uint64_t aligned_size = AlignUp(entry.item->size, entry.plan.page_size);
      if (cursor + aligned_size > config_.staging_buffer_size) return false;
      entry.staging_offset = cursor;
      cursor += aligned_size;
    }

    mori::io::MemoryDesc local_desc{};
    uint64_t base_offset = 0;
    if (entry.use_staging) {
      local_desc = staging_mem_;
      base_offset = entry.staging_offset;
    } else if (entry.zero_copy.has_value()) {
      local_desc = entry.zero_copy->first;
      base_offset = entry.zero_copy->second;
    } else {
      entry.failed = true;
      continue;
    }

    std::vector<TransferInstruction> entry_transfers;
    entry_transfers.reserve(entry.plan.pages.size());
    for (size_t p = 0; p < entry.plan.pages.size(); ++p) {
      const auto& page = entry.plan.pages[p];
      if (page.buffer_index >= peer.dram_memories.size() ||
          !IsValidMemoryDesc(peer.dram_memories[page.buffer_index])) {
        entry.failed = true;
        entry_transfers.clear();
        break;
      }
      TransferInstruction instr;
      instr.entry_index = idx;
      instr.local_desc = local_desc;
      instr.local_offset = base_offset + static_cast<uint64_t>(p) * entry.plan.page_size;
      instr.remote_desc = peer.dram_memories[page.buffer_index];
      instr.remote_offset = static_cast<uint64_t>(page.page_index) * entry.plan.page_size;
      instr.size =
          LogicalPageBytes(p, entry.plan.pages.size(), entry.plan.page_size, entry.item->size);
      entry_transfers.push_back(std::move(instr));
    }

    if (!entry_transfers.empty()) {
      transfers->insert(transfers->end(), entry_transfers.begin(), entry_transfers.end());
    }
  }

  *staging_bytes = cursor;
  return true;
}

void PoolClient::ExecuteRemoteGetTransfers(std::vector<RemoteGetEntry>& entries,
                                           std::vector<TransferInstruction>& transfers,
                                           uint64_t staging_bytes) {
  std::unique_lock<std::mutex> staging_lock(staging_mutex_, std::defer_lock);
  if (staging_bytes > 0) staging_lock.lock();

  std::vector<TransferInstruction> active_transfers;
  active_transfers.reserve(transfers.size());
  for (const auto& transfer : transfers) {
    if (!entries[transfer.entry_index].failed) {
      active_transfers.push_back(transfer);
    }
  }
  if (active_transfers.empty()) {
    if (staging_lock.owns_lock()) staging_lock.unlock();
    return;
  }

  const size_t N = active_transfers.size();
  mori::io::MemDescVec local_descs(N), remote_descs(N);
  mori::io::BatchSizeVec local_offsets(N), remote_offsets(N), sizes_v(N);
  std::vector<mori::io::TransferStatus> statuses(N);
  mori::io::TransferStatusPtrVec status_ptrs(N);
  mori::io::TransferUniqueIdVec ids(N);
  for (size_t i = 0; i < N; ++i) {
    const auto& transfer = active_transfers[i];
    remote_descs[i] = transfer.remote_desc;
    local_descs[i] = transfer.local_desc;
    remote_offsets[i].push_back(transfer.remote_offset);
    local_offsets[i].push_back(transfer.local_offset);
    sizes_v[i].push_back(transfer.size);
    status_ptrs[i] = &statuses[i];
    ids[i] = io_engine_->AllocateTransferUniqueId();
  }

  io_engine_->BatchRead(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v,
                        status_ptrs, ids);
  for (size_t i = 0; i < active_transfers.size(); ++i) {
    statuses[i].Wait();
    if (!statuses[i].Succeeded()) {
      entries[active_transfers[i].entry_index].failed = true;
    }
  }

  if (staging_lock.owns_lock()) {
    for (auto& entry : entries) {
      if (entry.failed || !entry.use_staging) continue;
      if (entry.staging_offset + entry.item->size > config_.staging_buffer_size) {
        entry.failed = true;
        continue;
      }
      std::memcpy(entry.item->dst, staging_buffer_.get() + entry.staging_offset, entry.item->size);
    }
    staging_lock.unlock();
  }
}

void PoolClient::FinalizeRemoteGetEntries(std::vector<RemoteGetEntry>& entries,
                                          std::vector<bool>* results) {
  for (auto& entry : entries) {
    if (entry.failed) {
      (*results)[entry.result_index] = false;
      continue;
    }
    master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                               MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP,
                               {{"traffic", "remote"}}, static_cast<double>(entry.item->size));
    (*results)[entry.result_index] = true;
  }
}

bool PoolClient::Exists(const std::string& key) {
  if (!initialized_) return false;
  std::optional<RouteGetResult> route;
  std::unordered_set<std::string> excludes;
  auto status = master_client_->RouteGet(key, excludes, &route);
  return status.ok() && route.has_value();
}

std::vector<bool> PoolClient::BatchExists(const std::vector<std::string>& keys) {
  std::vector<bool> out(keys.size(), false);
  if (!initialized_) return out;
  std::vector<std::optional<RouteGetResult>> routes;
  std::unordered_set<std::string> excludes;
  auto status = master_client_->BatchRouteGet(keys, excludes, &routes);
  if (!status.ok()) return out;
  for (size_t i = 0; i < keys.size() && i < routes.size(); ++i) {
    out[i] = routes[i].has_value();
  }
  return out;
}

// ---------------------------------------------------------------------------
//  External KV (unchanged)
// ---------------------------------------------------------------------------

bool PoolClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  return master_client_->ReportExternalKvBlocks(config_.master_config.node_id, hashes, tier).ok();
}

bool PoolClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  return master_client_->RevokeExternalKvBlocks(config_.master_config.node_id, hashes, tier).ok();
}

bool PoolClient::MatchExternalKv(const std::vector<std::string>& hashes,
                                 std::vector<MasterClient::ExternalKvNodeMatch>* out_matches) {
  if (!initialized_) return false;
  return master_client_->MatchExternalKv(hashes, out_matches).ok();
}

// ---------------------------------------------------------------------------
//  Peer connection cache
// ---------------------------------------------------------------------------

PoolClient::PeerConnection& PoolClient::GetOrConnectPeer(const std::string& node_id,
                                                         const std::string& peer_address) {
  std::lock_guard<std::mutex> lock(peers_mutex_);
  auto it = peers_.find(node_id);
  if (it != peers_.end()) return *it->second;

  auto conn = std::make_unique<PeerConnection>();
  conn->peer_address = peer_address;
  // engine_desc is hydrated lazily in EnsurePeerServiceConnection from
  // the peer's GetPeerInfo response.
  auto& ref = *conn;
  peers_[node_id] = std::move(conn);
  return ref;
}

void PoolClient::EnsureBufferDescsCached(PeerConnection& peer,
                                         const std::vector<BufferMemoryDescBytes>& descs) {
  if (!io_engine_) return;
  std::lock_guard<std::mutex> lock(peers_mutex_);
  for (const auto& d : descs) {
    if (peer.dram_memories.size() <= d.buffer_index) {
      peer.dram_memories.resize(d.buffer_index + 1);
    }
    if (IsValidMemoryDesc(peer.dram_memories[d.buffer_index])) continue;
    if (d.desc_bytes.empty()) continue;
    auto handle =
        msgpack::unpack(reinterpret_cast<const char*>(d.desc_bytes.data()), d.desc_bytes.size());
    peer.dram_memories[d.buffer_index] = handle.get().as<mori::io::MemoryDesc>();
  }
}

// ---------------------------------------------------------------------------
//  RDMA scatter helpers (unchanged from prior impl)
// ---------------------------------------------------------------------------

bool PoolClient::RemoteDramScatterWrite(PeerConnection& peer,
                                        const std::vector<PageLocation>& pages, uint64_t page_size,
                                        const void* src, size_t size, bool zero_copy) {
  if (!io_engine_) return false;
  if (pages.empty() || page_size == 0) return false;
  if (!SizeMatchesAllocation(size, pages.size(), page_size)) return false;

  mori::io::MemoryDesc local_mem;
  size_t local_base_offset = 0;
  bool used_zero_copy = false;
  std::unique_lock<std::mutex> staging_lock(staging_mutex_, std::defer_lock);
  if (zero_copy) {
    auto reg = FindRegisteredMemory(src, size);
    if (reg) {
      local_mem = reg->first;
      local_base_offset = reg->second;
      used_zero_copy = true;
    }
  }
  if (!used_zero_copy) {
    if (size > config_.staging_buffer_size) return false;
    staging_lock.lock();
    std::memcpy(staging_buffer_.get(), src, size);
    local_mem = staging_mem_;
    local_base_offset = 0;
  }

  auto groups = GroupPagesByBuffer(pages);
  const size_t N = groups.size();

  mori::io::MemDescVec remote_descs;
  remote_descs.reserve(N);
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (size_t k = 0; k < N; ++k) {
      const auto& g = groups[k];
      if (g.buffer_index >= peer.dram_memories.size() ||
          !IsValidMemoryDesc(peer.dram_memories[g.buffer_index])) {
        return false;
      }
      remote_descs.push_back(peer.dram_memories[g.buffer_index]);
    }
  }

  mori::io::MemDescVec local_descs(N, local_mem);
  mori::io::BatchSizeVec local_offsets(N), remote_offsets(N), sizes_v(N);
  for (size_t k = 0; k < N; ++k) {
    const auto& g = groups[k];
    local_offsets[k].reserve(g.src_page_indices.size());
    remote_offsets[k].reserve(g.src_page_indices.size());
    sizes_v[k].reserve(g.src_page_indices.size());
    for (size_t spi : g.src_page_indices) {
      local_offsets[k].push_back(local_base_offset + spi * page_size);
      remote_offsets[k].push_back(static_cast<uint64_t>(pages[spi].page_index) * page_size);
      sizes_v[k].push_back(LogicalPageBytes(spi, pages.size(), page_size, size));
    }
  }

  std::vector<mori::io::TransferStatus> statuses(N);
  mori::io::TransferStatusPtrVec status_ptrs(N);
  mori::io::TransferUniqueIdVec ids(N);
  for (size_t k = 0; k < N; ++k) {
    status_ptrs[k] = &statuses[k];
    ids[k] = io_engine_->AllocateTransferUniqueId();
  }
  io_engine_->BatchWrite(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v,
                         status_ptrs, ids);
  bool all_ok = true;
  for (auto& s : statuses) {
    s.Wait();
    if (!s.Succeeded()) all_ok = false;
  }
  return all_ok;
}

bool PoolClient::RemoteDramScatterRead(PeerConnection& peer, const std::vector<PageLocation>& pages,
                                       uint64_t page_size, void* dst, size_t size, bool zero_copy) {
  if (!io_engine_) return false;
  if (pages.empty() || page_size == 0) return false;
  if (!SizeMatchesAllocation(size, pages.size(), page_size)) return false;

  mori::io::MemoryDesc local_mem;
  size_t local_base_offset = 0;
  bool used_zero_copy = false;
  std::unique_lock<std::mutex> staging_lock(staging_mutex_, std::defer_lock);
  if (zero_copy) {
    auto reg = FindRegisteredMemory(dst, size);
    if (reg) {
      local_mem = reg->first;
      local_base_offset = reg->second;
      used_zero_copy = true;
    }
  }
  if (!used_zero_copy) {
    if (size > config_.staging_buffer_size) return false;
    staging_lock.lock();
    local_mem = staging_mem_;
    local_base_offset = 0;
  }

  auto groups = GroupPagesByBuffer(pages);
  const size_t N = groups.size();

  mori::io::MemDescVec remote_descs;
  remote_descs.reserve(N);
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (size_t k = 0; k < N; ++k) {
      const auto& g = groups[k];
      if (g.buffer_index >= peer.dram_memories.size() ||
          !IsValidMemoryDesc(peer.dram_memories[g.buffer_index])) {
        return false;
      }
      remote_descs.push_back(peer.dram_memories[g.buffer_index]);
    }
  }

  mori::io::MemDescVec local_descs(N, local_mem);
  mori::io::BatchSizeVec local_offsets(N), remote_offsets(N), sizes_v(N);
  for (size_t k = 0; k < N; ++k) {
    const auto& g = groups[k];
    local_offsets[k].reserve(g.src_page_indices.size());
    remote_offsets[k].reserve(g.src_page_indices.size());
    sizes_v[k].reserve(g.src_page_indices.size());
    for (size_t spi : g.src_page_indices) {
      local_offsets[k].push_back(local_base_offset + spi * page_size);
      remote_offsets[k].push_back(static_cast<uint64_t>(pages[spi].page_index) * page_size);
      sizes_v[k].push_back(LogicalPageBytes(spi, pages.size(), page_size, size));
    }
  }

  std::vector<mori::io::TransferStatus> statuses(N);
  mori::io::TransferStatusPtrVec status_ptrs(N);
  mori::io::TransferUniqueIdVec ids(N);
  for (size_t k = 0; k < N; ++k) {
    status_ptrs[k] = &statuses[k];
    ids[k] = io_engine_->AllocateTransferUniqueId();
  }
  io_engine_->BatchRead(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v,
                        status_ptrs, ids);
  bool all_ok = true;
  for (auto& s : statuses) {
    s.Wait();
    if (!s.Succeeded()) all_ok = false;
  }
  if (!all_ok) return false;
  if (!used_zero_copy) std::memcpy(dst, staging_buffer_.get(), size);
  return true;
}

// ---------------------------------------------------------------------------
//  SSD path (preserved from prior impl)
// ---------------------------------------------------------------------------

bool PoolClient::EnsurePeerServiceConnection(PeerConnection& peer) {
  std::lock_guard<std::mutex> lock(peer.ssd_op_mutex);
  if (peer.peer_address.empty()) {
    return false;
  }

  auto hydrate_from_peer = [&](::umbp::UMBPPeer::Stub* stub) -> bool {
    ::umbp::GetPeerInfoRequest req;
    ::umbp::GetPeerInfoResponse resp;
    grpc::ClientContext ctx;
    auto status = stub->GetPeerInfo(&ctx, req, &resp);
    if (!status.ok()) {
      MORI_UMBP_ERROR("[PoolClient] GetPeerInfo failed for '{}': {}", peer.peer_address,
                      status.error_message());
      return false;
    }

    if (!resp.engine_desc().empty()) {
      auto handle = msgpack::unpack(resp.engine_desc().data(), resp.engine_desc().size());
      peer.engine_desc = handle.get().as<mori::io::EngineDesc>();
      if (io_engine_) {
        io_engine_->RegisterRemoteEngine(peer.engine_desc);
        peer.engine_registered = true;
      }
    } else if (io_engine_ && !peer.engine_registered) {
      return false;
    }

    if (!resp.ssd_staging_mem_desc().empty()) {
      auto handle =
          msgpack::unpack(resp.ssd_staging_mem_desc().data(), resp.ssd_staging_mem_desc().size());
      peer.ssd_staging_mem = handle.get().as<mori::io::MemoryDesc>();
      peer.ssd_staging_size = resp.ssd_staging_size();
    }

    for (const auto& d : resp.dram_memory_descs()) {
      if (peer.dram_memories.size() <= d.buffer_index()) {
        peer.dram_memories.resize(d.buffer_index() + 1);
      }
      if (IsValidMemoryDesc(peer.dram_memories[d.buffer_index()])) continue;
      if (d.desc().empty()) continue;
      auto h = msgpack::unpack(d.desc().data(), d.desc().size());
      peer.dram_memories[d.buffer_index()] = h.get().as<mori::io::MemoryDesc>();
    }
    return true;
  };

  if (peer.peer_stub) {
    if (!peer.engine_registered && io_engine_) {
      auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());
      if (!hydrate_from_peer(stub)) {
        peer.peer_stub.reset();
        peer.engine_registered = false;
        return false;
      }
    }
    return true;
  }

  auto channel = grpc::CreateChannel(peer.peer_address, grpc::InsecureChannelCredentials());
  auto stub = ::umbp::UMBPPeer::NewStub(channel);
  if (!hydrate_from_peer(stub.get())) {
    return false;
  }

  peer.peer_stub = std::unique_ptr<void, void (*)(void*)>(
      stub.release(), +[](void* p) { delete static_cast<::umbp::UMBPPeer::Stub*>(p); });
  return true;
}

bool PoolClient::RemoteSsdWrite(PeerConnection& peer, const std::string& key, const void* src,
                                size_t size, bool zero_copy, uint32_t store_index) {
  if (!io_engine_) return false;
  if (!EnsurePeerServiceConnection(peer)) return false;
  if (!IsValidMemoryDesc(peer.ssd_staging_mem)) return false;
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  ::umbp::AllocateWriteSlotRequest alloc_req;
  alloc_req.set_size(size);
  ::umbp::AllocateWriteSlotResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  auto alloc_status = stub->AllocateWriteSlot(&alloc_ctx, alloc_req, &alloc_resp);
  if (!alloc_status.ok() || !alloc_resp.success()) return false;
  uint64_t write_offset = alloc_resp.staging_offset();

  bool used_zero_copy = false;
  if (zero_copy) {
    auto reg = FindRegisteredMemory(src, size);
    if (reg) {
      auto uid = io_engine_->AllocateTransferUniqueId();
      mori::io::TransferStatus status;
      io_engine_->Write(reg->first, reg->second, peer.ssd_staging_mem, write_offset, size, &status,
                        uid);
      status.Wait();
      if (!status.Succeeded()) return false;
      used_zero_copy = true;
    }
  }
  if (!used_zero_copy) {
    if (size > config_.staging_buffer_size) return false;
    std::lock_guard<std::mutex> lock(staging_mutex_);
    std::memcpy(staging_buffer_.get(), src, size);
    auto uid = io_engine_->AllocateTransferUniqueId();
    mori::io::TransferStatus status;
    io_engine_->Write(staging_mem_, 0, peer.ssd_staging_mem, write_offset, size, &status, uid);
    status.Wait();
    if (!status.Succeeded()) return false;
  }

  ::umbp::CommitSsdWriteRequest req;
  req.set_key(key);
  req.set_staging_offset(write_offset);
  req.set_size(size);
  req.set_store_index(store_index);
  req.set_lease_id(alloc_resp.lease_id());
  ::umbp::CommitSsdWriteResponse resp;
  grpc::ClientContext ctx;
  auto grpc_status = stub->CommitSsdWrite(&ctx, req, &resp);
  return grpc_status.ok() && resp.success();
}

bool PoolClient::RemoteSsdRead(PeerConnection& peer, const std::string& key,
                               const std::string& location_id, void* dst, size_t size,
                               bool zero_copy) {
  if (!io_engine_) return false;
  if (!EnsurePeerServiceConnection(peer)) return false;
  if (!IsValidMemoryDesc(peer.ssd_staging_mem)) return false;
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  ::umbp::PrepareSsdReadRequest req;
  req.set_key(key);
  req.set_ssd_location_id(location_id);
  req.set_size(size);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;
  auto grpc_status = stub->PrepareSsdRead(&ctx, req, &resp);
  if (!grpc_status.ok() || !resp.success()) return false;

  bool rdma_ok = false;
  if (zero_copy) {
    auto reg = FindRegisteredMemory(dst, size);
    if (reg) {
      auto uid = io_engine_->AllocateTransferUniqueId();
      mori::io::TransferStatus status;
      io_engine_->Read(reg->first, reg->second, peer.ssd_staging_mem, resp.staging_offset(), size,
                       &status, uid);
      status.Wait();
      rdma_ok = status.Succeeded();
    }
  }
  if (!rdma_ok) {
    if (size > config_.staging_buffer_size) return false;
    std::lock_guard<std::mutex> lock(staging_mutex_);
    auto uid = io_engine_->AllocateTransferUniqueId();
    mori::io::TransferStatus status;
    io_engine_->Read(staging_mem_, 0, peer.ssd_staging_mem, resp.staging_offset(), size, &status,
                     uid);
    status.Wait();
    if (status.Succeeded()) {
      std::memcpy(dst, staging_buffer_.get(), size);
      rdma_ok = true;
    }
  }

  if (resp.lease_id() > 0) {
    const uint32_t max_retries = ReleaseLeaseMaxRetries();
    for (uint32_t attempt = 0; attempt < max_retries; ++attempt) {
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

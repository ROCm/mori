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

#include <chrono>
#include <cstring>
#include <filesystem>
#include <msgpack.hpp>
#include <tuple>
#include <unordered_map>

#include "mori/io/backend.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/obs_counters.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {

// Bandwidth histogram bounds: 1 GiB/s steps from 0 to 3 TiB/s (3072 buckets).
const std::vector<double>& BatchBandwidthBoundsGiBps() {
  static const std::vector<double> kBounds = []() {
    std::vector<double> b;
    b.reserve(3 * 1024);
    for (int i = 1; i <= 3 * 1024; ++i) b.push_back(static_cast<double>(i));
    return b;
  }();
  return kBounds;
}

void ObserveBatchBandwidth(MasterClient& mc, const std::string& metric_name,
                           const std::string& metric_help, const std::string& client_id,
                           double bytes_local, double bytes_remote, double elapsed_s) {
  if (elapsed_s <= 0) return;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const auto& bounds = BatchBandwidthBoundsGiBps();
  if (bytes_local > 0)
    mc.Observe(metric_name, metric_help, {{"client", client_id}, {"traffic", "local"}}, bounds,
               bytes_local / kGiB / elapsed_s);
  if (bytes_remote > 0)
    mc.Observe(metric_name, metric_help, {{"client", client_id}, {"traffic", "remote"}}, bounds,
               bytes_remote / kGiB / elapsed_s);
}

std::optional<ParsedLocationId> ParseLocationIdWithLog(const std::string& location_id) {
  auto result = ParseLocationId(location_id);
  if (!result) {
    MORI_UMBP_ERROR("[PoolClient] Failed to parse location_id: {}", location_id);
  }
  return result;
}

std::optional<ParsedDramLocation> ParseDramLocationIdWithLog(const std::string& location_id) {
  auto result = ParseDramLocationId(location_id);
  if (!result) {
    MORI_UMBP_ERROR("[PoolClient] Failed to parse DRAM/HBM location_id: {}", location_id);
  }
  return result;
}

bool IsValidMemoryDesc(const mori::io::MemoryDesc& desc) { return desc.size > 0; }

// Bytes belonging to the i-th logical page in a Put/Get of `total_size`
// bytes spread across `num_pages` pages of `page_size` each.  Master rounds
// the request up: num_pages = ceil(total_size / page_size).  Thus every
// page except possibly the last is full; the last carries the remaining
// (total_size - (num_pages-1)*page_size) bytes, which is in (0, page_size].
//
// The caller MUST have already verified `total_size <= num_pages * page_size`
// and `total_size > (num_pages-1) * page_size` (i.e. master's rounding is
// consistent with what we asked for).  Passing total_size==0 / num_pages==0
// is undefined here; gate at the call site.
inline uint64_t LogicalPageBytes(size_t i, size_t num_pages, uint64_t page_size,
                                 size_t total_size) {
  return (i + 1 == num_pages) ? (total_size - i * page_size) : page_size;
}

// SizeMatchesAllocation is now defined in umbp/distributed/types.h so Master
// and Client share the exact same allocation-window predicate.

// Min interval between two batch-level "src not registered" WARNs from the
// same PoolClient instance.  Default 60s matches typical operator-noise
// tolerance; override via UMBP_BATCH_PUT_WARN_INTERVAL_SEC.
int64_t BatchPutStagingWarnIntervalNs() {
  static const int64_t v =
      GetEnvSeconds("UMBP_BATCH_PUT_WARN_INTERVAL_SEC", std::chrono::seconds(60),
                    /*min_allowed=*/1)
          .count() *
      1000LL * 1000LL * 1000LL;
  return v;
}

// Number of ReleaseSsdLease RPC attempts before giving up.  Default 2.
// Returns uint32_t to avoid a silent sign flip when the env is set to a
// value above INT_MAX; the loop variable below matches.
uint32_t ReleaseLeaseMaxRetries() {
  static const uint32_t v = GetEnvUint32("UMBP_RELEASE_LEASE_MAX_RETRIES", 2, /*min_allowed=*/1);
  return v;
}

}  // namespace

PoolClient::PoolClient(PoolClientConfig config) : config_(std::move(config)) {}

PoolClient::~PoolClient() { Shutdown(); }

bool PoolClient::Init() {
  bool expected = false;
  if (!initialized_.compare_exchange_strong(expected, true)) return true;

  master_client_ = std::make_unique<MasterClient>(config_.master_config);

  // Initialize IO Engine for RDMA data plane.
  // host non-empty signals intent to use RDMA; port=0 means OS-assigned ephemeral port.
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

  // PeerService is started by DistributedClient after PoolClient init. Advertise the
  // configured address here so the Master can route peer SSD traffic.
  std::string peer_address;
  if (config_.peer_service_port > 0 && !config_.ssd_stores.empty()) {
    std::string host = config_.io_engine.host.empty() ? config_.master_config.node_address
                                                      : config_.io_engine.host;
    peer_address = host + ":" + std::to_string(config_.peer_service_port);
  }

  std::vector<uint64_t> ssd_store_capacities;
  for (const auto& store : config_.ssd_stores) {
    ssd_store_capacities.push_back(store.capacity);
  }

  auto status = master_client_->RegisterSelf(
      config_.tier_capacities, peer_address, engine_desc_bytes, dram_memory_desc_bytes_list,
      dram_buffer_sizes, ssd_store_capacities, config_.dram_page_size);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] RegisterSelf failed: {}", status.error_message());
    initialized_ = false;
    return false;
  }

  if (config_.master_config.auto_heartbeat) {
    master_client_->StartHeartbeat();
  }

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
  // Reject null / zero-sized ranges up front.  IOEngine::RegisterMemory
  // does not null-check and downstream RDMA bookkeeping assumes a
  // non-empty region; surface this as a caller error instead of crashing.
  if (ptr == nullptr || size == 0) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory: invalid args ptr={}, size={}", ptr, size);
    return false;
  }
  std::lock_guard<std::mutex> lock(registered_mem_mutex_);
  for (auto& reg : registered_regions_) {
    if (reg.base == ptr) {
      MORI_UMBP_DEBUG("[PoolClient] RegisterMemory: ptr={} already registered, skipping", ptr);
      return true;
    }
  }
  auto mem_desc = io_engine_->RegisterMemory(ptr, size, -1, mori::io::MemoryLocationType::CPU);
  registered_regions_.push_back({ptr, size, mem_desc});
  MORI_UMBP_INFO("[PoolClient] RegisterMemory: ptr={}, size={}", ptr, size);
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
// DRAM-only methods for UMBPClient integration
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

  MORI_UMBP_OBS_INC(abort_allocation_calls_);
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

bool PoolClient::Exists(const std::string& key) {
  if (!initialized_) return false;

  bool found = false;
  auto status = master_client_->Lookup(key, &found);
  if (!status.ok()) return false;
  return found;
}

std::vector<bool> PoolClient::BatchExists(const std::vector<std::string>& keys) {
  std::vector<bool> results(keys.size(), false);
  if (!initialized_ || keys.empty()) return results;

  std::vector<bool> found;
  auto status = master_client_->BatchLookup(keys, &found);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchExists: BatchLookup failed: {}", status.error_message());
    return results;
  }
  if (found.size() != keys.size()) {
    MORI_UMBP_ERROR("[PoolClient] BatchExists: result count mismatch ({} vs {})", found.size(),
                    keys.size());
    return results;
  }
  results = std::move(found);
  return results;
}

bool PoolClient::Get(const std::string& key, void* dst, size_t size) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  std::optional<RouteGetResult> result;
  auto status = master_client_->RouteGet(key, &result);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] Get RouteGet failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) return false;

  const auto& loc = result->location;

  // Caller's `size` must match the byte size that was committed at Put time
  // (Master tracks it in Location.size).  The page-window check further
  // down would otherwise silently accept any `size` in ((N-1)*ps, N*ps],
  // and the partial-last-page logic would copy the wrong number of bytes,
  // either truncating valid data or pulling stale tail bytes into `dst`.
  if (size != loc.size) {
    MORI_UMBP_ERROR(
        "[PoolClient] Get: caller size {} != stored size {} for key='{}' "
        "(caller must pass the same size that was used at Put time)",
        size, loc.size, key);
    return false;
  }

  bool is_local = (loc.node_id == config_.master_config.node_id);
  if (loc.tier == TierType::DRAM || loc.tier == TierType::HBM) {
    auto parsed = ParseDramLocationIdWithLog(loc.location_id);
    if (!parsed) return false;
    if (result->page_size == 0) {
      MORI_UMBP_ERROR("[PoolClient] Get: master returned page_size=0 for DRAM/HBM target");
      return false;
    }
    const uint64_t page_size = result->page_size;
    const size_t num_pages = parsed->pages.size();
    if (num_pages == 0) {
      MORI_UMBP_ERROR("[PoolClient] Get: empty pages list, key='{}'", key);
      return false;
    }
    // Note: no `size in ((N-1)*ps, N*ps]` check here — it is implied by
    // `size == loc.size` (verified above) plus the master-side invariant
    // that Put stored num_pages = ceil(loc.size / page_size).  The
    // page-level OOB checks below catch any remaining inconsistency before
    // it can corrupt memory.
    if (is_local) {
      // Local-node short-circuit: copy each page from this node's own
      // exportable buffers; no RDMA round-trip needed.  The last logical
      // page may be partial — copy only its real bytes.
      char* dst_bytes = static_cast<char*>(dst);
      for (size_t i = 0; i < num_pages; ++i) {
        const auto& p = parsed->pages[i];
        if (p.buffer_index >= config_.dram_buffers.size()) {
          MORI_UMBP_ERROR("[PoolClient] local Get: invalid buffer_index {}", p.buffer_index);
          return false;
        }
        auto& dram = config_.dram_buffers[p.buffer_index];
        const uint64_t off = static_cast<uint64_t>(p.page_index) * page_size;
        if (!dram.buffer || page_size > dram.size || off > dram.size - page_size) {
          MORI_UMBP_ERROR("[PoolClient] local Get: OOB buf={} off={} page_size={} buf_size={}",
                          p.buffer_index, off, page_size, dram.size);
          return false;
        }
        const uint64_t bytes = LogicalPageBytes(i, num_pages, page_size, size);
        std::memcpy(dst_bytes + i * page_size, static_cast<const char*>(dram.buffer) + off, bytes);
      }
      MORI_UMBP_INFO("[PoolClient] Get: recording get_bytes key='{}' size={} traffic=local", key,
                     size);
      master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                                 MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP,
                                 {{"traffic", "local"}}, static_cast<double>(size));
      return true;
    }

    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address, result->engine_desc_bytes);
    EnsureBufferDescsCached(peer, result->dram_memory_descs);
    // zero_copy=true: try registered DRAM region first, fall back to staging
    // (with `staging_buffer_size` cap) only when caller did not pre-register.
    bool ok = RemoteDramScatterRead(peer, parsed->pages, page_size, dst, size, true);
    if (ok) {
      master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                                 MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP,
                                 {{"traffic", "remote"}}, static_cast<double>(size));
    }
    return ok;
  }

  if (loc.tier == TierType::SSD) {
    // SSD path: dram_memory_descs list is empty for SSD tier; only the
    // engine_desc is needed to establish the IOEngine endpoint.
    auto& peer = GetOrConnectPeer(loc.node_id, result->peer_address, result->engine_desc_bytes);
    bool ok = RemoteSsdRead(peer, key, loc.location_id, dst, size, true);
    if (ok) {
      master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                                 MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP,
                                 {{"traffic", "remote"}}, static_cast<double>(size));
    }
    return ok;
  }

  MORI_UMBP_WARN("[PoolClient] Get: key '{}' is on unsupported tier {}", key,
                 TierTypeName(loc.tier));
  return false;
}

bool PoolClient::Put(const std::string& key, const void* src, size_t size) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] Not initialized");
    return false;
  }

  std::optional<RoutePutResult> result;
  auto status = master_client_->RoutePut(key, size, &result);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] Put RoutePut failed: {}", status.error_message());
    return false;
  }
  if (!result.has_value()) {
    MORI_UMBP_ERROR("[PoolClient] Put: no suitable target");
    return false;
  }

  // DRAM-only path: SSD target routing is rejected here (Put-to-SSD goes
  // through a separate code path).
  if (result->tier != TierType::DRAM) {
    MORI_UMBP_WARN("[PoolClient] Put: target tier is {} (DRAM-only supported)",
                   TierTypeName(result->tier));
    return false;
  }

  // Full scatter-gather: single-page is the trivial N=1, K=1 case and goes
  // through the same code path as multi-page.
  // The three checks below are defensive: they assert Master-side allocation
  // invariants (non-zero page_size, non-empty pages list, size within the
  // allocation window).  Master's AllocateForPut enforces these before
  // creating a pending, so correct Master never produces a response that
  // trips them.  We still log+fail on violation to catch future regressions,
  // but do NOT call AbortAllocation: if Master is correct, there is no
  // pending to roll back; if Master is buggy enough to emit a malformed
  // response, the pending (if any) will be collected by the allocation-TTL
  // reaper.  See Plan `abort_allocation_cleanup` for rationale.
  if (result->page_size == 0) {
    MORI_UMBP_ERROR("[PoolClient] Put: master returned page_size=0 for DRAM target");
    return false;
  }
  const uint64_t page_size = result->page_size;
  const size_t num_pages = result->pages.size();
  if (num_pages == 0) {
    MORI_UMBP_ERROR("[PoolClient] Put: master returned empty pages list, key='{}'", key);
    return false;
  }
  if (!SizeMatchesAllocation(size, num_pages, page_size)) {
    MORI_UMBP_ERROR("[PoolClient] Put: size {} not in allocation window ({}..{}] for key='{}'",
                    size, (num_pages - 1) * page_size, num_pages * page_size, key);
    return false;
  }

  bool is_local = (result->node_id == config_.master_config.node_id);
  if (is_local) {
    // Local-node short-circuit: copy each page directly into our own
    // exportable buffers.  The two bounds checks below assert that Master's
    // PageBitmapAllocator stayed within the buffer topology this Client
    // registered with — they are master-invariant in nature (same family as
    // the page_size/pages/SizeMatchesAllocation guards above) and thus no
    // AbortAllocation is sent on violation: a correct Master never produces
    // an out-of-range page; a misconfigured Client (post-register dram_buffers
    // drift) is a local protocol violation that the master-side reaper will
    // collect via the allocation-TTL path.
    const char* src_bytes = static_cast<const char*>(src);
    for (size_t i = 0; i < num_pages; ++i) {
      const auto& p = result->pages[i];
      if (p.buffer_index >= config_.dram_buffers.size()) {
        MORI_UMBP_ERROR("[PoolClient] local Put: invalid buffer_index {}", p.buffer_index);
        return false;
      }
      auto& dram = config_.dram_buffers[p.buffer_index];
      const uint64_t off = static_cast<uint64_t>(p.page_index) * page_size;
      if (!dram.buffer || page_size > dram.size || off > dram.size - page_size) {
        MORI_UMBP_ERROR("[PoolClient] local Put: OOB buf={} off={} page_size={} buf_size={}",
                        p.buffer_index, off, page_size, dram.size);
        return false;
      }
      const uint64_t bytes = LogicalPageBytes(i, num_pages, page_size, size);
      std::memcpy(static_cast<char*>(dram.buffer) + off, src_bytes + i * page_size, bytes);
    }
  } else {
    // Hydrate every buffer's MemoryDesc up front from the response's
    // dram_memory_descs list (deduplicated, ascending), then issue a single
    // scatter-gather BatchWrite covering every page.
    auto& peer = GetOrConnectPeer(result->node_id, result->peer_address, result->engine_desc_bytes);
    EnsureBufferDescsCached(peer, result->dram_memory_descs);
    // zero_copy=true: try registered DRAM region first, fall back to staging
    // (with `staging_buffer_size` cap) only when caller did not pre-register.
    bool ok = RemoteDramScatterWrite(peer, result->pages, page_size, src, size, true);
    if (!ok) {
      MORI_UMBP_OBS_INC(abort_allocation_calls_);
      master_client_->AbortAllocation(result->node_id, result->allocation_id, size);
      return false;
    }
  }

  Location location;
  location.node_id = result->node_id;
  // Use the canonical page-bitmap location_id Master handed back so Finalize
  // matches what's stored in pending_allocations_ exactly.
  location.location_id = result->location_id;
  location.size = size;
  location.tier = result->tier;

  status = master_client_->FinalizeAllocation(key, location, result->allocation_id);
  if (!status.ok()) {
    // Two failure shapes funnel into !status.ok() here:
    //   (a) Master rejected with finalized==false: for tier/size/location_id
    //       mismatch master already auto-rolled back the pending; for the
    //       other rejection paths (pending-not-found, idempotent-mismatch,
    //       node_id mismatch, node not ALIVE) there is no pending owned by
    //       this node to collect.
    //   (b) gRPC transport error: master may or may not have processed the
    //       request; in the worst case a pending is left over and gets
    //       collected by the allocation-TTL reaper.  Accepted as a known
    //       trade-off (see Plan `abort_allocation_cleanup` §"已知行为变化").
    MORI_UMBP_ERROR("[PoolClient] Put FinalizeAllocation failed: {}", status.error_message());
    return false;
  }

  MORI_UMBP_INFO("[PoolClient] Put: recording put_bytes key='{}' size={} traffic={}", key, size,
                 is_local ? "local" : "remote");
  master_client_->AddCounter(
      MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL, MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL_HELP,
      {{"traffic", is_local ? "local" : "remote"}}, static_cast<double>(size));

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cluster_locations_[key] = location;
  }

  return true;
}

bool PoolClient::ShouldEmitBatchPutStagingWarn() {
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  int64_t prev = last_batch_put_staging_warn_ns_.load(std::memory_order_relaxed);
  // First emit unconditionally (prev == 0); afterwards gate on interval.
  while (prev == 0 || now - prev > BatchPutStagingWarnIntervalNs()) {
    if (last_batch_put_staging_warn_ns_.compare_exchange_weak(prev, now,
                                                              std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// BatchPut (3-phase RDMA fusion)
// ---------------------------------------------------------------------------
//
// Phase 0  : BatchRoutePut (one gRPC).
// Phase 1a : Per-item classify into LOCAL / REMOTE_ZC / REMOTE_STG / SKIPPED.
//            FindRegisteredMemory acquired here (registered_mem_mutex_),
//            BEFORE peers_mutex_, to keep a single global lock order.
// Phase 1b : Per-peer plan + snapshot under peers_mutex_, build FusedBucket.
// Phase 1c : Per-bucket SubmitFusedBucket (fire-and-return).
// Phase 2  : LOCAL memcpy + REMOTE_STG legacy serial path - CPU work that
//            overlaps with NIC RDMA in flight from Phase 1c.
// Phase 3  : Wait every submitted pair, map failures, BatchFinalize +
//            end-of-batch BatchAbortAllocation flush.
//
// Failure handling: SKIPPED skips Abort (master-invariant violation,
// TTL reaper covers stale pendings); REMOTE_ZC_PREP_FAILED / pair Wait
// failures / REMOTE_STG failures all funnel into pending_aborts for a
// single end-of-batch flush.

// Spin Wait on every submitted pair in a bucket.  TransferStatus::Wait
// only loops while IN_PROGRESS; INIT/SUCCESS/Failed return immediately,
// so MapBucketFailures must check Succeeded() to disambiguate.
void PoolClient::WaitFusedBucket(FusedBucket& bucket) {
  for (size_t k = 0; k < bucket.pairs.size(); ++k) {
    if (!bucket.submitted[k]) continue;
    bucket.statuses[k].Wait();
  }
}

// Classify each pair (success/failure) and route contributing items into
// results[] (true on success) or pending_aborts (on failure).  An item
// may contribute to MULTIPLE pairs (its pages span buffer_indices); it
// counts as a success only if EVERY one of its contributing pairs
// succeeds.  Two-pass: first compute per-item AND across pairs, then
// write results/pending_aborts once.  A pair that was never submitted
// (submit-time exception) also fails its items; master has reserved
// capacity, we owe rollback.
void PoolClient::MapBucketFailures(FusedBucket& bucket, const std::vector<BatchPutItemPlan>& plans,
                                   std::vector<bool>& results,
                                   std::vector<MasterClient::BatchAbortEntry>& pending_aborts) {
  // First pass: per-item AND across all pairs that touched it.  Iteration
  // order over an unordered_map is unspecified, but each item's final
  // value depends only on the AND, not on the visit sequence.
  std::unordered_map<size_t, bool> item_ok;
  for (size_t k = 0; k < bucket.pairs.size(); ++k) {
    const bool pair_ok = bucket.submitted[k] && bucket.statuses[k].Succeeded();
    if (!pair_ok && bucket.submitted[k]) {
      MORI_UMBP_ERROR("[PoolClient] BatchPut: fused pair {} (peer={}, buf={}) failed: uid={}, {}",
                      k, bucket.peer_node_id, bucket.pairs[k].remote_buf_idx, bucket.ids[k],
                      bucket.statuses[k].Message());
    }
    for (size_t orig_idx : bucket.pairs[k].contributing_items) {
      auto [it, inserted] = item_ok.try_emplace(orig_idx, pair_ok);
      if (!inserted) it->second = it->second && pair_ok;
    }
  }
  for (auto& [orig_idx, ok] : item_ok) {
    if (ok) {
      results[orig_idx] = true;
    } else {
      results[orig_idx] = false;
      pending_aborts.push_back({plans[orig_idx].peer_node_id, plans[orig_idx].allocation_id,
                                plans[orig_idx].bytes_total});
    }
  }
}

void PoolClient::PlanPeerForBatchPut(PeerConnection& peer,
                                     const std::vector<std::optional<RoutePutResult>>& routes,
                                     const std::vector<size_t>& peer_item_indices,
                                     const std::vector<std::string>& keys,
                                     const std::vector<size_t>& sizes,
                                     std::vector<BatchPutItemPlan>& plans,
                                     FusedBucket& out_bucket) {
  // Hydrate + snapshot under a single peers_mutex_ window.  Every read
  // of peer.dram_memories[] is contained in this scope; downstream code
  // sees only FusedPair.remote_mem (snapshotted by value).
  struct PerItemSnapshot {
    size_t orig_idx;
    std::vector<mori::io::MemoryDesc> remote_mems;  // one per page, indexed parallel to r.pages
  };
  std::vector<PerItemSnapshot> snaps;
  snaps.reserve(peer_item_indices.size());

  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (size_t orig_idx : peer_item_indices) {
      const auto& r = *routes[orig_idx];
      EnsureBufferDescsCachedLocked(peer, r.dram_memory_descs);
      PerItemSnapshot snap;
      snap.orig_idx = orig_idx;
      snap.remote_mems.reserve(r.pages.size());
      bool ok = true;
      for (const auto& p : r.pages) {
        if (p.buffer_index >= peer.dram_memories.size() ||
            !IsValidMemoryDesc(peer.dram_memories[p.buffer_index])) {
          MORI_UMBP_ERROR(
              "[PoolClient] BatchPut: buffer_index {} not hydrated on peer '{}' for key='{}'",
              p.buffer_index, peer.peer_address, keys[orig_idx]);
          ok = false;
          break;
        }
        snap.remote_mems.push_back(peer.dram_memories[p.buffer_index]);
      }
      if (!ok) {
        plans[orig_idx].kind = BatchPutItemPlan::Kind::REMOTE_ZC_PREP_FAILED;
        continue;
      }
      snaps.push_back(std::move(snap));
    }
  }

  // Build per-pair scatter-gather lists from snapshotted descs.  Pair key
  // is (local_mem.id, remote_buf_idx, page_size).  Item pages are
  // appended to their pair's SG vectors (real WR-level batching at the
  // backend layer).
  using PairKey = std::tuple<mori::io::MemoryUniqueId, uint32_t, uint64_t>;
  struct PairKeyHash {
    size_t operator()(const PairKey& k) const noexcept {
      return std::hash<uint64_t>{}(static_cast<uint64_t>(std::get<0>(k))) ^
             (std::hash<uint32_t>{}(std::get<1>(k)) << 1) ^
             (std::hash<uint64_t>{}(std::get<2>(k)) << 2);
    }
  };
  std::unordered_map<PairKey, size_t, PairKeyHash> key_to_pair;

  for (const auto& snap : snaps) {
    const auto& r = *routes[snap.orig_idx];
    const auto& plan = plans[snap.orig_idx];
    for (size_t pi = 0; pi < r.pages.size(); ++pi) {
      const auto& p = r.pages[pi];
      PairKey key{plan.local_mem.id, p.buffer_index, plan.page_size};
      auto it = key_to_pair.find(key);
      if (it == key_to_pair.end()) {
        FusedPair pair;
        pair.remote_buf_idx = p.buffer_index;
        pair.remote_mem = snap.remote_mems[pi];
        pair.local_mem = plan.local_mem;
        out_bucket.pairs.push_back(std::move(pair));
        it = key_to_pair.emplace(key, out_bucket.pairs.size() - 1).first;
      }
      auto& pair = out_bucket.pairs[it->second];
      const uint64_t local_off = plan.local_base_off + pi * plan.page_size;
      const uint64_t remote_off = static_cast<uint64_t>(p.page_index) * plan.page_size;
      const uint64_t bytes = LogicalPageBytes(pi, r.pages.size(), plan.page_size, plan.bytes_total);
      pair.local_offsets.push_back(local_off);
      pair.remote_offsets.push_back(remote_off);
      pair.sizes.push_back(bytes);
      // De-dup contributing_items: only push when this is the first page
      // of `orig_idx` we've routed to this pair.
      if (pair.contributing_items.empty() || pair.contributing_items.back() != snap.orig_idx) {
        pair.contributing_items.push_back(snap.orig_idx);
      }
    }
  }
}

void PoolClient::SubmitFusedBucket(FusedBucket& bucket) {
  const size_t N = bucket.pairs.size();
  bucket.submitted.assign(N, false);
  if (N == 0) return;  // engine.cpp asserts on batchSize==0

  // Statuses live in the bucket - their addresses must stay stable for
  // the whole BatchWrite + Wait window.  Default-fill in place.
  bucket.statuses = std::vector<mori::io::TransferStatus>(N);
  bucket.ids.assign(N, 0);

  // Per-pair prep with individual try-catch, so a failure on pair K
  // does NOT poison pairs 0..K-1 or K+1..N-1 (Phase 1 partial-submit
  // safety).  Successful preps are collected into ok_pairs and
  // submitted as a compacted sub-batch.
  std::vector<size_t> ok_pairs;
  ok_pairs.reserve(N);
  for (size_t k = 0; k < N; ++k) {
    try {
      bucket.ids[k] = io_engine_->AllocateTransferUniqueId();
      ok_pairs.push_back(k);
    } catch (const std::exception& e) {
      MORI_UMBP_ERROR("[PoolClient] BatchPut: submit prep failed for peer '{}' pair {}: {}",
                      bucket.peer_node_id, k, e.what());
      // bucket.submitted[k] stays false; Phase 3 routes its items to
      // pending_aborts.
    }
  }
  if (ok_pairs.empty()) return;

  // Build IOEngine-shape parameter vectors from ok_pairs only.  Status
  // pointers reference bucket.statuses[k] (still indexed by the
  // original k, so WaitFusedBucket / MapBucketFailures iterate by k
  // unchanged).
  const size_t M = ok_pairs.size();
  mori::io::MemDescVec local_descs(M);
  mori::io::MemDescVec remote_descs(M);
  mori::io::BatchSizeVec local_offsets(M);
  mori::io::BatchSizeVec remote_offsets(M);
  mori::io::BatchSizeVec sizes_v(M);
  mori::io::TransferStatusPtrVec status_ptrs(M);
  mori::io::TransferUniqueIdVec sub_ids(M);
  for (size_t i = 0; i < M; ++i) {
    const size_t k = ok_pairs[i];
    auto& pair = bucket.pairs[k];
    local_descs[i] = pair.local_mem;
    remote_descs[i] = pair.remote_mem;
    local_offsets[i] = pair.local_offsets;
    remote_offsets[i] = pair.remote_offsets;
    sizes_v[i] = pair.sizes;
    status_ptrs[i] = &bucket.statuses[k];
    sub_ids[i] = bucket.ids[k];
  }

  MORI_UMBP_OBS_INC(batch_put_io_engine_calls_);
  MORI_UMBP_OBS_ADD(batch_put_io_engine_pairs_, M);
  MORI_UMBP_DEBUG("[PoolClient] BatchPut: IssueBatchWrite peer='{}' pairs={}/{}",
                  bucket.peer_node_id, M, N);
  try {
    IssueBatchWrite(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v, status_ptrs,
                    sub_ids);
  } catch (...) {
    // IssueBatchWrite is foreach-per-pair under the hood; if it throws on
    // pair K, pairs 0..K-1 may have already posted to the NIC and now
    // reference bucket.statuses elements.  Drain everything before
    // unwind frees that storage (Wait is a no-op for INIT statuses, so
    // calling on every ok pair is safe regardless of post state).
    for (size_t k : ok_pairs) bucket.statuses[k].Wait();
    throw;
  }
  for (size_t k : ok_pairs) bucket.submitted[k] = true;
}

// ---------------------------------------------------------------------------
// BatchGet (3-phase RDMA fusion) helpers
// ---------------------------------------------------------------------------
//
// Mirror of BatchPut helpers above with three structural simplifications:
//   1. No allocation_id / pending_aborts: Get has no master reservation.
//   2. Kind set is one fewer (no REMOTE_ZC_PREP_FAILED) — snapshot miss
//      degrades to SKIPPED (no Abort owed).
//   3. MapBucketFailuresRead emits per-item GET_BYTES on Wait success
//      and accumulates bw_bytes_remote for the single end-of-batch
//      ObserveBatchBandwidth call.  Metrics fire ONLY when every
//      contributing pair for that item succeeded.
//
// FusedPair / FusedBucket / WaitFusedBucket are reused from BatchPut;
// in the Read direction, FusedPair.local_mem is the NIC *write*
// destination (caller dst), FusedPair.remote_mem is the NIC *read*
// source.

void PoolClient::PlanPeerForBatchGet(PeerConnection& peer,
                                     const std::vector<std::optional<RouteGetResult>>& routes,
                                     const std::vector<size_t>& peer_item_indices,
                                     const std::vector<std::string>& keys,
                                     std::vector<BatchGetItemPlan>& plans,
                                     FusedBucket& out_bucket) {
  // Hydrate + snapshot under a single peers_mutex_ window (mirror of
  // PlanPeerForBatchPut).  Pages were pre-parsed in Phase 1a into
  // plans[orig_idx].parsed_pages, avoiding a second ParseDramLocationId
  // here.
  struct PerItemSnapshot {
    size_t orig_idx;
    std::vector<mori::io::MemoryDesc> remote_mems;  // one per parsed_page
  };
  std::vector<PerItemSnapshot> snaps;
  snaps.reserve(peer_item_indices.size());

  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (size_t orig_idx : peer_item_indices) {
      const auto& r = *routes[orig_idx];
      EnsureBufferDescsCachedLocked(peer, r.dram_memory_descs);
      const auto& parsed_pages = plans[orig_idx].parsed_pages;
      PerItemSnapshot snap;
      snap.orig_idx = orig_idx;
      snap.remote_mems.reserve(parsed_pages.size());
      bool ok = true;
      for (const auto& p : parsed_pages) {
        if (p.buffer_index >= peer.dram_memories.size() ||
            !IsValidMemoryDesc(peer.dram_memories[p.buffer_index])) {
          MORI_UMBP_ERROR(
              "[PoolClient] BatchGet: buffer_index {} not hydrated on peer '{}' for key='{}'",
              p.buffer_index, peer.peer_address, keys[orig_idx]);
          ok = false;
          break;
        }
        snap.remote_mems.push_back(peer.dram_memories[p.buffer_index]);
      }
      if (!ok) {
        // Get has no Abort debt — degrade to SKIPPED so item is omitted
        // from any FusedPair and Phase 3 leaves results[orig_idx]=false.
        plans[orig_idx].kind = BatchGetItemPlan::Kind::SKIPPED;
        continue;
      }
      snaps.push_back(std::move(snap));
    }
  }

  using PairKey = std::tuple<mori::io::MemoryUniqueId, uint32_t, uint64_t>;
  struct PairKeyHash {
    size_t operator()(const PairKey& k) const noexcept {
      return std::hash<uint64_t>{}(static_cast<uint64_t>(std::get<0>(k))) ^
             (std::hash<uint32_t>{}(std::get<1>(k)) << 1) ^
             (std::hash<uint64_t>{}(std::get<2>(k)) << 2);
    }
  };
  std::unordered_map<PairKey, size_t, PairKeyHash> key_to_pair;

  for (const auto& snap : snaps) {
    const auto& plan = plans[snap.orig_idx];
    const auto& parsed_pages = plan.parsed_pages;
    for (size_t pi = 0; pi < parsed_pages.size(); ++pi) {
      const auto& p = parsed_pages[pi];
      PairKey key{plan.local_mem.id, p.buffer_index, plan.page_size};
      auto it = key_to_pair.find(key);
      if (it == key_to_pair.end()) {
        FusedPair pair;
        pair.remote_buf_idx = p.buffer_index;
        pair.remote_mem = snap.remote_mems[pi];
        pair.local_mem = plan.local_mem;
        out_bucket.pairs.push_back(std::move(pair));
        it = key_to_pair.emplace(key, out_bucket.pairs.size() - 1).first;
      }
      auto& pair = out_bucket.pairs[it->second];
      const uint64_t local_off = plan.local_base_off + pi * plan.page_size;
      const uint64_t remote_off = static_cast<uint64_t>(p.page_index) * plan.page_size;
      const uint64_t bytes =
          LogicalPageBytes(pi, parsed_pages.size(), plan.page_size, plan.bytes_total);
      pair.local_offsets.push_back(local_off);
      pair.remote_offsets.push_back(remote_off);
      pair.sizes.push_back(bytes);
      if (pair.contributing_items.empty() || pair.contributing_items.back() != snap.orig_idx) {
        pair.contributing_items.push_back(snap.orig_idx);
      }
    }
  }
}

void PoolClient::SubmitFusedBucketRead(FusedBucket& bucket) {
  const size_t N = bucket.pairs.size();
  bucket.submitted.assign(N, false);
  if (N == 0) return;

  bucket.statuses = std::vector<mori::io::TransferStatus>(N);
  bucket.ids.assign(N, 0);

  std::vector<size_t> ok_pairs;
  ok_pairs.reserve(N);
  for (size_t k = 0; k < N; ++k) {
    try {
      bucket.ids[k] = io_engine_->AllocateTransferUniqueId();
      ok_pairs.push_back(k);
    } catch (const std::exception& e) {
      MORI_UMBP_ERROR("[PoolClient] BatchGet: submit prep failed for peer '{}' pair {}: {}",
                      bucket.peer_node_id, k, e.what());
    }
  }
  if (ok_pairs.empty()) return;

  const size_t M = ok_pairs.size();
  mori::io::MemDescVec local_descs(M);
  mori::io::MemDescVec remote_descs(M);
  mori::io::BatchSizeVec local_offsets(M);
  mori::io::BatchSizeVec remote_offsets(M);
  mori::io::BatchSizeVec sizes_v(M);
  mori::io::TransferStatusPtrVec status_ptrs(M);
  mori::io::TransferUniqueIdVec sub_ids(M);
  for (size_t i = 0; i < M; ++i) {
    const size_t k = ok_pairs[i];
    auto& pair = bucket.pairs[k];
    local_descs[i] = pair.local_mem;
    remote_descs[i] = pair.remote_mem;
    local_offsets[i] = pair.local_offsets;
    remote_offsets[i] = pair.remote_offsets;
    sizes_v[i] = pair.sizes;
    status_ptrs[i] = &bucket.statuses[k];
    sub_ids[i] = bucket.ids[k];
  }

  MORI_UMBP_OBS_INC(batch_get_io_engine_calls_);
  MORI_UMBP_OBS_ADD(batch_get_io_engine_pairs_, M);
  MORI_UMBP_DEBUG("[PoolClient] BatchGet: IssueBatchRead peer='{}' pairs={}/{}",
                  bucket.peer_node_id, M, N);
  try {
    IssueBatchRead(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v, status_ptrs,
                   sub_ids);
  } catch (...) {
    for (size_t k : ok_pairs) bucket.statuses[k].Wait();
    throw;
  }
  for (size_t k : ok_pairs) bucket.submitted[k] = true;
}

void PoolClient::MapBucketFailuresRead(FusedBucket& bucket,
                                       const std::vector<BatchGetItemPlan>& plans,
                                       const std::vector<size_t>& sizes, std::vector<bool>& results,
                                       double* bw_bytes_remote_accum, MasterClient* master_client) {
  // Per-item AND across all pairs that touched it (mirror of
  // MapBucketFailures).  No pending_aborts collection — Get has no
  // reservation.  Per-item GET_BYTES counter + bw_bytes_remote
  // accumulation happen here only for items that succeeded across ALL
  // their contributing pairs (no metric for items whose any pair Failed,
  // matching the legacy single Get behavior).
  std::unordered_map<size_t, bool> item_ok;
  for (size_t k = 0; k < bucket.pairs.size(); ++k) {
    const bool pair_ok = bucket.submitted[k] && bucket.statuses[k].Succeeded();
    if (!pair_ok && bucket.submitted[k]) {
      MORI_UMBP_ERROR("[PoolClient] BatchGet: fused pair {} (peer={}, buf={}) failed: uid={}, {}",
                      k, bucket.peer_node_id, bucket.pairs[k].remote_buf_idx, bucket.ids[k],
                      bucket.statuses[k].Message());
    }
    for (size_t orig_idx : bucket.pairs[k].contributing_items) {
      auto [it, inserted] = item_ok.try_emplace(orig_idx, pair_ok);
      if (!inserted) it->second = it->second && pair_ok;
    }
  }
  for (auto& [orig_idx, ok] : item_ok) {
    if (ok) {
      results[orig_idx] = true;
      const double bytes = static_cast<double>(sizes[orig_idx]);
      *bw_bytes_remote_accum += bytes;
      if (master_client != nullptr) {
        master_client->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                                  MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP,
                                  {{"traffic", "remote"}}, bytes);
      }
    } else {
      results[orig_idx] = false;
      MORI_UMBP_WARN(
          "[PoolClient] BatchGet: fused ZC item idx={} on peer='{}' failed (caller dst content "
          "undefined per legacy contract)",
          orig_idx, plans[orig_idx].peer_node_id);
    }
  }
}

std::vector<bool> PoolClient::BatchPut(const std::vector<std::string>& keys,
                                       const std::vector<const void*>& srcs,
                                       const std::vector<size_t>& sizes,
                                       const std::vector<int>& depths) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (!initialized_ || n == 0) return results;
  if (srcs.size() != n || sizes.size() != n) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: mismatched vector sizes");
    return results;
  }

  ScopedTimer timer("BatchPut", mori::modules::UMBP);
  MORI_UMBP_OBS_INC(batch_put_calls_);

  // Phase 0: route.
  std::vector<uint64_t> block_sizes(n);
  for (size_t i = 0; i < n; ++i) block_sizes[i] = sizes[i];
  std::vector<std::optional<RoutePutResult>> routes;
  auto status = master_client_->BatchRoutePut(keys, block_sizes, &routes);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: BatchRoutePut failed: {}", status.error_message());
    return results;
  }
  if (routes.size() != n) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: route count mismatch ({} vs {})", routes.size(), n);
    return results;
  }

  std::vector<BatchPutItemPlan> plans(n);
  std::unordered_map<std::string, std::vector<size_t>> zc_by_peer;
  std::vector<size_t> stg_indices;
  std::vector<size_t> local_indices;
  std::vector<MasterClient::BatchAbortEntry> pending_aborts;
  pending_aborts.reserve(n);

  // Phase 1a: classify.  registered_mem_mutex_ acquired here (in
  // FindRegisteredMemory) BEFORE peers_mutex_ is taken in Phase 1b -
  // single global lock order.  Master-invariant violations stay SKIPPED.
  for (size_t i = 0; i < n; ++i) {
    plans[i].idx = i;
    if (!routes[i].has_value()) continue;
    const auto& r = *routes[i];
    if (r.tier != TierType::DRAM || r.pages.empty() || r.page_size == 0) {
      MORI_UMBP_ERROR(
          "[PoolClient] BatchPut: unexpected route for key='{}': tier={}, pages={}, page_size={}",
          keys[i], TierTypeName(r.tier), r.pages.size(), r.page_size);
      continue;
    }
    if (!SizeMatchesAllocation(sizes[i], r.pages.size(), r.page_size)) {
      MORI_UMBP_ERROR(
          "[PoolClient] BatchPut: size {} not in allocation window ({}..{}] for key='{}'", sizes[i],
          (r.pages.size() - 1) * r.page_size, r.pages.size() * r.page_size, keys[i]);
      continue;
    }
    plans[i].peer_node_id = r.node_id;
    plans[i].allocation_id = r.allocation_id;
    plans[i].page_size = r.page_size;
    plans[i].bytes_total = sizes[i];

    if (r.node_id == config_.master_config.node_id) {
      plans[i].kind = BatchPutItemPlan::Kind::LOCAL;
      local_indices.push_back(i);
      continue;
    }
    auto reg = FindRegisteredMemory(srcs[i], sizes[i]);
    if (reg.has_value()) {
      plans[i].kind = BatchPutItemPlan::Kind::REMOTE_ZC;
      plans[i].local_mem = reg->first;
      plans[i].local_base_off = reg->second;
      zc_by_peer[r.node_id].push_back(i);
      MORI_UMBP_OBS_INC(batch_put_items_);
    } else {
      plans[i].kind = BatchPutItemPlan::Kind::REMOTE_STG;
      stg_indices.push_back(i);
    }
  }

  // Throttled batch-level WARN if any remote item fell back to staging.
  // Substring "BatchPut: src not registered for key=" is matched by
  // existing tests; keep it stable.  Per-call WARNs inside
  // RemoteDramScatterWrite still fire from Phase 2 below.
  if (!stg_indices.empty() && ShouldEmitBatchPutStagingWarn()) {
    MORI_UMBP_WARN(
        "[PoolClient] BatchPut: src not registered for key='{}' (and possibly others), falling "
        "back to staging path (serial). Subsequent occurrences in the next 60s are suppressed. "
        "Call RegisterMemory() on the host buffer to enable zero-copy batch.",
        keys[stg_indices.front()]);
  }

  // Phase 1b/c: per-peer plan + submit.  buckets vector is reserved so
  // pushes don't relocate while statuses/ids inside are in flight (each
  // FusedBucket is moved-in once; statuses live as members and are
  // pointed to by status_ptrs created inside SubmitFusedBucket).
  //
  // The for-loop is wrapped in try-catch to defend against rare
  // std::bad_alloc / RPC-allocator throws from GetOrConnectPeer /
  // PlanPeerForBatchPut / SubmitFusedBucket.  Without the guard, stack
  // unwind would destroy already-submitted FusedBuckets (whose
  // statuses are pointed to by in-flight RDMA completion callbacks) ->
  // use-after-free.  On catch we drain in-flight RDMA on every bucket
  // already pushed to `buckets`, then mark all REMOTE_ZC items as
  // failed below in Phase 3.
  std::vector<FusedBucket> buckets;
  buckets.reserve(zc_by_peer.size());
  bool phase1_threw = false;
  try {
    for (auto& [peer_id, indices] : zc_by_peer) {
      if (indices.empty()) continue;
      const auto& r0 = *routes[indices[0]];
      auto& peer = GetOrConnectPeer(peer_id, r0.peer_address, r0.engine_desc_bytes);
      FusedBucket bucket;
      bucket.peer_node_id = peer_id;
      PlanPeerForBatchPut(peer, routes, indices, keys, sizes, plans, bucket);
      if (bucket.pairs.empty()) continue;
      SubmitFusedBucket(bucket);
      buckets.push_back(std::move(bucket));
    }
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: Phase 1 threw: {} ({} bucket(s) drained)", e.what(),
                    buckets.size());
    for (auto& bucket : buckets) WaitFusedBucket(bucket);
    phase1_threw = true;
  }

  // Items marked REMOTE_ZC_PREP_FAILED during PlanPeerForBatchPut owe an
  // Abort - master reserved their pages but they never reached a fused
  // pair.  Walk the plan table once to collect them.
  for (const auto& plan : plans) {
    if (plan.kind == BatchPutItemPlan::Kind::REMOTE_ZC_PREP_FAILED) {
      pending_aborts.push_back({plan.peer_node_id, plan.allocation_id, plan.bytes_total});
    }
  }

  // Phase 2a: LOCAL memcpy.  Runs while ZC RDMA is in flight.  OOB on
  // the local path is a master-invariant violation (Master's
  // PageBitmapAllocator stayed within registered buffer topology) - we
  // log and skip without sending Abort, matching legacy semantics.
  for (size_t orig_idx : local_indices) {
    const auto& r = *routes[orig_idx];
    const char* src_bytes = static_cast<const char*>(srcs[orig_idx]);
    bool oob = false;
    for (size_t k = 0; k < r.pages.size(); ++k) {
      const auto& p = r.pages[k];
      if (p.buffer_index >= config_.dram_buffers.size()) {
        MORI_UMBP_ERROR("[PoolClient] BatchPut local: invalid buffer_index {}", p.buffer_index);
        oob = true;
        break;
      }
      auto& dram = config_.dram_buffers[p.buffer_index];
      const uint64_t off = static_cast<uint64_t>(p.page_index) * r.page_size;
      if (!dram.buffer || r.page_size > dram.size || off > dram.size - r.page_size) {
        MORI_UMBP_ERROR("[PoolClient] BatchPut local: OOB buf={} off={} page_size={} buf_size={}",
                        p.buffer_index, off, r.page_size, dram.size);
        oob = true;
        break;
      }
      const uint64_t bytes = LogicalPageBytes(k, r.pages.size(), r.page_size, sizes[orig_idx]);
      std::memcpy(static_cast<char*>(dram.buffer) + off, src_bytes + k * r.page_size, bytes);
    }
    if (!oob) results[orig_idx] = true;
  }

  // Phase 2b: REMOTE_STG via legacy per-item scatter-write (zero_copy=false).
  for (size_t orig_idx : stg_indices) {
    const auto& r = *routes[orig_idx];
    auto& peer = GetOrConnectPeer(r.node_id, r.peer_address, r.engine_desc_bytes);
    EnsureBufferDescsCached(peer, r.dram_memory_descs);
    bool ok = RemoteDramScatterWrite(peer, r.pages, r.page_size, srcs[orig_idx], sizes[orig_idx],
                                     /*zero_copy=*/false);
    if (ok) {
      results[orig_idx] = true;
    } else {
      pending_aborts.push_back({r.node_id, r.allocation_id, sizes[orig_idx]});
    }
  }

  // Phase 3: Wait + map ZC failures.  When phase1_threw, the buckets
  // were already drained in the catch above and all REMOTE_ZC items are
  // considered failed (independent of which bucket they reached, if
  // any) - route them straight to pending_aborts.  REMOTE_ZC_PREP_FAILED
  // items have a different kind so the conditional below does not
  // double-add them.
  if (!phase1_threw) {
    for (auto& bucket : buckets) {
      WaitFusedBucket(bucket);
      MapBucketFailures(bucket, plans, results, pending_aborts);
    }
  } else {
    for (const auto& plan : plans) {
      if (plan.kind == BatchPutItemPlan::Kind::REMOTE_ZC) {
        pending_aborts.push_back({plan.peer_node_id, plan.allocation_id, plan.bytes_total});
      }
    }
  }

  // Build BatchFinalize input from items where results[i] is currently
  // true.  Order of plans (by orig_idx) determines fin_*[] alignment.
  std::vector<size_t> finalize_idx;
  finalize_idx.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (results[i]) finalize_idx.push_back(i);
  }

  double bw_bytes_local = 0, bw_bytes_remote = 0;

  if (!finalize_idx.empty()) {
    std::vector<std::string> fin_keys;
    std::vector<Location> fin_locs;
    std::vector<std::string> fin_aids;
    fin_keys.reserve(finalize_idx.size());
    fin_locs.reserve(finalize_idx.size());
    fin_aids.reserve(finalize_idx.size());
    for (size_t orig_idx : finalize_idx) {
      const auto& r = *routes[orig_idx];
      Location loc;
      loc.node_id = r.node_id;
      loc.location_id = r.location_id;
      loc.size = sizes[orig_idx];
      loc.tier = r.tier;
      fin_keys.push_back(keys[orig_idx]);
      fin_locs.push_back(loc);
      fin_aids.push_back(r.allocation_id);
    }

    std::vector<int32_t> fin_depths;
    if (!depths.empty()) {
      fin_depths.reserve(finalize_idx.size());
      for (size_t orig_idx : finalize_idx) {
        fin_depths.push_back(orig_idx < depths.size() ? static_cast<int32_t>(depths[orig_idx])
                                                      : -1);
      }
    }

    std::vector<bool> fin_results;
    MORI_UMBP_INFO("[PoolClient] BatchPut: calling BatchFinalizeAllocation for {} items",
                   finalize_idx.size());
    auto fin_status = master_client_->BatchFinalizeAllocation(fin_keys, fin_locs, fin_aids,
                                                              &fin_results, fin_depths);
    if (!fin_status.ok()) {
      MORI_UMBP_ERROR("[PoolClient] BatchPut: BatchFinalizeAllocation failed: {}",
                      fin_status.error_message());
    }
    size_t put_ok = 0, put_fail = 0;
    for (size_t i = 0; i < finalize_idx.size(); ++i) {
      const size_t orig_idx = finalize_idx[i];
      bool ok = (i < fin_results.size()) && fin_results[i];
      if (ok) {
        const bool is_local = fin_locs[i].node_id == config_.master_config.node_id;
        put_ok++;
        (is_local ? bw_bytes_local : bw_bytes_remote) += static_cast<double>(fin_locs[i].size);
        master_client_->AddCounter(
            MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL, MORI_UMBP_METRIC_CLIENT_PUT_BYTES_TOTAL_HELP,
            {{"traffic", is_local ? "local" : "remote"}}, static_cast<double>(fin_locs[i].size));
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cluster_locations_[fin_keys[i]] = fin_locs[i];
      } else {
        // Master auto-rolls back on field mismatch; other false paths
        // (pending-not-found / idempotent-mismatch / node_id mismatch /
        // non-ALIVE node) have nothing for us to do.  No Abort RPC.
        results[orig_idx] = false;
        put_fail++;
      }
    }
    MORI_UMBP_INFO(
        "[PoolClient] BatchPut: finalize done ok={} fail={} "
        "put_bytes_local={:.0f} put_bytes_remote={:.0f}",
        put_ok, put_fail, bw_bytes_local, bw_bytes_remote);
  }

  // End-of-batch abort flush (single gRPC across all target nodes).
  if (!pending_aborts.empty()) {
    MORI_UMBP_OBS_INC(batch_abort_allocation_calls_);
    MORI_UMBP_OBS_ADD(batch_abort_allocation_entries_, pending_aborts.size());
    std::vector<bool> abort_results;
    auto abort_status = master_client_->BatchAbortAllocation(pending_aborts, &abort_results);
    if (!abort_status.ok()) {
      MORI_UMBP_ERROR(
          "[PoolClient] BatchPut: BatchAbortAllocation failed: {} ({} entries -> TTL reaper)",
          abort_status.error_message(), pending_aborts.size());
    }
  }

  ObserveBatchBandwidth(*master_client_, MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH,
                        MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH_HELP, NodeId(), bw_bytes_local,
                        bw_bytes_remote, timer.ElapsedSeconds());

  return results;
}

// ---------------------------------------------------------------------------
// BatchGet (3-phase RDMA fusion; mirror of BatchPut)
// ---------------------------------------------------------------------------
//
// Phase 0  : BatchRouteGet (one gRPC).
// Phase 1a : Per-item classify into LOCAL / REMOTE_ZC / REMOTE_STG / SKIPPED;
//            pre-parse loc.location_id into plans[i].parsed_pages so the
//            per-peer loop in Phase 1b/c does not re-parse.  Validates
//            master partial-fill cases (peer_address / engine_desc_bytes /
//            dram_memory_descs / page_size) BEFORE tagging REMOTE_*.
//            FindRegisteredMemory acquired here BEFORE peers_mutex_ to
//            keep the same global lock order as BatchPut.
// Phase 1b : Per-peer plan + snapshot under peers_mutex_, build FusedBucket.
// Phase 1c : Per-bucket SubmitFusedBucketRead (fire-and-return).
// Phase 2  : LOCAL memcpy + REMOTE_STG legacy serial path - CPU work that
//            overlaps with NIC RDMA in flight from Phase 1c.  STG path is
//            self-contained (Read -> Wait -> memcpy(dst, staging) inside
//            RemoteDramScatterRead); its Wait is independent of Phase 3.
// Phase 3  : Wait every submitted ZC pair, map per-item success/failure,
//            emit per-item GET_BYTES counter for ZC successes only.
//
// Failure handling: Get has no master reservation, so SKIPPED / pair-failure /
// STG-failure all simply leave results[i]=false (no Abort RPC tail).
// caller dst[i] content on failure is undefined per legacy contract.
std::vector<bool> PoolClient::BatchGet(const std::vector<std::string>& keys,
                                       const std::vector<void*>& dsts,
                                       const std::vector<size_t>& sizes) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (!initialized_ || n == 0) return results;
  if (dsts.size() != n || sizes.size() != n) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: mismatched vector sizes");
    return results;
  }

  ScopedTimer timer("BatchGet", mori::modules::UMBP);
  MORI_UMBP_OBS_INC(batch_get_calls_);
  double bw_bytes_local = 0, bw_bytes_remote = 0;

  // Phase 0: route.
  std::vector<std::optional<RouteGetResult>> routes;
  auto status = master_client_->BatchRouteGet(keys, &routes);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: BatchRouteGet failed: {}", status.error_message());
    return results;
  }
  if (routes.size() != n) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: route count mismatch ({} vs {})", routes.size(), n);
    return results;
  }

  std::vector<BatchGetItemPlan> plans(n);
  std::unordered_map<std::string, std::vector<size_t>> zc_by_peer;
  std::vector<size_t> stg_indices;
  std::vector<size_t> local_indices;

  // Phase 1a: classify.  registered_mem_mutex_ acquired here (in
  // FindRegisteredMemory) BEFORE peers_mutex_ is taken in Phase 1b -
  // single global lock order, mirror of BatchPut.
  const std::string& self_node_id = config_.master_config.node_id;
  for (size_t i = 0; i < n; ++i) {
    plans[i].idx = i;
    if (!routes[i].has_value()) continue;
    const auto& r = *routes[i];
    const auto& loc = r.location;
    // Master tracks size at Put time; caller MUST pass the same size.
    // Issue #9 background: Lookup deliberately doesn't return size to
    // stay side-effect-free.  Without this check, the partial-last-page
    // logic in LogicalPageBytes would silently truncate or pull stale
    // tail bytes.  Per-item failure (continue), not whole-batch.
    if (sizes[i] != loc.size) {
      MORI_UMBP_ERROR("[PoolClient] BatchGet: caller size {} != stored size {} for key='{}'",
                      sizes[i], loc.size, keys[i]);
      continue;
    }
    if (loc.tier != TierType::DRAM && loc.tier != TierType::HBM) continue;

    auto parsed = ParseDramLocationIdWithLog(loc.location_id);
    if (!parsed) continue;
    if (parsed->pages.empty() || r.page_size == 0) {
      MORI_UMBP_ERROR(
          "[PoolClient] BatchGet: empty pages or zero page_size, key='{}' location_id='{}'",
          keys[i], loc.location_id);
      continue;
    }
    plans[i].parsed_pages = std::move(parsed->pages);
    plans[i].peer_node_id = loc.node_id;
    plans[i].page_size = r.page_size;
    plans[i].bytes_total = sizes[i];

    if (loc.node_id == self_node_id) {
      plans[i].kind = BatchGetItemPlan::Kind::LOCAL;
      local_indices.push_back(i);
      continue;
    }

    // Remote: validate master partial-fill BEFORE tagging REMOTE_*.
    // engine_desc_bytes and dram_memory_descs are the real RDMA
    // prerequisites; if either is empty, the BatchRead will fail with a
    // less actionable error inside the engine path.  peer_address is
    // ONLY used by the PeerService/SSD path (GetOrConnectPeer stashes
    // it but DRAM RDMA never reads it back), so an empty peer_address
    // is legitimate for clients that didn't expose PeerService and
    // MUST NOT be a SKIPPED trigger here.
    if (r.engine_desc_bytes.empty() || r.dram_memory_descs.empty()) {
      MORI_UMBP_ERROR(
          "[PoolClient] BatchGet: master partial-fill for key='{}' (engine_desc_bytes={}, "
          "dram_memory_descs={}); SKIPPED",
          keys[i], r.engine_desc_bytes.empty() ? "EMPTY" : "ok",
          r.dram_memory_descs.empty() ? "EMPTY" : "ok");
      plans[i].kind = BatchGetItemPlan::Kind::SKIPPED;
      continue;
    }

    auto reg = FindRegisteredMemory(dsts[i], sizes[i]);
    if (reg.has_value()) {
      plans[i].kind = BatchGetItemPlan::Kind::REMOTE_ZC;
      plans[i].local_mem = reg->first;
      plans[i].local_base_off = reg->second;
      zc_by_peer[loc.node_id].push_back(i);
      MORI_UMBP_OBS_INC(batch_get_items_);
    } else {
      plans[i].kind = BatchGetItemPlan::Kind::REMOTE_STG;
      stg_indices.push_back(i);
    }
  }

  // Phase 1b/c: per-peer plan + submit.  Mirror of BatchPut: outer
  // try/catch is REQUIRED.  On any throw, drain every already-pushed
  // bucket so in-flight RDMA completions don't dereference destroyed
  // bucket.statuses storage during stack unwind (UAF).
  std::vector<FusedBucket> buckets;
  buckets.reserve(zc_by_peer.size());
  bool phase1_threw = false;
  try {
    for (auto& [peer_id, indices] : zc_by_peer) {
      if (indices.empty()) continue;
      const auto& r0 = *routes[indices[0]];
      auto& peer = GetOrConnectPeer(peer_id, r0.peer_address, r0.engine_desc_bytes);
      FusedBucket bucket;
      bucket.peer_node_id = peer_id;
      PlanPeerForBatchGet(peer, routes, indices, keys, plans, bucket);
      if (bucket.pairs.empty()) continue;
      SubmitFusedBucketRead(bucket);
      buckets.push_back(std::move(bucket));
    }
  } catch (const std::exception& e) {
    MORI_UMBP_ERROR("[PoolClient] BatchGet: Phase 1 threw: {} ({} bucket(s) drained)", e.what(),
                    buckets.size());
    for (auto& bucket : buckets) WaitFusedBucket(bucket);
    phase1_threw = true;
  }

  // Phase 2a: LOCAL memcpy.  Runs while ZC RDMA is in flight on the NIC.
  // OOB on the local path is a master-invariant violation - log and skip
  // (results[i] stays false), matching legacy semantics.  Per-item
  // GET_BYTES counter emitted on success (traffic=local).
  for (size_t orig_idx : local_indices) {
    const auto& parsed_pages = plans[orig_idx].parsed_pages;
    const uint64_t page_size = plans[orig_idx].page_size;
    const size_t bytes_total = plans[orig_idx].bytes_total;
    char* dst_bytes = static_cast<char*>(dsts[orig_idx]);
    bool oob = false;
    for (size_t k = 0; k < parsed_pages.size() && !oob; ++k) {
      const auto& p = parsed_pages[k];
      if (p.buffer_index >= config_.dram_buffers.size()) {
        MORI_UMBP_ERROR("[PoolClient] BatchGet local: invalid buffer_index {}", p.buffer_index);
        oob = true;
        break;
      }
      auto& dram = config_.dram_buffers[p.buffer_index];
      const uint64_t off = static_cast<uint64_t>(p.page_index) * page_size;
      if (!dram.buffer || page_size > dram.size || off > dram.size - page_size) {
        MORI_UMBP_ERROR("[PoolClient] BatchGet local: OOB buf={} off={} page_size={} buf_size={}",
                        p.buffer_index, off, page_size, dram.size);
        oob = true;
        break;
      }
      const uint64_t bytes = LogicalPageBytes(k, parsed_pages.size(), page_size, bytes_total);
      std::memcpy(dst_bytes + k * page_size, static_cast<const char*>(dram.buffer) + off, bytes);
    }
    if (!oob) {
      results[orig_idx] = true;
      const double bytes_d = static_cast<double>(bytes_total);
      bw_bytes_local += bytes_d;
      master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                                 MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP,
                                 {{"traffic", "local"}}, bytes_d);
    }
  }

  // Phase 2b: REMOTE_STG via legacy per-item scatter-read (zero_copy=false).
  // Note: STG path is self-contained — RemoteDramScatterRead does Read -> Wait ->
  // memcpy(dst, staging) internally with staging_mutex_ held; its Wait is
  // INDEPENDENT of the Phase 3 ZC Wait below.
  for (size_t orig_idx : stg_indices) {
    const auto& r = *routes[orig_idx];
    auto& peer = GetOrConnectPeer(r.location.node_id, r.peer_address, r.engine_desc_bytes);
    EnsureBufferDescsCached(peer, r.dram_memory_descs);
    bool ok = RemoteDramScatterRead(peer, plans[orig_idx].parsed_pages, plans[orig_idx].page_size,
                                    dsts[orig_idx], sizes[orig_idx], /*zero_copy=*/false);
    if (ok) {
      results[orig_idx] = true;
      const double bytes_d = static_cast<double>(sizes[orig_idx]);
      bw_bytes_remote += bytes_d;
      master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL,
                                 MORI_UMBP_METRIC_CLIENT_GET_BYTES_TOTAL_HELP,
                                 {{"traffic", "remote"}}, bytes_d);
    } else {
      MORI_UMBP_WARN("[PoolClient] BatchGet: REMOTE_STG ScatterRead failed for key='{}' size={}",
                     keys[orig_idx], sizes[orig_idx]);
    }
  }

  // Phase 3: Wait + map ZC failures.  When phase1_threw, the buckets were
  // already drained in the catch above and all REMOTE_ZC items are
  // considered failed - leave results[i]=false (no Abort needed for Get).
  if (!phase1_threw) {
    for (auto& bucket : buckets) {
      WaitFusedBucket(bucket);
      MapBucketFailuresRead(bucket, plans, sizes, results, &bw_bytes_remote, master_client_.get());
    }
  }
  // No Master RPC tail (no Finalize, no Abort) — Get has no reservation.

  ObserveBatchBandwidth(*master_client_, MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH,
                        MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH_HELP, NodeId(), bw_bytes_local,
                        bw_bytes_remote, timer.ElapsedSeconds());

  return results;
}

MasterClient& PoolClient::Master() { return *master_client_; }

bool PoolClient::IsInitialized() const { return initialized_; }

bool PoolClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] ReportExternalKvBlocks: not initialized");
    return false;
  }
  auto status = master_client_->ReportExternalKvBlocks(config_.master_config.node_id, hashes, tier);
  return status.ok();
}

bool PoolClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] RevokeExternalKvBlocks: not initialized");
    return false;
  }
  auto status = master_client_->RevokeExternalKvBlocks(config_.master_config.node_id, hashes);
  return status.ok();
}

bool PoolClient::MatchExternalKv(const std::vector<std::string>& hashes,
                                 std::vector<MasterClient::ExternalKvNodeMatch>* out_matches) {
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] MatchExternalKv: not initialized");
    return false;
  }
  auto status = master_client_->MatchExternalKv(hashes, out_matches);
  return status.ok();
}

// ---------------------------------------------------------------------------
// Peer connection management
// ---------------------------------------------------------------------------

PoolClient::PeerConnection& PoolClient::GetOrConnectPeer(
    const std::string& node_id, const std::string& peer_address,
    const std::vector<uint8_t>& engine_desc_bytes) {
  std::lock_guard<std::mutex> lock(peers_mutex_);
  auto it = peers_.find(node_id);
  if (it != peers_.end()) {
    return *it->second;
  }

  auto peer = std::make_unique<PeerConnection>();
  peer->peer_address = peer_address;

  if (io_engine_ && !engine_desc_bytes.empty()) {
    try {
      auto handle = msgpack::unpack(reinterpret_cast<const char*>(engine_desc_bytes.data()),
                                    engine_desc_bytes.size());
      peer->engine_desc = handle.get().as<mori::io::EngineDesc>();
      io_engine_->RegisterRemoteEngine(peer->engine_desc);
      peer->engine_registered = true;
      MORI_UMBP_INFO("[PoolClient] Registered remote engine for node '{}'", node_id);
    } catch (const std::exception& e) {
      // Same containment policy as EnsureBufferDescsCachedLocked: a
      // malformed engine_desc shouldn't unwind the BatchPut/Get caller.
      // The peer is still inserted (leaving engine_registered=false);
      // subsequent RDMA on it will fail in the engine path with a
      // clear error rather than crash the process.
      MORI_UMBP_ERROR("[PoolClient] Failed to unpack EngineDesc for node '{}': {}", node_id,
                      e.what());
    }
  }

  // PeerService connection (stub + staging MemoryDesc) is lazy-initialized
  // on first SSD operation. DRAM path doesn't need PeerService.

  auto* raw = peer.get();
  peers_.emplace(node_id, std::move(peer));
  return *raw;
}

void PoolClient::EnsureBufferDescsCachedLocked(PeerConnection& peer,
                                               const std::vector<BufferMemoryDescBytes>& descs) {
  // Caller holds peers_mutex_.  Used by the BatchPut fused path so a
  // single critical section covers both hydrate and per-peer snapshot.
  if (descs.empty()) return;
  for (const auto& bd : descs) {
    if (bd.desc_bytes.empty()) continue;
    if (bd.buffer_index >= peer.dram_memories.size()) {
      peer.dram_memories.resize(bd.buffer_index + 1);
    }
    if (!IsValidMemoryDesc(peer.dram_memories[bd.buffer_index])) {
      // MemoryDesc is immutable: once cached, never overwrite the entry.
      try {
        auto handle = msgpack::unpack(reinterpret_cast<const char*>(bd.desc_bytes.data()),
                                      bd.desc_bytes.size());
        peer.dram_memories[bd.buffer_index] = handle.get().as<mori::io::MemoryDesc>();
      } catch (const std::exception& e) {
        MORI_UMBP_ERROR("[PoolClient] Failed to unpack DRAM MemoryDesc for buffer_index {}: {}",
                        bd.buffer_index, e.what());
      }
    }
  }
}

void PoolClient::EnsureBufferDescsCached(PeerConnection& peer,
                                         const std::vector<BufferMemoryDescBytes>& descs) {
  if (descs.empty()) return;
  std::lock_guard<std::mutex> lock(peers_mutex_);
  EnsureBufferDescsCachedLocked(peer, descs);
}

void PoolClient::IssueBatchWrite(const mori::io::MemDescVec& local_src,
                                 const mori::io::BatchSizeVec& local_offsets,
                                 const mori::io::MemDescVec& remote_dest,
                                 const mori::io::BatchSizeVec& remote_offsets,
                                 const mori::io::BatchSizeVec& sizes,
                                 mori::io::TransferStatusPtrVec& statuses,
                                 mori::io::TransferUniqueIdVec& ids) {
  // Default forward-to-engine implementation.  Test subclasses override
  // under MORI_UMBP_TESTING to inject pair-level failures.
  io_engine_->BatchWrite(local_src, local_offsets, remote_dest, remote_offsets, sizes, statuses,
                         ids);
}

void PoolClient::IssueBatchRead(const mori::io::MemDescVec& local_dest,
                                const mori::io::BatchSizeVec& local_offsets,
                                const mori::io::MemDescVec& remote_src,
                                const mori::io::BatchSizeVec& remote_offsets,
                                const mori::io::BatchSizeVec& sizes,
                                mori::io::TransferStatusPtrVec& statuses,
                                mori::io::TransferUniqueIdVec& ids) {
  // Default forward-to-engine implementation.  Test subclasses override
  // under MORI_UMBP_TESTING to inject pair-level Read failures.
  io_engine_->BatchRead(local_dest, local_offsets, remote_src, remote_offsets, sizes, statuses,
                        ids);
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
  // Snapshot the remote MemoryDesc under peers_mutex_ so subsequent RDMA does
  // not race with concurrent EnsureBufferDescsCached resize on dram_memories.
  // MemoryDesc is immutable ("once cached, never overwrite"), so a value copy
  // remains valid for the lifetime of this call.
  mori::io::MemoryDesc remote_mem;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    if (buffer_index >= peer.dram_memories.size() ||
        !IsValidMemoryDesc(peer.dram_memories[buffer_index])) {
      MORI_UMBP_ERROR("[PoolClient] RemoteDramWrite: invalid buffer_index {} (size={}, valid={})",
                      buffer_index, peer.dram_memories.size(),
                      buffer_index < peer.dram_memories.size()
                          ? IsValidMemoryDesc(peer.dram_memories[buffer_index])
                          : false);
      return false;
    }
    remote_mem = peer.dram_memories[buffer_index];
  }

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
  // Snapshot the remote MemoryDesc under peers_mutex_ so subsequent RDMA does
  // not race with concurrent EnsureBufferDescsCached resize on dram_memories.
  // MemoryDesc is immutable ("once cached, never overwrite"), so a value copy
  // remains valid for the lifetime of this call.
  mori::io::MemoryDesc remote_mem;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    if (buffer_index >= peer.dram_memories.size() ||
        !IsValidMemoryDesc(peer.dram_memories[buffer_index])) {
      MORI_UMBP_ERROR("[PoolClient] RemoteDramRead: invalid buffer_index {} (size={}, valid={})",
                      buffer_index, peer.dram_memories.size(),
                      buffer_index < peer.dram_memories.size()
                          ? IsValidMemoryDesc(peer.dram_memories[buffer_index])
                          : false);
      return false;
    }
    remote_mem = peer.dram_memories[buffer_index];
  }

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
// Remote DRAM scatter-gather path (multi-page Put/Get)
// ---------------------------------------------------------------------------

namespace {

// Group `pages` (positionally) by buffer_index, preserving the original
// page-list ordering as the *source* layout.  For each distinct
// buffer_index encountered (in first-seen order), emit one entry in
// `groups` containing the indices `i` (into `pages`) that target that
// buffer.  buffer_indices_out preserves the same first-seen order so the
// outer N for IOEngine batch ops follows the same iteration pattern.
struct ScatterGroup {
  uint32_t buffer_index;
  std::vector<size_t> src_page_indices;  // indices into the original `pages` vector
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

}  // namespace

// Build the IOEngine BatchWrite/Read parameter shape from a list of
// pages + a single local MemoryDesc + the page offset within `local_mem`.
//
// Returns `false` if any page targets a buffer_index that has not been
// hydrated in `peer.dram_memories` (meaning the caller forgot to
// EnsureBufferDescsCached).
bool PoolClient::RemoteDramScatterWrite(PeerConnection& peer,
                                        const std::vector<PageLocation>& pages, uint64_t page_size,
                                        const void* src, size_t size, bool zero_copy) {
  if (!io_engine_) return false;
  if (pages.empty() || page_size == 0) {
    MORI_UMBP_ERROR("[PoolClient] ScatterWrite: empty pages or page_size=0");
    return false;
  }
  // Allow partial last page: size in ((N-1)*ps, N*ps].
  if (!SizeMatchesAllocation(size, pages.size(), page_size)) {
    MORI_UMBP_ERROR("[PoolClient] ScatterWrite: size {} not in allocation window ({}..{}]", size,
                    (pages.size() - 1) * page_size, pages.size() * page_size);
    return false;
  }

  // Resolve local source: zero-copy registered region, else staging buffer.
  // FindRegisteredMemory checks the *whole* src range (not per-page) — that
  // matches both the spec and the existing RemoteDramWrite semantics.
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
    } else {
      MORI_UMBP_WARN(
          "[PoolClient] ScatterWrite zero_copy=true but pointer not registered, "
          "falling back to staging");
    }
  }
  if (!used_zero_copy) {
    if (size > config_.staging_buffer_size) {
      MORI_UMBP_ERROR("[PoolClient] ScatterWrite: size {} exceeds staging_buffer_size {}", size,
                      config_.staging_buffer_size);
      return false;
    }
    staging_lock.lock();
    std::memcpy(staging_buffer_.get(), src, size);
    local_mem = staging_mem_;
    local_base_offset = 0;
  }

  auto groups = GroupPagesByBuffer(pages);
  const size_t N = groups.size();

  // Snapshot remote MemoryDescs under peers_mutex_ to avoid racing with
  // concurrent EnsureBufferDescsCached resize on dram_memories.  MemoryDesc
  // is immutable ("once cached, never overwrite"), so value copies remain
  // valid for the rest of this call.
  mori::io::MemDescVec remote_descs;
  remote_descs.reserve(N);
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (size_t k = 0; k < N; ++k) {
      const auto& g = groups[k];
      if (g.buffer_index >= peer.dram_memories.size() ||
          !IsValidMemoryDesc(peer.dram_memories[g.buffer_index])) {
        MORI_UMBP_ERROR("[PoolClient] ScatterWrite: buffer_index {} not hydrated on peer",
                        g.buffer_index);
        return false;
      }
      remote_descs.push_back(peer.dram_memories[g.buffer_index]);
    }
  }

  // Build local/remote offsets and per-pair sizes off-lock.
  mori::io::MemDescVec local_descs(N, local_mem);
  mori::io::BatchSizeVec local_offsets(N), remote_offsets(N), sizes_v(N);
  for (size_t k = 0; k < N; ++k) {
    const auto& g = groups[k];
    auto& l_off = local_offsets[k];
    auto& r_off = remote_offsets[k];
    auto& sz = sizes_v[k];
    l_off.reserve(g.src_page_indices.size());
    r_off.reserve(g.src_page_indices.size());
    sz.reserve(g.src_page_indices.size());
    for (size_t spi : g.src_page_indices) {
      l_off.push_back(local_base_offset + spi * page_size);
      r_off.push_back(static_cast<uint64_t>(pages[spi].page_index) * page_size);
      // Last logical page may be partial — only RDMA-write its real bytes.
      sz.push_back(LogicalPageBytes(spi, pages.size(), page_size, size));
    }
  }

  // Allocate per-pair status + ids.  IOEngine writes status pointers, so
  // hold the storage in a vector with a stable address while waiting.
  std::vector<mori::io::TransferStatus> statuses(N);
  mori::io::TransferStatusPtrVec status_ptrs(N);
  mori::io::TransferUniqueIdVec ids(N);
  for (size_t k = 0; k < N; ++k) {
    status_ptrs[k] = &statuses[k];
    ids[k] = io_engine_->AllocateTransferUniqueId();
  }

  MORI_UMBP_DEBUG("[PoolClient] ScatterWrite start: pages={}, groups={}, zero_copy={}",
                  pages.size(), N, used_zero_copy);
  io_engine_->BatchWrite(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v,
                         status_ptrs, ids);

  bool all_ok = true;
  for (size_t k = 0; k < N; ++k) {
    statuses[k].Wait();
    if (!statuses[k].Succeeded()) {
      MORI_UMBP_ERROR("[PoolClient] ScatterWrite group {} (buf={}) failed: uid={}, {}", k,
                      groups[k].buffer_index, ids[k], statuses[k].Message());
      all_ok = false;
    }
  }
  if (all_ok) {
    MORI_UMBP_DEBUG("[PoolClient] ScatterWrite done: pages={}, groups={}", pages.size(), N);
  }
  return all_ok;
}

bool PoolClient::RemoteDramScatterRead(PeerConnection& peer, const std::vector<PageLocation>& pages,
                                       uint64_t page_size, void* dst, size_t size, bool zero_copy) {
  if (!io_engine_) return false;
  if (pages.empty() || page_size == 0) {
    MORI_UMBP_ERROR("[PoolClient] ScatterRead: empty pages or page_size=0");
    return false;
  }
  // Allow partial last page: size in ((N-1)*ps, N*ps].
  if (!SizeMatchesAllocation(size, pages.size(), page_size)) {
    MORI_UMBP_ERROR("[PoolClient] ScatterRead: size {} not in allocation window ({}..{}]", size,
                    (pages.size() - 1) * page_size, pages.size() * page_size);
    return false;
  }

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
    } else {
      MORI_UMBP_WARN(
          "[PoolClient] ScatterRead zero_copy=true but pointer not registered, "
          "falling back to staging");
    }
  }
  if (!used_zero_copy) {
    if (size > config_.staging_buffer_size) {
      MORI_UMBP_ERROR("[PoolClient] ScatterRead: size {} exceeds staging_buffer_size {}", size,
                      config_.staging_buffer_size);
      return false;
    }
    staging_lock.lock();
    local_mem = staging_mem_;
    local_base_offset = 0;
  }

  auto groups = GroupPagesByBuffer(pages);
  const size_t N = groups.size();

  // Snapshot remote MemoryDescs under peers_mutex_ to avoid racing with
  // concurrent EnsureBufferDescsCached resize on dram_memories.  MemoryDesc
  // is immutable ("once cached, never overwrite"), so value copies remain
  // valid for the rest of this call.
  mori::io::MemDescVec remote_descs;
  remote_descs.reserve(N);
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (size_t k = 0; k < N; ++k) {
      const auto& g = groups[k];
      if (g.buffer_index >= peer.dram_memories.size() ||
          !IsValidMemoryDesc(peer.dram_memories[g.buffer_index])) {
        MORI_UMBP_ERROR("[PoolClient] ScatterRead: buffer_index {} not hydrated on peer",
                        g.buffer_index);
        return false;
      }
      remote_descs.push_back(peer.dram_memories[g.buffer_index]);
    }
  }

  // Build local/remote offsets and per-pair sizes off-lock.
  mori::io::MemDescVec local_descs(N, local_mem);
  mori::io::BatchSizeVec local_offsets(N), remote_offsets(N), sizes_v(N);
  for (size_t k = 0; k < N; ++k) {
    const auto& g = groups[k];
    auto& l_off = local_offsets[k];
    auto& r_off = remote_offsets[k];
    auto& sz = sizes_v[k];
    l_off.reserve(g.src_page_indices.size());
    r_off.reserve(g.src_page_indices.size());
    sz.reserve(g.src_page_indices.size());
    for (size_t spi : g.src_page_indices) {
      l_off.push_back(local_base_offset + spi * page_size);
      r_off.push_back(static_cast<uint64_t>(pages[spi].page_index) * page_size);
      // Last logical page may be partial — only RDMA-read its real bytes.
      sz.push_back(LogicalPageBytes(spi, pages.size(), page_size, size));
    }
  }

  std::vector<mori::io::TransferStatus> statuses(N);
  mori::io::TransferStatusPtrVec status_ptrs(N);
  mori::io::TransferUniqueIdVec ids(N);
  for (size_t k = 0; k < N; ++k) {
    status_ptrs[k] = &statuses[k];
    ids[k] = io_engine_->AllocateTransferUniqueId();
  }

  MORI_UMBP_DEBUG("[PoolClient] ScatterRead start: pages={}, groups={}, zero_copy={}", pages.size(),
                  N, used_zero_copy);
  io_engine_->BatchRead(local_descs, local_offsets, remote_descs, remote_offsets, sizes_v,
                        status_ptrs, ids);

  bool all_ok = true;
  for (size_t k = 0; k < N; ++k) {
    statuses[k].Wait();
    if (!statuses[k].Succeeded()) {
      MORI_UMBP_ERROR("[PoolClient] ScatterRead group {} (buf={}) failed: uid={}, {}", k,
                      groups[k].buffer_index, ids[k], statuses[k].Message());
      all_ok = false;
    }
  }
  if (!all_ok) return false;

  // For staging-fallback path, copy from staging to caller's dst.  We held
  // the staging mutex across the whole BatchRead so the buffer is intact.
  if (!used_zero_copy) {
    std::memcpy(dst, staging_buffer_.get(), size);
  }
  MORI_UMBP_DEBUG("[PoolClient] ScatterRead done: pages={}, groups={}", pages.size(), N);
  return true;
}

// ---------------------------------------------------------------------------
// Remote SSD path (RDMA + PeerService gRPC coordination)
// ---------------------------------------------------------------------------

bool PoolClient::EnsurePeerServiceConnection(PeerConnection& peer) {
  std::lock_guard<std::mutex> lock(peer.ssd_op_mutex);
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
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  // Step 1: Pre-allocate a write slot on the remote peer
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

  // Step 2: RDMA write data into the allocated staging slot
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

  // Step 3: CommitSsdWrite with lease_id (slot is released by server on completion)
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
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  // Step 1: PrepareSsdRead — server allocates a slot and loads SSD data
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

  // Step 2: RDMA read from the allocated staging slot
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
        const uint32_t max_retries = ReleaseLeaseMaxRetries();
        for (uint32_t attempt = 0; attempt < max_retries; ++attempt) {
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

  // Step 3: Release staging slot (with lightweight retry)
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

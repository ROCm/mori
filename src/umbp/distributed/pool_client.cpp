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
#include <cstdlib>
#include <cstring>
#include <msgpack.hpp>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

#include "mori/io/backend.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp/distributed/peer/ssd_copy_pipeline.h"
#include "umbp/distributed/ssd_read_lease.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {

bool IsValidMemoryDesc(const mori::io::MemoryDesc& desc) { return desc.size > 0; }

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

const std::vector<double>& BatchBandwidthBucketsGiBps() {
  static const std::vector<double> buckets = {
      0.1,  0.2,  0.5,  1.0,   2.0,   3.0,   4.0,   6.0,   8.0,   12.0,  16.0,  24.0, 32.0,
      48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 320.0, 384.0, 448.0, 512.0, 640.0, 800.0};
  return buckets;
}

struct BatchBandwidthSplit {
  double local = 0.0;
  double remote = 0.0;
};

// Bandwidth predicate.  BatchGet uses `bool` (no dedup); BatchPut uses
// PutEntryOutcome (kAlreadyExists is success-to-caller but moves no
// bytes — excluded from bandwidth).
inline bool IsCountedForBandwidth(bool r) { return r; }
inline bool IsCountedForBandwidth(PoolClient::PutEntryOutcome r) {
  return r == PoolClient::PutEntryOutcome::kSucceeded;
}

template <typename Route, typename Result>
BatchBandwidthSplit ComputeBatchBandwidthBytes(const std::vector<Result>& results,
                                               const std::vector<size_t>& sizes,
                                               const std::vector<std::optional<Route>>& routes,
                                               std::string_view local_node_id) {
  // guard against mismatched sizes
  const size_t limit = std::min({results.size(), sizes.size(), routes.size()});
  BatchBandwidthSplit acc;
  for (size_t i = 0; i < limit; ++i) {
    if (!IsCountedForBandwidth(results[i])) continue;
    const double bytes = static_cast<double>(sizes[i]);
    // No route means the key was served from local storage (fallback path).
    const bool is_local = !routes[i].has_value() || routes[i]->node_id == local_node_id;
    (is_local ? acc.local : acc.remote) += bytes;
  }
  return acc;
}

void ObserveBatchBandwidth(MasterClient& master_client, double bytes, double seconds,
                           const char* metric_name, const char* metric_help,
                           std::string_view traffic) {
  if (bytes <= 0.0 || seconds <= 0.0) return;
  const double gibps = (bytes / seconds) / kGiB;
  if (gibps <= 0.0) return;
  MasterClient::Labels labels = {{"traffic", std::string(traffic)}};
  master_client.Observe(metric_name, metric_help, std::move(labels), BatchBandwidthBucketsGiBps(),
                        gibps);
}

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

// Crash-restart SSD leftover policy (v1 = discard): wipe physical SSD bytes a
// previous crashed process left behind at startup.  Default on; set
// UMBP_SSD_STARTUP_DISCARD=0/false to keep leftover (opt-out hook for a future
// rebuild-from-backend policy — there is no rebuild path yet).
bool SsdStartupDiscardEnabled() {
  const char* v = std::getenv("UMBP_SSD_STARTUP_DISCARD");
  if (v == nullptr) return true;
  const std::string s(v);
  return !(s == "0" || s == "false" || s == "FALSE" || s == "off");
}

// Total attempts for a remote SSD get.  Retryable outcomes (kRetry) are
// NO_SLOT (transient slot exhaustion, no slot claimed) and a reader-local lease
// expiry.  Defaults to 1 (no retry); operators opt into bounded retry to absorb
// slot contention.  A slow/failed PrepareSsdRead (DEADLINE_EXCEEDED etc.) is a
// hard failure, NOT retried — the peer may already hold a claimed slot, so a
// retry would only pile up more staging-slot occupation.  NOT_FOUND is always a
// definitive miss (no retry).
uint32_t SsdGetTransientMaxAttempts() {
  static const uint32_t v = GetEnvUint32("UMBP_SSD_GET_MAX_ATTEMPTS", 1, /*min_allowed=*/1);
  return v;
}

std::chrono::milliseconds SsdGetRetryBackoff() {
  static const auto v = std::chrono::milliseconds(
      GetEnvUint32("UMBP_SSD_GET_RETRY_BACKOFF_MS", 2, /*min_allowed=*/1));
  return v;
}

// Bounded client deadline for a single PrepareSsdRead RPC so a hung/slow peer
// cannot block the serial per-key remote SSD read loop indefinitely.  A read
// that cannot even return within this budget is past any useful lease window
// and is treated as a hard failure (NOT retried: the peer may already hold a
// claimed slot, and a retry would only claim another).  When the env is unset
// the caller falls back to the configured lease timeout (cluster-homogeneous
// assumption); 0 here means "use the fallback".
std::chrono::milliseconds SsdPrepareRpcTimeoutOverride() {
  static const auto v = GetEnvMilliseconds("UMBP_SSD_PREPARE_TIMEOUT_MS",
                                           std::chrono::milliseconds(0), /*min_allowed=*/0);
  return v;
}

// Bounded client deadline for a best-effort ReleaseSsdLease RPC.  Release is
// never on the critical path (the slot is also reclaimed by TTL), so cap it
// tightly to avoid blocking on a slow peer.
std::chrono::milliseconds ReleaseLeaseRpcTimeout() {
  static const auto v = GetEnvMilliseconds("UMBP_RELEASE_LEASE_TIMEOUT_MS",
                                           std::chrono::milliseconds(1000), /*min_allowed=*/1);
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
    // Local host base pointer, so PeerDramAllocator can resolve a committed
    // key's pages to readable segments for the SSD copy worker (Phase 2).
    cfg.buffer_bases.push_back(bufs[i].buffer);
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

  // Peer-side SSD tier owner.  Built only when SSD is enabled; registered as an
  // owned-location event source + SSD capacity provider.  Phase 2 populates it
  // via the copy-on-commit pipeline below; reads remain Phase 3.  Clear() now
  // quiesces the pipeline and clears peer_ssd_ alongside peer_alloc_.
  if (config_.ssd.enabled) {
    peer_ssd_ = std::make_unique<PeerSsdManager>(config_.ssd);
    // Crash-restart leftover (v1 = discard): wipe stale SSD bytes before the
    // pipeline starts so used capacity and the empty owned_ map start consistent
    // (env-gated; see SsdStartupDiscardEnabled / DiscardLeftoverOnStartup).
    if (SsdStartupDiscardEnabled()) peer_ssd_->DiscardLeftoverOnStartup();
    master_client_->SetPeerSsdManager(peer_ssd_.get());
    // Copy-on-commit pipeline: commits enqueue here, a worker pins the DRAM
    // pages and writes them to the local SSD tier.  Started below.
    ssd_copy_pipeline_ = std::make_unique<SsdCopyPipeline>(peer_alloc_.get(), peer_ssd_.get());
    ssd_copy_pipeline_->Start();
  }

  // Pack engine_desc for master registration.
  std::vector<uint8_t> engine_desc_bytes;
  if (io_engine_) {
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, io_engine_->GetEngineDesc());
    engine_desc_bytes.assign(sbuf.data(), sbuf.data() + sbuf.size());
  }

  // SSD staging buffer (one per process; not part of DRAM exports).  Phase 3
  // serves remote SSD reads out of it; allocated up front when SSD is enabled.
  if (config_.ssd.enabled) {
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
    // Wire the SSD read path: peer_ssd_ + the RDMA-registered SSD staging buffer
    // (both null/empty when SSD is disabled, leaving SsdRpcAvailable() false).
    peer_service_ = std::make_unique<PeerServiceServer>(
        peer_alloc_.get(), peer_ssd_.get(), ssd_staging_buffer_.get(),
        ssd_staging_buffer_ ? config_.staging_buffer_size : 0, ssd_staging_mem_desc_bytes_,
        config_.ssd_read_slots, config_.ssd_lease_timeout_s, engine_desc_bytes,
        master_client_.get(), ssd_copy_pipeline_.get());
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

  // Register the SSD metrics provider before RegisterSelf starts the metrics
  // thread (the list is read lock-free afterwards).  See PublishSsdMetrics.
  if (config_.ssd.enabled) {
    master_client_->AddMetricsProvider([this] { PublishSsdMetrics(); });
  }

  // Master register.  In the new design master holds no DRAM-side
  // metadata; only membership + capacity-snapshot (SSD capacity rides in
  // tier_capacities as TierType::SSD).
  auto status =
      master_client_->RegisterSelf(config_.tier_capacities, peer_address, engine_desc_bytes);
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
    // Stop the periodic metrics thread before tearing down the SSD components
    // its provider reads (peer_service_ / ssd_copy_pipeline_ / peer_ssd_), so no
    // provider callback runs against freed state.  Idempotent with ~MasterClient.
    master_client_->StopMetricsReporting();
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

  // Stop the copy pipeline AFTER the peer service (RPC commit path) is gone so
  // no commit can enqueue into a stopping pipeline.  Stop() drops queued tasks
  // and waits for the in-flight copy to finish + release its DRAM pin before
  // we tear down peer_alloc_ / peer_ssd_ below.
  if (ssd_copy_pipeline_) {
    ssd_copy_pipeline_->Stop();
    ssd_copy_pipeline_.reset();
  }

  if (peer_alloc_) {
    peer_alloc_->StopReaper();
    peer_alloc_.reset();
  }

  // Heartbeat thread is already stopped above, so MasterClient no longer reads
  // ssd_manager_; safe to drop the SSD tier owner here.
  peer_ssd_.reset();

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

bool PoolClient::Clear() {
  // Vacuously done: nothing has been initialized so there is no state to
  // clear and no master to notify.  Treat as success so callers in
  // shutdown / teardown paths do not surface a spurious error.
  if (!initialized_.load()) return true;
  // Quiesce the copy pipeline first: drop queued tasks and wait for any
  // in-flight copy to finish + release its pin.  Otherwise a copy completing
  // after the clear below would re-record an SSD location and emit ADD SSD
  // for a key we just collapsed.
  if (ssd_copy_pipeline_) ssd_copy_pipeline_->Quiesce();
  if (peer_alloc_) peer_alloc_->ClearLocal();
  if (peer_ssd_) peer_ssd_->ClearLocal();

  bool ok = true;
  if (master_client_) {
    ok = master_client_->ClearFullSync();
    if (!ok) MORI_UMBP_WARN("[PoolClient] Clear full-sync heartbeat failed");
  }

  // Always resume — even if the full-sync RPC failed — so the pipeline is
  // never left permanently paused (the heartbeat loop retries convergence).
  if (ssd_copy_pipeline_) ssd_copy_pipeline_->Resume();
  return ok;
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
  auto alloc_res = peer_alloc_->Allocate(key, size, tier);
  switch (alloc_res.outcome) {
    case PeerDramAllocator::Outcome::kSuccessAlreadyExists:
      return PutAttemptOutcome::kSuccessAlreadyExists;
    case PeerDramAllocator::Outcome::kFailed:
    case PeerDramAllocator::Outcome::kFailedNoSpace:
      // Peer allocator already logged the specific reason.
      return PutAttemptOutcome::kRetry;
    case PeerDramAllocator::Outcome::kSuccessAllocated:
      break;
  }
  const auto& pending = *alloc_res.slot;
  if (!LocalPutPages(pending.pages, peer_alloc_->PageSize(), src, size)) {
    peer_alloc_->Abort(pending.slot_id);
    return PutAttemptOutcome::kFatal;
  }
  uint64_t committed_bytes = 0;
  if (!peer_alloc_->Commit(pending.slot_id, key, committed_bytes)) {
    peer_alloc_->Abort(pending.slot_id);
    return PutAttemptOutcome::kFatal;
  }
  // Owner-side commit succeeded: best-effort async copy to local SSD.
  if (ssd_copy_pipeline_) {
    ssd_copy_pipeline_->Enqueue(SsdCopyTask{key, tier, size});
  }
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
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
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
  return GetAttemptOutcome::kSuccess;
}

PoolClient::GetAttemptOutcome PoolClient::ExecuteLocalSsdGet(const std::string& key, void* dst,
                                                             size_t size) {
  // reader == owner: read straight into the user buffer (no staging / RDMA /
  // lease — those serve remote readers).  kNotFound -> kRetry (miss, not error).
  if (!peer_ssd_) {
    MORI_UMBP_ERROR("[PoolClient] Local SSD Get requested but PeerSsdManager unavailable");
    return GetAttemptOutcome::kFatal;
  }
  SsdReadOutcome outcome = peer_ssd_->PrepareRead(key, dst, size);
  switch (outcome.status) {
    case SsdReadStatus::kOk:
      break;
    case SsdReadStatus::kNotFound:
      return GetAttemptOutcome::kRetry;
    case SsdReadStatus::kSizeTooLarge:
    case SsdReadStatus::kError:
      MORI_UMBP_WARN("[PoolClient] Local SSD Get key='{}' failed (status={}, size={})", key,
                     static_cast<int>(outcome.status), outcome.size);
      return GetAttemptOutcome::kFatal;
  }
  // Guard against a short read filling only part of the user buffer (mirrors the
  // remote path's size check).
  if (outcome.size != size) {
    MORI_UMBP_WARN("[PoolClient] Local SSD Get key='{}' size mismatch (wanted {}, got {})", key,
                   size, outcome.size);
    return GetAttemptOutcome::kFatal;
  }
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
  return GetAttemptOutcome::kSuccess;
}

std::unordered_map<std::string, std::vector<PoolClient::BatchPutItem>>
PoolClient::PartitionBatchPutTargets(const std::vector<std::string>& keys,
                                     const std::vector<const void*>& srcs,
                                     const std::vector<size_t>& sizes,
                                     const std::vector<std::optional<RoutePutResult>>& routes,
                                     std::vector<PutEntryOutcome>* results) {
  std::unordered_map<std::string, std::vector<BatchPutItem>> remote_groups;
  const size_t count = keys.size();
  for (size_t i = 0; i < count; ++i) {
    if (i >= routes.size() || !routes[i].has_value()) continue;
    const auto& route = routes[i].value();
    // Master-side dedup hit.
    if (route.outcome == RoutePutOutcome::kAlreadyExists) {
      (*results)[i] = PutEntryOutcome::kAlreadyExists;
      continue;
    }
    if (route.node_id == config_.master_config.node_id) {
      switch (ExecuteLocalPut(keys[i], srcs[i], sizes[i], route.tier)) {
        case PutAttemptOutcome::kSuccess:
          (*results)[i] = PutEntryOutcome::kSucceeded;
          break;
        case PutAttemptOutcome::kSuccessAlreadyExists:
          (*results)[i] = PutEntryOutcome::kAlreadyExists;
          break;
        case PutAttemptOutcome::kRetry:
        case PutAttemptOutcome::kFatal:
          break;  // default kFailed
      }
      continue;
    }
    if (route.tier != TierType::DRAM && route.tier != TierType::HBM) continue;
    remote_groups[route.node_id].push_back(BatchPutItem{
        .index = i, .key = &keys[i], .src = srcs[i], .size = sizes[i], .route = route});
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
  const auto call_start = std::chrono::steady_clock::now();
  if (keys.size() != srcs.size() || keys.size() != sizes.size()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: vector length mismatch");
    return std::vector<bool>(keys.size(), false);
  }
  if (!initialized_) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: client not initialized");
    return std::vector<bool>(keys.size(), false);
  }

  // Tri-state pipeline; projected to vector<bool> at return.
  std::vector<PutEntryOutcome> outcomes(keys.size(), PutEntryOutcome::kFailed);

  std::vector<uint64_t> block_sizes(keys.size());
  for (size_t i = 0; i < sizes.size(); ++i) block_sizes[i] = static_cast<uint64_t>(sizes[i]);
  std::vector<std::optional<RoutePutResult>> routes;
  std::unordered_set<std::string> excludes;
  auto status = master_client_->BatchRoutePut(keys, block_sizes, excludes, &routes);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] BatchPut: BatchRoutePut failed: {}", status.error_message());
    return std::vector<bool>(keys.size(), false);
  }
  if (routes.size() < keys.size()) routes.resize(keys.size());

  auto remote_groups = PartitionBatchPutTargets(keys, srcs, sizes, routes, &outcomes);
  for (auto& [node_id, items] : remote_groups) {
    ProcessRemoteBatchPut(items, &outcomes);
  }

  const auto call_end = std::chrono::steady_clock::now();
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(call_end - call_start).count();
  if (seconds > 0.0) {
    auto split = ComputeBatchBandwidthBytes(outcomes, sizes, routes, config_.master_config.node_id);
    ObserveBatchBandwidth(*master_client_, split.local, seconds,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH_HELP, "local");
    ObserveBatchBandwidth(*master_client_, split.remote, seconds,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH,
                          MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH_HELP, "remote");
  }

  std::vector<bool> results(outcomes.size());
  for (size_t i = 0; i < outcomes.size(); ++i) {
    results[i] = (outcomes[i] != PutEntryOutcome::kFailed);
  }
  return results;
}

std::vector<bool> PoolClient::BatchGet(const std::vector<std::string>& keys,
                                       const std::vector<void*>& dsts,
                                       const std::vector<size_t>& sizes) {
  const auto call_start = std::chrono::steady_clock::now();
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

  // DRAM/HBM remote reads go through the batched RDMA path (remote_groups);
  // remote SSD reads go through a per-key staging path (ssd_remote_groups)
  // because each needs its own peer-side staging slot + lease.
  std::unordered_map<std::string, std::vector<BatchGetItem>> remote_groups;
  std::unordered_map<std::string, std::vector<BatchGetItem>> ssd_remote_groups;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (i >= routes.size() || !routes[i].has_value()) {
      if (peer_alloc_) {
        auto outcome = ExecuteLocalGet(keys[i], const_cast<void*>(dsts[i]), sizes[i]);
        if (outcome == GetAttemptOutcome::kSuccess) {
          results[i] = true;
        }
      }
      continue;
    }
    const auto& route = routes[i].value();
    if (route.node_id == config_.master_config.node_id) {
      // Self-target fast path: SSD via PeerSsdManager, DRAM/HBM via the allocator.
      GetAttemptOutcome outcome =
          route.tier == TierType::SSD
              ? ExecuteLocalSsdGet(keys[i], const_cast<void*>(dsts[i]), sizes[i])
              : ExecuteLocalGet(keys[i], const_cast<void*>(dsts[i]), sizes[i]);
      if (outcome == GetAttemptOutcome::kSuccess) {
        results[i] = true;
      }
      continue;
    }
    BatchGetItem item{.index = i,
                      .key = &keys[i],
                      .dst = const_cast<void*>(dsts[i]),
                      .size = sizes[i],
                      .route = route};
    if (route.tier == TierType::SSD) {
      ssd_remote_groups[route.node_id].push_back(std::move(item));
    } else {
      remote_groups[route.node_id].push_back(std::move(item));
    }
  }

  for (auto& [node_id, items] : remote_groups) {
    ProcessRemoteBatchGet(items, &results);
  }
  for (auto& [node_id, items] : ssd_remote_groups) {
    ProcessRemoteSsdBatchGet(items, &results);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    if (results[i]) continue;
    results[i] = false;
  }

  const auto call_end = std::chrono::steady_clock::now();
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(call_end - call_start).count();
  if (seconds > 0.0) {
    auto split = ComputeBatchBandwidthBytes(results, sizes, routes, config_.master_config.node_id);
    ObserveBatchBandwidth(*master_client_, split.local, seconds,
                          MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH,
                          MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH_HELP, "local");
    ObserveBatchBandwidth(*master_client_, split.remote, seconds,
                          MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH,
                          MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH_HELP, "remote");
  }
  return results;
}

void PoolClient::ProcessRemoteBatchPut(const std::vector<BatchPutItem>& items,
                                       std::vector<PutEntryOutcome>* results) {
  if (items.empty()) return;
  auto fail_all = [&] {
    for (const auto& item : items) (*results)[item.index] = PutEntryOutcome::kFailed;
  };
  if (!io_engine_) {
    MORI_UMBP_ERROR("[PoolClient] ProcessRemoteBatchPut: io_engine_ not initialized (items={})",
                    items.size());
    fail_all();
    return;
  }
  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    MORI_UMBP_WARN(
        "[PoolClient] ProcessRemoteBatchPut: peer service connection unavailable, node='{}' "
        "addr='{}' items={}",
        first.route.node_id, first.route.peer_address, items.size());
    fail_all();
    return;
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());
  ExecuteRemoteBatchPut(items, results, peer, stub);
}

void PoolClient::ExecuteRemoteBatchPut(const std::vector<BatchPutItem>& items,
                                       std::vector<PutEntryOutcome>* results, PeerConnection& peer,
                                       ::umbp::UMBPPeer::Stub* stub) {
  if (items.empty()) return;
  if (!io_engine_) {
    for (const auto& item : items) (*results)[item.index] = PutEntryOutcome::kFailed;
    return;
  }

  std::vector<RemotePutEntry> entries;
  std::vector<uint64_t> abort_slots;
  if (!AllocateRemotePutEntries(items, stub, &entries, &abort_slots, results)) return;

  std::vector<TransferInstruction> transfers;
  uint64_t staging_bytes = 0;
  if (!BuildRemotePutTransfers(entries, peer, &transfers, &staging_bytes)) {
    MORI_UMBP_WARN(
        "[PoolClient] ExecuteRemoteBatchPut: BuildRemotePutTransfers failed, node='{}' "
        "entries={} staging_bytes={} -> aborting all slots",
        items.front().route.node_id, entries.size(), staging_bytes);
    for (auto& entry : entries) {
      abort_slots.push_back(entry.slot_id);
      (*results)[entry.result_index] = PutEntryOutcome::kFailed;
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
                                          std::vector<PutEntryOutcome>* results) {
  entries->clear();
  ::umbp::BatchAllocateSlotsRequest alloc_req;
  for (const auto& item : items) {
    auto* entry = alloc_req.add_entries();
    entry->set_size(item.size);
    entry->set_tier(static_cast<::umbp::TierType>(item.route.tier));
    entry->set_key(*item.key);
  }

  ::umbp::BatchAllocateSlotsResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  auto alloc_status = stub->BatchAllocateSlots(&alloc_ctx, alloc_req, &alloc_resp);
  if (!alloc_status.ok() || alloc_resp.entries_size() != static_cast<int>(items.size())) {
    MORI_UMBP_WARN("[PoolClient] BatchAllocateSlots failed on {}: {}", items.front().route.node_id,
                   alloc_status.error_message());
    for (const auto& item : items) (*results)[item.index] = PutEntryOutcome::kFailed;
    return false;
  }

  entries->reserve(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& resp_entry = alloc_resp.entries(static_cast<int>(i));
    const auto outcome = resp_entry.outcome();

    switch (outcome) {
      case ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS:
        (*results)[item.index] = PutEntryOutcome::kAlreadyExists;
        continue;
      case ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED:
      case ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE:
        // Peer allocator already logged the specific reason.
        (*results)[item.index] = PutEntryOutcome::kFailed;
        continue;
      case ::umbp::ALLOCATE_SLOT_OUTCOME_UNSPECIFIED:
      default:
        // Unset / unknown — proto version skew or wire corruption.
        // Must NOT fall through into slot processing below.
        MORI_UMBP_ERROR(
            "[PoolClient] BatchAllocateSlots: bad outcome={} ({}) for key='{}' on node='{}'",
            static_cast<int>(outcome), OutcomeName(outcome),
            item.key ? *item.key : std::string{"<null>"}, items.front().route.node_id);
        (*results)[item.index] = PutEntryOutcome::kFailed;
        continue;
      case ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED:
        break;
    }

    PoolClient::SlotPlan plan = FromAllocateSlotResponse(resp_entry);
    if (!SizeMatchesAllocation(item.size, plan.pages.size(), plan.page_size)) {
      MORI_UMBP_ERROR("[PoolClient] BatchPut: malformed slot for key='{}'", *item.key);
      abort_slots->push_back(plan.slot_id);
      (*results)[item.index] = PutEntryOutcome::kFailed;
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
      if (cursor + aligned_size > config_.staging_buffer_size) {
        MORI_UMBP_WARN(
            "[PoolClient] BuildRemotePutTransfers: staging buffer overflow at entry {}/{}, "
            "key='{}' size={} aligned={} cursor={} cap={}",
            idx, entries.size(),
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.item ? entry.item->size : 0, aligned_size, cursor, config_.staging_buffer_size);
        return false;
      }
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
        MORI_UMBP_ERROR(
            "[PoolClient] BuildRemotePutTransfers: invalid peer dram_memories slot, "
            "key='{}' buffer_index={} peer_dram_size={} page_index={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            page.buffer_index, peer.dram_memories.size(), page.page_index);
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
        MORI_UMBP_ERROR(
            "[PoolClient] ExecuteRemotePutTransfers: staging offset overflow (should not happen), "
            "key='{}' staging_offset={} size={} cap={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.staging_offset, entry.item ? entry.item->size : 0, config_.staging_buffer_size);
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
      const auto& tr = active_transfers[i];
      auto& entry = entries[tr.entry_index];
      MORI_UMBP_ERROR(
          "RemotePut BatchWrite failed: code={} msg='{}' peer_engine='{}' key='{}' "
          "slot_id={} size={} local_off={} remote_off={} use_staging={}",
          statuses[i].CodeUint32(), statuses[i].Message(), tr.remote_desc.engineKey,
          (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"}, entry.slot_id,
          tr.size, tr.local_offset, tr.remote_offset, entry.use_staging);
      entry.failed = true;
    }
  }

  if (staging_lock.owns_lock()) staging_lock.unlock();
}

void PoolClient::FinalizeRemotePutEntries(std::vector<RemotePutEntry>& entries,
                                          std::vector<uint64_t>& abort_slots,
                                          std::vector<PutEntryOutcome>* results,
                                          ::umbp::UMBPPeer::Stub* stub) {
  ::umbp::BatchCommitSlotsRequest commit_req;
  std::vector<size_t> commit_indices;
  commit_indices.reserve(entries.size());

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    if (entry.failed) {
      abort_slots.push_back(entry.slot_id);
      (*results)[entry.result_index] = PutEntryOutcome::kFailed;
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
        (*results)[entries[idx].result_index] = PutEntryOutcome::kFailed;
        entries[idx].failed = true;
      }
    } else {
      for (size_t i = 0; i < commit_indices.size(); ++i) {
        auto idx = commit_indices[i];
        auto& entry = entries[idx];
        if (commit_resp.success(static_cast<int>(i))) {
          master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL,
                                     MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL_HELP,
                                     {{"traffic", "remote"}},
                                     static_cast<double>(entry.item->size));
          (*results)[entry.result_index] = PutEntryOutcome::kSucceeded;
        } else {
          // Peer allocator already logged the reason (SLOT_GONE / PRE_CLEAR).
          abort_slots.push_back(entry.slot_id);
          (*results)[entry.result_index] = PutEntryOutcome::kFailed;
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
    MORI_UMBP_ERROR("[PoolClient] ProcessRemoteBatchGet: io_engine_ not initialized (items={})",
                    items.size());
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return;
  }

  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    MORI_UMBP_WARN(
        "[PoolClient] ProcessRemoteBatchGet: peer service connection unavailable, node='{}' "
        "addr='{}' items={}",
        first.route.node_id, first.route.peer_address, items.size());
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
    MORI_UMBP_WARN(
        "[PoolClient] ExecuteRemoteBatchGet: BuildRemoteGetTransfers failed, node='{}' "
        "entries={} staging_bytes={}",
        items.front().route.node_id, entries.size(), staging_bytes);
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
      if (cursor + aligned_size > config_.staging_buffer_size) {
        MORI_UMBP_WARN(
            "[PoolClient] BuildRemoteGetTransfers: staging buffer overflow at entry {}/{}, "
            "key='{}' size={} aligned={} cursor={} cap={}",
            idx, entries.size(),
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.item ? entry.item->size : 0, aligned_size, cursor, config_.staging_buffer_size);
        return false;
      }
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
        MORI_UMBP_ERROR(
            "[PoolClient] BuildRemoteGetTransfers: invalid peer dram_memories slot, "
            "key='{}' buffer_index={} peer_dram_size={} page_index={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            page.buffer_index, peer.dram_memories.size(), page.page_index);
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
      const auto& tr = active_transfers[i];
      auto& entry = entries[tr.entry_index];
      MORI_UMBP_ERROR(
          "RemoteGet BatchRead failed: code={} msg='{}' peer_engine='{}' key='{}' "
          "size={} local_off={} remote_off={} use_staging={}",
          statuses[i].CodeUint32(), statuses[i].Message(), tr.remote_desc.engineKey,
          (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"}, tr.size,
          tr.local_offset, tr.remote_offset, entry.use_staging);
      entry.failed = true;
    }
  }

  if (staging_lock.owns_lock()) {
    for (auto& entry : entries) {
      if (entry.failed || !entry.use_staging) continue;
      if (entry.staging_offset + entry.item->size > config_.staging_buffer_size) {
        MORI_UMBP_ERROR(
            "[PoolClient] ExecuteRemoteGetTransfers: staging offset overflow (should not happen), "
            "key='{}' staging_offset={} size={} cap={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.staging_offset, entry.item ? entry.item->size : 0, config_.staging_buffer_size);
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
    master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
                               MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP,
                               {{"traffic", "remote"}}, static_cast<double>(entry.item->size));
    master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                               MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP,
                               {{"traffic", "remote"}}, static_cast<double>(entry.item->size));
    (*results)[entry.result_index] = true;
  }
}

bool PoolClient::Exists(const std::string& key) {
  auto v = BatchExists({key});
  return !v.empty() && v.front();
}

std::vector<bool> PoolClient::BatchExists(const std::vector<std::string>& keys) {
  if (!initialized_ || keys.empty()) return std::vector<bool>(keys.size(), false);

  std::vector<bool> out;
  auto status = master_client_->BatchLookup(keys, &out);
  if (!status.ok() || out.size() != keys.size()) return std::vector<bool>(keys.size(), false);
  return out;
}

// ---------------------------------------------------------------------------
//  External KV
// ---------------------------------------------------------------------------

bool PoolClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  if (hashes.empty()) return true;
  return master_client_->ReportExternalKvBlocks(config_.master_config.node_id, hashes, tier).ok();
}

bool PoolClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  if (hashes.empty()) return true;
  return master_client_->RevokeExternalKvBlocks(config_.master_config.node_id, hashes, tier).ok();
}

bool PoolClient::RevokeAllExternalKvBlocksAtTier(TierType tier) {
  if (!initialized_) return false;
  return master_client_->RevokeAllExternalKvBlocksAtTier(config_.master_config.node_id, tier).ok();
}

bool PoolClient::MatchExternalKv(const std::vector<std::string>& hashes,
                                 std::vector<MasterClient::ExternalKvNodeMatch>* out_matches,
                                 bool count_as_hit) {
  if (!initialized_) return false;
  return master_client_->MatchExternalKv(hashes, out_matches, count_as_hit).ok();
}

bool PoolClient::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes,
    std::vector<MasterClient::ExternalKvHitCountEntry>* out_entries) {
  if (!initialized_) return false;
  return master_client_->GetExternalKvHitCounts(hashes, out_entries).ok();
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

// Remote SSD get for one key (reader != owner): key-based PrepareSsdRead on the
// peer reads the bytes into its serving staging slot; we RDMA them out of the
// published staging buffer, then issue a best-effort ReleaseSsdLease.  Outcomes:
// kMiss (NOT_FOUND, definitive miss); kRetry (retryable: NO_SLOT or a
// reader-local lease expiry); kError (not-served, not retried: rpc failure
// incl. DEADLINE_EXCEEDED, size mismatch, RDMA failure).
//
// Lease gating: the deadline is anchored at t_send (before the RPC) so it stays
// conservative against the peer, which counts the same TTL from request receipt
// (see ssd_read_lease.h).  A read is reported successful only if RDMA finished
// before that deadline; once expired we return a transient retry and do NOT
// release (the peer reclaims the slot by TTL).  A late-returning PrepareSsdRead
// is caught by the pre-RDMA expiry check; a hung one by the RPC deadline.
PoolClient::SsdGetOutcome PoolClient::RemoteSsdReadOnce(PeerConnection& peer,
                                                        const std::string& key, void* dst,
                                                        size_t size) {
  namespace lease = ssd_read_lease;
  if (!io_engine_) return SsdGetOutcome::kError;
  if (!EnsurePeerServiceConnection(peer)) return SsdGetOutcome::kError;
  if (!IsValidMemoryDesc(peer.ssd_staging_mem)) return SsdGetOutcome::kError;
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  // Anchor the lease deadline before sending; also bound the RPC itself so a
  // hung peer can't stall the serial batch.  Fall back to the configured lease
  // timeout when the dedicated timeout env is unset (cluster-homogeneous).
  const auto t_send = std::chrono::steady_clock::now();
  auto rpc_timeout = SsdPrepareRpcTimeoutOverride();
  if (rpc_timeout.count() == 0) {
    rpc_timeout = std::chrono::seconds(std::max(config_.ssd_lease_timeout_s, 1));
  }

  ::umbp::PrepareSsdReadRequest req;
  req.set_key(key);
  req.set_max_size(size);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;
  // gRPC deadlines are specialized for system_clock; the lease gating below
  // uses the monotonic steady_clock t_send for the actual validity window.
  ctx.set_deadline(std::chrono::system_clock::now() + rpc_timeout);
  const grpc::Status rpc = stub->PrepareSsdRead(&ctx, req, &resp);
  if (!rpc.ok()) {
    // Includes DEADLINE_EXCEEDED (peer slow / may already hold a claimed slot)
    // and UNAVAILABLE: hard failure, not retried — see SsdGetTransientMaxAttempts.
    MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead key='{}' PrepareSsdRead rpc failed (code={})", key,
                    static_cast<int>(rpc.error_code()));
    return SsdGetOutcome::kError;
  }

  switch (resp.status()) {
    case ::umbp::SSD_READ_OK:
      break;
    case ::umbp::SSD_READ_NOT_FOUND:
      return SsdGetOutcome::kMiss;
    case ::umbp::SSD_READ_NO_SLOT:
      // Transient slot exhaustion (no slot was claimed): retryable, not a miss.
      return SsdGetOutcome::kRetry;
    default:  // SIZE_TOO_LARGE / ERROR / unexpected
      MORI_UMBP_WARN("[PoolClient] RemoteSsdRead key='{}' status={}", key,
                     static_cast<int>(resp.status()));
      return SsdGetOutcome::kError;
  }

  const auto deadline_ttl = resp.lease_ttl_ms();

  // If the lease already elapsed (slow/late PrepareSsdRead return), don't even
  // RDMA — the staging bytes may already be getting recycled.  Transient, not a
  // miss; leave the slot for the peer's TTL reclaim (no release).
  if (lease::LeaseExpired(t_send, deadline_ttl, std::chrono::steady_clock::now())) {
    MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead key='{}' lease expired before RDMA (ttl_ms={})",
                    key, deadline_ttl);
    return SsdGetOutcome::kRetry;
  }

  if (resp.size() != size) {
    MORI_UMBP_WARN("[PoolClient] RemoteSsdRead key='{}' size mismatch (wanted {}, got {})", key,
                   size, resp.size());
    ReleaseSsdLeaseBestEffort(stub, resp.lease_id());
    return SsdGetOutcome::kError;
  }

  // RDMA the staged bytes home: zero-copy if the dst is registered, else stage
  // through our own buffer and memcpy.
  bool rdma_ok = false;
  auto reg = FindRegisteredMemory(dst, size);
  if (reg) {
    auto uid = io_engine_->AllocateTransferUniqueId();
    mori::io::TransferStatus status;
    io_engine_->Read(reg->first, reg->second, peer.ssd_staging_mem, resp.staging_offset(), size,
                     &status, uid);
    status.Wait();
    rdma_ok = status.Succeeded();
  }
  if (!rdma_ok && size <= config_.staging_buffer_size) {
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

  // Decide against the lease deadline: a read that finished after the deadline
  // is untrusted (the peer may have recycled the slot mid-RDMA), so it becomes
  // a transient retry and we skip release.  The caller uses the return value;
  // any bytes already written to dst on the expired path are not consumed.
  const bool expired = lease::LeaseExpired(t_send, deadline_ttl, std::chrono::steady_clock::now());
  const auto decision = lease::DecideSsdReadOutcome(expired, rdma_ok);
  if (decision.release) ReleaseSsdLeaseBestEffort(stub, resp.lease_id());
  if (expired && rdma_ok) {
    MORI_UMBP_DEBUG("[PoolClient] RemoteSsdRead key='{}' lease expired after RDMA (ttl_ms={})", key,
                    deadline_ttl);
  }
  switch (decision.outcome) {
    case lease::GateOutcome::kSuccess:
      return SsdGetOutcome::kSuccess;
    case lease::GateOutcome::kRetry:
      return SsdGetOutcome::kRetry;
    case lease::GateOutcome::kError:
      return SsdGetOutcome::kError;
  }
  return SsdGetOutcome::kError;
}

// Best-effort lease release.  Correctness does not depend on it: if it fails or
// is never called, the peer reclaims the slot when the lease TTL expires.  Each
// attempt is bounded by a short deadline so a slow peer can't stall the caller.
void PoolClient::ReleaseSsdLeaseBestEffort(::umbp::UMBPPeer::Stub* stub, uint64_t lease_id) {
  if (lease_id == 0) return;
  const uint32_t max_retries = ReleaseLeaseMaxRetries();
  const auto timeout = ReleaseLeaseRpcTimeout();
  for (uint32_t attempt = 0; attempt < max_retries; ++attempt) {
    ::umbp::ReleaseSsdLeaseRequest rel_req;
    rel_req.set_lease_id(lease_id);
    ::umbp::ReleaseSsdLeaseResponse rel_resp;
    grpc::ClientContext rel_ctx;
    rel_ctx.set_deadline(std::chrono::system_clock::now() + timeout);
    if (stub->ReleaseSsdLease(&rel_ctx, rel_req, &rel_resp).ok()) break;
  }
}

void PoolClient::ProcessRemoteSsdBatchGet(const std::vector<BatchGetItem>& items,
                                          std::vector<bool>* results) {
  if (items.empty()) return;
  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);

  const uint32_t max_attempts = SsdGetTransientMaxAttempts();
  uint64_t transient_not_served = 0;  // aggregated per batch → one AddCounter below
  for (const auto& item : items) {
    bool done = false;
    for (uint32_t attempt = 0; attempt < max_attempts && !done; ++attempt) {
      SsdGetOutcome outcome = RemoteSsdReadOnce(peer, *item.key, item.dst, item.size);
      switch (outcome) {
        case SsdGetOutcome::kSuccess:
          (*results)[item.index] = true;
          done = true;
          break;
        case SsdGetOutcome::kMiss:   // definitive miss
        case SsdGetOutcome::kError:  // hard failure (already logged)
          done = true;
          break;
        case SsdGetOutcome::kRetry:  // transient NO_SLOT / reader-local lease expiry; not a miss
          if (attempt + 1 < max_attempts) std::this_thread::sleep_for(SsdGetRetryBackoff());
          break;
      }
    }
    if (!done) {
      ++transient_not_served;
      MORI_UMBP_WARN(
          "[PoolClient] Remote SSD get key='{}' still transient-failing (NO_SLOT/lease-expired) "
          "after {} attempts; reporting as not-served this round (not a definitive miss)",
          *item.key, max_attempts);
    }
  }
  // Optional reader-side diagnostic: lease expiry is reader-local and never
  // shows up in the peer's ssd_read_total, so surface transient not-served here.
  // Aggregated to a single AddCounter per batch (cheap, no per-key lock churn).
  if (transient_not_served > 0 && master_client_) {
    master_client_->AddCounter(MORI_UMBP_METRIC_SSD_READ_CLIENT_TRANSIENT_TOTAL,
                               MORI_UMBP_METRIC_SSD_READ_CLIENT_TRANSIENT_TOTAL_HELP, {},
                               static_cast<double>(transient_not_served));
  }
}

void PoolClient::PublishSsdMetrics() {
  if (!master_client_) return;

  // Ship a monotonic peer-side counter as the delta since the last tick.  `last`
  // is updated even on a zero delta (or a defensive counter reset) so the next
  // tick stays correct.  Runs once per metrics flush in the metrics thread.
  auto ship_counter = [&](const char* name, const char* help, MasterClient::Labels labels,
                          uint64_t current, uint64_t& last) {
    if (current > last) {
      master_client_->AddCounter(name, help, std::move(labels),
                                 static_cast<double>(current - last));
    }
    last = current;
  };

  if (ssd_copy_pipeline_) {
    ship_counter(MORI_UMBP_METRIC_SSD_COPY_ENQUEUED_TOTAL,
                 MORI_UMBP_METRIC_SSD_COPY_ENQUEUED_TOTAL_HELP, {}, ssd_copy_pipeline_->Enqueued(),
                 ssd_metrics_last_.copy_enqueued);
    ship_counter(MORI_UMBP_METRIC_SSD_COPY_SUCCEEDED_TOTAL,
                 MORI_UMBP_METRIC_SSD_COPY_SUCCEEDED_TOTAL_HELP, {}, ssd_copy_pipeline_->CopiedOk(),
                 ssd_metrics_last_.copy_succeeded);
    ship_counter(MORI_UMBP_METRIC_SSD_COPY_FAILED_TOTAL,
                 MORI_UMBP_METRIC_SSD_COPY_FAILED_TOTAL_HELP, {}, ssd_copy_pipeline_->Failed(),
                 ssd_metrics_last_.copy_failed);
    ship_counter(MORI_UMBP_METRIC_SSD_COPY_DROPPED_TOTAL,
                 MORI_UMBP_METRIC_SSD_COPY_DROPPED_TOTAL_HELP, {{"reason", "queue_full"}},
                 ssd_copy_pipeline_->Dropped(), ssd_metrics_last_.copy_dropped_queue_full);
    ship_counter(MORI_UMBP_METRIC_SSD_COPY_DROPPED_TOTAL,
                 MORI_UMBP_METRIC_SSD_COPY_DROPPED_TOTAL_HELP, {{"reason", "stopped"}},
                 ssd_copy_pipeline_->DroppedStopped(), ssd_metrics_last_.copy_dropped_stopped);
  }

  if (peer_ssd_) {
    ship_counter(MORI_UMBP_METRIC_SSD_READ_TOTAL, MORI_UMBP_METRIC_SSD_READ_TOTAL_HELP,
                 {{"status", "ok"}}, peer_ssd_->ReadOk(), ssd_metrics_last_.read_ok);
    ship_counter(MORI_UMBP_METRIC_SSD_READ_TOTAL, MORI_UMBP_METRIC_SSD_READ_TOTAL_HELP,
                 {{"status", "not_found"}}, peer_ssd_->ReadNotFound(),
                 ssd_metrics_last_.read_not_found);
    ship_counter(MORI_UMBP_METRIC_SSD_READ_TOTAL, MORI_UMBP_METRIC_SSD_READ_TOTAL_HELP,
                 {{"status", "size_too_large"}}, peer_ssd_->ReadSizeTooLarge(),
                 ssd_metrics_last_.read_size_too_large);
    ship_counter(MORI_UMBP_METRIC_SSD_READ_TOTAL, MORI_UMBP_METRIC_SSD_READ_TOTAL_HELP,
                 {{"status", "error"}}, peer_ssd_->ReadError(), ssd_metrics_last_.read_error);

    // SSD IO byte counters -> bandwidth via rate() in Grafana.
    ship_counter(MORI_UMBP_METRIC_SSD_COPY_BYTES_TOTAL, MORI_UMBP_METRIC_SSD_COPY_BYTES_TOTAL_HELP,
                 {}, peer_ssd_->CopyBytes(), ssd_metrics_last_.copy_bytes);
    ship_counter(MORI_UMBP_METRIC_SSD_READ_BYTES_TOTAL, MORI_UMBP_METRIC_SSD_READ_BYTES_TOTAL_HELP,
                 {}, peer_ssd_->ReadBytes(), ssd_metrics_last_.read_bytes);

    ship_counter(MORI_UMBP_METRIC_SSD_EVICTION_ROUNDS_TOTAL,
                 MORI_UMBP_METRIC_SSD_EVICTION_ROUNDS_TOTAL_HELP, {}, peer_ssd_->EvictionRounds(),
                 ssd_metrics_last_.evict_rounds);
    ship_counter(MORI_UMBP_METRIC_SSD_EVICTION_VICTIMS_TOTAL,
                 MORI_UMBP_METRIC_SSD_EVICTION_VICTIMS_TOTAL_HELP, {}, peer_ssd_->EvictionVictims(),
                 ssd_metrics_last_.evict_victims);
    ship_counter(MORI_UMBP_METRIC_SSD_EVICTION_BYTES_FREED_TOTAL,
                 MORI_UMBP_METRIC_SSD_EVICTION_BYTES_FREED_TOTAL_HELP, {},
                 peer_ssd_->EvictionBytesFreed(), ssd_metrics_last_.evict_bytes_freed);
    ship_counter(MORI_UMBP_METRIC_SSD_EVICTION_BACKEND_FAILED_TOTAL,
                 MORI_UMBP_METRIC_SSD_EVICTION_BACKEND_FAILED_TOTAL_HELP, {},
                 peer_ssd_->EvictionBackendFailures(), ssd_metrics_last_.evict_backend_failed);
  }

  if (peer_service_) {
    const auto& m = peer_service_->Metrics();
    const uint64_t expired = m.expired_reclaims.load(std::memory_order_relaxed);
    const uint64_t slot_full = m.slot_full_rejects.load(std::memory_order_relaxed);
    ship_counter(MORI_UMBP_METRIC_SSD_STAGING_EXPIRED_RECLAIMS_TOTAL,
                 MORI_UMBP_METRIC_SSD_STAGING_EXPIRED_RECLAIMS_TOTAL_HELP, {}, expired,
                 ssd_metrics_last_.staging_expired_reclaims);
    ship_counter(MORI_UMBP_METRIC_SSD_STAGING_SLOT_FULL_REJECTS_TOTAL,
                 MORI_UMBP_METRIC_SSD_STAGING_SLOT_FULL_REJECTS_TOTAL_HELP, {}, slot_full,
                 ssd_metrics_last_.staging_slot_full_rejects);
    // A NO_SLOT read outcome IS a slot-full reject; surface it under the unified
    // read_total{status=no_slot} too (same event, peer-service view of reads).
    ship_counter(MORI_UMBP_METRIC_SSD_READ_TOTAL, MORI_UMBP_METRIC_SSD_READ_TOTAL_HELP,
                 {{"status", "no_slot"}}, slot_full, ssd_metrics_last_.read_no_slot);
    master_client_->SetGauge(MORI_UMBP_METRIC_SSD_STAGING_SLOTS_IN_USE,
                             MORI_UMBP_METRIC_SSD_STAGING_SLOTS_IN_USE_HELP, {},
                             static_cast<double>(peer_service_->SnapshotReadSlotsInUse()));
  }
}

}  // namespace mori::umbp

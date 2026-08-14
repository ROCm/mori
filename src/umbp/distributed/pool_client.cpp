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
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <msgpack.hpp>
#include <new>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/common/parallel_for.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/peer/backend/hbm_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/peer/backend/ssd_backend.h"
#include "umbp/distributed/peer/batch_resolve_codec.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {

// ---------------------------------------------------------------------------
//  Bandwidth metrics
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
//  Page / size math
// ---------------------------------------------------------------------------

// --- Cross-key parallelism for the self-target (local) paths. --------------
// In distributed mode 1 key == 1 page (master page_size == KV block size), so
// a self-target Put/Get copies one ~MiB-scale block per key.  The parallelism
// that pays is therefore ACROSS the many keys of one BatchPut/BatchGet
// (different threads -> different keys), not within one key's pages.  The
// per-key copy itself lives in LocalCopyEngine since Phase 6.  Threads via
// UMBP_DRAM_{READ,WRITE}_THREADS (same envs as local mode's DRAMTier).
inline int LocalCopyThreads(const char* env_name) {
  int t = 4;
  if (const char* e = std::getenv(env_name)) {
    int x = std::atoi(e);
    if (x >= 1) t = x;
  }
  unsigned hc = std::thread::hardware_concurrency();
  if (hc > 0 && t > static_cast<int>(hc)) t = static_cast<int>(hc);
  if (t < 1) t = 1;
  return t;
}

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

// ---------------------------------------------------------------------------
//  Lease env knobs
// ---------------------------------------------------------------------------

// Peer-side DRAM/HBM read lease: how long a single Resolve protects its key's
// pages from concurrent local Evict, covering one RDMA read.  Only needs to
// exceed one DRAM RDMA round trip (sub-ms), so 500 ms is already ~100x margin;
// exposed so operators can tighten it under eviction pressure.
std::chrono::milliseconds DramReadLeaseTtl() {
  static const auto v = GetEnvMilliseconds("UMBP_DRAM_READ_LEASE_MS",
                                           std::chrono::milliseconds(500), /*min_allowed=*/1);
  return v;
}

// ---------------------------------------------------------------------------
//  Config / proto translation
// ---------------------------------------------------------------------------

// Translate a peer-side ::umbp::AllocateSlotResponse / ResolveKeyResponse
// into the C++ shapes our code consumes.
PoolClient::SlotPlan FromAllocateSlotResponse(const ::umbp::AllocateSlotResponse& resp) {
  PoolClient::SlotPlan p;
  p.slot_id = resp.slot_id();
  p.page_size = resp.page_size();
  p.backend_id = resp.backend_id();
  p.pages.reserve(resp.pages_size());
  for (const auto& pp : resp.pages()) p.pages.push_back({pp.buffer_index(), pp.page_index()});
  p.descs.reserve(resp.descs_size());
  for (const auto& d : resp.descs()) {
    BufferMemoryDescBytes b;
    b.buffer_index = d.buffer_index();
    b.backend_id = d.backend_id();
    b.desc_bytes.assign(d.desc().begin(), d.desc().end());
    p.descs.push_back(std::move(b));
  }
  return p;
}

// ---------------------------------------------------------------------------
//  Engine outcome -> per-entry failure
//
//  A TransferItem's `tag` IS the index of the entry that produced it, so
//  mapping an engine failure back to keys is an index lookup.  A failed plan
//  fails EVERY key that contributed a segment to it (per-item AND) — the same
//  granularity the pre-refactor per-(localMR, remoteMR) group had, for the same
//  reason: the wire reports success per transfer, not per key.
//
//  Free templates rather than PoolClient members so both entry types share one
//  definition; deduction means neither ever has to be named here.
// ---------------------------------------------------------------------------

template <typename Entry>
void ApplyTransferFailures(std::vector<Entry>& entries,
                           const std::vector<TransferFailure>& failures, const char* what) {
  for (const auto& f : failures) {
    for (size_t tag : f.tags) {
      if (tag >= entries.size()) continue;
      auto& entry = entries[tag];
      MORI_UMBP_ERROR("{} transfer failed: code={} msg='{}' peer_engine='{}' key='{}'", what,
                      f.code, f.message, f.endpoint,
                      (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"});
      entry.failed = true;
    }
  }
}

// Items no engine could carry.  Distinct from a failed transfer: these never
// reached the wire, which usually means a peer descriptor was missing or the
// item was larger than the engine's bounce pool.
template <typename Entry>
void ApplyRejectedTags(std::vector<Entry>& entries, const std::vector<size_t>& tags,
                       const char* what) {
  for (size_t tag : tags) {
    if (tag >= entries.size()) continue;
    auto& entry = entries[tag];
    MORI_UMBP_WARN("{} transfer unplannable (no engine claimed the endpoints), key='{}'", what,
                   (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"});
    entry.failed = true;
  }
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

  // The one byte-moving path (design doc §4).  Order is preference order:
  // a pair both of whose endpoints are host-addressable never reaches the wire.
  //
  // This is the composition root, and the only place any concrete engine is
  // named — the same rule Phase 5 applies to backends.  Everything downstream
  // holds TransferEngine / MemoryRegistrar / PeerDirectory.
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  // Both-local pairs with a GPU endpoint.  Disjoint from LocalCopyEngine (which
  // requires both sides CPU) and from MoriIoEngine (which requires exactly one
  // side remote), so this is registration order as documentation, not as a
  // tie-break — but it must come before the wire engine on principle, since an
  // HBM pair that reached mori-io would be refused rather than served slowly.
  composite->AddEngine(std::make_unique<HbmCopyEngine>());
  if (!config_.io_engine.host.empty()) {
    mori::io::IOEngineConfig io_cfg;
    io_cfg.host = config_.io_engine.host;
    io_cfg.port = config_.io_engine.port;
    auto rdma = std::make_unique<MoriIoEngine>(config_.master_config.node_id, io_cfg,
                                               config_.staging_buffer_size);
    if (!rdma->Init()) {
      MORI_UMBP_ERROR("[PoolClient] MoriIoEngine init failed on {}:{}", config_.io_engine.host,
                      config_.io_engine.port);
      initialized_ = false;
      return false;
    }
    peer_directory_ = rdma.get();
    composite->AddEngine(std::move(rdma));
  }
  transfer_engine_ = std::move(composite);

  // Peer-side backend: ONE medium, chosen by config_.medium.
  //
  // This is the ONLY place in the tree that decides which medium is live;
  // everything downstream dispatches through registry_ (Phase 3) and never
  // names a tier.  It does not name a concrete backend TYPE either — each
  // factory hands back a MediumBackend, which is what closes Phase 5 Rule A.
  //
  // WHY ONE.  The registry holds a map keyed by tier and would happily take
  // all three, but UMBP's routing plane does not tier: master treats every
  // advertised tier as an equally valid put target (Phase 4 deleted the
  // hardcoded tier orders), so a node registering DRAM *and* SSD does not get
  // "DRAM in front of SSD" — it gets two independent pools that master picks
  // between by free capacity, i.e. mirroring, not promotion.  Local tiering is
  // a policy nobody has asked for; heterogeneity comes from different NODES
  // picking different media, which the routing plane already handles.  See
  // UMBPMedium in common/config.h.
  const uint64_t page_size =
      config_.dram_page_size > 0 ? config_.dram_page_size : 2ULL * 1024 * 1024;

  std::unique_ptr<MediumBackend> backend;
  switch (config_.medium) {
    case TierType::DRAM: {
      // Self-allocated at Init from sizing knobs only — PoolClient holds no
      // buffer pointer (design doc §1 item 4 / Phase 2b).
      PageBackend::OwnershipConfig dram_ownership;
      dram_ownership.buffer_sizes = config_.dram.buffer_sizes;
      dram_ownership.use_hugepages = config_.dram.use_hugepages;
      dram_ownership.hugepage_size = config_.dram.hugepage_size;
      dram_ownership.numa_node = config_.dram.numa_node;
      dram_ownership.prefault = config_.dram.prefault;
      backend = MakePageBackend(TierType::DRAM, page_size, std::move(dram_ownership),
                                /*pending_ttl=*/std::chrono::milliseconds{30000},
                                /*read_lease_ttl=*/DramReadLeaseTtl());
      break;
    }
    case TierType::HBM: {
      backend = MakeHbmBackend(page_size, config_.hbm.device, config_.hbm.buffer_sizes,
                               /*pending_ttl=*/std::chrono::milliseconds{30000},
                               /*read_lease_ttl=*/DramReadLeaseTtl());
      break;
    }
    case TierType::SSD: {
      // The one medium that is genuinely NOT like the others: its bytes are not
      // addressable, so it publishes a registered host staging arena and spills
      // behind it (see ssd_backend.h).  Everything outside that class — routing,
      // peer service, heartbeat, the batch executors — sees ordinary pages.
      SsdBackend::Config ssd_cfg;
      ssd_cfg.page_size = page_size;
      ssd_cfg.staging_pages = config_.ssd_staging_buffer_slots > 0
                                  ? static_cast<uint32_t>(config_.ssd_staging_buffer_slots)
                                  : 16;
      ssd_cfg.staging_use_hugepages = config_.ssd_staging_use_hugepages;
      ssd_cfg.staging_hugepage_size = config_.ssd_staging_hugepage_size;
      ssd_cfg.ssd = config_.ssd;
      ssd_cfg.ssd.enabled = true;  // selecting SSD IS the opt-in
      ssd_cfg.read_lease_ttl = DramReadLeaseTtl();
      backend = MakeSsdBackend(std::move(ssd_cfg));
      break;
    }
    case TierType::UNKNOWN:
      break;  // falls into the null check below
  }
  if (backend == nullptr) {
    MORI_UMBP_ERROR("[PoolClient] unknown medium {}", static_cast<int>(config_.medium));
    initialized_ = false;
    return false;
  }

  // Narrowed to MemoryRegistrar: a backend publishes endpoints, it does not
  // move bytes, and that is now a compile-time fact (design doc §5 Rule C).
  medium_ = backend->Tier();
  if (!backend->Init(static_cast<MemoryRegistrar*>(transfer_engine_.get()))) {
    MORI_UMBP_ERROR("[PoolClient] {} backend Init failed", TierTypeName(medium_));
    initialized_ = false;
    return false;
  }
  if (!registry_.Register(std::move(backend))) {
    initialized_ = false;
    return false;
  }

  master_client_->SetBackendRegistry(&registry_);

  // Medium-specific counters (SSD read outcomes, single-flight coalescing,
  // eviction, staging pressure) ride the existing metrics tick.  Backend-
  // agnostic by construction: PoolClient forwards whatever each backend names
  // and never learns which medium produced it.
  master_client_->AddMetricsProvider([this] { PublishBackendCounters(); });

  // Pack engine_desc for master registration.
  std::vector<uint8_t> engine_desc_bytes;
  if (peer_directory_ != nullptr) engine_desc_bytes = peer_directory_->PackedLocalEngineDesc();

  if (config_.peer_service_port > 0) {
    peer_service_ =
        std::make_unique<PeerServiceServer>(&registry_, engine_desc_bytes, master_client_.get());
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

  // Master register.  In the new design master holds no DRAM-side
  // metadata; only membership + capacity-snapshot.  Capacity is aggregated
  // over every registered backend instead of a PoolClientConfig literal
  // (design doc §1 item 4 / Phase 2b).  A backend reporting zero capacity is
  // omitted rather than advertised as a full tier: since Phase 4 the router
  // treats every advertised tier as a valid put target, so advertising an
  // empty one would invite placements it can never accept.
  std::map<TierType, TierCapacity> tier_caps;
  for (auto* backend : registry_.All()) {
    auto cap = backend->Capacity();
    if (cap.total_bytes > 0) tier_caps[backend->Tier()] = cap;
  }
  auto status = master_client_->RegisterSelf(tier_caps, peer_address, engine_desc_bytes);
  if (!status.ok()) {
    MORI_UMBP_ERROR("[PoolClient] RegisterSelf failed: {}", status.error_message());
    initialized_ = false;
    return false;
  }

  if (config_.master_config.auto_heartbeat) master_client_->StartHeartbeat();

  // Start the async re-cache worker only when the feature is on and this node
  // has an exportable local medium to install into.
  if (config_.cache_remote_fetches && registry_.Get(medium_) != nullptr) {
    {
      std::lock_guard<std::mutex> lk(recache_mutex_);
      recache_stop_ = false;
    }
    recache_worker_ = std::thread([this] { ReCacheWorkerLoop(); });
  }

  MORI_UMBP_INFO("[PoolClient] Initialized node_id='{}'", config_.master_config.node_id);
  return true;
}

void PoolClient::Shutdown() {
  if (!initialized_) return;
  initialized_ = false;

  // Stop the async re-cache worker first: it calls ExecuteLocalPut (which uses
  // the registry + master_client_), so it must be joined before those are torn
  // down below.
  {
    std::lock_guard<std::mutex> lk(recache_mutex_);
    recache_stop_ = true;
    recache_queue_.clear();
  }
  recache_cv_.notify_all();
  if (recache_worker_.joinable()) recache_worker_.join();

  if (master_client_) {
    master_client_->StopHeartbeat();
    // Idempotent with ~MasterClient.
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

  // Every backend deregisters its memory through transfer_engine_ inside its
  // own destructor — this MUST run before transfer_engine_ is torn down below.
  // MasterClient borrows the registry, so unbind it first.
  if (master_client_) master_client_->SetBackendRegistry(nullptr);
  registry_ = BackendRegistry{};

  if (transfer_engine_) {
    std::lock_guard<std::mutex> lock(registered_mem_mutex_);
    for (auto& reg : registered_regions_) transfer_engine_->Deregister(reg.ref);
    registered_regions_.clear();
  }
  peer_directory_ = nullptr;
  transfer_engine_.reset();

  master_client_.reset();
}

bool PoolClient::Clear() {
  // Vacuously done: nothing has been initialized so there is no state to
  // clear and no master to notify.  Treat as success so callers in
  // shutdown / teardown paths do not surface a spurious error.
  if (!initialized_.load()) return true;
  // Clear every medium, not just DRAM: a key mirrored across backends must
  // disappear from all of them before the full-sync empty snapshot goes out.
  for (auto* backend : registry_.All()) backend->ClearLocal();

  bool ok = true;
  if (master_client_) {
    ok = master_client_->ClearFullSync();
    if (!ok) MORI_UMBP_WARN("[PoolClient] Clear full-sync heartbeat failed");
  }
  return ok;
}

bool PoolClient::IsInitialized() const { return initialized_; }
MasterClient& PoolClient::Master() { return *master_client_; }
BackendRegistry& PoolClient::Backends() { return registry_; }

// ---------------------------------------------------------------------------
//  Memory registration
// ---------------------------------------------------------------------------

bool PoolClient::RegisterMemory(void* ptr, size_t size, mori::io::MemoryLocationType loc,
                                int device) {
  if (!transfer_engine_) {
    MORI_UMBP_ERROR("[PoolClient] RegisterMemory: transfer engine not available");
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
  registered_regions_.push_back(
      {ptr, size, transfer_engine_->RegisterMemory(ptr, size, loc, device)});
  return true;
}

void PoolClient::DeregisterMemory(void* ptr) {
  if (ptr == nullptr) return;
  std::lock_guard<std::mutex> lock(registered_mem_mutex_);
  auto it = std::find_if(registered_regions_.begin(), registered_regions_.end(),
                         [ptr](const RegisteredRegion& r) { return r.base == ptr; });
  if (it != registered_regions_.end()) {
    if (transfer_engine_) transfer_engine_->Deregister(it->ref);
    registered_regions_.erase(it);
  }
}

std::optional<std::pair<TransferRef, size_t>> PoolClient::FindRegisteredMemory(const void* ptr,
                                                                               size_t size) const {
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(registered_mem_mutex_);
  for (auto& reg : registered_regions_) {
    auto base = reinterpret_cast<uintptr_t>(reg.base);
    if (addr >= base && size <= reg.size && (addr - base) <= reg.size - size) {
      return std::pair{reg.ref, static_cast<size_t>(addr - base)};
    }
  }
  return std::nullopt;
}

std::pair<TransferRef, uint64_t> PoolClient::UserBufferRef(void* ptr, size_t size) const {
  auto reg = FindRegisteredMemory(ptr, size);
  if (reg.has_value()) return {reg->first, reg->second};
  return {TransferRef::HostBytes(ptr, size), 0};
}

// ---------------------------------------------------------------------------
//  Self-target paths
//
//  Not a "fast path" any more, and that is the headline of Phase 6: a local
//  access is just a transfer whose endpoints are both local, planned by the
//  same engine as everything else.  What used to make it special — a raw base
//  pointer per DRAM buffer, held by PoolClient, memcpy'd through directly — is
//  what made the local path host-DRAM-only and what forced PoolClient to name a
//  concrete backend type to obtain those pointers.
// ---------------------------------------------------------------------------

namespace {

// Offset of a page within its buffer, as a plain byte offset.
inline uint64_t PageOffset(const PageLocation& page, uint64_t page_size) {
  return static_cast<uint64_t>(page.page_index) * page_size;
}

}  // namespace

// Build one TransferItem per page between a caller buffer and a backend's own
// buffers.  `to_backend` is Put (user -> pages); false is Get (pages -> user).
// Returns false when the backend publishes no endpoint for a referenced buffer,
// which means this medium cannot serve the access in-process.
bool PoolClient::BuildLocalPageTransfers(MediumBackend* backend,
                                         const std::vector<PageLocation>& pages, uint64_t page_size,
                                         void* user, size_t size, bool to_backend,
                                         std::vector<TransferItem>* items) {
  if (pages.empty() || page_size == 0) return false;
  const TransferRef user_ref = TransferRef::HostBytes(user, size);
  items->reserve(pages.size());
  for (size_t i = 0; i < pages.size(); ++i) {
    TransferRef buf = backend->BufferRef(pages[i].buffer_index);
    if (!buf.HasHostPtr()) {
      MORI_UMBP_WARN("[PoolClient] local transfer: tier={} publishes no endpoint for buffer {}",
                     static_cast<int>(backend->Tier()), pages[i].buffer_index);
      return false;
    }
    TransferItem item;
    item.size = LogicalPageBytes(i, pages.size(), page_size, size);
    item.tag = i;
    if (to_backend) {
      item.src = user_ref;
      item.src_offset = i * page_size;
      item.dst = std::move(buf);
      item.dst_offset = PageOffset(pages[i], page_size);
    } else {
      item.src = std::move(buf);
      item.src_offset = PageOffset(pages[i], page_size);
      item.dst = user_ref;
      item.dst_offset = i * page_size;
    }
    items->push_back(std::move(item));
  }
  return true;
}

PoolClient::PutAttemptOutcome PoolClient::ExecuteLocalPut(const std::string& key, const void* src,
                                                          size_t size, TierType tier) {
  if (registry_.Empty()) {
    MORI_UMBP_ERROR("[PoolClient] Local Put requested but no storage backend is registered");
    return PutAttemptOutcome::kFatal;
  }
  // Dispatch on the routed tier (Phase 3) rather than assuming DRAM.  A tier
  // with no live backend here is not fatal: kRetry routes the key elsewhere,
  // the same way the pre-refactor allocator rejected an unconfigured tier.
  auto* backend = registry_.Get(tier);
  // A medium that publishes no buffer endpoints cannot be reached in-process at
  // all; route the key elsewhere rather than allocating a slot we cannot fill.
  if (backend == nullptr || backend->BufferCount() == 0) {
    MORI_UMBP_WARN("[PoolClient] Local Put: tier={} has no in-process-addressable backend",
                   static_cast<int>(tier));
    return PutAttemptOutcome::kRetry;
  }
  auto alloc_res = backend->BatchAllocate({AllocateRequest{key, size}}).front();
  switch (alloc_res.outcome) {
    case AllocateOutcome::kSuccessAlreadyExists:
      return PutAttemptOutcome::kSuccessAlreadyExists;
    case AllocateOutcome::kFailed:
    case AllocateOutcome::kFailedNoSpace:
      // Backend already logged the specific reason.
      return PutAttemptOutcome::kRetry;
    case AllocateOutcome::kSuccessAllocated:
      break;
  }
  std::vector<TransferItem> items;
  if (!BuildLocalPageTransfers(backend, alloc_res.pages, alloc_res.page_size,
                               const_cast<void*>(src), size, /*to_backend=*/true, &items) ||
      !transfer_engine_->Transfer(items, /*failed_tags=*/nullptr)) {
    backend->BatchAbort({alloc_res.slot_id});
    return PutAttemptOutcome::kFatal;
  }
  if (!backend->BatchCommit({CommitRequest{alloc_res.slot_id, key}}).front().success) {
    backend->BatchAbort({alloc_res.slot_id});
    return PutAttemptOutcome::kFatal;
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
  if (registry_.Empty()) {
    MORI_UMBP_ERROR("[PoolClient] Local Get requested but no storage backend is registered");
    return GetAttemptOutcome::kFatal;
  }
  // Get carries no tier — the key may be in any medium here, and mirrored
  // across several.  Walk this node's media and take the first hit, matching
  // what the peer service does for a remote reader.
  bool served = false;
  for (auto* backend : registry_.All()) {
    auto resolved = backend->BatchResolve({key}, /*include_descs=*/false).front();
    if (!resolved.found) continue;
    // Same guard the remote path applies after BatchResolveKeys: a stored size
    // that disagrees with the requested one is a different object, and copying
    // `size` bytes out of a slot sized for something else would read past it.
    if (resolved.size != size) {
      MORI_UMBP_WARN("[PoolClient] local Get: size mismatch for key='{}' (wanted {}, got {})", key,
                     size, resolved.size);
      return GetAttemptOutcome::kRetry;
    }
    std::vector<TransferItem> items;
    if (!BuildLocalPageTransfers(backend, resolved.pages, resolved.page_size, dst, size,
                                 /*to_backend=*/false, &items)) {
      // This medium holds the key but cannot be read in-process (no published
      // endpoint for its buffers).  Route elsewhere rather than reporting a
      // miss, which would make the client exclude a node that does hold it.
      return GetAttemptOutcome::kRetry;
    }
    if (!transfer_engine_->Transfer(items, /*failed_tags=*/nullptr)) {
      return GetAttemptOutcome::kFatal;
    }
    served = true;
    break;
  }
  // No medium here held the key.  This is reachable: PartitionBatchGetTargets
  // sends a key master has no route for down the local path as a fallback.
  // Falling through to kSuccess would report a HIT with an untouched dst — the
  // caller cannot tell the difference, and would hand stale bytes to its own
  // caller.  (Pre-Phase-6 the loop did exactly that.)
  if (!served) return GetAttemptOutcome::kRetry;
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
  master_client_->AddCounter(MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL,
                             MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP,
                             {{"traffic", "local"}}, static_cast<double>(size));
  return GetAttemptOutcome::kSuccess;
}

void PoolClient::MaybeReCacheAfterRemote(const std::string& key, const void* src, size_t size) {
  auto* local = registry_.Get(medium_);
  if (local == nullptr || local->BufferCount() == 0) return;  // no exportable local medium here
  // Admission gate (cache_remote_fetches / size==0 / NEVER / SIZE cap): shared
  // pure predicate, unit-tested in test_cache_remote_admission.cpp.
  if (!ShouldAdmitReCache(config_.cache_remote_fetches, config_.cache_remote_admission,
                          config_.admission_max_block_bytes, size)) {
    MORI_UMBP_DEBUG("[PoolClient] MaybeReCacheAfterRemote: key='{}' size={} not admitted", key,
                    size);
    return;
  }
  // ALWAYS and SIZE both delegate capacity enforcement to the peer allocator:
  // Allocate returns kFailedNoSpace when the medium is full, which we treat as
  // a best-effort miss (the remote read result is unaffected).

  // Prepare the job outside the queue lock: the source buffer is valid for this
  // call, but copying a multi-MiB block while holding recache_mutex_ would
  // serialize unrelated Get finalizers behind this memcpy.
  ReCacheJob job;
  job.key = key;
  job.bytes = std::unique_ptr<char[]>(new (std::nothrow) char[size]);
  if (!job.bytes) {
    MORI_UMBP_DEBUG("[PoolClient] MaybeReCacheAfterRemote: allocation failed for key='{}' size={}",
                    key, size);
    return;
  }
  job.size = size;
  HostCopyBlock(job.bytes.get(), src, size);

  // Enqueue for asynchronous install. The actual DRAM Allocate + copy +
  // Commit→KvEvent::ADD publish is performed by ReCacheWorkerLoop OFF the Get
  // critical path, so it does not add latency to concurrent Gets (the tail-round
  // TTFT blowup observed with a synchronous on-path install). Bounded queue →
  // drop-on-full keeps best-effort semantics.
  {
    std::lock_guard<std::mutex> lk(recache_mutex_);
    if (recache_stop_) return;
    if (recache_queue_.size() >= recache_queue_max_) {
      MORI_UMBP_DEBUG("[PoolClient] MaybeReCacheAfterRemote: queue full, dropping key='{}'", key);
      return;
    }
    recache_queue_.push_back(std::move(job));
  }
  recache_cv_.notify_one();
}

void PoolClient::ReCacheWorkerLoop() {
  for (;;) {
    ReCacheJob job;
    {
      std::unique_lock<std::mutex> lk(recache_mutex_);
      recache_cv_.wait(lk, [this] { return recache_stop_ || !recache_queue_.empty(); });
      if (recache_stop_ && recache_queue_.empty()) return;
      job = std::move(recache_queue_.front());
      recache_queue_.pop_front();
    }
    // Install into this node's medium. ExecuteLocalPut allocates a slot on that
    // backend, copies the bytes, and Commit queues a KvEvent::ADD that reaches
    // the master via heartbeat — mirroring the local Put publish path.
    // kSuccessAlreadyExists makes this idempotent for a repeat remote read of
    // the same key.
    switch (ExecuteLocalPut(job.key, job.bytes.get(), job.size, medium_)) {
      case PutAttemptOutcome::kSuccess:
        MORI_UMBP_DEBUG("[PoolClient] ReCacheWorker: re-cached key='{}' size={}", job.key,
                        job.size);
        break;
      case PutAttemptOutcome::kSuccessAlreadyExists:
        break;
      case PutAttemptOutcome::kRetry:
      case PutAttemptOutcome::kFatal:
        MORI_UMBP_DEBUG("[PoolClient] ReCacheWorker: local install failed for key='{}'", job.key);
        break;
    }
  }
}

// ---------------------------------------------------------------------------
//  BatchPut
// ---------------------------------------------------------------------------

bool PoolClient::Put(const std::string& key, const void* src, size_t size) {
  std::vector<std::string> keys{key};
  std::vector<const void*> srcs{src};
  std::vector<size_t> sizes{size};
  auto results = BatchPut(keys, srcs, sizes);
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

  BatchPutPlan plan = PartitionBatchPutTargets(keys, srcs, sizes, routes, &outcomes);
  ExecuteBatchPutPlan(plan, &outcomes);

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

PoolClient::BatchPutPlan PoolClient::PartitionBatchPutTargets(
    const std::vector<std::string>& keys, const std::vector<const void*>& srcs,
    const std::vector<size_t>& sizes, const std::vector<std::optional<RoutePutResult>>& routes,
    std::vector<PutEntryOutcome>* results) {
  BatchPutPlan plan;
  const size_t count = keys.size();
  for (size_t i = 0; i < count; ++i) {
    // Zero-size puts are meaningless: leave the result kFailed, never execute.
    if (sizes[i] == 0) {
      MORI_UMBP_WARN("[PoolClient] BatchPut: skipping zero-size put for key='{}'", keys[i]);
      continue;
    }
    if (i >= routes.size() || !routes[i].has_value()) continue;
    const auto& route = routes[i].value();
    // Master-side dedup hit.
    if (route.outcome == RoutePutOutcome::kAlreadyExists) {
      (*results)[i] = PutEntryOutcome::kAlreadyExists;
      continue;
    }
    if (route.node_id == config_.master_config.node_id) {
      // Self-target: deferred (with its tier) so ExecuteBatchPutPlan can run the
      // local memcpy inside the remote-DRAM submit..wait window.
      plan.local_items.push_back(BatchPutItem{
          .index = i, .key = &keys[i], .src = srcs[i], .size = sizes[i], .route = route});
      continue;
    }
    // No tier filter: every medium a peer advertises publishes registered
    // pages (SSD's are its staging arena — see ssd_backend.h), so the remote
    // put path is the same for all of them.  The old DRAM/HBM allowlist here
    // silently dropped puts master routed to a peer's SSD.
    plan.remote_groups[route.node_id].push_back(BatchPutItem{
        .index = i, .key = &keys[i], .src = srcs[i], .size = sizes[i], .route = route});
  }
  return plan;
}

void PoolClient::ExecuteBatchPutPlan(const BatchPutPlan& plan,
                                     std::vector<PutEntryOutcome>* results) {
  // Deferred local puts, parallel: per-key memcpy is lock-free (the allocator
  // serializes Allocate/Commit); results is not vector<bool>-bit-packed, so
  // workers write distinct indices directly. AddCounter / timing stay here.
  auto run_local_put = [&]() {
    const auto& local = plan.local_items;
    if (local.empty()) return;
    const int nthr = LocalCopyThreads("UMBP_DRAM_WRITE_THREADS");
    const auto t0 = std::chrono::steady_clock::now();
    ParallelFor(local.size(), nthr, [&](size_t k) {
      const auto& item = local[k];
      switch (ExecuteLocalPut(*item.key, item.src, item.size, item.route.tier)) {
        case PutAttemptOutcome::kSuccess:
          (*results)[item.index] = PutEntryOutcome::kSucceeded;
          break;
        case PutAttemptOutcome::kSuccessAlreadyExists:
          (*results)[item.index] = PutEntryOutcome::kAlreadyExists;
          break;
        case PutAttemptOutcome::kRetry:
        case PutAttemptOutcome::kFatal:
          break;
      }
    });
    if (std::getenv("UMBP_LOCAL_COPY_TIMING")) {
      double sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
      size_t tot = 0;
      for (const auto& item : local) tot += item.size;
      MORI_UMBP_INFO("[LocalCopy] PUT keys={} bytes={} threads={} elapsed_ms={:.3f} GiB_s={:.2f}",
                     local.size(), tot, nthr, sec * 1000.0,
                     tot / (sec > 0 ? sec : 1e-12) / (1024.0 * 1024 * 1024));
    }
  };

  // Submit every peer (not waited) to overlap the wire across peers, run the
  // local puts in that window, then wait all + commit.  On early exit the
  // engine handle's destructor drains; the wait does mapping + commit/abort.
  //
  // There is no longer an all-zero-copy / all-staging fork here.  Staging is
  // the engine's bounce pool, and a plan that needs it settles inside Submit,
  // so submit-all is unconditionally safe — a batch that mixes registered and
  // unregistered buffers, which used to be a contract violation the client
  // failed, now just works.
  std::vector<std::unique_ptr<RemotePutInFlight>> inflights;
  inflights.reserve(plan.remote_groups.size());
  for (const auto& [node_id, items] : plan.remote_groups) {
    if (auto f = SubmitRemoteBatchPut(items, results)) inflights.push_back(std::move(f));
  }
  run_local_put();
  for (auto& f : inflights) WaitRemoteBatchPut(*f, results);
}

std::unique_ptr<PoolClient::RemotePutInFlight> PoolClient::SubmitRemoteBatchPut(
    const std::vector<BatchPutItem>& items, std::vector<PutEntryOutcome>* results) {
  if (items.empty()) return nullptr;
  auto fail_all = [&] {
    for (const auto& item : items) (*results)[item.index] = PutEntryOutcome::kFailed;
  };
  if (peer_directory_ == nullptr) {
    MORI_UMBP_ERROR("[PoolClient] SubmitRemoteBatchPut: no RDMA engine configured (items={})",
                    items.size());
    fail_all();
    return nullptr;
  }

  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchPut: peer service connection unavailable, node='{}' "
        "addr='{}' items={}",
        first.route.node_id, first.route.peer_address, items.size());
    fail_all();
    return nullptr;
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  auto inflight = std::make_unique<RemotePutInFlight>();
  inflight->peer = &peer;
  inflight->stub = stub;

  // Abort already-allocated slots on a synchronous failure that returns nullptr
  // (no WaitRemoteBatchPut/finalize will run for them).
  auto abort_now = [&](std::vector<uint64_t> slot_ids) {
    if (slot_ids.empty()) return;
    ::umbp::BatchAbortSlotsRequest abort_req;
    for (uint64_t slot_id : slot_ids) abort_req.add_slot_ids(slot_id);
    ::umbp::BatchAbortSlotsResponse abort_resp;
    grpc::ClientContext abort_ctx;
    // Best-effort: a failed abort just leaves the slots for the peer reaper to
    // reclaim at pending_ttl. Warn to aid diagnosis but do not propagate.
    auto s = stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
    if (!s.ok()) {
      MORI_UMBP_WARN(
          "[PoolClient] SubmitRemoteBatchPut: BatchAbortSlots({} slots) failed on {}: {}",
          slot_ids.size(), first.route.node_id, s.error_message());
    }
  };

  // Allocate RPC + per-key dedup/failure mapping; malformed slots go to
  // inflight->abort_slots. On total failure results are written and the
  // malformed list already aborted inside the callee — nothing left in flight.
  if (!AllocateRemotePutEntries(items, stub, &inflight->entries, &inflight->abort_slots, results)) {
    return nullptr;
  }

  // Abort everything allocated and fail every key: used on the paths that
  // return nullptr, where no WaitRemoteBatchPut/finalize will run.
  auto abort_everything = [&] {
    std::vector<uint64_t> all = std::move(inflight->abort_slots);
    for (auto& entry : inflight->entries) {
      all.push_back(entry.slot_id);
      (*results)[entry.result_index] = PutEntryOutcome::kFailed;
    }
    abort_now(std::move(all));
  };

  std::vector<TransferItem> transfer_items;
  if (!BuildRemotePutTransfers(inflight->entries, first.route.node_id, &transfer_items)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchPut: BuildRemotePutTransfers failed, node='{}' entries={} "
        "-> aborting all slots",
        first.route.node_id, inflight->entries.size());
    abort_everything();
    return nullptr;
  }

  // Drop items whose entry failed during build.  Those failed entries ride in
  // inflight->entries and are aborted by FinalizeRemotePutEntries at wait time —
  // do NOT early-abort them here (they are not a whole-batch failure).
  std::vector<TransferItem> active;
  active.reserve(transfer_items.size());
  for (auto& item : transfer_items) {
    if (!inflight->entries[item.tag].failed) active.push_back(std::move(item));
  }
  if (active.empty()) {
    abort_everything();
    return nullptr;
  }

  TransferPlanSet planned = transfer_engine_->Plan(active);
  ApplyRejectedTags(inflight->entries, planned.rejected_tags, "RemotePut");
  if (planned.plans.empty()) {
    abort_everything();
    return nullptr;
  }
  // POST; do NOT wait.  Everything the post references — including any bytes
  // staged through the engine's bounce pool — is owned by the returned handle.
  inflight->handle = transfer_engine_->Submit(std::move(planned.plans));
  if (inflight->handle == nullptr) {
    abort_everything();
    return nullptr;
  }
  return inflight;
}

void PoolClient::WaitRemoteBatchPut(RemotePutInFlight& f, std::vector<PutEntryOutcome>* results) {
  if (f.drained) return;
  f.drained = true;
  std::vector<TransferFailure> failures;
  if (f.handle != nullptr) f.handle->Wait(&failures);
  ApplyTransferFailures(f.entries, failures, "RemotePut");
  FinalizeRemotePutEntries(f.entries, f.abort_slots, results, f.stub);
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
      // Best-effort: a failed abort just leaves the slots for the peer reaper to
      // reclaim at pending_ttl. Warn to aid diagnosis but do not propagate.
      auto abort_status = stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
      if (!abort_status.ok()) {
        MORI_UMBP_WARN(
            "[PoolClient] AllocateRemotePutEntries: BatchAbortSlots({} slots) failed: {}",
            abort_slots->size(), abort_status.error_message());
      }
      abort_slots->clear();
    }
    return false;
  }
  return true;
}

bool PoolClient::BuildRemotePutTransfers(std::vector<RemotePutEntry>& entries,
                                         const std::string& node_id,
                                         std::vector<TransferItem>* items) {
  items->clear();

  // Hydrate every entry's descs first, then snapshot the peer's buffers once
  // PER BACKEND the batch touches: the loop below indexes a snapshot instead of
  // taking the engine's remote lock per page.  A batch may span media (each
  // item carries its own route.tier), and buffer_index is backend-local, so one
  // snapshot per peer would index the wrong medium's buffers.  A concurrent
  // hydrate can only add buffers, so a snapshot taken here is never stale for
  // the indices these entries reference.
  for (const auto& entry : entries) {
    if (!entry.plan.descs.empty()) peer_directory_->CacheRemoteBuffers(node_id, entry.plan.descs);
  }
  std::array<std::vector<TransferRef>, kMaxBackendsPerPeer> snapshots;
  std::array<bool, kMaxBackendsPerPeer> snapped{};
  auto buffers_for = [&](uint32_t backend_id) -> const std::vector<TransferRef>& {
    static const std::vector<TransferRef> kNone;
    if (backend_id >= kMaxBackendsPerPeer) return kNone;
    if (!snapped[backend_id]) {
      snapshots[backend_id] = peer_directory_->RemoteBufferSnapshot(node_id, backend_id);
      snapped[backend_id] = true;
    }
    return snapshots[backend_id];
  };

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    // Whether this goes zero-copy or through the engine's bounce pool is the
    // engine's decision; this only names the endpoint.
    const auto [src, src_base] =
        UserBufferRef(const_cast<void*>(entry.item->src), entry.item->size);
    const std::vector<TransferRef>& remote = buffers_for(entry.plan.backend_id);

    std::vector<TransferItem> entry_items;
    entry_items.reserve(entry.plan.pages.size());
    for (size_t p = 0; p < entry.plan.pages.size(); ++p) {
      const auto& page = entry.plan.pages[p];
      if (page.buffer_index >= remote.size() || !remote[page.buffer_index].HasMemoryDesc()) {
        MORI_UMBP_ERROR(
            "[PoolClient] BuildRemotePutTransfers: peer published no buffer, "
            "key='{}' backend={} buffer_index={} peer_buffers={} page_index={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.plan.backend_id, page.buffer_index, remote.size(), page.page_index);
        entry.failed = true;
        entry_items.clear();
        break;
      }
      TransferItem item;
      item.tag = idx;
      item.src = src;
      item.src_offset = src_base + static_cast<uint64_t>(p) * entry.plan.page_size;
      item.dst = remote[page.buffer_index];
      item.dst_offset = static_cast<uint64_t>(page.page_index) * entry.plan.page_size;
      item.size =
          LogicalPageBytes(p, entry.plan.pages.size(), entry.plan.page_size, entry.item->size);
      entry_items.push_back(std::move(item));
    }

    if (!entry_items.empty()) {
      items->insert(items->end(), std::make_move_iterator(entry_items.begin()),
                    std::make_move_iterator(entry_items.end()));
    }
  }
  return true;
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
    // Best-effort: a failed abort just leaves the slots for the peer reaper to
    // reclaim at pending_ttl. Warn to aid diagnosis but do not propagate.
    auto abort_status = stub->BatchAbortSlots(&abort_ctx, abort_req, &abort_resp);
    if (!abort_status.ok()) {
      MORI_UMBP_WARN("[PoolClient] FinalizeRemotePutEntries: BatchAbortSlots({} slots) failed: {}",
                     abort_slots.size(), abort_status.error_message());
    }
    abort_slots.clear();
  }
}

// ---------------------------------------------------------------------------
//  BatchGet
// ---------------------------------------------------------------------------

bool PoolClient::Get(const std::string& key, void* dst, size_t size) {
  std::vector<std::string> keys{key};
  std::vector<void*> dsts{dst};
  std::vector<size_t> sizes{size};
  auto results = BatchGet(keys, dsts, sizes);
  return !results.empty() && results[0];
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

  BatchGetPlan plan = PartitionBatchGetTargets(keys, dsts, sizes, routes);
  ExecuteBatchGetPlan(plan, keys, dsts, sizes, &results);

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

PoolClient::BatchGetPlan PoolClient::PartitionBatchGetTargets(
    const std::vector<std::string>& keys, const std::vector<void*>& dsts,
    const std::vector<size_t>& sizes, const std::vector<std::optional<RouteGetResult>>& routes) {
  BatchGetPlan plan;
  for (size_t i = 0; i < keys.size(); ++i) {
    // Zero-size gets are rejected before local fallback or remote read: an
    // explicit skip is required here because a nullopt route below would
    // otherwise fall through to a local read (result stays false).
    if (sizes[i] == 0) {
      MORI_UMBP_WARN("[PoolClient] BatchGet: skipping zero-size get for key='{}'", keys[i]);
      continue;
    }
    if (i >= routes.size() || !routes[i].has_value()) {
      if (!registry_.Empty()) plan.local_indices.push_back(i);
      continue;
    }
    const auto& route = routes[i].value();
    if (route.node_id == config_.master_config.node_id) {
      // Self-target: deferred (collected as an index) so ExecuteBatchGetPlan
      // can place it inside the remote-DRAM in-flight window in the overlap
      // path.
      plan.local_indices.push_back(i);
      continue;
    }
    BatchGetItem item{.index = i,
                      .key = &keys[i],
                      .dst = const_cast<void*>(dsts[i]),
                      .size = sizes[i],
                      .route = route};
    plan.remote_groups[route.node_id].push_back(std::move(item));
  }
  return plan;
}

void PoolClient::ExecuteBatchGetPlan(const BatchGetPlan& plan, const std::vector<std::string>& keys,
                                     const std::vector<void*>& dsts,
                                     const std::vector<size_t>& sizes, std::vector<bool>* results) {
  // Parallel local reads: different threads handle different keys. Resolve
  // is mutex-serialized in the allocator; the per-key memcpy in
  // ExecuteLocalGet->LocalGetPages runs lock-free in parallel. results is
  // std::vector<bool> (bit-packed) so threads write a temp buffer; merge serially.
  auto run_local = [&]() {
    const auto& idx = plan.local_indices;
    if (idx.empty()) return;
    const int nthr = LocalCopyThreads("UMBP_DRAM_READ_THREADS");
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> ok(idx.size(), 0);
    ParallelFor(idx.size(), nthr, [&](size_t k) {
      const size_t i = idx[k];
      if (ExecuteLocalGet(keys[i], const_cast<void*>(dsts[i]), sizes[i]) ==
          GetAttemptOutcome::kSuccess) {
        ok[k] = 1;
      }
    });
    size_t tot = 0;
    for (size_t k = 0; k < idx.size(); ++k) {
      if (ok[k]) {
        (*results)[idx[k]] = true;
        tot += sizes[idx[k]];
      }
    }
    if (std::getenv("UMBP_LOCAL_COPY_TIMING")) {
      double sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
      MORI_UMBP_INFO("[LocalCopy] GET keys={} bytes={} threads={} elapsed_ms={:.3f} GiB_s={:.2f}",
                     idx.size(), tot, nthr, sec * 1000.0,
                     tot / (sec > 0 ? sec : 1e-12) / (1024.0 * 1024 * 1024));
    }
  };

  // Submit every peer (posted, not waited) to overlap wire time across peers,
  // run local reads in that window, then wait all.  On early/exceptional exit
  // the engine handle's destructor drains in-flight statuses (lifetime safety);
  // the wait loop does failure mapping + backfill.
  //
  // As in Put, there is no all-zero-copy / all-staging fork any more: staging
  // lives in the engine's bounce pool and a plan that needs it settles inside
  // Submit, so submit-all is unconditionally safe.
  std::vector<std::unique_ptr<RemoteGetInFlight>> inflights;
  inflights.reserve(plan.remote_groups.size());
  for (const auto& [node_id, items] : plan.remote_groups) {
    if (auto f = SubmitRemoteBatchGet(items, results)) inflights.push_back(std::move(f));
  }
  run_local();
  for (auto& f : inflights) WaitRemoteBatchGet(*f, results);
}

std::unique_ptr<PoolClient::RemoteGetInFlight> PoolClient::SubmitRemoteBatchGet(
    const std::vector<BatchGetItem>& items, std::vector<bool>* results) {
  if (items.empty()) return nullptr;
  auto fail_all = [&] {
    for (const auto& item : items) (*results)[item.index] = false;
  };
  if (peer_directory_ == nullptr) {
    MORI_UMBP_ERROR("[PoolClient] SubmitRemoteBatchGet: no RDMA engine configured (items={})",
                    items.size());
    fail_all();
    return nullptr;
  }

  const auto& first = items.front();
  auto& peer = GetOrConnectPeer(first.route.node_id, first.route.peer_address);
  if (!EnsurePeerServiceConnection(peer)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchGet: peer service connection unavailable, node='{}' "
        "addr='{}' items={}",
        first.route.node_id, first.route.peer_address, items.size());
    fail_all();
    return nullptr;
  }
  auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());

  auto inflight = std::make_unique<RemoteGetInFlight>();
  inflight->peer = &peer;

  // resolve RPC + per-key validation; failed keys already written to *results.
  if (!PrepareRemoteGetEntries(items, peer, stub, &inflight->entries, results)) {
    return nullptr;
  }

  std::vector<TransferItem> transfer_items;
  if (!BuildRemoteGetTransfers(inflight->entries, first.route.node_id, &transfer_items)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchGet: BuildRemoteGetTransfers failed, node='{}' entries={}",
        first.route.node_id, inflight->entries.size());
    for (auto& entry : inflight->entries) (*results)[entry.result_index] = false;
    return nullptr;
  }

  // Drop items whose entry failed during build (peer published no buffer etc.).
  std::vector<TransferItem> active;
  active.reserve(transfer_items.size());
  for (auto& item : transfer_items) {
    if (!inflight->entries[item.tag].failed) active.push_back(std::move(item));
  }
  if (active.empty()) {
    for (auto& entry : inflight->entries) {
      if (entry.failed) (*results)[entry.result_index] = false;
    }
    return nullptr;
  }

  TransferPlanSet planned = transfer_engine_->Plan(active);
  ApplyRejectedTags(inflight->entries, planned.rejected_tags, "RemoteGet");
  if (planned.plans.empty()) {
    for (auto& entry : inflight->entries) {
      if (entry.failed) (*results)[entry.result_index] = false;
    }
    return nullptr;
  }
  // POST; do NOT wait.  Everything the post references is owned by the returned
  // handle, including any bytes staged through the engine's bounce pool.
  inflight->handle = transfer_engine_->Submit(std::move(planned.plans));
  if (inflight->handle == nullptr) {
    for (auto& entry : inflight->entries) (*results)[entry.result_index] = false;
    return nullptr;
  }
  return inflight;
}

void PoolClient::WaitRemoteBatchGet(RemoteGetInFlight& f, std::vector<bool>* results) {
  if (f.drained) return;
  f.drained = true;
  std::vector<TransferFailure> failures;
  if (f.handle != nullptr) f.handle->Wait(&failures);
  ApplyTransferFailures(f.entries, failures, "RemoteGet");
  // Nothing to copy out here even for a staged read: the engine owns its bounce
  // pool and lands the bytes in the user's dst before its Wait returns.
  FinalizeRemoteGetEntries(f.entries, results);
}

bool PoolClient::PrepareRemoteGetEntries(const std::vector<BatchGetItem>& items,
                                         PeerConnection& peer, ::umbp::UMBPPeer::Stub* stub,
                                         std::vector<RemoteGetEntry>* entries,
                                         std::vector<bool>* results) {
  entries->clear();

  // Ask the peer to omit the buffer descriptors once we have already hydrated
  // them (from the GetPeerInfo handshake, or a prior resolve).  A wrong guess
  // is safe: a missing descriptor is caught by the transfer-build guard and the
  // entry degrades to a miss, never a corrupt read.
  const bool have_descs =
      peer_directory_ != nullptr && peer_directory_->HasRemoteBuffers(peer.node_id);

  ::umbp::BatchResolveKeysRequest resolve_req;
  for (const auto& item : items) resolve_req.add_keys(*item.key);
  resolve_req.set_omit_descs(have_descs);

  ::umbp::BatchResolveKeysResponse resolve_resp;
  grpc::ClientContext resolve_ctx;
  auto resolve_status = stub->BatchResolveKeys(&resolve_ctx, resolve_req, &resolve_resp);
  if (!resolve_status.ok() ||
      BatchResolveKeyCount(resolve_resp) != static_cast<int>(items.size())) {
    MORI_UMBP_WARN("[PoolClient] BatchResolveKeys failed on {}: {}", items.front().route.node_id,
                   resolve_status.error_message());
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return false;
  }

  DecodedBatchResolve decoded = DecodeBatchResolveResponse(resolve_resp);
  if (decoded.keys.size() != items.size()) {
    // Malformed (mismatched parallel arrays); fail the whole batch rather than
    // partially-read it.
    MORI_UMBP_WARN("[PoolClient] BatchResolveKeys malformed response on {}: {} keys for {} items",
                   items.front().route.node_id, decoded.keys.size(), items.size());
    for (const auto& item : items) {
      (*results)[item.index] = false;
    }
    return false;
  }
  // Hydrate the batch-level descriptors once (skipped when the peer honored
  // omit_descs and sent none).
  if (!decoded.descs.empty()) peer_directory_->CacheRemoteBuffers(peer.node_id, decoded.descs);

  entries->reserve(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& key = decoded.keys[i];
    if (!key.found) {
      (*results)[item.index] = false;
      continue;
    }
    if (key.size != item.size) {
      MORI_UMBP_WARN("[PoolClient] BatchGet: size mismatch for key='{}' (wanted {}, got {})",
                     *item.key, item.size, key.size);
      (*results)[item.index] = false;
      continue;
    }
    if (!SizeMatchesAllocation(item.size, key.pages.size(), decoded.page_size)) {
      MORI_UMBP_ERROR("[PoolClient] BatchGet: malformed slot for key='{}'", *item.key);
      (*results)[item.index] = false;
      continue;
    }

    RemoteGetEntry entry;
    entry.result_index = item.index;
    entry.item = &item;
    entry.plan.page_size = decoded.page_size;
    // Per key, because a resolve batch may now be served from several media at
    // once; the pages below are indices into THIS backend's buffers.
    entry.plan.backend_id = key.backend_id;
    entry.plan.pages = std::move(decoded.keys[i].pages);
    // Descriptors were hydrated batch-level above; the per-entry plan carries
    // none (BuildRemoteGetTransfers' EnsureBufferDescsCached call is a no-op on
    // an empty list and the read path resolves descriptors by buffer_index).
    entries->push_back(std::move(entry));
  }

  return !entries->empty();
}

bool PoolClient::BuildRemoteGetTransfers(std::vector<RemoteGetEntry>& entries,
                                         const std::string& node_id,
                                         std::vector<TransferItem>* items) {
  items->clear();

  // Batch-level descs were already hydrated by PrepareRemoteGetEntries; any
  // per-entry ones are folded in here before the snapshots (see
  // BuildRemotePutTransfers for why one snapshot per backend beats a lock per
  // page — and why one snapshot per PEER would index the wrong medium now that
  // a resolve batch can span media).
  for (const auto& entry : entries) {
    if (!entry.plan.descs.empty()) peer_directory_->CacheRemoteBuffers(node_id, entry.plan.descs);
  }
  std::array<std::vector<TransferRef>, kMaxBackendsPerPeer> snapshots;
  std::array<bool, kMaxBackendsPerPeer> snapped{};
  auto buffers_for = [&](uint32_t backend_id) -> const std::vector<TransferRef>& {
    static const std::vector<TransferRef> kNone;
    if (backend_id >= kMaxBackendsPerPeer) return kNone;
    if (!snapped[backend_id]) {
      snapshots[backend_id] = peer_directory_->RemoteBufferSnapshot(node_id, backend_id);
      snapped[backend_id] = true;
    }
    return snapshots[backend_id];
  };

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];
    const auto [dst, dst_base] = UserBufferRef(entry.item->dst, entry.item->size);
    const std::vector<TransferRef>& remote = buffers_for(entry.plan.backend_id);

    std::vector<TransferItem> entry_items;
    entry_items.reserve(entry.plan.pages.size());
    for (size_t p = 0; p < entry.plan.pages.size(); ++p) {
      const auto& page = entry.plan.pages[p];
      if (page.buffer_index >= remote.size() || !remote[page.buffer_index].HasMemoryDesc()) {
        MORI_UMBP_ERROR(
            "[PoolClient] BuildRemoteGetTransfers: peer published no buffer, "
            "key='{}' backend={} buffer_index={} peer_buffers={} page_index={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.plan.backend_id, page.buffer_index, remote.size(), page.page_index);
        entry.failed = true;
        entry_items.clear();
        break;
      }
      TransferItem item;
      item.tag = idx;
      item.src = remote[page.buffer_index];
      item.src_offset = static_cast<uint64_t>(page.page_index) * entry.plan.page_size;
      item.dst = dst;
      item.dst_offset = dst_base + static_cast<uint64_t>(p) * entry.plan.page_size;
      item.size =
          LogicalPageBytes(p, entry.plan.pages.size(), entry.plan.page_size, entry.item->size);
      entry_items.push_back(std::move(item));
    }

    if (!entry_items.empty()) {
      items->insert(items->end(), std::make_move_iterator(entry_items.begin()),
                    std::make_move_iterator(entry_items.end()));
    }
  }
  return true;
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

    // Re-cache the remotely-fetched block into local DRAM (best-effort): the
    // user dst is already populated (staging copy-out and zero-copy both land
    // before Finalize runs), so subsequent reads of this key route local.
    if (entry.item) {
      MaybeReCacheAfterRemote(*entry.item->key, entry.item->dst, entry.item->size);
    }
  }
}

// ---------------------------------------------------------------------------
//  Cluster-wide existence check
// ---------------------------------------------------------------------------

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
  conn->node_id = node_id;
  conn->peer_address = peer_address;
  // The peer's engine desc and buffer descriptors are hydrated lazily in
  // EnsurePeerServiceConnection, into the transfer engine rather than here.
  auto& ref = *conn;
  peers_[node_id] = std::move(conn);
  return ref;
}

// ---------------------------------------------------------------------------
//  Peer connection setup
// ---------------------------------------------------------------------------

bool PoolClient::EnsurePeerServiceConnection(PeerConnection& peer) {
  std::lock_guard<std::mutex> lock(peer.conn_mutex);
  if (peer.peer_address.empty()) {
    return false;
  }

  // GetPeerInfo returns two different kinds of fact and they now go to two
  // different owners: the stub is a control-plane connection kept here, while
  // the peer's engine desc and buffer descriptors are transfer-layer facts and
  // go straight into the engine's remote cache.
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

    if (peer_directory_ != nullptr) {
      if (!peer_directory_->EnsureRemoteEngine(peer.node_id, resp.engine_desc())) return false;

      // Every backend's buffers arrive in one list, each entry naming the
      // backend its buffer_index belongs to.  Before backend_id, this list held
      // one medium's buffers and the rest were unreachable — yet HasRemoteBuffers
      // still went true, so the next resolve asked the peer to omit descriptors
      // and the missing media's pages were read against the published one's
      // memory.
      std::vector<BufferMemoryDescBytes> descs;
      descs.reserve(resp.buffer_descs_size());
      for (const auto& d : resp.buffer_descs()) {
        if (d.desc().empty()) continue;
        BufferMemoryDescBytes b;
        b.buffer_index = d.buffer_index();
        b.backend_id = d.backend_id();
        b.desc_bytes.assign(d.desc().begin(), d.desc().end());
        descs.push_back(std::move(b));
      }
      peer_directory_->CacheRemoteBuffers(peer.node_id, descs);
    }
    return true;
  };

  const bool engine_known =
      peer_directory_ == nullptr || peer_directory_->HasRemoteEngine(peer.node_id);

  if (peer.peer_stub) {
    if (!engine_known) {
      auto* stub = static_cast<::umbp::UMBPPeer::Stub*>(peer.peer_stub.get());
      if (!hydrate_from_peer(stub)) {
        peer.peer_stub.reset();
        if (peer_directory_ != nullptr) peer_directory_->ForgetRemote(peer.node_id);
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

void PoolClient::PublishBackendCounters() {
  if (!master_client_) return;

  for (MediumBackend* backend : registry_.All()) {
    if (backend == nullptr) continue;
    const char* backend_name = backend->Name();
    for (auto& c : backend->Counters()) {
      if (c.name == nullptr) continue;

      // Identity = backend + metric + labels.  Two backends reporting the same
      // metric name must not share a delta baseline, or one would cancel the
      // other's progress out.
      std::string id = backend_name;
      id += '\0';
      id += c.name;
      for (const auto& [k, v] : c.labels) {
        id += '\0';
        id += k;
        id += '=';
        id += v;
      }

      uint64_t& last = backend_counter_last_[id];
      // Monotonic by contract; a decrease means the backend was rebuilt, so
      // rebase instead of shipping a negative delta.
      if (c.value > last) {
        master_client_->AddCounter(c.name, c.help, c.labels, static_cast<double>(c.value - last));
      }
      last = c.value;
    }
  }
}

}  // namespace mori::umbp

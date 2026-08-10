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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <msgpack.hpp>
#include <new>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <unordered_map>

#include "mori/io/backend.hpp"
#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/peer/batch_resolve_codec.h"
#include "umbp/distributed/peer/mock_backend.h"
#include "umbp/distributed/peer/page_backend.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/peer/transfer_engine.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace {

bool IsValidMemoryDesc(const mori::io::MemoryDesc& desc) { return desc.size > 0; }

// Key for grouping page transfers by their (local MR, remote MR) pair.
struct PairKey {
  mori::io::MemoryUniqueId local;
  mori::io::MemoryUniqueId remote;
  bool operator==(const PairKey& o) const noexcept {
    return local == o.local && remote == o.remote;
  }
};
struct PairKeyHash {
  size_t operator()(const PairKey& k) const noexcept {
    // hash_combine (boost-style): independent of size_t width.
    size_t h = std::hash<mori::io::MemoryUniqueId>{}(k.local);
    h ^= std::hash<mori::io::MemoryUniqueId>{}(k.remote) + 0x9e3779b97f4a7c15ULL + (h << 6) +
         (h >> 2);
    return h;
  }
};

// True iff [next, ...) is exactly adjacent after [base, base+len), with no
// size_t overflow in base+len.  Used to coalesce contiguous SG segments.
inline bool AdjacentNoOverflow(size_t base, size_t len, size_t next) {
  return len <= std::numeric_limits<size_t>::max() - base && base + len == next;
}

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

// Bytes belonging to the i-th logical page of a Put/Get spread across
// `num_pages` pages of `page_size` bytes.  Last page may be partial.
// --- Parallel + AVX2 NT block copy for self-target (local) pages. ----------
// In distributed mode 1 key == 1 page (master page_size == KV block size), so
// the distributed self-target path copies one ~4 MiB block per call. The local
// tier's (DRAMTier) multi-thread/AVX2 optimization never applied here because
// it parallelizes *within* a call's pages (always 1). We instead parallelize
// across the many keys of one BatchPut/BatchGet (different threads -> different
// keys); per-key NT-AVX2 copy gives the cache-bypass win. Threads via
// UMBP_DRAM_{READ,WRITE}_THREADS (same envs as DRAMTier).
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) inline void LocalNtCopyAvx2(char* d, const char* s, size_t n) {
  size_t head = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
  if (head > n) head = n;
  std::memcpy(d, s, head);
  size_t i = head;
  for (; i + 128 <= n; i += 128) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 32));
    __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 64));
    __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 96));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 32), b);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 64), c);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 96), e);
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
  }
  if (i < n) std::memcpy(d + i, s + i, n - i);
  _mm_sfence();
}
inline bool LocalAvx2Supported() { return __builtin_cpu_supports("avx2"); }
#else
inline void LocalNtCopyAvx2(char* d, const char* s, size_t n) { std::memcpy(d, s, n); }
inline bool LocalAvx2Supported() { return false; }
#endif

inline void LocalCopyBlock(void* dst, const void* src, size_t size) {
  static const bool kNt = LocalAvx2Supported() && !(std::getenv("UMBP_DRAM_NT_COPY") &&
                                                    std::getenv("UMBP_DRAM_NT_COPY")[0] == '0');
  static const size_t kNtMinBytes = 256ull << 10;
  if (kNt && size >= kNtMinBytes) {
    LocalNtCopyAvx2(static_cast<char*>(dst), static_cast<const char*>(src), size);
  } else {
    std::memcpy(dst, src, size);
  }
}

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

template <typename Fn>
inline void LocalParallelFor(size_t n, int num_threads, Fn&& fn) {
  if (n == 0) return;
  if (num_threads > static_cast<int>(n)) num_threads = static_cast<int>(n);
  if (num_threads <= 1) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    size_t i;
    while ((i = next.fetch_add(1)) < n) fn(i);
  };
  std::vector<std::thread> pool;
  pool.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
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

    rdma_cfg.qpPerTransfer = 4;
    rdma_cfg.enableTransferChunking = true;
    rdma_cfg.numNicsPerTransfer = 4;
    // UMBP only sets the config defaults above.  All RDMA knobs (qpPerTransfer
    // / postBatchSize / numWorkerThreads / pollCqMode / chunking / numNics /
    // ...) are overridable via MORI_IO_* env in the RDMA backend ctor, shared
    // by every IO backend user; no UMBP-specific entry points.
    io_engine_->CreateBackend(mori::io::BackendType::RDMA, rdma_cfg);

    staging_buffer_ = std::make_unique<char[]>(config_.staging_buffer_size);
    std::memset(staging_buffer_.get(), 0, config_.staging_buffer_size);
    staging_mem_ = io_engine_->RegisterMemory(staging_buffer_.get(), config_.staging_buffer_size,
                                              -1, mori::io::MemoryLocationType::CPU);

    MORI_UMBP_INFO("[PoolClient] IOEngine initialized on {}:{}", config_.io_engine.host,
                   config_.io_engine.port);
  }
  // Registration shim every backend's Init() receives.  Tolerates a null
  // io_engine_ (RegisterMemory then returns an invalid desc) — backends still
  // self-allocate their pool, they just aren't RDMA-reachable.
  transfer_engine_ = std::make_unique<TransferEngine>(io_engine_.get());

  // Peer-side backends: DRAM is the one real, self-allocated medium (design
  // doc §1 item 4 / Phase 2b — the backend now owns its memory, PoolClient no
  // longer holds a buffer pointer).  HBM gets a MockBackend so the registry
  // dispatches to more than one backend; it always advertises zero capacity
  // so master-side routing never picks it (design doc Phase 2 default
  // option b: "keeps the local path honestly single-medium").
  const uint64_t page_size =
      config_.dram_page_size > 0 ? config_.dram_page_size : 2ULL * 1024 * 1024;
  PageBackend::OwnershipConfig dram_ownership;
  dram_ownership.buffer_sizes = config_.dram.buffer_sizes;
  dram_ownership.use_hugepages = config_.dram.use_hugepages;
  dram_ownership.hugepage_size = config_.dram.hugepage_size;
  dram_ownership.numa_node = config_.dram.numa_node;
  dram_ownership.prefault = config_.dram.prefault;

  auto dram = std::make_unique<PageBackend>(TierType::DRAM, page_size, std::move(dram_ownership),
                                            /*pending_ttl=*/std::chrono::milliseconds{30000},
                                            /*read_lease_ttl=*/DramReadLeaseTtl());
  dram_backend_ = dram.get();
  if (!dram_backend_->Init(transfer_engine_.get())) {
    MORI_UMBP_ERROR("[PoolClient] DRAM backend Init failed");
    dram_backend_ = nullptr;
    initialized_ = false;
    return false;
  }
  dram_backend_->StartReaper();
  registry_.Register(std::move(dram));
  master_client_->SetPeerDramAllocator(dram_backend_);

  auto hbm_mock = std::make_unique<MockBackend>(TierType::HBM);
  hbm_mock->Init(transfer_engine_.get());
  registry_.Register(std::move(hbm_mock));

  // Pack engine_desc for master registration.
  std::vector<uint8_t> engine_desc_bytes;
  if (io_engine_) {
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, io_engine_->GetEngineDesc());
    engine_desc_bytes.assign(sbuf.data(), sbuf.data() + sbuf.size());
  }

  if (config_.peer_service_port > 0) {
    peer_service_ =
        std::make_unique<PeerServiceServer>(dram_backend_, engine_desc_bytes, master_client_.get());
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
  // (design doc §1 item 4 / Phase 2b) — a backend with zero capacity (e.g.
  // MockBackend) is omitted so it stays invisible to routing.
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

  // Start the async re-cache worker only when the feature is on and this node has
  // an exportable local DRAM tier to install into.
  if (config_.cache_remote_fetches && dram_backend_) {
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
  // dram_backend_ + master_client_), so it must be joined before those are torn
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

  // Every backend (including the DRAM one dram_backend_ points into)
  // deregisters its memory with transfer_engine_ inside its own destructor —
  // this MUST run before transfer_engine_ / io_engine_ are torn down below.
  dram_backend_ = nullptr;
  registry_ = BackendRegistry{};

  if (io_engine_) {
    {
      std::lock_guard<std::mutex> lock(registered_mem_mutex_);
      for (auto& reg : registered_regions_) io_engine_->DeregisterMemory(reg.mem_desc);
      registered_regions_.clear();
    }
    if (staging_buffer_) io_engine_->DeregisterMemory(staging_mem_);
    io_engine_.reset();
    staging_buffer_.reset();
  }
  transfer_engine_.reset();

  master_client_.reset();
}

bool PoolClient::Clear() {
  // Vacuously done: nothing has been initialized so there is no state to
  // clear and no master to notify.  Treat as success so callers in
  // shutdown / teardown paths do not surface a spurious error.
  if (!initialized_.load()) return true;
  if (dram_backend_) dram_backend_->ClearLocal();

  bool ok = true;
  if (master_client_) {
    ok = master_client_->ClearFullSync();
    if (!ok) MORI_UMBP_WARN("[PoolClient] Clear full-sync heartbeat failed");
  }
  return ok;
}

bool PoolClient::IsInitialized() const { return initialized_; }
MasterClient& PoolClient::Master() { return *master_client_; }
PageBackend* PoolClient::DramAllocator() { return dram_backend_; }

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
  if (!dram_backend_) return false;
  const char* src_bytes = static_cast<const char*>(src);
  for (size_t i = 0; i < pages.size(); ++i) {
    const auto& p = pages[i];
    auto buf = dram_backend_->LocalBufferView(p.buffer_index);
    const uint64_t off = static_cast<uint64_t>(p.page_index) * page_size;
    if (!buf.base || page_size > buf.size || off > buf.size - page_size) {
      MORI_UMBP_ERROR("[PoolClient] local Put: OOB buf={} off={}", p.buffer_index, off);
      return false;
    }
    const uint64_t bytes = LogicalPageBytes(i, pages.size(), page_size, size);
    LocalCopyBlock(static_cast<char*>(buf.base) + off, src_bytes + i * page_size, bytes);
  }
  return true;
}

bool PoolClient::LocalGetPages(const std::vector<PageLocation>& pages, uint64_t page_size,
                               void* dst, size_t size) {
  if (!dram_backend_) return false;
  char* dst_bytes = static_cast<char*>(dst);
  for (size_t i = 0; i < pages.size(); ++i) {
    const auto& p = pages[i];
    auto buf = dram_backend_->LocalBufferView(p.buffer_index);
    const uint64_t off = static_cast<uint64_t>(p.page_index) * page_size;
    if (!buf.base || page_size > buf.size || off > buf.size - page_size) {
      MORI_UMBP_ERROR("[PoolClient] local Get: OOB buf={} off={}", p.buffer_index, off);
      return false;
    }
    const uint64_t bytes = LogicalPageBytes(i, pages.size(), page_size, size);
    LocalCopyBlock(dst_bytes + i * page_size, static_cast<const char*>(buf.base) + off, bytes);
  }
  return true;
}

PoolClient::PutAttemptOutcome PoolClient::ExecuteLocalPut(const std::string& key, const void* src,
                                                          size_t size, TierType tier) {
  if (!dram_backend_) {
    MORI_UMBP_ERROR("[PoolClient] Local Put requested but DRAM backend unavailable");
    return PutAttemptOutcome::kFatal;
  }
  // The local fast path stays host-DRAM-only in this phase (design doc §5
  // Phase 2): a tier this backend does not serve is rejected the same way
  // the pre-refactor multi-tier allocator rejected an unconfigured tier
  // (kRetry, i.e. try routing elsewhere) rather than treated as fatal.
  if (tier != dram_backend_->Tier()) {
    MORI_UMBP_WARN("[PoolClient] Local Put: tier={} not served by this backend (tier={})",
                   static_cast<int>(tier), static_cast<int>(dram_backend_->Tier()));
    return PutAttemptOutcome::kRetry;
  }
  auto alloc_res = dram_backend_->Allocate(key, size);
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
  if (!LocalPutPages(alloc_res.pages, dram_backend_->PageSize(), src, size)) {
    dram_backend_->Abort(alloc_res.slot_id);
    return PutAttemptOutcome::kFatal;
  }
  uint64_t committed_bytes = 0;
  if (!dram_backend_->Commit(alloc_res.slot_id, key, committed_bytes)) {
    dram_backend_->Abort(alloc_res.slot_id);
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
  if (!dram_backend_) {
    MORI_UMBP_ERROR("[PoolClient] Local Get requested but DRAM backend unavailable");
    return GetAttemptOutcome::kFatal;
  }
  auto resolved = dram_backend_->Resolve(key);
  if (!resolved.found) {
    return GetAttemptOutcome::kRetry;
  }
  if (!LocalGetPages(resolved.pages, dram_backend_->PageSize(), dst, size)) {
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

void PoolClient::MaybeReCacheAfterRemote(const std::string& key, const void* src, size_t size) {
  if (!dram_backend_) return;  // no exportable local DRAM tier on this node
  // Admission gate (cache_remote_fetches / size==0 / NEVER / SIZE cap): shared
  // pure predicate, unit-tested in test_cache_remote_admission.cpp.
  if (!ShouldAdmitReCache(config_.cache_remote_fetches, config_.cache_remote_admission,
                          config_.admission_max_block_bytes, size)) {
    MORI_UMBP_DEBUG("[PoolClient] MaybeReCacheAfterRemote: key='{}' size={} not admitted", key,
                    size);
    return;
  }
  // ALWAYS and SIZE both delegate DRAM-capacity enforcement to the peer
  // allocator: Allocate returns kFailedNoSpace when the tier is full, which we
  // treat as a best-effort miss (the remote read result is unaffected).

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
  LocalCopyBlock(job.bytes.get(), src, size);

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
    // Install into local DRAM. ExecuteLocalPut allocates a slot on dram_backend_,
    // copies the bytes, and Commit queues a KvEvent::ADD that reaches the master
    // via heartbeat — mirroring the local Put publish path. kSuccessAlreadyExists
    // makes this idempotent for a repeat remote read of the same key.
    switch (ExecuteLocalPut(job.key, job.bytes.get(), job.size, TierType::DRAM)) {
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
    if (route.tier != TierType::DRAM && route.tier != TierType::HBM) continue;
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
    LocalParallelFor(local.size(), nthr, [&](size_t k) {
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

  const auto& remote_groups = plan.remote_groups;

  // All-ZC or all-staging (upper-layer invariant): probe one item's src.
  const bool is_zc =
      !remote_groups.empty() && FindRegisteredMemory(remote_groups.begin()->second.front().src,
                                                     remote_groups.begin()->second.front().size)
                                    .has_value();

  if (!is_zc) {
    // Staging / no remote: each peer submits then IMMEDIATELY waits. The
    // in-flight holds staging_mutex_ submit..wait, so a submit-all over multiple
    // staging peers would deadlock on the single lock.
    run_local_put();
    for (const auto& [node_id, items] : remote_groups) {
      if (auto f = SubmitRemoteBatchPut(items, results, /*permit_staging=*/true)) {
        WaitRemoteBatchPut(*f, results);
      }
    }
    return;
  }

  // Zero-copy: submit every peer (not waited) to overlap the wire across peers,
  // run local puts in that window, then wait all + commit. On early exit
  // ~RemoteDramPutInFlight drains statuses; the wait does mapping + commit/abort.
  std::vector<std::unique_ptr<RemoteDramPutInFlight>> inflights;
  inflights.reserve(remote_groups.size());
  for (const auto& [node_id, items] : remote_groups) {
    if (auto f = SubmitRemoteBatchPut(items, results, /*permit_staging=*/false)) {
      inflights.push_back(std::move(f));
    }
  }
  run_local_put();
  for (auto& f : inflights) WaitRemoteBatchPut(*f, results);
}

std::unique_ptr<PoolClient::RemoteDramPutInFlight> PoolClient::SubmitRemoteBatchPut(
    const std::vector<BatchPutItem>& items, std::vector<PutEntryOutcome>* results,
    bool permit_staging) {
  if (items.empty()) return nullptr;
  auto fail_all = [&] {
    for (const auto& item : items) (*results)[item.index] = PutEntryOutcome::kFailed;
  };
  if (!io_engine_) {
    MORI_UMBP_ERROR("[PoolClient] SubmitRemoteBatchPut: io_engine_ not initialized (items={})",
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

  auto inflight = std::make_unique<RemoteDramPutInFlight>();
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

  std::vector<TransferInstruction> transfers;
  uint64_t staging_bytes = 0;
  if (!BuildRemotePutTransfers(inflight->entries, peer, &transfers, &staging_bytes)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchPut: BuildRemotePutTransfers failed, node='{}' entries={} "
        "-> aborting all slots",
        first.route.node_id, inflight->entries.size());
    // Build failed wholesale: abort everything allocated.
    std::vector<uint64_t> all = std::move(inflight->abort_slots);
    for (auto& entry : inflight->entries) {
      all.push_back(entry.slot_id);
      (*results)[entry.result_index] = PutEntryOutcome::kFailed;
    }
    abort_now(std::move(all));
    return nullptr;
  }
  inflight->staging_bytes = staging_bytes;

  // Staging uses the shared staging_buffer_, so a cross-peer submit-all cannot
  // serialize on the single staging_mutex_ without deadlocking: the zero-copy
  // path passes permit_staging=false and fails such a batch (contract: a batch
  // is all-zc or all-staging). The serial staging path acquires staging_mutex_,
  // copies src->staging here (BEFORE BatchWrite, opposite of Get), and parks the
  // lock in the in-flight until WaitRemoteBatchPut releases it.
  if (staging_bytes > 0) {
    if (!permit_staging) {
      MORI_UMBP_ERROR(
          "[PoolClient] SubmitRemoteBatchPut: unexpected staging_bytes={} on zero-copy path "
          "node='{}' -> failing batch",
          staging_bytes, first.route.node_id);
      std::vector<uint64_t> all = std::move(inflight->abort_slots);
      for (auto& entry : inflight->entries) {
        all.push_back(entry.slot_id);
        (*results)[entry.result_index] = PutEntryOutcome::kFailed;
      }
      abort_now(std::move(all));
      return nullptr;
    }
    inflight->staging_lock = std::unique_lock<std::mutex>(staging_mutex_);
    for (auto& entry : inflight->entries) {
      if (entry.failed || !entry.use_staging) continue;
      // Defensive re-check before the memcpy (build already validated the cursor).
      if (entry.staging_offset + entry.item->size > config_.staging_buffer_size) {
        MORI_UMBP_ERROR(
            "[PoolClient] SubmitRemoteBatchPut: staging offset overflow (should not happen), "
            "key='{}' staging_offset={} size={} cap={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.staging_offset, entry.item ? entry.item->size : 0, config_.staging_buffer_size);
        entry.failed = true;
        continue;
      }
      std::memcpy(staging_buffer_.get() + entry.staging_offset, entry.item->src, entry.item->size);
    }
  }

  // Drop transfers whose entry failed during build. Those failed entries ride in
  // inflight->entries and are aborted by FinalizeRemotePutEntries at wait time —
  // do NOT early-abort them here (they're not a whole-batch failure).
  std::vector<TransferInstruction> active_transfers;
  active_transfers.reserve(transfers.size());
  for (const auto& t : transfers) {
    if (!inflight->entries[t.entry_index].failed) active_transfers.push_back(t);
  }
  if (active_transfers.empty()) {
    // Nothing to post: no in-flight returned, so finalize never runs. Release
    // staging_mutex_ before the abort RPC (no need to hold it over the wire),
    // then abort every allocated slot and fail every key.
    if (inflight->staging_lock.owns_lock()) inflight->staging_lock.unlock();
    std::vector<uint64_t> all = std::move(inflight->abort_slots);
    for (auto& entry : inflight->entries) {
      all.push_back(entry.slot_id);
      (*results)[entry.result_index] = PutEntryOutcome::kFailed;
    }
    abort_now(std::move(all));
    return nullptr;
  }

  // Collapse same-pair pages into one outer transfer, materialised INTO the
  // in-flight so the args outlive the post (see RemoteDramGetInFlight).
  inflight->groups = GroupTransfersByPair(active_transfers);
  const size_t G = inflight->groups.size();
  inflight->local_descs.resize(G);
  inflight->remote_descs.resize(G);
  inflight->local_offsets.resize(G);
  inflight->remote_offsets.resize(G);
  inflight->sizes_v.resize(G);
  inflight->statuses = std::vector<mori::io::TransferStatus>(G);  // built once, never resized/moved
  inflight->status_ptrs.resize(G);
  inflight->ids.resize(G);
  for (size_t g = 0; g < G; ++g) {
    inflight->local_descs[g] = inflight->groups[g].local_desc;
    inflight->remote_descs[g] = inflight->groups[g].remote_desc;
    inflight->local_offsets[g] = std::move(inflight->groups[g].local_offsets);
    inflight->remote_offsets[g] = std::move(inflight->groups[g].remote_offsets);
    inflight->sizes_v[g] = std::move(inflight->groups[g].sizes);
    inflight->status_ptrs[g] = &inflight->statuses[g];
    inflight->ids[g] = io_engine_->AllocateTransferUniqueId();
  }

  // POST the writes; do NOT wait. All args are owned by *inflight; for staging
  // the src bytes are already in staging_buffer_ under staging_lock.
  io_engine_->BatchWrite(inflight->local_descs, inflight->local_offsets, inflight->remote_descs,
                         inflight->remote_offsets, inflight->sizes_v, inflight->status_ptrs,
                         inflight->ids);
  return inflight;
}

void PoolClient::WaitRemoteBatchPut(RemoteDramPutInFlight& f,
                                    std::vector<PutEntryOutcome>* results) {
  if (f.drained) return;
  f.drained = true;  // set early (mirror Get) so the destructor never re-waits.
  // Wait every group (never break early); a non-success group fails all of its
  // contributing keys (per-item AND).
  const size_t G = f.groups.size();
  for (size_t g = 0; g < G; ++g) {
    f.statuses[g].Wait();
    if (f.statuses[g].Succeeded()) continue;
    for (size_t ei : f.groups[g].entry_indices) {
      auto& entry = f.entries[ei];
      MORI_UMBP_ERROR(
          "RemotePut BatchWrite failed: code={} msg='{}' peer_engine='{}' key='{}' slot_id={} "
          "use_staging={}",
          f.statuses[g].CodeUint32(), f.statuses[g].Message(), f.groups[g].remote_desc.engineKey,
          (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"}, entry.slot_id,
          entry.use_staging);
      entry.failed = true;
    }
  }

  // RDMA has consumed staging_buffer_; release staging_mutex_ (Put has no
  // staging->dst copy-out — the data went to the peer).
  if (f.staging_lock.owns_lock()) f.staging_lock.unlock();

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

bool PoolClient::BuildRemotePutTransfers(std::vector<RemotePutEntry>& entries, PeerConnection& peer,
                                         std::vector<TransferInstruction>* transfers,
                                         uint64_t* staging_bytes) {
  transfers->clear();
  uint64_t cursor = 0;

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    auto& entry = entries[idx];

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

    // Hydrate + snapshot remote descs inside ONE peers_mutex_ window: a
    // concurrent EnsureBufferDescsCached may resize dram_memories, so the
    // reads below must not race it.  zero_copy/staging above touch no peer
    // state and stay outside the lock.
    std::vector<TransferInstruction> entry_transfers;
    entry_transfers.reserve(entry.plan.pages.size());
    {
      std::lock_guard<std::mutex> lock(peers_mutex_);
      EnsureBufferDescsCachedLocked(peer, entry.plan.descs);
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
    }

    if (!entry_transfers.empty()) {
      transfers->insert(transfers->end(), entry_transfers.begin(), entry_transfers.end());
    }
  }

  *staging_bytes = cursor;
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
      if (dram_backend_) plan.local_indices.push_back(i);
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
    LocalParallelFor(idx.size(), nthr, [&](size_t k) {
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

  const auto& remote_groups = plan.remote_groups;

  // Zero-copy? The upper layer guarantees a batch is all-ZC or all-staging, so
  // probe one remote item's dst registration (unordered_map order is
  // unspecified, but any item answers the all-or-nothing question).
  const bool is_zc =
      !remote_groups.empty() && FindRegisteredMemory(remote_groups.begin()->second.front().dst,
                                                     remote_groups.begin()->second.front().size)
                                    .has_value();

  if (!is_zc) {
    // Staging (non-zero-copy), or no remote reads at all.  Order: local, then
    // staging remote (per peer).  Staging must submit then IMMEDIATELY wait
    // per peer: the in-flight holds staging_mutex_ from submit until the wait
    // copies out of the shared staging_buffer_, so accumulating staging
    // in-flights into a submit-all list would deadlock on that lock.
    run_local();
    for (const auto& [node_id, items] : remote_groups) {
      if (auto f = SubmitRemoteBatchGet(items, results, /*permit_staging=*/true)) {
        WaitRemoteBatchGet(*f, results);  // immediate; releases staging_mutex_
      }
    }
    return;
  }

  // Zero-copy remote: submit every peer (posted, not waited) to overlap wire
  // time across peers, run local reads in that window, then wait all. On
  // early/exceptional exit ~RemoteDramGetInFlight drains in-flight statuses
  // (lifetime safety); the wait loop does failure mapping + backfill.
  std::vector<std::unique_ptr<RemoteDramGetInFlight>> inflights;
  inflights.reserve(remote_groups.size());

  for (const auto& [node_id, items] : remote_groups) {
    if (auto f = SubmitRemoteBatchGet(items, results, /*permit_staging=*/false)) {
      inflights.push_back(std::move(f));
    }
  }
  run_local();
  for (auto& f : inflights) WaitRemoteBatchGet(*f, results);
}

std::unique_ptr<PoolClient::RemoteDramGetInFlight> PoolClient::SubmitRemoteBatchGet(
    const std::vector<BatchGetItem>& items, std::vector<bool>* results, bool permit_staging) {
  if (items.empty()) return nullptr;
  auto fail_all = [&] {
    for (const auto& item : items) (*results)[item.index] = false;
  };
  if (!io_engine_) {
    MORI_UMBP_ERROR("[PoolClient] SubmitRemoteBatchGet: io_engine_ not initialized (items={})",
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

  auto inflight = std::make_unique<RemoteDramGetInFlight>();
  inflight->peer = &peer;

  // resolve RPC + per-key validation; failed keys already written to *results.
  if (!PrepareRemoteGetEntries(items, peer, stub, &inflight->entries, results)) {
    return nullptr;
  }

  std::vector<TransferInstruction> transfers;
  uint64_t staging_bytes = 0;
  if (!BuildRemoteGetTransfers(inflight->entries, peer, &transfers, &staging_bytes)) {
    MORI_UMBP_WARN(
        "[PoolClient] SubmitRemoteBatchGet: BuildRemoteGetTransfers failed, node='{}' entries={}",
        first.route.node_id, inflight->entries.size());
    for (auto& entry : inflight->entries) (*results)[entry.result_index] = false;
    return nullptr;
  }
  inflight->staging_bytes = staging_bytes;
  // Staging needs the shared staging_buffer_, which a cross-peer submit-all
  // window cannot serialize on the single staging_mutex_ without deadlocking.
  // So the zero-copy submit-all caller passes permit_staging=false: a batch that
  // unexpectedly needs staging (upper layer guarantees all-zc or all-staging) is
  // failed rather than risking a deadlock.  The serial staging caller passes
  // true and we park staging_mutex_ in the in-flight until the wait copies out.
  if (staging_bytes > 0) {
    if (!permit_staging) {
      MORI_UMBP_ERROR(
          "[PoolClient] SubmitRemoteBatchGet: unexpected staging_bytes={} on zero-copy path "
          "node='{}' -> failing batch",
          staging_bytes, first.route.node_id);
      for (auto& entry : inflight->entries) (*results)[entry.result_index] = false;
      return nullptr;
    }
    inflight->staging_lock = std::unique_lock<std::mutex>(staging_mutex_);
  }

  // Drop transfers whose entry failed during build (invalid remote desc etc.).
  std::vector<TransferInstruction> active_transfers;
  active_transfers.reserve(transfers.size());
  for (const auto& t : transfers) {
    if (!inflight->entries[t.entry_index].failed) active_transfers.push_back(t);
  }
  if (active_transfers.empty()) {
    for (auto& entry : inflight->entries) {
      if (entry.failed) (*results)[entry.result_index] = false;
    }
    return nullptr;  // staging_lock (if held) released by the in-flight's destruction
  }

  // Collapse same-(localMR, remoteMR) pages into one outer transfer per pair,
  // then materialise the BatchRead args INTO the in-flight struct so they outlive
  // the post.
  inflight->groups = GroupTransfersByPair(active_transfers);
  const size_t G = inflight->groups.size();
  inflight->local_descs.resize(G);
  inflight->remote_descs.resize(G);
  inflight->local_offsets.resize(G);
  inflight->remote_offsets.resize(G);
  inflight->sizes_v.resize(G);
  // Build statuses at final size once and never resize/move (backend holds raw
  // TransferStatus*); status_ptrs alias into it.  vector move-assign steals the
  // buffer (default allocator), so elements are not moved and addresses are
  // stable for the in-flight lifetime.
  inflight->statuses = std::vector<mori::io::TransferStatus>(G);
  inflight->status_ptrs.resize(G);
  inflight->ids.resize(G);
  for (size_t g = 0; g < G; ++g) {
    inflight->local_descs[g] = inflight->groups[g].local_desc;
    inflight->remote_descs[g] = inflight->groups[g].remote_desc;
    inflight->local_offsets[g] = std::move(inflight->groups[g].local_offsets);
    inflight->remote_offsets[g] = std::move(inflight->groups[g].remote_offsets);
    inflight->sizes_v[g] = std::move(inflight->groups[g].sizes);
    inflight->status_ptrs[g] = &inflight->statuses[g];
    inflight->ids[g] = io_engine_->AllocateTransferUniqueId();
  }

  // POST the reads; do NOT wait.  Everything BatchRead may reference is owned by
  // *inflight; no allocation happens after this point in the submit window.  For
  // staging, the RDMA lands in staging_buffer_ under the held staging_lock.
  io_engine_->BatchRead(inflight->local_descs, inflight->local_offsets, inflight->remote_descs,
                        inflight->remote_offsets, inflight->sizes_v, inflight->status_ptrs,
                        inflight->ids);
  return inflight;
}

void PoolClient::WaitRemoteBatchGet(RemoteDramGetInFlight& f, std::vector<bool>* results) {
  if (f.drained) return;
  f.drained = true;
  // Wait every group (never break early): IN_PROGRESS groups complete here;
  // INIT/terminal return immediately.  A non-success group fails all of its
  // contributing keys (per-item AND).
  const size_t G = f.groups.size();
  for (size_t g = 0; g < G; ++g) {
    f.statuses[g].Wait();
    if (f.statuses[g].Succeeded()) continue;
    for (size_t ei : f.groups[g].entry_indices) {
      auto& entry = f.entries[ei];
      MORI_UMBP_ERROR(
          "RemoteGet BatchRead failed: code={} msg='{}' peer_engine='{}' key='{}' use_staging={}",
          f.statuses[g].CodeUint32(), f.statuses[g].Message(), f.groups[g].remote_desc.engineKey,
          (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
          entry.use_staging);
      entry.failed = true;
    }
  }

  // Staging (non-zero-copy): RDMA landed in staging_buffer_; copy each surviving
  // entry out to its user dst, then release staging_mutex_ (held since submit).
  if (f.staging_lock.owns_lock()) {
    for (auto& entry : f.entries) {
      if (entry.failed || !entry.use_staging) continue;
      if (entry.staging_offset + entry.item->size > config_.staging_buffer_size) {
        MORI_UMBP_ERROR(
            "[PoolClient] WaitRemoteBatchGet: staging offset overflow (should not happen), "
            "key='{}' staging_offset={} size={} cap={}",
            (entry.item && entry.item->key) ? *entry.item->key : std::string{"<null>"},
            entry.staging_offset, entry.item ? entry.item->size : 0, config_.staging_buffer_size);
        entry.failed = true;
        continue;
      }
      std::memcpy(entry.item->dst, staging_buffer_.get() + entry.staging_offset, entry.item->size);
    }
    f.staging_lock.unlock();
  }

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
  bool have_descs = false;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    have_descs = !peer.dram_memories.empty();
  }

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
  if (!decoded.descs.empty()) EnsureBufferDescsCached(peer, decoded.descs);

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
    entry.plan.pages = std::move(decoded.keys[i].pages);
    // Descriptors were hydrated batch-level above; the per-entry plan carries
    // none (BuildRemoteGetTransfers' EnsureBufferDescsCached call is a no-op on
    // an empty list and the read path resolves descriptors by buffer_index).
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

    // Hydrate + snapshot remote descs inside ONE peers_mutex_ window (see
    // BuildRemotePutTransfers).
    std::vector<TransferInstruction> entry_transfers;
    entry_transfers.reserve(entry.plan.pages.size());
    {
      std::lock_guard<std::mutex> lock(peers_mutex_);
      EnsureBufferDescsCachedLocked(peer, entry.plan.descs);
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
    }

    if (!entry_transfers.empty()) {
      transfers->insert(transfers->end(), entry_transfers.begin(), entry_transfers.end());
    }
  }

  *staging_bytes = cursor;
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
//  Remote DRAM transfer grouping (shared by Put and Get)
// ---------------------------------------------------------------------------

std::vector<PoolClient::PairGroup> PoolClient::GroupTransfersByPair(
    const std::vector<TransferInstruction>& active) {
  std::vector<PairGroup> groups;
  groups.reserve(active.size());
  // Group by (local MR id, remote MR id); first appearance defines order.
  std::unordered_map<PairKey, size_t, PairKeyHash> pair_to_group;
  pair_to_group.reserve(active.size() * 2);

  for (const auto& t : active) {
    const PairKey key{t.local_desc.id, t.remote_desc.id};
    auto it = pair_to_group.find(key);
    size_t gi;
    if (it == pair_to_group.end()) {
      gi = groups.size();
      pair_to_group.emplace(key, gi);
      PairGroup g;
      g.local_desc = t.local_desc;
      g.remote_desc = t.remote_desc;
      groups.push_back(std::move(g));
    } else {
      gi = it->second;
    }
    PairGroup& g = groups[gi];
    const size_t lo = static_cast<size_t>(t.local_offset);
    const size_t ro = static_cast<size_t>(t.remote_offset);
    const size_t sz = static_cast<size_t>(t.size);
    // Coalesce with the previous segment when BOTH local and remote are
    // exactly contiguous.  This is NOT redundant with the backend's WR
    // merging: the backend would fold these into the same WR either way, but
    // doing it here shrinks the inner SG vector it has to sort/merge
    // (O(M log M)) and allocate, which matters when M is large (big batch x
    // pages).  Same bytes, so per-pair failure granularity is unchanged.
    // The merged size must stay within uint32_t since the backend stores it
    // in ibv_sge.length (matches its own WR-merge cap in common.cpp).
    const bool can_coalesce =
        !g.sizes.empty() && AdjacentNoOverflow(g.local_offsets.back(), g.sizes.back(), lo) &&
        AdjacentNoOverflow(g.remote_offsets.back(), g.sizes.back(), ro) &&
        g.sizes.back() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - sz;
    if (can_coalesce) {
      g.sizes.back() += sz;
    } else {
      g.local_offsets.push_back(lo);
      g.remote_offsets.push_back(ro);
      g.sizes.push_back(sz);
    }
    // De-dup contributing entries: a single entry's pages arrive
    // consecutively within a pair, so a back() check suffices.
    if (g.entry_indices.empty() || g.entry_indices.back() != t.entry_index) {
      g.entry_indices.push_back(t.entry_index);
    }
  }
  return groups;
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
  conn->peer_address = peer_address;
  // engine_desc is hydrated lazily in EnsurePeerServiceConnection from
  // the peer's GetPeerInfo response.
  auto& ref = *conn;
  peers_[node_id] = std::move(conn);
  return ref;
}

void PoolClient::EnsureBufferDescsCachedLocked(PeerConnection& peer,
                                               const std::vector<BufferMemoryDescBytes>& descs) {
  // Caller holds peers_mutex_.
  if (!io_engine_) return;
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

void PoolClient::EnsureBufferDescsCached(PeerConnection& peer,
                                         const std::vector<BufferMemoryDescBytes>& descs) {
  std::lock_guard<std::mutex> lock(peers_mutex_);
  EnsureBufferDescsCachedLocked(peer, descs);
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
//  Peer connection setup
// ---------------------------------------------------------------------------

bool PoolClient::EnsurePeerServiceConnection(PeerConnection& peer) {
  std::lock_guard<std::mutex> lock(peer.conn_mutex);
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

}  // namespace mori::umbp

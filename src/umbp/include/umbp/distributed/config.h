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
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/pool/policy_config.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Forward declarations for strategy interfaces used by MasterServerConfig.
class RouteGetStrategy;
class RoutePutStrategy;
class MasterEvictStrategy;

// The user-facing medium selector, lowered to the tier id the routing plane
// and the wire already speak.  Two enums rather than one because the
// dependency is one-directional (distributed/config.h -> common/config.h) and
// UMBPMedium must be nameable by callers that never include the distributed
// headers.
inline TierType ToTierType(UMBPMedium medium) {
  switch (medium) {
    case UMBPMedium::HBM:
      return TierType::HBM;
    case UMBPMedium::SSD:
      return TierType::SSD;
    case UMBPMedium::DRAM:
      break;
  }
  return TierType::DRAM;
}

struct ClientRegistryConfig {
  std::chrono::seconds heartbeat_ttl{10};
  std::chrono::seconds reaper_interval{5};
  uint32_t max_missed_heartbeats = 3;

  // Overlay UMBP_* env vars on top of the defaults.  Fields are left
  // untouched when the corresponding env is unset or invalid.
  static ClientRegistryConfig FromEnvironment() {
    ClientRegistryConfig cfg;
    cfg.heartbeat_ttl =
        GetEnvSeconds("UMBP_HEARTBEAT_TTL_SEC", cfg.heartbeat_ttl, /*min_allowed=*/1);
    cfg.reaper_interval =
        GetEnvSeconds("UMBP_REAPER_INTERVAL_SEC", cfg.reaper_interval, /*min_allowed=*/1);
    cfg.max_missed_heartbeats =
        GetEnvUint32("UMBP_MAX_MISSED_HEARTBEATS", cfg.max_missed_heartbeats, /*min_allowed=*/1);
    return cfg;
  }

  // Sole source of truth for the DRAM/HBM page_size used by every
  // PageBitmapAllocator the registry creates when the registering Client
  // did not specify its own (RegisterClientRequest.dram_page_size == 0).
  // All nodes within the same tier must agree on page_size.  Upper layers
  // (UMBPDistributedConfig / PoolClientConfig) default their
  // `dram_page_size` to 0 and rely on this value to materialize.
  uint64_t default_dram_page_size = 2ULL * 1024 * 1024;  // 2 MiB

  // Current Masters advertise this so weighted peers may heartbeat aggregate
  // same-tier capacity together with max_allocatable_bytes. A pre-weighted
  // Master leaves it false (proto3 default); peers then keep heartbeats at the
  // first instance per tier so rolling upgrades cannot over-admit a value no
  // single backend can hold. Tests set this false to emulate that Master.
  bool advertise_max_allocatable_bytes = true;
  bool advertise_logical_tiers = true;
};

struct EvictionConfig {
  double high_watermark = 0.9;
  double low_watermark = 0.7;
  std::chrono::seconds check_interval{5};
  std::chrono::seconds lease_duration{2};
  size_t evict_batch_size = 32;

  // Only timing fields are env-overridable here; watermarks and batch size
  // have dedicated tuning paths and are intentionally excluded.
  static EvictionConfig FromEnvironment() {
    EvictionConfig cfg;
    cfg.check_interval =
        GetEnvSeconds("UMBP_EVICTION_CHECK_INTERVAL_SEC", cfg.check_interval, /*min_allowed=*/1);
    cfg.lease_duration =
        GetEnvSeconds("UMBP_LEASE_DURATION_SEC", cfg.lease_duration, /*min_allowed=*/1);
    return cfg;
  }
};

struct MasterServerConfig {
  std::string listen_address = "0.0.0.0:50051";
  int metrics_port = 0;  // 0 = disabled; set to a positive port to enable
  ClientRegistryConfig registry_config;
  EvictionConfig eviction_config;

  std::unique_ptr<RouteGetStrategy> get_strategy;
  std::unique_ptr<RoutePutStrategy> put_strategy;

  // Master-side DRAM/HBM eviction policy (optional code-level plugin).  Null
  // installs the default LruMasterEvictStrategy.  FromEnvironment() leaves it
  // null — only LRU exists today, so an env knob would be pseudo-config.
  std::unique_ptr<MasterEvictStrategy> evict_strategy;

  // Resolved put-strategy knobs, kept as strings for startup logging because a
  // unique_ptr<RoutePutStrategy> is not cheaply introspectable.  Populated by
  // FromEnvironment() alongside put_strategy.
  std::string route_put_algo = "most_available";
  std::string route_put_affinity = "none";

  // Composes ClientRegistryConfig::FromEnvironment() and
  // EvictionConfig::FromEnvironment().  listen_address is NOT read from env
  // here; callers (e.g. bin/master_main.cpp) apply argv overrides after
  // this call so the CLI remains the source of truth.
  //
  // Definition is out-of-line in master_server.cpp because this struct owns
  // unique_ptrs to forward-declared strategy types (RouteGetStrategy,
  // RoutePutStrategy, MasterEvictStrategy); an inline body would force
  // ~MasterServerConfig to be instantiated in every TU that includes this
  // header, where those types are incomplete.
  static MasterServerConfig FromEnvironment();

  // Special members are user-declared and defined out-of-line in
  // master_server.cpp.  This struct owns unique_ptrs to forward-declared
  // strategy types (RouteGetStrategy / RoutePutStrategy / MasterEvictStrategy),
  // so the destructor and move operations must be emitted in a TU where those
  // types are complete — not implicitly instantiated in every includer of this
  // header (which would require each to include all three strategy headers).
  MasterServerConfig();
  ~MasterServerConfig();
  MasterServerConfig(MasterServerConfig&&) noexcept;
  MasterServerConfig& operator=(MasterServerConfig&&) noexcept;
};

// Ownership config for this node's DRAM/HBM backend pool(s) — sizes only.
// PageBackend::Init(TransferEngine*) self-allocates one HostMemAllocator
// buffer per entry in `buffer_sizes` and registers it with the transfer
// engine; PoolClientConfig never sees a buffer pointer (backend-agnostic
// refactor Phase 2b — see design-backend-agnostic-refactor.md §1 item 4).
// `buffer_sizes` usually holds exactly one entry (one pool); more than one
// exercises PageBitmapAllocator's cross-buffer scatter/gather strategy.
struct DramOwnershipConfig {
  std::vector<uint64_t> buffer_sizes;
  bool use_hugepages = false;
  uint64_t hugepage_size = 2ULL * 1024 * 1024;
  int numa_node = -1;
  bool prefault = true;
};

// HBM-tier ownership knobs.  Deliberately NOT a superset of the DRAM ones:
// hipMalloc has no hugepage, NUMA or prefault dimension, and the device ordinal
// it does have is meaningless for host memory.  That asymmetry is the reason
// each medium brings its own PageMemorySource rather than sharing one options
// struct with per-medium fields nobody else reads.
//
// No `enabled` flag: PoolClientConfig::medium selects the one live medium, and
// these knobs are read only when it names HBM.
struct HbmOwnershipConfig {
  // GPU ordinal the pool is allocated on.  Fixed at Init, not inherited from
  // whichever thread allocates first (see HbmPageMemorySource).
  int device = 0;
  std::vector<uint64_t> buffer_sizes;
};

// SSD-tier construction parameters lowered from the user-facing UMBPConfig.
// SSDTier depends on UMBPSsdConfig (io backend/queue_depth, segment_size,
// durability, storage_dir, capacity, watermarks, backend selection), so the
// peer only needs that subset — not the whole global config.  ssd_backend
// (posix / spdk / spdk_proxy) lives inside UMBPSsdConfig, so PeerSsdManager
// picks the backend from cfg.ssd directly.
//
// `enabled` stays because PeerSsdManager's own contract is written against it
// (it refuses to open a tier that is off); PoolClient sets it from
// PoolClientConfig::medium rather than from user config.
struct PeerSsdConfig {
  bool enabled = false;
  UMBPSsdConfig ssd;
};

enum class PoolPlacementPolicy {
  SINGLE_BACKEND = 0,
  WEIGHTED = 1,
  TIERED = 2,
};

// One named backend instance on a peer. Only the ownership block selected by
// `tier` is read. Page size remains peer-global because the RPC wire carries a
// single page_size for batches that may span instances.
struct BackendInstanceConfig {
  std::string name;
  TierType tier = TierType::UNKNOWN;
  DramOwnershipConfig dram;
  HbmOwnershipConfig hbm;
  PeerSsdConfig ssd;
  int ssd_staging_buffer_slots = 16;
  // Relative admission share within this tier when WEIGHTED placement is
  // enabled. Must be positive; legacy and SINGLE_BACKEND paths ignore it.
  // Appended to preserve existing positional aggregate initializers.
  uint32_t placement_weight = 1;
};

struct PoolClientConfig {
  UMBPMasterClientConfig master_config;
  UMBPIoEngineConfig io_engine;

  size_t staging_buffer_size = 64ULL * 1024 * 1024;

  // Caller-owned, RDMA-registered host arenas for ranged I/O. DistributedClient
  // owns the backing mappings and frees them only after PoolClient::Shutdown has
  // deregistered the regions. Direct PoolClient tests may provide their own.
  //
  // Two separate arenas — one for remote ranged GET, one for remote ranged PUT
  // — each under its own mutex in PoolClient, so a remote get and a remote put
  // run concurrently instead of serializing on one lock (the load/offload
  // overlap sglang's direct linker wants). Each must hold at least one whole
  // object. Both zero keeps ranged remote I/O disabled with no host memory
  // registered; SupportsRangedIO() requires both to be set.
  void* ranged_get_scratch_buffer = nullptr;
  size_t ranged_get_scratch_size = 0;
  void* ranged_put_scratch_buffer = nullptr;
  size_t ranged_put_scratch_size = 0;

  // SSD read-staging tuning. More slots increase concurrent reads and writes;
  // each policy-created SsdBackend owns `slots * page_size` staging bytes. Its
  // read lease is resolved from UMBP_SSD_READ_LEASE_MS during PoolClient::Init.
  int ssd_staging_buffer_slots = 16;

  // Backs ssd_staging_buffer_, allocated only when ssd.enabled. A remote SSD
  // read fits one whole key value in a slot, so this / ssd_staging_buffer_slots
  // must be >= the largest single-key page KV (61-layer MLA page ~= 4.5 MB).
  size_t ssd_staging_buffer_size = 268435456;  // 256 MiB

  // Back the SSD staging arena with hugetlbfs pages. Every SSD backend this
  // client registers uses the same node-wide setting. Falls back to 4 KiB
  // pages when no hugetlb pages are free so the node can still start.
  bool ssd_staging_use_hugepages = false;
  size_t ssd_staging_hugepage_size = 2ULL * 1024 * 1024;  // 2 MiB

  // Legacy one-medium selector. Used only when `backends` is empty; in that
  // mode PoolClient::Init reads exactly one ownership block below.
  TierType medium = TierType::DRAM;

  DramOwnershipConfig dram;
  HbmOwnershipConfig hbm;
  PeerSsdConfig ssd;

  // Named multi-backend configuration. Empty selects the legacy fields above;
  // PoolClient lowers them at Init time so mutations made after
  // ToPoolClientConfig() remain authoritative.
  std::vector<BackendInstanceConfig> backends;

  uint16_t peer_service_port = 0;

  // Re-cache remotely-fetched blocks into local DRAM + publish to master so this
  // node becomes an additional serving replica (LMCache-style P2P pull).
  bool cache_remote_fetches = true;
  CacheRemoteAdmission cache_remote_admission = CacheRemoteAdmission::SIZE;
  size_t admission_max_block_bytes = 16ULL * 1024 * 1024;

  // Background whole-object pull after a remote ranged read; see
  // UMBPDistributedConfig::ranged_locality_prefetch for why it is not folded
  // into cache_remote_fetches.
  bool ranged_locality_prefetch = true;

  // Resolve on this node's own media before asking the master to route, and
  // skip the master entirely when the whole batch is local.  See
  // UMBPDistributedConfig::local_first for the reasoning and the multi-node
  // trade-off.
  bool local_first = true;

  // Page size used by Master's PageBitmapAllocator for this node's DRAM/HBM
  // tier.  Reported via RegisterClient.  Same value applies to both DRAM
  // and HBM.  Forwarded unmodified to MasterClient::RegisterSelf by
  // PoolClient::Init — PoolClient MUST NOT substitute a default here.
  // 0 = delegate to Master's ClientRegistryConfig::default_dram_page_size
  // (2 MiB by default).  Set to an explicit byte count to override.
  uint64_t dram_page_size = 0;

  UMBPCopyPipelineConfig copy_pipeline = [] {
    UMBPCopyPipelineConfig c;
    c.worker_threads = 1;
    return c;
  }();

  // Compatibility is the default. WEIGHTED distributes new keys among all
  // configured same-tier instances according to placement_weight. Appended to
  // preserve existing positional aggregate initializers.
  PoolPlacementPolicy placement_policy = PoolPlacementPolicy::SINGLE_BACKEND;

  // Opt-in ephemeral peer service for test/benchmark harnesses. When true,
  // PoolClient starts the service even when peer_service_port is zero and
  // advertises the port selected atomically by gRPC, avoiding probe/bind races.
  bool auto_peer_service_port = false;

  // Optional declarative policy. policy_config_path is loaded at Init and
  // lowered into named backends plus logical_tiers. Callers that load JSON
  // themselves may populate logical_tiers directly. Empty fields preserve the
  // legacy SINGLE_BACKEND/WEIGHTED behavior. Appended for aggregate-init
  // compatibility.
  std::string policy_config_path;
  std::vector<LogicalTierConfig> logical_tiers;

  // Optional production traffic recorder. Empty path disables recording.
  std::string workload_trace_path;
  uint32_t workload_trace_client_id = 0;
  uint64_t workload_trace_seed = 0;
};

// Lower a user-facing UMBPDistributedConfig to the internal PoolClientConfig.
// Kept as a free function (not a member of UMBPDistributedConfig) so that
// common/config.h does not need to include distributed/config.h — the
// dependency is one-directional: distributed/config.h -> common/config.h.
// `dram` is caller-supplied (sizes only, no pointer — see DramOwnershipConfig)
// because DistributedClient is the one place that knows the top-level
// UMBPConfig::dram ownership knobs (hugepages/numa/prefault), which sit
// beside — not inside — UMBPDistributedConfig.  PoolClient's DRAM PageBackend
// self-allocates from this at Init(); tier capacities are no longer
// caller-supplied at all — they are derived from BackendRegistry::Capacity()
// after Init (backend-agnostic refactor Phase 2).
inline PoolClientConfig ToPoolClientConfig(const UMBPDistributedConfig& dc,
                                           DramOwnershipConfig dram, PeerSsdConfig ssd = {}) {
  PoolClientConfig pc;
  pc.master_config = dc.master_config;
  pc.io_engine = dc.io_engine;
  pc.staging_buffer_size = dc.staging_buffer_size;
  // The ranged scratch buffers/sizes are set by DistributedClient after it
  // allocates and registers the two arenas (see DistributedClient ctor).
  pc.ssd_staging_buffer_size = dc.ssd_staging_buffer_size;
  pc.ssd_staging_buffer_slots = dc.ssd_staging_buffer_slots;
  pc.ssd_staging_use_hugepages = dc.ssd_staging_use_hugepages;
  pc.ssd_staging_hugepage_size = dc.ssd_staging_hugepage_size;
  pc.peer_service_port = dc.peer_service_port;
  pc.cache_remote_fetches = dc.cache_remote_fetches;
  pc.cache_remote_admission = dc.cache_remote_admission;
  pc.admission_max_block_bytes = dc.admission_max_block_bytes;
  pc.ranged_locality_prefetch = dc.ranged_locality_prefetch;
  pc.local_first = dc.local_first;
  // 0 propagates through PoolClient -> MasterClient::RegisterSelf ->
  // proto -> ClientRegistry, where it is interpreted as "use the
  // registry-wide default_dram_page_size".
  pc.dram_page_size = dc.dram_page_size;
  pc.policy_config_path = dc.backend_policy_path;
  pc.workload_trace_path = dc.workload_trace_path;
  pc.workload_trace_client_id = dc.workload_trace_client_id;
  pc.workload_trace_seed = dc.workload_trace_seed;
  pc.dram = std::move(dram);
  pc.ssd = std::move(ssd);
  // Unlike dram/ssd, UMBPHbmConfig carries no ownership knobs that live
  // outside UMBPDistributedConfig (no hugepages/NUMA/prefault dimension for
  // hipMalloc'd memory), so it can be lowered directly here instead of via a
  // caller-supplied parameter.
  pc.hbm.device = dc.hbm.device;
  if (dc.hbm.capacity_bytes > 0) pc.hbm.buffer_sizes = {dc.hbm.capacity_bytes};
  pc.medium = ToTierType(dc.medium);
  // The medium is what opts the SSD tier in; PeerSsdManager keys off this.
  pc.ssd.enabled = (pc.medium == TierType::SSD);
  // Deliberately leave pc.backends empty. PoolClient synthesizes the legacy
  // instance at Init so callers may still mutate medium/dram/hbm/ssd afterward.
  return pc;
}

}  // namespace mori::umbp

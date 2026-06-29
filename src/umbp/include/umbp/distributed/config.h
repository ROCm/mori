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
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Forward declarations for strategy interfaces used by MasterServerConfig.
class RouteGetStrategy;
class RoutePutStrategy;
class MasterEvictStrategy;

struct ClientRegistryConfig {
  std::chrono::seconds heartbeat_ttl{10};
  std::chrono::seconds reaper_interval{5};
  std::chrono::seconds allocation_ttl{30};
  std::chrono::seconds finalized_record_ttl{120};
  uint32_t max_missed_heartbeats = 3;

  // Overlay UMBP_* env vars on top of the defaults.  Fields are left
  // untouched when the corresponding env is unset or invalid.
  static ClientRegistryConfig FromEnvironment() {
    ClientRegistryConfig cfg;
    cfg.heartbeat_ttl =
        GetEnvSeconds("UMBP_HEARTBEAT_TTL_SEC", cfg.heartbeat_ttl, /*min_allowed=*/1);
    cfg.reaper_interval =
        GetEnvSeconds("UMBP_REAPER_INTERVAL_SEC", cfg.reaper_interval, /*min_allowed=*/1);
    cfg.allocation_ttl =
        GetEnvSeconds("UMBP_ALLOCATION_TTL_SEC", cfg.allocation_ttl, /*min_allowed=*/1);
    cfg.finalized_record_ttl =
        GetEnvSeconds("UMBP_FINALIZED_RECORD_TTL_SEC", cfg.finalized_record_ttl, /*min_allowed=*/1);
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
};

struct EvictionConfig {
  double high_watermark = 0.9;
  double low_watermark = 0.7;
  std::chrono::seconds check_interval{5};
  std::chrono::seconds lease_duration{10};
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

struct ExportableDram {
  void* buffer = nullptr;
  size_t size = 0;
};

// SSD-tier construction parameters lowered from the user-facing UMBPConfig.
// SSDTier depends on UMBPSsdConfig (io backend/queue_depth, segment_size,
// durability, storage_dir, capacity, watermarks, backend selection), so the
// peer only needs that subset — not the whole global config.  ssd_backend
// (posix / spdk / spdk_proxy) lives inside UMBPSsdConfig, so PeerSsdManager
// picks the backend from cfg.ssd directly.
struct PeerSsdConfig {
  bool enabled = false;
  UMBPSsdConfig ssd;
};

struct PoolClientConfig {
  UMBPMasterClientConfig master_config;
  UMBPIoEngineConfig io_engine;

  size_t staging_buffer_size = 64ULL * 1024 * 1024;

  // SSD read-staging tuning (peer side).  More slots reduce NO_SLOT under large
  // concurrent prefetch batches, but shrink per-slot size (= staging_buffer_size
  // / slots), which must stay >= the largest single SSD block.  The lease TTL
  // is the primary slot-reclaim mechanism (ReleaseSsdLease is best-effort), so
  // it should comfortably exceed one SSD read's latency.
  int ssd_staging_buffer_slots = 16;
  int ssd_lease_timeout_s = 10;

  // Backs ssd_staging_buffer_, allocated only when ssd.enabled. A remote SSD
  // read fits one whole key value in a slot, so this / ssd_staging_buffer_slots
  // must be >= the largest single-key page KV (61-layer MLA page ~= 4.5 MB).
  size_t ssd_staging_buffer_size = 268435456;  // 256 MiB

  std::vector<ExportableDram> dram_buffers;
  PeerSsdConfig ssd;

  std::map<TierType, TierCapacity> tier_capacities;

  uint16_t peer_service_port = 0;

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
};

// Lower a user-facing UMBPDistributedConfig to the internal PoolClientConfig.
// Kept as a free function (not a member of UMBPDistributedConfig) so that
// common/config.h does not need to include distributed/config.h — the
// dependency is one-directional: distributed/config.h -> common/config.h.
// DRAM buffers and tier capacities are caller-supplied because they live in
// DistributedClient (pool mmap'd memory), not in the user-facing config.
inline PoolClientConfig ToPoolClientConfig(const UMBPDistributedConfig& dc,
                                           std::vector<ExportableDram> dram_buffers,
                                           std::map<TierType, TierCapacity> tier_capacities,
                                           PeerSsdConfig ssd = {}) {
  PoolClientConfig pc;
  pc.master_config = dc.master_config;
  pc.io_engine = dc.io_engine;
  pc.staging_buffer_size = dc.staging_buffer_size;
  pc.ssd_staging_buffer_size = dc.ssd_staging_buffer_size;
  pc.ssd_staging_buffer_slots = dc.ssd_staging_buffer_slots;
  pc.peer_service_port = dc.peer_service_port;
  // 0 propagates through PoolClient -> MasterClient::RegisterSelf ->
  // proto -> ClientRegistry, where it is interpreted as "use the
  // registry-wide default_dram_page_size".
  pc.dram_page_size = dc.dram_page_size;
  pc.dram_buffers = std::move(dram_buffers);
  pc.tier_capacities = std::move(tier_capacities);
  pc.ssd = std::move(ssd);
  return pc;
}

}  // namespace mori::umbp

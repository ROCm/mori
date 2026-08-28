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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace mori::umbp {

enum class UMBPRole : int {
  Standalone = 0,
  SharedSSDLeader = 1,
  SharedSSDFollower = 2,
};

static constexpr uint32_t kAutoRankId = UINT32_MAX;

enum class UMBPSsdLayoutMode : int {
  SegmentedLog = 1,
};

enum class UMBPIoBackend : int {
  Posix = 0,
  IoUring = 1,
};

enum class UMBPDurabilityMode : int {
  Strict = 0,
  Relaxed = 1,
};

struct UMBPDramConfig {
  size_t capacity_bytes = 4ULL * 1024 * 1024 * 1024;
  bool use_shared_memory = false;
  std::string shm_name = "/umbp_dram";
  double high_watermark = 0.9;
  double low_watermark = 0.7;

  // Host memory options (ignored when use_shared_memory=true).
  bool use_hugepages = false;
  size_t hugepage_size = 2ULL * 1024 * 1024;  // 2 MiB
  int numa_node = -1;                         // -1 = no NUMA binding
  bool prefault = true;
};

struct UMBPIoConfig {
  UMBPIoBackend backend = UMBPIoBackend::IoUring;
  size_t queue_depth = 4096;
};

struct UMBPDurabilityConfig {
  UMBPDurabilityMode mode = UMBPDurabilityMode::Strict;
  bool enable_background_gc = true;
};

struct UMBPSsdConfig {
  bool enabled = true;
  // Comma-separated (same convention as spdk_nvme_pci_addr).  Naming >1 directory
  // — normally one mount point per drive — fans the tier across them via
  // ShardedSsdTier for their aggregate bandwidth.  capacity_bytes is the TOTAL,
  // split evenly among the directories.
  std::string storage_dir = "/tmp/umbp_ssd";
  size_t capacity_bytes = 32ULL * 1024 * 1024 * 1024;
  UMBPSsdLayoutMode layout_mode = UMBPSsdLayoutMode::SegmentedLog;
  size_t segment_size_bytes = 256ULL * 1024 * 1024;
  UMBPIoConfig io;
  UMBPDurabilityConfig durability;

  // Threads for the sharded tier's batch paths, i.e. ACROSS drives.  0 = one per
  // shard, which is what saturates N drives.  Ignored with a single directory.
  int shard_io_threads = 0;

  // Threads for the CPU-bound phases WITHIN one SSDTier (CRC verify on read, CRC
  // + record assembly on write).  Defaults to 4 to match the DRAM tier, so a
  // DRAM-vs-SSD comparison is not silently 4 threads against 1.
  int tier_io_threads = 4;

  // Bypass the page cache (O_DIRECT).  On by default: the tier is itself a cache,
  // so buffered I/O adds a second unmanaged DRAM cache that steals host memory
  // from the DRAM tier and makes any drive measurement meaningless (a run can be
  // served entirely from RAM, moving zero bytes to the device).  Safe as a default
  // because the tier probes O_DIRECT at startup and falls back to buffered with a
  // warning on tmpfs/overlayfs.  Set false to deliberately exploit the page cache.
  bool direct_io = true;

  // Coalesce concurrent reads of the same key: only the first touches the drive,
  // the rest are served by memcpy from its buffer.  For MLA + TP, where every
  // attention-TP rank GETs a byte-identical key; MHA keys carry a per-rank suffix
  // and never collide, so it is a no-op there.  NOTE: UMBP_SSD_SINGLE_FLIGHT only
  // reaches configs built by FromEnvironment() (the standalone server); sglang
  // constructs UMBPConfig directly and needs the pybind field.  [SsdPerf/peer]
  // reports merged=/merged_total=, which is what tells you it fired.
  bool single_flight_reads = true;

  // Checksum on write, verify on read.  Turning it off isolates how much of the
  // SSD path's cost is integrity work (the DRAM tier does none).  Records written
  // with it off are marked kFlagNoCrc and stay readable either way.
  bool verify_crc = true;

  // Split storage_dir on ',' — one entry per drive.  Always returns at least
  // one element (falls back to the default when the string is empty).
  std::vector<std::string> StorageDirs() const {
    std::vector<std::string> dirs;
    size_t start = 0;
    while (start <= storage_dir.size()) {
      size_t comma = storage_dir.find(',', start);
      if (comma == std::string::npos) comma = storage_dir.size();
      std::string part = storage_dir.substr(start, comma - start);
      // Trim surrounding whitespace so "a, b" works as well as "a,b".
      size_t b = part.find_first_not_of(" \t");
      size_t e = part.find_last_not_of(" \t");
      if (b != std::string::npos) dirs.push_back(part.substr(b, e - b + 1));
      if (comma == storage_dir.size()) break;
      start = comma + 1;
    }
    if (dirs.empty()) dirs.push_back("/tmp/umbp_ssd");
    return dirs;
  }

  // Local SSD-tier capacity watermarks for the distributed PeerSsdManager's
  // local eviction.  When used/total crosses high_watermark the owner
  // peer evicts its oldest keys down to low_watermark.  Mirrors the DRAM tier's
  // env-tunable convention (UMBP_DRAM_HIGH_WM / LOW_WM); NOT the master-side
  // EvictionConfig (whose watermarks are intentionally not env-tunable).
  double high_watermark = 0.9;
  double low_watermark = 0.7;

  // SSD backend selection. "file" uses the segmented-log SSDTier; "spdk" /
  // "spdk_proxy" use the SPDK NVMe path (direct SpdkSsdTier in standalone, or
  // SpdkProxyTier when sharing the device across processes).  Kept here (rather
  // than at UMBPConfig top level) so both the standalone LocalStorageManager and
  // the distributed PeerSsdManager select the backend from the same config.
  std::string ssd_backend = "file";       // "file", "spdk" or "spdk_proxy"
  std::string spdk_bdev_name;             // e.g. "Malloc0" or "NVMe0n1"
  std::string spdk_reactor_mask = "0x1";  // CPU core mask for SPDK reactors
  int spdk_mem_size_mb = 256;             // DPDK hugepage limit (MB)
  std::string spdk_nvme_pci_addr;         // PCI BDF, e.g. "0000:47:00.0"
  std::string spdk_nvme_ctrl_name = "NVMe0";
  int spdk_io_workers = 4;  // Internal I/O worker threads for SpdkSsdTier batch ops

  // SPDK Proxy configuration
  std::string spdk_proxy_shm_name = "/umbp_spdk_proxy";
  uint32_t spdk_proxy_tenant_id = 0;
  size_t spdk_proxy_tenant_quota_bytes = 0;
  uint32_t spdk_proxy_max_channels = 8;
  size_t spdk_proxy_data_per_channel_mb = 32;  // MB of SHM data region per channel
  std::string spdk_proxy_bin;                  // Path to spdk_proxy binary (empty = search PATH)
  int spdk_proxy_startup_timeout_ms = 30000;   // Max ms to wait for proxy READY
  bool spdk_proxy_auto_start = true;
  int spdk_proxy_idle_exit_timeout_ms = 30000;
  bool spdk_proxy_allow_borrow = false;
  size_t spdk_proxy_reserved_shared_bytes = 0;

  // Focused validation for the SSD tier alone (used by SSDTier, which depends
  // on UMBPSsdConfig rather than the whole UMBPConfig).  UMBPConfig::Validate()
  // remains the global validator.
  bool Validate(std::string* error_message = nullptr) const {
    // capacity_bytes == 0 is legal when SSD is not in use; only enforce sizing
    // when the tier is actually enabled (mirrors UMBPConfig::Validate's
    // `if (ssd.enabled)` gate).
    if (!enabled) return true;
    if (ssd_backend != "file" && ssd_backend != "spdk" && ssd_backend != "spdk_proxy" &&
        ssd_backend != "dummy_storage") {
      if (error_message)
        *error_message = "ssd.ssd_backend must be one of: file, spdk, spdk_proxy, dummy_storage";
      return false;
    }
    if (capacity_bytes == 0) {
      if (error_message) *error_message = "ssd.capacity_bytes must be > 0";
      return false;
    }
    if (segment_size_bytes == 0) {
      if (error_message) *error_message = "ssd.segment_size_bytes must be > 0";
      return false;
    }
    // Records are padded to segment::kRecordAlign (4096), so a segment whose
    // size is not a multiple of it would leave the append cursor unaligned at
    // the roll-over and break direct I/O on the next segment.
    if (segment_size_bytes % 4096 != 0) {
      if (error_message) *error_message = "ssd.segment_size_bytes must be a multiple of 4096";
      return false;
    }
    // Watermarks must satisfy 0 < low < high <= 1.  Fail fast on a misconfigured
    // value rather than silently clamping (a clamp would hide the config error).
    if (!(high_watermark > 0.0 && high_watermark <= 1.0 && low_watermark > 0.0 &&
          low_watermark < high_watermark)) {
      if (error_message)
        *error_message = "ssd watermarks must satisfy 0 < low_watermark < high_watermark <= 1";
      return false;
    }
    return true;
  }
};

struct UMBPEvictionConfig {
  std::string policy = "lru";
  size_t candidate_window = 16;
  bool auto_promote_on_read = true;
};

struct UMBPCopyPipelineConfig {
  bool async_enabled = true;
  size_t queue_depth = 4096;
  size_t worker_threads = 2;
  size_t batch_max_ops = 128;
};

// Master-control-plane client parameters.  Shared between user-facing
// UMBPDistributedConfig and the internal PoolClientConfig/MasterClient.
struct UMBPMasterClientConfig {
  std::string master_address;  // e.g. "master-host:50051"
  std::string node_id;         // unique node identifier
  std::string node_address;    // this node's reachable address for peers
  bool auto_heartbeat = true;  // start heartbeat thread on Init
  // Opaque key=value strings forwarded to master on RegisterClient and
  // attached to all metrics emitted for this node.  e.g. "sgl_role=prefill".
  std::vector<std::string> tags;
};

// RDMA IO-engine endpoint parameters.
struct UMBPIoEngineConfig {
  std::string host;   // RDMA engine hostname (formerly UMBPDistributedConfig::io_engine_host)
  uint16_t port = 0;  // RDMA engine port; 0 = OS-assigned ephemeral port (formerly io_engine_port)
};

// Admission policy for re-caching remotely-fetched blocks locally.
enum class CacheRemoteAdmission : int {
  SIZE = 0,    // admit if block size <= admission_max_block_bytes AND DRAM has room
  NEVER = 1,   // never re-cache (equivalent to cache_remote_fetches = false)
  ALWAYS = 2,  // always attempt re-cache (skip size gate)
};

// Pure admission predicate for the remote-fetch re-cache gate: decides whether a
// block of `size` bytes is eligible for local re-caching under the given policy,
// independent of runtime state (DRAM-capacity enforcement is left to the
// allocator). Extracted so the gate is unit-testable without a live PoolClient;
// used by PoolClient::MaybeReCacheAfterRemote.
inline bool ShouldAdmitReCache(bool cache_remote_fetches, CacheRemoteAdmission policy,
                               size_t admission_max_block_bytes, size_t size) {
  if (!cache_remote_fetches) return false;
  if (size == 0) return false;
  if (policy == CacheRemoteAdmission::NEVER) return false;
  if (policy == CacheRemoteAdmission::SIZE) {
    if (admission_max_block_bytes > 0 && size > admission_max_block_bytes) return false;
  }
  return true;  // ALWAYS, or SIZE within cap
}

// User-facing HBM-tier knobs for distributed mode. Unlike dram/ssd, HBM has
// no local (non-distributed) mode, so it lives only here rather than on the
// top-level UMBPConfig. Mirrors HbmOwnershipConfig's shape (device +
// buffer_sizes) deliberately: hipMalloc has no hugepage/NUMA/prefault
// dimension the way host memory does, so there is nothing else to expose.
// No `enabled` flag: UMBPDistributedConfig::medium is the single selector,
// and these knobs are read only when it names HBM.
struct UMBPHbmConfig {
  int device = 0;               // GPU ordinal the pool is allocated on
  uint64_t capacity_bytes = 0;  // single-buffer pool size
};

// The one storage medium a distributed node serves.
//
// UMBP's routing plane does not tier: every advertised medium is an equally
// valid put target (see medium_backend.h and Phase 4 of the backend-agnostic
// refactor), so a node registering two backends MIRRORS across them rather
// than promoting/demoting between them — which is not what "DRAM + SSD" reads
// like, and costs capacity for nothing. Rather than build a local tiering
// policy nobody asked for, a node picks exactly one medium and the cluster
// gets its heterogeneity from having different nodes pick differently.
//
// The medium each key lands on is therefore a property of the node master
// routed it to, not of a per-node tier order.
enum class UMBPMedium {
  DRAM,  // host memory (default; the pre-selector behaviour)
  HBM,   // device memory on UMBPHbmConfig::device
  SSD,   // local NVMe/file storage, staged through registered host memory
};

// User-facing distributed configuration. Set UMBPConfig::distributed to enable
// distributed mode. Internally translated to PoolClientConfig by DistributedClient.
struct UMBPDistributedConfig {
  UMBPMasterClientConfig master_config;
  UMBPIoEngineConfig io_engine;

  size_t staging_buffer_size = 64ULL * 1024 * 1024;  // 64 MB

  // Registered host arena used by ranged multi-buffer I/O. Remote objects are
  // fetched into disjoint slices here before being installed into the local
  // medium; ranged puts routed to another node assemble their scattered ranges
  // into a matching arena. Purely a remote-path resource — ranged I/O served by
  // this node's own medium never touches it.
  //
  // This sizes EACH of the two arenas UMBP allocates: a separate GET arena and
  // PUT arena, each under its own mutex, so a remote ranged get and a remote
  // ranged put run concurrently instead of serializing on one lock. Each must
  // hold at least one whole object.
  //
  // Zero keeps ranged remote I/O disabled without allocating or registering
  // additional host memory; callers that need it must opt in explicitly. An
  // existing distributed deployment that never issues ranged I/O therefore
  // stops paying for an arena it does not use.
  size_t ranged_scratch_size = 0;

  // Dedicated SSD read staging, allocated only when medium == SSD. Per-slot
  // (this / ssd_staging_buffer_slots) must be >= the largest single-key page KV
  // (61-layer MLA page ~= 4.5 MB).
  size_t ssd_staging_buffer_size = 268435456;  // 256 MiB

  // Remote SSD read staging slots; per-slot = ssd_staging_buffer_size / this.
  int ssd_staging_buffer_slots = 16;

  // Back the SSD staging arena with hugetlbfs pages.  Every byte the SSD
  // backend moves crosses that arena twice (device <-> arena, arena <-> wire),
  // so its TLB behaviour sits on the critical path in a way an ordinary
  // buffer's does not.  Falls back to 4 KiB pages when no hugetlb pages are
  // free — a node must still come up.
  bool ssd_staging_use_hugepages = false;
  size_t ssd_staging_hugepage_size = 2ULL * 1024 * 1024;  // 2 MiB

  uint16_t peer_service_port = 0;  // gRPC peer service port

  bool cache_remote_fetches = true;  // cache remotely-fetched blocks locally

  // After a remote RANGED read, pull the whole object into this node's medium
  // in the background so the next read of the key is a local hit.
  //
  // Separate from cache_remote_fetches on purpose.  That flag gates a re-cache
  // that copies out of the caller's destination buffer, which a GPU-destination
  // deployment cannot use and therefore has to turn off; this one never touches
  // the caller's buffer — it reads the peer straight into a freshly allocated
  // local slot — so the same deployment can keep locality.
  //
  // The traffic is duplicate: the layer-wise reader will pull the same object a
  // slice at a time regardless, so this costs up to one extra copy of the
  // object on the wire, in exchange for the object being local for the NEXT
  // request. Best-effort throughout — bounded queue, drop on full, admission
  // gated by cache_remote_admission / admission_max_block_bytes.
  bool ranged_locality_prefetch = true;

  // Ask this node's own media before asking the master.
  //
  // A read is a two-part question: "who holds this key" (master) and "give me
  // the bytes" (whoever holds it).  When this node holds the key itself, the
  // first part is answered by its own backends, and the routing RPC only
  // confirms what a local resolve already knew.  A single-node deployment —
  // one peer, master on localhost — never gets a different answer, so every
  // routing round trip on that path is pure latency.
  //
  // With this on, BatchGet and BatchExists resolve locally first and contact
  // the master ONLY for the keys this node missed; a fully-local batch issues
  // no RPC at all.  A local hit is conclusive (the backends own the bytes), so
  // this cannot invent a hit; a local MISS is not conclusive — another node may
  // hold the key — which is why the misses still go to the master.
  //
  // The cost is on multi-node deployments: local and remote reads stop
  // overlapping (the local half is served before the routing RPC is issued
  // rather than inside its in-flight window), and a batch whose keys are mostly
  // remote pays a local resolve that mostly misses.  Set false to restore
  // route-first ordering.
  //
  // One visible consequence: on a node that holds the key, Exists() now answers
  // as soon as the put commits, rather than a heartbeat publication interval
  // later once the master's index has it.  That is a more current answer about
  // the key, but it means Exists() on the HOLDER is no longer a barrier for
  // "the rest of the cluster can route to this".  Poll the node that will
  // actually read it -- which has to go to the master -- when that is what is
  // being waited for.
  bool local_first = true;

  // Admission gate for re-caching. Only consulted when cache_remote_fetches is true.
  CacheRemoteAdmission cache_remote_admission = CacheRemoteAdmission::SIZE;

  // Maximum block size (bytes) eligible for local re-cache under SIZE policy.
  // 0 means unlimited (no size gate). Default 16 MB.
  size_t admission_max_block_bytes = 16ULL * 1024 * 1024;

  // Page size used by Master's PageBitmapAllocator for this node's medium.
  // Reported via RegisterClient.  Forwarded to PoolClientConfig::dram_page_size
  // by DistributedClient unmodified.  (Named for DRAM because the wire field
  // is; it applies to whichever medium this node serves.)
  // 0 = delegate to Master's ClientRegistryConfig::default_dram_page_size
  // (2 MiB by default).  Set to an explicit byte count to override.
  uint64_t dram_page_size = 0;

  // Which medium this node's distributed data plane serves — exactly one.
  // Defaults to DRAM, so an existing deployment is bit-identical.
  //
  // The selected medium's sizing knobs come from the config it names:
  //   DRAM -> UMBPConfig::dram   (capacity, hugepages, NUMA, prefault)
  //   HBM  -> distributed.hbm    (device, capacity)
  //   SSD  -> UMBPConfig::ssd    (storage_dir, capacity, layout, io, backend)
  // The other two are ignored rather than validated, so a config that carries
  // all three (a common deployment template) selects by this field alone.
  UMBPMedium medium = UMBPMedium::DRAM;

  UMBPHbmConfig hbm;
};

// User-facing same-host standalone-process configuration.  Set
// UMBPConfig::standalone_process to make CreateUMBPClient construct a
// StandaloneProcessClient that talks to an umbp_standalone_server over UDS.
struct UMBPStandaloneProcessConfig {
  std::string address;             // e.g. unix:///run/umbp/standalone/node0.grpc.sock
  bool auto_start = false;         // opt-in fork+exec convenience path
  int startup_timeout_ms = 30000;  // readiness wait bound for auto_start

  // Optional distributed identity for external-KV reports routed through a
  // distributed-backed standalone server. Empty means external-KV identity is
  // not requested for this worker.
  std::string worker_node_id;
  std::string worker_node_address;
  std::vector<std::string> tags;
};

struct UMBPConfig {
  UMBPDramConfig dram;
  UMBPSsdConfig ssd;
  UMBPEvictionConfig eviction;
  UMBPCopyPipelineConfig copy_pipeline;

  // Role is the source of truth for runtime behavior.
  UMBPRole role = UMBPRole::Standalone;

  // Backward compatibility fields for older Python/C++ callers.
  // New code should set `role` instead.
  bool follower_mode = false;
  bool force_ssd_copy_on_write = false;

  // Optional distributed mode. When set, DistributedClient wraps PoolClient
  // that connects to the Master and sends periodic heartbeats.
  // nullopt (default) = local-only mode with no network dependencies.
  std::optional<UMBPDistributedConfig> distributed;

  // Optional same-host standalone-process mode.  Mutually exclusive with
  // distributed: this mode uses one local server process and shm fd handoff,
  // not the cross-node master/RDMA path.
  std::optional<UMBPStandaloneProcessConfig> standalone_process;

  UMBPRole ResolveRole() const {
    if (role != UMBPRole::Standalone) {
      return role;
    }
    if (follower_mode) {
      return UMBPRole::SharedSSDFollower;
    }
    if (force_ssd_copy_on_write) {
      return UMBPRole::SharedSSDLeader;
    }
    return UMBPRole::Standalone;
  }

  bool Validate(std::string* error_message = nullptr) const {
    if (dram.capacity_bytes == 0) {
      if (error_message) *error_message = "dram.capacity_bytes must be > 0";
      return false;
    }
    if (ssd.enabled) {
      if (ssd.capacity_bytes == 0) {
        if (error_message) *error_message = "ssd.capacity_bytes must be > 0";
        return false;
      }
      if (ssd.segment_size_bytes == 0) {
        if (error_message) *error_message = "ssd.segment_size_bytes must be > 0";
        return false;
      }
    }
    if (dram.use_hugepages && dram.hugepage_size != 0 &&
        (dram.hugepage_size & (dram.hugepage_size - 1)) != 0) {
      if (error_message) *error_message = "dram.hugepage_size must be a power of two";
      return false;
    }
    if (copy_pipeline.queue_depth == 0) {
      if (error_message) *error_message = "copy_pipeline.queue_depth must be > 0";
      return false;
    }
    if (copy_pipeline.worker_threads == 0) {
      if (error_message) *error_message = "copy_pipeline.worker_threads must be > 0";
      return false;
    }
    if (copy_pipeline.batch_max_ops == 0) {
      if (error_message) *error_message = "copy_pipeline.batch_max_ops must be > 0";
      return false;
    }
    if (ssd.spdk_proxy_max_channels == 0) {
      if (error_message) *error_message = "ssd.spdk_proxy_max_channels must be > 0";
      return false;
    }
    if (distributed.has_value()) {
      const auto& d = distributed.value();
      // master_address MAY be empty.  That is the single-node deployment: the
      // client keeps its whole data plane -- backends, transfer engine, peer
      // service -- and simply never consults a master.  Every read already
      // resolves locally first (see local_first), and on one node a local miss
      // is the final answer, so there is nothing a master could add.  Setting
      // the address later is what turns the same process into a cluster member.
      if (d.master_config.node_id.empty()) {
        if (error_message) *error_message = "distributed.master_config.node_id must not be empty";
        return false;
      }
      if (d.master_config.node_address.empty()) {
        if (error_message)
          *error_message = "distributed.master_config.node_address must not be empty";
        return false;
      }
      // Only the selected medium's sizing is checked: a node that serves HBM
      // still carries a defaulted dram/ssd block it never allocates from.
      if (d.medium == UMBPMedium::HBM && d.hbm.capacity_bytes == 0) {
        if (error_message)
          *error_message =
              "distributed.hbm.capacity_bytes must be > 0 when distributed.medium is HBM";
        return false;
      }
      // Not ssd.Validate(): that returns early on ssd.enabled == false, and
      // selecting SSD here IS the opt-in (DistributedClient enables the tier
      // from `medium`, so an unset ssd.enabled must not skip the sizing check).
      if (d.medium == UMBPMedium::SSD) {
        if (ssd.capacity_bytes == 0) {
          if (error_message)
            *error_message = "ssd.capacity_bytes must be > 0 when distributed.medium is SSD";
          return false;
        }
        if (ssd.segment_size_bytes == 0) {
          if (error_message)
            *error_message = "ssd.segment_size_bytes must be > 0 when distributed.medium is SSD";
          return false;
        }
      }
    }
    if (distributed.has_value() && standalone_process.has_value()) {
      if (error_message)
        *error_message = "distributed and standalone_process are mutually exclusive";
      return false;
    }
    if (standalone_process.has_value()) {
      const auto& sp = standalone_process.value();
      if (sp.address.empty()) {
        if (error_message) *error_message = "standalone_process.address must not be empty";
        return false;
      }
      if (sp.startup_timeout_ms <= 0) {
        if (error_message) *error_message = "standalone_process.startup_timeout_ms must be > 0";
        return false;
      }
    }
    return true;
  }

  static UMBPConfig FromEnvironment() {
    UMBPConfig cfg;
    auto getenv_str = [](const char* name, const std::string& def) -> std::string {
      const char* v = std::getenv(name);
      return v ? v : def;
    };
    auto getenv_size = [](const char* name, size_t def) -> size_t {
      const char* v = std::getenv(name);
      return v ? static_cast<size_t>(std::stoull(v)) : def;
    };
    auto getenv_int = [](const char* name, int def) -> int {
      const char* v = std::getenv(name);
      return v ? std::atoi(v) : def;
    };
    auto getenv_double = [](const char* name, double def) -> double {
      const char* v = std::getenv(name);
      return v ? std::atof(v) : def;
    };

    cfg.dram.capacity_bytes = getenv_size("UMBP_DRAM_CAPACITY", cfg.dram.capacity_bytes);
    cfg.ssd.enabled = getenv_int("UMBP_SSD_ENABLED", cfg.ssd.enabled ? 1 : 0) != 0;
    cfg.ssd.storage_dir = getenv_str("UMBP_SSD_DIR", cfg.ssd.storage_dir);
    cfg.ssd.capacity_bytes = getenv_size("UMBP_SSD_CAPACITY", cfg.ssd.capacity_bytes);
    cfg.ssd.shard_io_threads = getenv_int("UMBP_SSD_SHARD_IO_THREADS", cfg.ssd.shard_io_threads);
    cfg.ssd.tier_io_threads = getenv_int("UMBP_SSD_TIER_IO_THREADS", cfg.ssd.tier_io_threads);
    cfg.ssd.direct_io = getenv_int("UMBP_SSD_DIRECT_IO", cfg.ssd.direct_io ? 1 : 0) != 0;
    cfg.ssd.single_flight_reads =
        getenv_int("UMBP_SSD_SINGLE_FLIGHT", cfg.ssd.single_flight_reads ? 1 : 0) != 0;
    cfg.ssd.verify_crc = getenv_int("UMBP_SSD_VERIFY_CRC", cfg.ssd.verify_crc ? 1 : 0) != 0;
    // Durability of the SSD cache tier.  "strict" (default) fdatasync()s every
    // batch write; "relaxed" leaves the data in the page cache and lets the
    // kernel flush it.  Relaxed is safe for a pure cache — the bytes are
    // re-fetchable, and the distributed peer discards leftover segments at
    // startup anyway (see PeerSsdManager::DiscardLeftoverOnStartup) — so the
    // flush buys nothing there while costing a large share of write time.
    {
      const std::string durability =
          getenv_str("UMBP_SSD_DURABILITY",
                     cfg.ssd.durability.mode == UMBPDurabilityMode::Relaxed ? "relaxed" : "strict");
      if (durability == "relaxed" || durability == "RELAXED") {
        cfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
      } else if (durability == "strict" || durability == "STRICT") {
        cfg.ssd.durability.mode = UMBPDurabilityMode::Strict;
      }
      // Any other value leaves the configured mode untouched; Validate() does
      // not police this field, so silently keeping the default beats guessing.
    }
    cfg.eviction.policy = getenv_str("UMBP_EVICTION_POLICY", cfg.eviction.policy);
    cfg.dram.high_watermark = getenv_double("UMBP_DRAM_HIGH_WM", cfg.dram.high_watermark);
    cfg.dram.low_watermark = getenv_double("UMBP_DRAM_LOW_WM", cfg.dram.low_watermark);
    cfg.ssd.high_watermark = getenv_double("UMBP_SSD_HIGH_WM", cfg.ssd.high_watermark);
    cfg.ssd.low_watermark = getenv_double("UMBP_SSD_LOW_WM", cfg.ssd.low_watermark);
    cfg.dram.use_hugepages =
        getenv_int("UMBP_DRAM_USE_HUGEPAGES", cfg.dram.use_hugepages ? 1 : 0) != 0;
    cfg.dram.hugepage_size = getenv_size("UMBP_DRAM_HUGEPAGE_SIZE", cfg.dram.hugepage_size);
    cfg.dram.numa_node = getenv_int("UMBP_DRAM_NUMA_NODE", cfg.dram.numa_node);
    cfg.dram.prefault = getenv_int("UMBP_DRAM_PREFAULT", cfg.dram.prefault ? 1 : 0) != 0;

    cfg.ssd.ssd_backend = getenv_str("UMBP_SSD_BACKEND", cfg.ssd.ssd_backend);
    if (cfg.ssd.ssd_backend == "file" && !std::getenv("UMBP_SSD_BACKEND") &&
        std::getenv("UMBP_SPDK_NVME_PCI")) {
      cfg.ssd.ssd_backend = "spdk";
    }
    cfg.ssd.spdk_bdev_name = getenv_str("UMBP_SPDK_BDEV", cfg.ssd.spdk_bdev_name);
    cfg.ssd.spdk_reactor_mask = getenv_str("UMBP_SPDK_REACTOR_MASK", cfg.ssd.spdk_reactor_mask);
    cfg.ssd.spdk_mem_size_mb = getenv_int("UMBP_SPDK_MEM_MB", cfg.ssd.spdk_mem_size_mb);
    cfg.ssd.spdk_nvme_pci_addr = getenv_str("UMBP_SPDK_NVME_PCI", cfg.ssd.spdk_nvme_pci_addr);
    cfg.ssd.spdk_nvme_ctrl_name = getenv_str("UMBP_SPDK_NVME_CTRL", cfg.ssd.spdk_nvme_ctrl_name);
    cfg.ssd.spdk_io_workers = getenv_int("UMBP_SPDK_IO_WORKERS", cfg.ssd.spdk_io_workers);

    cfg.ssd.spdk_proxy_shm_name = getenv_str("UMBP_SPDK_PROXY_SHM", cfg.ssd.spdk_proxy_shm_name);
    cfg.ssd.spdk_proxy_tenant_id = static_cast<uint32_t>(
        getenv_int("UMBP_SPDK_PROXY_TENANT_ID", static_cast<int>(cfg.ssd.spdk_proxy_tenant_id)));
    cfg.ssd.spdk_proxy_tenant_quota_bytes =
        getenv_size("UMBP_SPDK_PROXY_TENANT_QUOTA_BYTES", cfg.ssd.spdk_proxy_tenant_quota_bytes);

    const char* max_channels_env = std::getenv("UMBP_SPDK_PROXY_MAX_CHANNELS");
    if (!max_channels_env) max_channels_env = std::getenv("UMBP_SPDK_PROXY_MAX_RANKS");
    if (max_channels_env) {
      cfg.ssd.spdk_proxy_max_channels = static_cast<uint32_t>(std::atoi(max_channels_env));
    }

    const char* data_mb_env = std::getenv("UMBP_SPDK_PROXY_DATA_PER_CHANNEL_MB");
    if (!data_mb_env) data_mb_env = std::getenv("UMBP_SPDK_PROXY_DATA_MB");
    if (data_mb_env) {
      cfg.ssd.spdk_proxy_data_per_channel_mb = static_cast<size_t>(std::stoull(data_mb_env));
    }

    cfg.ssd.spdk_proxy_bin = getenv_str("UMBP_SPDK_PROXY_BIN", cfg.ssd.spdk_proxy_bin);
    cfg.ssd.spdk_proxy_startup_timeout_ms =
        getenv_int("UMBP_SPDK_PROXY_TIMEOUT_MS", cfg.ssd.spdk_proxy_startup_timeout_ms);
    cfg.ssd.spdk_proxy_auto_start =
        getenv_int("UMBP_SPDK_PROXY_AUTO_START", cfg.ssd.spdk_proxy_auto_start ? 1 : 0) != 0;
    cfg.ssd.spdk_proxy_idle_exit_timeout_ms =
        getenv_int("UMBP_SPDK_PROXY_IDLE_EXIT_TIMEOUT_MS", cfg.ssd.spdk_proxy_idle_exit_timeout_ms);
    cfg.ssd.spdk_proxy_allow_borrow =
        getenv_int("UMBP_SPDK_PROXY_ALLOW_BORROW", cfg.ssd.spdk_proxy_allow_borrow ? 1 : 0) != 0;
    cfg.ssd.spdk_proxy_reserved_shared_bytes = getenv_size(
        "UMBP_SPDK_PROXY_RESERVED_SHARED_BYTES", cfg.ssd.spdk_proxy_reserved_shared_bytes);

    std::string role_str = getenv_str("UMBP_ROLE", "");
    if (role_str == "leader")
      cfg.role = UMBPRole::SharedSSDLeader;
    else if (role_str == "follower")
      cfg.role = UMBPRole::SharedSSDFollower;
    else if (role_str == "standalone")
      cfg.role = UMBPRole::Standalone;
    else if (role_str.empty() && cfg.role == UMBPRole::Standalone) {
      const char* local_rank = nullptr;
      for (const char* name :
           {"LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID"}) {
        local_rank = std::getenv(name);
        if (local_rank) break;
      }
      if (local_rank) {
        cfg.role =
            (std::atoi(local_rank) == 0) ? UMBPRole::SharedSSDLeader : UMBPRole::SharedSSDFollower;
      }
    }

    return cfg;
  }
};

}  // namespace mori::umbp

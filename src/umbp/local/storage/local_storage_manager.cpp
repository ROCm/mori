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
#include "umbp/local/storage/local_storage_manager.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "umbp/common/log.h"
#include "umbp/local/storage/dram_tier.h"
#include "umbp/local/storage/ssd_tier.h"
#ifdef USE_SPDK
#include "umbp/local/storage/spdk_ssd_tier.h"
#endif
#ifdef __linux__
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <unistd.h>

#include "umbp/local/storage/spdk_proxy_tier.h"
#include "umbp/proxy/spdk_proxy_shm.h"
#endif

namespace mori::umbp {

// ---------------------------------------------------------------------------
// ExtractBaseHash
//
// Strips the SGLang-generated key suffix to recover the 64-hex-char SHA256
// base hash.  Key formats (rightmost fields stripped first):
//
//   MHA, no PP:   {hash}_{tp_rank}_{k|v}       → strip _k/_v, strip _{digit}
//   MHA, with PP: {hash}_{tp_rank}_{pp_rank}_{k|v} → strip _k/_v, strip _{digit}, strip _{digit}
//   MLA, no PP:   {hash}__k                     → strip _k, strip trailing _
//   MLA, with PP: {hash}_{pp_rank}_{k}           → strip _k, strip _{digit}
//
// The SHA256 base hash is always exactly 64 lower-case hex characters.
// We use that invariant to anchor stripping: stop as soon as the remaining
// string length equals 64 and the remaining characters are all hex.
// ---------------------------------------------------------------------------
static bool IsHexChar(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static bool IsAllHex(const std::string& s, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    if (!IsHexChar(s[i])) return false;
  }
  return true;
}

std::string LocalStorageManager::ExtractBaseHash(const std::string& key) {
  // Fast-path: already looks like a bare hash.
  if (key.size() == 64 && IsAllHex(key, 64)) return key;

  std::string s = key;

  // Strip _k or _v suffix.
  if (s.size() >= 2 && s[s.size() - 2] == '_' && (s.back() == 'k' || s.back() == 'v')) {
    s.resize(s.size() - 2);
  } else {
    return key;  // Unrecognised format — return unchanged.
  }

  // Iteratively strip _{digit+} rank segments until the remainder is a 64-char
  // hex string or no more strippable segments remain.
  for (int pass = 0; pass < 2; ++pass) {
    if (s.size() == 64 && IsAllHex(s, 64)) return s;

    // Find the last '_' and check that everything after it is digits.
    size_t pos = s.rfind('_');
    if (pos == std::string::npos) break;

    bool all_digits = true;
    for (size_t i = pos + 1; i < s.size(); ++i) {
      if (s[i] < '0' || s[i] > '9') {
        all_digits = false;
        break;
      }
    }

    // Handle MLA no-PP double-underscore: after stripping _k we may have a
    // trailing '_' with nothing after it (pos == s.size()-1).
    if (pos == s.size() - 1) {
      // Empty segment after last '_' — this is the double-underscore case.
      s.resize(pos);
      continue;
    }

    if (!all_digits) break;
    s.resize(pos);
  }

  if (s.size() == 64 && IsAllHex(s, 64)) return s;

  // Fallback: return original key — group will contain only this key (size 1),
  // which is safe (single-key eviction).
  return key;
}

// ---------------------------------------------------------------------------
// Depth and group map helpers
// ---------------------------------------------------------------------------

void LocalStorageManager::RecordDepth(const std::string& key, int depth) {
  std::unique_lock<std::shared_mutex> lock(depth_mu_);
  depth_map_[key] = depth;
}

int LocalStorageManager::GetDepth(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(depth_mu_);
  auto it = depth_map_.find(key);
  return (it != depth_map_.end()) ? it->second : -1;
}

void LocalStorageManager::RemoveDepthAndGroup(const std::string& key) {
  std::unique_lock<std::shared_mutex> lock(depth_mu_);
  depth_map_.erase(key);

  std::string base = ExtractBaseHash(key);
  auto git = group_map_.find(base);
  if (git != group_map_.end()) {
    auto& vec = git->second;
    vec.erase(std::remove(vec.begin(), vec.end(), key), vec.end());
    if (vec.empty()) group_map_.erase(git);
  }
}

void LocalStorageManager::RecordGroup(const std::string& key) {
  std::unique_lock<std::shared_mutex> lock(depth_mu_);
  std::string base = ExtractBaseHash(key);
  auto& vec = group_map_[base];
  // Only append if not already present (idempotent for re-puts).
  if (std::find(vec.begin(), vec.end(), key) == vec.end()) {
    vec.push_back(key);
  }
}

std::vector<std::string> LocalStorageManager::GetGroup(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(depth_mu_);
  std::string base = ExtractBaseHash(key);
  auto it = group_map_.find(base);
  if (it != group_map_.end() && !it->second.empty()) return it->second;
  return {key};
}

// ---------------------------------------------------------------------------
// SelectVictim
//
// "lru" policy: O(1) — return plain LRU tail key.
// "prefix_aware_lru": scan up to eviction_candidate_window candidates from
// the LRU tail, score by depth (higher depth = preferred victim, deeper suffix
// block first).  Tie-break by LRU position (earlier in the candidate list =
// older = preferred victim among equal-depth candidates).
// Falls back to plain LRU when no depth metadata is available.
// ---------------------------------------------------------------------------
std::string LocalStorageManager::SelectVictim(TierBackend* tier) {
  if (config_.eviction.policy != "prefix_aware_lru") {
    return tier->GetLRUKey();
  }

  size_t window = config_.eviction.candidate_window;
  if (window == 0) window = 1;

  std::vector<std::string> candidates = tier->GetLRUCandidates(window);
  if (candidates.empty()) return "";

  // Score: (depth, position) where higher depth wins, lower position (older)
  // wins ties.  depth == -1 is treated as 0 for scoring so metadata-free
  // keys degrade to plain LRU ordering.
  int best_depth = -2;  // sentinel below any real score
  size_t best_pos = 0;
  std::string best_key;

  for (size_t i = 0; i < candidates.size(); ++i) {
    int d = GetDepth(candidates[i]);  // -1 if unknown
    int score = (d >= 0) ? d : 0;
    // Higher score wins; tie-break: smaller i (older LRU position) wins.
    if (best_key.empty() || score > best_depth || (score == best_depth && i < best_pos)) {
      best_depth = score;
      best_pos = i;
      best_key = candidates[i];
    }
  }

  return best_key;
}

// ---------------------------------------------------------------------------
// WriteFromPtrWithDepth
// ---------------------------------------------------------------------------
bool LocalStorageManager::WriteFromPtrWithDepth(const std::string& key, uintptr_t src, size_t size,
                                                int depth, StorageTier tier) {
  bool ok = WriteFromPtr(key, src, size, tier);
  if (ok && depth >= 0) {
    RecordDepth(key, depth);
    RecordGroup(key);
  }
  return ok;
}

LocalStorageManager::~LocalStorageManager() = default;

// ---------------------------------------------------------------------------
// Auto-fork proxy daemon (Linux only)
// ---------------------------------------------------------------------------
#ifdef __linux__
static std::string FindProxyBinary(const std::string& explicit_path) {
  auto try_candidate = [](const std::string& candidate) -> std::string {
    return (access(candidate.c_str(), X_OK) == 0) ? candidate : "";
  };

  if (!explicit_path.empty()) {
    if (auto hit = try_candidate(explicit_path); !hit.empty()) return hit;
    UMBP_LOG_WARN("LSM: UMBP_SPDK_PROXY_BIN='%s' not executable", explicit_path.c_str());
  }

  char exe_buf[4096];
  ssize_t len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (len > 0) {
    exe_buf[len] = '\0';
    std::string dir(exe_buf);
    size_t slash = dir.rfind('/');
    if (slash != std::string::npos) {
      dir.resize(slash + 1);
      if (auto hit = try_candidate(dir + "spdk_proxy"); !hit.empty()) return hit;

      // CMake build tree layout: test binaries live under
      // build_umbp/tests/... while spdk_proxy lives under build_umbp/src/umbp.
      std::string parent = dir;
      for (int depth = 0; depth < 8; ++depth) {
        if (auto hit = try_candidate(parent + "src/umbp/spdk_proxy"); !hit.empty()) {
          return hit;
        }
        if (parent.empty() || parent == "/") break;
        size_t end = parent.size();
        while (end > 0 && parent[end - 1] == '/') --end;
        if (end == 0) break;
        size_t prev = parent.rfind('/', end - 1);
        if (prev == std::string::npos) break;
        parent.resize(prev + 1);
      }
    }
  }

  return "spdk_proxy";
}

class ScopedBootstrapLock {
 public:
  explicit ScopedBootstrapLock(const std::string& shm_name) {
    path_ = ::umbp::proxy::ProxyShmRegion::BootstrapLockPath(shm_name);
    fd_ = open(path_.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ < 0) {
      UMBP_LOG_ERROR("LSM: open bootstrap lock '%s' failed: %s", path_.c_str(), strerror(errno));
      return;
    }
    if (flock(fd_, LOCK_EX) != 0) {
      UMBP_LOG_ERROR("LSM: flock bootstrap lock '%s' failed: %s", path_.c_str(), strerror(errno));
      close(fd_);
      fd_ = -1;
    }
  }

  ~ScopedBootstrapLock() {
    if (fd_ >= 0) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
  }

  bool IsValid() const { return fd_ >= 0; }

 private:
  std::string path_;
  int fd_ = -1;
};

static void SetEnvVarFromConfig(const char* name, const std::string& value) {
  if (!value.empty()) setenv(name, value.c_str(), 1);
}

static void SetEnvVarFromConfig(const char* name, size_t value) {
  setenv(name, std::to_string(value).c_str(), 1);
}

static void SetEnvVarFromConfig(const char* name, int value) {
  setenv(name, std::to_string(value).c_str(), 1);
}

static void SetEnvVarFromConfig(const char* name, bool value) {
  setenv(name, value ? "1" : "0", 1);
}

int LocalStorageManager::SpawnProxyDaemon(const std::string& shm_name) {
  std::string bin = FindProxyBinary(config_.spdk_proxy_bin);
  UMBP_LOG_INFO("LSM: launching spdk_proxy binary '%s' for shm '%s'", bin.c_str(),
                shm_name.c_str());
  pid_t pid = fork();
  if (pid < 0) {
    UMBP_LOG_ERROR("LSM: fork() failed: %s", strerror(errno));
    return -1;
  }

  if (pid == 0) {
    setsid();

    SetEnvVarFromConfig("UMBP_SPDK_PROXY_SHM", shm_name);
    SetEnvVarFromConfig("UMBP_SPDK_PROXY_MAX_CHANNELS",
                        static_cast<int>(config_.spdk_proxy_max_channels));
    SetEnvVarFromConfig("UMBP_SPDK_PROXY_DATA_PER_CHANNEL_MB",
                        config_.spdk_proxy_data_per_channel_mb);
    SetEnvVarFromConfig("UMBP_SPDK_PROXY_IDLE_EXIT_TIMEOUT_MS",
                        config_.spdk_proxy_idle_exit_timeout_ms);
    SetEnvVarFromConfig("UMBP_SPDK_PROXY_ALLOW_BORROW", config_.spdk_proxy_allow_borrow);
    SetEnvVarFromConfig("UMBP_SPDK_PROXY_RESERVED_SHARED_BYTES",
                        config_.spdk_proxy_reserved_shared_bytes);
    SetEnvVarFromConfig("UMBP_SSD_CAPACITY", config_.ssd.capacity_bytes);
    SetEnvVarFromConfig("UMBP_SPDK_NVME_PCI", config_.spdk_nvme_pci_addr);
    SetEnvVarFromConfig("UMBP_SPDK_NVME_CTRL", config_.spdk_nvme_ctrl_name);
    SetEnvVarFromConfig("UMBP_SPDK_BDEV", config_.spdk_bdev_name);
    SetEnvVarFromConfig("UMBP_SPDK_REACTOR_MASK", config_.spdk_reactor_mask);
    SetEnvVarFromConfig("UMBP_SPDK_MEM_MB", config_.spdk_mem_size_mb);
    SetEnvVarFromConfig("UMBP_SPDK_IO_WORKERS", config_.spdk_io_workers);

    if (UmbpLogLevel() >= 1) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
    }
    execlp(bin.c_str(), "spdk_proxy", static_cast<char*>(nullptr));
    fprintf(stderr, "[UMBP ERROR] execlp('%s') failed: %s\n", bin.c_str(), strerror(errno));
    _exit(127);
  }

  UMBP_LOG_INFO("LSM: spawned spdk_proxy service pid=%d shm='%s'", pid, shm_name.c_str());
  return 0;
}

int LocalStorageManager::EnsureProxyDaemon(const std::string& shm_name) {
  int probe = ::umbp::proxy::ProxyShmRegion::ProbeExisting(shm_name);
  if (probe == 1) return 0;
  if (probe == -2) {
    UMBP_LOG_ERROR("LSM: proxy protocol version mismatch on SHM '%s'", shm_name.c_str());
    return -1;
  }
  if (!config_.spdk_proxy_auto_start) {
    if (probe == -1) {
      return SpdkProxyTier::WaitForProxy(shm_name, config_.spdk_proxy_startup_timeout_ms) ? 0 : -1;
    }
    UMBP_LOG_ERROR("LSM: SPDK proxy absent on SHM '%s' and auto-start disabled", shm_name.c_str());
    return -1;
  }

  ScopedBootstrapLock lock(shm_name);
  if (!lock.IsValid()) return -1;

  probe = ::umbp::proxy::ProxyShmRegion::ProbeExisting(shm_name);
  if (probe == 1) return 0;
  if (probe == -2) {
    UMBP_LOG_ERROR("LSM: proxy protocol version mismatch on SHM '%s'", shm_name.c_str());
    return -1;
  }
  if (probe == -1) {
    if (SpdkProxyTier::WaitForProxy(shm_name, config_.spdk_proxy_startup_timeout_ms)) {
      return 0;
    }

    ::umbp::proxy::ProxyShmRegion existing;
    if (existing.Attach(shm_name) == 0) {
      auto* hdr = existing.Header();
      uint32_t st = hdr->state.load(std::memory_order_acquire);
      uint32_t pid = hdr->proxy_pid.load(std::memory_order_acquire);
      bool alive = false;
#ifdef __linux__
      alive = (pid > 0 && kill(static_cast<pid_t>(pid), 0) == 0);
#endif
      if (alive && st == static_cast<uint32_t>(::umbp::proxy::ProxyState::ERROR)) {
        UMBP_LOG_ERROR("LSM: proxy on SHM '%s' is alive but in ERROR state", shm_name.c_str());
        return -1;
      }
      if (alive && st != static_cast<uint32_t>(::umbp::proxy::ProxyState::SHUTDOWN)) {
        UMBP_LOG_ERROR("LSM: proxy on SHM '%s' did not become READY in time (state=%u)",
                       shm_name.c_str(), st);
        return -1;
      }
    }
  }

  ::umbp::proxy::ProxyShmRegion::CleanupStale(shm_name);
  if (SpawnProxyDaemon(shm_name) != 0) return -1;

  return SpdkProxyTier::WaitForProxy(shm_name, config_.spdk_proxy_startup_timeout_ms) ? 0 : -1;
}
#endif

LocalStorageManager::LocalStorageManager(const UMBPConfig& config,
                                         mori::umbp::LocalBlockIndex* index)
    : config_(config), role_(config.ResolveRole()), index_(index) {
  std::string error_message;
  if (!config_.Validate(&error_message)) {
    throw std::runtime_error("invalid UMBP config: " + error_message);
  }
  // DRAM tier is always present (fastest)
  tiers_.push_back(
      {StorageTier::CPU_DRAM,
       std::make_unique<DRAMTier>(config_.dram.capacity_bytes, config_.dram.use_shared_memory,
                                  config_.dram.shm_name)});

  // SSD tier is optional (slower)
  if (config_.ssd.enabled) {
    std::unique_ptr<TierBackend> ssd_backend;
    bool use_proxy = false;

    if (config_.ssd_backend == "spdk") {
#ifdef USE_SPDK
      if (role_ == UMBPRole::Standalone) {
        // Standalone with USE_SPDK compiled: try direct SPDK access first
        // (best perf, single process, no proxy overhead).
        auto spdk_tier = std::make_unique<SpdkSsdTier>(config_);
        if (spdk_tier->IsValid()) {
          ssd_backend = std::unique_ptr<TierBackend>(spdk_tier.release());
        } else {
          throw std::runtime_error(
              "UMBP_SSD_BACKEND=spdk: SpdkSsdTier init failed. "
              "Run 'tools/umbp_spdk_preflight.sh --pci <addr>' to diagnose.");
        }
      } else {
        // Non-Standalone (Leader/Follower): always use proxy for
        // multi-process coordination.
        use_proxy = true;
      }
#else
      // USE_SPDK not set — SpdkSsdTier (direct) is unavailable.
      // Fall back to proxy mode: SpdkProxyTier has zero SPDK dependency
      // (pure POSIX SHM), and the separate spdk_proxy daemon handles NVMe.
      use_proxy = true;
#endif
    } else if (config_.ssd_backend == "spdk_proxy") {
      use_proxy = true;
    }

#ifdef __linux__
    if (use_proxy && !ssd_backend) {
      std::string proxy_shm_name = config_.spdk_proxy_shm_name;
      if (proxy_shm_name.empty()) proxy_shm_name = ::umbp::proxy::kDefaultShmName;

      bool proxy_ready = false;
      if (config_.spdk_proxy_auto_start) {
        proxy_ready = (EnsureProxyDaemon(proxy_shm_name) == 0);
      } else {
        proxy_ready =
            SpdkProxyTier::WaitForProxy(proxy_shm_name, config_.spdk_proxy_startup_timeout_ms);
      }

      if (proxy_ready) {
        auto proxy_tier = std::make_unique<SpdkProxyTier>(config_);
        if (proxy_tier->IsValid()) {
          ssd_backend = std::unique_ptr<TierBackend>(proxy_tier.release());
        }
      }

      if (!ssd_backend) {
        bool explicit_spdk = (config_.ssd_backend == "spdk" || config_.ssd_backend == "spdk_proxy");
        if (explicit_spdk) {
          throw std::runtime_error(
              "UMBP_SSD_BACKEND=" + config_.ssd_backend + ": SPDK proxy connect failed (shm=" +
              proxy_shm_name + "). Run 'tools/umbp_spdk_preflight.sh --pci <addr>' to diagnose.");
        }
        fprintf(stderr, "[UMBP WARN] SPDK proxy connect failed. Falling back to POSIX SSD.\n");
      }
    }
#endif
    if (!ssd_backend) {
      SSDAccessMode segmented_access_mode = SSDAccessMode::ReadWrite;
      if (role_ == UMBPRole::SharedSSDFollower) {
        segmented_access_mode = SSDAccessMode::ReadOnlyShared;
      }
      ssd_backend = std::make_unique<SSDTier>(config_.ssd.storage_dir, config_.ssd.capacity_bytes,
                                              config_, segmented_access_mode);
    }
    tiers_.push_back({StorageTier::LOCAL_SSD, std::move(ssd_backend)});
  }
}

TierBackend* LocalStorageManager::GetTier(StorageTier tier) {
  for (auto& entry : tiers_) {
    if (entry.id == tier) return entry.backend.get();
  }
  return nullptr;
}

const TierBackend* LocalStorageManager::GetTier(StorageTier tier) const {
  for (const auto& entry : tiers_) {
    if (entry.id == tier) return entry.backend.get();
  }
  return nullptr;
}

TierBackend* LocalStorageManager::FindTierHolding(const std::string& key) {
  for (auto& entry : tiers_) {
    if (entry.backend->Exists(key)) return entry.backend.get();
  }
  return nullptr;
}

const TierBackend* LocalStorageManager::FindTierHolding(const std::string& key) const {
  for (const auto& entry : tiers_) {
    if (entry.backend->Exists(key)) return entry.backend.get();
  }
  return nullptr;
}

TierBackend* LocalStorageManager::NextSlowerTier(StorageTier current) {
  for (size_t i = 0; i + 1 < tiers_.size(); ++i) {
    if (tiers_[i].id == current) return tiers_[i + 1].backend.get();
  }
  return nullptr;
}

TierBackend* LocalStorageManager::NextFasterTier(StorageTier current) {
  for (size_t i = 1; i < tiers_.size(); ++i) {
    if (tiers_[i].id == current) return tiers_[i - 1].backend.get();
  }
  return nullptr;
}

void LocalStorageManager::SetOnTierChange(TierChangeCallback cb) {
  on_tier_change_ = std::move(cb);
}

std::optional<LocalStorageManager::TierLocationInfo> LocalStorageManager::BuildTierLocationInfo(
    TierBackend* tier, const std::string& key, size_t size) {
  if (!tier) {
    return std::nullopt;
  }

  TierLocationInfo info;
  info.size = size;

  // DRAM: read structured chunk location directly from DRAMTier.
  if (tier->tier_id() == StorageTier::CPU_DRAM) {
    auto* dram = dynamic_cast<DRAMTier*>(tier);
    if (dram) {
      auto loc = dram->GetSlotChunkLocation(key);
      if (!loc) return std::nullopt;
      info.location_id = std::to_string(loc->chunk_index) + ":" + std::to_string(loc->offset);
      return info;
    }
  }

  // Non-DRAM tiers: single-buffer "0:<raw_id>" format.
  auto raw_id = tier->GetLocationId(key);
  if (!raw_id.has_value()) {
    return std::nullopt;
  }
  info.location_id = "0:" + *raw_id;
  return info;
}

bool LocalStorageManager::MoveKey(const std::string& key, TierBackend* from, TierBackend* to) {
  // Fast path for DRAM->slower movement to avoid an intermediate vector copy.
  if (from->Capabilities().zero_copy_read) {
    size_t sz = 0;
    const void* ptr = from->ReadPtr(key, &sz);
    if (ptr && sz > 0 && to->Write(key, ptr, sz)) {
      StorageTier from_id = from->tier_id();
      auto new_location = BuildTierLocationInfo(to, key, sz);
      from->Evict(key);
      UpsertIndexTier(key, to->tier_id(), sz);
      if (on_tier_change_) on_tier_change_(key, from_id, to->tier_id(), new_location);
      return true;
    }
  }

  auto data = from->Read(key);
  if (data.empty()) return false;
  if (!to->Write(key, data.data(), data.size())) return false;
  StorageTier from_id = from->tier_id();
  auto new_location = BuildTierLocationInfo(to, key, data.size());
  from->Evict(key);
  UpsertIndexTier(key, to->tier_id(), data.size());
  if (on_tier_change_) on_tier_change_(key, from_id, to->tier_id(), new_location);
  return true;
}

bool LocalStorageManager::DemoteLRUForSpace(TierBackend* tier) {
  TierBackend* slower = NextSlowerTier(tier->tier_id());
  if (!slower) return false;
  if (role_ == UMBPRole::SharedSSDFollower && slower->tier_id() == StorageTier::LOCAL_SSD) {
    return false;
  }

  std::string victim = SelectVictim(tier);
  if (victim.empty()) return false;

  // Group-demote: move all keys sharing the same page together.
  // depth/group metadata is preserved (key still exists in slower tier).
  std::vector<std::string> group = GetGroup(victim);
  bool any_moved = false;
  for (const auto& k : group) {
    if (tier->Exists(k)) {
      if (MoveKey(k, tier, slower)) {
        any_moved = true;
      }
    }
  }
  return any_moved;
}

bool LocalStorageManager::Write(const std::string& key, const void* data, size_t size,
                                StorageTier tier) {
  TierBackend* target = GetTier(tier);
  if (!target) return false;

  // Try direct write
  if (target->Write(key, data, size)) return true;

  // Target full — try demoting LRU keys to next-slower tier
  while (DemoteLRUForSpace(target)) {
    if (target->Write(key, data, size)) return true;
  }
  TierBackend* faster_than_target = NextFasterTier(target->tier_id());
  while (true) {
    std::string victim = SelectVictim(target);
    if (victim.empty()) return false;
    std::vector<std::string> group = GetGroup(victim);
    for (const auto& k : group) {
      if (target->Exists(k)) {
        bool only_on_target = !faster_than_target || !faster_than_target->Exists(k);
        target->Evict(k);
        if (only_on_target) {
          if (on_tier_change_) on_tier_change_(k, target->tier_id(), std::nullopt, std::nullopt);
          if (index_) index_->Remove(k);
          RemoveDepthAndGroup(k);
        }
      }
    }
    if (target->Write(key, data, size)) return true;
  }
}

bool LocalStorageManager::WriteFromPtr(const std::string& key, uintptr_t src, size_t size,
                                       StorageTier tier) {
  return Write(key, reinterpret_cast<const void*>(src), size, tier);
}

bool LocalStorageManager::ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) {
  bool ok = ReadIntoPtrNoPromote(key, dst, size);
  if (ok && config_.eviction.auto_promote_on_read) {
    MaybeAutoPromote(key);
  }
  return ok;
}

bool LocalStorageManager::ReadIntoPtrNoPromote(const std::string& key, uintptr_t dst, size_t size) {
  if (index_) {
    auto loc = index_->Lookup(key);
    if (loc) {
      TierBackend* hinted = GetTier(loc->tier);
      if (hinted && hinted->Exists(key)) {
        bool ok = hinted->ReadIntoPtr(key, dst, size);
        if (ok) return true;
      }
    }
  }

  for (size_t i = 0; i < tiers_.size(); ++i) {
    if (tiers_[i].backend->Exists(key)) {
      bool ok = tiers_[i].backend->ReadIntoPtr(key, dst, size);
      return ok;
    }
  }
  return false;
}

std::vector<bool> LocalStorageManager::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                                        const std::vector<uintptr_t>& dst_ptrs,
                                                        const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;

  // Partition keys by tier using index hints.
  // tier_indices[t] holds original indices for keys hinted to tier t.
  // Use the integer value of StorageTier as array index (0=DRAM, 1=SSD).
  constexpr int kNumTiers = 2;
  std::vector<size_t> tier_indices[kNumTiers];
  std::vector<size_t> no_hint_indices;

  for (size_t i = 0; i < keys.size(); ++i) {
    bool has_hint = false;
    if (index_) {
      auto loc = index_->Lookup(keys[i]);
      if (loc) {
        int t = static_cast<int>(loc->tier);
        if (t >= 0 && t < kNumTiers) {
          tier_indices[t].push_back(i);
          has_hint = true;
        }
      }
    }
    if (!has_hint) {
      no_hint_indices.push_back(i);
    }
  }

  // Dispatch per-tier batch reads.
  for (int t = 0; t < kNumTiers; ++t) {
    auto& indices = tier_indices[t];
    if (indices.empty()) continue;

    StorageTier tier_id = static_cast<StorageTier>(t);
    TierBackend* tier = GetTier(tier_id);
    if (!tier) {
      for (size_t idx : indices) no_hint_indices.push_back(idx);
      continue;
    }

    if (tier->Capabilities().batch_read && indices.size() > 1) {
      std::vector<std::string> batch_keys;
      std::vector<uintptr_t> batch_ptrs;
      std::vector<size_t> batch_sizes;
      batch_keys.reserve(indices.size());
      batch_ptrs.reserve(indices.size());
      batch_sizes.reserve(indices.size());
      for (size_t idx : indices) {
        batch_keys.push_back(keys[idx]);
        batch_ptrs.push_back(dst_ptrs[idx]);
        batch_sizes.push_back(sizes[idx]);
      }
      auto batch_results = tier->ReadBatchIntoPtr(batch_keys, batch_ptrs, batch_sizes);
      for (size_t j = 0; j < indices.size(); ++j) {
        if (batch_results[j]) {
          results[indices[j]] = true;
          if (tier_id != StorageTier::CPU_DRAM && config_.eviction.auto_promote_on_read) {
            MaybeAutoPromote(keys[indices[j]]);
          }
        } else {
          no_hint_indices.push_back(indices[j]);
        }
      }
    } else {
      for (size_t idx : indices) {
        if (tier->Exists(keys[idx])) {
          bool ok = tier->ReadIntoPtr(keys[idx], dst_ptrs[idx], sizes[idx]);
          if (ok) {
            results[idx] = true;
            if (tier_id != StorageTier::CPU_DRAM && config_.eviction.auto_promote_on_read) {
              MaybeAutoPromote(keys[idx]);
            }
          } else {
            no_hint_indices.push_back(idx);
          }
        } else {
          no_hint_indices.push_back(idx);
        }
      }
    }
  }

  // Fallback: full tier scan for keys without hints or that failed hinted read.
  for (size_t idx : no_hint_indices) {
    if (results[idx]) continue;  // already resolved
    results[idx] = ReadIntoPtr(keys[idx], dst_ptrs[idx], sizes[idx]);
  }

  return results;
}

bool LocalStorageManager::Exists(const std::string& key) const {
  return FindTierHolding(key) != nullptr;
}

bool LocalStorageManager::Evict(const std::string& key) {
  TierBackend* tier = FindTierHolding(key);
  if (!tier) {
    if (index_) index_->Remove(key);
    RemoveDepthAndGroup(key);
    return false;
  }
  bool ok = tier->Evict(key);
  if (ok) {
    if (on_tier_change_) on_tier_change_(key, tier->tier_id(), std::nullopt, std::nullopt);
    if (index_) index_->Remove(key);
  }
  RemoveDepthAndGroup(key);
  return ok;
}

std::pair<size_t, size_t> LocalStorageManager::Capacity(StorageTier tier) const {
  const TierBackend* t = GetTier(tier);
  if (!t) return {0, 0};
  return t->Capacity();
}

bool LocalStorageManager::Demote(const std::string& key) {
  TierBackend* from = FindTierHolding(key);
  if (!from) return false;

  TierBackend* to = NextSlowerTier(from->tier_id());
  if (!to) return false;

  return MoveKey(key, from, to);
}

bool LocalStorageManager::Promote(const std::string& key) {
  TierBackend* from = FindTierHolding(key);
  if (!from) return false;

  TierBackend* to = NextFasterTier(from->tier_id());
  if (!to) return false;

  // Already in the faster tier
  if (to->Exists(key)) return true;

  // Try direct move
  auto data = from->Read(key);
  if (data.empty()) return false;

  // Shared SSD follower mode is read-only against SSD. Promotion into DRAM
  // must never demote DRAM victims into SSD.
  if (role_ == UMBPRole::SharedSSDFollower && from->tier_id() == StorageTier::LOCAL_SSD &&
      to->tier_id() == StorageTier::CPU_DRAM) {
    if (!to->Write(key, data.data(), data.size())) {
      while (true) {
        std::string victim = SelectVictim(to);
        if (victim.empty()) return false;
        std::vector<std::string> group = GetGroup(victim);
        for (const auto& k : group) {
          if (to->Exists(k)) {
            to->Evict(k);
            if (index_) index_->Remove(k);
            RemoveDepthAndGroup(k);
          }
        }
        if (to->Write(key, data.data(), data.size())) break;
      }
    }
    // Drop read-tracking metadata in source tier while preserving shared file.
    auto new_location = BuildTierLocationInfo(to, key, data.size());
    StorageTier from_id = from->tier_id();
    from->Evict(key);
    UpsertIndexTier(key, to->tier_id(), data.size());
    if (on_tier_change_) on_tier_change_(key, from_id, to->tier_id(), new_location);
    return true;
  }

  if (!to->Write(key, data.data(), data.size())) {
    // Target full — demote LRU keys from target to make room
    while (DemoteLRUForSpace(to)) {
      if (to->Write(key, data.data(), data.size())) {
        auto new_location = BuildTierLocationInfo(to, key, data.size());
        StorageTier from_id = from->tier_id();
        from->Evict(key);
        UpsertIndexTier(key, to->tier_id(), data.size());
        if (on_tier_change_) on_tier_change_(key, from_id, to->tier_id(), new_location);
        return true;
      }
    }
    return false;  // Cannot promote
  }

  auto new_location = BuildTierLocationInfo(to, key, data.size());
  StorageTier from_id = from->tier_id();
  from->Evict(key);
  UpsertIndexTier(key, to->tier_id(), data.size());
  if (on_tier_change_) on_tier_change_(key, from_id, to->tier_id(), new_location);
  return true;
}

bool LocalStorageManager::CopyToSSD(const std::string& key) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;

  TierBackend* dram = GetTier(StorageTier::CPU_DRAM);
  TierBackend* ssd = GetTier(StorageTier::LOCAL_SSD);
  if (!dram || !ssd) return false;

  // Already on SSD
  if (ssd->Exists(key)) return true;

  // Zero-copy: get a raw pointer into mmap'd DRAM instead of allocating a copy.
  size_t data_size = 0;
  const void* ptr = dram->ReadPtr(key, &data_size);
  if (!ptr || data_size == 0) return false;

  if (ssd->Write(key, ptr, data_size)) return true;

  // SSD full — evict SSD LRU entries to make room.
  // If a victim is only on SSD (no DRAM copy), we must also update the index.
  while (true) {
    std::string victim = SelectVictim(ssd);
    if (victim.empty()) return false;

    std::vector<std::string> group = GetGroup(victim);
    for (const auto& k : group) {
      if (!ssd->Exists(k)) continue;
      bool only_on_ssd = !dram->Exists(k);
      // Keep the index consistent: if the victim is only on SSD, evicting it
      // is data loss and the index must be updated.
      if (index_) {
        auto loc = index_->Lookup(k);
        if (loc && (loc->tier == StorageTier::LOCAL_SSD || only_on_ssd)) {
          index_->Remove(k);
        }
      }
      ssd->Evict(k);
      // Only clear metadata if key is gone from all tiers (data loss).
      if (only_on_ssd) RemoveDepthAndGroup(k);
    }
    if (ssd->Write(key, ptr, data_size)) return true;
  }
}

bool LocalStorageManager::CopyToSSDBatch(const std::vector<std::string>& keys) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;

  TierBackend* dram = GetTier(StorageTier::CPU_DRAM);
  TierBackend* ssd = GetTier(StorageTier::LOCAL_SSD);
  if (!dram || !ssd) return false;

  // Gather zero-copy pointers for keys not yet on SSD.
  std::vector<std::string> batch_keys;
  std::vector<const void*> batch_ptrs;
  std::vector<size_t> batch_sizes;
  std::vector<std::string> fallback_keys;

  for (const auto& key : keys) {
    if (ssd->Exists(key)) continue;
    size_t sz = 0;
    const void* ptr = dram->ReadPtr(key, &sz);
    if (!ptr || sz == 0) continue;
    batch_keys.push_back(key);
    batch_ptrs.push_back(ptr);
    batch_sizes.push_back(sz);
  }

  if (batch_keys.empty()) return true;

  // Try batch write if SSDTier is available.
  if (ssd->Capabilities().batch_write && batch_keys.size() > 1) {
    if (ssd->WriteBatch(batch_keys, batch_ptrs, batch_sizes)) return true;
    // Batch failed (likely capacity) — fall through to per-key CopyToSSD.
  }

  // Fallback: per-key CopyToSSD.
  bool all_ok = true;
  for (const auto& key : batch_keys) {
    if (!CopyToSSD(key)) all_ok = false;
  }
  return all_ok;
}

void LocalStorageManager::MaybeAutoPromote(const std::string& key) {
  if (!config_.eviction.auto_promote_on_read) return;

  // Only promote if key is not in the fastest tier
  if (tiers_.empty()) return;
  TierBackend* fastest = tiers_.front().backend.get();
  if (fastest->Exists(key)) return;

  // Check fastest tier watermark before promoting
  auto [used, total] = fastest->Capacity();
  double utilization = static_cast<double>(used) / total;
  if (utilization >= config_.dram.high_watermark) return;

  // Best-effort promote
  if (role_ == UMBPRole::SharedSSDFollower) {
    InsertReadCacheNoWriteback(key);
  } else {
    Promote(key);
  }
}

bool LocalStorageManager::InsertReadCacheNoWriteback(const std::string& key) {
  TierBackend* source = GetTier(StorageTier::LOCAL_SSD);
  TierBackend* dram = GetTier(StorageTier::CPU_DRAM);
  if (!source || !dram) return false;

  if (dram->Exists(key)) {
    UpsertIndexTier(key, StorageTier::CPU_DRAM, 0);
    return true;
  }
  if (!source->Exists(key)) return false;

  auto data = source->Read(key);
  if (data.empty()) return false;

  if (!dram->Write(key, data.data(), data.size())) {
    // Follower mode never writes back to SSD. Evict DRAM-only victims.
    while (true) {
      std::string victim = SelectVictim(dram);
      if (victim.empty()) return false;
      std::vector<std::string> group = GetGroup(victim);
      for (const auto& k : group) {
        if (dram->Exists(k)) {
          dram->Evict(k);
          if (index_) index_->Remove(k);
          RemoveDepthAndGroup(k);
        }
      }
      if (dram->Write(key, data.data(), data.size())) break;
    }
  }

  UpsertIndexTier(key, StorageTier::CPU_DRAM, data.size());
  return true;
}

void LocalStorageManager::UpsertIndexTier(const std::string& key, StorageTier tier,
                                          size_t size_hint) {
  if (!index_) return;
  if (!index_->UpdateTier(key, tier)) {
    index_->Insert(key, {tier, 0, size_hint});
  }
}

void LocalStorageManager::Clear() {
  for (auto& entry : tiers_) {
    entry.backend->Clear();
  }
  if (index_) index_->Clear();
  std::unique_lock<std::shared_mutex> lock(depth_mu_);
  depth_map_.clear();
  group_map_.clear();
}

bool LocalStorageManager::Flush() {
  bool ok = true;
  for (auto& entry : tiers_) {
    if (!entry.backend->Flush()) ok = false;
  }
  return ok;
}

std::vector<bool> LocalStorageManager::BatchWrite(const std::vector<std::string>& keys,
                                                  const std::vector<const void*>& data_ptrs,
                                                  const std::vector<size_t>& sizes,
                                                  StorageTier tier) {
  TierBackend* target = GetTier(tier);
  if (!target) return std::vector<bool>(keys.size(), false);

  auto results = target->BatchWrite(keys, data_ptrs, sizes);

  // Fallback: retry failed items one-by-one with eviction support
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i]) {
      results[i] = Write(keys[i], data_ptrs[i], sizes[i], tier);
    }
  }
  return results;
}

std::vector<bool> LocalStorageManager::BatchReadIntoPtr(const std::vector<std::string>& keys,
                                                        const std::vector<uintptr_t>& dst_ptrs,
                                                        const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = ReadIntoPtr(keys[i], dst_ptrs[i], sizes[i]);
  }
  return results;
}

}  // namespace mori::umbp

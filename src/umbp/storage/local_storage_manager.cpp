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
#include "umbp/storage/local_storage_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "umbp/common/log.h"
#include "umbp/storage/dram_tier.h"
#include "umbp/storage/ssd_tier.h"
#ifdef USE_SPDK
#include "umbp/storage/spdk_ssd_tier.h"
#endif
#ifdef __linux__
#include "umbp/storage/spdk_proxy_tier.h"
#include "umbp/proxy/spdk_proxy_shm.h"
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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
  if (config_.eviction_policy != "prefix_aware_lru") {
    return tier->GetLRUKey();
  }

  size_t window = config_.eviction_candidate_window;
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

// ---------------------------------------------------------------------------
// Destructor — graceful proxy shutdown if we spawned it
// ---------------------------------------------------------------------------
LocalStorageManager::~LocalStorageManager() {
#ifdef __linux__
  // Destroy all tiers FIRST so SpdkProxyTier deregisters (active_ranks--)
  // before we signal the proxy to shut down.
  tiers_.clear();

  if (proxy_child_pid_ > 0) {
    // Set the shutdown flag in SHM so proxy knows it can exit
    umbp::proxy::ProxyShmRegion shm;
    if (shm.Attach(proxy_shm_name_) == 0) {
      shm.Header()->shutdown_requested.store(1, std::memory_order_release);
    }

    // Wait for proxy to exit gracefully (up to 10 seconds)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool exited = false;
    while (std::chrono::steady_clock::now() < deadline) {
      int status = 0;
      pid_t r = waitpid(proxy_child_pid_, &status, WNOHANG);
      if (r == proxy_child_pid_ || r == -1) {
        exited = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!exited) {
      UMBP_LOG_WARN("LSM: proxy pid=%d did not exit in time, sending SIGKILL",
                     proxy_child_pid_);
      kill(proxy_child_pid_, SIGKILL);
      waitpid(proxy_child_pid_, nullptr, 0);
    }
    proxy_child_pid_ = -1;

    // Safety net: always remove SHM after proxy has exited.
    // Covers kill -9 on daemon where it couldn't run its own shm_unlink.
    umbp::proxy::ProxyShmRegion::CleanupStale(proxy_shm_name_);
  }
#endif
}

// ---------------------------------------------------------------------------
// Auto-fork proxy daemon (Linux only)
// ---------------------------------------------------------------------------
#ifdef __linux__
static std::string FindProxyBinary(const std::string& explicit_path) {
  if (!explicit_path.empty()) {
    if (access(explicit_path.c_str(), X_OK) == 0) return explicit_path;
    UMBP_LOG_WARN("LSM: UMBP_SPDK_PROXY_BIN='%s' not executable",
                   explicit_path.c_str());
  }

  // Try same directory as current executable
  char exe_buf[4096];
  ssize_t len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (len > 0) {
    exe_buf[len] = '\0';
    std::string dir(exe_buf);
    size_t slash = dir.rfind('/');
    if (slash != std::string::npos) {
      dir.resize(slash + 1);
      std::string candidate = dir + "spdk_proxy";
      if (access(candidate.c_str(), X_OK) == 0) return candidate;
    }
  }

  // Fall back to PATH search (execlp will handle it)
  return "spdk_proxy";
}

int LocalStorageManager::SpawnProxyDaemon() {
  // Always clean-start: kill any existing daemon, then fork a fresh one.
  int probe = umbp::proxy::ProxyShmRegion::ProbeExisting(proxy_shm_name_);
  if (probe == 1 || probe == -1) {
    UMBP_LOG_INFO("LSM: killing existing proxy on SHM '%s'", proxy_shm_name_.c_str());
    umbp::proxy::ProxyShmRegion tmp;
    if (tmp.Attach(proxy_shm_name_) == 0) {
      auto* hdr = tmp.Header();
      uint32_t old_pid = hdr->proxy_pid.load(std::memory_order_relaxed);
      if (old_pid > 0) {
        hdr->shutdown_requested.store(1, std::memory_order_release);
        kill(static_cast<pid_t>(old_pid), SIGTERM);
        for (int i = 0; i < 50 && kill(static_cast<pid_t>(old_pid), 0) == 0; ++i)
          usleep(100000);  // 5s total
        if (kill(static_cast<pid_t>(old_pid), 0) == 0) {
          UMBP_LOG_WARN("LSM: old proxy pid=%u did not exit, sending SIGKILL", old_pid);
          kill(static_cast<pid_t>(old_pid), SIGKILL);
        }
      }
    }
  }

  umbp::proxy::ProxyShmRegion::CleanupStale(proxy_shm_name_);

  std::string bin = FindProxyBinary(config_.spdk_proxy_bin);

  pid_t pid = fork();
  if (pid < 0) {
    UMBP_LOG_ERROR("LSM: fork() failed: %s", strerror(errno));
    return -1;
  }

  if (pid == 0) {
    // ---- Child process ----
    std::string spawner_pid_str = std::to_string(getppid());
    execlp(bin.c_str(), "spdk_proxy",
           "--spawner-pid", spawner_pid_str.c_str(),
           static_cast<char*>(nullptr));
    fprintf(stderr, "[UMBP ERROR] execlp('%s') failed: %s\n",
            bin.c_str(), strerror(errno));
    _exit(127);
  }

  proxy_child_pid_ = pid;
  UMBP_LOG_INFO("LSM: spawned spdk_proxy daemon pid=%d", pid);
  return 0;
}
#endif

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
LocalStorageManager::LocalStorageManager(const UMBPConfig& config, BlockIndexClient* index)
    : config_(config), role_(config.ResolveRole()), index_(index) {
  // DRAM tier is always present (fastest)
  tiers_.push_back({StorageTier::CPU_DRAM,
                    std::make_unique<DRAMTier>(config_.dram_capacity_bytes,
                                               config_.use_shared_memory, config_.shm_name)});

  // SSD tier is optional (slower)
  if (config_.ssd_enabled) {
    std::unique_ptr<TierBackend> ssd_backend;
    bool use_proxy = false;

    if (config_.ssd_backend == "spdk") {
      if (role_ == UMBPRole::Standalone) {
        // Direct in-process SPDK (best perf, single process only)
#ifdef USE_SPDK
        auto spdk_tier = std::make_unique<SpdkSsdTier>(config_);
        if (spdk_tier->IsValid()) {
          ssd_backend = std::move(spdk_tier);
        } else {
          fprintf(stderr, "[UMBP WARN] SpdkSsdTier init failed, falling back to POSIX SSD\n");
        }
#else
        fprintf(stderr,
                "[UMBP WARN] UMBP_SSD_BACKEND=spdk requested, but this build was compiled "
                "without SPDK support. Falling back to POSIX SSD.\n");
#endif
      } else {
        // Leader or Follower → use proxy daemon for multi-process SPDK
        use_proxy = true;
      }
    } else if (config_.ssd_backend == "spdk_proxy") {
      use_proxy = true;  // backward compatible explicit proxy
    }

#ifdef __linux__
    if (use_proxy && !ssd_backend) {
      proxy_shm_name_ = config_.spdk_proxy_shm_name;
      if (proxy_shm_name_.empty())
        proxy_shm_name_ = umbp::proxy::kDefaultShmName;

      // Leader always spawns. For Standalone with explicit "spdk_proxy" backend,
      // spawn if no daemon is already running (auto-elect as spawner).
      bool should_spawn = (role_ == UMBPRole::SharedSSDLeader);
      if (!should_spawn && config_.ssd_backend == "spdk_proxy" &&
          role_ == UMBPRole::Standalone) {
        int probe = umbp::proxy::ProxyShmRegion::ProbeExisting(proxy_shm_name_);
        if (probe == 0) should_spawn = true;  // no daemon → we spawn
      }

      if (should_spawn) {
        SpawnProxyDaemon();
      }

      bool proxy_ready = SpdkProxyTier::WaitForProxy(
          proxy_shm_name_, config_.spdk_proxy_startup_timeout_ms);

      if (proxy_ready) {
        auto proxy_tier = std::make_unique<SpdkProxyTier>(config_);
        if (proxy_tier->IsValid()) {
          ssd_backend = std::move(proxy_tier);
        }
      }

      if (!ssd_backend) {
        fprintf(stderr,
                "[UMBP WARN] SPDK proxy connect failed. Falling back to POSIX SSD.\n");
      }
    }
#endif
    if (!ssd_backend) {
      SSDAccessMode ssd_access_mode = SSDAccessMode::ReadWrite;
      if (role_ == UMBPRole::SharedSSDFollower) {
        ssd_access_mode = SSDAccessMode::ReadOnlyShared;
      }
      ssd_backend = std::make_unique<SSDTier>(
          config_.ssd_storage_dir, config_.ssd_capacity_bytes, ssd_access_mode);
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

bool LocalStorageManager::MoveKey(const std::string& key, TierBackend* from, TierBackend* to) {
  auto data = from->Read(key);
  if (data.empty()) return false;

  if (!to->Write(key, data.data(), data.size())) return false;

  from->Evict(key);

  UpsertIndexTier(key, to->tier_id(), data.size());
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
  for (size_t i = 0; i < tiers_.size(); ++i) {
    if (tiers_[i].backend->Exists(key)) {
      bool ok = tiers_[i].backend->ReadIntoPtr(key, dst, size);
      // Auto-promote if read from a slower tier
      if (ok && i > 0 && config_.auto_promote_on_read) {
        MaybeAutoPromote(key);
      }
      return ok;
    }
  }
  return false;
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
  if (ok && index_) index_->Remove(key);
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
    from->Evict(key);
    UpsertIndexTier(key, to->tier_id(), data.size());
    return true;
  }

  if (!to->Write(key, data.data(), data.size())) {
    // Target full — demote LRU keys from target to make room
    while (DemoteLRUForSpace(to)) {
      if (to->Write(key, data.data(), data.size())) {
        from->Evict(key);
        UpsertIndexTier(key, to->tier_id(), data.size());
        return true;
      }
    }
    return false;  // Cannot promote
  }

  from->Evict(key);
  UpsertIndexTier(key, to->tier_id(), data.size());
  return true;
}

bool LocalStorageManager::CopyToSSD(const std::string& key) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;

  TierBackend* dram = GetTier(StorageTier::CPU_DRAM);
  TierBackend* ssd = GetTier(StorageTier::LOCAL_SSD);
  if (!dram || !ssd) return false;

  // Already on SSD
  if (ssd->Exists(key)) return true;

  auto data = dram->Read(key);
  if (data.empty()) return false;

  if (ssd->Write(key, data.data(), data.size())) return true;

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
    if (ssd->Write(key, data.data(), data.size())) return true;
  }
}

void LocalStorageManager::MaybeAutoPromote(const std::string& key) {
  if (!config_.auto_promote_on_read) return;

  // Only promote if key is not in the fastest tier
  if (tiers_.empty()) return;
  TierBackend* fastest = tiers_.front().backend.get();
  if (fastest->Exists(key)) return;

  // Check fastest tier watermark before promoting
  auto [used, total] = fastest->Capacity();
  double utilization = static_cast<double>(used) / total;
  if (utilization >= config_.dram_high_watermark) return;

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

std::vector<bool> LocalStorageManager::BatchWrite(
    const std::vector<std::string>& keys,
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

std::vector<bool> LocalStorageManager::BatchReadIntoPtr(
    const std::vector<std::string>& keys,
    const std::vector<uintptr_t>& dst_ptrs,
    const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = ReadIntoPtr(keys[i], dst_ptrs[i], sizes[i]);
  }
  return results;
}

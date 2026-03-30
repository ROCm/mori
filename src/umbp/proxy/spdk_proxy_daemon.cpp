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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// spdk_proxy: multitenant daemon that owns SPDK + NVMe and serves many
// tenant-bound client sessions over shared memory channels.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "umbp/allocator/offset_allocator.hpp"
#include "umbp/common/config.h"
#include "umbp/common/log.h"
#include "umbp/local/storage/spdk_ssd_tier.h"
#include "umbp/proxy/spdk_proxy_protocol.h"
#include "umbp/proxy/spdk_proxy_shm.h"
#include "umbp/spdk/spdk_env.h"

#ifdef __linux__
#include <unistd.h>
#endif

using namespace mori::umbp;
using namespace umbp::proxy;

namespace {

#if defined(__x86_64__) || defined(_M_X64)
#define CPU_PAUSE() _mm_pause()
#else
#define CPU_PAUSE() ((void)0)
#endif

std::atomic<bool> g_running{true};
std::string g_shm_name;

size_t AlignUp(size_t value, size_t alignment) {
  if (alignment == 0) return value;
  return ((value + alignment - 1) / alignment) * alignment;
}

struct WbFlushTask {
  std::vector<std::string> keys;
  std::vector<std::vector<char>> staged_bufs;
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
};

class WbFlushQueue {
 public:
  void Start(SpdkSsdTier* tier) {
    tier_ = tier;
    thread_ = std::thread([this] { Run(); });
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  void Push(WbFlushTask&& task) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(task));
      ++pending_;
      ++submit_seq_;
    }
    cv_.notify_one();
  }

  void Drain() {
    std::unique_lock<std::mutex> lk(mu_);
    uint64_t target = submit_seq_;
    done_cv_.wait(lk, [&] { return done_seq_ >= target || stop_; });
  }

 private:
  void Run() {
    while (true) {
      WbFlushTask task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !queue_.empty() || stop_; });
        if (stop_ && queue_.empty()) break;
        task = std::move(queue_.front());
        queue_.pop();
      }
      tier_->BatchWriteStreaming(task.keys, task.ptrs, task.sizes, nullptr, task.offsets, nullptr,
                                 0);
      {
        std::lock_guard<std::mutex> lk(mu_);
        --pending_;
        ++done_seq_;
      }
      done_cv_.notify_all();
    }
  }

  SpdkSsdTier* tier_ = nullptr;
  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::queue<WbFlushTask> queue_;
  int pending_ = 0;
  uint64_t submit_seq_ = 0;
  uint64_t done_seq_ = 0;
  bool stop_ = false;
};

void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

void atexit_cleanup() {
#ifdef __linux__
  if (!g_shm_name.empty()) ProxyShmRegion::CleanupStale(g_shm_name);
#endif
}

void RingMemcpyIn(char* ring_data, uint64_t capacity, uint64_t abs_offset, const void* src,
                  size_t size) {
  size_t off = static_cast<size_t>(abs_offset % capacity);
  if (off + size <= capacity) {
    std::memcpy(ring_data + off, src, size);
  } else {
    size_t first = static_cast<size_t>(capacity - off);
    std::memcpy(ring_data + off, src, first);
    std::memcpy(ring_data, static_cast<const char*>(src) + first, size - first);
  }
}

void WriteShmCache(ProxyShmRegion& shm, uint32_t tenant_id, uint64_t cache_epoch,
                   const std::string& key, const void* data, size_t size) {
  auto* hdr = shm.Header();
  auto* ctrl = GetCacheRingControl(shm.Base(), hdr);
  if (!ctrl || size == 0) return;

  uint64_t cap = ctrl->capacity;
  if (size > cap) return;

  size_t aligned = (size + kCacheRingAlign - 1) & ~(kCacheRingAlign - 1);
  uint64_t pos = ctrl->write_pos.fetch_add(aligned, std::memory_order_relaxed);
  char* ring_data = GetCacheRingData(shm.Base(), hdr);
  RingMemcpyIn(ring_data, cap, pos, data, size);

  auto* index = GetCacheIndex(shm.Base(), hdr);
  uint32_t idx = CacheIndexHash(key.data(), static_cast<uint32_t>(key.size()), tenant_id,
                                hdr->cache_index_slots);
  auto& entry = index[idx];

  uint64_t old_gen = entry.gen.load(std::memory_order_relaxed);
  if (old_gen & 1) return;
  if (!entry.gen.compare_exchange_strong(old_gen, old_gen + 1, std::memory_order_acq_rel)) {
    return;
  }

  entry.tenant_id = tenant_id;
  entry.key_len = static_cast<uint32_t>(key.size());
  entry.tenant_cache_epoch = cache_epoch;
  std::memcpy(entry.key, key.data(), key.size());
  entry.ring_offset = pos;
  entry.data_size = size;

  entry.gen.store(old_gen + 2, std::memory_order_release);
}

void UpdateRingCacheIndex(ProxyShmRegion& shm, uint32_t tenant_id, uint64_t cache_epoch,
                          const std::string& key, uint64_t ring_offset, size_t size) {
  auto* hdr = shm.Header();
  auto* index = GetCacheIndex(shm.Base(), hdr);
  if (!index || hdr->cache_index_slots == 0) return;

  uint32_t idx = CacheIndexHash(key.data(), static_cast<uint32_t>(key.size()), tenant_id,
                                hdr->cache_index_slots);
  auto& entry = index[idx];

  uint64_t old_gen = entry.gen.load(std::memory_order_relaxed);
  if (old_gen & 1) return;
  if (!entry.gen.compare_exchange_strong(old_gen, old_gen + 1, std::memory_order_acq_rel)) {
    return;
  }

  entry.tenant_id = tenant_id;
  entry.key_len = static_cast<uint32_t>(key.size());
  entry.tenant_cache_epoch = cache_epoch;
  std::memcpy(entry.key, key.data(), key.size());
  entry.ring_offset = ring_offset;
  entry.data_size = size;

  entry.gen.store(old_gen + 2, std::memory_order_release);
}

struct TenantRuntime {
  uint32_t tenant_id = 0;
  uint32_t tenant_slot = 0;
  size_t quota_bytes = 0;
  size_t reserved_bytes = 0;
  uint32_t active_sessions = 0;
  uint64_t last_activity_ms = 0;
  uint64_t zero_sessions_since_ms = 0;
  std::atomic<uint64_t> cache_epoch{0};
  std::atomic<uint32_t> inflight_batch_workers{0};
  std::atomic<uint32_t> inflight_wb_workers{0};
  std::optional<umbp::offset_allocator::OffsetAllocationHandle> reservation;
  std::unique_ptr<SpdkSsdTier> tier;
  WbFlushQueue wb_queue;
  bool wb_queue_started = false;
};

class TenantRegistry {
 public:
  TenantRegistry(ProxyShmRegion& shm, const UMBPConfig& tier_cfg, size_t total_capacity_bytes,
                 size_t reserved_shared_bytes, size_t default_tenant_quota_bytes,
                 uint32_t block_size, bool allow_borrow, bool write_back, uint64_t tenant_grace_ms,
                 SpdkSsdTier::SharedDmaPool shared_dma_pool)
      : shm_(shm),
        tier_cfg_(tier_cfg),
        total_capacity_bytes_(total_capacity_bytes),
        reserved_shared_bytes_(std::min(reserved_shared_bytes, total_capacity_bytes)),
        usable_capacity_bytes_(total_capacity_bytes_ - reserved_shared_bytes_),
        default_tenant_quota_bytes_(default_tenant_quota_bytes),
        block_size_(std::max<uint32_t>(block_size, 4096)),
        allow_borrow_(allow_borrow),
        write_back_(write_back),
        tenant_grace_ms_(tenant_grace_ms),
        shared_dma_pool_(std::move(shared_dma_pool)) {
    if (usable_capacity_bytes_ > 0) {
      quota_allocator_ = umbp::offset_allocator::OffsetAllocator::createAligned(
          reserved_shared_bytes_, usable_capacity_bytes_, block_size_);
    }
  }

  ~TenantRegistry() {
    for (auto& [_, tenant] : tenants_) {
      WaitForTenantIdle(*tenant);
      if (tenant->wb_queue_started) {
        tenant->wb_queue.Stop();
        tenant->wb_queue_started = false;
      }
    }
  }

  size_t usable_capacity_bytes() const { return usable_capacity_bytes_; }
  bool HasInflightBatchWorkers() const {
    for (const auto& [_, tenant] : tenants_) {
      if (tenant->inflight_batch_workers.load(std::memory_order_acquire) > 0) {
        return true;
      }
      if (tenant->inflight_wb_workers.load(std::memory_order_acquire) > 0) {
        return true;
      }
    }
    return false;
  }

  TenantRuntime* FindTenant(uint32_t tenant_id) {
    auto it = tenants_.find(tenant_id);
    return (it != tenants_.end()) ? it->second.get() : nullptr;
  }

  TenantRuntime* FindTenantByChannel(const ClientChannel& ch) {
    auto it = tenants_.find(ch.tenant_id);
    if (it == tenants_.end()) return nullptr;
    if (it->second->tenant_slot != ch.tenant_slot) return nullptr;
    return it->second.get();
  }

  void SyncTenantTelemetry(TenantRuntime& tenant) { PublishTenantInfo(tenant); }

  ResultCode AttachChannel(ClientChannel& ch, uint64_t quota_hint, uint64_t* session_id_out) {
    auto* hdr = shm_.Header();
    uint64_t now_ms = NowEpochMs();
    auto it = tenants_.find(ch.tenant_id);
    if (it == tenants_.end()) {
      size_t quota = ResolveQuotaBytes(quota_hint);
      if (quota == 0 || !quota_allocator_) {
        return ResultCode::NO_SPACE;
      }

      auto reservation = quota_allocator_->allocate(quota);
      if (!reservation.has_value()) {
        return ResultCode::NO_SPACE;
      }

      uint32_t tenant_slot = AllocateTenantSlot();
      if (tenant_slot == UINT32_MAX) {
        return ResultCode::NO_SPACE;
      }

      auto tenant = std::make_unique<TenantRuntime>();
      tenant->tenant_id = ch.tenant_id;
      tenant->tenant_slot = tenant_slot;
      tenant->quota_bytes = quota;
      tenant->reserved_bytes = quota;
      tenant->last_activity_ms = now_ms;
      tenant->zero_sessions_since_ms = 0;
      tenant->cache_epoch.store(hdr->next_cache_epoch.fetch_add(1, std::memory_order_relaxed) + 1,
                                std::memory_order_relaxed);
      tenant->reservation = std::move(reservation);
      tenant->tier = std::make_unique<SpdkSsdTier>(tier_cfg_, tenant->reservation->address(),
                                                   tenant->quota_bytes, shared_dma_pool_);
      if (!tenant->tier || !tenant->tier->IsValid()) {
        ResetTenantInfo(tenant_slot);
        return ResultCode::ERROR;
      }
      if (write_back_) {
        tenant->wb_queue.Start(tenant->tier.get());
        tenant->wb_queue_started = true;
      }
      PublishTenantInfo(*tenant);
      it = tenants_.emplace(ch.tenant_id, std::move(tenant)).first;
    }

    TenantRuntime& tenant = *it->second;
    if (quota_hint > tenant.quota_bytes) {
      UMBP_LOG_WARN(
          "spdk_proxy: tenant %u quota_hint %zu exceeds allocated %zu; "
          "continuing with daemon quota",
          tenant.tenant_id, static_cast<size_t>(quota_hint), tenant.quota_bytes);
    }

    tenant.active_sessions += 1;
    tenant.zero_sessions_since_ms = 0;
    tenant.last_activity_ms = now_ms;
    uint64_t session_id = hdr->next_session_id.fetch_add(1, std::memory_order_relaxed) + 1;
    ch.tenant_slot = tenant.tenant_slot;
    ch.session_id.store(session_id, std::memory_order_release);
    ch.requested_quota_bytes = tenant.quota_bytes;
    ch.last_activity_ms.store(now_ms, std::memory_order_release);
    hdr->active_sessions.fetch_add(1, std::memory_order_release);
    hdr->last_activity_ms.store(now_ms, std::memory_order_relaxed);
    PublishTenantInfo(tenant);

    if (session_id_out) *session_id_out = session_id;
    return ResultCode::OK;
  }

  ResultCode DetachChannel(ClientChannel& ch) {
    uint64_t session_id = ch.session_id.load(std::memory_order_acquire);
    if (session_id == 0) return ResultCode::OK;

    auto it = tenants_.find(ch.tenant_id);
    if (it == tenants_.end()) return ResultCode::NOT_FOUND;

    TenantRuntime& tenant = *it->second;
    if (tenant.active_sessions > 0) {
      tenant.active_sessions -= 1;
      shm_.Header()->active_sessions.fetch_sub(1, std::memory_order_release);
    }
    tenant.last_activity_ms = NowEpochMs();
    if (tenant.active_sessions == 0) {
      tenant.zero_sessions_since_ms = tenant.last_activity_ms;
    }

    ch.session_id.store(0, std::memory_order_release);
    ch.tenant_slot = tenant.tenant_slot;
    ch.last_activity_ms.store(tenant.last_activity_ms, std::memory_order_release);
    shm_.Header()->last_activity_ms.store(tenant.last_activity_ms, std::memory_order_relaxed);
    PublishTenantInfo(tenant);
    return ResultCode::OK;
  }

  void ForceDetachChannel(ClientChannel& ch) {
    uint32_t t = ch.tail.load(std::memory_order_relaxed);
    uint32_t h = ch.head.load(std::memory_order_relaxed);
    while (t != h) {
      auto& slot = ch.slots[t % kRingSize];
      slot.result = static_cast<int32_t>(ResultCode::ERROR);
      slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED), std::memory_order_release);
      t = (t + 1) % kRingSize;
    }
    ch.tail.store(t, std::memory_order_release);

    (void)DetachChannel(ch);
    ResetChannelTransport(ch, /*release_owner=*/true);
  }

  void WaitForTenantWritebackDrained(TenantRuntime& tenant) {
    if (!write_back_ || !tenant.wb_queue_started) return;
    while (true) {
      while (tenant.inflight_wb_workers.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      tenant.wb_queue.Drain();
      if (tenant.inflight_wb_workers.load(std::memory_order_acquire) == 0) break;
    }
  }

  void WaitForTenantIdle(TenantRuntime& tenant) {
    while (tenant.inflight_batch_workers.load(std::memory_order_acquire) > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    WaitForTenantWritebackDrained(tenant);
    while (tenant.inflight_batch_workers.load(std::memory_order_acquire) > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  void BumpTenantCacheEpoch(TenantRuntime& tenant) {
    uint64_t epoch = shm_.Header()->next_cache_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    tenant.cache_epoch.store(epoch, std::memory_order_relaxed);
    PublishTenantInfo(tenant);
  }

  void SyncTelemetry() {
    size_t total_used = 0;
    for (auto& [_, tenant] : tenants_) {
      PublishTenantInfo(*tenant);
      auto [used, _quota] = tenant->tier->Capacity();
      total_used += used;
    }
    auto* hdr = shm_.Header();
    hdr->capacity_used.store(total_used, std::memory_order_relaxed);
    hdr->capacity_total.store(usable_capacity_bytes_, std::memory_order_relaxed);
  }

  void ReapInactiveTenants() {
    if (tenant_grace_ms_ == 0) return;
    uint64_t now_ms = NowEpochMs();
    std::vector<uint32_t> to_reap;
    for (const auto& [tenant_id, tenant] : tenants_) {
      if (tenant->active_sessions != 0 || tenant->zero_sessions_since_ms == 0) continue;
      if ((now_ms - tenant->zero_sessions_since_ms) < tenant_grace_ms_) continue;
      if (tenant->inflight_batch_workers.load(std::memory_order_acquire) > 0) continue;
      if (tenant->inflight_wb_workers.load(std::memory_order_acquire) > 0) continue;
      to_reap.push_back(tenant_id);
    }

    for (uint32_t tenant_id : to_reap) {
      auto it = tenants_.find(tenant_id);
      if (it == tenants_.end()) continue;
      TenantRuntime& tenant = *it->second;
      WaitForTenantIdle(tenant);
      if (tenant.wb_queue_started) {
        tenant.wb_queue.Stop();
        tenant.wb_queue_started = false;
      }
      tenant.tier->Clear();
      ResetTenantInfo(tenant.tenant_slot);
      tenants_.erase(it);
    }
  }

  static void ResetChannelTransport(ClientChannel& ch, bool release_owner) {
    ch.connected.store(0, std::memory_order_release);
    ch.client_pid = 0;
    ch.tenant_id = 0;
    ch.tenant_slot = 0;
    ch.requested_quota_bytes = 0;
    ch.session_id.store(0, std::memory_order_release);
    ch.last_activity_ms.store(0, std::memory_order_release);
    ch.head.store(0, std::memory_order_relaxed);
    ch.tail.store(0, std::memory_order_relaxed);
    for (uint32_t s = 0; s < kRingSize; ++s) {
      ch.slots[s].state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_relaxed);
    }
    if (release_owner) {
      ch.owner_pid.store(0, std::memory_order_release);
    }
  }

 private:
  size_t ResolveQuotaBytes(uint64_t quota_hint) const {
    uint64_t resolved = quota_hint;
    if (resolved == 0) {
      resolved =
          (default_tenant_quota_bytes_ > 0) ? default_tenant_quota_bytes_ : usable_capacity_bytes_;
    }
    if (resolved == 0 || usable_capacity_bytes_ == 0) return 0;
    resolved = std::min<uint64_t>(resolved, usable_capacity_bytes_);
    uint64_t aligned = ((resolved + block_size_ - 1) / block_size_) * block_size_;
    if (aligned > usable_capacity_bytes_) {
      aligned = (usable_capacity_bytes_ / block_size_) * block_size_;
    }
    return static_cast<size_t>(aligned);
  }

  uint32_t AllocateTenantSlot() {
    auto* hdr = shm_.Header();
    for (uint32_t slot = 0; slot < hdr->max_tenants; ++slot) {
      auto* info = shm_.Tenant(slot);
      if (info->flags.load(std::memory_order_acquire) == 0) {
        return slot;
      }
    }
    return UINT32_MAX;
  }

  void ResetTenantInfo(uint32_t tenant_slot) {
    auto* info = shm_.Tenant(tenant_slot);
    if (!info) return;
    info->tenant_id = 0;
    info->tenant_slot = tenant_slot;
    info->active_sessions.store(0, std::memory_order_relaxed);
    info->flags.store(0, std::memory_order_release);
    info->used_bytes.store(0, std::memory_order_relaxed);
    info->quota_bytes.store(0, std::memory_order_relaxed);
    info->reserved_bytes.store(0, std::memory_order_relaxed);
    info->evicted_bytes.store(0, std::memory_order_relaxed);
    info->hit_count.store(0, std::memory_order_relaxed);
    info->miss_count.store(0, std::memory_order_relaxed);
    info->last_activity_ms.store(0, std::memory_order_relaxed);
    info->cache_epoch.store(0, std::memory_order_relaxed);
  }

  void PublishTenantInfo(TenantRuntime& tenant) {
    auto* info = shm_.Tenant(tenant.tenant_slot);
    if (!info) return;

    auto [used, _quota] = tenant.tier->Capacity();
    auto stats = tenant.tier->GetStats();
    uint64_t now_ms = NowEpochMs();

    info->tenant_id = tenant.tenant_id;
    info->tenant_slot = tenant.tenant_slot;
    info->active_sessions.store(tenant.active_sessions, std::memory_order_relaxed);
    uint32_t flags = kTenantFlagActive;
    if (tenant.active_sessions == 0 && tenant.zero_sessions_since_ms != 0 &&
        (tenant_grace_ms_ == 0 || (now_ms - tenant.zero_sessions_since_ms) >= tenant_grace_ms_)) {
      flags |= kTenantFlagReaping;
    }
    info->flags.store(flags, std::memory_order_release);
    info->used_bytes.store(used, std::memory_order_relaxed);
    info->quota_bytes.store(tenant.quota_bytes, std::memory_order_relaxed);
    info->reserved_bytes.store(tenant.reserved_bytes, std::memory_order_relaxed);
    info->evicted_bytes.store(stats.evicted_bytes, std::memory_order_relaxed);
    info->hit_count.store(stats.hit_count, std::memory_order_relaxed);
    info->miss_count.store(stats.miss_count, std::memory_order_relaxed);
    info->last_activity_ms.store(tenant.last_activity_ms, std::memory_order_relaxed);
    info->cache_epoch.store(tenant.cache_epoch.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
  }

  ProxyShmRegion& shm_;
  UMBPConfig tier_cfg_;
  size_t total_capacity_bytes_ = 0;
  size_t reserved_shared_bytes_ = 0;
  size_t usable_capacity_bytes_ = 0;
  size_t default_tenant_quota_bytes_ = 0;
  uint32_t block_size_ = 4096;
  bool allow_borrow_ = false;
  bool write_back_ = false;
  uint64_t tenant_grace_ms_ = 0;
  SpdkSsdTier::SharedDmaPool shared_dma_pool_;
  std::shared_ptr<umbp::offset_allocator::OffsetAllocator> quota_allocator_;
  std::unordered_map<uint32_t, std::unique_ptr<TenantRuntime>> tenants_;
};

void ProcessSingleRequest(TenantRegistry& tenants, TenantRuntime& tenant, ProxyShmRegion& shm,
                          RingSlot& slot, void* data_region, size_t region_size, bool write_back) {
  std::string key(slot.key, slot.key_len);
  auto type = static_cast<RequestType>(slot.type);
  uint64_t cache_epoch = tenant.cache_epoch.load(std::memory_order_acquire);

  auto* ring_hdr = shm.Header();
  auto* ring_ctrl = GetCacheRingControl(shm.Base(), ring_hdr);
  char* ring_data_ptr = ring_ctrl ? GetCacheRingData(shm.Base(), ring_hdr) : nullptr;
  uint64_t ring_cap = ring_ctrl ? ring_ctrl->capacity : 0;

  switch (type) {
    case RequestType::PUT: {
      if (slot.data_size > region_size && slot.ring_data_base == 0) {
        slot.result = static_cast<int32_t>(ResultCode::ERROR);
        break;
      }
      const void* src = data_region;
      if (slot.ring_data_base != 0 && ring_data_ptr) {
        src = ring_data_ptr + static_cast<size_t>(slot.ring_data_base % ring_cap);
      }
      bool ok = tenant.tier->Write(key, src, slot.data_size);
      slot.result =
          ok ? static_cast<int32_t>(ResultCode::OK) : static_cast<int32_t>(ResultCode::NO_SPACE);
      if (ok) {
        if (slot.ring_data_base != 0 && ring_data_ptr) {
          UpdateRingCacheIndex(shm, tenant.tenant_id, cache_epoch, key, slot.ring_data_base,
                               slot.data_size);
        } else {
          WriteShmCache(shm, tenant.tenant_id, cache_epoch, key, src, slot.data_size);
        }
      }
      break;
    }
    case RequestType::GET: {
      size_t max_read = std::min(static_cast<size_t>(slot.data_size), region_size);
      if (max_read == 0) max_read = region_size;

      bool use_ring_read = false;
      uint64_t read_ring_base = 0;
      uintptr_t dst = reinterpret_cast<uintptr_t>(data_region);
      if (ring_ctrl) {
        read_ring_base = AllocRingContiguous(ring_ctrl, max_read);
        if (read_ring_base != UINT64_MAX) {
          dst = reinterpret_cast<uintptr_t>(ring_data_ptr +
                                            static_cast<size_t>(read_ring_base % ring_cap));
          use_ring_read = true;
        }
      }

      bool ok = tenant.tier->ReadIntoPtr(key, dst, max_read);
      if (ok) {
        tenant.tier->RecordHit();
        slot.result = static_cast<int32_t>(ResultCode::OK);
        slot.result_size = max_read;
        if (use_ring_read) {
          slot.ring_data_base = read_ring_base;
          UpdateRingCacheIndex(shm, tenant.tenant_id, cache_epoch, key, read_ring_base, max_read);
        } else {
          slot.ring_data_base = 0;
          WriteShmCache(shm, tenant.tenant_id, cache_epoch, key, reinterpret_cast<void*>(dst),
                        max_read);
        }
      } else {
        tenant.tier->RecordMiss();
        slot.result = static_cast<int32_t>(ResultCode::NOT_FOUND);
        slot.result_size = 0;
      }
      break;
    }
    case RequestType::EXISTS: {
      bool ok = tenant.tier->Exists(key);
      slot.result =
          ok ? static_cast<int32_t>(ResultCode::OK) : static_cast<int32_t>(ResultCode::NOT_FOUND);
      break;
    }
    case RequestType::REMOVE: {
      tenants.WaitForTenantIdle(tenant);
      bool ok = tenant.tier->Evict(key);
      slot.result =
          ok ? static_cast<int32_t>(ResultCode::OK) : static_cast<int32_t>(ResultCode::NOT_FOUND);
      if (ok) tenants.BumpTenantCacheEpoch(tenant);
      break;
    }
    case RequestType::CLEAR_TENANT: {
      tenants.WaitForTenantIdle(tenant);
      tenant.tier->Clear();
      tenants.BumpTenantCacheEpoch(tenant);
      slot.result = static_cast<int32_t>(ResultCode::OK);
      break;
    }
    case RequestType::CAPACITY_TENANT: {
      auto [used, total] = tenant.tier->Capacity();
      slot.result = static_cast<int32_t>(ResultCode::OK);
      slot.result_size = used;
      slot.result_aux = total;
      break;
    }
    case RequestType::FLUSH_TENANT: {
      if (write_back) {
        tenants.WaitForTenantWritebackDrained(tenant);
      }
      slot.result = static_cast<int32_t>(ResultCode::OK);
      break;
    }
    case RequestType::GET_LRU_KEY: {
      std::string lru = tenant.tier->GetLRUKey();
      if (lru.empty()) {
        slot.result = static_cast<int32_t>(ResultCode::NOT_FOUND);
        slot.result_size = 0;
      } else {
        size_t copy_len = std::min(lru.size(), region_size - 1);
        std::memcpy(data_region, lru.data(), copy_len);
        static_cast<char*>(data_region)[copy_len] = '\0';
        slot.result = static_cast<int32_t>(ResultCode::OK);
        slot.result_size = copy_len;
      }
      break;
    }
    case RequestType::GET_TENANT_STATS: {
      if (region_size < sizeof(TenantStatsPayload)) {
        slot.result = static_cast<int32_t>(ResultCode::ERROR);
        break;
      }
      auto [used, _quota] = tenant.tier->Capacity();
      auto stats = tenant.tier->GetStats();
      TenantStatsPayload payload{};
      payload.used_bytes = used;
      payload.quota_bytes = tenant.quota_bytes;
      payload.reserved_bytes = tenant.reserved_bytes;
      payload.evicted_bytes = stats.evicted_bytes;
      payload.hit_count = stats.hit_count;
      payload.miss_count = stats.miss_count;
      payload.last_activity_ms = tenant.last_activity_ms;
      payload.cache_epoch = tenant.cache_epoch.load(std::memory_order_relaxed);
      payload.active_sessions = tenant.active_sessions;
      payload.flags = kTenantFlagActive;
      payload.tenant_id = tenant.tenant_id;
      payload.tenant_slot = tenant.tenant_slot;
      std::memcpy(data_region, &payload, sizeof(payload));
      slot.result = static_cast<int32_t>(ResultCode::OK);
      slot.result_size = sizeof(payload);
      break;
    }
    default:
      slot.result = static_cast<int32_t>(ResultCode::PERMISSION_DENIED);
      break;
  }

  tenant.last_activity_ms = NowEpochMs();
}

void ProcessBatchRequest(SpdkSsdTier& tier, ProxyShmRegion& shm, uint32_t tenant_id,
                         uint64_t cache_epoch, RingSlot& slot, void* data_region,
                         size_t region_size, bool write_back, void** dma_bufs = nullptr,
                         int dma_count = 0, WbFlushTask* wb_out = nullptr) {
  auto type = static_cast<RequestType>(slot.type);
  auto* desc = static_cast<BatchDescriptor*>(data_region);
  uint32_t count = desc->count;

  if (count == 0) {
    slot.result = static_cast<int32_t>(ResultCode::OK);
    return;
  }

  size_t desc_total = sizeof(BatchDescriptor) + count * sizeof(BatchEntry);
  size_t data_base_offset = (desc_total + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
  char* data_base = static_cast<char*>(data_region) + data_base_offset;

  if (type == RequestType::BATCH_PUT) {
    std::vector<std::string> keys(count);
    std::vector<size_t> sizes(count);
    std::vector<size_t> shm_offsets(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto& e = desc->entries[i];
      keys[i] = std::string(e.key, e.key_len);
      sizes[i] = e.data_size;
      shm_offsets[i] = e.data_offset;
    }

    auto* hdr = shm.Header();
    uint64_t ring_write_base = desc->ring_data_base;
    char* src_base = nullptr;
    bool use_ring_write = false;

    if (ring_write_base != 0) {
      auto* ctrl = GetCacheRingControl(shm.Base(), hdr);
      if (ctrl) {
        char* ring_data = GetCacheRingData(shm.Base(), hdr);
        uint64_t cap = ctrl->capacity;
        src_base = ring_data + static_cast<size_t>(ring_write_base % cap);
        use_ring_write = true;
      }
    }
    if (!src_base) src_base = data_base;

    std::vector<const void*> cptrs(count);
    for (uint32_t i = 0; i < count; ++i) {
      cptrs[i] = src_base + desc->entries[i].data_offset;
    }

    if (write_back && wb_out) {
      uint64_t total = desc->total_data_size;
      while (desc->bytes_ready.load(std::memory_order_acquire) < total) CPU_PAUSE();

      if (use_ring_write) {
        for (uint32_t i = 0; i < count; ++i) {
          UpdateRingCacheIndex(shm, tenant_id, cache_epoch, keys[i],
                               ring_write_base + desc->entries[i].data_offset, sizes[i]);
          desc->entries[i].result = 1;
        }
      } else {
        for (uint32_t i = 0; i < count; ++i) {
          WriteShmCache(shm, tenant_id, cache_epoch, keys[i], cptrs[i], sizes[i]);
          desc->entries[i].result = 1;
        }
      }

      wb_out->keys = std::move(keys);
      wb_out->sizes = std::move(sizes);
      wb_out->offsets = std::move(shm_offsets);
      wb_out->staged_bufs.resize(count);
      wb_out->ptrs.resize(count);
      for (uint32_t i = 0; i < count; ++i) {
        wb_out->staged_bufs[i].assign(static_cast<const char*>(cptrs[i]),
                                      static_cast<const char*>(cptrs[i]) + wb_out->sizes[i]);
        wb_out->ptrs[i] = wb_out->staged_bufs[i].data();
      }
    } else {
      std::thread shm_cache_thread;
      if (!use_ring_write) {
        shm_cache_thread =
            std::thread([&shm, tenant_id, cache_epoch, &keys, &cptrs, &sizes, count, desc]() {
              uint64_t total = desc->total_data_size;
              while (desc->bytes_ready.load(std::memory_order_acquire) < total) CPU_PAUSE();
              for (uint32_t i = 0; i < count; ++i) {
                WriteShmCache(shm, tenant_id, cache_epoch, keys[i], cptrs[i], sizes[i]);
              }
            });
      }

      auto results = tier.BatchWriteStreaming(keys, cptrs, sizes, &desc->bytes_ready, shm_offsets,
                                              dma_bufs, dma_count);

      if (shm_cache_thread.joinable()) shm_cache_thread.join();

      if (use_ring_write) {
        for (uint32_t i = 0; i < count; ++i) {
          if (results[i]) {
            UpdateRingCacheIndex(shm, tenant_id, cache_epoch, keys[i],
                                 ring_write_base + desc->entries[i].data_offset, sizes[i]);
          }
        }
      }

      for (uint32_t i = 0; i < count; ++i) {
        desc->entries[i].result = results[i] ? 1 : 0;
      }
    }
  } else {
    std::vector<std::string> keys(count);
    std::vector<size_t> sizes(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto& e = desc->entries[i];
      keys[i] = std::string(e.key, e.key_len);
      sizes[i] = e.data_size;
    }

    desc->bytes_done.store(0, std::memory_order_relaxed);
    desc->items_done.store(0, std::memory_order_relaxed);

    auto* hdr = shm.Header();
    auto* ctrl = GetCacheRingControl(shm.Base(), hdr);
    uint64_t ring_base = UINT64_MAX;
    char* ring_data = nullptr;
    uint64_t ring_cap = 0;

    if (ctrl) {
      ring_cap = ctrl->capacity;
      ring_base = AllocRingContiguous(ctrl, desc->total_data_size);
      if (ring_base != UINT64_MAX) {
        ring_data = GetCacheRingData(shm.Base(), hdr);
      }
    }

    bool use_ring = (ring_base != UINT64_MAX && ring_data != nullptr);
    std::vector<uintptr_t> dma_ptrs(count);
    std::vector<size_t> shm_offsets(count);

    if (use_ring) {
      char* ring_block = ring_data + static_cast<size_t>(ring_base % ring_cap);
      for (uint32_t i = 0; i < count; ++i) {
        auto& e = desc->entries[i];
        dma_ptrs[i] = reinterpret_cast<uintptr_t>(ring_block + e.data_offset);
        shm_offsets[i] = e.data_offset;
      }
      desc->ring_data_base = ring_base;
    } else {
      for (uint32_t i = 0; i < count; ++i) {
        auto& e = desc->entries[i];
        dma_ptrs[i] = reinterpret_cast<uintptr_t>(data_base + e.data_offset);
        shm_offsets[i] = e.data_offset;
      }
      desc->ring_data_base = 0;
    }

    auto results =
        tier.BatchReadIntoPtrStreaming(keys, dma_ptrs, sizes, &desc->items_done, &desc->bytes_done,
                                       &shm_offsets, dma_bufs, dma_count);
    desc->bytes_done.store(desc->total_data_size, std::memory_order_release);

    uint64_t hits = 0;
    for (uint32_t i = 0; i < count; ++i) {
      desc->entries[i].result = results[i] ? 1 : 0;
      if (results[i]) ++hits;
    }
    if (hits > 0) tier.RecordHit(hits);
    if (hits < count) tier.RecordMiss(count - hits);

    if (use_ring) {
      for (uint32_t i = 0; i < count; ++i) {
        if (results[i]) {
          UpdateRingCacheIndex(shm, tenant_id, cache_epoch, keys[i],
                               ring_base + desc->entries[i].data_offset, sizes[i]);
        }
      }
    } else {
      for (uint32_t i = 0; i < count; ++i) {
        if (results[i]) {
          WriteShmCache(shm, tenant_id, cache_epoch, keys[i],
                        data_base + desc->entries[i].data_offset, sizes[i]);
        }
      }
    }
  }

  (void)region_size;
  slot.result = static_cast<int32_t>(ResultCode::OK);
}

static constexpr int kDmaBufsPerChannel = 128;
static constexpr size_t kDmaBufSize = 2ULL * 1024 * 1024;

struct ChannelDmaPool {
  void** bufs = nullptr;
  int count = 0;
};

void AllocChannelDmaPools(ChannelDmaPool* pools, int max_channels) {
  static constexpr int kTotalDmaBudget = 1024;
  int per_channel =
      std::min(kDmaBufsPerChannel, std::max(16, kTotalDmaBudget / std::max(1, max_channels)));

  auto& env = umbp::SpdkEnv::Instance();
  for (int c = 0; c < max_channels; ++c) {
    pools[c].bufs = new void*[per_channel];
    int got = env.DmaPoolAllocBatch(pools[c].bufs, kDmaBufSize, per_channel);
    pools[c].count = got;
    if (got < per_channel) {
      for (int i = got; i < per_channel; ++i) pools[c].bufs[i] = nullptr;
    }
  }
  UMBP_LOG_INFO("spdk_proxy: allocated %d DMA bufs/channel × %d channels (%zuMB each)", per_channel,
                max_channels, kDmaBufSize / (1024 * 1024));
}

void FreeChannelDmaPools(ChannelDmaPool* pools, int max_channels) {
  auto& env = umbp::SpdkEnv::Instance();
  for (int c = 0; c < max_channels; ++c) {
    if (pools[c].bufs && pools[c].count > 0) {
      env.DmaPoolFreeBatch(pools[c].bufs, kDmaBufSize, pools[c].count);
    }
    delete[] pools[c].bufs;
    pools[c].bufs = nullptr;
    pools[c].count = 0;
  }
}

void PollLoop(ProxyShmRegion& shm, TenantRegistry& tenants, ChannelDmaPool* channel_dma,
              bool write_back, int idle_exit_timeout_ms) {
  auto* hdr = shm.Header();
  uint32_t max_channels = hdr->max_channels;

  auto batch_inflight = std::make_unique<std::atomic<bool>[]>(max_channels);
  std::vector<std::thread> batch_workers(max_channels);
  for (uint32_t c = 0; c < max_channels; ++c) {
    batch_inflight[c].store(false, std::memory_order_relaxed);
  }

  UMBP_LOG_INFO("spdk_proxy: entering poll loop (channels=%u tenants=%u)", hdr->max_channels,
                hdr->max_tenants);

  auto last_heartbeat = std::chrono::steady_clock::now();
  auto last_reap = last_heartbeat;
  std::optional<uint64_t> idle_since_ms;

  while (g_running.load(std::memory_order_relaxed)) {
    bool any_work = false;

    for (uint32_t c = 0; c < max_channels; ++c) {
      if (batch_inflight[c].load(std::memory_order_acquire)) continue;

      auto* ch = shm.Channel(c);
      if (ch->connected.load(std::memory_order_acquire) == 0) continue;

      uint32_t t = ch->tail.load(std::memory_order_relaxed);
      uint32_t h = ch->head.load(std::memory_order_acquire);
      if (t == h) continue;

      auto& slot = ch->slots[t % kRingSize];
      uint32_t st = slot.state.load(std::memory_order_acquire);
      if (st != static_cast<uint32_t>(SlotState::PENDING)) continue;

      any_work = true;
      auto rtype = static_cast<RequestType>(slot.type);
      void* data_region = shm.DataRegion(c);
      size_t region_size = hdr->data_region_per_channel;

      if (rtype == RequestType::ATTACH_SESSION) {
        uint64_t session_id = 0;
        ResultCode rc = tenants.AttachChannel(*ch, slot.request_aux, &session_id);
        slot.result = static_cast<int32_t>(rc);
        slot.result_aux = session_id;
        slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED), std::memory_order_release);
        ch->tail.store((t + 1) % kRingSize, std::memory_order_release);
        tenants.SyncTelemetry();
        continue;
      }

      if (rtype == RequestType::DETACH_SESSION) {
        ResultCode rc = tenants.DetachChannel(*ch);
        slot.result = static_cast<int32_t>(rc);
        slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED), std::memory_order_release);
        ch->tail.store((t + 1) % kRingSize, std::memory_order_release);
        tenants.SyncTelemetry();
        continue;
      }

      TenantRuntime* tenant = tenants.FindTenantByChannel(*ch);
      if (!tenant || ch->session_id.load(std::memory_order_acquire) == 0) {
        slot.result = static_cast<int32_t>(ResultCode::PERMISSION_DENIED);
        slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED), std::memory_order_release);
        ch->tail.store((t + 1) % kRingSize, std::memory_order_release);
        continue;
      }

      tenant->last_activity_ms = NowEpochMs();
      hdr->last_activity_ms.store(tenant->last_activity_ms, std::memory_order_relaxed);

      if (rtype == RequestType::BATCH_PUT || rtype == RequestType::BATCH_GET) {
        if (batch_workers[c].joinable()) batch_workers[c].join();
        batch_inflight[c].store(true, std::memory_order_release);
        tenant->inflight_batch_workers.fetch_add(1, std::memory_order_acq_rel);
        bool is_wb_write = write_back && rtype == RequestType::BATCH_PUT;
        if (is_wb_write) {
          tenant->inflight_wb_workers.fetch_add(1, std::memory_order_acq_rel);
        }
        uint64_t cache_epoch = tenant->cache_epoch.load(std::memory_order_acquire);
        void** dma_bufs = channel_dma[c].bufs;
        int dma_count = channel_dma[c].count;
        batch_workers[c] = std::thread([&shm, &slot, data_region, region_size, ch, t, tenant,
                                        &batch_inflight, dma_bufs, dma_count, is_wb_write, &tenants,
                                        write_back, cache_epoch, c]() {
          WbFlushTask wb_task;
          ProcessBatchRequest(*tenant->tier, shm, tenant->tenant_id, cache_epoch, slot, data_region,
                              region_size, write_back, dma_bufs, dma_count,
                              is_wb_write ? &wb_task : nullptr);
          slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED), std::memory_order_release);
          ch->tail.store((t + 1) % kRingSize, std::memory_order_release);
          if (is_wb_write) {
            if (!wb_task.keys.empty()) {
              tenant->wb_queue.Push(std::move(wb_task));
            }
            tenant->inflight_wb_workers.fetch_sub(1, std::memory_order_acq_rel);
          }
          tenant->inflight_batch_workers.fetch_sub(1, std::memory_order_acq_rel);
          batch_inflight[c].store(false, std::memory_order_release);
        });
      } else {
        ProcessSingleRequest(tenants, *tenant, shm, slot, data_region, region_size, write_back);
        slot.state.store(static_cast<uint32_t>(SlotState::COMPLETED), std::memory_order_release);
        ch->tail.store((t + 1) % kRingSize, std::memory_order_release);
        tenants.SyncTenantTelemetry(*tenant);
      }
    }

    auto now = std::chrono::steady_clock::now();
    uint64_t now_ms = NowEpochMs();

    auto since_hb = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat);
    if (since_hb.count() > 500) {
      hdr->proxy_heartbeat_ms.store(now_ms, std::memory_order_relaxed);
      last_heartbeat = now;
    }

    auto since_reap = std::chrono::duration_cast<std::chrono::seconds>(now - last_reap);
    if (since_reap.count() >= 5) {
#ifdef __linux__
      for (uint32_t c = 0; c < max_channels; ++c) {
        if (batch_inflight[c].load(std::memory_order_relaxed)) continue;
        auto* ch = shm.Channel(c);
        uint32_t owner_pid = ch->owner_pid.load(std::memory_order_relaxed);
        if (owner_pid > 0 && kill(static_cast<pid_t>(owner_pid), 0) != 0) {
          UMBP_LOG_WARN("spdk_proxy: client channel %u pid %u dead, reclaiming", c, owner_pid);
          tenants.ForceDetachChannel(*ch);
        }
      }
#endif
      tenants.ReapInactiveTenants();
      tenants.SyncTelemetry();
      last_reap = now;
    }

    uint32_t active_sessions = hdr->active_sessions.load(std::memory_order_acquire);
    if (active_sessions == 0) {
      if (!idle_since_ms.has_value()) idle_since_ms = now_ms;
      if (idle_exit_timeout_ms > 0 &&
          (now_ms - *idle_since_ms) >= static_cast<uint64_t>(idle_exit_timeout_ms)) {
        hdr->state.store(static_cast<uint32_t>(ProxyState::DRAINING), std::memory_order_release);
        if (hdr->active_sessions.load(std::memory_order_acquire) == 0 &&
            !tenants.HasInflightBatchWorkers()) {
          hdr->state.store(static_cast<uint32_t>(ProxyState::SHUTDOWN), std::memory_order_release);
          break;
        }
        hdr->state.store(static_cast<uint32_t>(ProxyState::READY), std::memory_order_release);
        idle_since_ms = now_ms;
      }
    } else {
      idle_since_ms.reset();
      if (hdr->state.load(std::memory_order_acquire) ==
          static_cast<uint32_t>(ProxyState::DRAINING)) {
        hdr->state.store(static_cast<uint32_t>(ProxyState::READY), std::memory_order_release);
      }
    }

    if (!any_work) {
#if defined(__x86_64__) || defined(_M_X64)
      for (int i = 0; i < 32; ++i) _mm_pause();
#endif
    }
  }

  for (auto& worker : batch_workers) {
    if (worker.joinable()) worker.join();
  }

  UMBP_LOG_INFO("spdk_proxy: exiting poll loop");
}

std::string getenv_str(const char* name, const char* def) {
  const char* v = std::getenv(name);
  return v ? v : def;
}

bool try_getenv_size(const char* name, size_t* value_out) {
  const char* v = std::getenv(name);
  if (!v || !v[0]) return false;

  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(v, &end, 10);
  if (errno != 0 || end == v || (end && *end != '\0')) {
    UMBP_LOG_WARN("spdk_proxy: ignoring invalid numeric env %s='%s'", name, v);
    return false;
  }
  *value_out = static_cast<size_t>(parsed);
  return true;
}

int getenv_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

size_t getenv_size(const char* name, size_t def) {
  size_t value = 0;
  return try_getenv_size(name, &value) ? value : def;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  std::string nvme_pci = getenv_str("UMBP_SPDK_NVME_PCI", "");
  std::string nvme_ctrl = getenv_str("UMBP_SPDK_NVME_CTRL", "NVMe0");
  std::string bdev_name = getenv_str("UMBP_SPDK_BDEV", "");
  std::string reactor_mask = getenv_str("UMBP_SPDK_REACTOR_MASK", "0xF");
  int mem_mb = getenv_int("UMBP_SPDK_MEM_MB", 4096);
  int io_workers = getenv_int("UMBP_SPDK_IO_WORKERS", 4);
  size_t ssd_cap = getenv_size("UMBP_SSD_CAPACITY", 0);

  std::string shm_name = getenv_str("UMBP_SPDK_PROXY_SHM", kDefaultShmName);
  int max_channels =
      getenv_int("UMBP_SPDK_PROXY_MAX_CHANNELS", getenv_int("UMBP_SPDK_PROXY_MAX_RANKS", 8));
  int max_tenants = getenv_int("UMBP_SPDK_PROXY_MAX_TENANTS", max_channels);
  bool write_back = getenv_int("UMBP_SPDK_PROXY_WRITE_BACK", 0) != 0;
  int idle_exit_timeout_ms = getenv_int("UMBP_SPDK_PROXY_IDLE_EXIT_TIMEOUT_MS", 30000);
  size_t default_tenant_quota_bytes = getenv_size("UMBP_SPDK_PROXY_DEFAULT_TENANT_QUOTA_BYTES", 0);
  bool allow_borrow = getenv_int("UMBP_SPDK_PROXY_ALLOW_BORROW", 0) != 0;
  size_t reserved_shared_bytes = getenv_size("UMBP_SPDK_PROXY_RESERVED_SHARED_BYTES", 0);
  uint64_t tenant_grace_ms =
      static_cast<uint64_t>(getenv_int("UMBP_SPDK_PROXY_TENANT_GRACE_MS", 30000));

  if (nvme_pci.empty() && bdev_name.empty()) {
    fprintf(stderr, "spdk_proxy: UMBP_SPDK_NVME_PCI or UMBP_SPDK_BDEV required\n");
    return 1;
  }

  g_shm_name = shm_name;
  std::atexit(atexit_cleanup);

  static constexpr size_t kMinRingMb = 1024;
  size_t ring_mb = 0;
  if (try_getenv_size("UMBP_SPDK_RING_MB", &ring_mb)) {
    // Prefer explicit ring size.
  } else if (try_getenv_size("UMBP_SPDK_PROXY_CACHE_MB", &ring_mb)) {
    // Compatibility fallback.
  } else {
    ring_mb = 32768;
  }

  if (ring_mb < kMinRingMb) {
    fprintf(stderr,
            "spdk_proxy: UMBP_SPDK_RING_MB=%zu is below minimum %zu. "
            "Set UMBP_SPDK_RING_MB >= %zu.\n",
            ring_mb, kMinRingMb, kMinRingMb);
    return 1;
  }

  size_t data_per_channel_mb = 32;
  if (!try_getenv_size("UMBP_SPDK_PROXY_DATA_PER_CHANNEL_MB", &data_per_channel_mb)) {
    (void)try_getenv_size("UMBP_SPDK_PROXY_DATA_MB", &data_per_channel_mb);
  }
  size_t data_per_channel = data_per_channel_mb * 1024 * 1024;

  ProxyShmRegion shm;
  int rc = shm.Create(shm_name, max_channels, max_tenants, data_per_channel,
                      /*try_hugepage=*/true, ring_mb);
  if (rc != 0) {
    fprintf(stderr, "spdk_proxy: failed to create SHM '%s' rc=%d\n", shm_name.c_str(), rc);
    return 1;
  }

  auto* hdr = shm.Header();
  hdr->proxy_pid.store(static_cast<uint32_t>(getpid()), std::memory_order_relaxed);
  hdr->proxy_heartbeat_ms.store(NowEpochMs(), std::memory_order_relaxed);

  umbp::SpdkEnvConfig env_cfg;
  env_cfg.bdev_name = bdev_name;
  env_cfg.reactor_mask = reactor_mask;
  env_cfg.mem_size_mb = mem_mb;
  env_cfg.nvme_pci_addr = nvme_pci;
  env_cfg.nvme_ctrl_name = nvme_ctrl;

  auto& env = umbp::SpdkEnv::Instance();
  rc = env.Init(env_cfg);
  if (rc != 0 || !env.IsInitialized()) {
    fprintf(stderr, "spdk_proxy: SpdkEnv init failed rc=%d\n", rc);
    hdr->state.store(static_cast<uint32_t>(ProxyState::ERROR), std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    shm.Detach();
    return 1;
  }

  uint32_t block_size = env.GetBlockSize();
  if (block_size == 0) block_size = 4096;
  size_t device_size = static_cast<size_t>(env.GetBdevSize());
  size_t service_capacity = (ssd_cap > 0) ? std::min(ssd_cap, device_size) : device_size;
  reserved_shared_bytes = AlignUp(reserved_shared_bytes, block_size);
  if (service_capacity <= reserved_shared_bytes) {
    fprintf(stderr, "spdk_proxy: reserved shared bytes exhaust service capacity\n");
    hdr->state.store(static_cast<uint32_t>(ProxyState::ERROR), std::memory_order_release);
    shm.Detach();
    return 1;
  }

  UMBPConfig tier_cfg;
  tier_cfg.ssd_backend = "spdk";
  tier_cfg.spdk_bdev_name = bdev_name;
  tier_cfg.spdk_reactor_mask = reactor_mask;
  tier_cfg.spdk_mem_size_mb = mem_mb;
  tier_cfg.spdk_nvme_pci_addr = nvme_pci;
  tier_cfg.spdk_nvme_ctrl_name = nvme_ctrl;
  tier_cfg.spdk_io_workers = io_workers;
  tier_cfg.ssd.capacity_bytes = service_capacity;

  auto shared_dma_pool =
      SpdkSsdTier::CreateSharedDmaPool(2ULL * 1024 * 1024, 128 * std::max(1, io_workers));

  hdr->bdev_size = service_capacity;
  hdr->block_size = block_size;
  hdr->capacity_used.store(0, std::memory_order_relaxed);
  hdr->capacity_total.store(service_capacity - reserved_shared_bytes, std::memory_order_relaxed);
  hdr->last_activity_ms.store(NowEpochMs(), std::memory_order_relaxed);

  auto* channel_dma = new ChannelDmaPool[max_channels];
  AllocChannelDmaPools(channel_dma, max_channels);

  TenantRegistry tenants(shm, tier_cfg, service_capacity, reserved_shared_bytes,
                         default_tenant_quota_bytes, block_size, allow_borrow, write_back,
                         tenant_grace_ms, std::move(shared_dma_pool));
  tenants.SyncTelemetry();
  hdr->state.store(static_cast<uint32_t>(ProxyState::READY), std::memory_order_release);

  UMBP_LOG_INFO(
      "spdk_proxy: READY — capacity=%zuMB usable=%zuMB ring=%zuMB "
      "channels=%d tenants=%d data_region=%zuMB/channel idle_exit=%dms "
      "borrow=%s write_back=%s",
      service_capacity / (1024 * 1024), tenants.usable_capacity_bytes() / (1024 * 1024), ring_mb,
      max_channels, max_tenants, data_per_channel_mb, idle_exit_timeout_ms,
      allow_borrow ? "ON" : "OFF", write_back ? "ON" : "OFF");
  fflush(stdout);

  PollLoop(shm, tenants, channel_dma, write_back, idle_exit_timeout_ms);

  tenants.SyncTelemetry();
  FreeChannelDmaPools(channel_dma, max_channels);
  delete[] channel_dma;

  hdr->state.store(static_cast<uint32_t>(ProxyState::SHUTDOWN), std::memory_order_release);

  UMBP_LOG_INFO("spdk_proxy: shutting down");
  shm.Detach();
  return 0;
}

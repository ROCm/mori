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

#include "umbp/storage/spdk/proxy/spdk_proxy_shm.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#ifdef __linux__
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace umbp {
namespace proxy {

ProxyShmRegion::~ProxyShmRegion() { Detach(); }

static std::string HugepagePath(const std::string& name) {
  if (!name.empty() && name[0] == '/') return "/dev/hugepages" + name;
  return "/dev/hugepages/" + name;
}

static uint64_t Fnv1a64(const std::string& value) {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : value) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

static size_t AlignToHugepage(size_t size) {
  return (size + kHugepageSize - 1) & ~(kHugepageSize - 1);
}

static bool ValidateHeaderLayoutImpl(const ProxyShmHeader* hdr, size_t mapped_size,
                                     std::string* error_message) {
  auto fail = [&](const char* message) {
    if (error_message) *error_message = message;
    return false;
  };

  if (!hdr) return fail("null header");
  if (hdr->magic != kProxyShmMagic) return fail("bad magic");
  if (hdr->version != kProxyVersion) return fail("protocol version mismatch");
  if (hdr->max_channels == 0 || hdr->max_channels > kMaxChannels) {
    return fail("invalid max_channels");
  }
  if (hdr->max_tenants == 0 || hdr->max_tenants > kMaxTenants) {
    return fail("invalid max_tenants");
  }
  if (hdr->data_region_per_channel == 0) return fail("invalid data_region_per_channel");
  if (hdr->total_shm_size == 0 || hdr->total_shm_size > mapped_size) {
    return fail("invalid total_shm_size");
  }

  size_t min_size =
      ComputeShmSize(hdr->max_channels, hdr->max_tenants, hdr->data_region_per_channel, 0);
  if (hdr->total_shm_size < min_size) return fail("shared memory too small");

  size_t header_size = sizeof(ProxyShmHeader);
  if (hdr->tenant_table_offset < header_size || hdr->tenant_table_offset >= hdr->total_shm_size) {
    return fail("invalid tenant_table_offset");
  }
  if (hdr->channels_offset < hdr->tenant_table_offset + sizeof(TenantInfo) * hdr->max_tenants ||
      hdr->channels_offset >= hdr->total_shm_size) {
    return fail("invalid channels_offset");
  }
  if (hdr->data_region_offset < hdr->channels_offset + sizeof(ClientChannel) * hdr->max_channels ||
      hdr->data_region_offset >= hdr->total_shm_size) {
    return fail("invalid data_region_offset");
  }
  size_t data_region_end =
      hdr->data_region_offset + hdr->data_region_per_channel * hdr->max_channels;
  if (data_region_end > hdr->total_shm_size) return fail("data region exceeds shm");

  if (hdr->cache_region_offset != 0) {
    if (hdr->cache_region_offset < data_region_end ||
        hdr->cache_region_offset >= hdr->total_shm_size) {
      return fail("invalid cache_region_offset");
    }
    if (hdr->cache_index_slots == 0 || hdr->cache_ring_capacity == 0) {
      return fail("cache region missing index/ring");
    }
    size_t cache_data_offset = CacheRingDataOffset(hdr);
    if (cache_data_offset < hdr->cache_region_offset || cache_data_offset > hdr->total_shm_size) {
      return fail("invalid cache data offset");
    }
    size_t cache_end = cache_data_offset + hdr->cache_ring_capacity;
    if (cache_end > hdr->total_shm_size) return fail("cache region exceeds shm");
  }

  return true;
}

int ProxyShmRegion::Create(const std::string& name, uint32_t max_channels, uint32_t max_tenants,
                           size_t data_per_channel, bool try_hugepage, size_t cache_budget_mb) {
#ifndef __linux__
  return -ENOTSUP;
#else
  if (max_channels == 0 || max_tenants == 0 || max_channels > kMaxChannels ||
      max_tenants > kMaxTenants) {
    return -EINVAL;
  }

  uint32_t cache_index_slots = 0;
  size_t cache_ring_cap = 0;
  size_t cache_total = 0;
  if (cache_budget_mb > 0) {
    size_t budget = cache_budget_mb * 1024ULL * 1024ULL;
    cache_index_slots = static_cast<uint32_t>(budget / 32768);
    if (cache_index_slots < 4096) cache_index_slots = 4096;
    size_t ctrl_and_idx = sizeof(CacheRingControl) +
                          static_cast<size_t>(cache_index_slots) * sizeof(CacheRingIndexEntry);
    size_t ctrl_and_idx_aligned = (ctrl_and_idx + 4095) & ~4095ULL;
    cache_ring_cap = (budget > ctrl_and_idx_aligned) ? budget - ctrl_and_idx_aligned : 0;
    cache_total = ctrl_and_idx_aligned + cache_ring_cap;
  }
  size_ = ComputeShmSize(max_channels, max_tenants, data_per_channel, cache_total);
  name_ = name;
  is_server_ = true;
  is_hugepage_ = false;

  if (try_hugepage) {
    hp_path_ = HugepagePath(name);
    size_t hp_size = AlignToHugepage(size_);

    unlink(hp_path_.c_str());

    fd_ = open(hp_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd_ >= 0) {
      if (ftruncate(fd_, static_cast<off_t>(hp_size)) == 0) {
        base_ = mmap(nullptr, hp_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, 0);
        if (base_ != MAP_FAILED) {
          is_hugepage_ = true;
          size_ = hp_size;
        } else {
          base_ = nullptr;
        }
      }
      if (!is_hugepage_) {
        close(fd_);
        unlink(hp_path_.c_str());
        fd_ = -1;
        hp_path_.clear();
      }
    } else {
      hp_path_.clear();
    }
  }

  if (!is_hugepage_) {
    shm_unlink(name_.c_str());
    fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
    if (fd_ < 0) return -errno;

    if (ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
      int err = errno;
      close(fd_);
      shm_unlink(name_.c_str());
      fd_ = -1;
      return -err;
    }

    base_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED) {
      int err = errno;
      close(fd_);
      shm_unlink(name_.c_str());
      base_ = nullptr;
      fd_ = -1;
      return -err;
    }
  }

  std::memset(base_, 0, size_);

  size_t header_sz = sizeof(ProxyShmHeader);
  size_t tenants_offset = (header_sz + 4095) & ~4095ULL;
  size_t tenants_size = sizeof(TenantInfo) * max_tenants;
  size_t channels_offset = (tenants_offset + tenants_size + 4095) & ~4095ULL;
  size_t channels_size = sizeof(ClientChannel) * max_channels;
  size_t data_offset = (channels_offset + channels_size + 4095) & ~4095ULL;

  auto* hdr = Header();
  hdr->magic = kProxyShmMagic;
  hdr->version = kProxyVersion;
  hdr->state.store(static_cast<uint32_t>(ProxyState::UNINIT), std::memory_order_relaxed);
  hdr->service_epoch = NowEpochMs();
  hdr->max_channels = max_channels;
  hdr->max_tenants = max_tenants;
  hdr->block_size = 0;
  hdr->hugepage = is_hugepage_ ? 1 : 0;
  hdr->bdev_size = 0;
  hdr->tenant_table_offset = tenants_offset;
  hdr->channels_offset = channels_offset;
  hdr->data_region_offset = data_offset;
  hdr->data_region_per_channel = data_per_channel;
  hdr->total_shm_size = size_;
  hdr->next_session_id.store(1, std::memory_order_relaxed);
  hdr->next_cache_epoch.store(1, std::memory_order_relaxed);
  hdr->capacity_used.store(0, std::memory_order_relaxed);
  hdr->capacity_total.store(0, std::memory_order_relaxed);

  hdr->proxy_pid.store(0, std::memory_order_relaxed);
  hdr->active_sessions.store(0, std::memory_order_relaxed);
  hdr->proxy_heartbeat_ms.store(0, std::memory_order_relaxed);
  hdr->last_activity_ms.store(hdr->service_epoch, std::memory_order_relaxed);

  for (uint32_t t = 0; t < max_tenants; ++t) {
    auto* tenant = Tenant(t);
    tenant->tenant_id = 0;
    tenant->tenant_slot = t;
    tenant->active_sessions.store(0, std::memory_order_relaxed);
    tenant->flags.store(0, std::memory_order_relaxed);
    tenant->used_bytes.store(0, std::memory_order_relaxed);
    tenant->quota_bytes.store(0, std::memory_order_relaxed);
    tenant->reserved_bytes.store(0, std::memory_order_relaxed);
    tenant->evicted_bytes.store(0, std::memory_order_relaxed);
    tenant->hit_count.store(0, std::memory_order_relaxed);
    tenant->miss_count.store(0, std::memory_order_relaxed);
    tenant->last_activity_ms.store(0, std::memory_order_relaxed);
    tenant->cache_epoch.store(0, std::memory_order_relaxed);
  }

  if (cache_index_slots > 0 && cache_ring_cap > 0) {
    size_t base_before_cache = ComputeShmSize(max_channels, max_tenants, data_per_channel, 0);
    hdr->cache_region_offset = (base_before_cache + 4095) & ~4095ULL;
    hdr->cache_ring_capacity = cache_ring_cap;
    hdr->cache_index_slots = cache_index_slots;

    auto* ctrl =
        reinterpret_cast<CacheRingControl*>(static_cast<char*>(base_) + hdr->cache_region_offset);
    ctrl->write_pos.store(0, std::memory_order_relaxed);
    ctrl->capacity = cache_ring_cap;
  } else {
    hdr->cache_region_offset = 0;
    hdr->cache_ring_capacity = 0;
    hdr->cache_index_slots = 0;
  }

  for (uint32_t c = 0; c < max_channels; ++c) {
    auto* ch = Channel(c);
    ch->head.store(0, std::memory_order_relaxed);
    ch->tail.store(0, std::memory_order_relaxed);
    ch->owner_pid.store(0, std::memory_order_relaxed);
    ch->connected.store(0, std::memory_order_relaxed);
    ch->channel_id = c;
    ch->client_pid = 0;
    ch->tenant_id = 0;
    ch->tenant_slot = 0;
    ch->session_id.store(0, std::memory_order_relaxed);
    ch->last_activity_ms.store(0, std::memory_order_relaxed);
    ch->requested_quota_bytes = 0;
    for (uint32_t s = 0; s < kRingSize; ++s) {
      ch->slots[s].state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_relaxed);
    }
  }

  return 0;
#endif
}

int ProxyShmRegion::Attach(const std::string& name) {
#ifndef __linux__
  return -ENOTSUP;
#else
  name_ = name;
  is_server_ = false;
  is_hugepage_ = false;

  hp_path_ = HugepagePath(name);
  fd_ = open(hp_path_.c_str(), O_RDWR);
  if (fd_ >= 0) {
    is_hugepage_ = true;
  } else {
    hp_path_.clear();
    fd_ = shm_open(name_.c_str(), O_RDWR, 0666);
    if (fd_ < 0) return -errno;
  }

  ProxyShmHeader tmp_hdr;
  if (pread(fd_, &tmp_hdr, sizeof(tmp_hdr), 0) != static_cast<ssize_t>(sizeof(tmp_hdr))) {
    int err = errno;
    close(fd_);
    fd_ = -1;
    return -err;
  }

  if (tmp_hdr.magic != kProxyShmMagic) {
    close(fd_);
    fd_ = -1;
    return -EINVAL;
  }

  size_ = tmp_hdr.total_shm_size;

  int mmap_flags = MAP_SHARED;
  if (is_hugepage_) mmap_flags |= MAP_POPULATE;

  base_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, mmap_flags, fd_, 0);
  if (base_ == MAP_FAILED) {
    int err = errno;
    close(fd_);
    base_ = nullptr;
    fd_ = -1;
    return -err;
  }

  return 0;
#endif
}

bool ProxyShmRegion::ValidateHeaderLayout(const ProxyShmHeader* hdr, size_t mapped_size,
                                          std::string* error_message) {
  return ValidateHeaderLayoutImpl(hdr, mapped_size, error_message);
}

void ProxyShmRegion::Detach() {
#ifdef __linux__
  if (base_ && size_ > 0) {
    munmap(base_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
  if (is_server_) {
    if (is_hugepage_ && !hp_path_.empty()) {
      unlink(hp_path_.c_str());
    } else if (!name_.empty()) {
      shm_unlink(name_.c_str());
    }
  }
#endif
  base_ = nullptr;
  size_ = 0;
  fd_ = -1;
}

// ---------------------------------------------------------------------------
// ProbeExisting — check if a live proxy is already running on this SHM name.
// ---------------------------------------------------------------------------
int ProxyShmRegion::ProbeExisting(const std::string& name) {
#ifndef __linux__
  return 0;
#else
  ProxyShmRegion probe;
  int rc = probe.Attach(name);
  if (rc != 0) return 0;

  auto* hdr = probe.Header();
  uint32_t pid = hdr->proxy_pid.load(std::memory_order_acquire);
  if (pid == 0) return 0;

  if (kill(static_cast<pid_t>(pid), 0) != 0) return 0;

  std::string error_message;
  if (!ValidateHeaderLayout(hdr, probe.Size(), &error_message)) return -2;

  uint64_t hb = hdr->proxy_heartbeat_ms.load(std::memory_order_acquire);
  if (hb != 0 && (NowEpochMs() - hb) >= kHeartbeatStaleMs) return -1;

  uint32_t st = hdr->state.load(std::memory_order_acquire);
  if (st == static_cast<uint32_t>(ProxyState::READY)) return 1;

  return -1;
#endif
}

// ---------------------------------------------------------------------------
// CleanupStale — forcefully remove SHM artifacts from a dead proxy.
// ---------------------------------------------------------------------------
void ProxyShmRegion::CleanupStale(const std::string& name) {
#ifdef __linux__
  std::string hp = HugepagePath(name);
  unlink(hp.c_str());
  shm_unlink(name.c_str());
#endif
}

std::string ProxyShmRegion::BootstrapLockPath(const std::string& name) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "/tmp/umbp_proxy_%016llx.lock",
                static_cast<unsigned long long>(Fnv1a64(name)));
  return std::string(buf);
}

}  // namespace proxy
}  // namespace umbp

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
// SpdkProxyTier: TierBackend that communicates with a multitenant spdk_proxy
// daemon via POSIX shared memory.

#include "umbp/local/tiers/spdk_proxy_tier.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "umbp/common/env_time.h"
#include "umbp/common/log.h"

#ifdef __linux__
#include <unistd.h>

#include <csignal>
#endif

namespace mori {
namespace umbp {

using namespace ::umbp::proxy;

namespace {
uint64_t HeartbeatStaleMs() {
  static const uint64_t v =
      static_cast<uint64_t>(GetEnvMilliseconds("UMBP_SPDK_PROXY_HEARTBEAT_STALE_MS",
                                               std::chrono::milliseconds(kDefaultHeartbeatStaleMs),
                                               /*min_allowed=*/1)
                                .count());
  return v;
}

std::chrono::milliseconds ProxyPollInterval() {
  static const auto v = GetEnvMilliseconds("UMBP_SPDK_PROXY_POLL_INTERVAL_MS",
                                           std::chrono::milliseconds(100), /*min_allowed=*/1);
  return v;
}
}  // namespace

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#else
#define CPU_PAUSE() ((void)0)
#endif

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------
SpdkProxyTier::SpdkProxyTier(const UMBPConfig& config) : TierBackend(StorageTier::LOCAL_SSD) {
  std::string shm_name = config.spdk_proxy_shm_name;
  if (shm_name.empty()) shm_name = kDefaultShmName;
  tenant_id_ = config.spdk_proxy_tenant_id;

  int rc = shm_.Attach(shm_name);
  if (rc != 0) {
    UMBP_LOG_ERROR("SpdkProxyTier: cannot attach to SHM '%s' rc=%d", shm_name.c_str(), rc);
    return;
  }

  auto* hdr = shm_.Header();
  if (hdr->version != kProxyVersion) {
    UMBP_LOG_ERROR("SpdkProxyTier: protocol mismatch on SHM '%s' (have=%u want=%u)",
                   shm_name.c_str(), hdr->version, kProxyVersion);
    shm_.Detach();
    return;
  }
  std::string layout_error;
  if (!ProxyShmRegion::ValidateHeaderLayout(hdr, shm_.Size(), &layout_error)) {
    UMBP_LOG_ERROR("SpdkProxyTier: invalid proxy SHM layout on '%s': %s", shm_name.c_str(),
                   layout_error.c_str());
    shm_.Detach();
    return;
  }

  uint32_t state = hdr->state.load(std::memory_order_acquire);
  if (state != static_cast<uint32_t>(ProxyState::READY)) {
    UMBP_LOG_ERROR("SpdkProxyTier: proxy not READY (state=%u)", state);
    shm_.Detach();
    return;
  }

  uint32_t my_pid = 0;
#ifdef __linux__
  my_pid = static_cast<uint32_t>(getpid());
#endif

  channel_id_ = hdr->max_channels;
  for (uint32_t c = 0; c < hdr->max_channels; ++c) {
    auto* ch = shm_.Channel(c);
    uint32_t expected = 0;
    if (ch->owner_pid.compare_exchange_strong(expected, my_pid, std::memory_order_acq_rel)) {
      channel_id_ = c;
      break;
    }
#ifdef __linux__
    if (expected > 0 && kill(static_cast<pid_t>(expected), 0) != 0) {
      if (ch->owner_pid.compare_exchange_strong(expected, my_pid, std::memory_order_acq_rel)) {
        ch->connected.store(0, std::memory_order_relaxed);
        ch->head.store(0, std::memory_order_relaxed);
        ch->tail.store(0, std::memory_order_relaxed);
        ch->session_id.store(0, std::memory_order_relaxed);
        channel_id_ = c;
        UMBP_LOG_WARN("SpdkProxyTier: reclaimed dead channel %u (pid %u)", c, expected);
        break;
      }
    }
#endif
  }

  if (channel_id_ >= hdr->max_channels) {
    UMBP_LOG_ERROR("SpdkProxyTier: all %u proxy channels are occupied", hdr->max_channels);
    shm_.Detach();
    return;
  }

  auto* ch = shm_.Channel(channel_id_);
  ch->head.store(0, std::memory_order_relaxed);
  ch->tail.store(0, std::memory_order_relaxed);
  ch->connected.store(1, std::memory_order_release);
  ch->channel_id = channel_id_;
  ch->client_pid = my_pid;
  ch->tenant_id = tenant_id_;
  ch->tenant_slot = 0;
  ch->requested_quota_bytes = config.spdk_proxy_tenant_quota_bytes;
  ch->last_activity_ms.store(NowEpochMs(), std::memory_order_relaxed);
  ch->session_id.store(0, std::memory_order_relaxed);
  for (uint32_t s = 0; s < kRingSize; ++s) {
    ch->slots[s].state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_relaxed);
  }

  connected_ = true;
  uint64_t attach_session_id = 0;
  auto attach_rc = SubmitAndWait(RequestType::ATTACH_SESSION, "", nullptr, 0, nullptr, 0,
                                 config.spdk_proxy_tenant_quota_bytes, nullptr, &attach_session_id);
  if (attach_rc != ResultCode::OK) {
    UMBP_LOG_ERROR("SpdkProxyTier: ATTACH_SESSION failed tenant=%u rc=%d", tenant_id_,
                   static_cast<int>(attach_rc));
    connected_ = false;
    ReleaseChannel();
    shm_.Detach();
    return;
  }

  session_id_ = attach_session_id;
  tenant_slot_ = ch->tenant_slot;
  if (session_id_ == 0) session_id_ = ch->session_id.load(std::memory_order_acquire);

  UMBP_LOG_INFO("SpdkProxyTier: attached channel=%u tenant=%u session=%lu shm='%s'", channel_id_,
                tenant_id_, static_cast<unsigned long>(session_id_), shm_name.c_str());
}

SpdkProxyTier::~SpdkProxyTier() {
  if (connected_ && shm_.IsValid()) {
    if (session_id_ != 0 && IsProxyAlive()) {
      (void)SubmitAndWait(RequestType::DETACH_SESSION, "", nullptr, 0, nullptr, 0);
    }
    ReleaseChannel();
    connected_ = false;
    session_id_ = 0;
  }
  shm_.Detach();
}

// ---------------------------------------------------------------------------
// WaitForProxy — static, called before constructing SpdkProxyTier
// ---------------------------------------------------------------------------
bool SpdkProxyTier::WaitForProxy(const std::string& shm_name, int timeout_ms) {
  auto start = std::chrono::steady_clock::now();

  while (true) {
    ProxyShmRegion probe;
    int rc = probe.Attach(shm_name);
    if (rc == 0) {
      auto* hdr = probe.Header();
      if (hdr->version != kProxyVersion) {
        UMBP_LOG_ERROR("SpdkProxyTier: protocol mismatch on SHM '%s' (have=%u want=%u)",
                       shm_name.c_str(), hdr->version, kProxyVersion);
        return false;
      }
      std::string layout_error;
      if (!ProxyShmRegion::ValidateHeaderLayout(hdr, probe.Size(), &layout_error)) {
        UMBP_LOG_ERROR("SpdkProxyTier: invalid proxy SHM layout on '%s': %s", shm_name.c_str(),
                       layout_error.c_str());
        return false;
      }

      uint32_t st = hdr->state.load(std::memory_order_acquire);
      if (st == static_cast<uint32_t>(ProxyState::READY)) {
#ifdef __linux__
        uint32_t pid = hdr->proxy_pid.load(std::memory_order_relaxed);
        if (pid > 0 && kill(static_cast<pid_t>(pid), 0) != 0) {
          probe.Detach();
        } else
#endif
        {
          uint64_t hb = hdr->proxy_heartbeat_ms.load(std::memory_order_acquire);
          if (hb == 0 || (NowEpochMs() - hb) < HeartbeatStaleMs()) {
            return true;
          }
        }
      } else if (st == static_cast<uint32_t>(ProxyState::ERROR)) {
        UMBP_LOG_ERROR("SpdkProxyTier: proxy reported ERROR state");
        return false;
      }
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeout_ms) {
      UMBP_LOG_ERROR("SpdkProxyTier: timed out waiting for proxy READY (%d ms)", timeout_ms);
      return false;
    }

    std::this_thread::sleep_for(ProxyPollInterval());
  }
}

// ---------------------------------------------------------------------------
// IsProxyAlive — check proxy heartbeat
// ---------------------------------------------------------------------------
bool SpdkProxyTier::IsProxyAlive() const {
  if (!shm_.IsValid()) return false;
  auto* hdr = shm_.Header();
  uint64_t hb = hdr->proxy_heartbeat_ms.load(std::memory_order_relaxed);
  if (hb == 0) return true;
  uint64_t now = NowEpochMs();
  return (now - hb) < HeartbeatStaleMs();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
uint32_t SpdkProxyTier::NextSeqId() const { return static_cast<uint32_t>(++seq_counter_); }

void SpdkProxyTier::ReleaseChannel() const {
  if (!shm_.IsValid()) return;
  auto* ch = shm_.Channel(channel_id_);
  if (!ch) return;

  ch->connected.store(0, std::memory_order_release);
  ch->client_pid = 0;
  ch->tenant_id = 0;
  ch->tenant_slot = 0;
  ch->requested_quota_bytes = 0;
  ch->session_id.store(0, std::memory_order_release);
  ch->last_activity_ms.store(0, std::memory_order_release);
  ch->head.store(0, std::memory_order_relaxed);
  ch->tail.store(0, std::memory_order_relaxed);
  for (uint32_t s = 0; s < kRingSize; ++s) {
    ch->slots[s].state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_relaxed);
  }
  ch->owner_pid.store(0, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Single-op submit/wait (with heartbeat timeout)
// ---------------------------------------------------------------------------
ResultCode SpdkProxyTier::SubmitAndWait(RequestType type, const std::string& key,
                                        const void* write_data, size_t write_size, void* read_buf,
                                        size_t read_buf_size, uint64_t request_aux,
                                        uint64_t* out_result_size, uint64_t* out_result_aux) const {
  if (!connected_) return ResultCode::ERROR;
  if (type != RequestType::ATTACH_SESSION && session_id_ == 0) {
    return ResultCode::PERMISSION_DENIED;
  }

  std::lock_guard<std::mutex> lk(submit_mu_);

  auto* ch = shm_.Channel(channel_id_);
  uint32_t h = ch->head.load(std::memory_order_relaxed);
  uint32_t t = ch->tail.load(std::memory_order_acquire);

  if (((h + 1) % kRingSize) == t) {
    UMBP_LOG_ERROR("SpdkProxyTier: ring full");
    return ResultCode::ERROR;
  }

  auto& slot = ch->slots[h % kRingSize];
  void* data_region = shm_.DataRegion(channel_id_);
  uint64_t ring_base_for_slot = 0;

  if (write_data && write_size > 0) {
    auto* hdr = shm_.Header();
    auto* ctrl = GetCacheRingControl(shm_.Base(), hdr);
    if (ctrl && type == RequestType::PUT) {
      uint64_t rbase = AllocRingContiguous(ctrl, write_size);
      if (rbase != UINT64_MAX) {
        char* ring_data = GetCacheRingData(shm_.Base(), hdr);
        uint64_t cap = ctrl->capacity;
        std::memcpy(ring_data + static_cast<size_t>(rbase % cap), write_data, write_size);
        ring_base_for_slot = rbase;
      }
    }
    if (ring_base_for_slot == 0) {
      size_t max_size = shm_.Header()->data_region_per_channel;
      if (write_size > max_size) {
        UMBP_LOG_ERROR("SpdkProxyTier: data too large %zu > %zu", write_size, max_size);
        return ResultCode::ERROR;
      }
      std::memcpy(data_region, write_data, write_size);
    }
  }

  slot.type = static_cast<uint32_t>(type);
  slot.seq_id = NextSeqId();
  slot.key_len = std::min(static_cast<uint32_t>(key.size()), kMaxKeyLen - 1);
  if (slot.key_len > 0) {
    std::memcpy(slot.key, key.data(), slot.key_len);
  }
  slot.key[slot.key_len] = '\0';
  slot.data_offset = 0;
  slot.data_size = write_size;
  slot.request_aux = request_aux;
  slot.batch_count = 0;
  slot.flags = 0;
  slot.ring_data_base = ring_base_for_slot;
  slot.result = static_cast<int32_t>(ResultCode::ERROR);
  slot.result_size = 0;
  slot.result_aux = 0;

  ch->last_activity_ms.store(NowEpochMs(), std::memory_order_relaxed);
  slot.state.store(static_cast<uint32_t>(SlotState::PENDING), std::memory_order_release);
  ch->head.store((h + 1) % kRingSize, std::memory_order_release);

  int spin = 0;
  while (true) {
    uint32_t st = slot.state.load(std::memory_order_acquire);
    if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
    CPU_PAUSE();
    if (++spin % 8192 == 0 && !IsProxyAlive()) {
      UMBP_LOG_ERROR("SpdkProxyTier: proxy heartbeat stale, aborting");
      slot.state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_release);
      return ResultCode::ERROR;
    }
  }

  ResultCode rc = static_cast<ResultCode>(slot.result);
  if (out_result_size) *out_result_size = slot.result_size;
  if (out_result_aux) *out_result_aux = slot.result_aux;

  if (read_buf && slot.result_size > 0 && rc == ResultCode::OK) {
    size_t copy_sz = std::min(static_cast<size_t>(slot.result_size), read_buf_size);
    if (slot.ring_data_base != 0) {
      auto* hdr = shm_.Header();
      auto* ctrl = GetCacheRingControl(shm_.Base(), hdr);
      if (ctrl) {
        const char* ring_data = GetCacheRingData(shm_.Base(), hdr);
        uint64_t cap = ctrl->capacity;
        std::memcpy(read_buf, ring_data + static_cast<size_t>(slot.ring_data_base % cap), copy_sz);
      } else {
        std::memcpy(read_buf, data_region, copy_sz);
      }
    } else {
      std::memcpy(read_buf, data_region, copy_sz);
    }
  }

  slot.state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_release);
  return rc;
}

// ---------------------------------------------------------------------------
// Batch submit (with streaming, heartbeat timeout, and correct indices)
// ---------------------------------------------------------------------------
std::vector<bool> SpdkProxyTier::SubmitBatch(RequestType type, const std::vector<std::string>& keys,
                                             const std::vector<const void*>& data_ptrs,
                                             const std::vector<uintptr_t>& dst_ptrs,
                                             const std::vector<size_t>& sizes) const {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);
  if (!connected_ || session_id_ == 0 || count == 0) return results;

  std::lock_guard<std::mutex> lk(submit_mu_);

  auto* hdr = shm_.Header();
  void* data_region = shm_.DataRegion(channel_id_);
  const size_t region_size = hdr->data_region_per_channel;

  int base = 0;
  while (base < count) {
    size_t desc_overhead = sizeof(BatchDescriptor);
    size_t data_start = 0;
    int sub_count = 0;
    size_t total_data = 0;

    for (int i = base; i < count; ++i) {
      size_t entry_overhead = sizeof(BatchEntry);
      size_t new_desc = desc_overhead + (sub_count + 1) * entry_overhead;
      size_t new_data_start = (new_desc + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
      size_t aligned_data = (sizes[i] + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
      size_t new_total = new_data_start + total_data + aligned_data;
      if (new_total > region_size && sub_count > 0) break;
      sub_count++;
      total_data += aligned_data;
      data_start = new_data_start;
    }

    if (sub_count == 0) {
      results[base] = false;
      base++;
      continue;
    }

    auto* desc = static_cast<BatchDescriptor*>(data_region);
    desc->count = sub_count;
    desc->total_data_size = total_data;
    desc->ring_data_base = 0;
    desc->items_ready.store(0, std::memory_order_relaxed);
    desc->items_done.store(0, std::memory_order_relaxed);
    desc->bytes_ready.store(0, std::memory_order_relaxed);
    desc->bytes_done.store(0, std::memory_order_relaxed);

    size_t data_cursor = 0;
    char* data_base = static_cast<char*>(data_region) + data_start;

    for (int i = 0; i < sub_count; ++i) {
      int gi = base + i;
      auto& entry = desc->entries[i];
      entry.key_len =
          std::min(static_cast<uint16_t>(keys[gi].size()), static_cast<uint16_t>(kMaxKeyLen - 1));
      std::memcpy(entry.key, keys[gi].data(), entry.key_len);
      entry.key[entry.key_len] = '\0';
      entry.data_offset = data_cursor;
      entry.data_size = sizes[gi];
      entry.result = 0;
      size_t aligned = (sizes[gi] + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
      data_cursor += aligned;
    }

    char* write_target = data_base;
    if (type == RequestType::BATCH_PUT) {
      auto* ctrl = GetCacheRingControl(shm_.Base(), hdr);
      if (ctrl) {
        uint64_t rbase = AllocRingContiguous(ctrl, total_data);
        // `ring_data_base == 0` is reserved as the protocol sentinel for
        // "payload lives in the per-channel data region".  Avoid using the
        // ring for that first allocation so client and proxy interpret the
        // descriptor consistently.
        if (rbase != UINT64_MAX && rbase != 0) {
          char* ring_data = GetCacheRingData(shm_.Base(), hdr);
          uint64_t cap = ctrl->capacity;
          write_target = ring_data + static_cast<size_t>(rbase % cap);
          desc->ring_data_base = rbase;
        }
      }
    }

    auto* ch = shm_.Channel(channel_id_);
    uint32_t h = ch->head.load(std::memory_order_relaxed);

    int ring_spin = 0;
    while (((h + 1) % kRingSize) == ch->tail.load(std::memory_order_acquire)) {
      CPU_PAUSE();
      if (++ring_spin % 8192 == 0 && !IsProxyAlive()) {
        UMBP_LOG_ERROR("SpdkProxyTier: proxy dead while waiting for ring slot");
        return results;
      }
    }

    auto& slot = ch->slots[h % kRingSize];
    slot.type = static_cast<uint32_t>(type);
    slot.seq_id = NextSeqId();
    slot.key_len = 0;
    slot.data_offset = 0;
    slot.data_size = data_start + data_cursor;
    slot.request_aux = 0;
    slot.batch_count = sub_count;
    slot.flags = 0;
    slot.ring_data_base = 0;
    slot.result = static_cast<int32_t>(ResultCode::ERROR);
    slot.result_size = 0;
    slot.result_aux = 0;

    ch->last_activity_ms.store(NowEpochMs(), std::memory_order_relaxed);
    slot.state.store(static_cast<uint32_t>(SlotState::PENDING), std::memory_order_release);
    ch->head.store((h + 1) % kRingSize, std::memory_order_release);

    if (type == RequestType::BATCH_PUT) {
      constexpr size_t kCopyChunk = 2ULL * 1024 * 1024;
      for (int i = 0; i < sub_count; ++i) {
        int gi = base + i;
        if (!data_ptrs.empty() && data_ptrs[gi] != nullptr) {
          const char* src = static_cast<const char*>(data_ptrs[gi]);
          char* dst = write_target + desc->entries[i].data_offset;
          size_t item_sz = sizes[gi];
          size_t copied = 0;
          while (copied < item_sz) {
            size_t chunk = std::min(kCopyChunk, item_sz - copied);
            std::memcpy(dst + copied, src + copied, chunk);
            copied += chunk;
            desc->bytes_ready.store(desc->entries[i].data_offset + copied,
                                    std::memory_order_release);
          }
        }
        desc->items_ready.store(static_cast<uint32_t>(i + 1), std::memory_order_release);
      }
      desc->bytes_ready.store(desc->total_data_size, std::memory_order_release);

      int spin = 0;
      while (true) {
        uint32_t st = slot.state.load(std::memory_order_acquire);
        if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
        CPU_PAUSE();
        if (++spin % 8192 == 0 && !IsProxyAlive()) {
          UMBP_LOG_ERROR("SpdkProxyTier: proxy dead during batch write");
          slot.state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_release);
          return results;
        }
      }

      for (int i = 0; i < sub_count; ++i) results[base + i] = (desc->entries[i].result != 0);

    } else {
      constexpr size_t kReadChunk = 2ULL * 1024 * 1024;
      char* read_base = nullptr;
      int current_item = 0;
      size_t item_copied = 0;
      int spin = 0;

      while (current_item < sub_count) {
        int gi = base + current_item;
        auto& entry = desc->entries[current_item];
        size_t item_sz = sizes[gi];

        if (!dst_ptrs.empty() && item_sz > 0) {
          char* dst = reinterpret_cast<char*>(dst_ptrs[gi]);

          while (item_copied < item_sz) {
            size_t want = std::min(kReadChunk, item_sz - item_copied);
            uint64_t need = entry.data_offset + item_copied + want;
            uint64_t bd = desc->bytes_done.load(std::memory_order_acquire);

            if (bd >= need) {
              if (!read_base) {
                if (desc->ring_data_base != 0) {
                  auto* ctrl = GetCacheRingControl(shm_.Base(), hdr);
                  if (ctrl) {
                    char* rd = GetCacheRingData(shm_.Base(), hdr);
                    uint64_t cap = ctrl->capacity;
                    read_base = rd + static_cast<size_t>(desc->ring_data_base % cap);
                  } else {
                    read_base = data_base;
                  }
                } else {
                  read_base = data_base;
                }
              }
              char* src = read_base + entry.data_offset;
              std::memcpy(dst + item_copied, src + item_copied, want);
              item_copied += want;
              spin = 0;
            } else {
              CPU_PAUSE();
              if (++spin % 8192 == 0 && !IsProxyAlive()) {
                UMBP_LOG_ERROR("SpdkProxyTier: proxy dead during batch read");
                slot.state.store(static_cast<uint32_t>(SlotState::EMPTY),
                                 std::memory_order_release);
                return results;
              }
            }
          }
        }

        ++current_item;
        item_copied = 0;
      }

      spin = 0;
      while (true) {
        uint32_t st = slot.state.load(std::memory_order_acquire);
        if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
        CPU_PAUSE();
        if (++spin % 8192 == 0 && !IsProxyAlive()) {
          UMBP_LOG_ERROR("SpdkProxyTier: proxy dead waiting for read completion");
          slot.state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_release);
          return results;
        }
      }

      for (int i = 0; i < sub_count; ++i) {
        int gi = base + i;
        results[gi] = (desc->entries[i].result != 0);
      }
    }

    slot.state.store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_release);
    base += sub_count;
  }

  return results;
}

// ---------------------------------------------------------------------------
// TierBackend interface
// ---------------------------------------------------------------------------
bool SpdkProxyTier::Write(const std::string& key, const void* data, size_t size) {
  auto rc = SubmitAndWait(RequestType::PUT, key, data, size, nullptr, 0);
  return rc == ResultCode::OK;
}

bool SpdkProxyTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  if (TryShmCacheReadOne(key, dst_ptr, size)) return true;

  uint64_t actual_size = 0;
  auto rc = SubmitAndWait(RequestType::GET, key, nullptr, 0, reinterpret_cast<void*>(dst_ptr), size,
                          0, &actual_size);
  return rc == ResultCode::OK;
}

bool SpdkProxyTier::Exists(const std::string& key) const {
  auto rc = SubmitAndWait(RequestType::EXISTS, key, nullptr, 0, nullptr, 0);
  return rc == ResultCode::OK;
}

bool SpdkProxyTier::Evict(const std::string& key) {
  auto rc = SubmitAndWait(RequestType::REMOVE, key, nullptr, 0, nullptr, 0);
  return rc == ResultCode::OK;
}

std::pair<size_t, size_t> SpdkProxyTier::Capacity() const {
  uint64_t result_size = 0, result_aux = 0;
  auto rc = SubmitAndWait(RequestType::CAPACITY_TENANT, "", nullptr, 0, nullptr, 0, 0, &result_size,
                          &result_aux);
  if (rc != ResultCode::OK) return {0, 0};
  return {static_cast<size_t>(result_size), static_cast<size_t>(result_aux)};
}

void SpdkProxyTier::Clear() {
  (void)SubmitAndWait(RequestType::CLEAR_TENANT, "", nullptr, 0, nullptr, 0);
}

bool SpdkProxyTier::Flush() {
  auto rc = SubmitAndWait(RequestType::FLUSH_TENANT, "", nullptr, 0, nullptr, 0);
  return rc == ResultCode::OK;
}

// ---------------------------------------------------------------------------
// Per-item ring cache read — validate tenant, key, and tenant cache epoch.
// ---------------------------------------------------------------------------
bool SpdkProxyTier::TryShmCacheReadOne(const std::string& key, uintptr_t dst, size_t size) const {
  if (!connected_ || cold_read_) return false;

  auto* hdr = shm_.Header();
  if (hdr->cache_index_slots == 0 || tenant_slot_ >= hdr->max_tenants) return false;

  auto* ctrl = GetCacheRingControl(shm_.Base(), hdr);
  if (!ctrl) return false;

  const auto* tenant = shm_.Tenant(tenant_slot_);
  if (!tenant) return false;
  if ((tenant->flags.load(std::memory_order_acquire) & kTenantFlagActive) == 0) return false;
  if (tenant->tenant_id != tenant_id_) return false;
  uint64_t cache_epoch = tenant->cache_epoch.load(std::memory_order_acquire);
  if (cache_epoch == 0) return false;

  auto* index = GetCacheIndex(shm_.Base(), hdr);
  uint32_t idx = CacheIndexHash(key.data(), static_cast<uint32_t>(key.size()), tenant_id_,
                                hdr->cache_index_slots);
  auto& entry = index[idx];

  uint64_t g1 = entry.gen.load(std::memory_order_acquire);
  if (g1 == 0 || (g1 & 1)) return false;

  if (entry.tenant_id != tenant_id_ || entry.key_len != key.size() ||
      entry.tenant_cache_epoch != cache_epoch ||
      std::memcmp(entry.key, key.data(), key.size()) != 0 || entry.data_size != size) {
    return false;
  }

  uint64_t ring_off = entry.ring_offset;
  uint64_t wp = ctrl->write_pos.load(std::memory_order_acquire);
  if (wp - ring_off > ctrl->capacity) return false;

  const char* ring_data = GetCacheRingData(shm_.Base(), hdr);
  uint64_t cap = ctrl->capacity;
  size_t off = static_cast<size_t>(ring_off % cap);
  auto* out = reinterpret_cast<char*>(dst);

  if (off + size <= cap) {
    std::memcpy(out, ring_data + off, size);
  } else {
    size_t first = static_cast<size_t>(cap - off);
    std::memcpy(out, ring_data + off, first);
    std::memcpy(out + first, ring_data, size - first);
  }

  uint64_t g2 = entry.gen.load(std::memory_order_acquire);
  if (g1 != g2) return false;

  uint64_t wp2 = ctrl->write_pos.load(std::memory_order_acquire);
  if (wp2 - ring_off > ctrl->capacity) return false;

  return true;
}

// ---------------------------------------------------------------------------
// Capabilities — advertise batch support so CopyToSSDBatch uses WriteBatch.
// ---------------------------------------------------------------------------
TierCapabilities SpdkProxyTier::Capabilities() const {
  return {/*.zero_copy_read=*/false,
          /*.batch_write=*/true,
          /*.batch_read=*/true};
}

bool SpdkProxyTier::WriteBatch(const std::vector<std::string>& keys,
                               const std::vector<const void*>& data_ptrs,
                               const std::vector<size_t>& sizes) {
  auto results = BatchWrite(keys, data_ptrs, sizes);
  for (bool ok : results) {
    if (!ok) return false;
  }
  return true;
}

std::vector<bool> SpdkProxyTier::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                                  const std::vector<uintptr_t>& dst_ptrs,
                                                  const std::vector<size_t>& sizes) {
  return BatchReadIntoPtr(keys, dst_ptrs, sizes);
}

std::vector<bool> SpdkProxyTier::BatchWrite(const std::vector<std::string>& keys,
                                            const std::vector<const void*>& data_ptrs,
                                            const std::vector<size_t>& sizes) {
  return SubmitBatch(RequestType::BATCH_PUT, keys, data_ptrs, {}, sizes);
}

std::vector<bool> SpdkProxyTier::BatchReadIntoPtr(const std::vector<std::string>& keys,
                                                  const std::vector<uintptr_t>& dst_ptrs,
                                                  const std::vector<size_t>& sizes) {
  const int count = static_cast<int>(keys.size());
  std::vector<bool> results(count, false);

  std::vector<int> misses;
  misses.reserve(count);
  for (int i = 0; i < count; ++i) {
    if (TryShmCacheReadOne(keys[i], dst_ptrs[i], sizes[i])) {
      results[i] = true;
    } else {
      misses.push_back(i);
    }
  }
  if (misses.empty()) return results;

  std::vector<std::string> m_keys(misses.size());
  std::vector<uintptr_t> m_dst(misses.size());
  std::vector<size_t> m_sizes(misses.size());
  for (size_t j = 0; j < misses.size(); ++j) {
    int i = misses[j];
    m_keys[j] = keys[i];
    m_dst[j] = dst_ptrs[i];
    m_sizes[j] = sizes[i];
  }
  auto m_results = SubmitBatch(RequestType::BATCH_GET, m_keys, {}, m_dst, m_sizes);
  for (size_t j = 0; j < misses.size(); ++j) {
    int i = misses[j];
    results[i] = m_results[j];
  }
  return results;
}

std::string SpdkProxyTier::GetLRUKey() const {
  if (!connected_) return "";
  uint64_t result_size = 0;
  char buf[kMaxKeyLen] = {};
  auto rc =
      SubmitAndWait(RequestType::GET_LRU_KEY, "", nullptr, 0, buf, sizeof(buf), 0, &result_size);
  if (rc != ResultCode::OK || result_size == 0) return "";
  return std::string(buf, std::min(static_cast<size_t>(result_size), sizeof(buf) - 1));
}

std::vector<std::string> SpdkProxyTier::GetLRUCandidates(size_t max_candidates) const {
  (void)max_candidates;
  std::vector<std::string> result;
  std::string k = GetLRUKey();
  if (!k.empty()) result.push_back(std::move(k));
  return result;
}

}  // namespace umbp
}  // namespace mori

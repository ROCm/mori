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
#include "umbp/distributed/peer/peer_ssd_manager.h"

#include <cstring>

#include "mori/utils/mori_log.hpp"
#include "umbp/local/tiers/ssd_tier.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

PeerSsdManager::PeerSsdManager(const PeerSsdConfig& cfg) {
  if (!cfg.enabled) {
    MORI_UMBP_INFO("[PeerSsdManager] constructed disabled (no SSD backend)");
    return;
  }
  // v1 builds the POSIX/io_uring SSDTier directly from UMBPSsdConfig.  SPDK
  // backend selection is not wired into the distributed peer path yet (see
  // PeerSsdConfig TODO); when it is, branch on the backend here.
  const auto& ssd = cfg.ssd;
  backend_ = std::make_unique<SSDTier>(ssd.storage_dir, ssd.capacity_bytes, ssd);
  MORI_UMBP_INFO("[PeerSsdManager] SSDTier ready dir={} capacity={}B", ssd.storage_dir,
                 ssd.capacity_bytes);
}

PeerSsdManager::~PeerSsdManager() = default;

std::pair<size_t, size_t> PeerSsdManager::Capacity() const {
  if (!backend_) return {0, 0};
  return backend_->Capacity();
}

bool PeerSsdManager::Exists(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return owned_.find(key) != owned_.end();
}

bool PeerSsdManager::Write(const std::string& key,
                           const std::vector<std::pair<const void*, size_t>>& segments,
                           size_t total_size) {
  if (!backend_) return false;

  // Assemble (possibly non-contiguous) DRAM source segments into one
  // contiguous buffer before handing it to the backend.  v1 accepts the extra
  // memcpy; a writev/batch path can replace this later.  When the source is
  // already a single contiguous segment of the right size we write it directly.
  const void* data = nullptr;
  std::vector<char> scratch;
  if (segments.size() == 1 && segments[0].second == total_size) {
    data = segments[0].first;
  } else {
    scratch.resize(total_size);
    size_t off = 0;
    for (const auto& [ptr, len] : segments) {
      if (off + len > total_size) {
        MORI_UMBP_ERROR("[PeerSsdManager] Write key={} segments exceed total_size={}", key,
                        total_size);
        return false;
      }
      if (len > 0) std::memcpy(scratch.data() + off, ptr, len);
      off += len;
    }
    if (off != total_size) {
      MORI_UMBP_ERROR("[PeerSsdManager] Write key={} assembled {} != total_size={}", key, off,
                      total_size);
      return false;
    }
    data = scratch.data();
  }

  // Backend IO outside our mutex_ — SSDTier is internally synchronized.
  if (!backend_->Write(key, data, total_size)) {
    MORI_UMBP_WARN("[PeerSsdManager] backend Write failed key={} size={}", key, total_size);
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  owned_[key] = total_size;
  pending_events_.push_back(KvEvent{KvEvent::Kind::ADD, key, TierType::SSD, total_size});
  return true;
}

bool PeerSsdManager::Evict(const std::string& key) {
  if (!backend_) return false;
  backend_->Evict(key);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = owned_.find(key);
  if (it == owned_.end()) return false;
  owned_.erase(it);
  pending_events_.push_back(KvEvent{KvEvent::Kind::REMOVE, key, TierType::SSD, 0});
  return true;
}

std::vector<std::string> PeerSsdManager::SelectVictims(size_t /*bytes_to_free*/) {
  // TODO(Phase 4): local watermark + LRU victim selection.
  return {};
}

SsdReadOutcome PeerSsdManager::PrepareRead(const std::string& /*key*/, void* /*staging_ptr*/,
                                           size_t /*staging_cap*/) {
  // TODO(Phase 3): key-based read into staging (local vs remote SSD get).
  return SsdReadOutcome{SsdReadStatus::kNotFound, 0};
}

std::vector<KvEvent> PeerSsdManager::DrainPendingEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> drained;
  drained.swap(pending_events_);
  return drained;
}

std::vector<KvEvent> PeerSsdManager::SnapshotOwnedKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> out;
  out.reserve(owned_.size());
  for (const auto& [key, size] : owned_) {
    out.push_back(KvEvent{KvEvent::Kind::ADD, key, TierType::SSD, size});
  }
  return out;
}

}  // namespace mori::umbp

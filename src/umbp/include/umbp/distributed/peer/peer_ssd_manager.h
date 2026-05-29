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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/config.h"  // PeerSsdConfig
#include "umbp/distributed/peer/owned_location_source.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class TierBackend;  // umbp/local/tiers/tier_backend.h — kept out of this header.

enum class SsdReadStatus { kOk, kNotFound, kSizeTooLarge, kError };
struct SsdReadOutcome {
  SsdReadStatus status = SsdReadStatus::kError;
  size_t size = 0;
};

// Peer-side owner of the local SSD tier in the master-as-advisor design.
// Single responsibility: manage one SSD TierBackend + the key->SSD-location
// map + capacity + the owned-location event outbox (and, from Phase 3, the
// read-prepare path).  It deliberately reuses ONLY the low-level TierBackend
// (SSDTier); it must NOT pull in LocalStorageManager / LocalBlockIndex (which
// carry their own DRAM tier + demote/promote) — peer DRAM is owned by
// PeerDramAllocator and two DRAM concepts would scramble ownership.
//
// Phase 1 builds the ownership model + reporting channel but does NOT move
// data: nothing writes to SSD yet (copy-on-commit = Phase 2), so Write() is
// only driven by unit tests; reads are a stub (Phase 3).
class PeerSsdManager : public OwnedLocationSource {
 public:
  explicit PeerSsdManager(const PeerSsdConfig& cfg);
  ~PeerSsdManager() override;

  PeerSsdManager(const PeerSsdManager&) = delete;
  PeerSsdManager& operator=(const PeerSsdManager&) = delete;

  // {used_bytes, total_bytes}.  Reported via heartbeat as TierType::SSD.
  std::pair<size_t, size_t> Capacity() const;

  bool Exists(const std::string& key) const;

  // Write the key's bytes (assembled from possibly non-contiguous DRAM source
  // segments) to the SSD backend.  On success records the SSD location and
  // queues an ADD SSD event; on failure records nothing and queues nothing
  // (best-effort clean).  In Phase 1 this is exercised only by unit tests; the
  // copy worker that drives it lands in Phase 2.
  bool Write(const std::string& key, const std::vector<std::pair<const void*, size_t>>& segments,
             size_t total_size);

  // Phase 4: local eviction.  Removes the key from the backend + map and
  // queues a REMOVE SSD event.  Implemented here but not yet wired to any
  // caller in Phase 1.
  bool Evict(const std::string& key);

  // Phase 4: local LRU victim selection.  Minimal stub in Phase 1.
  std::vector<std::string> SelectVictims(size_t bytes_to_free);

  // Distributed Clear: drop the logical owned-location map + undrained
  // events so a post-Clear full-sync snapshot is empty and no stale ADD
  // SSD is shipped.  Physical backend bytes are intentionally left in
  // place (best-effort cache; reclaimed by Phase 4 local eviction).
  // Callers MUST quiesce the SSD copy pipeline first so no in-flight copy
  // re-populates owned_ right after this returns.
  void ClearLocal();

  // Phase 3: read the key's bytes into a staging slot.  Minimal stub in
  // Phase 1 (always kNotFound); the real key-based read path lands in Phase 3.
  SsdReadOutcome PrepareRead(const std::string& key, void* staging_ptr, size_t staging_cap);

  // OwnedLocationSource — all events carry TierType::SSD.
  std::vector<KvEvent> DrainPendingEvents() override;
  std::vector<KvEvent> SnapshotOwnedKeys() const override;

 private:
  mutable std::mutex mutex_;
  std::unique_ptr<TierBackend> backend_;             // null when cfg.enabled == false
  std::unordered_map<std::string, uint64_t> owned_;  // key -> size (SSD location)
  std::vector<KvEvent> pending_events_;
};

}  // namespace mori::umbp

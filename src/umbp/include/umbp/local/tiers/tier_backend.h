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
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "umbp/local/storage_tier.h"

namespace mori::umbp {

struct TierCapabilities {
  bool zero_copy_read = false;
  bool batch_write = false;
  bool batch_read = false;
  // Whether ReadBatchRangesIntoPtr is really implemented.  The base class
  // provides an all-false default, which is indistinguishable from "every key
  // missed"; callers route on this flag instead so an unsupported tier falls
  // through to another one rather than reporting a phantom miss.
  bool ranged_read = false;
  // Whether EvictionOrder() is the order to evict in.  Set by tiers that free
  // disk in units bigger than one key, where evicting keys in plain recency
  // order can leave every unit partly live and free nothing.
  bool orders_eviction = false;
};

// A key offered for eviction, with the bytes evicting it is credited with
// freeing.
struct EvictionHint {
  std::string key;
  uint64_t reclaim_bytes = 0;
};

// Disk use of a tier that deletes files as space frees.  live_bytes and files
// are current; the rest count since start.
struct TierSpaceStats {
  uint64_t live_bytes = 0;     // records still mapped to a key
  uint64_t files = 0;          // segment files on disk
  uint64_t files_deleted = 0;  // emptied segment files deleted
  uint64_t bytes_deleted = 0;  // disk those files held
  uint64_t no_room = 0;        // new segments refused: the filesystem was full
};

// Physical location of a value inside a segment file, for a zero-copy reader
// (GdsEngine) that DMAs the bytes itself with hipFileRead instead of asking the
// tier to copy them.  Only the segment-file SSD tiers can answer; every other
// tier reports "no stable file location".
struct RecordLocation {
  int fd = -1;                 // O_DIRECT segment descriptor
  uint64_t value_offset = 0;   // 4 KiB-aligned start of the value in the file
  uint64_t readable_size = 0;  // 4 KiB-aligned padded length safe to read
  uint64_t value_size = 0;     // logical value length (<= readable_size)
  bool direct_io = false;      // fd is O_DIRECT — a hipFile fastpath precondition
};

// Room for one value that a file engine writes in place at `location`.
struct RecordReservation {
  RecordLocation location;
  uint64_t token = 0;  // tier-private
};

// Abstract base class for storage tier backends (SSD, SPDK, ...).
// All tiers share a common interface for write/read/evict; tier-specific
// extensions (e.g. SSDTier::direct_io_active) live in the concrete subclass.
class TierBackend {
 public:
  virtual ~TierBackend() = default;

  // Non-copyable
  TierBackend(const TierBackend&) = delete;
  TierBackend& operator=(const TierBackend&) = delete;

  // --- Core interface (pure virtual) ---

  // Write data. Returns false if no space available.
  // Eviction semantics are tier-defined: some tiers may auto-evict,
  // others return false and let the upper layer decide.
  virtual bool Write(const std::string& key, const void* data, size_t size) = 0;

  // Copy data for |key| into the buffer at |dst_ptr|.
  virtual bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) = 0;

  virtual bool Exists(const std::string& key) const = 0;
  virtual bool Evict(const std::string& key) = 0;

  // Returns (used_bytes, total_bytes).
  virtual std::pair<size_t, size_t> Capacity() const = 0;

  virtual void Clear() = 0;

  // --- Extended interface (with default implementations) ---

  // Write from a raw pointer. Default casts to const void* and calls Write().
  virtual bool WriteFromPtr(const std::string& key, uintptr_t src_ptr, size_t size);

  // Read full data into a newly allocated buffer.
  // Default returns empty vector. Override in subclasses that track per-key sizes.
  virtual std::vector<char> Read(const std::string& key);

  // Optional capability-based extensions.
  virtual TierCapabilities Capabilities() const;
  virtual bool WriteBatch(const std::vector<std::string>& keys,
                          const std::vector<const void*>& data_ptrs,
                          const std::vector<size_t>& sizes);
  virtual std::vector<bool> ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& dst_ptrs,
                                             const std::vector<size_t>& sizes);
  virtual std::vector<bool> ReadBatchRangesIntoPtr(
      const std::vector<std::string>& keys, const std::vector<std::vector<uintptr_t>>& dst_ptrs,
      const std::vector<std::vector<size_t>>& sizes,
      const std::vector<std::vector<size_t>>& src_offsets);

  // Return the LRU key, or empty string if empty.
  // Default returns "". Override in tiers with LRU tracking.
  virtual std::string GetLRUKey() const;

  // Return up to |max_candidates| keys from the LRU tail (oldest first).
  // Default delegates to GetLRUKey(); override in tiers with LRU tracking.
  virtual std::vector<std::string> GetLRUCandidates(size_t max_candidates) const;

  // Up to |max_keys| keys in the order that frees space soonest, for tiers
  // that set TierCapabilities::orders_eviction.  Frees nothing.  Default: none.
  virtual std::vector<EvictionHint> EvictionOrder(size_t /*max_keys*/) { return {}; }

  // For tiers that delete whole files as space is freed.  |on_close| runs with
  // each deleted file's fd just before it is closed, and the close waits
  // |grace| after the delete: an fd handed out by LocateRecord() may still be
  // in a zero-copy reader's hands, which caches per-fd state and reads without
  // telling the tier when it is done.  Default: no files are ever deleted.
  virtual void SetRetiredFileHook(std::function<void(int fd)> /*on_close*/,
                                  std::chrono::milliseconds /*grace*/) {}

  // Default: a tier that deletes no files reports nothing.
  virtual TierSpaceStats SpaceStats() const { return {}; }

  // Batch write: write multiple keys in one call.
  // Default loops over Write(). Override for deep-queue NVMe pipeline.
  virtual std::vector<bool> BatchWrite(const std::vector<std::string>& keys,
                                       const std::vector<const void*>& data_ptrs,
                                       const std::vector<size_t>& sizes);

  // Batch read into user pointers. Default loops over ReadIntoPtr().
  virtual std::vector<bool> BatchReadIntoPtr(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& dst_ptrs,
                                             const std::vector<size_t>& sizes);

  // Ensure all prior writes are durable and visible to readers.
  // Default is a no-op (write-through tiers are always consistent).
  // SpdkProxyTier overrides this to drain pending write-back NVMe flushes.
  virtual bool Flush() { return true; }

  // Physical location of |key|'s value for a zero-copy reader that DMAs it
  // itself (GdsEngine).  Default: no stable file location — only the
  // segment-file SSD tiers override it.  Acquires no lock the caller must hold.
  // Touches LRU: the caller reads the record, just not through this tier.
  virtual std::optional<RecordLocation> LocateRecord(const std::string& /*key*/) {
    return std::nullopt;
  }

  // Write-in-place: reserve room for |key|, let a file engine fill
  // `location`, then CommitRecord makes it readable or AbortRecord drops it.
  // Default: unsupported (no stable file location).
  virtual std::optional<RecordReservation> ReserveRecord(const std::string& /*key*/,
                                                         size_t /*size*/) {
    return std::nullopt;
  }
  virtual bool CommitRecord(const std::string& /*key*/, const RecordReservation& /*r*/) {
    return false;
  }
  virtual void AbortRecord(const std::string& /*key*/, const RecordReservation& /*r*/) {}

  // Which StorageTier does this backend represent?
  StorageTier tier_id() const { return tier_id_; }

 protected:
  explicit TierBackend(StorageTier id) : tier_id_(id) {}

 private:
  StorageTier tier_id_;
};

}  // namespace mori::umbp

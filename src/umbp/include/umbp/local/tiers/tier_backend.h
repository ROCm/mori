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
};

// Abstract base class for storage tier backends (DRAM, SSD, NVM, ...).
// All tiers share a common interface for write/read/evict; tier-specific
// extensions (e.g., DRAMTier::ReadPtr) live in the concrete subclass.
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
  virtual const void* ReadPtr(const std::string& key, size_t* out_size);
  virtual bool WriteBatch(const std::vector<std::string>& keys,
                          const std::vector<const void*>& data_ptrs,
                          const std::vector<size_t>& sizes);
  virtual std::vector<bool> ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& dst_ptrs,
                                             const std::vector<size_t>& sizes);

  // Return the LRU key, or empty string if empty.
  // Default returns "". Override in tiers with LRU tracking.
  virtual std::string GetLRUKey() const;

  // Return up to |max_candidates| keys from the LRU tail (oldest first).
  // Default delegates to GetLRUKey(); override in tiers with LRU tracking.
  virtual std::vector<std::string> GetLRUCandidates(size_t max_candidates) const;

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

  // Toggle cold-read mode: bypass all caches, read directly from storage.
  // SSDTier: O_DIRECT; SpdkProxyTier: skip ring buffer cache.
  virtual void SetColdRead(bool /*enable*/) {}

  // Return an opaque location identifier for a previously written key.
  // Callers prepend the store index before publishing to the Master.
  virtual std::optional<std::string> GetLocationId(const std::string& key) const;

  // Which StorageTier does this backend represent?
  StorageTier tier_id() const { return tier_id_; }

 protected:
  explicit TierBackend(StorageTier id) : tier_id_(id) {}

 private:
  StorageTier tier_id_;
};

}  // namespace mori::umbp

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
#include <string>
#include <utility>
#include <vector>

#include "umbp/common/storage_tier.h"

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

  // Return the LRU key, or empty string if empty.
  // Default returns "". Override in tiers with LRU tracking.
  virtual std::string GetLRUKey() const;

  // Which StorageTier does this backend represent?
  StorageTier tier_id() const { return tier_id_; }

 protected:
  explicit TierBackend(StorageTier id) : tier_id_(id) {}

 private:
  StorageTier tier_id_;
};

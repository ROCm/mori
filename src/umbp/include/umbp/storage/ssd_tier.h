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
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/storage/tier_backend.h"

enum class SSDAccessMode : int {
  ReadWrite = 0,
  ReadOnlyShared = 1,
};

// SSD Tier: per-key file storage
class SSDTier : public TierBackend {
 public:
  SSDTier(const std::string& dir, size_t capacity,
          SSDAccessMode access_mode = SSDAccessMode::ReadWrite);
  ~SSDTier() override;

  // Non-copyable
  SSDTier(const SSDTier&) = delete;
  SSDTier& operator=(const SSDTier&) = delete;

  // TierBackend interface
  bool Write(const std::string& key, const void* data, size_t size) override;
  bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;

  // Extended interface overrides
  std::vector<char> Read(const std::string& key) override;
  std::string GetLRUKey() const override;

 private:
  std::string dir_;
  size_t capacity_, used_;
  SSDAccessMode access_mode_;
  std::unordered_map<std::string, size_t> keys_;  // key -> file size
  std::list<std::string> lru_list_;
  std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;
  mutable std::mutex mu_;

  bool IsReadOnlyShared() const { return access_mode_ == SSDAccessMode::ReadOnlyShared; }
  std::string KeyToPath(const std::string& key) const;
  bool FileExistsOnDisk(const std::string& key) const;
  void TouchLRU(const std::string& key);
  void EvictLRU();
};

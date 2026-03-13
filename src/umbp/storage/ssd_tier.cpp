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
#include "umbp/storage/ssd_tier.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {
inline size_t SaturatingSub(size_t a, size_t b) { return (a >= b) ? (a - b) : 0; }
}  // namespace

SSDTier::SSDTier(const std::string& dir, size_t capacity, SSDAccessMode access_mode)
    : TierBackend(StorageTier::LOCAL_SSD),
      dir_(dir),
      capacity_(capacity),
      used_(0),
      access_mode_(access_mode) {
  fs::create_directories(dir_);
}

SSDTier::~SSDTier() {
  // Don't clean up files on destruction — let Clear() handle explicit cleanup
}

std::string SSDTier::KeyToPath(const std::string& key) const { return dir_ + "/" + key + ".bin"; }

bool SSDTier::FileExistsOnDisk(const std::string& key) const {
  std::string path = KeyToPath(key);
  struct stat st;
  return (::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

void SSDTier::TouchLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();
}

void SSDTier::EvictLRU() {
  if (lru_list_.empty()) return;

  const std::string& victim = lru_list_.back();
  auto key_it = keys_.find(victim);
  if (key_it != keys_.end()) {
    if (!IsReadOnlyShared()) {
      std::string path = KeyToPath(victim);
      std::remove(path.c_str());
    }
    used_ = SaturatingSub(used_, key_it->second);
    keys_.erase(key_it);
  }
  lru_map_.erase(victim);
  lru_list_.pop_back();
}

bool SSDTier::Write(const std::string& key, const void* data, size_t size) {
  std::lock_guard<std::mutex> lock(mu_);

  if (IsReadOnlyShared()) {
    return false;
  }

  // Existing key will be atomically replaced by rename().
  auto existing = keys_.find(key);
  size_t existing_size = (existing == keys_.end()) ? 0 : existing->second;

  // Do NOT self-evict — return false if no space.
  // Upper layer (LocalStorageManager) is responsible for demoting keys
  // with index synchronization, just as it does for DRAMTier.
  if (SaturatingSub(used_, existing_size) + size > capacity_) {
    return false;
  }

  // Atomic write: write to .tmp then rename, so followers never see partial files
  std::string path = KeyToPath(key);
  std::string tmp_path = path + ".tmp";
  int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;

  size_t written = 0;
  while (written < size) {
    ssize_t n = ::write(fd, static_cast<const char*>(data) + written, size - written);
    if (n < 0) {
      close(fd);
      std::remove(tmp_path.c_str());
      return false;
    }
    written += n;
  }
  // Flush data to disk before rename so followers see complete content
  // once the file appears under its final name.
  fsync(fd);
  close(fd);

  if (rename(tmp_path.c_str(), path.c_str()) != 0) {
    std::remove(tmp_path.c_str());
    return false;
  }

  if (existing != keys_.end()) {
    auto lru_it = lru_map_.find(key);
    if (lru_it != lru_map_.end()) {
      lru_list_.erase(lru_it->second);
      lru_map_.erase(lru_it);
    }
  }

  keys_[key] = size;
  used_ = SaturatingSub(used_, existing_size) + size;
  TouchLRU(key);
  return true;
}

bool SSDTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = keys_.find(key);
  if (it != keys_.end()) {
    // Fast path: key tracked locally
    if (size != it->second) return false;

    std::string path = KeyToPath(key);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      // File disappeared; drop local tracking to avoid permanent false positives.
      used_ = SaturatingSub(used_, it->second);
      keys_.erase(it);
      auto lru_it = lru_map_.find(key);
      if (lru_it != lru_map_.end()) {
        lru_list_.erase(lru_it->second);
        lru_map_.erase(lru_it);
      }
      return false;
    }

    size_t total_read = 0;
    while (total_read < size) {
      ssize_t n =
          pread(fd, reinterpret_cast<char*>(dst_ptr) + total_read, size - total_read, total_read);
      if (n <= 0) {
        close(fd);
        used_ = SaturatingSub(used_, it->second);
        keys_.erase(key);
        auto lru_it = lru_map_.find(key);
        if (lru_it != lru_map_.end()) {
          lru_list_.erase(lru_it->second);
          lru_map_.erase(lru_it);
        }
        return false;
      }
      total_read += n;
    }
    close(fd);

    TouchLRU(key);
    return true;
  }

  // Read-only shared fallback: key not in local map, try filesystem
  if (!IsReadOnlyShared()) return false;

  std::string path = KeyToPath(key);
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;

  struct stat st;
  if (fstat(fd, &st) < 0 || static_cast<size_t>(st.st_size) != size) {
    close(fd);
    return false;
  }

  size_t total_read = 0;
  while (total_read < size) {
    ssize_t n =
        pread(fd, reinterpret_cast<char*>(dst_ptr) + total_read, size - total_read, total_read);
    if (n <= 0) {
      close(fd);
      return false;
    }
    total_read += n;
  }
  close(fd);

  // Register in local map for future fast-path lookups.
  // Do not increment used_ — disk space is owned by leader.
  keys_[key] = size;
  TouchLRU(key);
  return true;
}

std::vector<char> SSDTier::Read(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = keys_.find(key);
  if (it != keys_.end()) {
    // Fast path: known key
    std::string path = KeyToPath(key);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      // File disappeared; drop local tracking to avoid permanent false positives.
      used_ = SaturatingSub(used_, it->second);
      keys_.erase(it);
      auto lru_it = lru_map_.find(key);
      if (lru_it != lru_map_.end()) {
        lru_list_.erase(lru_it->second);
        lru_map_.erase(lru_it);
      }
      return {};
    }

    size_t file_size = it->second;
    std::vector<char> buf(file_size);
    size_t total_read = 0;
    while (total_read < file_size) {
      ssize_t n = ::read(fd, buf.data() + total_read, file_size - total_read);
      if (n <= 0) {
        close(fd);
        used_ = SaturatingSub(used_, it->second);
        keys_.erase(key);
        auto lru_it = lru_map_.find(key);
        if (lru_it != lru_map_.end()) {
          lru_list_.erase(lru_it->second);
          lru_map_.erase(lru_it);
        }
        return {};
      }
      total_read += n;
    }
    close(fd);

    TouchLRU(key);
    return buf;
  }

  // Read-only shared fallback
  if (!IsReadOnlyShared()) return {};

  std::string path = KeyToPath(key);
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return {};

  struct stat st;
  if (fstat(fd, &st) < 0 || st.st_size <= 0) {
    close(fd);
    return {};
  }

  size_t file_size = static_cast<size_t>(st.st_size);
  std::vector<char> buf(file_size);
  size_t total_read = 0;
  while (total_read < file_size) {
    ssize_t n = ::read(fd, buf.data() + total_read, file_size - total_read);
    if (n <= 0) {
      close(fd);
      return {};
    }
    total_read += n;
  }
  close(fd);

  keys_[key] = file_size;
  TouchLRU(key);
  return buf;
}

std::vector<std::string> SSDTier::GetLRUCandidates(size_t max_candidates) const {
  if (max_candidates == 0) max_candidates = 1;
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> result;
  result.reserve(std::min(max_candidates, lru_list_.size()));
  auto it = lru_list_.rbegin();
  for (size_t i = 0; i < max_candidates && it != lru_list_.rend(); ++i, ++it) {
    result.push_back(*it);
  }
  return result;
}

std::string SSDTier::GetLRUKey() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

bool SSDTier::Exists(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (keys_.count(key) > 0) {
    if (!IsReadOnlyShared()) return true;
    // In read-only shared mode, local keys_ may be stale if leader evicted the file.
    // Always consult the filesystem as the source of truth.
    return FileExistsOnDisk(key);
  }
  if (IsReadOnlyShared()) return FileExistsOnDisk(key);
  return false;
}

bool SSDTier::Evict(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = keys_.find(key);
  if (it == keys_.end()) return false;

  if (!IsReadOnlyShared()) {
    // Leader/normal: delete the file
    std::string path = KeyToPath(key);
    std::remove(path.c_str());
  }
  // Read-only shared: only remove from local tracking, do NOT delete file

  used_ = SaturatingSub(used_, it->second);
  keys_.erase(it);

  auto lru_it = lru_map_.find(key);
  if (lru_it != lru_map_.end()) {
    lru_list_.erase(lru_it->second);
    lru_map_.erase(lru_it);
  }
  return true;
}

std::pair<size_t, size_t> SSDTier::Capacity() const {
  std::lock_guard<std::mutex> lock(mu_);
  return {used_, capacity_};
}

void SSDTier::Clear() {
  std::lock_guard<std::mutex> lock(mu_);

  if (!IsReadOnlyShared()) {
    // Leader/normal: delete all files
    for (auto& [key, _] : keys_) {
      std::string path = KeyToPath(key);
      std::remove(path.c_str());
    }
  }
  // Read-only shared: only clear local tracking
  keys_.clear();
  lru_list_.clear();
  lru_map_.clear();
  used_ = 0;
}

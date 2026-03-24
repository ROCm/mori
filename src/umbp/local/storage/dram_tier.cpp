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
#include "umbp/local/storage/dram_tier.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mori::umbp {

DRAMTier::DRAMTier(size_t capacity, bool use_shm, const std::string& shm_name)
    : TierBackend(StorageTier::CPU_DRAM),
      base_ptr_(nullptr),
      capacity_(capacity),
      used_(0),
      shm_fd_(-1),
      use_shm_(use_shm),
      shm_name_(shm_name) {
  if (use_shm_) {
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ < 0) {
      throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
    }
    if (ftruncate(shm_fd_, capacity_) < 0) {
      close(shm_fd_);
      shm_unlink(shm_name_.c_str());
      throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
    }
    base_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  } else {
    base_ptr_ =
        mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  }

  if (base_ptr_ == MAP_FAILED) {
    if (use_shm_ && shm_fd_ >= 0) {
      close(shm_fd_);
      shm_unlink(shm_name_.c_str());
    }
    throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
  }

  // Initialize free list with entire capacity
  free_list_.push_back({0, capacity_});
}

DRAMTier::~DRAMTier() {
  if (base_ptr_ && base_ptr_ != MAP_FAILED) {
    munmap(base_ptr_, capacity_);
  }
  if (use_shm_) {
    if (shm_fd_ >= 0) close(shm_fd_);
    shm_unlink(shm_name_.c_str());
  }
}

size_t DRAMTier::Allocate(size_t size) {
  // First-fit allocation
  for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
    if (it->size >= size) {
      size_t offset = it->offset;
      if (it->size == size) {
        free_list_.erase(it);
      } else {
        it->offset += size;
        it->size -= size;
      }
      return offset;
    }
  }
  return static_cast<size_t>(-1);  // Allocation failed
}

void DRAMTier::Deallocate(size_t offset, size_t size) {
  // Insert into sorted position and coalesce adjacent blocks
  auto it = free_list_.begin();
  while (it != free_list_.end() && it->offset < offset) {
    ++it;
  }

  auto new_it = free_list_.insert(it, {offset, size});

  // Coalesce with next block
  auto next = std::next(new_it);
  if (next != free_list_.end() && new_it->offset + new_it->size == next->offset) {
    new_it->size += next->size;
    free_list_.erase(next);
  }

  // Coalesce with previous block
  if (new_it != free_list_.begin()) {
    auto prev = std::prev(new_it);
    if (prev->offset + prev->size == new_it->offset) {
      prev->size += new_it->size;
      free_list_.erase(new_it);
    }
  }
}

void DRAMTier::TouchLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();
}

void DRAMTier::EvictLRU() {
  if (lru_list_.empty()) return;

  const std::string& victim = lru_list_.back();
  auto slot_it = slots_.find(victim);
  if (slot_it != slots_.end()) {
    Deallocate(slot_it->second.offset, slot_it->second.size);
    used_ -= slot_it->second.size;
    slots_.erase(slot_it);
  }
  lru_map_.erase(victim);
  lru_list_.pop_back();
}

bool DRAMTier::Write(const std::string& key, const void* data, size_t size) {
  std::lock_guard<std::mutex> lock(mu_);

  // If key already exists, free its old slot first
  auto existing = slots_.find(key);
  if (existing != slots_.end()) {
    Deallocate(existing->second.offset, existing->second.size);
    used_ -= existing->second.size;
    slots_.erase(existing);
    auto lru_it = lru_map_.find(key);
    if (lru_it != lru_map_.end()) {
      lru_list_.erase(lru_it->second);
      lru_map_.erase(lru_it);
    }
  }

  // Try to allocate — do NOT self-evict.
  // If no space, return false so upper layer can demote keys to SSD.
  size_t offset = Allocate(size);
  if (offset == static_cast<size_t>(-1)) {
    return false;
  }

  std::memcpy(static_cast<char*>(base_ptr_) + offset, data, size);
  slots_[key] = {offset, size};
  used_ += size;
  TouchLRU(key);
  return true;
}

bool DRAMTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  // Reject if caller's buffer size does not match the stored block size.
  // A mismatch indicates a caller bug (wrong page size); silently truncating
  // would produce a partially-filled KV block with no error signal.
  if (size != it->second.size) return false;

  std::memcpy(reinterpret_cast<void*>(dst_ptr), static_cast<char*>(base_ptr_) + it->second.offset,
              size);
  TouchLRU(key);
  return true;
}

const void* DRAMTier::ReadPtr(const std::string& key, size_t* out_size) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return nullptr;

  if (out_size) *out_size = it->second.size;
  TouchLRU(key);
  return static_cast<char*>(base_ptr_) + it->second.offset;
}

std::vector<char> DRAMTier::Read(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return {};

  size_t sz = it->second.size;
  std::vector<char> buf(sz);
  std::memcpy(buf.data(), static_cast<char*>(base_ptr_) + it->second.offset, sz);
  TouchLRU(key);
  return buf;
}

TierCapabilities DRAMTier::Capabilities() const {
  TierCapabilities caps;
  caps.zero_copy_read = true;
  return caps;
}

bool DRAMTier::Exists(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  return slots_.count(key) > 0;
}

bool DRAMTier::Evict(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  Deallocate(it->second.offset, it->second.size);
  used_ -= it->second.size;
  slots_.erase(it);

  auto lru_it = lru_map_.find(key);
  if (lru_it != lru_map_.end()) {
    lru_list_.erase(lru_it->second);
    lru_map_.erase(lru_it);
  }
  return true;
}

std::pair<size_t, size_t> DRAMTier::Capacity() const {
  std::lock_guard<std::mutex> lock(mu_);
  return {used_, capacity_};
}

void DRAMTier::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  slots_.clear();
  lru_list_.clear();
  lru_map_.clear();
  free_list_.clear();
  free_list_.push_back({0, capacity_});
  used_ = 0;
}

std::vector<std::string> DRAMTier::GetLRUCandidates(size_t max_candidates) const {
  if (max_candidates == 0) max_candidates = 1;
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> result;
  result.reserve(std::min(max_candidates, lru_list_.size()));
  // Walk from the back (LRU end) up to max_candidates entries.
  auto it = lru_list_.rbegin();
  for (size_t i = 0; i < max_candidates && it != lru_list_.rend(); ++i, ++it) {
    result.push_back(*it);
  }
  return result;
}

std::string DRAMTier::GetLRUKey() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::optional<size_t> DRAMTier::GetSlotOffset(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = slots_.find(key);
  if (it == slots_.end()) return std::nullopt;
  return it->second.offset;
}

}  // namespace mori::umbp

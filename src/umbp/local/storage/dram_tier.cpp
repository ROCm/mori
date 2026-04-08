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

  // Default to a single full-arena chunk.  Distributed mode may call
  // ConfigureChunks() later to re-slice for RDMA MR size limits.
  DramChunk chunk;
  chunk.base = base_ptr_;
  chunk.size = capacity_;
  chunk.free_list.push_back({0, capacity_});
  chunks_.push_back(std::move(chunk));
  chunks_configured_ = true;
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

void DRAMTier::ConfigureChunks(size_t chunk_size) {
  std::lock_guard<std::mutex> lock(mu_);
  if (chunks_sealed_) {
    throw std::runtime_error(
        "DRAMTier::ConfigureChunks: chunk layout has been sealed and cannot be reconfigured");
  }

  chunks_.clear();

  // chunk_size == 0, SIZE_MAX, or >= capacity_ → single full-arena chunk.
  if (chunk_size == 0 || chunk_size >= capacity_) {
    DramChunk chunk;
    chunk.base = base_ptr_;
    chunk.size = capacity_;
    chunk.free_list.push_back({0, capacity_});
    chunks_.push_back(std::move(chunk));
  } else {
    for (size_t off = 0; off < capacity_; off += chunk_size) {
      size_t sz = std::min(chunk_size, capacity_ - off);
      DramChunk chunk;
      chunk.base = static_cast<char*>(base_ptr_) + off;
      chunk.size = sz;
      chunk.free_list.push_back({0, sz});
      chunks_.push_back(std::move(chunk));
    }
  }

  chunks_configured_ = true;
}

void DRAMTier::SealChunkLayout() {
  std::lock_guard<std::mutex> lock(mu_);
  chunks_sealed_ = true;
}

std::vector<ExportableDram> DRAMTier::GetExportableChunks() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<ExportableDram> result;
  result.reserve(chunks_.size());
  for (const auto& chunk : chunks_) {
    result.push_back({chunk.base, chunk.size});
  }
  return result;
}

std::optional<DRAMTier::ChunkLocation> DRAMTier::GetSlotChunkLocation(
    const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = slots_.find(key);
  if (it == slots_.end()) return std::nullopt;
  return ChunkLocation{it->second.chunk_index, it->second.offset};
}

std::optional<std::pair<uint32_t, size_t>> DRAMTier::Allocate(size_t size) {
  if (!chunks_configured_) {
    return std::nullopt;
  }

  // Iterate chunks in ascending index order (first-fit across chunks).
  for (uint32_t ci = 0; ci < static_cast<uint32_t>(chunks_.size()); ++ci) {
    auto& chunk = chunks_[ci];

    // Fast reject: block cannot exceed this chunk's size.
    if (size > chunk.size) continue;

    // First-fit within this chunk's free list.
    for (auto it = chunk.free_list.begin(); it != chunk.free_list.end(); ++it) {
      if (it->size < size) continue;

      size_t offset = it->offset;
      if (it->size == size) {
        chunk.free_list.erase(it);
      } else {
        it->offset += size;
        it->size -= size;
      }
      return std::make_pair(ci, offset);
    }
  }
  return std::nullopt;
}

void DRAMTier::Deallocate(uint32_t chunk_index, size_t offset, size_t size) {
  auto& free_list = chunks_[chunk_index].free_list;

  // Insert into sorted position and coalesce adjacent blocks.
  auto it = free_list.begin();
  while (it != free_list.end() && it->offset < offset) {
    ++it;
  }

  auto new_it = free_list.insert(it, {offset, size});

  // Coalesce with next block
  auto next = std::next(new_it);
  if (next != free_list.end() && new_it->offset + new_it->size == next->offset) {
    new_it->size += next->size;
    free_list.erase(next);
  }

  // Coalesce with previous block
  if (new_it != free_list.begin()) {
    auto prev = std::prev(new_it);
    if (prev->offset + prev->size == new_it->offset) {
      prev->size += new_it->size;
      free_list.erase(new_it);
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
    Deallocate(slot_it->second.chunk_index, slot_it->second.offset, slot_it->second.size);
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
    Deallocate(existing->second.chunk_index, existing->second.offset, existing->second.size);
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
  auto alloc = Allocate(size);
  if (!alloc) {
    return false;
  }

  chunks_sealed_ = true;
  auto [chunk_index, offset] = *alloc;
  std::memcpy(static_cast<char*>(chunks_[chunk_index].base) + offset, data, size);
  slots_[key] = {chunk_index, offset, size};
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

  const auto& slot = it->second;
  std::memcpy(reinterpret_cast<void*>(dst_ptr),
              static_cast<char*>(chunks_[slot.chunk_index].base) + slot.offset, size);
  TouchLRU(key);
  return true;
}

const void* DRAMTier::ReadPtr(const std::string& key, size_t* out_size) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return nullptr;

  const auto& slot = it->second;
  if (out_size) *out_size = slot.size;
  TouchLRU(key);
  return static_cast<char*>(chunks_[slot.chunk_index].base) + slot.offset;
}

std::vector<char> DRAMTier::Read(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return {};

  const auto& slot = it->second;
  std::vector<char> buf(slot.size);
  std::memcpy(buf.data(), static_cast<char*>(chunks_[slot.chunk_index].base) + slot.offset,
              slot.size);
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

  Deallocate(it->second.chunk_index, it->second.offset, it->second.size);
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
  used_ = 0;

  if (chunks_configured_) {
    // Re-initialize each chunk's free list to cover its full size.
    for (auto& chunk : chunks_) {
      chunk.free_list.clear();
      chunk.free_list.push_back({0, chunk.size});
    }
  }
  // If chunks were never configured, the next Write() will auto-configure.
}

std::vector<std::string> DRAMTier::GetLRUCandidates(size_t max_candidates) const {
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

std::string DRAMTier::GetLRUKey() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::optional<size_t> DRAMTier::GetSlotOffset(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = slots_.find(key);
  if (it == slots_.end()) return std::nullopt;
  const auto& slot = it->second;
  // Reconstruct global offset from chunk base pointer.
  return static_cast<size_t>(static_cast<char*>(chunks_[slot.chunk_index].base) -
                             static_cast<char*>(base_ptr_)) +
         slot.offset;
}

std::optional<std::string> DRAMTier::GetLocationId(const std::string& key) const {
  auto offset = GetSlotOffset(key);
  if (!offset.has_value()) {
    return std::nullopt;
  }
  return std::to_string(*offset);
}

}  // namespace mori::umbp

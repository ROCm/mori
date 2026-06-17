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
#include "umbp/local/tiers/dummy_ssd_tier.h"

#include <algorithm>

namespace mori::umbp {

DummySsdTier::DummySsdTier(size_t capacity)
    : TierBackend(StorageTier::LOCAL_SSD), capacity_(capacity) {}

bool DummySsdTier::Write(const std::string& key, const void* /*data*/, size_t size) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = keys_.find(key);
  if (it != keys_.end()) {
    TouchLRU(key);
    return true;
  }
  if (used_ + size > capacity_) return false;
  keys_[key] = size;
  used_ += size;
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();
  return true;
}

bool DummySsdTier::ReadIntoPtr(const std::string& key, uintptr_t /*dst_ptr*/, size_t /*size*/) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = keys_.find(key);
  if (it == keys_.end()) return false;
  TouchLRU(key);
  return true;
}

bool DummySsdTier::Exists(const std::string& key) const {
  std::lock_guard<std::mutex> lk(mu_);
  return keys_.count(key) > 0;
}

bool DummySsdTier::Evict(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = keys_.find(key);
  if (it == keys_.end()) return false;
  used_ -= it->second;
  keys_.erase(it);
  auto lru_it = lru_map_.find(key);
  if (lru_it != lru_map_.end()) {
    lru_list_.erase(lru_it->second);
    lru_map_.erase(lru_it);
  }
  return true;
}

std::pair<size_t, size_t> DummySsdTier::Capacity() const {
  std::lock_guard<std::mutex> lk(mu_);
  return {used_, capacity_};
}

void DummySsdTier::Clear() {
  std::lock_guard<std::mutex> lk(mu_);
  keys_.clear();
  lru_list_.clear();
  lru_map_.clear();
  used_ = 0;
}

std::vector<char> DummySsdTier::Read(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = keys_.find(key);
  if (it == keys_.end()) return {};
  TouchLRU(key);
  return std::vector<char>(it->second, 0);
}

std::string DummySsdTier::GetLRUKey() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::vector<std::string> DummySsdTier::GetLRUCandidates(size_t max_candidates) const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::string> result;
  result.reserve(std::min(max_candidates, lru_list_.size()));
  for (auto rit = lru_list_.rbegin(); rit != lru_list_.rend() && result.size() < max_candidates;
       ++rit) {
    result.push_back(*rit);
  }
  return result;
}

void DummySsdTier::TouchLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
    lru_list_.push_front(key);
    it->second = lru_list_.begin();
  }
}

}  // namespace mori::umbp

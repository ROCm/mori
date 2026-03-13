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
#include "umbp/block_index/block_index.h"

#include <mutex>
#include <shared_mutex>

bool BlockIndexClient::MayExist(const std::string& key) const {
  std::shared_lock lock(mu_);
  return index_.count(key) > 0;
}

std::optional<LocalLocation> BlockIndexClient::Lookup(const std::string& key) const {
  std::shared_lock lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return std::nullopt;
  return it->second;
}

void BlockIndexClient::Insert(const std::string& key, const LocalLocation& loc) {
  std::unique_lock lock(mu_);
  index_[key] = loc;
}

std::optional<LocalLocation> BlockIndexClient::Remove(const std::string& key) {
  std::unique_lock lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return std::nullopt;
  auto loc = it->second;
  index_.erase(it);
  return loc;
}

bool BlockIndexClient::UpdateTier(const std::string& key, StorageTier new_tier) {
  std::unique_lock lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  it->second.tier = new_tier;
  return true;
}

size_t BlockIndexClient::Count() const {
  std::shared_lock lock(mu_);
  return index_.size();
}

void BlockIndexClient::Clear() {
  std::unique_lock lock(mu_);
  index_.clear();
}

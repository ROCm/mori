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
//
// Peer-side SSD eviction policy plug-in.  Covers the default LruSsdEvictStrategy
// behavior (oldest-first by lru_rank until the byte budget is met) and the
// PeerSsdManager DI seam (SelectVictims routes through the injected strategy and
// returns exactly what it returns).
#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp/distributed/peer/ssd_evict_strategy.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {
namespace {

// -------------------- 1. Default LRU behavior --------------------

TEST(LruSsdEvictStrategy, PicksOldestFirstUntilBudgetMet) {
  LruSsdEvictStrategy strategy;
  // lru_rank: 0 == oldest.  Provided out of order to prove the policy sorts.
  std::vector<SsdEvictCandidate> candidates = {
      SsdEvictCandidate{"new", 100, 2},
      SsdEvictCandidate{"old", 100, 0},
      SsdEvictCandidate{"mid", 100, 1},
  };
  auto victims = strategy.SelectVictims(candidates, /*bytes_to_free=*/150);
  ASSERT_EQ(victims.size(), 2u);
  EXPECT_EQ(victims[0], "old");
  EXPECT_EQ(victims[1], "mid");
}

TEST(LruSsdEvictStrategy, ZeroBudgetSelectsNothing) {
  LruSsdEvictStrategy strategy;
  std::vector<SsdEvictCandidate> candidates = {SsdEvictCandidate{"k", 100, 0}};
  EXPECT_TRUE(strategy.SelectVictims(candidates, 0).empty());
}

// -------------------- 2. PeerSsdManager DI seam --------------------

// Minimal in-memory backend: just enough of TierBackend to land Writes so the
// manager builds a candidate snapshot.
class InMemoryBackend : public TierBackend {
 public:
  explicit InMemoryBackend(size_t capacity)
      : TierBackend(StorageTier::LOCAL_SSD), capacity_(capacity) {}

  bool Write(const std::string& key, const void* data, size_t size) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    size_t prev = (it == store_.end()) ? 0 : it->second.size();
    if (used_ - prev + size > capacity_) return false;
    store_[key].assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
    used_ = used_ - prev + size;
    return true;
  }
  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    if (it == store_.end() || it->second.size() != size) return false;
    std::memcpy(reinterpret_cast<void*>(dst), it->second.data(), size);
    return true;
  }
  bool Exists(const std::string& key) const override {
    std::lock_guard<std::mutex> lk(mu_);
    return store_.count(key) != 0;
  }
  bool Evict(const std::string& key) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    if (it == store_.end()) return false;
    used_ -= it->second.size();
    store_.erase(it);
    return true;
  }
  std::pair<size_t, size_t> Capacity() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return {used_, capacity_};
  }
  void Clear() override {
    std::lock_guard<std::mutex> lk(mu_);
    store_.clear();
    used_ = 0;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::vector<char>> store_;
  size_t used_ = 0;
  size_t capacity_;
};

// Fixed-output policy: lets the test assert the manager ran it and returned its
// result verbatim.
class FakeSsdEvictStrategy : public SsdEvictStrategy {
 public:
  std::vector<std::string> SelectVictims(std::vector<SsdEvictCandidate> candidates,
                                         size_t /*bytes_to_free*/) override {
    ++call_count;
    received_candidates = candidates.size();
    return canned;
  }
  int call_count = 0;
  size_t received_candidates = 0;
  std::vector<std::string> canned;
};

std::vector<std::pair<const void*, size_t>> OneSeg(const std::string& s) {
  return {{s.data(), s.size()}};
}

TEST(PeerSsdManagerPlugin, RoutesVictimSelectionThroughInjectedStrategy) {
  auto fake = std::make_unique<FakeSsdEvictStrategy>();
  fake->canned = {"B"};
  FakeSsdEvictStrategy* fake_raw = fake.get();

  // Roomy capacity so no watermark eviction fires during the writes.
  PeerSsdManager mgr(std::make_unique<InMemoryBackend>(1'000'000), /*high=*/0.9, /*low=*/0.7,
                     std::move(fake));
  ASSERT_TRUE(mgr.Write("A", OneSeg("aaaa"), 4));
  ASSERT_TRUE(mgr.Write("B", OneSeg("bbbb"), 4));
  ASSERT_TRUE(mgr.Write("C", OneSeg("cccc"), 4));

  // bytes_to_free (100) exceeds the 12 bytes resident, so the snapshot walk
  // covers all three keys before the early-stop would trigger.
  auto victims = mgr.SelectVictims(/*bytes_to_free=*/100);

  EXPECT_EQ(fake_raw->call_count, 1);
  EXPECT_EQ(fake_raw->received_candidates, 3u);  // A, B, C all evictable
  ASSERT_EQ(victims.size(), 1u);
  EXPECT_EQ(victims[0], "B");  // verbatim from the strategy
}

}  // namespace
}  // namespace mori::umbp

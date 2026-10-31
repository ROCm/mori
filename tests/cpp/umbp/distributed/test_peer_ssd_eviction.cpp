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
// The SSD medium's Evict() and candidate order, over an in-memory TierBackend.
// Watermark eviction is PeerPool's, tested there.
#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/ssd/peer_ssd_manager.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {
namespace {

// In-memory TierBackend with test hooks: blockable reads (to hold the in-flight
// guard), forced evict failure, and call counters.
class FakeBackend : public TierBackend {
 public:
  explicit FakeBackend(size_t capacity)
      : TierBackend(StorageTier::LOCAL_SSD), capacity_(capacity) {}

  bool Write(const std::string& key, const void* data, size_t size) override {
    std::lock_guard<std::mutex> lk(mu_);
    ++write_calls_;
    auto it = store_.find(key);
    size_t prev = (it == store_.end()) ? 0 : it->second.size();
    if (used_ - prev + size > capacity_) return false;  // ENOSPC
    store_[key].assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
    used_ = used_ - prev + size;
    return true;
  }

  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) override {
    {
      std::unique_lock<std::mutex> lk(gate_mu_);
      ++reads_started_;
      started_cv_.notify_all();
      gate_cv_.wait(lk, [this] { return !read_blocked_; });
    }
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
    if (fail_evict_) return false;
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
    ++clear_calls_;
    store_.clear();
    used_ = 0;
  }

  // --- test controls ---
  void BlockReads() {
    std::lock_guard<std::mutex> lk(gate_mu_);
    read_blocked_ = true;
  }
  void UnblockReads() {
    {
      std::lock_guard<std::mutex> lk(gate_mu_);
      read_blocked_ = false;
    }
    gate_cv_.notify_all();
  }
  void WaitReadsStarted(int n) {
    std::unique_lock<std::mutex> lk(gate_mu_);
    started_cv_.wait(lk, [&] { return reads_started_ >= n; });
  }
  void SetFailEvict(bool f) {
    std::lock_guard<std::mutex> lk(mu_);
    fail_evict_ = f;
  }
  int write_calls() const {
    std::lock_guard<std::mutex> lk(mu_);
    return write_calls_;
  }
  int clear_calls() const {
    std::lock_guard<std::mutex> lk(mu_);
    return clear_calls_;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::vector<char>> store_;
  size_t used_ = 0;
  size_t capacity_;
  int write_calls_ = 0;
  int clear_calls_ = 0;
  bool fail_evict_ = false;

  std::mutex gate_mu_;
  std::condition_variable gate_cv_;
  std::condition_variable started_cv_;
  bool read_blocked_ = false;
  int reads_started_ = 0;
};

bool Put(PeerSsdManager& mgr, const std::string& key, const std::string& value) {
  return mgr.WriteBatch({key}, {value.data()}, {value.size()})[0];
}

// Manager owning a FakeBackend we keep a raw pointer to for test inspection.
struct Harness {
  FakeBackend* backend;
  std::unique_ptr<PeerSsdManager> mgr;
};

Harness MakeHarness(size_t capacity) {
  auto be = std::make_unique<FakeBackend>(capacity);
  FakeBackend* raw = be.get();
  return Harness{raw, std::make_unique<PeerSsdManager>(std::move(be))};
}

// A tier that orders its own eviction, records the touches it gets, and can
// hold an eviction while it is under way.
class OrderingBackend : public FakeBackend {
 public:
  using FakeBackend::FakeBackend;

  TierCapabilities Capabilities() const override {
    TierCapabilities caps = FakeBackend::Capabilities();
    caps.orders_eviction = true;
    return caps;
  }

  size_t Touch(const std::vector<std::string>& keys) override {
    std::lock_guard<std::mutex> lk(touch_mu_);
    touched_.insert(touched_.end(), keys.begin(), keys.end());
    return keys.size();
  }

  bool Evict(const std::string& key) override {
    {
      std::unique_lock<std::mutex> lk(evict_mu_);
      ++evicts_started_;
      evict_started_cv_.notify_all();
      evict_cv_.wait(lk, [this] { return !evict_blocked_; });
    }
    return FakeBackend::Evict(key);
  }

  std::vector<std::string> touched() const {
    std::lock_guard<std::mutex> lk(touch_mu_);
    return touched_;
  }
  void BlockEvicts() {
    std::lock_guard<std::mutex> lk(evict_mu_);
    evict_blocked_ = true;
  }
  void UnblockEvicts() {
    {
      std::lock_guard<std::mutex> lk(evict_mu_);
      evict_blocked_ = false;
    }
    evict_cv_.notify_all();
  }
  void WaitEvictStarted() {
    std::unique_lock<std::mutex> lk(evict_mu_);
    evict_started_cv_.wait(lk, [this] { return evicts_started_ > 0; });
  }

 private:
  mutable std::mutex touch_mu_;
  std::vector<std::string> touched_;
  std::mutex evict_mu_;
  std::condition_variable evict_cv_;
  std::condition_variable evict_started_cv_;
  bool evict_blocked_ = false;
  int evicts_started_ = 0;
};

int CountKind(const std::vector<KvEvent>& events, KvEvent::Kind kind) {
  int n = 0;
  for (const auto& e : events) {
    if (e.kind == kind && e.tier == TierType::SSD) ++n;
  }
  return n;
}

// ---------------------------------------------------------------------------

TEST(PeerSsdEviction, WriteAndPrepareReadRefreshLru) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "A", "aaaa"));
  ASSERT_TRUE(Put(*h.mgr, "B", "bbbb"));
  ASSERT_TRUE(Put(*h.mgr, "C", "cccc"));

  // LRU is A, B, C; reading A makes B the coldest.
  std::vector<char> buf(4);
  auto out = h.mgr->PrepareRead("A", buf.data(), buf.size());
  ASSERT_EQ(out.status, SsdReadStatus::kOk);
  EXPECT_EQ(std::string(buf.data(), out.size), "aaaa");

  auto candidates = h.mgr->EvictionCandidates(/*max_count=*/1);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates.front().key, "B");
  EXPECT_EQ(candidates.front().bytes, 4u);
  EXPECT_EQ(h.mgr->EvictionCandidates(100).size(), 3u);
  EXPECT_TRUE(h.mgr->Exists("B")) << "asking for candidates freed one";
}

TEST(PeerSsdEviction, TouchRefreshesLruAsAReadDoes) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "A", "aaaa"));
  ASSERT_TRUE(Put(*h.mgr, "B", "bbbb"));
  ASSERT_TRUE(Put(*h.mgr, "C", "cccc"));

  // LRU is A, B, C; touching A makes B the coldest without reading anything.
  EXPECT_EQ(h.mgr->Touch({"A", "missing"}), 1u);
  EXPECT_EQ(h.mgr->Touch({}), 0u);
  auto candidates = h.mgr->EvictionCandidates(/*max_count=*/1);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates.front().key, "B");
  EXPECT_EQ(h.mgr->EvictionCandidates(100).size(), 3u) << "a touch stored the unknown key";
}

TEST(PeerSsdEviction, TouchReachesATierThatOrdersItsOwnEviction) {
  auto backend = std::make_unique<OrderingBackend>(/*capacity=*/1'000'000);
  OrderingBackend* tier = backend.get();
  PeerSsdManager mgr(std::move(backend));
  ASSERT_TRUE(mgr.OrdersEviction());
  ASSERT_TRUE(Put(mgr, "A", "aaaa"));
  ASSERT_TRUE(Put(mgr, "B", "bbbb"));

  EXPECT_EQ(mgr.Touch({"A", "missing", "B"}), 2u);
  EXPECT_EQ(tier->touched(), (std::vector<std::string>{"A", "B"}))
      << "only held keys reach the tier";

  // A key whose eviction is under way is no longer held.
  tier->BlockEvicts();
  std::thread evictor([&] { EXPECT_TRUE(mgr.Evict("A")); });
  tier->WaitEvictStarted();
  EXPECT_EQ(mgr.Touch({"A", "B"}), 1u);
  tier->UnblockEvicts();
  evictor.join();
  EXPECT_EQ(tier->touched(), (std::vector<std::string>{"A", "B", "B"}));
  EXPECT_EQ(mgr.Touch({"A"}), 0u);
}

TEST(PeerSsdEviction, WritesNeverEvict) {
  // Neither 90% usage nor ENOSPC makes a write evict earlier keys.
  auto h = MakeHarness(/*capacity=*/1000);
  std::string val(100, 'x');
  for (int i = 1; i <= 9; ++i) {
    ASSERT_TRUE(Put(*h.mgr, "k" + std::to_string(i), val));
  }
  EXPECT_EQ(h.mgr->Capacity().first, 900u);
  EXPECT_FALSE(Put(*h.mgr, "big", std::string(300, 'c')));
  EXPECT_FALSE(h.mgr->Exists("big"));

  // The batched path fails the oversized key alone.
  const std::string big(300, 'c');
  const std::string small(100, 's');
  auto ok =
      h.mgr->WriteBatch({"big", "small"}, {big.data(), small.data()}, {big.size(), small.size()});
  EXPECT_EQ(ok, (std::vector<bool>{false, true}));
  for (int i = 1; i <= 9; ++i) EXPECT_TRUE(h.mgr->Exists("k" + std::to_string(i)));
  EXPECT_EQ(CountKind(h.mgr->DrainPendingEvents(), KvEvent::Kind::REMOVE), 0);
}

TEST(PeerSsdEviction, InFlightReadIsNotEvicted) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  const std::string val = "payload-payload";
  ASSERT_TRUE(Put(*h.mgr, "K", val));

  h.backend->BlockReads();
  std::vector<char> buf(val.size());
  SsdReadOutcome out{};
  std::thread reader([&] { out = h.mgr->PrepareRead("K", buf.data(), buf.size()); });
  h.backend->WaitReadsStarted(1);  // PrepareRead has marked K in-flight and is blocked in the read

  // Eviction must skip a key that is being read, and must not offer it.
  EXPECT_FALSE(h.mgr->Evict("K"));
  EXPECT_TRUE(h.mgr->EvictionCandidates(100).empty());
  EXPECT_TRUE(h.mgr->Exists("K"));

  h.backend->UnblockReads();
  reader.join();
  EXPECT_EQ(out.status, SsdReadStatus::kOk);
  EXPECT_EQ(std::string(buf.data(), out.size), val);

  // Once the read finished, the key can be evicted.
  EXPECT_TRUE(h.mgr->Evict("K"));
  EXPECT_FALSE(h.mgr->Exists("K"));
}

TEST(PeerSsdEviction, StaleRouteReadAfterEvictIsNotFound) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "K", "data"));
  ASSERT_TRUE(h.mgr->Evict("K"));

  std::vector<char> buf(4);
  auto out = h.mgr->PrepareRead("K", buf.data(), buf.size());
  EXPECT_EQ(out.status, SsdReadStatus::kNotFound);
}

TEST(PeerSsdEviction, DuplicateWriteIsIdempotent) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "K", "data"));
  ASSERT_TRUE(Put(*h.mgr, "K", "data"));  // same content-addressed key

  EXPECT_EQ(h.backend->write_calls(), 1);  // no second backend write
  auto events = h.mgr->DrainPendingEvents();
  EXPECT_EQ(CountKind(events, KvEvent::Kind::ADD), 1);  // no duplicate ADD SSD
}

TEST(PeerSsdEviction, BackendEvictFailureKeepsMetadata) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "K", "data"));
  h.mgr->DrainPendingEvents();  // discard ADD

  h.backend->SetFailEvict(true);
  EXPECT_FALSE(h.mgr->Evict("K"));
  EXPECT_TRUE(h.mgr->Exists("K"));                   // kept for retry
  EXPECT_TRUE(h.mgr->DrainPendingEvents().empty());  // no REMOVE emitted

  h.backend->SetFailEvict(false);
  EXPECT_TRUE(h.mgr->Evict("K"));  // retry succeeds
  EXPECT_FALSE(h.mgr->Exists("K"));
  auto events = h.mgr->DrainPendingEvents();
  EXPECT_EQ(CountKind(events, KvEvent::Kind::REMOVE), 1);
}

TEST(PeerSsdEviction, ConcurrentEvictOfSameKeyRemovesOnce) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "K", "data"));
  h.mgr->DrainPendingEvents();

  std::atomic<int> wins{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      if (h.mgr->Evict("K")) wins.fetch_add(1);
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_EQ(wins.load(), 1);  // exactly one evictor wins
  EXPECT_FALSE(h.mgr->Exists("K"));
  auto events = h.mgr->DrainPendingEvents();
  EXPECT_EQ(CountKind(events, KvEvent::Kind::REMOVE), 1);  // no double REMOVE
}

TEST(PeerSsdEviction, ClearLocalWipesPhysicalBytes) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "a", "1111"));
  ASSERT_TRUE(Put(*h.mgr, "b", "2222"));

  h.mgr->ClearLocal();

  EXPECT_EQ(h.backend->clear_calls(), 1);  // physical wipe happened
  EXPECT_FALSE(h.mgr->Exists("a"));
  EXPECT_FALSE(h.mgr->Exists("b"));
  EXPECT_TRUE(h.mgr->SnapshotOwnedKeys().empty());
  auto [used, total] = h.mgr->Capacity();
  EXPECT_EQ(used, 0u);
}

TEST(PeerSsdEviction, ClearLocalWaitsForInFlightRead) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  const std::string val = "read-priority";
  ASSERT_TRUE(Put(*h.mgr, "K", val));

  h.backend->BlockReads();
  std::vector<char> buf(val.size());
  SsdReadOutcome out{};
  std::thread reader([&] { out = h.mgr->PrepareRead("K", buf.data(), buf.size()); });
  h.backend->WaitReadsStarted(1);

  std::thread clearer([&] { h.mgr->ClearLocal(); });

  // The read is in flight; let it complete, then ClearLocal may wipe.  If
  // ClearLocal had wiped the backend before the read finished, the read would
  // return kError instead of the correct bytes — so kOk proves read priority.
  h.backend->UnblockReads();
  reader.join();
  clearer.join();

  EXPECT_EQ(out.status, SsdReadStatus::kOk);
  EXPECT_EQ(std::string(buf.data(), out.size), val);
  EXPECT_EQ(h.backend->clear_calls(), 1);
  EXPECT_FALSE(h.mgr->Exists("K"));
}

TEST(PeerSsdEviction, EvictionCandidatesBoundaries) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(Put(*h.mgr, "K", "data"));

  EXPECT_TRUE(h.mgr->EvictionCandidates(0).empty());  // nothing asked for

  // All candidates in flight -> none offered, no spin.
  h.backend->BlockReads();
  std::vector<char> buf(4);
  SsdReadOutcome out{};
  std::thread reader([&] { out = h.mgr->PrepareRead("K", buf.data(), buf.size()); });
  h.backend->WaitReadsStarted(1);
  EXPECT_TRUE(h.mgr->EvictionCandidates(100).empty());
  h.backend->UnblockReads();
  reader.join();
  EXPECT_EQ(out.status, SsdReadStatus::kOk);
}

TEST(PeerSsdEviction, DisabledManagerIsInert) {
  PeerSsdConfig cfg;
  cfg.enabled = false;
  PeerSsdManager mgr(cfg);

  EXPECT_FALSE(Put(mgr, "K", "data"));
  EXPECT_FALSE(mgr.Evict("K"));
  EXPECT_TRUE(mgr.EvictionCandidates(100).empty());
  std::vector<char> buf(4);
  EXPECT_EQ(mgr.PrepareRead("K", buf.data(), buf.size()).status, SsdReadStatus::kNotFound);
  mgr.ClearLocal();  // no backend -> no crash, no-op
  EXPECT_TRUE(mgr.SnapshotOwnedKeys().empty());
}

}  // namespace
}  // namespace mori::umbp

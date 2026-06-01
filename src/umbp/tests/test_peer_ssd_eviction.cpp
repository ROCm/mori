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
// Phase 4: SSD local capacity management + eviction.  Drives PeerSsdManager
// through a controllable in-memory TierBackend (the test-only constructor) so
// LRU ordering, watermark eviction, the in-flight-read guard, idempotent Write,
// backend-evict failure, concurrent eviction, and physical Clear are all
// deterministic without real disk IO.
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

#include "umbp/distributed/peer/peer_ssd_manager.h"
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

std::vector<std::pair<const void*, size_t>> OneSeg(const std::string& s) {
  return {{s.data(), s.size()}};
}

// Manager owning a FakeBackend we keep a raw pointer to for test inspection.
struct Harness {
  FakeBackend* backend;
  std::unique_ptr<PeerSsdManager> mgr;
};

Harness MakeHarness(size_t capacity, double high = 0.9, double low = 0.7) {
  auto be = std::make_unique<FakeBackend>(capacity);
  FakeBackend* raw = be.get();
  return Harness{raw, std::make_unique<PeerSsdManager>(std::move(be), high, low)};
}

int CountKind(const std::vector<KvEvent>& events, KvEvent::Kind kind) {
  int n = 0;
  for (const auto& e : events) {
    if (e.kind == kind && e.tier == TierType::SSD) ++n;
  }
  return n;
}

bool HasRemove(const std::vector<KvEvent>& events, const std::string& key) {
  for (const auto& e : events) {
    if (e.kind == KvEvent::Kind::REMOVE && e.tier == TierType::SSD && e.key == key) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------

TEST(PeerSsdEviction, WriteAndPrepareReadRefreshLru) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(h.mgr->Write("A", OneSeg("aaaa"), 4));
  ASSERT_TRUE(h.mgr->Write("B", OneSeg("bbbb"), 4));
  ASSERT_TRUE(h.mgr->Write("C", OneSeg("cccc"), 4));

  // LRU now (oldest->newest): A, B, C.  Reading A promotes it to MRU, so the
  // oldest becomes B and SelectVictims must pick B first (not the just-read A).
  std::vector<char> buf(4);
  auto out = h.mgr->PrepareRead("A", buf.data(), buf.size());
  ASSERT_EQ(out.status, SsdReadStatus::kOk);
  EXPECT_EQ(std::string(buf.data(), out.size), "aaaa");

  auto victims = h.mgr->SelectVictims(/*bytes_to_free=*/1);
  ASSERT_FALSE(victims.empty());
  EXPECT_EQ(victims.front(), "B");
  EXPECT_NE(victims.front(), "A");
}

TEST(PeerSsdEviction, WatermarkTriggersEvictionDownToLow) {
  // capacity 1000, high 0.9 (=>900), low 0.7 (=>700); 100-byte values.
  auto h = MakeHarness(/*capacity=*/1000, /*high=*/0.9, /*low=*/0.7);
  std::string val(100, 'x');
  for (int i = 1; i <= 9; ++i) {
    ASSERT_TRUE(h.mgr->Write("k" + std::to_string(i), OneSeg(val), val.size()));
  }
  // After k9: used hit 900 >= high -> evict oldest down to <= 700.
  auto [used, total] = h.mgr->Capacity();
  EXPECT_EQ(total, 1000u);
  EXPECT_LE(used, 700u);

  // Oldest (k1, k2) evicted first; newest still present.
  EXPECT_FALSE(h.mgr->Exists("k1"));
  EXPECT_FALSE(h.mgr->Exists("k2"));
  EXPECT_TRUE(h.mgr->Exists("k9"));

  auto events = h.mgr->DrainPendingEvents();
  EXPECT_TRUE(HasRemove(events, "k1"));
  EXPECT_TRUE(HasRemove(events, "k2"));
  EXPECT_EQ(CountKind(events, KvEvent::Kind::REMOVE), 2);
}

TEST(PeerSsdEviction, EnospcTriggersEvictThenRetry) {
  // Fill to 800/1000 (below the 0.9 high watermark, so no watermark eviction
  // fires during the fill), then write a 300-byte value that overflows the
  // device: backend Write -> ENOSPC -> one evict round (frees the oldest down
  // to the 0.5 low watermark) -> retry succeeds.  After the retry used is 800,
  // still below high, so no second round disturbs the just-written key.
  auto h = MakeHarness(/*capacity=*/1000, /*high=*/0.9, /*low=*/0.5);
  for (int i = 1; i <= 8; ++i) {
    ASSERT_TRUE(h.mgr->Write("k" + std::to_string(i), OneSeg(std::string(100, 'a')), 100));
  }
  ASSERT_TRUE(h.mgr->Write("big", OneSeg(std::string(300, 'c')), 300));
  EXPECT_TRUE(h.mgr->Exists("big"));
  EXPECT_FALSE(h.mgr->Exists("k1"));  // oldest reclaimed to make room
  EXPECT_LE(h.mgr->Capacity().first, 1000u);
}

TEST(PeerSsdEviction, InFlightReadIsNotEvicted) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  const std::string val = "payload-payload";
  ASSERT_TRUE(h.mgr->Write("K", OneSeg(val), val.size()));

  h.backend->BlockReads();
  std::vector<char> buf(val.size());
  SsdReadOutcome out{};
  std::thread reader([&] { out = h.mgr->PrepareRead("K", buf.data(), buf.size()); });
  h.backend->WaitReadsStarted(1);  // PrepareRead has marked K in-flight and is blocked in the read

  // Eviction must skip a key that is being read.
  EXPECT_FALSE(h.mgr->Evict("K"));
  EXPECT_TRUE(h.mgr->SelectVictims(1'000'000).empty());
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
  ASSERT_TRUE(h.mgr->Write("K", OneSeg("data"), 4));
  ASSERT_TRUE(h.mgr->Evict("K"));

  std::vector<char> buf(4);
  auto out = h.mgr->PrepareRead("K", buf.data(), buf.size());
  EXPECT_EQ(out.status, SsdReadStatus::kNotFound);
}

TEST(PeerSsdEviction, DuplicateWriteIsIdempotent) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(h.mgr->Write("K", OneSeg("data"), 4));
  ASSERT_TRUE(h.mgr->Write("K", OneSeg("data"), 4));  // same content-addressed key

  EXPECT_EQ(h.backend->write_calls(), 1);  // no second backend write
  auto events = h.mgr->DrainPendingEvents();
  EXPECT_EQ(CountKind(events, KvEvent::Kind::ADD), 1);  // no duplicate ADD SSD
}

TEST(PeerSsdEviction, BackendEvictFailureKeepsMetadata) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(h.mgr->Write("K", OneSeg("data"), 4));
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
  ASSERT_TRUE(h.mgr->Write("K", OneSeg("data"), 4));
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
  ASSERT_TRUE(h.mgr->Write("a", OneSeg("1111"), 4));
  ASSERT_TRUE(h.mgr->Write("b", OneSeg("2222"), 4));

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
  ASSERT_TRUE(h.mgr->Write("K", OneSeg(val), val.size()));

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

TEST(PeerSsdEviction, InvalidWatermarksThrow) {
  // low >= high, and high > 1 are both rejected (fail-fast, no silent clamp).
  EXPECT_THROW(PeerSsdManager(std::make_unique<FakeBackend>(1000), 0.5, 0.7), std::runtime_error);
  EXPECT_THROW(PeerSsdManager(std::make_unique<FakeBackend>(1000), 1.5, 0.7), std::runtime_error);
  EXPECT_THROW(PeerSsdManager(std::make_unique<FakeBackend>(1000), 0.9, 0.0), std::runtime_error);
}

TEST(PeerSsdEviction, SelectVictimsBoundaries) {
  auto h = MakeHarness(/*capacity=*/1'000'000);
  ASSERT_TRUE(h.mgr->Write("K", OneSeg("data"), 4));

  EXPECT_TRUE(h.mgr->SelectVictims(0).empty());  // nothing to free

  // All candidates in flight -> no victim, no spin.
  h.backend->BlockReads();
  std::vector<char> buf(4);
  SsdReadOutcome out{};
  std::thread reader([&] { out = h.mgr->PrepareRead("K", buf.data(), buf.size()); });
  h.backend->WaitReadsStarted(1);
  EXPECT_TRUE(h.mgr->SelectVictims(1'000'000).empty());
  h.backend->UnblockReads();
  reader.join();
  EXPECT_EQ(out.status, SsdReadStatus::kOk);
}

TEST(PeerSsdEviction, DisabledManagerIsInert) {
  PeerSsdConfig cfg;
  cfg.enabled = false;
  PeerSsdManager mgr(cfg);

  EXPECT_FALSE(mgr.Write("K", OneSeg("data"), 4));
  EXPECT_FALSE(mgr.Evict("K"));
  EXPECT_TRUE(mgr.SelectVictims(100).empty());
  std::vector<char> buf(4);
  EXPECT_EQ(mgr.PrepareRead("K", buf.data(), buf.size()).status, SsdReadStatus::kNotFound);
  mgr.ClearLocal();  // no backend -> no crash, no-op
  EXPECT_TRUE(mgr.SnapshotOwnedKeys().empty());
}

}  // namespace
}  // namespace mori::umbp

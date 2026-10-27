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

// Single-flight read coalescing in PeerSsdManager.
//
// Motivation: with MLA + TP every attention-TP rank GETs a byte-identical key,
// so one logical page was being read from the drive tp_size times over.  A
// production TP8 trace measured 85.6% of reads redundant, and per-key device
// latency rising ~linearly with same-key fan-out (3.3ms solo -> 27.7ms at 6
// concurrent).  These tests pin the behaviour that removes it.

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
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

constexpr int kFanout = 8;  // TP8: one leader + 7 followers

// In-memory backend whose reads can be held open, so every reader is guaranteed
// to be inside the merge window at the same time.
class GatedBackend : public TierBackend {
 public:
  explicit GatedBackend(size_t capacity)
      : TierBackend(StorageTier::LOCAL_SSD), capacity_(capacity) {}

  bool Write(const std::string& key, const void* data, size_t size) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (used_ + size > capacity_) return false;
    store_[key].assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
    used_ += size;
    return true;
  }

  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) override {
    {
      std::unique_lock<std::mutex> lk(gate_mu_);
      ++reads_started_;
      started_cv_.notify_all();
      gate_cv_.wait(lk, [this] { return !blocked_; });
      if (fail_) return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    if (it == store_.end() || it->second.size() != size) return false;
    std::memcpy(reinterpret_cast<void*>(dst), it->second.data(), size);
    return true;
  }

  // Extent reads, gated like ReadIntoPtr: one gate pass per key.
  TierCapabilities Capabilities() const override {
    TierCapabilities caps;
    caps.ranged_read = true;
    return caps;
  }
  std::vector<bool> ReadBatchRangesIntoPtr(
      const std::vector<std::string>& keys, const std::vector<std::vector<uintptr_t>>& dst_ptrs,
      const std::vector<std::vector<size_t>>& sizes,
      const std::vector<std::vector<size_t>>& src_offsets) override {
    std::vector<bool> out(keys.size(), false);
    for (size_t k = 0; k < keys.size(); ++k) {
      {
        std::unique_lock<std::mutex> lk(gate_mu_);
        ++reads_started_;
        started_cv_.notify_all();
        gate_cv_.wait(lk, [this] { return !blocked_; });
        if (fail_) continue;
      }
      std::lock_guard<std::mutex> lk(mu_);
      auto it = store_.find(keys[k]);
      if (it == store_.end()) continue;
      for (size_t r = 0; r < dst_ptrs[k].size(); ++r) {
        std::memcpy(reinterpret_cast<void*>(dst_ptrs[k][r]), it->second.data() + src_offsets[k][r],
                    sizes[k][r]);
        bytes_read_ += sizes[k][r];
      }
      out[k] = true;
    }
    return out;
  }
  size_t bytes_read() const {
    std::lock_guard<std::mutex> lk(mu_);
    return bytes_read_;
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

  void BlockReads() {
    std::lock_guard<std::mutex> lk(gate_mu_);
    blocked_ = true;
  }
  void UnblockReads() {
    {
      std::lock_guard<std::mutex> lk(gate_mu_);
      blocked_ = false;
    }
    gate_cv_.notify_all();
  }
  void SetFailReads(bool f) {
    std::lock_guard<std::mutex> lk(gate_mu_);
    fail_ = f;
  }
  void WaitReadsStarted(int n) {
    std::unique_lock<std::mutex> lk(gate_mu_);
    started_cv_.wait(lk, [&] { return reads_started_ >= n; });
  }
  int reads_started() const {
    std::lock_guard<std::mutex> lk(gate_mu_);
    return reads_started_;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::vector<char>> store_;
  size_t used_ = 0;
  size_t bytes_read_ = 0;  // ranged reads only
  size_t capacity_;

  mutable std::mutex gate_mu_;
  std::condition_variable gate_cv_;
  std::condition_variable started_cv_;
  bool blocked_ = false;
  bool fail_ = false;
  int reads_started_ = 0;
};

struct Harness {
  GatedBackend* backend;
  std::unique_ptr<PeerSsdManager> mgr;
};

Harness MakeHarness(bool single_flight = true) {
  auto be = std::make_unique<GatedBackend>(1'000'000);
  GatedBackend* raw = be.get();
  return Harness{raw, std::make_unique<PeerSsdManager>(std::move(be), single_flight)};
}

bool Put(PeerSsdManager& mgr, const std::string& key, const std::string& value) {
  return mgr.WriteBatch({key}, {value.data()}, {value.size()})[0];
}

// Spin until `n` requesters have found a read already in flight.  Deterministic
// stand-in for "all followers have attached" — polling a public counter beats
// sleeping, which would make the merge window a race.
void WaitForDups(const PeerSsdManager& mgr, uint64_t n) {
  while (mgr.ReadDup() < n) std::this_thread::yield();
}

// All fan-out readers of one key collapse to a single device read, and every
// one of them still receives the correct bytes.
TEST(PeerSsdSingleFlight, ConcurrentSameKeyReadsIssueOneDeviceRead) {
  auto h = MakeHarness();
  const std::string payload = "kv-page-payload";
  ASSERT_TRUE(Put(*h.mgr, "K", payload));

  h.backend->BlockReads();
  std::vector<std::vector<char>> bufs(kFanout, std::vector<char>(payload.size(), '\0'));
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back(
        [&, i] { outs[i] = h.mgr->PrepareRead("K", bufs[i].data(), bufs[i].size()); });
  }
  h.backend->WaitReadsStarted(1);    // leader is on the device
  WaitForDups(*h.mgr, kFanout - 1);  // all 7 followers have attached
  h.backend->UnblockReads();
  for (auto& t : readers) t.join();

  EXPECT_EQ(h.backend->reads_started(), 1) << "followers must not touch the drive";
  EXPECT_EQ(h.mgr->ReadMerged(), static_cast<uint64_t>(kFanout - 1));
  for (int i = 0; i < kFanout; ++i) {
    EXPECT_EQ(outs[i].status, SsdReadStatus::kOk) << "reader " << i;
    EXPECT_EQ(std::string(bufs[i].data(), outs[i].size), payload) << "reader " << i;
  }
}

// With coalescing off every requester reads the drive — the pre-single-flight
// behaviour, and the A/B baseline the UMBP_SSD_SINGLE_FLIGHT knob selects.
TEST(PeerSsdSingleFlight, DisabledIssuesOneDeviceReadPerRequester) {
  auto h = MakeHarness(/*single_flight=*/false);
  const std::string payload = "kv-page-payload";
  ASSERT_TRUE(Put(*h.mgr, "K", payload));

  h.backend->BlockReads();
  std::vector<std::vector<char>> bufs(kFanout, std::vector<char>(payload.size(), '\0'));
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back(
        [&, i] { outs[i] = h.mgr->PrepareRead("K", bufs[i].data(), bufs[i].size()); });
  }
  h.backend->WaitReadsStarted(kFanout);  // all of them reach the device
  h.backend->UnblockReads();
  for (auto& t : readers) t.join();

  EXPECT_EQ(h.backend->reads_started(), kFanout);
  EXPECT_EQ(h.mgr->ReadMerged(), 0u);
  for (int i = 0; i < kFanout; ++i) {
    EXPECT_EQ(outs[i].status, SsdReadStatus::kOk);
    EXPECT_EQ(std::string(bufs[i].data(), outs[i].size), payload);
  }
}

// A leader whose read fails must release its followers with an error rather
// than leaving them to hit the 30s timeout.
TEST(PeerSsdSingleFlight, LeaderFailureReleasesFollowersWithError) {
  auto h = MakeHarness();
  const std::string payload = "kv-page-payload";
  ASSERT_TRUE(Put(*h.mgr, "K", payload));

  h.backend->BlockReads();
  h.backend->SetFailReads(true);
  std::vector<std::vector<char>> bufs(kFanout, std::vector<char>(payload.size(), '\0'));
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back(
        [&, i] { outs[i] = h.mgr->PrepareRead("K", bufs[i].data(), bufs[i].size()); });
  }
  h.backend->WaitReadsStarted(1);
  WaitForDups(*h.mgr, kFanout - 1);
  h.backend->UnblockReads();
  for (auto& t : readers) t.join();  // hangs here if the failure is not published

  for (int i = 0; i < kFanout; ++i) {
    EXPECT_EQ(outs[i].status, SsdReadStatus::kError) << "reader " << i;
  }
}

// Distinct keys share no episode, so nothing merges — this is the MHA case,
// where keys carry a per-rank suffix and there is no redundancy to remove.
TEST(PeerSsdSingleFlight, DistinctKeysDoNotMerge) {
  auto h = MakeHarness();
  for (int i = 0; i < kFanout; ++i) {
    ASSERT_TRUE(Put(*h.mgr, "K" + std::to_string(i), "payload!"));
  }

  h.backend->BlockReads();
  std::vector<std::vector<char>> bufs(kFanout, std::vector<char>(8, '\0'));
  std::vector<SsdReadOutcome> outs(kFanout);
  std::vector<std::thread> readers;
  for (int i = 0; i < kFanout; ++i) {
    readers.emplace_back([&, i] {
      outs[i] = h.mgr->PrepareRead("K" + std::to_string(i), bufs[i].data(), bufs[i].size());
    });
  }
  h.backend->WaitReadsStarted(kFanout);
  h.backend->UnblockReads();
  for (auto& t : readers) t.join();

  EXPECT_EQ(h.backend->reads_started(), kFanout);
  EXPECT_EQ(h.mgr->ReadMerged(), 0u);
  for (int i = 0; i < kFanout; ++i) {
    EXPECT_EQ(outs[i].status, SsdReadStatus::kOk);
    EXPECT_EQ(std::string(bufs[i].data(), outs[i].size), "payload!");
  }
}

// Back-to-back rounds on one key: a follower that has not yet woken when the
// next leader claims the key must still report its own leader's result.  This
// is the generation hazard that per-episode state exists to prevent.
TEST(PeerSsdSingleFlight, SequentialRoundsStayIndependent) {
  auto h = MakeHarness();
  const std::string payload = "kv-page-payload";
  ASSERT_TRUE(Put(*h.mgr, "K", payload));

  for (int round = 0; round < 20; ++round) {
    h.backend->BlockReads();
    std::vector<std::vector<char>> bufs(kFanout, std::vector<char>(payload.size(), '\0'));
    std::vector<SsdReadOutcome> outs(kFanout);
    std::vector<std::thread> readers;
    for (int i = 0; i < kFanout; ++i) {
      readers.emplace_back(
          [&, i] { outs[i] = h.mgr->PrepareRead("K", bufs[i].data(), bufs[i].size()); });
    }
    h.backend->WaitReadsStarted(round + 1);  // reads_started() is cumulative
    h.backend->UnblockReads();
    for (auto& t : readers) t.join();
    for (int i = 0; i < kFanout; ++i) {
      EXPECT_EQ(outs[i].status, SsdReadStatus::kOk) << "round " << round << " reader " << i;
      EXPECT_EQ(std::string(bufs[i].data(), outs[i].size), payload);
    }
  }
}

// ---------------------------------------------------------------------------
//  Extent reads (design-ssd-ranged-io.md D1c): a reader joins an in-flight read
//  only when that read covers its bytes.
// ---------------------------------------------------------------------------

const std::string kObject = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-";

SsdReadOutcome ReadExtent(PeerSsdManager& mgr, const std::string& key, uint64_t offset,
                          std::vector<char>* buf) {
  const std::vector<ObjectExtent> extent = {{offset, buf->size()}};
  return mgr.PrepareReadBatch({key}, {buf->data()}, {buf->size()}, &extent)[0];
}

// Holds `first` on the drive, starts `second`, and releases both once `second`
// has attached to it (`second_follows`) or reached the drive itself.
void RaceTwoReads(Harness& h, const std::function<void()>& first,
                  const std::function<void()>& second, bool second_follows) {
  h.backend->BlockReads();
  std::thread a(first);
  h.backend->WaitReadsStarted(1);
  std::thread b(second);
  if (second_follows) {
    WaitForDups(*h.mgr, 1);
  } else {
    h.backend->WaitReadsStarted(2);
  }
  h.backend->UnblockReads();
  a.join();
  b.join();
}

// Attaching on the key alone would hand the second reader the first one's bytes.
TEST(PeerSsdSingleFlight, DisjointExtentsOfOneKeyEachGetTheirOwnBytes) {
  auto h = MakeHarness();
  ASSERT_TRUE(Put(*h.mgr, "K", kObject));
  std::vector<char> front(16), back(16);
  SsdReadOutcome out_front, out_back;
  RaceTwoReads(
      h, [&] { out_front = ReadExtent(*h.mgr, "K", 0, &front); },
      [&] { out_back = ReadExtent(*h.mgr, "K", 40, &back); }, /*second_follows=*/false);

  EXPECT_EQ(h.mgr->ReadMerged(), 0u);
  ASSERT_EQ(out_front.status, SsdReadStatus::kOk);
  ASSERT_EQ(out_back.status, SsdReadStatus::kOk);
  EXPECT_EQ(out_front.size, kObject.size()) << "outcome size is the object's, not the extent's";
  EXPECT_EQ(std::string(front.begin(), front.end()), kObject.substr(0, 16));
  EXPECT_EQ(std::string(back.begin(), back.end()), kObject.substr(40, 16));
  EXPECT_EQ(h.backend->bytes_read(), 32u) << "only the two extents came off the drive";
}

TEST(PeerSsdSingleFlight, ExtentInsideAWholeReadFollowsAndTakesItsSlice) {
  auto h = MakeHarness();
  ASSERT_TRUE(Put(*h.mgr, "K", kObject));
  std::vector<char> whole(kObject.size()), slice(20);
  SsdReadOutcome out_whole, out_slice;
  RaceTwoReads(
      h, [&] { out_whole = h.mgr->PrepareRead("K", whole.data(), whole.size()); },
      [&] { out_slice = ReadExtent(*h.mgr, "K", 23, &slice); }, /*second_follows=*/true);

  EXPECT_EQ(h.backend->reads_started(), 1) << "the extent reader must not touch the drive";
  EXPECT_EQ(h.mgr->ReadMerged(), 1u);
  ASSERT_EQ(out_whole.status, SsdReadStatus::kOk);
  ASSERT_EQ(out_slice.status, SsdReadStatus::kOk);
  EXPECT_EQ(std::string(whole.begin(), whole.end()), kObject);
  EXPECT_EQ(std::string(slice.begin(), slice.end()), kObject.substr(23, 20));
}

TEST(PeerSsdSingleFlight, WholeReadDoesNotFollowAnExtentLeader) {
  auto h = MakeHarness();
  ASSERT_TRUE(Put(*h.mgr, "K", kObject));
  std::vector<char> slice(8), whole(kObject.size());
  SsdReadOutcome out_slice, out_whole;
  RaceTwoReads(
      h, [&] { out_slice = ReadExtent(*h.mgr, "K", 8, &slice); },
      [&] { out_whole = h.mgr->PrepareRead("K", whole.data(), whole.size()); },
      /*second_follows=*/false);

  EXPECT_EQ(h.mgr->ReadMerged(), 0u);
  ASSERT_EQ(out_slice.status, SsdReadStatus::kOk);
  ASSERT_EQ(out_whole.status, SsdReadStatus::kOk);
  EXPECT_EQ(std::string(slice.begin(), slice.end()), kObject.substr(8, 8));
  EXPECT_EQ(std::string(whole.begin(), whole.end()), kObject);
}

TEST(PeerSsdSingleFlight, ExtentPastTheObjectIsAnError) {
  auto h = MakeHarness();
  ASSERT_TRUE(Put(*h.mgr, "K", kObject));
  std::vector<char> buf(16);
  EXPECT_EQ(ReadExtent(*h.mgr, "K", kObject.size() - 8, &buf).status, SsdReadStatus::kError);
  EXPECT_EQ(h.backend->reads_started(), 0);
}

}  // namespace
}  // namespace mori::umbp

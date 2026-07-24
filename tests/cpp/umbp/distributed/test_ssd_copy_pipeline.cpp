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
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp/distributed/peer/ssd_copy_pipeline.h"
#include "umbp/local/block_index/local_block_index.h"
#include "umbp/local/tiers/copy_pipeline.h"
#include "umbp/local/tiers/local_storage_manager.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

constexpr uint64_t kPageSize = 1024;

// Concatenate a pin's segments into one buffer for content comparison.
std::string Concat(const PeerDramAllocator::DramCopyPin& pin) {
  std::string out;
  for (const auto& [ptr, len] : pin.segments) {
    out.append(static_cast<const char*>(ptr), len);
  }
  return out;
}

// ---- DramCopyPin unit tests (direct on PeerDramAllocator) -------------------

class DramCopyPinTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backing_.assign(kPageSize * 8, 0);
    PeerDramAllocator::TierConfig dram;
    dram.buffer_sizes = {kPageSize * 8};
    dram.buffer_descs = {{0x01, 0x02}};
    dram.buffer_bases = {backing_.data()};
    dram_ = std::make_unique<PeerDramAllocator>(kPageSize, std::move(dram),
                                                PeerDramAllocator::TierConfig{},
                                                /*pending_ttl=*/std::chrono::milliseconds{5000},
                                                /*read_lease_ttl=*/std::chrono::milliseconds{0});
  }

  // Allocate, write `value` into its pages, commit.  Backing memory is owned
  // by the test, so we resolve pages -> offset exactly like the real writer.
  void PutLocal(const std::string& key, const std::string& value) {
    auto res = dram_->Allocate(key, value.size(), TierType::DRAM);
    ASSERT_EQ(res.outcome, PeerDramAllocator::Outcome::kSuccessAllocated);
    const auto& slot = *res.slot;
    size_t off = 0;
    for (const auto& p : slot.pages) {
      const size_t bytes = std::min<size_t>(kPageSize, value.size() - off);
      std::memcpy(backing_.data() + static_cast<size_t>(p.page_index) * kPageSize,
                  value.data() + off, bytes);
      off += bytes;
    }
    uint64_t committed = 0;
    ASSERT_TRUE(dram_->Commit(slot.slot_id, key, committed));
    ASSERT_EQ(committed, value.size());
  }

  std::vector<char> backing_;
  std::unique_ptr<PeerDramAllocator> dram_;
};

TEST_F(DramCopyPinTest, AcquireResolvesSegmentsToCommittedBytes) {
  const std::string value(kPageSize + 17, 'Z');  // spans 2 pages, last partial
  PutLocal("k", value);

  auto pin = dram_->AcquireDramCopyPin("k");
  ASSERT_TRUE(pin.has_value());
  EXPECT_EQ(pin->total_size, value.size());
  EXPECT_EQ(Concat(*pin), value);
  dram_->ReleaseDramCopyPin("k", pin->pin_token);
}

TEST_F(DramCopyPinTest, AcquireMissingKeyReturnsNullopt) {
  EXPECT_FALSE(dram_->AcquireDramCopyPin("never").has_value());
}

TEST_F(DramCopyPinTest, DuplicatePinReturnsNullopt) {
  PutLocal("k", "payload");
  auto first = dram_->AcquireDramCopyPin("k");
  ASSERT_TRUE(first.has_value());
  EXPECT_FALSE(dram_->AcquireDramCopyPin("k").has_value());  // already pinned
  dram_->ReleaseDramCopyPin("k", first->pin_token);
}

TEST_F(DramCopyPinTest, EvictBlockedWhilePinnedThenAllowedAfterRelease) {
  PutLocal("k", "payload");
  dram_->DrainPendingEvents();  // discard the commit's ADD DRAM event
  auto pin = dram_->AcquireDramCopyPin("k");
  ASSERT_TRUE(pin.has_value());

  // Pinned: Evict must not free, not emit REMOVE, keep ownership.
  auto evicted = dram_->Evict({"k"});
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0].bytes_freed, 0u);
  EXPECT_TRUE(dram_->DrainPendingEvents().empty());  // no REMOVE DRAM
  ASSERT_EQ(dram_->SnapshotOwnedKeys().size(), 1u);  // still owned

  // Release -> next Evict frees and emits REMOVE.
  dram_->ReleaseDramCopyPin("k", pin->pin_token);
  auto evicted2 = dram_->Evict({"k"});
  ASSERT_EQ(evicted2.size(), 1u);
  EXPECT_GT(evicted2[0].bytes_freed, 0u);
  auto events = dram_->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::REMOVE);
  EXPECT_EQ(events[0].tier, TierType::DRAM);
}

TEST(DramCopyPinNonContiguous, SegmentsSpanMultipleBuffers) {
  // Two 1-page buffers force a cross-buffer page set for a 2-page key.
  std::vector<char> b0(kPageSize, 0), b1(kPageSize, 0);
  PeerDramAllocator::TierConfig dram;
  dram.buffer_sizes = {kPageSize, kPageSize};
  dram.buffer_descs = {{0x01}, {0x02}};
  dram.buffer_bases = {b0.data(), b1.data()};
  PeerDramAllocator alloc(kPageSize, std::move(dram), PeerDramAllocator::TierConfig{},
                          std::chrono::milliseconds{5000}, std::chrono::milliseconds{0});

  const std::string value(kPageSize + 5, 'Q');
  auto res = alloc.Allocate("k", value.size(), TierType::DRAM);
  ASSERT_EQ(res.outcome, PeerDramAllocator::Outcome::kSuccessAllocated);
  const auto& slot = *res.slot;
  ASSERT_EQ(slot.pages.size(), 2u);
  std::vector<char*> bases = {b0.data(), b1.data()};
  size_t off = 0;
  for (const auto& p : slot.pages) {
    const size_t bytes = std::min<size_t>(kPageSize, value.size() - off);
    std::memcpy(bases[p.buffer_index] + static_cast<size_t>(p.page_index) * kPageSize,
                value.data() + off, bytes);
    off += bytes;
  }
  uint64_t committed = 0;
  ASSERT_TRUE(alloc.Commit(slot.slot_id, "k", committed));

  auto pin = alloc.AcquireDramCopyPin("k");
  ASSERT_TRUE(pin.has_value());
  EXPECT_EQ(pin->segments.size(), 2u);
  EXPECT_EQ(Concat(*pin), value);
  alloc.ReleaseDramCopyPin("k", pin->pin_token);
}

// ---- Pipeline integration tests (allocator + SSD manager + pipeline) --------

class SsdCopyPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = fs::temp_directory_path() / ("umbp_copy_test_" + std::to_string(::getpid()) + "_" +
                                        std::to_string(counter.fetch_add(1)));
    fs::remove_all(dir_);

    backing_.assign(kPageSize * 16, 0);
    PeerDramAllocator::TierConfig dram;
    dram.buffer_sizes = {kPageSize * 16};
    dram.buffer_descs = {{0x01}};
    dram.buffer_bases = {backing_.data()};
    dram_ = std::make_unique<PeerDramAllocator>(
        kPageSize, std::move(dram), PeerDramAllocator::TierConfig{},
        std::chrono::milliseconds{5000}, std::chrono::milliseconds{0});

    PeerSsdConfig ssd_cfg;
    ssd_cfg.enabled = true;
    ssd_cfg.ssd.enabled = true;
    ssd_cfg.ssd.storage_dir = dir_.string();
    ssd_cfg.ssd.capacity_bytes = 64ULL * 1024 * 1024;
    ssd_cfg.ssd.io.backend = UMBPIoBackend::Posix;  // avoid io_uring container flakiness
    ssd_ = std::make_unique<PeerSsdManager>(ssd_cfg);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  void PutLocal(const std::string& key, const std::string& value) {
    auto res = dram_->Allocate(key, value.size(), TierType::DRAM);
    ASSERT_EQ(res.outcome, PeerDramAllocator::Outcome::kSuccessAllocated);
    const auto& slot = *res.slot;
    size_t off = 0;
    for (const auto& p : slot.pages) {
      const size_t bytes = std::min<size_t>(kPageSize, value.size() - off);
      std::memcpy(backing_.data() + static_cast<size_t>(p.page_index) * kPageSize,
                  value.data() + off, bytes);
      off += bytes;
    }
    uint64_t committed = 0;
    ASSERT_TRUE(dram_->Commit(slot.slot_id, key, committed));
  }

  bool WaitForSsd(const std::string& key, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (ssd_->Exists(key)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return ssd_->Exists(key);
  }

  fs::path dir_;
  std::vector<char> backing_;
  std::unique_ptr<PeerDramAllocator> dram_;
  std::unique_ptr<PeerSsdManager> ssd_;
};

TEST_F(SsdCopyPipelineTest, CommitCopiesToSsdAndEmitsAddEvent) {
  SsdCopyPipeline pipeline(dram_.get(), ssd_.get());
  pipeline.Start();

  PutLocal("k", "hello-ssd-copy-on-commit");
  ASSERT_TRUE(pipeline.Enqueue(SsdCopyTask{"k", TierType::DRAM, 24}));

  ASSERT_TRUE(WaitForSsd("k", std::chrono::seconds(2)));
  EXPECT_GE(pipeline.CopiedOk(), 1u);
  EXPECT_GE(pipeline.Enqueued(), 1u);  // observability: task was accepted
  EXPECT_EQ(pipeline.Failed(), 0u);

  auto events = ssd_->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[0].tier, TierType::SSD);
  EXPECT_EQ(events[0].key, "k");

  pipeline.Stop();
}

TEST_F(SsdCopyPipelineTest, QueuedTaskForEvictedKeyIsDropped) {
  SsdCopyPipeline pipeline(dram_.get(), ssd_.get());

  PutLocal("gone", "data");
  // Evict before the copy ever runs (no pin held) -> key removed from owned_.
  auto ev = dram_->Evict({"gone"});
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_GT(ev[0].bytes_freed, 0u);

  // Now start draining: the worker's AcquireDramCopyPin returns nullopt -> drop.
  ASSERT_TRUE(pipeline.Enqueue(SsdCopyTask{"gone", TierType::DRAM, 4}));
  pipeline.Start();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(ssd_->Exists("gone"));
  EXPECT_EQ(pipeline.CopiedOk(), 0u);
  pipeline.Stop();
}

TEST_F(SsdCopyPipelineTest, FullQueueDropsWithoutBlocking) {
  // queue_depth=2, no workers started -> nothing drains, so the 3rd+ enqueue
  // overflows and is dropped (and returns immediately).
  SsdCopyPipeline pipeline(dram_.get(), ssd_.get(), /*queue_depth=*/2, /*workers=*/1);
  EXPECT_TRUE(pipeline.Enqueue(SsdCopyTask{"a", TierType::DRAM, 1}));
  EXPECT_TRUE(pipeline.Enqueue(SsdCopyTask{"b", TierType::DRAM, 1}));
  EXPECT_FALSE(pipeline.Enqueue(SsdCopyTask{"c", TierType::DRAM, 1}));  // full -> drop
  EXPECT_FALSE(pipeline.Enqueue(SsdCopyTask{"d", TierType::DRAM, 1}));
  EXPECT_EQ(pipeline.Dropped(), 2u);
  EXPECT_EQ(pipeline.Enqueued(), 2u);        // only the two accepted tasks
  EXPECT_EQ(pipeline.DroppedStopped(), 0u);  // these are queue-full, not stopped, drops
}

TEST_F(SsdCopyPipelineTest, EnqueueRejectedWhileStopped) {
  SsdCopyPipeline pipeline(dram_.get(), ssd_.get());
  pipeline.Start();
  pipeline.Stop();
  EXPECT_FALSE(pipeline.Enqueue(SsdCopyTask{"k", TierType::DRAM, 1}));
  EXPECT_EQ(pipeline.Dropped(), 0u);         // stopped path is not counted as a full-drop
  EXPECT_EQ(pipeline.DroppedStopped(), 1u);  // counted under the stopped reason instead
}

TEST_F(SsdCopyPipelineTest, StopAfterCopyIsCleanAndReleasesPin) {
  SsdCopyPipeline pipeline(dram_.get(), ssd_.get());
  pipeline.Start();

  const std::string value(8 * 1024, 'X');
  PutLocal("big", value);
  ASSERT_TRUE(pipeline.Enqueue(SsdCopyTask{"big", TierType::DRAM, value.size()}));

  // Let the copy run, then Stop().  Stop() joins the worker; the RAII pin guard
  // guarantees the pin is released before the worker exits (Stop() never
  // force-frees an in-flight pin — that join is the in-flight-wait guarantee).
  ASSERT_TRUE(WaitForSsd("big", std::chrono::seconds(2)));
  pipeline.Stop();

  EXPECT_EQ(pipeline.CopiedOk(), 1u);
  // Pin released -> the key is now evictable (no copy holding its pages).
  auto ev = dram_->Evict({"big"});
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_GT(ev[0].bytes_freed, 0u);
}

TEST_F(SsdCopyPipelineTest, QuiesceThenClearLeavesNoStaleSsdState) {
  SsdCopyPipeline pipeline(dram_.get(), ssd_.get());
  pipeline.Start();

  PutLocal("k", "payload");
  ASSERT_TRUE(pipeline.Enqueue(SsdCopyTask{"k", TierType::DRAM, 7}));
  ASSERT_TRUE(WaitForSsd("k", std::chrono::seconds(2)));

  // Clear path: quiesce (drain in-flight) then clear both tiers.
  pipeline.Quiesce();
  dram_->ClearLocal();
  ssd_->ClearLocal();
  pipeline.Resume();

  EXPECT_FALSE(ssd_->Exists("k"));
  EXPECT_TRUE(ssd_->SnapshotOwnedKeys().empty());
  EXPECT_TRUE(ssd_->DrainPendingEvents().empty());

  pipeline.Stop();
}

TEST(LocalCopyPipelineDrainTest, DrainWaitsForQueuedCopies) {
  UMBPConfig cfg;
  cfg.role = UMBPRole::SharedSSDLeader;
  cfg.force_ssd_copy_on_write = true;
  cfg.dram.capacity_bytes = 1 << 20;
  cfg.ssd.enabled = true;
  cfg.ssd.ssd_backend = "dummy_storage";
  cfg.ssd.capacity_bytes = 1 << 20;
  cfg.copy_pipeline.async_enabled = true;
  cfg.copy_pipeline.worker_threads = 1;
  cfg.copy_pipeline.queue_depth = 8;

  LocalBlockIndex index;
  LocalStorageManager storage(cfg, &index);
  CopyPipeline pipeline(storage, cfg.copy_pipeline, cfg.ResolveRole());

  std::string payload = "copy-pipeline-drain";
  ASSERT_TRUE(
      storage.WriteFromPtr("k", reinterpret_cast<uintptr_t>(payload.data()), payload.size()));
  index.Insert("k", {StorageTier::CPU_DRAM, 0, payload.size()});

  ASSERT_TRUE(pipeline.MaybeCopyToSharedSSD("k"));
  EXPECT_TRUE(pipeline.Drain(std::chrono::seconds(2)));
  auto* ssd = storage.GetTier(StorageTier::LOCAL_SSD);
  ASSERT_NE(ssd, nullptr);
  EXPECT_TRUE(ssd->Exists("k"));
}

}  // namespace
}  // namespace mori::umbp

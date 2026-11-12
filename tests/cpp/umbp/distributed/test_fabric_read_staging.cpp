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

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/backend/mock_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool/fabric_read_push.h"
#include "umbp/distributed/pool/fabric_read_staging.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/range_map.h"
#include "umbp/distributed/transfer/plan_builder.h"

namespace mori::umbp {
namespace {
using Location = mori::io::MemoryLocationType;
constexpr uint64_t kPage = 16;

struct ArenaStats {
  void* pointer = nullptr;
  int allocations = 0;
  int releases = 0;
  int deregistrations = 0;
  int live_handles = 0;
};

class TestArena final : public PageMemorySource {
 public:
  explicit TestArena(std::shared_ptr<ArenaStats> stats) : stats_(std::move(stats)) {}
  ~TestArena() override { Release(); }
  bool Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) override {
    if (sizes.size() != 1 || bytes_) return false;
    bytes_ = std::make_unique<char[]>(static_cast<size_t>(sizes.front()));
    stats_->pointer = bytes_.get();
    ++stats_->allocations;
    out->push_back({bytes_.get(), sizes.front()});
    return true;
  }
  void Release() override {
    if (!bytes_) return;
    bytes_.reset();
    stats_->pointer = nullptr;
    ++stats_->releases;
  }
  Location LocationType() const override { return Location::GPU; }
  int Device() const override { return 0; }
  const char* Name() const override { return "test-exportable-arena"; }

 private:
  std::shared_ptr<ArenaStats> stats_;
  std::unique_ptr<char[]> bytes_;
};

class SourceBackend final : public MockBackend {
 public:
  explicit SourceBackend(Location loc = Location::CPU)
      : MockBackend(loc == Location::CPU ? TierType::DRAM : TierType::HBM, kPage),
        loc_(loc),
        bytes_(8 * kPage) {
    for (size_t i = 0; i < bytes_.size(); ++i) bytes_[i] = static_cast<char>(i);
  }
  TransferRef BufferRef(uint32_t index) const override {
    ++lookups;
    if (index != 0) return {};
    return TransferRef::HostBytes(const_cast<char*>(bytes_.data()), bytes_.size(), loc_,
                                  loc_ == Location::CPU ? -1 : 0);
  }
  size_t BufferCount() const override { return 1; }
  std::vector<ResolvedEntry> BatchResolve(const std::vector<std::string>& keys, bool, bool = false,
                                          ResolveReader = ResolveReader::kRemote,
                                          const std::vector<ObjectExtent>* = nullptr) override {
    return std::vector<ResolvedEntry>(keys.size(), forced_resolution);
  }
  ResolvedEntry forced_resolution;
  mutable std::atomic<int> lookups{0};

 private:
  Location loc_;
  std::vector<char> bytes_;
};

class CopyEngine final : public TransferEngine {
 public:
  enum class Fault { kNone, kDrained, kUnconfirmed, kThrow, kSubmitThrow, kNull };
  explicit CopyEngine(std::shared_ptr<ArenaStats> stats) : stats_(std::move(stats)) {}
  const char* Name() const override { return "test-H2D"; }
  TransferRef RegisterMemory(void* base, size_t size, Location location, int device) override {
    return RegisterMemoryWithBackends(base, size, location, device, mori::io::kAllBackendMask);
  }
  TransferRef RegisterMemoryWithBackends(void* base, size_t size, Location location, int device,
                                         uint32_t mask) override {
    auto ref = TransferRef::HostBytes(base, size, location, device);
    ref.mem.loc = location;
    ref.mem.size = size;
    ref.mem.data = reinterpret_cast<uintptr_t>(base);
    ref.mem.deviceId = device;
    ref.mem.engineKey = "test-stage-owner";
    ref.mem.registeredBackendMask = mask;
    if (exportable) ref.mem.fabricHandle[0] = 1;
    return ref;
  }
  void Deregister(const TransferRef&) override { ++stats_->deregistrations; }
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override {
    if (push_mode) return src.HasHostPtr() && dst.HasMemoryDesc() && !dst.HasHostPtr();
    return src.HasHostPtr() && dst.HasHostPtr() && src.loc == Location::CPU &&
           dst.loc == Location::GPU;
  }
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t) const override {
    return MakePairPlanBuilder(*this, [](TransferPlan* plan, const TransferItem& item) {
      AppendSegment(plan, item.src_offset, item.dst_offset, item.size, item.tag);
    });
  }
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override {
    ++submissions;
    if (fault == Fault::kNull) return nullptr;
    if (fault == Fault::kSubmitThrow) throw std::runtime_error("unknown H2D submission");
    class Handle final : public TransferHandle {
     public:
      Handle(std::vector<TransferPlan> work, Fault fault, std::shared_ptr<ArenaStats> stats,
             uint64_t* copied, std::function<void()> before_wait)
          : work_(std::move(work)),
            fault_(fault),
            stats_(std::move(stats)),
            copied_(copied),
            before_wait_(std::move(before_wait)) {
        ++stats_->live_handles;
      }
      ~Handle() override { --stats_->live_handles; }
      void Wait(std::vector<TransferFailure>* failures) override {
        if (fault_ == Fault::kThrow) throw std::runtime_error("unknown H2D completion");
        if (waited_) return;
        waited_ = true;
        if (before_wait_) before_wait_();
        if (fault_ != Fault::kNone) {
          TransferFailure failure;
          failure.code = 1;
          failure.resources_quarantined = fault_ == Fault::kUnconfirmed;
          for (const auto& plan : work_)
            failure.tags.insert(failure.tags.end(), plan.tags.begin(), plan.tags.end());
          failures->push_back(std::move(failure));
          return;
        }
        for (const auto& plan : work_)
          for (size_t i = 0; i < plan.sizes.size(); ++i) {
            std::memcpy(
                static_cast<char*>(plan.dst.host_ptr ? plan.dst.host_ptr
                                                     : reinterpret_cast<void*>(plan.dst.mem.data)) +
                    plan.dst_offsets[i],
                static_cast<const char*>(plan.src.host_ptr) + plan.src_offsets[i], plan.sizes[i]);
            *copied_ += plan.sizes[i];
          }
      }

     private:
      std::vector<TransferPlan> work_;
      Fault fault_;
      std::shared_ptr<ArenaStats> stats_;
      uint64_t* copied_;
      bool waited_ = false;
      std::function<void()> before_wait_;
    };
    return std::make_unique<Handle>(std::move(plans), fault, stats_, &copied_bytes, before_wait);
  }
  bool exportable = true;
  bool push_mode = false;
  Fault fault = Fault::kNone;
  size_t submissions = 0;
  uint64_t copied_bytes = 0;
  std::function<void()> before_wait;

 private:
  std::shared_ptr<ArenaStats> stats_;
};

class FabricReadStagingTest : public testing::Test {
 protected:
  FabricReadStagingTest() : engine(stats) {
    auto backend = std::make_unique<SourceBackend>();
    cpu = backend.get();
    registry.Register("cpu", std::move(backend));
    registry.Register("gpu", std::make_unique<SourceBackend>(Location::GPU));
  }
  std::unique_ptr<FabricReadStaging> MakeStaging(uint64_t capacity = 4 * kPage) {
    auto staging = std::make_unique<FabricReadStaging>(&engine, std::make_unique<TestArena>(stats),
                                                       kPage, capacity);
    EXPECT_TRUE(staging->Init(&registry));
    return staging;
  }
  PoolResolvedEntry Entry(uint64_t bytes, std::vector<uint32_t> pages = {0, 1, 2, 3},
                          uint32_t backend = 0) {
    PoolResolvedEntry entry;
    entry.backend_id = backend;
    entry.tier = backend == 0 ? TierType::DRAM : TierType::HBM;
    entry.resolved.outcome = ResolveOutcome::kFound;
    entry.resolved.found = true;
    entry.resolved.size = bytes;
    entry.resolved.page_size = kPage;
    entry.resolved.hold = std::make_shared<int>(7);
    for (uint32_t page : pages) entry.resolved.pages.push_back({0, page});
    return entry;
  }
  std::vector<char> StagedBytes(const PoolResolvedEntry& entry, uint64_t size) {
    std::vector<char> out;
    auto* arena = static_cast<char*>(stats->pointer);
    for (const auto& page : entry.resolved.pages) {
      const uint64_t count = std::min(kPage, size - out.size());
      const char* begin = arena + page.page_index * kPage;
      out.insert(out.end(), begin, begin + count);
      if (out.size() == size) break;
    }
    return out;
  }
  std::vector<char> PackedPayload(const PoolResolvedEntry& entry) {
    const auto& ranges = entry.resolved.packed_ranges;
    const auto bytes = StagedBytes(entry, PackedRangeBytes(ranges));
    std::vector<char> payload;
    for (const auto& range : ranges) {
      size_t offset = range.offset;
      EXPECT_TRUE(RebaseRangeOntoPackedRanges(ranges, &offset, range.length));
      payload.insert(payload.end(), bytes.begin() + offset, bytes.begin() + offset + range.length);
    }
    return payload;
  }
  std::shared_ptr<ArenaStats> stats = std::make_shared<ArenaStats>();
  CopyEngine engine;
  BackendRegistry registry;
  SourceBackend* cpu = nullptr;
};

class PushPeers final : public PeerDirectory {
 public:
  std::vector<uint8_t> PackedLocalEngineDesc() const override { return {}; }
  bool EnsureRemoteEngine(const std::string&, const std::string&) override { return true; }
  bool HasRemoteEngine(const std::string&) const override { return true; }
  void ForgetRemote(const std::string&) override {}
  void CacheRemoteBuffers(const std::string&, const std::vector<BufferMemoryDescBytes>&) override {}
  bool HasRemoteBuffers(const std::string&) const override { return true; }
  std::vector<TransferRef> RemoteBufferSnapshot(const std::string&, uint32_t) const override {
    return {};
  }
};

TEST_F(FabricReadStagingTest, OwnerPushCopiesFinalRangesAndRejectsBoundsBeforePosting) {
  PushPeers peers;
  engine.push_mode = true;
  bool quarantined = false;
  FabricReadPush push(
      &engine, &peers, &registry, 0,
      [](const void* ptr, size_t, int) { return const_cast<void*>(ptr); },
      [&] { quarantined = true; });
  std::vector<char> destination(80, char(0x7f));
  auto dst = engine.RegisterMemory(destination.data(), destination.size(), Location::GPU, 0);
  dst.host_ptr = nullptr;
  dst.mem.engineKey = "reader";
  FabricReadPushRequest request;
  request.ranges = {{{13, 19, dst, 3}, {48, 8, dst, 50}}};
  std::vector<PoolResolvedEntry> entries{Entry(64, {1, 3, 0, 2})};
  const auto result = push.Execute(&entries, request);
  ASSERT_TRUE(result.supported);
  EXPECT_TRUE(result.drained);
  EXPECT_EQ(result.submitted_bytes, 27);
  EXPECT_EQ(engine.copied_bytes, 27);
  for (size_t i = 0; i < destination.size(); ++i) {
    char expected = char(0x7f);
    if (i >= 3 && i < 22) expected = i < 6 ? char(29 + i - 3) : char(48 + i - 6);
    if (i >= 50 && i < 58) expected = char(32 + i - 50);
    EXPECT_EQ(destination[i], expected) << i;
  }
  request.ranges[0][1].destination_offset = 79;
  EXPECT_FALSE(push.Execute(&entries, request).supported);
  EXPECT_EQ(engine.submissions, 1);
  EXPECT_FALSE(quarantined);
}

TEST_F(FabricReadStagingTest, OwnerPushRetryJoinsOnceAndFencesChangedDestinationAndRelease) {
  PushPeers peers;
  engine.push_mode = true;
  PeerPool pool(&registry, MakeSingleBackendPolicy(), &engine);
  pool.SetFabricReadPush(std::make_shared<FabricReadPush>(
      &engine, &peers, &registry, 0,
      [](const void* ptr, size_t, int) { return const_cast<void*>(ptr); }, [] {}));
  cpu->forced_resolution = Entry(64).resolved;
  std::vector<char> destination(64);
  auto dst = engine.RegisterMemory(destination.data(), destination.size(), Location::GPU, 0);
  dst.host_ptr = nullptr;
  FabricReadPushRequest request;
  request.signature = "immutable request";
  request.ranges = {{{1, 31, dst, 2}}};
  const auto epoch = pool.Generation();
  auto acquire = [&] {
    return pool.AcquireReadBatch({"key"}, "push-op", epoch, false, nullptr, false, nullptr,
                                 &request);
  };
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  engine.before_wait = [&] {
    entered.set_value();
    released.wait();
  };
  auto first_call = std::async(std::launch::async, acquire);
  entered.get_future().wait();
  auto retry_call = std::async(std::launch::async, acquire);
  EXPECT_EQ(retry_call.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
  EXPECT_FALSE(pool.AbandonReadOperations({"push-op"})[0]);
  release.set_value();
  auto first = first_call.get();
  const auto retry = retry_call.get();
  ASSERT_TRUE(first.accepted);
  EXPECT_EQ(retry.token, first.token);
  ASSERT_TRUE(first.push_supported);
  EXPECT_TRUE(first.push_drained);
  EXPECT_EQ(acquire().token, first.token);
  EXPECT_EQ(engine.submissions, 1);
  request.signature = "different destination";
  EXPECT_FALSE(acquire().accepted);
  request.signature = "immutable request";
  EXPECT_FALSE(pool.AbandonReadOperations({"push-op"})[0]);
  EXPECT_TRUE(pool.ReleaseReadPins({first.token})[0]);
  EXPECT_FALSE(acquire().accepted);
  EXPECT_EQ(engine.submissions, 1);
}

TEST_F(FabricReadStagingTest, ClearDuringPushCannotReleaseSourceBeforeGpuDrain) {
  PushPeers peers;
  engine.push_mode = true;
  PeerPool pool(&registry, MakeSingleBackendPolicy(), &engine);
  pool.SetFabricReadPush(std::make_shared<FabricReadPush>(
      &engine, &peers, &registry, 0,
      [](const void* ptr, size_t, int) { return const_cast<void*>(ptr); }, [] {}));
  cpu->forced_resolution = Entry(64).resolved;
  std::weak_ptr<const void> source_pin = cpu->forced_resolution.hold;
  std::vector<char> destination(64);
  auto dst = engine.RegisterMemory(destination.data(), destination.size(), Location::GPU, 0);
  dst.host_ptr = nullptr;
  FabricReadPushRequest request;
  request.signature = "cancel-push";
  request.ranges = {{{0, 64, dst, 0}}};
  engine.before_wait = [&] {
    cpu->forced_resolution.hold.reset();
    pool.ClearLocal();
    EXPECT_FALSE(source_pin.expired());
  };
  const auto grant = pool.AcquireReadBatch({"key"}, "cancel-push", pool.Generation(), false,
                                           nullptr, false, nullptr, &request);
  EXPECT_FALSE(grant.accepted);
  EXPECT_TRUE(source_pin.expired());
  EXPECT_EQ(engine.copied_bytes, 64);
  EXPECT_FALSE(pool.AcquireReadBatch({"key"}, "cancel-push", pool.Generation(), false, nullptr,
                                     false, nullptr, &request)
                   .accepted);
  EXPECT_EQ(engine.submissions, 1);
}

TEST_F(FabricReadStagingTest, OwnerPushBusyBatchDoesNotWriteSuccessfulPrefix) {
  PushPeers peers;
  engine.push_mode = true;
  FabricReadPush push(
      &engine, &peers, &registry, 0,
      [](const void* ptr, size_t, int) { return const_cast<void*>(ptr); }, [] {});
  std::vector<char> destination(128, 7);
  auto dst = engine.RegisterMemory(destination.data(), destination.size(), Location::GPU, 0);
  dst.host_ptr = nullptr;
  FabricReadPushRequest request;
  request.ranges = {{{0, 64, dst, 0}}, {{0, 64, dst, 64}}};
  std::vector<PoolResolvedEntry> entries{Entry(64), {}};
  entries[1].resolved.outcome = ResolveOutcome::kBusy;
  const auto result = push.Execute(&entries, request);
  EXPECT_FALSE(result.supported);
  EXPECT_TRUE(result.drained);
  EXPECT_EQ(engine.submissions, 0);
  EXPECT_TRUE(std::all_of(destination.begin(), destination.end(), [](char c) { return c == 7; }));
}

TEST_F(FabricReadStagingTest, OwnerPushUnsupportedMappingDoesNotWriteAndUnknownDrainRetains) {
  PushPeers peers;
  engine.push_mode = true;
  bool mapped = false, quarantined = false;
  FabricReadPush push(
      &engine, &peers, &registry, 0,
      [&](const void* ptr, size_t, int) { return mapped ? const_cast<void*>(ptr) : nullptr; },
      [&] { quarantined = true; });
  std::vector<char> destination(64, 7);
  auto dst = engine.RegisterMemory(destination.data(), destination.size(), Location::GPU, 0);
  dst.host_ptr = nullptr;
  FabricReadPushRequest request;
  request.ranges = {{{0, 64, dst, 0}}};
  std::vector<PoolResolvedEntry> entries{Entry(64)};
  EXPECT_FALSE(push.Execute(&entries, request).supported);
  EXPECT_EQ(engine.submissions, 0);
  mapped = true;
  engine.fault = CopyEngine::Fault::kUnconfirmed;
  std::weak_ptr<const void> hold = entries.front().resolved.hold;
  const auto result = push.Execute(&entries, request);
  EXPECT_TRUE(result.supported);
  EXPECT_FALSE(result.drained);
  entries.clear();
  EXPECT_FALSE(hold.expired());
  EXPECT_TRUE(quarantined);
  EXPECT_EQ(stats->live_handles, 1);
}

TEST_F(FabricReadStagingTest, CopiesSubsetAcrossNoncontiguousSourcePagesAndRecyclesOnLastPin) {
  auto staging = MakeStaging();
  std::vector<PoolResolvedEntry> entries{Entry(64, {1, 3, 0, 2})};
  std::weak_ptr<const void> source_hold = entries.front().resolved.hold;
  const std::vector<ObjectExtent> extents{{13, 19}};
  ASSERT_TRUE(staging->Stage(&entries, &extents));
  const auto& result = entries.front();
  EXPECT_EQ(result.backend_id, FabricReadStaging::kBackendId);
  EXPECT_EQ(result.tier, TierType::DRAM);
  EXPECT_EQ(result.resolved.size, 64);
  EXPECT_EQ(result.resolved.covered_offset, 13);
  EXPECT_EQ(result.resolved.covered_size, 19);
  EXPECT_EQ(result.resolved.descs.size(), 1);
  EXPECT_EQ(result.resolved.descs.front().backend_id, FabricReadStaging::kBackendId);
  EXPECT_EQ(engine.copied_bytes, 19);
  EXPECT_EQ(cpu->lookups.load(), 1);
  EXPECT_TRUE(source_hold.expired());
  std::vector<char> expected{29, 30, 31};
  for (char byte = 48; byte < 64; ++byte) expected.push_back(byte);
  EXPECT_EQ(StagedBytes(result, 19), expected);
  auto second_pin = result.resolved.hold;
  entries.clear();
  EXPECT_EQ(staging->InUseBytes(), 2 * kPage);
  second_pin.reset();
  EXPECT_EQ(staging->InUseBytes(), 0);
  EXPECT_FALSE(staging->HasQuarantinedTransfers());
  staging.reset();
  EXPECT_EQ(stats->allocations, 1);
  EXPECT_EQ(stats->releases, 1);
  EXPECT_EQ(stats->deregistrations, 1);
}

TEST_F(FabricReadStagingTest, ExistingCoverageUsesObjectRelativeOffsetsAndOmittedDescriptors) {
  auto staging = MakeStaging();
  std::vector<PoolResolvedEntry> entries{Entry(256, {2, 5})};
  entries.front().tier = TierType::SSD;
  entries.front().resolved.covered_offset = 17;
  entries.front().resolved.covered_size = 29;
  const std::vector<ObjectExtent> extents{{20, 9}};
  ASSERT_TRUE(staging->Stage(&entries, &extents, false));
  EXPECT_EQ(entries.front().tier, TierType::SSD);
  EXPECT_EQ(entries.front().resolved.size, 256);
  EXPECT_EQ(entries.front().resolved.covered_offset, 20);
  EXPECT_EQ(entries.front().resolved.covered_size, 9);
  EXPECT_TRUE(entries.front().resolved.descs.empty());
  EXPECT_EQ(staging->Descriptors().size(), 1);
  std::vector<char> expected;
  for (char byte = 35; byte < 44; ++byte) expected.push_back(byte);
  EXPECT_EQ(StagedBytes(entries.front(), 9), expected);
  entries.clear();
  EXPECT_EQ(staging->InUseBytes(), 0);
}

TEST_F(FabricReadStagingTest, SparseUnionCopiesOnlyRequestedBytesIntoCompactPages) {
  auto staging = MakeStaging(2 * kPage);
  std::vector<PoolResolvedEntry> entries{Entry(64, {1, 3, 0, 2})};
  const std::vector<ObjectExtent> windows{{1, 60}};
  // Unsorted and overlapping spans must share their bytes, even across a
  // source page boundary. The 60-byte envelope cannot fit this 32-byte arena.
  const std::vector<std::vector<ObjectExtent>> ranges{{{58, 3}, {13, 6}, {1, 4}, {15, 6}}};
  ASSERT_TRUE(staging->Stage(&entries, &windows, true, &ranges));
  EXPECT_EQ(engine.copied_bytes, 15);
  EXPECT_EQ(staging->InUseBytes(), 2 * kPage);
  const auto& result = entries.front();
  EXPECT_EQ(result.resolved.size, 64);
  EXPECT_EQ(result.resolved.covered_offset, 0);
  EXPECT_EQ(result.resolved.covered_size, 0);
  EXPECT_EQ(result.resolved.packed_ranges, (std::vector<ObjectExtent>{{1, 4}, {13, 8}, {58, 3}}));
  EXPECT_EQ(PackedPayload(result),
            (std::vector<char>{17, 18, 19, 20, 29, 30, 31, 48, 49, 50, 51, 52, 42, 43, 44}));
}

TEST_F(FabricReadStagingTest, SparseRequestOutsideWindowOrObjectNeverPosts) {
  auto staging = MakeStaging();
  for (const auto& spans : std::vector<std::vector<ObjectExtent>>{
           {{0, 2}}, {{31, 3}}, {{63, 2}}, {{17, 0}}, {{UINT64_MAX, 4}}}) {
    std::vector<PoolResolvedEntry> entries{Entry(64)};
    const std::vector<ObjectExtent> windows{{16, 16}};
    const std::vector<std::vector<ObjectExtent>> ranges{spans};
    EXPECT_FALSE(staging->Stage(&entries, &windows, true, &ranges));
    EXPECT_FALSE(entries.front().resolved.found);
    EXPECT_EQ(engine.submissions, 0);
    EXPECT_EQ(staging->InUseBytes(), 0);
  }
}

TEST_F(FabricReadStagingTest, PacksRangesFromAnAlreadyStagedSsdWindow) {
  auto staging = MakeStaging(kPage);
  std::vector<PoolResolvedEntry> entries{Entry(256, {2, 5})};
  entries.front().tier = TierType::SSD;
  entries.front().resolved.covered_offset = 17;
  entries.front().resolved.covered_size = 29;
  const std::vector<ObjectExtent> windows{{20, 24}};
  const std::vector<std::vector<ObjectExtent>> ranges{{{20, 3}, {40, 4}}};
  ASSERT_TRUE(staging->Stage(&entries, &windows, false, &ranges));
  EXPECT_EQ(engine.copied_bytes, 7);
  EXPECT_TRUE(entries.front().resolved.descs.empty());
  EXPECT_EQ(PackedPayload(entries.front()), (std::vector<char>{35, 36, 37, 87, 88, 89, 90}));
}

TEST_F(FabricReadStagingTest, NoExtentCopiesOnlyExistingCoverage) {
  auto staging = MakeStaging();
  std::vector<PoolResolvedEntry> entries{Entry(256, {2, 5})};
  entries.front().resolved.covered_offset = 17;
  entries.front().resolved.covered_size = 29;
  ASSERT_TRUE(staging->Stage(&entries));
  EXPECT_EQ(engine.copied_bytes, 29);
  EXPECT_EQ(entries.front().resolved.covered_offset, 17);
  EXPECT_EQ(entries.front().resolved.covered_size, 29);
  entries.clear();
}

TEST_F(FabricReadStagingTest, CacheReusesImmutableBytesAfterReleaseAndRejectsNewContentVersion) {
  auto staging = MakeStaging();
  staging->EnableCache(4 * kPage);
  auto make = [&](uint64_t version, uint32_t first) {
    std::vector<PoolResolvedEntry> entries{Entry(32, {first, first + 1})};
    entries.front().resolved.content_version = version;
    return entries;
  };
  auto entries = make(1, 0);
  ASSERT_TRUE(staging->Stage(&entries));
  const auto original = StagedBytes(entries.front(), 32);
  entries.clear();
  EXPECT_EQ(staging->InUseBytes(), 32);
  entries = make(1, 0);
  ASSERT_TRUE(staging->Stage(&entries, nullptr, false));
  EXPECT_EQ(engine.copied_bytes, 32);
  EXPECT_EQ(StagedBytes(entries.front(), 32), original);
  EXPECT_TRUE(entries.front().resolved.descs.empty());
  entries.clear();
  entries = make(2, 4);
  ASSERT_TRUE(staging->Stage(&entries));
  EXPECT_EQ(engine.copied_bytes, 64);
  EXPECT_NE(StagedBytes(entries.front(), 32), original);
  EXPECT_EQ(staging->GetCacheStats().hits, 1);
  entries.clear();
  staging.reset();
  EXPECT_EQ(stats->releases, 1);
  EXPECT_EQ(stats->deregistrations, 1);
}

TEST_F(FabricReadStagingTest, CacheEvictionPreservesActiveGrantsAndReclaimsIdlePages) {
  auto staging = MakeStaging(2 * kPage);
  staging->EnableCache(2 * kPage);
  std::vector<PoolResolvedEntry> first{Entry(32, {0, 1})};
  first.front().resolved.content_version = 1;
  ASSERT_TRUE(staging->Stage(&first));
  const auto expected = StagedBytes(first.front(), 32);
  std::vector<PoolResolvedEntry> second{Entry(32, {4, 5})};
  second.front().resolved.content_version = 2;
  EXPECT_FALSE(staging->Stage(&second));
  EXPECT_EQ(second.front().resolved.outcome, ResolveOutcome::kBusy);
  EXPECT_EQ(StagedBytes(first.front(), 32), expected);
  EXPECT_EQ(engine.copied_bytes, 32);
  first.clear();
  second = {Entry(32, {4, 5})};
  second.front().resolved.content_version = 2;
  ASSERT_TRUE(staging->Stage(&second));
  EXPECT_EQ(engine.copied_bytes, 64);
  second.clear();
  first = {Entry(32, {0, 1})};
  first.front().resolved.content_version = 1;
  ASSERT_TRUE(staging->Stage(&first));
  EXPECT_EQ(engine.copied_bytes, 96);
  EXPECT_EQ(StagedBytes(first.front(), 32), expected);
}

TEST_F(FabricReadStagingTest, MetadataCacheHitRequiresLiveMatchingContentAndCompleteCoverage) {
  auto staging = MakeStaging();
  staging->EnableCache(4 * kPage);
  std::vector<PoolResolvedEntry> original{Entry(64)};
  original.front().resolved.content_version = 11;
  const std::vector<ObjectExtent> windows{{0, 64}};
  const std::vector<std::vector<ObjectExtent>> ranges{{{0, 4}, {48, 4}}};
  ASSERT_TRUE(staging->Stage(&original, &windows, true, &ranges));
  auto metadata = [&] {
    auto entry = Entry(64);
    entry.resolved.content_version = 11;
    entry.resolved.pages.clear();
    entry.resolved.hold.reset();
    return std::vector<PoolResolvedEntry>{entry};
  };
  auto hit = metadata();
  ASSERT_TRUE(staging->TryCached(&hit, &windows, true, &ranges));
  EXPECT_EQ(PackedPayload(hit.front()), PackedPayload(original.front()));
  EXPECT_TRUE(hit.front().resolved.hold);
  EXPECT_EQ(engine.copied_bytes, 8);
  auto changed = metadata();
  changed.front().resolved.content_version = 12;
  EXPECT_FALSE(staging->TryCached(&changed, &windows, true, &ranges));
  changed = metadata();
  changed.front().resolved.found = false;
  EXPECT_FALSE(staging->TryCached(&changed, &windows, true, &ranges));
  changed = metadata();
  const std::vector<std::vector<ObjectExtent>> different{{{0, 4}, {47, 5}}};
  EXPECT_FALSE(staging->TryCached(&changed, &windows, true, &different));
}

TEST_F(FabricReadStagingTest, ConcurrentIdenticalReadsJoinOneH2dPreparation) {
  auto staging = MakeStaging(2 * kPage);
  staging->EnableCache(2 * kPage);
  std::promise<void> started, release;
  const auto released = release.get_future().share();
  engine.before_wait = [&] {
    started.set_value();
    released.wait();
  };
  std::vector<PoolResolvedEntry> first{Entry(32, {0, 1})}, second = first;
  first.front().resolved.content_version = second.front().resolved.content_version = 1;
  auto leader = std::async(std::launch::async, [&] { return staging->Stage(&first); });
  started.get_future().wait();
  auto follower = std::async(std::launch::async, [&] { return staging->Stage(&second); });
  for (int i = 0; i < 1000 && staging->GetCacheStats().joins == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool joined = staging->GetCacheStats().joins == 1;
  release.set_value();
  EXPECT_TRUE(joined);
  EXPECT_TRUE(leader.get());
  EXPECT_TRUE(follower.get());
  EXPECT_EQ(engine.submissions, 1);
  EXPECT_EQ(engine.copied_bytes, 32);
  EXPECT_EQ(StagedBytes(first.front(), 32), StagedBytes(second.front(), 32));
}

TEST_F(FabricReadStagingTest, FailedCachePreparationWakesJoinerAndAllowsRetry) {
  auto staging = MakeStaging(2 * kPage);
  staging->EnableCache(2 * kPage);
  engine.fault = CopyEngine::Fault::kDrained;
  std::promise<void> started, release;
  const auto released = release.get_future().share();
  engine.before_wait = [&] {
    started.set_value();
    released.wait();
  };
  std::vector<PoolResolvedEntry> first{Entry(32)}, second = first;
  first.front().resolved.content_version = second.front().resolved.content_version = 1;
  auto leader = std::async(std::launch::async, [&] { return staging->Stage(&first); });
  started.get_future().wait();
  auto follower = std::async(std::launch::async, [&] { return staging->Stage(&second); });
  for (int i = 0; i < 1000 && staging->GetCacheStats().joins == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool joined = staging->GetCacheStats().joins == 1;
  release.set_value();
  EXPECT_TRUE(joined);
  EXPECT_FALSE(leader.get());
  EXPECT_FALSE(follower.get());
  EXPECT_EQ(first.front().resolved.outcome, ResolveOutcome::kFailed);
  EXPECT_EQ(second.front().resolved.outcome, ResolveOutcome::kFailed);
  EXPECT_EQ(engine.submissions, 1);
  EXPECT_EQ(staging->GetCacheStats().bytes, 0);
  EXPECT_EQ(staging->InUseBytes(), 0);
  engine.fault = CopyEngine::Fault::kNone;
  engine.before_wait = {};
  first = {Entry(32)};
  first.front().resolved.content_version = 1;
  ASSERT_TRUE(staging->Stage(&first));
  EXPECT_EQ(engine.submissions, 2);
  EXPECT_EQ(engine.copied_bytes, 32);
}

TEST_F(FabricReadStagingTest, CachedSparseCoverageDoesNotServeDifferentRangesOrUnknownVersions) {
  auto staging = MakeStaging();
  staging->EnableCache(4 * kPage);
  const std::vector<ObjectExtent> windows{{0, 64}};
  const std::vector<std::vector<ObjectExtent>> low{{{0, 4}, {48, 4}}}, high{{{8, 4}, {48, 4}}};
  for (const auto* ranges : {&low, &low, &high}) {
    std::vector<PoolResolvedEntry> entries{Entry(64)};
    entries.front().resolved.content_version = 1;
    ASSERT_TRUE(staging->Stage(&entries, &windows, true, ranges));
    EXPECT_EQ(entries.front().resolved.packed_ranges, ranges->front());
  }
  EXPECT_EQ(engine.submissions, 2);
  EXPECT_EQ(engine.copied_bytes, 16);
  std::vector<PoolResolvedEntry> unknown{Entry(16, {6})};
  ASSERT_TRUE(staging->Stage(&unknown));
  EXPECT_EQ(engine.submissions, 3);
}

TEST_F(FabricReadStagingTest, PressureReturnsBusyWithoutPostingAndAdmissionIsAtomic) {
  auto staging = MakeStaging(3 * kPage);
  std::vector<PoolResolvedEntry> pinned{Entry(32)};
  ASSERT_TRUE(staging->Stage(&pinned));
  const size_t submitted = engine.submissions;
  std::vector<PoolResolvedEntry> next{Entry(16), Entry(16)};
  EXPECT_FALSE(staging->Stage(&next));
  EXPECT_EQ(engine.submissions, submitted);
  for (const auto& entry : next) {
    EXPECT_FALSE(entry.resolved.found);
    EXPECT_EQ(entry.resolved.outcome, ResolveOutcome::kBusy);
  }
  EXPECT_EQ(staging->InUseBytes(), 2 * kPage);
  pinned.clear();
  EXPECT_EQ(staging->InUseBytes(), 0);
  next = {Entry(16), Entry(16)};
  ASSERT_TRUE(staging->Stage(&next));
  EXPECT_EQ(staging->InUseBytes(), 2 * kPage);
  next.clear();
}

TEST_F(FabricReadStagingTest, OversizedBatchRejectsEveryCpuEntryAndPreservesGpu) {
  auto staging = MakeStaging(3 * kPage);
  std::vector<PoolResolvedEntry> entries{Entry(32), Entry(32), Entry(16, {0}, 1), Entry(16)};
  EXPECT_FALSE(staging->Stage(&entries));
  EXPECT_EQ(engine.submissions, 0);
  EXPECT_EQ(staging->InUseBytes(), 0);
  for (size_t i : {0u, 1u, 3u}) {
    EXPECT_FALSE(entries[i].resolved.found);
    EXPECT_EQ(entries[i].resolved.outcome, ResolveOutcome::kFailed);
  }
  EXPECT_TRUE(entries[2].resolved.found);
  EXPECT_EQ(entries[2].backend_id, 1);
}

TEST_F(FabricReadStagingTest, InvalidSourceWindowFailsBeforeCopying) {
  auto staging = MakeStaging();
  std::vector<PoolResolvedEntry> entries{Entry(128, {0})};
  entries.front().resolved.covered_offset = 32;
  entries.front().resolved.covered_size = 16;
  const std::vector<ObjectExtent> extents{{31, 2}};
  EXPECT_FALSE(staging->Stage(&entries, &extents));
  EXPECT_FALSE(entries.front().resolved.found);
  EXPECT_EQ(entries.front().resolved.outcome, ResolveOutcome::kFailed);
  EXPECT_EQ(engine.submissions, 0);
  EXPECT_EQ(staging->InUseBytes(), 0);
}

TEST_F(FabricReadStagingTest, DrainedOrUnpostedFailureReleasesSourceAndStagePages) {
  auto staging = MakeStaging();
  for (auto fault : {CopyEngine::Fault::kDrained, CopyEngine::Fault::kNull}) {
    engine.fault = fault;
    std::vector<PoolResolvedEntry> entries{Entry(32)};
    std::weak_ptr<const void> hold = entries.front().resolved.hold;
    EXPECT_FALSE(staging->Stage(&entries));
    EXPECT_FALSE(entries.front().resolved.found);
    EXPECT_EQ(entries.front().resolved.outcome, ResolveOutcome::kFailed);
    EXPECT_TRUE(hold.expired());
    EXPECT_EQ(staging->InUseBytes(), 0);
    EXPECT_FALSE(staging->HasQuarantinedTransfers());
    EXPECT_EQ(stats->live_handles, 0);
  }
}

TEST_F(FabricReadStagingTest, UnknownCompletionRetainsSourcesHandleAndArena) {
  for (auto fault : {CopyEngine::Fault::kUnconfirmed, CopyEngine::Fault::kThrow,
                     CopyEngine::Fault::kSubmitThrow}) {
    auto staging = MakeStaging();
    engine.fault = fault;
    std::vector<PoolResolvedEntry> entries{Entry(32)};
    std::weak_ptr<const void> hold = entries.front().resolved.hold;
    EXPECT_FALSE(staging->Stage(&entries));
    EXPECT_TRUE(staging->HasQuarantinedTransfers());
    EXPECT_FALSE(hold.expired());
    EXPECT_EQ(staging->InUseBytes(), 32);
    EXPECT_TRUE(staging->Descriptors().empty());
    const size_t submitted = engine.submissions;
    entries = {Entry(16)};
    EXPECT_FALSE(staging->Stage(&entries));
    EXPECT_EQ(engine.submissions, submitted);
    staging.reset();
    EXPECT_EQ(stats->releases, 0);
    EXPECT_EQ(stats->deregistrations, 0);
  }
  EXPECT_EQ(stats->live_handles, 2);
}

TEST_F(FabricReadStagingTest, ManagerShutdownWithOutstandingReadPinRetainsArena) {
  auto staging = MakeStaging();
  std::vector<PoolResolvedEntry> entries{Entry(16)};
  ASSERT_TRUE(staging->Stage(&entries));
  staging.reset();
  EXPECT_EQ(stats->releases, 0);
  entries.clear();
  EXPECT_EQ(stats->releases, 0);
  EXPECT_EQ(stats->deregistrations, 0);
}

TEST_F(FabricReadStagingTest, InitRequiresExportableArenaAndAlignedBudget) {
  engine.exportable = false;
  {
    FabricReadStaging staging(&engine, std::make_unique<TestArena>(stats), kPage, 64);
    EXPECT_FALSE(staging.Init(&registry));
    EXPECT_EQ(staging.CapacityBytes(), 0);
    EXPECT_EQ(staging.InUseBytes(), 0);
  }
  EXPECT_EQ(stats->releases, 1);
  EXPECT_EQ(stats->deregistrations, 1);
  FabricReadStaging invalid(&engine, std::make_unique<TestArena>(stats), kPage, 65);
  EXPECT_FALSE(invalid.Init(&registry));
  EXPECT_EQ(stats->allocations, 1);
}

}  // namespace
}  // namespace mori::umbp

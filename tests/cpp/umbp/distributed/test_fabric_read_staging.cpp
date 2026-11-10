// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/backend/mock_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool/fabric_read_staging.h"
#include "umbp/distributed/pool/peer_pool.h"
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
  mutable int lookups = 0;

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
             uint64_t* copied)
          : work_(std::move(work)), fault_(fault), stats_(std::move(stats)), copied_(copied) {
        ++stats_->live_handles;
      }
      ~Handle() override { --stats_->live_handles; }
      void Wait(std::vector<TransferFailure>* failures) override {
        if (fault_ == Fault::kThrow) throw std::runtime_error("unknown H2D completion");
        if (waited_) return;
        waited_ = true;
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
            std::memcpy(static_cast<char*>(plan.dst.host_ptr) + plan.dst_offsets[i],
                        static_cast<const char*>(plan.src.host_ptr) + plan.src_offsets[i],
                        plan.sizes[i]);
            *copied_ += plan.sizes[i];
          }
      }

     private:
      std::vector<TransferPlan> work_;
      Fault fault_;
      std::shared_ptr<ArenaStats> stats_;
      uint64_t* copied_;
      bool waited_ = false;
    };
    return std::make_unique<Handle>(std::move(plans), fault, stats_, &copied_bytes);
  }
  bool exportable = true;
  Fault fault = Fault::kNone;
  size_t submissions = 0;
  uint64_t copied_bytes = 0;

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
  std::shared_ptr<ArenaStats> stats = std::make_shared<ArenaStats>();
  CopyEngine engine;
  BackendRegistry registry;
  SourceBackend* cpu = nullptr;
};

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
  EXPECT_EQ(cpu->lookups, 1);
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

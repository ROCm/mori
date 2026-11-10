// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/peer/backend/instrumented_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/transfer/engine_transfer_engine.h"
#include "umbp/engine/engine_client.h"
#include "umbp/engine/engine_control.h"
#include "umbp/engine/engine_dispatch.h"
#include "umbp/engine/engine_executor.h"
#include "umbp/engine/engine_session.h"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/ipc.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

TEST(EnginePoolLoopTest, LocalExecutorMovesAnEngineOwnedSpan) {
  std::array<uint8_t, 64> source{};
  std::array<uint8_t, 64> target{};
  for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i + 1);

  engine::EngineExecutor executor({"executor-a", -1});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "source",
      TransferRef::HostBytes(source.data(), source.size(), mori::io::MemoryLocationType::CPU, -1)));
  TransferRef remote_source;
  remote_source.size = source.size();
  remote_source.loc = mori::io::MemoryLocationType::CPU;
  remote_source.executor_id = "executor-a";
  remote_source.region_id = "source";
  remote_source.executor_address = reinterpret_cast<uintptr_t>(source.data());

  EngineTransferEngine transfer(&executor);
  TransferItem item;
  item.src = remote_source;
  item.src_offset = 7;
  item.dst =
      TransferRef::HostBytes(target.data(), target.size(), mori::io::MemoryLocationType::CPU, -1);
  item.dst_offset = 11;
  item.size = 29;
  item.tag = 3;
  std::vector<size_t> failed;
  ASSERT_TRUE(transfer.Transfer({item}, &failed));
  EXPECT_TRUE(failed.empty());
  EXPECT_EQ(std::memcmp(target.data() + 11, source.data() + 7, 29), 0);
}

TEST(EnginePoolLoopTest, EnginePlannerMergesManySegmentsPerEndpointPair) {
  std::array<uint8_t, 512> source{};
  std::array<uint8_t, 512> target{};
  engine::EngineExecutor executor({"merge-executor", -1});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "source",
      TransferRef::HostBytes(source.data(), source.size(), mori::io::MemoryLocationType::CPU, -1)));
  TransferRef engine_source;
  engine_source.size = source.size();
  engine_source.loc = mori::io::MemoryLocationType::CPU;
  engine_source.executor_id = "merge-executor";
  engine_source.region_id = "source";
  engine_source.executor_address = reinterpret_cast<uintptr_t>(source.data());
  EngineTransferEngine transfer(&executor);
  std::vector<TransferItem> items;
  for (size_t i = 0; i < 64; ++i) {
    items.push_back(TransferItem{
        engine_source, i * 4, TransferRef::HostBytes(target.data(), target.size()), i * 4, 4, i});
  }
  auto planned = transfer.Plan(items);
  ASSERT_TRUE(planned.rejected_tags.empty());
  ASSERT_EQ(planned.plans.size(), 1u);
  EXPECT_EQ(planned.plans.front().sizes.size(), items.size());
}

TEST(EnginePoolLoopTest, EnginePlannerBindsSharedRegionToOwningExecutor) {
  TransferRef source;
  source.size = 64;
  source.loc = mori::io::MemoryLocationType::GPU;
  source.device = 3;
  source.executor_id = "rank-executor";
  source.region_id = "rank-vmm";
  source.executor_address = 0x1000;
  TransferRef shared = TransferRef::HostBytes(reinterpret_cast<void*>(0x2000), 64);
  shared.region_id = "node-shared-dram";
  EngineTransferEngine transfer;
  auto planned = transfer.Plan({TransferItem{source, 0, shared, 0, 64, 7}});
  ASSERT_TRUE(planned.rejected_tags.empty());
  ASSERT_EQ(planned.plans.size(), 1u);
  EXPECT_EQ(planned.plans.front().dst.executor_id, "rank-executor");
  EXPECT_EQ(planned.plans.front().dst.region_id, "node-shared-dram");
  EXPECT_EQ(planned.plans.front().dst.executor_address, 0u);
}

TEST(EnginePoolLoopTest, HeterogeneousGeometryPlansF3AndLoadWithoutRejection) {
  engine::PageGeometry geometry;
  geometry.page_tokens = 256;
  geometry.num_pages = 3;
  struct Shape {
    uint64_t row_bytes;
    uint64_t rows;
  };
  const std::array<Shape, 3> shapes{{{8, 3}, {10, 2}, {7, 1}}};
  uint64_t logical = 0;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const uint64_t page_bytes = shapes[i].row_bytes * shapes[i].rows;
    TransferRef ref;
    ref.size = (geometry.num_pages + 1) * page_bytes;
    ref.loc = mori::io::MemoryLocationType::GPU;
    ref.device = 2;
    ref.executor_id = "heterogeneous-rank";
    ref.region_id = "region-" + std::to_string(i);
    ref.executor_address = 0x100000 + i * 0x10000;
    geometry.regions.push_back(engine::PageRegion{ref.region_id, ref, shapes[i].row_bytes,
                                                  shapes[i].rows, logical, static_cast<int32_t>(i),
                                                  "component"});
    logical += page_bytes;
  }
  geometry.object_bytes = logical;
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  ASSERT_TRUE(geometry.Validate());
  const ObjectView source = geometry.View(3);
  ASSERT_TRUE(ValidateObjectView(source));

  TransferRef dram =
      TransferRef::HostBytes(reinterpret_cast<void*>(0x400000), 4 * geometry.object_bytes);
  dram.region_id = "shared-dram-region";
  const ObjectView target =
      BuildPageObjectView({PageLocation{0, 2}}, geometry.object_bytes, geometry.object_bytes,
                          "page-backend-v1", [&](uint32_t) { return dram; });
  ASSERT_TRUE(ValidateObjectView(target));
  std::vector<TransferItem> f3;
  ASSERT_TRUE(BuildObjectViewTransfers(source, target, &f3));
  ASSERT_EQ(f3.size(), shapes.size());
  EngineTransferEngine engine;
  auto planned = engine.Plan(f3);
  EXPECT_TRUE(planned.rejected_tags.empty());
  ASSERT_EQ(planned.plans.size(), shapes.size());
  uint64_t at = 0;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const uint64_t bytes = shapes[i].row_bytes * shapes[i].rows;
    EXPECT_EQ(f3[i].src_offset, 3 * bytes);
    EXPECT_EQ(f3[i].dst_offset, 2 * geometry.object_bytes + at);
    EXPECT_EQ(f3[i].size, bytes);
    EXPECT_EQ(planned.plans[i].dst.executor_id, "heterogeneous-rank");
    at += bytes;
  }

  std::vector<TransferItem> load;
  ASSERT_TRUE(BuildObjectViewTransfers(target, source, &load));
  auto load_planned = engine.Plan(load);
  EXPECT_TRUE(load_planned.rejected_tags.empty());
  ASSERT_EQ(load_planned.plans.size(), shapes.size());
  for (const auto& plan : load_planned.plans) {
    EXPECT_EQ(plan.src.executor_id, "heterogeneous-rank");
  }

  TransferItem bad = f3[1];
  bad.dst_offset = bad.dst.size;
  std::vector<size_t> failed_tags;
  std::vector<TransferFailure> failure_details;
  EXPECT_FALSE(engine.Transfer({bad}, &failed_tags, nullptr, nullptr, nullptr, &failure_details));
  ASSERT_EQ(failure_details.size(), 1u);
  EXPECT_NE(failure_details.front().message.find("target offset+bytes exceeds size"),
            std::string::npos);
  EXPECT_NE(failure_details.front().message.find("executor='heterogeneous-rank'"),
            std::string::npos);
  EXPECT_NE(failure_details.front().message.find("region='region-1'"), std::string::npos);

  TransferItem bad_source = f3[1];
  bad_source.src.size = bad_source.src_offset + 1;
  failure_details.clear();
  EXPECT_FALSE(
      engine.Transfer({bad_source}, &failed_tags, nullptr, nullptr, nullptr, &failure_details));
  ASSERT_EQ(failure_details.size(), 1u);
  EXPECT_NE(failure_details.front().message.find("source offset+bytes exceeds size"),
            std::string::npos);
  EXPECT_NE(failure_details.front().message.find("reason=source offset+bytes exceeds size"),
            std::string::npos);
  EXPECT_NE(failure_details.front().message.find("size=61 offset=60 bytes=20"), std::string::npos);
}

TEST(EnginePoolLoopTest, LoadGatherRangeValidationRejectsEveryOutOfBoundsFragment) {
  std::array<uint8_t, 64> source{};
  std::array<uint8_t, 64> target{};
  TransferPlan plan;
  plan.src = TransferRef::HostBytes(source.data(), source.size());
  plan.dst = TransferRef::HostBytes(target.data(), target.size());
  plan.src_offsets = {8, 32};
  plan.dst_offsets = {4, 40};
  plan.sizes = {16, 8};
  EXPECT_TRUE(engine::ValidateLoadPlanRanges({plan}));
  plan.src_offsets[1] = 60;
  plan.sizes[1] = 8;
  EXPECT_FALSE(engine::ValidateLoadPlanRanges({plan}));
  plan.src_offsets[1] = 32;
  plan.dst_offsets[1] = 63;
  EXPECT_FALSE(engine::ValidateLoadPlanRanges({plan}));
  plan.dst_offsets[1] = 40;
  plan.dst.host_ptr = reinterpret_cast<void*>(UINTPTR_MAX - 1);
  EXPECT_FALSE(engine::ValidateLoadPlanRanges({plan}));
}

TEST(EnginePoolLoopTest, RegisteredHostSpanUsesDeviceAliasAndRejectsUnmappedRanges) {
  constexpr uintptr_t kHostBase = 0x100000;
  constexpr uintptr_t kDeviceBase = 0x900000;
  constexpr size_t kMapped = 0x4000;
  auto translated = engine::TranslateRegisteredHostSpan(kHostBase + 0x800, 0x1000, kHostBase,
                                                        kDeviceBase, kMapped);
  ASSERT_TRUE(translated.has_value());
  EXPECT_EQ(*translated, kDeviceBase + 0x800);
  EXPECT_NE(*translated, kHostBase + 0x800) << "kernel must receive the device alias";

  EXPECT_FALSE(
      engine::TranslateRegisteredHostSpan(kHostBase - 1, 1, kHostBase, kDeviceBase, kMapped));
  EXPECT_FALSE(engine::TranslateRegisteredHostSpan(kHostBase + kMapped - 1, 2, kHostBase,
                                                   kDeviceBase, kMapped));
  EXPECT_FALSE(engine::TranslateRegisteredHostSpan(kHostBase, 1, kHostBase, 0, kMapped));
  EXPECT_FALSE(engine::TranslateRegisteredHostSpan(kHostBase + 0x20, 1, kHostBase,
                                                   UINTPTR_MAX - 0x10, kMapped));
}

TEST(EnginePoolLoopTest, EarlyLoadRejectionSettlesEveryLayer) {
  engine::EngineExecutor executor({"unattached-load-executor", -1});
  engine::PreparedLoad prepared;
  prepared.success = true;
  prepared.id = 17;
  prepared.layers.resize(3);
  auto handle = executor.SubmitLoad(prepared);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->outcome(), engine::LoadOutcome::kFailed);
  EXPECT_FALSE(handle->error().empty());
  for (size_t layer = 0; layer < handle->layer_count(); ++layer) {
    EXPECT_TRUE(handle->layer_done(layer));
    EXPECT_EQ(handle->layer_outcome(layer), engine::LoadOutcome::kFailed);
    EXPECT_FALSE(handle->layer_error(layer).empty());
  }
}

TEST(EnginePoolLoopTest, SessionLossCompletesRemoteTransferAsUnknown) {
  const std::string executor_id = "lost-executor";
  std::array<uint8_t, 16> session{};
  session[0] = 7;
  auto& registry = engine::EngineDispatchRegistry::Global();
  ASSERT_TRUE(registry.Register(executor_id, session));

  std::array<uint8_t, 16> source{};
  std::array<uint8_t, 16> target{};
  TransferRef engine_source;
  engine_source.size = source.size();
  engine_source.loc = mori::io::MemoryLocationType::CPU;
  engine_source.executor_id = executor_id;
  engine_source.region_id = "source";
  engine_source.executor_address = reinterpret_cast<uintptr_t>(source.data());
  EngineTransferEngine transfer;
  TransferItem item{
      engine_source,
      0,
      TransferRef::HostBytes(target.data(), target.size(), mori::io::MemoryLocationType::CPU, -1),
      0,
      source.size(),
      0};

  auto pending = std::async(std::launch::async, [&] {
    bool quiesced = true;
    const bool ok = transfer.Transfer({item}, nullptr, nullptr, &quiesced);
    return std::pair<bool, bool>{ok, quiesced};
  });
  std::optional<engine::DispatchedTransfer> work;
  for (int i = 0; i < 100 && !work.has_value(); ++i) {
    work = registry.Fetch(executor_id, session);
    if (!work.has_value()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(work.has_value());
  registry.Unregister(executor_id, session);
  ASSERT_EQ(pending.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  auto [ok, quiesced] = pending.get();
  EXPECT_FALSE(ok);
  EXPECT_FALSE(quiesced);
}

TEST(EnginePoolLoopTest, SessionLossMarksNeverFetchedTransferNeverSubmitted) {
  const std::string executor_id = "never-fetched-executor";
  std::array<uint8_t, 16> session{};
  session[0] = 8;
  auto& registry = engine::EngineDispatchRegistry::Global();
  ASSERT_TRUE(registry.Register(executor_id, session));
  std::array<uint8_t, 16> source{};
  std::array<uint8_t, 16> target{};
  TransferRef engine_source;
  engine_source.size = source.size();
  engine_source.loc = mori::io::MemoryLocationType::CPU;
  engine_source.executor_id = executor_id;
  engine_source.region_id = "source";
  engine_source.executor_address = reinterpret_cast<uintptr_t>(source.data());
  EngineTransferEngine transfer;
  TransferItem item{
      engine_source,
      0,
      TransferRef::HostBytes(target.data(), target.size(), mori::io::MemoryLocationType::CPU, -1),
      0,
      source.size(),
      0};
  auto pending = std::async(std::launch::async, [&] {
    bool quiesced = false;
    bool submitted = true;
    const bool ok = transfer.Transfer({item}, nullptr, nullptr, &quiesced, &submitted);
    return std::tuple<bool, bool, bool>{ok, quiesced, submitted};
  });
  for (int i = 0; i < 100 && registry.PendingCountForTest(executor_id) == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(registry.PendingCountForTest(executor_id), 1u);
  registry.Unregister(executor_id, session);
  ASSERT_EQ(pending.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  auto [ok, quiesced, submitted] = pending.get();
  EXPECT_FALSE(ok);
  EXPECT_TRUE(quiesced);
  EXPECT_FALSE(submitted);
}

TEST(EnginePoolLoopTest, DispatchEventFdWakesImmediatelyAndSendFailureIsNeverSubmitted) {
  const std::string executor_id = "eventfd-executor";
  std::array<uint8_t, 16> session{};
  session[0] = 9;
  auto& registry = engine::EngineDispatchRegistry::Global();
  ASSERT_TRUE(registry.Register(executor_id, session));
  auto ticket = registry.Submit(executor_id, {TransferPlan{}});
  const int event_fd = registry.EventFd(executor_id, session);
  ASSERT_GE(event_fd, 0);
  pollfd ready{event_fd, POLLIN, 0};
  ASSERT_EQ(poll(&ready, 1, 100), 1);
  auto work = registry.Fetch(executor_id, session);
  ASSERT_TRUE(work.has_value());
  ASSERT_TRUE(registry.FailFetched(executor_id, session, work->id,
                                   engine::DispatchOutcome::kNeverSubmitted));
  std::vector<TransferFailure> failures;
  ticket->Wait(&failures);
  ASSERT_EQ(failures.size(), 1u);
  EXPECT_TRUE(failures[0].quiesced);
  EXPECT_FALSE(failures[0].submitted);
  registry.Unregister(executor_id, session);
}

TEST(EnginePoolLoopTest, DispatchTimeoutCancelsNeverFetchedWorkWithAllTags) {
  const std::string executor_id = "timeout-executor";
  std::array<uint8_t, 16> session{};
  session[0] = 10;
  auto& registry = engine::EngineDispatchRegistry::Global();
  ASSERT_TRUE(registry.Register(executor_id, session));
  TransferPlan plan;
  plan.tags = {17, 23};
  auto ticket = registry.Submit(executor_id, {plan});
  std::vector<TransferFailure> failures;
  EXPECT_FALSE(ticket->WaitFor(std::chrono::milliseconds(1), &failures));
  ASSERT_EQ(failures.size(), 1u);
  EXPECT_TRUE(failures[0].quiesced);
  EXPECT_FALSE(failures[0].submitted);
  EXPECT_EQ(failures[0].tags, std::vector<size_t>({17, 23}));
  EXPECT_EQ(registry.PendingCountForTest(executor_id), 0u);
  EXPECT_FALSE(registry.Fetch(executor_id, session).has_value());
  registry.Unregister(executor_id, session);
}

TEST(EnginePoolLoopTest, DispatchTimeoutAfterFetchIsUnknownWithAllTags) {
  const std::string executor_id = "fetched-timeout-executor";
  std::array<uint8_t, 16> session{};
  session[0] = 11;
  auto& registry = engine::EngineDispatchRegistry::Global();
  ASSERT_TRUE(registry.Register(executor_id, session));
  TransferPlan plan;
  plan.tags = {31, 37};
  auto ticket = registry.Submit(executor_id, {plan});
  auto work = registry.Fetch(executor_id, session);
  ASSERT_TRUE(work.has_value());
  std::vector<TransferFailure> failures;
  EXPECT_FALSE(ticket->WaitFor(std::chrono::milliseconds(1), &failures));
  ASSERT_EQ(failures.size(), 1u);
  EXPECT_FALSE(failures[0].quiesced);
  EXPECT_TRUE(failures[0].submitted);
  EXPECT_EQ(failures[0].tags, std::vector<size_t>({31, 37}));
  EXPECT_TRUE(
      registry.Complete(executor_id, session, work->id, engine::DispatchOutcome::kCompleted, {}))
      << "a late completion for a timed-out fetched DMA must hit its tombstone";
  EXPECT_FALSE(
      registry.Complete(executor_id, session, work->id, engine::DispatchOutcome::kCompleted, {}))
      << "the tombstone is consumed exactly once";
  registry.Unregister(executor_id, session);
}

TEST(EnginePoolLoopTest, SharedDramSlicesMapAndRegisterOneMemfdOnce) {
  HostMemAllocator allocator;
  HostBufferOptions options;
  options.backing = HostBufferBacking::kAnonymousShm;
  options.prefault = false;
  auto storage = allocator.Alloc(8192, options);
  ASSERT_TRUE(storage.valid());
  auto allocation = HostMemAllocator::LookupShmAllocation(reinterpret_cast<uintptr_t>(storage.ptr),
                                                          storage.mapped_size);
  ASSERT_TRUE(allocation.has_value());

  engine::SharedDramRegion first;
  first.region_id = "first";
  first.fd = allocation->fd;
  first.length = 4096;
  first.mapped_length = allocation->mapped_size;
  engine::SharedDramRegion second = first;
  second.region_id = "second";
  second.fd_offset = 4096;

  engine::EngineExecutor executor({"dram-mapper", -1});
  std::string error;
  ASSERT_TRUE(executor.MapSharedDram({first, second}, &error)) << error;
  EXPECT_EQ(executor.mapped_region_count(), 1u);
  executor.Close();
  allocator.Free(storage);
}

TEST(EnginePoolLoopTest, ExecutorCloseWaitsForInFlightCopyBeforeUnmapping) {
  std::array<uint8_t, 64> source{};
  std::array<uint8_t, 64> target{};
  engine::EngineExecutor executor({"close-race", -1});
  ASSERT_TRUE(executor.RegisterOriginalRegion(
      "source",
      TransferRef::HostBytes(source.data(), source.size(), mori::io::MemoryLocationType::CPU, -1)));
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool release = false;
  executor.SetBeforeExecuteHookForTest([&] {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release; });
  });
  TransferPlan plan;
  plan.src.size = source.size();
  plan.src.loc = mori::io::MemoryLocationType::CPU;
  plan.src.executor_id = "close-race";
  plan.src.region_id = "source";
  plan.src.executor_address = reinterpret_cast<uintptr_t>(source.data());
  plan.dst = TransferRef::HostBytes(target.data(), target.size());
  plan.src_offsets = {0};
  plan.dst_offsets = {0};
  plan.sizes = {source.size()};
  plan.tags = {0};
  auto execute = std::async(std::launch::async, [&] {
    std::vector<TransferFailure> failures;
    return executor.ExecutePlans({plan}, &failures);
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1), [&] { return entered; }));
  }
  auto close = std::async(std::launch::async, [&] { executor.Close(); });
  EXPECT_EQ(close.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
  }
  cv.notify_all();
  EXPECT_EQ(execute.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(close.wait_for(std::chrono::seconds(1)), std::future_status::ready);
}

TEST(EnginePoolLoopTest, ClosedExecutorReportsEveryTagAsNeverSubmitted) {
  engine::EngineExecutor executor({"closed-executor", -1});
  executor.Close();
  EXPECT_TRUE(executor.PackedEngineDesc().empty());
  TransferPlan plan;
  std::array<uint8_t, 16> source{};
  std::array<uint8_t, 16> target{};
  plan.src = TransferRef::HostBytes(source.data(), source.size());
  plan.dst = TransferRef::HostBytes(target.data(), target.size());
  plan.src_offsets = {0};
  plan.dst_offsets = {0};
  plan.sizes = {source.size()};
  plan.tags = {7, 9};
  std::vector<TransferFailure> failures;
  EXPECT_FALSE(executor.ExecutePlans({plan}, &failures));
  ASSERT_EQ(failures.size(), 1u);
  EXPECT_EQ(failures[0].tags, (std::vector<size_t>{7, 9}));
  EXPECT_TRUE(failures[0].quiesced);
  EXPECT_FALSE(failures[0].submitted);
}

class PoolEngineControl final : public engine::EngineControl {
 public:
  explicit PoolEngineControl(PoolClient* client) : client_(client) {}
  bool RegisterEngine(EngineRegistration registration) override {
    return client_ != nullptr && client_->ActivateEngine(std::move(registration));
  }
  std::optional<uint32_t> BackendId(engine::RankId rank) const override {
    return client_ == nullptr ? std::nullopt : client_->EngineBackendId(rank);
  }
  std::vector<engine::SharedDramRegion> ExportSharedDram(engine::PoolEpoch epoch) override {
    return client_ == nullptr ? std::vector<engine::SharedDramRegion>{}
                              : client_->ExportSharedDram(epoch);
  }
  PeerPool* Pool() const override { return client_ == nullptr ? nullptr : client_->EnginePool(); }
  engine::ReadPlan Lookup(const std::vector<std::string>& keys, std::string name_space,
                          std::string layout_id, uint32_t requester_backend_id) override {
    return client_ == nullptr ? engine::ReadPlan{}
                              : client_->LookupEngine(keys, std::move(name_space),
                                                      std::move(layout_id), requester_backend_id);
  }
  engine::PreparedLoad PrepareLoad(const engine::ReadPlan& plan, std::string request_id,
                                   const std::vector<engine::AllocationHandle>& targets,
                                   uint32_t n_pages, const ReaderIdentity& reader,
                                   engine::OperationId operation) override {
    return client_ == nullptr ? engine::PreparedLoad{}
                              : client_->PrepareEngineLoad(plan, std::move(request_id), targets,
                                                           n_pages, reader, operation);
  }
  engine::CancelLoadResult CancelLoad(uint64_t prepared_id,
                                      engine::OperationId operation) override {
    return client_ == nullptr ? engine::CancelLoadResult::kRejected
                              : client_->CancelEngineLoad(prepared_id, operation);
  }
  bool CompleteLoad(uint64_t prepared_id, engine::LoadOutcome outcome,
                    engine::OperationId operation) override {
    return client_ != nullptr && client_->CompleteEngineLoad(prepared_id, outcome, operation);
  }
  void SessionFailed(engine::RankId rank) override {
    if (client_ != nullptr) client_->FailEngineSession(rank);
  }

 private:
  PoolClient* client_;
};

class EngineTwoNodeLoop : public ::testing::Test {
 protected:
  void SetUp() override {
    MasterServerConfig master_config;
    master_config.listen_address = "0.0.0.0:0";
    master_config.registry_config.heartbeat_ttl = std::chrono::seconds(1);
    master_ = std::make_unique<MasterServer>(std::move(master_config));
    master_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 100 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);
    const std::string master_address = "localhost:" + std::to_string(master_->GetBoundPort());

    PoolClientConfig first;
    first.master_config.node_id = "engine-node";
    first.master_config.node_address = "127.0.0.1";
    first.master_config.master_address = master_address;
    first.io_engine.host = "0.0.0.0";
    first.io_engine.port = 0;
    first.auto_peer_service_port = true;
    first.dram.buffer_sizes = {1 << 20};
    first.dram.shared = true;
    first.dram.use_hugepages = false;
    first.dram.prefault = false;
    first.dram_page_size = 4096;
    first.engine_hbm_ranks = 8;
    first.eviction_mode = PeerEvictionMode::kMaster;
    first_ = std::make_unique<PoolClient>(std::move(first));
    ASSERT_TRUE(first_->Init());

    PoolClientConfig second;
    second.master_config.node_id = "reader-node";
    second.master_config.node_address = "127.0.0.1";
    second.master_config.master_address = master_address;
    second.io_engine.host = "0.0.0.0";
    second.io_engine.port = 0;
    second.auto_peer_service_port = true;
    second.dram.buffer_sizes = {1 << 20};
    second.dram.shared = true;
    second.dram.use_hugepages = false;
    second.dram.prefault = false;
    second.dram_page_size = 4096;
    second.engine_hbm_ranks = 8;
    second.eviction_mode = PeerEvictionMode::kMaster;
    second_ = std::make_unique<PoolClient>(std::move(second));
    ASSERT_TRUE(second_->Init());

    read_buffer_.resize(4096);
    ASSERT_TRUE(second_->RegisterMemory(read_buffer_.data(), read_buffer_.size()));

    const std::string grpc_address =
        "unix:///tmp/umbp_engine_two_node_" + std::to_string(getpid()) + ".grpc.sock";
    engine_address_ = grpc_address;
    control_ = std::make_unique<PoolEngineControl>(first_.get());
    sessions_ = std::make_unique<engine::EngineSessionServer>(control_.get());
    std::string error;
    ASSERT_TRUE(sessions_->Start(standalone::DeriveEngineSocketPath(grpc_address), &error))
        << error;
  }

  void TearDown() override {
    if (engine_client_) engine_client_->Close();
    if (sessions_) sessions_->Shutdown();
    if (second_) second_->Shutdown();
    if (first_) first_->Shutdown();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
    if (!engine_address_.empty()) {
      unlink(standalone::DeriveEngineSocketPath(engine_address_).c_str());
    }
  }

  static bool WaitUntil(const std::function<bool()>& predicate,
                        std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (predicate()) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return predicate();
  }

  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> first_;
  std::unique_ptr<PoolClient> second_;
  std::unique_ptr<PoolEngineControl> control_;
  std::unique_ptr<engine::EngineSessionServer> sessions_;
  std::unique_ptr<engine::EngineClient> engine_client_;
  std::unique_ptr<engine::EngineExecutor> executor_;
  std::vector<uint8_t> hbm_;
  std::vector<uint8_t> read_buffer_;
  std::string engine_address_;
};

TEST_F(EngineTwoNodeLoop, HandoffAndDramLoadReachMasterThroughProductionReadQueue) {
  auto* pool = first_->EnginePool();
  const std::string key = "handoff-master-queue";
  std::array<uint8_t, 128> bytes{};
  EngineRegistration registration;
  registration.pool_id = "recency-pool";
  registration.owner_rank = 0;
  registration.geometry.page_tokens = 1;
  registration.geometry.num_pages = 1;
  registration.geometry.object_bytes = 64;
  registration.geometry.regions = {
      {"kv", TransferRef::HostBytes(bytes.data(), bytes.size()), 64, 1, 0, 0, "kv"}};
  registration.geometry.layout_id = engine::ComputeLayoutId(registration.geometry);
  ASSERT_TRUE(first_->ActivateEngine(registration));
  const auto backend_id = *first_->EngineBackendId(0);
  engine::OperationId op;
  op.session_id[0] = 93;
  op.seq = 1;
  auto pages = pool->DelegateEnginePages(backend_id, 0, 1, op);
  ASSERT_TRUE(pages.success);
  ++pages.handles[0].generation;
  ++op.seq;
  ASSERT_TRUE(pool->PublishResident(backend_id, {{key, {}, pages.handles[0]}}, op).success);
  first_->Master().FlushHeartbeat();
  const auto* store = master_->MetadataStoreForTest();
  ASSERT_TRUE(WaitUntil([&] { return store->LookupBlock(key).size() == 1; }));
  const uint32_t dram_id = first_->Backends().BackendId(first_->Backends().Get(TierType::DRAM));
  auto allocated = first_->Backends().Get(dram_id)->BatchAllocate({{key, 64}});
  ASSERT_EQ(allocated[0].outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(first_->Backends().Get(dram_id)->BatchCommit({{allocated[0].slot_id, key}})[0].success);
  pool->RefreshBackendResidencies(backend_id);
  first_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitUntil([&] { return store->LookupBlock(key).size() == 2; }));
  const NodeTierKey bucket{"engine-node", TierType::DRAM};
  auto timestamp = [&]() {
    const auto all = store->EnumerateEvictionCandidates(
        {bucket}, EvictionOrder::kLeastRecentlyAccessed, 100, std::chrono::system_clock::now());
    auto found = all.find(bucket);
    if (found != all.end()) {
      for (const auto& candidate : found->second) {
        if (candidate.key == key) return candidate.last_accessed_at;
      }
    }
    return std::chrono::system_clock::time_point{};
  };
  const auto old = timestamp();
  ASSERT_NE(old, std::chrono::system_clock::time_point{});
  ++op.seq;
  const auto ticket = pool->RetireEnginePages(backend_id, pages.handles, op);
  ASSERT_TRUE(WaitUntil([&] { return !pool->PollReadyEngineRetires(backend_id).empty(); }));
  ASSERT_TRUE(pool->AcknowledgeEngineReleased(backend_id, ticket.id));
  first_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitUntil([&] { return timestamp() > old; }))
      << "successful handoff must reach the in-memory master store via NoteLocalReads/heartbeat";
  const auto after_handoff = timestamp();
  ++op.seq;
  auto grant = pool->AcquireLoadSources({"reader", "request"}, {key}, {dram_id},
                                       {TierType::DRAM}, op, GrantTable::Clock::now() + std::chrono::seconds(10));
  ASSERT_TRUE(grant[0].granted);
  first_->Master().FlushHeartbeat();
  EXPECT_TRUE(WaitUntil([&] { return timestamp() > after_handoff; }))
      << "actual DRAM grant must use the same production master report queue";
}

TEST_F(EngineTwoNodeLoop, MasterDiscoversRetiredReplicaAndSecondNodeRunsRemoteRead) {
  constexpr size_t kBytes = 4096;
  // Cache GetPeerInfo before the rank registers. Ordinary DRAM reads must keep
  // using that snapshot instead of issuing the metadata RPC on every use.
  std::vector<uint8_t> preconnect(kBytes, 0x3d);
  ASSERT_TRUE(first_->RegisterMemory(preconnect.data(), preconnect.size()));
  ASSERT_TRUE(first_->Put("pre-engine-dram-key", preconnect.data(), preconnect.size()))
      << "ordinary Put on an engine node must enter DRAM";
  first_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitUntil([&] { return second_->Exists("pre-engine-dram-key"); }));
  std::optional<RouteGetResult> preconnect_route;
  ASSERT_TRUE(WaitUntil([&] {
    preconnect_route.reset();
    return second_->Master().RouteGet("pre-engine-dram-key", {}, &preconnect_route).ok() &&
           preconnect_route.has_value();
  }));
  ASSERT_TRUE(
      second_->RefreshPeerInfoForTest(preconnect_route->node_id, preconnect_route->peer_address));
  ASSERT_TRUE(
      WaitUntil([&] { return second_->Get("pre-engine-dram-key", read_buffer_.data(), kBytes); }));
  ASSERT_EQ(second_->PeerInfoRefreshCountForTest(), 1u);

  hbm_.resize(3 * kBytes);
  executor_ = std::make_unique<engine::EngineExecutor>(
      engine::EngineExecutorConfig{"two-node-executor", -1});
  ASSERT_TRUE(executor_->RegisterOriginalRegion(
      "kv",
      TransferRef::HostBytes(hbm_.data(), hbm_.size(), mori::io::MemoryLocationType::CPU, -1)));
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 2;
  geometry.object_bytes = kBytes;
  geometry.regions.push_back(engine::PageRegion{
      "kv", TransferRef::HostBytes(hbm_.data(), hbm_.size(), mori::io::MemoryLocationType::CPU, -1),
      kBytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);

  engine_client_ = std::make_unique<engine::EngineClient>(
      engine::EngineClientConfig{engine_address_, "two-node-executor", 500, 2000});
  engine_client_->AttachExecutor(executor_.get());
  engine::PoolEpoch epoch;
  epoch.boot_id[0] = 0x72;
  auto registration = engine_client_->RegisterPool(0, "two-node-pool", epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  std::string map_error;
  ASSERT_TRUE(executor_->MapSharedDram(registration.dram_regions, &map_error)) << map_error;

  std::atomic<int> execute_batches{0};
  executor_->SetBeforeExecuteHookForTest([&] { ++execute_batches; });
  std::vector<std::shared_ptr<engine::DispatchTicket>> coalesced;
  for (int i = 0; i < 4; ++i) {
    coalesced.push_back(engine::EngineDispatchRegistry::Global().Submit("two-node-executor", {}));
  }
  for (auto& ticket : coalesced) {
    std::vector<TransferFailure> failures;
    ASSERT_TRUE(ticket->WaitFor(std::chrono::seconds(1), &failures));
    EXPECT_TRUE(failures.empty());
  }
  EXPECT_GT(execute_batches.load(), 0);
  EXPECT_LT(execute_batches.load(), 4)
      << "the rank progress loop must still coalesce a burst after Serve drops its fixed sleep";
  executor_->SetBeforeExecuteHookForTest({});

  auto reservation = engine_client_->Reserve("two-node-request", 1, 0);
  ASSERT_TRUE(reservation.success);
  auto taken = engine_client_->Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(taken.size(), 1u);
  ASSERT_TRUE(engine_client_->Refund(reservation.id));
  auto handle = taken.front();
  for (size_t i = 0; i < kBytes; ++i) {
    hbm_[handle.page_id * kBytes + i] = static_cast<uint8_t>(i * 17 + 3);
  }
  const std::string key = "engine-two-node-key";
  ASSERT_TRUE(engine_client_->Publish({ResidentRecord{key, {}, handle}}).success);
  auto* wrapped =
      dynamic_cast<InstrumentedBackend*>(first_->Backends().Get(EngineHbmBackendName(0)));
  ASSERT_NE(wrapped, nullptr);
  auto* engine_backend = dynamic_cast<EngineHbmBackend*>(wrapped->Inner());
  ASSERT_NE(engine_backend, nullptr);
  ASSERT_TRUE(WaitUntil([&] {
    const auto history = engine_backend->HistoryStats();
    return history.delegate_records == 0 && history.publish_records == 0;
  })) << "heartbeat ACK watermark did not reclaim completed operation records";
  first_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitUntil([&] { return second_->Exists(key); }));

  std::optional<RouteGetResult> route;
  ASSERT_TRUE(WaitUntil([&] {
    route.reset();
    return second_->Master().RouteGet(key, {}, &route).ok() && route.has_value();
  }));
  auto late_backend = first_->EngineBackendId(0);
  ASSERT_TRUE(late_backend.has_value());
  ASSERT_TRUE(
      second_->EnsurePeerBackendForTest(route->node_id, route->peer_address, *late_backend));
  ASSERT_EQ(second_->PeerInfoRefreshCountForTest(), 2u)
      << "an unknown late executor/backend must trigger exactly one production refresh";
  auto peer = ::umbp::UMBPPeer::NewStub(
      grpc::CreateChannel(route->peer_address, grpc::InsecureChannelCredentials()));
  ::umbp::GetPeerInfoRequest peer_request;
  ::umbp::GetPeerInfoResponse peer_response;
  grpc::ClientContext peer_context;
  ASSERT_TRUE(peer->GetPeerInfo(&peer_context, peer_request, &peer_response).ok());
  ASSERT_EQ(peer_response.executor_infos_size(), 1);
  EXPECT_EQ(peer_response.executor_infos(0).executor_id(), "two-node-executor");
  ASSERT_EQ(peer_response.backend_geometries_size(), 1);
  EXPECT_EQ(peer_response.backend_geometries(0).layout_id(), geometry.layout_id);
  EXPECT_EQ(peer_response.protocol_capabilities() & 0x7, 0x7u);

  ::umbp::BatchAcquireEngineRequest acquire_request;
  acquire_request.set_reader_session("reader-session");
  acquire_request.set_reader_request("reader-request");
  acquire_request.mutable_operation()->set_session_id(std::string(16, 'r'));
  acquire_request.mutable_operation()->set_seq(1);
  acquire_request.add_keys(key);
  acquire_request.set_lease_ms(1000);
  ::umbp::BatchAcquireEngineResponse acquire_response;
  grpc::ClientContext acquire_context;
  ASSERT_TRUE(peer->BatchAcquireEngine(&acquire_context, acquire_request, &acquire_response).ok());
  ASSERT_EQ(acquire_response.grants_size(), 1);
  ASSERT_TRUE(acquire_response.grants(0).granted());
  ASSERT_EQ(acquire_response.grants(0).view().spans_size(), 1);
  EXPECT_EQ(acquire_response.grants(0).view().spans(0).buffer().executor_id(), "two-node-executor");

  ::umbp::BatchReleaseEngineRequest release_request;
  release_request.set_reader_session("reader-session");
  release_request.set_reader_request("reader-request");
  release_request.mutable_operation()->set_session_id(std::string(16, 'r'));
  release_request.mutable_operation()->set_seq(2);
  release_request.mutable_ack_watermark()->set_session_id(std::string(16, 'r'));
  release_request.mutable_ack_watermark()->set_seq(1);
  release_request.set_backend_id(acquire_response.grants(0).backend_id());
  release_request.add_grant_ids(acquire_response.grants(0).grant_id());
  release_request.set_proof(static_cast<uint32_t>(engine::CompletionProof::kCompleted));
  ::umbp::BatchReleaseEngineResponse release_response;
  grpc::ClientContext release_context;
  ASSERT_TRUE(peer->BatchReleaseEngine(&release_context, release_request, &release_response).ok());
  ASSERT_EQ(release_response.released_size(), 1);
  EXPECT_TRUE(release_response.released(0));
  ::umbp::BatchReleaseEngineRequest invalid_release = release_request;
  invalid_release.mutable_operation()->set_seq(3);
  invalid_release.set_proof(999);
  ::umbp::BatchReleaseEngineResponse invalid_response;
  grpc::ClientContext invalid_context;
  auto invalid_status =
      peer->BatchReleaseEngine(&invalid_context, invalid_release, &invalid_response);
  EXPECT_EQ(invalid_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  const auto retire_started = std::chrono::steady_clock::now();
  ASSERT_TRUE(engine_client_->ReturnPages({handle}));
  std::array<uint8_t, kBytes> local_read{};
  MediumBackend* debug_dram = nullptr;
  ASSERT_TRUE(WaitUntil([&] {
    for (auto* medium : first_->Backends().All()) {
      if (medium != nullptr && medium->Tier() == TierType::DRAM && medium->Contains(key)) {
        debug_dram = medium;
        return true;
      }
    }
    return false;
  })) << "F3 did not commit a DRAM replica; state="
      << static_cast<int>(engine_backend->State(handle.page_id).value_or(engine::SlotState::kFree))
      << " retiring=" << engine_client_->Accounting().retiring
      << " quarantined=" << engine_client_->Accounting().quarantined;
  ASSERT_TRUE(WaitUntil([&] { return first_->Get(key, local_read.data(), local_read.size()); }));
  EXPECT_LT(std::chrono::steady_clock::now() - retire_started, std::chrono::milliseconds(300))
      << "node-pushed execution must not wait for the 500 ms heartbeat";
  ASSERT_EQ(std::memcmp(local_read.data(), hbm_.data() + handle.page_id * kBytes, kBytes), 0);

  std::vector<engine::RetireTicket> ready;
  ASSERT_TRUE(WaitUntil([&] {
    ready = engine_client_->PollRetires();
    return !ready.empty();
  }));
  MediumBackend* preserved_dram = nullptr;
  for (auto* medium : first_->Backends().All()) {
    if (medium != nullptr && medium->Tier() == TierType::DRAM && medium->Contains(key)) {
      preserved_dram = medium;
      break;
    }
  }
  ASSERT_NE(preserved_dram, nullptr);
  EXPECT_TRUE(preserved_dram->Contains(key));
  EXPECT_TRUE(WaitUntil([&] {
    return engine_backend->State(handle.page_id) == engine::SlotState::kDelegated;
  })) << "the rank progress thread must consume Ready and ACK Released in the background";
  ASSERT_TRUE(engine_client_->AcknowledgeReleased({ready.front().id}));
  first_->Master().FlushHeartbeat();

  std::fill(read_buffer_.begin(), read_buffer_.end(), 0);
  // MORI_IO_SIM_ENABLE validates routing, peer bootstrap, resolve and transfer
  // completion but deliberately does not mutate the destination buffer (the
  // existing CrossNodeSmoke has the same CPU-sim boundary). The local engine
  // executor byte check above covers the HBM->DRAM copy; real RDMA bytes remain
  // in the approved GPU/network validation list.
  ASSERT_TRUE(WaitUntil([&] { return second_->Get(key, read_buffer_.data(), kBytes); }));
  EXPECT_EQ(second_->PeerInfoRefreshCountForTest(), 2u)
      << "healthy cached metadata must not be refreshed on every peer use";

  auto notice_reservation = engine_client_->Reserve("node-quarantine-notice", 1, 0);
  ASSERT_TRUE(notice_reservation.success);
  auto notice_page =
      engine_client_->Take(notice_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(notice_page.size(), 1u);
  ASSERT_TRUE(engine_client_->Refund(notice_reservation.id));
  ASSERT_TRUE(
      engine_client_->Publish({ResidentRecord{"node-quarantine-notice", {}, notice_page.front()}})
          .success);
  ASSERT_TRUE(engine_backend->Quarantine(notice_page));
  ASSERT_TRUE(WaitUntil([&] { return engine_client_->Accounting().quarantined == 1; }))
      << "node-originated quarantine must be delivered to the owning rank";
  auto quarantine_events = engine_client_->PollRetires();
  ASSERT_FALSE(quarantine_events.empty());
  EXPECT_TRUE(std::any_of(
      quarantine_events.begin(), quarantine_events.end(),
      [](const auto& event) { return event.state == engine::RetireState::kQuarantined; }));

  auto terminal_reservation = engine_client_->Reserve("terminal-pin", 1, 0);
  ASSERT_TRUE(terminal_reservation.success);
  auto terminal_page =
      engine_client_->Take(terminal_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(terminal_page.size(), 1u);
  ASSERT_TRUE(engine_client_->Refund(terminal_reservation.id));
  const std::string terminal_key = "terminal-pin-key";
  ASSERT_TRUE(
      engine_client_->Publish({ResidentRecord{terminal_key, {}, terminal_page.front()}}).success);
  std::mutex terminal_mutex;
  std::condition_variable terminal_cv;
  bool terminal_ack_waiting = false;
  bool release_terminal_ack = false;
  engine_client_->SetBeforeCallSendHookForTest([&](const ::umbp::engine::EngineFrame& request) {
    if (!request.has_acknowledge_released()) return;
    std::unique_lock<std::mutex> lock(terminal_mutex);
    terminal_ack_waiting = true;
    terminal_cv.notify_all();
    terminal_cv.wait(lock, [&] { return release_terminal_ack; });
  });
  ASSERT_TRUE(engine_client_->ReturnPages(terminal_page));
  {
    std::unique_lock<std::mutex> lock(terminal_mutex);
    ASSERT_TRUE(
        terminal_cv.wait_for(lock, std::chrono::seconds(5), [&] { return terminal_ack_waiting; }));
  }
  std::vector<engine::RetireTicket> terminal_ready;
  terminal_ready = engine_client_->PollRetires();
  ASSERT_EQ(terminal_ready.size(), 1u);
  ASSERT_EQ(terminal_ready[0].state, engine::RetireState::kReady);
  MediumBackend* terminal_dram = nullptr;
  for (auto* medium : first_->Backends().All()) {
    if (medium != nullptr && medium->Tier() == TierType::DRAM && medium->Contains(terminal_key)) {
      terminal_dram = medium;
      break;
    }
  }
  ASSERT_NE(terminal_dram, nullptr);
  EXPECT_EQ(terminal_dram->Evict({terminal_key}).front().bytes_freed, 0u);
  ASSERT_TRUE(engine_backend->Quarantine(terminal_page));
  {
    std::lock_guard<std::mutex> lock(terminal_mutex);
    release_terminal_ack = true;
  }
  terminal_cv.notify_all();
  engine_client_->SetBeforeCallSendHookForTest({});
  ASSERT_TRUE(WaitUntil([&] {
    auto events = engine_client_->PollRetires();
    return std::any_of(events.begin(), events.end(), [](const auto& event) {
      return event.state == engine::RetireState::kQuarantined;
    });
  })) << "the terminal Quarantined ticket must clear the client's pending Released state";
  EXPECT_GT(terminal_dram->Evict({terminal_key}).front().bytes_freed, 0u)
      << "a terminal quarantined ticket has no future ACK, so its DRAM pin must be released";
  first_->DeregisterMemory(preconnect.data());
}

TEST_F(EngineTwoNodeLoop, LookupPrepareAndLayeredLoadUsesHbmThenDram) {
  constexpr size_t kHalf = 32;
  constexpr size_t kBytes = 2 * kHalf;
  constexpr uint32_t kPages = 4;
  std::array<std::vector<uint8_t>, 2> source_regions{std::vector<uint8_t>((kPages + 1) * kHalf),
                                                     std::vector<uint8_t>((kPages + 1) * kHalf)};
  std::array<std::vector<uint8_t>, 2> target_regions{std::vector<uint8_t>((kPages + 1) * kHalf),
                                                     std::vector<uint8_t>((kPages + 1) * kHalf)};
  engine::EngineExecutor source_executor({"load-source-executor", -1});
  engine::EngineExecutor target_executor({"load-target-executor", -1});
  ASSERT_TRUE(source_executor.RegisterOriginalRegion(
      "layer-0", TransferRef::HostBytes(source_regions[0].data(), source_regions[0].size())));
  ASSERT_TRUE(source_executor.RegisterOriginalRegion(
      "layer-1", TransferRef::HostBytes(source_regions[1].data(), source_regions[1].size())));
  ASSERT_TRUE(target_executor.RegisterOriginalRegion(
      "layer-0", TransferRef::HostBytes(target_regions[0].data(), target_regions[0].size())));
  ASSERT_TRUE(target_executor.RegisterOriginalRegion(
      "layer-1", TransferRef::HostBytes(target_regions[1].data(), target_regions[1].size())));
  ASSERT_TRUE(target_executor.MapPeerRegion(
      "load-source-executor", "layer-0",
      TransferRef::HostBytes(source_regions[0].data(), source_regions[0].size())));
  ASSERT_TRUE(target_executor.MapPeerRegion(
      "load-source-executor", "layer-1",
      TransferRef::HostBytes(source_regions[1].data(), source_regions[1].size())));

  auto geometry_for = [&](auto& regions) {
    engine::PageGeometry geometry;
    geometry.page_tokens = 1;
    geometry.num_pages = kPages;
    geometry.object_bytes = kBytes;
    geometry.regions = {
        {"layer-0", TransferRef::HostBytes(regions[0].data(), regions[0].size()), kHalf, 1, 0, 0,
         "kv"},
        {"layer-1", TransferRef::HostBytes(regions[1].data(), regions[1].size()), kHalf, 1, kHalf,
         1, "kv"},
    };
    geometry.layout_id = engine::ComputeLayoutId(geometry);
    return geometry;
  };
  auto source_geometry = geometry_for(source_regions);
  auto target_geometry = geometry_for(target_regions);
  ASSERT_EQ(source_geometry.layout_id, target_geometry.layout_id);

  engine::EngineClient source_client({engine_address_, "load-source-executor", 10, 1000});
  engine::EngineClient target_client({engine_address_, "load-target-executor", 10, 1000});
  source_client.AttachExecutor(&source_executor);
  target_client.AttachExecutor(&target_executor);
  engine::PoolEpoch epoch;
  auto source_registration =
      source_client.RegisterPool(0, "layer-load-pool", epoch, source_geometry);
  ASSERT_TRUE(source_registration.success) << source_registration.error;
  auto target_registration =
      target_client.RegisterPool(1, "layer-load-pool", epoch, target_geometry);
  ASSERT_TRUE(target_registration.success) << target_registration.error;
  std::string map_error;
  ASSERT_TRUE(target_executor.MapSharedDram(target_registration.dram_regions, &map_error))
      << map_error;

  auto source_reservation = source_client.Reserve("source", 1, 0);
  ASSERT_TRUE(source_reservation.success);
  auto source_page =
      source_client.Take(source_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(source_page.size(), 1u);
  ASSERT_TRUE(source_client.Refund(source_reservation.id));
  for (size_t i = 0; i < kHalf; ++i) {
    source_regions[0][source_page[0].page_id * kHalf + i] = static_cast<uint8_t>(0x20 + i);
    source_regions[1][source_page[0].page_id * kHalf + i] = static_cast<uint8_t>(0x80 + i);
  }
  const std::string hbm_key = "load-hbm";
  ASSERT_TRUE(source_client.Publish({ResidentRecord{hbm_key, {}, source_page[0]}}).success);

  std::array<uint8_t, kBytes> dram_bytes{};
  for (size_t i = 0; i < dram_bytes.size(); ++i) dram_bytes[i] = static_cast<uint8_t>(i + 3);
  const std::string dram_key = "load-dram";
  PoolPlacementRequest dram_request;
  dram_request.key = dram_key;
  dram_request.size = dram_bytes.size();
  dram_request.tier = TierType::DRAM;
  auto dram_allocation = first_->EnginePool()->BatchAllocate({dram_request}).front();
  ASSERT_EQ(dram_allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(ValidateObjectView(dram_allocation.allocation.view));
  for (const auto& span : dram_allocation.allocation.view.spans) {
    ASSERT_TRUE(span.buffer.HasHostPtr());
    std::memcpy(static_cast<uint8_t*>(span.buffer.host_ptr) + span.offset,
                dram_bytes.data() + span.logical_offset, span.length);
  }
  ASSERT_TRUE(first_->EnginePool()
                  ->BatchCommit({PoolCommitRequest{
                      {dram_allocation.backend_id, dram_allocation.allocation.slot_id}, dram_key}})
                  .front()
                  .commit.success);
  MediumBackend* hbm_shadow_dram = nullptr;
  for (auto* medium : first_->Backends().All()) {
    if (medium != nullptr && medium->Tier() == TierType::DRAM) {
      hbm_shadow_dram = medium;
      break;
    }
  }
  ASSERT_NE(hbm_shadow_dram, nullptr);
  auto shadow = hbm_shadow_dram->BatchAllocate({AllocateRequest{hbm_key, kBytes}}).front();
  ASSERT_EQ(shadow.outcome, AllocateOutcome::kSuccessAllocated);
  for (const auto& span : shadow.view.spans) {
    std::memset(static_cast<uint8_t*>(span.buffer.host_ptr) + span.offset, 0xee, span.length);
  }
  ASSERT_TRUE(
      hbm_shadow_dram->BatchCommit({CommitRequest{shadow.slot_id, hbm_key}}).front().success);

  auto plan = target_client.Lookup({hbm_key, dram_key, "missing"}, "model@rev/layout");
  ASSERT_TRUE(plan.success) << plan.error;
  ASSERT_EQ(plan.sources.size(), 3u);
  EXPECT_EQ(plan.sources[0].kind, engine::ReadSourceKind::kLocalHbm);
  EXPECT_EQ(plan.sources[1].kind, engine::ReadSourceKind::kLocalDram);
  EXPECT_EQ(plan.sources[2].kind, engine::ReadSourceKind::kMissing);
  EXPECT_EQ(plan.valid_prefix, 2u);
  EXPECT_EQ(plan.legal_boundaries, std::vector<uint32_t>({1, 2}));

  auto target_reservation = target_client.Reserve("target", 2, 0);
  ASSERT_TRUE(target_reservation.success);
  auto targets = target_client.Take(target_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_TRUE(target_client.Refund(target_reservation.id));
  auto prepared = target_client.PrepareLoad(plan, "request-layer-load", targets, 2);
  ASSERT_TRUE(prepared.success) << prepared.error;
  ASSERT_EQ(prepared.layers.size(), 2u);
  EXPECT_EQ(prepared.layers[0].layer_id, 0);
  EXPECT_EQ(prepared.layers[1].layer_id, 1);
  auto load = target_executor.SubmitLoad(prepared);
  ASSERT_NE(load, nullptr);
  EXPECT_EQ(load->Wait(), engine::LoadOutcome::kCompleted);
  EXPECT_EQ(load->layer_count(), 2u);
  for (size_t layer = 0; layer < load->layer_count(); ++layer) {
    EXPECT_TRUE(load->layer_done(layer));
    EXPECT_EQ(load->layer_outcome(layer), engine::LoadOutcome::kCompleted);
  }
  EXPECT_EQ(std::memcmp(target_regions[0].data() + targets[0].page_id * kHalf,
                        source_regions[0].data() + source_page[0].page_id * kHalf, kHalf),
            0);
  EXPECT_EQ(std::memcmp(target_regions[1].data() + targets[0].page_id * kHalf,
                        source_regions[1].data() + source_page[0].page_id * kHalf, kHalf),
            0);
  EXPECT_EQ(
      std::memcmp(target_regions[0].data() + targets[1].page_id * kHalf, dram_bytes.data(), kHalf),
      0);
  EXPECT_EQ(std::memcmp(target_regions[1].data() + targets[1].page_id * kHalf,
                        dram_bytes.data() + kHalf, kHalf),
            0);
  auto completions = target_executor.PollLoads();
  ASSERT_EQ(completions.size(), 1u);
  EXPECT_EQ(completions[0].first, prepared.id);
  EXPECT_EQ(completions[0].second, engine::LoadOutcome::kCompleted);
  EXPECT_TRUE(target_client.ReturnPages(targets));

  auto grouped_source_reservation = source_client.Reserve("grouped-sources", 3, 0);
  ASSERT_TRUE(grouped_source_reservation.success);
  auto grouped_sources =
      source_client.Take(grouped_source_reservation.id, engine::ReservationPurpose::kLoad, 3);
  ASSERT_EQ(grouped_sources.size(), 3u);
  ASSERT_TRUE(source_client.Refund(grouped_source_reservation.id));
  ASSERT_TRUE(source_client
                  .Publish({{"grouped-a", {}, grouped_sources[0]},
                            {"grouped-b", {}, grouped_sources[1]},
                            {"grouped-c", {}, grouped_sources[2]}})
                  .success);
  auto grouped_plan =
      target_client.Lookup({"grouped-a", "grouped-b", "grouped-c"}, "model@rev/layout");
  ASSERT_EQ(grouped_plan.valid_prefix, 3u);
  auto grouped_target_reservation = target_client.Reserve("grouped-targets", 3, 0);
  ASSERT_TRUE(grouped_target_reservation.success);
  auto grouped_targets =
      target_client.Take(grouped_target_reservation.id, engine::ReservationPurpose::kLoad, 3);
  ASSERT_TRUE(target_client.Refund(grouped_target_reservation.id));
  auto grouped_prepared =
      target_client.PrepareLoad(grouped_plan, "grouped-targets", grouped_targets, 3);
  ASSERT_TRUE(grouped_prepared.success) << grouped_prepared.error;
  ASSERT_EQ(grouped_prepared.layers.size(), 2u);
  for (const auto& layer : grouped_prepared.layers) {
    ASSERT_EQ(layer.plans.size(), 1u) << "same endpoint pair must be grouped once per layer";
    EXPECT_EQ(layer.plans[0].sizes.size(), 3u)
        << "per-page spans belong in one indexed plan, not one plan per page";
  }
  auto grouped_handle = target_executor.SubmitLoad(grouped_prepared);
  ASSERT_EQ(grouped_handle->Wait(), engine::LoadOutcome::kCompleted);
  ASSERT_TRUE(target_client.ReturnPages(grouped_targets));
  ASSERT_TRUE(source_client.ReturnPages(grouped_sources));
  ASSERT_TRUE(WaitUntil([&] { return source_client.Accounting().cached >= 3; }));
  (void)source_client.PollRetires();

  auto event_failure_plan = target_client.Lookup({hbm_key}, "model@rev/layout");
  ASSERT_TRUE(event_failure_plan.success);
  auto event_failure_reservation = target_client.Reserve("event-create-failure", 1, 0);
  ASSERT_TRUE(event_failure_reservation.success);
  auto event_failure_target =
      target_client.Take(event_failure_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(event_failure_target.size(), 1u);
  ASSERT_TRUE(target_client.Refund(event_failure_reservation.id));
  auto event_failure_prepared = target_client.PrepareLoad(
      event_failure_plan, "event-create-failure", event_failure_target, 1);
  ASSERT_TRUE(event_failure_prepared.success) << event_failure_prepared.error;
  target_executor.SetLoadEventCreateFailureForTest(true);
  auto event_failure = target_executor.SubmitLoad(event_failure_prepared);
  ASSERT_NE(event_failure, nullptr);
  EXPECT_EQ(event_failure->outcome(), engine::LoadOutcome::kFailed);
  for (size_t layer = 0; layer < event_failure->layer_count(); ++layer) {
    EXPECT_TRUE(event_failure->layer_done(layer));
    EXPECT_EQ(event_failure->layer_outcome(layer), engine::LoadOutcome::kFailed);
  }
  target_executor.SetLoadEventCreateFailureForTest(false);
  EXPECT_TRUE(target_client.ReturnPages(event_failure_target))
      << "event creation rejection must cancel the prepare and release its target hold";

  auto duplicate_target_plan = target_client.Lookup({hbm_key, dram_key}, "model@rev/layout");
  ASSERT_EQ(duplicate_target_plan.valid_prefix, 2u);
  auto duplicate_target_reservation = target_client.Reserve("duplicate-load-target", 1, 0);
  ASSERT_TRUE(duplicate_target_reservation.success);
  auto duplicate_target =
      target_client.Take(duplicate_target_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(duplicate_target_reservation.id));
  auto duplicate_rejected =
      target_client.PrepareLoad(duplicate_target_plan, "duplicate-load-target",
                                {duplicate_target.front(), duplicate_target.front()}, 2);
  EXPECT_FALSE(duplicate_rejected.success);
  EXPECT_TRUE(target_client.ReturnPages(duplicate_target));

  auto published_target_reservation = target_client.Reserve("published-load-target", 1, 0);
  ASSERT_TRUE(published_target_reservation.success);
  auto published_target =
      target_client.Take(published_target_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(published_target_reservation.id));
  ASSERT_TRUE(
      target_client.Publish({{"must-not-overwrite-published", {}, published_target.front()}})
          .success);
  auto published_target_plan = target_client.Lookup({hbm_key}, "model@rev/layout");
  ASSERT_TRUE(published_target_plan.success);
  auto published_rejected = target_client.PrepareLoad(published_target_plan,
                                                      "published-load-target", published_target, 1);
  EXPECT_FALSE(published_rejected.success)
      << "a load target already visible to readers must never be overwritten";
  EXPECT_TRUE(target_client.ReturnPages(published_target));

  auto cancel_race_plan = target_client.Lookup({hbm_key}, "model@rev/layout");
  auto cancel_race_reservation = target_client.Reserve("cancel-submit-race", 1, 0);
  ASSERT_TRUE(cancel_race_reservation.success);
  auto cancel_race_target =
      target_client.Take(cancel_race_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(cancel_race_reservation.id));
  auto cancel_race_prepared =
      target_client.PrepareLoad(cancel_race_plan, "cancel-submit-race", cancel_race_target, 1);
  ASSERT_TRUE(cancel_race_prepared.success);
  std::mutex cancel_race_mutex;
  std::condition_variable cancel_race_cv;
  bool cancel_at_send = false;
  bool release_cancel = false;
  target_client.SetBeforeCallSendHookForTest([&](const ::umbp::engine::EngineFrame& request) {
    if (!request.has_cancel_load()) return;
    std::unique_lock<std::mutex> lock(cancel_race_mutex);
    cancel_at_send = true;
    cancel_race_cv.notify_all();
    cancel_race_cv.wait(lock, [&] { return release_cancel; });
  });
  auto cancel_future = std::async(std::launch::async,
                                  [&] { return target_client.CancelLoad(cancel_race_prepared); });
  {
    std::unique_lock<std::mutex> lock(cancel_race_mutex);
    ASSERT_TRUE(
        cancel_race_cv.wait_for(lock, std::chrono::seconds(1), [&] { return cancel_at_send; }));
  }
  auto submit_future = std::async(std::launch::async,
                                  [&] { return target_executor.SubmitLoad(cancel_race_prepared); });
  {
    std::lock_guard<std::mutex> lock(cancel_race_mutex);
    release_cancel = true;
  }
  cancel_race_cv.notify_all();
  EXPECT_EQ(cancel_future.get(), engine::CancelLoadResult::kNeverSubmitted);
  auto cancel_race_handle = submit_future.get();
  ASSERT_NE(cancel_race_handle, nullptr);
  EXPECT_EQ(cancel_race_handle->outcome(), engine::LoadOutcome::kFailed);
  target_client.SetBeforeCallSendHookForTest({});
  EXPECT_TRUE(target_client.ReturnPages(cancel_race_target))
      << "cancel winning the ordered race must release the target hold";

  for (int cycle = 0; cycle < 8; ++cycle) {
    auto cycle_plan = target_client.Lookup({hbm_key}, "model@rev/layout");
    ASSERT_TRUE(cycle_plan.success);
    auto cycle_reservation = target_client.Reserve("grant-history-" + std::to_string(cycle), 1, 0);
    ASSERT_TRUE(cycle_reservation.success);
    auto cycle_target =
        target_client.Take(cycle_reservation.id, engine::ReservationPurpose::kLoad, 1);
    ASSERT_TRUE(target_client.Refund(cycle_reservation.id));
    auto cycle_prepared = target_client.PrepareLoad(
        cycle_plan, "grant-history-" + std::to_string(cycle), cycle_target, 1);
    ASSERT_TRUE(cycle_prepared.success) << cycle_prepared.error;
    auto cycle_handle = target_executor.SubmitLoad(cycle_prepared);
    ASSERT_EQ(cycle_handle->Wait(), engine::LoadOutcome::kCompleted);
    ASSERT_TRUE(target_client.ReturnPages(cycle_target));
  }
  auto* history_wrapped =
      dynamic_cast<InstrumentedBackend*>(first_->Backends().Get(EngineHbmBackendName(0)));
  auto* history_backend = history_wrapped == nullptr
                              ? nullptr
                              : dynamic_cast<EngineHbmBackend*>(history_wrapped->Inner());
  ASSERT_NE(history_backend, nullptr);
  const auto load_history = history_backend->HistoryStats().grants;
  EXPECT_LE(load_history.acquire_records, 2u);
  EXPECT_LE(load_history.release_records, 2u)
      << "load acquire/release replay records must be pruned by completion watermarks";

  auto abandoned_plan = target_client.Lookup({hbm_key}, "model@rev/layout");
  auto abandoned_reservation = target_client.Reserve("abandoned-prepare", 1, 0);
  ASSERT_TRUE(abandoned_reservation.success);
  auto abandoned_target =
      target_client.Take(abandoned_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(abandoned_reservation.id));
  auto abandoned =
      target_client.PrepareLoad(abandoned_plan, "abandoned-prepare", abandoned_target, 1);
  ASSERT_TRUE(abandoned.success);
  ASSERT_TRUE(target_client.ReturnPages(abandoned_target));
  EXPECT_GT(target_client.Accounting().pending_free, 0u);
  ASSERT_TRUE(WaitUntil([&] { return target_client.Accounting().pending_free == 0; },
                        std::chrono::seconds(7)))
      << "an unsubmitted prepare must be cancelled as NeverSubmitted after its TTL";

  auto cancel_plan = target_client.Lookup({hbm_key}, "model@rev/layout");
  ASSERT_TRUE(cancel_plan.success);
  auto cancel_reservation = target_client.Reserve("cancel-target", 1, 0);
  ASSERT_TRUE(cancel_reservation.success);
  auto cancel_target =
      target_client.Take(cancel_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(cancel_target.size(), 1u);
  ASSERT_TRUE(target_client.Refund(cancel_reservation.id));
  auto cancel_prepared =
      target_client.PrepareLoad(cancel_plan, "cancel-before-submit", cancel_target, 1);
  ASSERT_TRUE(cancel_prepared.success) << cancel_prepared.error;
  ASSERT_TRUE(source_client.ReturnPages(source_page));
  EXPECT_TRUE(source_client.PollRetires().empty())
      << "source Retire must wait for the prepared load grant";
  EXPECT_FALSE(target_client.QuiesceReset().ready)
      << "an unpublished load target hold must make reset Busy";

  auto resume_reservation = source_client.Reserve("busy-resumes-publish", 1, 0);
  ASSERT_TRUE(resume_reservation.success);
  auto resume_page =
      source_client.Take(resume_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(resume_page.size(), 1u);
  ASSERT_TRUE(source_client.Refund(resume_reservation.id));
  const uint64_t resume_ticket = source_client.QueuePublish(
      {ResidentRecord{"busy-resumes-publish", {}, resume_page.front()}}, false);
  ASSERT_NE(resume_ticket, 0u);
  auto busy_lease = source_client.QuiesceReset();
  EXPECT_FALSE(busy_lease.ready) << "the prepared-load grant keeps the source reset Busy";
  ASSERT_TRUE(source_client.MarkPublishReady(resume_ticket, engine::CompletionProof::kCompleted));
  ASSERT_TRUE(WaitUntil([&] {
    auto receipt = source_client.PublishStatus(resume_ticket);
    return receipt.has_value() && receipt->state == engine::PublishReceiptState::kAccepted;
  })) << "a Busy reset must resume, not cancel, an unsent publication";
  EXPECT_EQ(source_client.Accounting().published, 1u);

  EXPECT_EQ(target_client.CancelLoad(cancel_prepared), engine::CancelLoadResult::kNeverSubmitted);
  EXPECT_TRUE(target_client.ReturnPages(cancel_target));
  std::vector<engine::RetireTicket> canceled_source_ready;
  ASSERT_TRUE(WaitUntil([&] {
    canceled_source_ready = source_client.PollRetires();
    return !canceled_source_ready.empty();
  }));
  ASSERT_TRUE(source_client.AcknowledgeReleased({canceled_source_ready.front().id}));

  auto in_flight_plan = target_client.Lookup({dram_key}, "model@rev/layout");
  ASSERT_TRUE(in_flight_plan.success);
  auto in_flight_reservation = target_client.Reserve("in-flight-target", 1, 0);
  ASSERT_TRUE(in_flight_reservation.success);
  auto in_flight_target =
      target_client.Take(in_flight_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(in_flight_target.size(), 1u);
  ASSERT_TRUE(target_client.Refund(in_flight_reservation.id));
  auto in_flight_prepared =
      target_client.PrepareLoad(in_flight_plan, "cancel-after-submit", in_flight_target, 1);
  ASSERT_TRUE(in_flight_prepared.success) << in_flight_prepared.error;
  MediumBackend* held_dram = nullptr;
  for (auto* medium : first_->Backends().All()) {
    if (medium != nullptr && medium->Tier() == TierType::DRAM && medium->Contains(dram_key)) {
      held_dram = medium;
      break;
    }
  }
  ASSERT_NE(held_dram, nullptr);
  EXPECT_EQ(held_dram->Evict({dram_key}).front().bytes_freed, 0u)
      << "prepare_load must pin a DRAM source until physical completion";
  std::mutex load_mutex;
  std::condition_variable load_cv;
  bool load_started = false;
  bool release_load = false;
  target_executor.SetBeforeExecuteHookForTest([&] {
    std::unique_lock<std::mutex> lock(load_mutex);
    load_started = true;
    load_cv.notify_all();
    load_cv.wait(lock, [&] { return release_load; });
  });
  auto in_flight_handle = target_executor.SubmitLoad(in_flight_prepared);
  ASSERT_NE(in_flight_handle, nullptr);
  {
    std::unique_lock<std::mutex> lock(load_mutex);
    ASSERT_TRUE(load_cv.wait_for(lock, std::chrono::seconds(1), [&] { return load_started; }));
  }
  EXPECT_EQ(in_flight_handle->outcome(), engine::LoadOutcome::kPending);
  EXPECT_FALSE(in_flight_handle->layer_done(0));
  EXPECT_EQ(target_client.CancelLoad(in_flight_prepared), engine::CancelLoadResult::kInFlight);
  const auto drop_started = std::chrono::steady_clock::now();
  in_flight_handle.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - drop_started, std::chrono::milliseconds(50))
      << "dropping a load handle must not join its blocked worker";
  {
    std::lock_guard<std::mutex> lock(load_mutex);
    release_load = true;
  }
  load_cv.notify_all();
  ASSERT_TRUE(WaitUntil([&] {
    auto done = target_executor.PollLoads();
    return std::any_of(done.begin(), done.end(), [&](const auto& item) {
      return item.first == in_flight_prepared.id && item.second == engine::LoadOutcome::kCompleted;
    });
  }));
  target_executor.SetBeforeExecuteHookForTest({});
  EXPECT_TRUE(target_client.ReturnPages(in_flight_target));
  EXPECT_GT(held_dram->Evict({dram_key}).front().bytes_freed, 0u)
      << "the DRAM grant must release after the completed load";

  auto rollback_sources_reservation = source_client.Reserve("rollback-sources", 2, 0);
  ASSERT_TRUE(rollback_sources_reservation.success);
  auto rollback_sources =
      source_client.Take(rollback_sources_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_EQ(rollback_sources.size(), 2u);
  ASSERT_TRUE(source_client.Refund(rollback_sources_reservation.id));
  ASSERT_TRUE(source_client
                  .Publish({{"rollback-a", {}, rollback_sources[0]},
                            {"rollback-b", {}, rollback_sources[1]}})
                  .success);
  auto rollback_plan = target_client.Lookup({"rollback-a", "rollback-b"}, "model@rev/layout");
  ASSERT_EQ(rollback_plan.valid_prefix, 2u);
  auto rollback_targets_reservation = target_client.Reserve("rollback-targets", 2, 0);
  ASSERT_TRUE(rollback_targets_reservation.success);
  auto rollback_targets =
      target_client.Take(rollback_targets_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_TRUE(target_client.Refund(rollback_targets_reservation.id));
  ASSERT_TRUE(source_client.ReturnPages({rollback_sources[1]}));
  std::vector<engine::OperationId> rollback_ready;
  ASSERT_TRUE(WaitUntil([&] {
    for (const auto& ticket : source_client.PollRetires()) {
      if (ticket.state == engine::RetireState::kReady) rollback_ready.push_back(ticket.id);
    }
    return !rollback_ready.empty();
  }));
  auto rejected =
      target_client.PrepareLoad(rollback_plan, "partial-grant-reject", rollback_targets, 2);
  EXPECT_FALSE(rejected.success);
  EXPECT_TRUE(target_client.ReturnPages(rollback_targets));
  ASSERT_TRUE(source_client.ReturnPages({rollback_sources[0]}))
      << "the first grant must have been rolled back after the second source rejected";
  ASSERT_TRUE(WaitUntil([&] {
    for (const auto& ticket : source_client.PollRetires()) {
      if (ticket.state == engine::RetireState::kReady) rollback_ready.push_back(ticket.id);
    }
    return rollback_ready.size() >= 2;
  }));
  ASSERT_TRUE(source_client.AcknowledgeReleased(rollback_ready));

  auto unknown_source_reservation = source_client.Reserve("unknown-source", 1, 0);
  ASSERT_TRUE(unknown_source_reservation.success);
  auto unknown_source =
      source_client.Take(unknown_source_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(source_client.Refund(unknown_source_reservation.id));
  ASSERT_TRUE(source_client.Publish({{"unknown-load", {}, unknown_source.front()}}).success);
  auto unknown_plan = target_client.Lookup({"unknown-load"}, "model@rev/layout");
  auto unknown_target_reservation = target_client.Reserve("unknown-target", 1, 0);
  ASSERT_TRUE(unknown_target_reservation.success);
  auto unknown_target =
      target_client.Take(unknown_target_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(unknown_target_reservation.id));
  auto unknown_prepared =
      target_client.PrepareLoad(unknown_plan, "unknown-load", unknown_target, 1);
  ASSERT_TRUE(unknown_prepared.success) << unknown_prepared.error;
  target_executor.SetExecuteOverrideForTest(
      [](const std::vector<TransferPlan>& plans, std::vector<TransferFailure>* failures) {
        std::vector<size_t> tags;
        for (const auto& plan : plans) tags.insert(tags.end(), plan.tags.begin(), plan.tags.end());
        failures->push_back({std::move(tags), 91, "injected load unknown", {}, false, true});
        return false;
      });
  auto unknown_handle = target_executor.SubmitLoad(unknown_prepared);
  ASSERT_NE(unknown_handle, nullptr);
  EXPECT_EQ(unknown_handle->Wait(), engine::LoadOutcome::kUnknown);
  EXPECT_NE(unknown_handle->error().find("code=91 message='injected load unknown'"),
            std::string::npos);
  ASSERT_EQ(unknown_handle->layer_count(), 2u);
  for (size_t layer = 0; layer < unknown_handle->layer_count(); ++layer) {
    EXPECT_TRUE(unknown_handle->layer_done(layer));
    EXPECT_EQ(unknown_handle->layer_outcome(layer), engine::LoadOutcome::kUnknown)
        << "a terminal layer failure must settle every later layer";
    EXPECT_NE(unknown_handle->layer_error(layer).find("injected load unknown"), std::string::npos);
  }
  target_executor.SetExecuteOverrideForTest({});
  EXPECT_EQ(target_client.Accounting().quarantined, 1u);
  auto* source_wrapped =
      dynamic_cast<InstrumentedBackend*>(first_->Backends().Get(EngineHbmBackendName(0)));
  auto* source_backend = source_wrapped == nullptr
                             ? nullptr
                             : dynamic_cast<EngineHbmBackend*>(source_wrapped->Inner());
  ASSERT_NE(source_backend, nullptr);
  EXPECT_EQ(source_backend->State(unknown_source.front().page_id), engine::SlotState::kQuarantined)
      << "Unknown retains and quarantines the source grant as well as the target";

  auto mixed_retire_reservation = source_client.Reserve("mixed-retire", 2, 0);
  auto mixed_retire_pages =
      source_client.Take(mixed_retire_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_TRUE(source_client.Refund(mixed_retire_reservation.id));
  ASSERT_TRUE(source_client
                  .Publish({{"mixed-retire-a", {}, mixed_retire_pages[0]},
                            {"mixed-retire-b", {}, mixed_retire_pages[1]}})
                  .success);
  std::mutex mixed_retire_mutex;
  std::condition_variable mixed_retire_cv;
  size_t retire_send_count = 0;
  bool allow_first_retire = false;
  source_client.SetBeforeRetireSendHookForTest([&] {
    std::unique_lock<std::mutex> lock(mixed_retire_mutex);
    if (++retire_send_count == 1) {
      mixed_retire_cv.notify_all();
      mixed_retire_cv.wait(lock, [&] { return allow_first_retire; });
    }
  });
  auto first_retire =
      std::async(std::launch::async, [&] { return source_client.Retire({mixed_retire_pages[0]}); });
  {
    std::unique_lock<std::mutex> lock(mixed_retire_mutex);
    ASSERT_TRUE(mixed_retire_cv.wait_for(lock, std::chrono::seconds(1),
                                         [&] { return retire_send_count == 1; }));
  }
  auto mixed_retire = source_client.Retire(mixed_retire_pages);
  EXPECT_NE(mixed_retire.state, engine::RetireState::kBusy);
  EXPECT_EQ(source_client.Accounting().retiring, 2u)
      << "the unsent sibling must not be stranded behind an already-sent page";
  {
    std::lock_guard<std::mutex> lock(mixed_retire_mutex);
    allow_first_retire = true;
  }
  mixed_retire_cv.notify_all();
  (void)first_retire.get();
  source_client.SetBeforeRetireSendHookForTest({});
  source_client.Close();
  target_client.Close();
  source_executor.Close();
  target_executor.Close();
}

TEST_F(EngineTwoNodeLoop, OversizedPreparedLoadIsRejectedBeforeBuildingAnUnboundedFrame) {
  std::vector<engine::AllocationHandle> targets(257);
  engine::ReadPlan plan;
  plan.id = 1;
  auto rejected = first_->PrepareEngineLoad(plan, "oversized-prepare", targets, targets.size(),
                                            {"reader", "oversized"},
                                            engine::OperationId{std::array<uint8_t, 16>{}, 1});
  EXPECT_FALSE(rejected.success);
  EXPECT_EQ(rejected.error, "invalid prepare_load request");
}

TEST_F(EngineTwoNodeLoop, LookupRoutingExcludesSelfAndTreatsSsdAsMissing) {
  EXPECT_EQ(PoolClient::EngineLookupExcludes("reader-node"),
            std::unordered_set<std::string>{"reader-node"});
  EXPECT_EQ(PoolClient::EngineRemoteReadKind(TierType::SSD), engine::ReadSourceKind::kMissing);
  EXPECT_EQ(PoolClient::EngineRemoteReadKind(TierType::HBM), engine::ReadSourceKind::kRemoteHbm);
  EXPECT_EQ(PoolClient::EngineRemoteReadKind(TierType::DRAM), engine::ReadSourceKind::kRemoteDram);
}

TEST_F(EngineTwoNodeLoop, LookupAndPrepareRemoteHbmUsesPeerGrantAndExecutorPlan) {
  constexpr size_t kHalf = 32;
  constexpr size_t kBytes = 2 * kHalf;
  constexpr uint32_t kPages = 2;
  std::array<std::vector<uint8_t>, 2> source_regions{std::vector<uint8_t>((kPages + 1) * kHalf),
                                                     std::vector<uint8_t>((kPages + 1) * kHalf)};
  std::array<std::vector<uint8_t>, 2> target_regions{std::vector<uint8_t>((kPages + 1) * kHalf),
                                                     std::vector<uint8_t>((kPages + 1) * kHalf)};
  auto geometry_for = [&](auto& regions) {
    engine::PageGeometry geometry;
    geometry.page_tokens = 1;
    geometry.num_pages = kPages;
    geometry.object_bytes = kBytes;
    geometry.regions = {
        {"remote-layer-0", TransferRef::HostBytes(regions[0].data(), regions[0].size()), kHalf, 1,
         0, 0, "kv"},
        {"remote-layer-1", TransferRef::HostBytes(regions[1].data(), regions[1].size()), kHalf, 1,
         kHalf, 1, "kv"},
    };
    geometry.layout_id = engine::ComputeLayoutId(geometry);
    return geometry;
  };
  auto source_geometry = geometry_for(source_regions);
  auto target_geometry = geometry_for(target_regions);
  engine::EngineExecutor source_executor({"remote-source-executor", -1, "127.0.0.1", 0, 0});
  engine::EngineExecutor target_executor({"remote-target-executor", -1, "127.0.0.1", 0, 0});
  for (size_t layer = 0; layer < 2; ++layer) {
    const std::string region = "remote-layer-" + std::to_string(layer);
    ASSERT_TRUE(source_executor.RegisterOriginalRegion(
        region,
        TransferRef::HostBytes(source_regions[layer].data(), source_regions[layer].size())));
    ASSERT_TRUE(target_executor.RegisterOriginalRegion(
        region,
        TransferRef::HostBytes(target_regions[layer].data(), target_regions[layer].size())));
  }
  engine::EngineClient source_client({engine_address_, "remote-source-executor", 10, 5000});
  source_client.AttachExecutor(&source_executor);
  engine::PoolEpoch epoch;
  auto source_registration =
      source_client.RegisterPool(0, "remote-load-source", epoch, source_geometry);
  ASSERT_TRUE(source_registration.success) << source_registration.error;

  const std::string target_address =
      "unix:///tmp/umbp_engine_remote_target_" + std::to_string(getpid()) + ".grpc.sock";
  PoolEngineControl target_control(second_.get());
  engine::EngineSessionServer target_sessions(&target_control);
  std::string server_error;
  ASSERT_TRUE(
      target_sessions.Start(standalone::DeriveEngineSocketPath(target_address), &server_error))
      << server_error;
  engine::EngineClient target_client({target_address, "remote-target-executor", 10, 5000});
  target_client.AttachExecutor(&target_executor);
  auto target_registration =
      target_client.RegisterPool(0, "remote-load-target", epoch, target_geometry);
  ASSERT_TRUE(target_registration.success) << target_registration.error;

  auto source_reservation = source_client.Reserve("remote-source", 1, 0);
  ASSERT_TRUE(source_reservation.success);
  auto source_page =
      source_client.Take(source_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(source_client.Refund(source_reservation.id));
  for (size_t i = 0; i < kHalf; ++i) {
    source_regions[0][source_page[0].page_id * kHalf + i] = static_cast<uint8_t>(0x31 + i);
    source_regions[1][source_page[0].page_id * kHalf + i] = static_cast<uint8_t>(0x91 + i);
  }
  const std::string key = "remote-hbm-load";
  ASSERT_TRUE(source_client.Publish({ResidentRecord{key, {}, source_page[0]}}).success);
  first_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitUntil([&] { return second_->Exists(key); }));

  auto plan = target_client.Lookup({key}, "model@rev/remote-layout");
  ASSERT_TRUE(plan.success) << plan.error;
  ASSERT_EQ(plan.sources.size(), 1u);
  EXPECT_EQ(plan.sources[0].kind, engine::ReadSourceKind::kRemoteHbm);
  EXPECT_EQ(plan.sources[0].node_id, "engine-node");
  auto target_reservation = target_client.Reserve("remote-target", 1, 0);
  ASSERT_TRUE(target_reservation.success);
  auto target_page =
      target_client.Take(target_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(target_reservation.id));
  SetLoadRpcDelayForTest(1500);
  const auto delayed_started = std::chrono::steady_clock::now();
  auto delayed = target_client.PrepareLoad(plan, "deadline-bounded-prepare", target_page, 1);
  const auto delayed_elapsed = std::chrono::steady_clock::now() - delayed_started;
  SetLoadRpcDelayForTest(0);
  EXPECT_FALSE(delayed.success);
  EXPECT_LT(delayed_elapsed, std::chrono::milliseconds(2800))
      << "prepare_load retries must share one total deadline";
  ASSERT_TRUE(target_client.ReturnPages(target_page));

  for (int attempt = 0; attempt < 200; ++attempt) {
    plan = target_client.Lookup({key}, "model@rev/remote-layout");
    if (plan.success) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(plan.success);
  target_reservation = target_client.Reserve("remote-target-lost-reply", 1, 0);
  target_page = target_client.Take(target_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(target_reservation.id));
  FailNextLoadAcquireReplyForTest();
  auto lost_acquire = target_client.PrepareLoad(plan, "lost-acquire-reply", target_page, 1);
  EXPECT_FALSE(lost_acquire.success)
      << "a replayed acquire must be compensated, not accepted after a lost first reply";
  ASSERT_TRUE(target_client.ReturnPages(target_page));

  plan = target_client.Lookup({key}, "model@rev/remote-layout");
  ASSERT_TRUE(plan.success);
  target_reservation = target_client.Reserve("remote-target-retry", 1, 0);
  ASSERT_TRUE(target_reservation.success);
  target_page = target_client.Take(target_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(target_reservation.id));
  auto prepared = target_client.PrepareLoad(plan, "remote-request", target_page, 1);
  ASSERT_TRUE(prepared.success) << prepared.error;
  ASSERT_TRUE(std::any_of(prepared.grants.begin(), prepared.grants.end(),
                          [](const auto& grant) { return grant.remote && grant.engine; }));
  ASSERT_TRUE(std::any_of(
      prepared.bootstraps.begin(), prepared.bootstraps.end(), [](const auto& bootstrap) {
        return bootstrap.executor_id == "remote-source-executor" && !bootstrap.engine_desc.empty();
      }));
  for (const auto& layer : prepared.layers) {
    for (const auto& transfer : layer.plans) {
      EXPECT_TRUE(transfer.src.HasMemoryDesc())
          << "remote HBM spans must carry the owner rank's registered MR";
      EXPECT_TRUE(transfer.dst.HasMemoryDesc())
          << "load targets must carry the requesting rank's registered MR";
      EXPECT_NE(transfer.src.mem.engineKey, transfer.dst.mem.engineKey);
    }
  }
  target_executor.SetExecuteOverrideForTest(
      [](const std::vector<TransferPlan>& plans, std::vector<TransferFailure>*) {
        for (const auto& plan : plans) {
          for (size_t i = 0; i < plan.sizes.size(); ++i) {
            auto* src =
                reinterpret_cast<const uint8_t*>(plan.src.executor_address) + plan.src_offsets[i];
            auto* dst = reinterpret_cast<uint8_t*>(plan.dst.executor_address) + plan.dst_offsets[i];
            std::memcpy(dst, src, plan.sizes[i]);
          }
        }
        return true;
      });
  FailNextLoadReleaseReplyForTest();
  auto handle = target_executor.SubmitLoad(prepared);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->Wait(), engine::LoadOutcome::kCompleted);
  for (size_t layer = 0; layer < 2; ++layer) {
    EXPECT_EQ(std::memcmp(target_regions[layer].data() + target_page[0].page_id * kHalf,
                          source_regions[layer].data() + source_page[0].page_id * kHalf, kHalf),
              0);
  }
  target_executor.SetExecuteOverrideForTest({});
  EXPECT_TRUE(target_client.ReturnPages(target_page));

  auto failed_session_plan = target_client.Lookup({key}, "model@rev/remote-layout");
  ASSERT_TRUE(failed_session_plan.success);
  auto failed_session_reservation = target_client.Reserve("failed-session-target", 1, 0);
  auto failed_session_target =
      target_client.Take(failed_session_reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_TRUE(target_client.Refund(failed_session_reservation.id));
  auto failed_session_prepared = target_client.PrepareLoad(
      failed_session_plan, "failed-session-request", failed_session_target, 1);
  ASSERT_TRUE(failed_session_prepared.success) << failed_session_prepared.error;
  second_->FailEngineSession(0);
  ASSERT_TRUE(source_client.ReturnPages(source_page));
  std::vector<engine::RetireTicket> remote_source_ready;
  ASSERT_TRUE(WaitUntil([&] {
    remote_source_ready = source_client.PollRetires();
    return !remote_source_ready.empty();
  }));
  ASSERT_TRUE(source_client.AcknowledgeReleased({remote_source_ready.front().id}))
      << "remote completion must release the source-node grant";
  source_client.Close();
  target_client.Close();
  target_sessions.Shutdown();
  source_executor.Close();
  target_executor.Close();
  unlink(standalone::DeriveEngineSocketPath(target_address).c_str());
}

TEST_F(EngineTwoNodeLoop, EightRanksRetiringSameKeyScheduleOneF3Preservation) {
  constexpr size_t kBytes = 64;
  const std::string key = "eight-rank-last-replica";
  std::vector<std::vector<uint8_t>> storage(8, std::vector<uint8_t>(2 * kBytes));
  std::vector<std::unique_ptr<engine::EngineExecutor>> executors;
  std::vector<std::unique_ptr<engine::EngineClient>> clients;
  std::vector<engine::AllocationHandle> handles;
  executors.reserve(8);
  clients.reserve(8);
  for (uint32_t rank = 0; rank < 8; ++rank) {
    const std::string executor_id = "rank-executor-" + std::to_string(rank);
    executors.push_back(
        std::make_unique<engine::EngineExecutor>(engine::EngineExecutorConfig{executor_id, -1}));
    ASSERT_TRUE(executors.back()->RegisterOriginalRegion(
        "kv", TransferRef::HostBytes(storage[rank].data(), storage[rank].size(),
                                     mori::io::MemoryLocationType::CPU, -1)));
    engine::PageGeometry geometry;
    geometry.page_tokens = 1;
    geometry.num_pages = 1;
    geometry.object_bytes = kBytes;
    geometry.regions.push_back(
        engine::PageRegion{"kv",
                           TransferRef::HostBytes(storage[rank].data(), storage[rank].size(),
                                                  mori::io::MemoryLocationType::CPU, -1),
                           kBytes, 1, 0, 0, "kv"});
    geometry.layout_id = engine::ComputeLayoutId(geometry);
    clients.push_back(std::make_unique<engine::EngineClient>(engine::EngineClientConfig{
        engine_address_, executor_id, /*heartbeat_ms=*/10, /*heartbeat_timeout_ms=*/1000}));
    clients.back()->AttachExecutor(executors.back().get());
    engine::PoolEpoch ignored_rank_epoch;
    auto registration =
        clients.back()->RegisterPool(rank, "eight-rank-pool", ignored_rank_epoch, geometry);
    ASSERT_TRUE(registration.success) << registration.error;
    std::string map_error;
    ASSERT_TRUE(executors.back()->MapSharedDram(registration.dram_regions, &map_error))
        << map_error;
    auto reservation = clients.back()->Reserve("request", 1, 0);
    ASSERT_TRUE(reservation.success);
    auto taken = clients.back()->Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
    ASSERT_EQ(taken.size(), 1u);
    ASSERT_TRUE(clients.back()->Refund(reservation.id));
    handles.push_back(taken.front());
    for (size_t i = 0; i < kBytes; ++i) {
      storage[rank][handles.back().page_id * kBytes + i] = static_cast<uint8_t>(i + 9);
    }
    ASSERT_TRUE(clients.back()->Publish({ResidentRecord{key, {}, handles.back()}}).success);
  }

  std::vector<std::thread> retires;
  for (size_t rank = 0; rank < clients.size(); ++rank) {
    retires.emplace_back([&, rank] { EXPECT_TRUE(clients[rank]->ReturnPages({handles[rank]})); });
  }
  for (auto& thread : retires) thread.join();

  std::array<uint8_t, kBytes> read{};
  ASSERT_TRUE(WaitUntil([&] { return first_->Get(key, read.data(), read.size()); }));
  for (size_t i = 0; i < kBytes; ++i) EXPECT_EQ(read[i], static_cast<uint8_t>(i + 9));
  EXPECT_LE(first_->TransitionMetrics().attempted, 1u);

  for (size_t rank = 0; rank < clients.size(); ++rank) {
    auto& client = clients[rank];
    std::vector<engine::RetireTicket> tickets;
    ASSERT_TRUE(WaitUntil([&] {
      tickets = client->PollRetires();
      return !tickets.empty();
    })) << "rank="
        << rank << " retiring=" << client->Accounting().retiring
        << " quarantined=" << client->Accounting().quarantined;
    ASSERT_EQ(tickets.front().state, engine::RetireState::kReady);
    ASSERT_TRUE(client->AcknowledgeReleased({tickets.front().id}));
  }
}

TEST_F(EngineTwoNodeLoop, UnknownF3LeaderRetriesFromHealthyRetiringFollower) {
  constexpr size_t kBytes = 64;
  const std::string key = "f3-leader-retry";
  std::vector<std::vector<uint8_t>> storage(2, std::vector<uint8_t>(2 * kBytes));
  std::vector<std::unique_ptr<engine::EngineExecutor>> executors;
  std::vector<std::unique_ptr<engine::EngineClient>> clients;
  std::vector<engine::AllocationHandle> handles;
  for (uint32_t rank = 0; rank < 2; ++rank) {
    const std::string executor_id = "retry-executor-" + std::to_string(rank);
    executors.push_back(
        std::make_unique<engine::EngineExecutor>(engine::EngineExecutorConfig{executor_id, -1}));
    ASSERT_TRUE(executors.back()->RegisterOriginalRegion(
        "kv", TransferRef::HostBytes(storage[rank].data(), storage[rank].size())));
    engine::PageGeometry geometry;
    geometry.page_tokens = 1;
    geometry.num_pages = 1;
    geometry.object_bytes = kBytes;
    geometry.regions.push_back(
        engine::PageRegion{"kv", TransferRef::HostBytes(storage[rank].data(), storage[rank].size()),
                           kBytes, 1, 0, 0, "kv"});
    geometry.layout_id = engine::ComputeLayoutId(geometry);
    clients.push_back(std::make_unique<engine::EngineClient>(engine::EngineClientConfig{
        engine_address_, executor_id, /*heartbeat_ms=*/10, /*heartbeat_timeout_ms=*/1000}));
    clients.back()->AttachExecutor(executors.back().get());
    engine::PoolEpoch epoch;
    auto registration = clients.back()->RegisterPool(rank, "retry-pool", epoch, geometry);
    ASSERT_TRUE(registration.success) << registration.error;
    std::string map_error;
    ASSERT_TRUE(executors.back()->MapSharedDram(registration.dram_regions, &map_error))
        << map_error;
    auto reservation = clients.back()->Reserve("retry", 1, 0);
    auto taken = clients.back()->Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
    ASSERT_EQ(taken.size(), 1u);
    ASSERT_TRUE(clients.back()->Refund(reservation.id));
    handles.push_back(taken.front());
    std::fill(storage[rank].begin() + handles.back().page_id * kBytes,
              storage[rank].begin() + (handles.back().page_id + 1) * kBytes, 0x4d);
    ASSERT_TRUE(clients.back()->Publish({ResidentRecord{key, {}, handles.back()}}).success);
  }
  std::atomic<int> failed_rank{-1};
  for (size_t rank = 0; rank < executors.size(); ++rank) {
    executors[rank]->SetExecuteOverrideForTest(
        [&, rank](const std::vector<TransferPlan>& plans, std::vector<TransferFailure>* failures) {
          int expected = -1;
          if (!failed_rank.compare_exchange_strong(expected, static_cast<int>(rank))) return true;
          for (auto& executor : executors) executor->SetExecuteOverrideForTest({});
          std::vector<size_t> tags;
          for (const auto& plan : plans)
            tags.insert(tags.end(), plan.tags.begin(), plan.tags.end());
          failures->push_back({std::move(tags), 9, "injected unknown F3", {}, false, true});
          return false;
        });
  }

  ASSERT_TRUE(clients[0]->ReturnPages({handles[0]}));
  ASSERT_TRUE(clients[1]->ReturnPages({handles[1]}));
  std::array<uint8_t, kBytes> read{};
  ASSERT_TRUE(WaitUntil([&] { return first_->Get(key, read.data(), read.size()); }));
  EXPECT_TRUE(std::all_of(read.begin(), read.end(), [](uint8_t value) { return value == 0x4d; }));

  std::array<std::vector<engine::RetireTicket>, 2> observed;
  for (size_t rank = 0; rank < clients.size(); ++rank) {
    ASSERT_TRUE(WaitUntil([&, rank] {
      observed[rank] = clients[rank]->PollRetires();
      return !observed[rank].empty();
    }));
  }
  ASSERT_GE(failed_rank.load(), 0);
  const size_t failed = static_cast<size_t>(failed_rank.load());
  const size_t follower = 1 - failed;
  EXPECT_EQ(observed[failed].front().state, engine::RetireState::kQuarantined);
  EXPECT_EQ(observed[follower].front().state, engine::RetireState::kReady);
  ASSERT_TRUE(clients[follower]->AcknowledgeReleased({observed[follower].front().id}));
  for (auto& client : clients) client->Close();
  sessions_->Shutdown();
  EXPECT_TRUE(first_->Clear())
      << "a follower finalized after its Ready ACK must release, not retain, its DRAM hold";
  executors[failed]->Close();
  EXPECT_GT(executors[failed]->mapped_region_count(), 0u)
      << "Unknown DMA must retain mapped and registered native resources";
  executors[follower]->Close();
}

TEST_F(EngineTwoNodeLoop, UnknownDuringF3QuarantinesTargetAndSource) {
  constexpr size_t kBytes = 64;
  std::vector<uint8_t> storage(2 * kBytes);
  executor_ = std::make_unique<engine::EngineExecutor>(
      engine::EngineExecutorConfig{"unknown-f3-executor", -1});
  ASSERT_TRUE(executor_->RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(storage.data(), storage.size(),
                                   mori::io::MemoryLocationType::CPU, -1)));
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 1;
  geometry.object_bytes = kBytes;
  geometry.regions.push_back(engine::PageRegion{
      "kv",
      TransferRef::HostBytes(storage.data(), storage.size(), mori::io::MemoryLocationType::CPU, -1),
      kBytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  engine_client_ = std::make_unique<engine::EngineClient>(
      engine::EngineClientConfig{engine_address_, "unknown-f3-executor", 500, 2000});
  engine_client_->AttachExecutor(executor_.get());
  engine::PoolEpoch ignored;
  auto registration = engine_client_->RegisterPool(0, "unknown-f3-pool", ignored, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  std::string map_error;
  ASSERT_TRUE(executor_->MapSharedDram(registration.dram_regions, &map_error)) << map_error;
  auto reservation = engine_client_->Reserve("unknown", 1, 0);
  auto taken = engine_client_->Take(reservation.id, engine::ReservationPurpose::kLoad, 1);
  ASSERT_EQ(taken.size(), 1u);
  ASSERT_TRUE(engine_client_->Refund(reservation.id));
  auto handle = taken.front();
  std::fill(storage.begin() + handle.page_id * kBytes,
            storage.begin() + (handle.page_id + 1) * kBytes, 0x6b);
  const std::string key = "unknown-f3-key";
  ASSERT_TRUE(engine_client_->Publish({ResidentRecord{key, {}, handle}}).success);

  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool release = false;
  executor_->SetBeforeExecuteHookForTest([&] {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release; });
  });
  const uint64_t maintenance_before = first_->EnginePool()->EngineMaintenanceRoundsForTest();
  ASSERT_TRUE(engine_client_->ReturnPages({handle}));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1), [&] { return entered; }));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  EXPECT_GT(first_->EnginePool()->EngineMaintenanceRoundsForTest(), maintenance_before)
      << "grant maintenance must advance while a dispatch ticket is stuck";
  sessions_->Shutdown();
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
  }
  cv.notify_all();

  auto* wrapped =
      dynamic_cast<InstrumentedBackend*>(first_->Backends().Get(EngineHbmBackendName(0)));
  ASSERT_NE(wrapped, nullptr);
  auto* engine_backend = dynamic_cast<EngineHbmBackend*>(wrapped->Inner());
  ASSERT_NE(engine_backend, nullptr);
  ASSERT_TRUE(WaitUntil(
      [&] { return engine_backend->State(handle.page_id) == engine::SlotState::kQuarantined; }));
  bool target_quarantined = false;
  for (auto* medium : first_->Backends().All()) {
    auto* instrumented = dynamic_cast<InstrumentedBackend*>(medium);
    auto* page =
        instrumented == nullptr ? nullptr : dynamic_cast<PageBackend*>(instrumented->Inner());
    if (page != nullptr && page->Tier() == TierType::DRAM &&
        page->QuarantinedPendingCountForTest() > 0) {
      target_quarantined = true;
    }
  }
  EXPECT_TRUE(target_quarantined);
  EXPECT_TRUE(first_->EnginePool()->ResidencyBackends(key).empty())
      << "quarantine must remove stale placement and residency projections";
  EXPECT_TRUE(first_->Clear())
      << "after the engine session fails, Clear must release F3 records and backing pins";
}

TEST_F(EngineTwoNodeLoop, UnknownF3QuarantinesOnlyItsSourceInAMultiPageRetire) {
  constexpr size_t kBytes = 64;
  std::vector<uint8_t> storage(4 * kBytes);
  executor_ = std::make_unique<engine::EngineExecutor>(
      engine::EngineExecutorConfig{"multi-page-unknown-executor", -1});
  ASSERT_TRUE(executor_->RegisterOriginalRegion(
      "kv", TransferRef::HostBytes(storage.data(), storage.size())));
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 3;
  geometry.object_bytes = kBytes;
  geometry.regions.push_back(engine::PageRegion{
      "kv", TransferRef::HostBytes(storage.data(), storage.size()), kBytes, 1, 0, 0, "kv"});
  geometry.layout_id = engine::ComputeLayoutId(geometry);
  engine_client_ = std::make_unique<engine::EngineClient>(
      engine::EngineClientConfig{engine_address_, "multi-page-unknown-executor", 10, 1000});
  engine_client_->AttachExecutor(executor_.get());
  engine::PoolEpoch epoch;
  auto registration = engine_client_->RegisterPool(0, "multi-page-unknown-pool", epoch, geometry);
  ASSERT_TRUE(registration.success) << registration.error;
  std::string map_error;
  ASSERT_TRUE(executor_->MapSharedDram(registration.dram_regions, &map_error)) << map_error;
  auto reservation = engine_client_->Reserve("multi-page-unknown", 2, 0);
  ASSERT_TRUE(reservation.success);
  auto pages = engine_client_->Take(reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_EQ(pages.size(), 2u);
  ASSERT_TRUE(engine_client_->Refund(reservation.id));
  ASSERT_TRUE(engine_client_
                  ->Publish({{"multi-unknown-a", {}, pages[0]}, {"multi-unknown-b", {}, pages[1]}})
                  .success);
  const uint64_t failed_offset = static_cast<uint64_t>(pages[0].page_id) * kBytes;
  executor_->SetExecuteOverrideForTest([failed_offset](const std::vector<TransferPlan>& plans,
                                                       std::vector<TransferFailure>* failures) {
    bool failed = false;
    for (const auto& plan : plans) {
      if (std::find(plan.src_offsets.begin(), plan.src_offsets.end(), failed_offset) ==
          plan.src_offsets.end()) {
        continue;
      }
      failures->push_back({plan.tags, 19, "injected one-page unknown", {}, false, true});
      failed = true;
    }
    return !failed;
  });
  const uint64_t schedule_checks_before = first_->EnginePool()->F3ScheduleChecksForTest();
  ASSERT_TRUE(engine_client_->ReturnPages(pages));

  std::vector<engine::RetireTicket> observed;
  ASSERT_TRUE(WaitUntil([&] {
    auto polled = engine_client_->PollRetires();
    observed.insert(observed.end(), polled.begin(), polled.end());
    return observed.size() >= 2;
  }));
  auto quarantined = std::find_if(observed.begin(), observed.end(), [](const auto& ticket) {
    return ticket.state == engine::RetireState::kQuarantined;
  });
  auto ready = std::find_if(observed.begin(), observed.end(), [](const auto& ticket) {
    return ticket.state == engine::RetireState::kReady;
  });
  ASSERT_NE(quarantined, observed.end());
  ASSERT_NE(ready, observed.end());
  ASSERT_EQ(quarantined->handles.size(), 1u);
  EXPECT_EQ(quarantined->handles[0].page_id, pages[0].page_id);
  ASSERT_EQ(ready->handles.size(), 1u);
  EXPECT_EQ(ready->handles[0].page_id, pages[1].page_id);
  ASSERT_TRUE(engine_client_->AcknowledgeReleased({ready->id}));
  EXPECT_EQ(engine_client_->Accounting().quarantined, 1u);
  EXPECT_EQ(engine_client_->Accounting().cached, 2u);
  EXPECT_LE(first_->EnginePool()->F3ScheduleChecksForTest() - schedule_checks_before, 4u)
      << "two pending keys must be consumed from the FIFO, not trigger record-wide rescans";

  executor_->SetExecuteOverrideForTest({});
  auto missing_reservation = engine_client_->Reserve("multi-page-source-missing", 2, 0);
  ASSERT_TRUE(missing_reservation.success);
  auto missing_pages =
      engine_client_->Take(missing_reservation.id, engine::ReservationPurpose::kLoad, 2);
  ASSERT_EQ(missing_pages.size(), 2u);
  ASSERT_TRUE(engine_client_->Refund(missing_reservation.id));
  const std::string missing_key = "multi-source-missing-a";
  ASSERT_TRUE(engine_client_
                  ->Publish({{missing_key, {}, missing_pages[0]},
                             {"multi-source-missing-b", {}, missing_pages[1]}})
                  .success);
  auto* wrapped =
      dynamic_cast<InstrumentedBackend*>(first_->Backends().Get(EngineHbmBackendName(0)));
  ASSERT_NE(wrapped, nullptr);
  auto* engine_backend = dynamic_cast<EngineHbmBackend*>(wrapped->Inner());
  ASSERT_NE(engine_backend, nullptr);
  std::atomic<bool> removed{false};
  first_->EnginePool()->SetBeforeF3RunHookForTest([&](const std::string& key) {
    if (key == missing_key && !removed.exchange(true)) {
      EXPECT_TRUE(engine_backend->Quarantine({missing_pages[0]}));
    }
  });
  ASSERT_TRUE(engine_client_->ReturnPages(missing_pages));
  std::vector<engine::RetireTicket> missing_observed;
  ASSERT_TRUE(WaitUntil([&] {
    auto polled = engine_client_->PollRetires();
    missing_observed.insert(missing_observed.end(), polled.begin(), polled.end());
    return missing_observed.size() >= 2;
  }));
  first_->EnginePool()->SetBeforeF3RunHookForTest({});
  auto missing_quarantined = std::find_if(
      missing_observed.begin(), missing_observed.end(),
      [](const auto& ticket) { return ticket.state == engine::RetireState::kQuarantined; });
  auto missing_ready =
      std::find_if(missing_observed.begin(), missing_observed.end(),
                   [](const auto& ticket) { return ticket.state == engine::RetireState::kReady; });
  ASSERT_NE(missing_quarantined, missing_observed.end());
  ASSERT_NE(missing_ready, missing_observed.end());
  ASSERT_EQ(missing_quarantined->handles.size(), 1u);
  EXPECT_EQ(missing_quarantined->handles[0].page_id, missing_pages[0].page_id);
  ASSERT_EQ(missing_ready->handles.size(), 1u);
  EXPECT_EQ(missing_ready->handles[0].page_id, missing_pages[1].page_id)
      << "source_missing removes only its key; the sibling remains independently releasable";
  ASSERT_TRUE(engine_client_->AcknowledgeReleased({missing_ready->id}));
  EXPECT_EQ(engine_client_->Accounting().quarantined, 2u);
  EXPECT_EQ(engine_client_->Accounting().cached, 1u);
}

}  // namespace
}  // namespace mori::umbp

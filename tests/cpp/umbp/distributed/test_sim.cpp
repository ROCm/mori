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

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <future>
#include <thread>

#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/pool/tier_graph.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/sim/sim.h"
#include "umbp/sim/trace.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {
namespace {
using MemLoc = mori::io::MemoryLocationType;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
constexpr size_t kPage = 4096;

class SimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/umbp-sim-test-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    dir_ = path;
    Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":1000000,
                           "bandwidth_bytes_per_second":4096000}})");
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }
  void Profile(const std::string& paths) {
    std::ofstream(dir_ + "/profile.json") << R"({"id":"test","paths":)" << paths << "}";
  }
  UMBPConfig Config() {
    UMBPConfig config;
    config.sim.enabled = true;
    config.sim.profile_path = dir_ + "/profile.json";
    config.sim.trace_path = dir_ + "/trace.jsonl";
    config.dram.capacity_bytes = 32 * kPage;
    config.page_size = kPage;
    config.ssd.enabled = false;
    return config;
  }
  std::unique_ptr<PageBackend> Backend(SimTransferEngine& engine, MemLoc loc, size_t pages = 4) {
    auto backend = std::make_unique<PageBackend>(
        loc == MemLoc::GPU ? TierType::HBM : TierType::DRAM, kPage,
        std::make_unique<SimPageMemorySource>(loc, loc == MemLoc::GPU ? 0 : -1),
        std::vector<uint64_t>{pages * kPage + 1}, 40ms, 40ms, 5ms);
    EXPECT_TRUE(backend->Init(&engine));
    return backend;
  }
  uintptr_t Buffer(size_t bytes) {
    std::vector<PageMemorySource::Buffer> buffers;
    if (!memory_.Allocate({bytes}, &buffers)) throw std::bad_alloc();
    return reinterpret_cast<uintptr_t>(buffers.front().base);
  }
  template <class Predicate>
  bool Await(Predicate ready) {
    const auto deadline = Clock::now() + 2s;
    while (!ready()) {
      if (Clock::now() >= deadline) return false;
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }
  SimPageMemorySource memory_{MemLoc::CPU, -1};
  std::string dir_;
};

TEST_F(SimTest, PublicApisNeverAccessPayloadAndRecordOutcomes) {
  auto config = Config();
  const auto ptr = Buffer(8 * kPage);
  auto client = CreateUMBPClient(config);
  ASSERT_TRUE(client->RegisterMemory(ptr, 8 * kPage));
  EXPECT_TRUE(client->Put("scalar", ptr, 2 * kPage));
  EXPECT_TRUE(client->Get("scalar", ptr, 2 * kPage));
  EXPECT_FALSE(client->Get("missing", ptr, kPage));
  EXPECT_EQ(client->BatchPut({"a", "b"}, {ptr, ptr + kPage}, {kPage, kPage}),
            (std::vector<bool>{true, true}));
  EXPECT_EQ(client->BatchPutWithDepth({"depth"}, {ptr}, {kPage}, {3}), std::vector<bool>{true});
  EXPECT_EQ(client->BatchGet({"a", "missing"}, {ptr, ptr}, {kPage, kPage}),
            (std::vector<bool>{true, false}));
  ASSERT_TRUE(client->SupportsRangedIO());
  EXPECT_EQ(client->BatchPutRanges({"range"}, {2 * kPage}, {{ptr, ptr + kPage}}, {{kPage, kPage}},
                                   {{0, kPage}}),
            std::vector<bool>{true});
  EXPECT_EQ(client->BatchGetRanges({"range"}, {{ptr, ptr + kPage}}, {{128, 128}},
                                   {{kPage - 64, 2 * kPage - 128}}),
            std::vector<bool>{true});
  EXPECT_EQ(client->BatchPutRanges({"gap"}, {kPage}, {{ptr}}, {{128}}, {{0}}),
            std::vector<bool>{false});
  EXPECT_FALSE(client->Exists("gap"));
  EXPECT_EQ(client->BatchExistsConsecutive({"a", "b", "missing", "depth"}), 2u);
  EXPECT_TRUE(client->Clear());
  EXPECT_EQ(client->BatchExists({"a", "scalar", "range"}), std::vector<bool>(3, false));
  client->DeregisterMemory(ptr);
  client->Close();
  EXPECT_FALSE(client->Get("closed", ptr, kPage));
  EXPECT_TRUE(client->Flush());

  std::ifstream trace(config.sim.trace_path);
  std::string line;
  uint64_t id = 0;
  bool saw_put = false, saw_partial = false;
  while (std::getline(trace, line)) {
    google::protobuf::Struct record;
    ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(line, &record).ok());
    const auto& f = record.fields();
    EXPECT_EQ(f.at("call_id").number_value(), id++);
    EXPECT_FALSE(f.at("exception").bool_value());
    if (f.at("api").string_value() == "get")
      EXPECT_EQ(f.at("result_count").number_value(), f.at("bytes").number_value() == 2 * kPage);
    if (f.at("api").string_value() == "close" || f.at("api").string_value() == "deregister_memory")
      EXPECT_EQ(f.at("result_count").number_value(), -1);
    if (f.at("api").string_value() == "put") {
      saw_put = true;
      EXPECT_EQ(f.at("bytes").number_value(), 2 * kPage);
      EXPECT_EQ(f.at("result_count").number_value(), 1);
      EXPECT_GE(f.at("elapsed_ns").number_value(), 3000000);
    }
    if (f.at("api").string_value() == "batch_get") {
      saw_partial = true;
      EXPECT_EQ(f.at("result_count").number_value(), 1);
    }
  }
  EXPECT_TRUE(saw_put && saw_partial);
}

TEST_F(SimTest, HbmAndRegisteredGpuCallerNeedNoDevice) {
  // CPU paths deliberately absent: accidentally classifying the caller as host must fail.
  Profile(R"({"gpu_gpu":{"channel":"hbm","latency_ns":0,
                           "bandwidth_bytes_per_second":1e12}})");
  auto config = Config();
  config.distributed.emplace();
  config.distributed->medium = UMBPMedium::HBM;
  config.distributed->hbm.capacity_bytes = 32 * kPage;
  const auto ptr = Buffer(4 * kPage);
  auto client = CreateUMBPClient(config);
  ASSERT_TRUE(client->RegisterMemory(ptr, 4 * kPage, MemLoc::GPU, 0));
  EXPECT_TRUE(client->Put("gpu", ptr + kPage, kPage));
  EXPECT_TRUE(client->Get("gpu", ptr + 2 * kPage, kPage));
  EXPECT_EQ(client->BatchGetRanges({"gpu"}, {{ptr + 3 * kPage}}, {{128}}, {{0}}),
            std::vector<bool>{true});
}

TEST_F(SimTest, BufferedTracePreservesEscapingAndIntegerPrecision) {
  auto config = Config().sim;
  config.profile_path = "quoted\"path\\with\nnewline";
  SimTrace trace(config);
  constexpr uint64_t id = (uint64_t{1} << 53) + 1;
  trace.Write(id, "get", 1, kPage, id, 123, -1, true);
  trace.Write(id + 1, "flush", 0, 0, 0, 0, 1, false);
  std::ifstream input(config.trace_path);
  std::string line;
  ASSERT_TRUE(std::getline(input, line));
  EXPECT_NE(line.find(std::to_string(id)), std::string::npos);
  google::protobuf::Struct record;
  ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(line, &record).ok());
  EXPECT_EQ(record.fields().at("profile_path").string_value(), config.profile_path);
  EXPECT_TRUE(record.fields().at("exception").bool_value());
  EXPECT_EQ(record.fields().at("result_count").number_value(), -1);
  ASSERT_TRUE(std::getline(input, line));
  EXPECT_FALSE(std::getline(input, line));
  EXPECT_THROW(
      {
        SimCall call(&trace, "get", 1, kPage);
        throw std::runtime_error("test exception");
      },
      std::runtime_error);
  trace.Write(id + 2, "flush", 0, 0, 0, 0, 1, false);
  input.clear();
  ASSERT_TRUE(std::getline(input, line));
  ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(line, &record).ok());
  EXPECT_TRUE(record.fields().at("exception").bool_value());
  EXPECT_EQ(record.fields().at("result_count").number_value(), -1);
}

TEST_F(SimTest, ParallelCopiesKeepSingleStreamFloorAndSharedCapacity) {
  // A 1 MiB copy needs 80 ms alone, while four copies share a 4x faster channel.
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,
      "bandwidth_bytes_per_second":[13107200,26214400,39321600,52428800]}})");
  constexpr size_t bytes = 1 << 20;
  if (LocalCopyThreads(4 * bytes, 4) < 4) GTEST_SKIP() << "requires four copy threads";
  SimTransferEngine engine(Config().sim.profile_path);
  const auto ptr = Buffer(4 * bytes);
  std::vector<TransferItem> items;
  for (size_t i = 0; i < 4; ++i) {
    auto ref = TransferRef::HostBytes(reinterpret_cast<void*>(ptr + i * bytes), bytes);
    items.push_back({ref, 0, ref, 0, bytes, i});
  }
  auto start = Clock::now();
  EXPECT_TRUE(engine.Transfer({items[0]}, nullptr));
  EXPECT_GE(Clock::now() - start, 80ms);
  start = Clock::now();
  EXPECT_TRUE(engine.Transfer(items, nullptr));
  EXPECT_GE(Clock::now() - start, 80ms);
  EXPECT_LT(Clock::now() - start, 250ms);  // Old serial model would take 320 ms.

  std::promise<void> go;
  const auto gate = go.get_future().share();
  std::vector<std::future<bool>> pending;
  for (size_t i = 0; i < 8; ++i) {
    pending.push_back(std::async(std::launch::async, [&, i] {
      gate.wait();
      return engine.Transfer({items[i % 4]}, nullptr);
    }));
  }
  start = Clock::now();
  go.set_value();
  for (auto& result : pending) EXPECT_TRUE(result.get());
  EXPECT_GE(Clock::now() - start, 160ms);  // Concurrent streams still share the cap.
  EXPECT_LT(Clock::now() - start, 500ms);

  // A registered caller arena must not merge the API's distinct KV blocks into
  // one copy: real BuildLocalPageTransfers retains a per-key buffer view.
  auto config = Config();
  config.page_size = bytes;
  config.dram.capacity_bytes = 8 * bytes;
  auto client = CreateUMBPClient(config);
  ASSERT_TRUE(client->RegisterMemory(ptr, 4 * bytes));
  const std::vector<std::string> keys{"a", "b", "c", "d"};
  const std::vector<uintptr_t> ptrs{ptr, ptr + bytes, ptr + 2 * bytes, ptr + 3 * bytes};
  const std::vector<size_t> sizes(4, bytes);
  ASSERT_EQ(client->BatchPut(keys, ptrs, sizes), std::vector<bool>(4, true));
  start = Clock::now();
  EXPECT_EQ(client->BatchGet(keys, ptrs, sizes), std::vector<bool>(4, true));
  EXPECT_GE(Clock::now() - start, 80ms);
  EXPECT_LT(Clock::now() - start, 250ms);
}

TEST_F(SimTest, PerCopyCostUsesCoalescedSegments) {
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,"copy_latency_ns":40000000,
                           "bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine engine(Config().sim.profile_path);
  auto backend = Backend(engine, MemLoc::CPU);
  const auto ref = backend->BufferRef(0);
  auto start = Clock::now();
  EXPECT_TRUE(
      engine.Transfer({{ref, 0, ref, 0, kPage, 0}, {ref, kPage, ref, kPage, kPage, 1}}, nullptr));
  EXPECT_GE(Clock::now() - start, 40ms);
  EXPECT_LT(Clock::now() - start, 70ms);
  start = Clock::now();
  EXPECT_TRUE(engine.Transfer(
      {{ref, 0, ref, 0, kPage, 0}, {ref, 2 * kPage, ref, 2 * kPage, kPage, 1}}, nullptr));
  EXPECT_GE(Clock::now() - start, 80ms);
}

TEST_F(SimTest, ReusesCapacityLruLeasesEventsAndProductionReaper) {
  SimTransferEngine engine(Config().sim.profile_path);
  auto backend = Backend(engine, MemLoc::CPU);
  EXPECT_EQ(backend->Capacity().total_bytes, 4 * kPage);
  backend->TrackEvictionOrder();
  uint64_t committed = 0;
  for (const auto* key : {"a", "b"}) {
    auto slot = backend->Allocate(key, kPage);
    ASSERT_EQ(slot.outcome, AllocateOutcome::kSuccessAllocated);
    ASSERT_TRUE(backend->Commit(slot.slot_id, key, committed));
  }
  EXPECT_EQ(backend->DrainPendingEvents().size(), 2u);
  EXPECT_TRUE(backend->Resolve("a").found);
  auto candidates = backend->EvictionCandidates(2);
  ASSERT_FALSE(candidates.empty());
  EXPECT_EQ(candidates.front().key, "b");
  EXPECT_EQ(backend->Evict({"a"})[0].bytes_freed, 0u);
  EXPECT_EQ(backend->Evict({"b"})[0].bytes_freed, kPage);
  auto pending = backend->Allocate("uncommitted", 2 * kPage);
  ASSERT_EQ(pending.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_FALSE(backend->Contains("uncommitted"));
  // No manual reaper call: the production Init must have started it.
  ASSERT_TRUE(Await([&] { return backend->Capacity().available_bytes == 3 * kPage; }));
  EXPECT_EQ(backend->Evict({"a"})[0].bytes_freed, kPage);
  EXPECT_EQ(backend->Capacity().available_bytes, 4 * kPage);
}

TEST_F(SimTest, ModelCountsBytesOnceAndSerializesSharedChannel) {
  Profile(R"({"cpu_cpu":{"channel":"shared","latency_ns":20000000,
                           "bandwidth_bytes_per_second":409600},
              "cpu_gpu":{"channel":"shared","latency_ns":20000000,
                           "bandwidth_bytes_per_second":409600}})");
  SimTransferEngine engine(Config().sim.profile_path);
  auto a = Backend(engine, MemLoc::CPU);
  auto b = Backend(engine, MemLoc::GPU);
  const auto src = a->BufferRef(0), dst = b->BufferRef(0);
  // Two fragments on one path: 20 ms + 8192 / 409600 = 40 ms, not 60 ms.
  auto plans = engine.Plan({{src, 0, dst, 0, kPage, 0}, {src, kPage, dst, 2 * kPage, kPage, 1}});
  auto start = Clock::now();
  auto handle = engine.Submit(std::move(plans.plans));
  ASSERT_NE(handle, nullptr);
  const auto elapsed = Clock::now() - start;
  EXPECT_GE(elapsed, 40ms);
  EXPECT_LT(elapsed, 500ms);
  std::vector<TransferFailure> failures;
  handle->Wait(&failures);
  handle->Wait(&failures);
  EXPECT_TRUE(failures.empty());

  std::promise<void> go;
  auto gate = go.get_future().share();
  auto run = [&](TransferRef target) {
    gate.wait();
    return engine.Transfer({{src, 0, target, 0, kPage, 0}}, nullptr);
  };
  auto first = std::async(std::launch::async, run, src);
  auto second = std::async(std::launch::async, run, dst);
  start = Clock::now();
  go.set_value();
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
  EXPECT_GE(Clock::now() - start, 60ms);
  EXPECT_FALSE(engine.CanHandle(src, TransferRef::File(1, 0, kPage)));
  auto remote = src;
  remote.host_ptr = nullptr;
  EXPECT_FALSE(engine.CanHandle(src, remote));
  EXPECT_EQ(engine.Plan({{src, 4 * kPage, dst, 0, kPage, 7}}).rejected_tags,
            std::vector<size_t>{7});
}

TEST_F(SimTest, TierDemotionUsesVirtualBuffersAndSimEngine) {
  Profile(R"({"gpu_cpu":{"channel":"link","latency_ns":10000000,
                           "bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine engine(Config().sim.profile_path);
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("hot", Backend(engine, MemLoc::GPU)));
  ASSERT_TRUE(registry.Register("cold", Backend(engine, MemLoc::CPU)));
  LogicalTierConfig hot, cold;
  hot.name = "hot";
  hot.entry = true;
  hot.members = {{"hot", 100}};
  hot.offload_to = {"cold"};
  hot.trigger = PoolOffloadTrigger::kOnEvict;
  cold.name = "cold";
  cold.members = {{"cold", 100}};
  auto compiled = LogicalTierGraph::Compile({hot, cold}, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  PoolPlacementRequest request;
  request.key = "move";
  request.size = kPage;
  auto slot = pool.BatchAllocate({request}).front();
  ASSERT_EQ(slot.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(pool.BatchCommit({{{slot.backend_id, slot.allocation.slot_id}, "move"}})
                  .front()
                  .commit.success);
  const auto start = Clock::now();
  EXPECT_EQ(pool.Evict({"move"}, PoolEvictMode::kReclaim).front().outcome,
            PoolEvictOutcome::kDemotionQueued);
  ASSERT_TRUE(Await([&] { return !registry.Get("hot")->Contains("move"); }));
  EXPECT_TRUE(registry.Get("cold")->Contains("move"));
  EXPECT_GE(Clock::now() - start, 10ms);
}

TEST_F(SimTest, RejectsUnsupportedConfigurationsAndBadProfiles) {
  auto config = Config();
  config.standalone_process.emplace();
  EXPECT_THROW(CreateUMBPClient(config), std::invalid_argument);
  config = Config();
  config.sim.profile_path.clear();
  EXPECT_THROW(CreateUMBPClient(config), std::runtime_error);
  config = Config();
  config.distributed.emplace();
  config.distributed->medium = UMBPMedium::SSD;
  EXPECT_THROW(CreateUMBPClient(config), std::runtime_error);
  config.distributed->medium = UMBPMedium::DRAM;
  config.distributed->io_engine.host = "127.0.0.1";
  EXPECT_NO_THROW(CreateUMBPClient(config));  // Sim replaces the configured RDMA engine.
  config.distributed->ranged_scratch_size = kPage;
  EXPECT_NO_THROW(CreateUMBPClient(config));
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,
                           "bandwidth_bytes_per_second":0}})");
  EXPECT_THROW(SimTransferEngine(Config().sim.profile_path), std::invalid_argument);
  for (const auto* curve : {"[]", "[1,0]", "[1,-1]", "[1,\"fast\"]"}) {
    Profile(std::string(R"({"cpu_cpu":{"channel":"host","latency_ns":0,
                           "bandwidth_bytes_per_second":)") +
            curve + "}}");
    EXPECT_THROW(SimTransferEngine(Config().sim.profile_path), std::invalid_argument);
  }
  PoolClientConfig pool_config;
  pool_config.sim = Config().sim;
  pool_config.dram.buffer_sizes = {32 * kPage};
  pool_config.dram_page_size = kPage;
  PoolClient pool(pool_config);
  EXPECT_FALSE(pool.Init());
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,
                           "bandwidth_bytes_per_second":1e12}})");
  EXPECT_TRUE(pool.Init()) << "invalid profile must not leave the client initialized";
}

TEST_F(SimTest, MissingPathAbortsReservationAndDefaultModeStillCopies) {
  Profile(R"({"gpu_gpu":{"channel":"hbm","latency_ns":0,
                           "bandwidth_bytes_per_second":1e12}})");
  auto client = CreateUMBPClient(Config());
  char src = 'x', dst = 'y';
  EXPECT_FALSE(client->Put("unsupported", reinterpret_cast<uintptr_t>(&src), 1));
  EXPECT_FALSE(client->Exists("unsupported"));
  auto config = Config();
  config.sim.enabled = false;
  config.sim.profile_path = "/does/not/exist";
  config.sim.trace_path = "/does/not/exist";
  auto real = CreateUMBPClient(config);
  EXPECT_TRUE(real->Put("real", reinterpret_cast<uintptr_t>(&src), 1));
  EXPECT_TRUE(real->Get("real", reinterpret_cast<uintptr_t>(&dst), 1));
  EXPECT_EQ(dst, src);
}

std::string Packed(const PeerDirectory& engine) {
  const auto bytes = engine.PackedLocalEngineDesc();
  return {bytes.begin(), bytes.end()};
}

TEST_F(SimTest, RemotePlansPreservePeerIdentityAndAsynchronousCompletion) {
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,"bandwidth_bytes_per_second":1e12},
    "remote_read":{"channel":"nic-rx","latency_ns":80000000,"bandwidth_bytes_per_second":1e12},
    "remote_write":{"channel":"nic-tx","latency_ns":80000000,"bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine caller(Config().sim.profile_path, "caller", 2 * kPage);
  SimTransferEngine first(Config().sim.profile_path, "first");
  SimTransferEngine second(Config().sim.profile_path, "second");
  ASSERT_TRUE(caller.EnsureRemoteEngine("first", Packed(first)));
  ASSERT_TRUE(caller.EnsureRemoteEngine("second", Packed(second)));
  auto* buffer = reinterpret_cast<void*>(Buffer(4 * kPage));
  auto local = caller.RegisterMemory(buffer, 4 * kPage, MemLoc::CPU, -1);
  // Deliberately equal virtual addresses and memory IDs on different peers.
  auto a = TransferRef::Remote(first.RegisterMemory(buffer, kPage, MemLoc::CPU, -1).mem);
  auto b = TransferRef::Remote(second.RegisterMemory(buffer, kPage, MemLoc::CPU, -1).mem);
  auto plans = caller.Plan({{a, 0, local, 0, kPage, 0}, {b, 0, local, kPage, kPage, 1}});
  ASSERT_EQ(plans.plans.size(), 2u);
  EXPECT_NE(plans.plans[0].src.mem.engineKey, plans.plans[1].src.mem.engineKey);
  EXPECT_EQ(plans.plans[0].dir, TransferDirection::kPull);
  auto start = Clock::now();
  auto read = caller.Submit(std::move(plans.plans));
  auto writes = caller.Plan({{local, 0, a, 0, kPage, 0}});
  auto write = caller.Submit(std::move(writes.plans));
  ASSERT_NE(read, nullptr);
  ASSERT_NE(write, nullptr);
  EXPECT_LT(Clock::now() - start, 60ms);  // Both posted, neither has completed yet.
  std::vector<TransferFailure> failures;
  read->Wait(&failures);
  write->Wait(&failures);
  EXPECT_GE(Clock::now() - start, 80ms);
  EXPECT_TRUE(failures.empty());
  EXPECT_FALSE(caller.CanHandle(a, b));
  EXPECT_EQ(caller.Plan({{local, 0, a, kPage, 1, 0}}).rejected_tags.size(), 1u);

  auto unregistered = TransferRef::HostBytes(buffer, 4 * kPage);
  auto staged = caller.Plan({{unregistered, 0, a, 0, kPage, 0}});
  ASSERT_EQ(staged.plans.size(), 1u);
  EXPECT_TRUE(staged.plans[0].uses_bounce);
  start = Clock::now();
  auto staged_handle = caller.Submit(std::move(staged.plans));
  ASSERT_NE(staged_handle, nullptr);
  EXPECT_GE(Clock::now() - start, 80ms);  // Staging completes inside Submit, as in RDMA.
  staged_handle->Wait(&failures);
  EXPECT_TRUE(failures.empty());
  EXPECT_TRUE(caller.Transfer({{a, 0, unregistered, 0, kPage, 0}}, nullptr));

  caller.ForgetRemote("first");
  EXPECT_FALSE(caller.CanHandle(a, local));
  mori::io::IOEngineConfig io;
  MoriIoEngine real("real", io, 0);  // No RDMA initialization needed for handshake rejection.
  EXPECT_FALSE(real.EnsureRemoteEngine("sim", Packed(caller)));
  mori::io::EngineDesc real_desc{};
  real_desc.key = "real";
  msgpack::sbuffer packed;
  msgpack::pack(packed, real_desc);
  EXPECT_FALSE(caller.EnsureRemoteEngine("real", {packed.data(), packed.size()}));
}

class SimClusterTest : public SimTest {
 protected:
  void SetUp() override {
    SimTest::SetUp();
    Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,"bandwidth_bytes_per_second":1e10},
      "remote_read":{"channel":"nic","latency_ns":1000000,"bandwidth_bytes_per_second":4096000},
      "remote_write":{"channel":"nic","latency_ns":1000000,"bandwidth_bytes_per_second":4096000}})");
    MasterServerConfig config;
    config.listen_address = "127.0.0.1:0";
    master_ = std::make_unique<MasterServer>(std::move(config));
    thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 200 && master_->GetBoundPort() == 0; ++i) std::this_thread::sleep_for(5ms);
    ASSERT_NE(master_->GetBoundPort(), 0);
    ptr_ = Buffer(64 * kPage);
  }
  void TearDown() override {
    master_->Shutdown();
    if (thread_.joinable()) thread_.join();
    SimTest::TearDown();
  }
  UMBPConfig Node(const std::string& id, size_t pages = 32) {
    auto config = Config();
    config.dram.capacity_bytes = pages * kPage;
    config.distributed.emplace();
    auto& dc = *config.distributed;
    dc.master_config.node_id = id;
    dc.master_config.node_address = "127.0.0.1";
    dc.master_config.master_address = "127.0.0.1:" + std::to_string(master_->GetBoundPort());
    dc.auto_peer_service_port = true;
    dc.io_engine.host = "0.0.0.0";  // Existing deployment setting, no actual RDMA started.
    dc.ranged_scratch_size = 32 * kPage;
    dc.cache_remote_fetches = false;
    dc.ranged_locality_prefetch = false;
    config.sim.trace_path = dir_ + "/" + id + ".jsonl";
    return config;
  }
  std::unique_ptr<PoolClient> Pool(const std::string& id, bool recache) {
    const auto config = Node(id);
    PoolClientConfig pc;
    pc.sim = config.sim;
    pc.master_config = config.distributed->master_config;
    pc.auto_peer_service_port = true;
    pc.dram.buffer_sizes = {32 * kPage};
    pc.dram_page_size = kPage;
    pc.cache_remote_fetches = recache;
    auto client = std::make_unique<PoolClient>(pc);
    EXPECT_TRUE(client->Init());
    EXPECT_TRUE(client->RegisterMemory(reinterpret_cast<void*>(ptr_), 64 * kPage));
    return client;
  }
  uintptr_t ptr_ = 0;
  std::unique_ptr<MasterServer> master_;
  std::thread thread_;
};

TEST_F(SimClusterTest, PublicShareAndRangedOffloadRunRealControlRpcs) {
  auto owner = CreateUMBPClient(Node("owner", 64));
  ASSERT_TRUE(owner->Put("shared", ptr_, 2 * kPage));
  ASSERT_TRUE(owner->Flush());
  auto reader = CreateUMBPClient(Node("reader", 1));
  ASSERT_TRUE(reader->RegisterMemory(ptr_, 64 * kPage));
  ASSERT_TRUE(Await([&] { return reader->Exists("shared"); }));
  auto start = Clock::now();
  ASSERT_TRUE(reader->Get("shared", ptr_, 2 * kPage));
  EXPECT_GE(Clock::now() - start, 3ms);
  EXPECT_EQ(reader->BatchGet({"shared", "missing"}, {ptr_, ptr_}, {2 * kPage, kPage}),
            (std::vector<bool>{true, false}));
  EXPECT_EQ(reader->BatchGetRanges({"shared"}, {{ptr_, ptr_ + kPage}}, {{128, 128}},
                                   {{kPage - 64, 2 * kPage - 128}}),
            std::vector<bool>{true});
  // Object exceeds reader capacity: Master must place both forms on owner.
  EXPECT_TRUE(reader->Put("offload", ptr_, 4 * kPage));
  EXPECT_EQ(reader->BatchPutRanges({"ranged-offload"}, {4 * kPage}, {{ptr_, ptr_ + 2 * kPage}},
                                   {{2 * kPage, 2 * kPage}}, {{0, 2 * kPage}}),
            std::vector<bool>{true});
  EXPECT_TRUE(owner->Get("offload", ptr_, 4 * kPage));
  EXPECT_TRUE(owner->Get("ranged-offload", ptr_, 4 * kPage));
  ASSERT_TRUE(owner->Clear());
  EXPECT_FALSE(reader->Get("shared", ptr_, 2 * kPage));
  EXPECT_FALSE(reader->Get("offload", ptr_, 4 * kPage));
  EXPECT_FALSE(reader->Get("ranged-offload", ptr_, 4 * kPage));
  reader->Close();
  std::ifstream trace(dir_ + "/reader.jsonl");
  std::string line;
  bool timed_remote_get = false;
  while (std::getline(trace, line)) {
    google::protobuf::Struct record;
    ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(line, &record).ok());
    const auto& f = record.fields();
    if (f.at("api").string_value() == "get" && f.at("result_count").number_value() == 1) {
      timed_remote_get = true;
      EXPECT_GE(f.at("elapsed_ns").number_value(), 3000000);
    }
  }
  EXPECT_TRUE(timed_remote_get);
}

TEST_F(SimClusterTest, RemoteRecacheAndEvictionKeepRealMetadata) {
  auto owner = Pool("owner", false);
  auto* payload = reinterpret_cast<void*>(ptr_);
  ASSERT_TRUE(owner->Put("cached", payload, kPage));
  ASSERT_TRUE(owner->Put("evicted", payload, kPage));
  owner->Master().FlushHeartbeat();
  auto reader = Pool("reader", true);
  ASSERT_TRUE(Await([&] { return reader->Exists("cached"); }));
  ASSERT_TRUE(reader->Get("cached", payload, kPage));
  auto* backend = reader->Backends().Get(TierType::DRAM);
  ASSERT_TRUE(Await([&] { return backend->Contains("cached"); }));
  EXPECT_EQ(owner->Backends().Get(TierType::DRAM)->Evict({"evicted"}).front().bytes_freed, kPage);
  owner->Master().FlushHeartbeat();
  EXPECT_FALSE(reader->Get("evicted", payload, kPage));
  ASSERT_TRUE(owner->Clear());
  EXPECT_TRUE(reader->Get("cached", payload, kPage));
  reader->Master().FlushHeartbeat();
  auto observer = Pool("observer", false);
  ASSERT_TRUE(Await([&] { return observer->Exists("cached"); }));
  EXPECT_TRUE(observer->Get("cached", payload, kPage));  // Re-cached replica serves another peer.
}

TEST_F(SimClusterTest, MissingNetworkPathAbortsRemoteWrite) {
  auto owner = Pool("owner", false);
  auto* backend = owner->Backends().Get(TierType::DRAM);
  const auto available = backend->Capacity().available_bytes;
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,"bandwidth_bytes_per_second":1e10}})");
  auto reader = CreateUMBPClient(Node("reader", 1));
  ASSERT_TRUE(reader->RegisterMemory(ptr_, 64 * kPage));
  EXPECT_FALSE(reader->Put("failed", ptr_, 2 * kPage));
  EXPECT_FALSE(owner->Exists("failed"));
  EXPECT_EQ(backend->Capacity().available_bytes, available);
  EXPECT_TRUE(owner->Put("failed", reinterpret_cast<void*>(ptr_), 2 * kPage));
}

TEST_F(SimTest, RemoteDescriptorsKeepBackendAndGenerationIdentity) {
  SimTransferEngine caller(Config().sim.profile_path, "caller");
  SimTransferEngine owner(Config().sim.profile_path, "owner");
  ASSERT_TRUE(caller.EnsureRemoteEngine("owner", Packed(owner)));
  auto first = owner.RegisterMemory(reinterpret_cast<void*>(0x10000), kPage, MemLoc::CPU, -1);
  auto second = owner.RegisterMemory(reinterpret_cast<void*>(0x20000), kPage, MemLoc::GPU, 0);
  auto descriptor = [](const TransferRef& ref, uint32_t backend) {
    msgpack::sbuffer bytes;
    msgpack::pack(bytes, ref.mem);
    BufferMemoryDescBytes desc;
    desc.backend_id = backend;
    desc.buffer_index = 0;
    desc.desc_bytes.assign(bytes.data(), bytes.data() + bytes.size());
    return desc;
  };
  caller.CacheRemoteBuffers("owner", {descriptor(first, 0), descriptor(second, 1)});
  ASSERT_TRUE(caller.HasRemoteBuffers("owner"));
  EXPECT_EQ(caller.RemoteBufferSnapshot("owner", 0).front().mem.id, first.mem.id);
  EXPECT_EQ(caller.RemoteBufferSnapshot("owner", 1).front().mem.id, second.mem.id);
  caller.CacheRemoteBuffers("owner", {descriptor(second, 0)});
  EXPECT_EQ(caller.RemoteBufferSnapshot("owner", 0).front().mem.id, first.mem.id);
  SimTransferEngine restarted(Config().sim.profile_path, "owner");
  ASSERT_TRUE(caller.EnsureRemoteEngine("owner", Packed(restarted)));
  EXPECT_FALSE(caller.HasRemoteBuffers("owner"));
  caller.CacheRemoteBuffers("owner", {descriptor(first, 0)});  // Old process generation.
  EXPECT_FALSE(caller.HasRemoteBuffers("owner"));
}

TEST_F(SimClusterTest, VirtualHbmOwnerServesRegisteredGpuCaller) {
  Profile(R"({"gpu_gpu":{"channel":"hbm","latency_ns":0,"bandwidth_bytes_per_second":1e10},
    "remote_read":{"channel":"nic","latency_ns":1000000,"bandwidth_bytes_per_second":4096000},
    "remote_write":{"channel":"nic","latency_ns":1000000,"bandwidth_bytes_per_second":4096000}})");
  auto config = Node("hbm-owner", 64);
  config.distributed->medium = UMBPMedium::HBM;
  config.distributed->hbm.capacity_bytes = 64 * kPage;
  auto owner = CreateUMBPClient(config);
  ASSERT_TRUE(owner->RegisterMemory(ptr_, 64 * kPage, MemLoc::GPU, 0));
  ASSERT_TRUE(owner->Put("hbm-shared", ptr_, 2 * kPage));
  ASSERT_TRUE(owner->Flush());
  auto reader = CreateUMBPClient(Node("gpu-reader", 1));
  ASSERT_TRUE(reader->RegisterMemory(ptr_, 64 * kPage, MemLoc::GPU, 0));
  ASSERT_TRUE(Await([&] { return reader->Exists("hbm-shared"); }));
  EXPECT_TRUE(reader->Get("hbm-shared", ptr_, 2 * kPage));
  EXPECT_TRUE(reader->Put("hbm-offload", ptr_, 4 * kPage));
  EXPECT_TRUE(owner->Get("hbm-offload", ptr_, 4 * kPage));
}

TEST_F(SimTest, BounceCopiesUseSerialCpuBandwidth) {
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,
    "bandwidth_bytes_per_second":[10485760,20971520,31457280,41943040]},
    "remote_write":{"channel":"nic","latency_ns":0,"bandwidth_bytes_per_second":1e12}})");
  constexpr size_t bytes = 1 << 20;
  SimTransferEngine caller(Config().sim.profile_path, "caller", bytes);
  SimTransferEngine owner(Config().sim.profile_path, "owner");
  ASSERT_TRUE(caller.EnsureRemoteEngine("owner", Packed(owner)));
  auto* buffer = reinterpret_cast<void*>(Buffer(bytes));
  auto local = TransferRef::HostBytes(buffer, bytes);
  auto remote = TransferRef::Remote(owner.RegisterMemory(buffer, bytes, MemLoc::CPU, -1).mem);
  std::vector<TransferItem> items;
  for (size_t i = 0; i < 4; ++i)
    items.push_back({local, i * bytes / 4, remote, i * bytes / 4, bytes / 4, i});
  const auto start = Clock::now();
  ASSERT_TRUE(caller.Transfer(items, nullptr));
  EXPECT_GE(Clock::now() - start, 100ms);  // CPU staging is serial even with four copy threads.
}

}  // namespace
}  // namespace mori::umbp

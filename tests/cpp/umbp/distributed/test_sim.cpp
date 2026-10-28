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

#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/pool/tier_graph.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/sim/sim.h"
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
  std::string dir_;
};

TEST_F(SimTest, PublicApisNeverAccessPayloadAndRecordOutcomes) {
  auto config = Config();
  SimPageMemorySource caller(MemLoc::CPU, -1);
  std::vector<PageMemorySource::Buffer> buffers;
  ASSERT_TRUE(caller.Allocate({8 * kPage}, &buffers));
  const auto ptr = reinterpret_cast<uintptr_t>(buffers[0].base);
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
  SimPageMemorySource caller(MemLoc::GPU, 0);
  std::vector<PageMemorySource::Buffer> buffers;
  ASSERT_TRUE(caller.Allocate({4 * kPage}, &buffers));
  const auto ptr = reinterpret_cast<uintptr_t>(buffers[0].base);
  auto client = CreateUMBPClient(config);
  ASSERT_TRUE(client->RegisterMemory(ptr, 4 * kPage, MemLoc::GPU, 0));
  EXPECT_TRUE(client->Put("gpu", ptr + kPage, kPage));
  EXPECT_TRUE(client->Get("gpu", ptr + 2 * kPage, kPage));
  EXPECT_EQ(client->BatchGetRanges({"gpu"}, {{ptr + 3 * kPage}}, {{128}}, {{0}}),
            std::vector<bool>{true});
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
  const auto deadline = Clock::now() + 2s;
  while (backend->Capacity().available_bytes != 3 * kPage && Clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_EQ(backend->Capacity().available_bytes, 3 * kPage);
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
  while (registry.Get("hot")->Contains("move") && Clock::now() - start < 2s) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_FALSE(registry.Get("hot")->Contains("move"));
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
  EXPECT_THROW(CreateUMBPClient(config), std::runtime_error);
  config.distributed->io_engine.host.clear();
  config.distributed->ranged_scratch_size = kPage;
  EXPECT_THROW(CreateUMBPClient(config), std::invalid_argument);
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,
                           "bandwidth_bytes_per_second":0}})");
  EXPECT_THROW(SimTransferEngine(Config().sim.profile_path), std::invalid_argument);
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

}  // namespace
}  // namespace mori::umbp

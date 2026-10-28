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
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/sim/sim.h"
#include "umbp/standalone/standalone_server.h"
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
  void TearDown() override {
    servers_.clear();
    std::filesystem::remove_all(dir_);
  }
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
  UMBPConfig RankConfig() {
    UMBPConfig config;
    config.sim.enabled = true;
    config.sim.trace_path = dir_ + "/rank.jsonl";
    config.standalone_process.emplace();
    config.standalone_process->address = "unix:" + dir_ + "/server.sock";
    return config;
  }
  std::unique_ptr<IUMBPClient> Client(UMBPConfig config, bool use_standalone) {
    if (use_standalone) {
      auto rank = RankConfig();
      rank.sim.trace_path = config.sim.trace_path;
      rank.standalone_process->address =
          "unix:" + dir_ + "/server-" + std::to_string(servers_.size()) + ".sock";
      config.sim.trace_path += ".server";
      auto server =
          std::make_unique<standalone::StandaloneServer>(config, rank.standalone_process->address);
      if (!server->Start()) throw std::runtime_error("standalone server failed to start");
      servers_.push_back(std::move(server));
      config = rank;
    }
    return CreateUMBPClient(config);
  }
  std::unique_ptr<PageBackend> Backend(SimTransferEngine& engine, size_t pages = 4) {
    auto backend = std::make_unique<PageBackend>(
        TierType::DRAM, kPage, std::make_unique<SimPageMemorySource>(MemLoc::CPU, -1),
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
  std::vector<std::unique_ptr<standalone::StandaloneServer>> servers_;
};

// Exercise the same API contract embedded and through the standalone IPC boundary.
class SimApiTest : public SimTest, public ::testing::WithParamInterface<bool> {};
INSTANTIATE_TEST_SUITE_P(EmbeddedAndStandalone, SimApiTest, ::testing::Bool());

TEST_P(SimApiTest, PublicApisNeverAccessPayloadAndRecordOutcomes) {
  auto config = Config();
  const auto ptr = Buffer(8 * kPage);
  auto client = Client(config, GetParam());
  ASSERT_TRUE(client->RegisterMemory(ptr, 8 * kPage));
  EXPECT_TRUE(client->Put("scalar", ptr, 2 * kPage));
  EXPECT_TRUE(client->Get("scalar", ptr, 2 * kPage));
  EXPECT_FALSE(client->Get("missing", ptr, kPage));
  EXPECT_EQ(client->BatchPut({"a", "b"}, {ptr, ptr + kPage}, {kPage, kPage}),
            (std::vector<bool>{true, true}));
  EXPECT_EQ(client->BatchGet({"a", "missing"}, {ptr, ptr}, {kPage, kPage}),
            (std::vector<bool>{true, false}));
  ASSERT_TRUE(client->SupportsRangedIO());
  EXPECT_EQ(client->BatchPutRanges({"range"}, {2 * kPage}, {{ptr, ptr + kPage}}, {{kPage, kPage}},
                                   {{0, kPage}}),
            std::vector<bool>{true});
  EXPECT_EQ(client->BatchGetRanges({"range"}, {{ptr, ptr + kPage}}, {{128, 128}},
                                   {{kPage - 64, 2 * kPage - 128}}),
            std::vector<bool>{true});
  EXPECT_TRUE(client->Clear());
  EXPECT_EQ(client->BatchExists({"a", "scalar", "range"}), std::vector<bool>(3, false));
  client->Close();

  std::ifstream trace(config.sim.trace_path);
  std::string line;
  bool saw_put = false, saw_partial = false;
  while (std::getline(trace, line)) {
    google::protobuf::Struct record;
    ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(line, &record).ok());
    const auto& f = record.fields();
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

TEST_P(SimApiTest, VirtualGpuCallerLoadsAndOffloadsWithoutDevice) {
  for (auto medium : {UMBPMedium::DRAM, UMBPMedium::HBM}) {
    Profile(
        medium == UMBPMedium::HBM
            ? R"({"gpu_gpu":{"channel":"hbm","latency_ns":0,"bandwidth_bytes_per_second":1e12}})"
            : R"({"gpu_cpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e12},
                       "cpu_gpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e12}})");
    auto config = Config();
    config.distributed.emplace();
    config.distributed->medium = medium;
    config.distributed->hbm.capacity_bytes = 32 * kPage;
    auto client = Client(config, GetParam());
    const auto ptr = Buffer(kPage);
    ASSERT_TRUE(client->RegisterMemory(ptr, kPage, MemLoc::GPU, 0));
    EXPECT_TRUE(client->Put("gpu", ptr, kPage));
    EXPECT_TRUE(client->Get("gpu", ptr, kPage));
  }
}

TEST_F(SimTest, StandaloneRanksShareCapacityLruAndRegistrationLifetime) {
  auto config = Config();
  config.dram.high_watermark = 0.6;
  config.dram.low_watermark = 0.5;
  standalone::StandaloneServer server(config, RankConfig().standalone_process->address);
  ASSERT_TRUE(server.Start());
  auto rank0 = CreateUMBPClient(RankConfig());
  auto other = RankConfig();
  other.sim.trace_path = dir_ + "/rank1.jsonl";
  auto rank1 = CreateUMBPClient(other);
  const auto ptr = Buffer(8 * kPage);
  // Identical virtual addresses in separate ranks must remain independent registrations.
  ASSERT_TRUE(rank0->RegisterMemory(ptr, 8 * kPage));
  ASSERT_TRUE(rank1->RegisterMemory(ptr, 8 * kPage));
  ASSERT_TRUE(rank0->Put("a", ptr, 8 * kPage));
  ASSERT_TRUE(rank1->Put("b", ptr, 8 * kPage));
  ASSERT_TRUE(rank1->Get("a", ptr, 8 * kPage));  // Touch a; b is now the oldest.
  ASSERT_TRUE(rank1->Put("c", ptr, 8 * kPage));
  ASSERT_TRUE(Await([&] { return !rank0->Exists("b"); }));
  EXPECT_EQ(rank1->BatchExists({"a", "b", "c"}), (std::vector<bool>{true, false, true}));
  rank0->Close();
  EXPECT_TRUE(rank1->Exists("a"));  // Rank exit does not clear the shared tier.
  rank1->DeregisterMemory(ptr);
  EXPECT_FALSE(rank1->Get("a", ptr, 8 * kPage));
  EXPECT_TRUE(rank1->Clear());
  EXPECT_FALSE(rank1->Exists("c"));
}

TEST_F(SimTest, StandaloneRejectsMismatchedModes) {
  for (bool sim : {false, true}) {
    auto config = Config();
    config.sim.enabled = sim;
    standalone::StandaloneServer server(config, RankConfig().standalone_process->address);
    ASSERT_TRUE(server.Start());
    auto rank = RankConfig();
    rank.sim.enabled = !sim;
    EXPECT_THROW(CreateUMBPClient(rank), std::runtime_error);
  }
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

TEST_F(SimTest, ModelCountsBytesOnceAndUsesCoalescedCopyCosts) {
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":40000000,"copy_latency_ns":40000000,
                           "bandwidth_bytes_per_second":409600}})");
  SimTransferEngine engine(Config().sim.profile_path);
  auto backend = Backend(engine);
  const auto ref = backend->BufferRef(0);
  auto start = Clock::now();
  EXPECT_TRUE(
      engine.Transfer({{ref, 0, ref, 0, kPage, 0}, {ref, kPage, ref, kPage, kPage, 1}}, nullptr));
  EXPECT_GE(Clock::now() - start, 100ms);  // 40 ms fixed + 40 ms/copy + 20 ms bytes.
  EXPECT_LT(Clock::now() - start, 130ms);
  start = Clock::now();
  EXPECT_TRUE(engine.Transfer(
      {{ref, 0, ref, 0, kPage, 0}, {ref, 2 * kPage, ref, 2 * kPage, kPage, 1}}, nullptr));
  EXPECT_GE(Clock::now() - start, 140ms);  // Same bytes and fixed cost, one extra copy.
  EXPECT_LT(Clock::now() - start, 170ms);
}

TEST_F(SimTest, ReusesCapacityLeasesAndProductionReaper) {
  SimTransferEngine engine(Config().sim.profile_path);
  auto backend = Backend(engine);
  EXPECT_EQ(backend->Capacity().total_bytes, 4 * kPage);
  uint64_t committed = 0;
  auto slot = backend->Allocate("a", kPage);
  ASSERT_EQ(slot.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(backend->Commit(slot.slot_id, "a", committed));
  EXPECT_TRUE(backend->Resolve("a").found);
  EXPECT_EQ(backend->Evict({"a"})[0].bytes_freed, 0u);
  auto pending = backend->Allocate("uncommitted", 2 * kPage);
  ASSERT_EQ(pending.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_FALSE(backend->Contains("uncommitted"));
  // No manual reaper call: the production Init must have started it.
  ASSERT_TRUE(Await([&] { return backend->Capacity().available_bytes == 3 * kPage; }));
  EXPECT_EQ(backend->Evict({"a"})[0].bytes_freed, kPage);
  EXPECT_EQ(backend->Capacity().available_bytes, 4 * kPage);
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

TEST_F(SimTest, MixedSubmitWaitsForLocalPrefixAndReservesRemoteCompletion) {
  for (bool shared : {false, true}) {
    SCOPED_TRACE(shared ? "shared channel" : "independent channels");
    Profile(std::string(R"({"cpu_cpu":{"channel":"host","latency_ns":20000000,
      "bandwidth_bytes_per_second":1e12},"remote_write":{"channel":")") +
            (shared ? "host" : "nic") +
            R"(","latency_ns":100000000,"bandwidth_bytes_per_second":1e12}})");
    SimTransferEngine caller(Config().sim.profile_path, "caller");
    SimTransferEngine owner(Config().sim.profile_path, "owner");
    SimTransferEngine other(Config().sim.profile_path, "other");
    ASSERT_TRUE(caller.EnsureRemoteEngine("owner", Packed(owner)));
    ASSERT_TRUE(caller.EnsureRemoteEngine("other", Packed(other)));
    auto* buffer = reinterpret_cast<void*>(Buffer(kPage));
    const auto local = caller.RegisterMemory(buffer, kPage, MemLoc::CPU, -1);
    const auto remote =
        TransferRef::Remote(owner.RegisterMemory(buffer, kPage, MemLoc::CPU, -1).mem);
    const auto another =
        TransferRef::Remote(other.RegisterMemory(buffer, kPage, MemLoc::CPU, -1).mem);
    auto plans = caller.Plan({{local, 0, local, 0, kPage, 0},
                              {local, 0, remote, 0, kPage, 1},
                              {local, 0, another, 0, kPage, 2}});
    ASSERT_EQ(plans.plans.size(), 3u);  // Equal addresses on different peers stay distinct.
    EXPECT_NE(plans.plans[1].dst.mem.engineKey, plans.plans[2].dst.mem.engineKey);
    const auto start = Clock::now();
    auto mixed = caller.Submit(std::move(plans.plans));
    ASSERT_NE(mixed, nullptr);
    EXPECT_GE(Clock::now() - start, 20ms);
    EXPECT_LT(Clock::now() - start, 80ms);  // The remote part remains asynchronous.
    auto next = caller.Plan({{local, 0, remote, 0, kPage, 0}});
    auto queued = caller.Submit(std::move(next.plans));
    ASSERT_NE(queued, nullptr);
    std::vector<TransferFailure> failures;
    mixed->Wait(&failures);
    EXPECT_GE(Clock::now() - start, shared ? 120ms : 100ms);
    queued->Wait(&failures);
    EXPECT_GE(Clock::now() - start, shared ? 220ms : 200ms);
    EXPECT_TRUE(failures.empty());
  }
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
    SimTest::TearDown();
    master_->Shutdown();
    if (thread_.joinable()) thread_.join();
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

class SimClusterApiTest : public SimClusterTest, public ::testing::WithParamInterface<bool> {};
INSTANTIATE_TEST_SUITE_P(EmbeddedAndStandalone, SimClusterApiTest, ::testing::Bool());

TEST_P(SimClusterApiTest, PublicShareAndRangedOffloadRunRealControlRpcs) {
  auto owner = Client(Node("owner", 64), GetParam());
  ASSERT_TRUE(owner->RegisterMemory(ptr_, 64 * kPage));
  ASSERT_TRUE(owner->Put("shared", ptr_, 2 * kPage));
  ASSERT_TRUE(owner->Flush());
  auto reader = Client(Node("reader", 1), GetParam());
  ASSERT_TRUE(reader->SupportsRangedIO());
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
}

TEST_P(SimClusterApiTest, RemoteRangedIoRequiresScratch) {
  auto owner = Client(Node("owner", 64), GetParam());
  ASSERT_TRUE(owner->RegisterMemory(ptr_, 64 * kPage));
  ASSERT_TRUE(owner->Put("shared", ptr_, 2 * kPage));
  ASSERT_TRUE(owner->Flush());
  auto config = Node("reader", 1);
  config.distributed->ranged_scratch_size = 0;
  auto reader = Client(config, GetParam());
  EXPECT_FALSE(reader->SupportsRangedIO());
  ASSERT_TRUE(reader->RegisterMemory(ptr_, 64 * kPage));
  ASSERT_TRUE(reader->Get("shared", ptr_, 2 * kPage));
  EXPECT_EQ(reader->BatchGetRanges({"shared"}, {{ptr_}}, {{kPage}}, {{0}}),
            std::vector<bool>{false});
  EXPECT_EQ(reader->BatchPutRanges({"offload"}, {2 * kPage}, {{ptr_}}, {{2 * kPage}}, {{0}}),
            std::vector<bool>{false});
  EXPECT_FALSE(owner->Exists("offload"));
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

TEST_F(SimTest, PeerDescriptorsRejectStaleGenerationsAndRealEngines) {
  SimTransferEngine caller(Config().sim.profile_path, "caller");
  SimTransferEngine owner(Config().sim.profile_path, "owner");
  ASSERT_TRUE(caller.EnsureRemoteEngine("owner", Packed(owner)));
  auto first = owner.RegisterMemory(reinterpret_cast<void*>(0x10000), kPage, MemLoc::CPU, -1);
  auto descriptor = [](const TransferRef& ref, uint32_t backend) {
    msgpack::sbuffer bytes;
    msgpack::pack(bytes, ref.mem);
    BufferMemoryDescBytes desc;
    desc.backend_id = backend;
    desc.buffer_index = 0;
    desc.desc_bytes.assign(bytes.data(), bytes.data() + bytes.size());
    return desc;
  };
  caller.CacheRemoteBuffers("owner", {descriptor(first, 0)});
  ASSERT_TRUE(caller.HasRemoteBuffers("owner"));
  EXPECT_EQ(caller.RemoteBufferSnapshot("owner", 0).front().mem.id, first.mem.id);
  SimTransferEngine restarted(Config().sim.profile_path, "owner");
  ASSERT_TRUE(caller.EnsureRemoteEngine("owner", Packed(restarted)));
  EXPECT_FALSE(caller.HasRemoteBuffers("owner"));
  caller.CacheRemoteBuffers("owner", {descriptor(first, 0)});  // Old process generation.
  EXPECT_FALSE(caller.HasRemoteBuffers("owner"));
  MoriIoEngine real("real", mori::io::IOEngineConfig{}, 0);
  EXPECT_FALSE(real.EnsureRemoteEngine("sim", Packed(caller)));
  mori::io::EngineDesc real_desc{};
  real_desc.key = "real";
  msgpack::sbuffer packed;
  msgpack::pack(packed, real_desc);
  EXPECT_FALSE(caller.EnsureRemoteEngine("real", {packed.data(), packed.size()}));
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
    "remote_read":{"channel":"nic","latency_ns":0,"bandwidth_bytes_per_second":1e12},
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
  EXPECT_TRUE(caller.Transfer({{remote, 0, local, 0, bytes, 0}}, nullptr));
}

}  // namespace
}  // namespace mori::umbp

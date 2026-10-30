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
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <thread>
#include <tuple>

#include "umbp/common/device_gather.h"
#include "umbp/common/host_registration.h"
#include "umbp/common/parallel_for.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/pool_platform.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/sim/sim.h"
#include "umbp/sim/ssd.h"
#include "umbp/standalone/ipc.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"
#include "umbp_standalone.grpc.pb.h"

namespace mori::umbp {
namespace {
using MemLoc = mori::io::MemoryLocationType;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
constexpr size_t kPage = 4096;

class ScopedEnv {
 public:
  ScopedEnv(const char* key, const char* value) : key_(key) {
    if (const char* old = std::getenv(key)) old_ = old;
    setenv(key, value, 1);
  }
  ~ScopedEnv() {
    if (old_)
      setenv(key_, old_->c_str(), 1);
    else
      unsetenv(key_);
  }

 private:
  const char* key_;
  std::optional<std::string> old_;
};

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
  SsdBackend::Config SsdConfig() {
    SsdBackend::Config config;
    config.page_size = kPage;
    config.staging_pages = 8;
    config.ssd.enabled = true;
    config.ssd.ssd.storage_dir = dir_ + "/ssd";
    config.ssd.ssd.capacity_bytes = 32 * kPage;
    config.ssd.ssd.segment_size_bytes = 4 * kPage;
    config.read_lease_ttl = 40ms;
    config.reaper_interval = 5ms;
    return config;
  }
  void SsdProfile() {
    Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,"bandwidth_bytes_per_second":1e12},
      "gpu_cpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e12},
      "cpu_gpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e12},
      "cpu_file":{"channel":"disk","latency_ns":1000000,"bandwidth_bytes_per_second":1e12},
      "file_cpu":{"channel":"disk","latency_ns":3000000,"bandwidth_bytes_per_second":1e12},
      "file_gpu":{"channel":"disk","latency_ns":7000000,"bandwidth_bytes_per_second":1e12}})");
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

TEST_F(SimTest, PlatformPreservesRegionSplittingAndNumaPlacement) {
  auto platform = MakePoolPlatform(Config().sim);
  PoolClientConfig config;
  BackendInstanceConfig dram;
  dram.tier = TierType::DRAM;
  dram.dram.buffer_sizes = {4 * kPage, 4 * kPage};
  dram.dram.max_mr_bytes = 2 * kPage;
  dram.dram.numa_nodes = {1, 3};
  auto engines = platform->CreateTransferEngine(config, {dram});
  auto backend = platform->CreateBackend(dram, kPage, config);
  ASSERT_NE(backend, nullptr);
  ASSERT_TRUE(backend->Init(engines.transfer.get()));
  ASSERT_EQ(backend->BufferCount(), 4u);
  EXPECT_EQ(backend->Capacity().total_bytes, 8 * kPage);
  for (uint32_t i = 0; i < backend->BufferCount(); ++i)
    EXPECT_EQ(backend->BufferRef(i).size, 2 * kPage);

  const auto source = TransferRef::HostBytes(reinterpret_cast<void*>(Buffer(2 * kPage)), 2 * kPage);
  for (uint32_t buffer : {2u, 3u, 0u}) {
    const auto key = std::to_string(buffer);
    const int numa = buffer >= 2 ? 3 : -1;
    const auto slot = backend->BatchAllocate({AllocateRequest{key, 2 * kPage, numa}}).front();
    ASSERT_EQ(slot.outcome, AllocateOutcome::kSuccessAllocated);
    ASSERT_EQ(slot.pages.size(), 2u);
    ASSERT_EQ(slot.pages.front().buffer_index, buffer);
    // Both source and tier are PROT_NONE; a payload access would fault.
    ASSERT_TRUE(engines.transfer->Transfer(
        {{source, 0, backend->BufferRef(buffer), 0, 2 * kPage, 0}}, nullptr));
    ASSERT_TRUE(backend->BatchCommit({{slot.slot_id, key}}).front().success);
    EXPECT_TRUE(backend->Contains(key));
  }
}

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

TEST_P(SimApiTest, SsdApisKeepSharedStorageVirtualAndGpuRegistrationsWork) {
  SsdProfile();
  auto config = Config();
  config.ssd = SsdConfig().ssd.ssd;
  config.distributed.emplace();
  config.distributed->medium = UMBPMedium::SSD;
  config.distributed->master_config.node_id = "ssd-api";
  config.distributed->master_config.node_address = "127.0.0.1:0";
  config.distributed->ssd_staging_buffer_slots = 8;
  const auto ptr = Buffer(8 * kPage);
  for (bool gds : {false, true}) {
    ScopedEnv mode("UMBP_ENABLE_GDS", gds ? "1" : "0");
    auto client = Client(config, GetParam());
    ASSERT_TRUE(client->RegisterMemory(ptr, 8 * kPage));
    EXPECT_EQ(client->BatchPutRanges({"range"}, {2 * kPage}, {{ptr, ptr + kPage}}, {{kPage, kPage}},
                                     {{0, kPage}}),
              std::vector<bool>{true});
    EXPECT_EQ(client->BatchGetRanges({"range"}, {{ptr}}, {{128}}, {{kPage - 64}}),
              std::vector<bool>{true});
    EXPECT_FALSE(client->Get("range", ptr, 3 * kPage));
    EXPECT_FALSE(client->Get("missing", ptr, kPage));
    client->DeregisterMemory(ptr);
    ASSERT_TRUE(client->RegisterMemory(ptr, 8 * kPage, MemLoc::GPU, 0));
    ASSERT_TRUE(client->Put("gpu", ptr, kPage));
    const auto start = Clock::now();
    EXPECT_TRUE(client->Get("gpu", ptr, kPage));
    EXPECT_GE(Clock::now() - start, gds ? 7ms : 3ms);
    if (gds) {
      const auto ranged_start = Clock::now();
      EXPECT_EQ(client->BatchGetRanges({"gpu"}, {{ptr}}, {{128}}, {{64}}), std::vector<bool>{true});
      EXPECT_GE(Clock::now() - ranged_start, 7ms);
    }
    EXPECT_TRUE(client->Clear());
    EXPECT_EQ(client->BatchExists({"gpu", "range"}), std::vector<bool>(2, false));
    EXPECT_FALSE(std::filesystem::exists(config.ssd.storage_dir));
  }
}

TEST_F(SimTest, SsdSegmentCapacityReservationsAndClearPreserveControlContracts) {
  SsdProfile();
  SimTransferEngine engine(Config().sim.profile_path);
  auto backend = MakeSimSsdBackend(SsdConfig(), engine);
  ASSERT_TRUE(backend->Init(&engine));
  const auto available = [&] { return backend->Capacity().available_bytes; };
  const auto capacity = available();
  auto reserve = [&](const std::string& key) {
    return backend->BatchAllocate({AllocateRequest{key, kPage, -1, true}}).front();
  };
  // A 4 KiB value occupies a padded 4 KiB prefix and 4 KiB value extent.
  for (const char* key : {"a", "b", "c"}) {
    auto slot = reserve(key);
    ASSERT_EQ(slot.outcome, AllocateOutcome::kSuccessAllocated);
    ASSERT_TRUE(slot.file_ref.IsFile());
    ASSERT_TRUE(backend->BatchCommit({{slot.slot_id, key}}).front().success);
  }
  EXPECT_EQ(available(), capacity - 6 * kPage);
  EXPECT_EQ(backend->DrainPendingEvents().size(), 3u);
  EXPECT_TRUE(backend->OwnsEvictionOrder());
  backend->Evict({"a"});
  EXPECT_EQ(available(), capacity - 6 * kPage);  // b keeps a's segment live.
  backend->Evict({"b"});
  EXPECT_EQ(available(), capacity - 2 * kPage);
  auto aborted = reserve("pending");
  ASSERT_TRUE(aborted.file_ref.IsFile());
  EXPECT_TRUE(backend->BatchAbort({aborted.slot_id}).front());
  EXPECT_FALSE(backend->Contains("pending"));
  EXPECT_EQ(available(), capacity - 2 * kPage);
  auto stale = reserve("pending");
  backend->ClearLocal();
  EXPECT_TRUE(backend->IsClearFullSyncPending());
  EXPECT_NE(reserve("blocked").outcome, AllocateOutcome::kSuccessAllocated);
  backend->ClearFullSyncAcked();
  auto current = reserve("pending");
  ASSERT_TRUE(current.file_ref.IsFile());
  EXPECT_NE(stale.file_ref.file_fd, current.file_ref.file_fd);
  EXPECT_FALSE(backend->BatchCommit({{stale.slot_id, "pending"}}).front().success);
  EXPECT_TRUE(backend->BatchAbort({stale.slot_id}).front());  // Abort remains idempotent.
  EXPECT_TRUE(backend->BatchCommit({{current.slot_id, "pending"}}).front().success);
  EXPECT_TRUE(backend->Contains("pending"));
  EXPECT_FALSE(std::filesystem::exists(SsdConfig().ssd.ssd.storage_dir));
}

TEST_F(SimTest, SsdMissingModelRollsBackAndUnsupportedBackendsFailFast) {
  SimTransferEngine engine(Config().sim.profile_path);  // cpu_cpu only.
  auto config = SsdConfig();
  auto backend = MakeSimSsdBackend(config, engine);
  ASSERT_TRUE(backend->Init(&engine));
  const auto available = backend->Capacity().available_bytes;
  // The failed prefix reservation falls back to staging; the failed commit
  // must return both disk accounting and the staging slot.
  auto slot = backend->BatchAllocate({AllocateRequest{"failed", kPage, -1, true}}).front();
  ASSERT_EQ(slot.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_FALSE(slot.file_ref.IsFile());
  EXPECT_FALSE(backend->BatchCommit({{slot.slot_id, "failed"}}).front().success);
  EXPECT_FALSE(backend->Contains("failed"));
  EXPECT_EQ(backend->Capacity().available_bytes, available);
  EXPECT_TRUE(backend->DrainPendingEvents().empty());
  for (const char* kind : {"spdk", "spdk_proxy", "dummy_storage"}) {
    config.ssd.ssd.ssd_backend = kind;
    EXPECT_THROW(MakeSimSsdBackend(config, engine), std::invalid_argument);
  }
  config.ssd.ssd.ssd_backend = "file";
  config.ssd.enabled = false;
  EXPECT_THROW(MakeSimSsdBackend(config, engine), std::invalid_argument);
}

TEST_F(SimTest, FilePathsPreserveGroupingBoundsAndRejectGpuWrites) {
  SsdProfile();
  SimTransferEngine engine(Config().sim.profile_path);
  auto* ptr = reinterpret_cast<void*>(Buffer(2 * kPage));
  const auto small = TransferRef::HostBytes(ptr, kPage);
  const auto large = TransferRef::HostBytes(ptr, 2 * kPage);
  const auto file = engine.RegisterFile(123, 0, 2 * kPage);
  const auto gpu = TransferRef::HostBytes(ptr, 2 * kPage, MemLoc::GPU, 0);
  EXPECT_FALSE(engine.CanHandle(gpu, file));
  auto items =
      std::vector<TransferItem>{{small, 0, engine.RegisterFile(123, 0, kPage), 0, kPage, 0},
                                {large, kPage, file, kPage, kPage, 1}};
  auto plans = engine.Plan(items);
  ASSERT_EQ(plans.plans.size(), 1u);
  EXPECT_EQ(plans.plans.front().src.size, 2 * kPage);
  EXPECT_EQ(plans.plans.front().dst.size, 2 * kPage);
  EXPECT_TRUE(engine.Transfer(items, nullptr));
  items.push_back({large, 0, engine.RegisterFile(124, 0, 2 * kPage), 0, kPage, 2});
  EXPECT_EQ(engine.Plan(items).plans.size(), 2u);  // Distinct files cannot merge.
  EXPECT_FALSE(engine.Transfer({{file, 0, small, kPage, 1, 0}}, nullptr));
  EXPECT_FALSE(
      engine.Transfer({{engine.RegisterFile(125, INT64_MAX, kPage), 0, small, 0, 1, 0}}, nullptr));
  const auto start = Clock::now();
  EXPECT_TRUE(engine.Transfer({{file, 0, gpu, 0, kPage, 0}}, nullptr));
  EXPECT_GE(Clock::now() - start, 7ms);
  EXPECT_FALSE(engine.Transfer({{TransferRef::File(123, 0, kPage), 0, gpu, 0, kPage, 0}},
                               nullptr));  // GDS reads require a registered handle.
}

TEST_F(SimTest, SsdPayloadCurveIsOptionalAndSeparateFromDirectIo) {
  SsdProfile();
  SimTransferEngine legacy(Config().sim.profile_path);
  EXPECT_TRUE(legacy.SimulateStorage(true, kPage));
  Profile(R"({"file_cpu_staged":{"channel":"disk","latency_ns":0,
      "bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine staged(Config().sim.profile_path);
  EXPECT_TRUE(staged.SimulateStorage(true, kPage));
  EXPECT_TRUE(staged.SimulateStorage(true, kPage, 1, false));  // Legacy staged-only estimate.
  EXPECT_FALSE(
      staged.CanHandle(TransferRef::File(1, 0, kPage),
                       TransferRef::HostBytes(reinterpret_cast<void*>(Buffer(kPage)), kPage)));
  EXPECT_FALSE(staged.SimulateStorage(false, kPage));
  Profile(R"({"file_cpu_staged":{"channel":"disk","latency_ns":50000000,
      "bandwidth_bytes_per_second":1e12},
      "file_cpu_staged_no_crc":{"channel":"disk","latency_ns":0,
      "log_size_latency_ns":20000000,"batch_log_size_latency_ns":5000000,
      "bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine variants(Config().sim.profile_path);
  auto start = Clock::now();
  EXPECT_TRUE(variants.SimulateStorage(true, 2 * kPage, 1, false));
  EXPECT_GE(Clock::now() - start, 25ms);
  EXPECT_LT(Clock::now() - start, 40ms);
  start = Clock::now();
  EXPECT_TRUE(variants.SimulateStorage(true, 4 * kPage, 2, false));
  EXPECT_GE(Clock::now() - start, 45ms);  // Batch latency is charged once, not per I/O.
  EXPECT_LT(Clock::now() - start, 60ms);
  start = Clock::now();
  EXPECT_TRUE(variants.SimulateStorage(true, 2 * kPage));
  EXPECT_GE(Clock::now() - start, 50ms);
}

TEST_F(SimTest, DisabledSsdChecksumsUseTheNoCrcCurve) {
  Profile(R"({"cpu_file_staged":{"channel":"disk","latency_ns":0,
      "bandwidth_bytes_per_second":1e12},
      "file_cpu_staged_no_crc":{"channel":"disk","latency_ns":0,
      "bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine engine(Config().sim.profile_path);
  auto config = SsdConfig();
  config.ssd.ssd.verify_crc = false;
  auto backend = MakeSimSsdBackend(config, engine);
  ASSERT_TRUE(backend->Init(&engine));
  const auto slot = backend->BatchAllocate({AllocateRequest{"no-crc", kPage, -1, true}}).front();
  ASSERT_TRUE(slot.file_ref.IsFile());
  ASSERT_TRUE(backend->BatchCommit({{slot.slot_id, "no-crc"}}).front().success);
  EXPECT_TRUE(backend->BatchResolve({"no-crc"}, false, false, ResolveReader::kLocal).front().found);
}

TEST_F(SimTest, GdsSingleCallerKeepsBatchCopyAndLogCosts) {
  Profile(R"({"file_gpu":{"channel":"disk","latency_ns":20000000,
    "copy_latency_ns":15000000,"log_size_latency_ns":8000000,
    "batch_log_size_latency_ns":10000000,"bandwidth_bytes_per_second":512000}})");
  SimTransferEngine engine(Config().sim.profile_path);
  const auto file = engine.RegisterFile(1, 0, 4 * kPage);
  const auto gpu =
      TransferRef::HostBytes(reinterpret_cast<void*>(Buffer(4 * kPage)), 4 * kPage, MemLoc::GPU, 0);
  const auto started = Clock::now();
  ASSERT_TRUE(engine.Transfer(
      {{file, 0, gpu, 0, 2 * kPage, 0}, {file, 2 * kPage, gpu, 2 * kPage, 2 * kPage, 1}}, nullptr));
  // 30ms batch + 2 * 23ms private copy + 32ms byte service, all inline.
  EXPECT_GE(Clock::now() - started, 105ms);
  EXPECT_LT(Clock::now() - started, 150ms);
}

TEST_F(SimTest, GdsNewReaderDelaysActiveCompletionAndRejectsOverflow) {
  Profile(R"({"file_gpu":{"channel":"disk","latency_ns":0,
    "bandwidth_bytes_per_second":20480}})");
  SimTransferEngine engine(Config().sim.profile_path);
  const auto file = engine.RegisterFile(1, 0, kPage);
  const auto gpu =
      TransferRef::HostBytes(reinterpret_cast<void*>(Buffer(kPage)), kPage, MemLoc::GPU, 0);
  std::promise<Clock::time_point> entered;
  auto first = std::async(std::launch::async, [&] {
    entered.set_value(Clock::now());
    return engine.Transfer({{file, 0, gpu, 0, kPage, 0}}, nullptr);
  });
  const auto started = entered.get_future().get();
  std::this_thread::sleep_for(50ms);
  const auto huge = static_cast<size_t>(std::numeric_limits<int64_t>::max());
  EXPECT_FALSE(
      engine.Transfer({{engine.RegisterFile(2, 0, huge), 0,
                        TransferRef::HostBytes(gpu.host_ptr, huge, MemLoc::GPU, 0), 0, huge, 1}},
                      nullptr));
  auto second = std::async(std::launch::async,
                           [&] { return engine.Transfer({{file, 0, gpu, 0, kPage, 2}}, nullptr); });
  EXPECT_EQ(first.wait_for(230ms), std::future_status::timeout);
  EXPECT_TRUE(first.get());
  EXPECT_GE(Clock::now() - started, 310ms);  // First now finishes near 350ms, not 200ms.
  EXPECT_TRUE(second.get());
  EXPECT_LT(Clock::now() - started, 550ms);
}

TEST_F(SimTest, GdsAndOrdinarySameChannelReservationsDoNotOverlap) {
  Profile(R"({"file_gpu":{"channel":"disk","latency_ns":0,"bandwidth_bytes_per_second":20480},
    "cpu_file":{"channel":"disk","latency_ns":0,"bandwidth_bytes_per_second":20480}})");
  SimTransferEngine engine(Config().sim.profile_path);
  const auto file = engine.RegisterFile(1, 0, kPage);
  const auto gpu =
      TransferRef::HostBytes(reinterpret_cast<void*>(Buffer(kPage)), kPage, MemLoc::GPU, 0);
  std::promise<Clock::time_point> entered;
  auto first = std::async(std::launch::async, [&] {
    entered.set_value(Clock::now());
    return engine.Transfer({{file, 0, gpu, 0, kPage, 0}}, nullptr);
  });
  const auto started = entered.get_future().get();
  std::this_thread::sleep_for(50ms);
  auto write = std::async(std::launch::async,
                          [&] { return engine.SimulateStorage(false, kPage, 1, false); });
  std::this_thread::sleep_for(20ms);
  auto later = std::async(std::launch::async,
                          [&] { return engine.Transfer({{file, 0, gpu, 0, kPage, 1}}, nullptr); });
  EXPECT_EQ(write.wait_for(250ms), std::future_status::timeout);
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(write.get());
  EXPECT_GE(Clock::now() - started, 350ms);
  EXPECT_TRUE(later.get());
  EXPECT_GE(Clock::now() - started, 550ms);  // 200ms GDS, 200ms write, then 200ms GDS.
  EXPECT_LT(Clock::now() - started, 900ms);
}

TEST_F(SimTest, NativeSsdQueuesIoAndKeepsChecksumWorkPrivate) {
  Profile(R"({"file_cpu_staged":{"channel":"disk","latency_ns":200000000,
      "bandwidth_bytes_per_second":1e12,"serialize_latency":true},
      "file_cpu_staged_no_crc":{"channel":"disk","latency_ns":80000000,
      "bandwidth_bytes_per_second":1e12,"serialize_latency":true}})");
  SimTransferEngine engine(Config().sim.profile_path);
  std::promise<void> start;
  const auto gate = start.get_future().share();
  std::atomic<int> ready{0};
  auto read = [&] {
    ++ready;
    gate.wait();
    return engine.SimulateStorage(true, kPage);
  };
  auto first = std::async(std::launch::async, read);
  auto second = std::async(std::launch::async, read);
  const bool both_ready = Await([&] { return ready == 2; });
  const auto submitted = Clock::now();
  start.set_value();
  ASSERT_TRUE(both_ready);
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
  // Two 80 ms I/Os queue; the last caller then pays its private 120 ms checksum.
  EXPECT_GE(Clock::now() - submitted, 240ms);
  EXPECT_LT(Clock::now() - submitted, 360ms);
}

TEST_F(SimTest, SsdReadLatencyOverlapsButBandwidthAndWritesQueue) {
  for (const auto [read, bandwidth, serial] :
       {std::tuple{true, false, false}, std::tuple{true, true, false},
        std::tuple{false, false, false}, std::tuple{true, false, true}}) {
    const auto curve = std::string(R"({"channel":"disk","latency_ns":)") +
                       (bandwidth ? "0" : "100000000") + R"(,"bandwidth_bytes_per_second":)" +
                       (bandwidth ? "40960" : "1e12") + "}";
    const auto read_curve =
        serial ? curve.substr(0, curve.size() - 1) + ",\"serialize_latency\":true}" : curve;
    Profile("{\"file_cpu\":" + read_curve + ",\"cpu_file\":" + curve + "}");
    SimTransferEngine engine(Config().sim.profile_path);
    std::promise<void> start;
    const auto gate = start.get_future().share();
    std::vector<std::future<bool>> pending;
    for (int i = 0; i < 4; ++i)
      pending.push_back(std::async(std::launch::async, [&] {
        gate.wait();
        return engine.SimulateStorage(read, kPage, 1, false);
      }));
    const auto submitted = Clock::now();
    start.set_value();
    for (auto& call : pending) EXPECT_TRUE(call.get());
    if (read && !bandwidth && !serial) {
      EXPECT_GE(Clock::now() - submitted, 100ms);
      EXPECT_LT(Clock::now() - submitted, 300ms);
    } else {
      EXPECT_GE(Clock::now() - submitted, 400ms);
    }
  }
  SimTransferEngine mixed(Config().sim.profile_path);
  const auto host = TransferRef::HostBytes(reinterpret_cast<void*>(Buffer(kPage)), kPage);
  const auto file = TransferRef::File(1, 0, kPage);
  const auto start = Clock::now();
  EXPECT_TRUE(
      mixed.Transfer({{file, 0, host, 0, kPage, 0}, {host, 0, file, 0, kPage, 1}}, nullptr));
  EXPECT_GE(Clock::now() - start, 200ms);
}

TEST_F(SimTest, SsdWriteCpuPreparationPrecedesAndOverlapsTheIoQueue) {
  Profile(R"({"cpu_file_staged":{"channel":"disk","latency_ns":100000000,
      "bandwidth_bytes_per_second":1e12,"cpu_bandwidth_bytes_per_second":40960,
      "cpu_threads":4,"cpu_launch_latency_ns":10000000}})");
  SimTransferEngine engine(Config().sim.profile_path);
  std::promise<void> start;
  const auto gate = start.get_future().share();
  std::vector<std::future<bool>> pending;
  for (int i = 0; i < 4; ++i)
    pending.push_back(std::async(std::launch::async, [&] {
      gate.wait();
      return engine.SimulateStorage(false, kPage);
    }));
  const auto submitted = Clock::now();
  start.set_value();
  for (auto& call : pending) EXPECT_TRUE(call.get());
  EXPECT_GE(Clock::now() - submitted, 500ms);  // CPU 100ms overlaps, then four I/O 100ms.
  EXPECT_LT(Clock::now() - submitted, 700ms);  // Serializing CPU too would take 800ms.
  for (int threads : {4, 2}) {
    const int workers = EffectiveParallelThreads(threads);
    const auto expected = 100ms + 100ms * (1 + 3 / workers) + 10ms * (workers - 1);
    const auto begun = Clock::now();
    EXPECT_TRUE(engine.SimulateStorage(false, 4 * kPage, 4, true, threads));
    EXPECT_GE(Clock::now() - begun, expected);
    EXPECT_LT(Clock::now() - begun, expected + 100ms);
  }
  const int workers = EffectiveParallelThreads(2);
  const auto expected = 100ms + 100ms * (1 + 2 / workers) + 10ms * (workers - 1);
  const auto begun = Clock::now();
  EXPECT_TRUE(engine.SimulateStorage(false, 3 * kPage, 3, true, 2));
  EXPECT_GE(Clock::now() - begun, expected);  // Whole records cannot divide evenly across workers.
  EXPECT_LT(Clock::now() - begun, expected + 100ms);
}

TEST_F(SimTest, SsdCleanupCompletesAfterIoWithoutHoldingItsQueue) {
  Profile(R"({"cpu_file_staged":{"channel":"disk","latency_ns":100000000,
      "bandwidth_bytes_per_second":1e12,"cpu_bandwidth_bytes_per_second":1e12,
      "cpu_cleanup_bandwidth_bytes_per_second":40960}})");
  SimTransferEngine engine(Config().sim.profile_path);
  std::promise<void> start;
  const auto gate = start.get_future().share();
  auto write = [&] {
    gate.wait();
    return engine.SimulateStorage(false, kPage);
  };
  auto first = std::async(std::launch::async, write);
  auto second = std::async(std::launch::async, write);
  const auto begun = Clock::now();
  start.set_value();
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
  EXPECT_GE(Clock::now() - begun, 260ms);
  EXPECT_LT(Clock::now() - begun, 360ms);  // Two I/O 100ms plus the last private cleanup 100ms.
  for (const auto* invalid :
       {"\"cpu_launch_latency_ns\":-1", "\"cpu_cleanup_bandwidth_bytes_per_second\":0"}) {
    Profile(std::string(R"({"cpu_file_staged":{"channel":"disk","latency_ns":0,
        "bandwidth_bytes_per_second":1e12,"cpu_bandwidth_bytes_per_second":1e12,)") +
            invalid + "}}");
    EXPECT_THROW(SimTransferEngine(Config().sim.profile_path), std::invalid_argument);
  }
}

TEST_F(SimTest, GdsRegistrationCostEndsWithTheLastReference) {
  Profile(R"({"file_gpu":{"channel":"disk","latency_ns":0,
      "bandwidth_bytes_per_second":1e12,"registration_latency_ns":100000000}})");
  SimTransferEngine engine(Config().sim.profile_path);
  const auto first_start = Clock::now();
  auto first = engine.RegisterFile(1, 0, kPage);
  EXPECT_GE(Clock::now() - first_start, 100ms);
  const auto cached_start = Clock::now();
  auto second = engine.RegisterFile(1, kPage, kPage);
  engine.Deregister(first);
  auto third = engine.RegisterFile(1, 0, kPage);
  EXPECT_EQ(first.gds_handle, third.gds_handle);
  EXPECT_LT(Clock::now() - cached_start, 100ms);
  engine.Deregister(second);
  engine.Deregister(third);
  const auto next_start = Clock::now();
  EXPECT_TRUE(engine.RegisterFile(1, 0, kPage).gds_handle);
  EXPECT_GE(Clock::now() - next_start, 100ms);
}

TEST_F(SimTest, SsdSingleFlightCopyFailureWakesWholeAndRangeFollowers) {
  // No cpu_cpu model: the leader reads successfully, but simulated follower
  // fan-out fails without accessing its PROT_NONE staging pages.
  Profile(R"({"cpu_file":{"channel":"disk","latency_ns":0,"bandwidth_bytes_per_second":1e12},
      "file_cpu":{"channel":"disk","latency_ns":200000000,"bandwidth_bytes_per_second":1e12}})");
  for (bool ranged : {false, true}) {
    SimTransferEngine engine(Config().sim.profile_path);
    auto backend = MakeSimSsdBackend(SsdConfig(), engine);
    ASSERT_TRUE(backend->Init(&engine));
    auto slot = backend->BatchAllocate({AllocateRequest{"shared", 4 * kPage, -1, true}}).front();
    ASSERT_TRUE(slot.file_ref.IsFile());
    ASSERT_TRUE(backend->BatchCommit({{slot.slot_id, "shared"}}).front().success);
    std::vector<ObjectExtent> extents{{kPage, kPage}};
    auto read = [&] {
      return backend
          ->BatchResolve({"shared"}, false, false, ResolveReader::kLocal,
                         ranged ? &extents : nullptr)
          .front();
    };
    auto metric = [&](const char* name) {
      for (const auto& sample : backend->SampleMetrics())
        if (sample.labels.front().second == name) return sample.value;
      return uint64_t{0};
    };
    auto leader = std::async(std::launch::async, read);
    ASSERT_TRUE(Await([&] { return metric("single_flight_lead") != 0; }));
    auto follower = std::async(std::launch::async, read);
    ASSERT_TRUE(Await([&] { return metric("single_flight_merged") != 0; }));
    EXPECT_EQ(backend->Evict({"shared"}).front().bytes_freed, 0u);
    ASSERT_EQ(follower.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(follower.get().outcome, ResolveOutcome::kFailed);
    auto loaded = leader.get();
    ASSERT_TRUE(loaded.found);
    // Whole reads lease the key; extent holds retain only their private staging
    // span, so deleting the original SSD record is safe while the extent lives.
    if (ranged)
      EXPECT_NE(backend->Evict({"shared"}).front().bytes_freed, 0u);
    else
      EXPECT_EQ(backend->Evict({"shared"}).front().bytes_freed, 0u);
    loaded.hold.reset();
    if (!ranged)
      ASSERT_TRUE(Await([&] { return backend->Evict({"shared"}).front().bytes_freed != 0; }));
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

TEST_F(SimTest, StandaloneRegistrationRejectsWrongKindsAndVirtualFdHandoffs) {
  for (bool sim : {false, true}) {
    auto config = Config();
    config.sim.enabled = sim;
    const auto address = RankConfig().standalone_process->address;
    standalone::StandaloneServer server(config, address);
    ASSERT_TRUE(server.Start());
    auto stub = ::umbp::UMBPStandalone::NewStub(
        grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
    HostMemAllocator allocator;
    HostBufferOptions options;
    options.backing = HostBufferBacking::kAnonymousShm;
    options.prefault = false;
    auto memory = allocator.Alloc(kPage, options);
    ASSERT_TRUE(memory.valid());
    const auto ptr = reinterpret_cast<uintptr_t>(memory.ptr);
    const auto shm = HostMemAllocator::LookupShmAllocation(ptr, kPage);
    ASSERT_TRUE(shm);
    std::string error;
    const auto status =
        standalone::SendFdRegistration(standalone::DeriveFdSocketPath(address), shm->fd, "rank",
                                       ptr, memory.mapped_size, 1000, &error);
    EXPECT_EQ(status, sim ? -1 : 0) << error;

    ::umbp::RegisterMemoryRequest request;
    request.set_client_id("rank");
    request.set_worker_base(ptr);
    request.set_size(kPage);
    request.set_device_id(-1);
    auto register_kind = [&](::umbp::MemoryKind kind) {
      request.set_kind(kind);
      grpc::ClientContext context;
      ::umbp::BoolResponse response;
      EXPECT_TRUE(stub->RegisterMemory(&context, request, &response).ok());
      return response.ok();
    };
    EXPECT_FALSE(register_kind(sim ? ::umbp::MEMORY_KIND_HOST_SHM : ::umbp::MEMORY_KIND_VIRTUAL));
    if (sim) EXPECT_FALSE(register_kind(::umbp::MEMORY_KIND_GPU_IPC));
    EXPECT_TRUE(register_kind(sim ? ::umbp::MEMORY_KIND_VIRTUAL : ::umbp::MEMORY_KIND_HOST_SHM));
    if (sim) {
      request.set_ipc_handle("invalid");
      EXPECT_FALSE(register_kind(::umbp::MEMORY_KIND_VIRTUAL));
    }
    allocator.Free(memory);
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

TEST_F(SimTest, GatherModelRequiresRegisteredHostAndEligibleDeviceBatch) {
  Profile(R"({"cpu_gpu":{"channel":"link","latency_ns":30000000,"bandwidth_bytes_per_second":1e12},
    "cpu_gpu_gather":{"channel":"link","latency_ns":5000000,"bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine engine(Config().sim.profile_path);
  constexpr size_t bytes = 256 << 10;
  const auto host = Buffer(2 * bytes), gpu = Buffer(2 * bytes);
  auto item = [&](size_t offset, size_t size, int device = 0) {
    return TransferItem{
        TransferRef::HostBytes(reinterpret_cast<void*>(host + offset), size),
        0,
        TransferRef::HostBytes(reinterpret_cast<void*>(gpu + offset), size, MemLoc::GPU, device),
        0,
        size,
        offset};
  };
  auto run = [&](std::vector<TransferItem> items, bool gather) {
    const auto start = Clock::now();
    ASSERT_TRUE(engine.Transfer(items, nullptr));
    if (gather)
      EXPECT_LT(Clock::now() - start, 20ms);
    else
      EXPECT_GE(Clock::now() - start, 30ms);
  };
  run({item(0, kPage), item(kPage, kPage)}, false);
  engine.AddHostGatherRegion(reinterpret_cast<void*>(host), 2 * bytes);
  run({item(0, kPage), item(kPage, kPage)}, DeviceGatherEnabled() && HostRegistrationEnabled());
  run({item(0, kPage)}, false);                         // One fragment takes memcpy.
  run({item(0, kPage), item(kPage, kPage, 1)}, false);  // One fragment per device.
  run({item(0, 128 << 10), item(128 << 10, 128 << 10)}, false);
  engine.ClearHostGatherRegions();
  run({item(0, kPage), item(kPage, kPage)}, false);
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

TEST_F(SimTest, DefaultGpuPointerClassificationKeepsRealAndSimCopies) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) GTEST_SKIP() << "needs a GPU";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  void* raw = nullptr;
  ASSERT_EQ(hipMalloc(&raw, kPage), hipSuccess);
  std::unique_ptr<char, decltype(&hipFree)> buffer(static_cast<char*>(raw), hipFree);
  std::vector<char> expected(kPage, 'x'), actual(kPage);
  ASSERT_EQ(hipMemcpy(raw, expected.data(), kPage, hipMemcpyHostToDevice), hipSuccess);
  auto config = Config();
  config.sim.enabled = false;
  auto client = CreateUMBPClient(config);
  const auto ptr = reinterpret_cast<uintptr_t>(raw);
  ASSERT_TRUE(client->RegisterMemory(ptr, kPage));  // Existing two-argument API.
  ASSERT_TRUE(client->Put("gpu", ptr, kPage));
  ASSERT_EQ(hipMemset(raw, 0, kPage), hipSuccess);
  ASSERT_TRUE(client->Get("gpu", ptr, kPage));
  ASSERT_EQ(hipMemcpy(actual.data(), raw, kPage, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(actual, expected);
  Profile(R"({"cpu_gpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e12},
      "gpu_cpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e12}})");
  for (bool standalone : {false, true}) {
    auto simulated = Client(Config(), standalone);
    if (standalone) ASSERT_TRUE(simulated->RegisterMemory(ptr, kPage));
    // Embedded callers may omit registration; standalone keeps its existing requirement.
    EXPECT_TRUE(simulated->Put("gpu", ptr, kPage));
    EXPECT_TRUE(simulated->Get("gpu", ptr, kPage));
  }
  SsdProfile();
  auto ssd = Config();
  ssd.ssd = SsdConfig().ssd.ssd;
  ssd.distributed.emplace();
  ssd.distributed->medium = UMBPMedium::SSD;
  ssd.distributed->master_config.node_id = "ssd-gpu";
  ssd.distributed->master_config.node_address = "127.0.0.1:0";
  ssd.distributed->ssd_staging_buffer_slots = 8;
  for (bool standalone : {false, true}) {
    for (bool gds : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "standalone=" << standalone << " gds=" << gds);
      ScopedEnv mode("UMBP_ENABLE_GDS", gds ? "1" : "0");
      auto simulated = Client(ssd, standalone);
      // Distributed registration's location is explicit; the standalone client
      // also detects physical GPU pointers when the location is omitted.
      ASSERT_TRUE(standalone ? simulated->RegisterMemory(ptr, kPage)
                             : simulated->RegisterMemory(ptr, kPage, MemLoc::GPU, 0));
      ASSERT_TRUE(simulated->Put("gpu", ptr, kPage));
      const auto start = Clock::now();
      EXPECT_TRUE(simulated->Get("gpu", ptr, kPage));
      EXPECT_GE(Clock::now() - start, gds ? 7ms : 3ms);
    }
  }
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
      "cpu_gpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e10},
      "gpu_cpu":{"channel":"link","latency_ns":0,"bandwidth_bytes_per_second":1e10},
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

class SimRecacheTest : public SimClusterTest, public ::testing::WithParamInterface<MemLoc> {};
INSTANTIATE_TEST_SUITE_P(CpuAndGpuCallers, SimRecacheTest,
                         ::testing::Values(MemLoc::CPU, MemLoc::GPU));

TEST_P(SimRecacheTest, RemoteRecacheAndEvictionKeepRealMetadata) {
  auto owner = Pool("owner", false);
  auto* payload = reinterpret_cast<void*>(ptr_);
  ASSERT_TRUE(owner->Put("cached", payload, kPage));
  ASSERT_TRUE(owner->Put("evicted", payload, kPage));
  owner->Master().FlushHeartbeat();
  auto reader = Pool("reader", true);
  ASSERT_TRUE(
      reader->RegisterMemory(payload, 64 * kPage, GetParam(), GetParam() == MemLoc::GPU ? 0 : -1));
  ASSERT_TRUE(Await([&] { return reader->Exists("cached"); }));
  ASSERT_TRUE(reader->Get("cached", reinterpret_cast<void*>(ptr_ + kPage), kPage));
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

TEST_F(SimTest, CalibratedStagingUsesItsOwnCopyCost) {
  Profile(R"({"cpu_cpu":{"channel":"host","latency_ns":0,"bandwidth_bytes_per_second":1e12},
    "cpu_staging":{"channel":"host","latency_ns":20000000,"bandwidth_bytes_per_second":1e12},
    "cpu_staging_read":{"channel":"host","latency_ns":40000000,"bandwidth_bytes_per_second":1e12},
    "remote_read":{"channel":"nic","latency_ns":0,"bandwidth_bytes_per_second":1e12},
    "remote_write":{"channel":"nic","latency_ns":0,"bandwidth_bytes_per_second":1e12}})");
  SimTransferEngine caller(Config().sim.profile_path, "caller", kPage);
  SimTransferEngine owner(Config().sim.profile_path, "owner");
  ASSERT_TRUE(caller.EnsureRemoteEngine("owner", Packed(owner)));
  auto* buffer = reinterpret_cast<void*>(Buffer(kPage));
  auto local = TransferRef::HostBytes(buffer, kPage);
  auto remote = TransferRef::Remote(owner.RegisterMemory(buffer, kPage, MemLoc::CPU, -1).mem);
  const auto start = Clock::now();
  ASSERT_TRUE(caller.Transfer({{local, 0, remote, 0, kPage, 0}}, nullptr));
  EXPECT_GE(Clock::now() - start, 20ms);
  const auto read_start = Clock::now();
  ASSERT_TRUE(caller.Transfer({{remote, 0, local, 0, kPage, 0}}, nullptr));
  EXPECT_GE(Clock::now() - read_start, 40ms);
}

}  // namespace
}  // namespace mori::umbp

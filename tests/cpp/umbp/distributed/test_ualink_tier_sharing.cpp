// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// Real local FABRIC integration for DRAM/SSD owners. Logical peers share one
// fabric-ready GPU here; the hardware driver covers deployment across trays.
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/pool_platform.h"

namespace mori::umbp {
namespace {
constexpr size_t kPage = 4096;
constexpr size_t kObject = 6 * kPage + 2048;
constexpr size_t kObjectPages = (kObject + kPage - 1) / kPage;

class ObservedTierPlatform final : public PoolPlatform {
 public:
  ObservedTierPlatform() : inner_(MakePoolPlatform({})) {}
  Engines CreateTransferEngine(const PoolClientConfig& config,
                               const std::vector<BackendInstanceConfig>& backends) override {
    auto engines = inner_->CreateTransferEngine(config, backends);
    engine_ = engines.transfer.get();
    return engines;
  }
  std::unique_ptr<MediumBackend> CreateBackend(const BackendInstanceConfig& backend, uint64_t page,
                                               const PoolClientConfig& config) override {
    return inner_->CreateBackend(backend, page, config);
  }
  TransferRef LocalUserBytes(void* ptr, size_t bytes, const TransferRef& ref) const override {
    return inner_->LocalUserBytes(ptr, bytes, ref);
  }
  HostBufferHandle AllocateHost(size_t bytes, const HostBufferOptions& options = {}) override {
    return inner_->AllocateHost(bytes, options);
  }
  void FreeHost(HostBufferHandle& buffer) override { inner_->FreeHost(buffer); }
  void AddHostGatherRegion(void* ptr, size_t bytes) override {
    inner_->AddHostGatherRegion(ptr, bytes);
  }
  void ClearHostGatherRegions() override { inner_->ClearHostGatherRegions(); }
  struct Counters {
    uint64_t fabric = 0;
    uint64_t other = 0;
    uint64_t plans = 0;
  };
  Counters ReadCounters() const {
    Counters counters;
    if (!engine_) return counters;
    for (const auto& metric : engine_->SampleMetrics()) {
      if (!metric.name) continue;
      std::string backend, direction;
      for (const auto& [label, value] : metric.labels) {
        if (label == "backend") backend = value;
        if (label == "direction") direction = value;
      }
      if (direction != "pull") continue;
      if (std::string(metric.name) == "mori_umbp_transport_submitted_bytes_total") {
        if (backend == "fabric")
          counters.fabric += metric.value;
        else
          counters.other += metric.value;
      }
      if (std::string(metric.name) == "mori_umbp_transport_submitted_plans_total" &&
          backend == "fabric")
        counters.plans += metric.value;
    }
    return counters;
  }

 private:
  std::shared_ptr<PoolPlatform> inner_;
  TransferEngine* engine_ = nullptr;
};

class ScopedBusyTimeout {
 public:
  ScopedBusyTimeout() {
    if (const char* value = std::getenv("UMBP_RESOLVE_BUSY_TIMEOUT_MS")) old_ = value;
    setenv("UMBP_RESOLVE_BUSY_TIMEOUT_MS", "120", 1);
  }
  ~ScopedBusyTimeout() {
    if (old_)
      setenv("UMBP_RESOLVE_BUSY_TIMEOUT_MS", old_->c_str(), 1);
    else
      unsetenv("UMBP_RESOLVE_BUSY_TIMEOUT_MS");
  }

 private:
  std::optional<std::string> old_;
};

std::filesystem::path CopyModule() {
  if (const char* path = std::getenv("UMBP_FABRIC_KERNEL_PATH")) return path;
  std::error_code error;
  auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
  if (!error) {
    auto directory = executable.parent_path();
    while (!directory.empty()) {
      auto candidate = directory / "src/io/fabric_copy.hsaco";
      if (std::filesystem::exists(candidate)) return candidate;
      const auto parent = directory.parent_path();
      if (parent == directory) break;
      directory = parent;
    }
  }
  return "build-ualink/src/io/fabric_copy.hsaco";
}

class UalinkTierSharing : public ::testing::TestWithParam<TierType> {
 protected:
  void SetUp() override {
    if (const char* sim = std::getenv("MORI_IO_SIM_ENABLE"))
      if (std::string(sim) != "0") GTEST_SKIP() << "Requires the real FABRIC backend";
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) GTEST_SKIP() << "No HIP GPU";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    kernel_ = CopyModule().string();
    if (!std::filesystem::exists(kernel_)) GTEST_SKIP() << "No fabric_copy.hsaco: " << kernel_;
    mori::io::IOEngineConfig io;
    io.host = "127.0.0.1";
    try {
      mori::io::IOEngine probe("tier-sharing-capability", io);
      probe.CreateBackend(mori::io::BackendType::FABRIC, mori::io::FabricBackendConfig{});
      probe.LoadFabricCopyModule(kernel_);
      if (!probe.IsFabricReady(0)) GTEST_SKIP() << "GPU 0 is not in a copy-ready FABRIC domain";
    } catch (const std::exception& error) {
      GTEST_SKIP() << "FABRIC capability unavailable: " << error.what();
    }
    char directory[] = "/tmp/mori-ualink-tier-XXXXXX";
    const char* created = mkdtemp(directory);
    ASSERT_NE(created, nullptr);
    ssd_directory_ = created;
    MasterServerConfig master_config;
    master_config.listen_address = "127.0.0.1:0";
    master_config.registry_config.heartbeat_ttl = std::chrono::seconds(1);
    master_ = std::make_unique<MasterServer>(std::move(master_config));
    master_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 200 && master_->GetBoundPort() == 0; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_NE(master_->GetBoundPort(), 0);
    ASSERT_EQ(hipMalloc(&source_, 2 * kObject), hipSuccess);
    ASSERT_EQ(hipMalloc(&destination_, 2 * kObject), hipSuccess);
    expected_.resize(2 * kObject);
    for (size_t key = 0; key < 2; ++key)
      for (size_t offset = 0; offset < kObject; ++offset)
        expected_[key * kObject + offset] = (key * 43 + offset * 13 + (offset >> 8) * 7) % 251;
    ASSERT_EQ(hipMemcpy(source_, expected_.data(), expected_.size(), hipMemcpyHostToDevice),
              hipSuccess);
  }
  void TearDown() override {
    const bool quarantined = (reader_ && reader_->HasQuarantinedTransfers()) ||
                             (owner_ && owner_->HasQuarantinedTransfers());
    if (reader_) reader_->Shutdown();
    if (owner_) owner_->Shutdown();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
    if (!quarantined) {
      if (destination_) (void)hipFree(destination_);
      if (source_) (void)hipFree(source_);
    }
    if (!ssd_directory_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(ssd_directory_, ignored);
    }
  }
  PoolClientConfig Config(bool owner, size_t staging_bytes, bool mixed = false) {
    PoolClientConfig config;
    config.master_config.master_address = "127.0.0.1:" + std::to_string(master_->GetBoundPort());
    config.master_config.node_id = owner ? "tier-owner" : "tier-reader";
    config.master_config.node_address = "127.0.0.1";
    config.io_engine.host = "127.0.0.1";
    config.io_engine.mode = UMBPTransferMode::kFabricStrict;
    config.io_engine.fabric_kernel_path = kernel_;
    config.auto_peer_service_port = true;
    config.dram_page_size = kPage;
    config.medium = owner ? GetParam() : TierType::HBM;
    config.hbm.device = 0;
    config.hbm.allocation = UMBPHbmAllocation::kFabricVmm;
    config.hbm.buffer_sizes = {2 * kPage};
    config.dram.buffer_sizes = {64 * kPage};
    config.kv_put_placement = UMBPKvPutPlacement::kOwnerLocal;
    config.kv_ranged_path = UMBPKvRangedPath::kGpuDirect;
    config.kv_max_inflight_reads = 4;
    config.kv_max_inflight_bytes = 8 * kObject;
    config.cache_remote_fetches = false;
    config.ranged_locality_prefetch = false;
    config.prefetch_budget_bytes = 0;
    config.eviction_mode = PeerEvictionMode::kMaster;
    if (owner) {
      config.fabric_read_staging_bytes = staging_bytes;
      config.fabric_read_staging_device = 0;
      config.ssd.enabled = GetParam() == TierType::SSD;
      config.ssd.ssd.storage_dir = ssd_directory_;
      config.ssd.ssd.capacity_bytes = 128 * kPage;
      config.ssd.ssd.io.backend = UMBPIoBackend::Posix;
      config.ssd.ssd.direct_io = false;
      config.ssd_staging_buffer_slots = 64;
      if (mixed) {
        BackendInstanceConfig native;
        native.name = "native";
        native.tier = TierType::HBM;
        native.hbm = config.hbm;
        native.hbm.buffer_sizes = {kObjectPages * kPage};
        BackendInstanceConfig cold;
        cold.name = "cold";
        cold.tier = GetParam();
        cold.dram = config.dram;
        cold.ssd = config.ssd;
        cold.ssd_staging_buffer_slots = config.ssd_staging_buffer_slots;
        config.backends = {native, cold};
        config.placement_policy = PoolPlacementPolicy::TIERED;
        LogicalTierConfig hot_tier;
        hot_tier.name = "hot";
        hot_tier.members = {{"native", 1}};
        hot_tier.entry = true;
        hot_tier.offload_to = {"cold"};
        LogicalTierConfig cold_tier;
        cold_tier.name = "cold";
        cold_tier.members = {{"cold", 1}};
        config.logical_tiers = {hot_tier, cold_tier};
      }
    }
    return config;
  }
  void StartPeers(size_t staging_bytes, bool mixed = false) {
    mixed_ = mixed;
    owner_ = std::make_unique<PoolClient>(Config(true, staging_bytes, mixed));
    ASSERT_TRUE(owner_->Init());
    ASSERT_TRUE(owner_->RegisterMemory(source_, 2 * kObject, mori::io::MemoryLocationType::GPU, 0,
                                       MemoryRegistration::kLocalCopyOnly));
    observed_ = std::make_shared<ObservedTierPlatform>();
    reader_ = std::make_unique<PoolClient>(Config(false, 0), observed_);
    ASSERT_TRUE(reader_->Init());
    ASSERT_TRUE(reader_->RegisterMemory(destination_, 2 * kObject,
                                        mori::io::MemoryLocationType::GPU, 0,
                                        MemoryRegistration::kFabricLocal));
    std::vector<bool> outcomes;
    if (mixed && GetParam() == TierType::SSD) {
      // Populate committed SSD storage through its local staging protocol.
      // Tiered protected PUT deliberately rejects SSD; this test covers mixed
      // reads and does not extend the remote protected-write protocol.
      outcomes = {owner_->Put("tier-key-0", source_, kObject), false};
      auto* cold = owner_->Backends().Get(TierType::SSD);
      ASSERT_NE(cold, nullptr);
      auto slots = cold->BatchAllocate({AllocateRequest{"tier-key-1", kObject}});
      ASSERT_EQ(slots.size(), 1u);
      ASSERT_EQ(slots[0].outcome, AllocateOutcome::kSuccessAllocated);
      size_t copied = 0;
      for (const auto& page : slots[0].pages) {
        const auto ref = cold->BufferRef(page.buffer_index);
        const auto length = std::min(kPage, kObject - copied);
        ASSERT_EQ(ref.loc, mori::io::MemoryLocationType::CPU);
        ASSERT_EQ(hipMemcpy(static_cast<char*>(ref.host_ptr) + page.page_index * kPage,
                            static_cast<char*>(source_) + kObject + copied, length,
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        copied += length;
      }
      ASSERT_EQ(copied, kObject);
      const auto committed = cold->BatchCommit({CommitRequest{slots[0].slot_id, "tier-key-1"}});
      ASSERT_EQ(committed.size(), 1u);
      outcomes[1] = committed[0].success;
    } else {
      outcomes =
          owner_->BatchPut({"tier-key-0", "tier-key-1"},
                           {source_, static_cast<char*>(source_) + kObject}, {kObject, kObject});
    }
    ASSERT_EQ(outcomes, (std::vector<bool>{true, true}));
    // Force reads to use the committed DRAM/SSD objects rather than the
    // producer's original GPU bytes.
    ASSERT_EQ(hipMemset(source_, 0xa7, 2 * kObject), hipSuccess);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::vector<std::optional<RouteGetResult>> routes;
    do {
      const auto status =
          reader_->Master().BatchRouteGet({"tier-key-0", "tier-key-1"}, {}, &routes);
      if (status.ok() && routes.size() == 2 && routes[0] && routes[1]) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } while (std::chrono::steady_clock::now() < deadline);
    ASSERT_EQ(routes.size(), 2u);
    for (size_t key = 0; key < routes.size(); ++key) {
      const auto& route = routes[key];
      ASSERT_TRUE(route.has_value());
      EXPECT_EQ(route->node_id, owner_->NodeId());
      EXPECT_EQ(route->tier, mixed_ && key == 0 ? TierType::HBM : GetParam());
      EXPECT_EQ(route->size, kObject);
    }
    VerifyOwnership();
  }
  void VerifyOwnership() {
    auto* storage = owner_->Backends().Get(GetParam());
    ASSERT_NE(storage, nullptr);
    for (const auto* backend : owner_->Backends().All()) {
      const auto* first = mixed_ ? owner_->Backends().Get(TierType::HBM) : storage;
      EXPECT_EQ(backend->Contains("tier-key-0"), backend == first);
      EXPECT_EQ(backend->Contains("tier-key-1"), backend == storage);
    }
    for (const auto* backend : reader_->Backends().All()) {
      EXPECT_FALSE(backend->Contains("tier-key-0"));
      EXPECT_FALSE(backend->Contains("tier-key-1"));
    }
  }
  std::shared_ptr<PoolClient::KvReadHandle> Read(size_t key) {
    auto* base = static_cast<char*>(destination_) + key * kObject;
    return reader_->SubmitBatchGetRangesGpuDirect(
        {"tier-key-" + std::to_string(key)},
        {{base + 13, base + 2 * kPage + 33, base + kObject - 498}}, {{kPage + 37, kPage + 19, 257}},
        {{0, 2 * kPage + 17, kObject - 513}});
  }
  void VerifyBytes(size_t key) {
    std::vector<unsigned char> actual(kObject);
    ASSERT_EQ(hipMemcpy(actual.data(), static_cast<char*>(destination_) + key * kObject, kObject,
                        hipMemcpyDeviceToHost),
              hipSuccess);
    const size_t destination_offsets[] = {13, 2 * kPage + 33, kObject - 498};
    const size_t source_offsets[] = {0, 2 * kPage + 17, kObject - 513};
    const size_t lengths[] = {kPage + 37, kPage + 19, 257};
    for (size_t offset = 0; offset < kObject; ++offset) {
      unsigned char expected = 0xd3;
      for (size_t range = 0; range < 3; ++range)
        if (offset >= destination_offsets[range] &&
            offset - destination_offsets[range] < lengths[range])
          expected = expected_[key * kObject + source_offsets[range] + offset -
                               destination_offsets[range]];
      ASSERT_EQ(actual[offset], expected) << "key=" << key << " destination offset=" << offset;
    }
  }
  void VerifyFabric(const ObservedTierPlatform::Counters& before, size_t reads) {
    const auto after = observed_->ReadCounters();
    EXPECT_EQ(after.fabric - before.fabric, reads * (2 * kPage + 313));
    EXPECT_EQ(after.other, before.other);
    EXPECT_GT(after.plans, before.plans);
  }

  std::string kernel_, ssd_directory_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> owner_, reader_;
  std::shared_ptr<ObservedTierPlatform> observed_;
  void* source_ = nullptr;
  void* destination_ = nullptr;
  std::vector<unsigned char> expected_;
  bool mixed_ = false;
};

TEST_P(UalinkTierSharing, MultiPageSubsetColdAndWarmStayInOwnerTier) {
  StartPeers(4 * kObjectPages * kPage);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    ASSERT_EQ(hipMemset(destination_, 0xd3, 2 * kObject), hipSuccess);
    const auto before = observed_->ReadCounters();
    auto first = Read(0);
    auto second = Read(1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->Wait(), (std::vector<bool>{true}));
    EXPECT_EQ(second->Wait(), (std::vector<bool>{true}));
    EXPECT_FALSE(first->ResourcesQuarantined());
    EXPECT_FALSE(second->ResourcesQuarantined());
    VerifyBytes(0);
    VerifyBytes(1);
    VerifyFabric(before, 2);
    VerifyOwnership();
  }
}

TEST_P(UalinkTierSharing, OutstandingGrantBoundsStagingAndReleaseEnablesRetry) {
  StartPeers(kObjectPages * kPage);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  ASSERT_EQ(hipMemset(destination_, 0xd3, 2 * kObject), hipSuccess);
  const auto before = observed_->ReadCounters();
  auto held = Read(0);
  ASSERT_NE(held, nullptr);
  // Even if its GPU copy has finished, an unwaited handle still owns the
  // owner's grant. A second different key cannot reuse those staging pages.
  {
    ScopedBusyTimeout busy_timeout;
    auto pressured = Read(1);
    ASSERT_NE(pressured, nullptr);
    EXPECT_EQ(pressured->Wait(), (std::vector<bool>{false}));
    EXPECT_FALSE(pressured->ResourcesQuarantined());
  }
  EXPECT_EQ(held->Wait(), (std::vector<bool>{true}));
  EXPECT_FALSE(held->ResourcesQuarantined());
  auto retry = Read(1);
  ASSERT_NE(retry, nullptr);
  EXPECT_EQ(retry->Wait(), (std::vector<bool>{true}));
  EXPECT_FALSE(retry->ResourcesQuarantined());
  VerifyBytes(0);
  VerifyBytes(1);
  VerifyFabric(before, 2);
  VerifyOwnership();
}

TEST_P(UalinkTierSharing, NonzeroWindowBatchThenWholeObjects) {
  StartPeers(4 * kObjectPages * kPage);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  ASSERT_EQ(hipMemset(destination_, 0xd3, 2 * kObject), hipSuccess);
  constexpr size_t offsets[] = {kPage + 31, 3 * kPage + 19};
  constexpr size_t destinations[] = {13, 2 * kPage + 29};
  constexpr size_t lengths[] = {kPage + 7, 223};
  std::vector<std::vector<void*>> ranges;
  for (size_t key = 0; key < 2; ++key) {
    auto* base = static_cast<char*>(destination_) + key * kObject;
    ranges.push_back({base + destinations[0], base + destinations[1]});
  }
  auto before = observed_->ReadCounters();
  auto read = reader_->SubmitBatchGetRangesGpuDirect(
      {"tier-key-0", "tier-key-1"}, ranges, {{lengths[0], lengths[1]}, {lengths[0], lengths[1]}},
      {{offsets[0], offsets[1]}, {offsets[0], offsets[1]}});
  ASSERT_NE(read, nullptr);
  EXPECT_EQ(read->Wait(), (std::vector<bool>{true, true}));
  EXPECT_FALSE(read->ResourcesQuarantined());
  auto after = observed_->ReadCounters();
  EXPECT_EQ(after.fabric - before.fabric, 2 * (lengths[0] + lengths[1]));
  EXPECT_EQ(after.other, before.other);
  std::vector<unsigned char> actual(2 * kObject);
  ASSERT_EQ(hipMemcpy(actual.data(), destination_, actual.size(), hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t key = 0; key < 2; ++key) {
    for (size_t offset = 0; offset < kObject; ++offset) {
      unsigned char expected = 0xd3;
      for (size_t range = 0; range < 2; ++range) {
        if (offset >= destinations[range] && offset - destinations[range] < lengths[range])
          expected = expected_[key * kObject + offsets[range] + offset - destinations[range]];
      }
      ASSERT_EQ(actual[key * kObject + offset], expected) << "key=" << key << " offset=" << offset;
    }
  }
  VerifyOwnership();

  before = observed_->ReadCounters();
  EXPECT_EQ(reader_->BatchGet({"tier-key-0", "tier-key-1"},
                              {destination_, static_cast<char*>(destination_) + kObject},
                              {kObject, kObject}),
            (std::vector<bool>{true, true}));
  after = observed_->ReadCounters();
  EXPECT_EQ(after.fabric - before.fabric, 2 * kObject);
  EXPECT_EQ(after.other, before.other);
  ASSERT_EQ(hipMemcpy(actual.data(), destination_, actual.size(), hipMemcpyDeviceToHost),
            hipSuccess);
  EXPECT_EQ(actual, expected_);
  VerifyOwnership();
}

TEST_P(UalinkTierSharing, MixedNativeFullAndColdPartialReuseSeparateDescriptors) {
  StartPeers(4 * kObjectPages * kPage, true);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  ASSERT_EQ(owner_->Backends().BackendId(owner_->Backends().Get(TierType::HBM)), 0u);
  constexpr size_t kOffset = kPage + 31;
  constexpr size_t kBytes = kPage + 157;
  for (size_t iteration = 0; iteration < 2; ++iteration) {
    ASSERT_EQ(hipMemset(destination_, 0xd3, 2 * kObject), hipSuccess);
    const auto before = observed_->ReadCounters();
    auto read = reader_->SubmitBatchGetRangesGpuDirect(
        {"tier-key-0", "tier-key-1"},
        {{destination_}, {static_cast<char*>(destination_) + kObject + 13}}, {{kObject}, {kBytes}},
        {{0}, {kOffset}});
    ASSERT_NE(read, nullptr);
    EXPECT_EQ(read->Wait(), (std::vector<bool>{true, true}));
    EXPECT_FALSE(read->ResourcesQuarantined());
    const auto after = observed_->ReadCounters();
    EXPECT_EQ(after.fabric - before.fabric, kObject + kBytes);
    EXPECT_EQ(after.other, before.other);
    std::vector<unsigned char> actual(2 * kObject);
    ASSERT_EQ(hipMemcpy(actual.data(), destination_, actual.size(), hipMemcpyDeviceToHost),
              hipSuccess);
    for (size_t offset = 0; offset < kObject; ++offset) {
      ASSERT_EQ(actual[offset], expected_[offset]);
      const auto expected = offset >= 13 && offset - 13 < kBytes
                                ? expected_[kObject + kOffset + offset - 13]
                                : static_cast<unsigned char>(0xd3);
      ASSERT_EQ(actual[kObject + offset], expected) << "cold destination offset=" << offset;
    }
    VerifyOwnership();
  }
}

INSTANTIATE_TEST_SUITE_P(DramAndSsdOwners, UalinkTierSharing,
                         ::testing::Values(TierType::DRAM, TierType::SSD),
                         [](const ::testing::TestParamInfo<TierType>& info) {
                           return info.param == TierType::DRAM ? "DRAM" : "SSD";
                         });
}  // namespace
}  // namespace mori::umbp

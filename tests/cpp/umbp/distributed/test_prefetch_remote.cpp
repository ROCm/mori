// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <future>

#include "umbp/distributed/pool_client.h"

namespace mori::umbp {

// Only the transport is simulated. Allocation, contiguous-slot validation,
// publication, leases, abort and subsequent GET all use the real backends.
class PrefetchRemoteTest : public ::testing::Test {
 protected:
  std::unique_ptr<PoolClient> MakeClient(std::vector<size_t> buffers = {16384}) {
    PoolClientConfig config;
    config.master_config.node_id = "prefetch-remote-test";
    config.dram_page_size = 4096;
    config.dram.buffer_sizes.assign(buffers.begin(), buffers.end());
    auto client = std::make_unique<PoolClient>(config);
    EXPECT_TRUE(client->Init());
    return client;
  }

  auto Pull(PoolClient& client, const std::vector<std::string>& keys, size_t size = 4096) {
    std::vector<PoolClient::ReCacheJob> jobs;
    for (const auto& key : keys) {
      PoolClient::ReCacheJob job;
      job.key = key;
      job.size = size;
      job.route = RouteGetResult{};
      job.route->node_id = "remote";
      job.route->size = size;
      jobs.push_back(std::move(job));
    }
    return client.FetchWholeObjectsIntoMedium(jobs);
  }

  bool PutOnSsd(PoolClient& client, const std::string& key, const std::vector<char>& bytes) {
    return client.ExecuteLocalPut(key, bytes.data(), bytes.size(), TierType::SSD, "cold") ==
           PoolClient::PutAttemptOutcome::kSuccess;
  }

  void Transport(PoolClient& client, std::function<bool(const std::string&, void*, size_t)> copy) {
    client.prefetch_transport_for_test_ = [copy = std::move(copy)](const auto& plan,
                                                                   auto* fetched) {
      for (const auto& [node, items] : plan.remote_groups) {
        for (const auto& item : items) {
          auto* dst = static_cast<char*>(item.dst_ref->host_ptr) + item.dst_ref_offset;
          (*fetched)[item.index] = copy(*item.key, dst, item.size);
        }
      }
    };
  }
};

TEST_F(PrefetchRemoteTest, OnlyCommittedTransfersAreReadyAndFailedSlotsAreReusable) {
  auto client = MakeClient();
  Transport(*client, [](const auto& key, void* dst, size_t size) {
    if (key == "failed") return false;
    std::memset(dst, 0x37, size);
    return true;
  });
  auto results = Pull(*client, {"ready", "failed"});
  ASSERT_EQ(results.size(), 2u);
  EXPECT_TRUE(results[0].Ready());
  EXPECT_EQ(results[1].code, PrefetchCode::kTransferFailed);
  EXPECT_FALSE(client->Backends().Get(TierType::DRAM)->Contains("failed"));
  std::vector<char> bytes(4096);
  ASSERT_TRUE(client->Get("ready", bytes.data(), bytes.size()));
  EXPECT_EQ(bytes, std::vector<char>(4096, 0x37));
  // Reuse every other page. A leaked failed allocation would prevent this.
  auto retry = Pull(*client, {"b", "c", "d"});
  for (const auto& result : retry) EXPECT_TRUE(result.Ready());
}

TEST_F(PrefetchRemoteTest, CrossBufferSlotFailsBeforeTransportAndReleasesItsPages) {
  auto client = MakeClient({4096, 4096});
  size_t calls = 0;
  Transport(*client, [&](const auto&, void*, size_t) {
    ++calls;
    return true;
  });
  auto result = Pull(*client, {"cross-mr"}, 8192);
  EXPECT_EQ(result[0].code, PrefetchCode::kFragmented);
  EXPECT_EQ(calls, 0u);
  EXPECT_FALSE(client->Backends().Get(TierType::DRAM)->Contains("cross-mr"));
  auto small = Pull(*client, {"a", "b"});
  for (const auto& r : small) EXPECT_TRUE(r.Ready());
}

TEST_F(PrefetchRemoteTest, CapacityFailureAndAlreadyResidentKeepPerInputResults) {
  auto client = MakeClient({4096});
  size_t calls = 0;
  Transport(*client, [&](const auto&, void* dst, size_t size) {
    ++calls;
    std::memset(dst, 0x18, size);
    return true;
  });
  ASSERT_TRUE(Pull(*client, {"resident"})[0].Ready());
  auto mixed = Pull(*client, {"resident", "no-space"});
  EXPECT_TRUE(mixed[0].Ready());
  EXPECT_EQ(mixed[1].code, PrefetchCode::kNoSpace);
  EXPECT_EQ(calls, 1u);
}

TEST_F(PrefetchRemoteTest, CommitRejectionIsNotReportedReady) {
  auto client = MakeClient();
  Transport(*client, [&](const auto&, void* dst, size_t size) {
    std::memset(dst, 0x51, size);
    // Invalidate the allocation generation after the fake copy completes.
    client->Backends().Get(TierType::DRAM)->ClearLocal();
    return true;
  });
  EXPECT_EQ(Pull(*client, {"rejected"})[0].code, PrefetchCode::kTransferFailed);
  EXPECT_FALSE(client->Backends().Get(TierType::DRAM)->Contains("rejected"));
}

TEST_F(PrefetchRemoteTest, TransportExceptionAbortsEveryPendingDestination) {
  auto client = MakeClient({4096, 4096});
  Transport(*client, [](const auto&, void*, size_t) -> bool {
    throw std::runtime_error("transport failure");
  });
  EXPECT_THROW(Pull(*client, {"a", "b"}), std::runtime_error);
  Transport(*client, [](const auto&, void*, size_t) { return true; });
  for (const auto& result : Pull(*client, {"c", "d"})) EXPECT_TRUE(result.Ready());
}

TEST_F(PrefetchRemoteTest, RemoteCopiesUsePolicyOrderAndFallBackWhenFirstBackendIsFull) {
  PoolClientConfig config;
  config.master_config.node_id = "prefetch-tiered";
  config.dram_page_size = 4096;
  BackendInstanceConfig cold;
  cold.name = "cold";
  cold.tier = TierType::DRAM;
  cold.dram.buffer_sizes = {4096};
  BackendInstanceConfig hot = cold;
  hot.name = "hot";
  config.backends = {cold, hot};  // registry order intentionally opposes policy
  config.placement_policy = PoolPlacementPolicy::TIERED;
  LogicalTierConfig hot_tier;
  hot_tier.name = "hot";
  hot_tier.entry = true;
  hot_tier.members = {{"hot", 1}};
  hot_tier.offload_to = {"cold"};
  LogicalTierConfig cold_tier;
  cold_tier.name = "cold";
  cold_tier.members = {{"cold", 1}};
  config.logical_tiers = {hot_tier, cold_tier};
  PoolClient client(config);
  ASSERT_TRUE(client.Init());
  Transport(client, [](const auto&, void* dst, size_t size) {
    std::memset(dst, 0x42, size);
    return true;
  });
  ASSERT_TRUE(Pull(client, {"first"})[0].Ready());
  EXPECT_TRUE(client.Backends().Get("hot")->Contains("first"));
  ASSERT_TRUE(Pull(client, {"second"})[0].Ready());
  EXPECT_TRUE(client.Backends().Get("cold")->Contains("second"));
}

TEST_F(PrefetchRemoteTest, MasterlessSsdPrefetchReadsActualFileBytesIntoDram) {
  char path[] = "/tmp/umbp-prefetch-ssd-XXXXXX";
  ASSERT_NE(mkdtemp(path), nullptr);
  struct Directory {
    std::string path;
    ~Directory() { std::filesystem::remove_all(path); }
  } directory{path};
  PoolClientConfig config;
  config.master_config.node_id = "prefetch-ssd-file";
  config.dram_page_size = 4096;
  BackendInstanceConfig hot;
  hot.name = "hot";
  hot.tier = TierType::DRAM;
  hot.dram.buffer_sizes = {1 << 20};
  BackendInstanceConfig cold;
  cold.name = "cold";
  cold.tier = TierType::SSD;
  cold.ssd.enabled = true;
  cold.ssd.ssd.enabled = true;
  cold.ssd.ssd.storage_dir = path;
  cold.ssd.ssd.capacity_bytes = 8 << 20;
  cold.ssd.ssd.segment_size_bytes = 1 << 20;
  cold.ssd.ssd.io.backend = UMBPIoBackend::Posix;
  cold.ssd.ssd.direct_io = false;  // this test may run on tmpfs
  cold.ssd_staging_buffer_slots = 4;
  config.backends = {hot, cold};
  config.placement_policy = PoolPlacementPolicy::TIERED;
  LogicalTierConfig hot_tier;
  hot_tier.name = "hot";
  hot_tier.entry = true;
  hot_tier.members = {{"hot", 1}};
  hot_tier.offload_to = {"cold"};
  LogicalTierConfig cold_tier;
  cold_tier.name = "cold";
  cold_tier.members = {{"cold", 1}};
  config.logical_tiers = {hot_tier, cold_tier};
  PoolClient client(config);
  ASSERT_TRUE(client.Init());
  std::vector<char> expected(4096);
  for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<char>((i * 37 + 11) & 255);
  ASSERT_TRUE(PutOnSsd(client, "file-key", expected));
  EXPECT_FALSE(client.Backends().Get("hot")->Contains("file-key"));
  auto submitted = client.SubmitPrefetch({"file-key", "missing", "file-key"});
  ASSERT_TRUE(submitted.accepted);
  PrefetchStatus status;
  for (int i = 0; i < 200; ++i) {
    status = client.GetPrefetchStatus(submitted.request_id);
    if (PrefetchTerminal(status.state)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_EQ(status.state, PrefetchState::kPartial);
  EXPECT_EQ(status.completed_keys, 2u);
  EXPECT_EQ(status.results[0].source, PrefetchSource::kSsd);
  EXPECT_TRUE(client.Backends().Get("hot")->Contains("file-key"));
  EXPECT_TRUE(client.Backends().Get("cold")->Contains("file-key"));
  std::vector<char> actual(expected.size());
  ASSERT_TRUE(client.Get("file-key", actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);
}

TEST_F(PrefetchRemoteTest, PendingDestinationIsNotPublishedDuringTransfer) {
  auto client = MakeClient();
  std::promise<void> entered, released;
  auto release = released.get_future().share();
  Transport(*client, [&](const auto&, void* dst, size_t size) {
    entered.set_value();
    EXPECT_EQ(release.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    std::memset(dst, 0x49, size);
    return true;
  });
  auto pull = std::async(std::launch::async, [&] { return Pull(*client, {"pending"}); });
  EXPECT_EQ(entered.get_future().wait_for(std::chrono::seconds(2)), std::future_status::ready);
  EXPECT_FALSE(client->Backends().Get(TierType::DRAM)->Contains("pending"));
  released.set_value();
  auto result = pull.get();
  EXPECT_TRUE(result[0].Ready());
  EXPECT_TRUE(client->Backends().Get(TierType::DRAM)->Contains("pending"));
}

}  // namespace mori::umbp

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

#include <cstring>
#include <filesystem>

#include "umbp/distributed/pool/peer_pool.h"
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
    client.FetchWholeObjectsIntoMedium(jobs);
    return client.default_pool_->LeasePrefetched(keys, std::chrono::milliseconds(500));
  }

  bool PutOnSsd(PoolClient& client, const std::string& key, const std::vector<char>& bytes) {
    return client.ExecuteLocalPut(key, bytes.data(), bytes.size(), TierType::SSD) ==
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
  EXPECT_TRUE(results[0]);
  EXPECT_EQ(results[1], false);
  EXPECT_FALSE(client->Backends().Get(TierType::DRAM)->Contains("failed"));
  std::vector<char> bytes(4096);
  ASSERT_TRUE(client->Get("ready", bytes.data(), bytes.size()));
  EXPECT_EQ(bytes, std::vector<char>(4096, 0x37));
  // Reuse every other page. A leaked failed allocation would prevent this.
  auto retry = Pull(*client, {"b", "c", "d"});
  for (const auto& result : retry) EXPECT_TRUE(result);
}

TEST_F(PrefetchRemoteTest, CrossBufferSlotFailsBeforeTransportAndReleasesItsPages) {
  auto client = MakeClient({4096, 4096});
  size_t calls = 0;
  Transport(*client, [&](const auto&, void*, size_t) {
    ++calls;
    return true;
  });
  auto result = Pull(*client, {"cross-mr"}, 8192);
  EXPECT_EQ(result[0], false);
  EXPECT_EQ(calls, 0u);
  EXPECT_FALSE(client->Backends().Get(TierType::DRAM)->Contains("cross-mr"));
  auto small = Pull(*client, {"a", "b"});
  for (const auto& r : small) EXPECT_TRUE(r);
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
  LogicalTierConfig tier;
  tier.members = {{"hot", 1}, {"cold", 1}};
  config.logical_tiers = {tier};
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
  EXPECT_EQ(status.ok, (std::vector<bool>{true, false, true}));
  EXPECT_TRUE(client.Backends().Get("hot")->Contains("file-key"));
  EXPECT_TRUE(client.Backends().Get("cold")->Contains("file-key"));
  std::vector<char> actual(expected.size());
  ASSERT_TRUE(client.Get("file-key", actual.data(), actual.size()));
  EXPECT_EQ(actual, expected);
}

}  // namespace mori::umbp

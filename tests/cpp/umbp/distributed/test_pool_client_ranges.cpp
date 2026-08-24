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
#include <hip/hip_runtime_api.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "umbp/common/device_gather.h"
#include "umbp/common/host_registration.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kObjectSize = 2 * kPageSize;
constexpr size_t kCallerCapacity = 16 * kPageSize;
constexpr size_t kTargetCapacity = 256 * kPageSize;
constexpr size_t kScratchSize = kObjectSize;  // Forces one object per sub-batch.

// The kernel dereferences the tier directly, so it needs the host region
// registered for GPU access; without it the client uses the copy engine.
bool GatherPathAvailable() {
  static const bool available = [] {
    std::vector<char> probe(64 * 1024);
    return HostTierRegistration(probe.data(), probe.size()).RegisteredBytes() != 0;
  }();
  return available;
}

uint16_t FreePort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd >= 0) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = 0;
    socklen_t len = sizeof(addr);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
      const uint16_t port = ntohs(addr.sin_port);
      ::close(fd);
      return port;
    }
    ::close(fd);
  }
  static std::atomic<uint16_t> next{56000};
  return next.fetch_add(1);
}

bool WaitForExists(PoolClient* client, const std::string& key) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (client->Exists(key)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return client->Exists(key);
}

bool WaitForExists(IUMBPClient* client, const std::string& key) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto found = client->BatchExists({key});
    if (found.size() == 1 && found[0]) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  const auto found = client->BatchExists({key});
  return found.size() == 1 && found[0];
}

class PoolClientRangesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    // The registry derives a 2 s client heartbeat from this TTL: short enough
    // for normal publication tests, while the 12 s expiry window keeps a stopped
    // caller alive through the stale-self assertion.
    master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds(4);
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    master_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 100 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);
    master_address_ = "localhost:" + std::to_string(master_->GetBoundPort());

    caller_get_scratch_.resize(kScratchSize);
    caller_put_scratch_.resize(kScratchSize);
    target_get_scratch_.resize(kScratchSize);
    target_put_scratch_.resize(kScratchSize);
    caller_ =
        MakeClient("ranges-caller", kCallerCapacity, caller_get_scratch_, caller_put_scratch_);
    target_ =
        MakeClient("ranges-target", kTargetCapacity, target_get_scratch_, target_put_scratch_);
  }

  void TearDown() override {
    if (caller_) caller_->Shutdown();
    if (target_) target_->Shutdown();
    caller_.reset();
    target_.reset();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
    master_.reset();
  }

  std::unique_ptr<PoolClient> MakeClient(const std::string& node_id, size_t dram_capacity,
                                         std::vector<char>& get_scratch,
                                         std::vector<char>& put_scratch) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_address_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = FreePort();
    cfg.dram_page_size = kPageSize;
    // Smaller than one object: ranged remote I/O can pass only through the
    // registered arena's zero-copy descriptor, never the legacy staging path.
    cfg.staging_buffer_size = 1024;
    // The DRAM backend self-allocates its own pool (refactor Phase 2b), so the
    // test hands it a size rather than a buffer.
    cfg.dram.buffer_sizes = {dram_capacity};
    // Separate GET and PUT arenas so remote ranged get/put run concurrently.
    cfg.ranged_get_scratch_buffer = get_scratch.data();
    cfg.ranged_get_scratch_size = get_scratch.size();
    cfg.ranged_put_scratch_buffer = put_scratch.data();
    cfg.ranged_put_scratch_size = put_scratch.size();
    cfg.cache_remote_fetches = false;
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    EXPECT_TRUE(client->Init());
    EXPECT_TRUE(client->RegisterMemory(get_scratch.data(), get_scratch.size()));
    EXPECT_TRUE(client->RegisterMemory(put_scratch.data(), put_scratch.size()));
    return client;
  }

  // A full DistributedClient (not the bare PoolClient the other cases drive),
  // because the opt-in this asserts lives in DistributedClient's constructor
  // and in SupportsRangedIO(), neither of which PoolClient sees.
  UMBPConfig DistributedConfig(const std::string& node_id, size_t ranged_scratch_size) {
    UMBPConfig config;
    config.dram.capacity_bytes = 4 * kTargetCapacity;
    config.dram.use_hugepages = false;
    config.ssd.enabled = false;

    UMBPDistributedConfig distributed;
    distributed.master_config.node_id = node_id;
    distributed.master_config.node_address = "127.0.0.1";
    distributed.master_config.master_address = master_address_;
    distributed.io_engine.host = "0.0.0.0";
    distributed.io_engine.port = 0;
    distributed.peer_service_port = FreePort();
    distributed.dram_page_size = kPageSize;
    distributed.staging_buffer_size = 4 * kObjectSize;
    distributed.ranged_scratch_size = ranged_scratch_size;
    distributed.cache_remote_fetches = false;
    config.distributed = std::move(distributed);
    return config;
  }

  // Publish a replica on one specific node, bypassing RoutePut's de-duplication
  // so two nodes can hold the same key. Writes through the backend's published
  // endpoint rather than a caller-owned buffer, since the medium owns its pool.
  void PutLocalReplica(PoolClient* client, const std::string& key,
                       const std::vector<char>& object) {
    auto* backend = client->Backends().Get(TierType::DRAM);
    ASSERT_NE(backend, nullptr);
    auto allocated = backend->BatchAllocate({AllocateRequest{key, object.size()}}).front();
    ASSERT_EQ(allocated.outcome, AllocateOutcome::kSuccessAllocated);

    size_t copied = 0;
    for (const auto& page : allocated.pages) {
      TransferRef buf = backend->BufferRef(page.buffer_index);
      ASSERT_TRUE(buf.HasHostPtr());
      const size_t bytes = std::min(kPageSize, object.size() - copied);
      const size_t offset = static_cast<size_t>(page.page_index) * kPageSize;
      ASSERT_LE(offset + bytes, buf.size);
      std::memcpy(static_cast<char*>(buf.host_ptr) + offset, object.data() + copied, bytes);
      copied += bytes;
    }
    ASSERT_EQ(copied, object.size());

    auto committed = backend->BatchCommit({CommitRequest{allocated.slot_id, key}}).front();
    ASSERT_TRUE(committed.success);
    ASSERT_EQ(committed.bytes_committed, object.size());
  }

  // Two DRAM instances under a tiered policy whose entry tier is the SECOND
  // one registered, so selecting by TierType and asking the policy disagree.
  std::unique_ptr<PoolClient> MakeTieredClient(const std::string& node_id) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_address_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = FreePort();
    cfg.dram_page_size = kPageSize;
    cfg.staging_buffer_size = 1024;
    tiered_get_scratch_.resize(kScratchSize);
    tiered_put_scratch_.resize(kScratchSize);
    cfg.ranged_get_scratch_buffer = tiered_get_scratch_.data();
    cfg.ranged_get_scratch_size = tiered_get_scratch_.size();
    cfg.ranged_put_scratch_buffer = tiered_put_scratch_.data();
    cfg.ranged_put_scratch_size = tiered_put_scratch_.size();
    cfg.cache_remote_fetches = false;

    BackendInstanceConfig cold;
    cold.name = "cold";
    cold.tier = TierType::DRAM;
    cold.dram.buffer_sizes = {kTargetCapacity};
    BackendInstanceConfig hot = cold;
    hot.name = "hot";
    cfg.backends = {cold, hot};

    cfg.placement_policy = PoolPlacementPolicy::TIERED;
    LogicalTierConfig hot_tier;
    hot_tier.name = "hot";
    hot_tier.members = {{"hot", 1}};
    hot_tier.entry = true;
    hot_tier.offload_to = {"cold"};
    LogicalTierConfig cold_tier;
    cold_tier.name = "cold";
    cold_tier.members = {{"cold", 1}};
    cfg.logical_tiers = {hot_tier, cold_tier};

    auto client = std::make_unique<PoolClient>(std::move(cfg));
    EXPECT_TRUE(client->Init());
    EXPECT_TRUE(client->RegisterMemory(tiered_get_scratch_.data(), tiered_get_scratch_.size()));
    EXPECT_TRUE(client->RegisterMemory(tiered_put_scratch_.data(), tiered_put_scratch_.size()));
    return client;
  }

  bool WaitForRoute(PoolClient* client, const std::string& key,
                    const std::unordered_set<std::string>& excludes,
                    const std::string& expected_node) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      std::vector<std::optional<RouteGetResult>> routes;
      const auto status = client->Master().BatchRouteGet({key}, excludes, &routes);
      if (status.ok() && routes.size() == 1 && routes[0].has_value() &&
          routes[0]->node_id == expected_node) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return false;
  }

  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::string master_address_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
  std::vector<char> caller_get_scratch_;
  std::vector<char> caller_put_scratch_;
  std::vector<char> target_get_scratch_;
  std::vector<char> target_put_scratch_;
  std::vector<char> tiered_get_scratch_;
  std::vector<char> tiered_put_scratch_;
};

// ---------------------------------------------------------------------------
// Ranged I/O must go through the pool, not straight at the backend registry.
// A tiered policy shows both directions at once: the entry tier is
// deliberately not the first-registered DRAM instance, so a write that picked
// a backend by TierType lands on the wrong one, and a read that scanned the
// registry serves the bytes but leaves the pool's placement index and read
// accounting empty -- which is what makes watermark offload and
// promote-on-read inert.
// ---------------------------------------------------------------------------
TEST_F(PoolClientRangesTest, LocalRangedIoIsPlacedAndAccountedByThePool) {
  caller_->Shutdown();
  target_->Shutdown();
  auto tiered = MakeTieredClient("ranges-tiered");
  ASSERT_NE(tiered, nullptr);

  auto* cold = tiered->Backends().Get("cold");
  auto* hot = tiered->Backends().Get("hot");
  ASSERT_NE(cold, nullptr);
  ASSERT_NE(hot, nullptr);
  // Selecting by TierType would hand back "cold": it is the first DRAM entry.
  ASSERT_EQ(tiered->Backends().Get(TierType::DRAM), cold);

  const std::string key = "tiered-range-object";
  std::vector<char> head(1024, 0x41);
  std::vector<char> tail(kObjectSize - head.size(), 0x42);
  const std::vector<const void*> srcs = {head.data(), tail.data()};
  const std::vector<size_t> sizes = {head.size(), tail.size()};
  const std::vector<size_t> offsets = {0, head.size()};
  ASSERT_EQ(tiered->BatchPutRanges({key}, {kObjectSize}, {srcs}, {sizes}, {offsets}),
            std::vector<bool>({true}));

  // Placed where the policy's entry tier says, not where the routed TierType
  // would have put it.
  EXPECT_TRUE(hot->Contains(key));
  EXPECT_FALSE(cold->Contains(key));

  std::vector<char> readback(kObjectSize, 0);
  const std::vector<void*> dsts = {readback.data()};
  ASSERT_EQ(tiered->BatchGetRanges({key}, {dsts}, {{kObjectSize}}, {{0}}),
            std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(readback.data(), head.data(), head.size()), 0);
  EXPECT_EQ(std::memcmp(readback.data() + head.size(), tail.data(), tail.size()), 0);

  // Served through the pool, so the read is attributed to the logical tier
  // that held the key. A registry scan would leave this map empty.
  const auto hits = tiered->TierReadHits();
  const auto hot_hits = hits.find("hot");
  ASSERT_NE(hot_hits, hits.end());
  EXPECT_GE(hot_hits->second, 1u);

  tiered->Shutdown();
}

TEST_F(PoolClientRangesTest, RemoteRoundTripSubBatchesAndInstallsLocally) {
  constexpr size_t kKeys = 2;
  std::vector<std::string> keys = {"range-object-0", "range-object-1"};
  std::vector<std::vector<char>> objects(kKeys, std::vector<char>(kObjectSize));
  for (size_t k = 0; k < kKeys; ++k) {
    for (size_t i = 0; i < kObjectSize; ++i) {
      objects[k][i] = static_cast<char>((i * 13 + k * 37) & 0xff);
    }
  }

  std::vector<size_t> object_sizes(kKeys, kObjectSize);
  std::vector<std::vector<const void*>> put_ptrs(kKeys);
  std::vector<std::vector<size_t>> put_sizes(kKeys);
  std::vector<std::vector<size_t>> put_offsets(kKeys);
  for (size_t k = 0; k < kKeys; ++k) {
    put_ptrs[k] = {objects[k].data(), objects[k].data() + 1500, objects[k].data() + 6000};
    put_sizes[k] = {1500, 4500, kObjectSize - 6000};
    put_offsets[k] = {0, 1500, 6000};
  }

  auto put = caller_->BatchPutRanges(keys, object_sizes, put_ptrs, put_sizes, put_offsets);
  ASSERT_EQ(put.size(), kKeys);
  EXPECT_TRUE(put[0]);
  EXPECT_TRUE(put[1]);
  caller_->Master().FlushHeartbeat();
  target_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitForExists(caller_.get(), keys[0]));
  ASSERT_TRUE(WaitForExists(caller_.get(), keys[1]));

  std::vector<std::vector<char>> out_a(kKeys, std::vector<char>(900, 0));
  std::vector<std::vector<char>> out_b(kKeys, std::vector<char>(700, 0));
  std::vector<std::vector<void*>> get_ptrs(kKeys);
  std::vector<std::vector<size_t>> get_sizes(kKeys, {900, 700});
  std::vector<std::vector<size_t>> get_offsets(kKeys, {200, 5000});
  for (size_t k = 0; k < kKeys; ++k) get_ptrs[k] = {out_a[k].data(), out_b[k].data()};

  auto first_get = caller_->BatchGetRanges(keys, get_ptrs, get_sizes, get_offsets);
  ASSERT_EQ(first_get.size(), kKeys);
  for (size_t k = 0; k < kKeys; ++k) {
    ASSERT_TRUE(first_get[k]) << "key=" << keys[k];
    EXPECT_EQ(std::memcmp(out_a[k].data(), objects[k].data() + 200, 900), 0);
    EXPECT_EQ(std::memcmp(out_b[k].data(), objects[k].data() + 5000, 700), 0);
  }

  // The first remote get synchronously installs both objects in caller DRAM.
  // A second ranged get must succeed even before the ADD events reach master.
  for (auto& v : out_a) std::fill(v.begin(), v.end(), 0);
  for (auto& v : out_b) std::fill(v.begin(), v.end(), 0);
  auto second_get = caller_->BatchGetRanges(keys, get_ptrs, get_sizes, get_offsets);
  EXPECT_EQ(second_get, std::vector<bool>({true, true}));
  for (size_t k = 0; k < kKeys; ++k) {
    EXPECT_EQ(std::memcmp(out_a[k].data(), objects[k].data() + 200, 900), 0);
    EXPECT_EQ(std::memcmp(out_b[k].data(), objects[k].data() + 5000, 700), 0);
  }
}

TEST_F(PoolClientRangesTest, InvalidRangesFailWithoutPublishing) {
  std::vector<char> src(kObjectSize, 0x5a);
  std::vector<std::string> keys = {"range-gap", "range-overlap"};
  std::vector<size_t> object_sizes = {kObjectSize, kObjectSize};
  std::vector<std::vector<const void*>> ptrs = {{src.data(), src.data() + 4096},
                                                {src.data(), src.data() + 2048}};
  std::vector<std::vector<size_t>> sizes = {{4096, 2048}, {4096, 4096}};
  std::vector<std::vector<size_t>> offsets = {{0, 6144}, {0, 2048}};
  auto result = caller_->BatchPutRanges(keys, object_sizes, ptrs, sizes, offsets);
  EXPECT_EQ(result, std::vector<bool>({false, false}));
  EXPECT_FALSE(caller_->Exists(keys[0]));
  EXPECT_FALSE(caller_->Exists(keys[1]));
}

TEST_F(PoolClientRangesTest, DistributedScratchIsExplicitlyOptedIn) {
  EXPECT_EQ(UMBPDistributedConfig{}.ranged_scratch_size, 0u);

  auto config = DistributedConfig("ranges-no-scratch", /*ranged_scratch_size=*/0);
  std::string validation_error;
  ASSERT_TRUE(config.Validate(&validation_error)) << validation_error;
  auto client = CreateUMBPClient(config);
  ASSERT_NE(client, nullptr);
  EXPECT_EQ(client->GetDeploymentMode(), UMBPDeploymentMode::Distributed);
  EXPECT_FALSE(client->SupportsRangedIO());

  const std::string key = "ordinary-io-without-ranged-scratch";
  std::vector<char> source(kObjectSize);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<char>((i * 17 + 3) & 0xff);
  }
  auto put = client->BatchPut({key}, {reinterpret_cast<uintptr_t>(source.data())}, {source.size()});
  ASSERT_EQ(put, std::vector<bool>({true}));
  ASSERT_TRUE(client->Flush());
  ASSERT_TRUE(WaitForExists(client.get(), key));

  std::vector<char> restored(source.size(), 0);
  auto get =
      client->BatchGet({key}, {reinterpret_cast<uintptr_t>(restored.data())}, {restored.size()});
  ASSERT_EQ(get, std::vector<bool>({true}));
  EXPECT_EQ(restored, source);

  // Bypassing the advertised capability remains a safe per-key failure. Use a
  // missing key so the call reaches the remote-scratch guard after its local probe.
  std::vector<char> ranged_out(16, 0);
  auto ranged = client->BatchGetRanges({"missing-without-ranged-scratch"},
                                       {{reinterpret_cast<uintptr_t>(ranged_out.data())}},
                                       {{ranged_out.size()}}, {{0}});
  EXPECT_EQ(ranged, std::vector<bool>({false}));
  client->Close();
  client.reset();

  auto opted_in_config = DistributedConfig("ranges-with-scratch", kScratchSize);
  ASSERT_TRUE(opted_in_config.Validate(&validation_error)) << validation_error;
  auto opted_in = CreateUMBPClient(opted_in_config);
  ASSERT_NE(opted_in, nullptr);
  EXPECT_TRUE(opted_in->SupportsRangedIO());
  opted_in->Close();
}

TEST_F(PoolClientRangesTest, StaleSelfLocationIsExcludedBeforeRemoteFetch) {
  const std::string key = "range-stale-self";
  std::vector<char> stale_object(kObjectSize, 0x11);
  std::vector<char> remote_object(kObjectSize);
  for (size_t i = 0; i < remote_object.size(); ++i) {
    remote_object[i] = static_cast<char>((i * 29 + 7) & 0xff);
  }

  // Publish two replicas without going through RoutePut's de-duplication. The
  // caller heartbeat is then stopped before local eviction, deliberately
  // leaving its location stale at master while the target remains readable.
  PutLocalReplica(caller_.get(), key, stale_object);
  PutLocalReplica(target_.get(), key, remote_object);
  caller_->Master().FlushHeartbeat();
  target_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitForRoute(caller_.get(), key, {"ranges-target"}, "ranges-caller"));
  ASSERT_TRUE(WaitForRoute(caller_.get(), key, {"ranges-caller"}, "ranges-target"));

  caller_->Master().StopHeartbeat();
  auto evicted = caller_->Backends().Get(TierType::DRAM)->Evict({key});
  ASSERT_EQ(evicted.size(), 1u);
  ASSERT_EQ(evicted[0].bytes_freed, kObjectSize);
  ASSERT_TRUE(WaitForRoute(caller_.get(), key, {}, "ranges-caller"))
      << "master must still prefer the caller's deliberately stale location";

  std::vector<char> out_a(777, 0);
  std::vector<char> out_b(999, 0);
  auto result =
      caller_->BatchGetRanges({key}, {{out_a.data(), out_b.data()}}, {{777, 999}}, {{123, 5000}});

  ASSERT_EQ(result, std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(out_a.data(), remote_object.data() + 123, out_a.size()), 0);
  EXPECT_EQ(std::memcmp(out_b.data(), remote_object.data() + 5000, out_b.size()), 0);
}

TEST_F(PoolClientRangesTest, GpuRangesUseGatherKernel) {
  struct HipAllocation {
    void* ptr = nullptr;
    ~HipAllocation() {
      if (ptr) (void)hipFree(ptr);
    }
  };

  const std::vector<size_t> part_sizes = {1024, 2048, kObjectSize - 3072};
  std::vector<HipAllocation> gpu_src(part_sizes.size());
  std::vector<std::vector<char>> host_src(part_sizes.size());
  std::vector<const void*> src_ptrs;
  for (size_t i = 0; i < part_sizes.size(); ++i) {
    host_src[i].resize(part_sizes[i], static_cast<char>(0x31 + i));
    ASSERT_EQ(hipMalloc(&gpu_src[i].ptr, part_sizes[i]), hipSuccess);
    ASSERT_EQ(hipMemcpy(gpu_src[i].ptr, host_src[i].data(), part_sizes[i], hipMemcpyHostToDevice),
              hipSuccess);
    src_ptrs.push_back(gpu_src[i].ptr);
  }

  const bool gather_available = DeviceGatherEnabled() && GatherPathAvailable();
  const uint64_t launches_before = DeviceGatherLaunchCount();
  const std::vector<std::string> gpu_keys = {"gpu-range-object-0", "gpu-range-object-1"};
  auto put = target_->BatchPutRanges(gpu_keys, {kObjectSize, kObjectSize}, {src_ptrs, src_ptrs},
                                     {part_sizes, part_sizes}, {{0, 1024, 3072}, {0, 1024, 3072}});
  ASSERT_EQ(put, std::vector<bool>({true, true}));
  if (gather_available) EXPECT_EQ(DeviceGatherLaunchCount(), launches_before + 1);

  const std::vector<size_t> get_sizes = {600, 700, 800};
  const std::vector<size_t> get_offsets = {100, 2200, 6500};
  std::vector<HipAllocation> gpu_dst(2 * get_sizes.size());
  std::vector<std::vector<void*>> dst_ptrs(2);
  for (size_t key = 0; key < 2; ++key) {
    for (size_t i = 0; i < get_sizes.size(); ++i) {
      auto& allocation = gpu_dst[key * get_sizes.size() + i];
      ASSERT_EQ(hipMalloc(&allocation.ptr, get_sizes[i]), hipSuccess);
      ASSERT_EQ(hipMemset(allocation.ptr, 0, get_sizes[i]), hipSuccess);
      dst_ptrs[key].push_back(allocation.ptr);
    }
  }
  auto get = target_->BatchGetRanges(gpu_keys, dst_ptrs, {get_sizes, get_sizes},
                                     {get_offsets, get_offsets});
  ASSERT_EQ(get, std::vector<bool>({true, true}));
  if (gather_available) EXPECT_EQ(DeviceGatherLaunchCount(), launches_before + 2);

  std::vector<char> full(kObjectSize);
  std::memset(full.data(), 0x31, 1024);
  std::memset(full.data() + 1024, 0x32, 2048);
  std::memset(full.data() + 3072, 0x33, kObjectSize - 3072);
  for (size_t key = 0; key < 2; ++key) {
    for (size_t i = 0; i < get_sizes.size(); ++i) {
      std::vector<char> actual(get_sizes[i]);
      ASSERT_EQ(hipMemcpy(actual.data(), gpu_dst[key * get_sizes.size() + i].ptr, get_sizes[i],
                          hipMemcpyDeviceToHost),
                hipSuccess);
      EXPECT_EQ(std::memcmp(actual.data(), full.data() + get_offsets[i], get_sizes[i]), 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Separate GET / PUT scratch arenas: a remote ranged get and put use different
// buffers under different mutexes, so they overlap and must never clobber each
// other.  The fixture already gives every client a separate get arena and put
// arena (each one object); these exercise the two together.
// ---------------------------------------------------------------------------

// A remote ranged put (PUT arena) then a remote ranged get (GET arena) of the
// same keys round-trips the exact bytes.
TEST_F(PoolClientRangesTest, SplitRemotePutThenGetRoundTrips) {
  constexpr size_t kKeys = 3;
  std::vector<std::string> keys = {"split-0", "split-1", "split-2"};
  std::vector<std::vector<char>> objects(kKeys, std::vector<char>(kObjectSize));
  for (size_t k = 0; k < kKeys; ++k) {
    for (size_t i = 0; i < kObjectSize; ++i) {
      objects[k][i] = static_cast<char>((i * 7 + k * 101) & 0xff);
    }
  }

  std::vector<size_t> object_sizes(kKeys, kObjectSize);
  std::vector<std::vector<const void*>> put_ptrs(kKeys);
  std::vector<std::vector<size_t>> put_sizes(kKeys);
  std::vector<std::vector<size_t>> put_offsets(kKeys);
  for (size_t k = 0; k < kKeys; ++k) {
    put_ptrs[k] = {objects[k].data(), objects[k].data() + 2048};
    put_sizes[k] = {2048, kObjectSize - 2048};
    put_offsets[k] = {0, 2048};
  }
  auto put = caller_->BatchPutRanges(keys, object_sizes, put_ptrs, put_sizes, put_offsets);
  ASSERT_EQ(put, std::vector<bool>(kKeys, true));
  caller_->Master().FlushHeartbeat();
  target_->Master().FlushHeartbeat();
  for (const auto& key : keys) ASSERT_TRUE(WaitForExists(caller_.get(), key));

  std::vector<std::vector<char>> out(kKeys, std::vector<char>(kObjectSize, 0));
  std::vector<std::vector<void*>> get_ptrs(kKeys);
  std::vector<std::vector<size_t>> get_sizes(kKeys, {kObjectSize});
  std::vector<std::vector<size_t>> get_offsets(kKeys, {0});
  for (size_t k = 0; k < kKeys; ++k) get_ptrs[k] = {out[k].data()};

  auto got = caller_->BatchGetRanges(keys, get_ptrs, get_sizes, get_offsets);
  ASSERT_EQ(got, std::vector<bool>(kKeys, true));
  for (size_t k = 0; k < kKeys; ++k) {
    EXPECT_EQ(std::memcmp(out[k].data(), objects[k].data(), kObjectSize), 0) << "key=" << keys[k];
  }
}

// Concurrent remote gets (GET arena) and remote puts (PUT arena) on disjoint key
// sets: every payload must survive byte-for-byte, proving the two arenas never
// overwrite each other under real contention.
TEST_F(PoolClientRangesTest, SplitConcurrentGetPutDoNotClobber) {
  constexpr size_t kSeed = 8;  // keys pre-seeded on target for the get threads
  constexpr size_t kPutPerThread = 8;
  constexpr size_t kGetThreads = 3;
  constexpr size_t kPutThreads = 3;

  // Seed keys the get threads will fetch remotely; each has a distinct fill.
  std::vector<std::string> seed_keys(kSeed);
  std::vector<std::vector<char>> seed_objs(kSeed, std::vector<char>(kObjectSize));
  for (size_t s = 0; s < kSeed; ++s) {
    for (size_t i = 0; i < kObjectSize; ++i)
      seed_objs[s][i] = static_cast<char>((i + s * 17) & 0xff);
    seed_keys[s] = "seed-" + std::to_string(s);
    PutLocalReplica(target_.get(), seed_keys[s], seed_objs[s]);
  }
  target_->Master().FlushHeartbeat();
  caller_->Master().FlushHeartbeat();
  for (const auto& k : seed_keys) ASSERT_TRUE(WaitForExists(caller_.get(), k));

  std::atomic<bool> ok{true};
  std::vector<std::thread> threads;

  // GET threads: read whole seeded objects, verify bytes (GET arena).
  for (size_t t = 0; t < kGetThreads; ++t) {
    threads.emplace_back([&, t] {
      for (size_t it = 0; it < 12; ++it) {
        const size_t s = (t + it) % kSeed;
        std::vector<char> out(kObjectSize, 0);
        std::vector<std::vector<void*>> ptrs = {{out.data()}};
        std::vector<std::vector<size_t>> sizes = {{kObjectSize}};
        std::vector<std::vector<size_t>> offsets = {{0}};
        auto r = caller_->BatchGetRanges({seed_keys[s]}, ptrs, sizes, offsets);
        if (r.size() != 1 || !r[0] || std::memcmp(out.data(), seed_objs[s].data(), kObjectSize)) {
          ok.store(false);
        }
      }
    });
  }

  // PUT threads: write unique objects (PUT arena) concurrently with the gets.
  // Read-back verification happens after join to avoid racing master's
  // ADD-event propagation.
  auto put_key = [](size_t t, size_t p) {
    return "cput-" + std::to_string(t) + "-" + std::to_string(p);
  };
  auto put_fill = [](size_t t, size_t p) { return static_cast<int>((t * 31 + p * 3 + 1) & 0xff); };
  for (size_t t = 0; t < kPutThreads; ++t) {
    threads.emplace_back([&, t] {
      for (size_t p = 0; p < kPutPerThread; ++p) {
        std::vector<char> obj(kObjectSize);
        std::memset(obj.data(), put_fill(t, p), kObjectSize);
        std::vector<std::vector<const void*>> src = {{obj.data(), obj.data() + 4096}};
        std::vector<std::vector<size_t>> sizes = {{4096, kObjectSize - 4096}};
        std::vector<std::vector<size_t>> offs = {{0, 4096}};
        auto pr = caller_->BatchPutRanges({put_key(t, p)}, {kObjectSize}, src, sizes, offs);
        if (pr.size() != 1 || !pr[0]) ok.store(false);
      }
    });
  }

  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());

  // After the concurrent phase, every put's bytes must be intact — a clobber
  // between the two arenas would corrupt these.
  caller_->Master().FlushHeartbeat();
  target_->Master().FlushHeartbeat();
  for (size_t t = 0; t < kPutThreads; ++t) {
    for (size_t p = 0; p < kPutPerThread; ++p) {
      const std::string key = put_key(t, p);
      ASSERT_TRUE(WaitForExists(caller_.get(), key));
      std::vector<char> back(kObjectSize, 0xAB);
      std::vector<std::vector<void*>> gp = {{back.data()}};
      std::vector<std::vector<size_t>> gs = {{kObjectSize}};
      std::vector<std::vector<size_t>> go = {{0}};
      auto gr = caller_->BatchGetRanges({key}, gp, gs, go);
      ASSERT_EQ(gr.size(), 1u);
      ASSERT_TRUE(gr[0]) << "key=" << key;
      std::vector<char> expect(kObjectSize, static_cast<char>(put_fill(t, p)));
      EXPECT_EQ(std::memcmp(back.data(), expect.data(), kObjectSize), 0) << "key=" << key;
    }
  }
}

}  // namespace
}  // namespace mori::umbp

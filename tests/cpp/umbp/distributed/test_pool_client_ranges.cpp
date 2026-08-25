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
                                         std::vector<char>& put_scratch,
                                         bool locality_prefetch = true) {
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
    cfg.ranged_locality_prefetch = locality_prefetch;
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

  // Publish `object` on the target only, so the caller's sole route to it is
  // remote.  PutLocalReplica writes the target's medium directly, which is what
  // keeps a copy off the caller.
  void SeedRemoteObject(const std::string& key, const std::vector<char>& object) {
    PutLocalReplica(target_.get(), key, object);
    target_->Master().FlushHeartbeat();
    caller_->Master().FlushHeartbeat();
    ASSERT_TRUE(WaitForExists(caller_.get(), key));
  }

  // Is `key` readable from this node's own medium?  The same question phase 1
  // of BatchGetRanges asks, and therefore the thing that decides whether a
  // later read goes remote.
  static bool LocallyResident(PoolClient* client, const std::string& key) {
    auto* backend = client->Backends().Get(client->Medium());
    return backend != nullptr &&
           backend->BatchResolve({key}, /*include_descs=*/false).front().found;
  }

  static bool WaitLocallyResident(PoolClient* client, const std::string& key,
                                  std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (LocallyResident(client, key)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return LocallyResident(client, key);
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
};

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

  // Reading the same ranges again must give the same bytes whichever way it is
  // served: still remotely, or locally once the background prefetch has landed.
  // Both are correct, so this asserts the payload rather than the route --
  // which route was taken is what LocalityPrefetch* below pins down.
  for (auto& v : out_a) std::fill(v.begin(), v.end(), 0);
  for (auto& v : out_b) std::fill(v.begin(), v.end(), 0);
  auto second_get = caller_->BatchGetRanges(keys, get_ptrs, get_sizes, get_offsets);
  EXPECT_EQ(second_get, std::vector<bool>({true, true}));
  for (size_t k = 0; k < kKeys; ++k) {
    EXPECT_EQ(std::memcmp(out_a[k].data(), objects[k].data() + 200, 900), 0);
    EXPECT_EQ(std::memcmp(out_b[k].data(), objects[k].data() + 5000, 700), 0);
  }
}

TEST_F(PoolClientRangesTest, LocalCopyOnlyRegistrationServesRangesAndSharesOnePlan) {
  // kLocalCopyOnly records a region without handing it to the IO engine. It
  // exists because the two halves of "register" are separable: a region that is
  // only ever a local copy endpoint has no use for an RDMA MR, but it still has
  // to be in the table -- a range that misses it is described by its own
  // address instead of its region base, so instead of every range sharing one
  // (src, dst) pair and collapsing into a single transfer plan, each range
  // becomes its own plan. That is a latency bug with no wrong answer attached,
  // which is exactly why it needs a test rather than a reviewer.
  const std::string key = "local-copy-only-ranges";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < object.size(); ++i) object[i] = static_cast<char>((i * 31 + 7) & 0xff);
  PutLocalReplica(caller_.get(), key, object);

  // One buffer, several disjoint ranges landing in it -- the layer-wise restore
  // shape, and the case the pair has to collapse for.
  std::vector<char> dst(kObjectSize, 0);
  ASSERT_TRUE(caller_->RegisterMemory(dst.data(), dst.size(), mori::io::MemoryLocationType::CPU, -1,
                                      MemoryRegistration::kLocalCopyOnly));

  // Disjoint and inside the 2-page object; the middle one straddles the page
  // boundary so the walk emits more than one fragment for a single range.
  const std::vector<size_t> sizes = {2048, 2048, 1024};
  const std::vector<size_t> offsets = {0, 3072, 6144};
  std::vector<void*> dsts;
  size_t cursor = 0;
  for (size_t bytes : sizes) {
    dsts.push_back(dst.data() + cursor);
    cursor += bytes;
  }
  ASSERT_EQ(caller_->BatchGetRanges({key}, {dsts}, {sizes}, {offsets}), std::vector<bool>({true}));

  cursor = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    EXPECT_EQ(std::memcmp(dst.data() + cursor, object.data() + offsets[i], sizes[i]), 0)
        << "range " << i << " restored the wrong bytes";
    cursor += sizes[i];
  }

  // Idempotent for a size already covered, and still refuses to silently shrink
  // a region -- same contract as the pinned path.
  EXPECT_TRUE(caller_->RegisterMemory(dst.data(), dst.size() / 2, mori::io::MemoryLocationType::CPU,
                                      -1, MemoryRegistration::kLocalCopyOnly));
  EXPECT_FALSE(caller_->RegisterMemory(dst.data(), dst.size() * 2,
                                       mori::io::MemoryLocationType::CPU, -1,
                                       MemoryRegistration::kLocalCopyOnly));
  caller_->DeregisterMemory(dst.data());
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

// ---------------------------------------------------------------------------
//  Range-granular remote fetch
// ---------------------------------------------------------------------------

// The arena is the whole evidence: whatever the wire brought back is sitting in
// it, and whatever it did not bring back still holds the poison.  So filling it
// with a known byte and inspecting it afterwards answers both "did only the
// requested ranges cross" and "were they packed the way the copy-out expects"
// in one assertion, with no instrumentation inside PoolClient.
TEST_F(PoolClientRangesTest, RemoteRangedFetchMovesOnlyRequestedBytes) {
  const std::string key = "ranged-only-requested";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 7 + 3) & 0xff);
  SeedRemoteObject(key, object);

  constexpr size_t kSpanA = 900;
  constexpr size_t kSpanB = 700;
  constexpr size_t kOffsetA = 200;
  constexpr size_t kOffsetB = 5000;
  constexpr size_t kPacked = kSpanA + kSpanB;
  constexpr char kPoison = static_cast<char>(0xAA);
  std::fill(caller_get_scratch_.begin(), caller_get_scratch_.end(), kPoison);

  std::vector<char> out_a(kSpanA, 0), out_b(kSpanB, 0);
  std::vector<std::vector<void*>> ptrs = {{out_a.data(), out_b.data()}};
  std::vector<std::vector<size_t>> sizes = {{kSpanA, kSpanB}};
  std::vector<std::vector<size_t>> offsets = {{kOffsetA, kOffsetB}};
  auto got = caller_->BatchGetRanges({key}, ptrs, sizes, offsets);
  ASSERT_EQ(got.size(), 1u);
  ASSERT_TRUE(got[0]);

  EXPECT_EQ(std::memcmp(out_a.data(), object.data() + kOffsetA, kSpanA), 0);
  EXPECT_EQ(std::memcmp(out_b.data(), object.data() + kOffsetB, kSpanB), 0);

  // Packed in caller order, back to back.
  EXPECT_EQ(std::memcmp(caller_get_scratch_.data(), object.data() + kOffsetA, kSpanA), 0);
  EXPECT_EQ(std::memcmp(caller_get_scratch_.data() + kSpanA, object.data() + kOffsetB, kSpanB), 0);
  // ...and nothing else moved.  A whole-object fetch would have overwritten
  // every byte of the arena up to kObjectSize.
  for (size_t i = kPacked; i < caller_get_scratch_.size(); ++i) {
    ASSERT_EQ(caller_get_scratch_[i], kPoison) << "arena byte " << i << " was overwritten";
  }
}

// A range that starts mid-page, crosses into the next, and a range that ends
// exactly at the end of a SHORT last page.  The last-page clamp is the one
// piece of the remote builder's arithmetic that nothing else reaches, and
// getting it wrong reads bytes the object does not own.
TEST_F(PoolClientRangesTest, RemoteRangedFetchCrossesPageBoundary) {
  constexpr size_t kShortObject = kPageSize + 1000;  // last page holds 1000 B
  const std::string key = "ranged-short-tail";
  std::vector<char> object(kShortObject);
  for (size_t i = 0; i < kShortObject; ++i) object[i] = static_cast<char>((i * 11 + 5) & 0xff);
  SeedRemoteObject(key, object);

  constexpr size_t kCrossOffset = kPageSize - 300;  // 300 B in page 0, 500 B in page 1
  constexpr size_t kCrossSize = 800;
  constexpr size_t kTailSize = 400;
  constexpr size_t kTailOffset = kShortObject - kTailSize;  // ends on the last byte

  std::vector<char> cross(kCrossSize, 0), tail(kTailSize, 0);
  std::vector<std::vector<void*>> ptrs = {{cross.data(), tail.data()}};
  std::vector<std::vector<size_t>> sizes = {{kCrossSize, kTailSize}};
  std::vector<std::vector<size_t>> offsets = {{kCrossOffset, kTailOffset}};
  auto got = caller_->BatchGetRanges({key}, ptrs, sizes, offsets);
  ASSERT_EQ(got.size(), 1u);
  ASSERT_TRUE(got[0]);
  EXPECT_EQ(std::memcmp(cross.data(), object.data() + kCrossOffset, kCrossSize), 0);
  EXPECT_EQ(std::memcmp(tail.data(), object.data() + kTailOffset, kTailSize), 0);
}

// What the sglang direct linker actually does: the same key set, once per layer
// group, each call taking a different slice of every object.
TEST_F(PoolClientRangesTest, MultiLayerGroupsOverSameKeys) {
  constexpr size_t kKeys = 3;
  constexpr size_t kGroups = 8;
  constexpr size_t kSliceSize = kObjectSize / kGroups;

  std::vector<std::string> keys;
  std::vector<std::vector<char>> objects;
  for (size_t k = 0; k < kKeys; ++k) {
    keys.push_back("layerwise-" + std::to_string(k));
    std::vector<char> object(kObjectSize);
    for (size_t i = 0; i < kObjectSize; ++i) {
      object[i] = static_cast<char>((i * 3 + k * 101) & 0xff);
    }
    SeedRemoteObject(keys.back(), object);
    objects.push_back(std::move(object));
  }

  std::vector<std::vector<char>> restored(kKeys, std::vector<char>(kObjectSize, 0));
  for (size_t g = 0; g < kGroups; ++g) {
    const size_t slice_offset = g * kSliceSize;
    std::vector<std::vector<void*>> ptrs(kKeys);
    std::vector<std::vector<size_t>> sizes(kKeys, {kSliceSize});
    std::vector<std::vector<size_t>> offsets(kKeys, {slice_offset});
    for (size_t k = 0; k < kKeys; ++k) ptrs[k] = {restored[k].data() + slice_offset};

    auto got = caller_->BatchGetRanges(keys, ptrs, sizes, offsets);
    ASSERT_EQ(got.size(), kKeys);
    for (size_t k = 0; k < kKeys; ++k) ASSERT_TRUE(got[k]) << "group=" << g << " key=" << keys[k];
  }

  // Every group, wherever it was served from, contributed the right bytes.
  for (size_t k = 0; k < kKeys; ++k) {
    EXPECT_EQ(std::memcmp(restored[k].data(), objects[k].data(), kObjectSize), 0)
        << "key=" << keys[k];
  }
}

// Spans totalling more than the arena are split across sub-batches rather than
// failing the key -- the packing unit is a span prefix, not an object.
TEST_F(PoolClientRangesTest, PartialReadSplitsWhenSpansExceedArena) {
  // The spans have to total MORE than the arena to split, which a single object
  // the size of the arena can never do -- so the object is bigger, and the read
  // is deliberately NOT a tiling one.  That keeps holds_whole_object false, so
  // this exercises the split on the arena path rather than the slot path
  // ObjectLargerThanArenaIsServedAcrossSubBatches already covers.
  constexpr size_t kBigObject = 5 * kPageSize;  // 20480, arena is 8192
  const std::string key = "ranged-split";
  std::vector<char> object(kBigObject);
  for (size_t i = 0; i < kBigObject; ++i) object[i] = static_cast<char>((i * 17 + 9) & 0xff);
  SeedRemoteObject(key, object);

  // 3 x 4000 = 12000 > 8192, with gaps between them so they cannot tile.
  constexpr size_t kSpan = 4000;
  const size_t starts[3] = {0, 6000, 13000};
  std::vector<std::vector<char>> out(3, std::vector<char>(kSpan, 0));
  std::vector<std::vector<void*>> ptrs(1);
  std::vector<std::vector<size_t>> sizes(1), offsets(1);
  for (size_t i = 0; i < 3; ++i) {
    ptrs[0].push_back(out[i].data());
    sizes[0].push_back(kSpan);
    offsets[0].push_back(starts[i]);
  }

  auto got = caller_->BatchGetRanges({key}, ptrs, sizes, offsets);
  ASSERT_EQ(got.size(), 1u);
  ASSERT_TRUE(got[0]);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(std::memcmp(out[i].data(), object.data() + starts[i], kSpan), 0) << "span " << i;
  }
}

// One span bigger than the whole arena is the only remaining unservable shape,
// and it must fail that key alone.
// A single range bigger than the whole arena is the one shape that is still
// unservable.  It must fail that key ALONE, and -- the part worth testing --
// it must not leave the key half-delivered: the object here is larger than the
// arena, so the earlier ranges have already been split into their own fetch
// unit by the time the oversized one is seen.  Committing those units would
// fetch part of the key and still report it whole, which the caller has no way
// to detect.
TEST_F(PoolClientRangesTest, OversizedRangeFailsItsKeyWholeAndAlone) {
  constexpr size_t kBigObject = 5 * kPageSize;  // > kScratchSize
  static_assert(kBigObject > kScratchSize, "the split path needs an oversized object");
  const std::string bad_key = "oversize-bad";
  const std::string ok_key = "oversize-good";
  std::vector<char> big(kBigObject), small(kObjectSize);
  for (size_t i = 0; i < kBigObject; ++i) big[i] = static_cast<char>((i * 59 + 4) & 0xff);
  for (size_t i = 0; i < kObjectSize; ++i) small[i] = static_cast<char>((i * 61 + 9) & 0xff);
  SeedRemoteObject(bad_key, big);
  SeedRemoteObject(ok_key, small);

  // 5000 + 5000 forces a split (the arena is 8192), then 10480 cannot be served
  // at all.  Every one of the three has to be refused.
  constexpr char kPoison = static_cast<char>(0x5a);
  std::vector<char> a(5000, kPoison), b(5000, kPoison), c(kBigObject - 10000, kPoison);
  std::vector<char> good(600, 0);
  std::vector<std::vector<void*>> ptrs = {{a.data(), b.data(), c.data()}, {good.data()}};
  std::vector<std::vector<size_t>> sizes = {{a.size(), b.size(), c.size()}, {600}};
  std::vector<std::vector<size_t>> offsets = {{0, 5000, 10000}, {900}};

  auto got = caller_->BatchGetRanges({bad_key, ok_key}, ptrs, sizes, offsets);
  ASSERT_EQ(got.size(), 2u);
  EXPECT_FALSE(got[0]);
  EXPECT_TRUE(got[1]);

  // Nothing partially delivered: the split-off prefix must not have been
  // fetched either.
  EXPECT_EQ(a, std::vector<char>(a.size(), kPoison)) << "first range was delivered anyway";
  EXPECT_EQ(b, std::vector<char>(b.size(), kPoison));
  EXPECT_EQ(c, std::vector<char>(c.size(), kPoison));
  EXPECT_EQ(std::memcmp(good.data(), small.data() + 900, 600), 0);
}

// The same split path when every range IS servable: an object larger than the
// arena, read whole, has to come back correct across several sub-batches.
TEST_F(PoolClientRangesTest, ObjectLargerThanArenaIsServedAcrossSubBatches) {
  constexpr size_t kBigObject = 5 * kPageSize;
  const std::string key = "oversize-object";
  std::vector<char> object(kBigObject);
  for (size_t i = 0; i < kBigObject; ++i) object[i] = static_cast<char>((i * 67 + 5) & 0xff);
  SeedRemoteObject(key, object);

  constexpr size_t kChunk = 4096;
  std::vector<char> out(kBigObject, 0);
  std::vector<std::vector<void*>> ptrs(1);
  std::vector<std::vector<size_t>> sizes(1), offsets(1);
  for (size_t off = 0; off < kBigObject; off += kChunk) {
    const size_t len = std::min(kChunk, kBigObject - off);
    ptrs[0].push_back(out.data() + off);
    sizes[0].push_back(len);
    offsets[0].push_back(off);
  }
  ASSERT_TRUE(caller_->BatchGetRanges({key}, ptrs, sizes, offsets).front());
  EXPECT_EQ(std::memcmp(out.data(), object.data(), kBigObject), 0);
}

// ---------------------------------------------------------------------------
//  Locality prefetch
// ---------------------------------------------------------------------------

TEST_F(PoolClientRangesTest, LocalityPrefetchInstallsWholeObjectAfterRangedRead) {
  const std::string key = "prefetch-on";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 23 + 7) & 0xff);
  SeedRemoteObject(key, object);
  ASSERT_FALSE(LocallyResident(caller_.get(), key));

  std::vector<char> out(512, 0);
  std::vector<std::vector<void*>> ptrs = {{out.data()}};
  std::vector<std::vector<size_t>> sizes = {{512}};
  std::vector<std::vector<size_t>> offsets = {{1024}};
  ASSERT_TRUE(caller_->BatchGetRanges({key}, ptrs, sizes, offsets).front());

  // Asynchronous by design: the read that scheduled it must not have waited.
  ASSERT_TRUE(WaitLocallyResident(caller_.get(), key));

  // And what landed is the WHOLE object, not just the slice that was read.
  std::vector<char> whole(kObjectSize, 0);
  std::vector<std::vector<void*>> whole_ptrs = {{whole.data()}};
  std::vector<std::vector<size_t>> whole_sizes = {{kObjectSize}};
  std::vector<std::vector<size_t>> whole_offsets = {{0}};
  ASSERT_TRUE(caller_->BatchGetRanges({key}, whole_ptrs, whole_sizes, whole_offsets).front());
  EXPECT_EQ(std::memcmp(whole.data(), object.data(), kObjectSize), 0);
}

TEST_F(PoolClientRangesTest, LocalityPrefetchOffLeavesNothingLocal) {
  std::vector<char> no_prefetch_get(kScratchSize), no_prefetch_put(kScratchSize);
  auto quiet = MakeClient("ranges-no-prefetch", kCallerCapacity, no_prefetch_get, no_prefetch_put,
                          /*locality_prefetch=*/false);

  const std::string key = "prefetch-off";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 29 + 11) & 0xff);
  PutLocalReplica(target_.get(), key, object);
  target_->Master().FlushHeartbeat();
  quiet->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitForExists(quiet.get(), key));

  std::vector<char> out(512, 0);
  std::vector<std::vector<void*>> ptrs = {{out.data()}};
  std::vector<std::vector<size_t>> sizes = {{512}};
  std::vector<std::vector<size_t>> offsets = {{1024}};
  ASSERT_TRUE(quiet->BatchGetRanges({key}, ptrs, sizes, offsets).front());
  EXPECT_EQ(std::memcmp(out.data(), object.data() + 1024, 512), 0);

  // Give a prefetch every chance to appear, then assert it did not.
  EXPECT_FALSE(WaitLocallyResident(quiet.get(), key, std::chrono::milliseconds(750)));
  quiet->Shutdown();
}

// Eight layer groups over one cold key must schedule ONE whole-object pull, not
// one per group: the dedup set is what keeps the duplicate traffic bounded at
// 1x the object instead of 8x.
// The switch buys off the SECOND fetch, not locality itself.  A whole-object
// read already has every byte in hand, so landing it in a slot costs nothing on
// the wire and must still happen -- gating it would silently undo what the
// pre-ranged code did on every remote ranged read.
TEST_F(PoolClientRangesTest, WholeObjectReadStillInstallsWithPrefetchOff) {
  std::vector<char> quiet_get(kScratchSize), quiet_put(kScratchSize);
  auto quiet = MakeClient("ranges-quiet-whole", kCallerCapacity, quiet_get, quiet_put,
                          /*locality_prefetch=*/false);

  const std::string key = "prefetch-off-whole";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 83 + 7) & 0xff);
  PutLocalReplica(target_.get(), key, object);
  target_->Master().FlushHeartbeat();
  quiet->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitForExists(quiet.get(), key));
  ASSERT_FALSE(LocallyResident(quiet.get(), key));

  // Four ascending gapless quarters: the ranges tile the object, so the fetch
  // brings all of it and the slot path applies.
  constexpr size_t kQuarter = kObjectSize / 4;
  std::vector<char> out(kObjectSize, 0);
  std::vector<std::vector<void*>> ptrs(1);
  std::vector<std::vector<size_t>> sizes(1), offsets(1);
  for (size_t q = 0; q < 4; ++q) {
    ptrs[0].push_back(out.data() + q * kQuarter);
    sizes[0].push_back(kQuarter);
    offsets[0].push_back(q * kQuarter);
  }
  ASSERT_TRUE(quiet->BatchGetRanges({key}, ptrs, sizes, offsets).front());
  EXPECT_EQ(std::memcmp(out.data(), object.data(), kObjectSize), 0);

  // Synchronous: the slot is committed before the call returns, so no waiting.
  EXPECT_TRUE(LocallyResident(quiet.get(), key))
      << "a switch named for the background pull must not disable the free install";
  quiet->Shutdown();
}

// The sglang shape against a cold key: eight layer groups, each naming the same
// key, with the object only ever arriving from the peer.  Every group must
// return its own bytes and the key must end up local.
//
// This does NOT pin the in-flight dedup (prefetch_inflight_).  Dedup has no
// external observable -- a duplicate pull is absorbed by the worker's
// BatchResolve check and by kSuccessAlreadyExists, so removing the set changes
// nothing a caller can see.  Naming it here would claim coverage that does not
// exist; it is a throughput guard, and the bench is where it shows up.
TEST_F(PoolClientRangesTest, LayerWiseGroupsOverOneColdKeyAllReturnTheirBytes) {
  const std::string key = "prefetch-dedup";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 31 + 13) & 0xff);
  SeedRemoteObject(key, object);

  constexpr size_t kGroups = 8;
  constexpr size_t kSliceSize = kObjectSize / kGroups;
  std::vector<char> restored(kObjectSize, 0);
  for (size_t g = 0; g < kGroups; ++g) {
    const size_t slice_offset = g * kSliceSize;
    std::vector<std::vector<void*>> ptrs = {{restored.data() + slice_offset}};
    std::vector<std::vector<size_t>> sizes = {{kSliceSize}};
    std::vector<std::vector<size_t>> offsets = {{slice_offset}};
    ASSERT_TRUE(caller_->BatchGetRanges({key}, ptrs, sizes, offsets).front()) << "group=" << g;
  }
  EXPECT_EQ(std::memcmp(restored.data(), object.data(), kObjectSize), 0);

  ASSERT_TRUE(WaitLocallyResident(caller_.get(), key));
  // A second pull would have had to allocate a second slot for the same key;
  // the medium reports one, and the bytes are intact.
  std::vector<char> whole(kObjectSize, 0);
  std::vector<std::vector<void*>> whole_ptrs = {{whole.data()}};
  std::vector<std::vector<size_t>> whole_sizes = {{kObjectSize}};
  std::vector<std::vector<size_t>> whole_offsets = {{0}};
  ASSERT_TRUE(caller_->BatchGetRanges({key}, whole_ptrs, whole_sizes, whole_offsets).front());
  EXPECT_EQ(std::memcmp(whole.data(), object.data(), kObjectSize), 0);
}

// When the caller's ranges tile the object in order, the arena slice IS the
// object, so locality costs one local copy and nothing on the wire.  This is
// the whole-object reader's shape (one call naming every layer).
TEST_F(PoolClientRangesTest, CompleteArenaObjectIsInstalledWithoutRefetching) {
  const std::string key = "arena-complete";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 47 + 3) & 0xff);
  SeedRemoteObject(key, object);
  ASSERT_FALSE(LocallyResident(caller_.get(), key));

  // Four ascending, gapless ranges covering [0, kObjectSize).
  constexpr size_t kQuarter = kObjectSize / 4;
  std::vector<char> out(kObjectSize, 0);
  std::vector<std::vector<void*>> ptrs(1);
  std::vector<std::vector<size_t>> sizes(1), offsets(1);
  for (size_t q = 0; q < 4; ++q) {
    ptrs[0].push_back(out.data() + q * kQuarter);
    sizes[0].push_back(kQuarter);
    offsets[0].push_back(q * kQuarter);
  }
  ASSERT_TRUE(caller_->BatchGetRanges({key}, ptrs, sizes, offsets).front());
  EXPECT_EQ(std::memcmp(out.data(), object.data(), kObjectSize), 0);

  // Installed from the arena, synchronously -- no waiting for a worker, and no
  // second trip to the peer.
  EXPECT_TRUE(LocallyResident(caller_.get(), key));

  std::vector<char> again(kObjectSize, 0);
  std::vector<std::vector<void*>> again_ptrs = {{again.data()}};
  std::vector<std::vector<size_t>> again_sizes = {{kObjectSize}};
  std::vector<std::vector<size_t>> again_offsets = {{0}};
  ASSERT_TRUE(caller_->BatchGetRanges({key}, again_ptrs, again_sizes, again_offsets).front());
  EXPECT_EQ(std::memcmp(again.data(), object.data(), kObjectSize), 0);
}

// A whole-object read does not need the arena at all: the slot it is going to
// be cached in has the same layout, so the peer writes straight into it.  The
// arena is the witness -- poison it, and if a single byte changed the read went
// the long way round.
TEST_F(PoolClientRangesTest, WholeObjectReadBypassesTheArena) {
  const std::string key = "slot-direct";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 71 + 13) & 0xff);
  SeedRemoteObject(key, object);
  ASSERT_FALSE(LocallyResident(caller_.get(), key));

  constexpr char kPoison = static_cast<char>(0xC3);
  std::fill(caller_get_scratch_.begin(), caller_get_scratch_.end(), kPoison);

  constexpr size_t kQuarter = kObjectSize / 4;
  std::vector<char> out(kObjectSize, 0);
  std::vector<std::vector<void*>> ptrs(1);
  std::vector<std::vector<size_t>> sizes(1), offsets(1);
  for (size_t q = 0; q < 4; ++q) {
    ptrs[0].push_back(out.data() + q * kQuarter);
    sizes[0].push_back(kQuarter);
    offsets[0].push_back(q * kQuarter);
  }
  ASSERT_TRUE(caller_->BatchGetRanges({key}, ptrs, sizes, offsets).front());

  EXPECT_EQ(std::memcmp(out.data(), object.data(), kObjectSize), 0);
  EXPECT_TRUE(LocallyResident(caller_.get(), key)) << "committing the slot IS the install";
  EXPECT_EQ(caller_get_scratch_, std::vector<char>(kScratchSize, kPoison))
      << "the arena was used for a read that did not need it";
}

// ...and when the medium cannot give it a slot, the same read still has to
// work.  The fallback is the arena, decided on the failed allocation rather
// than after a failed transfer.
TEST_F(PoolClientRangesTest, WholeObjectReadFallsBackToArenaWithoutASlot) {
  // One page of DRAM, objects are two: BatchAllocate can never satisfy one.
  std::vector<char> tiny_get(kScratchSize), tiny_put(kScratchSize);
  auto cramped = MakeClient("ranges-cramped", kPageSize, tiny_get, tiny_put);

  const std::string key = "slot-unavailable";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 73 + 19) & 0xff);
  PutLocalReplica(target_.get(), key, object);
  target_->Master().FlushHeartbeat();
  cramped->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitForExists(cramped.get(), key));

  std::vector<char> out(kObjectSize, 0);
  std::vector<std::vector<void*>> ptrs = {{out.data()}};
  std::vector<std::vector<size_t>> sizes = {{kObjectSize}};
  std::vector<std::vector<size_t>> offsets = {{0}};
  ASSERT_TRUE(cramped->BatchGetRanges({key}, ptrs, sizes, offsets).front())
      << "no slot must not mean no read";
  EXPECT_EQ(std::memcmp(out.data(), object.data(), kObjectSize), 0);
  EXPECT_FALSE(LocallyResident(cramped.get(), key)) << "nowhere to cache it, and that is fine";
  cramped->Shutdown();
}

// Ranges that cover every byte but arrive out of order do NOT make the arena
// equal the object -- it is packed in caller order -- so this must fall back to
// the background pull rather than install a scrambled slice.
TEST_F(PoolClientRangesTest, OutOfOrderFullCoverageDoesNotInstallFromArena) {
  const std::string key = "arena-scrambled";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 53 + 17) & 0xff);
  SeedRemoteObject(key, object);

  constexpr size_t kHalf = kObjectSize / 2;
  std::vector<char> out(kObjectSize, 0);
  // Second half named first.
  std::vector<std::vector<void*>> ptrs = {{out.data() + kHalf, out.data()}};
  std::vector<std::vector<size_t>> sizes = {{kHalf, kHalf}};
  std::vector<std::vector<size_t>> offsets = {{kHalf, 0}};
  ASSERT_TRUE(caller_->BatchGetRanges({key}, ptrs, sizes, offsets).front());
  EXPECT_EQ(std::memcmp(out.data(), object.data(), kObjectSize), 0);

  // It still becomes local, but via the pull -- so what lands must be the real
  // object and not the arena's caller-order layout.
  ASSERT_TRUE(WaitLocallyResident(caller_.get(), key));
  std::vector<char> again(kObjectSize, 0);
  std::vector<std::vector<void*>> again_ptrs = {{again.data()}};
  std::vector<std::vector<size_t>> again_sizes = {{kObjectSize}};
  std::vector<std::vector<size_t>> again_offsets = {{0}};
  ASSERT_TRUE(caller_->BatchGetRanges({key}, again_ptrs, again_sizes, again_offsets).front());
  EXPECT_EQ(std::memcmp(again.data(), object.data(), kObjectSize), 0);
}

TEST_F(PoolClientRangesTest, MixedLocalAndRemoteRangedKeys) {
  const std::string local_key = "mixed-local";
  const std::string remote_key = "mixed-remote";
  std::vector<char> local_object(kObjectSize), remote_object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) {
    local_object[i] = static_cast<char>((i * 37 + 2) & 0xff);
    remote_object[i] = static_cast<char>((i * 41 + 6) & 0xff);
  }
  PutLocalReplica(caller_.get(), local_key, local_object);
  SeedRemoteObject(remote_key, remote_object);

  std::vector<char> out_local(600, 0), out_remote(600, 0);
  std::vector<std::vector<void*>> ptrs = {{out_local.data()}, {out_remote.data()}};
  std::vector<std::vector<size_t>> sizes = {{600}, {600}};
  std::vector<std::vector<size_t>> offsets = {{300}, {4500}};

  auto got = caller_->BatchGetRanges({local_key, remote_key}, ptrs, sizes, offsets);
  ASSERT_EQ(got.size(), 2u);
  EXPECT_TRUE(got[0]);
  EXPECT_TRUE(got[1]);
  EXPECT_EQ(std::memcmp(out_local.data(), local_object.data() + 300, 600), 0);
  EXPECT_EQ(std::memcmp(out_remote.data(), remote_object.data() + 4500, 600), 0);
}

// A key whose ranges do not fit its object must fail alone, and must not
// disturb a healthy sibling in the same call.
TEST_F(PoolClientRangesTest, PartialFailureAttributionAcrossKeys) {
  const std::string bad_key = "attrib-bad";
  const std::string good_key = "attrib-good";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 43 + 8) & 0xff);
  SeedRemoteObject(bad_key, object);
  SeedRemoteObject(good_key, object);

  std::vector<char> bad_out(600, 0x5a), good_out(600, 0);
  std::vector<std::vector<void*>> ptrs = {{bad_out.data()}, {good_out.data()}};
  std::vector<std::vector<size_t>> sizes = {{600}, {600}};
  // Runs 300 B past the end of the object.
  std::vector<std::vector<size_t>> offsets = {{kObjectSize - 300}, {1200}};

  auto got = caller_->BatchGetRanges({bad_key, good_key}, ptrs, sizes, offsets);
  ASSERT_EQ(got.size(), 2u);
  EXPECT_FALSE(got[0]);
  EXPECT_TRUE(got[1]);
  EXPECT_EQ(std::memcmp(good_out.data(), object.data() + 1200, 600), 0);
  EXPECT_EQ(bad_out, std::vector<char>(600, 0x5a)) << "failed key's buffer was written";
}

// Everything above reads one or two keys from one thread.  The paths this
// change added only diverge under load: which keys are local when a call
// starts, whether a slot can be had, whether the background pull lands before
// the next group asks -- all of it decided by a race.  So drive every shape at
// once over an overlapping key set and check the bytes afterwards.
//
// The caller's medium deliberately cannot hold all the keys, so slot
// allocation fails part of the time and the arena fallback is exercised too.
TEST_F(PoolClientRangesTest, ConcurrentMixedShapesAllReturnCorrectBytes) {
  constexpr size_t kKeys = 12;  // > kCallerCapacity / kObjectSize, so slots run out
  constexpr size_t kThreads = 4;
  constexpr size_t kRounds = 3;
  constexpr size_t kGroups = 8;
  constexpr size_t kSlice = kObjectSize / kGroups;

  std::vector<std::string> keys;
  std::vector<std::vector<char>> objects;
  for (size_t k = 0; k < kKeys; ++k) {
    keys.push_back("stress-" + std::to_string(k));
    std::vector<char> object(kObjectSize);
    for (size_t i = 0; i < kObjectSize; ++i) {
      object[i] = static_cast<char>((i * 97 + k * 211 + 5) & 0xff);
    }
    SeedRemoteObject(keys.back(), object);
    objects.push_back(std::move(object));
  }

  std::atomic<size_t> mismatches{0};
  std::atomic<size_t> failures{0};
  std::vector<std::thread> threads;
  for (size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      // Half the threads read layer-wise (a slice per call, never the whole
      // object); half read the whole object in one call (the slot path).  Both
      // touch the same keys.
      const bool layer_wise = (t % 2) == 0;
      std::vector<std::vector<char>> restored(kKeys, std::vector<char>(kObjectSize, 0));
      std::vector<char> served(kKeys, 1);
      for (size_t round = 0; round < kRounds; ++round) {
        for (auto& buffer : restored) std::fill(buffer.begin(), buffer.end(), 0);
        std::fill(served.begin(), served.end(), 1);

        if (layer_wise) {
          for (size_t g = 0; g < kGroups; ++g) {
            const size_t offset = g * kSlice;
            std::vector<std::vector<void*>> ptrs(kKeys);
            std::vector<std::vector<size_t>> sizes(kKeys, {kSlice});
            std::vector<std::vector<size_t>> offsets(kKeys, {offset});
            for (size_t k = 0; k < kKeys; ++k) ptrs[k] = {restored[k].data() + offset};
            const auto got = caller_->BatchGetRanges(keys, ptrs, sizes, offsets);
            for (size_t k = 0; k < kKeys; ++k) {
              if (!got[k]) {
                served[k] = false;
                failures.fetch_add(1, std::memory_order_relaxed);
              }
            }
          }
        } else {
          std::vector<std::vector<void*>> ptrs(kKeys);
          std::vector<std::vector<size_t>> sizes(kKeys), offsets(kKeys);
          for (size_t k = 0; k < kKeys; ++k) {
            for (size_t g = 0; g < kGroups; ++g) {
              ptrs[k].push_back(restored[k].data() + g * kSlice);
              sizes[k].push_back(kSlice);
              offsets[k].push_back(g * kSlice);
            }
          }
          const auto got = caller_->BatchGetRanges(keys, ptrs, sizes, offsets);
          for (size_t k = 0; k < kKeys; ++k) {
            if (!got[k]) {
              served[k] = false;
              failures.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }

        // Only a key the client SAID it served is checked.  A refused read
        // leaves its buffer untouched, and counting that as corruption would
        // turn every capacity refusal into a false alarm.
        for (size_t k = 0; k < kKeys; ++k) {
          if (!served[k]) continue;
          if (std::memcmp(restored[k].data(), objects[k].data(), kObjectSize) != 0) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    });
  }
  for (auto& thread : threads) thread.join();

  // A read may legitimately fail here -- the medium is intentionally too small,
  // so a slot or an arena round can be refused -- but it must never return the
  // wrong bytes.
  EXPECT_EQ(mismatches.load(), 0u) << "a caller buffer came back wrong";
  // Failures are reported, not asserted away: the medium really is too small
  // here, so a refusal is a legitimate outcome and pinning it to zero would
  // make this test fail for the one reason it is allowed to.
  if (failures.load() != 0) {
    GTEST_LOG_(INFO) << failures.load()
                     << " reads were refused (capacity), none returned bad bytes";
  }
}

}  // namespace
}  // namespace mori::umbp

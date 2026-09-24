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

#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "mori/application/utils/cpu_affinity.hpp"
#include "umbp/distributed/peer/backend/mock_backend.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {
constexpr size_t kPage = 4096;

PoolClientConfig ReplicaConfig(size_t pages = 128) {
  PoolClientConfig cfg;
  cfg.numa_replication = true;
  cfg.master_config.node_id = "numa-replication";
  cfg.io_engine.host.clear();
  cfg.dram_page_size = kPage;
  cfg.dram.buffer_sizes = {pages * kPage, pages * kPage};
  cfg.dram.numa_nodes = {0, 1};
  cfg.dram.prefault = false;
  cfg.local_evict_high_watermark = 1;
  cfg.local_evict_low_watermark = 1;
  return cfg;
}

uint64_t UsedBytes(MediumBackend* backend) {
  const auto capacity = backend->Capacity();
  return capacity.total_bytes - capacity.available_bytes;
}

std::vector<char> StoredBytes(MediumBackend* backend, const std::string& key) {
  const auto stored = backend->BatchResolve({key}, false).front();
  EXPECT_TRUE(stored.found);
  std::vector<char> bytes(stored.size);
  size_t offset = 0;
  for (const auto& page : stored.pages) {
    const auto ref = backend->BufferRef(page.buffer_index);
    const size_t size = std::min<uint64_t>(stored.page_size, stored.size - offset);
    std::memcpy(bytes.data() + offset,
                static_cast<const char*>(ref.host_ptr) + page.page_index * stored.page_size, size);
    offset += size;
  }
  return bytes;
}

TEST(NumaReplication, WholeAndRangedWritesShareKeysAndPreservePartialReads) {
  PoolClient client(ReplicaConfig());
  ASSERT_TRUE(client.Init());
  auto* backend = client.Backends().Get(TierType::DRAM);
  std::vector<char> expected(2 * kPage + 31);
  for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<char>(i * 37 + 19);
  ASSERT_TRUE(client.Put("whole", expected.data(), expected.size()));
  ASSERT_EQ(client.BatchPutRanges({"ranged"}, {expected.size()},
                                  {{expected.data(), expected.data() + 113}},
                                  {{113, expected.size() - 113}}, {{0, 113}}),
            std::vector<bool>{true});
  for (const std::string key : {"whole", "ranged"}) {
    for (int replica = 0; replica < 2; ++replica) {
      const auto physical = key + "#n" + std::to_string(replica);
      EXPECT_EQ(StoredBytes(backend, physical), expected);
      const auto resolved = backend->BatchResolve({physical}, false).front();
      for (const auto& page : resolved.pages) EXPECT_EQ(page.buffer_index, replica);
    }
  }
  std::vector<char> a(19, '\x55'), b(47, '\x55');
  ASSERT_EQ(client.BatchGetRanges({"ranged"}, {{a.data(), b.data()}}, {{a.size(), b.size()}},
                                  {{7, kPage - 8}}),
            std::vector<bool>{true});
  EXPECT_EQ(a, std::vector<char>(expected.begin() + 7, expected.begin() + 26));
  EXPECT_EQ(b, std::vector<char>(expected.begin() + kPage - 8, expected.begin() + kPage + 39));
  // A client key that looks like a physical key remains a different object.
  std::vector<char> other(expected.size(), '\x23');
  ASSERT_TRUE(client.Put("whole#n0", other.data(), other.size()));
  EXPECT_EQ(StoredBytes(backend, "whole#n0#n0"), other);
  EXPECT_EQ(StoredBytes(backend, "whole#n0"), expected);
}

TEST(NumaReplication, SurvivingReplicaRemainsAuthoritativeAndReadable) {
  auto cfg = ReplicaConfig();
  cfg.local_first = false;  // no-master mode must still resolve local replicas
  PoolClient client(cfg);
  ASSERT_TRUE(client.Init());
  auto* backend = client.Backends().Get(TierType::DRAM);
  std::vector<char> original(kPage, '\x31'), replacement(kPage, '\x72'), read(kPage, '\x55');
  ASSERT_TRUE(client.Put("key", original.data(), original.size()));
  ASSERT_EQ(backend->Evict({"key#n0"}).front().bytes_freed, kPage);
  const auto used = UsedBytes(backend);
  ASSERT_EQ(client.BatchExists({"key", "missing"}), (std::vector<bool>{true, false}));
  ASSERT_TRUE(client.Put("key", replacement.data(), replacement.size()));
  EXPECT_FALSE(backend->Contains("key#n0"));
  EXPECT_EQ(UsedBytes(backend), used);
  ASSERT_TRUE(client.Get("key", read.data(), read.size()));
  EXPECT_EQ(read, original);
  ASSERT_TRUE(client.Clear());
  EXPECT_FALSE(client.Exists("key"));
}

TEST(NumaReplication, PartialAllocationFailureReleasesBothReservations) {
  PoolClient client(ReplicaConfig(2));
  ASSERT_TRUE(client.Init());
  auto* backend = client.Backends().Get(TierType::DRAM);
  std::vector<char> bytes(3 * kPage, '\x31');
  EXPECT_FALSE(client.Put("too-large", bytes.data(), bytes.size()));
  EXPECT_EQ(UsedBytes(backend), 0u);
  EXPECT_FALSE(client.Exists("too-large"));
  // Reusing the same key detects leaked pending-key reservations as well as pages.
  EXPECT_TRUE(client.Put("too-large", bytes.data(), kPage));
  EXPECT_EQ(UsedBytes(backend), 2 * kPage);
}

TEST(NumaReplication, ConcurrentDifferentInputsNeverSplitReplicaContents) {
  PoolClient client(ReplicaConfig());
  ASSERT_TRUE(client.Init());
  constexpr size_t count = 32;
  std::vector<std::string> keys;
  for (size_t i = 0; i < count; ++i) keys.push_back("race-" + std::to_string(i));
  std::vector<char> a(kPage, '\x31'), b(kPage, '\x72');
  std::vector<size_t> sizes(count, kPage);
  std::promise<void> go;
  auto start = go.get_future().share();
  auto put = [&](const std::vector<char>& bytes) {
    start.wait();
    return client.BatchPut(keys, std::vector<const void*>(count, bytes.data()), sizes);
  };
  auto first = std::async(std::launch::async, [&] { return put(a); });
  auto second = std::async(std::launch::async, [&] { return put(b); });
  go.set_value();
  auto first_result = first.get(), second_result = second.get();
  auto* backend = client.Backends().Get(TierType::DRAM);
  for (size_t i = 0; i < count; ++i) {
    ASSERT_TRUE(first_result[i] || second_result[i]);
    auto zero = StoredBytes(backend, keys[i] + "#n0");
    EXPECT_EQ(zero, StoredBytes(backend, keys[i] + "#n1"));
    EXPECT_TRUE(zero == a || zero == b);
  }
}

TEST(NumaReplication, InvalidConfigurationsFailBeforeCreatingBackends) {
  auto cfg = ReplicaConfig();
  cfg.master_config.master_address = "127.0.0.1:1";
  EXPECT_FALSE(PoolClient(cfg).Init());
  cfg.master_config.master_address.clear();
  cfg.dram.numa_nodes = {0};
  EXPECT_FALSE(PoolClient(cfg).Init());
  cfg = ReplicaConfig();
  cfg.medium = TierType::HBM;
  EXPECT_FALSE(PoolClient(cfg).Init());
}

TEST(NumaReplication, PairDedupAndPendingChecksDoNotAllocateTheOtherMember) {
  class CountingBackend : public MockBackend {
   public:
    CountingBackend() : MockBackend(TierType::DRAM) {}
    size_t allocations = 0;
    std::vector<AllocateResult> BatchAllocate(
        const std::vector<AllocateRequest>& entries) override {
      allocations += entries.size();
      return MockBackend::BatchAllocate(entries);
    }
  };
  BackendRegistry registry;
  auto owned = std::make_unique<CountingBackend>();
  auto* backend = owned.get();
  ASSERT_TRUE(registry.Register("dram", std::move(owned)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  PoolPlacementRequest zero{"K#n0", 64, TierType::DRAM}, one{"K#n1", 64, TierType::DRAM};
  auto first = pool.BatchAllocate({one}).front();
  ASSERT_TRUE(pool.BatchCommit({{{first.backend_id, first.allocation.slot_id}, one.key}})
                  .front()
                  .commit.success);
  zero.paired_request = 1;
  one.paired_request = 0;
  backend->allocations = 0;
  auto exists = pool.BatchAllocate({zero, one});
  EXPECT_EQ(exists[0].allocation.outcome, AllocateOutcome::kSuccessAlreadyExists);
  EXPECT_EQ(exists[1].allocation.outcome, AllocateOutcome::kSuccessAlreadyExists);
  EXPECT_EQ(backend->allocations, 0u);
  pool.Evict({one.key}, PoolEvictMode::kReclaim);
  auto pending = pool.BatchAllocate({zero, one});
  ASSERT_EQ(backend->allocations, 2u);
  auto busy = pool.BatchAllocate({zero, one});
  EXPECT_EQ(busy[0].allocation.outcome, AllocateOutcome::kFailed);
  EXPECT_EQ(busy[1].allocation.outcome, AllocateOutcome::kFailed);
  EXPECT_EQ(backend->allocations, 2u);
  pool.BatchAbort({{pending[0].backend_id, pending[0].allocation.slot_id},
                   {pending[1].backend_id, pending[1].allocation.slot_id}});
}

TEST(NumaReplicationGpu, WritesFromEitherSocketAndReadsBothReplicas) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) GTEST_SKIP() << "requires GPUs";
  std::vector<int> devices, nodes;
  for (int device = 0; device < count && devices.size() < 2; ++device) {
    int node = -1;
    mori::application::detail::GpuLocalCpuList(device, node);
    if (node >= 0 && (nodes.empty() || node != nodes.front())) {
      devices.push_back(device);
      nodes.push_back(node);
    }
  }
  if (devices.size() != 2) GTEST_SKIP() << "requires GPUs on two NUMA nodes";
  auto cfg = ReplicaConfig();
  cfg.dram.numa_nodes = nodes;
  PoolClient client(cfg);
  ASSERT_TRUE(client.Init());
  for (size_t writer = 0; writer < 2; ++writer) {
    std::vector<char> expected(2 * kPage + 17, static_cast<char>(0x31 + writer));
    ASSERT_EQ(hipSetDevice(devices[writer]), hipSuccess);
    void* gpu = nullptr;
    ASSERT_EQ(hipMalloc(&gpu, expected.size()), hipSuccess);
    std::unique_ptr<void, decltype(&hipFree)> buffer(gpu, hipFree);
    ASSERT_TRUE(client.RegisterMemory(gpu, expected.size(), mori::io::MemoryLocationType::GPU,
                                      devices[writer], MemoryRegistration::kLocalCopyOnly));
    ASSERT_EQ(hipMemcpy(gpu, expected.data(), expected.size(), hipMemcpyHostToDevice), hipSuccess);
    for (size_t reader = 0; reader < 2; ++reader) {
      const auto key = "gpu-" + std::to_string(writer) + "-" + std::to_string(reader);
      ASSERT_EQ(hipSetDevice(devices[writer]), hipSuccess);
      if (reader == 0) {
        ASSERT_TRUE(client.Put(key, gpu, expected.size()));
      } else {
        ASSERT_EQ(
            client.BatchPutRanges({key}, {expected.size()}, {{gpu}}, {{expected.size()}}, {{0}}),
            std::vector<bool>{true});
      }
      ASSERT_EQ(hipSetDevice(devices[reader]), hipSuccess);
      void* dst = nullptr;
      ASSERT_EQ(hipMalloc(&dst, expected.size()), hipSuccess);
      std::unique_ptr<void, decltype(&hipFree)> target(dst, hipFree);
      ASSERT_TRUE(client.RegisterMemory(dst, expected.size(), mori::io::MemoryLocationType::GPU,
                                        devices[reader], MemoryRegistration::kLocalCopyOnly));
      ASSERT_EQ(hipMemset(dst, 0x55, expected.size()), hipSuccess);
      ASSERT_EQ(client.BatchGetRanges({key}, {{dst}}, {{expected.size()}}, {{0}}),
                std::vector<bool>{true});
      std::vector<char> actual(expected.size());
      ASSERT_EQ(hipMemcpy(actual.data(), dst, actual.size(), hipMemcpyDeviceToHost), hipSuccess);
      EXPECT_EQ(actual, expected);
      client.DeregisterMemory(dst);
      // Only the selected replica acquires a read lease. This distinguishes
      // NUMA-local selection from always reading replica zero with correct bytes.
      auto* backend = client.Backends().Get(TierType::DRAM);
      EXPECT_EQ(backend->Evict({key + "#n" + std::to_string(nodes[reader])}).front().bytes_freed,
                0u);
      EXPECT_EQ(
          backend->Evict({key + "#n" + std::to_string(nodes[1 - reader])}).front().bytes_freed,
          expected.size());
    }
    client.DeregisterMemory(gpu);
    ASSERT_EQ(hipSetDevice(devices[writer]), hipSuccess);
  }
}
}  // namespace
}  // namespace mori::umbp

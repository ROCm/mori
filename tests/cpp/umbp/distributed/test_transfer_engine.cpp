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

// Phase 6 transfer layer, without gRPC or RDMA: every engine's planner,
// LocalCopyEngine's executor, and CompositeTransferEngine's routing.
// MoriIoEngine's transfers need a fabric: see test_cross_node_smoke.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/aligned_buffer.h"
#include "umbp/common/device_gather.h"
#include "umbp/common/host_registration.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/distributed/transfer/posix_engine.h"
#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {
namespace {

TransferItem MakeItem(const TransferRef& src, uint64_t src_off, const TransferRef& dst,
                      uint64_t dst_off, uint64_t size, size_t tag) {
  TransferItem item;
  item.src = src;
  item.src_offset = src_off;
  item.dst = dst;
  item.dst_offset = dst_off;
  item.size = size;
  item.tag = tag;
  return item;
}

// A ref that looks like a peer's memory: reachable only through mori-io, with
// an engine key that is not ours.
TransferRef RemoteRef(const std::string& engine_key, mori::io::MemoryUniqueId id, uint64_t size) {
  mori::io::MemoryDesc desc;
  desc.engineKey = engine_key;
  desc.id = id;
  desc.size = size;
  desc.loc = mori::io::MemoryLocationType::CPU;
  return TransferRef::Remote(std::move(desc));
}

// TransferPlanBuilder's rules: every non-empty item is rejected or planned, and
// the plans carry exactly the accepted bytes.
void ExpectPlanContract(const std::vector<TransferItem>& items, const TransferPlanSet& planned) {
  const std::set<size_t> rejected(planned.rejected_tags.begin(), planned.rejected_tags.end());
  std::set<size_t> carried;
  uint64_t planned_bytes = 0;
  for (const auto& plan : planned.plans) {
    carried.insert(plan.tags.begin(), plan.tags.end());
    planned_bytes += std::accumulate(plan.sizes.begin(), plan.sizes.end(), uint64_t{0});
  }
  uint64_t accepted_bytes = 0;
  for (const auto& item : items) {
    if (item.size == 0) continue;
    EXPECT_TRUE(rejected.count(item.tag) != 0 || carried.count(item.tag) != 0)
        << "item tag " << item.tag << " is neither rejected nor planned";
    if (rejected.count(item.tag) == 0) accepted_bytes += item.size;
  }
  EXPECT_EQ(accepted_bytes, planned_bytes);
}

// ---------------------------------------------------------------------------
//  LocalCopyEngine
// ---------------------------------------------------------------------------

TEST(LocalCopyEngine, CopiesEveryScatteredPage) {
  LocalCopyEngine engine;
  std::vector<char> src(4096), dst(4096, 0);
  std::iota(src.begin(), src.end(), 1);

  TransferRef s =
      engine.RegisterMemory(src.data(), src.size(), mori::io::MemoryLocationType::CPU, -1);
  TransferRef d =
      engine.RegisterMemory(dst.data(), dst.size(), mori::io::MemoryLocationType::CPU, -1);

  // Deliberately out of order and non-adjacent, so nothing coalesces.
  std::vector<TransferItem> items{
      MakeItem(s, 2048, d, 0, 1024, 0),
      MakeItem(s, 0, d, 2048, 1024, 1),
  };
  std::vector<size_t> failed;
  ASSERT_TRUE(engine.Transfer(items, &failed));
  EXPECT_TRUE(failed.empty());
  EXPECT_EQ(0, std::memcmp(dst.data(), src.data() + 2048, 1024));
  EXPECT_EQ(0, std::memcmp(dst.data() + 2048, src.data(), 1024));
}

TEST(LocalCopyEngine, CoalescesContiguousSegmentsIntoOnePlan) {
  LocalCopyEngine engine;
  std::vector<char> src(4096), dst(4096, 0);
  std::iota(src.begin(), src.end(), 7);
  TransferRef s = TransferRef::HostBytes(src.data(), src.size());
  TransferRef d = TransferRef::HostBytes(dst.data(), dst.size());

  // Four adjacent pages, same buffer pair: one plan, one segment.
  std::vector<TransferItem> items;
  for (size_t i = 0; i < 4; ++i) {
    items.push_back(MakeItem(s, i * 1024, d, i * 1024, 1024, i));
  }
  TransferPlanSet planned = engine.Plan(items);
  ASSERT_EQ(1u, planned.plans.size());
  EXPECT_TRUE(planned.rejected_tags.empty());
  EXPECT_EQ(1u, planned.plans[0].sizes.size());
  EXPECT_EQ(4096u, planned.plans[0].sizes[0]);
  // Every contributing key is still named, so a failure maps back to all four.
  EXPECT_EQ(std::vector<size_t>({0, 1, 2, 3}), planned.plans[0].tags);

  auto handle = engine.Submit(std::move(planned.plans));
  ASSERT_NE(nullptr, handle);
  std::vector<TransferFailure> failures;
  handle->Wait(&failures);
  EXPECT_TRUE(failures.empty());
  EXPECT_EQ(src, dst);
}

TEST(LocalCopyEngine, SeparatePlanPerBufferPair) {
  LocalCopyEngine engine;
  std::vector<char> a(1024, 'a'), b(1024, 'b'), out1(1024, 0), out2(1024, 0);
  TransferRef ra = TransferRef::HostBytes(a.data(), a.size());
  TransferRef rb = TransferRef::HostBytes(b.data(), b.size());
  TransferRef r1 = TransferRef::HostBytes(out1.data(), out1.size());
  TransferRef r2 = TransferRef::HostBytes(out2.data(), out2.size());

  std::vector<TransferItem> items{MakeItem(ra, 0, r1, 0, 1024, 0), MakeItem(rb, 0, r2, 0, 1024, 1)};
  TransferPlanSet planned = engine.Plan(items);
  EXPECT_EQ(2u, planned.plans.size());
}

TEST(LocalCopyEngine, RejectsOutOfBoundsWithoutCopying) {
  LocalCopyEngine engine;
  std::vector<char> src(1024, 'x'), dst(1024, 0);
  TransferRef s = TransferRef::HostBytes(src.data(), src.size());
  TransferRef d = TransferRef::HostBytes(dst.data(), dst.size());

  std::vector<TransferItem> items{
      MakeItem(s, 0, d, 0, 512, 0),      // fine
      MakeItem(s, 512, d, 512, 1024, 1)  // runs past the end of both
  };
  TransferPlanSet planned = engine.Plan(items);
  EXPECT_EQ(std::vector<size_t>({1}), planned.rejected_tags);
  ASSERT_EQ(1u, planned.plans.size());
  EXPECT_EQ(std::vector<size_t>({0}), planned.plans[0].tags);

  // Transfer() surfaces the rejection as a failure without touching the bytes
  // the rejected item would have written.
  std::vector<size_t> failed;
  EXPECT_FALSE(engine.Transfer(items, &failed));
  EXPECT_EQ(std::vector<size_t>({1}), failed);
  for (size_t i = 512; i < dst.size(); ++i) EXPECT_EQ(0, dst[i]) << "byte " << i;
}

TEST(LocalCopyEngine, RefusesAnEndpointItCannotAddress) {
  LocalCopyEngine engine;
  std::vector<char> local(1024, 'x');
  TransferRef host = TransferRef::HostBytes(local.data(), local.size());
  TransferRef remote = RemoteRef("some-peer", /*id=*/7, 1024);

  EXPECT_TRUE(engine.CanHandle(host, host));
  EXPECT_FALSE(engine.CanHandle(host, remote));
  EXPECT_FALSE(engine.CanHandle(remote, host));
  EXPECT_FALSE(engine.CanHandle(remote, remote));
}

// Interleaved pairs still merge within their own plans; empty items are dropped.
TEST(LocalCopyEngine, MergesWithinEachPairWhenPairsInterleave) {
  LocalCopyEngine engine;
  std::vector<char> a(256, 'a'), b(256, 'b'), out1(256, 0), out2(256, 0);
  auto ra = TransferRef::HostBytes(a.data(), a.size());
  auto rb = TransferRef::HostBytes(b.data(), b.size());
  auto r1 = TransferRef::HostBytes(out1.data(), out1.size());
  auto r2 = TransferRef::HostBytes(out2.data(), out2.size());
  const std::vector<TransferItem> items{
      MakeItem(ra, 0, r1, 0, 64, 0),     MakeItem(rb, 0, r2, 0, 32, 1),
      MakeItem(ra, 64, r1, 64, 64, 0),   MakeItem(rb, 32, r2, 64, 32, 1),
      MakeItem(ra, 200, r1, 200, 64, 2), MakeItem(ra, 128, r1, 128, 16, 3),
      MakeItem(ra, 0, r1, 0, 0, 4),
  };
  const auto built = engine.Plan(items);
  ExpectPlanContract(items, built);
  EXPECT_EQ(std::vector<size_t>({2}), built.rejected_tags);
  ASSERT_EQ(2u, built.plans.size());
  EXPECT_EQ(std::vector<size_t>({144}), built.plans[0].sizes);  // 0, 64, then 128 all abut
  EXPECT_EQ(std::vector<size_t>({0, 3}), built.plans[0].tags);
  EXPECT_EQ(std::vector<size_t>({32, 32}), built.plans[1].sizes);  // dst gap: two segments
  EXPECT_EQ(std::vector<size_t>({0, 64}), built.plans[1].dst_offsets);
}

// ---------------------------------------------------------------------------
//  HbmCopyEngine
// ---------------------------------------------------------------------------

TEST(HbmCopyEngine, BuilderMergesAdjacentRanges) {
  HbmCopyEngine engine;
  std::vector<char> src(64, 'x'), dst(64, 0);
  auto host = TransferRef::HostBytes(src.data(), src.size());
  // Planning uses descriptions only; these fake GPU endpoints are never submitted.
  auto gpu = TransferRef::HostBytes(dst.data(), dst.size(), mori::io::MemoryLocationType::GPU, 0);
  auto builder = engine.CreatePlanBuilder();
  builder->Add(MakeItem(host, 0, gpu, 0, 8, 1));
  builder->Add(MakeItem(host, 8, gpu, 8, 8, 1));
  builder->Add(MakeItem(host, 32, gpu, 40, 4, 2));
  auto planned = builder->Finish();
  EXPECT_TRUE(planned.rejected_tags.empty());
  ASSERT_EQ(1u, planned.plans.size());
  EXPECT_EQ(std::vector<size_t>({0, 32}), planned.plans[0].src_offsets);
  EXPECT_EQ(std::vector<size_t>({0, 40}), planned.plans[0].dst_offsets);
  EXPECT_EQ(std::vector<size_t>({16, 4}), planned.plans[0].sizes);
  EXPECT_EQ(std::vector<size_t>({1, 2}), planned.plans[0].tags);
}

TEST(HbmCopyEngine, BuilderCreatesPlanPerBufferPair) {
  HbmCopyEngine engine;
  std::vector<char> src(64, 'x'), dst(64, 0), other_dst(64, 0);
  auto host = TransferRef::HostBytes(src.data(), src.size());
  auto gpu = TransferRef::HostBytes(dst.data(), dst.size(), mori::io::MemoryLocationType::GPU, 0);
  auto other_gpu = TransferRef::HostBytes(other_dst.data(), other_dst.size(),
                                          mori::io::MemoryLocationType::GPU, 0);
  auto builder = engine.CreatePlanBuilder();
  builder->Add(MakeItem(host, 0, gpu, 0, 4, 1));
  builder->Add(MakeItem(host, 16, other_gpu, 2, 4, 2));
  auto planned = builder->Finish();

  ASSERT_EQ(2u, planned.plans.size());
  EXPECT_EQ(dst.data(), planned.plans[0].dst.host_ptr);
  EXPECT_EQ(other_dst.data(), planned.plans[1].dst.host_ptr);
}

TEST(HbmCopyEngine, BuilderRejectsOutOfBoundsRanges) {
  HbmCopyEngine engine;
  std::vector<char> src(64, 'x'), dst(64, 0);
  auto host = TransferRef::HostBytes(src.data(), src.size());
  auto gpu = TransferRef::HostBytes(dst.data(), dst.size(), mori::io::MemoryLocationType::GPU, 0);
  auto builder = engine.CreatePlanBuilder();
  builder->Add(MakeItem(host, 64, gpu, 0, 1, 1));
  builder->Add(MakeItem(host, 0, gpu, 64, 1, 2));
  builder->Add(MakeItem(host, 0, gpu, 0, 0, 3));
  auto planned = builder->Finish();

  EXPECT_EQ(std::vector<size_t>({1, 2}), planned.rejected_tags);
  EXPECT_TRUE(planned.plans.empty());
}

// A segment never crosses a host registration piece.
TEST(HbmCopyEngine, BuilderSplitsAtHostRegistrationPieces) {
  HbmCopyEngine engine;
  const uintptr_t piece = HostRegistrationChunkBytes();
  // Never dereferenced: planning reads addresses only.
  auto host = TransferRef::HostBytes(reinterpret_cast<void*>(piece - 4096), 65536);
  std::vector<char> stub(65536);
  auto gpu = TransferRef::HostBytes(stub.data(), stub.size(), mori::io::MemoryLocationType::GPU, 0);
  const std::vector<TransferItem> items{
      MakeItem(host, 0, gpu, 0, 8192, 1),        // straddles the boundary
      MakeItem(host, 8192, gpu, 8192, 4096, 2),  // continues the second piece
  };
  const auto planned = engine.Plan(items);
  ASSERT_EQ(1u, planned.plans.size());
  EXPECT_EQ(std::vector<size_t>({0, 4096}), planned.plans[0].src_offsets);
  EXPECT_EQ(std::vector<size_t>({4096, 8192}), planned.plans[0].sizes);
  EXPECT_EQ(std::vector<size_t>({1, 2}), planned.plans[0].tags);
  ExpectPlanContract(items, planned);
}

// Gathers submitted from a succession of short-lived threads, the way the
// standalone server's gRPC pool retires and spawns threads under load, share a
// pooled stream.  A stream per thread grew until HIP could not create another.
TEST(HbmCopyEngine, GatherStreamsOutliveTheirThreads) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) GTEST_SKIP() << "no GPU";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";

  constexpr size_t kBytes = 1 << 20;
  constexpr size_t kFragment = 4096;
  constexpr int kThreads = 16;
  AlignedBuffer host(kBytes);
  for (size_t i = 0; i < kBytes; ++i) host.data()[i] = static_cast<char>(i * 131 + 7);
  HbmCopyEngine engine;
  engine.AddHostGatherRegion(host.data(), host.size());
  void* device = nullptr;
  ASSERT_EQ(hipMalloc(&device, kBytes), hipSuccess);
  ASSERT_EQ(hipMemset(device, 0, kBytes), hipSuccess);
  const auto src = TransferRef::HostBytes(host.data(), host.size());
  const auto dst = TransferRef::HostBytes(device, kBytes, mori::io::MemoryLocationType::GPU, 0);

  const uint64_t launches_before = DeviceGatherLaunchCount();
  const uint64_t streams_before = HbmCopyEngine::GatherStreamsCreated();
  for (int t = 0; t < kThreads; ++t) {
    std::thread([&, t] {
      // Two fragments far apart, so they stay two segments: one is hipMemcpy's case.
      const uint64_t first = t * 2 * kFragment;
      const uint64_t second = kBytes / 2 + first;
      auto planned = engine.Plan({MakeItem(src, first, dst, first, kFragment, 1),
                                  MakeItem(src, second, dst, second, kFragment, 2)});
      std::vector<TransferFailure> failures;
      if (auto handle = engine.Submit(std::move(planned.plans))) handle->Wait(&failures);
      EXPECT_TRUE(failures.empty());
    }).join();
  }
  if (DeviceGatherLaunchCount() == launches_before) {
    (void)hipFree(device);
    GTEST_SKIP() << "host region not gatherable on this device";
  }
  EXPECT_EQ(DeviceGatherLaunchCount(), launches_before + kThreads);
  EXPECT_LE(HbmCopyEngine::GatherStreamsCreated() - streams_before, 1u);

  std::vector<char> copied(kBytes);
  ASSERT_EQ(hipMemcpy(copied.data(), device, kBytes, hipMemcpyDeviceToHost), hipSuccess);
  for (int t = 0; t < kThreads; ++t) {
    for (const size_t offset : {t * 2 * kFragment, kBytes / 2 + t * 2 * kFragment}) {
      EXPECT_EQ(std::memcmp(copied.data() + offset, host.data() + offset, kFragment), 0)
          << "thread " << t << " offset " << offset;
    }
  }
  (void)hipFree(device);
}

TEST(HbmCopyEngine, GathersRunOnPrewarmedStreams) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) GTEST_SKIP() << "no GPU";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";

  constexpr size_t kBytes = 1 << 20;
  constexpr size_t kFragment = 4096;
  constexpr size_t kPrewarmed = 3;
  AlignedBuffer host(kBytes);
  for (size_t i = 0; i < kBytes; ++i) host.data()[i] = static_cast<char>(i * 37 + 3);
  HbmCopyEngine engine;
  engine.AddHostGatherRegion(host.data(), host.size());
  void* device = nullptr;
  ASSERT_EQ(hipMalloc(&device, kBytes), hipSuccess);
  ASSERT_EQ(hipMemset(device, 0, kBytes), hipSuccess);
  const auto src = TransferRef::HostBytes(host.data(), host.size());
  const auto dst = TransferRef::HostBytes(device, kBytes, mori::io::MemoryLocationType::GPU, 0);

  const uint64_t streams_before = HbmCopyEngine::GatherStreamsCreated();
  HbmCopyEngine::Prewarm(kPrewarmed);
  const uint64_t streams_prewarmed = HbmCopyEngine::GatherStreamsCreated();
  EXPECT_LE(streams_prewarmed - streams_before, kPrewarmed * devices);

  // Twice as many gathers in flight on device 0 as were prewarmed there: the
  // extra ones wait for a stream rather than create one.
  constexpr size_t kGathers = 2 * kPrewarmed;
  const uint64_t launches_before = DeviceGatherLaunchCount();
  std::vector<std::thread> threads;
  for (size_t t = 0; t < kGathers; ++t) {
    threads.emplace_back([&, t] {
      const uint64_t first = t * 2 * kFragment;
      const uint64_t second = kBytes / 2 + first;
      auto planned = engine.Plan({MakeItem(src, first, dst, first, kFragment, 1),
                                  MakeItem(src, second, dst, second, kFragment, 2)});
      std::vector<TransferFailure> failures;
      if (auto handle = engine.Submit(std::move(planned.plans))) handle->Wait(&failures);
      EXPECT_TRUE(failures.empty());
    });
  }
  for (auto& thread : threads) thread.join();
  if (DeviceGatherLaunchCount() == launches_before) {
    (void)hipFree(device);
    GTEST_SKIP() << "host region not gatherable on this device";
  }
  EXPECT_EQ(DeviceGatherLaunchCount(), launches_before + kGathers);
  EXPECT_EQ(HbmCopyEngine::GatherStreamsCreated(), streams_prewarmed);

  std::vector<char> copied(kBytes);
  ASSERT_EQ(hipMemcpy(copied.data(), device, kBytes, hipMemcpyDeviceToHost), hipSuccess);
  for (size_t t = 0; t < kGathers; ++t) {
    for (const size_t offset : {t * 2 * kFragment, kBytes / 2 + t * 2 * kFragment}) {
      EXPECT_EQ(std::memcmp(copied.data() + offset, host.data() + offset, kFragment), 0)
          << "thread " << t << " offset " << offset;
    }
  }
  (void)hipFree(device);
}

// ---------------------------------------------------------------------------
//  MoriIoEngine (planning only: the engine is never Init()ed)
// ---------------------------------------------------------------------------

// Registered with this node's engine: the zero-copy local side.
TransferRef LocalRegistered(void* base, uint64_t size, mori::io::MemoryUniqueId id) {
  TransferRef ref = TransferRef::HostBytes(base, size);
  ref.mem.engineKey = "local";
  ref.mem.id = id;
  ref.mem.size = size;
  ref.mem.loc = mori::io::MemoryLocationType::CPU;
  return ref;
}

TEST(MoriIoEngine, GroupsZeroCopyByEndpointsAndCapsSegments) {
  MoriIoEngine engine("local", mori::io::IOEngineConfig{}, /*bounce_bytes=*/0);
  std::vector<char> buf(1 << 16);
  TransferRef local = LocalRegistered(buf.data(), buf.size(), 1);
  // Same buffer id on two peers: two plans.
  TransferRef peer_a = RemoteRef("peer-a", 5, 1 << 16);
  TransferRef peer_b = RemoteRef("peer-b", 5, 1 << 16);
  // Sizes only; never dereferenced.
  TransferRef big_local = LocalRegistered(buf.data(), 8ULL << 30, 2);
  TransferRef big_peer = RemoteRef("peer-a", 6, 8ULL << 30);
  const std::vector<TransferItem> items{
      MakeItem(local, 0, peer_a, 0, 4096, 0),        // push to A
      MakeItem(local, 0, peer_b, 0, 4096, 1),        // push to B: its own plan
      MakeItem(local, 4096, peer_a, 4096, 4096, 2),  // adjacent: merges into A's plan
      MakeItem(peer_a, 0, local, 8192, 4096, 3),     // a pull: its own plan
      MakeItem(peer_a, 0, peer_b, 0, 4096, 4),       // both remote
      MakeItem(local, 0, local, 0, 4096, 5),         // both local
      MakeItem(big_local, 0, big_peer, 0, 3ULL << 30, 6),
      MakeItem(big_local, 3ULL << 30, big_peer, 3ULL << 30, 3ULL << 30, 7),
  };
  const TransferPlanSet planned = engine.Plan(items);
  EXPECT_EQ(std::vector<size_t>({4, 5}), planned.rejected_tags);
  ASSERT_EQ(4u, planned.plans.size());
  EXPECT_EQ(std::vector<size_t>({8192}), planned.plans[0].sizes);
  EXPECT_EQ(std::vector<size_t>({0, 2}), planned.plans[0].tags);
  EXPECT_EQ("peer-b", planned.plans[1].dst.mem.engineKey);
  EXPECT_TRUE(planned.plans[2].dir == TransferDirection::kPull);
  // Merged, 6 GiB would overflow ibv_sge.length.
  EXPECT_EQ(std::vector<size_t>({3ULL << 30, 3ULL << 30}), planned.plans[3].sizes);
  ExpectPlanContract(items, planned);
}

// Unregistered memory is staged through the bounce pool; a full pool starts a
// new plan.
TEST(MoriIoEngine, StagesUnregisteredMemoryThroughTheBouncePool) {
  constexpr uint64_t kPool = 1 << 20;
  constexpr uint64_t kItem = 384 << 10;
  MoriIoEngine engine("local", mori::io::IOEngineConfig{}, kPool);
  std::vector<char> host(4 << 20), gpu_stub(4096);
  TransferRef unregistered = TransferRef::HostBytes(host.data(), host.size());
  TransferRef gpu = TransferRef::HostBytes(gpu_stub.data(), gpu_stub.size(),
                                           mori::io::MemoryLocationType::GPU, 0);
  TransferRef peer = RemoteRef("peer-a", 5, 4 << 20);
  const std::vector<TransferItem> items{
      MakeItem(unregistered, 0, peer, 0, kItem, 0),
      MakeItem(unregistered, kItem, peer, kItem, kItem, 1),          // same pool
      MakeItem(unregistered, 2 * kItem, peer, 2 * kItem, kItem, 2),  // pool full
      MakeItem(peer, 0, unregistered, 3 << 20, 4096, 3),             // a pull
      MakeItem(unregistered, 0, peer, 0, kPool + 1, 4),              // bigger than the pool
      MakeItem(gpu, 0, peer, 0, 4096, 5),                            // GPU: not stageable
  };
  const TransferPlanSet planned = engine.Plan(items);
  EXPECT_EQ(std::vector<size_t>({4, 5}), planned.rejected_tags);
  ASSERT_EQ(3u, planned.plans.size());
  const TransferPlan& first = planned.plans[0];
  EXPECT_TRUE(first.uses_bounce);
  EXPECT_EQ(std::vector<size_t>({2 * kItem}), first.sizes);  // pool offsets 0, kItem
  ASSERT_EQ(2u, first.bounce_copies.size());
  EXPECT_EQ(static_cast<void*>(host.data() + kItem), first.bounce_copies[1].host);
  EXPECT_EQ(std::vector<size_t>({2}), planned.plans[1].tags);
  EXPECT_TRUE(planned.plans[2].dir == TransferDirection::kPull);
  ExpectPlanContract(items, planned);

  // No pool: rejected.
  MoriIoEngine no_pool("local", mori::io::IOEngineConfig{}, /*bounce_bytes=*/0);
  EXPECT_EQ(std::vector<size_t>({0}), no_pool.Plan({items[0]}).rejected_tags);
}

// ---------------------------------------------------------------------------
//  CompositeTransferEngine
// ---------------------------------------------------------------------------

// Records what it was asked to do; claims exactly the pairs it is told to.
class SpyEngine final : public TransferEngine {
 public:
  explicit SpyEngine(bool claims_remote) : claims_remote_(claims_remote) {}

  const char* Name() const override { return "SpyEngine"; }

  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    ++registrations;
    TransferRef ref = TransferRef::HostBytes(base, size, loc, device);
    // Pretend to be a transport that also produces a mori-io registration.
    ref.mem.engineKey = "spy";
    ref.mem.id = next_id_++;
    ref.mem.size = size;
    return ref;
  }
  void Deregister(const TransferRef&) override { ++deregistrations; }

  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override {
    const bool any_remote = (src.host_ptr == nullptr) || (dst.host_ptr == nullptr);
    return claims_remote_ == any_remote;
  }

  // One plan for all items; Finish() records the count.
  class Builder final : public TransferPlanBuilder {
   public:
    explicit Builder(const SpyEngine& spy) : spy_(spy) {}
    void Add(const TransferItem& item) override {
      plan_.src = item.src;
      plan_.dst = item.dst;
      plan_.sizes.push_back(item.size);
      plan_.src_offsets.push_back(item.src_offset);
      plan_.dst_offsets.push_back(item.dst_offset);
      plan_.tags.push_back(item.tag);
    }
    TransferPlanSet Finish() override {
      spy_.planned_batches.push_back(plan_.sizes.size());
      TransferPlanSet out;
      if (!plan_.sizes.empty()) out.plans.push_back(std::move(plan_));
      return out;
    }

   private:
    const SpyEngine& spy_;
    TransferPlan plan_;
  };
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t) const override {
    return std::make_unique<Builder>(*this);
  }

  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override {
    submitted += plans.size();
    return nullptr;  // "posted nothing" — the composite must fail these tags
  }

  int registrations = 0;
  int deregistrations = 0;
  size_t submitted = 0;
  mutable std::vector<size_t> planned_batches;

 private:
  bool claims_remote_;
  mori::io::MemoryUniqueId next_id_ = 1;
};

TEST(CompositeTransferEngine, RegistrationFansOutAndMergesHandles) {
  auto composite = std::make_unique<CompositeTransferEngine>();
  // LocalCopyEngine contributes only a host pointer; the spy contributes a
  // mori-io registration.  A merged ref must carry BOTH — that is the whole
  // reason TransferRef is a struct of per-transport handles and not a variant.
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  auto spy_owned = std::make_unique<SpyEngine>(/*claims_remote=*/true);
  SpyEngine* spy = spy_owned.get();
  composite->AddEngine(std::move(spy_owned));

  std::vector<char> buf(256, 'z');
  TransferRef ref =
      composite->RegisterMemory(buf.data(), buf.size(), mori::io::MemoryLocationType::CPU, -1);
  EXPECT_EQ(1, spy->registrations);
  EXPECT_TRUE(ref.HasHostPtr());
  EXPECT_EQ(buf.data(), ref.host_ptr);
  EXPECT_TRUE(ref.HasMemoryDesc());
  EXPECT_EQ("spy", ref.mem.engineKey);

  composite->Deregister(ref);
  EXPECT_EQ(1, spy->deregistrations);
}

TEST(CompositeTransferEngine, SelectsPerPairAndPlansOncePerEngine) {
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  auto spy_owned = std::make_unique<SpyEngine>(/*claims_remote=*/true);
  SpyEngine* spy = spy_owned.get();
  composite->AddEngine(std::move(spy_owned));

  std::vector<char> src(2048, 'q'), dst(2048, 0);
  TransferRef s = TransferRef::HostBytes(src.data(), src.size());
  TransferRef d = TransferRef::HostBytes(dst.data(), dst.size());
  TransferRef remote = RemoteRef("peer", /*id=*/3, 2048);

  std::vector<TransferItem> items{
      MakeItem(s, 0, d, 0, 512, 0),         // local -> local: LocalCopyEngine
      MakeItem(s, 512, remote, 0, 512, 1),  // local -> remote: SpyEngine
      MakeItem(s, 1024, d, 1024, 512, 2),   // local -> local
  };

  TransferPlanSet planned = composite->Plan(items);
  EXPECT_TRUE(planned.rejected_tags.empty());
  // Each engine gets ONE builder per call, fed all of its items.
  ASSERT_EQ(1u, spy->planned_batches.size());
  EXPECT_EQ(1u, spy->planned_batches[0]);
  // Every plan is tagged with the engine that produced it.
  ASSERT_FALSE(planned.plans.empty());
  for (const auto& plan : planned.plans) EXPECT_NE(nullptr, plan.engine);
}
TEST(CompositeTransferEngine, BuilderGroupsHbmAndBatchesOtherEngines) {
  CompositeTransferEngine composite;
  auto hbm = std::make_unique<HbmCopyEngine>();
  auto* hbm_ptr = hbm.get();
  composite.AddEngine(std::move(hbm));
  composite.AddEngine(std::make_unique<LocalCopyEngine>());
  auto spy_owned = std::make_unique<SpyEngine>(/*claims_remote=*/true);
  auto* spy = spy_owned.get();
  composite.AddEngine(std::move(spy_owned));
  std::vector<char> src(64, 'x'), dst(64, 0), gpu_bytes(64, 0);
  auto host = TransferRef::HostBytes(src.data(), src.size());
  auto cpu = TransferRef::HostBytes(dst.data(), dst.size());
  auto gpu = TransferRef::HostBytes(gpu_bytes.data(), gpu_bytes.size(),
                                    mori::io::MemoryLocationType::GPU, 0);
  auto remote = RemoteRef("peer", 3, 64);
  const std::vector<TransferItem> items{
      MakeItem(host, 0, gpu, 0, 4, 10),     MakeItem(host, 0, cpu, 0, 4, 20),
      MakeItem(host, 8, remote, 0, 4, 30),  MakeItem(host, 4, gpu, 4, 4, 11),
      MakeItem(host, 12, remote, 4, 4, 31), MakeItem(host, 64, gpu, 0, 1, 40),
  };
  auto builder = composite.CreatePlanBuilder();
  for (const auto& item : items) builder->Add(item);
  auto planned = builder->Finish();
  EXPECT_EQ(std::vector<size_t>({40}), planned.rejected_tags);
  EXPECT_EQ(std::vector<size_t>({2}), spy->planned_batches);
  ASSERT_EQ(3u, planned.plans.size());
  // Engines in first-seen order.
  EXPECT_EQ(hbm_ptr, planned.plans[0].engine);
  EXPECT_EQ(std::vector<size_t>({8}), planned.plans[0].sizes);
  EXPECT_EQ(std::vector<size_t>({10, 11}), planned.plans[0].tags);
  EXPECT_EQ(spy, planned.plans[2].engine);
  for (const auto& plan : planned.plans) EXPECT_NE(nullptr, plan.engine);
  EXPECT_EQ(0u, spy->submitted);
  EXPECT_EQ(std::vector<char>(64, 0), dst);
  ExpectPlanContract(items, planned);
}

TEST(CompositeTransferEngine, RejectsAPairNoEngineClaims) {
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());

  std::vector<char> src(512, 'r');
  TransferRef s = TransferRef::HostBytes(src.data(), src.size());
  TransferRef remote = RemoteRef("peer", /*id=*/9, 512);

  std::vector<TransferItem> items{MakeItem(s, 0, remote, 0, 512, 42)};
  TransferPlanSet planned = composite->Plan(items);
  EXPECT_TRUE(planned.plans.empty());
  EXPECT_EQ(std::vector<size_t>({42}), planned.rejected_tags);
}

TEST(CompositeTransferEngine, FailsTagsWhenASubEngineSubmitsNothing) {
  // A sub-engine that posts nothing must not silently drop keys: the caller
  // would wait for a completion that never comes and report success.
  auto composite = std::make_unique<CompositeTransferEngine>();
  auto spy_owned = std::make_unique<SpyEngine>(/*claims_remote=*/true);
  composite->AddEngine(std::move(spy_owned));

  std::vector<char> src(512, 'r');
  TransferRef s = TransferRef::HostBytes(src.data(), src.size());
  TransferRef remote = RemoteRef("peer", /*id=*/9, 512);

  std::vector<size_t> failed;
  EXPECT_FALSE(composite->Transfer({MakeItem(s, 0, remote, 0, 512, 5)}, &failed));
  EXPECT_EQ(std::vector<size_t>({5}), failed);
}

// ---------------------------------------------------------------------------
//  File endpoints (GdsEngine's shape, without HIP)
// ---------------------------------------------------------------------------

// Stands in for GdsEngine: the only engine that registers and claims files.
class FakeFileEngine final : public TransferEngine {
 public:
  const char* Name() const override { return "FakeFileEngine"; }
  TransferRef RegisterMemory(void*, size_t, mori::io::MemoryLocationType, int) override {
    return TransferRef{};  // not a memory engine
  }
  TransferRef RegisterFile(int fd, uint64_t offset, uint64_t size) override {
    ++file_registrations;
    return TransferRef::File(fd, offset, size, reinterpret_cast<void*>(0xF11E));
  }
  void Deregister(const TransferRef& ref) override {
    if (ref.IsFile()) ++file_deregistrations;
  }
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override {
    return src.IsFile() && dst.loc == mori::io::MemoryLocationType::GPU;
  }
  // Plans nothing: only registration and selection are tested.
  class NoPlans final : public TransferPlanBuilder {
   public:
    void Add(const TransferItem&) override {}
    TransferPlanSet Finish() override { return {}; }
  };
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t) const override {
    return std::make_unique<NoPlans>();
  }
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan>) override { return nullptr; }

  int file_registrations = 0;
  int file_deregistrations = 0;
};

TEST(TransferRef, FileEndpointIsValidAndIgnoredByMemoryEngines) {
  TransferRef f = TransferRef::File(/*fd=*/7, /*offset=*/4096, /*n=*/65536);
  EXPECT_TRUE(f.IsFile());
  EXPECT_TRUE(f.Valid());
  EXPECT_FALSE(f.HasHostPtr());
  EXPECT_FALSE(f.HasMemoryDesc());
  EXPECT_EQ(7, f.file_fd);
  EXPECT_EQ(4096u, f.file_offset);
  EXPECT_EQ(65536u, f.size);

  // A memory engine never claims a file endpoint, so per-pair selection leaves
  // it for the file engine.
  LocalCopyEngine mem;
  std::vector<char> stub(64);
  TransferRef gpu =
      TransferRef::HostBytes(stub.data(), stub.size(), mori::io::MemoryLocationType::GPU, 0);
  EXPECT_FALSE(mem.CanHandle(f, gpu));
}

TEST(CompositeTransferEngine, RegisterFileFansOutToTheFileEngine) {
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  auto file_owned = std::make_unique<FakeFileEngine>();
  FakeFileEngine* file = file_owned.get();
  composite->AddEngine(std::move(file_owned));

  TransferRef ref = composite->RegisterFile(/*fd=*/9, /*offset=*/8192, /*size=*/4096);
  EXPECT_EQ(1, file->file_registrations);
  ASSERT_TRUE(ref.IsFile());
  EXPECT_EQ(9, ref.file_fd);
  EXPECT_EQ(8192u, ref.file_offset);
  EXPECT_EQ(reinterpret_cast<void*>(0xF11E), ref.gds_handle);

  composite->Deregister(ref);
  EXPECT_EQ(1, file->file_deregistrations);
}

TEST(CompositeTransferEngine, RegisterFileIsInvalidWithoutAFileEngine) {
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  TransferRef ref = composite->RegisterFile(/*fd=*/3, /*offset=*/0, /*size=*/4096);
  EXPECT_FALSE(ref.IsFile());
  EXPECT_FALSE(ref.Valid());
}

// Scattered pages with an unaligned tail go to a file range and back, direct
// where the filesystem allows it; the tail's on-disk padding is zeroed.
TEST(PosixEngine, RoundTripsScatteredPagesThroughAFile) {
  constexpr size_t kPage = 4096;
  constexpr size_t kValue = 2 * kPage + 100;
  const std::string path = ::testing::TempDir() + "/umbp_posix_engine.bin";
  for (int flags : {O_DIRECT, 0}) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | flags, 0644);
    if (fd < 0) continue;  // no O_DIRECT here
    CompositeTransferEngine composite;
    composite.AddEngine(std::make_unique<LocalCopyEngine>());
    composite.AddEngine(std::make_unique<PosixEngine>());

    AlignedBuffer dram(4 * kPage);
    for (size_t i = 0; i < dram.size(); ++i) dram.data()[i] = static_cast<char>(i * 7 + 1);
    const TransferRef mem = TransferRef::HostBytes(dram.data(), dram.size());
    const TransferRef file = TransferRef::File(fd, kPage, 3 * kPage);
    EXPECT_STREQ("PosixEngine", composite.SelectEngine(mem, file)->Name());
    EXPECT_STREQ("PosixEngine", composite.SelectEngine(file, mem)->Name());

    // The value lives in pages {2, 0, 3}; it reads back into pages {1, 3, 0}.
    const size_t from[] = {2, 0, 3};
    const size_t to[] = {1, 3, 0};
    std::vector<TransferItem> put, get;
    for (size_t i = 0; i < 3; ++i) {
      const size_t n = std::min(kPage, kValue - i * kPage);
      put.push_back(MakeItem(mem, from[i] * kPage, file, i * kPage, n, i));
      get.push_back(MakeItem(file, i * kPage, mem, to[i] * kPage, n, i));
    }
    std::vector<char> want(kValue);
    for (size_t i = 0; i < 3; ++i) {
      std::memcpy(want.data() + i * kPage, dram.data() + from[i] * kPage,
                  std::min(kPage, kValue - i * kPage));
    }
    ASSERT_TRUE(composite.Transfer(put, nullptr));
    std::memset(dram.data(), 0, dram.size());
    ASSERT_TRUE(composite.Transfer(get, nullptr));
    for (size_t i = 0; i < 3; ++i) {
      const size_t n = std::min(kPage, kValue - i * kPage);
      EXPECT_EQ(0, std::memcmp(want.data() + i * kPage, dram.data() + to[i] * kPage, n));
    }

    AlignedBuffer tail(kPage);
    ASSERT_EQ(static_cast<ssize_t>(kPage), ::pread(fd, tail.data(), kPage, 3 * kPage));
    EXPECT_EQ(std::string(kPage - 100, '\0'), std::string(tail.data() + 100, kPage - 100));
    ::close(fd);
  }
  ::unlink(path.c_str());
}

TEST(CompositeTransferEngine, SelectsTheFileEngineForAFileToGpuPair) {
  auto composite = std::make_unique<CompositeTransferEngine>();
  composite->AddEngine(std::make_unique<LocalCopyEngine>());
  composite->AddEngine(std::make_unique<FakeFileEngine>());

  TransferRef src =
      TransferRef::File(/*fd=*/5, /*offset=*/0, /*n=*/4096, reinterpret_cast<void*>(0x1));
  std::vector<char> gpu(4096);
  TransferRef dst =
      TransferRef::HostBytes(gpu.data(), gpu.size(), mori::io::MemoryLocationType::GPU, 0);
  TransferEngine* sel = composite->SelectEngine(src, dst);
  ASSERT_NE(nullptr, sel);
  EXPECT_STREQ("FakeFileEngine", sel->Name());
}

}  // namespace
}  // namespace mori::umbp

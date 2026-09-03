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

// Phase 6 transfer layer, without gRPC or RDMA: LocalCopyEngine's planner and
// executor, and CompositeTransferEngine's fan-out registration + per-pair
// engine selection.
//
// The mori-io engine is exercised end-to-end by test_cross_node_smoke (which
// needs a real fabric); what is unit-testable here is everything that decides
// WHICH engine runs and WHAT it is asked to run.

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
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

  TransferPlanSet Plan(const std::vector<TransferItem>& items) const override {
    planned_batches.push_back(items.size());
    TransferPlanSet out;
    TransferPlan plan;
    for (const auto& item : items) {
      plan.src = item.src;
      plan.dst = item.dst;
      plan.sizes.push_back(item.size);
      plan.src_offsets.push_back(item.src_offset);
      plan.dst_offsets.push_back(item.dst_offset);
      plan.tags.push_back(item.tag);
    }
    if (!plan.sizes.empty()) out.plans.push_back(std::move(plan));
    return out;
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
  // The two local items were planned in ONE call to the local engine, not one
  // call per item: partition-then-plan is what keeps grouping meaningful.
  ASSERT_EQ(1u, spy->planned_batches.size());
  EXPECT_EQ(1u, spy->planned_batches[0]);
  // Every plan is tagged with the engine that produced it.
  ASSERT_FALSE(planned.plans.empty());
  for (const auto& plan : planned.plans) EXPECT_NE(nullptr, plan.engine);
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
  TransferPlanSet Plan(const std::vector<TransferItem>&) const override { return {}; }
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

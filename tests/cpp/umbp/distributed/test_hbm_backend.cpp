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

// HbmCopyEngine + the HBM backend: the second medium, and the engine that
// exists because no engine in tree could serve its local pairs.
//
// The planner/selection half needs no GPU and always runs.  The byte-moving
// half needs a real device and SKIPS (rather than fails) without one, so the
// suite stays runnable on a CPU-only box — matching how the integration label
// is used for tests that need a fabric.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "umbp/distributed/peer/backend/hbm_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"

namespace mori::umbp {
namespace {

bool HaveGpu() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

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

TransferRef HostRef(void* p, uint64_t n) {
  return TransferRef::HostBytes(p, n, mori::io::MemoryLocationType::CPU, -1);
}
TransferRef GpuRef(void* p, uint64_t n, int device = 0) {
  return TransferRef::HostBytes(p, n, mori::io::MemoryLocationType::GPU, device);
}

// RAII device buffer so a failed assertion cannot leak VRAM across cases.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) {
    if (hipMalloc(&ptr_, bytes) != hipSuccess) ptr_ = nullptr;
    size_ = ptr_ != nullptr ? bytes : 0;
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) (void)hipFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  size_t size() const { return size_; }
  bool valid() const { return ptr_ != nullptr; }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

// ---------------------------------------------------------------------------
//  Selection — the reason this engine exists
// ---------------------------------------------------------------------------

// The gap this engine was written to close: before it, a both-local pair with a
// GPU endpoint was claimed by NO engine, so an HBM backend's local Put/Get
// could not complete.  Asserted against the real composite, not by inspection.
TEST(HbmCopyEngine, ClaimsTheLocalGpuPairNoOtherEngineWould) {
  int host = 0;
  int fake_device = 0;  // no GPU needed: selection never dereferences.
  const TransferRef h = HostRef(&host, sizeof(host));
  const TransferRef g = GpuRef(&fake_device, sizeof(fake_device));

  LocalCopyEngine local;
  HbmCopyEngine hbm;

  // The precondition: the host-only engine refuses every pair with a GPU side.
  EXPECT_FALSE(local.CanHandle(h, g));
  EXPECT_FALSE(local.CanHandle(g, h));
  EXPECT_FALSE(local.CanHandle(g, g));

  // And this engine picks up exactly those three.
  EXPECT_TRUE(hbm.CanHandle(h, g));  // H2D
  EXPECT_TRUE(hbm.CanHandle(g, h));  // D2H
  EXPECT_TRUE(hbm.CanHandle(g, g));  // D2D
}

// Disjoint, not merely ordered: the two local engines must never both claim a
// pair, or composite registration order would silently become a performance
// decision.
TEST(HbmCopyEngine, DoesNotOverlapLocalCopyEngine) {
  int a = 0, b = 0;
  const TransferRef h1 = HostRef(&a, sizeof(a));
  const TransferRef h2 = HostRef(&b, sizeof(b));

  LocalCopyEngine local;
  HbmCopyEngine hbm;

  EXPECT_TRUE(local.CanHandle(h1, h2));
  EXPECT_FALSE(hbm.CanHandle(h1, h2));  // both-CPU stays with the NT-AVX2 path
}

TEST(HbmCopyEngine, CompositeRoutesGpuPairsHere) {
  int host = 0;
  int fake_device = 0;
  CompositeTransferEngine composite;
  composite.AddEngine(std::make_unique<LocalCopyEngine>());
  composite.AddEngine(std::make_unique<HbmCopyEngine>());

  const TransferRef h = HostRef(&host, sizeof(host));
  const TransferRef g = GpuRef(&fake_device, sizeof(fake_device));

  TransferEngine* for_host = composite.SelectEngine(h, h);
  TransferEngine* for_gpu = composite.SelectEngine(h, g);
  ASSERT_NE(for_host, nullptr);
  ASSERT_NE(for_gpu, nullptr);
  EXPECT_STREQ(for_host->Name(), "LocalCopyEngine");
  EXPECT_STREQ(for_gpu->Name(), "HbmCopyEngine");
}

// ---------------------------------------------------------------------------
//  Planner
// ---------------------------------------------------------------------------

TEST(HbmCopyEngine, CoalescesAdjacentSegments) {
  int host = 0, dev = 0;
  HbmCopyEngine engine;
  const TransferRef h = HostRef(&host, 4096);
  const TransferRef g = GpuRef(&dev, 4096);

  // Three adjacent pages on both sides collapse into one hipMemcpy — the win
  // that matters more here than for memcpy, since each call has launch cost.
  const auto set = engine.Plan({MakeItem(h, 0, g, 0, 1024, 0), MakeItem(h, 1024, g, 1024, 1024, 1),
                                MakeItem(h, 2048, g, 2048, 1024, 2)});

  ASSERT_EQ(set.plans.size(), 1u);
  EXPECT_TRUE(set.rejected_tags.empty());
  ASSERT_EQ(set.plans[0].sizes.size(), 1u);
  EXPECT_EQ(set.plans[0].sizes[0], 3072u);
  EXPECT_EQ(set.plans[0].tags.size(), 3u);
}

TEST(HbmCopyEngine, RejectsOutOfBoundsRatherThanCorruptingThePool) {
  int host = 0, dev = 0;
  HbmCopyEngine engine;
  const TransferRef h = HostRef(&host, 1024);
  const TransferRef g = GpuRef(&dev, 1024);

  const auto set = engine.Plan({MakeItem(h, 0, g, 512, 1024, 7)});
  EXPECT_TRUE(set.plans.empty());
  ASSERT_EQ(set.rejected_tags.size(), 1u);
  EXPECT_EQ(set.rejected_tags[0], 7u);
}

TEST(HbmCopyEngine, RejectsPairsItCannotCarry) {
  int a = 0, b = 0;
  HbmCopyEngine engine;
  const auto set = engine.Plan({MakeItem(HostRef(&a, 64), 0, HostRef(&b, 64), 0, 64, 3)});
  EXPECT_TRUE(set.plans.empty());
  ASSERT_EQ(set.rejected_tags.size(), 1u);
  EXPECT_EQ(set.rejected_tags[0], 3u);
}

// ---------------------------------------------------------------------------
//  Real bytes (needs a GPU)
// ---------------------------------------------------------------------------

TEST(HbmCopyEngine, RoundTripsHostToDeviceAndBack) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";

  constexpr size_t kBytes = 256 * 1024;
  DeviceBuffer device(kBytes);
  ASSERT_TRUE(device.valid());

  std::vector<char> src(kBytes), dst(kBytes, 0);
  std::iota(src.begin(), src.end(), 1);

  HbmCopyEngine engine;
  const TransferRef s = HostRef(src.data(), kBytes);
  const TransferRef g = GpuRef(device.get(), kBytes);
  const TransferRef d = HostRef(dst.data(), kBytes);

  std::vector<size_t> failed;
  ASSERT_TRUE(engine.Transfer({MakeItem(s, 0, g, 0, kBytes, 0)}, &failed)) << "H2D failed";
  EXPECT_TRUE(failed.empty());

  ASSERT_TRUE(engine.Transfer({MakeItem(g, 0, d, 0, kBytes, 0)}, &failed)) << "D2H failed";
  EXPECT_TRUE(failed.empty());

  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kBytes), 0);
}

TEST(HbmCopyEngine, RoundTripsScatteredPagesThroughOneDeviceBuffer) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";

  constexpr size_t kPage = 4096;
  constexpr size_t kPages = 4;
  DeviceBuffer device(kPage * kPages);
  ASSERT_TRUE(device.valid());

  std::vector<char> src(kPage * kPages), dst(kPage * kPages, 0);
  std::iota(src.begin(), src.end(), 7);

  HbmCopyEngine engine;
  const TransferRef s = HostRef(src.data(), src.size());
  const TransferRef g = GpuRef(device.get(), kPage * kPages);
  const TransferRef d = HostRef(dst.data(), dst.size());

  // Deliberately non-adjacent order so the planner cannot coalesce everything.
  // The stride must be coprime with kPages or this stops being a permutation
  // and silently leaves pages uncopied (which is how this test first failed).
  static_assert(kPages == 4, "stride 3 below is chosen coprime with kPages");
  std::vector<TransferItem> up;
  for (size_t i = 0; i < kPages; ++i) {
    const size_t page = (i * 3) % kPages;  // {0, 3, 2, 1}
    up.push_back(MakeItem(s, page * kPage, g, page * kPage, kPage, i));
  }
  std::vector<size_t> failed;
  ASSERT_TRUE(engine.Transfer(up, &failed));

  std::vector<TransferItem> down;
  for (size_t i = 0; i < kPages; ++i) {
    down.push_back(MakeItem(g, i * kPage, d, i * kPage, kPage, i));
  }
  ASSERT_TRUE(engine.Transfer(down, &failed));

  EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
}

TEST(HbmCopyEngine, CopiesDeviceToDevice) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";

  constexpr size_t kBytes = 64 * 1024;
  DeviceBuffer a(kBytes), b(kBytes);
  ASSERT_TRUE(a.valid());
  ASSERT_TRUE(b.valid());

  std::vector<char> src(kBytes), dst(kBytes, 0);
  std::iota(src.begin(), src.end(), 3);
  ASSERT_EQ(hipMemcpy(a.get(), src.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);

  HbmCopyEngine engine;
  std::vector<size_t> failed;
  ASSERT_TRUE(engine.Transfer(
      {MakeItem(GpuRef(a.get(), kBytes), 0, GpuRef(b.get(), kBytes), 0, kBytes, 0)}, &failed));

  ASSERT_EQ(hipMemcpy(dst.data(), b.get(), kBytes, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), kBytes), 0);
}

// ---------------------------------------------------------------------------
//  The backend
// ---------------------------------------------------------------------------

// A registrar that only hands back process-local refs — enough to exercise the
// backend's ownership path without an IOEngine, and it is what
// CompositeTransferEngine degrades to on a node with no RDMA configured.
class LocalOnlyRegistrar final : public MemoryRegistrar {
 public:
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    ++registrations;
    last_loc = loc;
    last_device = device;
    return TransferRef::HostBytes(base, size, loc, device);
  }
  void Deregister(const TransferRef&) override { ++deregistrations; }

  int registrations = 0;
  int deregistrations = 0;
  mori::io::MemoryLocationType last_loc = mori::io::MemoryLocationType::CPU;
  int last_device = -1;
};

// The registration contract: the backend supplies the facts a descriptor cannot
// recover.  Getting this wrong is invisible until a transfer picks the wrong
// engine, which is exactly what this asserts against.
TEST(HbmBackend, RegistersItsPoolAsGpuMemoryOnItsDevice) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";

  constexpr uint64_t kPageSize = 64 * 1024;
  auto backend = MakeHbmBackend(kPageSize, /*device=*/0, {kPageSize * 8},
                                std::chrono::milliseconds{30000}, std::chrono::milliseconds{500});
  LocalOnlyRegistrar registrar;
  ASSERT_TRUE(backend->Init(&registrar));

  EXPECT_EQ(backend->Tier(), TierType::HBM);
  EXPECT_EQ(registrar.registrations, 1);
  EXPECT_EQ(registrar.last_loc, mori::io::MemoryLocationType::GPU);
  EXPECT_EQ(registrar.last_device, 0);

  // And the endpoint it publishes says so too, which is what routes a local
  // transfer to HbmCopyEngine instead of LocalCopyEngine.
  ASSERT_EQ(backend->BufferCount(), 1u);
  const TransferRef ref = backend->BufferRef(0);
  EXPECT_TRUE(ref.Valid());
  EXPECT_EQ(ref.loc, mori::io::MemoryLocationType::GPU);
  EXPECT_EQ(ref.device, 0);

  backend->Shutdown();
  EXPECT_EQ(registrar.deregistrations, 1);
}

// The point of the whole exercise: a Put and a Get against HBM, moving real
// bytes, with the composite choosing the engine — no tier branch anywhere.
TEST(HbmBackend, PutsAndGetsThroughTheCompositeEngine) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";

  constexpr uint64_t kPageSize = 64 * 1024;
  auto backend = MakeHbmBackend(kPageSize, /*device=*/0, {kPageSize * 4},
                                std::chrono::milliseconds{30000}, std::chrono::milliseconds{500});
  LocalOnlyRegistrar registrar;
  ASSERT_TRUE(backend->Init(&registrar));

  CompositeTransferEngine composite;
  composite.AddEngine(std::make_unique<LocalCopyEngine>());
  composite.AddEngine(std::make_unique<HbmCopyEngine>());

  // --- Put ---
  auto allocated = backend->BatchAllocate({AllocateRequest{"key-a", kPageSize}});
  ASSERT_EQ(allocated.size(), 1u);
  ASSERT_EQ(allocated[0].outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_EQ(allocated[0].pages.size(), 1u);

  std::vector<char> payload(kPageSize);
  std::iota(payload.begin(), payload.end(), 11);

  const TransferRef pool = backend->BufferRef(allocated[0].pages[0].buffer_index);
  const uint64_t page_off = static_cast<uint64_t>(allocated[0].pages[0].page_index) * kPageSize;

  std::vector<size_t> failed;
  ASSERT_TRUE(composite.Transfer(
      {MakeItem(HostRef(payload.data(), kPageSize), 0, pool, page_off, kPageSize, 0)}, &failed))
      << "host -> HBM put failed";

  auto committed = backend->BatchCommit({CommitRequest{allocated[0].slot_id, "key-a"}});
  ASSERT_EQ(committed.size(), 1u);
  EXPECT_TRUE(committed[0].success);

  // --- Get ---
  auto resolved = backend->BatchResolve({"key-a"}, /*include_descs=*/false);
  ASSERT_EQ(resolved.size(), 1u);
  ASSERT_TRUE(resolved[0].found);
  ASSERT_EQ(resolved[0].pages.size(), 1u);
  EXPECT_EQ(resolved[0].size, kPageSize);

  std::vector<char> readback(kPageSize, 0);
  const TransferRef read_pool = backend->BufferRef(resolved[0].pages[0].buffer_index);
  const uint64_t read_off = static_cast<uint64_t>(resolved[0].pages[0].page_index) * kPageSize;
  ASSERT_TRUE(composite.Transfer(
      {MakeItem(read_pool, read_off, HostRef(readback.data(), kPageSize), 0, kPageSize, 0)},
      &failed))
      << "HBM -> host get failed";

  EXPECT_EQ(std::memcmp(payload.data(), readback.data(), kPageSize), 0);

  backend->Shutdown();
}

}  // namespace
}  // namespace mori::umbp

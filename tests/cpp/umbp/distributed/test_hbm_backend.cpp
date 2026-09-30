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
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "umbp/common/device_gather.h"
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

// Page-aligned host memory, as HostTierRegistration expects.  Declare it before
// the engine that registers it: the engine unregisters on destruction.
class HostPages {
 public:
  explicit HostPages(size_t bytes) : size_(bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ptr_ = p == MAP_FAILED ? nullptr : static_cast<char*>(p);
  }
  ~HostPages() {
    if (ptr_ != nullptr) munmap(ptr_, size_);
  }
  HostPages(const HostPages&) = delete;
  HostPages& operator=(const HostPages&) = delete;

  char* get() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  char* ptr_ = nullptr;
  size_t size_ = 0;
};

constexpr size_t kRestoreSegment = 8448;  // median restore fragment in production

// Host offset of segment i of a permuted, gapped layout: nothing coalesces.
size_t ScatteredOffset(size_t i, size_t segments, size_t stride) {
  return ((i * stride) % segments) * 2 * kRestoreSegment;
}

// A registered host region takes the gather kernel in both directions: one
// launch per batch however many scattered segments it holds.
TEST(HbmCopyEngine, GatherKernelRoundTripsScatteredSegments) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";

  constexpr size_t kSegments = 64;
  HostPages up(4 << 20), down(4 << 20);
  ASSERT_NE(up.get(), nullptr);
  ASSERT_NE(down.get(), nullptr);
  for (size_t i = 0; i < up.size(); ++i) up.get()[i] = static_cast<char>(i * 131 + 7);
  DeviceBuffer device(kSegments * kRestoreSegment);
  ASSERT_TRUE(device.valid());

  HbmCopyEngine engine;
  engine.AddHostGatherRegion(up.get(), up.size());
  engine.AddHostGatherRegion(down.get(), down.size());
  const TransferRef u = HostRef(up.get(), up.size());
  const TransferRef g = GpuRef(device.get(), device.size());
  const TransferRef d = HostRef(down.get(), down.size());

  std::vector<TransferItem> to_gpu, to_host;
  for (size_t i = 0; i < kSegments; ++i) {
    const size_t host = ScatteredOffset(i, kSegments, 37);
    to_gpu.push_back(MakeItem(u, host, g, i * kRestoreSegment, kRestoreSegment, i));
    to_host.push_back(MakeItem(g, i * kRestoreSegment, d, host, kRestoreSegment, i));
  }
  const uint64_t launches = DeviceGatherLaunchCount();
  std::vector<size_t> failed;
  ASSERT_TRUE(engine.Transfer(to_gpu, &failed));
  ASSERT_TRUE(engine.Transfer(to_host, &failed));
  EXPECT_EQ(DeviceGatherLaunchCount(), launches + 2);
  for (size_t i = 0; i < kSegments; ++i) {
    const size_t host = ScatteredOffset(i, kSegments, 37);
    EXPECT_EQ(std::memcmp(up.get() + host, down.get() + host, kRestoreSegment), 0)
        << "segment " << i;
  }
}

// Submit runs on many threads at once (PoolClient executors, the standalone
// server's gRPC threads).  Concurrent gather batches draw their own stream and
// descriptor staging, so no batch may copy another batch's segments.
TEST(HbmCopyEngine, ConcurrentGatherBatchesStayIndependent) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";

  constexpr size_t kThreads = 8;
  constexpr size_t kSegments = 32;
  constexpr size_t kRounds = 20;
  constexpr size_t kSlice = kSegments * 2 * kRestoreSegment;
  HostPages up(kThreads * kSlice), down(kThreads * kSlice);
  ASSERT_NE(up.get(), nullptr);
  ASSERT_NE(down.get(), nullptr);
  for (size_t i = 0; i < up.size(); ++i) up.get()[i] = static_cast<char>(i * 29 + 3);
  std::vector<std::unique_ptr<DeviceBuffer>> devices;
  for (size_t t = 0; t < kThreads; ++t) {
    devices.push_back(std::make_unique<DeviceBuffer>(kSegments * kRestoreSegment));
    ASSERT_TRUE(devices.back()->valid());
  }

  HbmCopyEngine engine;
  engine.AddHostGatherRegion(up.get(), up.size());
  engine.AddHostGatherRegion(down.get(), down.size());
  const TransferRef u = HostRef(up.get(), up.size());
  const TransferRef d = HostRef(down.get(), down.size());

  std::atomic<size_t> mismatches{0}, failures{0};
  std::vector<std::thread> threads;
  for (size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      const TransferRef g = GpuRef(devices[t]->get(), devices[t]->size());
      std::vector<char> seen(kSegments * kRestoreSegment);
      for (size_t round = 0; round < kRounds; ++round) {
        // A different odd stride each round is a different permutation of the
        // slice, so a batch that picked up stale or foreign descriptors lands
        // the wrong bytes.
        const size_t stride = 2 * round + 1;
        std::vector<TransferItem> to_gpu, to_host;
        for (size_t i = 0; i < kSegments; ++i) {
          const size_t host = t * kSlice + ScatteredOffset(i, kSegments, stride);
          to_gpu.push_back(MakeItem(u, host, g, i * kRestoreSegment, kRestoreSegment, i));
          to_host.push_back(MakeItem(g, i * kRestoreSegment, d, host, kRestoreSegment, i));
        }
        std::vector<size_t> failed;
        if (!engine.Transfer(to_gpu, &failed)) ++failures;
        (void)hipSetDevice(0);
        if (hipMemcpy(seen.data(), devices[t]->get(), seen.size(), hipMemcpyDeviceToHost) !=
            hipSuccess) {
          ++failures;
          continue;
        }
        for (size_t i = 0; i < kSegments; ++i) {
          const size_t host = t * kSlice + ScatteredOffset(i, kSegments, stride);
          if (std::memcmp(seen.data() + i * kRestoreSegment, up.get() + host, kRestoreSegment) !=
              0) {
            ++mismatches;
          }
        }
        if (!engine.Transfer(to_host, &failed)) ++failures;
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(failures.load(), 0u);
  EXPECT_EQ(mismatches.load(), 0u);
  // Every slot's first segment went up and came back; the gaps stay untouched.
  for (size_t slot = 0; slot < kThreads * kSegments; ++slot) {
    const size_t host = slot * 2 * kRestoreSegment;
    EXPECT_EQ(std::memcmp(up.get() + host, down.get() + host, kRestoreSegment), 0)
        << "slot " << slot;
  }
}

// Copies `segments` fragments of `bytes` each between a registered host
// region and a device buffer, in the direction given, and reports how many
// gather kernels the transfer launched.  Fails the test on wrong bytes.
uint64_t GatherLaunchesFor(size_t segments, size_t bytes, bool to_device) {
  HostPages host(segments * bytes * 2);
  EXPECT_NE(host.get(), nullptr);
  if (host.get() == nullptr) return 0;
  DeviceBuffer device(segments * bytes);
  EXPECT_TRUE(device.valid());
  if (!device.valid()) return 0;
  std::vector<char> pattern(segments * bytes);
  for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<char>(i * 13 + 5);

  HbmCopyEngine engine;
  engine.AddHostGatherRegion(host.get(), host.size());
  const TransferRef h = HostRef(host.get(), host.size());
  const TransferRef g = GpuRef(device.get(), device.size());
  // Every other slot of the host region, in reverse, so nothing coalesces.
  std::vector<TransferItem> items;
  for (size_t i = 0; i < segments; ++i) {
    const size_t host_offset = (segments - 1 - i) * 2 * bytes;
    items.push_back(to_device ? MakeItem(h, host_offset, g, i * bytes, bytes, i)
                              : MakeItem(g, i * bytes, h, host_offset, bytes, i));
  }
  if (to_device) {
    for (size_t i = 0; i < segments; ++i) {
      std::memcpy(host.get() + (segments - 1 - i) * 2 * bytes, pattern.data() + i * bytes, bytes);
    }
  } else {
    EXPECT_EQ(hipMemcpy(device.get(), pattern.data(), pattern.size(), hipMemcpyHostToDevice),
              hipSuccess);
  }

  const uint64_t before = DeviceGatherLaunchCount();
  std::vector<size_t> failed;
  EXPECT_TRUE(engine.Transfer(items, &failed));
  const uint64_t launches = DeviceGatherLaunchCount() - before;

  std::vector<char> seen(pattern.size());
  if (to_device) {
    EXPECT_EQ(hipMemcpy(seen.data(), device.get(), seen.size(), hipMemcpyDeviceToHost), hipSuccess);
  } else {
    for (size_t i = 0; i < segments; ++i) {
      std::memcpy(seen.data() + i * bytes, host.get() + (segments - 1 - i) * 2 * bytes, bytes);
    }
  }
  EXPECT_EQ(std::memcmp(seen.data(), pattern.data(), pattern.size()), 0);
  return launches;
}

// The engine resolves a plan's device alias once, for the whole host endpoint.
// An endpoint larger than the registration that covers its segments fails that
// lookup and must fall back to one lookup per segment, still on the kernel.
TEST(HbmCopyEngine, GatherResolvesSegmentsWhenTheEndpointOutgrowsItsRegistration) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";

  constexpr size_t kSegments = 16;
  HostPages host(4 << 20);
  ASSERT_NE(host.get(), nullptr);
  for (size_t i = 0; i < host.size(); ++i) host.get()[i] = static_cast<char>(i * 17 + 1);
  DeviceBuffer device(kSegments * kRestoreSegment);
  ASSERT_TRUE(device.valid());

  HbmCopyEngine engine;
  engine.AddHostGatherRegion(host.get(), host.size() / 2);  // segments live in this half
  const TransferRef h = HostRef(host.get(), host.size());   // the endpoint names all of it
  const TransferRef g = GpuRef(device.get(), device.size());
  std::vector<TransferItem> items;
  for (size_t i = 0; i < kSegments; ++i) {
    items.push_back(
        MakeItem(h, ScatteredOffset(i, kSegments, 5), g, i * kRestoreSegment, kRestoreSegment, i));
  }
  const uint64_t before = DeviceGatherLaunchCount();
  std::vector<size_t> failed;
  ASSERT_TRUE(engine.Transfer(items, &failed));
  EXPECT_EQ(DeviceGatherLaunchCount(), before + 1);

  std::vector<char> seen(kSegments * kRestoreSegment);
  ASSERT_EQ(hipMemcpy(seen.data(), device.get(), seen.size(), hipMemcpyDeviceToHost), hipSuccess);
  for (size_t i = 0; i < kSegments; ++i) {
    EXPECT_EQ(std::memcmp(seen.data() + i * kRestoreSegment,
                          host.get() + ScatteredOffset(i, kSegments, 5), kRestoreSegment),
              0)
        << "segment " << i;
  }
}

// Restores of large per-layer state keep the kernel: with every GPU copying at
// once it outruns the copy engine at every fragment size measured.
TEST(HbmCopyEngine, LargeRestoreFragmentsStayOnTheGatherKernel) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";
  EXPECT_EQ(GatherLaunchesFor(4, 1 << 20, /*to_device=*/true), 1u);
  EXPECT_EQ(GatherLaunchesFor(3, 4 << 20, /*to_device=*/true), 1u);
}

// Offloads switch to the copy engine once the mean fragment reaches 4 MiB.
TEST(HbmCopyEngine, LargeOffloadFragmentsGoToTheCopyEngine) {
  if (!HaveGpu()) GTEST_SKIP() << "no GPU visible";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";
  EXPECT_EQ(GatherLaunchesFor(4, 1 << 20, /*to_device=*/false), 1u);
  EXPECT_EQ(GatherLaunchesFor(3, 4 << 20, /*to_device=*/false), 0u);
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

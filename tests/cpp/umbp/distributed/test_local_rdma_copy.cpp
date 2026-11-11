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
// test_local_rdma_copy — UMBP_LOCAL_RDMA_COPY end to end: a registered host
// buffer and a registered GPU buffer are copied through the composite engine
// with MoriIoEngine's NIC loopback, both ways, and the bytes are checked.
//
// Needs a GPU and an RDMA NIC.  It skips, rather than fails, on a machine
// without either: test_transfer_engine covers the planning without hardware.
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {
namespace {

// One KV pool in miniature: pages are contiguous in host memory, every layer of
// a page [layer][bytes]; the GPU keeps one tensor per layer, [layer][slot][bytes].
constexpr size_t kLayers = 8;
constexpr size_t kLayerBytes = 16384;
constexpr size_t kPages = 32;
constexpr size_t kPageBytes = kLayers * kLayerBytes;
constexpr size_t kPoolBytes = kPages * kPageBytes;

struct Fixture {
  void* host = nullptr;
  void* host_back = nullptr;
  void* gpu = nullptr;
  CompositeTransferEngine composite;
  MoriIoEngine* rdma = nullptr;
  TransferRef host_ref, back_ref, gpu_ref;

  ~Fixture() {
    if (gpu != nullptr) {
      composite.Deregister(gpu_ref);
      (void)hipFree(gpu);
    }
    if (host != nullptr) {
      composite.Deregister(host_ref);
      (void)hipHostFree(host);
    }
    if (host_back != nullptr) {
      composite.Deregister(back_ref);
      (void)hipHostFree(host_back);
    }
  }
};

// Returns an empty string on success, else why the test cannot run here.
std::string Build(Fixture* f, uint64_t min_item_bytes) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) return "no GPU";
  if (hipSetDevice(0) != hipSuccess) return "hipSetDevice failed";
  if (hipHostMalloc(&f->host, kPoolBytes) != hipSuccess ||
      hipHostMalloc(&f->host_back, kPoolBytes) != hipSuccess ||
      hipMalloc(&f->gpu, kPoolBytes) != hipSuccess) {
    return "allocation failed";
  }

  mori::io::IOEngineConfig io_cfg;
  io_cfg.host = "127.0.0.1";
  io_cfg.port = static_cast<uint16_t>(20000 + (getpid() % 2000));
  auto rdma = std::make_unique<MoriIoEngine>("local-rdma-copy-test", io_cfg, /*bounce_bytes=*/0);
  rdma->SetLocalLoopback(true, min_item_bytes);
  try {
    if (!rdma->Init()) return "MoriIoEngine::Init failed (no RDMA device?)";
  } catch (const std::exception& e) {
    return std::string("MoriIoEngine::Init threw: ") + e.what();
  }
  if (!rdma->LocalLoopback()) return "local loopback did not come up";
  f->rdma = rdma.get();

  // The same composition RealPoolPlatform builds.
  auto hbm = std::make_unique<HbmCopyEngine>();
  hbm->YieldRegisteredHostGpuPairs(true, min_item_bytes);
  f->composite.AddEngine(std::make_unique<LocalCopyEngine>());
  f->composite.AddEngine(std::move(hbm));
  f->composite.AddEngine(std::move(rdma));

  f->host_ref = f->composite.RegisterMemory(f->host, kPoolBytes, mori::io::MemoryLocationType::CPU,
                                            /*device=*/-1);
  f->back_ref = f->composite.RegisterMemory(f->host_back, kPoolBytes,
                                            mori::io::MemoryLocationType::CPU, /*device=*/-1);
  f->gpu_ref = f->composite.RegisterMemory(f->gpu, kPoolBytes, mori::io::MemoryLocationType::GPU,
                                           /*device=*/0);
  if (!f->host_ref.HasMemoryDesc() || !f->back_ref.HasMemoryDesc() || !f->gpu_ref.HasMemoryDesc()) {
    return "memory was not registered with mori-io";
  }
  return {};
}

TransferItem Item(const TransferRef& src, uint64_t src_off, const TransferRef& dst,
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

// Load one layer group (all kLayers here) of every page: host pages are picked at
// random, GPU slots too, so no two fragments are adjacent on the GPU side.
TEST(LocalRdmaCopy, MovesScatteredKvFragmentsBothWaysThroughTheNic) {
  Fixture f;
  const std::string why = Build(&f, /*min_item_bytes=*/0);  // every item on the NIC
  if (!why.empty()) GTEST_SKIP() << why;

  std::mt19937_64 rng(11);
  auto* h = static_cast<uint64_t*>(f.host);
  for (size_t i = 0; i < kPoolBytes / 8; ++i) h[i] = rng();
  std::memset(f.host_back, 0, kPoolBytes);
  ASSERT_EQ(hipSuccess, hipMemset(f.gpu, 0, kPoolBytes));

  std::vector<size_t> pages(kPages), slots(kPages);
  std::iota(pages.begin(), pages.end(), size_t{0});
  std::iota(slots.begin(), slots.end(), size_t{0});
  std::shuffle(pages.begin(), pages.end(), rng);
  std::shuffle(slots.begin(), slots.end(), rng);

  std::vector<TransferItem> load, offload;
  size_t tag = 0;
  for (size_t p = 0; p < kPages; ++p) {
    for (size_t layer = 0; layer < kLayers; ++layer) {
      const uint64_t host_off = pages[p] * kPageBytes + layer * kLayerBytes;
      const uint64_t gpu_off = (layer * kPages + slots[p]) * kLayerBytes;
      load.push_back(Item(f.host_ref, host_off, f.gpu_ref, gpu_off, kLayerBytes, tag));
      offload.push_back(Item(f.gpu_ref, gpu_off, f.back_ref, host_off, kLayerBytes, tag));
      ++tag;
    }
  }

  // Both directions must be planned onto the NIC, not the copy engine.
  for (const auto* items : {&load, &offload}) {
    TransferPlanSet planned = f.composite.Plan(*items);
    EXPECT_TRUE(planned.rejected_tags.empty());
    ASSERT_FALSE(planned.plans.empty());
    for (const auto& plan : planned.plans) {
      ASSERT_NE(nullptr, plan.engine);
      EXPECT_STREQ("MoriIoEngine", plan.engine->Name());
    }
    if (items == &load) {
      EXPECT_TRUE(planned.plans.front().dir == TransferDirection::kPull);
    } else {
      EXPECT_TRUE(planned.plans.front().dir == TransferDirection::kPush);
    }
  }

  std::vector<size_t> failed;
  ASSERT_TRUE(f.composite.Transfer(load, &failed)) << failed.size() << " items failed";
  ASSERT_TRUE(failed.empty());
  ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

  // GPU side: every fragment landed where the item said.
  std::vector<char> gpu_bytes(kPoolBytes);
  ASSERT_EQ(hipSuccess, hipMemcpy(gpu_bytes.data(), f.gpu, kPoolBytes, hipMemcpyDeviceToHost));
  for (size_t p = 0; p < kPages; ++p) {
    for (size_t layer = 0; layer < kLayers; ++layer) {
      const uint64_t host_off = pages[p] * kPageBytes + layer * kLayerBytes;
      const uint64_t gpu_off = (layer * kPages + slots[p]) * kLayerBytes;
      ASSERT_EQ(0, std::memcmp(gpu_bytes.data() + gpu_off, static_cast<char*>(f.host) + host_off,
                               kLayerBytes))
          << "load: page " << p << " layer " << layer;
    }
  }

  ASSERT_TRUE(f.composite.Transfer(offload, &failed)) << failed.size() << " items failed";
  ASSERT_TRUE(failed.empty());
  for (size_t p = 0; p < kPages; ++p) {
    const uint64_t host_off = pages[p] * kPageBytes;
    ASSERT_EQ(0, std::memcmp(static_cast<char*>(f.host_back) + host_off,
                             static_cast<char*>(f.host) + host_off, kPageBytes))
        << "offload: page " << p;
  }
}

// One batch, two fragment sizes, a 32 KiB floor: the small items must take the
// copy engine and the large ones the NIC, and every byte must still arrive.
TEST(LocalRdmaCopy, SplitsOneBatchBySizeBetweenTheCopyEngineAndTheNic) {
  constexpr uint64_t kFloor = 32 << 10;
  constexpr size_t kBig = 64 << 10, kSmall = 4 << 10, kCount = 16;
  Fixture f;
  const std::string why = Build(&f, kFloor);
  if (!why.empty()) GTEST_SKIP() << why;

  std::mt19937_64 rng(5);
  auto* h = static_cast<uint64_t*>(f.host);
  for (size_t i = 0; i < kPoolBytes / 8; ++i) h[i] = rng();
  ASSERT_EQ(hipSuccess, hipMemset(f.gpu, 0, kPoolBytes));

  // Big items fill the first half of the GPU buffer, small ones are strided
  // through the second half; host sources are scattered the same way.
  std::vector<TransferItem> items;
  std::vector<std::pair<uint64_t, uint64_t>> placed;  // (host_off, gpu_off)
  size_t tag = 0;
  for (size_t i = 0; i < kCount; ++i) {
    const uint64_t host_off = (kCount - 1 - i) * kBig;
    const uint64_t gpu_off = i * kBig;
    items.push_back(Item(f.host_ref, host_off, f.gpu_ref, gpu_off, kBig, tag++));
    placed.emplace_back(host_off, gpu_off);
  }
  for (size_t i = 0; i < kCount; ++i) {
    const uint64_t host_off = kPoolBytes / 2 + i * 2 * kSmall;
    const uint64_t gpu_off = kPoolBytes / 2 + i * 3 * kSmall;
    items.push_back(Item(f.host_ref, host_off, f.gpu_ref, gpu_off, kSmall, tag++));
    placed.emplace_back(host_off, gpu_off);
  }

  TransferPlanSet planned = f.composite.Plan(items);
  EXPECT_TRUE(planned.rejected_tags.empty());
  size_t nic_items = 0, copy_items = 0;
  for (const auto& plan : planned.plans) {
    ASSERT_NE(nullptr, plan.engine);
    const std::string name = plan.engine->Name();
    if (name == "MoriIoEngine") {
      nic_items += plan.tags.size();
    } else if (name == "HbmCopyEngine") {
      copy_items += plan.tags.size();
    } else {
      ADD_FAILURE() << "unexpected engine " << name;
    }
  }
  EXPECT_EQ(kCount, nic_items);
  EXPECT_EQ(kCount, copy_items);

  std::vector<size_t> failed;
  ASSERT_TRUE(f.composite.Transfer(std::move(planned), &failed)) << failed.size() << " failed";
  ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

  std::vector<char> gpu_bytes(kPoolBytes);
  ASSERT_EQ(hipSuccess, hipMemcpy(gpu_bytes.data(), f.gpu, kPoolBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < items.size(); ++i) {
    ASSERT_EQ(0, std::memcmp(gpu_bytes.data() + placed[i].second,
                             static_cast<char*>(f.host) + placed[i].first, items[i].size))
        << "item " << i << " (" << items[i].size << " B)";
  }
}

}  // namespace
}  // namespace mori::umbp

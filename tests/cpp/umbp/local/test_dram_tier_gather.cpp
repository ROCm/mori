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
//
// The GPU gather path for page-granular objects. Registered twice in ctest,
// once with UMBP_DRAM_GATHER_KERNEL=0, so the same byte-level expectations are
// enforced against both the kernel and the copy engine. The launch counter
// assertions are what stop this from passing when the kernel silently never
// runs -- on any machine that can register host memory for GPU access, which is
// the one precondition the kernel path has and the one thing a test cannot
// assert its way into.
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "device_gather.h"
#include "host_registration.h"
#include "umbp/local/tiers/dram_tier.h"

namespace mori::umbp {
namespace {

bool GatherKernelExpected() {
  const char* value = std::getenv("UMBP_DRAM_GATHER_KERNEL");
  return value == nullptr || (std::string(value) != "0");
}

// Whether this machine can make host memory GPU-addressable at all. The tier
// registers itself exactly this way, and where the registration does not take
// the tier falls back to the copy engine on purpose -- so the launch counter
// below would be measuring the machine rather than the planner's decision.
// Probed once; the reason for a failure is logged by the registration itself at
// warn level.
bool GatherPathAvailable() {
  static const bool available = [] {
    std::vector<char> probe(64 * 1024);
    return HostTierRegistration(probe.data(), probe.size()).RegisteredBytes() != 0;
  }();
  return available;
}

// The kernel-off variant asserts the absence of launches and holds anywhere;
// only the kernel-on variant needs the registration to have taken.
bool GatherKernelReachable() { return !GatherKernelExpected() || GatherPathAvailable(); }

constexpr const char* kGatherPathUnreachable =
    "host memory cannot be registered for GPU access on this machine, so the tier stays on the "
    "copy engine; rerun with MORI_UMBP_LOG_LEVEL=warn for the reason the registration failed";

bool HasDevice() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return count > 0;
}

// Distinct per byte so a fragment landing at the wrong offset cannot match.
char ByteAt(size_t object, size_t offset) {
  return static_cast<char>((object * 131u + offset * 17u + (offset >> 8)) & 0xff);
}

// hipMemset on device memory is asynchronous with respect to the host, and the
// tier copies on a non-blocking stream that the null stream does not order
// against. Without this the zeroing can land after the copy under test.
void ZeroDevice(void* device, size_t bytes) {
  ASSERT_EQ(hipMemset(device, 0, bytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

// The production shape: `objects` page objects of `layers` layers each, one
// layer read out of every object into one contiguous device buffer.
class PageObjectTier {
 public:
  PageObjectTier(size_t objects, size_t layers, size_t layer_bytes)
      : objects_(objects),
        layers_(layers),
        layer_bytes_(layer_bytes),
        object_bytes_(layers * layer_bytes),
        tier_(objects * layers * layer_bytes * 4) {
    for (size_t object = 0; object < objects_; ++object) {
      std::vector<char> payload(object_bytes_);
      for (size_t i = 0; i < object_bytes_; ++i) payload[i] = ByteAt(object, i);
      EXPECT_TRUE(tier_.Write(Key(object), payload.data(), payload.size()));
    }
  }

  std::string Key(size_t object) const { return "page_" + std::to_string(object); }

  DRAMTier& tier() { return tier_; }
  size_t objects() const { return objects_; }
  size_t layer_bytes() const { return layer_bytes_; }
  size_t object_bytes() const { return object_bytes_; }

  // Reads layer `layer` of every object into consecutive slots of `device`.
  std::vector<bool> ReadLayer(size_t layer, char* device) {
    std::vector<std::string> keys;
    std::vector<std::vector<uintptr_t>> destinations;
    std::vector<std::vector<size_t>> sizes;
    std::vector<std::vector<size_t>> offsets;
    for (size_t object = 0; object < objects_; ++object) {
      keys.push_back(Key(object));
      destinations.push_back({reinterpret_cast<uintptr_t>(device + object * layer_bytes_)});
      sizes.push_back({layer_bytes_});
      offsets.push_back({layer * layer_bytes_});
    }
    return tier_.ReadBatchRangesIntoPtr(keys, destinations, sizes, offsets);
  }

  std::vector<char> ExpectedLayer(size_t layer) const {
    std::vector<char> expected(objects_ * layer_bytes_);
    for (size_t object = 0; object < objects_; ++object) {
      for (size_t i = 0; i < layer_bytes_; ++i) {
        expected[object * layer_bytes_ + i] = ByteAt(object, layer * layer_bytes_ + i);
      }
    }
    return expected;
  }

 private:
  size_t objects_;
  size_t layers_;
  size_t layer_bytes_;
  size_t object_bytes_;
  DRAMTier tier_;
};

TEST(DeviceGatherTest, StridedLayerReadIsByteCorrect) {
  if (!HasDevice()) GTEST_SKIP() << "No HIP device available";
  if (!GatherKernelReachable()) GTEST_SKIP() << kGatherPathUnreachable;

  constexpr size_t kObjects = 32;
  constexpr size_t kLayers = 8;
  constexpr size_t kLayerBytes = 4096;
  PageObjectTier fixture(kObjects, kLayers, kLayerBytes);

  char* device = nullptr;
  ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&device), kObjects * kLayerBytes), hipSuccess);

  for (size_t layer = 0; layer < kLayers; ++layer) {
    ZeroDevice(device, kObjects * kLayerBytes);
    const uint64_t launches_before = DeviceGatherLaunchCount();

    EXPECT_EQ(fixture.ReadLayer(layer, device), std::vector<bool>(kObjects, true));

    // 32 fragments of 4 KiB is well inside the fragmented regime, so the path
    // is not a matter of tuning: it must be the kernel unless disabled.
    if (GatherKernelExpected()) {
      EXPECT_GT(DeviceGatherLaunchCount(), launches_before);
    } else {
      EXPECT_EQ(DeviceGatherLaunchCount(), launches_before);
    }

    std::vector<char> readback(kObjects * kLayerBytes, 0);
    ASSERT_EQ(hipMemcpy(readback.data(), device, readback.size(), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(readback, fixture.ExpectedLayer(layer)) << "layer " << layer;
  }

  EXPECT_EQ(hipFree(device), hipSuccess);
}

TEST(DeviceGatherTest, UnalignedFragmentsAreByteCorrect) {
  if (!HasDevice()) GTEST_SKIP() << "No HIP device available";
  if (!GatherKernelReachable()) GTEST_SKIP() << kGatherPathUnreachable;

  // Neither the size nor the resulting offsets are multiples of 16, which sends
  // the kernel down its byte loop instead of the vectorized one.
  constexpr size_t kObjects = 16;
  constexpr size_t kLayers = 5;
  constexpr size_t kLayerBytes = 3001;
  PageObjectTier fixture(kObjects, kLayers, kLayerBytes);

  char* device = nullptr;
  ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&device), kObjects * kLayerBytes), hipSuccess);
  ZeroDevice(device, kObjects * kLayerBytes);

  const uint64_t launches_before = DeviceGatherLaunchCount();
  EXPECT_EQ(fixture.ReadLayer(3, device), std::vector<bool>(kObjects, true));
  if (GatherKernelExpected()) EXPECT_GT(DeviceGatherLaunchCount(), launches_before);

  std::vector<char> readback(kObjects * kLayerBytes, 0);
  ASSERT_EQ(hipMemcpy(readback.data(), device, readback.size(), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(readback, fixture.ExpectedLayer(3));

  EXPECT_EQ(hipFree(device), hipSuccess);
}

TEST(DeviceGatherTest, StridedOffloadWriteIsByteCorrect) {
  if (!HasDevice()) GTEST_SKIP() << "No HIP device available";
  if (!GatherKernelReachable()) GTEST_SKIP() << kGatherPathUnreachable;

  // The offload direction: contiguous device source scattered into one layer
  // slot of each page object.
  constexpr size_t kObjects = 24;
  constexpr size_t kLayers = 6;
  constexpr size_t kLayerBytes = 2048;
  constexpr size_t kObjectBytes = kLayers * kLayerBytes;

  DRAMTier tier(kObjects * kObjectBytes * 4);

  // The device pool is layer-major and the tier object is page-major, which is
  // exactly what makes the offload a scatter: consecutive device fragments land
  // one object size apart in the tier, so no run merges.
  auto device_offset = [](size_t object, size_t layer) {
    return layer * kObjects * kLayerBytes + object * kLayerBytes;
  };
  std::vector<char> payload(kObjects * kObjectBytes);
  for (size_t object = 0; object < kObjects; ++object) {
    for (size_t layer = 0; layer < kLayers; ++layer) {
      for (size_t i = 0; i < kLayerBytes; ++i) {
        payload[device_offset(object, layer) + i] = ByteAt(object, layer * kLayerBytes + i);
      }
    }
  }

  char* device = nullptr;
  ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&device), payload.size()), hipSuccess);
  ASSERT_EQ(hipMemcpy(device, payload.data(), payload.size(), hipMemcpyHostToDevice), hipSuccess);

  std::vector<std::string> keys;
  std::vector<size_t> object_sizes;
  std::vector<std::vector<const void*>> sources;
  std::vector<std::vector<size_t>> sizes;
  std::vector<std::vector<size_t>> offsets;
  for (size_t object = 0; object < kObjects; ++object) {
    keys.push_back("offload_" + std::to_string(object));
    object_sizes.push_back(kObjectBytes);
    std::vector<const void*> object_sources;
    std::vector<size_t> object_sizes_per_layer;
    std::vector<size_t> object_offsets;
    for (size_t layer = 0; layer < kLayers; ++layer) {
      object_sources.push_back(device + device_offset(object, layer));
      object_sizes_per_layer.push_back(kLayerBytes);
      object_offsets.push_back(layer * kLayerBytes);
    }
    sources.push_back(object_sources);
    sizes.push_back(object_sizes_per_layer);
    offsets.push_back(object_offsets);
  }

  const uint64_t launches_before = DeviceGatherLaunchCount();
  EXPECT_EQ(tier.BatchWriteRanges(keys, object_sizes, sources, sizes, offsets),
            std::vector<bool>(kObjects, true));
  if (GatherKernelExpected()) EXPECT_GT(DeviceGatherLaunchCount(), launches_before);

  for (size_t object = 0; object < kObjects; ++object) {
    std::vector<char> expected(kObjectBytes);
    for (size_t i = 0; i < kObjectBytes; ++i) expected[i] = ByteAt(object, i);
    std::vector<char> readback(kObjectBytes, 0);
    ASSERT_TRUE(tier.ReadIntoPtr(keys[object], reinterpret_cast<uintptr_t>(readback.data()),
                                 readback.size()));
    EXPECT_EQ(readback, expected) << "object " << object;
  }

  EXPECT_EQ(hipFree(device), hipSuccess);
}

TEST(DeviceGatherTest, OneLargeContiguousRunStaysOnTheCopyEngine) {
  if (!HasDevice()) GTEST_SKIP() << "No HIP device available";

  // A single object read whole is one run of 512 KiB, past the point where a
  // kernel beats hipMemcpyAsync. Guards the threshold from drifting to "always
  // use the kernel", which measured slower for this shape.
  constexpr size_t kBytes = 512 * 1024;
  DRAMTier tier(kBytes * 4);
  std::vector<char> payload(kBytes);
  for (size_t i = 0; i < kBytes; ++i) payload[i] = ByteAt(7, i);
  ASSERT_TRUE(tier.Write("bulk", payload.data(), payload.size()));

  char* device = nullptr;
  ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&device), kBytes), hipSuccess);
  ZeroDevice(device, kBytes);

  const uint64_t launches_before = DeviceGatherLaunchCount();
  EXPECT_EQ(tier.ReadBatchIntoPtr({"bulk"}, {reinterpret_cast<uintptr_t>(device)}, {kBytes}),
            std::vector<bool>({true}));
  EXPECT_EQ(DeviceGatherLaunchCount(), launches_before);

  std::vector<char> readback(kBytes, 0);
  ASSERT_EQ(hipMemcpy(readback.data(), device, kBytes, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(readback, payload);

  EXPECT_EQ(hipFree(device), hipSuccess);
}

TEST(HostTierRegistrationTest, CoversOnlyRegisteredBytesAndRejectsOverflow) {
  std::vector<char> region(64 * 1024);
  HostTierRegistration registration(region.data(), region.size());

  if (registration.RegisteredBytes() == 0) {
    // No device, or registration unavailable: the safe answer is always false,
    // which is exactly what keeps the kernel path off.
    EXPECT_FALSE(registration.Covers(region.data(), 1));
    return;
  }

  // All-or-nothing, never a partial prefix. Registering the region in pieces
  // makes hipMemcpy reject any copy that straddles two of them, which in the
  // tier surfaces as a write failure that the storage manager mistakes for a
  // full tier and answers by evicting everything.
  ASSERT_EQ(registration.RegisteredBytes(), region.size());
  EXPECT_TRUE(registration.Covers(region.data(), region.size()));
  EXPECT_TRUE(registration.Covers(region.data() + region.size() - 1, 1));
  EXPECT_FALSE(registration.Covers(region.data(), region.size() + 1));
  EXPECT_FALSE(registration.Covers(region.data() + region.size(), 1));
  // Built from an integer, not as `region.data() - 1`: forming a pointer before
  // the start of an array is undefined even without dereferencing it.
  EXPECT_FALSE(registration.Covers(
      reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(region.data()) - 1), 1));
  EXPECT_FALSE(registration.Covers(nullptr, 1));
  // A size that would wrap when added to the offset must not read as covered.
  EXPECT_FALSE(registration.Covers(region.data(), static_cast<size_t>(-1)));
}

}  // namespace
}  // namespace mori::umbp

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
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "umbp/common/aligned_buffer.h"
#include "umbp/common/device_gather.h"
#include "umbp/common/host_registration.h"
#include "umbp/distributed/transfer/restore_scheduler.h"
#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {
namespace {

// Rejection tests never dereference these endpoint addresses or initialize HIP.
TransferPlan HostToDevicePlan(int destination = 0) {
  TransferPlan plan;
  plan.src = TransferRef::HostBytes(reinterpret_cast<void*>(0x1000), 8);
  plan.dst = TransferRef::HostBytes(reinterpret_cast<void*>(0x2000), 8,
                                    mori::io::MemoryLocationType::GPU, destination);
  plan.src_offsets = {0};
  plan.dst_offsets = {0};
  plan.sizes = {8};
  return plan;
}

void* UnexpectedMapping(const void*, size_t, int) {
  ADD_FAILURE() << "Rejected transfers must not map host memory or submit GPU work";
  return nullptr;
}

TEST(RestoreScheduler, RejectsEmptyOrDisabledTransfers) {
  EXPECT_FALSE(TryStripedRestore({}, UnexpectedMapping, 4));
  EXPECT_FALSE(TryStripedRestore({HostToDevicePlan()}, UnexpectedMapping, 0));
}

TEST(RestoreScheduler, RejectsDirectionsOtherThanHostToDevice) {
  auto plan = HostToDevicePlan();
  plan.dst.loc = mori::io::MemoryLocationType::CPU;
  EXPECT_FALSE(TryStripedRestore({plan}, UnexpectedMapping, 4));
  plan.src.loc = mori::io::MemoryLocationType::GPU;
  EXPECT_FALSE(TryStripedRestore({plan}, UnexpectedMapping, 4));
  plan.dst.loc = mori::io::MemoryLocationType::GPU;
  EXPECT_FALSE(TryStripedRestore({plan}, UnexpectedMapping, 4));
}

TEST(RestoreScheduler, RejectsBatchesWithMultipleDestinations) {
  EXPECT_FALSE(TryStripedRestore({HostToDevicePlan(0), HostToDevicePlan(1)}, UnexpectedMapping, 4));
}

TEST(RestoreScheduler, RejectsIncompleteRangeDescriptors) {
  auto plan = HostToDevicePlan();
  plan.src_offsets.clear();
  EXPECT_FALSE(TryStripedRestore({plan}, UnexpectedMapping, 4));
  plan = HostToDevicePlan();
  plan.dst_offsets.clear();
  EXPECT_FALSE(TryStripedRestore({plan}, UnexpectedMapping, 4));
}

TEST(RestoreScheduler, RejectsBatchesWithNoBytes) {
  auto plan = HostToDevicePlan();
  plan.sizes = {0};
  EXPECT_FALSE(TryStripedRestore({plan}, UnexpectedMapping, 4));
}

TEST(RestoreScheduler, RestoresHostDataIntoDestinationGpu) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) GTEST_SKIP() << "no GPU";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  constexpr size_t kBytes = 1 << 20;
  constexpr size_t kHalf = kBytes / 2;
  AlignedBuffer source(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    source.data()[i] = static_cast<char>((i * 37 + i / 4096) % 251);
  HostTierRegistration registration(source.data(), source.size());
  if (!registration.GatherableOn(0)) GTEST_SKIP() << "host region not gatherable";

  void* ptr = nullptr;
  ASSERT_EQ(hipMalloc(&ptr, kBytes), hipSuccess);
  std::unique_ptr<void, decltype(&hipFree)> destination(ptr, hipFree);
  ASSERT_EQ(hipMemset(destination.get(), 0, kBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  TransferPlan plan;
  plan.src = TransferRef::HostBytes(source.data(), source.size());
  plan.dst =
      TransferRef::HostBytes(destination.get(), kBytes, mori::io::MemoryLocationType::GPU, 0);
  // Swap the two halves to check that source and destination offsets are honored.
  plan.src_offsets = {0, kHalf};
  plan.dst_offsets = {kHalf, 0};
  plan.sizes = {kHalf, kHalf};
  auto map_host = [&](const void* host, size_t bytes, int device) {
    return registration.DeviceAddress(host, bytes, device);
  };
  // Call the striped path directly: there is no copy-engine fallback in this test.
  ASSERT_TRUE(TryStripedRestore({plan}, map_host, 64 << 10));

  std::vector<char> copied(kBytes);
  ASSERT_EQ(hipMemcpy(copied.data(), destination.get(), kBytes, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(std::memcmp(copied.data(), source.data() + kHalf, kHalf), 0);
  EXPECT_EQ(std::memcmp(copied.data() + kHalf, source.data(), kHalf), 0);
}

TEST(RestoreScheduler, ConcurrentRestoresReachTheirOwnDestinations) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 2) GTEST_SKIP() << "needs two GPUs";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";

  constexpr size_t kBytes = 8 << 20;
  AlignedBuffer sources[] = {AlignedBuffer(kBytes), AlignedBuffer(kBytes)};
  for (int destination = 0; destination < 2; ++destination)
    for (size_t i = 0; i < kBytes; ++i)
      sources[destination].data()[i] = static_cast<char>((i + destination) % 251);
  HostTierRegistration registrations[] = {{sources[0].data(), kBytes}, {sources[1].data(), kBytes}};
  if (!registrations[0].GatherableOn(0) || !registrations[1].GatherableOn(1))
    GTEST_SKIP() << "host regions not gatherable";

  std::unique_ptr<void, decltype(&hipFree)> destinations[] = {{nullptr, hipFree},
                                                              {nullptr, hipFree}};
  TransferPlan plans[2];
  // Finish setup before starting either thread, so setup failures cannot strand a caller.
  for (int destination = 0; destination < 2; ++destination) {
    ASSERT_EQ(hipSetDevice(destination), hipSuccess);
    void* ptr = nullptr;
    ASSERT_EQ(hipMalloc(&ptr, kBytes), hipSuccess);
    destinations[destination].reset(ptr);
    ASSERT_EQ(hipMemset(ptr, 0, kBytes), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto& plan = plans[destination];
    plan.src = TransferRef::HostBytes(sources[destination].data(), kBytes);
    plan.dst = TransferRef::HostBytes(ptr, kBytes, mori::io::MemoryLocationType::GPU, destination);
    plan.src_offsets = {0};
    plan.dst_offsets = {0};
    plan.sizes = {kBytes};
  }

  std::atomic<int> ready{0};
  auto restore = [&](int destination) {
    ready.fetch_add(1);
    while (ready.load() != 2) std::this_thread::yield();
    auto map_host = [&](const void* host, size_t bytes, int device) {
      return registrations[destination].DeviceAddress(host, bytes, device);
    };
    ASSERT_TRUE(TryStripedRestore({plans[destination]}, map_host, 64 << 10));
  };
  std::thread first(restore, 0);
  std::thread second(restore, 1);
  first.join();
  second.join();

  std::vector<char> copied(kBytes);
  for (int destination = 0; destination < 2; ++destination) {
    ASSERT_EQ(hipSetDevice(destination), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(copied.data(), destinations[destination].get(), kBytes, hipMemcpyDeviceToHost),
        hipSuccess);
    EXPECT_EQ(std::memcmp(copied.data(), sources[destination].data(), kBytes), 0) << destination;
  }
}

TEST(RestoreScheduler, StripedRestoreDrainsBeforeDirectFallback) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 2) GTEST_SKIP() << "needs two GPUs";
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "gather kernel disabled";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  constexpr size_t kChunk = 64 << 10;
  AlignedBuffer source(kChunk), missing(kChunk);
  std::memset(source.data(), 0x31, kChunk);
  std::memset(missing.data(), 0x72, kChunk);
  HostTierRegistration registration(source.data(), source.size());
  if (!registration.GatherableOn(0)) GTEST_SKIP() << "host region not gatherable";

  void* ptr = nullptr;
  ASSERT_EQ(hipMalloc(&ptr, 2 * kChunk), hipSuccess);
  std::unique_ptr<void, decltype(&hipFree)> destination(ptr, hipFree);
  ASSERT_EQ(hipMemset(ptr, 0, 2 * kChunk), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  TransferPlan first;
  first.src = TransferRef::HostBytes(source.data(), kChunk);
  first.dst = TransferRef::HostBytes(ptr, 2 * kChunk, mori::io::MemoryLocationType::GPU, 0);
  first.src_offsets = {0};
  first.dst_offsets = {0};
  first.sizes = {kChunk};
  TransferPlan second = first;
  second.src = TransferRef::HostBytes(missing.data(), kChunk);
  second.dst_offsets = {kChunk};

  const auto launches_before = DeviceGatherLaunchCount();
  bool failed_after_submission = false;
  auto map_host = [&](const void* host, size_t bytes, int device) {
    void* alias = registration.DeviceAddress(host, bytes, device);
    if (alias == nullptr) failed_after_submission = DeviceGatherLaunchCount() > launches_before;
    return alias;
  };
  // With at least two GPUs, the second chunk fails while the first is still
  // tracked as in flight. Assert that this cannot pass by rejecting all work.
  ASSERT_FALSE(TryStripedRestore({first, second}, map_host, kChunk));
  ASSERT_TRUE(failed_after_submission);
  EXPECT_EQ(DeviceGatherLaunchCount(), launches_before + 1);

  // Verify the submitted chunk completed; the unmapped chunk was not copied.
  std::vector<char> copied(2 * kChunk), expected(2 * kChunk, 0);
  std::memcpy(expected.data(), source.data(), kChunk);
  ASSERT_EQ(hipMemcpy(copied.data(), ptr, copied.size(), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(copied, expected);

  // The caller can now safely overwrite the destination via direct fallback.
  ASSERT_EQ(hipMemcpy(ptr, source.data(), kChunk, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(
      hipMemcpy(static_cast<char*>(ptr) + kChunk, missing.data(), kChunk, hipMemcpyHostToDevice),
      hipSuccess);
  std::memcpy(expected.data() + kChunk, missing.data(), kChunk);
  ASSERT_EQ(hipMemcpy(copied.data(), ptr, copied.size(), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(copied, expected);
}

}  // namespace
}  // namespace mori::umbp

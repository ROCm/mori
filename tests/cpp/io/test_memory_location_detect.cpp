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

// IOEngine::RegisterMemory's MemoryLocationType::Unknown auto-detection.
//
// Callers that hold an opaque pointer (UMBP's PoolClient, for one) cannot say
// whether it is host or device memory.  Passing Unknown makes the engine work
// it out; the alternative -- guessing CPU -- silently misinforms NIC selection
// and sends the NUMA probe at a device VA.

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdlib>

#include "mori/io/engine.hpp"

using namespace mori::io;

namespace {

bool HasGpu() {
  int n = 0;
  const hipError_t err = hipGetDeviceCount(&n);
  (void)hipGetLastError();
  return err == hipSuccess && n > 0;
}

// No backend is created: registration is pure bookkeeping (the RDMA MR is built
// lazily on first transfer), so classification is observable without a NIC.
IOEngine MakeEngine() { return IOEngine("detect-test", IOEngineConfig{"127.0.0.1", 0}); }

}  // namespace

TEST(MemoryLocationDetect, HostPointerDetectedAsCpu) {
  IOEngine engine = MakeEngine();
  constexpr size_t kSize = 4096;
  void* host = std::malloc(kSize);
  ASSERT_NE(host, nullptr);

  const MemoryDesc desc = engine.RegisterMemory(host, kSize, -1, MemoryLocationType::Unknown);
  EXPECT_EQ(desc.loc, MemoryLocationType::CPU);

  engine.DeregisterMemory(desc);
  std::free(host);
}

// The probe fails on host pointers and leaves the sticky per-thread hip error
// set.  If RegisterMemory does not clear it, the next unrelated hip call in the
// process reports a failure it did not cause -- so assert the clear explicitly.
TEST(MemoryLocationDetect, HostPointerLeavesNoStickyHipError) {
  IOEngine engine = MakeEngine();
  constexpr size_t kSize = 4096;
  void* host = std::malloc(kSize);
  ASSERT_NE(host, nullptr);

  (void)hipGetLastError();  // start from a clean slate
  const MemoryDesc desc = engine.RegisterMemory(host, kSize, -1, MemoryLocationType::Unknown);
  EXPECT_EQ(hipGetLastError(), hipSuccess);

  engine.DeregisterMemory(desc);
  std::free(host);
}

TEST(MemoryLocationDetect, DevicePointerDetectedAsGpuWithOrdinal) {
  if (!HasGpu()) GTEST_SKIP() << "no HIP device";
  IOEngine engine = MakeEngine();
  constexpr size_t kSize = 4096;
  void* dev = nullptr;
  ASSERT_EQ(hipMalloc(&dev, kSize), hipSuccess);

  // device = -1: the ordinal is unknown to the caller too, and detection is
  // expected to fill it in.
  const MemoryDesc desc = engine.RegisterMemory(dev, kSize, -1, MemoryLocationType::Unknown);
  EXPECT_EQ(desc.loc, MemoryLocationType::GPU);
  EXPECT_GE(desc.deviceId, 0);
  EXPECT_FALSE(desc.deviceBusId.empty());

  engine.DeregisterMemory(desc);
  ASSERT_EQ(hipFree(dev), hipSuccess);
}

// An explicitly-supplied ordinal must survive detection: only an unset (< 0)
// one is filled in.
TEST(MemoryLocationDetect, ExplicitDeviceOrdinalIsNotOverwritten) {
  if (!HasGpu()) GTEST_SKIP() << "no HIP device";
  IOEngine engine = MakeEngine();
  constexpr size_t kSize = 4096;
  void* dev = nullptr;
  ASSERT_EQ(hipMalloc(&dev, kSize), hipSuccess);

  const MemoryDesc desc = engine.RegisterMemory(dev, kSize, 0, MemoryLocationType::Unknown);
  EXPECT_EQ(desc.loc, MemoryLocationType::GPU);
  EXPECT_EQ(desc.deviceId, 0);

  engine.DeregisterMemory(desc);
  ASSERT_EQ(hipFree(dev), hipSuccess);
}

// Backward compatibility: an explicit location is taken at face value and never
// re-derived, so no existing caller changes behaviour.  Registering device
// memory as CPU stays possible -- that is exactly what UMBP did before this.
TEST(MemoryLocationDetect, ExplicitLocationIsNotSecondGuessed) {
  if (!HasGpu()) GTEST_SKIP() << "no HIP device";
  IOEngine engine = MakeEngine();
  constexpr size_t kSize = 4096;
  void* dev = nullptr;
  ASSERT_EQ(hipMalloc(&dev, kSize), hipSuccess);

  const MemoryDesc desc = engine.RegisterMemory(dev, kSize, -1, MemoryLocationType::CPU);
  EXPECT_EQ(desc.loc, MemoryLocationType::CPU);

  engine.DeregisterMemory(desc);
  ASSERT_EQ(hipFree(dev), hipSuccess);
}

TEST(MemoryLocationDetect, NullPointerWithUnknownStaysUnknown) {
  IOEngine engine = MakeEngine();
  // Nothing to probe, so detection is skipped rather than guessing.
  const MemoryDesc desc = engine.RegisterMemory(nullptr, 0, -1, MemoryLocationType::Unknown);
  EXPECT_EQ(desc.loc, MemoryLocationType::Unknown);
  engine.DeregisterMemory(desc);
}

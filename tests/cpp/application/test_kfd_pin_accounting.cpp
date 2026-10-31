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

// Registering HIP VRAM for RDMA must leave the free VRAM that hipMemGetInfo
// reports, and new allocations are checked against, unchanged: whichever path
// registers the MRs, however many cover one allocation, and in whatever order
// they and their contexts go away. amdgpu-dkms 6.14/6.16 count a dma-buf pin
// twice and add every peer-memory pin back.
//
// Needs a GPU and two RDMA devices with peer-memory support; pick the devices
// with MORI_RDMA_DEVICES.
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>

#include <memory>
#include <vector>

#include "infiniband/verbs.h"
#include "mori/application/transport/rdma/rdma.hpp"

using namespace mori::application;

namespace {

// The workaround is opt-in; turn it on unless the caller chose a setting, e.g.
// MORI_KFD_PIN_FIX=0 to watch these tests fail on an affected driver.
[[maybe_unused]] const bool kPinFixRequested =
    setenv("MORI_KFD_PIN_FIX", "1", /*overwrite=*/0) == 0;

constexpr size_t kBufferSize = 4ull << 30;
// hipMemGetInfo moves in 2 MiB steps, and the HIP runtime makes small
// allocations of its own.
constexpr size_t kSlack = 64ull << 20;

size_t FreeVram() {
  size_t free = 0, total = 0;
  EXPECT_EQ(hipMemGetInfo(&free, &total), hipSuccess);
  return free;
}

void ExpectFreeVram(size_t expected, const char* step) {
  size_t now = FreeVram();
  EXPECT_LE(expected, now + kSlack) << step << ": registered VRAM counted twice";
  EXPECT_LE(now, expected + kSlack) << step << ": registered VRAM credited more than once";
}

std::unique_ptr<RdmaDeviceContext> NewContext(RdmaDevice* device, bool dmabufFirst) {
  if (dmabufFirst) setenv("MORI_ENABLE_DMABUF_REG", "1", 1);
  std::unique_ptr<RdmaDeviceContext> ctx(device->CreateRdmaDeviceContext());
  if (dmabufFirst) unsetenv("MORI_ENABLE_DMABUF_REG");
  return ctx;
}

class KfdPinAccountingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    context = std::make_unique<RdmaContext>(RdmaBackendType::IBVerbs);
    for (const ActiveDevicePort& port : GetActiveDevicePortList(context->GetRdmaDeviceList())) {
      if (devices.empty() || devices.back() != port.first) devices.push_back(port.first);
    }
    if (devices.size() < 2) GTEST_SKIP() << "needs two active RDMA devices";

    beforeAlloc = FreeVram();
    ASSERT_EQ(hipMalloc(&buf, kBufferSize), hipSuccess);
    std::unique_ptr<RdmaDeviceContext> check(devices[0]->CreateRdmaDeviceContext());
    ibv_mr* mr = ibv_reg_mr(check->GetIbvPd(), buf, 4096, IBV_ACCESS_LOCAL_WRITE);
    if (mr == nullptr) GTEST_SKIP() << "no peer-memory support for VRAM";
    ibv_dereg_mr(mr);
    allocated = FreeVram();
  }

  void TearDown() override {
    if (buf == nullptr) return;
    EXPECT_EQ(hipFree(buf), hipSuccess);
    // An MR or anchor left behind would keep the allocation alive.
    ExpectFreeVram(beforeAlloc, "after hipFree");
  }

  std::unique_ptr<RdmaContext> context;
  std::vector<RdmaDevice*> devices;
  void* buf = nullptr;
  size_t beforeAlloc = 0;
  size_t allocated = 0;
};

TEST_F(KfdPinAccountingTest, DmabufMr) {
  auto ctx = NewContext(devices[0], /*dmabufFirst=*/true);
  ctx->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  ExpectFreeVram(allocated, "dma-buf MR registered");
  ctx->DeregisterRdmaMemoryRegion(buf);
  ExpectFreeVram(allocated, "dma-buf MR deregistered");
}

TEST_F(KfdPinAccountingTest, TwoNics) {
  auto a = NewContext(devices[0], false);
  auto b = NewContext(devices[1], false);
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  b->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  ExpectFreeVram(allocated, "registered on two NICs");
  a->DeregisterRdmaMemoryRegion(buf);
  ExpectFreeVram(allocated, "MR holding the pin deregistered first");
  b->DeregisterRdmaMemoryRegion(buf);
  ExpectFreeVram(allocated, "both deregistered");
}

TEST_F(KfdPinAccountingTest, SubBuffersOfOneAllocation) {
  auto a = NewContext(devices[0], false);
  char* hi = static_cast<char*>(buf) + kBufferSize / 2;
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize / 2);
  a->RegisterRdmaMemoryRegionAuto(hi, kBufferSize / 2);
  a->RegisterRdmaMemoryRegion(hi + kBufferSize / 4, kBufferSize / 4);
  ExpectFreeVram(allocated, "three sub-buffers registered");
  a->DeregisterRdmaMemoryRegion(buf);
  ExpectFreeVram(allocated, "lower half deregistered");
  a->DeregisterRdmaMemoryRegion(hi);
  a->DeregisterRdmaMemoryRegion(hi + kBufferSize / 4);
  ExpectFreeVram(allocated, "all deregistered");
}

TEST_F(KfdPinAccountingTest, PinHolderContextDestroyedFirst) {
  auto a = NewContext(devices[0], false);
  auto b = NewContext(devices[1], false);
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  b->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  ExpectFreeVram(allocated, "registered on two NICs");
  a.reset();
  ExpectFreeVram(allocated, "context holding the pin destroyed");
  b.reset();
  ExpectFreeVram(allocated, "both contexts destroyed");
}

TEST_F(KfdPinAccountingTest, AnchorContextDestroyedFirst) {
  auto a = NewContext(devices[0], /*dmabufFirst=*/true);
  auto b = NewContext(devices[1], /*dmabufFirst=*/true);
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  b->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  ExpectFreeVram(allocated, "dma-buf MRs on two NICs");
  a.reset();
  ExpectFreeVram(allocated, "context holding the anchor destroyed");
  b->DeregisterRdmaMemoryRegion(buf);
  ExpectFreeVram(allocated, "remaining MR deregistered");
}

}  // namespace

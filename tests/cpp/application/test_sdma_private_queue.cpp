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

#include <chrono>
#include <cstring>
#include <memory>
#include <vector>

#include "mori/application/transport/sdma/anvil.hpp"

namespace {

int Node(int hipDev) { return static_cast<int>(anvil::AnvilLib::nodeForHipDevice(hipDev)); }

int DeviceCount() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess ? n : 0;
}

TEST(SdmaPrivateQueue, CreateIdle) {
  if (DeviceCount() < 1) GTEST_SKIP() << "no GPU";
  auto q = anvil::anvil.createPrivateQueue(Node(0), Node(0));
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->deviceHandle(), nullptr);
  const anvil::SdmaQueueHostHandle h = q->hostHandle();
  EXPECT_NE(h.ring, nullptr);
  EXPECT_EQ(h.ringBytes, anvil::SDMA_QUEUE_SIZE);
  ASSERT_NE(h.rptr, nullptr);
  ASSERT_NE(h.wptr, nullptr);
  EXPECT_NE(h.doorbell, nullptr);
  EXPECT_EQ(*h.rptr, *h.wptr);
}

TEST(SdmaPrivateQueue, RejectsBadArguments) {
  if (DeviceCount() < 1) GTEST_SKIP() << "no GPU";
  const int n = Node(0);
  EXPECT_EQ(anvil::anvil.createPrivateQueue(n, n, 0, 2048), nullptr);
  EXPECT_EQ(anvil::anvil.createPrivateQueue(n, n, 0, 12288), nullptr);
  EXPECT_EQ(anvil::anvil.createPrivateQueue(n, n, -1), nullptr);
  EXPECT_NE(anvil::anvil.createPrivateQueue(n, n, 0, 4096), nullptr);
}

TEST(SdmaPrivateQueue, EngineOffsetMovesTheQueue) {
  if (DeviceCount() < 2) GTEST_SKIP() << "needs two GPUs";
  auto q0 = anvil::anvil.createPrivateQueue(Node(0), Node(1), 0);
  ASSERT_NE(q0, nullptr);
  auto q1 = anvil::anvil.createPrivateQueue(Node(0), Node(1), 1);
  if (q1 == nullptr) GTEST_SKIP() << "offset 1 refused here (e.g. MI308X)";
  EXPECT_NE(q1->hostHandle().engineId, q0->hostHandle().engineId);
}

TEST(SdmaPrivateQueue, HostDrivenCopy) {
  if (DeviceCount() < 1) GTEST_SKIP() << "no GPU";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  auto q = anvil::anvil.createPrivateQueue(Node(0), Node(0));
  ASSERT_NE(q, nullptr);
  const anvil::SdmaQueueHostHandle h = q->hostHandle();
  if (h.hdpFlush == nullptr) GTEST_SKIP() << "GPU exposes no HDP flush register";

  constexpr size_t kBytes = 1 << 20;
  std::vector<uint32_t> want(kBytes / 4), got(kBytes / 4, 0);
  for (size_t i = 0; i < want.size(); i++) want[i] = static_cast<uint32_t>(i * 2654435761u);
  void *src = nullptr, *dst = nullptr;
  ASSERT_EQ(hipMalloc(&src, kBytes), hipSuccess);
  ASSERT_EQ(hipMalloc(&dst, kBytes), hipSuccess);
  ASSERT_EQ(hipMemcpy(src, want.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemset(dst, 0, kBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  SDMA_PKT_COPY_LINEAR p{};
  p.HEADER_UNION.op = SDMA_OP_COPY;
  p.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
  p.COUNT_UNION.count = kBytes - 1;
  p.SRC_ADDR_LO_UNION.src_addr_31_0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src));
  p.SRC_ADDR_HI_UNION.src_addr_63_32 =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src) >> 32);
  p.DST_ADDR_LO_UNION.dst_addr_31_0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst));
  p.DST_ADDR_HI_UNION.dst_addr_63_32 =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst) >> 32);
  uint32_t unit[8] = {};  // one 32-byte unit; the zero tail dwords are NOPs
  static_assert(sizeof(p) <= sizeof(unit), "COPY_LINEAR fits one unit");
  std::memcpy(unit, &p, sizeof(p));

  const uint64_t start = *h.wptr;
  auto* slot = reinterpret_cast<volatile uint32_t*>(reinterpret_cast<char*>(h.ring) +
                                                    (start & (h.ringBytes - 1)));
  for (int i = 0; i < 8; i++) slot[i] = unit[i];
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  *h.hdpFlush = 1u;
  (void)*h.hdpFlush;  // the read-back completes only after the flush has landed
  const uint64_t end = start + sizeof(unit);
  __atomic_store_n(h.wptr, end, __ATOMIC_RELEASE);
  __atomic_store_n(h.doorbell, end, __ATOMIC_RELEASE);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (__atomic_load_n(h.rptr, __ATOMIC_ACQUIRE) < end &&
         std::chrono::steady_clock::now() < deadline) {
  }
  ASSERT_GE(*h.rptr, end) << "engine did not consume the packet";
  ASSERT_EQ(hipMemcpy(got.data(), dst, kBytes, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(std::memcmp(got.data(), want.data(), kBytes), 0);
  EXPECT_EQ(hipFree(src), hipSuccess);
  EXPECT_EQ(hipFree(dst), hipSuccess);
}

}  // namespace

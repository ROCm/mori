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

// GdsEngine: routing/planning are pure and run anywhere; the read round trip
// needs a GPU and the hipFile driver and skips without them.  Point TMPDIR at
// an ext4/xfs mount (e.g. TMPDIR=/mnt/gds) so the O_DIRECT open succeeds and the
// fastpath is actually exercised; on other filesystems it skips or falls back.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <string>
#include <vector>

#include "umbp/distributed/transfer/gds_engine.h"

namespace mori::umbp {
namespace {

TransferRef GpuBuf(void* p, uint64_t n, int dev = 0) {
  return TransferRef::HostBytes(p, n, mori::io::MemoryLocationType::GPU, dev);
}

TransferItem MakeItem(const TransferRef& src, uint64_t src_off, const TransferRef& dst,
                      uint64_t dst_off, uint64_t size, size_t tag) {
  TransferItem it;
  it.src = src;
  it.src_offset = src_off;
  it.dst = dst;
  it.dst_offset = dst_off;
  it.size = size;
  it.tag = tag;
  return it;
}

// ---------------------------------------------------------------------------
//  Routing / planning — no GPU, no hipFile driver
// ---------------------------------------------------------------------------

TEST(GdsEngine, ClaimsFileToGpuOnly) {
  GdsEngine engine;
  TransferRef file =
      TransferRef::File(/*fd=*/7, /*offset=*/0, /*n=*/4096, reinterpret_cast<void*>(0x1));
  std::vector<char> stub(4096);
  TransferRef gpu = GpuBuf(stub.data(), stub.size());
  TransferRef host = TransferRef::HostBytes(stub.data(), stub.size());  // CPU

  EXPECT_TRUE(engine.CanHandle(file, gpu));    // the read path
  EXPECT_FALSE(engine.CanHandle(file, host));  // host dst -> staged path
  EXPECT_FALSE(engine.CanHandle(gpu, file));   // write path, a later change
  EXPECT_FALSE(engine.CanHandle(gpu, gpu));    // no file endpoint at all
}

TEST(GdsEngine, PlansOneRangePerItemAndBoundsToBuffer) {
  GdsEngine engine;
  TransferRef file =
      TransferRef::File(/*fd=*/7, /*offset=*/8192, /*n=*/1 << 20, reinterpret_cast<void*>(0x1));
  std::vector<char> stub(4096);
  TransferRef gpu = GpuBuf(stub.data(), stub.size());

  std::vector<TransferItem> items{
      MakeItem(file, 0, gpu, 0, 2048, 0), MakeItem(file, 2048, gpu, 2048, 2048, 1),
      MakeItem(file, 0, gpu, 0, 8192, 2),  // overruns the 4096-byte buffer
  };
  TransferPlanSet planned = engine.Plan(items);
  EXPECT_EQ(std::vector<size_t>({2}), planned.rejected_tags);
  ASSERT_EQ(2u, planned.plans.size());
  for (const auto& p : planned.plans) EXPECT_EQ(1u, p.sizes.size());
}

// ---------------------------------------------------------------------------
//  End to end — needs a GPU and the hipFile driver
// ---------------------------------------------------------------------------

bool HaveGpu() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

TEST(GdsEngine, ReadsFileRangeIntoDeviceMemory) {
  if (!HaveGpu()) GTEST_SKIP() << "no HIP device";

  const size_t kSize = 8192;  // 4 KiB-aligned so the fastpath can accept it
  const std::string path = std::string(::testing::TempDir()) + "/gds_engine_rt.bin";
  std::vector<uint8_t> payload(kSize);
  for (size_t i = 0; i < kSize; ++i) payload[i] = static_cast<uint8_t>(i * 131 + 7);

  {
    int wfd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    ASSERT_GE(wfd, 0);
    ASSERT_EQ(static_cast<ssize_t>(kSize), ::write(wfd, payload.data(), kSize));
    ::close(wfd);
  }

  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    ::unlink(path.c_str());
    GTEST_SKIP() << "filesystem rejects O_DIRECT (errno=" << errno << "); point TMPDIR at ext4/xfs";
  }

  GdsEngine engine;
  TransferRef file = engine.RegisterFile(fd, /*offset=*/0, kSize);
  if (!file.IsFile()) {
    ::close(fd);
    ::unlink(path.c_str());
    GTEST_SKIP() << "hipFileHandleRegister unavailable on this host";
  }

  void* dbuf = nullptr;
  ASSERT_EQ(hipSuccess, hipMalloc(&dbuf, kSize));
  ASSERT_EQ(hipSuccess, hipMemset(dbuf, 0, kSize));
  TransferRef gpu = GpuBuf(dbuf, kSize);

  std::vector<size_t> failed;
  ASSERT_TRUE(engine.Transfer({MakeItem(file, 0, gpu, 0, kSize, 0)}, &failed));
  EXPECT_TRUE(failed.empty());

  std::vector<uint8_t> got(kSize, 0);
  ASSERT_EQ(hipSuccess, hipMemcpy(got.data(), dbuf, kSize, hipMemcpyDeviceToHost));
  EXPECT_EQ(payload, got);

  hipFree(dbuf);
  engine.Deregister(file);
  ::close(fd);
  ::unlink(path.c_str());
}

}  // namespace
}  // namespace mori::umbp

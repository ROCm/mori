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

// How HipMemoryApi::CopyBatch behaves when the runtime fails under it, on real
// devices (skipped with fewer than two). hipMemcpyAsync and
// hipStreamCreateWithFlags are wrapped at link time (-Wl,--wrap) so a test can
// make one call fail.
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "umbp/l1/l1_pool.h"

namespace {
std::atomic<int> fail_async_at{0};  // fail the Nth hipMemcpyAsync (1-based); 0: none
std::atomic<int> async_calls{0};
std::atomic<int> failing_stream_creations{0};
}  // namespace

extern "C" {
hipError_t __real_hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind,
                                 hipStream_t stream);
hipError_t __wrap_hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind,
                                 hipStream_t stream) {
  const int fail_at = fail_async_at.load();
  if (fail_at != 0 && async_calls.fetch_add(1) + 1 == fail_at) return hipErrorOutOfMemory;
  return __real_hipMemcpyAsync(dst, src, bytes, kind, stream);
}
hipError_t __real_hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t __wrap_hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  if (failing_stream_creations.load() > 0) {
    failing_stream_creations.fetch_sub(1);
    return hipErrorOutOfMemory;
  }
  return __real_hipStreamCreateWithFlags(stream, flags);
}
}

namespace mori::umbp::l1 {
namespace {

constexpr int kSrcDevice = 0;
constexpr int kDstDevice = 1;
// Large enough that a copy between two devices is still running long after a
// call that did not wait for it has returned.
constexpr uint64_t kBytes = uint64_t{1} << 30;

class L1HipCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices < 2) {
      GTEST_SKIP() << "needs two GPUs";
    }
    api = MakeHipMemoryApi();
    ExportedMemory exported;
    std::string error;
    ASSERT_TRUE(api->Allocate(kSrcDevice, kBytes, &src, &exported, &error)) << error;
    ASSERT_TRUE(api->Allocate(kDstDevice, kBytes, &dst, &exported, &error)) << error;
    ASSERT_EQ(hipSetDevice(kSrcDevice), hipSuccess);
    ASSERT_EQ(hipMemset(src, 0xAB, kBytes), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  }
  void TearDown() override {
    fail_async_at.store(0);
    failing_stream_creations.store(0);
    if (src != nullptr) api->Free(kSrcDevice, src, kBytes);
    if (dst != nullptr) api->Free(kDstDevice, dst, kBytes);
  }

  // The destination's last byte as it is now, without waiting for any stream.
  uint8_t LastByteOfDst() {
    uint8_t byte = 0;
    EXPECT_EQ(hipSetDevice(kDstDevice), hipSuccess);
    EXPECT_EQ(hipMemcpy(&byte, static_cast<char*>(dst) + kBytes - 1, 1, hipMemcpyDeviceToHost),
              hipSuccess);
    return byte;
  }

  std::shared_ptr<MemoryApi> api;
  void* src = nullptr;
  void* dst = nullptr;
};

TEST_F(L1HipCopyTest, AFailedFragmentWaitsForTheFragmentsAlreadyQueued) {
  // Fragment 0 fills the destination; queuing fragment 1 fails.
  std::vector<MemoryApi::Fragment> fragments{
      MemoryApi::Fragment{dst, src, kBytes, kSrcDevice},
      MemoryApi::Fragment{dst, src, 4096, kSrcDevice},
  };
  async_calls.store(0);
  fail_async_at.store(2);
  std::string error;
  EXPECT_FALSE(api->CopyBatch(kDstDevice, fragments, &error));
  EXPECT_NE(error.find("hipMemcpyAsync"), std::string::npos) << error;
  // The caller frees the sources and destinations as soon as the call fails:
  // by then nothing may still be copying into them.
  EXPECT_EQ(LastByteOfDst(), 0xAB) << "fragment 0 was still copying after CopyBatch returned";
}

TEST_F(L1HipCopyTest, AFailedStreamCreationIsTriedAgainOnTheNextCopy) {
  std::vector<MemoryApi::Fragment> fragments{MemoryApi::Fragment{dst, src, 1 << 20, kSrcDevice}};
  failing_stream_creations.store(1);
  std::string error;
  EXPECT_FALSE(api->CopyBatch(kDstDevice, fragments, &error));
  EXPECT_EQ(failing_stream_creations.load(), 0);
  error.clear();
  EXPECT_TRUE(api->CopyBatch(kDstDevice, fragments, &error)) << error;
}

}  // namespace
}  // namespace mori::umbp::l1

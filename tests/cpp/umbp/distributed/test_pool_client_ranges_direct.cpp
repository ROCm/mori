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

#include <algorithm>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "pool_client_ranges_test_fixture.h"
#include "umbp/common/device_gather.h"

namespace mori::umbp {
namespace {

// A destination the NIC registered is read straight into: the bytes land where
// the caller asked, and the arena, still all poison, never saw them.
TEST_F(PoolClientRangesTest, RegisteredDestinationsAreReadDirectly) {
  const std::string key = "ranged-direct-read";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 11 + 5) & 0xff);
  SeedRemoteObject(key, object);

  constexpr char kPoison = static_cast<char>(0xAA);
  std::vector<char> get_arena(kScratchSize, kPoison), put_arena(kScratchSize, kPoison);
  auto client = MakeClient("ranges-direct-read", kCallerCapacity, get_arena, put_arena,
                           /*locality_prefetch=*/true, /*direct=*/true);
  std::vector<char> dst(3000, 0);
  ASSERT_TRUE(client->RegisterMemory(dst.data(), dst.size()));
  ASSERT_TRUE(WaitForExists(client.get(), key));

  auto got = client->BatchGetRanges({key}, {{dst.data() + 1000, dst.data()}}, {{2000, 1000}},
                                    {{4000, 100}});
  ASSERT_EQ(got, std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(dst.data(), object.data() + 100, 1000), 0);
  EXPECT_EQ(std::memcmp(dst.data() + 1000, object.data() + 4000, 2000), 0);
  EXPECT_EQ(get_arena, std::vector<char>(kScratchSize, kPoison));
  client->Shutdown();
}

// Sources the NIC registered are gathered straight into the remote slot, here
// out of order: the object reads back whole, and the PUT arena is untouched.
TEST_F(PoolClientRangesTest, RegisteredSourcesAreWrittenDirectly) {
  const std::string key = "ranged-direct-write";
  constexpr char kPoison = static_cast<char>(0xAA);
  std::vector<char> get_arena(kScratchSize, kPoison), put_arena(kScratchSize, kPoison);
  auto client = MakeClient("ranges-direct-write", kCallerCapacity, get_arena, put_arena,
                           /*locality_prefetch=*/true, /*direct=*/true);
  std::vector<char> src(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) src[i] = static_cast<char>((i * 13 + 1) & 0xff);
  ASSERT_TRUE(client->RegisterMemory(src.data(), src.size()));

  const size_t half = kObjectSize / 2;
  auto put = client->BatchPutRanges({key}, {kObjectSize}, {{src.data() + half, src.data()}},
                                    {{kObjectSize - half, half}}, {{half, 0}});
  ASSERT_EQ(put, std::vector<bool>({true}));
  EXPECT_EQ(put_arena, std::vector<char>(kScratchSize, kPoison));

  client->Master().FlushHeartbeat();
  target_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitForExists(caller_.get(), key));
  std::vector<char> out(kObjectSize, 0);
  ASSERT_TRUE(caller_->BatchGetRanges({key}, {{out.data()}}, {{kObjectSize}}, {{0}}).front());
  EXPECT_EQ(out, src);
  client->Shutdown();
}

// The same between registered GPU memory and the peer's tier: written from
// device ranges, read back into other device ranges, neither through an arena.
TEST_F(PoolClientRangesTest, RegisteredGpuMemoryMovesDirectlyBothWays) {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  void* gpu = nullptr;
  ASSERT_EQ(hipMalloc(&gpu, 2 * kObjectSize), hipSuccess);
  char* const base = static_cast<char*>(gpu);
  constexpr char kPoison = static_cast<char>(0xAA);
  std::vector<char> get_arena(kScratchSize, kPoison), put_arena(kScratchSize, kPoison);
  auto client = MakeClient("ranges-direct-gpu", kCallerCapacity, get_arena, put_arena,
                           /*locality_prefetch=*/true, /*direct=*/true);
  ASSERT_TRUE(client->RegisterMemory(gpu, 2 * kObjectSize, mori::io::MemoryLocationType::GPU, 0));

  const std::string key = "ranged-direct-gpu";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 17 + 9) & 0xff);
  ASSERT_EQ(hipMemcpy(base, object.data(), kObjectSize, hipMemcpyHostToDevice), hipSuccess);
  const size_t third = kObjectSize / 3;
  auto put = client->BatchPutRanges({key}, {kObjectSize}, {{base, base + third}},
                                    {{third, kObjectSize - third}}, {{0, third}});
  ASSERT_EQ(put, std::vector<bool>({true}));
  EXPECT_EQ(put_arena, std::vector<char>(kScratchSize, kPoison));

  client->Master().FlushHeartbeat();
  target_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitForExists(client.get(), key));
  ASSERT_EQ(hipMemset(base + kObjectSize, 0, kObjectSize), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  char* const dst = base + kObjectSize;
  auto got =
      client->BatchGetRanges({key}, {{dst + 4096, dst}}, {{kObjectSize - 4096, 4096}}, {{4096, 0}});
  ASSERT_EQ(got, std::vector<bool>({true}));
  EXPECT_EQ(get_arena, std::vector<char>(kScratchSize, kPoison));
  std::vector<char> out(kObjectSize, 0);
  ASSERT_EQ(hipMemcpy(out.data(), dst, kObjectSize, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(out, object);
  client->Shutdown();
  (void)hipFree(gpu);
}

// The NIC writes device memory without passing through the GPU's caches, so a
// kernel must see what a direct read wrote -- a host copy-back cannot tell, it
// bypasses them too.  Kernels first write and read the destination, leaving
// its lines cached, then it is read into directly and a kernel reads it again.
TEST_F(PoolClientRangesTest, KernelsSeeWhatADirectReadWrote) {
  if (!DeviceGatherEnabled()) GTEST_SKIP() << "no gather kernel to read device memory with";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  const std::string key = "ranged-direct-kernel";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 19 + 7) & 0xff);
  SeedRemoteObject(key, object);

  std::vector<char> get_arena(kScratchSize, 0), put_arena(kScratchSize, 0);
  auto client = MakeClient("ranges-direct-kernel", kCallerCapacity, get_arena, put_arena,
                           /*locality_prefetch=*/false, /*direct=*/true);
  void* gpu = nullptr;
  ASSERT_EQ(hipMalloc(&gpu, kObjectSize), hipSuccess);
  ASSERT_TRUE(client->RegisterMemory(gpu, kObjectSize, mori::io::MemoryLocationType::GPU, 0));
  ASSERT_TRUE(WaitForExists(client.get(), key));

  constexpr char kStale = 0x5C;
  std::vector<char> host(kObjectSize, kStale);
  ASSERT_EQ(hipHostRegister(host.data(), host.size(), hipHostRegisterMapped), hipSuccess);
  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
  auto gather = [&](const void* src, void* dst) {
    const DeviceGatherFragment fragment{src, dst, kObjectSize};
    return LaunchDeviceGather(&fragment, 1, 0, stream) &&
           hipStreamSynchronize(stream) == hipSuccess;
  };
  ASSERT_TRUE(gather(host.data(), gpu));
  ASSERT_TRUE(gather(gpu, host.data()));
  ASSERT_EQ(host, std::vector<char>(kObjectSize, kStale));

  auto got = client->BatchGetRanges({key}, {{gpu}}, {{kObjectSize}}, {{0}});
  ASSERT_EQ(got, std::vector<bool>({true}));
  ASSERT_TRUE(gather(gpu, host.data()));
  EXPECT_EQ(host, object);

  client->Shutdown();
  (void)hipStreamDestroy(stream);
  (void)hipHostUnregister(host.data());
  (void)hipFree(gpu);
}

// One call, every way a key can go: read directly (all its destinations
// registered), staged (none of them, or only some), or missing.  Each read key
// lands whole, and no byte of the direct one passes through the arena.
TEST_F(PoolClientRangesTest, MixedBatchStagesOnlyWhatTheNicCannotReach) {
  const std::vector<std::string> keys = {"ranged-mixed-direct", "ranged-mixed-plain",
                                         "ranged-mixed-part", "ranged-mixed-missing"};
  std::vector<std::vector<char>> objects(3, std::vector<char>(kObjectSize));
  for (size_t k = 0; k < objects.size(); ++k) {
    for (size_t i = 0; i < kObjectSize; ++i) {
      objects[k][i] = static_cast<char>((i * (7 + 2 * k) + 3 * k + 1) & 0xff);
    }
    SeedRemoteObject(keys[k], objects[k]);
  }
  constexpr char kPoison = static_cast<char>(0xAA);
  std::vector<char> get_arena(kScratchSize, kPoison), put_arena(kScratchSize, kPoison);
  auto client = MakeClient("ranges-mixed", kCallerCapacity, get_arena, put_arena,
                           /*locality_prefetch=*/false, /*direct=*/true);
  std::vector<char> registered(3000, 0), plain(3000, 0), spare(1000, 0);
  ASSERT_TRUE(client->RegisterMemory(registered.data(), registered.size()));
  for (size_t k = 0; k < objects.size(); ++k) ASSERT_TRUE(WaitForExists(client.get(), keys[k]));

  auto got = client->BatchGetRanges(keys,
                                    {{registered.data(), registered.data() + 1000},
                                     {plain.data(), plain.data() + 1000},
                                     {registered.data() + 2000, plain.data() + 2000},
                                     {spare.data()}},
                                    {{1000, 1000}, {1000, 1000}, {1000, 1000}, {1000}},
                                    {{3000, 100}, {3000, 100}, {3000, 100}, {0}});
  ASSERT_EQ(got, std::vector<bool>({true, true, true, false}));
  EXPECT_EQ(std::memcmp(registered.data(), objects[0].data() + 3000, 1000), 0);
  EXPECT_EQ(std::memcmp(registered.data() + 1000, objects[0].data() + 100, 1000), 0);
  EXPECT_EQ(std::memcmp(plain.data(), objects[1].data() + 3000, 1000), 0);
  EXPECT_EQ(std::memcmp(plain.data() + 1000, objects[1].data() + 100, 1000), 0);
  EXPECT_EQ(std::memcmp(registered.data() + 2000, objects[2].data() + 3000, 1000), 0);
  EXPECT_EQ(std::memcmp(plain.data() + 2000, objects[2].data() + 100, 1000), 0);

  EXPECT_NE(get_arena, std::vector<char>(kScratchSize, kPoison));
  for (size_t offset : {size_t{100}, size_t{3000}}) {
    const auto direct_bytes = objects[0].begin() + static_cast<std::ptrdiff_t>(offset);
    EXPECT_EQ(std::search(get_arena.begin(), get_arena.end(), direct_bytes, direct_bytes + 1000),
              get_arena.end());
  }
  client->Shutdown();
}

// A read whose destinations the NIC registered never touches the arena: with
// the GET arena unregistered it still succeeds, and the key beside it that
// would stage fails alone.
TEST_F(PoolClientRangesTest, DirectReadDoesNotTouchTheArena) {
  const std::string direct_key = "ranged-direct-no-arena";
  const std::string staged_key = "ranged-staged-no-arena";
  std::vector<char> object(kObjectSize);
  for (size_t i = 0; i < kObjectSize; ++i) object[i] = static_cast<char>((i * 5 + 3) & 0xff);
  SeedRemoteObject(direct_key, object);
  SeedRemoteObject(staged_key, object);

  std::vector<char> get_arena(kScratchSize, 0), put_arena(kScratchSize, 0);
  auto client = MakeClient("ranges-direct-no-arena", kCallerCapacity, get_arena, put_arena,
                           /*locality_prefetch=*/false, /*direct=*/true);
  client->DeregisterMemory(get_arena.data());
  std::vector<char> registered(2000, 0), plain(2000, 0);
  ASSERT_TRUE(client->RegisterMemory(registered.data(), registered.size()));
  ASSERT_TRUE(WaitForExists(client.get(), direct_key));
  ASSERT_TRUE(WaitForExists(client.get(), staged_key));

  auto got = client->BatchGetRanges({direct_key, staged_key}, {{registered.data()}, {plain.data()}},
                                    {{2000}, {2000}}, {{500}, {500}});
  ASSERT_EQ(got, std::vector<bool>({true, false}));
  EXPECT_EQ(std::memcmp(registered.data(), object.data() + 500, 2000), 0);
  client->Shutdown();
}

}  // namespace
}  // namespace mori::umbp

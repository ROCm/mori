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

// MORI's single-process multi-thread (SPMT) mode, as JAX runs it: one process with one thread
// per GPU, and each thread creates its own IOEngine. RAS starts once for all of them, and the
// process identity lists one attachment per engine and GPU.
//
// The binary holds this one test, so the process-wide runtime it starts serves nothing else.
// It needs two GPUs and skips with fewer. With no RDMA backend, the attachments have no NICs.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/engine.hpp"
#include "mori/ras/ras.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

using namespace mori::io;
using mori::ras::GpuAttachment;
using mori::ras::RasRuntime;
using mori::ras::RunState;
using ras_test::WaitFor;

namespace {

constexpr size_t kBufferBytes = 1 << 20;

std::vector<GpuAttachment> Attachments() {
  RasRuntime* rt = RasRuntime::Get();
  return rt != nullptr ? rt->Snapshot().self.attachments : std::vector<GpuAttachment>{};
}

}  // namespace

TEST(RasSpmt, EachGpuThreadsEngineAttachesItsGpu) {
  int visible = 0;
  if (hipGetDeviceCount(&visible) != hipSuccess || visible < 2) {
    GTEST_SKIP() << "needs two GPUs; found " << visible;
  }
  const int n = std::min(visible, 8);
  setenv("MORI_RAS_ENABLE", "1", 1);

  std::vector<std::unique_ptr<IOEngine>> engines(n);
  std::vector<void*> buffers(n, nullptr);
  std::vector<MemoryDesc> mems(n);
  std::vector<int> done(n, 0);
  std::atomic<int> waiting{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (int i = 0; i < n; ++i) {
    threads.emplace_back([&, i] {
      bool ready =
          hipSetDevice(i) == hipSuccess && hipMalloc(&buffers[i], kBufferBytes) == hipSuccess;
      waiting.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      if (!ready) return;
      // Every thread creates its first engine at once, so RAS's first starts race.
      engines[i] =
          std::make_unique<IOEngine>("spmt-" + std::to_string(i), IOEngineConfig{"127.0.0.1", 0});
      mems[i] = engines[i]->RegisterMemory(buffers[i], kBufferBytes, i, MemoryLocationType::GPU);
      // Another registration on the same GPU adds no attachment.
      engines[i]->RegisterMemory(static_cast<char*>(buffers[i]) + 4096, 4096, i,
                                 MemoryLocationType::GPU);
      done[i] = 1;
    });
  }
  while (waiting.load() < n) std::this_thread::yield();
  go.store(true, std::memory_order_release);
  for (std::thread& t : threads) t.join();
  for (int i = 0; i < n; ++i) ASSERT_EQ(done[i], 1) << "GPU " << i << " didn't set up";

  // One runtime: every engine advertises the same RAS endpoint.
  ASSERT_NE(RasRuntime::Get(), nullptr);
  EXPECT_EQ(RasRuntime::Status().state, RunState::kRunning);
  const RasDesc first = engines[0]->GetEngineDesc().ras;
  EXPECT_NE(first.port, 0);
  EXPECT_FALSE(first.uuidHi == 0 && first.uuidLo == 0);
  for (int i = 1; i < n; ++i) EXPECT_EQ(engines[i]->GetEngineDesc().ras, first) << "engine " << i;

  // One attachment per engine and GPU, ordered by engine.
  ASSERT_TRUE(WaitFor([&] { return Attachments().size() == static_cast<size_t>(n); }, 2000))
      << Attachments().size() << " attachments";
  std::vector<GpuAttachment> a = Attachments();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(a[i].owner, "spmt-" + std::to_string(i));
    EXPECT_EQ(a[i].gpu, i);
    EXPECT_FALSE(a[i].pciBusId.empty());
    EXPECT_EQ(a[i].pciBusId, mems[i].deviceBusId);
    EXPECT_TRUE(a[i].nics.empty());
  }

  // One thread's engine goes away while the others live on: only its attachment goes.
  engines[0].reset();
  ASSERT_TRUE(WaitFor([&] { return Attachments().size() == static_cast<size_t>(n - 1); }, 2000));
  EXPECT_EQ(Attachments()[0].owner, "spmt-1");

  for (int i = 1; i < n; ++i) engines[i].reset();
  EXPECT_TRUE(WaitFor([&] { return Attachments().empty(); }, 2000));
  for (int i = 0; i < n; ++i) {
    if (hipSetDevice(i) == hipSuccess) (void)hipFree(buffers[i]);
  }
}

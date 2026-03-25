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
// Plain C++ test for umbp::SpdkEnv.
// Requires a real SPDK bdev or malloc bdev available at runtime.
// Usage: UMBP_SPDK_BDEV=Malloc0 ./test_spdk_env
//
// Uses assert() — no Google Test dependency.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "umbp/spdk/spdk_env.h"

static umbp::SpdkEnvConfig ConfigFromEnv() {
  umbp::SpdkEnvConfig cfg;
  const char* bdev = std::getenv("UMBP_SPDK_BDEV");
  const char* nvme_pci = std::getenv("UMBP_SPDK_NVME_PCI");
  if (nvme_pci && nvme_pci[0]) {
    cfg.nvme_pci_addr = nvme_pci;
    const char* ctrl = std::getenv("UMBP_SPDK_NVME_CTRL");
    if (ctrl) cfg.nvme_ctrl_name = ctrl;
    cfg.bdev_name = bdev ? bdev : (cfg.nvme_ctrl_name + "n1");
  } else if (bdev) {
    cfg.bdev_name = bdev;
  } else {
    cfg.use_malloc_bdev = true;
    cfg.bdev_name = "Malloc0";
    cfg.malloc_num_blocks = 262144;  // 1 GB @ 4KB blocks
    cfg.malloc_block_size = 4096;
  }
  const char* mask = std::getenv("UMBP_SPDK_REACTOR_MASK");
  if (mask) cfg.reactor_mask = mask;
  const char* mem = std::getenv("UMBP_SPDK_MEM_MB");
  if (mem)
    cfg.mem_size_mb = std::atoi(mem);
  else
    cfg.mem_size_mb = 256;
  return cfg;
}

static void test_init_shutdown() {
  printf("  test_init_shutdown...\n");
  auto& env = umbp::SpdkEnv::Instance();
  auto cfg = ConfigFromEnv();
  int rc = env.Init(cfg);
  assert(rc == 0 && "SpdkEnv::Init should succeed");
  assert(env.IsInitialized());
  assert(env.GetBlockSize() > 0);
  assert(env.GetBdevSize() > 0);
  assert(env.GetNumReactors() > 0);
  printf("    block_size=%u bdev_size=%lu reactors=%d\n", env.GetBlockSize(),
         static_cast<unsigned long>(env.GetBdevSize()), env.GetNumReactors());
  printf("  PASS\n");
}

static void test_dma_alloc() {
  printf("  test_dma_alloc...\n");
  auto& env = umbp::SpdkEnv::Instance();
  assert(env.IsInitialized());

  void* buf = env.DmaMalloc(4096);
  assert(buf != nullptr);
  std::memset(buf, 0xAB, 4096);
  env.DmaFree(buf);
  printf("  PASS\n");
}

static void test_dma_pool() {
  printf("  test_dma_pool...\n");
  auto& env = umbp::SpdkEnv::Instance();

  constexpr int N = 16;
  constexpr size_t BUF_SIZE = 65536;
  void* bufs[N];
  int got = env.DmaPoolAllocBatch(bufs, BUF_SIZE, N);
  assert(got == N);
  for (int i = 0; i < N; ++i) {
    assert(bufs[i] != nullptr);
    std::memset(bufs[i], static_cast<unsigned char>(i), BUF_SIZE);
  }
  env.DmaPoolFreeBatch(bufs, BUF_SIZE, N);

  void* b = env.DmaPoolAlloc(BUF_SIZE);
  assert(b != nullptr);
  env.DmaPoolFree(b, BUF_SIZE);
  printf("  PASS\n");
}

static void test_sync_io() {
  printf("  test_sync_io...\n");
  auto& env = umbp::SpdkEnv::Instance();
  uint32_t bs = env.GetBlockSize();

  void* wbuf = env.DmaMalloc(bs);
  void* rbuf = env.DmaMalloc(bs);
  assert(wbuf && rbuf);

  std::memset(wbuf, 0x42, bs);
  std::memset(rbuf, 0, bs);

  // Synchronous write
  umbp::SpdkIoRequest wreq{};
  wreq.op = umbp::SpdkIoRequest::WRITE;
  wreq.buf = wbuf;
  wreq.offset = 0;
  wreq.nbytes = bs;
  env.SubmitIo(&wreq);
  assert(wreq.success && "Sync write should succeed");

  // Synchronous read
  umbp::SpdkIoRequest rreq{};
  rreq.op = umbp::SpdkIoRequest::READ;
  rreq.buf = rbuf;
  rreq.offset = 0;
  rreq.nbytes = bs;
  env.SubmitIo(&rreq);
  assert(rreq.success && "Sync read should succeed");

  assert(std::memcmp(wbuf, rbuf, bs) == 0 && "Data mismatch");

  env.DmaFree(wbuf);
  env.DmaFree(rbuf);
  printf("  PASS\n");
}

static void test_async_batch() {
  printf("  test_async_batch...\n");
  auto& env = umbp::SpdkEnv::Instance();
  uint32_t bs = env.GetBlockSize();
  constexpr int N = 8;

  std::vector<void*> bufs(N);
  std::vector<umbp::SpdkIoRequest> reqs(N);
  std::vector<umbp::SpdkIoRequest*> ptrs(N);

  for (int i = 0; i < N; ++i) {
    bufs[i] = env.DmaMalloc(bs);
    assert(bufs[i]);
    std::memset(bufs[i], static_cast<unsigned char>(i + 1), bs);

    reqs[i].op = umbp::SpdkIoRequest::WRITE;
    reqs[i].buf = bufs[i];
    reqs[i].offset = static_cast<uint64_t>(i) * bs;
    reqs[i].nbytes = bs;
    ptrs[i] = &reqs[i];
  }

  int rc = env.SubmitIoBatchAsync(ptrs.data(), N);
  assert(rc == 0);

  for (int i = 0; i < N; ++i) {
    while (!reqs[i].completed.load(std::memory_order_acquire)) {
    }
    assert(reqs[i].success);
  }

  // Read back
  for (int i = 0; i < N; ++i) {
    std::memset(bufs[i], 0, bs);
    reqs[i].op = umbp::SpdkIoRequest::READ;
    reqs[i].buf = bufs[i];
    reqs[i].offset = static_cast<uint64_t>(i) * bs;
    reqs[i].nbytes = bs;
    reqs[i].completed.store(false);
    reqs[i].success = false;
    ptrs[i] = &reqs[i];
  }

  rc = env.SubmitIoBatchAsync(ptrs.data(), N);
  assert(rc == 0);

  for (int i = 0; i < N; ++i) {
    while (!reqs[i].completed.load(std::memory_order_acquire)) {
    }
    assert(reqs[i].success);
    unsigned char expected = static_cast<unsigned char>(i + 1);
    auto* data = static_cast<unsigned char*>(bufs[i]);
    assert(data[0] == expected && "Readback data mismatch");
  }

  for (int i = 0; i < N; ++i) env.DmaFree(bufs[i]);
  printf("  PASS\n");
}

static void test_shutdown() {
  printf("  test_shutdown...\n");
  auto& env = umbp::SpdkEnv::Instance();
  env.DmaPoolDrain();
  env.Shutdown();
  assert(!env.IsInitialized());
  printf("  PASS\n");
}

int main() {
  printf("=== test_spdk_env ===\n");
  test_init_shutdown();
  test_dma_alloc();
  test_dma_pool();
  test_sync_io();
  test_async_batch();
  test_shutdown();
  printf("=== ALL PASSED ===\n");
  return 0;
}

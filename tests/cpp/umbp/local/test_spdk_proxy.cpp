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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Integration test for SPDK Proxy IPC.
// Tests the shared memory protocol and SpdkProxyTier client.
// Rank slot is auto-allocated via CAS — no manual RANK env needed.
//
// Requires: spdk_proxy daemon running with the default SHM name.
// Run: ./test_spdk_proxy
//
// If spdk_proxy is not running, all tests are skipped gracefully.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/local/storage/spdk_proxy_tier.h"

using namespace mori::umbp;

static UMBPConfig MakeProxyConfig() {
  auto cfg = UMBPConfig::FromEnvironment();
  cfg.ssd_backend = "spdk_proxy";
  // rank auto-allocated via CAS (spdk_proxy_rank_id == kAutoRankId from env)
  if (cfg.spdk_proxy_shm_name.empty()) cfg.spdk_proxy_shm_name = "/umbp_spdk_proxy";
  return cfg;
}

static void test_connect() {
  printf("  test_connect...\n");
  auto cfg = MakeProxyConfig();
  SpdkProxyTier tier(cfg);
  if (!tier.IsValid()) {
    printf("    SKIPPED (proxy not running)\n");
    return;
  }
  printf("  PASS\n");
}

static void test_single_write_read() {
  printf("  test_single_write_read...\n");
  auto cfg = MakeProxyConfig();
  SpdkProxyTier tier(cfg);
  if (!tier.IsValid()) {
    printf("    SKIPPED\n");
    return;
  }

  std::string key = "proxy_test_key_1";
  std::vector<char> data(4096, 0xAB);
  assert(tier.Write(key, data.data(), data.size()));
  assert(tier.Exists(key));

  std::vector<char> buf(4096, 0);
  assert(tier.ReadIntoPtr(key, reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);
  printf("  PASS\n");
}

static void test_batch_write_read() {
  printf("  test_batch_write_read...\n");
  auto cfg = MakeProxyConfig();
  SpdkProxyTier tier(cfg);
  if (!tier.IsValid()) {
    printf("    SKIPPED\n");
    return;
  }

  const int count = 100;
  const size_t val_size = 32768;  // 32KB

  std::vector<std::string> keys;
  std::vector<std::vector<char>> datas;
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;

  for (int i = 0; i < count; ++i) {
    keys.push_back("proxy_batch_" + std::to_string(i));
    datas.emplace_back(val_size, static_cast<char>(i & 0xFF));
    ptrs.push_back(datas.back().data());
    sizes.push_back(val_size);
  }

  auto write_results = tier.BatchWrite(keys, ptrs, sizes);
  int write_ok = 0;
  for (int i = 0; i < count; ++i)
    if (write_results[i]) ++write_ok;
  printf("    batch wrote %d/%d\n", write_ok, count);
  assert(write_ok == count);

  std::vector<std::vector<char>> read_bufs(count, std::vector<char>(val_size, 0));
  std::vector<uintptr_t> dst_ptrs;
  for (int i = 0; i < count; ++i)
    dst_ptrs.push_back(reinterpret_cast<uintptr_t>(read_bufs[i].data()));

  auto read_results = tier.BatchReadIntoPtr(keys, dst_ptrs, sizes);
  int read_ok = 0;
  for (int i = 0; i < count; ++i) {
    if (read_results[i]) {
      ++read_ok;
      assert(read_bufs[i] == datas[i]);
    }
  }
  printf("    batch read %d/%d\n", read_ok, count);
  assert(read_ok == count);
  printf("  PASS\n");
}

static void test_evict() {
  printf("  test_evict...\n");
  auto cfg = MakeProxyConfig();
  SpdkProxyTier tier(cfg);
  if (!tier.IsValid()) {
    printf("    SKIPPED\n");
    return;
  }

  std::string key = "proxy_evict_test";
  std::vector<char> data(4096, 0xCD);
  assert(tier.Write(key, data.data(), data.size()));
  assert(tier.Exists(key));
  assert(tier.Evict(key));
  assert(!tier.Exists(key));
  printf("  PASS\n");
}

static void test_capacity() {
  printf("  test_capacity...\n");
  auto cfg = MakeProxyConfig();
  SpdkProxyTier tier(cfg);
  if (!tier.IsValid()) {
    printf("    SKIPPED\n");
    return;
  }

  auto [used, total] = tier.Capacity();
  printf("    used=%zuMB total=%zuMB\n", used / (1024 * 1024), total / (1024 * 1024));
  assert(total > 0);
  printf("  PASS\n");
}

int main() {
  printf("=== test_spdk_proxy ===\n");
  test_connect();
  test_single_write_read();
  test_batch_write_read();
  test_evict();
  test_capacity();
  printf("=== ALL PASSED ===\n");
  return 0;
}

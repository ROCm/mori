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
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "umbp/local/umbp_client.h"

using namespace mori::umbp;

void test_put_get() {
  std::cout << "test_put_get... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = false;

  UMBPClient client(config);

  std::vector<char> data(4096, 'A');
  assert(client.Put("key1", data.data(), data.size()));
  assert(client.Exists("key1"));

  std::vector<char> buf(4096, 0);
  assert(client.GetIntoPtr("key1", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);

  std::cout << "PASSED" << std::endl;
}

void test_put_dedup() {
  std::cout << "test_put_dedup... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = false;

  UMBPClient client(config);

  std::vector<char> data(4096, 'B');
  assert(client.Put("key1", data.data(), data.size()));

  // Second put with same key should return true (dedup)
  std::vector<char> data2(4096, 'C');
  assert(client.Put("key1", data2.data(), data2.size()));

  // Data should still be original (dedup skipped write)
  std::vector<char> buf(4096, 0);
  assert(client.GetIntoPtr("key1", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == data);  // Original data, not data2

  std::cout << "PASSED" << std::endl;
}

void test_remove() {
  std::cout << "test_remove... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = false;

  UMBPClient client(config);

  std::vector<char> data(4096, 'D');
  assert(client.Put("key1", data.data(), data.size()));
  assert(client.Exists("key1"));

  assert(client.Remove("key1"));
  assert(!client.Exists("key1"));
  assert(!client.Remove("key1"));  // Already removed

  std::cout << "PASSED" << std::endl;
}

void test_batch_put_get() {
  std::cout << "test_batch_put_get... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = false;

  UMBPClient client(config);

  const int N = 10;
  std::vector<std::string> keys;
  std::vector<std::vector<char>> all_data;
  std::vector<uintptr_t> ptrs;
  std::vector<size_t> sizes;

  for (int i = 0; i < N; ++i) {
    keys.push_back("batch_key_" + std::to_string(i));
    all_data.emplace_back(1024, static_cast<char>('0' + i));
    ptrs.push_back(reinterpret_cast<uintptr_t>(all_data.back().data()));
    sizes.push_back(1024);
  }

  auto put_results = client.BatchPutFromPtr(keys, ptrs, sizes);
  for (int i = 0; i < N; ++i) {
    assert(put_results[i]);
  }

  // Batch exists
  auto exists_results = client.BatchExists(keys);
  for (int i = 0; i < N; ++i) {
    assert(exists_results[i]);
  }

  // Batch get
  std::vector<std::vector<char>> read_bufs(N, std::vector<char>(1024, 0));
  std::vector<uintptr_t> dst_ptrs;
  for (int i = 0; i < N; ++i) {
    dst_ptrs.push_back(reinterpret_cast<uintptr_t>(read_bufs[i].data()));
  }

  auto get_results = client.BatchGetIntoPtr(keys, dst_ptrs, sizes);
  for (int i = 0; i < N; ++i) {
    assert(get_results[i]);
    assert(read_bufs[i] == all_data[i]);
  }

  std::cout << "PASSED" << std::endl;
}

void test_clear() {
  std::cout << "test_clear... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = false;

  UMBPClient client(config);

  std::vector<char> data(4096, 'E');
  client.Put("key1", data.data(), data.size());
  client.Put("key2", data.data(), data.size());
  assert(client.Exists("key1"));
  assert(client.Exists("key2"));

  client.Clear();
  assert(!client.Exists("key1"));
  assert(!client.Exists("key2"));

  std::cout << "PASSED" << std::endl;
}

void test_put_from_ptr_get_into_ptr() {
  std::cout << "test_put_from_ptr_get_into_ptr... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1 * 1024 * 1024;
  config.ssd.enabled = false;

  UMBPClient client(config);

  std::vector<char> data(8192, 'Z');
  uintptr_t src = reinterpret_cast<uintptr_t>(data.data());
  assert(client.PutFromPtr("ptr_key", src, data.size()));

  std::vector<char> buf(8192, 0);
  uintptr_t dst = reinterpret_cast<uintptr_t>(buf.data());
  assert(client.GetIntoPtr("ptr_key", dst, buf.size()));
  assert(buf == data);

  std::cout << "PASSED" << std::endl;
}

void test_dram_full_demote_with_index() {
  std::cout << "test_dram_full_demote_with_index... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1024;  // 1 KB
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_client_demote";
  config.ssd.capacity_bytes = 10 * 1024 * 1024;

  UMBPClient client(config);

  // Fill DRAM: 2 x 512 bytes
  std::vector<char> d1(512, 'A');
  assert(client.Put("k1", d1.data(), d1.size()));
  std::vector<char> d2(512, 'B');
  assert(client.Put("k2", d2.data(), d2.size()));

  // This write forces auto-demote of k1 (LRU) to SSD
  std::vector<char> d3(512, 'C');
  assert(client.Put("k3", d3.data(), d3.size()));

  // k1 should still be accessible (demoted to SSD, not lost)
  assert(client.Exists("k1"));

  // k1's index should say LOCAL_SSD
  auto loc = client.Index().Lookup("k1");
  assert(loc.has_value());
  assert(loc->tier == StorageTier::LOCAL_SSD);

  // k3's index should say CPU_DRAM
  auto loc3 = client.Index().Lookup("k3");
  assert(loc3.has_value());
  assert(loc3->tier == StorageTier::CPU_DRAM);

  // Data integrity: read k1 back from SSD
  std::vector<char> buf(512, 0);
  assert(client.GetIntoPtr("k1", reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  assert(buf == d1);

  client.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_batch_get_ssd_tier() {
  std::cout << "test_batch_get_ssd_tier... ";

  UMBPConfig config;
  config.dram.capacity_bytes = 1024;  // Tiny DRAM: forces SSD demotion
  config.ssd.enabled = true;
  config.ssd.storage_dir = "/tmp/umbp_test_batch_get_ssd";
  config.ssd.capacity_bytes = 10 * 1024 * 1024;

  UMBPClient client(config);

  // Write 10 x 512-byte values; DRAM holds ~2, rest auto-demote to SSD.
  constexpr int N = 10;
  constexpr size_t kSize = 512;
  std::vector<std::string> keys(N);
  std::vector<std::vector<char>> data(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = "ssd_batch_" + std::to_string(i);
    data[i].assign(kSize, static_cast<char>('A' + i));
    assert(client.Put(keys[i], data[i].data(), data[i].size()));
  }

  // Batch read all 10 — mix of DRAM and SSD.
  std::vector<std::vector<char>> bufs(N, std::vector<char>(kSize, 0));
  std::vector<uintptr_t> ptrs(N);
  std::vector<size_t> sizes(N, kSize);
  for (int i = 0; i < N; ++i) {
    ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
  }

  auto results = client.BatchGetIntoPtr(keys, ptrs, sizes);
  for (int i = 0; i < N; ++i) {
    assert(results[i]);
    assert(bufs[i] == data[i]);
  }

  client.Clear();
  std::cout << "PASSED" << std::endl;
}

void test_batch_get_follower() {
  std::cout << "test_batch_get_follower... ";

  const std::string ssd_dir = "/tmp/umbp_test_batch_get_follower";

  // Leader: writes to DRAM + copies to shared SSD.
  UMBPConfig leader_cfg;
  leader_cfg.dram.capacity_bytes = 1024 * 1024;
  leader_cfg.ssd.enabled = true;
  leader_cfg.ssd.storage_dir = ssd_dir;
  leader_cfg.ssd.capacity_bytes = 10 * 1024 * 1024;
  leader_cfg.role = UMBPRole::SharedSSDLeader;
  leader_cfg.force_ssd_copy_on_write = true;
  leader_cfg.copy_pipeline.async_enabled = false;  // sync copy for determinism

  constexpr int N = 5;
  constexpr size_t kSize = 2048;
  std::vector<std::string> keys(N);
  std::vector<std::vector<char>> data(N);

  {
    UMBPClient leader(leader_cfg);
    for (int i = 0; i < N; ++i) {
      keys[i] = "follower_batch_" + std::to_string(i);
      data[i].assign(kSize, static_cast<char>('P' + i));
      assert(leader.Put(keys[i], data[i].data(), data[i].size()));
    }
  }  // leader destructor drains copy pipeline

  // Follower: reads from shared SSD via BatchGetIntoPtr.
  UMBPConfig follower_cfg;
  follower_cfg.dram.capacity_bytes = 1024 * 1024;
  follower_cfg.ssd.enabled = true;
  follower_cfg.ssd.storage_dir = ssd_dir;
  follower_cfg.ssd.capacity_bytes = 10 * 1024 * 1024;
  follower_cfg.role = UMBPRole::SharedSSDFollower;
  follower_cfg.follower_mode = true;

  UMBPClient follower(follower_cfg);

  std::vector<std::vector<char>> bufs(N, std::vector<char>(kSize, 0));
  std::vector<uintptr_t> ptrs(N);
  std::vector<size_t> sizes(N, kSize);
  for (int i = 0; i < N; ++i) {
    ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
  }

  auto results = follower.BatchGetIntoPtr(keys, ptrs, sizes);
  for (int i = 0; i < N; ++i) {
    assert(results[i]);
    assert(bufs[i] == data[i]);
  }

  follower.Clear();
  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== UMBPClient Tests ===" << std::endl;
  test_put_get();
  test_put_dedup();
  test_remove();
  test_batch_put_get();
  test_clear();
  test_put_from_ptr_get_into_ptr();
  test_dram_full_demote_with_index();
  test_batch_get_ssd_tier();
  test_batch_get_follower();
  std::cout << "All UMBPClient tests passed!" << std::endl;
  return 0;
}
